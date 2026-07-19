#!/usr/bin/env python3
"""Filesystem-isolated tests for the one-rung Flight A supervisor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


MODULE_PATH = Path(__file__).with_name("splatlab-flight-a-supervisor.py")
SPEC = importlib.util.spec_from_file_location(
    "splatlab_flight_a_supervisor", MODULE_PATH
)
assert SPEC and SPEC.loader
flight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = flight
SPEC.loader.exec_module(flight)


BOOT_ID = "12345678-1234-4567-89ab-123456789abc"
RECEIPT_NAME = "gpu-hardware-acceptance-20260713T120000Z.json"
ORIGINAL_MARKER = b'SPLAT_TRAINING_DISABLED_REASON="attended hardware hold"\n'
WATCHER_MARKER = (
    b"# Automatically asserted GPU hardware-maintenance sentinel.\n"
    b'SPLAT_TRAINING_DISABLED_REASON="watcher fault"\n'
)


def iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class FakeOps:
    def __init__(self, config: flight.Config, now: float):
        self.config = config
        self.now_value = now
        self.monotonic_value = 120_000_000_000
        self.active = True
        self.empty = False
        self.submit_calls = 0
        self.stop_calls = 0
        self.start_calls = 0
        self.status_calls = 0
        self.kernel_calls = 0
        self.resource_calls = 0
        self.sleep_calls = 0
        self.submit_error = False
        self.signal_on_submit = False
        self.monitor_kernel_fault = False
        self.kernel_fault_on_call: int | None = None
        self.marker_race_on_start = False
        self.watcher_active_reads = 0
        self.monitor_status = "completed"
        self.jump_after_monitor_status = False
        self.resource_fail_on_call: int | None = None
        self.holder_override: dict[str, Any] | None = None
        self.response_override: dict[str, Any] | None = None
        self.baseline_jobs: list[dict[str, Any]] = []
        self.compute_pids: list[int] = []
        self.cgroup_members: set[int] = {111}
        self.busy_on_status_call: int | None = None
        self.signal_on_status_call: int | None = None
        self.terminate_fail_on_call: int | None = None
        self.delay_on_kernel_call: int | None = None
        self.service_probe_error = False
        self.competing_calls = 0
        self.competing_busy_on_call: int | None = None
        self.competing_busy_unit = "comfyui.service"
        self.competing_states = {
            unit: "inactive" for unit in flight.COMPETING_WORKLOAD_UNITS
        }
        self.interactive_ai_scopes: set[str] = set()

    def now(self) -> float:
        return self.now_value

    def monotonic_ns(self) -> int:
        return self.monotonic_value

    def sleep(self, seconds: float) -> None:
        self.sleep_calls += 1
        self.now_value += seconds
        self.monotonic_value += int(seconds * 1_000_000_000)

    def watcher_unit_state(self) -> dict[str, str]:
        if self.watcher_active_reads > 0:
            self.watcher_active_reads -= 1
            return {
                "InvocationID": "a" * 32,
                "Result": "success",
                "ExecMainStatus": "0",
                "ActiveState": "activating",
                "SubState": "start",
            }
        return {
            "InvocationID": "a" * 32,
            "Result": "success",
            "ExecMainStatus": "0",
            "ActiveState": "inactive",
            "SubState": "dead",
        }

    def backups_idle(self) -> tuple[bool, dict[str, str]]:
        return True, {"backup": "inactive"}

    def auxiliary_units_idle(self) -> tuple[bool, dict[str, str]]:
        return True, {"auxiliary": "inactive"}

    def competing_workloads_idle(self) -> tuple[bool, dict[str, str]]:
        self.competing_calls += 1
        states = dict(self.competing_states)
        if self.competing_busy_on_call == self.competing_calls:
            states[self.competing_busy_unit] = "active"
        states.update({scope: "active" for scope in self.interactive_ai_scopes})
        return (
            not self.interactive_ai_scopes
            and all(
                states[unit] in {"inactive", "failed"}
                for unit in flight.COMPETING_WORKLOAD_UNITS
            ),
            states,
        )

    def service_active(self) -> bool:
        if self.service_probe_error:
            raise flight.SafetyError("simulated service-state probe failure")
        return self.active

    def cgroup_empty(self) -> bool:
        return self.empty

    def cgroup_pids(self) -> set[int]:
        return set() if self.empty else set(self.cgroup_members)

    def service_resource_safety(self) -> dict[str, int]:
        self.resource_calls += 1
        if self.resource_fail_on_call == self.resource_calls:
            raise flight.SafetyError("SplatLab began using swap")
        return {"memory_bytes": 1024**3, "swap_bytes": 0, "tasks": 8}

    def cpu_package_safety(self) -> dict[str, float]:
        return {"temperature_c": 45.0}

    def cpu_throttle_counts(self) -> dict[str, int]:
        return {"cpu0/thermal_throttle/package_throttle_count": 0}

    def ups_safety(self) -> dict[str, Any]:
        return {"status": "ONLINE", "load_percent": 35.0}

    def terminate_service_cgroup(self) -> None:
        self.stop_calls += 1
        if self.terminate_fail_on_call == self.stop_calls:
            raise flight.SafetyError("simulated cgroup termination failure")
        self.active = False
        self.empty = True

    def start_service(self) -> None:
        self.start_calls += 1
        if self.marker_race_on_start and not os.path.lexists(self.config.marker):
            self.config.marker.write_bytes(WATCHER_MARKER)
            os.chmod(self.config.marker, 0o644)
        self.active = True
        self.empty = False

    def kernel_faults(self, since_epoch: float | None = None) -> list[str]:
        self.kernel_calls += 1
        if self.delay_on_kernel_call == self.kernel_calls:
            self.sleep(flight.MAX_FINAL_ADMISSION_AGE_SECONDS + 1)
        if (
            self.kernel_fault_on_call is not None
            and self.kernel_calls >= self.kernel_fault_on_call
        ):
            return ["PCIe Bus Error: RxErr"]
        if since_epoch is not None and self.monitor_kernel_fault:
            return ["PCIe Bus Error: RxErr"]
        return []

    def gpu_safety(self, *, idle_required: bool) -> dict[str, Any]:
        compute_pids = [] if idle_required else list(self.compute_pids)
        return {
            "temperature_c": 35.0,
            "power_draw_w": 40.0,
            "power_limit_w": 400.0,
            "compute_process_count": len(compute_pids),
            "compute_pids": compute_pids,
        }

    @staticmethod
    def _job(job_id: str, status: str) -> dict[str, Any]:
        return {
            **flight.FLIGHT_A_PAYLOAD,
            "output_dir": str(flight.EXPECTED_OUTPUT_ROOT / job_id),
            "job_id": job_id,
            "status": status,
        }

    def api_status(self) -> dict[str, Any]:
        self.status_calls += 1
        if self.signal_on_status_call == self.status_calls:
            os.kill(os.getpid(), signal.SIGTERM)
        if self.busy_on_status_call == self.status_calls:
            return {
                "active_jobs": 1,
                "gpu": {
                    "locked": True,
                    "lane": "splat",
                    "job_id": "splat_deadbeef00:train",
                },
                "jobs": [self._job("splat_deadbeef00", "running")],
            }
        if self.status_calls <= 4:
            return {
                "active_jobs": 0,
                "gpu": {"locked": False},
                "jobs": list(self.baseline_jobs),
            }
        job = self._job("splat_0123456789", self.monitor_status)
        if self.jump_after_monitor_status:
            self.now_value += flight.MAX_FLIGHT_RUNTIME_SECONDS + 1
            self.jump_after_monitor_status = False
        locked = self.monitor_status != "completed"
        holder = self.holder_override or {
            "locked": locked,
            "lane": "splat" if locked else None,
            "job_id": "splat_0123456789:train" if locked else None,
        }
        return {
            "active_jobs": 0 if self.monitor_status == "completed" else 1,
            "gpu": holder,
            "jobs": [job, *self.baseline_jobs],
        }

    def submit_flight_a(self) -> dict[str, Any]:
        self.submit_calls += 1
        if self.signal_on_submit:
            os.kill(os.getpid(), signal.SIGTERM)
        if self.submit_error:
            raise flight.SafetyError("simulated lost response")
        if self.response_override is not None:
            return self.response_override
        return self._job("splat_0123456789", "starting")


@pytest.fixture
def rig(tmp_path: Path) -> tuple[flight.Supervisor, FakeOps, flight.Config]:
    now = 1_800_000_000.0
    marker = tmp_path / "config" / "gpu-hardware-maintenance.conf"
    acceptance = tmp_path / "reports" / "acceptance"
    watcher = tmp_path / "state" / "watcher" / "gpu_health_watch_status.json"
    state = tmp_path / "state" / "flight-a"
    runtime = tmp_path / "runtime"
    boot = tmp_path / "boot-id"
    for directory in (marker.parent, acceptance, watcher.parent, runtime):
        directory.mkdir(parents=True, mode=0o700)
        os.chmod(directory, 0o700)
    for directory in (watcher.parents[2], watcher.parents[1], watcher.parents[0]):
        os.chmod(directory, 0o700)
    marker.write_bytes(ORIGINAL_MARKER)
    os.chmod(marker, 0o644)
    boot.write_text(f"{BOOT_ID}\n", encoding="ascii")

    config = flight.Config(
        marker=marker,
        acceptance_dir=acceptance,
        watcher_status=watcher,
        state_dir=state,
        heavy_work_lock=runtime / "nexus-heavy-work.lock",
        boot_id=boot,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
        receipt_max_age_seconds=1800,
        watcher_max_age_seconds=480,
        monitor_interval_seconds=0.01,
    )
    ops = FakeOps(config, now)
    supervisor = flight.Supervisor(config, ops)

    marker_info = flight.marker_snapshot(marker, os.getuid())
    checks: list[dict[str, Any]] = []
    for name, count in flight.ACCEPTANCE_CHECK_COUNTS.items():
        for _index in range(count):
            checks.append(
                {
                    "name": name,
                    "passed": True,
                    "detail": "test evidence passed",
                    "evidence": {},
                    "source": "measured",
                }
            )

    def named(name: str) -> list[dict[str, Any]]:
        return [check for check in checks if check["name"] == name]

    marker_evidence = {
        "path": str(marker),
        **flight.asdict(marker_info),
        "reason_present": True,
    }
    named("maintenance_marker_start")[0]["evidence"] = marker_evidence
    named("maintenance_marker_final")[0]["evidence"] = dict(marker_evidence)
    named("nexus_heavy_work_lock")[0]["evidence"] = {
        "path": str(config.heavy_work_lock),
        "device": 1,
        "inode": 2,
        "uid": config.expected_uid,
        "gid": config.expected_gid,
        "mode": 0o600,
        "link_count": 1,
        "exclusive": True,
    }
    named("boot_identity")[0]["evidence"] = {"boot_id": BOOT_ID}
    named("motherboard_model")[0]["evidence"] = {"observed": flight.EXPECTED_BOARD}
    named("bios_version")[0]["evidence"] = {
        "observed": flight.EXPECTED_BIOS,
        "bios_date": "07/01/2026",
    }
    named("intel_me_version_measured")[0]["evidence"] = {
        "sysfs_path": "/sys/class/mei/mei0/fw_ver",
        "observed_versions": [flight.EXPECTED_ME, flight.EXPECTED_ME],
        "expected_version": flight.EXPECTED_ME,
    }
    named("firmware_package")[0]["evidence"] = {
        "size_bytes": 1024,
        "sha256": flight.EXPECTED_BIOS_PACKAGE_SHA256,
        "expected_name": flight.EXPECTED_BIOS_PACKAGE_NAME,
        "expected_sha256": flight.EXPECTED_BIOS_PACKAGE_SHA256,
    }
    private_artifact = {
        "path": "/private/evidence.txt",
        "uid": flight.EXPECTED_UID,
        "gid": flight.EXPECTED_UID,
        "mode": 0o600,
        "link_count": 1,
        "size_bytes": 100,
        "sha256": "4" * 64,
    }

    def structured_evidence(
        evidence_type: str,
        assertions: dict[str, Any],
        digest: str,
    ) -> dict[str, Any]:
        return {
            "path": f"/private/{evidence_type}.json",
            "uid": flight.EXPECTED_UID,
            "gid": flight.EXPECTED_UID,
            "mode": 0o600,
            "link_count": 1,
            "size_bytes": 100,
            "sha256": digest,
            "schema": "splatlab.operator-evidence.v1",
            "evidence_type": evidence_type,
            "host": flight.platform.node(),
            "boot_id": BOOT_ID,
            "recorded_at": iso(now - 30),
            "operator": "rtoony",
            "operator_uid": flight.EXPECTED_UID,
            "assertions": assertions,
            "artifacts": [dict(private_artifact)],
        }

    named("firmware_evidence")[0]["evidence"] = structured_evidence(
        "firmware",
        {
            "bios_defaults_loaded": True,
            "memory_auto_jedec_xmp_disabled": True,
            "asus_ai_and_multicore_overclocking_disabled": True,
            "firmware_package_name": flight.EXPECTED_BIOS_PACKAGE_NAME,
            "firmware_package_sha256": flight.EXPECTED_BIOS_PACKAGE_SHA256,
        },
        "1" * 64,
    )
    named("physical_inspection_evidence")[0]["evidence"] = structured_evidence(
        "physical",
        {
            "gpu_reseated": True,
            "gpu_support_checked": True,
            "native_12v_2x6_inspected_and_reseated": True,
            "eps_power_reseated": True,
            "connectors_undamaged": True,
        },
        "2" * 64,
    )
    named("memtest86_evidence")[0]["evidence"] = structured_evidence(
        "memtest86",
        {"completed": True, "test_mode": "full", "passes": 4, "errors": 0},
        "3" * 64,
    )
    for name in (
        "firmware_evidence",
        "physical_inspection_evidence",
        "memtest86_evidence",
    ):
        named(name)[0]["source"] = "structured_operator_evidence"
    aer_snapshot = {
        "gpu": {"aer_dev_correctable": {"TOTAL": 0}},
        "root_port": {"aer_rootport_total_err_cor": {"TOTAL": 0}},
    }
    named("aer_counters_boot_start")[0]["evidence"] = {"snapshot": aer_snapshot}
    named("aer_counters_after_idle")[0]["evidence"] = {"snapshot": aer_snapshot}
    pcie_evidence = {
        "gpu": {
            "vendor": "0x10de",
            "current_link_width": "16",
            "max_link_width": "16",
        },
        "root_port": {
            "vendor": "0x8086",
            "current_link_width": "16",
            "max_link_width": "16",
            "max_link_speed": "16.0 GT/s PCIe",
        },
    }
    for check in named("pcie_link"):
        check["evidence"] = pcie_evidence
    gpu_evidence = {
        "name": "NVIDIA GeForce RTX 5090",
        "pci_bus_id": "00000000:02:00.0",
        "power_limit_w": 400.0,
        "persistence_mode": "Disabled",
        "compute_process_count": 0,
        "pcie_replays_since_reset": 0,
        "hardware_throttle_states": ["not active", "not active"],
    }
    for check in named("gpu_safety_state"):
        check["evidence"] = dict(gpu_evidence)
    for name in ("nvidia_compute_idle", "nvidia_compute_idle_after_observation"):
        named(name)[0]["evidence"] = {"process_count": 0, "processes": []}
    inactive_units = {
        unit: "inactive" for unit in flight.EXPECTED_COMPETING_WORKLOAD_UNITS
    }
    for name in (
        "auxiliary_compute_units",
        "auxiliary_compute_units_after_observation",
    ):
        named(name)[0]["evidence"] = {"states": inactive_units}
    for check in named("compute_gate_blocked"):
        check["evidence"] = {"exit_code": 75}
    rapl_limits = {
        "long_term": flight.EXPECTED_RAPL_LONG_TERM_UW,
        "short_term": flight.EXPECTED_RAPL_SHORT_TERM_UW,
    }
    power_guard = {
        "limits_uw": rapl_limits,
        "expected_limits_uw": rapl_limits,
        "systemd": {
            "LoadState": "loaded",
            "UnitFileState": "enabled",
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecStart": (
                "{ argv[]=/usr/bin/bash "
                "/home/rtoony/scripts/aipc-cpu-power-guard.sh "
                "apply-live --pl1 125 --pl2 177 ; }"
            ),
            "FragmentPath": "/etc/systemd/system/aipc-cpu-power-guard.service",
        },
    }
    for name in ("cpu_power_guard_start", "cpu_power_guard_final"):
        named(name)[0]["evidence"] = power_guard
    for name in ("kernel_faults_current_boot", "kernel_faults_during_idle"):
        named(name)[0]["evidence"] = {"match_count": 0, "matches": []}
    for check in named("splatlab_browse_health"):
        check["evidence"] = {
            "http_status": 200,
            "ok": True,
            "service": "splatlab",
        }
    watcher_evidence = {
        "schema": flight.EXPECTED_WATCHER_SCHEMA,
        "boot_id": BOOT_ID,
        "interlock_status": "already-active",
        "journal_ok": True,
        "run_success": True,
        "last_error_is_null": True,
        "validation_errors": [],
        "fault_counts": {
            "gpu_unreadable": 0,
            "xid": 0,
            "aer_current": 0,
            "aer_previous": 3,
            "aer_severe": 0,
            "platform_fatal": 0,
        },
    }
    for name in (
        "gpu_health_watcher_status",
        "gpu_health_watcher_status_after_observation",
    ):
        named(name)[0]["evidence"] = dict(watcher_evidence)
    samples = [
        {
            "elapsed_seconds": float(index * 10),
            "sample_gap_seconds": 10.0,
            "boot_id": BOOT_ID,
            "marker_unchanged": True,
            "compute_process_count": 0,
            "aer_nonzero": [],
            "aer_unchanged": True,
            "compute_unit_states": inactive_units,
            "cpu_rapl_limits_uw": rapl_limits,
            "watcher_invocation_id": "a" * 32,
            "watcher_finished_at_epoch": now - 10,
            "watcher_monotonic_age_seconds": 0.0,
        }
        for index in range(1, 91)
    ]
    named("continuous_idle_samples")[0]["evidence"] = {
        "sample_count": 90,
        "required_sample_count": 90,
        "maximum_permitted_gap_seconds": 12.0,
        "samples": samples,
    }
    named("idle_observation_duration")[0]["evidence"] = {
        "observed_monotonic_seconds": 900.0,
        "required_seconds": 900,
    }
    named("watcher_receipt_continuity")[0]["evidence"] = {
        "unique_invocation_ids": ["a" * 32],
        "post_start_invocation_ids": ["a" * 32],
    }
    report = {
        "schema_version": 2,
        "tool": flight.EXPECTED_ACCEPTANCE_TOOL,
        "host": flight.platform.node(),
        "started_at": iso(now - 920),
        "finished_at": iso(now - 10),
        "verdict": "PASS_PRE_FLIGHT_A",
        "required_idle_observation_seconds": 900,
        "maintenance_marker_path": str(marker),
        "maintenance_marker_action": "retained; this tool never clears it",
        "operator_authorization_required_for_flight_a": True,
        "flight_a_was_not_run": True,
        "checks": checks,
    }
    receipt = acceptance / RECEIPT_NAME
    receipt_payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    receipt.write_bytes(receipt_payload)
    os.chmod(receipt, 0o600)
    checksum = acceptance / f"{RECEIPT_NAME}.sha256"
    checksum.write_text(
        f"{hashlib.sha256(receipt_payload).hexdigest()}  {RECEIPT_NAME}\n",
        encoding="ascii",
    )
    os.chmod(checksum, 0o600)

    watcher_payload = {
        "schema": flight.EXPECTED_WATCHER_SCHEMA,
        "tool": flight.EXPECTED_WATCHER_TOOL,
        "unit": "nexus-gpu-health-watch.service",
        "boot_id": BOOT_ID,
        "invocation_id": "a" * 32,
        "started_at_epoch": int(now - 20),
        "finished_at_epoch": int(now - 10),
        "started_at_monotonic_ns": 100_000_000_000,
        "finished_at_monotonic_ns": 110_000_000_000,
        "journal_ok": True,
        "previous_journal_ok": True,
        "probe_counts": {
            "gpu_attempted": 1,
            "gpu_ok": 1,
            "kernel_journal_attempted": 1,
            "kernel_journal_ok": 1,
            "previous_journal_attempted": 1,
            "previous_journal_ok": 1,
        },
        "fault_counts": {
            "gpu_unreadable": 0,
            "xid": 0,
            "aer_current": 0,
            "aer_previous": 3,
            "aer_severe": 0,
            "platform_fatal": 0,
        },
        "interlock_status": "already-active",
        "run_success": True,
        "last_error": None,
    }
    watcher.write_text(json.dumps(watcher_payload) + "\n", encoding="utf-8")
    os.chmod(watcher, 0o600)
    os.utime(watcher, (now - 10, now - 10))
    return supervisor, ops, config


def rewrite_receipt(config: flight.Config, payload: bytes) -> None:
    receipt = config.acceptance_dir / RECEIPT_NAME
    receipt.write_bytes(payload)
    os.chmod(receipt, 0o600)
    checksum = Path(f"{receipt}.sha256")
    checksum.write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {receipt.name}\n",
        encoding="ascii",
    )
    os.chmod(checksum, 0o600)


def create_archived_state(
    supervisor: flight.Supervisor,
    ops: FakeOps,
    config: flight.Config,
    *,
    transition_id: str,
    receipt_sha: str,
    phase: str = "submitting",
    submit_attempted: bool = True,
    job_id: str | None = None,
) -> dict[str, Any]:
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    preserved = supervisor._preserved_path(transition_id)
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    os.link(config.marker, preserved, follow_symlinks=False)
    config.marker.unlink()
    state: dict[str, Any] = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": transition_id,
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": phase,
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": receipt_sha,
        "consumed_receipt_path": str(config.consumed / f"{receipt_sha}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(preserved),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": submit_attempted,
        "job_id": job_id,
        "marker_archived_epoch": ops.now() - 10,
    }
    if submit_attempted:
        state["monitor_started_epoch"] = ops.now() - 5
    flight.atomic_write_json(config.pending, state, config.expected_uid)
    return state


def test_success_submits_once_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    result = supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert ops.start_calls == 2  # fresh ungated start, then gated browse restart
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()
    payload = json.loads(result.read_text())
    assert payload["outcome"] == "completed"
    assert payload["marker_relocked_before_review"] is True
    assert payload["payload_sha256"] == flight.payload_sha256()
    assert len(list(config.consumed.glob("*.json"))) == 1


def test_recovery_termination_failure_relocks_and_retains_pending(rig) -> None:
    supervisor, ops, config = rig
    ops.terminate_fail_on_call = 2

    with pytest.raises(flight.SafetyError, match="recovery did not complete"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert ops.stop_calls == 3
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert ops.active is False
    assert ops.empty is True
    assert config.pending.exists()


def test_api_job_race_at_post_boundary_consumes_but_never_submits(rig) -> None:
    supervisor, ops, config = rig
    ops.busy_on_status_call = 4

    with pytest.raises(flight.SafetyError, match="active job"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert len(list(config.consumed.glob("*.json"))) == 1


def test_signal_during_final_api_check_never_submits(rig) -> None:
    supervisor, ops, config = rig
    ops.signal_on_status_call = 4

    with pytest.raises(flight.SafetyError, match="SIGTERM"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert len(list(config.consumed.glob("*.json"))) == 1


def test_expired_final_admission_never_consumes_or_submits(rig) -> None:
    supervisor, ops, config = rig
    ops.delay_on_kernel_call = 4

    with pytest.raises(flight.SafetyError, match="final admission expired"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not list(config.consumed.glob("*.json"))


def test_monitor_fault_terminates_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.monitor_kernel_fault = True

    with pytest.raises(flight.SafetyError, match="kernel fault"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert ops.stop_calls >= 2
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert ops.active is True
    assert not config.pending.exists()


def test_final_admission_fault_before_archive_never_submits(rig) -> None:
    supervisor, ops, config = rig
    ops.kernel_fault_on_call = 2

    with pytest.raises(flight.SafetyError, match="final admission"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()


def test_resource_stop_after_submit_terminates_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.resource_fail_on_call = 3

    with pytest.raises(flight.SafetyError, match="swap"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_runtime_bound_terminates_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.monitor_status = "running"
    ops.jump_after_monitor_status = True

    with pytest.raises(flight.SafetyError, match="runtime bound"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_foreign_compute_process_aborts_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.compute_pids = [999]

    with pytest.raises(flight.SafetyError, match="foreign NVIDIA compute"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_wrong_gpu_arbiter_holder_aborts_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.holder_override = {
        "locked": True,
        "lane": "llm",
        "job_id": "foreign:operation",
    }

    with pytest.raises(flight.SafetyError, match="unauthorized operation"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_signal_after_submit_never_retries_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.signal_on_submit = True

    with pytest.raises(flight.SafetyError, match="SIGTERM"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()
    result = json.loads(next(config.results.glob("*.json")).read_text())
    assert result["outcome"] == "ambiguous_submission"


def test_ambiguous_submission_is_not_retried(rig) -> None:
    supervisor, ops, config = rig
    ops.submit_error = True

    with pytest.raises(flight.AmbiguousSubmission, match="will not be retried"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    result_files = list(config.results.glob("*.json"))
    assert len(result_files) == 1
    assert json.loads(result_files[0].read_text())["outcome"] == "ambiguous_submission"

    ops.submit_error = False
    ops.status_calls = 0
    with pytest.raises(flight.SafetyError, match="already consumed"):
        supervisor.run(RECEIPT_NAME)
    assert ops.submit_calls == 1


def test_watcher_marker_wins_relock_race(rig) -> None:
    supervisor, ops, config = rig
    ops.marker_race_on_start = True

    with pytest.raises(flight.SafetyError, match="reasserted"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == WATCHER_MARKER
    assert not list(config.marker.parent.glob(".*.flight-a-preserved.*"))


def test_payload_mismatch_fails_after_only_one_submission(rig) -> None:
    supervisor, ops, config = rig
    response = FakeOps._job("splat_0123456789", "starting")
    response["max_num_iterations"] = 7001
    ops.response_override = response

    with pytest.raises(flight.SafetyError, match="payload mismatch"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_normalized_output_path_mismatch_fails_after_one_submission(rig) -> None:
    supervisor, ops, config = rig
    response = FakeOps._job("splat_0123456789", "starting")
    response["output_dir"] = "outputs/3d"
    ops.response_override = response

    with pytest.raises(flight.SafetyError, match="output_dir"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_active_job_blocks_before_marker_transition(rig) -> None:
    supervisor, ops, config = rig
    ops.baseline_jobs = [FakeOps._job("splat_deadbeef00", "running")]

    def busy_status() -> dict[str, Any]:
        return {
            "active_jobs": 1,
            "gpu": {"locked": True},
            "jobs": ops.baseline_jobs,
        }

    ops.api_status = busy_status  # type: ignore[method-assign]
    with pytest.raises(flight.SafetyError, match="active job"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER


@pytest.mark.parametrize("unit", flight.COMPETING_WORKLOAD_UNITS)
def test_each_competing_service_blocks_before_marker_transition(rig, unit: str) -> None:
    supervisor, ops, config = rig
    ops.competing_states[unit] = "active"

    with pytest.raises(flight.SafetyError, match="competing workload"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()


def test_interactive_ai_scope_blocks_before_marker_transition(rig) -> None:
    supervisor, ops, config = rig
    ops.interactive_ai_scopes.add("aipc-safe-run-1000-123-1800000000.scope")

    with pytest.raises(flight.SafetyError, match="competing workload"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()


def test_competing_service_after_submit_aborts_and_relocks(rig) -> None:
    supervisor, ops, config = rig
    ops.competing_busy_on_call = 5
    ops.competing_busy_unit = "media-batch-transcode.service"

    with pytest.raises(flight.SafetyError, match="began during Flight A"):
        supervisor.run(RECEIPT_NAME)

    assert ops.submit_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()


def test_nexus_heavy_work_lock_blocks_before_transition(rig) -> None:
    supervisor, ops, config = rig
    fd = os.open(config.heavy_work_lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(flight.HeavyWorkBusy, match="Nexus heavy workload"):
            supervisor.run(RECEIPT_NAME)
    finally:
        os.close(fd)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()


def test_nexus_heavy_work_lock_covers_final_recovery(rig) -> None:
    supervisor, ops, config = rig
    original_start_service = ops.start_service
    lock_probes = 0

    def start_service_with_lock_probe() -> None:
        nonlocal lock_probes
        fd = os.open(config.heavy_work_lock, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        lock_probes += 1
        original_start_service()

    ops.start_service = start_service_with_lock_probe  # type: ignore[method-assign]

    supervisor.run(RECEIPT_NAME)

    assert lock_probes == 2  # fresh Flight A start and final gated recovery restart
    fd = os.open(config.heavy_work_lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_host_ops_detects_exact_active_ai_scope(monkeypatch) -> None:
    ops = flight.HostOps(flight.Config())

    def fake_run(argv: list[str], *, timeout: int = 45):
        assert argv[-1] == "aipc-safe-run-*.scope"
        assert timeout == 15
        return subprocess.CompletedProcess(
            argv,
            0,
            "aipc-safe-run-1000-123-1800000000.scope loaded active running test\n",
            "",
        )

    monkeypatch.setattr(ops, "run", fake_run)

    assert ops.active_interactive_ai_scopes() == [
        "aipc-safe-run-1000-123-1800000000.scope"
    ]


def test_host_ops_rejects_malformed_ai_scope_inventory(monkeypatch) -> None:
    ops = flight.HostOps(flight.Config())

    def fake_run(argv: list[str], *, timeout: int = 45):
        return subprocess.CompletedProcess(
            argv,
            0,
            "unrelated.scope loaded active running test\n",
            "",
        )

    monkeypatch.setattr(ops, "run", fake_run)

    with pytest.raises(flight.SafetyError, match="scope state is malformed"):
        ops.active_interactive_ai_scopes()


def test_transition_flock_rejects_second_supervisor(rig) -> None:
    supervisor, _ops, _config = rig
    other = flight.Supervisor(
        supervisor.config, FakeOps(supervisor.config, 1_800_000_000)
    )

    with supervisor.transition_lock():
        with pytest.raises(flight.SafetyError, match="owns the lock"):
            with other.transition_lock():
                pytest.fail("second supervisor acquired the transition lock")


def test_boot_dependency_allows_live_lock_owner_without_recovery(rig) -> None:
    supervisor, ops, config = rig
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    transition_id = "9" * 32
    preserved = supervisor._preserved_path(transition_id)
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    os.link(config.marker, preserved, follow_symlinks=False)
    config.marker.unlink()
    state = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": transition_id,
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": "marker_archived",
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": "9" * 64,
        "consumed_receipt_path": str(config.consumed / f"{'9' * 64}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(preserved),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": False,
        "job_id": None,
        "marker_archived_epoch": ops.now(),
    }
    flight.atomic_write_json(config.pending, state, config.expected_uid)

    with supervisor.transition_lock():
        assert (
            supervisor.recover(restart_service=False, allow_active_transition=True)
            is True
        )

    assert ops.stop_calls == 0
    assert not config.marker.exists()
    assert preserved.exists()


def test_boot_dependency_rejects_lock_without_pending_state(rig) -> None:
    supervisor, _ops, _config = rig

    with supervisor.transition_lock():
        with pytest.raises(flight.SafetyError, match="no durable pending"):
            supervisor.recover(restart_service=False, allow_active_transition=True)


def test_boot_dependency_allows_final_gated_recovery_start(rig) -> None:
    supervisor, ops, config = rig
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    transition_id = "3" * 32
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    state = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": transition_id,
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": "recovering",
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": "3" * 64,
        "consumed_receipt_path": str(config.consumed / f"{'3' * 64}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(supervisor._preserved_path(transition_id)),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": False,
        "job_id": None,
        "recovery_detail": "final gated restart",
    }
    flight.atomic_write_json(config.pending, state, config.expected_uid)

    with supervisor.transition_lock():
        assert (
            supervisor.recover(restart_service=False, allow_active_transition=True)
            is True
        )

    assert ops.stop_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_boot_recovery_restores_without_submission(rig) -> None:
    supervisor, ops, config = rig
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    transition_id = "b" * 32
    preserved = supervisor._preserved_path(transition_id)
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    os.link(config.marker, preserved, follow_symlinks=False)
    config.marker.unlink()
    state = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": transition_id,
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": "submitting",
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": "c" * 64,
        "consumed_receipt_path": str(config.consumed / f"{'c' * 64}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(preserved),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": True,
        "job_id": None,
        "marker_archived_epoch": ops.now() - 10,
        "monitor_started_epoch": ops.now() - 5,
    }
    flight.atomic_write_json(config.pending, state, config.expected_uid)

    assert supervisor.recover() is True
    assert ops.submit_calls == 0
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.pending.exists()
    result = json.loads(next(config.results.glob("*.json")).read_text())
    assert result["outcome"] == "recovered_without_resubmission"


def test_recovery_preserves_already_published_result(rig) -> None:
    supervisor, ops, config = rig
    state = create_archived_state(
        supervisor,
        ops,
        config,
        transition_id="7" * 32,
        receipt_sha="7" * 64,
        phase="monitoring",
        job_id="splat_0123456789",
    )
    result = {
        "schema": flight.RESULT_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": state["transition_id"],
        "boot_id": state["boot_id"],
        "finished_at": flight.utc_now(),
        "outcome": "completed",
        "detail": "original durable completion",
        "job_id": "splat_0123456789",
        "submit_attempted": True,
        "payload_sha256": state["payload_sha256"],
        "receipt_name": state["receipt_name"],
        "receipt_sha256": state["receipt_sha256"],
        "marker_disposition": "preserved_marker_restored",
        "marker_relocked_before_review": True,
    }
    path = config.results / f"flight-a-{state['transition_id']}.json"
    flight.publish_exclusive_json(path, result, config.expected_uid)

    assert supervisor.recover() is True
    observed = json.loads(path.read_text())
    assert observed["outcome"] == "completed"
    assert observed["detail"] == "original durable completion"
    assert not config.pending.exists()


def test_recovery_rejects_invalid_existing_result_fields(rig) -> None:
    supervisor, ops, config = rig
    state = create_archived_state(
        supervisor,
        ops,
        config,
        transition_id="1" * 32,
        receipt_sha="1" * 64,
    )
    invalid_result = {
        "schema": flight.RESULT_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": state["transition_id"],
        "boot_id": state["boot_id"],
        "finished_at": flight.utc_now(),
        "outcome": "not-a-real-outcome",
        "detail": "invalid durable result",
        "job_id": None,
        "submit_attempted": True,
        "payload_sha256": state["payload_sha256"],
        "receipt_name": state["receipt_name"],
        "receipt_sha256": state["receipt_sha256"],
        "marker_disposition": "preserved_marker_restored",
        "marker_relocked_before_review": True,
    }
    path = config.results / f"flight-a-{state['transition_id']}.json"
    flight.publish_exclusive_json(path, invalid_result, config.expected_uid)

    assert supervisor.recover() is False
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert config.pending.exists()


def test_recovery_rejects_duplicate_key_in_existing_result(rig) -> None:
    supervisor, ops, config = rig
    state = create_archived_state(
        supervisor,
        ops,
        config,
        transition_id="8" * 32,
        receipt_sha="8" * 64,
    )
    flight.ensure_private_directory(config.results, config.expected_uid)
    path = config.results / f"flight-a-{state['transition_id']}.json"
    payload = {
        "schema": flight.RESULT_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": state["transition_id"],
        "boot_id": state["boot_id"],
        "finished_at": flight.utc_now(),
        "outcome": "completed",
        "detail": "original",
        "job_id": None,
        "submit_attempted": True,
        "payload_sha256": state["payload_sha256"],
        "receipt_name": state["receipt_name"],
        "receipt_sha256": state["receipt_sha256"],
        "marker_disposition": "preserved_marker_restored",
        "marker_relocked_before_review": True,
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    needle = f'"schema": "{flight.RESULT_SCHEMA}"'.encode()
    encoded = encoded.replace(needle, needle + b', "schema": "invalid"', 1)
    path.write_bytes(encoded)
    os.chmod(path, 0o600)

    assert supervisor.recover() is False
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert config.pending.exists()


def test_hostile_pending_transition_id_never_constructs_result_path(rig) -> None:
    supervisor, ops, config = rig
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    state = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": "../../escape",
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": "validated",
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": "6" * 64,
        "consumed_receipt_path": str(config.consumed / f"{'6' * 64}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(config.marker.parent / "escape"),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": False,
        "job_id": None,
    }
    flight.atomic_write_json(config.pending, state, config.expected_uid)

    assert supervisor.recover() is False
    assert ops.stop_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert not config.results.exists()


def test_recovery_rejects_duplicate_key_in_pending_state(rig) -> None:
    supervisor, ops, config = rig
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    transition_id = "5" * 32
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    state = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": transition_id,
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": "validated",
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": "5" * 64,
        "consumed_receipt_path": str(config.consumed / f"{'5' * 64}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(supervisor._preserved_path(transition_id)),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": False,
        "job_id": None,
    }
    encoded = (json.dumps(state, sort_keys=True) + "\n").encode()
    needle = f'"transition_id": "{transition_id}"'.encode()
    encoded = encoded.replace(
        needle, needle + b', "transition_id": "' + b"4" * 32 + b'"', 1
    )
    config.pending.write_bytes(encoded)
    os.chmod(config.pending, 0o600)

    assert supervisor.recover() is False
    assert ops.stop_calls == 1
    assert config.marker.read_bytes() == ORIGINAL_MARKER


def test_boot_recovery_can_defer_gated_service_start(rig) -> None:
    supervisor, ops, config = rig
    flight.ensure_private_directory(config.state_dir, config.expected_uid)
    transition_id = "d" * 32
    preserved = supervisor._preserved_path(transition_id)
    snapshot = flight.marker_snapshot(config.marker, config.expected_uid)
    os.link(config.marker, preserved, follow_symlinks=False)
    config.marker.unlink()
    state = {
        "schema": flight.STATE_SCHEMA,
        "tool": flight.TOOL_NAME,
        "transition_id": transition_id,
        "created_at": flight.utc_now(),
        "updated_at": flight.utc_now(),
        "phase": "marker_archived",
        "boot_id": BOOT_ID,
        "receipt_name": RECEIPT_NAME,
        "receipt_sha256": "e" * 64,
        "consumed_receipt_path": str(config.consumed / f"{'e' * 64}.json"),
        "marker_snapshot": flight.asdict(snapshot),
        "preserved_marker_path": str(preserved),
        "payload_sha256": flight.payload_sha256(),
        "baseline_job_ids": [],
        "cpu_throttle_counts": {"cpu0/thermal_throttle/package_throttle_count": 0},
        "submit_attempted": False,
        "job_id": None,
        "marker_archived_epoch": ops.now() - 10,
    }
    flight.atomic_write_json(config.pending, state, config.expected_uid)

    ops.active = False
    ops.empty = True
    assert supervisor.recover(restart_service=False) is True
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert ops.active is False
    assert ops.start_calls == 0
    assert ops.stop_calls == 0
    assert ops.submit_calls == 0


def test_deferred_recovery_probe_failure_still_attempts_termination(rig) -> None:
    supervisor, ops, config = rig
    create_archived_state(
        supervisor,
        ops,
        config,
        transition_id="2" * 32,
        receipt_sha="2" * 64,
        phase="marker_archived",
        submit_attempted=False,
    )
    ops.service_probe_error = True

    assert supervisor.recover(restart_service=False) is False
    assert config.marker.read_bytes() == ORIGINAL_MARKER
    assert ops.stop_calls == 1
    assert ops.active is False
    assert ops.empty is True
    assert config.pending.exists()


def test_previous_journal_visibility_is_diagnostic_not_a_gate(rig) -> None:
    supervisor, ops, config = rig
    payload = json.loads(config.watcher_status.read_text())
    payload["previous_journal_ok"] = False
    payload["probe_counts"]["previous_journal_ok"] = 0
    config.watcher_status.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.chmod(config.watcher_status, 0o600)
    os.utime(config.watcher_status, (ops.now() - 10, ops.now() - 10))

    status = supervisor.validate_watcher_status(BOOT_ID)

    assert status["previous_journal_ok"] is False


def test_watcher_active_oneshot_is_retried_then_identity_bound(rig) -> None:
    supervisor, ops, _config = rig
    ops.watcher_active_reads = 2

    status = supervisor.validate_watcher_status(BOOT_ID)

    assert status["invocation_id"] == "a" * 32
    assert ops.sleep_calls == 2


def test_watcher_status_rejects_schema_extension(rig) -> None:
    supervisor, ops, config = rig
    payload = json.loads(config.watcher_status.read_text())
    payload["unexpected"] = "field"
    config.watcher_status.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    os.chmod(config.watcher_status, 0o600)
    os.utime(config.watcher_status, (ops.now() - 10, ops.now() - 10))

    with pytest.raises(flight.SafetyError, match="top-level keys"):
        supervisor.validate_watcher_status(BOOT_ID)


def test_acceptance_duplicate_key_is_rejected_before_transition(rig) -> None:
    supervisor, ops, config = rig
    receipt = config.acceptance_dir / RECEIPT_NAME
    payload = receipt.read_bytes()
    needle = b'"verdict": "PASS_PRE_FLIGHT_A"'
    duplicate = needle + b',\n  "verdict": "PASS_PRE_FLIGHT_A"'
    rewrite_receipt(config, payload.replace(needle, duplicate, 1))

    with pytest.raises(flight.SafetyError, match="duplicate key"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0


def test_acceptance_requires_all_90_absolute_samples(rig) -> None:
    supervisor, ops, config = rig
    receipt = config.acceptance_dir / RECEIPT_NAME
    report = json.loads(receipt.read_text())
    check = next(
        value
        for value in report["checks"]
        if value["name"] == "continuous_idle_samples"
    )
    check["evidence"]["samples"].pop()
    check["evidence"]["sample_count"] = 89
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    rewrite_receipt(config, payload)

    with pytest.raises(flight.SafetyError, match="exactly 90"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0


@pytest.mark.parametrize(
    "duration_evidence",
    [
        {},
        {
            "observed_monotonic_seconds": 899.499,
            "required_seconds": 900,
        },
        {
            "observed_monotonic_seconds": 900.0,
            "required_seconds": 900,
            "unexpected": True,
        },
    ],
)
def test_acceptance_requires_exact_monotonic_duration_evidence(
    rig, duration_evidence: dict[str, Any]
) -> None:
    supervisor, ops, config = rig
    receipt = config.acceptance_dir / RECEIPT_NAME
    report = json.loads(receipt.read_text())
    check = next(
        value
        for value in report["checks"]
        if value["name"] == "idle_observation_duration"
    )
    check["evidence"] = duration_evidence
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    rewrite_receipt(config, payload)

    with pytest.raises(flight.SafetyError, match="monotonic idle-duration"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0


def test_acceptance_rejects_nonzero_active_aer_evidence(rig) -> None:
    supervisor, ops, config = rig
    receipt = config.acceptance_dir / RECEIPT_NAME
    report = json.loads(receipt.read_text())
    check = next(
        value
        for value in report["checks"]
        if value["name"] == "gpu_health_watcher_status"
    )
    check["evidence"]["fault_counts"]["aer_current"] = 1
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    rewrite_receipt(config, payload)

    with pytest.raises(flight.SafetyError, match="watcher.*evidence"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0


def test_generated_units_keep_recovery_uninstalled_but_wired() -> None:
    unit_root = Path(__file__).parents[1] / "deploy" / "systemd" / "user"
    flight_unit = (unit_root / "splatlab-flight-a@.service").read_text()
    recovery_unit = (unit_root / "splatlab-flight-a-boot-recovery.service").read_text()
    drop_in = (
        unit_root / "splatlab.service.d" / "90-flight-a-recovery.conf"
    ).read_text()

    assert (
        "ExecStopPost=/home/rtoony/projects/splatlab/tools/splatlab-flight-a-supervisor.py recover"
        in flight_unit
    )
    assert "--authorize-exact-flight-a" in flight_unit
    assert "recover --defer-service-start --allow-active-transition" in recovery_unit
    assert "Before=splatlab.service" in recovery_unit
    assert "Requires=splatlab-flight-a-boot-recovery.service" in drop_in
    assert "TimeoutStartSec=2h15min" in flight_unit
    assert (
        "ExecStartPre=/usr/bin/timeout --signal=TERM --kill-after=2s 15s "
        "/home/rtoony/bin/nexus-svc-inject" in flight_unit
    )


def test_sensitive_file_opens_are_nonblocking(rig, monkeypatch) -> None:
    _supervisor, _ops, config = rig
    real_open = flight.os.open
    observed: dict[str, int] = {}

    def recording_open(path, flags, *args, **kwargs):
        if path == config.marker:
            observed["regular"] = flags
        if path == RECEIPT_NAME and "dir_fd" in kwargs:
            observed["private_child"] = flags
        if path == config.watcher_status.name and "dir_fd" in kwargs:
            observed["watcher"] = flags
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(flight.os, "open", recording_open)
    flight.read_regular_file(config.marker, uid=config.expected_uid)
    flight.read_private_directory_child(
        config.acceptance_dir,
        RECEIPT_NAME,
        uid=config.expected_uid,
        gid=config.expected_gid,
        max_bytes=4 * 1024 * 1024,
    )
    flight.secure_read_watcher_status(
        config.watcher_status,
        uid=config.expected_uid,
        gid=config.expected_gid,
    )

    assert observed["regular"] & os.O_NONBLOCK
    assert observed["private_child"] & os.O_NONBLOCK
    assert observed["watcher"] & os.O_NONBLOCK


def test_run_cli_requires_generated_systemd_unit(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.write_text(
        "0::/user.slice/app.slice/splatlab.slice/"
        "splatlab-flight-a@gpu-hardware-acceptance-test.service\n",
        encoding="ascii",
    )
    environment = {
        "INVOCATION_ID": "a" * 32,
        "SYSTEMD_EXEC_PID": str(os.getpid()),
    }

    flight.require_systemd_flight_invocation(
        environment=environment, cgroup_path=cgroup
    )
    with pytest.raises(flight.SafetyError, match="only through"):
        flight.require_systemd_flight_invocation(environment={}, cgroup_path=cgroup)


def test_stale_or_wrong_boot_receipt_blocks_without_transition(rig) -> None:
    supervisor, ops, config = rig
    receipt = config.acceptance_dir / RECEIPT_NAME
    report = json.loads(receipt.read_text())
    boot_check = next(
        check for check in report["checks"] if check["name"] == "boot_identity"
    )
    boot_check["evidence"]["boot_id"] = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    receipt.write_bytes(payload)
    checksum = Path(f"{receipt}.sha256")
    checksum.write_text(
        f"{hashlib.sha256(payload).hexdigest()}  {receipt.name}\n", encoding="ascii"
    )

    with pytest.raises(flight.SafetyError, match="another boot"):
        supervisor.run(RECEIPT_NAME)

    assert ops.stop_calls == 0
    assert ops.submit_calls == 0


def test_cli_exposes_no_payload_overrides() -> None:
    args = flight.parse_args(
        ["run", "--receipt", RECEIPT_NAME, "--authorize-exact-flight-a"]
    )
    assert args.command == "run"
    assert vars(args) == {
        "command": "run",
        "receipt": RECEIPT_NAME,
        "authorize_exact_flight_a": True,
    }
    with pytest.raises(SystemExit):
        flight.parse_args(
            [
                "run",
                "--receipt",
                RECEIPT_NAME,
                "--authorize-exact-flight-a",
                "--max-num-iterations",
                "8000",
            ]
        )


def test_watcher_long_activation_survives_race_budget(rig) -> None:
    supervisor, ops, _config = rig
    ops.watcher_active_reads = 20

    status = supervisor.validate_watcher_status(BOOT_ID)

    assert status["invocation_id"] == "a" * 32
    assert ops.sleep_calls == 20
