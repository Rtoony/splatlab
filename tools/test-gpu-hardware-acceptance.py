#!/usr/bin/env python3
"""Isolated unit tests for gpu-hardware-acceptance.py."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


TOOL = Path(__file__).with_name("gpu-hardware-acceptance.py")
SPEC = importlib.util.spec_from_file_location("gpu_hardware_acceptance", TOOL)
assert SPEC and SPEC.loader
acceptance = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = acceptance
SPEC.loader.exec_module(acceptance)


class AcceptanceToolTests(unittest.TestCase):
    BOOT_ID = "11111111-2222-4333-8444-555555555555"
    INVOCATION_ID = "a" * 32

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_paths(self) -> acceptance.Paths:
        dmi = self.root / "dmi"
        gpu = self.root / "gpu"
        root_port = self.root / "root-port"
        local = self.root / ".local"
        state = local / "state"
        watcher_dir = state / "nexus-watchers"
        dmi.mkdir()
        gpu.mkdir()
        root_port.mkdir()
        local.mkdir(mode=0o700)
        state.mkdir(mode=0o700)
        watcher_dir.mkdir(mode=0o700)
        for directory in (local, state, watcher_dir):
            directory.chmod(0o700)
        (dmi / "board_name").write_text(acceptance.EXPECTED_BOARD + "\n")
        (dmi / "bios_version").write_text(acceptance.EXPECTED_BIOS + "\n")
        (dmi / "bios_date").write_text("07/01/2026\n")
        (self.root / "me-fw-version").write_text(
            f"0:{acceptance.EXPECTED_ME}\n1:{acceptance.EXPECTED_ME}\n"
        )
        rapl = self.root / "rapl"
        rapl.mkdir()
        (rapl / "constraint_0_name").write_text("long_term\n")
        (rapl / "constraint_0_power_limit_uw").write_text(
            f"{acceptance.EXPECTED_RAPL_LONG_TERM_UW}\n"
        )
        (rapl / "constraint_1_name").write_text("short_term\n")
        (rapl / "constraint_1_power_limit_uw").write_text(
            f"{acceptance.EXPECTED_RAPL_SHORT_TERM_UW}\n"
        )
        (self.root / "boot-id").write_text(self.BOOT_ID + "\n")
        (self.root / "marker").write_text('SPLAT_TRAINING_DISABLED_REASON="test"\n')
        (self.root / "marker").chmod(0o644)
        values = {
            "current_link_speed": "2.5 GT/s PCIe\n",
            "current_link_width": "16\n",
            "max_link_width": "16\n",
            "vendor": "0x10de\n",
            "device": "0x2b85\n",
        }
        for name, value in {**values, "max_link_speed": "32.0 GT/s PCIe\n"}.items():
            (gpu / name).write_text(value)
        for name, value in {
            **values,
            "vendor": "0x8086\n",
            "device": "0xae4d\n",
            "max_link_speed": "16.0 GT/s PCIe\n",
        }.items():
            (root_port / name).write_text(value)
        for device, names in (
            (gpu, acceptance.AER_DEVICE_FILES),
            (root_port, acceptance.AER_ROOT_FILES),
        ):
            for name in names:
                value = (
                    "0\n" if name.startswith("aer_rootport_total") else "TOTAL_ERR 0\n"
                )
                (device / name).write_text(value)
        return acceptance.Paths(
            board_name=dmi / "board_name",
            bios_version=dmi / "bios_version",
            bios_date=dmi / "bios_date",
            me_fw_version=self.root / "me-fw-version",
            rapl_package=rapl,
            boot_id=self.root / "boot-id",
            gpu_device=gpu,
            root_port=root_port,
            marker=self.root / "marker",
            watcher_status=watcher_dir / "gpu_health_watch_status.json",
            report_dir=self.root / "reports",
        )

    def watcher_payload(
        self,
        *,
        invocation_id: str | None = None,
        finished_epoch: float = 1_000.0,
        finished_monotonic_ns: int = 1_000_000_000_000,
    ) -> dict[str, object]:
        return {
            "schema": acceptance.WATCHER_STATUS_SCHEMA,
            "tool": acceptance.WATCHER_STATUS_TOOL,
            "unit": acceptance.WATCHER_STATUS_UNIT,
            "boot_id": self.BOOT_ID,
            "invocation_id": invocation_id or self.INVOCATION_ID,
            "started_at_epoch": finished_epoch - 1.0,
            "finished_at_epoch": finished_epoch,
            "started_at_monotonic_ns": finished_monotonic_ns - 1_000_000_000,
            "finished_at_monotonic_ns": finished_monotonic_ns,
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
                "aer_previous": 9,
                "aer_severe": 0,
                "platform_fatal": 0,
            },
            "interlock_status": "already-active",
            "run_success": True,
            "last_error": None,
        }

    def power_guard_unit_state(self) -> dict[str, str]:
        return {
            "LoadState": "loaded",
            "UnitFileState": "enabled",
            "ActiveState": "inactive",
            "SubState": "dead",
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecStart": (
                "{ path=/usr/bin/bash ; argv[]=/usr/bin/bash "
                f"{acceptance.EXPECTED_POWER_GUARD_SCRIPT} "
                "apply-live --pl1 125 --pl2 177 ; }"
            ),
            "FragmentPath": str(acceptance.EXPECTED_POWER_GUARD_FRAGMENT),
        }

    def write_operator_record(
        self,
        *,
        evidence_type: str,
        boot_id: str | None = None,
        now: float | None = None,
    ) -> tuple[Path, dict[str, object]]:
        now = time.time() if now is None else now
        directory = self.root / "operator-evidence"
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
        artifact = directory / f"{evidence_type}-artifact.txt"
        artifact.write_text(f"attended {evidence_type} evidence\n")
        artifact.chmod(0o600)
        os.utime(artifact, (now, now))
        if evidence_type == "physical":
            assertions: dict[str, object] = {
                name: True for name in acceptance.PHYSICAL_ASSERTIONS
            }
        elif evidence_type == "firmware":
            assertions = {
                "bios_defaults_loaded": True,
                "memory_auto_jedec_xmp_disabled": True,
                "asus_ai_and_multicore_overclocking_disabled": True,
                "firmware_package_name": acceptance.EXPECTED_BIOS_PACKAGE_NAME,
                "firmware_package_sha256": acceptance.EXPECTED_BIOS_PACKAGE_SHA256,
            }
        else:
            assertions = {
                "completed": True,
                "test_mode": "full",
                "passes": 4,
                "errors": 0,
            }
        record = {
            "schema": acceptance.OPERATOR_EVIDENCE_SCHEMA,
            "evidence_type": evidence_type,
            "host": acceptance.platform.node(),
            "boot_id": boot_id or self.BOOT_ID,
            "recorded_at": datetime.fromtimestamp(now, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "operator": acceptance.pwd.getpwuid(acceptance.EXPECTED_UID).pw_name,
            "operator_uid": acceptance.EXPECTED_UID,
            "assertions": assertions,
            "artifacts": [
                {
                    "filename": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
        }
        path = directory / f"{evidence_type}.json"
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        path.chmod(0o600)
        os.utime(path, (now, now))
        return path, assertions

    def write_watcher_payload(
        self,
        paths: acceptance.Paths,
        payload: dict[str, object],
    ) -> None:
        paths.watcher_status.write_text(json.dumps(payload, sort_keys=True) + "\n")
        paths.watcher_status.chmod(0o600)
        finished_epoch = float(payload["finished_at_epoch"])
        timestamp_ns = int(finished_epoch * 1_000_000_000)
        os.utime(paths.watcher_status, ns=(timestamp_ns, timestamp_ns))

    def watcher_unit_state(
        self,
        *,
        invocation_id: str | None = None,
        result: str = "success",
        exit_status: str = "0",
        active_state: str = "inactive",
        sub_state: str = "dead",
    ) -> dict[str, str]:
        return {
            "InvocationID": invocation_id or self.INVOCATION_ID,
            "Result": result,
            "ExecMainStatus": exit_status,
            "ActiveState": active_state,
            "SubState": sub_state,
        }

    def test_marker_snapshot_records_identity_and_hash(self) -> None:
        paths = self.make_paths()
        check, snapshot = acceptance.check_marker(paths)
        self.assertTrue(check.passed)
        self.assertEqual(snapshot["mode"], 0o644)
        self.assertEqual(len(snapshot["sha256"]), 64)

    def test_marker_symlink_is_rejected(self) -> None:
        paths = self.make_paths()
        target = self.root / "marker-target"
        target.write_text("active\n")
        paths.marker.unlink()
        paths.marker.symlink_to(target)
        check, snapshot = acceptance.check_marker(paths)
        self.assertFalse(check.passed)
        self.assertIsNone(snapshot)

    def test_expected_platform_and_pcie_link_pass(self) -> None:
        paths = self.make_paths()
        self.assertTrue(all(item.passed for item in acceptance.check_platform(paths)))
        self.assertTrue(acceptance.check_pcie(paths).passed)

    def test_intel_me_version_is_measured_and_must_match(self) -> None:
        paths = self.make_paths()
        me = acceptance.check_platform(paths)[2]
        self.assertTrue(me.passed)
        self.assertEqual(me.source, "measured")
        self.assertIn(acceptance.EXPECTED_ME, me.evidence["observed_versions"])
        paths.me_fw_version.write_text("0:19.0.0.1\n")
        self.assertFalse(acceptance.check_platform(paths)[2].passed)

    def test_cpu_power_guard_requires_persistence_and_exact_limits(self) -> None:
        paths = self.make_paths()
        with patch.object(
            acceptance,
            "read_power_guard_unit_state",
            side_effect=self.power_guard_unit_state,
        ):
            self.assertTrue(acceptance.check_cpu_power_guard(paths, "cpu_guard").passed)
            (paths.rapl_package / "constraint_1_power_limit_uw").write_text(
                "200000000\n"
            )
            self.assertFalse(
                acceptance.check_cpu_power_guard(paths, "cpu_guard").passed
            )

        state = self.power_guard_unit_state()
        state["UnitFileState"] = "disabled"
        with patch.object(
            acceptance,
            "read_power_guard_unit_state",
            return_value=state,
        ):
            self.assertFalse(
                acceptance.check_cpu_power_guard(paths, "cpu_guard").passed
            )

    def test_heavy_work_lock_is_private_exclusive_and_nonblocking(self) -> None:
        lock = self.root / "nexus-heavy-work.lock"
        with acceptance.hold_heavy_work_lock(lock) as evidence:
            self.assertTrue(evidence["exclusive"])
            self.assertEqual(evidence["mode"], 0o600)
            with self.assertRaises(acceptance.HeavyWorkBusy):
                with acceptance.hold_heavy_work_lock(lock):
                    self.fail("contending acceptance acquired the shared lock")

        lock.chmod(0o644)
        with self.assertRaises(acceptance.SafetyError):
            with acceptance.hold_heavy_work_lock(lock):
                self.fail("unsafe public lock was accepted")

    def test_competing_workload_check_matches_supervisor_denylist(self) -> None:
        inactive = {unit: "inactive" for unit in acceptance.KNOWN_COMPUTE_UNITS}
        with patch.object(
            acceptance, "read_compute_unit_states", return_value=inactive
        ):
            self.assertTrue(acceptance.check_compute_units().passed)
        active_scope = {**inactive, "aipc-safe-run-codex.scope": "active"}
        with patch.object(
            acceptance, "read_compute_unit_states", return_value=active_scope
        ):
            self.assertFalse(acceptance.check_compute_units().passed)

    def test_structured_operator_evidence_is_private_fresh_and_bound(self) -> None:
        for evidence_type in ("physical", "firmware", "memtest86"):
            with self.subTest(evidence_type=evidence_type):
                path, assertions = self.write_operator_record(
                    evidence_type=evidence_type
                )
                evidence = acceptance.validate_operator_evidence(
                    path,
                    evidence_type=evidence_type,
                    boot_id=self.BOOT_ID,
                    expected_assertions=assertions,
                )
                self.assertEqual(evidence["mode"], 0o600)
                self.assertEqual(evidence["artifacts"][0]["mode"], 0o600)
                self.assertEqual(evidence["boot_id"], self.BOOT_ID)

    def test_operator_evidence_rejects_public_stale_unbound_and_bad_hash(self) -> None:
        path, assertions = self.write_operator_record(evidence_type="memtest86")
        path.chmod(0o644)
        with self.assertRaisesRegex(
            acceptance.OperatorEvidenceError, "private, nonempty 0600"
        ):
            acceptance.validate_operator_evidence(
                path,
                evidence_type="memtest86",
                boot_id=self.BOOT_ID,
                expected_assertions=assertions,
            )

        path.chmod(0o600)
        stale = time.time() - acceptance.OPERATOR_EVIDENCE_MAX_AGE_SECONDS - 1
        os.utime(path, (stale, stale))
        with self.assertRaisesRegex(acceptance.OperatorEvidenceError, "stale"):
            acceptance.validate_operator_evidence(
                path,
                evidence_type="memtest86",
                boot_id=self.BOOT_ID,
                expected_assertions=assertions,
            )

        os.utime(path, None)
        with self.assertRaisesRegex(acceptance.OperatorEvidenceError, "host and boot"):
            acceptance.validate_operator_evidence(
                path,
                evidence_type="memtest86",
                boot_id="99999999-8888-4777-8666-555555555555",
                expected_assertions=assertions,
            )

        record = json.loads(path.read_text())
        record["artifacts"][0]["sha256"] = "f" * 64
        path.write_text(json.dumps(record) + "\n")
        path.chmod(0o600)
        with self.assertRaisesRegex(acceptance.OperatorEvidenceError, "hash"):
            acceptance.validate_operator_evidence(
                path,
                evidence_type="memtest86",
                boot_id=self.BOOT_ID,
                expected_assertions=assertions,
            )

    def test_generated_operator_templates_are_private_and_incomplete(self) -> None:
        paths = self.make_paths()
        directory = self.root / "templates"
        written = acceptance.write_operator_evidence_templates(directory, paths)
        self.assertEqual(len(written), 3)
        self.assertEqual(os.stat(directory).st_mode & 0o777, 0o700)
        for path in written:
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["artifacts"], [])

    def test_firmware_package_requires_exact_name_and_hash(self) -> None:
        package = self.root / acceptance.EXPECTED_BIOS_PACKAGE_NAME
        package.write_bytes(b"standard-board-3202-package")
        digest = hashlib.sha256(package.read_bytes()).hexdigest()
        with patch.object(acceptance, "EXPECTED_BIOS_PACKAGE_SHA256", digest):
            self.assertTrue(acceptance.check_firmware_package(package).passed)

            wrong_name = self.root / "ROG-MAXIMUS-Z890-HERO-BTF-ASUS-3202.ZIP"
            wrong_name.write_bytes(package.read_bytes())
            self.assertFalse(acceptance.check_firmware_package(wrong_name).passed)

            link = self.root / acceptance.EXPECTED_BIOS_PACKAGE_NAME.lower()
            link.symlink_to(package)
            self.assertFalse(acceptance.check_firmware_package(link).passed)

        package.write_bytes(b"wrong-package")
        self.assertFalse(acceptance.check_firmware_package(package).passed)

    def test_wrong_bios_and_width_fail(self) -> None:
        paths = self.make_paths()
        paths.bios_version.write_text("3002\n")
        (paths.gpu_device / "current_link_width").write_text("8\n")
        self.assertFalse(acceptance.check_platform(paths)[1].passed)
        self.assertFalse(acceptance.check_pcie(paths).passed)

    def test_aer_counters_require_absolute_zero(self) -> None:
        paths = self.make_paths()
        check, baseline = acceptance.check_aer_counters(paths, "aer")
        self.assertTrue(check.passed)
        self.assertIsNotNone(baseline)
        (paths.root_port / "aer_rootport_total_err_cor").write_text("3\n")
        check, _ = acceptance.check_aer_counters(paths, "aer", baseline)
        self.assertFalse(check.passed)
        self.assertIn("TOTAL=3", check.detail)

    def test_watcher_receipt_accepts_fresh_latest_invocation(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)
        check, snapshot = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertTrue(check.passed)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["fault_counts"]["aer_previous"], 9)

        exact_boundary, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_360.0,
            monotonic_ns_fn=lambda: 1_360_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertTrue(exact_boundary.passed)

    def test_consumer_contract_accepts_producer_payload(self) -> None:
        producer_tool = TOOL.parents[3] / "nexus-watchers" / "gpu_health_watch.py"
        producer_spec = importlib.util.spec_from_file_location(
            "gpu_health_watch_contract_test",
            producer_tool,
        )
        self.assertIsNotNone(producer_spec)
        self.assertIsNotNone(producer_spec.loader)
        producer = importlib.util.module_from_spec(producer_spec)
        with patch.dict(os.environ, {"INVOCATION_ID": self.INVOCATION_ID}):
            producer_spec.loader.exec_module(producer)
            receipt = producer._new_status_receipt(self.BOOT_ID)
        receipt.update(
            {
                "finished_at_epoch": receipt["started_at_epoch"] + 1,
                "finished_at_monotonic_ns": (
                    receipt["started_at_monotonic_ns"] + 1_000_000_000
                ),
                "journal_ok": True,
                "previous_journal_ok": False,
                "interlock_status": "already-active",
                "run_success": True,
                "last_error": None,
            }
        )
        receipt["probe_counts"].update(
            {
                "gpu_attempted": 1,
                "gpu_ok": 1,
                "kernel_journal_attempted": 1,
                "kernel_journal_ok": 1,
                "previous_journal_attempted": 1,
                "previous_journal_ok": 0,
            }
        )
        payload = producer._status_payload(receipt)

        self.assertEqual(set(payload), acceptance.WATCHER_STATUS_KEYS)
        self.assertEqual(acceptance._watcher_status_structure_errors(payload), [])
        self.assertIn("private_state_failure", acceptance.WATCHER_ERROR_STATES)

    def test_acceptance_and_flight_supervisor_safety_constants_match(self) -> None:
        supervisor_path = TOOL.with_name("splatlab-flight-a-supervisor.py")
        supervisor_spec = importlib.util.spec_from_file_location(
            "splatlab_flight_a_acceptance_contract_test",
            supervisor_path,
        )
        self.assertIsNotNone(supervisor_spec)
        self.assertIsNotNone(supervisor_spec.loader)
        supervisor = importlib.util.module_from_spec(supervisor_spec)
        sys.modules[supervisor_spec.name] = supervisor
        supervisor_spec.loader.exec_module(supervisor)
        self.assertEqual(
            set(acceptance.KNOWN_COMPUTE_UNITS),
            set(supervisor.COMPETING_WORKLOAD_UNITS),
        )
        self.assertEqual(
            acceptance.NEXUS_HEAVY_WORK_LOCK,
            supervisor.NEXUS_HEAVY_WORK_LOCK,
        )
        self.assertEqual(
            acceptance.EXPECTED_RAPL_LONG_TERM_UW,
            supervisor.EXPECTED_RAPL_LONG_TERM_UW,
        )
        self.assertEqual(
            acceptance.EXPECTED_RAPL_SHORT_TERM_UW,
            supervisor.EXPECTED_RAPL_SHORT_TERM_UW,
        )

    def test_watcher_receipt_rejects_stale_or_mismatched_identity(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)

        stale, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_361.0,
            monotonic_ns_fn=lambda: 1_361_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(stale.passed)
        self.assertIn("status_wall_age", stale.detail)

        payload["boot_id"] = "99999999-8888-4777-8666-555555555555"
        self.write_watcher_payload(paths, payload)
        wrong_boot, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(wrong_boot.passed)
        self.assertIn("status_boot_id_mismatch", wrong_boot.detail)

        payload["boot_id"] = self.BOOT_ID
        self.write_watcher_payload(paths, payload)
        wrong_invocation, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=lambda: self.watcher_unit_state(invocation_id="b" * 32),
            race_retries=0,
        )
        self.assertFalse(wrong_invocation.passed)
        self.assertIn("status_invocation_mismatch", wrong_invocation.detail)

    def test_watcher_receipt_rejects_faults_and_failed_service(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        payload["fault_counts"]["aer_current"] = 1
        payload["interlock_status"] = "asserted"
        self.write_watcher_payload(paths, payload)
        faulted, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(faulted.passed)
        self.assertIn("status_active_fault_aer_current", faulted.detail)
        self.assertIn("status_interlock_not_maintenance_locked", faulted.detail)

        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)
        failed_unit, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=lambda: self.watcher_unit_state(
                result="exit-code",
                exit_status="1",
            ),
            race_retries=0,
        )
        self.assertFalse(failed_unit.passed)
        self.assertIn("watcher_unit_result", failed_unit.detail)

    def test_watcher_receipt_allows_unavailable_previous_journal_only(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        payload["previous_journal_ok"] = False
        payload["probe_counts"]["previous_journal_ok"] = 0
        self.write_watcher_payload(paths, payload)
        check, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertTrue(check.passed)

        payload["previous_journal_ok"] = True
        self.write_watcher_payload(paths, payload)
        mismatch, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(mismatch.passed)
        self.assertIn("status_previous_journal_mismatch", mismatch.detail)

    def test_watcher_receipt_rejects_malformed_schema_and_types(self) -> None:
        paths = self.make_paths()

        def extra_key(payload: dict[str, object]) -> None:
            payload["unexpected"] = True

        def boolean_fault(payload: dict[str, object]) -> None:
            payload["fault_counts"]["aer_current"] = False

        def invalid_probe(payload: dict[str, object]) -> None:
            payload["probe_counts"]["gpu_ok"] = -1

        def unknown_error(payload: dict[str, object]) -> None:
            payload["last_error"] = "raw exception text"

        for mutation in (extra_key, boolean_fault, invalid_probe, unknown_error):
            with self.subTest(mutation=mutation.__name__):
                payload = self.watcher_payload()
                mutation(payload)
                self.write_watcher_payload(paths, payload)
                check, _ = acceptance.check_watcher_status_receipt(
                    paths,
                    wall_time_fn=lambda: 1_000.0,
                    monotonic_ns_fn=lambda: 1_000_000_000_000,
                    unit_state_fn=self.watcher_unit_state,
                    race_retries=0,
                )
                self.assertFalse(check.passed)

    def test_watcher_receipt_requires_each_freshness_clock(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)

        monotonic_stale, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_361_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(monotonic_stale.passed)
        self.assertIn("status_monotonic_age", monotonic_stale.detail)

        stale_mtime_ns = 639_000_000_000
        os.utime(paths.watcher_status, ns=(stale_mtime_ns, stale_mtime_ns))
        mtime_stale, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(mtime_stale.passed)
        self.assertIn("status_mtime_age", mtime_stale.detail)

        self.write_watcher_payload(paths, payload)
        monotonic_future, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 999_000_000_000,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(monotonic_future.passed)
        self.assertIn("status_monotonic_age", monotonic_future.detail)

    def test_watcher_receipt_rejects_hardlink_and_parent_symlink(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)
        alias = paths.watcher_status.with_name("status-hardlink.json")
        os.link(paths.watcher_status, alias)
        hardlink, _ = acceptance.check_watcher_status_receipt(
            paths,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(hardlink.passed)
        self.assertIn("status_file_wrong_link_count", hardlink.detail)
        alias.unlink()

        alias_local = self.root / "alias-local"
        alias_local.symlink_to(self.root / ".local", target_is_directory=True)
        alias_status = (
            alias_local / "state" / "nexus-watchers" / paths.watcher_status.name
        )
        with self.assertRaises(acceptance.WatcherStatusError) as raised:
            acceptance.secure_read_watcher_status(alias_status)
        self.assertIn(
            raised.exception.code,
            {"status_parent_not_directory", "status_file_io"},
        )

    def test_watcher_receipt_retries_a_running_oneshot_race(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)
        states = [
            self.watcher_unit_state(active_state="activating", sub_state="start"),
            self.watcher_unit_state(),
            self.watcher_unit_state(),
        ]

        def unit_state() -> dict[str, str]:
            return states.pop(0)

        retried, _ = acceptance.check_watcher_status_receipt(
            paths,
            wall_time_fn=lambda: 1_000.0,
            monotonic_ns_fn=lambda: 1_000_000_000_000,
            sleep_fn=lambda _seconds: None,
            unit_state_fn=unit_state,
            race_retries=1,
        )
        self.assertTrue(retried.passed)

        running, _ = acceptance.check_watcher_status_receipt(
            paths,
            sleep_fn=lambda _seconds: None,
            unit_state_fn=lambda: self.watcher_unit_state(
                active_state="activating",
                sub_state="start",
            ),
            race_retries=1,
        )
        self.assertFalse(running.passed)
        self.assertIn("watcher_unit_running", running.detail)

    def test_watcher_receipt_rejects_unsafe_file_and_duplicate_json(self) -> None:
        paths = self.make_paths()
        payload = self.watcher_payload()
        self.write_watcher_payload(paths, payload)
        paths.watcher_status.chmod(0o640)
        wrong_mode, _ = acceptance.check_watcher_status_receipt(
            paths,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(wrong_mode.passed)
        self.assertIn("status_file_wrong_mode", wrong_mode.detail)

        paths.watcher_status.chmod(0o600)
        paths.watcher_status.parent.chmod(0o775)
        parent_mode, _ = acceptance.check_watcher_status_receipt(
            paths,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(parent_mode.passed)
        self.assertIn("status_parent_wrong_mode", parent_mode.detail)
        paths.watcher_status.parent.chmod(0o700)

        target = paths.watcher_status.with_name("status-target.json")
        paths.watcher_status.rename(target)
        paths.watcher_status.symlink_to(target.name)
        symlink, _ = acceptance.check_watcher_status_receipt(
            paths,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(symlink.passed)
        self.assertIn("status_file_not_regular", symlink.detail)

        paths.watcher_status.unlink()
        duplicate = (
            '{"schema":"nexus.gpu-health-watch.status.v1",'
            '"schema":"nexus.gpu-health-watch.status.v1",'
            + json.dumps(payload, sort_keys=True)[1:]
        )
        paths.watcher_status.write_text(duplicate)
        paths.watcher_status.chmod(0o600)
        timestamp_ns = 1_000_000_000_000
        os.utime(paths.watcher_status, ns=(timestamp_ns, timestamp_ns))
        duplicate_check, _ = acceptance.check_watcher_status_receipt(
            paths,
            unit_state_fn=self.watcher_unit_state,
            race_retries=0,
        )
        self.assertFalse(duplicate_check.passed)
        self.assertIn("status_duplicate_json_key", duplicate_check.detail)

    def test_watcher_receipt_fifo_swap_is_nonblocking_and_rejected(self) -> None:
        paths = self.make_paths()
        self.write_watcher_payload(paths, self.watcher_payload())
        real_open = os.open
        swapped = False

        def swap_before_open(
            name: object,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal swapped
            if name == paths.watcher_status.name and dir_fd is not None and not swapped:
                swapped = True
                self.assertTrue(flags & os.O_NONBLOCK)
                paths.watcher_status.unlink()
                os.mkfifo(paths.watcher_status, 0o600)
            return real_open(name, flags, mode, dir_fd=dir_fd)

        with patch.object(acceptance.os, "open", side_effect=swap_before_open):
            with self.assertRaises(acceptance.WatcherStatusError) as raised:
                acceptance.secure_read_watcher_status(paths.watcher_status)
        self.assertEqual(raised.exception.code, "status_file_race")

    def test_watcher_receipt_rejects_parent_mode_race(self) -> None:
        paths = self.make_paths()
        self.write_watcher_payload(paths, self.watcher_payload())
        real_lstat = os.lstat

        def change_mode_before_final_parent_check(path: object) -> os.stat_result:
            if Path(path) == paths.watcher_status.parent:
                paths.watcher_status.parent.chmod(0o770)
            return real_lstat(path)

        try:
            with patch.object(
                acceptance.os,
                "lstat",
                side_effect=change_mode_before_final_parent_check,
            ):
                with self.assertRaises(acceptance.WatcherStatusError) as raised:
                    acceptance.secure_read_watcher_status(paths.watcher_status)
            self.assertEqual(raised.exception.code, "status_parent_wrong_mode")
        finally:
            paths.watcher_status.parent.chmod(0o700)

    def test_watcher_unit_timeout_is_categorical(self) -> None:
        timeout = acceptance.subprocess.TimeoutExpired(["systemctl"], 30)
        with patch.object(acceptance, "run_command", side_effect=timeout):
            with self.assertRaises(acceptance.WatcherStatusError) as raised:
                acceptance.read_watcher_unit_state()
        self.assertEqual(raised.exception.code, "watcher_unit_query_failed")

    def test_watcher_receipt_path_ignores_environment_redirect(self) -> None:
        redirected = str(self.root / "attacker-status.json")
        with patch.dict(os.environ, {"GPU_HEALTH_STATUS_FILE": redirected}):
            self.assertEqual(
                acceptance.Paths().watcher_status,
                acceptance.WATCHER_STATUS_PATH,
            )

    def test_900_second_observation_uses_continuous_clean_samples(self) -> None:
        paths = self.make_paths()
        _, marker = acceptance.check_marker(paths)
        _, aer = acceptance.check_aer_counters(paths, "aer")
        self.assertIsNotNone(marker)
        self.assertIsNotNone(aer)
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        initial_watcher = {"invocation_id": "0" * 32}

        def watcher_status(*_args: object, **_kwargs: object):
            invocation_id = f"{int(clock[0] // 180):032x}"
            snapshot = {
                "invocation_id": invocation_id,
                "finished_at_epoch": 1_000.0 + clock[0],
                "finished_at_monotonic_ns": int(clock[0] * 1_000_000_000),
                "monotonic_age_seconds": 0.0,
            }
            # Model nonzero final-probe runtime. Duration evidence must remain
            # bound to the sample timestamp, not this post-sample tail.
            clock[0] += 0.1
            return acceptance.Check("watcher", True, "ok"), snapshot

        passing = acceptance.Check("stub", True, "ok")
        inactive = {unit: "inactive" for unit in acceptance.KNOWN_COMPUTE_UNITS}
        with (
            patch.object(acceptance, "read_compute_processes", return_value=[]),
            patch.object(acceptance, "read_compute_unit_states", return_value=inactive),
            patch.object(acceptance, "check_kernel_journal", return_value=passing),
            patch.object(acceptance, "check_pcie", return_value=passing),
            patch.object(acceptance, "check_gpu_state", return_value=passing),
            patch.object(acceptance, "check_compute_idle", return_value=passing),
            patch.object(acceptance, "check_compute_units", return_value=passing),
            patch.object(acceptance, "check_compute_gate", return_value=passing),
            patch.object(acceptance, "check_watcher_timer", return_value=passing),
            patch.object(
                acceptance,
                "check_watcher_status_receipt",
                side_effect=watcher_status,
            ),
            patch.object(acceptance, "check_splatlab_health", return_value=passing),
        ):
            checks = acceptance.observe_idle_window(
                paths,
                marker,
                aer,
                initial_watcher,
                seconds=900,
                sleep_fn=sleep,
                monotonic_fn=lambda: clock[0],
                monotonic_ns_fn=lambda: int(clock[0] * 1_000_000_000),
                wall_time_fn=lambda: 1_000.0 + clock[0],
            )
        duration = next(
            item for item in checks if item.name == "idle_observation_duration"
        )
        samples = next(
            item for item in checks if item.name == "continuous_idle_samples"
        )
        self.assertTrue(duration.passed)
        self.assertEqual(
            duration.evidence,
            {
                "observed_monotonic_seconds": 900.0,
                "required_seconds": 900,
            },
        )
        self.assertTrue(samples.passed)
        self.assertEqual(samples.evidence["sample_count"], 90)
        continuity = next(
            item for item in checks if item.name == "watcher_receipt_continuity"
        )
        self.assertTrue(continuity.passed)

    def test_idle_observation_rejects_one_900_second_sampling_gap(self) -> None:
        paths = self.make_paths()
        _, marker = acceptance.check_marker(paths)
        _, aer = acceptance.check_aer_counters(paths, "aer")
        self.assertIsNotNone(marker)
        self.assertIsNotNone(aer)
        clock = [0.0]

        def stalled_sleep(_seconds: float) -> None:
            clock[0] += 900.0

        watcher_snapshot = {
            "invocation_id": "b" * 32,
            "finished_at_epoch": 1_900.0,
            "finished_at_monotonic_ns": 900_000_000_000,
            "monotonic_age_seconds": 0.0,
        }
        passing = acceptance.Check("stub", True, "ok")
        with (
            patch.object(acceptance, "check_kernel_journal", return_value=passing),
            patch.object(acceptance, "check_pcie", return_value=passing),
            patch.object(acceptance, "check_gpu_state", return_value=passing),
            patch.object(acceptance, "check_compute_idle", return_value=passing),
            patch.object(acceptance, "check_compute_units", return_value=passing),
            patch.object(acceptance, "check_compute_gate", return_value=passing),
            patch.object(acceptance, "check_watcher_timer", return_value=passing),
            patch.object(
                acceptance,
                "check_watcher_status_receipt",
                return_value=(
                    acceptance.Check("watcher", True, "ok"),
                    watcher_snapshot,
                ),
            ),
            patch.object(acceptance, "check_splatlab_health", return_value=passing),
        ):
            checks = acceptance.observe_idle_window(
                paths,
                marker,
                aer,
                {"invocation_id": "a" * 32},
                seconds=900,
                sleep_fn=stalled_sleep,
                monotonic_fn=lambda: clock[0],
                monotonic_ns_fn=lambda: int(clock[0] * 1_000_000_000),
                wall_time_fn=lambda: 1_000.0 + clock[0],
            )

        samples = next(
            item for item in checks if item.name == "continuous_idle_samples"
        )
        self.assertFalse(samples.passed)
        self.assertEqual(samples.evidence["sample_count"], 0)
        self.assertIn("sampling gap 900.000s", samples.detail)

    def test_idle_observation_aborts_when_watcher_receipt_stales(self) -> None:
        paths = self.make_paths()
        _, marker = acceptance.check_marker(paths)
        _, aer = acceptance.check_aer_counters(paths, "aer")
        self.assertIsNotNone(marker)
        self.assertIsNotNone(aer)
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        def watcher_status(*_args: object, **_kwargs: object):
            if clock[0] > 360.0:
                return (
                    acceptance.Check(
                        "watcher",
                        False,
                        "watcher status blocked: status_monotonic_age",
                    ),
                    None,
                )
            return (
                acceptance.Check("watcher", True, "ok"),
                {
                    "invocation_id": "a" * 32,
                    "finished_at_epoch": 1_000.0,
                    "finished_at_monotonic_ns": 0,
                    "monotonic_age_seconds": clock[0],
                },
            )

        passing = acceptance.Check("stub", True, "ok")
        inactive = {unit: "inactive" for unit in acceptance.KNOWN_COMPUTE_UNITS}
        with (
            patch.object(acceptance, "read_compute_processes", return_value=[]),
            patch.object(acceptance, "read_compute_unit_states", return_value=inactive),
            patch.object(acceptance, "check_kernel_journal", return_value=passing),
            patch.object(acceptance, "check_pcie", return_value=passing),
            patch.object(acceptance, "check_gpu_state", return_value=passing),
            patch.object(acceptance, "check_compute_idle", return_value=passing),
            patch.object(acceptance, "check_compute_units", return_value=passing),
            patch.object(acceptance, "check_compute_gate", return_value=passing),
            patch.object(acceptance, "check_watcher_timer", return_value=passing),
            patch.object(
                acceptance,
                "check_watcher_status_receipt",
                side_effect=watcher_status,
            ),
            patch.object(acceptance, "check_splatlab_health", return_value=passing),
        ):
            checks = acceptance.observe_idle_window(
                paths,
                marker,
                aer,
                {"invocation_id": "a" * 32},
                seconds=900,
                sleep_fn=sleep,
                monotonic_fn=lambda: clock[0],
                monotonic_ns_fn=lambda: int(clock[0] * 1_000_000_000),
                wall_time_fn=lambda: 1_000.0 + clock[0],
            )
        samples = next(
            item for item in checks if item.name == "continuous_idle_samples"
        )
        self.assertFalse(samples.passed)
        self.assertIn("status_monotonic_age", samples.detail)

    def test_fault_classifier_ignores_benign_lines(self) -> None:
        lines = [
            "kernel: pcieport: AER: enabled with IRQ 125",
            "kernel: thermal zone initialized",
            "kernel: NVRM: loading NVIDIA UNIX module",
        ]
        self.assertEqual(acceptance.classify_fault_lines(lines), [])

    def test_fault_classifier_finds_aer_xid_oom_and_platform_fatal(self) -> None:
        lines = [
            "kernel: AER: Correctable error message received",
            "kernel: NVRM: Xid (PCI:0000:02:00): 79",
            "kernel: Out of memory: Killed process 123",
            "kernel: CPER record reports fatal CATERR",
        ]
        matches = acceptance.classify_fault_lines(lines)
        self.assertEqual(len(matches), 4)
        self.assertIn("pcie_aer", matches[0]["categories"])
        self.assertIn("nvidia_xid", matches[1]["categories"])
        self.assertIn("oom", matches[2]["categories"])
        self.assertIn("platform_fatal", matches[3]["categories"])

    def test_report_is_exclusive_and_contains_no_marker_action(self) -> None:
        report = {
            "started_at": "2026-07-13T20:00:00Z",
            "verdict": "BLOCKED",
            "maintenance_marker_action": "retained; this tool never clears it",
        }
        directory = self.root / "reports"
        first = acceptance.write_report(report, directory)
        second = acceptance.write_report(report, directory)
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(first.read_text())["verdict"], "BLOCKED")
        self.assertEqual(os.stat(directory).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(first).st_mode & 0o777, 0o600)
        checksum = Path(f"{first}.sha256")
        self.assertTrue(checksum.is_file())
        self.assertEqual(os.stat(checksum).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
