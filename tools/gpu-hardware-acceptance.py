#!/usr/bin/env python3
"""Collect fail-closed evidence after attended RTX 5090 maintenance.

This tool cannot enable SplatLab. It verifies the physical-maintenance
attestation, platform state, PCIe link, GPU safety controls, kernel journal, and
a fixed 15-minute idle window. It writes a non-secret JSON report and leaves the
canonical hardware-maintenance marker untouched on both pass and failure.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import pwd
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TOOL_NAME = "splatlab-gpu-hardware-acceptance"
EXPECTED_BOARD = "ROG MAXIMUS Z890 HERO"
EXPECTED_BIOS = "3202"
EXPECTED_ME = "19.0.5.2175"
EXPECTED_BIOS_PACKAGE_NAME = "ROG-MAXIMUS-Z890-HERO-ASUS-3202.ZIP"
EXPECTED_BIOS_PACKAGE_SHA256 = (
    "4fb0aaa3981ba1c070398929dbdb6a678e959eee2c2b2f4eba7166927205de02"
)
EXPECTED_GPU_BDF = "02:00.0"
EXPECTED_POWER_LIMIT_W = 400.0
EXPECTED_UID = 1000
EXPECTED_GID = 1000
EXPECTED_RAPL_LONG_TERM_UW = 125_000_000
EXPECTED_RAPL_SHORT_TERM_UW = 177_000_000
EXPECTED_POWER_GUARD_UNIT = "aipc-cpu-power-guard.service"
EXPECTED_POWER_GUARD_FRAGMENT = Path("/etc/systemd/system/aipc-cpu-power-guard.service")
EXPECTED_POWER_GUARD_SCRIPT = Path("/home/rtoony/scripts/aipc-cpu-power-guard.sh")
NEXUS_HEAVY_WORK_LOCK = Path("/run/user/1000/nexus-heavy-work.lock")
OBSERVATION_SECONDS = 15 * 60
IDLE_SAMPLE_INTERVAL_SECONDS = 10.0
IDLE_SAMPLE_MAX_GAP_SECONDS = 12.0
MARKER_PATH = Path("/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf")
WATCHER_STATUS_PATH = Path(
    "/home/rtoony/.local/state/nexus-watchers/gpu_health_watch_status.json"
)
WATCHER_STATUS_SCHEMA = "nexus.gpu-health-watch.status.v1"
WATCHER_STATUS_TOOL = "nexus-gpu-health-watch"
WATCHER_STATUS_UNIT = "nexus-gpu-health-watch.service"
WATCHER_STATUS_MAX_AGE_SECONDS = 360.0
WATCHER_STATUS_MAX_RUNTIME_SECONDS = 95.0
# The watcher oneshot is not inactive/dead for its whole activation: the vault
# ExecStartPre alone is allowed 10s (observed ~3-4s total, 2026-07-18). The
# retry window (retries x 0.5s sleep) must outlast a normal activation, yet
# stay under IDLE_SAMPLE_MAX_GAP_SECONDS or a mid-observation retry would trip
# the sampling-gap check instead.
WATCHER_UNIT_RACE_RETRIES = 16
WATCHER_STATUS_MAX_BYTES = 64 * 1024
WATCHER_STATUS_UID = 1000
WATCHER_STATUS_GID = 1000
WATCHER_STATUS_KEYS = frozenset(
    {
        "schema",
        "tool",
        "unit",
        "boot_id",
        "invocation_id",
        "started_at_epoch",
        "finished_at_epoch",
        "started_at_monotonic_ns",
        "finished_at_monotonic_ns",
        "journal_ok",
        "previous_journal_ok",
        "probe_counts",
        "fault_counts",
        "interlock_status",
        "run_success",
        "last_error",
    }
)
WATCHER_PROBE_KEYS = frozenset(
    {
        "gpu_attempted",
        "gpu_ok",
        "kernel_journal_attempted",
        "kernel_journal_ok",
        "previous_journal_attempted",
        "previous_journal_ok",
    }
)
WATCHER_FAULT_KEYS = frozenset(
    {
        "gpu_unreadable",
        "xid",
        "aer_current",
        "aer_previous",
        "aer_severe",
        "platform_fatal",
    }
)
WATCHER_INTERLOCK_STATES = frozenset(
    {
        "not-evaluated",
        "not-required",
        "asserted",
        "already-active",
        "asserted-incomplete",
        "failed",
    }
)
WATCHER_ERROR_STATES = frozenset(
    {
        "run_incomplete",
        "gpu_unreadable",
        "kernel_journal_unreadable",
        "maintenance_interlock_failed",
        "private_state_failure",
        "unexpected_internal_error",
    }
)
WATCHER_ACTIVE_FAULT_KEYS = (
    "gpu_unreadable",
    "xid",
    "aer_current",
    "aer_severe",
    "platform_fatal",
)
WATCHER_INVOCATION_RE = re.compile(r"^[0-9a-f]{32}$")
BOOT_ID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
REPORT_DIR = Path("/home/rtoony/reports/splatlab-safe-evaluation-2026-07-11/acceptance")
HEALTH_URL = "http://127.0.0.1:3416/healthz"
COMPUTE_GATE = Path("/home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh")
KNOWN_COMPUTE_UNITS = (
    "splatlab-langfield.service",
    "splatlab-mesh-autoresearch.service",
    "media-batch-transcode.service",
    "media-batch-transcode-extend.service",
    "pulse-whisper.service",
    "comfyui.service",
    "sam3d-body.service",
    "sam-video-lab.service",
    "vllm-diffusiongemma.service",
)
INTERACTIVE_AI_SCOPE_RE = re.compile(r"aipc-safe-run-[A-Za-z0-9_.:@-]+\.scope")
OPERATOR_EVIDENCE_SCHEMA = "splatlab.operator-evidence.v1"
OPERATOR_EVIDENCE_TYPES = frozenset({"firmware", "physical", "memtest86"})
OPERATOR_EVIDENCE_MAX_BYTES = 256 * 1024
OPERATOR_EVIDENCE_MAX_AGE_SECONDS = 24 * 60 * 60
OPERATOR_EVIDENCE_CLOCK_SLOP_SECONDS = 5.0
OPERATOR_EVIDENCE_MTIME_SLOP_SECONDS = 5 * 60
OPERATOR_EVIDENCE_MAX_ARTIFACTS = 16
OPERATOR_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "evidence_type",
        "host",
        "boot_id",
        "recorded_at",
        "operator",
        "operator_uid",
        "assertions",
        "artifacts",
    }
)
OPERATOR_ARTIFACT_KEYS = frozenset({"filename", "sha256"})
PHYSICAL_ASSERTIONS = frozenset(
    {
        "gpu_reseated",
        "gpu_support_checked",
        "native_12v_2x6_inspected_and_reseated",
        "eps_power_reseated",
        "connectors_undamaged",
    }
)
FIRMWARE_ASSERTIONS = frozenset(
    {
        "bios_defaults_loaded",
        "memory_auto_jedec_xmp_disabled",
        "asus_ai_and_multicore_overclocking_disabled",
        "firmware_package_name",
        "firmware_package_sha256",
    }
)
MEMTEST_ASSERTIONS = frozenset({"completed", "test_mode", "passes", "errors"})
AER_DEVICE_FILES = (
    "aer_dev_correctable",
    "aer_dev_nonfatal",
    "aer_dev_fatal",
)
AER_ROOT_FILES = AER_DEVICE_FILES + (
    "aer_rootport_total_err_cor",
    "aer_rootport_total_err_nonfatal",
    "aer_rootport_total_err_fatal",
)

FAULT_PATTERNS = (
    ("pcie_aer", re.compile(r"AER: .*error|PCIe Bus Error|\bRxErr\b", re.I)),
    ("nvidia_xid", re.compile(r"NVRM:.*\bXid\b|GPU has fallen off", re.I)),
    ("machine_check", re.compile(r"Machine check|mce:.*Hardware Error", re.I)),
    ("bert", re.compile(r"\bBERT\b.*(?:error|fatal)", re.I)),
    ("cper_fatal", re.compile(r"\bCPER\b.*(?:fatal|CATERR|IERR)", re.I)),
    ("platform_fatal", re.compile(r"\b(?:CATERR|IERR)\b", re.I)),
    ("hardware_error", re.compile(r"\bHardware Error\b", re.I)),
    ("oom", re.compile(r"Out of memory|oom-kill|Killed process .* total-vm", re.I)),
    (
        "kernel_panic",
        re.compile(r"Kernel panic|BUG: soft lockup|watchdog:.*lockup", re.I),
    ),
    ("thermal", re.compile(r"thermal.*(?:critical|shutdown)", re.I)),
)


@dataclass(frozen=True)
class Paths:
    board_name: Path = Path("/sys/class/dmi/id/board_name")
    bios_version: Path = Path("/sys/class/dmi/id/bios_version")
    bios_date: Path = Path("/sys/class/dmi/id/bios_date")
    me_fw_version: Path = Path("/sys/class/mei/mei0/fw_ver")
    rapl_package: Path = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
    boot_id: Path = Path("/proc/sys/kernel/random/boot_id")
    gpu_device: Path = Path("/sys/bus/pci/devices/0000:02:00.0")
    root_port: Path = Path("/sys/bus/pci/devices/0000:00:06.0")
    marker: Path = MARKER_PATH
    watcher_status: Path = WATCHER_STATUS_PATH
    report_dir: Path = REPORT_DIR


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    source: str = "measured"


class WatcherStatusError(Exception):
    """Categorical watcher receipt failure safe to include in a report."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class SafetyError(RuntimeError):
    """Fail-closed local admission or evidence validation failure."""


class HeavyWorkBusy(SafetyError):
    """Another host-wide heavy workload owns the shared lock."""


class OperatorEvidenceError(SafetyError):
    """Operator evidence is weak, stale, public, malformed, or unbound."""


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def run_command(argv: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _lock_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
    )


@contextlib.contextmanager
def hold_heavy_work_lock(
    path: Path = NEXUS_HEAVY_WORK_LOCK,
    expected_uid: int = EXPECTED_UID,
    expected_gid: int = EXPECTED_GID,
):
    """Hold the host-wide exclusive workload lock for the complete run."""
    try:
        parent_before = os.lstat(path.parent)
    except OSError as exc:
        raise SafetyError(
            f"Nexus heavy-work lock parent is unavailable: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or (parent_before.st_uid, parent_before.st_gid) != (expected_uid, expected_gid)
        or stat.S_IMODE(parent_before.st_mode) & 0o077
    ):
        raise SafetyError("Nexus heavy-work lock parent is not private")

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SafetyError(f"cannot open Nexus heavy-work lock: {exc}") from exc
    try:
        opened = os.fstat(fd)
        named = os.lstat(path)
        parent_after = os.lstat(path.parent)
        if (
            _lock_identity(opened) != _lock_identity(named)
            or (parent_before.st_dev, parent_before.st_ino, parent_before.st_mode)
            != (parent_after.st_dev, parent_after.st_ino, parent_after.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_uid, opened.st_gid) != (expected_uid, expected_gid)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_nlink != 1
        ):
            raise SafetyError("Nexus heavy-work lock path is unsafe")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HeavyWorkBusy(
                "another Nexus heavy workload is active; acceptance was not run"
            ) from exc
        yield {
            "path": str(path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "uid": opened.st_uid,
            "gid": opened.st_gid,
            "mode": stat.S_IMODE(opened.st_mode),
            "link_count": opened.st_nlink,
            "exclusive": True,
        }
    finally:
        os.close(fd)


def read_me_versions(path: Path) -> list[str]:
    rows = read_text(path).splitlines()
    versions: list[str] = []
    for row in rows:
        match = re.fullmatch(r"\d+:(\d+\.\d+\.\d+\.\d+)", row.strip())
        if match is None:
            raise ValueError("unexpected Intel ME firmware version row")
        versions.append(match.group(1))
    if not versions:
        raise ValueError("Intel ME firmware version list is empty")
    return versions


def read_rapl_limits(root: Path) -> dict[str, int]:
    limits: dict[str, int] = {}
    try:
        name_paths = sorted(root.glob("constraint_*_name"))
        for name_path in name_paths:
            name = read_text(name_path)
            if name not in {"long_term", "short_term"}:
                continue
            if name in limits:
                raise ValueError(f"duplicate RAPL constraint: {name}")
            index = name_path.name.removeprefix("constraint_").removesuffix("_name")
            value = int(read_text(root / f"constraint_{index}_power_limit_uw"))
            if value <= 0:
                raise ValueError(f"invalid RAPL constraint value: {name}")
            limits[name] = value
    except OSError as exc:
        raise ValueError("RAPL package controls are unreadable") from exc
    if set(limits) != {"long_term", "short_term"}:
        raise ValueError("RAPL long-term or short-term constraint is missing")
    return limits


def read_power_guard_unit_state() -> dict[str, str]:
    properties = (
        "LoadState",
        "UnitFileState",
        "ActiveState",
        "SubState",
        "Result",
        "ExecMainStatus",
        "ExecStart",
        "FragmentPath",
    )
    argv = ["systemctl", "show", EXPECTED_POWER_GUARD_UNIT]
    for property_name in properties:
        argv.extend(["--property", property_name])
    result = run_command(argv, timeout=15)
    if result.returncode != 0:
        raise OSError(
            result.stderr.strip()[:200] or f"systemctl exited {result.returncode}"
        )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    if set(values) != set(properties):
        raise OSError("CPU power guard systemd state is incomplete")
    return values


def check_cpu_power_guard(paths: Paths, name: str) -> Check:
    try:
        limits = read_rapl_limits(paths.rapl_package)
        unit = read_power_guard_unit_state()
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return Check(name, False, f"CPU power guard is unreadable: {exc}")

    expected_limits = {
        "long_term": EXPECTED_RAPL_LONG_TERM_UW,
        "short_term": EXPECTED_RAPL_SHORT_TERM_UW,
    }
    expected_exec = f"{EXPECTED_POWER_GUARD_SCRIPT} apply-live --pl1 125 --pl2 177"
    unit_ok = (
        unit["LoadState"] == "loaded"
        and unit["UnitFileState"] == "enabled"
        and unit["Result"] == "success"
        and unit["ExecMainStatus"] == "0"
        and unit["FragmentPath"] == str(EXPECTED_POWER_GUARD_FRAGMENT)
        and expected_exec in unit["ExecStart"]
        and (unit["ActiveState"], unit["SubState"])
        in {("inactive", "dead"), ("active", "exited")}
    )
    passed = limits == expected_limits and unit_ok
    return Check(
        name,
        passed,
        (
            "persistent CPU guard is enabled and RAPL PL1/PL2 are 125/177 W"
            if passed
            else "CPU guard persistence or RAPL PL1/PL2 validation failed"
        ),
        {
            "limits_uw": limits,
            "expected_limits_uw": expected_limits,
            "systemd": unit,
        },
    )


def marker_snapshot(path: Path) -> dict[str, Any]:
    """Read one regular marker without following a final-component symlink."""
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise OSError("maintenance marker is not a regular file")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode):
            raise OSError("maintenance marker changed type while being read")
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise OSError("maintenance marker changed while being read")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            payload = handle.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise OSError("maintenance marker is unexpectedly large")
    finally:
        os.close(fd)

    if not payload:
        raise OSError("maintenance marker is empty")
    try:
        marker_text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise OSError("maintenance marker is not valid UTF-8") from exc
    reason_match = re.search(
        r"^SPLAT_TRAINING_DISABLED_REASON=(.+)$", marker_text, re.MULTILINE
    )
    reason = reason_match.group(1).strip().strip("\"'") if reason_match else ""
    return {
        "device": current.st_dev,
        "inode": current.st_ino,
        "uid": current.st_uid,
        "gid": current.st_gid,
        "mode": stat.S_IMODE(current.st_mode),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "reason_present": bool(reason),
    }


def check_marker(
    paths: Paths, name: str = "maintenance_marker_start"
) -> tuple[Check, dict[str, Any] | None]:
    try:
        snapshot = marker_snapshot(paths.marker)
    except OSError as exc:
        return Check(name, False, f"marker unavailable or unsafe: {exc}"), None
    safe_mode = snapshot["mode"] & 0o022 == 0
    safe_owner = snapshot["uid"] == 1000
    reason_present = snapshot["reason_present"]
    passed = safe_mode and safe_owner and reason_present
    return (
        Check(
            name,
            passed,
            (
                "marker is owned by UID 1000, has a reason, and is not group/world writable"
                if passed
                else (
                    f"unsafe marker: uid={snapshot['uid']} mode={snapshot['mode']:04o} "
                    f"reason_present={reason_present}"
                )
            ),
            {"path": str(paths.marker), **snapshot},
        ),
        snapshot,
    )


def check_attestations(args: argparse.Namespace) -> list[Check]:
    attestations = {
        "gpu_reseated": args.confirm_gpu_reseated,
        "gpu_support_checked": args.confirm_gpu_support,
        "native_12v_2x6_inspected_and_reseated": args.confirm_12v_2x6,
        "eps_power_reseated": args.confirm_eps,
        "connectors_undamaged": args.confirm_connectors_undamaged,
        "bios_defaults_loaded": args.confirm_bios_defaults,
        "memory_auto_jedec_xmp_disabled": args.confirm_jedec_memory,
        "asus_ai_and_multicore_overclocking_disabled": args.confirm_ai_overclocking_disabled,
    }
    return [
        Check(
            f"attestation_{name}",
            bool(value),
            "operator explicitly confirmed"
            if value
            else "operator confirmation missing",
            source="operator_attested",
        )
        for name, value in attestations.items()
    ]


def check_platform(paths: Paths) -> list[Check]:
    values: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name, path in (
        ("board", paths.board_name),
        ("bios", paths.bios_version),
        ("bios_date", paths.bios_date),
    ):
        try:
            values[name] = read_text(path)
        except OSError as exc:
            errors[name] = str(exc)
    try:
        me_versions = read_me_versions(paths.me_fw_version)
    except (OSError, ValueError) as exc:
        me_versions = []
        errors["intel_me"] = str(exc)

    board = values.get("board", "")
    bios = values.get("bios", "")
    return [
        Check(
            "motherboard_model",
            board == EXPECTED_BOARD,
            f"expected {EXPECTED_BOARD}; observed {board or errors.get('board', 'unreadable')}",
            {"observed": board},
        ),
        Check(
            "bios_version",
            bios == EXPECTED_BIOS,
            f"expected {EXPECTED_BIOS}; observed {bios or errors.get('bios', 'unreadable')}",
            {"observed": bios, "bios_date": values.get("bios_date", "")},
        ),
        Check(
            "intel_me_version_measured",
            EXPECTED_ME in me_versions,
            (
                f"expected Intel ME {EXPECTED_ME}; measured {me_versions}"
                if me_versions
                else f"Intel ME version unreadable: {errors.get('intel_me', 'unknown')}"
            ),
            {
                "sysfs_path": str(paths.me_fw_version),
                "observed_versions": me_versions,
                "expected_version": EXPECTED_ME,
            },
        ),
    ]


def sha256_regular_file(path: Path) -> tuple[os.stat_result, str]:
    """Hash a regular file without following a final-component symlink."""
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("evidence is not a regular file")
        if info.st_size <= 0:
            raise OSError("evidence file is empty")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    finally:
        os.close(fd)
    return info, digest.hexdigest()


def _private_file_identity(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
    )


def secure_read_private_evidence_file(
    path: Path,
    *,
    now_epoch: float,
    max_bytes: int = OPERATOR_EVIDENCE_MAX_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    """Read a fresh 0600 file from a same-user 0700 directory without links."""
    if not path.is_absolute():
        raise OperatorEvidenceError("evidence path must be absolute")
    try:
        parent_before = os.lstat(path.parent)
        before = os.lstat(path)
    except OSError as exc:
        raise OperatorEvidenceError("evidence path is unavailable") from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or (parent_before.st_uid, parent_before.st_gid) != (EXPECTED_UID, EXPECTED_GID)
        or stat.S_IMODE(parent_before.st_mode) != 0o700
    ):
        raise OperatorEvidenceError("evidence directory must be private 0700")
    if (
        not stat.S_ISREG(before.st_mode)
        or (before.st_uid, before.st_gid) != (EXPECTED_UID, EXPECTED_GID)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_nlink != 1
        or not 0 < before.st_size <= max_bytes
    ):
        raise OperatorEvidenceError(
            "evidence file must be a private, nonempty 0600 regular file"
        )
    mtime_epoch = before.st_mtime_ns / 1_000_000_000
    mtime_age = now_epoch - mtime_epoch
    if not (
        -OPERATOR_EVIDENCE_CLOCK_SLOP_SECONDS
        <= mtime_age
        <= OPERATOR_EVIDENCE_MAX_AGE_SECONDS
    ):
        raise OperatorEvidenceError("evidence file is stale or future-dated")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise OperatorEvidenceError("evidence file cannot be opened safely") from exc
    try:
        opened = os.fstat(fd)
        if _private_file_identity(before) != _private_file_identity(opened):
            raise OperatorEvidenceError("evidence file changed before read")
        payload = b""
        while len(payload) <= max_bytes:
            chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        after = os.fstat(fd)
        named_after = os.lstat(path)
        parent_after = os.lstat(path.parent)
    except OSError as exc:
        raise OperatorEvidenceError("evidence file read failed") from exc
    finally:
        os.close(fd)
    if len(payload) > max_bytes:
        raise OperatorEvidenceError("evidence file is too large")
    if (
        _private_file_identity(opened) != _private_file_identity(after)
        or _private_file_identity(opened) != _private_file_identity(named_after)
        or (parent_before.st_dev, parent_before.st_ino, parent_before.st_mode)
        != (parent_after.st_dev, parent_after.st_ino, parent_after.st_mode)
    ):
        raise OperatorEvidenceError("evidence path changed during read")
    return payload, {
        "path": str(path),
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "uid": opened.st_uid,
        "gid": opened.st_gid,
        "mode": stat.S_IMODE(opened.st_mode),
        "link_count": opened.st_nlink,
        "size_bytes": opened.st_size,
        "mtime_ns": opened.st_mtime_ns,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _reject_duplicate_evidence_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OperatorEvidenceError("operator evidence has a duplicate JSON key")
        result[key] = value
    return result


def parse_operator_recorded_at(value: object) -> float:
    if type(value) is not str:
        raise OperatorEvidenceError("operator evidence recorded_at must be text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise OperatorEvidenceError(
            "operator evidence recorded_at must be canonical UTC"
        ) from exc
    return parsed.timestamp()


def validate_operator_assertions(
    evidence_type: str,
    assertions: object,
    expected_assertions: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(assertions, dict):
        raise OperatorEvidenceError("operator assertions must be an object")
    expected_keys = {
        "physical": PHYSICAL_ASSERTIONS,
        "firmware": FIRMWARE_ASSERTIONS,
        "memtest86": MEMTEST_ASSERTIONS,
    }[evidence_type]
    if set(assertions) != expected_keys:
        raise OperatorEvidenceError("operator assertion schema does not match type")
    if any(assertions.get(key) != value for key, value in expected_assertions.items()):
        raise OperatorEvidenceError("operator assertions disagree with explicit inputs")
    if evidence_type == "physical":
        passed = all(assertions.get(key) is True for key in PHYSICAL_ASSERTIONS)
    elif evidence_type == "firmware":
        boolean_keys = FIRMWARE_ASSERTIONS - {
            "firmware_package_name",
            "firmware_package_sha256",
        }
        passed = (
            all(assertions.get(key) is True for key in boolean_keys)
            and assertions.get("firmware_package_name") == EXPECTED_BIOS_PACKAGE_NAME
            and assertions.get("firmware_package_sha256")
            == EXPECTED_BIOS_PACKAGE_SHA256
        )
    else:
        passed = (
            assertions.get("completed") is True
            and assertions.get("test_mode") == "full"
            and type(assertions.get("passes")) is int
            and assertions["passes"] >= 4
            and assertions.get("errors") == 0
        )
    if not passed:
        raise OperatorEvidenceError("operator assertions do not meet the gate")
    return assertions


def validate_operator_evidence(
    path: Path,
    *,
    evidence_type: str,
    boot_id: str,
    expected_assertions: dict[str, Any],
    now_epoch: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now_epoch is None else now_epoch
    if evidence_type not in OPERATOR_EVIDENCE_TYPES:
        raise OperatorEvidenceError("unsupported operator evidence type")
    payload_bytes, record_metadata = secure_read_private_evidence_file(
        path, now_epoch=now
    )
    try:
        payload_text = payload_bytes.decode("utf-8")
        record = json.loads(
            payload_text,
            object_pairs_hook=_reject_duplicate_evidence_keys,
        )
    except OperatorEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OperatorEvidenceError(
            "operator evidence is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(record, dict) or set(record) != OPERATOR_EVIDENCE_KEYS:
        raise OperatorEvidenceError("operator evidence has an unexpected schema")
    if record.get("schema") != OPERATOR_EVIDENCE_SCHEMA:
        raise OperatorEvidenceError("operator evidence schema version is unsupported")
    if record.get("evidence_type") != evidence_type:
        raise OperatorEvidenceError("operator evidence type does not match its role")
    if record.get("host") != platform.node() or record.get("boot_id") != boot_id:
        raise OperatorEvidenceError(
            "operator evidence is not bound to this host and boot"
        )
    if os.getuid() != EXPECTED_UID or os.getgid() != EXPECTED_GID:
        raise OperatorEvidenceError(
            "acceptance must run as the expected local operator"
        )
    try:
        expected_operator = pwd.getpwuid(EXPECTED_UID).pw_name
    except KeyError as exc:
        raise OperatorEvidenceError(
            "expected local operator account is unavailable"
        ) from exc
    if (
        record.get("operator") != expected_operator
        or record.get("operator_uid") != EXPECTED_UID
    ):
        raise OperatorEvidenceError("operator evidence identity is invalid")

    recorded_epoch = parse_operator_recorded_at(record.get("recorded_at"))
    recorded_age = now - recorded_epoch
    record_mtime = record_metadata["mtime_ns"] / 1_000_000_000
    if (
        not (
            -OPERATOR_EVIDENCE_CLOCK_SLOP_SECONDS
            <= recorded_age
            <= OPERATOR_EVIDENCE_MAX_AGE_SECONDS
        )
        or abs(record_mtime - recorded_epoch) > OPERATOR_EVIDENCE_MTIME_SLOP_SECONDS
    ):
        raise OperatorEvidenceError(
            "operator evidence timestamp is stale, future-dated, or unbound to the file"
        )
    assertions = validate_operator_assertions(
        evidence_type,
        record.get("assertions"),
        expected_assertions,
    )

    artifact_records = record.get("artifacts")
    if (
        not isinstance(artifact_records, list)
        or not 1 <= len(artifact_records) <= OPERATOR_EVIDENCE_MAX_ARTIFACTS
    ):
        raise OperatorEvidenceError("operator evidence must bind at least one artifact")
    artifact_metadata: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for artifact in artifact_records:
        if not isinstance(artifact, dict) or set(artifact) != OPERATOR_ARTIFACT_KEYS:
            raise OperatorEvidenceError("operator artifact entry is malformed")
        filename = artifact.get("filename")
        expected_sha = artifact.get("sha256")
        if (
            type(filename) is not str
            or Path(filename).name != filename
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename) is None
            or filename in seen_names
            or filename == path.name
            or type(expected_sha) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            raise OperatorEvidenceError("operator artifact identity is unsafe")
        seen_names.add(filename)
        _artifact_payload, metadata = secure_read_private_evidence_file(
            path.parent / filename,
            now_epoch=now,
            max_bytes=16 * 1024 * 1024,
        )
        if metadata["sha256"] != expected_sha:
            raise OperatorEvidenceError("operator artifact hash does not match")
        artifact_metadata.append(metadata)
    return {
        **record_metadata,
        "schema": OPERATOR_EVIDENCE_SCHEMA,
        "evidence_type": evidence_type,
        "host": record["host"],
        "boot_id": record["boot_id"],
        "recorded_at": record["recorded_at"],
        "operator": record["operator"],
        "operator_uid": record["operator_uid"],
        "assertions": assertions,
        "artifacts": artifact_metadata,
    }


def check_operator_evidence(
    name: str,
    path: Path,
    *,
    evidence_type: str,
    boot_id: str,
    expected_assertions: dict[str, Any],
) -> Check:
    try:
        evidence = validate_operator_evidence(
            path,
            evidence_type=evidence_type,
            boot_id=boot_id,
            expected_assertions=expected_assertions,
        )
    except OperatorEvidenceError as exc:
        return Check(
            name,
            False,
            str(exc),
            {"path": str(path), "evidence_type": evidence_type},
            source="structured_operator_evidence",
        )
    return Check(
        name,
        True,
        "private structured operator evidence is fresh and host/boot/artifact bound",
        evidence,
        source="structured_operator_evidence",
    )


def check_firmware_package(path: Path) -> Check:
    """Require the exact ASUS package for this standard-board BIOS release."""
    try:
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("firmware package is not a regular file or is a symlink")
        info, digest = sha256_regular_file(path)
        if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
            raise OSError("firmware package changed while being read")
    except OSError as exc:
        return Check(
            "firmware_package",
            False,
            str(exc),
            {"path": str(path)},
            source="measured_hashed_artifact",
        )

    filename_matches = path.name == EXPECTED_BIOS_PACKAGE_NAME
    digest_matches = digest == EXPECTED_BIOS_PACKAGE_SHA256
    passed = filename_matches and digest_matches
    return Check(
        "firmware_package",
        passed,
        (
            "exact standard-board ASUS BIOS package verified"
            if passed
            else (
                f"expected {EXPECTED_BIOS_PACKAGE_NAME} with SHA-256 "
                f"{EXPECTED_BIOS_PACKAGE_SHA256}; observed {path.name} with {digest}"
            )
        ),
        {
            "path": str(path.absolute()),
            "size_bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "sha256": digest,
            "expected_name": EXPECTED_BIOS_PACKAGE_NAME,
            "expected_sha256": EXPECTED_BIOS_PACKAGE_SHA256,
        },
        source="measured_hashed_artifact",
    )


def read_pcie_device(path: Path) -> dict[str, str]:
    names = (
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
        "vendor",
        "device",
    )
    return {name: read_text(path / name) for name in names}


def check_pcie(paths: Paths) -> Check:
    try:
        gpu = read_pcie_device(paths.gpu_device)
        root = read_pcie_device(paths.root_port)
    except OSError as exc:
        return Check("pcie_link", False, f"PCIe sysfs read failed: {exc}")

    conditions = (
        gpu["vendor"].lower() == "0x10de",
        gpu["current_link_width"] == "16",
        gpu["max_link_width"] == "16",
        root["vendor"].lower() == "0x8086",
        root["current_link_width"] == "16",
        root["max_link_width"] == "16",
        root["max_link_speed"].startswith("16.0 GT/s"),
    )
    detail = (
        "GPU/root link is full-width x16 and root supports Gen4"
        if all(conditions)
        else "GPU/root link width, vendor, or host capability is not the expected x16 Gen4 path"
    )
    return Check("pcie_link", all(conditions), detail, {"gpu": gpu, "root_port": root})


def parse_aer_counter_text(text: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    lines = text.splitlines()
    if len(lines) == 1 and len(lines[0].split()) == 1:
        return {"TOTAL": int(lines[0])}
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"unexpected AER counter line: {line!r}")
        counters[parts[0]] = int(parts[1])
    if not counters:
        raise ValueError("empty AER counter file")
    return counters


def aer_snapshot(paths: Paths) -> dict[str, dict[str, dict[str, int]]]:
    snapshot: dict[str, dict[str, dict[str, int]]] = {}
    for label, device, names in (
        ("gpu", paths.gpu_device, AER_DEVICE_FILES),
        ("root_port", paths.root_port, AER_ROOT_FILES),
    ):
        snapshot[label] = {}
        for name in names:
            snapshot[label][name] = parse_aer_counter_text(read_text(device / name))
    return snapshot


def aer_nonzero(snapshot: dict[str, dict[str, dict[str, int]]]) -> list[str]:
    nonzero: list[str] = []
    for device, files in snapshot.items():
        for filename, counters in files.items():
            for counter, value in counters.items():
                if value:
                    nonzero.append(f"{device}/{filename}/{counter}={value}")
    return nonzero


def check_aer_counters(
    paths: Paths,
    name: str,
    initial: dict[str, dict[str, dict[str, int]]] | None = None,
) -> tuple[Check, dict[str, dict[str, dict[str, int]]] | None]:
    try:
        snapshot = aer_snapshot(paths)
    except (OSError, ValueError) as exc:
        return Check(name, False, f"AER counters unreadable: {exc}"), None
    nonzero = aer_nonzero(snapshot)
    unchanged = initial is None or snapshot == initial
    passed = not nonzero and unchanged
    if nonzero:
        detail = f"AER counters are not zero: {', '.join(nonzero[:10])}"
    elif not unchanged:
        detail = "AER counter map changed during observation"
    else:
        detail = "all GPU and root-port AER counters are zero"
    return Check(name, passed, detail, {"snapshot": snapshot}), snapshot


def normalize_bdf(value: str) -> str:
    return value.strip().lower().removeprefix("00000000:")


def read_compute_processes() -> list[str]:
    result = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        raise OSError(
            result.stderr.strip()[:200] or f"nvidia-smi exited {result.returncode}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_compute_idle(name: str = "nvidia_compute_idle") -> Check:
    try:
        rows = read_compute_processes()
    except OSError as exc:
        return Check(name, False, f"compute process query failed: {exc}")
    return Check(
        name,
        not rows,
        "no NVIDIA compute processes"
        if not rows
        else f"found {len(rows)} compute process(es)",
        {"process_count": len(rows), "processes": rows[:20]},
    )


def check_gpu_state() -> Check:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,pci.bus_id,power.limit,persistence_mode",
            "--format=csv,noheader,nounits",
        ]
    )
    compute = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    throttle = run_command(
        [
            "nvidia-smi",
            "--query-gpu=clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_slowdown",
            "--format=csv,noheader",
        ]
    )
    detail = run_command(["nvidia-smi", "-q"])
    commands = (query, compute, throttle, detail)
    if any(item.returncode != 0 for item in commands):
        failures = [
            item.stderr.strip()[:200] for item in commands if item.returncode != 0
        ]
        return Check(
            "gpu_safety_state", False, f"nvidia-smi failed: {'; '.join(failures)}"
        )

    rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        return Check(
            "gpu_safety_state",
            False,
            f"expected exactly one NVIDIA GPU; observed {len(rows)}",
        )
    fields = [value.strip() for value in rows[0].split(",")]
    if len(fields) < 4:
        return Check("gpu_safety_state", False, "unexpected nvidia-smi CSV response")
    name, bdf, power_text, persistence = fields[:4]
    try:
        power_limit = float(power_text)
    except ValueError:
        power_limit = -1.0
    compute_rows = [
        line.strip() for line in compute.stdout.splitlines() if line.strip()
    ]
    replay_match = re.search(r"Replays Since Reset\s*:\s*(\d+)", detail.stdout)
    replay_count = int(replay_match.group(1)) if replay_match else -1
    throttle_states = [value.strip().lower() for value in throttle.stdout.split(",")]

    conditions = (
        "RTX 5090" in name,
        normalize_bdf(bdf) == EXPECTED_GPU_BDF,
        abs(power_limit - EXPECTED_POWER_LIMIT_W) <= 0.1,
        persistence.lower() == "disabled",
        not compute_rows,
        replay_count == 0,
        len(throttle_states) >= 2,
        all(value == "not active" for value in throttle_states[:2]),
    )
    evidence = {
        "name": name,
        "pci_bus_id": bdf,
        "power_limit_w": power_limit,
        "persistence_mode": persistence,
        "compute_process_count": len(compute_rows),
        "pcie_replays_since_reset": replay_count,
        "hardware_throttle_states": throttle_states[:2],
    }
    return Check(
        "gpu_safety_state",
        all(conditions),
        (
            "RTX 5090 is idle, capped at 400 W, full telemetry readable, and has no replays"
            if all(conditions)
            else "GPU identity, cap, persistence, compute-idle, replay, or throttle check failed"
        ),
        evidence,
    )


def check_compute_gate() -> Check:
    result = run_command(["bash", str(COMPUTE_GATE), "--check"])
    passed = result.returncode == 75
    return Check(
        "compute_gate_blocked",
        passed,
        f"gate exit={result.returncode}; expected maintenance exit 75",
        {"exit_code": result.returncode},
    )


def read_compute_unit_states() -> dict[str, str]:
    states: dict[str, str] = {}
    for unit in KNOWN_COMPUTE_UNITS:
        result = run_command(["systemctl", "--user", "is-active", unit])
        state = result.stdout.strip() or "unknown"
        states[unit] = state
    result = run_command(
        [
            "systemctl",
            "--user",
            "list-units",
            "--type=scope",
            "--state=active",
            "--plain",
            "--no-legend",
            "--no-pager",
            "aipc-safe-run-*.scope",
        ],
        timeout=15,
    )
    if result.returncode != 0:
        states["interactive_ai_scope_inventory"] = "unknown"
        return states
    scopes: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) < 4
            or INTERACTIVE_AI_SCOPE_RE.fullmatch(fields[0]) is None
            or fields[1] != "loaded"
            or fields[2] != "active"
        ):
            states["interactive_ai_scope_inventory"] = "unknown"
            return states
        if fields[0] in scopes:
            states["interactive_ai_scope_inventory"] = "unknown"
            return states
        scopes.add(fields[0])
    states.update({scope: "active" for scope in sorted(scopes)})
    return states


def check_compute_units(name: str = "auxiliary_compute_units") -> Check:
    states = read_compute_unit_states()
    passed = set(states) == set(KNOWN_COMPUTE_UNITS) and all(
        state == "inactive" for state in states.values()
    )
    return Check(
        name,
        passed,
        "all supervisor-denied workloads and interactive AI scopes are inactive"
        if passed
        else "a supervisor-denied workload is active, malformed, or unknown",
        {"states": states},
    )


def check_watcher_timer() -> Check:
    enabled = run_command(
        ["systemctl", "--user", "is-enabled", "nexus-gpu-health-watch.timer"]
    )
    active = run_command(
        ["systemctl", "--user", "is-active", "nexus-gpu-health-watch.timer"]
    )
    enabled_text = enabled.stdout.strip()
    active_text = active.stdout.strip()
    passed = (
        enabled.returncode == 0
        and active.returncode == 0
        and enabled_text == "enabled"
        and active_text == "active"
    )
    return Check(
        "gpu_health_watcher_timer",
        passed,
        f"enabled={enabled_text or 'unknown'} active={active_text or 'unknown'}",
    )


def _directory_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_ctime_ns,
    )


def _file_identity(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
    )


def _validate_private_status_directory(info: os.stat_result) -> None:
    if not stat.S_ISDIR(info.st_mode):
        raise WatcherStatusError("status_parent_not_directory")
    if (info.st_uid, info.st_gid) != (WATCHER_STATUS_UID, WATCHER_STATUS_GID):
        raise WatcherStatusError("status_parent_wrong_owner")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise WatcherStatusError("status_parent_wrong_mode")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WatcherStatusError("status_duplicate_json_key")
        result[key] = value
    return result


def secure_read_watcher_status(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the receipt through private no-follow parent directory handles."""
    try:
        parents = (path.parents[2], path.parents[1], path.parents[0])
    except IndexError as exc:
        raise WatcherStatusError("status_path_invalid") from exc

    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC

    directory_fds: list[int] = []
    directory_stats: list[os.stat_result] = []
    file_fd: int | None = None
    try:
        first_before = os.lstat(parents[0])
        _validate_private_status_directory(first_before)
        first_fd = os.open(parents[0], directory_flags)
        first_open = os.fstat(first_fd)
        if _directory_identity(first_before) != _directory_identity(first_open):
            raise WatcherStatusError("status_parent_race")
        _validate_private_status_directory(first_open)
        directory_fds.append(first_fd)
        directory_stats.append(first_open)

        for parent in parents[1:]:
            before = os.stat(
                parent.name,
                dir_fd=directory_fds[-1],
                follow_symlinks=False,
            )
            _validate_private_status_directory(before)
            current_fd = os.open(parent.name, directory_flags, dir_fd=directory_fds[-1])
            opened = os.fstat(current_fd)
            if _directory_identity(before) != _directory_identity(opened):
                raise WatcherStatusError("status_parent_race")
            _validate_private_status_directory(opened)
            directory_fds.append(current_fd)
            directory_stats.append(opened)

        before = os.stat(
            path.name,
            dir_fd=directory_fds[-1],
            follow_symlinks=False,
        )
        if not stat.S_ISREG(before.st_mode):
            raise WatcherStatusError("status_file_not_regular")
        if (before.st_uid, before.st_gid) != (WATCHER_STATUS_UID, WATCHER_STATUS_GID):
            raise WatcherStatusError("status_file_wrong_owner")
        if stat.S_IMODE(before.st_mode) != 0o600:
            raise WatcherStatusError("status_file_wrong_mode")
        if before.st_nlink != 1:
            raise WatcherStatusError("status_file_wrong_link_count")
        if not 0 < before.st_size <= WATCHER_STATUS_MAX_BYTES:
            raise WatcherStatusError("status_file_wrong_size")

        file_flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            file_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NONBLOCK"):
            file_flags |= os.O_NONBLOCK
        file_fd = os.open(path.name, file_flags, dir_fd=directory_fds[-1])
        opened = os.fstat(file_fd)
        if _file_identity(before) != _file_identity(opened):
            raise WatcherStatusError("status_file_race")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                file_fd, min(16 * 1024, WATCHER_STATUS_MAX_BYTES + 1 - total)
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > WATCHER_STATUS_MAX_BYTES:
                raise WatcherStatusError("status_file_wrong_size")
        payload_bytes = b"".join(chunks)
        after_open = os.fstat(file_fd)
        current = os.stat(
            path.name,
            dir_fd=directory_fds[-1],
            follow_symlinks=False,
        )
        if _file_identity(opened) != _file_identity(after_open) or _file_identity(
            opened
        ) != _file_identity(current):
            raise WatcherStatusError("status_file_race")

        for parent, opened_parent in zip(parents, directory_stats, strict=True):
            current_parent = os.lstat(parent)
            _validate_private_status_directory(current_parent)
            if _directory_identity(opened_parent) != _directory_identity(
                current_parent
            ):
                raise WatcherStatusError("status_parent_race")
    except WatcherStatusError:
        raise
    except OSError as exc:
        raise WatcherStatusError("status_file_io") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)

    if not payload_bytes:
        raise WatcherStatusError("status_file_wrong_size")
    try:
        payload_text = payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WatcherStatusError("status_invalid_utf8") from exc
    try:
        payload = json.loads(
            payload_text, object_pairs_hook=_reject_duplicate_json_keys
        )
    except WatcherStatusError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise WatcherStatusError("status_invalid_json") from exc
    if not isinstance(payload, dict):
        raise WatcherStatusError("status_root_not_object")

    metadata = {
        "path": str(path),
        "device": opened.st_dev,
        "inode": opened.st_ino,
        "uid": opened.st_uid,
        "gid": opened.st_gid,
        "mode": stat.S_IMODE(opened.st_mode),
        "link_count": opened.st_nlink,
        "size_bytes": opened.st_size,
        "mtime_ns": opened.st_mtime_ns,
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    return payload, metadata


def read_watcher_unit_state() -> dict[str, str]:
    properties = (
        "InvocationID",
        "Result",
        "ExecMainStatus",
        "ActiveState",
        "SubState",
    )
    argv = ["systemctl", "--user", "show", WATCHER_STATUS_UNIT]
    for property_name in properties:
        argv.extend(["-p", property_name])
    try:
        result = run_command(argv)
    except subprocess.TimeoutExpired as exc:
        raise WatcherStatusError("watcher_unit_query_failed") from exc
    if result.returncode != 0:
        raise WatcherStatusError("watcher_unit_query_failed")
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    if set(values) != set(properties):
        raise WatcherStatusError("watcher_unit_state_incomplete")
    return values


def _strict_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _watcher_status_structure_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != WATCHER_STATUS_KEYS:
        return ["status_top_level_keys"]

    probes = payload["probe_counts"]
    faults = payload["fault_counts"]
    if not isinstance(probes, dict) or set(probes) != WATCHER_PROBE_KEYS:
        errors.append("status_probe_keys")
    if not isinstance(faults, dict) or set(faults) != WATCHER_FAULT_KEYS:
        errors.append("status_fault_keys")
    if errors:
        return errors

    string_fields = ("schema", "tool", "unit", "boot_id", "interlock_status")
    if any(type(payload[name]) is not str for name in string_fields):
        errors.append("status_string_types")
    if type(payload["invocation_id"]) is not str:
        errors.append("status_invocation_type")
    for name in ("started_at_epoch", "finished_at_epoch"):
        if not _strict_number(payload[name]) or payload[name] < 0:
            errors.append(f"status_{name}_type")
    for name in ("started_at_monotonic_ns", "finished_at_monotonic_ns"):
        if type(payload[name]) is not int or payload[name] < 0:
            errors.append(f"status_{name}_type")
    if payload["journal_ok"] is not None and type(payload["journal_ok"]) is not bool:
        errors.append("status_journal_type")
    if (
        payload["previous_journal_ok"] is not None
        and type(payload["previous_journal_ok"]) is not bool
    ):
        errors.append("status_previous_journal_type")
    if type(payload["run_success"]) is not bool:
        errors.append("status_run_success_type")
    if payload["last_error"] is not None and type(payload["last_error"]) is not str:
        errors.append("status_last_error_type")
    if (
        isinstance(payload["last_error"], str)
        and payload["last_error"] not in WATCHER_ERROR_STATES
    ):
        errors.append("status_last_error_value")
    if isinstance(probes, dict):
        for name, value in probes.items():
            if type(value) is not int or value not in (0, 1):
                errors.append(f"status_probe_type_{name}")
    if isinstance(faults, dict):
        for name, value in faults.items():
            if type(value) is not int or value < 0:
                errors.append(f"status_fault_type_{name}")
    return errors


def validate_watcher_status(
    payload: dict[str, Any],
    metadata: dict[str, Any],
    unit_state: dict[str, str],
    boot_id: str,
    now_epoch: float,
    now_monotonic_ns: int,
) -> tuple[list[str], dict[str, Any], dict[str, Any] | None]:
    errors = _watcher_status_structure_errors(payload)
    base_evidence = {
        "file": metadata,
        "systemd": unit_state,
        "validation_errors": errors,
    }
    if errors:
        return errors, base_evidence, None

    started_epoch = float(payload["started_at_epoch"])
    finished_epoch = float(payload["finished_at_epoch"])
    started_monotonic_ns = payload["started_at_monotonic_ns"]
    finished_monotonic_ns = payload["finished_at_monotonic_ns"]
    mtime_epoch = metadata["mtime_ns"] / 1_000_000_000
    wall_age = now_epoch - finished_epoch
    monotonic_age = (now_monotonic_ns - finished_monotonic_ns) / 1_000_000_000
    mtime_age = now_epoch - mtime_epoch
    wall_duration = finished_epoch - started_epoch
    monotonic_duration = (finished_monotonic_ns - started_monotonic_ns) / 1_000_000_000

    if payload["schema"] != WATCHER_STATUS_SCHEMA:
        errors.append("status_schema")
    if payload["tool"] != WATCHER_STATUS_TOOL:
        errors.append("status_tool")
    if payload["unit"] != WATCHER_STATUS_UNIT:
        errors.append("status_unit")
    if not BOOT_ID_RE.fullmatch(payload["boot_id"]):
        errors.append("status_boot_id_format")
    if not BOOT_ID_RE.fullmatch(boot_id):
        errors.append("current_boot_id_format")
    elif payload["boot_id"] != boot_id:
        errors.append("status_boot_id_mismatch")
    if not WATCHER_INVOCATION_RE.fullmatch(payload["invocation_id"]):
        errors.append("status_invocation_format")
    elif payload["invocation_id"] != unit_state["InvocationID"]:
        errors.append("status_invocation_mismatch")

    if started_epoch > finished_epoch or started_monotonic_ns > finished_monotonic_ns:
        errors.append("status_clock_order")
    if not 0 <= wall_duration <= WATCHER_STATUS_MAX_RUNTIME_SECONDS:
        errors.append("status_wall_runtime")
    if not 0 <= monotonic_duration <= WATCHER_STATUS_MAX_RUNTIME_SECONDS:
        errors.append("status_monotonic_runtime")
    if abs(wall_duration - monotonic_duration) > 5.0:
        errors.append("status_clock_duration_mismatch")
    if not -5.0 <= wall_age <= WATCHER_STATUS_MAX_AGE_SECONDS:
        errors.append("status_wall_age")
    if not 0.0 <= monotonic_age <= WATCHER_STATUS_MAX_AGE_SECONDS:
        errors.append("status_monotonic_age")
    if not -5.0 <= mtime_age <= WATCHER_STATUS_MAX_AGE_SECONDS:
        errors.append("status_mtime_age")
    if abs(mtime_epoch - finished_epoch) > 5.0:
        errors.append("status_mtime_binding")

    if unit_state["Result"] != "success":
        errors.append("watcher_unit_result")
    if unit_state["ExecMainStatus"] != "0":
        errors.append("watcher_unit_exit_status")
    if unit_state["ActiveState"] != "inactive" or unit_state["SubState"] != "dead":
        errors.append("watcher_unit_not_idle")
    if payload["run_success"] is not True:
        errors.append("status_run_unsuccessful")
    if payload["journal_ok"] is not True:
        errors.append("status_journal_unavailable")
    if payload["last_error"] is not None:
        errors.append("status_last_error")
    if payload["interlock_status"] not in WATCHER_INTERLOCK_STATES:
        errors.append("status_interlock_unknown")
    elif payload["interlock_status"] != "already-active":
        errors.append("status_interlock_not_maintenance_locked")

    probes = payload["probe_counts"]
    for name in (
        "gpu_attempted",
        "gpu_ok",
        "kernel_journal_attempted",
        "kernel_journal_ok",
        "previous_journal_attempted",
    ):
        if probes[name] != 1:
            errors.append(f"status_probe_incomplete_{name}")
    if payload["previous_journal_ok"] is None:
        errors.append("status_previous_journal_not_recorded")
    elif payload["previous_journal_ok"] is not bool(probes["previous_journal_ok"]):
        errors.append("status_previous_journal_mismatch")
    faults = payload["fault_counts"]
    for name in WATCHER_ACTIVE_FAULT_KEYS:
        if faults[name] != 0:
            errors.append(f"status_active_fault_{name}")

    errors = list(dict.fromkeys(errors))
    snapshot = {
        "schema": payload["schema"],
        "boot_id": payload["boot_id"],
        "invocation_id": payload["invocation_id"],
        "started_at_epoch": started_epoch,
        "finished_at_epoch": finished_epoch,
        "started_at_monotonic_ns": started_monotonic_ns,
        "finished_at_monotonic_ns": finished_monotonic_ns,
        "journal_ok": payload["journal_ok"],
        "previous_journal_ok": payload["previous_journal_ok"],
        "probe_counts": probes,
        "fault_counts": faults,
        "interlock_status": payload["interlock_status"],
        "run_success": payload["run_success"],
        "last_error_is_null": payload["last_error"] is None,
        "wall_age_seconds": round(wall_age, 3),
        "monotonic_age_seconds": round(monotonic_age, 3),
        "mtime_age_seconds": round(mtime_age, 3),
        "file": metadata,
        "systemd": unit_state,
    }
    evidence = {**snapshot, "validation_errors": errors}
    return errors, evidence, snapshot if not errors else None


def check_watcher_status_receipt(
    paths: Paths,
    name: str = "gpu_health_watcher_status",
    wall_time_fn: Callable[[], float] = time.time,
    monotonic_ns_fn: Callable[[], int] = time.monotonic_ns,
    sleep_fn: Callable[[float], None] = time.sleep,
    unit_state_fn: Callable[[], dict[str, str]] = read_watcher_unit_state,
    race_retries: int = WATCHER_UNIT_RACE_RETRIES,
) -> tuple[Check, dict[str, Any] | None]:
    last_race = "status_unavailable"
    for attempt in range(race_retries + 1):
        try:
            unit_before = unit_state_fn()
        except (OSError, WatcherStatusError):
            return Check(
                name, False, "watcher status blocked: watcher_unit_query_failed"
            ), None

        if (
            unit_before.get("ActiveState") != "inactive"
            or unit_before.get("SubState") != "dead"
        ):
            last_race = "watcher_unit_running"
            if attempt < race_retries:
                sleep_fn(0.5)
                continue
            return Check(name, False, f"watcher status blocked: {last_race}"), None

        try:
            payload, metadata = secure_read_watcher_status(paths.watcher_status)
        except WatcherStatusError as exc:
            last_race = exc.code
            if (
                exc.code in {"status_file_race", "status_parent_race"}
                and attempt < race_retries
            ):
                sleep_fn(0.5)
                continue
            return Check(name, False, f"watcher status blocked: {exc.code}"), None

        try:
            unit_after = unit_state_fn()
        except (OSError, WatcherStatusError):
            return Check(
                name, False, "watcher status blocked: watcher_unit_query_failed"
            ), None
        if unit_before != unit_after:
            last_race = "watcher_unit_race"
            if attempt < race_retries:
                sleep_fn(0.5)
                continue
            return Check(name, False, f"watcher status blocked: {last_race}"), None

        try:
            boot_id = current_boot_id(paths)
        except OSError:
            return Check(
                name, False, "watcher status blocked: current_boot_id_unreadable"
            ), None
        errors, evidence, snapshot = validate_watcher_status(
            payload,
            metadata,
            unit_after,
            boot_id,
            wall_time_fn(),
            monotonic_ns_fn(),
        )
        passed = not errors and snapshot is not None
        detail = (
            "fresh watcher receipt matches this boot and latest successful invocation"
            if passed
            else f"watcher status blocked: {', '.join(errors)}"
        )
        return Check(name, passed, detail, evidence), snapshot

    return Check(name, False, f"watcher status blocked: {last_race}"), None


def check_splatlab_health(url: str = HEALTH_URL) -> Check:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.status
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return Check("splatlab_browse_health", False, f"health request failed: {exc}")
    passed = status == 200 and payload.get("ok") is True
    return Check(
        "splatlab_browse_health",
        passed,
        f"HTTP {status}, ok={payload.get('ok')!r}",
        {
            "http_status": status,
            "ok": payload.get("ok"),
            "service": payload.get("service"),
        },
    )


def classify_fault_lines(lines: list[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for line in lines:
        categories = [name for name, pattern in FAULT_PATTERNS if pattern.search(line)]
        if categories:
            matches.append(
                {"categories": ",".join(categories), "line": line.strip()[:1000]}
            )
    return matches


def check_kernel_journal(
    name: str,
    since_epoch: float | None = None,
) -> Check:
    argv = ["journalctl", "-k", "-b", "0", "--no-pager", "-o", "short-monotonic"]
    if since_epoch is not None:
        argv.append(f"--since=@{int(since_epoch)}")
    result = run_command(argv, timeout=45)
    if result.returncode != 0:
        error = result.stderr.strip()[:300] or f"journalctl exited {result.returncode}"
        return Check(name, False, f"kernel journal unreadable: {error}")
    matches = classify_fault_lines(result.stdout.splitlines())
    return Check(
        name,
        not matches,
        "no relevant kernel fault records"
        if not matches
        else f"found {len(matches)} fault record(s)",
        {"match_count": len(matches), "matches": matches[-100:]},
    )


def current_boot_id(paths: Paths) -> str:
    return read_text(paths.boot_id)


def check_boot_identity(paths: Paths) -> Check:
    try:
        boot_id = current_boot_id(paths)
    except OSError as exc:
        return Check("boot_identity", False, f"boot ID unreadable: {exc}")
    return Check(
        "boot_identity",
        bool(boot_id),
        f"acceptance is bound to boot ID {boot_id}",
        {"boot_id": boot_id},
    )


def same_marker(initial: dict[str, Any], final: dict[str, Any]) -> bool:
    keys = ("device", "inode", "uid", "gid", "mode", "size", "sha256", "reason_present")
    return all(initial.get(key) == final.get(key) for key in keys)


def observe_idle_window(
    paths: Paths,
    initial_marker: dict[str, Any],
    initial_aer: dict[str, dict[str, dict[str, int]]],
    initial_watcher_status: dict[str, Any],
    seconds: int = OBSERVATION_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    monotonic_ns_fn: Callable[[], int] = time.monotonic_ns,
    wall_time_fn: Callable[[], float] = time.time,
) -> list[Check]:
    started_wall = wall_time_fn()
    started_monotonic = monotonic_fn()
    started_monotonic_ns = monotonic_ns_fn()
    try:
        initial_boot = current_boot_id(paths)
    except OSError as exc:
        return [Check("idle_observation", False, f"boot ID unreadable: {exc}")]

    observation_error = ""
    samples: list[dict[str, Any]] = []
    watcher_invocations = {initial_watcher_status["invocation_id"]}
    post_start_watcher_invocations: set[str] = set()
    deadline = started_monotonic + seconds
    required_sample_count = math.ceil(max(0, seconds) / IDLE_SAMPLE_INTERVAL_SECONDS)
    last_sample_monotonic = started_monotonic
    next_sample_deadline = min(
        started_monotonic + IDLE_SAMPLE_INTERVAL_SECONDS,
        deadline,
    )
    while next_sample_deadline <= deadline and len(samples) < required_sample_count:
        now = monotonic_fn()
        if now < next_sample_deadline:
            sleep_fn(next_sample_deadline - now)
        sampled_monotonic = monotonic_fn()
        sample_gap = sampled_monotonic - last_sample_monotonic
        if sample_gap > IDLE_SAMPLE_MAX_GAP_SECONDS:
            observation_error = (
                f"idle sampling gap {sample_gap:.3f}s exceeded "
                f"{IDLE_SAMPLE_MAX_GAP_SECONDS:.1f}s"
            )
            break
        try:
            marker = marker_snapshot(paths.marker)
            boot_id = current_boot_id(paths)
            compute_rows = read_compute_processes()
            aer = aer_snapshot(paths)
            units = read_compute_unit_states()
            rapl_limits = read_rapl_limits(paths.rapl_package)
        except (OSError, ValueError) as exc:
            observation_error = str(exc)
            break
        watcher_check, watcher_status = check_watcher_status_receipt(
            paths,
            name="gpu_health_watcher_status_during_idle",
            wall_time_fn=wall_time_fn,
            monotonic_ns_fn=monotonic_ns_fn,
            sleep_fn=sleep_fn,
        )
        if not watcher_check.passed or watcher_status is None:
            observation_error = watcher_check.detail
            break
        watcher_invocations.add(watcher_status["invocation_id"])
        if watcher_status["finished_at_monotonic_ns"] > started_monotonic_ns:
            post_start_watcher_invocations.add(watcher_status["invocation_id"])
        sample = {
            "elapsed_seconds": round(sampled_monotonic - started_monotonic, 3),
            "sample_gap_seconds": round(sample_gap, 3),
            "boot_id": boot_id,
            "marker_unchanged": same_marker(initial_marker, marker),
            "compute_process_count": len(compute_rows),
            "aer_nonzero": aer_nonzero(aer),
            "aer_unchanged": aer == initial_aer,
            "compute_unit_states": units,
            "cpu_rapl_limits_uw": rapl_limits,
            "watcher_invocation_id": watcher_status["invocation_id"],
            "watcher_finished_at_epoch": watcher_status["finished_at_epoch"],
            "watcher_monotonic_age_seconds": watcher_status["monotonic_age_seconds"],
        }
        samples.append(sample)
        if boot_id != initial_boot:
            observation_error = "boot ID changed during observation"
        elif not sample["marker_unchanged"]:
            observation_error = "marker content or identity changed during observation"
        elif compute_rows:
            observation_error = (
                "NVIDIA compute process appeared during idle observation"
            )
        elif sample["aer_nonzero"] or not sample["aer_unchanged"]:
            observation_error = (
                "AER counters were nonzero or changed during observation"
            )
        elif any(state != "inactive" for state in units.values()):
            observation_error = "supervisor-denied workload became active or unknown"
        elif set(units) != set(KNOWN_COMPUTE_UNITS):
            observation_error = "competing workload inventory changed or is malformed"
        elif rapl_limits != {
            "long_term": EXPECTED_RAPL_LONG_TERM_UW,
            "short_term": EXPECTED_RAPL_SHORT_TERM_UW,
        }:
            observation_error = "CPU RAPL guard changed during observation"
        if observation_error:
            break
        last_sample_monotonic = sampled_monotonic
        if next_sample_deadline >= deadline:
            break
        next_sample_deadline = min(
            next_sample_deadline + IDLE_SAMPLE_INTERVAL_SECONDS,
            deadline,
        )

    elapsed = monotonic_fn() - started_monotonic
    observed_monotonic_seconds = (
        float(samples[-1]["elapsed_seconds"]) if samples else round(elapsed, 3)
    )
    sample_count_ok = len(samples) >= required_sample_count
    continuous_samples_ok = not observation_error and sample_count_ok
    checks = [
        Check(
            "idle_observation_duration",
            elapsed >= max(0, seconds - 0.5),
            f"observed {elapsed:.1f}s; required {seconds}s",
            {
                "observed_monotonic_seconds": observed_monotonic_seconds,
                "required_seconds": seconds,
            },
        ),
        Check(
            "continuous_idle_samples",
            continuous_samples_ok,
            (
                f"{len(samples)} clean sample(s); required {required_sample_count}; "
                f"maximum permitted gap {IDLE_SAMPLE_MAX_GAP_SECONDS:.1f}s"
                if continuous_samples_ok
                else observation_error
                or (
                    f"only {len(samples)} clean sample(s); "
                    f"required {required_sample_count}"
                )
            ),
            {
                "sample_count": len(samples),
                "required_sample_count": required_sample_count,
                "maximum_permitted_gap_seconds": IDLE_SAMPLE_MAX_GAP_SECONDS,
                "samples": samples,
            },
        ),
    ]
    try:
        final_boot = current_boot_id(paths)
        boot_same = initial_boot == final_boot
        boot_detail = f"boot_id={final_boot}"
    except OSError as exc:
        boot_same = False
        boot_detail = f"boot ID unreadable: {exc}"
    checks.append(Check("single_boot_observation", boot_same, boot_detail))
    checks.append(check_kernel_journal("kernel_faults_during_idle", started_wall - 1.0))
    final_watcher_check, final_watcher_status = check_watcher_status_receipt(
        paths,
        name="gpu_health_watcher_status_after_observation",
        wall_time_fn=wall_time_fn,
        monotonic_ns_fn=monotonic_ns_fn,
        sleep_fn=sleep_fn,
    )
    if (
        final_watcher_status is not None
        and final_watcher_status["finished_at_monotonic_ns"] > started_monotonic_ns
    ):
        post_start_watcher_invocations.add(final_watcher_status["invocation_id"])
        watcher_invocations.add(final_watcher_status["invocation_id"])
    receipt_continuity_passed = final_watcher_check.passed and bool(
        post_start_watcher_invocations
    )
    checks.extend(
        [
            final_watcher_check,
            Check(
                "watcher_receipt_continuity",
                receipt_continuity_passed,
                (
                    "watcher remained fresh and completed at least one invocation "
                    "after the idle observation began"
                    if receipt_continuity_passed
                    else (
                        "watcher did not provide a fresh successful post-start "
                        "receipt throughout the observation"
                    )
                ),
                {
                    "unique_invocation_ids": sorted(watcher_invocations),
                    "post_start_invocation_ids": sorted(post_start_watcher_invocations),
                },
            ),
        ]
    )
    aer_check, _ = check_aer_counters(paths, "aer_counters_after_idle", initial_aer)
    checks.extend(
        [
            aer_check,
            check_pcie(paths),
            check_gpu_state(),
            check_compute_idle("nvidia_compute_idle_after_observation"),
            check_compute_units("auxiliary_compute_units_after_observation"),
            check_cpu_power_guard(paths, "cpu_power_guard_final"),
            check_compute_gate(),
            check_watcher_timer(),
            check_splatlab_health(),
        ]
    )
    return checks


def write_report(report: dict[str, Any], directory: Path) -> Path:
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise OSError(f"unsafe report directory: {directory}")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory_mode = stat.S_IMODE(os.stat(directory, follow_symlinks=False).st_mode)
    if directory_mode & 0o077:
        raise OSError(
            f"report directory must be private (0700); observed {directory_mode:04o}"
        )
    stamp = report["started_at"].replace(":", "").replace("-", "")
    candidate = directory / f"gpu-hardware-acceptance-{stamp}.json"
    suffix = 0
    while True:
        path = (
            candidate
            if suffix == 0
            else candidate.with_name(f"{candidate.stem}-{suffix}.json")
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
            break
        except FileExistsError:
            suffix += 1

    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    checksum_path = Path(f"{path}.sha256")
    checksum_payload = (f"{hashlib.sha256(payload).hexdigest()}  {path.name}\n").encode(
        "ascii"
    )
    checksum_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        checksum_flags |= os.O_NOFOLLOW
    try:
        checksum_fd = os.open(checksum_path, checksum_flags, 0o600)
        with os.fdopen(checksum_fd, "wb") as handle:
            handle.write(checksum_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def write_operator_evidence_templates(directory: Path, paths: Paths) -> list[Path]:
    """Create deliberately incomplete private records for attended completion."""
    if not directory.is_absolute():
        raise OSError("operator evidence template directory must be absolute")
    if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
        raise OSError(f"unsafe operator evidence directory: {directory}")
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = os.lstat(directory)
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid) != (EXPECTED_UID, EXPECTED_GID)
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise OSError(
            "operator evidence directory must be owned by UID/GID 1000 at 0700"
        )
    boot_id = current_boot_id(paths)
    operator = pwd.getpwuid(EXPECTED_UID).pw_name
    common = {
        "schema": OPERATOR_EVIDENCE_SCHEMA,
        "host": platform.node(),
        "boot_id": boot_id,
        "recorded_at": "REPLACE_WITH_UTC_COMPLETION_TIME",
        "operator": operator,
        "operator_uid": EXPECTED_UID,
        "artifacts": [],
    }
    templates = {
        "firmware-evidence.template.json": {
            **common,
            "evidence_type": "firmware",
            "assertions": {
                "bios_defaults_loaded": False,
                "memory_auto_jedec_xmp_disabled": False,
                "asus_ai_and_multicore_overclocking_disabled": False,
                "firmware_package_name": EXPECTED_BIOS_PACKAGE_NAME,
                "firmware_package_sha256": EXPECTED_BIOS_PACKAGE_SHA256,
            },
        },
        "physical-evidence.template.json": {
            **common,
            "evidence_type": "physical",
            "assertions": {name: False for name in sorted(PHYSICAL_ASSERTIONS)},
        },
        "memtest86-evidence.template.json": {
            **common,
            "evidence_type": "memtest86",
            "assertions": {
                "completed": False,
                "test_mode": "REPLACE_WITH_full",
                "passes": 0,
                "errors": -1,
            },
        },
    }
    written: list[Path] = []
    for filename, payload in templates.items():
        path = directory / filename
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(
                    (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
                )
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        written.append(path)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect post-maintenance evidence for the RTX 5090. Passing does not "
            "remove the SplatLab maintenance marker."
        )
    )
    parser.add_argument(
        "--write-evidence-templates",
        type=Path,
        help="write incomplete 0600 structured templates into a private 0700 directory",
    )
    parser.add_argument(
        "--firmware-package",
        type=Path,
        help=f"exact ASUS package {EXPECTED_BIOS_PACKAGE_NAME}",
    )
    parser.add_argument(
        "--firmware-evidence",
        type=Path,
        help="private structured firmware evidence JSON",
    )
    parser.add_argument(
        "--physical-evidence",
        type=Path,
        help="private structured physical-inspection evidence JSON",
    )
    parser.add_argument(
        "--memtest-result",
        type=Path,
        help="private structured four-pass MemTest86 evidence JSON",
    )
    parser.add_argument(
        "--memtest-passes",
        type=int,
        help="completed MemTest86 pass count",
    )
    parser.add_argument("--memtest-errors", type=int, help="MemTest86 error count")
    parser.add_argument("--confirm-gpu-reseated", action="store_true")
    parser.add_argument("--confirm-gpu-support", action="store_true")
    parser.add_argument("--confirm-12v-2x6", action="store_true")
    parser.add_argument("--confirm-eps", action="store_true")
    parser.add_argument("--confirm-connectors-undamaged", action="store_true")
    parser.add_argument("--confirm-bios-defaults", action="store_true")
    parser.add_argument("--confirm-jedec-memory", action="store_true")
    parser.add_argument("--confirm-ai-overclocking-disabled", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)
    acceptance_paths = (
        args.firmware_package,
        args.firmware_evidence,
        args.physical_evidence,
        args.memtest_result,
    )
    confirmations = (
        args.confirm_gpu_reseated,
        args.confirm_gpu_support,
        args.confirm_12v_2x6,
        args.confirm_eps,
        args.confirm_connectors_undamaged,
        args.confirm_bios_defaults,
        args.confirm_jedec_memory,
        args.confirm_ai_overclocking_disabled,
    )
    if args.write_evidence_templates is not None:
        if any(value is not None for value in acceptance_paths) or any(confirmations):
            parser.error(
                "template generation cannot be combined with acceptance inputs"
            )
        if args.memtest_passes is not None or args.memtest_errors is not None:
            parser.error("template generation cannot include MemTest results")
        return args
    if any(value is None for value in acceptance_paths):
        parser.error("all firmware, physical, and MemTest evidence paths are required")
    if args.memtest_passes is None or args.memtest_errors is None:
        parser.error("MemTest pass and error counts are required")
    if not all(confirmations):
        parser.error("every attended physical and firmware confirmation is required")
    return args


def run_acceptance(args: argparse.Namespace, lock_evidence: dict[str, Any]) -> int:
    started_at = utc_now()
    paths = Paths(report_dir=args.report_dir)
    checks = [
        Check(
            "nexus_heavy_work_lock",
            True,
            "exclusive host-wide heavy-work lock is held for the complete run",
            lock_evidence,
        ),
        *check_attestations(args),
    ]
    marker_check, initial_marker = check_marker(paths)
    checks.append(marker_check)
    checks.append(check_boot_identity(paths))
    try:
        boot_id = current_boot_id(paths)
    except OSError:
        boot_id = ""
    checks.extend(check_platform(paths))
    checks.append(check_firmware_package(args.firmware_package))
    checks.append(
        check_operator_evidence(
            "firmware_evidence",
            args.firmware_evidence,
            evidence_type="firmware",
            boot_id=boot_id,
            expected_assertions={
                "bios_defaults_loaded": args.confirm_bios_defaults,
                "memory_auto_jedec_xmp_disabled": args.confirm_jedec_memory,
                "asus_ai_and_multicore_overclocking_disabled": (
                    args.confirm_ai_overclocking_disabled
                ),
                "firmware_package_name": EXPECTED_BIOS_PACKAGE_NAME,
                "firmware_package_sha256": EXPECTED_BIOS_PACKAGE_SHA256,
            },
        )
    )
    checks.append(
        check_operator_evidence(
            "physical_inspection_evidence",
            args.physical_evidence,
            evidence_type="physical",
            boot_id=boot_id,
            expected_assertions={
                "gpu_reseated": args.confirm_gpu_reseated,
                "gpu_support_checked": args.confirm_gpu_support,
                "native_12v_2x6_inspected_and_reseated": args.confirm_12v_2x6,
                "eps_power_reseated": args.confirm_eps,
                "connectors_undamaged": args.confirm_connectors_undamaged,
            },
        )
    )
    checks.append(
        check_operator_evidence(
            "memtest86_evidence",
            args.memtest_result,
            evidence_type="memtest86",
            boot_id=boot_id,
            expected_assertions={
                "passes": args.memtest_passes,
                "errors": args.memtest_errors,
            },
        )
    )
    aer_start_check, initial_aer = check_aer_counters(paths, "aer_counters_boot_start")
    watcher_status_check, initial_watcher_status = check_watcher_status_receipt(paths)
    checks.extend(
        [
            aer_start_check,
            check_pcie(paths),
            check_gpu_state(),
            check_compute_idle(),
            check_compute_units(),
            check_cpu_power_guard(paths, "cpu_power_guard_start"),
            check_compute_gate(),
            check_watcher_timer(),
            watcher_status_check,
            check_splatlab_health(),
            check_kernel_journal("kernel_faults_current_boot"),
        ]
    )

    if (
        initial_marker is not None
        and initial_aer is not None
        and initial_watcher_status is not None
        and all(check.passed for check in checks)
    ):
        checks.extend(
            observe_idle_window(
                paths,
                initial_marker,
                initial_aer,
                initial_watcher_status,
            )
        )
    else:
        checks.append(
            Check(
                "idle_observation_duration",
                False,
                "15-minute observation skipped because a preflight check failed",
            )
        )

    final_marker_check, final_marker = check_marker(paths, "maintenance_marker_final")
    if initial_marker is not None and final_marker is not None:
        final_marker_check.passed = final_marker_check.passed and same_marker(
            initial_marker, final_marker
        )
        final_marker_check.detail = (
            "maintenance marker remained present and unchanged; manual review is required"
            if final_marker_check.passed
            else "maintenance marker changed during acceptance"
        )
    checks.append(final_marker_check)

    verdict = (
        "PASS_PRE_FLIGHT_A" if all(check.passed for check in checks) else "BLOCKED"
    )
    report = {
        "schema_version": 2,
        "tool": TOOL_NAME,
        "host": platform.node(),
        "started_at": started_at,
        "finished_at": utc_now(),
        "verdict": verdict,
        "required_idle_observation_seconds": OBSERVATION_SECONDS,
        "maintenance_marker_path": str(paths.marker),
        "maintenance_marker_action": "retained; this tool never clears it",
        "operator_authorization_required_for_flight_a": True,
        "flight_a_was_not_run": True,
        "checks": [asdict(check) for check in checks],
    }
    try:
        report_path = write_report(report, paths.report_dir)
    except OSError as exc:
        print(f"{TOOL_NAME}: report write failed: {exc}", file=sys.stderr)
        return 2

    print(f"{TOOL_NAME}: {verdict}")
    print(f"report={report_path}")
    print(f"maintenance_marker={paths.marker} (retained)")
    return 0 if verdict == "PASS_PRE_FLIGHT_A" else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = Paths(report_dir=args.report_dir)
    if args.write_evidence_templates is not None:
        try:
            written = write_operator_evidence_templates(
                args.write_evidence_templates,
                paths,
            )
        except (OSError, KeyError) as exc:
            print(f"{TOOL_NAME}: template write failed: {exc}", file=sys.stderr)
            return 2
        for path in written:
            print(f"template={path}")
        print("templates are intentionally incomplete and cannot satisfy acceptance")
        return 0
    try:
        with hold_heavy_work_lock() as lock_evidence:
            return run_acceptance(args, lock_evidence)
    except SafetyError as exc:
        print(f"{TOOL_NAME}: blocked before acceptance: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
