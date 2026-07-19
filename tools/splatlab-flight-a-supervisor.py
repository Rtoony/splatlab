#!/usr/bin/env python3
"""Run one crash-safe, strictly bounded SplatLab Flight A through its API.

This is a transition supervisor, not a general launcher.  It accepts no payload
overrides, never takes the GPU arbiter locks itself, and never retries an
ambiguous submission.  It holds the host-wide Nexus heavy-work lock through
admission and releases it immediately before job submission — since the
07-13/14 pause-hardening the SplatLab GPU arbiter's host layer is this same
file, so holding it across the flight deadlocks the job's first GPU stage
(observed 2026-07-19, splat_da70e534a3).  The existing SplatLab API remains the sole owner of the
host and Redis GPU leases.
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
import re
import signal
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


TOOL_NAME = "splatlab-flight-a-supervisor"
STATE_SCHEMA = "splatlab.flight-a-transition.v1"
RESULT_SCHEMA = "splatlab.flight-a-result.v1"
EXPECTED_ACCEPTANCE_TOOL = "splatlab-gpu-hardware-acceptance"
EXPECTED_WATCHER_SCHEMA = "nexus.gpu-health-watch.status.v1"
EXPECTED_WATCHER_TOOL = "nexus-gpu-health-watch"
EXPECTED_UID = 1000
RECEIPT_MAX_AGE_SECONDS = 30 * 60
WATCHER_MAX_AGE_SECONDS = 6 * 60
WATCHER_MAX_RUNTIME_SECONDS = 95
MONITOR_INTERVAL_SECONDS = 2.0
# The gpu-health-watch oneshot holds "activating" ~4s per 3-min timer firing
# (vault ExecStartPre + probe), up to ~12s when the inject hits its 10s timeout.
# 24 x 0.5s polls cover that worst case; a ~3s budget loses the race and aborts
# the flight (observed 2026-07-19, same class as the acceptance-tool fix).
WATCHER_UNIT_RACE_RETRIES = 24
MAX_FLIGHT_RUNTIME_SECONDS = 2 * 60 * 60
MAX_CPU_TEMP_C = 85.0
MAX_GPU_TEMP_C = 85.0
MAX_UPS_LOAD_PERCENT = 80.0
# apcaccess can stall past its 15s timeout when apcupsd's 60s USB poll lands on
# a sharp UPS load transition (the UPS MCU answers HID reports slowly and
# apcupsd holds its status lock for the whole device poll; observed aborting
# flight attempt 9 on 2026-07-19 seconds after a ~300W load step). Telemetry
# being briefly unreadable is not a danger signal, so retry before aborting.
UPS_TELEMETRY_ATTEMPTS = 3
UPS_TELEMETRY_RETRY_DELAY_SECONDS = 2.0
MAX_SERVICE_MEMORY_BYTES = 31 * 1024**3
MAX_SERVICE_SWAP_BYTES = 0
MAX_SERVICE_TASKS = 480
MAX_FINAL_ADMISSION_AGE_SECONDS = 10.0
EXPECTED_POWER_LIMIT_W = 400.0
EXPECTED_RAPL_LONG_TERM_UW = 125_000_000
EXPECTED_RAPL_SHORT_TERM_UW = 177_000_000
EXPECTED_BOARD = "ROG MAXIMUS Z890 HERO"
EXPECTED_BIOS = "3202"
EXPECTED_ME = "19.0.5.2175"
EXPECTED_BIOS_PACKAGE_NAME = "ROG-MAXIMUS-Z890-HERO-ASUS-3202.ZIP"
EXPECTED_BIOS_PACKAGE_SHA256 = (
    "4fb0aaa3981ba1c070398929dbdb6a678e959eee2c2b2f4eba7166927205de02"
)
EXPECTED_OUTPUT_ROOT = Path("/home/rtoony/projects/splatcli/outputs/3d")
NEXUS_HEAVY_WORK_LOCK = Path("/run/user/1000/nexus-heavy-work.lock")
EXPECTED_COMPETING_WORKLOAD_UNITS = frozenset(
    {
        "splatlab-langfield.service",
        "splatlab-mesh-autoresearch.service",
        "media-batch-transcode.service",
        "media-batch-transcode-extend.service",
        "pulse-whisper.service",
        "comfyui.service",
        "sam3d-body.service",
        "sam-video-lab.service",
        "vllm-diffusiongemma.service",
    }
)
ACCEPTANCE_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "tool",
        "host",
        "started_at",
        "finished_at",
        "verdict",
        "required_idle_observation_seconds",
        "maintenance_marker_path",
        "maintenance_marker_action",
        "operator_authorization_required_for_flight_a",
        "flight_a_was_not_run",
        "checks",
    }
)
ACCEPTANCE_CHECK_KEYS = frozenset({"name", "passed", "detail", "evidence", "source"})
ACCEPTANCE_CHECK_COUNTS = Counter(
    {
        "nexus_heavy_work_lock": 1,
        "attestation_gpu_reseated": 1,
        "attestation_gpu_support_checked": 1,
        "attestation_native_12v_2x6_inspected_and_reseated": 1,
        "attestation_eps_power_reseated": 1,
        "attestation_connectors_undamaged": 1,
        "attestation_bios_defaults_loaded": 1,
        "attestation_memory_auto_jedec_xmp_disabled": 1,
        "attestation_asus_ai_and_multicore_overclocking_disabled": 1,
        "maintenance_marker_start": 1,
        "boot_identity": 1,
        "motherboard_model": 1,
        "bios_version": 1,
        "intel_me_version_measured": 1,
        "firmware_package": 1,
        "firmware_evidence": 1,
        "physical_inspection_evidence": 1,
        "memtest86_evidence": 1,
        "aer_counters_boot_start": 1,
        "pcie_link": 2,
        "gpu_safety_state": 2,
        "nvidia_compute_idle": 1,
        "auxiliary_compute_units": 1,
        "cpu_power_guard_start": 1,
        "compute_gate_blocked": 2,
        "gpu_health_watcher_timer": 2,
        "gpu_health_watcher_status": 1,
        "splatlab_browse_health": 2,
        "kernel_faults_current_boot": 1,
        "idle_observation_duration": 1,
        "continuous_idle_samples": 1,
        "single_boot_observation": 1,
        "kernel_faults_during_idle": 1,
        "gpu_health_watcher_status_after_observation": 1,
        "watcher_receipt_continuity": 1,
        "aer_counters_after_idle": 1,
        "nvidia_compute_idle_after_observation": 1,
        "auxiliary_compute_units_after_observation": 1,
        "cpu_power_guard_final": 1,
        "maintenance_marker_final": 1,
    }
)
RESULT_KEYS = frozenset(
    {
        "schema",
        "tool",
        "transition_id",
        "boot_id",
        "finished_at",
        "outcome",
        "detail",
        "job_id",
        "submit_attempted",
        "payload_sha256",
        "receipt_name",
        "receipt_sha256",
        "marker_disposition",
        "marker_relocked_before_review",
    }
)
RESULT_OUTCOMES = frozenset(
    {
        "completed",
        "failed",
        "ambiguous_submission",
        "recovered_without_resubmission",
        "recovered_before_new_authorization",
    }
)
MARKER_DISPOSITIONS = frozenset(
    {
        "authoritative_existing_marker_retained",
        "authoritative_racing_marker_retained",
        "preserved_marker_restored",
    }
)
STATE_BASE_KEYS = frozenset(
    {
        "schema",
        "tool",
        "transition_id",
        "created_at",
        "updated_at",
        "phase",
        "boot_id",
        "receipt_name",
        "receipt_sha256",
        "consumed_receipt_path",
        "marker_snapshot",
        "preserved_marker_path",
        "payload_sha256",
        "baseline_job_ids",
        "cpu_throttle_counts",
        "submit_attempted",
        "job_id",
    }
)
STATE_OPTIONAL_KEYS = frozenset(
    {
        "marker_archived_epoch",
        "monitor_started_epoch",
        "recovery_detail",
        "recovery_error",
    }
)
STATE_PHASES = frozenset(
    {
        "validated",
        "service_stopped",
        "marker_preserved",
        "marker_archived",
        "fresh_service_started",
        "submitting",
        "monitoring",
        "recovering",
        "recovery_failed",
    }
)

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
WATCHER_ACTIVE_FAULT_KEYS = (
    "gpu_unreadable",
    "xid",
    "aer_current",
    "aer_severe",
    "platform_fatal",
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

MARKER_PATH = Path("/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf")
ACCEPTANCE_DIR = Path(
    "/home/rtoony/reports/splatlab-safe-evaluation-2026-07-11/acceptance"
)
WATCHER_STATUS_PATH = Path(
    "/home/rtoony/.local/state/nexus-watchers/gpu_health_watch_status.json"
)
STATE_DIR = Path("/home/rtoony/.local/state/splatlab-flight-a")
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
SERVICE = "splatlab.service"
HEALTH_URL = "http://127.0.0.1:3416/healthz"
API_BASE = "http://127.0.0.1:3416/api/splat"

# This is the only job this tool can submit.  A caller cannot replace or extend
# any field through CLI arguments, environment variables, or an input file.
FLIGHT_A_PAYLOAD: dict[str, Any] = {
    "mode": "3d",
    "input_path": (
        "/mnt/storage/AI_Hub/datasets/splatlab/"
        "2026-07-11-transfer-archive/VID_20260514_073947_00_002.insv"
    ),
    "output_dir": "outputs/3d",
    "capture_format": "equirectangular360",
    "images_per_equirect": 8,
    "crop_bottom": 0.15,
    "num_frames_target": 90,
    "max_num_iterations": 7000,
    "insv_fov": 204.0,
    "trim_start_s": None,
    "trim_duration_s": 30.0,
    "sfm_backend": "rig",
    "language_field": False,
    "capture_mode": "standard",
    "source_type": "capture",
}

BACKUP_UNITS = (
    ("system", "restic-backup-core.service"),
    ("user", "restic-tier0-offsite.service"),
    ("user", "restic-tier0-offsite-cold.service"),
    ("user", "nexus-backup.service"),
    ("user", "backup-docker-services.service"),
    ("user", "vaultwarden-backup.service"),
    ("user", "vm300-databases-backup.service"),
)
AUXILIARY_COMPUTE_UNITS = (
    "splatlab-langfield.service",
    "splatlab-mesh-autoresearch.service",
)
COMPETING_WORKLOAD_UNITS = (
    *AUXILIARY_COMPUTE_UNITS,
    "media-batch-transcode.service",
    "media-batch-transcode-extend.service",
    "pulse-whisper.service",
    "comfyui.service",
    "sam3d-body.service",
    "sam-video-lab.service",
    "vllm-diffusiongemma.service",
)
INTERACTIVE_AI_SCOPE_RE = re.compile(r"aipc-safe-run-[A-Za-z0-9_.:@-]+\.scope")
FAULT_PATTERNS = (
    re.compile(r"AER: .*error|PCIe Bus Error|\bRxErr\b", re.I),
    re.compile(r"NVRM:.*\bXid\b|GPU has fallen off", re.I),
    re.compile(r"Machine check|mce:.*Hardware Error", re.I),
    re.compile(r"\bBERT\b.*(?:error|fatal)", re.I),
    re.compile(r"\bCPER\b.*(?:fatal|CATERR|IERR)", re.I),
    re.compile(r"\b(?:CATERR|IERR)\b", re.I),
    re.compile(r"Out of memory|oom-kill|Killed process .* total-vm", re.I),
    re.compile(r"Kernel panic|BUG: soft lockup|watchdog:.*lockup", re.I),
    re.compile(r"thermal.*(?:critical|shutdown)", re.I),
)


class SafetyError(RuntimeError):
    """A fail-closed admission or monitoring condition was observed."""


class AmbiguousSubmission(SafetyError):
    """The one allowed POST may have reached the API and must not be retried."""


class TransitionBusy(SafetyError):
    """Another process owns the durable Flight A transition lock."""


class HeavyWorkBusy(SafetyError):
    """Another process owns the host-wide Nexus heavy-work lock."""


@dataclass(frozen=True)
class Config:
    marker: Path = MARKER_PATH
    acceptance_dir: Path = ACCEPTANCE_DIR
    watcher_status: Path = WATCHER_STATUS_PATH
    state_dir: Path = STATE_DIR
    heavy_work_lock: Path = NEXUS_HEAVY_WORK_LOCK
    boot_id: Path = BOOT_ID_PATH
    service: str = SERVICE
    health_url: str = HEALTH_URL
    api_base: str = API_BASE
    expected_uid: int = EXPECTED_UID
    expected_gid: int = EXPECTED_UID
    receipt_max_age_seconds: int = RECEIPT_MAX_AGE_SECONDS
    watcher_max_age_seconds: int = WATCHER_MAX_AGE_SECONDS
    monitor_interval_seconds: float = MONITOR_INTERVAL_SECONDS

    @property
    def pending(self) -> Path:
        return self.state_dir / "transition.pending"

    @property
    def lock(self) -> Path:
        return self.state_dir / "transition.lock"

    @property
    def results(self) -> Path:
        return self.state_dir / "results"

    @property
    def consumed(self) -> Path:
        return self.state_dir / "consumed-receipts"


@dataclass(frozen=True)
class FileSnapshot:
    device: int
    inode: int
    uid: int
    gid: int
    mode: int
    size: int
    sha256: str


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_utc(value: object) -> float:
    if not isinstance(value, str) or not value:
        raise SafetyError("receipt timestamp is missing")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise SafetyError(f"invalid receipt timestamp: {value!r}") from exc


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_private_directory(path: Path, uid: int) -> None:
    existed = os.path.lexists(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
        raise SafetyError(f"unsafe state directory: {path}")
    os.chmod(path, 0o700)
    if not existed:
        fsync_directory(path)
        fsync_directory(path.parent)


def read_regular_file(
    path: Path,
    *,
    uid: int,
    max_bytes: int = 4 * 1024 * 1024,
    require_private: bool = False,
) -> tuple[FileSnapshot, bytes]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise SafetyError(f"cannot inspect {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise SafetyError(f"not a regular file: {path}")
    if before.st_uid != uid:
        raise SafetyError(f"unexpected owner for {path}: UID {before.st_uid}")
    mode = stat.S_IMODE(before.st_mode)
    if mode & 0o022:
        raise SafetyError(f"group/world-writable file rejected: {path} ({mode:04o})")
    if require_private and mode & 0o077:
        raise SafetyError(f"private file must be 0600: {path} ({mode:04o})")

    def identity(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    flags = os.O_RDONLY | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SafetyError(f"cannot safely open {path}: {exc}") from exc
    try:
        current = os.fstat(fd)
        if not stat.S_ISREG(current.st_mode):
            raise SafetyError(f"file changed type while opening: {path}")
        if identity(before) != identity(current):
            raise SafetyError(f"file changed while opening: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        after = os.fstat(fd)
        named = os.lstat(path)
        if identity(current) != identity(after) or identity(current) != identity(named):
            raise SafetyError(f"file changed while reading: {path}")
    except SafetyError:
        raise
    except OSError as exc:
        raise SafetyError(f"cannot safely read {path}: {exc}") from exc
    finally:
        os.close(fd)
    if not payload or len(payload) > max_bytes:
        raise SafetyError(f"empty or oversized file rejected: {path}")
    return (
        FileSnapshot(
            device=current.st_dev,
            inode=current.st_ino,
            uid=current.st_uid,
            gid=current.st_gid,
            mode=stat.S_IMODE(current.st_mode),
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        payload,
    )


def _read_private_directory_child(
    directory: Path,
    name: str,
    *,
    uid: int,
    gid: int,
    max_bytes: int,
) -> tuple[FileSnapshot, bytes]:
    """Read one 0600 child through a stable, private no-follow directory FD."""
    if Path(name).name != name:
        raise SafetyError("private artifact name is not a basename")
    directory_before = os.lstat(directory)
    if (
        not stat.S_ISDIR(directory_before.st_mode)
        or (directory_before.st_uid, directory_before.st_gid) != (uid, gid)
        or stat.S_IMODE(directory_before.st_mode) != 0o700
    ):
        raise SafetyError(f"private artifact directory is unsafe: {directory}")
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(directory, directory_flags)
    file_fd: int | None = None
    try:
        directory_open = os.fstat(directory_fd)
        directory_identity = (
            directory_open.st_dev,
            directory_open.st_ino,
            directory_open.st_mode,
            directory_open.st_uid,
            directory_open.st_gid,
            directory_open.st_ctime_ns,
        )
        if directory_identity != (
            directory_before.st_dev,
            directory_before.st_ino,
            directory_before.st_mode,
            directory_before.st_uid,
            directory_before.st_gid,
            directory_before.st_ctime_ns,
        ):
            raise SafetyError("private artifact directory changed while opening")
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_uid, before.st_gid) != (uid, gid)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or not 0 < before.st_size <= max_bytes
        ):
            raise SafetyError(f"private artifact metadata is unsafe: {name}")
        file_flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        file_fd = os.open(name, file_flags, dir_fd=directory_fd)
        opened = os.fstat(file_fd)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity != (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise SafetyError("private artifact changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise SafetyError("private artifact is oversized")
        payload = b"".join(chunks)
        after = os.fstat(file_fd)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        directory_current = os.lstat(directory)
        if (
            not payload
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_gid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or identity
            != (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_gid,
                current.st_nlink,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            or directory_identity
            != (
                directory_current.st_dev,
                directory_current.st_ino,
                directory_current.st_mode,
                directory_current.st_uid,
                directory_current.st_gid,
                directory_current.st_ctime_ns,
            )
        ):
            raise SafetyError("private artifact or directory changed while reading")
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)
    return (
        FileSnapshot(
            device=opened.st_dev,
            inode=opened.st_ino,
            uid=opened.st_uid,
            gid=opened.st_gid,
            mode=stat.S_IMODE(opened.st_mode),
            size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        payload,
    )


def read_private_directory_child(
    directory: Path,
    name: str,
    *,
    uid: int,
    gid: int,
    max_bytes: int,
) -> tuple[FileSnapshot, bytes]:
    """Convert filesystem races and access failures into fail-closed errors."""
    try:
        return _read_private_directory_child(
            directory,
            name,
            uid=uid,
            gid=gid,
            max_bytes=max_bytes,
        )
    except SafetyError:
        raise
    except OSError as exc:
        raise SafetyError(f"cannot safely read private artifact {name}: {exc}") from exc


def atomic_write_json(path: Path, value: dict[str, Any], uid: int) -> None:
    ensure_private_directory(path.parent, uid)
    with contextlib.suppress(FileNotFoundError):
        existing = os.lstat(path)
        if not stat.S_ISREG(existing.st_mode) or existing.st_uid != uid:
            raise SafetyError(f"unsafe state target: {path}")
    temp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def create_exclusive_json(path: Path, value: dict[str, Any], uid: int) -> None:
    """Durably create one ledger record without replacing any prior record."""
    ensure_private_directory(path.parent, uid)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise SafetyError(
            "this acceptance receipt already authorized a submission"
        ) from exc
    try:
        os.fchmod(fd, 0o600)
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        # A partially created ledger record remains consumed fail-closed.
        raise
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def publish_exclusive_json(path: Path, value: dict[str, Any], uid: int) -> None:
    """Atomically publish one durable JSON object without replacing a target."""
    ensure_private_directory(path.parent, uid)
    temp = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temp, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path, follow_symlinks=False)
        fsync_directory(path.parent)
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()
        fsync_directory(path.parent)


def unlink_and_sync(path: Path) -> None:
    path.unlink()
    fsync_directory(path.parent)


def decode_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise SafetyError(f"{label} must be a JSON object")
    return value


def decode_json_strict(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SafetyError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except SafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SafetyError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise SafetyError(f"{label} must be a JSON object")
    return value


def secure_read_watcher_status(
    path: Path, *, uid: int, gid: int
) -> tuple[dict[str, Any], os.stat_result]:
    """Read through three private no-follow parent directory handles."""
    try:
        parents = (path.parents[2], path.parents[1], path.parents[0])
    except IndexError as exc:
        raise SafetyError("GPU watcher status path is invalid") from exc

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

    def validate_directory(info: os.stat_result) -> None:
        if (
            not stat.S_ISDIR(info.st_mode)
            or (info.st_uid, info.st_gid) != (uid, gid)
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise SafetyError("GPU watcher status parent is not private")

    def directory_identity(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_ctime_ns,
        )

    def file_identity(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_gid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    try:
        first_before = os.lstat(parents[0])
        validate_directory(first_before)
        first_fd = os.open(parents[0], directory_flags)
        first_open = os.fstat(first_fd)
        if directory_identity(first_before) != directory_identity(first_open):
            raise SafetyError("GPU watcher status parent changed")
        validate_directory(first_open)
        directory_fds.append(first_fd)
        directory_stats.append(first_open)

        for parent in parents[1:]:
            before = os.stat(
                parent.name, dir_fd=directory_fds[-1], follow_symlinks=False
            )
            validate_directory(before)
            current_fd = os.open(parent.name, directory_flags, dir_fd=directory_fds[-1])
            opened = os.fstat(current_fd)
            if directory_identity(before) != directory_identity(opened):
                raise SafetyError("GPU watcher status parent changed")
            validate_directory(opened)
            directory_fds.append(current_fd)
            directory_stats.append(opened)

        before = os.stat(path.name, dir_fd=directory_fds[-1], follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or (before.st_uid, before.st_gid) != (uid, gid)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or not 0 < before.st_size <= 64 * 1024
        ):
            raise SafetyError("GPU watcher status file metadata is unsafe")
        file_flags = os.O_RDONLY | os.O_NONBLOCK
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            file_flags |= os.O_CLOEXEC
        file_fd = os.open(path.name, file_flags, dir_fd=directory_fds[-1])
        opened = os.fstat(file_fd)
        if file_identity(before) != file_identity(opened):
            raise SafetyError("GPU watcher status changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, min(16 * 1024, 64 * 1024 + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 64 * 1024:
                raise SafetyError("GPU watcher status is oversized")
        payload = b"".join(chunks)
        after = os.fstat(file_fd)
        current = os.stat(path.name, dir_fd=directory_fds[-1], follow_symlinks=False)
        if (
            not payload
            or len(payload) > 64 * 1024
            or file_identity(opened) != file_identity(after)
            or file_identity(opened) != file_identity(current)
        ):
            raise SafetyError("GPU watcher status changed while reading")
        for parent, opened_parent in zip(parents, directory_stats, strict=True):
            if directory_identity(opened_parent) != directory_identity(
                os.lstat(parent)
            ):
                raise SafetyError("GPU watcher status parent changed while reading")
    except SafetyError:
        raise
    except OSError as exc:
        raise SafetyError(f"GPU watcher status cannot be read safely: {exc}") from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        for directory_fd in reversed(directory_fds):
            os.close(directory_fd)
    return decode_json_strict(payload, "GPU watcher status"), opened


def marker_snapshot(path: Path, uid: int) -> FileSnapshot:
    snapshot, payload = read_regular_file(path, uid=uid, max_bytes=1024 * 1024)
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise SafetyError("maintenance marker is not UTF-8") from exc
    match = re.search(r"^SPLAT_TRAINING_DISABLED_REASON=(.+)$", text, re.MULTILINE)
    if not match or not match.group(1).strip().strip("\"'"):
        raise SafetyError("maintenance marker has no disable reason")
    return snapshot


def same_file_snapshot(left: FileSnapshot, right: FileSnapshot) -> bool:
    return left == right


def payload_sha256() -> str:
    encoded = json.dumps(
        FLIGHT_A_PAYLOAD, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class HostOps:
    """All process/service/network interactions, kept replaceable for tests."""

    def __init__(self, config: Config):
        self.config = config

    def now(self) -> float:
        return time.time()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def run(
        self, argv: list[str], *, timeout: int = 45
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SafetyError(f"command failed: {argv[0]}: {exc}") from exc

    def unit_state(self, scope: str, unit: str) -> str:
        argv = ["systemctl"]
        if scope == "user":
            argv.append("--user")
        argv.extend(["is-active", unit])
        result = self.run(argv, timeout=15)
        return result.stdout.strip() or "unknown"

    def backups_idle(self) -> tuple[bool, dict[str, str]]:
        states = {unit: self.unit_state(scope, unit) for scope, unit in BACKUP_UNITS}
        return all(state in {"inactive", "failed"} for state in states.values()), states

    def auxiliary_units_idle(self) -> tuple[bool, dict[str, str]]:
        states = {
            unit: self.unit_state("user", unit) for unit in AUXILIARY_COMPUTE_UNITS
        }
        return all(state in {"inactive", "failed"} for state in states.values()), states

    def active_interactive_ai_scopes(self) -> list[str]:
        result = self.run(
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
            raise SafetyError("cannot inspect interactive AI workload scopes")
        scopes: list[str] = []
        for line in result.stdout.splitlines():
            fields = line.split()
            if (
                len(fields) < 4
                or INTERACTIVE_AI_SCOPE_RE.fullmatch(fields[0]) is None
                or fields[1] != "loaded"
                or fields[2] != "active"
            ):
                raise SafetyError("interactive AI workload scope state is malformed")
            scopes.append(fields[0])
        if len(scopes) != len(set(scopes)):
            raise SafetyError("interactive AI workload scope inventory is duplicated")
        return sorted(scopes)

    def competing_workloads_idle(self) -> tuple[bool, dict[str, str]]:
        states = {
            unit: self.unit_state("user", unit) for unit in COMPETING_WORKLOAD_UNITS
        }
        scopes = self.active_interactive_ai_scopes()
        states.update({scope: "active" for scope in scopes})
        idle = not scopes and all(
            state in {"inactive", "failed"}
            for unit, state in states.items()
            if unit in COMPETING_WORKLOAD_UNITS
        )
        return idle, states

    def watcher_unit_state(self) -> dict[str, str]:
        properties = (
            "InvocationID",
            "Result",
            "ExecMainStatus",
            "ActiveState",
            "SubState",
        )
        argv = ["systemctl", "--user", "show", "nexus-gpu-health-watch.service"]
        for name in properties:
            argv.extend(["--property", name])
        result = self.run(argv, timeout=15)
        if result.returncode != 0:
            raise SafetyError("cannot inspect the latest GPU watcher invocation")
        values: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        if set(values) != set(properties):
            raise SafetyError("GPU watcher unit state is incomplete")
        return values

    def service_active(self) -> bool:
        return self.unit_state("user", self.config.service) == "active"

    def _control_group(self) -> str:
        result = self.run(
            [
                "systemctl",
                "--user",
                "show",
                self.config.service,
                "--property=ControlGroup",
                "--value",
            ],
            timeout=15,
        )
        if result.returncode != 0:
            raise SafetyError("cannot inspect SplatLab control group")
        return result.stdout.strip()

    def _cgroup_root(self) -> Path | None:
        group = self._control_group()
        if not group:
            return None
        return Path("/sys/fs/cgroup") / group.lstrip("/")

    def cgroup_empty(self) -> bool:
        return not self.cgroup_pids()

    def cgroup_pids(self) -> set[int]:
        root = self._cgroup_root()
        if root is None or not root.exists():
            return set()
        pids: set[int] = set()
        try:
            for procs in root.rglob("cgroup.procs"):
                for value in procs.read_text(encoding="ascii").split():
                    pids.add(int(value))
        except OSError as exc:
            raise SafetyError(f"cannot inspect SplatLab cgroup: {exc}") from exc
        except ValueError as exc:
            raise SafetyError("SplatLab cgroup contained a malformed PID") from exc
        return pids

    def service_resource_safety(self) -> dict[str, int]:
        root = self._cgroup_root()
        if root is None or not root.is_dir():
            raise SafetyError("SplatLab cgroup resource counters are unavailable")

        def integer(name: str) -> int:
            try:
                value = int((root / name).read_text(encoding="ascii").strip())
            except (OSError, ValueError) as exc:
                raise SafetyError(
                    f"SplatLab cgroup counter is unreadable: {name}"
                ) from exc
            if value < 0:
                raise SafetyError(f"SplatLab cgroup counter is invalid: {name}")
            return value

        memory = integer("memory.current")
        swap = integer("memory.swap.current")
        tasks = integer("pids.current")
        try:
            events = {
                key: int(value)
                for key, value in (
                    line.split(None, 1)
                    for line in (root / "memory.events")
                    .read_text(encoding="ascii")
                    .splitlines()
                )
            }
        except (OSError, ValueError) as exc:
            raise SafetyError("SplatLab memory event counters are unreadable") from exc
        if memory >= MAX_SERVICE_MEMORY_BYTES:
            raise SafetyError(f"SplatLab memory reached {memory / 1024**3:.1f} GiB")
        if swap > MAX_SERVICE_SWAP_BYTES:
            raise SafetyError(f"SplatLab began using swap ({swap} bytes)")
        if tasks >= MAX_SERVICE_TASKS:
            raise SafetyError(f"SplatLab task count reached {tasks}")
        if events.get("oom", 0) or events.get("oom_kill", 0):
            raise SafetyError("SplatLab cgroup reports an OOM event")
        return {"memory_bytes": memory, "swap_bytes": swap, "tasks": tasks}

    def cpu_package_safety(self) -> dict[str, float]:
        result = self.run(["sensors", "-j"], timeout=15)
        # lm-sensors may return nonzero for an unrelated unreadable subfeature
        # while still emitting valid coretemp JSON. The package value itself is
        # the fail-closed signal here.
        if not result.stdout.strip():
            raise SafetyError("CPU package telemetry is unreadable")
        payload = decode_json_strict(result.stdout.encode(), "sensors telemetry")
        temperatures: list[float] = []

        def walk(value: object, label: str = "") -> None:
            if not isinstance(value, dict):
                return
            normalized = label.strip().lower()
            if normalized in {"package id 0", "cpu package", "tctl"}:
                for key, reading in value.items():
                    if key.endswith("_input") and type(reading) in (int, float):
                        temperatures.append(float(reading))
                    if key.endswith("_alarm") and reading not in (0, 0.0, False):
                        raise SafetyError("CPU package thermal alarm became active")
            for key, child in value.items():
                walk(child, str(key))

        walk(payload)
        if not temperatures or any(not math.isfinite(value) for value in temperatures):
            raise SafetyError("CPU package temperature is unavailable")
        temperature = max(temperatures)
        if temperature >= MAX_CPU_TEMP_C:
            raise SafetyError(f"CPU package temperature reached {temperature:.1f} C")
        rapl = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
        limits: dict[str, int] = {}
        try:
            for name_path in rapl.glob("constraint_*_name"):
                name = name_path.read_text(encoding="ascii").strip()
                index = name_path.name.removeprefix("constraint_").removesuffix("_name")
                if name in {"long_term", "short_term"}:
                    limits[name] = int(
                        (rapl / f"constraint_{index}_power_limit_uw")
                        .read_text(encoding="ascii")
                        .strip()
                    )
        except (OSError, ValueError) as exc:
            raise SafetyError("CPU RAPL guard is unreadable") from exc
        if limits != {
            "long_term": EXPECTED_RAPL_LONG_TERM_UW,
            "short_term": EXPECTED_RAPL_SHORT_TERM_UW,
        }:
            raise SafetyError(f"CPU RAPL guard changed: {limits}")
        return {
            "temperature_c": temperature,
            "rapl_long_term_w": limits["long_term"] / 1_000_000,
            "rapl_short_term_w": limits["short_term"] / 1_000_000,
        }

    def cpu_throttle_counts(self) -> dict[str, int]:
        root = Path("/sys/devices/system/cpu")
        paths = sorted(root.glob("cpu[0-9]*/thermal_throttle/package_throttle_count"))
        if not paths:
            raise SafetyError("CPU package throttle counters are unavailable")
        counts: dict[str, int] = {}
        try:
            for path in paths:
                value = int(path.read_text(encoding="ascii").strip())
                if value < 0:
                    raise ValueError("negative throttle count")
                counts[str(path.relative_to(root))] = value
        except (OSError, ValueError) as exc:
            raise SafetyError("CPU package throttle counters are unreadable") from exc
        return counts

    def ups_safety(self) -> dict[str, Any]:
        result: subprocess.CompletedProcess[str] | None = None
        last_failure = ""
        for attempt in range(1, UPS_TELEMETRY_ATTEMPTS + 1):
            try:
                candidate = self.run(["/usr/sbin/apcaccess", "status"], timeout=15)
            except SafetyError as exc:
                last_failure = str(exc)
            else:
                if candidate.returncode == 0:
                    result = candidate
                    break
                last_failure = f"apcaccess exited {candidate.returncode}"
            if attempt < UPS_TELEMETRY_ATTEMPTS:
                self.sleep(UPS_TELEMETRY_RETRY_DELAY_SECONDS)
        if result is None:
            raise SafetyError(
                "UPS telemetry is unreadable after "
                f"{UPS_TELEMETRY_ATTEMPTS} attempts: {last_failure}"
            )
        fields: dict[str, str] = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key.strip().upper()] = value.strip()
        match = re.fullmatch(
            r"([0-9]+(?:\.[0-9]+)?)\s+Percent", fields.get("LOADPCT", "")
        )
        if match is None:
            raise SafetyError("UPS load telemetry is malformed")
        load = float(match.group(1))
        if fields.get("STATUS", "").upper() != "ONLINE":
            raise SafetyError(f"UPS is not online: {fields.get('STATUS', 'unknown')}")
        if load >= MAX_UPS_LOAD_PERCENT:
            raise SafetyError(f"UPS load reached {load:.1f} percent")
        return {"status": "ONLINE", "load_percent": load}

    def terminate_service_cgroup(self) -> None:
        # Stop admission first, then kill the entire configured control group.
        self.run(
            ["systemctl", "--user", "stop", "--no-block", self.config.service],
            timeout=15,
        )
        self.run(
            [
                "systemctl",
                "--user",
                "kill",
                "--kill-whom=all",
                "--signal=SIGKILL",
                self.config.service,
            ],
            timeout=15,
        )
        deadline = self.now() + 20
        while self.now() < deadline:
            if not self.service_active() and self.cgroup_empty():
                return
            self.sleep(0.25)
        raise SafetyError("SplatLab service cgroup did not become empty")

    def start_service(self) -> None:
        result = self.run(
            ["systemctl", "--user", "start", self.config.service], timeout=60
        )
        if result.returncode != 0:
            raise SafetyError(
                f"failed to start {self.config.service}: {result.stderr.strip()[:200]}"
            )
        deadline = self.now() + 30
        while self.now() < deadline:
            if self.service_active():
                try:
                    health = self.http_json("GET", self.config.health_url, auth=False)
                except SafetyError:
                    health = {}
                if health.get("ok") is True:
                    return
            self.sleep(0.5)
        raise SafetyError("fresh SplatLab service did not become healthy")

    def http_json(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        auth: bool = True,
        timeout: int = 10,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": f"{TOOL_NAME}/1"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, separators=(",", ":")).encode()
        if auth:
            token = os.environ.get("PORTAL_TOKEN", "")
            if not token:
                raise SafetyError("PORTAL_TOKEN is unavailable in process memory")
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read(4 * 1024 * 1024 + 1)
                if response.status < 200 or response.status >= 300:
                    raise SafetyError(f"API returned HTTP {response.status}")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise SafetyError(
                f"local API request failed: {type(exc).__name__}"
            ) from exc
        if len(payload) > 4 * 1024 * 1024:
            raise SafetyError("local API response was oversized")
        return decode_json_strict(payload, "local API response")

    def api_status(self) -> dict[str, Any]:
        return self.http_json("GET", f"{self.config.api_base}/status")

    def submit_flight_a(self) -> dict[str, Any]:
        return self.http_json(
            "POST",
            f"{self.config.api_base}/train",
            body=dict(FLIGHT_A_PAYLOAD),
            timeout=20,
        )

    def kernel_faults(self, since_epoch: float | None = None) -> list[str]:
        argv = ["journalctl", "-k", "-b", "0", "--no-pager", "-o", "short-monotonic"]
        if since_epoch is not None:
            argv.append(f"--since=@{int(since_epoch)}")
        result = self.run(argv, timeout=45)
        if result.returncode != 0:
            raise SafetyError("current-boot kernel journal is unreadable")
        return [
            line[:1000]
            for line in result.stdout.splitlines()
            if any(pattern.search(line) for pattern in FAULT_PATTERNS)
        ][-100:]

    def gpu_safety(self, *, idle_required: bool) -> dict[str, Any]:
        query = self.run(
            [
                "nvidia-smi",
                "--query-gpu=name,pci.bus_id,temperature.gpu,power.draw,power.limit,"
                "clocks_throttle_reasons.hw_thermal_slowdown,"
                "clocks_throttle_reasons.hw_slowdown,"
                "clocks_throttle_reasons.hw_power_brake_slowdown,"
                "clocks_throttle_reasons.sw_thermal_slowdown",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        )
        if query.returncode != 0:
            raise SafetyError("NVIDIA telemetry is unreadable")
        rows = [line for line in query.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise SafetyError(f"expected one NVIDIA GPU; observed {len(rows)}")
        fields = [field.strip() for field in rows[0].split(",")]
        if len(fields) != 9:
            raise SafetyError("unexpected NVIDIA telemetry response")
        name, bdf = fields[:2]
        try:
            temperature = float(fields[2])
            power_draw = float(fields[3])
            power_limit = float(fields[4])
        except ValueError as exc:
            raise SafetyError("non-numeric NVIDIA telemetry") from exc
        throttles = [value.lower() for value in fields[5:9]]
        if "RTX 5090" not in name or not bdf.lower().endswith("02:00.0"):
            raise SafetyError("unexpected NVIDIA GPU identity")
        if temperature >= MAX_GPU_TEMP_C:
            raise SafetyError(f"GPU temperature reached {temperature:.0f} C")
        if abs(power_limit - EXPECTED_POWER_LIMIT_W) > 0.1:
            raise SafetyError(f"GPU power cap changed to {power_limit:.1f} W")
        if power_draw > EXPECTED_POWER_LIMIT_W + 10:
            raise SafetyError(
                f"GPU power draw exceeded monitor tolerance: {power_draw:.1f} W"
            )
        if any(value == "active" for value in throttles):
            raise SafetyError("GPU hardware slowdown/throttle became active")

        compute = self.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            timeout=20,
        )
        if compute.returncode != 0:
            raise SafetyError("NVIDIA compute-process query failed")
        processes = [
            line.strip() for line in compute.stdout.splitlines() if line.strip()
        ]
        if idle_required and processes:
            raise SafetyError(f"GPU is not idle ({len(processes)} compute process(es))")
        compute_pids: list[int] = []
        for row in processes:
            try:
                compute_pids.append(int(row.split(",", 1)[0].strip()))
            except ValueError as exc:
                raise SafetyError("NVIDIA compute-process PID was malformed") from exc
        return {
            "temperature_c": temperature,
            "power_draw_w": power_draw,
            "power_limit_w": power_limit,
            "compute_process_count": len(processes),
            "compute_pids": compute_pids,
        }


class SignalLatch:
    def __init__(self) -> None:
        self.received: int | None = None
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.received = signum

    def __enter__(self) -> SignalLatch:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def __exit__(self, *_args: Any) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def check(self) -> None:
        if self.received is not None:
            raise SafetyError(f"received signal {signal.Signals(self.received).name}")


class Supervisor:
    def __init__(self, config: Config | None = None, ops: HostOps | None = None):
        self.config = config or Config()
        self.ops = ops or HostOps(self.config)

    @contextlib.contextmanager
    def transition_lock(self) -> Iterator[None]:
        ensure_private_directory(self.config.state_dir, self.config.expected_uid)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.config.lock, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TransitionBusy(
                    "another Flight A transition owns the lock"
                ) from exc
            yield
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def heavy_work_lock(self) -> Iterator[None]:
        path = self.config.heavy_work_lock
        try:
            parent = os.lstat(path.parent)
        except OSError as exc:
            raise SafetyError(
                f"Nexus heavy-work lock directory is unavailable: {exc}"
            ) from exc
        if (
            not stat.S_ISDIR(parent.st_mode)
            or (parent.st_uid, parent.st_gid)
            != (self.config.expected_uid, self.config.expected_gid)
            or stat.S_IMODE(parent.st_mode) & 0o077
        ):
            raise SafetyError("Nexus heavy-work lock directory is unsafe")

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise SafetyError(f"cannot open Nexus heavy-work lock: {exc}") from exc
        self._heavy_work_fd = fd
        try:
            opened = os.fstat(fd)
            named = os.lstat(path)
            identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_uid,
                opened.st_gid,
                opened.st_nlink,
            )
            if (
                identity
                != (
                    named.st_dev,
                    named.st_ino,
                    named.st_mode,
                    named.st_uid,
                    named.st_gid,
                    named.st_nlink,
                )
                or not stat.S_ISREG(opened.st_mode)
                or (opened.st_uid, opened.st_gid)
                != (self.config.expected_uid, self.config.expected_gid)
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_nlink != 1
            ):
                raise SafetyError("Nexus heavy-work lock file is unsafe")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HeavyWorkBusy(
                    "another Nexus heavy workload is active; Flight A was not started"
                ) from exc
            yield
        finally:
            if self._heavy_work_fd == fd:
                self._heavy_work_fd = None
                os.close(fd)

    _heavy_work_fd: int | None = None

    def _release_heavy_work_lock(self) -> None:
        """Release the host heavy-work flock before job submission.

        Since the 07-13/14 pause-hardening, the SplatLab backend's GPU
        arbiter uses this same file as its host-level lock; holding it for
        the complete transition deadlocks the submitted job's first GPU
        stage (observed 2026-07-19, flight splat_da70e534a3).  The monitor
        loop still aborts on backups and competing workloads, so the
        exclusion this flock provided during admission stays enforced for
        the flight by the monitor instead.
        """
        fd, self._heavy_work_fd = self._heavy_work_fd, None
        if fd is None:
            return
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def current_boot_id(self) -> str:
        try:
            value = self.config.boot_id.read_text(encoding="ascii").strip().lower()
        except OSError as exc:
            raise SafetyError(f"boot ID is unreadable: {exc}") from exc
        if not re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            value,
        ):
            raise SafetyError("boot ID is malformed")
        return value

    def _receipt_path(self, receipt_name: str) -> Path:
        if Path(receipt_name).name != receipt_name:
            raise SafetyError("receipt must be a basename in the acceptance directory")
        if not re.fullmatch(
            r"gpu-hardware-acceptance-[A-Za-z0-9T.Z-]+\.json", receipt_name
        ):
            raise SafetyError("receipt basename is not an acceptance report")
        return self.config.acceptance_dir / receipt_name

    @staticmethod
    def _zero_aer_snapshot(value: object) -> bool:
        leaves = 0
        stack = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, dict) and current:
                stack.extend(current.values())
            elif type(current) is int:
                leaves += 1
                if current != 0:
                    return False
            else:
                return False
        return leaves > 0

    @staticmethod
    def _hashed_evidence_is_valid(evidence: object) -> bool:
        return (
            isinstance(evidence, dict)
            and type(evidence.get("size_bytes")) is int
            and evidence["size_bytes"] > 0
            and isinstance(evidence.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"]) is not None
        )

    @classmethod
    def _private_hashed_evidence_is_valid(cls, evidence: object) -> bool:
        return (
            cls._hashed_evidence_is_valid(evidence)
            and isinstance(evidence, dict)
            and evidence.get("uid") == EXPECTED_UID
            and evidence.get("gid") == EXPECTED_UID
            and evidence.get("mode") == 0o600
            and evidence.get("link_count") == 1
        )

    def _validate_acceptance_evidence(
        self,
        checks_by_name: dict[str, list[dict[str, Any]]],
        marker: FileSnapshot,
        boot_id: str,
    ) -> None:
        def one(name: str) -> dict[str, Any]:
            values = checks_by_name[name]
            if len(values) != 1:
                raise SafetyError(f"acceptance check multiplicity is invalid: {name}")
            return values[0]

        def evidence(name: str) -> dict[str, Any]:
            return one(name)["evidence"]

        lock = evidence("nexus_heavy_work_lock")
        if (
            lock.get("path") != str(self.config.heavy_work_lock)
            or lock.get("uid") != self.config.expected_uid
            or lock.get("gid") != self.config.expected_gid
            or lock.get("mode") != 0o600
            or lock.get("link_count") != 1
            or lock.get("exclusive") is not True
            or type(lock.get("device")) is not int
            or type(lock.get("inode")) is not int
        ):
            raise SafetyError("acceptance heavy-work lock evidence is invalid")

        expected_marker = asdict(marker)
        marker_evidence: list[dict[str, Any]] = []
        for name in ("maintenance_marker_start", "maintenance_marker_final"):
            observed = evidence(name)
            if (
                observed.get("path") != str(self.config.marker)
                or observed.get("reason_present") is not True
                or any(
                    observed.get(key) != value for key, value in expected_marker.items()
                )
            ):
                raise SafetyError(f"{name} evidence does not match the live marker")
            marker_evidence.append(observed)
        if marker_evidence[0] != marker_evidence[1]:
            raise SafetyError("acceptance marker evidence is not continuous")

        if evidence("boot_identity").get("boot_id") != boot_id:
            raise SafetyError("acceptance receipt belongs to another boot")
        if evidence("motherboard_model").get("observed") != EXPECTED_BOARD:
            raise SafetyError("acceptance motherboard evidence is invalid")
        if evidence("bios_version").get("observed") != EXPECTED_BIOS:
            raise SafetyError("acceptance BIOS evidence is invalid")
        me_evidence = evidence("intel_me_version_measured")
        me_versions = me_evidence.get("observed_versions")
        if (
            me_evidence.get("sysfs_path") != "/sys/class/mei/mei0/fw_ver"
            or me_evidence.get("expected_version") != EXPECTED_ME
            or not isinstance(me_versions, list)
            or EXPECTED_ME not in me_versions
            or any(
                not isinstance(version, str)
                or re.fullmatch(r"\d+\.\d+\.\d+\.\d+", version) is None
                for version in me_versions
            )
        ):
            raise SafetyError("acceptance Intel ME evidence is invalid")

        firmware = evidence("firmware_package")
        if (
            not self._hashed_evidence_is_valid(firmware)
            or firmware.get("expected_name") != EXPECTED_BIOS_PACKAGE_NAME
            or firmware.get("expected_sha256") != EXPECTED_BIOS_PACKAGE_SHA256
            or firmware.get("sha256") != EXPECTED_BIOS_PACKAGE_SHA256
        ):
            raise SafetyError("acceptance firmware package evidence is invalid")
        structured = {
            "firmware_evidence": "firmware",
            "physical_inspection_evidence": "physical",
            "memtest86_evidence": "memtest86",
        }
        for name, evidence_type in structured.items():
            if one(name).get("source") != "structured_operator_evidence":
                raise SafetyError(f"acceptance {name} source is invalid")
            observed = evidence(name)
            artifacts = observed.get("artifacts")
            if (
                not self._private_hashed_evidence_is_valid(observed)
                or observed.get("schema") != "splatlab.operator-evidence.v1"
                or observed.get("evidence_type") != evidence_type
                or observed.get("host") != platform.node()
                or observed.get("boot_id") != boot_id
                or observed.get("operator_uid") != EXPECTED_UID
                or not isinstance(observed.get("operator"), str)
                or not observed["operator"]
                or not isinstance(observed.get("recorded_at"), str)
                or not isinstance(artifacts, list)
                or not artifacts
                or any(
                    not self._private_hashed_evidence_is_valid(artifact)
                    for artifact in artifacts
                )
            ):
                raise SafetyError(f"acceptance {name} structured evidence is invalid")
            parse_utc(observed["recorded_at"])

        firmware_assertions = evidence("firmware_evidence").get("assertions")
        if (
            not isinstance(firmware_assertions, dict)
            or set(firmware_assertions)
            != {
                "bios_defaults_loaded",
                "memory_auto_jedec_xmp_disabled",
                "asus_ai_and_multicore_overclocking_disabled",
                "firmware_package_name",
                "firmware_package_sha256",
            }
            or firmware_assertions["bios_defaults_loaded"] is not True
            or firmware_assertions["memory_auto_jedec_xmp_disabled"] is not True
            or firmware_assertions["asus_ai_and_multicore_overclocking_disabled"]
            is not True
            or firmware_assertions["firmware_package_name"]
            != EXPECTED_BIOS_PACKAGE_NAME
            or firmware_assertions["firmware_package_sha256"]
            != EXPECTED_BIOS_PACKAGE_SHA256
        ):
            raise SafetyError("acceptance firmware assertions are invalid")

        physical_assertions = evidence("physical_inspection_evidence").get("assertions")
        if (
            not isinstance(physical_assertions, dict)
            or set(physical_assertions)
            != {
                "gpu_reseated",
                "gpu_support_checked",
                "native_12v_2x6_inspected_and_reseated",
                "eps_power_reseated",
                "connectors_undamaged",
            }
            or any(value is not True for value in physical_assertions.values())
        ):
            raise SafetyError("acceptance physical assertions are invalid")

        memtest = evidence("memtest86_evidence")
        memtest_assertions = memtest.get("assertions")
        if (
            not isinstance(memtest_assertions, dict)
            or set(memtest_assertions) != {"completed", "test_mode", "passes", "errors"}
            or memtest_assertions["completed"] is not True
            or memtest_assertions["test_mode"] != "full"
            or type(memtest_assertions["passes"]) is not int
            or memtest_assertions["passes"] < 4
            or memtest_assertions["errors"] != 0
        ):
            raise SafetyError("acceptance MemTest86 evidence is invalid")

        expected_rapl = {
            "long_term": EXPECTED_RAPL_LONG_TERM_UW,
            "short_term": EXPECTED_RAPL_SHORT_TERM_UW,
        }
        expected_guard_exec = (
            "/home/rtoony/scripts/aipc-cpu-power-guard.sh "
            "apply-live --pl1 125 --pl2 177"
        )
        for name in ("cpu_power_guard_start", "cpu_power_guard_final"):
            observed = evidence(name)
            systemd = observed.get("systemd")
            if (
                observed.get("limits_uw") != expected_rapl
                or observed.get("expected_limits_uw") != expected_rapl
                or not isinstance(systemd, dict)
                or systemd.get("LoadState") != "loaded"
                or systemd.get("UnitFileState") != "enabled"
                or systemd.get("Result") != "success"
                or systemd.get("ExecMainStatus") != "0"
                or systemd.get("FragmentPath")
                != "/etc/systemd/system/aipc-cpu-power-guard.service"
                or expected_guard_exec not in str(systemd.get("ExecStart", ""))
                or (systemd.get("ActiveState"), systemd.get("SubState"))
                not in {("inactive", "dead"), ("active", "exited")}
            ):
                raise SafetyError(f"acceptance {name} evidence is invalid")

        aer_start = evidence("aer_counters_boot_start").get("snapshot")
        aer_final = evidence("aer_counters_after_idle").get("snapshot")
        if (
            aer_start != aer_final
            or not self._zero_aer_snapshot(aer_start)
            or not self._zero_aer_snapshot(aer_final)
        ):
            raise SafetyError("acceptance AER evidence is nonzero or changed")

        for check in checks_by_name["pcie_link"]:
            observed = check["evidence"]
            gpu = observed.get("gpu")
            root = observed.get("root_port")
            if (
                not isinstance(gpu, dict)
                or not isinstance(root, dict)
                or str(gpu.get("vendor", "")).lower() != "0x10de"
                or gpu.get("current_link_width") != "16"
                or gpu.get("max_link_width") != "16"
                or str(root.get("vendor", "")).lower() != "0x8086"
                or root.get("current_link_width") != "16"
                or root.get("max_link_width") != "16"
                or not str(root.get("max_link_speed", "")).startswith("16.0 GT/s")
            ):
                raise SafetyError("acceptance PCIe link evidence is invalid")

        for check in checks_by_name["gpu_safety_state"]:
            observed = check["evidence"]
            throttles = observed.get("hardware_throttle_states")
            if (
                "RTX 5090" not in str(observed.get("name", ""))
                or not str(observed.get("pci_bus_id", "")).lower().endswith("02:00.0")
                or observed.get("power_limit_w") != EXPECTED_POWER_LIMIT_W
                or str(observed.get("persistence_mode", "")).lower() != "disabled"
                or observed.get("compute_process_count") != 0
                or observed.get("pcie_replays_since_reset") != 0
                or not isinstance(throttles, list)
                or throttles[:2] != ["not active", "not active"]
            ):
                raise SafetyError("acceptance GPU safety evidence is invalid")
        for name in (
            "nvidia_compute_idle",
            "nvidia_compute_idle_after_observation",
        ):
            observed = evidence(name)
            if observed.get("process_count") != 0 or observed.get("processes") != []:
                raise SafetyError(f"acceptance {name} evidence is invalid")
        for name in (
            "auxiliary_compute_units",
            "auxiliary_compute_units_after_observation",
        ):
            states = evidence(name).get("states")
            if (
                not isinstance(states, dict)
                or set(states) != EXPECTED_COMPETING_WORKLOAD_UNITS
                or any(value != "inactive" for value in states.values())
            ):
                raise SafetyError(f"acceptance {name} evidence is invalid")
        for check in checks_by_name["compute_gate_blocked"]:
            if check["evidence"].get("exit_code") != 75:
                raise SafetyError("acceptance compute-gate evidence is invalid")
        for name in ("kernel_faults_current_boot", "kernel_faults_during_idle"):
            observed = evidence(name)
            if observed.get("match_count") != 0 or observed.get("matches") != []:
                raise SafetyError(f"acceptance {name} evidence is invalid")
        for check in checks_by_name["splatlab_browse_health"]:
            if (
                check["evidence"].get("http_status") != 200
                or check["evidence"].get("ok") is not True
            ):
                raise SafetyError("acceptance browse-health evidence is invalid")

        for name in (
            "gpu_health_watcher_status",
            "gpu_health_watcher_status_after_observation",
        ):
            observed = evidence(name)
            faults = observed.get("fault_counts")
            if (
                observed.get("schema") != EXPECTED_WATCHER_SCHEMA
                or observed.get("boot_id") != boot_id
                or observed.get("interlock_status") != "already-active"
                or observed.get("journal_ok") is not True
                or observed.get("run_success") is not True
                or observed.get("last_error_is_null") is not True
                or observed.get("validation_errors") != []
                or not isinstance(faults, dict)
                or any(faults.get(key) != 0 for key in WATCHER_ACTIVE_FAULT_KEYS)
            ):
                raise SafetyError(f"acceptance {name} evidence is invalid")

        duration_evidence = evidence("idle_observation_duration")
        if set(duration_evidence) != {
            "observed_monotonic_seconds",
            "required_seconds",
        }:
            raise SafetyError("acceptance monotonic idle-duration evidence is invalid")
        observed_monotonic_seconds = duration_evidence["observed_monotonic_seconds"]
        if (
            type(observed_monotonic_seconds) not in (int, float)
            or not math.isfinite(float(observed_monotonic_seconds))
            or not 899.5 <= float(observed_monotonic_seconds) <= 912.0
            or duration_evidence["required_seconds"] != 900
        ):
            raise SafetyError("acceptance monotonic idle-duration evidence is invalid")

        samples_evidence = evidence("continuous_idle_samples")
        samples = samples_evidence.get("samples")
        if (
            samples_evidence.get("sample_count") != 90
            or samples_evidence.get("required_sample_count") != 90
            or samples_evidence.get("maximum_permitted_gap_seconds") != 12.0
            or not isinstance(samples, list)
            or len(samples) != 90
        ):
            raise SafetyError("acceptance did not contain exactly 90 idle samples")
        previous_elapsed = 0.0
        sample_keys = {
            "elapsed_seconds",
            "sample_gap_seconds",
            "boot_id",
            "marker_unchanged",
            "compute_process_count",
            "aer_nonzero",
            "aer_unchanged",
            "compute_unit_states",
            "cpu_rapl_limits_uw",
            "watcher_invocation_id",
            "watcher_finished_at_epoch",
            "watcher_monotonic_age_seconds",
        }
        for sample in samples:
            if not isinstance(sample, dict) or set(sample) != sample_keys:
                raise SafetyError("acceptance idle sample schema is invalid")
            elapsed = sample["elapsed_seconds"]
            gap = sample["sample_gap_seconds"]
            if (
                type(elapsed) not in (int, float)
                or type(gap) not in (int, float)
                or not math.isfinite(float(elapsed))
                or not math.isfinite(float(gap))
                or not 0 < float(gap) <= 12.0
                or float(elapsed) <= previous_elapsed
                or abs(float(elapsed) - previous_elapsed - float(gap)) > 0.02
                or sample["boot_id"] != boot_id
                or sample["marker_unchanged"] is not True
                or sample["compute_process_count"] != 0
                or sample["aer_nonzero"] != []
                or sample["aer_unchanged"] is not True
                or not isinstance(sample["compute_unit_states"], dict)
                or set(sample["compute_unit_states"])
                != EXPECTED_COMPETING_WORKLOAD_UNITS
                or sample["cpu_rapl_limits_uw"]
                != {
                    "long_term": EXPECTED_RAPL_LONG_TERM_UW,
                    "short_term": EXPECTED_RAPL_SHORT_TERM_UW,
                }
                or any(
                    value != "inactive"
                    for value in sample["compute_unit_states"].values()
                )
                or not isinstance(sample["watcher_invocation_id"], str)
                or re.fullmatch(r"[0-9a-f]{32}", sample["watcher_invocation_id"])
                is None
                or type(sample["watcher_monotonic_age_seconds"]) not in (int, float)
                or not 0
                <= float(sample["watcher_monotonic_age_seconds"])
                <= WATCHER_MAX_AGE_SECONDS
            ):
                raise SafetyError("acceptance idle sample evidence is invalid")
            previous_elapsed = float(elapsed)
        if not 899.5 <= previous_elapsed <= 912.0:
            raise SafetyError("acceptance idle samples do not span 15 minutes")
        if abs(float(observed_monotonic_seconds) - previous_elapsed) > 0.02:
            raise SafetyError(
                "acceptance monotonic duration disagrees with idle samples"
            )

        continuity = evidence("watcher_receipt_continuity")
        post_start = continuity.get("post_start_invocation_ids")
        unique = continuity.get("unique_invocation_ids")
        if (
            not isinstance(post_start, list)
            or not post_start
            or not isinstance(unique, list)
            or not set(post_start).issubset(set(unique))
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{32}", value) is None
                for value in [*post_start, *unique]
            )
        ):
            raise SafetyError("acceptance watcher continuity evidence is invalid")

    def validate_acceptance(
        self, receipt_name: str, marker: FileSnapshot, boot_id: str
    ) -> tuple[dict[str, Any], str]:
        path = self._receipt_path(receipt_name)
        _snapshot, payload = read_private_directory_child(
            self.config.acceptance_dir,
            path.name,
            uid=self.config.expected_uid,
            gid=self.config.expected_gid,
            max_bytes=4 * 1024 * 1024,
        )
        digest = hashlib.sha256(payload).hexdigest()
        checksum_path = Path(f"{path}.sha256")
        _checksum_snapshot, checksum_payload = read_private_directory_child(
            self.config.acceptance_dir,
            checksum_path.name,
            uid=self.config.expected_uid,
            gid=self.config.expected_gid,
            max_bytes=512,
        )
        try:
            checksum_line = checksum_payload.decode("ascii").strip()
        except UnicodeError as exc:
            raise SafetyError("acceptance checksum is not ASCII") from exc
        if checksum_line != f"{digest}  {path.name}":
            raise SafetyError("acceptance checksum does not match the receipt")

        report = decode_json_strict(payload, "acceptance receipt")
        if set(report) != ACCEPTANCE_REPORT_KEYS:
            raise SafetyError("acceptance receipt has unexpected top-level keys")
        if (
            type(report.get("schema_version")) is not int
            or report["schema_version"] != 2
        ):
            raise SafetyError("unsupported acceptance receipt schema")
        if report.get("tool") != EXPECTED_ACCEPTANCE_TOOL:
            raise SafetyError("receipt was not produced by the acceptance tool")
        if report.get("host") != platform.node():
            raise SafetyError("acceptance receipt belongs to another host")
        if report.get("verdict") != "PASS_PRE_FLIGHT_A":
            raise SafetyError("acceptance verdict is not PASS_PRE_FLIGHT_A")
        if report.get("flight_a_was_not_run") is not True:
            raise SafetyError(
                "acceptance receipt does not prove Flight A remained unrun"
            )
        if report.get("operator_authorization_required_for_flight_a") is not True:
            raise SafetyError(
                "acceptance receipt lacks the operator-authorization gate"
            )
        if report.get("required_idle_observation_seconds") != 900:
            raise SafetyError("acceptance did not use the fixed 15-minute idle window")
        if report.get("maintenance_marker_path") != str(self.config.marker):
            raise SafetyError("acceptance receipt names a different marker")
        if not str(report.get("maintenance_marker_action", "")).startswith("retained"):
            raise SafetyError("acceptance receipt did not retain the marker")

        now = self.ops.now()
        started = parse_utc(report.get("started_at"))
        finished = parse_utc(report.get("finished_at"))
        if finished - started < 899.5 or finished > now + 5:
            raise SafetyError("acceptance timestamps are inconsistent")
        age = now - finished
        if age < 0 or age > self.config.receipt_max_age_seconds:
            raise SafetyError(f"acceptance receipt is stale ({age:.0f} seconds old)")

        checks = report.get("checks")
        if not isinstance(checks, list) or not checks:
            raise SafetyError("acceptance receipt has no checks")
        allowed_sources = {
            "measured",
            "operator_attested",
            "measured_hashed_artifact",
            "structured_operator_evidence",
        }
        for check in checks:
            if (
                not isinstance(check, dict)
                or set(check) != ACCEPTANCE_CHECK_KEYS
                or type(check.get("name")) is not str
                or check.get("passed") is not True
                or type(check.get("detail")) is not str
                or not check["detail"]
                or not isinstance(check.get("evidence"), dict)
                or check.get("source") not in allowed_sources
            ):
                raise SafetyError(
                    "acceptance receipt contains a failed or malformed check"
                )
        observed_counts = Counter(check["name"] for check in checks)
        if observed_counts != ACCEPTANCE_CHECK_COUNTS:
            raise SafetyError("acceptance receipt check set does not match schema v2")
        by_name: dict[str, list[dict[str, Any]]] = {}
        for check in checks:
            by_name.setdefault(check["name"], []).append(check)
        self._validate_acceptance_evidence(by_name, marker, boot_id)
        return report, digest

    def validate_watcher_status(
        self,
        boot_id: str,
        *,
        allowed_interlocks: frozenset[str] = frozenset({"already-active"}),
    ) -> dict[str, Any]:
        for attempt in range(WATCHER_UNIT_RACE_RETRIES):
            unit_before = self.ops.watcher_unit_state()
            if (
                unit_before.get("ActiveState") != "inactive"
                or unit_before.get("SubState") != "dead"
            ):
                if attempt < WATCHER_UNIT_RACE_RETRIES - 1:
                    self.ops.sleep(0.5)
                    continue
                raise SafetyError("GPU watcher is still running; no stable receipt")
            status_payload, metadata = secure_read_watcher_status(
                self.config.watcher_status,
                uid=self.config.expected_uid,
                gid=self.config.expected_gid,
            )
            unit_state = self.ops.watcher_unit_state()
            if unit_before == unit_state:
                break
            if attempt < WATCHER_UNIT_RACE_RETRIES - 1:
                self.ops.sleep(0.5)
        else:  # pragma: no cover - loop either breaks or raises above
            raise SafetyError("GPU watcher invocation raced receipt validation")
        if unit_before != unit_state:
            raise SafetyError("GPU watcher invocation raced receipt validation")
        if set(status_payload) != WATCHER_STATUS_KEYS:
            raise SafetyError("GPU watcher status has unexpected top-level keys")
        probes = status_payload["probe_counts"]
        faults = status_payload["fault_counts"]
        if not isinstance(probes, dict) or set(probes) != WATCHER_PROBE_KEYS:
            raise SafetyError("GPU watcher probe schema is invalid")
        if not isinstance(faults, dict) or set(faults) != WATCHER_FAULT_KEYS:
            raise SafetyError("GPU watcher fault schema is invalid")

        string_fields = ("schema", "tool", "unit", "boot_id", "interlock_status")
        if any(type(status_payload[name]) is not str for name in string_fields):
            raise SafetyError("GPU watcher status has invalid string fields")
        invocation_id = status_payload["invocation_id"]
        if type(invocation_id) is not str or not re.fullmatch(
            r"[0-9a-f]{32}", invocation_id
        ):
            raise SafetyError("GPU watcher invocation ID is malformed")
        for name in ("started_at_epoch", "finished_at_epoch"):
            value = status_payload[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise SafetyError(f"GPU watcher {name} is malformed")
        for name in ("started_at_monotonic_ns", "finished_at_monotonic_ns"):
            value = status_payload[name]
            if type(value) is not int or value < 0:
                raise SafetyError(f"GPU watcher {name} is malformed")
        if type(status_payload["journal_ok"]) is not bool:
            raise SafetyError("GPU watcher current-journal state is malformed")
        if type(status_payload["previous_journal_ok"]) is not bool:
            raise SafetyError("GPU watcher previous-journal state is malformed")
        if type(status_payload["run_success"]) is not bool:
            raise SafetyError("GPU watcher success state is malformed")
        if (
            status_payload["last_error"] is not None
            and type(status_payload["last_error"]) is not str
        ):
            raise SafetyError("GPU watcher error state is malformed")
        if any(
            type(value) is not int or value not in (0, 1) for value in probes.values()
        ):
            raise SafetyError("GPU watcher probe counts are malformed")
        if any(type(value) is not int or value < 0 for value in faults.values()):
            raise SafetyError("GPU watcher fault counts are malformed")

        expected = {
            "schema": EXPECTED_WATCHER_SCHEMA,
            "tool": EXPECTED_WATCHER_TOOL,
            "unit": "nexus-gpu-health-watch.service",
            "boot_id": boot_id,
            "run_success": True,
            "journal_ok": True,
            "last_error": None,
        }
        for key, value in expected.items():
            if status_payload[key] != value:
                raise SafetyError(
                    f"GPU watcher status rejected: {key}={status_payload[key]!r}"
                )
        interlock = status_payload["interlock_status"]
        if (
            interlock not in WATCHER_INTERLOCK_STATES
            or interlock not in allowed_interlocks
        ):
            raise SafetyError(f"GPU watcher interlock state rejected: {interlock!r}")

        if unit_state != {
            "InvocationID": invocation_id,
            "Result": "success",
            "ExecMainStatus": "0",
            "ActiveState": "inactive",
            "SubState": "dead",
        }:
            raise SafetyError(
                "GPU watcher receipt is not the latest successful invocation"
            )

        started_epoch = float(status_payload["started_at_epoch"])
        finished_epoch = float(status_payload["finished_at_epoch"])
        started_monotonic = status_payload["started_at_monotonic_ns"]
        finished_monotonic = status_payload["finished_at_monotonic_ns"]
        wall_duration = finished_epoch - started_epoch
        monotonic_duration = (finished_monotonic - started_monotonic) / 1_000_000_000
        wall_age = self.ops.now() - finished_epoch
        monotonic_age = (self.ops.monotonic_ns() - finished_monotonic) / 1_000_000_000
        mtime_epoch = metadata.st_mtime_ns / 1_000_000_000
        mtime_age = self.ops.now() - mtime_epoch
        if (
            not 0 <= wall_duration <= WATCHER_MAX_RUNTIME_SECONDS
            or not 0 <= monotonic_duration <= WATCHER_MAX_RUNTIME_SECONDS
            or abs(wall_duration - monotonic_duration) > 5
            or not -5 <= wall_age <= self.config.watcher_max_age_seconds
            or not 0 <= monotonic_age <= self.config.watcher_max_age_seconds
            or not -5 <= mtime_age <= self.config.watcher_max_age_seconds
            or abs(mtime_epoch - finished_epoch) > 5
        ):
            raise SafetyError("GPU watcher receipt freshness or clock binding failed")

        for name in (
            "gpu_attempted",
            "gpu_ok",
            "kernel_journal_attempted",
            "kernel_journal_ok",
            "previous_journal_attempted",
        ):
            if probes[name] != 1:
                raise SafetyError(f"GPU watcher did not complete probe {name}")
        if status_payload["previous_journal_ok"] is not bool(
            probes["previous_journal_ok"]
        ):
            raise SafetyError("GPU watcher previous-journal fields disagree")
        for name in WATCHER_ACTIVE_FAULT_KEYS:
            if faults[name] != 0:
                raise SafetyError(f"GPU watcher reports {name}={faults[name]}")
        return status_payload

    @staticmethod
    def _validate_idle_api_status(status_payload: dict[str, Any]) -> None:
        if status_payload.get("active_jobs") != 0:
            raise SafetyError("SplatLab reports an active job")
        gpu = status_payload.get("gpu")
        if not isinstance(gpu, dict) or gpu.get("locked") is not False:
            raise SafetyError("shared GPU arbiter is locked or unreadable")
        jobs = status_payload.get("jobs")
        if not isinstance(jobs, list):
            raise SafetyError("SplatLab job status is malformed")

    @classmethod
    def _validate_idle_job_inventory(
        cls,
        status_payload: dict[str, Any],
        expected_ids: list[str] | None = None,
    ) -> list[str]:
        cls._validate_idle_api_status(status_payload)
        observed: list[str] = []
        for job in status_payload["jobs"]:
            if not isinstance(job, dict):
                raise SafetyError("SplatLab job inventory contains a malformed job")
            job_id = job.get("job_id")
            if (
                not isinstance(job_id, str)
                or re.fullmatch(r"splat_(?:[0-9a-f]{10}|[0-9a-f]{8})", job_id) is None
            ):
                raise SafetyError("SplatLab job inventory contains an invalid job ID")
            observed.append(job_id)
        if len(observed) != len(set(observed)):
            raise SafetyError("SplatLab job inventory contains duplicate job IDs")
        inventory = sorted(observed)
        if expected_ids is not None and inventory != expected_ids:
            raise SafetyError("job inventory changed during the marker transition")
        return inventory

    def preflight(self, receipt_name: str) -> dict[str, Any]:
        boot_id = self.current_boot_id()
        marker = marker_snapshot(self.config.marker, self.config.expected_uid)
        if os.lstat(self.config.marker).st_nlink != 1:
            raise SafetyError("maintenance marker has an unexpected hard-link count")
        _report, receipt_digest = self.validate_acceptance(
            receipt_name, marker, boot_id
        )
        consumed_path = self.config.consumed / f"{receipt_digest}.json"
        if os.path.lexists(consumed_path):
            raise SafetyError(
                "this acceptance receipt was already consumed; a new acceptance is required"
            )
        self.validate_watcher_status(boot_id)
        faults = self.ops.kernel_faults()
        if faults:
            raise SafetyError(
                f"current boot has {len(faults)} relevant kernel fault record(s)"
            )
        self.ops.gpu_safety(idle_required=True)
        self.ops.cpu_package_safety()
        throttle_counts = self.ops.cpu_throttle_counts()
        self.ops.ups_safety()
        backups_idle, backup_states = self.ops.backups_idle()
        if not backups_idle:
            raise SafetyError(f"backup interlock is active: {backup_states}")
        workloads_idle, workload_states = self.ops.competing_workloads_idle()
        if not workloads_idle:
            raise SafetyError(
                f"competing workload is active or unreadable: {workload_states}"
            )
        if not self.ops.service_active():
            raise SafetyError(
                f"{self.config.service} is not active for gated preflight"
            )
        if self.ops.cgroup_empty():
            raise SafetyError("active SplatLab service has an empty control group")
        self.ops.service_resource_safety()
        status_payload = self.ops.api_status()
        baseline_job_ids = self._validate_idle_job_inventory(status_payload)
        final_marker = marker_snapshot(self.config.marker, self.config.expected_uid)
        if not same_file_snapshot(marker, final_marker):
            raise SafetyError("maintenance marker changed during transition preflight")
        return {
            "boot_id": boot_id,
            "marker": asdict(marker),
            "receipt_name": receipt_name,
            "receipt_sha256": receipt_digest,
            "consumed_receipt_path": str(consumed_path),
            "baseline_job_ids": baseline_job_ids,
            "cpu_throttle_counts": throttle_counts,
        }

    def _write_pending(self, state: dict[str, Any], phase: str, **updates: Any) -> None:
        state.update(updates)
        state["phase"] = phase
        state["updated_at"] = utc_now()
        atomic_write_json(self.config.pending, state, self.config.expected_uid)

    def _consume_receipt(self, state: dict[str, Any]) -> None:
        path = Path(state["consumed_receipt_path"])
        expected = self.config.consumed / f"{state['receipt_sha256']}.json"
        if path != expected:
            raise SafetyError("pending transition has an invalid consumed-receipt path")
        create_exclusive_json(
            path,
            {
                "schema": "splatlab.flight-a-consumed-receipt.v1",
                "tool": TOOL_NAME,
                "transition_id": state["transition_id"],
                "receipt_name": state["receipt_name"],
                "receipt_sha256": state["receipt_sha256"],
                "payload_sha256": state["payload_sha256"],
                "consumed_at": utc_now(),
                "reason": "the one authorized API POST may follow; never reuse",
            },
            self.config.expected_uid,
        )

    def _load_pending(self) -> dict[str, Any]:
        _snapshot, payload = read_private_directory_child(
            self.config.state_dir,
            self.config.pending.name,
            uid=self.config.expected_uid,
            gid=self.config.expected_gid,
            max_bytes=1024 * 1024,
        )
        state = decode_json_strict(payload, "pending transition")
        keys = set(state)
        if (
            not STATE_BASE_KEYS.issubset(keys)
            or not keys.issubset(STATE_BASE_KEYS | STATE_OPTIONAL_KEYS)
            or state.get("schema") != STATE_SCHEMA
            or state.get("tool") != TOOL_NAME
        ):
            raise SafetyError("pending transition schema is invalid")
        transition_id = state["transition_id"]
        if (
            type(transition_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", transition_id) is None
        ):
            raise SafetyError("pending transition ID is invalid")
        for name in ("created_at", "updated_at"):
            parse_utc(state[name])
        boot_id = state["boot_id"]
        if (
            type(boot_id) is not str
            or re.fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                boot_id,
            )
            is None
        ):
            raise SafetyError("pending boot identity is invalid")
        receipt_name = state["receipt_name"]
        if type(receipt_name) is not str:
            raise SafetyError("pending receipt name is invalid")
        self._receipt_path(receipt_name)
        receipt_sha = state["receipt_sha256"]
        if (
            type(receipt_sha) is not str
            or re.fullmatch(r"[0-9a-f]{64}", receipt_sha) is None
        ):
            raise SafetyError("pending receipt digest is invalid")
        if state["payload_sha256"] != payload_sha256():
            raise SafetyError("pending payload digest is invalid")
        if type(state["consumed_receipt_path"]) is not str:
            raise SafetyError("pending consumed-receipt path is invalid")
        if Path(state["consumed_receipt_path"]) != (
            self.config.consumed / f"{receipt_sha}.json"
        ):
            raise SafetyError("pending consumed-receipt path is invalid")
        if type(state["preserved_marker_path"]) is not str:
            raise SafetyError("pending marker-preservation path is invalid")
        if Path(state["preserved_marker_path"]) != self._preserved_path(transition_id):
            raise SafetyError("pending marker-preservation path is invalid")

        marker = state["marker_snapshot"]
        if not isinstance(marker, dict) or set(marker) != {
            "device",
            "inode",
            "uid",
            "gid",
            "mode",
            "size",
            "sha256",
        }:
            raise SafetyError("pending marker snapshot is invalid")
        for name in ("device", "inode", "uid", "gid", "mode", "size"):
            if type(marker[name]) is not int or marker[name] < 0:
                raise SafetyError("pending marker snapshot is invalid")
        if (
            marker["uid"] != self.config.expected_uid
            or type(marker["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", marker["sha256"]) is None
        ):
            raise SafetyError("pending marker snapshot is invalid")
        baseline = state["baseline_job_ids"]
        if (
            not isinstance(baseline, list)
            or any(type(value) is not str for value in baseline)
            or len(baseline) != len(set(baseline))
        ):
            raise SafetyError("pending baseline job inventory is invalid")
        throttle_counts = state["cpu_throttle_counts"]
        if (
            not isinstance(throttle_counts, dict)
            or not throttle_counts
            or any(
                type(key) is not str
                or re.fullmatch(
                    r"cpu[0-9]+/thermal_throttle/package_throttle_count", key
                )
                is None
                or type(value) is not int
                or value < 0
                for key, value in throttle_counts.items()
            )
        ):
            raise SafetyError("pending CPU throttle baseline is invalid")
        if type(state["submit_attempted"]) is not bool:
            raise SafetyError("pending submission state is invalid")
        job_id = state["job_id"]
        if job_id is not None and (
            type(job_id) is not str
            or re.fullmatch(r"splat_[0-9a-f]{10}", job_id) is None
        ):
            raise SafetyError("pending Flight A job ID is invalid")
        phase = state["phase"]
        if type(phase) is not str or phase not in STATE_PHASES:
            raise SafetyError("pending transition phase is invalid")
        for name in ("marker_archived_epoch", "monitor_started_epoch"):
            if name in state and (
                type(state[name]) not in (int, float)
                or not math.isfinite(float(state[name]))
                or float(state[name]) < 0
            ):
                raise SafetyError(f"pending {name} is invalid")
        for name in ("recovery_detail", "recovery_error"):
            if name in state and type(state[name]) is not str:
                raise SafetyError(f"pending {name} is invalid")
        if phase in {
            "marker_archived",
            "fresh_service_started",
            "submitting",
            "monitoring",
        }:
            if "marker_archived_epoch" not in state:
                raise SafetyError("pending state lacks marker archive time")
        if state["submit_attempted"] and "monitor_started_epoch" not in state:
            raise SafetyError("pending state lacks submission start time")
        if phase == "monitoring" and job_id is None:
            raise SafetyError("pending monitoring state lacks a job ID")
        return state

    def _preserved_path(self, transition_id: str) -> Path:
        return self.config.marker.parent / (
            f".{self.config.marker.name}.flight-a-preserved.{transition_id}"
        )

    def _preserve_marker(self, state: dict[str, Any]) -> None:
        expected = FileSnapshot(**state["marker_snapshot"])
        current = marker_snapshot(self.config.marker, self.config.expected_uid)
        if not same_file_snapshot(expected, current):
            raise SafetyError("maintenance marker changed before preservation")
        preserved = Path(state["preserved_marker_path"])
        if (
            preserved.parent != self.config.marker.parent
            or not preserved.name.startswith(
                f".{self.config.marker.name}.flight-a-preserved."
            )
        ):
            raise SafetyError("unsafe preserved-marker path")
        os.link(self.config.marker, preserved, follow_symlinks=False)
        fsync_directory(self.config.marker.parent)
        preserved_snapshot = marker_snapshot(preserved, self.config.expected_uid)
        current = marker_snapshot(self.config.marker, self.config.expected_uid)
        if not same_file_snapshot(
            expected, preserved_snapshot
        ) or not same_file_snapshot(expected, current):
            raise SafetyError("hard-link marker preservation verification failed")
        self._write_pending(state, "marker_preserved")

    def _archive_marker(self, state: dict[str, Any]) -> None:
        expected = FileSnapshot(**state["marker_snapshot"])
        current = marker_snapshot(self.config.marker, self.config.expected_uid)
        if not same_file_snapshot(expected, current):
            raise SafetyError("maintenance marker changed before archive")
        # The watcher creates with O_EXCL and never replaces an existing marker.
        # It therefore cannot install a replacement until after this unlink, and
        # any replacement it creates afterward is never removed by this tool.
        archived_epoch = self.ops.now()
        unlink_and_sync(self.config.marker)
        self._write_pending(
            state, "marker_archived", marker_archived_epoch=archived_epoch
        )

    def _dynamic_admission_check(
        self, state: dict[str, Any], *, marker_expected: bool
    ) -> None:
        if self.current_boot_id() != state["boot_id"]:
            raise SafetyError("final admission observed a boot identity change")
        expected_marker = FileSnapshot(**state["marker_snapshot"])
        _report, receipt_digest = self.validate_acceptance(
            state["receipt_name"], expected_marker, state["boot_id"]
        )
        if receipt_digest != state["receipt_sha256"]:
            raise SafetyError("acceptance receipt changed during final admission")

        def verify_marker() -> None:
            if marker_expected:
                current = marker_snapshot(self.config.marker, self.config.expected_uid)
                if not same_file_snapshot(expected_marker, current):
                    raise SafetyError(
                        "final admission observed a changed maintenance marker"
                    )
            elif os.path.lexists(self.config.marker):
                raise SafetyError(
                    "final admission observed a reasserted maintenance marker"
                )

        verify_marker()
        faults = self.ops.kernel_faults()
        if faults:
            raise SafetyError(
                f"final admission found {len(faults)} current-boot kernel fault(s)"
            )
        self.ops.gpu_safety(idle_required=True)
        self.ops.cpu_package_safety()
        if self.ops.cpu_throttle_counts() != state["cpu_throttle_counts"]:
            raise SafetyError("final admission observed new CPU throttling")
        self.ops.ups_safety()
        backups_idle, backup_states = self.ops.backups_idle()
        if not backups_idle:
            raise SafetyError(
                f"final admission found an active backup: {backup_states}"
            )
        workloads_idle, workload_states = self.ops.competing_workloads_idle()
        if not workloads_idle:
            raise SafetyError(
                f"final admission found competing workload: {workload_states}"
            )

        if marker_expected:
            self.validate_watcher_status(
                state["boot_id"], allowed_interlocks=frozenset({"already-active"})
            )
        else:
            watcher = self.validate_watcher_status(
                state["boot_id"],
                allowed_interlocks=frozenset({"already-active", "not-required"}),
            )
            archived_epoch = float(state["marker_archived_epoch"])
            expected_interlock = (
                "not-required"
                if float(watcher["started_at_epoch"]) >= archived_epoch
                else "already-active"
            )
            if watcher["interlock_status"] != expected_interlock:
                raise SafetyError(
                    "final admission watcher interlock disagrees with marker timing"
                )
        verify_marker()

    def _abort_compute(self, detail: str) -> None:
        with contextlib.suppress(Exception):
            self.ops.terminate_service_cgroup()
        raise SafetyError(detail)

    def _verify_job_payload(self, job: dict[str, Any]) -> None:
        persisted_keys = (
            "mode",
            "input_path",
            "capture_format",
            "images_per_equirect",
            "crop_bottom",
            "num_frames_target",
            "max_num_iterations",
            "insv_fov",
            "trim_start_s",
            "trim_duration_s",
            "sfm_backend",
            "language_field",
            "capture_mode",
            "source_type",
        )
        for key in persisted_keys:
            if job.get(key) != FLIGHT_A_PAYLOAD[key]:
                raise SafetyError(
                    f"submitted job payload mismatch: {key}={job.get(key)!r}"
                )
        job_id = job.get("job_id")
        if (
            not isinstance(job_id, str)
            or re.fullmatch(r"splat_[0-9a-f]{10}", job_id) is None
        ):
            raise SafetyError("submitted job payload has an invalid job ID")
        expected_output_dir = str(EXPECTED_OUTPUT_ROOT / job_id)
        if job.get("output_dir") != expected_output_dir:
            raise SafetyError(
                f"submitted job payload mismatch: output_dir={job.get('output_dir')!r}"
            )

    def _monitor_once(
        self, state: dict[str, Any], signal_latch: SignalLatch
    ) -> str | None:
        try:
            signal_latch.check()
            elapsed = self.ops.now() - float(state["monitor_started_epoch"])
            if elapsed < 0 or elapsed > MAX_FLIGHT_RUNTIME_SECONDS:
                self._abort_compute(
                    f"Flight A exceeded its {MAX_FLIGHT_RUNTIME_SECONDS}-second runtime bound"
                )
            if os.path.lexists(self.config.marker):
                self._abort_compute("maintenance marker was reasserted during Flight A")
            if self.current_boot_id() != state["boot_id"]:
                self._abort_compute("boot identity changed during Flight A")
            if not self.ops.service_active() or self.ops.cgroup_empty():
                self._abort_compute("SplatLab service/cgroup left the running state")
            faults = self.ops.kernel_faults(float(state["monitor_started_epoch"]))
            if faults:
                self._abort_compute(
                    f"kernel fault observed during Flight A: {faults[-1]}"
                )
            gpu_safety = self.ops.gpu_safety(idle_required=False)
            compute_pids = set(gpu_safety.get("compute_pids", []))
            if gpu_safety.get("compute_process_count") != len(compute_pids):
                self._abort_compute("NVIDIA compute-process identity is incomplete")
            foreign_pids = sorted(compute_pids - self.ops.cgroup_pids())
            if foreign_pids:
                self._abort_compute(
                    f"foreign NVIDIA compute process(es) appeared: {foreign_pids}"
                )
            self.ops.cpu_package_safety()
            if self.ops.cpu_throttle_counts() != state["cpu_throttle_counts"]:
                self._abort_compute("CPU package throttle count changed")
            self.ops.ups_safety()
            self.ops.service_resource_safety()
            backups_idle, states = self.ops.backups_idle()
            if not backups_idle:
                self._abort_compute(f"backup began during Flight A: {states}")
            workloads_idle, states = self.ops.competing_workloads_idle()
            if not workloads_idle:
                self._abort_compute(
                    f"competing workload began during Flight A: {states}"
                )
            watcher_status = self.validate_watcher_status(
                state["boot_id"],
                allowed_interlocks=frozenset({"already-active", "not-required"}),
            )
            archived_epoch = float(state["marker_archived_epoch"])
            watcher_started = float(watcher_status["started_at_epoch"])
            expected_interlock = (
                "not-required"
                if watcher_started >= archived_epoch
                else "already-active"
            )
            if watcher_status["interlock_status"] != expected_interlock:
                self._abort_compute(
                    "GPU watcher interlock state does not match marker-transition timing"
                )
            status_payload = self.ops.api_status()
            jobs = status_payload.get("jobs")
            if not isinstance(jobs, list):
                self._abort_compute("SplatLab returned malformed job status")
            job_id = state["job_id"]
            gpu_holder = status_payload.get("gpu")
            if not isinstance(gpu_holder, dict):
                self._abort_compute("SplatLab GPU holder status is malformed")
            if gpu_holder.get("locked") is True:
                holder_lane = gpu_holder.get("lane")
                holder_job = gpu_holder.get("job_id")
                holder_matches = (
                    holder_lane == "splat"
                    and isinstance(holder_job, str)
                    and holder_job.startswith(f"{job_id}:")
                )
                if not holder_matches:
                    self._abort_compute(
                        "shared GPU arbiter is held by an unauthorized operation"
                    )
            elif gpu_holder.get("locked") is False:
                if compute_pids:
                    self._abort_compute(
                        "NVIDIA compute is active without the shared GPU arbiter"
                    )
            else:
                self._abort_compute("SplatLab GPU lock state is malformed")
            matches = [
                job
                for job in jobs
                if isinstance(job, dict) and job.get("job_id") == job_id
            ]
            if len(matches) != 1:
                self._abort_compute(
                    "the authorized Flight A job disappeared from status"
                )
            job = matches[0]
            self._verify_job_payload(job)
            active_jobs = status_payload.get("active_jobs")
            if not isinstance(active_jobs, int) or active_jobs > 1 or active_jobs < 0:
                self._abort_compute(
                    "SplatLab active-job count violated the one-job invariant"
                )
            status = job.get("status")
            if status in {"starting", "running"}:
                if active_jobs != 1:
                    self._abort_compute(
                        "Flight A is running but active-job count is not one"
                    )
                return None
            if active_jobs != 0:
                self._abort_compute("another SplatLab job became active")
            if status == "completed":
                return "completed"
            self._abort_compute(f"Flight A ended with status {status!r}")
        except SafetyError:
            with contextlib.suppress(Exception):
                if self.ops.service_active() or not self.ops.cgroup_empty():
                    self.ops.terminate_service_cgroup()
            raise
        return None

    def _restore_marker(self, state: dict[str, Any]) -> str:
        preserved = Path(str(state.get("preserved_marker_path", "")))
        expected = FileSnapshot(**state["marker_snapshot"])
        if os.path.lexists(self.config.marker):
            marker_snapshot(self.config.marker, self.config.expected_uid)
            disposition = "authoritative_existing_marker_retained"
        else:
            if (
                preserved.parent != self.config.marker.parent
                or not preserved.name.startswith(
                    f".{self.config.marker.name}.flight-a-preserved."
                )
            ):
                raise SafetyError("pending state has an unsafe preservation path")
            saved = marker_snapshot(preserved, self.config.expected_uid)
            if not same_file_snapshot(expected, saved):
                raise SafetyError("preserved maintenance marker no longer matches")
            try:
                os.link(preserved, self.config.marker, follow_symlinks=False)
            except FileExistsError:
                marker_snapshot(self.config.marker, self.config.expected_uid)
                disposition = "authoritative_racing_marker_retained"
            else:
                fsync_directory(self.config.marker.parent)
                restored = marker_snapshot(self.config.marker, self.config.expected_uid)
                if not same_file_snapshot(expected, restored):
                    raise SafetyError("restored maintenance marker failed verification")
                disposition = "preserved_marker_restored"

        # Only remove the known, verified preservation link after a canonical,
        # safe marker exists.  A different watcher marker remains untouched.
        if os.path.lexists(preserved):
            saved = marker_snapshot(preserved, self.config.expected_uid)
            if not same_file_snapshot(expected, saved):
                raise SafetyError("refusing to remove changed preservation link")
            unlink_and_sync(preserved)
        return disposition

    def _record_result(
        self,
        state: dict[str, Any],
        *,
        outcome: str,
        detail: str,
        marker_disposition: str,
    ) -> Path:
        ensure_private_directory(self.config.results, self.config.expected_uid)
        result = {
            "schema": RESULT_SCHEMA,
            "tool": TOOL_NAME,
            "transition_id": state["transition_id"],
            "boot_id": state["boot_id"],
            "finished_at": utc_now(),
            "outcome": outcome,
            "detail": detail,
            "job_id": state.get("job_id"),
            "submit_attempted": state.get("submit_attempted") is True,
            "payload_sha256": state["payload_sha256"],
            "receipt_name": state["receipt_name"],
            "receipt_sha256": state["receipt_sha256"],
            "marker_disposition": marker_disposition,
            "marker_relocked_before_review": True,
        }
        path = self.config.results / f"flight-a-{state['transition_id']}.json"

        def validate_existing() -> None:
            _snapshot, payload = read_private_directory_child(
                self.config.results,
                path.name,
                uid=self.config.expected_uid,
                gid=self.config.expected_gid,
                max_bytes=1024 * 1024,
            )
            existing = decode_json_strict(payload, "Flight A result")
            expected = {
                "schema": RESULT_SCHEMA,
                "tool": TOOL_NAME,
                "transition_id": state["transition_id"],
                "boot_id": state["boot_id"],
                "payload_sha256": state["payload_sha256"],
                "receipt_name": state["receipt_name"],
                "receipt_sha256": state["receipt_sha256"],
                "job_id": state.get("job_id"),
                "submit_attempted": state.get("submit_attempted") is True,
                "marker_relocked_before_review": True,
            }
            if set(existing) != RESULT_KEYS or any(
                existing.get(key) != value for key, value in expected.items()
            ):
                raise SafetyError("existing Flight A result is invalid")
            try:
                finished_at = parse_utc(existing["finished_at"])
                created_at = parse_utc(state["created_at"])
            except SafetyError as exc:
                raise SafetyError("existing Flight A result is invalid") from exc
            outcome = existing["outcome"]
            detail_value = existing["detail"]
            job_id = existing["job_id"]
            submit_attempted = existing["submit_attempted"]
            marker_disposition = existing["marker_disposition"]
            if (
                not isinstance(outcome, str)
                or outcome not in RESULT_OUTCOMES
                or not isinstance(detail_value, str)
                or not detail_value
                or len(detail_value) > 2000
                or type(submit_attempted) is not bool
                or not isinstance(marker_disposition, str)
                or marker_disposition not in MARKER_DISPOSITIONS
                or existing["marker_relocked_before_review"] is not True
                or finished_at < created_at
                or finished_at > self.ops.now() + 5
                or (
                    job_id is not None
                    and (
                        not isinstance(job_id, str)
                        or re.fullmatch(r"splat_[0-9a-f]{10}", job_id) is None
                    )
                )
                or (not submit_attempted and job_id is not None)
                or (outcome == "completed" and (not submit_attempted or job_id is None))
                or (
                    outcome == "ambiguous_submission"
                    and (not submit_attempted or job_id is not None)
                )
                or (outcome == "failed" and submit_attempted and job_id is None)
            ):
                raise SafetyError("existing Flight A result is invalid")

        if os.path.lexists(path):
            validate_existing()
            return path
        try:
            publish_exclusive_json(path, result, self.config.expected_uid)
        except FileExistsError:
            validate_existing()
        return path

    def _recover_state(
        self,
        state: dict[str, Any],
        *,
        outcome: str,
        detail: str,
        restart_service: bool = True,
    ) -> tuple[bool, Path | None]:
        # Failure to prove an idle service/cgroup must still attempt termination.
        terminate_required = True
        marker_safe = False
        try:
            with contextlib.suppress(Exception):
                self._write_pending(state, "recovering", recovery_detail=detail[:1000])
            marker_disposition = self._restore_marker(state)
            marker_snapshot(self.config.marker, self.config.expected_uid)
            marker_safe = True
            if not restart_service:
                service_active = self.ops.service_active()
                cgroup_empty = self.ops.cgroup_empty()
                terminate_required = service_active or not cgroup_empty
            if terminate_required:
                self.ops.terminate_service_cgroup()
                if self.ops.service_active() or not self.ops.cgroup_empty():
                    raise SafetyError(
                        "SplatLab service/cgroup remained active during recovery"
                    )
            if restart_service:
                self.ops.start_service()
            marker_snapshot(self.config.marker, self.config.expected_uid)
            result_path = self._record_result(
                state,
                outcome=outcome,
                detail=detail[:2000],
                marker_disposition=marker_disposition,
            )
            unlink_and_sync(self.config.pending)
            return True, result_path
        except Exception as exc:  # noqa: BLE001 - recovery must retain pending state
            with contextlib.suppress(Exception):
                self._write_pending(
                    state,
                    "recovery_failed",
                    recovery_error=f"{type(exc).__name__}: {exc}"[:1000],
                )
            if terminate_required or not marker_safe:
                with contextlib.suppress(Exception):
                    self.ops.terminate_service_cgroup()
            return False, None

    def recover(
        self,
        *,
        restart_service: bool = True,
        allow_active_transition: bool = False,
    ) -> bool:
        try:
            with self.transition_lock():
                if not os.path.lexists(self.config.pending):
                    return True
                try:
                    state = self._load_pending()
                except SafetyError:
                    with contextlib.suppress(Exception):
                        self.ops.terminate_service_cgroup()
                    return False
                recovered, _result = self._recover_state(
                    state,
                    outcome="recovered_without_resubmission",
                    detail=(
                        "durable pending transition recovered; no API submission "
                        "was retried"
                    ),
                    restart_service=restart_service,
                )
                return recovered
        except TransitionBusy:
            # The boot dependency is also pulled into the fresh service start
            # requested by the live supervisor. Only that generated unit uses
            # this option. A crashed/powered-off supervisor releases flock, so
            # actual recovery still takes this path's lock and relocks first.
            if allow_active_transition:
                if not os.path.lexists(self.config.pending):
                    raise SafetyError(
                        "active transition lock has no durable pending state"
                    )
                state = self._load_pending()
                phase = state["phase"]
                if phase == "marker_archived":
                    if os.path.lexists(self.config.marker):
                        raise SafetyError(
                            "fresh-start dependency found a reasserted marker"
                        )
                    preserved = marker_snapshot(
                        Path(state["preserved_marker_path"]),
                        self.config.expected_uid,
                    )
                    if not same_file_snapshot(
                        FileSnapshot(**state["marker_snapshot"]), preserved
                    ):
                        raise SafetyError(
                            "fresh-start dependency found invalid marker preservation"
                        )
                elif phase == "recovering":
                    marker_snapshot(self.config.marker, self.config.expected_uid)
                else:
                    raise SafetyError(
                        "active transition is not at a service-start dependency phase"
                    )
                return True
            raise

    def run(self, receipt_name: str) -> Path:
        with self.heavy_work_lock(), self.transition_lock():
            if os.path.lexists(self.config.pending):
                try:
                    state = self._load_pending()
                except SafetyError:
                    with contextlib.suppress(Exception):
                        self.ops.terminate_service_cgroup()
                    raise
                recovered, _ = self._recover_state(
                    state,
                    outcome="recovered_before_new_authorization",
                    detail="older pending transition recovered without starting a new Flight A",
                )
                if not recovered:
                    raise SafetyError("older pending transition could not be recovered")
                raise SafetyError(
                    "older transition was recovered; issue a new explicit authorization to continue"
                )

            preflight = self.preflight(receipt_name)
            transition_id = uuid.uuid4().hex
            state: dict[str, Any] = {
                "schema": STATE_SCHEMA,
                "tool": TOOL_NAME,
                "transition_id": transition_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "phase": "validated",
                "boot_id": preflight["boot_id"],
                "receipt_name": receipt_name,
                "receipt_sha256": preflight["receipt_sha256"],
                "consumed_receipt_path": preflight["consumed_receipt_path"],
                "marker_snapshot": preflight["marker"],
                "preserved_marker_path": str(self._preserved_path(transition_id)),
                "payload_sha256": payload_sha256(),
                "baseline_job_ids": preflight["baseline_job_ids"],
                "cpu_throttle_counts": preflight["cpu_throttle_counts"],
                "submit_attempted": False,
                "job_id": None,
            }
            atomic_write_json(self.config.pending, state, self.config.expected_uid)

            outcome = "failed"
            detail = "Flight A did not start"
            result_path: Path | None = None
            error: BaseException | None = None
            with SignalLatch() as signal_latch:
                try:
                    signal_latch.check()
                    self.ops.terminate_service_cgroup()
                    if not self.ops.cgroup_empty():
                        raise SafetyError(
                            "SplatLab cgroup remained populated after stop"
                        )
                    self._write_pending(state, "service_stopped")
                    signal_latch.check()
                    self._preserve_marker(state)
                    signal_latch.check()
                    self._dynamic_admission_check(state, marker_expected=True)
                    signal_latch.check()
                    self._archive_marker(state)
                    signal_latch.check()
                    self._dynamic_admission_check(state, marker_expected=False)
                    signal_latch.check()
                    self.ops.start_service()
                    self._write_pending(state, "fresh_service_started")
                    signal_latch.check()
                    if os.path.lexists(self.config.marker):
                        raise SafetyError(
                            "maintenance marker was reasserted before submission"
                        )
                    fresh_status = self.ops.api_status()
                    self._validate_idle_job_inventory(
                        fresh_status, state["baseline_job_ids"]
                    )
                    baseline = set(state["baseline_job_ids"])
                    signal_latch.check()
                    final_admission_started_ns = self.ops.monotonic_ns()
                    self._dynamic_admission_check(state, marker_expected=False)
                    self.ops.service_resource_safety()
                    self._validate_idle_job_inventory(
                        self.ops.api_status(), state["baseline_job_ids"]
                    )
                    admission_age = (
                        self.ops.monotonic_ns() - final_admission_started_ns
                    ) / 1_000_000_000
                    if (
                        admission_age < 0
                        or admission_age > MAX_FINAL_ADMISSION_AGE_SECONDS
                    ):
                        raise SafetyError(
                            "final admission expired before receipt consumption"
                        )
                    signal_latch.check()
                    self._consume_receipt(state)
                    self._write_pending(
                        state,
                        "submitting",
                        submit_attempted=True,
                        monitor_started_epoch=self.ops.now(),
                    )
                    signal_latch.check()
                    if os.path.lexists(self.config.marker):
                        raise SafetyError(
                            "maintenance marker was reasserted immediately before submission"
                        )
                    self._validate_idle_job_inventory(
                        self.ops.api_status(), state["baseline_job_ids"]
                    )
                    admission_age = (
                        self.ops.monotonic_ns() - final_admission_started_ns
                    ) / 1_000_000_000
                    if (
                        admission_age < 0
                        or admission_age > MAX_FINAL_ADMISSION_AGE_SECONDS
                    ):
                        raise SafetyError(
                            "final admission expired immediately before submission"
                        )
                    signal_latch.check()
                    self._release_heavy_work_lock()
                    try:
                        response = self.ops.submit_flight_a()
                    except Exception as exc:  # one POST may have reached the server
                        raise AmbiguousSubmission(
                            "the single API submission was ambiguous and will not be retried"
                        ) from exc
                    signal_latch.check()
                    job_id = response.get("job_id")
                    if not isinstance(job_id, str) or not re.fullmatch(
                        r"splat_[0-9a-f]{10}", job_id
                    ):
                        raise AmbiguousSubmission(
                            "submission response lacked a valid new job ID; no retry is allowed"
                        )
                    if job_id in baseline:
                        raise SafetyError("API returned a pre-existing job ID")
                    self._verify_job_payload(response)
                    self._write_pending(state, "monitoring", job_id=job_id)
                    while True:
                        terminal = self._monitor_once(state, signal_latch)
                        if terminal is not None:
                            outcome = terminal
                            detail = "the one authorized Flight A job completed"
                            break
                        self.ops.sleep(self.config.monitor_interval_seconds)
                except (
                    BaseException
                ) as exc:  # relock on errors, signals, and interrupts
                    error = exc
                    outcome = (
                        "ambiguous_submission"
                        if isinstance(exc, AmbiguousSubmission)
                        or (
                            state.get("submit_attempted") is True
                            and state.get("phase") == "submitting"
                        )
                        else "failed"
                    )
                    detail = f"{type(exc).__name__}: {exc}"
                finally:
                    recovered, result_path = self._recover_state(
                        state, outcome=outcome, detail=detail
                    )
                    if not recovered:
                        raise SafetyError(
                            "Flight A recovery did not complete; pending state was retained"
                        ) from error

            if error is not None:
                raise error
            if result_path is None:
                raise SafetyError("Flight A result receipt was not written")
            return result_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or recover the one fixed, crash-safe SplatLab Flight A transition."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--receipt",
        required=True,
        help="acceptance report basename from the fixed private acceptance directory",
    )
    run_parser.add_argument(
        "--authorize-exact-flight-a",
        required=True,
        action="store_true",
        help="explicitly authorize only the built-in 90-frame/7000-iteration payload",
    )
    recover_parser = subparsers.add_parser("recover")
    recover_parser.add_argument(
        "--defer-service-start",
        action="store_true",
        help="restore the marker but let systemd ordering start the gated service",
    )
    recover_parser.add_argument(
        "--allow-active-transition",
        action="store_true",
        help="boot dependency only: succeed if the live supervisor owns flock",
    )
    return parser.parse_args(argv)


def require_systemd_flight_invocation(
    *,
    environment: dict[str, str] | None = None,
    cgroup_path: Path = Path("/proc/self/cgroup"),
) -> None:
    values = os.environ if environment is None else environment
    invocation_id = values.get("INVOCATION_ID", "")
    exec_pid = values.get("SYSTEMD_EXEC_PID", "")
    try:
        cgroup = cgroup_path.read_text(encoding="ascii")
    except OSError as exc:
        raise SafetyError("cannot prove the generated systemd Flight A unit") from exc
    if (
        re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None
        or exec_pid != str(os.getpid())
        or "splatlab-flight-a@" not in cgroup
        or ".service" not in cgroup
    ):
        raise SafetyError(
            "Flight A run is allowed only through the generated systemd template"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    supervisor = Supervisor()
    try:
        if args.command == "recover":
            if not supervisor.recover(
                restart_service=not args.defer_service_start,
                allow_active_transition=args.allow_active_transition,
            ):
                print(
                    f"{TOOL_NAME}: recovery failed; SplatLab remains stopped",
                    file=sys.stderr,
                )
                return 2
            print(f"{TOOL_NAME}: recovery complete; no job was submitted")
            return 0
        require_systemd_flight_invocation()
        result = supervisor.run(args.receipt)
        print(f"{TOOL_NAME}: Flight A completed and maintenance marker was restored")
        print(f"result={result}")
        return 0
    except SafetyError as exc:
        print(f"{TOOL_NAME}: BLOCKED: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:  # signals/interrupts are already recovered above
        print(
            f"{TOOL_NAME}: interrupted after relock: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
