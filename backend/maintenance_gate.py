"""Dynamic, fail-closed SplatLab hardware-maintenance admission."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAINTENANCE_FILE = Path("/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf")
SUPERVISED_UNLOCK_FILE = Path("/home/rtoony/.config/splatlab/gpu-compute-unlock.json")
WATCHER_STATUS_FILE = Path("/home/rtoony/.local/state/nexus-watchers/gpu_health_watch_status.json")
REASON_KEY = "SPLAT_TRAINING_DISABLED_REASON"
DEFAULT_REASON = "SplatLab GPU hardware maintenance is active."
UNLOCK_SCHEMA = "splatlab.compute-unlock.v1"
UNLOCK_MAX_SECONDS = 2 * 60 * 60
WATCHER_MAX_AGE_SECONDS = 6 * 60
WATCHER_ACTIVE_FAULT_KEYS = (
    "gpu_unreadable",
    "xid",
    "aer_current",
    "aer_severe",
    "platform_fatal",
)


def _marker_reason(path: Path) -> str | None:
    """Return None only when the marker definitely does not exist."""
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        return DEFAULT_REASON

    try:
        if not path.is_file():
            return DEFAULT_REASON
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return DEFAULT_REASON

    prefix = f"{REASON_KEY}="
    for line in lines:
        if not line.startswith(prefix):
            continue
        value = line[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].strip()
        return value or DEFAULT_REASON
    return DEFAULT_REASON


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def supervised_unlock_status(path: Path | None = None) -> dict[str, Any]:
    """Return the current supervised-unlock state.

    A valid unlock deliberately bypasses the hardware marker, but only when it
    is explicit, short-lived, and scoped to the supervised transition mode.
    Malformed/stale files are ignored so the marker remains fail-closed.
    """
    unlock_path = path or SUPERVISED_UNLOCK_FILE
    status: dict[str, Any] = {
        "active": False,
        "path": str(unlock_path),
        "schema": UNLOCK_SCHEMA,
        "mode": None,
        "reason": None,
        "operator": None,
        "created_at": None,
        "expires_at": None,
        "seconds_remaining": 0,
        "max_active_jobs": 1,
        "detail": None,
        "watcher": None,
    }
    try:
        unlock_path.lstat()
    except FileNotFoundError:
        status["detail"] = "absent"
        return status
    except OSError:
        status["detail"] = "unreadable"
        return status

    try:
        if not unlock_path.is_file():
            status["detail"] = "not a regular file"
            return status
        raw = json.loads(unlock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        status["detail"] = "invalid JSON"
        return status
    if not isinstance(raw, dict):
        status["detail"] = "unlock must be a JSON object"
        return status

    status.update(
        {
            "mode": raw.get("mode"),
            "reason": raw.get("reason"),
            "operator": raw.get("operator"),
            "created_at": raw.get("created_at"),
            "expires_at": raw.get("expires_at"),
            "max_active_jobs": raw.get("max_active_jobs", 1),
        }
    )
    if raw.get("schema") != UNLOCK_SCHEMA:
        status["detail"] = "schema mismatch"
        return status
    if raw.get("enabled") is not True or raw.get("mode") != "supervised":
        status["detail"] = "not enabled for supervised mode"
        return status
    if not isinstance(raw.get("reason"), str) or not raw["reason"].strip():
        status["detail"] = "missing reason"
        return status
    if not isinstance(raw.get("operator"), str) or not raw["operator"].strip():
        status["detail"] = "missing operator"
        return status
    if raw.get("max_active_jobs", 1) != 1:
        status["detail"] = "max_active_jobs must be 1"
        return status

    expires_at = _parse_utc(raw.get("expires_at"))
    now = datetime.now(timezone.utc)
    if expires_at is None:
        status["detail"] = "invalid expires_at"
        return status
    seconds_remaining = int((expires_at - now).total_seconds())
    status["seconds_remaining"] = max(0, seconds_remaining)
    if seconds_remaining <= 0:
        status["detail"] = "expired"
        return status
    if seconds_remaining > UNLOCK_MAX_SECONDS:
        status["detail"] = "expiry exceeds supervised limit"
        return status

    watcher = gpu_watcher_status()
    status["watcher"] = watcher
    if not watcher["ok"]:
        status["detail"] = f"watcher guard blocked: {watcher['detail']}"
        return status

    status["active"] = True
    status["reason"] = raw["reason"].strip()
    status["operator"] = raw["operator"].strip()
    status["detail"] = "active"
    return status


def gpu_watcher_status(path: Path | None = None) -> dict[str, Any]:
    status_path = path or WATCHER_STATUS_FILE
    status: dict[str, Any] = {
        "ok": False,
        "path": str(status_path),
        "detail": None,
        "age_seconds": None,
        "fault_counts": None,
    }
    try:
        raw = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        status["detail"] = "missing or invalid watcher status"
        return status
    if not isinstance(raw, dict):
        status["detail"] = "watcher status must be a JSON object"
        return status
    try:
        finished_at = float(raw.get("finished_at_epoch"))
    except (TypeError, ValueError):
        status["detail"] = "watcher status has invalid timestamp"
        return status
    age = max(0.0, time.time() - finished_at)
    status["age_seconds"] = round(age, 1)
    if age > WATCHER_MAX_AGE_SECONDS:
        status["detail"] = "watcher status is stale"
        return status
    if raw.get("run_success") is not True:
        status["detail"] = "watcher run did not succeed"
        return status
    fault_counts = raw.get("fault_counts")
    if not isinstance(fault_counts, dict):
        status["detail"] = "watcher fault counts missing"
        return status
    status["fault_counts"] = fault_counts
    active_faults = {
        key: int(fault_counts.get(key, 0) or 0)
        for key in WATCHER_ACTIVE_FAULT_KEYS
    }
    tripped = {key: value for key, value in active_faults.items() if value > 0}
    if tripped:
        status["detail"] = f"active GPU faults: {tripped}"
        return status
    status["ok"] = True
    status["detail"] = "fresh watcher status with no active faults"
    return status


def maintenance_reason(fallback_reason: str = "") -> str:
    """Read dynamic controls on every call and fail closed unless explicitly armed."""
    if supervised_unlock_status()["active"]:
        return ""
    marker_reason = _marker_reason(MAINTENANCE_FILE)
    if marker_reason is not None:
        return marker_reason
    return os.environ.get(REASON_KEY, "").strip() or fallback_reason.strip()
