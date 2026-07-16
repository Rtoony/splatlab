"""Dynamic, fail-closed SplatLab hardware-maintenance admission."""

from __future__ import annotations

import os
from pathlib import Path

MAINTENANCE_FILE = Path("/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf")
REASON_KEY = "SPLAT_TRAINING_DISABLED_REASON"
DEFAULT_REASON = "SplatLab GPU hardware maintenance is active."


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


def maintenance_reason(fallback_reason: str = "") -> str:
    """Read the canonical marker on every call; marker presence is authoritative."""
    marker_reason = _marker_reason(MAINTENANCE_FILE)
    if marker_reason is not None:
        return marker_reason
    return os.environ.get(REASON_KEY, "").strip() or fallback_reason.strip()
