"""Black-box recorder for GPU-coordination deaths.

Born 2026-07-31, after diagnosing four identical "lease ownership lost" kills
took Redis MONITOR taps, keyspace-event listeners, PSI, sar archaeology and
cgroup forensics across four corpses. Every one of those signals was readable
at the moment of death — nothing captured them. This module does: the lease
heartbeat feeds a cadence ring buffer, and when the arbiter kills an
operation it writes one JSON record with everything the investigation needed:

  heartbeat cadence  -> was the holder starving?  (gaps vs HEARTBEAT_SEC)
  PSI cpu/memory/io  -> was the HOST stalling?    (the 30% full-stall smoking gun)
  cgroup memory.*    -> was the SERVICE throttled? (the 35M-hit smoking gun)
  nvidia-smi         -> who was on the card?       (the 24.5G Ollama squatter)
  redis ping latency -> was Redis itself slow?

The recorder must never make a failure worse: every probe is individually
guarded, write_record returns None instead of raising, and the spool is
pruned so it cannot grow unbounded.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

SPOOL_ENV = "SPLATLAB_FLIGHTREC_DIR"
_DEFAULT_SPOOL = Path.home() / ".local" / "state" / "splatlab" / "flightrec"
KEEP_RECORDS = 20
HEARTBEAT_RING = 64

_beats: deque[dict[str, float | str]] = deque(maxlen=HEARTBEAT_RING)
_beats_lock = threading.Lock()


def spool_dir() -> Path:
    configured = os.environ.get(SPOOL_ENV, "").strip()
    return Path(configured) if configured else _DEFAULT_SPOOL


def record_heartbeat(token_tail: str, gap_secs: float, refresh_secs: float) -> None:
    """Called by the lease heartbeat thread once per beat. Cheap and lock-tight."""
    with _beats_lock:
        _beats.append(
            {
                "at": time.time(),
                "token": token_tail,
                "gap_secs": round(gap_secs, 3),
                "refresh_secs": round(refresh_secs, 3),
            }
        )


def _read_text(path: Path) -> str | None:
    with contextlib.suppress(OSError):
        return path.read_text().strip()
    return None


def _psi() -> dict[str, str | None]:
    base = Path("/proc/pressure")
    return {kind: _read_text(base / kind) for kind in ("cpu", "memory", "io")}


def _own_cgroup() -> dict[str, str | None]:
    out: dict[str, str | None] = {"dir": None}
    raw = _read_text(Path("/proc/self/cgroup")) or ""
    for line in raw.splitlines():
        if line.startswith("0::"):
            d = Path("/sys/fs/cgroup") / line.split("::", 1)[1].strip().lstrip("/")
            out["dir"] = str(d)
            for name in ("memory.current", "memory.high", "memory.max"):
                out[name] = _read_text(d / name)
            events = _read_text(d / "memory.events")
            out["memory.events"] = " ".join((events or "").split("\n"))
            break
    return out


def _nvidia_snapshot() -> str | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip() or "(no compute apps)"
    except Exception:  # noqa: BLE001 - a dead probe must not kill the record
        return None


def _redis_ping_ms() -> float | None:
    # Local import: the recorder must stay importable even if gpu_arbiter's
    # redis stack is unavailable in a stripped test environment.
    try:
        import gpu_arbiter

        r = gpu_arbiter._redis()
        if r is None:
            return None
        started = time.monotonic()
        r.exists(gpu_arbiter.LOCK_KEY)
        return round((time.monotonic() - started) * 1000, 2)
    except Exception:  # noqa: BLE001
        return None


def snapshot(reason: str, operation_id: str | None) -> dict[str, Any]:
    """Assemble the record. Every section degrades to None independently."""
    with _beats_lock:
        beats = list(_beats)
    load = None
    with contextlib.suppress(OSError):
        load = os.getloadavg()
    return {
        "at": time.time(),
        "reason": reason,
        "operation_id": operation_id,
        "heartbeats": beats,
        "psi": _psi(),
        "cgroup": _own_cgroup(),
        "meminfo_available": _read_text(Path("/proc/meminfo")) and next(
            (
                line
                for line in (_read_text(Path("/proc/meminfo")) or "").splitlines()
                if line.startswith("MemAvailable")
            ),
            None,
        ),
        "loadavg": load,
        "gpu_compute_apps": _nvidia_snapshot(),
        "redis_ping_ms": _redis_ping_ms(),
    }


def _prune(spool: Path) -> None:
    with contextlib.suppress(OSError):
        records = sorted(spool.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in records[:-KEEP_RECORDS]:
            with contextlib.suppress(OSError):
                old.unlink()


def write_record(reason: str, operation_id: str | None) -> str | None:
    """Write one flight record; return its path, or None. NEVER raises."""
    try:
        spool = spool_dir()
        spool.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_op = "".join(c for c in (operation_id or "unknown") if c.isalnum() or c in "-_")[:60]
        path = spool / f"{stamp}-{safe_op}.json"
        path.write_text(json.dumps(snapshot(reason, operation_id), indent=2))
        _prune(spool)
        return str(path)
    except Exception:  # noqa: BLE001 - the recorder must never worsen a failure
        return None
