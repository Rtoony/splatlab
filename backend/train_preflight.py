"""Pre-train memory admission: refuse in one second what quicksand kills in hours.

Born 2026-07-31. A 4920-image equirect capture needed ~50 GB of anonymous
memory (uint8 image cache + torch working set) inside a service cgroup whose
MemoryHigh was 32 GiB. Nothing checked; the kernel throttled every allocation
(memory.events: 35 million `high` hits), "caching" crawled for 8 hours, the
throttle starved the GPU-lease heartbeat, and four consecutive trainings died
"lease ownership lost". Every input of that prediction was knowable before
the train stage started.

This module predicts the train stage's memory need from the processed dataset
(image count x dimensions, uint8 CPU cache — both pinned by the train
command) and compares it against the cgroup's own limits. The verdict either
passes (with a WARN band), or refuses with the exact numbers and the exact
knobs: raise the service guard, or shrink the dataset.

Calibration receipts (2026-07-31, this box):
- 4920 imgs @ 1440x1440: cache 30.6 GB, observed anonymous peak ~50 GB on a
  64 GiB limit -> fit (barely; no re-throttle events).
- Same job on the old 32 GiB limit -> 8h quicksand (must refuse).
- 2560 imgs @ 1440x1440 (largest historical success): predicted 26.3 GB vs
  28 GB room on the old limit -> must still pass.
OVERHEAD_FACTOR/BASE_BYTES below thread exactly that needle.

The gate must never break the pipeline: any inspection failure (missing
Pillow, unreadable dir, exotic cgroup layout) logs a note and SKIPS — the
same philosophy as the stitch sanity gate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # Pillow is an optional backend dependency; the gate degrades to skip
    from PIL import Image
except Exception:  # noqa: BLE001 - any import failure means "no gate"
    Image = None  # type: ignore[assignment]

# Anonymous-memory model: uint8 cache (N*W*H*3) plus torch/dataloader working
# set scaling with the cache, plus a flat process base. Tuned to the receipts
# in the module docstring — change only with a new measured peak.
OVERHEAD_FACTOR = 1.4
BASE_BYTES = 4 * 1024**3

# Refuse when predicted need exceeds the room outright; warn above this share.
WARN_ROOM_SHARE = 0.80

CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass
class PreflightVerdict:
    ok: bool
    log_lines: list[str] = field(default_factory=list)


def _read_cgroup_bytes(path: Path) -> int | None:
    """Parse a cgroup v2 memory file: integer bytes, or None for 'max'/absent."""
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _own_cgroup_dir(proc_cgroup: Path = Path("/proc/self/cgroup")) -> Path | None:
    """Resolve this process's cgroup v2 directory (children share it via slice)."""
    try:
        for line in proc_cgroup.read_text().splitlines():
            # cgroup v2: "0::/user.slice/.../splatlab.service"
            if line.startswith("0::"):
                rel = line.split("::", 1)[1].strip().lstrip("/")
                d = CGROUP_ROOT / rel
                return d if d.is_dir() else None
    except OSError:
        return None
    return None


def predict_train_bytes(images_dir: Path) -> tuple[int, int, int, int] | None:
    """Return (n_images, width, height, predicted_bytes), or None if unknowable."""
    if Image is None:
        return None
    try:
        names = [n for n in os.listdir(images_dir) if not n.startswith(".")]
    except OSError:
        return None
    if not names:
        return None
    try:
        with Image.open(images_dir / sorted(names)[0]) as im:
            width, height = im.size
    except Exception:  # noqa: BLE001 - unreadable first image => no gate
        return None
    n = len(names)
    cache = n * width * height * 3  # uint8 RGB — pinned by the train command
    predicted = int(cache * OVERHEAD_FACTOR) + BASE_BYTES
    return n, width, height, predicted


def memory_room_bytes(cgroup_dir: Path | None = None) -> tuple[int, str] | None:
    """Effective anonymous-memory room: binding limit minus current usage.

    Returns (room_bytes, described_limit) or None when no limit binds (then
    the host is the only bound and the kernel's reclaim handles it).
    """
    d = cgroup_dir if cgroup_dir is not None else _own_cgroup_dir()
    if d is None:
        return None
    high = _read_cgroup_bytes(d / "memory.high")
    maximum = _read_cgroup_bytes(d / "memory.max")
    limits = [(v, name) for v, name in ((high, "memory.high"), (maximum, "memory.max")) if v is not None]
    if not limits:
        return None
    limit, limit_name = min(limits)
    current = _read_cgroup_bytes(d / "memory.current") or 0
    return max(0, limit - current), f"{limit_name}={limit / 1024**3:.0f}G (current use {current / 1024**3:.1f}G)"


def train_preflight(processed_dir: Path, cgroup_dir: Path | None = None) -> PreflightVerdict:
    """Gate the train stage on predicted memory vs the cgroup's real room."""
    prediction = predict_train_bytes(processed_dir / "images")
    if prediction is None:
        return PreflightVerdict(
            ok=True,
            log_lines=["memory preflight skipped (dataset unreadable or Pillow missing)"],
        )
    n, width, height, predicted = prediction
    room = memory_room_bytes(cgroup_dir)
    summary = (
        f"memory preflight: {n} images @ {width}x{height} -> "
        f"predicted need ~{predicted / 1024**3:.1f}G"
    )
    if room is None:
        return PreflightVerdict(ok=True, log_lines=[f"{summary}; no cgroup limit binds — proceeding"])
    room_bytes, limit_desc = room
    if predicted > room_bytes:
        return PreflightVerdict(
            ok=False,
            log_lines=[
                f"{summary} EXCEEDS the service memory room ~{room_bytes / 1024**3:.1f}G ({limit_desc}).",
                "Refusing before training instead of letting the kernel throttle it for hours.",
                "Fix EITHER side: raise the guard "
                "(systemctl --user set-property splatlab.service splatlab.slice MemoryHigh=<n>G) "
                "or shrink the dataset (images_per_equirect / num_frames_target).",
            ],
        )
    lines = [f"{summary} vs room ~{room_bytes / 1024**3:.1f}G ({limit_desc}) — OK"]
    if predicted > room_bytes * WARN_ROOM_SHARE:
        lines.append(
            f"memory preflight WARNING: predicted need is above {WARN_ROOM_SHARE:.0%} "
            "of the room — a modestly larger capture will be refused; consider raising the guard."
        )
    return PreflightVerdict(ok=True, log_lines=lines)
