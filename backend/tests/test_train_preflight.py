"""Tests for the pre-train memory admission gate (born of the 2026-07-31
four-dead-trains incident: a 4920-image cache vs a 32 GiB cgroup guard)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_preflight as tp  # noqa: E402

PIL = pytest.importorskip("PIL", reason="preflight skips itself without Pillow")
from PIL import Image  # noqa: E402


def _make_dataset(tmp_path: Path, n: int, size: tuple[int, int]) -> Path:
    processed = tmp_path / "processed"
    images = processed / "images"
    images.mkdir(parents=True)
    # Only the FIRST image is opened for dimensions; the rest just need to
    # exist for the count. Keeps fixtures fast at historical scale (4920).
    Image.new("RGB", size).save(images / "frame_00001.jpg")
    for i in range(2, n + 1):
        (images / f"frame_{i:05d}.jpg").touch()
    return processed


def _make_cgroup(tmp_path: Path, high: str, maximum: str, current: str) -> Path:
    d = tmp_path / "cg"
    d.mkdir()
    (d / "memory.high").write_text(high + "\n")
    (d / "memory.max").write_text(maximum + "\n")
    (d / "memory.current").write_text(current + "\n")
    return d


GIB = 1024**3


def test_disaster_dataset_refused_on_old_guard(tmp_path: Path) -> None:
    """The exact 2026-07-31 configuration must refuse: 4920 x 1440x1440
    against MemoryHigh=32G with a ~2G service baseline."""
    processed = _make_dataset(tmp_path, 4920, (1440, 1440))
    cg = _make_cgroup(tmp_path, str(32 * GIB), str(48 * GIB), str(2 * GIB))
    v = tp.train_preflight(processed, cgroup_dir=cg)
    assert not v.ok
    joined = " ".join(v.log_lines)
    assert "EXCEEDS" in joined
    assert "MemoryHigh" in joined  # the message names the exact knob
    assert "images_per_equirect" in joined  # ...and the dataset-side knob


def test_disaster_dataset_passes_on_raised_guard(tmp_path: Path) -> None:
    """Same dataset on the raised 64G guard (the proven-working config)."""
    processed = _make_dataset(tmp_path, 4920, (1440, 1440))
    cg = _make_cgroup(tmp_path, str(64 * GIB), str(80 * GIB), str(2 * GIB))
    v = tp.train_preflight(processed, cgroup_dir=cg)
    assert v.ok
    assert any("OK" in line for line in v.log_lines)


def test_largest_historical_success_still_passes_old_guard(tmp_path: Path) -> None:
    """2560 x 1440x1440 completed on the OLD 32G guard — the gate must not
    retroactively refuse configurations that demonstrably worked."""
    processed = _make_dataset(tmp_path, 2560, (1440, 1440))
    cg = _make_cgroup(tmp_path, str(32 * GIB), str(48 * GIB), str(2 * GIB))
    v = tp.train_preflight(processed, cgroup_dir=cg)
    assert v.ok


def test_warn_band_above_80_percent(tmp_path: Path) -> None:
    processed = _make_dataset(tmp_path, 2560, (1440, 1440))
    # Room ~27G; prediction ~26.3G -> >80% share -> pass WITH warning.
    cg = _make_cgroup(tmp_path, str(29 * GIB), "max", str(2 * GIB))
    v = tp.train_preflight(processed, cgroup_dir=cg)
    assert v.ok
    assert any("WARNING" in line for line in v.log_lines)


def test_no_binding_limit_proceeds(tmp_path: Path) -> None:
    processed = _make_dataset(tmp_path, 4920, (1440, 1440))
    cg = _make_cgroup(tmp_path, "max", "max", str(2 * GIB))
    v = tp.train_preflight(processed, cgroup_dir=cg)
    assert v.ok
    assert any("no cgroup limit binds" in line for line in v.log_lines)


def test_unreadable_dataset_skips_never_blocks(tmp_path: Path) -> None:
    v = tp.train_preflight(tmp_path / "nope", cgroup_dir=None)
    assert v.ok
    assert any("skipped" in line for line in v.log_lines)


def test_tightest_limit_binds(tmp_path: Path) -> None:
    """memory.max below memory.high (unusual but legal) must bind."""
    processed = _make_dataset(tmp_path, 4920, (1440, 1440))
    cg = _make_cgroup(tmp_path, str(64 * GIB), str(30 * GIB), str(2 * GIB))
    v = tp.train_preflight(processed, cgroup_dir=cg)
    assert not v.ok
    assert "memory.max" in " ".join(v.log_lines)


def test_prediction_math_exact(tmp_path: Path) -> None:
    processed = _make_dataset(tmp_path, 3, (64, 32))
    got = tp.predict_train_bytes(processed / "images")
    assert got is not None
    n, w, h, predicted = got
    assert (n, w, h) == (3, 64, 32)
    assert predicted == int(3 * 64 * 32 * 3 * tp.OVERHEAD_FACTOR) + tp.BASE_BYTES


def test_hidden_files_not_counted(tmp_path: Path) -> None:
    processed = _make_dataset(tmp_path, 2, (16, 16))
    (processed / "images" / ".DS_Store").touch()
    got = tp.predict_train_bytes(processed / "images")
    assert got is not None
    assert got[0] == 2
