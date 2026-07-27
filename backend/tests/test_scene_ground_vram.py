"""VRAM admission for the semantic-ground lane.

Why this exists: SCENE_GROUND_VRAM_MB was a flat 6_000, but the pass does
`torch.tensor(d["gauss_emb"]).float().to(DEV)` — a rows x dim FLOAT32 tensor.
On the 1.46M-gaussian bicycle scene that is 2,351,565 x 1152 x 4 B = 10.09 GiB,
so the arbiter admitted the job against 6.4 GB of real headroom and it died
with `torch.OutOfMemoryError: Tried to allocate 10.09 GiB`. Admission has to be
measured against what the job will actually allocate.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


def _write_npz(path: Path, rows: int, dim: int) -> None:
    np.savez(path, gauss_emb=np.zeros((rows, dim), dtype=np.float16), seen=np.zeros(rows, np.uint8))


def test_shape_is_read_without_decompressing(tmp_path: Path) -> None:
    p = tmp_path / "gauss_emb.npz"
    _write_npz(p, 1000, 1152)
    assert splat_route._npz_member_shape(p, "gauss_emb") == (1000, 1152)
    # the point of the header peek: it must not need the payload
    assert splat_route._npz_member_shape(p, "seen") == (1000,)


def test_vram_matches_the_real_float32_tensor(tmp_path: Path) -> None:
    # The exact scene that OOM'd.
    rows, dim = 2_351_565, 1152
    p = tmp_path / "gauss_emb.npz"
    _write_npz(p, rows, dim)
    got = splat_route._scene_ground_vram_mb(p)
    tensor_mb = rows * dim * 4 / (1024 * 1024)
    assert tensor_mb == pytest.approx(10_332, rel=0.01)  # 10.09 GiB
    # Must exceed the tensor itself — admitting at exactly the tensor size
    # leaves nothing for the encoder or the allocator's slack.
    assert got > tensor_mb
    assert got == pytest.approx(14_417, rel=0.02)
    # ...and must be far above the old flat constant that caused the OOM.
    assert got > splat_route.SCENE_GROUND_VRAM_MB


def test_small_scene_still_reserves_the_floor(tmp_path: Path) -> None:
    p = tmp_path / "gauss_emb.npz"
    _write_npz(p, 5_000, 1152)
    # A tiny embedding must not shrink the reservation below what the text
    # encoder and workspace need regardless of scene size.
    assert splat_route._scene_ground_vram_mb(p) == splat_route.SCENE_GROUND_VRAM_MB


def test_unreadable_npz_falls_back_to_the_floor(tmp_path: Path) -> None:
    junk = tmp_path / "gauss_emb.npz"
    junk.write_bytes(b"not a zip")
    assert splat_route._npz_member_shape(junk, "gauss_emb") is None
    assert splat_route._scene_ground_vram_mb(junk) == splat_route.SCENE_GROUND_VRAM_MB
    assert splat_route._scene_ground_vram_mb(tmp_path / "missing.npz") == splat_route.SCENE_GROUND_VRAM_MB


def test_member_missing_from_npz(tmp_path: Path) -> None:
    p = tmp_path / "gauss_emb.npz"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("other.npy", b"x")
    assert splat_route._npz_member_shape(p, "gauss_emb") is None
