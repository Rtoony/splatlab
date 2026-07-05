"""langfield_align: byte-exact ply->ckpt row mapping (CPU-only, synthetic plys).

The invariant under test: /relevancy must serve rows in EXPORTED-PLY order,
because ns-export filters gaussians (Garden: 1,326,611 ckpt -> 1,321,833 ply)
and the raw gauss_emb/ckpt order silently mistints every splat after the first
dropped row. A partial or ambiguous match must yield NO map, never a wrong one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import langfield_align  # noqa: E402


def _write_ply(path: Path, xyz: np.ndarray, extra_props: int = 2) -> None:
    """Minimal binary_little_endian 3DGS-style ply: x/y/z + N filler floats."""
    n = xyz.shape[0]
    props = ["x", "y", "z"] + [f"f_dc_{i}" for i in range(extra_props)]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    rows = np.zeros((n, len(props)), dtype="<f4")
    rows[:, :3] = xyz
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rows.tobytes())


def _rand_xyz(n: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, 3)).astype(np.float32)


def test_read_ply_xyz_roundtrip(tmp_path):
    xyz = _rand_xyz(500)
    p = tmp_path / "splat.ply"
    _write_ply(p, xyz)
    got = langfield_align.read_ply_xyz(p)
    assert got.dtype == np.float32
    assert np.array_equal(got, xyz)


def test_map_recovers_export_filtering_and_order(tmp_path):
    ckpt = _rand_xyz(1000)
    keep = np.random.default_rng(3).permutation(1000)[:850]  # drop 150, reorder
    ply = ckpt[keep]
    m = langfield_align.build_index_map(ply, ckpt)
    assert m is not None
    assert np.array_equal(m, keep)  # exact recovery
    rel_ckpt = np.arange(1000, dtype=np.float32)
    assert np.array_equal(rel_ckpt[m], keep.astype(np.float32))  # relevancy rows follow


def test_no_partial_map_on_unmatched_rows():
    ckpt = _rand_xyz(100)
    ply = ckpt[:50].copy()
    ply[10] += 1e-3  # a transformed position: byte match must fail
    assert langfield_align.build_index_map(ply, ckpt) is None


def test_load_or_build_caches_and_identity(tmp_path):
    ckpt = _rand_xyz(200)
    lfdir = tmp_path / "_langfield"
    lfdir.mkdir()
    (tmp_path / "_preview").mkdir()
    _write_ply(tmp_path / "_preview" / "splat.ply", ckpt)  # exporter dropped nothing
    m = langfield_align.load_or_build_map(lfdir, ckpt)
    assert m is not None and np.array_equal(m, np.arange(200))
    assert (lfdir / langfield_align.MAP_FILENAME).is_file()
    # cached path (delete the ply: the cache must satisfy the second call alone)
    (tmp_path / "_preview" / "splat.ply").unlink()
    m2 = langfield_align.load_or_build_map(lfdir, ckpt)
    assert m2 is not None and np.array_equal(m2, m)


def test_load_or_build_no_ply_returns_none(tmp_path):
    lfdir = tmp_path / "_langfield"
    lfdir.mkdir()
    assert langfield_align.load_or_build_map(lfdir, _rand_xyz(10)) is None


def test_non_float_property_rejected(tmp_path):
    p = tmp_path / "bad.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nend_header\n"
    )
    with open(p, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(b"\x00" * 13)
    with pytest.raises(ValueError):
        langfield_align.read_ply_xyz(p)
