"""First tests for mesh/object_texture.py — the rasterize→bake→dilate chain and
the coverage-honesty contract (`coverage` = shipped atlas, `coverage_rasterized`
= pre-dilation rasterizer hits).

object_texture runs in the dn-splatter-probe env in production; its heavy deps
(trimesh, plyfile, xatlas) are deliberately absent from this app's venv. The
functions under test are pure numpy/scipy, so the module is imported with
empty import-time placeholders for deps these tests never exercise — nothing
stubbed is ever called. The opt-in end-to-end test at the bottom runs the real
script through the real env (SPLATLAB_RUN_MESH_ENV_TESTS=1), the same way
test_blender_headless_integration gates real-binary work.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

for _name, _attrs in (("trimesh", ()), ("plyfile", ("PlyData",))):
    try:
        __import__(_name)
    except ImportError:
        _mod = types.ModuleType(_name)
        for _a in _attrs:
            setattr(_mod, _a, object)
        sys.modules[_name] = _mod

import object_texture as ot  # noqa: E402

MESH_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"


# ---------------------------------------------------------------------------
# dilate_atlas — returns (texture, shipped_mask)
# ---------------------------------------------------------------------------

def test_dilate_atlas_returns_texture_and_shipped_mask():
    size = 16
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    tex[8, 8] = (200, 100, 50)
    mask[8, 8] = True
    tex_in, mask_in = tex.copy(), mask.copy()

    out, shipped = ot.dilate_atlas(tex, mask, passes=2)

    # inputs are not mutated — the caller's rasterized mask stays meaningful
    assert np.array_equal(tex, tex_in)
    assert np.array_equal(mask, mask_in)
    # shipped is a strict superset of the rasterized mask
    assert shipped[mask].all()
    assert shipped.sum() > mask.sum()
    # 2 passes of 4-neighbour growth = Manhattan radius 2, no further
    assert shipped[8, 10] and not shipped[8, 11]
    # bleed carries the neighbour colour; everything outside shipped stays black
    assert tuple(out[8, 9]) == (200, 100, 50)
    assert not out[~shipped].any()


def test_dilate_atlas_full_mask_is_noop():
    rng = np.random.default_rng(7)
    tex = rng.integers(1, 255, size=(8, 8, 3), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    out, shipped = ot.dilate_atlas(tex, mask, passes=4)
    assert np.array_equal(out, tex)
    assert shipped.all()


def test_dilate_atlas_coverage_semantics_shipped_geq_rasterized():
    # The report contract: `coverage` (shipped) >= `coverage_rasterized`.
    size = 32
    tex = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    tex[4:9, 4:9] = 120
    mask[4:9, 4:9] = True
    out, shipped = ot.dilate_atlas(tex, mask, passes=3)
    assert float(shipped.mean()) >= float(mask.mean())


# ---------------------------------------------------------------------------
# _face_slabs — splits between faces, never inside one
# ---------------------------------------------------------------------------

def test_face_slabs_covers_every_face_once_within_budget():
    npix = np.array([4, 4, 10, 3, 1], dtype=np.int64)
    ends = np.cumsum(npix)
    slabs = list(ot._face_slabs(ends, budget=8))

    # contiguous, complete, in order
    assert slabs[0][0] == 0
    assert slabs[-1][1] == len(ends)
    for (_, b), (c, _) in zip(slabs, slabs[1:]):
        assert b == c
    # each slab fits the budget unless it is a single over-budget face,
    # which still gets its own slab rather than stalling the generator
    for lo, hi in slabs:
        base = int(ends[lo - 1]) if lo else 0
        texels = int(ends[hi - 1]) - base
        assert texels <= 8 or hi - lo == 1
    # the 10-texel face was isolated
    assert (2, 3) in slabs


# ---------------------------------------------------------------------------
# _rasterize_atlas — every on-atlas face owns at least one texel
# ---------------------------------------------------------------------------

def test_rasterize_atlas_guarantees_texel_for_subpixel_face():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    # UV triangle entirely inside pixel (2,2) but containing no pixel centre
    uv_px = np.array([[2.1, 2.1], [2.4, 2.1], [2.2, 2.4]], dtype=np.float32)

    px, py, pos, nrm = ot._rasterize_atlas(verts, faces, uv_px, size=8, vnormals=None)

    assert len(px) == 1
    assert (int(px[0]), int(py[0])) == (2, 2)
    # fallback samples the face centroid in scene space
    np.testing.assert_allclose(pos[0], verts.mean(axis=0), atol=1e-5)
    assert nrm is None


def test_rasterize_atlas_empty_faces_returns_nones():
    out = ot._rasterize_atlas(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
        np.zeros((0, 2), np.float32), size=8, vnormals=None)
    assert out == (None, None, None, None)


# ---------------------------------------------------------------------------
# bake_texture — mask covers exactly the mapped region, colours follow gaussians
# ---------------------------------------------------------------------------

def _quad_bake(size: int = 32):
    # unit quad in 3D, UV-mapped onto the left half of the atlas
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[0, 0], [0.5, 0], [0.5, 1], [0, 1]], dtype=np.float32)
    # gaussian grid over the quad: left half red, right half blue
    g = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    gxx, gyy = np.meshgrid(g, g)
    gx = np.stack([gxx.ravel(), gyy.ravel(), np.zeros(gxx.size, np.float32)], axis=1)
    gc = np.where(gx[:, :1] < 0.5,
                  np.array([[1.0, 0.1, 0.1]], np.float32),
                  np.array([[0.1, 0.1, 1.0]], np.float32))
    tex, mask = ot.bake_texture(verts, faces, uvs, gx, gc, size)
    return tex, mask


def test_bake_texture_masks_only_the_mapped_region():
    size = 32
    tex, mask = _quad_bake(size)
    assert tex is not None
    # the quad maps to the left ~half of the atlas
    assert 0.40 < float(mask.mean()) < 0.60
    # unmasked texels were never written
    assert not tex[~mask].any()
    # colour follows the gaussians: u≈0.11 → x≈0.23 (red side),
    # u≈0.44 → x≈0.87 (blue side)
    red = tex[16, 3].astype(int)
    blue = tex[16, 13].astype(int)
    assert red[0] > red[2] + 50
    assert blue[2] > blue[0] + 50


def test_bake_then_dilate_shipped_mask_is_superset():
    tex, mask = _quad_bake(32)
    out, shipped = ot.dilate_atlas(tex, mask, passes=3)
    assert shipped[mask].all()
    assert float(shipped.mean()) > float(mask.mean())
    assert not out[~shipped].any()


# ---------------------------------------------------------------------------
# Opt-in end-to-end: the real script through the real env, coverage keys proven
# ---------------------------------------------------------------------------

def _write_cube_mesh_ply(path: Path) -> None:
    v = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    lines = ["ply", "format ascii 1.0", f"element vertex {len(v)}",
             "property float x", "property float y", "property float z",
             f"element face {len(f)}", "property list uchar int vertex_indices",
             "end_header"]
    lines += [f"{a} {b} {c}" for a, b, c in v]
    lines += [f"3 {a} {b} {c}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n")


def _write_splat_ply(path: Path, n_per_face: int = 300) -> None:
    rng = np.random.default_rng(11)
    pts = []
    for axis in range(3):
        for level in (0.0, 1.0):
            p = rng.random((n_per_face, 3), dtype=np.float32)
            p[:, axis] = level
            pts.append(p)
    xyz = np.concatenate(pts)
    dc = 0.5 / ot.C0  # rgb == 1.0 after the SH DC transform
    lines = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}",
             "property float x", "property float y", "property float z",
             "property float opacity",
             "property float f_dc_0", "property float f_dc_1",
             "property float f_dc_2", "end_header"]
    lines += [f"{x:.6f} {y:.6f} {z:.6f} 5.0 {dc:.6f} 0.0 0.0"
              for x, y, z in xyz]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1",
    reason="opt-in: runs the real object_texture.py in the dn-splatter-probe env",
)
def test_end_to_end_report_carries_both_coverage_keys(tmp_path):
    if not MESH_ENV_PYTHON.is_file():
        pytest.skip("dn-splatter-probe env not present on this machine")
    mesh_ply = tmp_path / "mesh.ply"
    splat_ply = tmp_path / "splat.ply"
    out_glb = tmp_path / "out.glb"
    report_path = tmp_path / "report.json"
    _write_cube_mesh_ply(mesh_ply)
    _write_splat_ply(splat_ply)

    script = Path(__file__).resolve().parents[1] / "mesh" / "object_texture.py"
    proc = subprocess.run(
        [str(MESH_ENV_PYTHON), str(script), str(mesh_ply), str(splat_ply),
         str(out_glb), "--no-crop", "--no-remesh", "--no-reconstruct",
         "--texture-size", "128", "--report", str(report_path)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]

    report = json.loads(report_path.read_text())
    tex = report["texture"]
    assert tex["baked"] is True
    assert set(tex) >= {"coverage", "coverage_rasterized"}
    # shipped coverage is the dilated atlas; it can never read below the
    # rasterized fraction, and on a well-unwrapped cube both are meaningful
    assert tex["coverage"] >= tex["coverage_rasterized"] > 0.0
    assert out_glb.is_file() and report["glb_bytes"] > 0
