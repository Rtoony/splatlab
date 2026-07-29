"""surface_patch_core: clustering, seeded RANSAC (refusals BY NUMBER),
paint-bounded footprints, and quad-grid meshing — pure numpy/scipy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

import surface_patch_core as spc  # noqa: E402


def _plane_points(n=600, extent=1.0, z=0.3, noise=0.005, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, extent, size=(n, 2))
    zs = z + rng.normal(0, noise, size=n)
    return np.column_stack([xy, zs])


# ── clustering ──────────────────────────────────────────────────────────────────


def test_two_walls_become_two_clusters():
    wall_a = _plane_points(seed=1)
    wall_b = _plane_points(seed=2) + np.array([0.0, 0.0, 1.0])  # 1.0 apart
    clusters = spc.grid_clusters(np.vstack([wall_a, wall_b]), cell_units=0.1)
    assert len(clusters) == 2
    assert {len(c) for c in clusters} == {600}
    # Index arrays partition the input exactly.
    assert sorted(np.concatenate(clusters).tolist()) == list(range(1200))


def test_empty_input_clusters_to_nothing():
    assert spc.grid_clusters(np.zeros((0, 3))) == []


# ── plane fitting ───────────────────────────────────────────────────────────────


def test_ransac_finds_the_plane_through_noise():
    plane = _plane_points(n=500, noise=0.01)
    outliers = np.column_stack([
        np.random.default_rng(7).uniform(0, 1, size=(25, 2)),
        np.full(25, 2.0)])
    fit = spc.fit_plane_ransac(np.vstack([plane, outliers]), dist_units=0.05)
    assert abs(fit["normal"][2]) > 0.999          # ~ +-z
    assert len(fit["inliers"]) >= 490
    assert (fit["inliers"] < 500).all()           # no outlier crept in
    assert fit["rms_units"] < 0.02


def test_ransac_is_deterministic():
    pts = _plane_points()
    a = spc.fit_plane_ransac(pts)
    b = spc.fit_plane_ransac(pts)
    assert np.array_equal(a["inliers"], b["inliers"])
    assert a["rms_units"] == b["rms_units"]
    assert np.allclose(a["normal"], b["normal"])


def test_a_curved_surface_is_refused_by_inlier_fraction():
    """The curvature detector is INLIER FRACTION, not rms: rms-over-inliers
    is capped at ~0.577x the band and cannot catch a curve (this test, run
    against the rms-only design, is what exposed that).

    The sample must be a 2-D SURFACE — theta and y drawn independently. The
    first version zipped linspace(theta) with linspace(y), which is a 1-D
    HELIX, and a helix admits a good plane (its linear ramp cancels the
    sinusoid's slope — caught live by the commit gate). A true half-cylinder
    shell's best 0.05-band plane explains only ~40% of the points."""
    rng = np.random.default_rng(9)
    theta = rng.uniform(0, np.pi, 2000)
    height = rng.uniform(0, 1.0, 2000)
    cylinder = np.column_stack([0.5 * np.cos(theta), height,
                                0.5 * np.sin(theta)])
    with pytest.raises(spc.SurfacePatchError,
                       match="not planar.*paint flatter"):
        spc.fit_plane_ransac(cylinder, dist_units=0.05, min_inlier_frac=0.7)


def test_a_flat_wall_passes_the_inlier_fraction_floor():
    fit = spc.fit_plane_ransac(_plane_points(noise=0.01), dist_units=0.05,
                               min_inlier_frac=0.7)
    assert fit["inlier_frac"] >= 0.95
    assert fit["rms_units"] < 0.02


def test_collinear_points_are_refused():
    line = np.column_stack([np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)])
    with pytest.raises(spc.SurfacePatchError, match="collinear|degenerate"):
        spc.fit_plane_ransac(line)


def test_too_few_points_are_refused_with_the_count():
    with pytest.raises(spc.SurfacePatchError, match="only 2 points"):
        spc.fit_plane_ransac(np.zeros((2, 3)))


# ── plane basis ─────────────────────────────────────────────────────────────────


def test_plane_basis_is_orthonormal_right_handed_and_deterministic():
    for normal in ([0, 0, 1], [1, 0, 0], [0.3, -0.7, 0.648]):
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        u, v = spc.plane_basis(n)
        u2, v2 = spc.plane_basis(n)
        assert np.allclose(u, u2) and np.allclose(v, v2)
        assert abs(np.linalg.norm(u) - 1) < 1e-12
        assert abs(u @ v) < 1e-12 and abs(u @ n) < 1e-12
        # cross(u, v) == n is what makes the mesh winding face the normal.
        assert np.allclose(np.cross(u, v), n, atol=1e-12)


# ── footprint ───────────────────────────────────────────────────────────────────


def _annulus_uv(inner=0.2, outer=0.5, n=4000, seed=3):
    rng = np.random.default_rng(seed)
    r = np.sqrt(rng.uniform(inner ** 2, outer ** 2, size=n))
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)]) + outer


def test_footprint_closes_interior_gaps_but_not_beyond_the_paint():
    uv = _annulus_uv()
    grid, (i0, j0) = spc.footprint_cells(uv, cell_units=0.05, close_cells=3,
                                         pad_cells=1)
    height, width = grid.shape
    centre = (int(np.floor(0.5 / 0.05)) - i0, int(np.floor(0.5 / 0.05)) - j0)
    assert grid[centre]                        # the hole was closed
    # ...but the grid never grew past the painted bbox + pad.
    ij = np.floor(uv / 0.05).astype(np.int64)
    assert height == ij[:, 0].max() - ij[:, 0].min() + 3   # +2*pad +1
    assert width == ij[:, 1].max() - ij[:, 1].min() + 3


def test_footprint_keeps_only_the_largest_component():
    blob = np.random.default_rng(5).uniform(0, 0.4, size=(500, 2))
    speck = np.array([[3.0, 3.0]])
    grid, (i0, j0) = spc.footprint_cells(np.vstack([blob, speck]),
                                         cell_units=0.05, close_cells=0,
                                         pad_cells=0)
    speck_cell = (int(np.floor(3.0 / 0.05)) - i0, int(np.floor(3.0 / 0.05)) - j0)
    assert not grid[speck_cell]
    assert grid.sum() > 0


# ── meshing ─────────────────────────────────────────────────────────────────────


def test_cells_to_mesh_shares_vertices_and_faces_the_normal():
    grid = np.ones((3, 4), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    u, v = spc.plane_basis(normal)
    verts, faces = spc.cells_to_mesh(grid, (0, 0), 0.1,
                                     np.array([0.0, 0.0, 0.3]), u, v)
    assert len(verts) == 4 * 5                 # shared corners, not 4/cell
    assert len(faces) == 2 * 12
    for tri in faces:
        a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        assert np.cross(b - a, c - a) @ normal > 0
    # Every vertex sits on the plane.
    assert np.allclose(verts[:, 2], 0.3)


def test_empty_footprint_refuses_to_mesh():
    u, v = spc.plane_basis(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(spc.SurfacePatchError, match="empty footprint"):
        spc.cells_to_mesh(np.zeros((2, 2), bool), (0, 0), 0.1,
                          np.zeros(3), u, v)


# ── UVs ─────────────────────────────────────────────────────────────────────────


def test_planar_uvs_are_normalized_and_axis_aligned():
    normal = np.array([0.0, 0.0, 1.0])
    u, v = spc.plane_basis(normal)
    grid = np.ones((2, 2), bool)
    verts, _ = spc.cells_to_mesh(grid, (0, 0), 0.5, np.zeros(3), u, v)
    uv = spc.planar_uvs(verts, np.zeros(3), u, v)
    assert uv.min() == 0.0 and uv.max() == 1.0
    assert uv.shape == (len(verts), 2)


def test_surface_patches_cli_end_to_end_mesh_env(tmp_path):
    """Opt-in e2e of the REAL CLI in the mesh env: synthetic structure paint
    on a sparse wall -> tagged patch ply + capture-coloured preview GLB."""
    import os
    import subprocess
    if os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1":
        pytest.skip("set SPLATLAB_RUN_MESH_ENV_TESTS=1 for the mesh-env e2e")
    mesh_python = Path.home() / "miniconda3/envs/dn-splatter-probe/bin/python"
    if not mesh_python.is_file():
        pytest.skip("dn-splatter-probe env is unavailable")
    import json as _json

    backend = Path(__file__).resolve().parents[1]
    lfdir = tmp_path / "_langfield"
    lfdir.mkdir()
    # A sparse vertical wall (x-z plane at y=0.5), painted as structure.
    rng = np.random.default_rng(11)
    wall = np.column_stack([rng.uniform(0, 2, 900), np.full(900, 0.5),
                            rng.uniform(0, 1.2, 900)]).astype(np.float32)
    np.save(lfdir / "class_xyz_ab12cd34.npy", wall)
    (lfdir / "class_labels.json").write_text(_json.dumps([
        {"id": "ab12cd34", "class_id": "structure", "count": 900}]))

    # A minimal ascii splat.ply for the colour bake (solid white gaussians).
    n = 400
    pts = np.column_stack([rng.uniform(0, 2, n), np.full(n, 0.5),
                           rng.uniform(0, 1.2, n)])
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in
                        ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
                         "opacity"))
              + "end_header\n")
    rows = "".join(f"{x} {y} {z} 1.0 1.0 1.0 4.0\n" for x, y, z in pts)
    splat = tmp_path / "splat.ply"
    splat.write_text(header + rows)

    out_dir = tmp_path / "surfaces"
    proc = subprocess.run(
        [str(mesh_python), str(backend / "mesh" / "surface_patches.py"),
         str(lfdir), str(splat), str(backend / "class_taxonomy.json"),
         str(out_dir), "--min-cluster-points", "100",
         "--texture-size", "256"],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = _json.loads((out_dir / "surface_patches.json").read_text())
    assert report["provenance"] == "paint-synthesized"
    assert len(report["patches"]) >= 1
    assert report["patches"][0]["plane"]["rms_units"] < 0.05
    assert report["texture"]["baked"] is True
    assert (out_dir / "patch_00.ply").is_file()
    assert (out_dir / "surfaces_preview.glb").is_file()
    # The provenance tag rides INSIDE the ply header.
    assert b"generative" in (out_dir / "patch_00.ply").read_bytes()[:2048].lower()


def test_tile_packing_keeps_patches_disjoint_in_uv():
    charts = [np.array([[0.0, 0.0], [1.0, 1.0]]) for _ in range(5)]
    packed = spc.pack_tiles(charts, inset=0.01)
    assert len(packed) == 5
    boxes = [(p.min(axis=0), p.max(axis=0)) for p in packed]
    for a in range(5):
        assert (boxes[a][0] >= -1e-9).all() and (boxes[a][1] <= 1 + 1e-9).all()
        for b in range(a + 1, 5):
            lo = np.maximum(boxes[a][0], boxes[b][0])
            hi = np.minimum(boxes[a][1], boxes[b][1])
            assert (lo >= hi).any(), (a, b)    # no overlap in at least one axis
