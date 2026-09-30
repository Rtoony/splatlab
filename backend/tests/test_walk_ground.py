"""walk_ground.py on a synthetic outdoor scene: ground from the path, a parked car, a pond, sky floaters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Mesh-env only (dn-splatter-probe: plyfile + scipy + trimesh). The main test env has none of them, and a bare
# importorskip("plyfile") can be fooled by a namespace package on the path — import the real names or skip.
try:
    import scipy  # noqa: F401
    import trimesh
    from plyfile import PlyData, PlyElement
except ImportError:
    pytest.skip("walk_ground tests need the mesh env (plyfile, scipy, trimesh)", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import walk_ground as wg  # noqa: E402


def _scene(tmp: Path) -> Path:
    """A 20 m straight walk along +x at camera height 2.0 m over flat ground (z=0).
    A 2 x 1.5 m car block sits 3 m beside the path; a 2 x 2 m pond (no splats) 3 m on the other side;
    a haze of solid 'sky' splats 4-6 m up over everything (the thing that buried the old outdoor walks)."""
    rng = np.random.default_rng(0)
    job = tmp / "job"
    (job / "_preview").mkdir(parents=True)
    (job / "_spirula" / "cameras").mkdir(parents=True)
    xs = np.linspace(0, 20, 60)
    frames = [{"file_path": f"images/cam0/f{i:04d}.jpg",
               "transform_matrix": [[1, 0, 0, float(x)], [0, 1, 0, 0.0], [0, 0, 1, 2.0], [0, 0, 0, 1]]}
              for i, x in enumerate(xs)]
    (job / "_spirula" / "cameras" / "transforms.json").write_text(json.dumps({"frames": frames}))
    ground = np.c_[rng.uniform(-3, 23, 120_000), rng.uniform(-7, 7, 120_000), rng.normal(0, 0.01, 120_000)]
    pond = (np.abs(ground[:, 0] - 10) < 1) & (np.abs(ground[:, 1] + 3) < 1)
    ground = ground[~pond]
    car = np.c_[rng.uniform(4, 6, 20_000), rng.uniform(2.25, 3.75, 20_000), rng.uniform(0.3, 1.5, 20_000)]
    sky = np.c_[rng.uniform(-3, 23, 40_000), rng.uniform(-7, 7, 40_000), rng.uniform(4, 6, 40_000)]
    xyz = np.concatenate([ground, car, sky]).astype(np.float32)
    v = np.zeros(len(xyz), dtype=[(k, "f4") for k in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")])
    v["x"], v["y"], v["z"] = xyz.T
    v["opacity"] = 3.0                              # sigmoid -> 0.95: solid
    v["scale_0"] = v["scale_1"] = v["scale_2"] = np.log(0.03)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(job / "_preview" / "web.ply"))
    return job


def test_ground_car_pond_and_sky(tmp_path):
    job = _scene(tmp_path)
    assert wg.classify(wg.load_cameras(job), wg.load_solid_gaussians(job))["indoor"] is False
    m = wg.build(job)
    st, g, walk, obstacle, ground = m["stats"], m["grid"], m["walk"], m["obstacle"], m["ground"]
    assert st["cam_height_source"] == "learned" and abs(st["cam_height_used_m"] - 2.0) < 0.05
    assert st["path_walkable_frac"] == 1.0 and st["path_clear_frac"] == 1.0

    def cell(x, y):
        return int((x - g.x0) / wg.CELL), int((y - g.y0) / wg.CELL)

    assert abs(ground[cell(12, 0)]) < 0.05                                # ground under the path at z=0
    assert obstacle[cell(5, 3)] and not walk[cell(5, 3)]                  # the car blocks
    assert not walk[cell(10, -3)]                                         # the pond is fenced, not a floor
    assert walk[cell(15, 3)] and walk[cell(15, -3)]                       # open ground beside the path walks
    assert not walk[cell(12, 6.8)] or m["dist"][cell(12, 6.8)] <= wg.CORRIDOR_M   # corridor bound respected


def test_mesh_is_up_facing_in_y_up_and_the_spawn_stands_on_it(tmp_path):
    job = _scene(tmp_path)
    m = wg.build(job)
    verts, faces, walls = wg.build_mesh(m)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    n_floor = int(m["walk"].sum()) * 2
    floor_normals = mesh.face_normals[:n_floor]
    assert (floor_normals[:, 1] > 0.0).all()                              # every floor face points UP in Y-up
    assert (floor_normals[:, 1] > 0.9).mean() > 0.99                      # and is near-flat (cm noise on 15 cm cells)
    assert walls > 0 and len(faces) > n_floor
    seed, floor = wg.pick_spawn(m)
    assert abs(floor) < 0.05 and abs(seed[1] - (floor + 1.0)) < 1e-3   # seed is stored to 4 decimals
    locs, _, tri = mesh.ray.intersects_location([seed], [[0, -1, 0]])
    assert len(locs) and mesh.face_normals[tri[np.argmax(locs[:, 1])]][1] > 0.9


def test_implausible_camera_height_falls_back_to_the_owner_eye_height(tmp_path):
    job = _scene(tmp_path)
    frames = json.loads((job / "_spirula/cameras/transforms.json").read_text())
    for f in frames["frames"]:
        f["transform_matrix"][2][3] = 0.6                                 # the garden-passage case
    (job / "_spirula/cameras/transforms.json").write_text(json.dumps(frames))
    st = wg.build(job)["stats"]
    assert st["cam_height_used_m"] == wg.EYE_M == 1.68 and st["cam_height_source"] != "learned"


def test_navmesh_axes_match_the_walker(tmp_path):
    job = _scene(tmp_path)
    m = wg.build(job)
    nav = wg.navmesh(m, 0.0)
    g = m["grid"]
    assert nav["shape"] == [g.nx, g.ny] and len(nav["rows"]) == g.nx and len(nav["rows"][0]) == g.ny
    # the walker maps Y-up (X, Z) -> cell (i, j) with i = (X - ox)/cell, j = (Z - oz)/cell; Z = -y
    x, y = 5.0, 3.0                                                        # inside the car: not walkable
    i = int((x - nav["origin"][0]) / nav["cell"]); j = int((-y - nav["origin"][1]) / nav["cell"])
    assert nav["rows"][i][j] == "0"
    x, y = 15.0, -3.0
    i = int((x - nav["origin"][0]) / nav["cell"]); j = int((-y - nav["origin"][1]) / nav["cell"])
    assert nav["rows"][i][j] == "1"
