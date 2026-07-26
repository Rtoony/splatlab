"""P4 class-textured ground: procedural tiles (deterministic, bounded) and the
world-space sampler. The full ground_mesh_build vote + ground_texture bake run
as opt-in e2e through the probe env (open3d/trimesh live there)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "mesh"))
import class_textures as cx  # noqa: E402

MESH_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"
TAXONOMY = BACKEND / "class_taxonomy.json"


def _grass():
    return json.loads(TAXONOMY.read_text())["classes"][0]


def test_tile_is_deterministic_and_bounded() -> None:
    a = cx.tile_for(_grass(), 64)
    b = cx.tile_for(_grass(), 64)
    assert np.array_equal(a, b)  # seeded from the class id — receipt-friendly
    assert a.shape == (64, 64, 3) and a.dtype == np.uint8
    # stays recognisably the class colour: green channel dominates
    mean = a.reshape(-1, 3).mean(axis=0)
    assert mean[1] > mean[0] and mean[1] > mean[2]


def test_different_classes_get_different_tiles() -> None:
    classes = json.loads(TAXONOMY.read_text())["classes"]
    grass = cx.tile_for(classes[0], 64)
    dirt = cx.tile_for(next(c for c in classes if c["id"] == "dirt"), 64)
    assert not np.array_equal(grass, dirt)


def test_sample_world_wraps_and_is_position_stable() -> None:
    tile = cx.tile_for(_grass(), 64)
    x = np.array([0.0, 1.0, 100.0])
    y = np.array([0.0, 0.5, -3.25])
    s1 = cx.sample_world(tile, x, y, texels_per_unit=32.0)
    s2 = cx.sample_world(tile, x, y, texels_per_unit=32.0)
    assert np.array_equal(s1, s2)
    # world-periodic: one full tile period apart samples identically
    period = 64 / 32.0
    s3 = cx.sample_world(tile, x + period, y, texels_per_unit=32.0)
    assert np.array_equal(s1, s3)


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1",
    reason="opt-in: runs ground_mesh_build + ground_texture in the probe env",
)
def test_end_to_end_classed_ground(tmp_path: Path) -> None:
    if not MESH_ENV_PYTHON.is_file():
        pytest.skip("dn-splatter-probe env not present")
    rng = np.random.default_rng(9)
    # two flat patches: grass x<1, pavement x>1 — enough cells to survive the
    # connectivity filter
    n = 4000
    xy = rng.random((n, 2), dtype=np.float32) * np.array([2.0, 1.0], np.float32)
    xyz = np.column_stack([xy, rng.normal(0, 0.005, n).astype(np.float32)])
    class_ids = ["grass", "pavement"]
    class_rel = np.zeros((n, 2), np.float16)
    grass_side = xyz[:, 0] < 1.0
    class_rel[grass_side, 0] = 0.9
    class_rel[~grass_side, 1] = 0.9
    gg = tmp_path / "ground_gaussians.npz"
    np.savez_compressed(gg, xyz=xyz, rel=np.full(n, 0.9, np.float32),
                        seen=np.ones(n, bool), class_rel=class_rel,
                        class_ids=np.array(class_ids))

    out_dir = tmp_path / "ground"
    r1 = subprocess.run(
        [str(MESH_ENV_PYTHON), str(BACKEND / "mesh" / "ground_mesh_build.py"),
         str(gg), str(out_dir), "--cell-units", "0.1"],
        capture_output=True, text=True, timeout=300)
    assert r1.returncode == 0, r1.stderr[-1500:]
    cells = np.load(out_dir / "ground_class_cells.npz")
    build = json.loads((out_dir / "ground_mesh_build.json").read_text())
    assert set(build["class_cells"]) == {"grass", "pavement"}
    assert (cells["class_idx"] >= 0).all()

    out_glb = out_dir / "ground_classed.glb"
    r2 = subprocess.run(
        [str(MESH_ENV_PYTHON), str(BACKEND / "mesh" / "ground_texture.py"),
         str(out_dir / "ground_mesh_raw.ply"),
         str(out_dir / "ground_class_cells.npz"),
         str(TAXONOMY), str(out_glb)],
        capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr[-1500:]
    report = json.loads((out_dir / "ground_texture_report.json").read_text())
    assert report["provenance"] == "captured-geometry/class-textured"
    fr = report["class_texel_fractions"]
    assert fr["grass"] > 0.2 and fr["pavement"] > 0.2  # two clean partitions
    assert out_glb.is_file() and (out_dir / "ground_atlas.png").is_file()
