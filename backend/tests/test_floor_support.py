"""The reconciled floor-support metric, proven on synthetic rooms.

Opt-in (mesh env: open3d + trimesh + scipy), same gate as the other
mesh-env lanes. Pins the two semantics that motivated the 2026-07-28
reconciliation:

  1. a genuine floor hole under a ceiling IS caught (the failure mode the
     old lowest-surface design existed for), and
  2. standable furniture over occluded floor is SUPPORT, not a hole (the
     false-negative the old design produced on the bonsai perimeter).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

MESH_ENV_PYTHON = (
    Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"
)

SCRIPT = """
import sys
sys.path.insert(0, {mesh_dir!r})
import json
import numpy as np
import trimesh
import floor_support

def room(hole_radius=0.0, furniture=False, drop_floor_under_furniture=False):
    parts = []
    floor = trimesh.creation.box(extents=(10, 0.1, 10))
    if hole_radius > 0.0:
        # remove a disc of floor: rebuild floor from a grid of tiles minus disc
        tiles = []
        for i in range(20):
            for j in range(20):
                cx, cz = -4.75 + i * 0.5, -4.75 + j * 0.5
                if np.hypot(cx - 2.0, cz - 2.0) < hole_radius:
                    continue
                t = trimesh.creation.box(extents=(0.5, 0.1, 0.5))
                t.apply_translation((cx, 0.0, cz))
                tiles.append(t)
        parts.append(trimesh.util.concatenate(tiles))
    else:
        parts.append(floor)
    ceiling = trimesh.creation.box(extents=(10, 0.1, 10))
    ceiling.apply_translation((0, 3.0, 0))
    parts.append(ceiling)
    for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        wall = trimesh.creation.box(
            extents=(0.1, 3.0, 10) if sx else (10, 3.0, 0.1))
        wall.apply_translation((sx * 4.95, 1.5, sz * 4.95))
        parts.append(wall)
    if furniture:
        desk = trimesh.creation.box(extents=(2.0, 0.1, 2.0))
        desk.apply_translation((2.0, 0.9, 2.0))  # desktop at 0.9 m
        parts.append(desk)
        if drop_floor_under_furniture:
            pass  # combined with hole_radius covering the same spot
    m = trimesh.util.concatenate(parts)
    return np.asarray(m.vertices), np.asarray(m.faces)

results = {{}}
v, f = room()
results["intact"] = floor_support.measure_floor_support(v, f, "Y")
v, f = room(hole_radius=1.2)
results["hole_under_ceiling"] = floor_support.measure_floor_support(v, f, "Y")
v, f = room(furniture=True)
results["furniture_over_floor"] = floor_support.measure_floor_support(v, f, "Y")
v, f = room(hole_radius=1.2, furniture=True)
results["furniture_patching_hole"] = floor_support.measure_floor_support(v, f, "Y")
print(json.dumps({{k: {{"coverage": r["coverage"],
                        "largest_gap_frac": r["largest_gap_frac"],
                        "ground_band_coverage": r["ground_band_coverage"]}}
                  for k, r in results.items()}}))
"""


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1",
    reason="set SPLATLAB_RUN_MESH_ENV_TESTS=1 for the real mesh-env floor test",
)
def test_floor_support_semantics() -> None:
    if not MESH_ENV_PYTHON.is_file():
        pytest.skip("dn-splatter-probe env is unavailable")
    mesh_dir = str(Path(__file__).resolve().parents[1] / "mesh")
    proc = subprocess.run(
        [str(MESH_ENV_PYTHON), "-c", SCRIPT.format(mesh_dir=mesh_dir)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    r = json.loads(proc.stdout.splitlines()[-1])

    # 1. An intact room is fully supported.
    assert r["intact"]["coverage"] >= 0.99

    # 2. A floor hole under a ceiling IS caught: coverage drops and the void
    #    is contiguous. (The ceiling cannot fake a floor — rays start below it.)
    assert r["hole_under_ceiling"]["coverage"] < 0.99
    assert r["hole_under_ceiling"]["largest_gap_frac"] > 0.03

    # 3. Standable furniture over intact floor changes nothing (the up-cast
    #    still finds the floor beneath it, so both metrics agree here).
    assert r["furniture_over_floor"]["coverage"] >= 0.99

    # 4. THE semantic change, pinned: a standable slab over a floor hole is
    #    SUPPORT (you land on the desk, you do not fall through the world)...
    assert r["furniture_patching_hole"]["coverage"] > r["hole_under_ceiling"]["coverage"]
    #    ...and THIS is the case the old strict metric punished — the lowest
    #    surface in those columns is the desk, far above the ground band:
    assert (r["furniture_patching_hole"]["ground_band_coverage"]
            < r["furniture_patching_hole"]["coverage"])
