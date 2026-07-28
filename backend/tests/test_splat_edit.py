"""Splat-edit engine semantics, proven on a synthetic 3DGS PLY (mesh env)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

MESH_ENV_PYTHON = (
    Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"
)
ENGINE = Path(__file__).resolve().parents[1] / "mesh" / "splat_edit.py"

BUILD_PLY = """
import numpy as np
from plyfile import PlyData, PlyElement
import sys

out = sys.argv[1]
n = 100
rng = np.random.default_rng(7)
names = (["x", "y", "z", "nx", "ny", "nz"]
         + [f"f_dc_{i}" for i in range(3)]
         + [f"f_rest_{i}" for i in range(45)]
         + ["opacity"] + [f"scale_{i}" for i in range(3)]
         + [f"rot_{i}" for i in range(4)])
data = np.zeros(n, dtype=[(name, "<f4") for name in names])
data["x"] = np.linspace(-1, 1, n); data["y"] = 0.1; data["z"] = 0.2
for i in range(3):
    data[f"scale_{i}"] = -5.0
data["opacity"] = 3.0  # sigmoid(3) ~ 0.95: solidly opaque
# 10 low-opacity ghosts, 5 giant floaters, 3 far-field outliers:
data["opacity"][:10] = -4.0        # sigmoid(-4) ~ 0.018
data["scale_0"][10:15] = 1.0       # exp(1) = 2.7 units: huge
data["x"][15:18] = 50.0            # far from the median cloud
PlyData([PlyElement.describe(data, "vertex")], text=False).write(out)
print(n)
"""


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1",
    reason="set SPLATLAB_RUN_MESH_ENV_TESTS=1 for the real mesh-env splat test",
)
def test_splat_edit_ops_end_to_end(tmp_path: Path) -> None:
    if not MESH_ENV_PYTHON.is_file():
        pytest.skip("dn-splatter-probe env is unavailable")
    py = str(MESH_ENV_PYTHON)
    src = tmp_path / "splat.ply"
    assert _run([py, "-c", BUILD_PLY, str(src)]).returncode == 0

    # clean: ghosts + floaters + far-field all drop, everything else stays
    cleaned = tmp_path / "cleaned.ply"
    proc = _run([py, str(ENGINE), str(src), str(cleaned), "--op", "clean",
                 "--min-opacity", "0.05", "--max-scale", "1.0",
                 "--max-dist", "10.0"])
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.splitlines()[-1])
    assert report["removed"] == 18 and report["out_count"] == 82
    assert report["properties"] == 62  # full canonical schema preserved

    # crop_box: keep the left half
    cropped = tmp_path / "cropped.ply"
    proc = _run([py, str(ENGINE), str(src), str(cropped), "--op", "crop_box",
                 "--min=-2,-1,-1", "--max=0,1,1"])
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.splitlines()[-1])
    assert 0 < report["out_count"] < 100

    # transform: 2x scale doubles positions and shifts log scales by ln 2
    scaled = tmp_path / "scaled.ply"
    proc = _run([py, str(ENGINE), str(src), str(scaled), "--op", "transform",
                 "--scale", "2.0"])
    assert proc.returncode == 0, proc.stderr
    check = _run([py, "-c", f"""
import numpy as np
from plyfile import PlyData
a = PlyData.read({str(src)!r})["vertex"]
b = PlyData.read({str(scaled)!r})["vertex"]
assert np.allclose(b["x"], np.asarray(a["x"]) * 2.0, atol=1e-5)
assert np.allclose(b["scale_0"], np.asarray(a["scale_0"]) + np.log(2.0), atol=1e-5)
assert [p.name for p in b.properties] == [p.name for p in a.properties]
print("ok")
"""])
    assert check.returncode == 0, check.stderr

    # an edit that would empty the splat is refused, loudly
    empty = tmp_path / "empty.ply"
    proc = _run([py, str(ENGINE), str(src), str(empty), "--op", "crop_box",
                 "--min=100,100,100", "--max=101,101,101"])
    assert proc.returncode == 1 and "refusing" in proc.stderr
    assert not empty.exists()
