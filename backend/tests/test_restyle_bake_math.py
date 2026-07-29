"""Restyle bake: the pure math helpers (app env) and an opt-in mesh-env
end-to-end run of the real CLI on real textured GLBs.

The math tests pin the bake to the walker preview's conventions
(frontend/src/lib/world-restyle.ts): the texels-per-unit formula, the
triplanar plane selection, and linear-space tint multiplication. The e2e
gates behind SPLATLAB_RUN_MESH_ENV_TESTS=1 like every trimesh-needing test.
"""

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
sys.path.insert(0, str(BACKEND))

import class_textures as ctx  # noqa: E402
import restyle_bake as rb  # noqa: E402

MESH_ENV_PYTHON = (
    Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"
)


def test_srgb_round_trip_is_exact_for_all_bytes() -> None:
    b = np.arange(256, dtype=np.uint8)
    assert (rb.linear_to_srgb(rb.srgb_to_linear(b)) == b).all()


def test_texels_per_glb_unit_matches_frontend_formula() -> None:
    # unitsPerTile = material_scale_m * unitsPerMetre; tpu = tile_px / that.
    assert rb.texels_per_glb_unit(1.5, 1.0, 512) == pytest.approx(512 / 1.5)
    assert rb.texels_per_glb_unit(0.5, 1.9184, 512) == pytest.approx(
        512 / (0.5 * 1.9184))
    # The shader's own 1e-4 floor on unitsPerTile.
    assert rb.texels_per_glb_unit(0.0, 1.0, 512) == pytest.approx(512 / 1e-4)


def test_triplanar_axis_selection_matches_the_shader() -> None:
    size = 8
    tile = np.zeros((size, size, 3), np.uint8)
    tile[..., 0] = (np.arange(size) * 10)[None, :]  # R encodes u (column)
    tile[..., 1] = (np.arange(size) * 10)[:, None]  # G encodes v (row)
    pos = np.array([[1.0, 2.0, 3.0]], np.float32)

    # Plane selection per the injected shader: X->(z,y), Y->(x,z), Z->(x,y).
    cases = [
        (np.array([[1.0, 0.0, 0.0]]), ctx.sample_world(tile, pos[:, 2], pos[:, 1], 1.0)),
        (np.array([[0.0, 1.0, 0.0]]), ctx.sample_world(tile, pos[:, 0], pos[:, 2], 1.0)),
        (np.array([[0.0, 0.0, 1.0]]), ctx.sample_world(tile, pos[:, 0], pos[:, 1], 1.0)),
        (np.array([[0.0, -1.0, 0.0]]), ctx.sample_world(tile, pos[:, 0], pos[:, 2], 1.0)),
    ]
    for nrm, expected in cases:
        got = ctx.sample_world_triplanar(tile, pos, nrm.astype(np.float32), 1.0)
        # The epsilon weight leaks ~1e-5 of the other planes — sub-integer.
        assert np.allclose(got, expected.astype(np.float32), atol=0.1), nrm

    # A 45-degree normal blends the two planes at half weight each.
    nrm = np.array([[1.0, 0.0, 1.0]], np.float32)
    got = ctx.sample_world_triplanar(tile, pos, nrm, 1.0)
    cx = ctx.sample_world(tile, pos[:, 2], pos[:, 1], 1.0).astype(np.float32)
    cz = ctx.sample_world(tile, pos[:, 0], pos[:, 1], 1.0).astype(np.float32)
    assert np.allclose(got, 0.5 * cx + 0.5 * cz, atol=0.1)


def test_tint_multiplies_in_linear_space() -> None:
    arr = np.full((2, 2, 3), 128, np.uint8)
    assert (rb.tint_atlas(arr, "#ffffff") == arr).all()

    half = rb.tint_atlas(arr, "#808080")
    expected = rb.linear_to_srgb(
        rb.srgb_to_linear(np.uint8(128)) * rb.srgb_to_linear(np.uint8(128)))
    assert (half == expected).all()
    # And it is NOT the naive byte multiply (128*128/255 = 64).
    assert int(half[0, 0, 0]) != 64


def test_black_floor_clamp_keeps_painted_texels_nonzero() -> None:
    rgb = np.zeros((2, 2, 3), np.uint8)
    painted = np.array([[True, False], [False, True]])
    out = rb.floor_clamp_painted(rgb, painted)
    assert (out[0, 0] == 1).all() and (out[1, 1] == 1).all()
    assert (out[0, 1] == 0).all() and (out[1, 0] == 0).all()

    # A painted texel with any colour at all is left untouched.
    rgb2 = np.zeros((1, 1, 3), np.uint8)
    rgb2[0, 0, 1] = 7
    assert (rb.floor_clamp_painted(rgb2, np.ones((1, 1), bool)) == rgb2).all()


def test_bake_plan_refuses_empty_and_filters_slugs() -> None:
    doc = {
        "elements": {
            "a": {"tint": "#ff0000"},
            "b": {"material": "lava", "material_scale": 2.0},
            "c": {},
        },
        "lighting": {"preset": "dungeon", "intensity": 1},
    }
    plan = rb.bake_plan(doc)
    assert [p["slug"] for p in plan] == ["a", "b"]
    assert plan[1]["material_scale"] == 2.0
    assert plan[0]["material_scale"] == 1.0  # the preview's `?? 1` default

    assert [p["slug"] for p in rb.bake_plan(doc, {"b"})] == ["b"]

    with pytest.raises(ValueError, match="nothing to bake"):
        rb.bake_plan({"elements": {}, "lighting": {"preset": "dungeon"}})
    with pytest.raises(ValueError, match="nothing to bake"):
        rb.bake_plan(doc, {"c"})  # lighting/empty entries never qualify


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1",
    reason="set SPLATLAB_RUN_MESH_ENV_TESTS=1 for the real mesh-env bake test",
)
def test_mesh_env_restyle_bake_end_to_end(tmp_path: Path) -> None:
    """Two textured crates -> material bake (lava) + tint bake (green) via the
    real CLI; staged atlases must carry the new surfaces, geometry unchanged."""
    if not MESH_ENV_PYTHON.is_file():
        pytest.skip("dn-splatter-probe env is unavailable")

    job_dir = tmp_path / "splat_e2e001"
    world_elements = job_dir / "_world" / "elements"
    build = f"""
import sys
sys.path.insert(0, {str(BACKEND / "mesh")!r})
from pathlib import Path
import numpy as np
import trimesh
import generated_texture

out = Path({str(world_elements)!r})
out.mkdir(parents=True, exist_ok=True)
for slug in ("crate", "crate2"):
    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide()
    colors = np.zeros((len(m.vertices), 4), dtype=np.uint8)
    colors[:, 0] = 200; colors[:, 1] = 60; colors[:, 2] = 30; colors[:, 3] = 255
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=colors)
    generated_texture.texture_generated_mesh(
        m, out / (slug + ".glb"), mpu=1.0, tex_size=128)
print("built")
"""
    proc = subprocess.run([str(MESH_ENV_PYTHON), "-c", build],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]

    (job_dir / "_world" / "restyle.json").write_text(json.dumps({
        "v": 1, "job_id": "splat_e2e001",
        "elements": {"crate": {"material": "lava", "material_scale": 0.5},
                     "crate2": {"tint": "#00ff00"}},
        "lighting": {"preset": "dungeon", "intensity": 1.0},
    }))

    staging = tmp_path / "staging"
    proc = subprocess.run(
        [str(MESH_ENV_PYTHON), str(BACKEND / "mesh" / "restyle_bake.py"),
         str(job_dir), "--staging-dir", str(staging),
         "--units-per-metre", "1.0", "--tile-size", "128"],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (proc.stdout, proc.stderr[-2000:])
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["ok"] is True
    by_slug = {e["slug"]: e for e in report["elements"]}
    assert by_slug["crate"]["verbs"] == ["material"]
    assert by_slug["crate"]["covered_frac"] > 0.3
    assert by_slug["crate"]["roughness"] == pytest.approx(0.6)  # lava's own
    assert by_slug["crate2"]["verbs"] == ["tint"]
    for slug in ("crate", "crate2"):
        assert (staging / f"{slug}.glb").is_file()
        assert (staging / f"{slug}_atlas.png").is_file()

    verify = f"""
import json
import numpy as np
from PIL import Image
def mean_painted(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
    m = a.max(axis=2) > 0
    return a[m].mean(axis=0).tolist()
print(json.dumps({{
    "lava": mean_painted({str(staging / "crate_atlas.png")!r}),
    "tinted": mean_painted({str(staging / "crate2_atlas.png")!r}),
}}))
"""
    proc = subprocess.run([str(MESH_ENV_PYTHON), "-c", verify],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    means = json.loads(proc.stdout.strip().splitlines()[-1])
    # Lava base_rgb [0.85, 0.28, 0.06]*255 = [217, 71, 15]; jitter/speckle
    # wobble the mean but red must dominate hard.
    lava = means["lava"]
    assert lava[0] > 150 and lava[1] < 110 and lava[2] < 60, lava
    # Green tint on a red-ish capture: linear multiply kills R and B, keeps G.
    tinted = means["tinted"]
    assert tinted[1] > 40 and tinted[0] < 10 and tinted[2] < 10, tinted
