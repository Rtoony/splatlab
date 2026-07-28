"""Tests for the generated-prop texture leg and world.json patch mode.

The island policy and the patch merge are pure functions, tested in the app
env. The bake itself needs trimesh/xatlas (dn-splatter-probe only), so the
end-to-end run is opt-in via SPLATLAB_RUN_MESH_ENV_TESTS=1, the same gate
test_object_texture.py uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

for _name in ("trimesh",):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

import generated_texture  # noqa: E402
import scene_solidify  # noqa: E402

MESH_ENV_PYTHON = (
    Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"
)


def test_island_drop_plan_recovers_the_bonsai_cardboard_box() -> None:
    # The measured component structure that failed the gate at frac 0.7107.
    plan = generated_texture.island_drop_plan([5686, 1290, 546, 460, 18])
    # Smallest-first drops until the gate criterion holds; the substantial
    # 1290-face interior wall is KEPT.
    dropped = sorted(plan["drop"])
    assert dropped == [2, 3, 4]
    assert plan["largest"] == 0
    assert plan["achieved_frac"] == pytest.approx(5686 / 6976, abs=1e-4)
    assert plan["achieved_frac"] >= 0.8


def test_island_drop_plan_sweeps_crumbs() -> None:
    plan = generated_texture.island_drop_plan([7968, 8, 8, 6, 6, 4])
    assert len(plan["drop"]) == 0  # 7968/8000 = 0.996 already passes
    plan = generated_texture.island_drop_plan(
        [7968, 8, 8, 6, 6, 4], target_frac=0.999
    )
    assert len(plan["drop"]) == 5


def test_island_drop_plan_refuses_substantial_geometry() -> None:
    # A 40% component is not debris no matter what the target says.
    plan = generated_texture.island_drop_plan([60, 40], target_frac=0.9)
    assert plan["drop"] == []
    assert plan["achieved_frac"] == 0.6
    assert generated_texture.island_drop_plan([]) == {
        "drop": [],
        "largest": None,
        "achieved_frac": 0.0,
    }


def _world_doc() -> dict:
    return {
        "v": 1,
        "job_id": "splat_test",
        "counts": {"instances": 6, "props_built": 5, "props_failed": 1,
                   "shell_built": True},
        "shell": {"role": "shell", "source": "voxel", "built": True},
        "elements": [
            {"slug": "red-bicycle", "built": True},
            {"slug": "cardboard-box", "built": True, "texture": None},
            {"slug": "bench", "built": False, "reason": "not isolated yet"},
        ],
    }


def test_patch_world_doc_replaces_only_matching_slugs() -> None:
    doc = _world_doc()
    patched = scene_solidify.patch_world_doc(
        doc,
        [{"slug": "cardboard-box", "built": True, "texture": True}],
        shell=None,
    )
    assert [e["slug"] for e in patched["elements"]] == [
        "red-bicycle", "cardboard-box", "bench",
    ]
    assert patched["elements"][1]["texture"] is True
    assert patched["shell"]["built"] is True  # untouched
    assert patched["counts"]["props_built"] == 2
    assert patched["counts"]["props_failed"] == 1
    assert patched["counts"]["shell_built"] is True


def test_patch_world_doc_appends_new_and_guards_the_shell_record() -> None:
    doc = _world_doc()
    # A FAILED shell attempt must never displace a good record (the review
    # finding: an element patch that accidentally triggered a shell rebuild
    # could destroy a working shell).
    patched = scene_solidify.patch_world_doc(
        doc,
        [{"slug": "new-prop", "built": False, "reason": "boom"}],
        shell={"role": "shell", "source": "voxel", "built": False},
    )
    assert patched["elements"][-1]["slug"] == "new-prop"
    assert patched["counts"]["props_built"] == 2
    assert patched["counts"]["props_failed"] == 2
    assert patched["shell"]["built"] is True  # good record preserved
    assert patched["counts"]["shell_built"] is True

    # A shell that actually BUILT does replace.
    rebuilt = scene_solidify.patch_world_doc(
        _world_doc(),
        [],
        shell={"role": "shell", "source": "voxel", "built": True, "faces": 9},
    )
    assert rebuilt["shell"]["faces"] == 9

    # And a failed attempt may land only when there is no good record to lose.
    doc_no_shell = _world_doc()
    doc_no_shell["shell"] = {"role": "shell", "built": False}
    fresh = scene_solidify.patch_world_doc(
        doc_no_shell,
        [],
        shell={"role": "shell", "source": "voxel", "built": False, "reason": "x"},
    )
    assert fresh["shell"]["reason"] == "x"
    assert fresh["counts"]["shell_built"] is False


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_MESH_ENV_TESTS") != "1",
    reason="set SPLATLAB_RUN_MESH_ENV_TESTS=1 for the real mesh-env bake test",
)
def test_mesh_env_texture_generated_mesh_end_to_end(tmp_path: Path) -> None:
    if not MESH_ENV_PYTHON.is_file():
        pytest.skip("dn-splatter-probe env is unavailable")
    out_glb = tmp_path / "prop.glb"
    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "mesh")!r})
import json
import numpy as np
import trimesh
import generated_texture

# A big coloured box plus a tiny far-away crumb that must be dropped.
# Subdivided so the crumb is a small FRACTION of faces — equal-sized
# components are (correctly) refused by the max_single_drop guard.
big = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide().subdivide()
crumb = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
crumb.apply_translation((5.0, 0.0, 0.0))
mesh = trimesh.util.concatenate([big, crumb])
colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
colors[:, 0] = 200; colors[:, 1] = 60; colors[:, 2] = 30; colors[:, 3] = 255
mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)

report = generated_texture.texture_generated_mesh(
    mesh, __import__("pathlib").Path({str(out_glb)!r}),
    mpu=2.0, tex_size=128, target_frac=0.95)

back = trimesh.load({str(out_glb)!r}, force="mesh")
report["check_uv"] = getattr(getattr(back.visual, "uv", None), "shape", None) is not None
report["check_tex"] = getattr(getattr(back.visual, "material", None), "baseColorTexture", None) is not None
print(json.dumps(report))
"""
    proc = subprocess.run(
        [str(MESH_ENV_PYTHON), "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(proc.stdout.splitlines()[-1])
    assert report["check_uv"] and report["check_tex"]
    assert report["islands"]["dropped_components"] == 1
    assert report["texture"]["baked"] is True
    assert report["texture"]["coverage"] > 0.1
    assert (tmp_path / "prop_atlas.png").is_file()


def test_world_scale_fields_describe_the_baked_geometry() -> None:
    # Calibrated bake: exports are METRES, so the world's own mpu is 1.0 and
    # the capture factor is provenance (the old behaviour recorded the capture
    # factor as mpu, double-applying calibration in every consumer).
    calibrated = scene_solidify.world_scale_fields(0.9497)
    assert calibrated == {
        "units": "meters",
        "meters_per_unit": 1.0,
        "calibrated_from_meters_per_unit": 0.9497,
    }
    assert scene_solidify.world_scale_fields(None) == {
        "units": "scene-units (uncalibrated)",
        "meters_per_unit": None,
    }
