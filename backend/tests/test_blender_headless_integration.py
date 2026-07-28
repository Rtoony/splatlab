"""Opt-in smoke test against the installed Blender binary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc import blender_workflow  # noqa: E402


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_BLENDER_TESTS") != "1",
    reason="set SPLATLAB_RUN_BLENDER_TESTS=1 for the real Blender smoke test",
)
def test_headless_snapshot_and_inspect(tmp_path: Path) -> None:
    blender = blender_workflow.BLENDER_BIN
    if not blender.is_file():
        pytest.skip("Blender binary is unavailable")

    source = tmp_path / "source.blend"
    create = subprocess.run(
        [
            str(blender),
            "--disable-autoexec",
            "--background",
            "--factory-startup",
            "--python-expr",
            (
                "import bpy; "
                "bpy.context.active_object.name='SplatLabCube'; "
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r}, check_existing=False)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=blender_workflow._sanitized_env(),
    )
    assert create.returncode == 0, create.stdout + create.stderr

    output = tmp_path / "version.blend"
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "schema": "dev.splatlab.blender-action/v1",
                "job_id": "splat_smoke01",
                "action": "transform_object",
                "params": {"object": "SplatLabCube", "location": [1, 2, 3]},
                "source_blend": str(source),
                "output_blend": str(output),
            }
        )
    )
    mutate = subprocess.run(
        [
            str(blender),
            "--disable-autoexec",
            "--background",
            str(source),
            "--python",
            str(blender_workflow.ACTION_SCRIPT),
            "--",
            str(request),
            str(response),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=blender_workflow._sanitized_env(),
    )
    assert mutate.returncode == 0, mutate.stdout + mutate.stderr
    assert output.is_file()
    result = json.loads(response.read_text())
    assert result["status"] == "ok"
    assert result["result"]["location"] == [1.0, 2.0, 3.0]


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_BLENDER_TESTS") != "1",
    reason="set SPLATLAB_RUN_BLENDER_TESTS=1 for the real Blender smoke test",
)
def test_headless_export_glb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blender = blender_workflow.BLENDER_BIN
    if not blender.is_file():
        pytest.skip("Blender binary is unavailable")

    outputs = tmp_path / "outputs"
    job_dir = outputs / "splat_b1e999"
    (job_dir / "_regen").mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": "splat_b1e999", "status": "completed",
        "output_dir": str(job_dir),
    }))
    source = job_dir / "_regen" / "scene.blend"
    create = subprocess.run(
        [
            str(blender), "--disable-autoexec", "--background",
            "--factory-startup", "--python-expr",
            (
                "import bpy; "
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r}, check_existing=False)"
            ),
        ],
        check=False, capture_output=True, text=True, timeout=120,
        env=blender_workflow._sanitized_env(),
    )
    assert create.returncode == 0, create.stdout + create.stderr

    monkeypatch.setattr(blender_workflow, "OUTPUT_ROOT", outputs.resolve())
    receipt = blender_workflow.export_glb("splat_b1e999", note="integration")
    exported = job_dir / "_blender" / "exports" / "scene-v0000.glb"
    assert exported.is_file()
    assert receipt["gltf"]["meshes"] >= 1  # factory startup scene has the cube
    assert receipt["result"]["exported"] is True


@pytest.mark.skipif(
    os.environ.get("SPLATLAB_RUN_BLENDER_TESTS") != "1",
    reason="set SPLATLAB_RUN_BLENDER_TESTS=1 for the real Blender smoke test",
)
def test_headless_polish_primitives_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The polish loop's Blender legs: import element -> cleanup -> selective export."""
    blender = blender_workflow.BLENDER_BIN
    if not blender.is_file():
        pytest.skip("Blender binary is unavailable")

    outputs = tmp_path / "outputs"
    job_dir = outputs / "splat_b1e998"
    (job_dir / "_regen").mkdir(parents=True)
    element = job_dir / "_world" / "elements" / "test-prop.glb"
    element.parent.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": "splat_b1e998", "status": "completed",
        "output_dir": str(job_dir),
    }))
    source = job_dir / "_regen" / "scene.blend"
    create = subprocess.run(
        [
            str(blender), "--disable-autoexec", "--background",
            "--factory-startup", "--python-expr",
            (
                "import bpy; "
                # The element GLB: the factory cube, exported alone.
                f"bpy.ops.export_scene.gltf(filepath={str(element)!r}, "
                "export_format='GLB', export_yup=True); "
                # The workflow scene: emptied, so the import is unambiguous.
                "[bpy.data.objects.remove(o, do_unlink=True) "
                "for o in list(bpy.data.objects)]; "
                f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r}, "
                "check_existing=False)"
            ),
        ],
        check=False, capture_output=True, text=True, timeout=120,
        env=blender_workflow._sanitized_env(),
    )
    assert create.returncode == 0, create.stdout + create.stderr

    monkeypatch.setattr(blender_workflow, "OUTPUT_ROOT", outputs.resolve())

    imported = blender_workflow.run_action(
        "splat_b1e998", "import_world_element", {"slug": "test-prop"}
    )
    assert imported["version"] == 1
    assert imported["result"]["object"] == "polish_test-prop"
    assert imported["result"]["faces"] == 12  # the factory cube, glTF-triangulated

    cleaned = blender_workflow.run_action(
        "splat_b1e998",
        "cleanup_mesh",
        {
            "object": "polish_test-prop",
            "merge_distance": 1e-4,
            "min_component_frac": 0.01,
            "shade_smooth": True,
        },
    )
    assert cleaned["version"] == 2
    assert cleaned["result"]["faces_after"] == 12
    assert cleaned["result"]["components_removed"] == 0  # one island: kept

    receipt = blender_workflow.export_glb(
        "splat_b1e998", base_version=2, object_name="polish_test-prop"
    )
    exported = (
        job_dir / "_blender" / "exports" / "scene-v0002-polish-test-prop.glb"
    )
    assert exported.is_file()
    assert receipt["gltf"]["meshes"] == 1
    assert receipt["params"] == {"object": "polish_test-prop"}

    # Re-importing the same slug must hand the canonical name to the FRESH
    # import (the stale one is renamed aside), or slug-derived tool calls
    # would silently target old geometry.
    again = blender_workflow.run_action(
        "splat_b1e998", "import_world_element", {"slug": "test-prop"}
    )
    assert again["result"]["object"] == "polish_test-prop"
    assert again["result"]["superseded"] == "polish_test-prop.superseded"
