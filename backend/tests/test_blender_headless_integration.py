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
import glb_check


@pytest.mark.skipif(os.environ.get("SPLATLAB_RUN_BLENDER_TESTS") != "1", reason="Explicit CPU Blender rotation test")
def test_typed_euler_rotation_controls_quaternion_imports(tmp_path, monkeypatch):
    blender = blender_workflow.BLENDER_BIN
    if not blender.is_file():
        pytest.skip("Blender is unavailable")
    outputs = tmp_path / "outputs"
    job = outputs / "splat_a11ce"
    (job / "_regen").mkdir(parents=True)
    (job / "meta.json").write_text(json.dumps({"job_id": job.name, "status": "completed", "output_dir": str(job), "meters_per_unit": 1}))
    source = job / "_regen/scene.blend"
    created = subprocess.run([str(blender), "--disable-autoexec", "--background", "--threads", "4", "--factory-startup", "--python-expr",
        "import bpy, math; from mathutils import Quaternion; "
        "object=bpy.context.active_object; object.name='QuaternionBox'; object.scale=(1,2,3); "
        "object.rotation_mode='QUATERNION'; object.rotation_quaternion=Quaternion((0,0,1),math.radians(30)); "
        f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r}, check_existing=False)"],
        capture_output=True, text=True, timeout=120, env=blender_workflow._sanitized_env())
    assert created.returncode == 0, created.stdout + created.stderr
    monkeypatch.setattr(blender_workflow, "OUTPUT_ROOT", outputs.resolve())
    inspected = blender_workflow.run_action(job.name, "inspect", {})
    before = next(item for item in inspected["result"]["objects"] if item["name"] == "QuaternionBox")
    assert before["rotation_mode"] == "QUATERNION"
    assert before["rotation_degrees"] == pytest.approx([0, 0, 30], abs=1e-4)
    rotated = blender_workflow.run_action(job.name, "transform_object", {"object": "QuaternionBox", "rotation_degrees": [0, 0, 90]})
    exported = blender_workflow.export_glb(job.name, base_version=rotated["version"], object_name="QuaternionBox", bake_world_transform=True)
    bounds = glb_check.position_bounds(job / exported["output"]["path"])
    assert bounds["identity_transforms"]
    assert bounds["extent"] == pytest.approx([4, 6, 2], abs=1e-5)


@pytest.mark.skipif(os.environ.get("SPLATLAB_RUN_BLENDER_TESTS") != "1", reason="Explicit CPU Blender smoke test")
def test_architecture_opening_room_export_and_undo(tmp_path, monkeypatch):
    blender = blender_workflow.BLENDER_BIN
    if not blender.is_file():
        pytest.skip("Blender is unavailable")
    outputs = tmp_path / "outputs"
    job_dir = outputs / "splat_a11ce"
    (job_dir / "_regen").mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({"job_id": "splat_a11ce", "status": "completed", "output_dir": str(job_dir), "meters_per_unit": 1}))
    source = job_dir / "_regen" / "scene.blend"
    created = subprocess.run([str(blender), "--disable-autoexec", "--background", "--threads", "4", "--factory-startup", "--python-expr",
        "import bpy; bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(); "
        f"bpy.ops.wm.save_as_mainfile(filepath={str(source)!r}, check_existing=False)"],
        capture_output=True, text=True, timeout=120, env=blender_workflow._sanitized_env())
    assert created.returncode == 0, created.stdout + created.stderr
    monkeypatch.setattr(blender_workflow, "OUTPUT_ROOT", outputs.resolve())
    wall = blender_workflow.run_action("splat_a11ce", "create_wall", {"name": "test-wall", "height": 3})
    opened = blender_workflow.run_action("splat_a11ce", "cut_opening", {"object": wall["result"]["object"], "width": 1, "height": 2}, base_version=1)
    assert opened["version"] == 2
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="version changed"):
        blender_workflow.run_action("splat_a11ce", "cut_opening", {"object": wall["result"]["object"]}, base_version=1)
    room = blender_workflow.run_action("splat_a11ce", "create_room", {"name": "extension", "width": 4, "depth": 3, "height": 3, "thickness": 0.2, "door_width": 1, "door_height": 2})
    assert room["version"] == 3
    material = blender_workflow.run_action("splat_a11ce", "assign_material", {"object": room["result"]["object"], "color": [0.3, 0.6, 0.8]})
    assert material["version"] == 4
    exported = blender_workflow.export_glb("splat_a11ce", base_version=4, object_name=room["result"]["object"], bake_world_transform=True)
    assert exported["gltf"]["meshes"] == 1
    artifact = job_dir / "_blender" / "exports" / "scene-v0004-architecture-extension.glb"
    bounds = glb_check.position_bounds(artifact)
    assert bounds["identity_transforms"] is True
    assert bounds["extent"] == pytest.approx([4.4, 3.2, 3.2], abs=1e-5)
    restored = blender_workflow.restore_version("splat_a11ce", 1, "Undo the authored extension")
    assert restored["version"] == 5
    assert (job_dir / "_blender" / "versions" / "scene-v0005.blend").read_bytes() == (job_dir / "_blender" / "versions" / "scene-v0001.blend").read_bytes()


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
