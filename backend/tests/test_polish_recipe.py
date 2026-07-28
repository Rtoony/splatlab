"""Polish recipe composition tests — fake Blender runner, real staging."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc import blender_workflow, polish_recipe  # noqa: E402


def _minimal_glb_bytes(meshes: int = 1) -> bytes:
    doc = json.dumps(
        {
            "asset": {"version": "2.0", "generator": "fake-blender"},
            "meshes": [{"primitives": []} for _ in range(meshes)],
            "nodes": [{}],
        }
    ).encode()
    doc += b" " * ((4 - len(doc) % 4) % 4)
    body = struct.pack("<I4s", len(doc), b"JSON") + doc
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _make_job(outputs: Path, job_id: str = "splat_b1e001") -> Path:
    job_dir = outputs / job_id
    (job_dir / "_regen").mkdir(parents=True)
    (job_dir / "_regen" / "scene.blend").write_bytes(b"BLENDER-v0")
    element = job_dir / "_world" / "elements" / "red-bicycle.glb"
    element.parent.mkdir(parents=True)
    element.write_bytes(_minimal_glb_bytes())
    (job_dir / "meta.json").write_text(
        json.dumps({"job_id": job_id, "status": "completed",
                    "output_dir": str(job_dir)})
    )
    return job_dir


@pytest.fixture()
def workflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(blender_workflow, "OUTPUT_ROOT", outputs.resolve())
    fake_blender = tmp_path / "blender"
    fake_blender.write_bytes(b"bin")
    monkeypatch.setattr(blender_workflow, "BLENDER_BIN", fake_blender)
    return outputs


def _recipe_aware_runner(monkeypatch: pytest.MonkeyPatch, *, fail_on: str = ""):
    def fake_run(command: list[str], timeout: int = 0) -> tuple[int, str]:
        request = json.loads(Path(command[-2]).read_text())
        action = request["action"]
        if action == fail_on:
            return 5, f"{action} exploded"
        if action == "import_world_element":
            result = {"object": f"polish_{request['params']['slug']}",
                      "imported": [f"polish_{request['params']['slug']}"],
                      "faces": 8000}
        elif action == "cleanup_mesh":
            result = {"object": request["params"]["object"],
                      "faces_before": 8000, "faces_after": 7884,
                      "components_removed": 1}
        elif action == "export_glb":
            Path(request["output_glb"]).write_bytes(_minimal_glb_bytes())
            result = {"exported": True, "objects": 1}
        else:
            result = request["params"]
        if request.get("output_blend"):
            Path(request["output_blend"]).write_bytes(b"BLENDER-out")
        Path(command[-1]).write_text(json.dumps({
            "status": "ok", "result": result,
            "blender": {"version": "4.5.11 LTS", "background": True},
        }))
        return 0, "ok"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_run)


def test_recipe_composes_all_four_steps(
    workflow_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow_env)
    _recipe_aware_runner(monkeypatch)
    receipt = polish_recipe.run_recipe("splat_b1e001", "red-bicycle")

    assert [s["step"] for s in receipt["steps"]] == [
        "snapshot", "import_world_element", "cleanup_mesh", "export_glb",
    ]
    assert receipt["object"] == "polish_red-bicycle"
    assert receipt["cleanup"]["components_removed"] == 1
    assert receipt["export"]["path"] == (
        "_blender/exports/scene-v0003-polish-red-bicycle.glb"
    )
    assert (job_dir / receipt["export"]["path"]).is_file()
    assert receipt["upload"]["route"].endswith(
        "/jobs/splat_b1e001/world/elements/red-bicycle/polish"
    )
    # Three mutations -> versions 1..3; export is derived, not a version.
    versions = [v["version"] for v in blender_workflow.list_versions("splat_b1e001")]
    assert versions == [1, 2, 3]


def test_recipe_failure_preserves_partial_receipts(
    workflow_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_job(workflow_env)
    _recipe_aware_runner(monkeypatch, fail_on="cleanup_mesh")
    with pytest.raises(blender_workflow.BlenderWorkflowError) as excinfo:
        polish_recipe.run_recipe("splat_b1e001", "red-bicycle")
    done = [s["step"] for s in excinfo.value.partial_receipts]
    assert done == ["snapshot", "import_world_element"]
