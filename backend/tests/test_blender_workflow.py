"""Restricted Blender workflow tests; no Blender process required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcc import blender_workflow  # noqa: E402


def _make_job(
    outputs: Path,
    job_id: str = "splat_b1e001",
    *,
    output_dir: Path | None = None,
) -> Path:
    job_dir = output_dir or outputs / job_id
    (job_dir / "_regen").mkdir(parents=True)
    (job_dir / "_regen" / "scene.blend").write_bytes(b"BLENDER-v0")
    (job_dir / "dimensions.json").write_text(
        json.dumps([{"id": "d1", "a": [0, 0, 0], "b": [1, 0, 0]}])
    )
    (job_dir / "meta.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "completed",
                "output_dir": str(job_dir),
                "meters_per_unit": 1.5,
            }
        )
    )
    return job_dir


@pytest.fixture()
def workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(blender_workflow, "OUTPUT_ROOT", outputs.resolve())
    fake_blender = tmp_path / "blender"
    fake_blender.write_bytes(b"bin")
    monkeypatch.setattr(blender_workflow, "BLENDER_BIN", fake_blender)
    return outputs


def test_inspect_job_reports_approved_artifacts(workflow: Path) -> None:
    _make_job(workflow)
    result = blender_workflow.inspect_job("splat_b1e001")
    assert result["blend"] == "_regen/scene.blend"
    assert result["coordinate_system"]["source_frame"]["meters_per_unit"] == 1.5
    assert result["dimensions"][0]["id"] == "d1"
    assert result["versions"] == []


def test_job_output_outside_approved_root_is_rejected(
    workflow: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside" / "splat_b1e001"
    _make_job(workflow, output_dir=outside)
    # _read_meta resolves through DEFAULT_3D_ROOT, so publish only the hostile meta
    # there while its output_dir points outside the approved root.
    canonical = workflow / "splat_b1e001"
    canonical.mkdir(parents=True)
    (canonical / "meta.json").write_text((outside / "meta.json").read_text())
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="approved root"):
        blender_workflow.inspect_job("splat_b1e001")


def test_symlinked_scene_input_is_rejected(workflow: Path, tmp_path: Path) -> None:
    job_dir = _make_job(workflow)
    scene = job_dir / "_regen" / "scene.blend"
    outside = tmp_path / "outside.blend"
    outside.write_bytes(b"BLENDER-outside")
    scene.unlink()
    scene.symlink_to(outside)

    with pytest.raises(blender_workflow.BlenderWorkflowError, match="symlinked"):
        blender_workflow.inspect_job("splat_b1e001")


def test_parameter_surface_rejects_code_and_bad_vectors() -> None:
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="unsupported"):
        blender_workflow._sanitize_params("execute_code", {"code": "import os"})
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="three"):
        blender_workflow._sanitize_params(
            "transform_object", {"object": "Cube", "scale": [1, 2]}
        )
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="positive"):
        blender_workflow._sanitize_params(
            "transform_object", {"object": "Cube", "scale": [1, -1, 1]}
        )


def test_mutations_create_versions_and_receipts(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)

    def fake_run(command: list[str], timeout: int = 0) -> tuple[int, str]:
        assert "--disable-autoexec" in command
        request_path = Path(command[-2])
        response_path = Path(command[-1])
        request = json.loads(request_path.read_text())
        output = Path(request["output_blend"])
        output.write_bytes(
            Path(request["source_blend"]).read_bytes()
            + f"\n{request['action']}".encode()
        )
        response_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "result": request["params"],
                    "blender": {"version": "4.5.11 LTS", "background": True},
                }
            )
        )
        return 0, "ok"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_run)
    first = blender_workflow.run_action("splat_b1e001", "snapshot", note="baseline")
    assert first["version"] == 1
    assert first["base"]["version"] is None

    second = blender_workflow.run_action(
        "splat_b1e001",
        "transform_object",
        {"object": "Hydrant", "location": [1, 2, 3]},
    )
    assert second["version"] == 2
    assert second["base"]["version"] == 1
    assert second["params"]["object"] == "Hydrant"
    assert (job_dir / "_blender" / "versions" / "scene-v0002.blend").is_file()

    restored = blender_workflow.restore_version("splat_b1e001", 1, "undo")
    assert restored["version"] == 3
    assert restored["action"] == "restore_version"
    assert (job_dir / "_blender" / "versions" / "scene-v0003.blend").read_bytes() == (
        job_dir / "_blender" / "versions" / "scene-v0001.blend"
    ).read_bytes()

    versions = blender_workflow.list_versions("splat_b1e001")
    assert [item["version"] for item in versions] == [1, 2, 3]
    assert versions[2]["receipt"]["params"] == {"restored_version": 1}


def test_failed_action_leaves_no_version(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)

    def fake_failure(command: list[str], timeout: int = 0) -> tuple[int, str]:
        return 9, "failure"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_failure)
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="failed"):
        blender_workflow.run_action("splat_b1e001", "snapshot")
    assert not list((job_dir / "_blender" / "versions").glob("*.blend"))


def test_failed_restore_copy_leaves_no_partial_version(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)
    versions = job_dir / "_blender" / "versions"
    versions.mkdir(parents=True)
    (versions / "scene-v0001.blend").write_bytes(b"BLENDER-v1")

    def fail_copy(source: Path, destination: Path) -> None:
        del source
        destination.write_bytes(b"partial")
        raise OSError("copy interrupted")

    monkeypatch.setattr(blender_workflow.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="interrupted"):
        blender_workflow.restore_version("splat_b1e001", 1)

    assert not (versions / "scene-v0002.blend").exists()
    assert not list(versions.glob(".building-*"))


def test_blender_subprocess_environment_drops_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PORTAL_TOKEN", "do-not-pass")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-not-pass")
    monkeypatch.setenv("LANG", "C.UTF-8")
    env = blender_workflow._sanitized_env()
    assert "PORTAL_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert env["LANG"] == "C.UTF-8"


def test_typed_parameters_reject_implicit_boolean_and_vector_coercion() -> None:
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="boolean"):
        blender_workflow._sanitize_params(
            "toggle_collection", {"collection": "Scene", "visible": "false"}
        )
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="three numbers"):
        blender_workflow._sanitize_params(
            "transform_object", {"object": "Cube", "location": [True, 2, 3]}
        )


def test_cleanup_mesh_parameters_are_range_checked() -> None:
    for params in (
        {"object": "polish_x", "merge_distance": 0.0},
        {"object": "polish_x", "merge_distance": 0.2},
        {"object": "polish_x", "decimate_ratio": 0.0},
        {"object": "polish_x", "decimate_ratio": 1.5},
        {"object": "polish_x", "min_component_frac": 0.5},
        {"object": "polish_x", "min_component_frac": -0.1},
    ):
        with pytest.raises(blender_workflow.BlenderWorkflowError, match="must be in"):
            blender_workflow._sanitize_params("cleanup_mesh", params)
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="number"):
        blender_workflow._sanitize_params(
            "cleanup_mesh", {"object": "polish_x", "merge_distance": True}
        )
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="boolean"):
        blender_workflow._sanitize_params(
            "cleanup_mesh", {"object": "polish_x", "shade_smooth": "yes"}
        )
    with pytest.raises(
        blender_workflow.BlenderWorkflowError, match="at least one"
    ):
        blender_workflow._sanitize_params("cleanup_mesh", {"object": "polish_x"})
    sanitized = blender_workflow._sanitize_params(
        "cleanup_mesh",
        {"object": "polish_x", "merge_distance": 1e-4, "shade_smooth": True},
    )
    assert sanitized == {
        "object": "polish_x",
        "merge_distance": 1e-4,
        "shade_smooth": True,
    }


def test_import_world_element_slug_is_validated() -> None:
    for bad in ("", "../shell", "Red Bicycle", "a" * 41, "UPPER"):
        with pytest.raises(blender_workflow.BlenderWorkflowError, match="slug"):
            blender_workflow._sanitize_params("import_world_element", {"slug": bad})
    assert blender_workflow._sanitize_params(
        "import_world_element", {"slug": "red-bicycle"}
    ) == {"slug": "red-bicycle"}
    assert blender_workflow._sanitize_params(
        "import_world_element", {"slug": "shell"}
    ) == {"slug": "shell"}


def _fake_action_runner(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    requests: list[dict] = []

    def fake_run(command: list[str], timeout: int = 0) -> tuple[int, str]:
        request = json.loads(Path(command[-2]).read_text())
        requests.append(request)
        if request["output_blend"]:
            Path(request["output_blend"]).write_bytes(b"BLENDER-out")
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "result": request["params"],
                    "blender": {"version": "4.5.11 LTS", "background": True},
                }
            )
        )
        return 0, "ok"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_run)
    return requests


def test_import_world_element_resolves_path_host_side(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)
    element = job_dir / "_world" / "elements" / "red-bicycle.glb"
    element.parent.mkdir(parents=True)
    element.write_bytes(_minimal_glb_bytes())
    requests = _fake_action_runner(monkeypatch)

    receipt = blender_workflow.run_action(
        "splat_b1e001", "import_world_element", {"slug": "red-bicycle"}
    )
    assert receipt["version"] == 1
    assert receipt["params"] == {"slug": "red-bicycle"}
    assert requests[0]["import_glb"] == str(element)


def test_import_world_element_missing_or_symlinked_is_rejected(
    workflow: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)
    _fake_action_runner(monkeypatch)
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="solidify"):
        blender_workflow.run_action(
            "splat_b1e001", "import_world_element", {"slug": "red-bicycle"}
        )

    outside = tmp_path / "outside.glb"
    outside.write_bytes(b"glTF-outside")
    element = job_dir / "_world" / "elements" / "red-bicycle.glb"
    element.parent.mkdir(parents=True)
    element.symlink_to(outside)
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="symlink"):
        blender_workflow.run_action(
            "splat_b1e001", "import_world_element", {"slug": "red-bicycle"}
        )


def test_selective_export_gets_distinct_stem_and_receipt(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)

    def fake_run(command: list[str], timeout: int = 0) -> tuple[int, str]:
        request = json.loads(Path(command[-2]).read_text())
        assert request["action"] == "export_glb"
        Path(request["output_glb"]).write_bytes(_minimal_glb_bytes(meshes=1))
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "result": {"exported": True, "objects": 1},
                    "blender": {"version": "4.5.11 LTS", "background": True},
                }
            )
        )
        return 0, "ok"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_run)
    whole = blender_workflow.export_glb("splat_b1e001")
    selective = blender_workflow.export_glb(
        "splat_b1e001", object_name="polish_red-bicycle"
    )
    exports = job_dir / "_blender" / "exports"
    assert (exports / "scene-v0000.glb").is_file()
    assert (exports / "scene-v0000-polish-red-bicycle.glb").is_file()
    assert whole["params"] == {}
    assert selective["params"] == {"object": "polish_red-bicycle"}
    assert selective["output"]["path"] == (
        "_blender/exports/scene-v0000-polish-red-bicycle.glb"
    )
    receipt = json.loads(
        (exports / "scene-v0000-polish-red-bicycle.json").read_text()
    )
    assert receipt["params"] == {"object": "polish_red-bicycle"}


def _minimal_glb_bytes(meshes: int = 1) -> bytes:
    import struct

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


def test_export_glb_creates_validated_artifact_and_receipt(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)

    def fake_run(command: list[str], timeout: int = 0) -> tuple[int, str]:
        request = json.loads(Path(command[-2]).read_text())
        assert request["action"] == "export_glb"
        assert request["output_glb"].endswith(".glb")
        Path(request["output_glb"]).write_bytes(_minimal_glb_bytes(meshes=2))
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "result": {"exported": True, "meshes": 2},
                    "blender": {"version": "4.5.11 LTS", "background": True},
                }
            )
        )
        return 0, "ok"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_run)
    receipt = blender_workflow.export_glb("splat_b1e001", note="for UE")
    exports = job_dir / "_blender" / "exports"
    assert (exports / "scene-v0000.glb").is_file()
    assert receipt["gltf"]["meshes"] == 2
    assert receipt["output"]["path"] == "_blender/exports/scene-v0000.glb"
    assert receipt["base"]["version"] is None
    assert json.loads((exports / "scene-v0000.json").read_text())["schema"] == (
        "dev.splatlab.blender-export-receipt/v1"
    )
    assert not list(exports.glob(".building-*"))


def test_export_glb_invalid_output_is_rejected_loudly(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)

    def fake_run(command: list[str], timeout: int = 0) -> tuple[int, str]:
        request = json.loads(Path(command[-2]).read_text())
        Path(request["output_glb"]).write_bytes(b"not a glb")
        Path(command[-1]).write_text(json.dumps({"status": "ok", "result": {}}))
        return 0, "ok"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_run)
    with pytest.raises(
        blender_workflow.BlenderWorkflowError, match="structural validation"
    ):
        blender_workflow.export_glb("splat_b1e001")
    exports = job_dir / "_blender" / "exports"
    assert not list(exports.glob("*.glb"))
    assert not list(exports.glob(".building-*"))


def test_export_glb_failed_process_leaves_nothing(
    workflow: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _make_job(workflow)

    def fake_failure(command: list[str], timeout: int = 0) -> tuple[int, str]:
        return 7, "blender exploded"

    monkeypatch.setattr(blender_workflow, "_run_process", fake_failure)
    with pytest.raises(blender_workflow.BlenderWorkflowError, match="export_glb"):
        blender_workflow.export_glb("splat_b1e001")
    exports = job_dir / "_blender" / "exports"
    assert not exports.is_dir() or not list(exports.iterdir())
