import importlib.util
import io
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
import pytest

import architectural_edits as architecture
import artifact_manifest as manifests
import scene_revisions as scenes
from reconstruction_evidence import EvidenceError
from test_architectural_edits import architectural_scene, spec, stage_result
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene


TOOL = Path(__file__).resolve().parents[2] / "tools/prove-spatial-ui.py"
MODULE_SPEC = importlib.util.spec_from_file_location("architectural_proof_runner", TOOL)
runner = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("passed", [True, False])
def test_proof_input_audit_requires_current_passing_geometry_and_never_activates(architectural_scene, passed):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Synthetic architectural proof fixture")
    result = stage_result(job, receipt, passed)
    if passed:
        snapshot = runner.architectural_proof_inputs(job, receipt["architecture_id"], 0)
        assert snapshot["base"] == pointer and snapshot["result_sha256"] == result["sha256"]
        assert snapshot["artifacts"] == scenes.read_revision(job, pointer["revision_id"])["artifacts"]
    else:
        with pytest.raises(RuntimeError, match="passing geometry"):
            runner.architectural_proof_inputs(job, receipt["architecture_id"], 0)
    assert scenes.active(job) == pointer


def test_proof_input_audit_refuses_a_changed_output_blob(architectural_scene):
    job, _ = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Synthetic proof fixture")
    stage_result(job, receipt)
    blob = architecture.artifact(job, receipt["architecture_id"], "room.glb", True)
    blob.chmod(0o600)
    blob.write_bytes(blob.read_bytes() + b"changed")
    with pytest.raises(EvidenceError, match="corrupt"):
        runner.architectural_proof_inputs(job, receipt["architecture_id"], 0)


def test_proof_input_count_must_match_verified_joined_master(architectural_scene):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Synthetic joined room proof fixture", selected_geometry=architecture.JOINED_GEOMETRY)
    result = stage_result(job, receipt)
    snapshot = runner.architectural_proof_inputs(job, receipt["architecture_id"], 0)
    assert snapshot["authored_room_triangles"] == result["metrics"]["room_triangles"]
    assert snapshot["authored_room_triangles"] != 144
    result["metrics"]["room_triangles"] = 144
    result["sha256"] = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in result.items() if key != "sha256"})).hexdigest()
    manifests.atomic_write_json(architecture.directory(job, receipt["architecture_id"]) / "result.json", result)
    with pytest.raises(RuntimeError, match="triangle count"):
        runner.architectural_proof_inputs(job, receipt["architecture_id"], 0)
    assert scenes.active(job) == pointer


@pytest.fixture
def restored_proof(architectural_scene, tmp_path):
    job, pointer = architectural_scene
    receipt = architecture.prepare(job, spec(), 0, "Synthetic proof fixture, not a real browser or Boolean run")
    result = stage_result(job, receipt)
    snapshot = runner.architectural_proof_inputs(job, receipt["architecture_id"], 0)
    proposal = scenes.propose(job, {"kind": "extend-room", "architecture_id": receipt["architecture_id"], "slug": "extension"}, receipt["instruction"], 0)
    applied = scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, pointer["revision_id"], applied["generation"])
    output = tmp_path / "proof"
    output.mkdir()
    groups = [
        ["architecture-baseline", "architecture-guide-hidden", "architecture-restored"],
        ["architecture-preview", "architecture-splat-reclipped", "architecture-after-traversal", "architecture-applied-reloaded"],
        ["architecture-mesh-cut", "architecture-mesh-reclipped"], ["architecture-draft"],
        ["architecture-preview-oblique"], ["architecture-splat-unclipped-diagnostic"], ["architecture-mesh-uncut-diagnostic"],
    ]
    frames = []
    for ordinal, labels in enumerate(groups):
        image = Image.new("RGB", (960, 640), (ordinal * 20, 5, 6))
        for index in range(16):
            image.putpixel((index, 0), (index + ordinal * 20, ordinal * 15, 80))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        for label in labels:
            image_path = output / (label + ".png")
            image_path.write_bytes(buffer.getvalue())
            frames.append({"label": label, "width": 960, "height": 640, "png_sha256": manifests.sha256_file(image_path)})
    traces, rendered_traces = [], []
    for profile in ("retained-worker", "viewer-default-dimensions"):
        for direction in ("outward", "return"):
            filename = f"architecture-trace-{profile}-{direction}.json"
            manifests.atomic_write_json(output / filename, {"scope": "Synthetic artifact audit fixture, not physical traversal"})
            traces.append({"filename": filename, "sha256": manifests.sha256_file(output / filename)})
            rendered_name = filename.replace("architecture-trace-", "architecture-rendered-")
            snapshots = []
            for index in range(3):
                image_name = rendered_name.replace(".json", f"-{index}.png")
                (output / image_name).write_bytes((output / "architecture-preview.png").read_bytes())
                walking_snapshot = {"filename": image_name, "sha256": manifests.sha256_file(output / image_name)}
                add_render_control(output, walking_snapshot)
                snapshots.append(walking_snapshot)
            manifests.atomic_write_json(output / rendered_name, {"snapshots": snapshots, "rendering_during_substeps": True,
                "fixed_step_simulation": False, "completed": True, "error": None,
                "scope": "Synthetic artifact audit fixture only, not physical or rendered traversal"})
            rendered_traces.append({"filename": rendered_name, "sha256": manifests.sha256_file(output / rendered_name)})
    report = {"status": "passed", "architecture_id": receipt["architecture_id"], "result_sha256": result["sha256"], "traces": traces,
        "base": pointer, "final": restored, "exact_state_undo": True, "exact_pixel_undo": True, "frames": frames, "rendered_traces": rendered_traces,
        "scope": "Synthetic local-audit unit fixture only"}
    manifests.atomic_write_json(output / "architectural-browser-proof.json", report)
    return job, snapshot, output, report


def test_independent_audit_verifies_saved_pngs_and_exact_original_artifact_maps(restored_proof):
    job, snapshot, output, _ = restored_proof
    audit = runner.audit_architectural_restoration(job, snapshot, output)
    assert audit["verified_frames"] == 13 and audit["exact_state_and_artifact_restore"]
    assert audit["verified_artifacts"] == len(snapshot["artifacts"])
    assert audit["verified_rendered_traces"] == 4 and audit["verified_walking_images"] == 12
    assert audit["verified_walking_control_images"] == 24 and len(audit["walking_render_controls"]) == 12


def test_restoration_audit_refuses_resealed_trace_without_walking_control(restored_proof):
    job, snapshot, output, report = restored_proof
    record = report["rendered_traces"][0]
    document = manifests.read_json(output / record["filename"])
    document["snapshots"][0].pop("render_control")
    manifests.atomic_write_json(output / record["filename"], document)
    record["sha256"] = manifests.sha256_file(output / record["filename"])
    manifests.atomic_write_json(output / "architectural-browser-proof.json", report)
    with pytest.raises(RuntimeError, match="scene visibility control"):
        runner.audit_architectural_restoration(job, snapshot, output)
    assert not (output / "architectural-restoration-audit.json").exists()


def add_render_control(output, snapshot):
    control = {"method": "scene-visibility-ab/v1", "same_camera_body_and_collision": True, "scene_visible_after": True}
    for label in ("hidden", "restored"):
        filename = snapshot["filename"].replace(".png", f"-scene-{label}.png")
        if label == "hidden":
            Image.new("RGB", (960, 640)).save(output / filename)
        else:
            (output / filename).write_bytes((output / snapshot["filename"]).read_bytes())
        control[label] = {"filename": filename, "sha256": manifests.sha256_file(output / filename)}
    snapshot["render_control"] = control


@pytest.mark.parametrize("color", [(163, 173, 187), (255, 255, 255), (8, 0, 0)])
def test_flat_rendered_wall_requires_significant_visibility_control_not_color_noise(tmp_path, color):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (960, 640), color).save(image_path)
    snapshot = {"filename": image_path.name, "sha256": manifests.sha256_file(image_path)}
    add_render_control(tmp_path, snapshot)
    audit = runner.audit_architectural_walking_image(tmp_path, snapshot)
    assert audit["scene_contribution_pixels"] == 960 * 640 and audit["exact_pixel_restore"] is True


@pytest.mark.parametrize("defect", ["missing", "method", "pose", "hidden-state", "missing-file", "hash", "dimensions",
    "path", "symlink", "restore", "blank", "noise", "few-pixels", "old-color-heuristic"])
def test_walking_render_control_refuses_missing_unrestored_or_insignificant_images(tmp_path, defect):
    image_path = tmp_path / "frame.png"
    original = Image.new("RGB", (960, 640), (163, 173, 187))
    if defect in {"blank", "noise", "few-pixels", "old-color-heuristic"}:
        original = Image.new("RGB", (960, 640), (7, 7, 7) if defect == "noise" else (0, 0, 0))
        if defect in {"few-pixels", "old-color-heuristic"}:
            for index in range(20):
                original.putpixel((index, 0), (20 + index, 128, 255))
    original.save(image_path)
    snapshot = {"filename": image_path.name, "sha256": manifests.sha256_file(image_path)}
    add_render_control(tmp_path, snapshot)
    control = snapshot["render_control"]
    if defect == "missing": snapshot.pop("render_control")
    if defect == "method": control["method"] = "color-count/v1"
    if defect == "pose": control["same_camera_body_and_collision"] = False
    if defect == "hidden-state": control["scene_visible_after"] = False
    if defect == "path": control["hidden"]["filename"] = "../frame.png"
    if defect == "hash": control["hidden"]["sha256"] = "0" * 64
    if defect == "missing-file": (tmp_path / control["hidden"]["filename"]).unlink()
    if defect == "symlink":
        hidden = tmp_path / control["hidden"]["filename"]
        content = hidden.read_bytes()
        hidden.unlink()
        (tmp_path / "other.png").write_bytes(content)
        hidden.symlink_to(tmp_path / "other.png")
    if defect in {"dimensions", "restore", "old-color-heuristic"}:
        record = control["hidden" if defect != "restore" else "restored"]
        picture = Image.new("RGB", (480, 320) if defect == "dimensions" else (960, 640), (0, 0, 0))
        if defect == "old-color-heuristic": picture = original
        picture.save(tmp_path / record["filename"])
        record["sha256"] = manifests.sha256_file(tmp_path / record["filename"])
    with pytest.raises(RuntimeError):
        runner.audit_architectural_walking_image(tmp_path, snapshot)


@pytest.mark.parametrize("defect", ["generation", "result", "claim", "missing-frame", "checksum", "dimensions", "pixel-pair", "blank", "missing-trace", "trace-checksum", "missing-rendered", "rendered-checksum"])
def test_independent_audit_refuses_incomplete_or_contradictory_proof(restored_proof, defect):
    job, snapshot, output, report = restored_proof
    if defect == "generation":
        report["final"]["generation"] += 1
    elif defect == "result":
        report["result_sha256"] = "0" * 64
    elif defect == "claim":
        report["exact_state_undo"] = False
    elif defect == "missing-frame":
        report["frames"].pop()
    elif defect == "checksum":
        report["frames"][0]["png_sha256"] = "0" * 64
    elif defect == "dimensions":
        report["frames"][0]["width"] = 480
    elif defect == "missing-trace":
        report["traces"].pop()
    elif defect == "trace-checksum":
        report["traces"][0]["sha256"] = "0" * 64
    elif defect == "missing-rendered":
        report["rendered_traces"].pop()
    elif defect == "rendered-checksum":
        report["rendered_traces"][0]["sha256"] = "0" * 64
    else:
        frame = report["frames"][0]
        image_path = output / (frame["label"] + ".png")
        if defect == "pixel-pair":
            image_path.write_bytes((output / "architecture-preview.png").read_bytes())
        else:
            Image.new("RGB", (960, 640)).save(image_path)
        frame["png_sha256"] = manifests.sha256_file(image_path)
    manifests.atomic_write_json(output / "architectural-browser-proof.json", report)
    with pytest.raises(RuntimeError):
        runner.audit_architectural_restoration(job, snapshot, output)
    assert not (output / "architectural-restoration-audit.json").exists()


def test_audit_rehashes_bound_original_sources_even_if_size_and_mtime_are_preserved(restored_proof):
    job, snapshot, output, _ = restored_proof
    relative = next(name for name in snapshot["sources"] if name.endswith(".bin"))
    source = job / relative
    before = source.stat()
    content = source.read_bytes()
    source.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(EvidenceError, match="checksum"):
        runner.audit_architectural_restoration(job, snapshot, output)
    assert not (output / "architectural-restoration-audit.json").exists()


@pytest.mark.parametrize("defect", ["gpu", "root", "mixed"])
def test_architectural_cli_rejects_unsafe_modes_before_loading_a_server_or_gpu(tmp_path, defect):
    root = TOOL.parents[1] / "data/spatial/scene-proof/outputs"
    command = [sys.executable, str(TOOL), "--dist", str(tmp_path / "absent-dist"), "--playwright-module", "absent",
        "--output", str(tmp_path / "new-proof"), "--studio-job", "splat_c0ffee", "--studio-architecture", "architecture_" + "a" * 24,
        "--outputs-root", str(tmp_path if defect == "root" else root)]
    if defect != "gpu":
        command.append("--browser-gpu")
    if defect == "mixed":
        command.append("--studio-selection")
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "Architectural" in result.stderr
    assert not (tmp_path / "new-proof").exists()


@pytest.mark.parametrize("defect", ["gpu", "root", "mixed", "missing-inspect"])
def test_historical_appearance_diagnostic_refuses_unsafe_or_mixed_modes(tmp_path, defect):
    root = TOOL.parents[1] / "data/spatial/scene-proof/outputs"
    command = [sys.executable, str(TOOL), "--dist", str(tmp_path / "absent-dist"), "--playwright-module", "absent",
        "--output", str(tmp_path / "new-proof"), "--studio-job", "splat_c0ffee",
        "--outputs-root", str(tmp_path if defect == "root" else root)]
    if defect != "gpu": command.append("--browser-gpu")
    if defect != "missing-inspect": command.append("--studio-inspect")
    if defect == "mixed": command.append("--studio-removal")
    environment = dict(os.environ, SPATIAL_INSPECT_ARCHITECTURE_PROOF=str(tmp_path))
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=environment)
    assert result.returncode == 2 and "Historical architectural" in result.stderr
    assert not (tmp_path / "new-proof").exists()


def test_preview_only_cli_requires_an_explicit_architectural_study(tmp_path):
    command = [sys.executable, str(TOOL), "--dist", str(tmp_path), "--playwright-module", str(tmp_path),
        "--output", str(tmp_path / "new-proof"), "--studio-architecture-preview"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "preview-only" in result.stderr
    assert not (tmp_path / "new-proof").exists()


def test_projection_comparison_requires_explicit_private_inspection_before_any_server(tmp_path):
    command = [sys.executable, str(TOOL), "--dist", str(tmp_path), "--playwright-module", str(tmp_path),
        "--output", str(tmp_path / "new-proof"), "--architectural-projection-comparison"]
    environment = dict(os.environ)
    environment.pop("SPATIAL_INSPECT_ARCHITECTURE_PROOF", None)
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=environment)
    assert result.returncode == 2 and "projection comparison" in result.stderr
    assert not (tmp_path / "new-proof").exists()
