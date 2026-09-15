from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import generated_color
import generated_gaussians as gaussians
import generated_objects as objects
import glb_transform
import scene_revisions as scenes
import scene_studio_route
import selection_collision as collision
import selection_reviews as reviews
from mesh.provenance import assert_not_generative, GenerativeInputRefused
from test_generated_color import fixture_glb
from test_generated_gaussians import sample_rows
from test_generated_objects import generated_scene, support_scene, observed_scene, selected_scene
from test_selection_collision import seal_collision


@pytest.fixture
def gaussian_scene(generated_scene):
    job, pointer, receipt = generated_scene
    object_id = receipt["generated_object_id"]
    directory = objects.directory(job, object_id)
    document, binary = fixture_glb()
    for name in ("master.glb", "placed-master.glb", "delivery.glb"):
        (directory / name).write_bytes(glb_transform._assemble(document, binary))
    gaussians.write_placed(directory / "master-splat.ply", sample_rows())
    parent = reviews.seal(job, directory, "result.json", {
        "generated_object_id": object_id, "prepared_receipt_sha256": receipt["sha256"],
        "status": "needs-review", "render_vr_only": True, "placement_resolved": True,
        "model": {"name": "local Meta SAM-3D Objects", "worker": {"ok": True}, "scope": "synthetic contract fixture, no inference"},
        "placement": {"generated_to_raw": np.eye(4).tolist()},
    }, [directory / name for name in ("master.glb", "placed-master.glb", "delivery.glb", "master-splat.ply")])
    identifier = "gaussians_" + "c" * 24
    output = gaussians.directory(job, object_id, identifier)
    output.mkdir(parents=True)
    matrix = gaussians.world_matrix(receipt, parent)
    gaussians.write_placed(output / "placed-splat.ply", gaussians.transform_rows(sample_rows(), matrix))
    color = generated_color.derive(directory / "delivery.glb", output / "appearance-delivery.glb")
    result = reviews.seal(job, output, "result.json", {
        "gaussians_id": identifier, "generated_object_id": object_id, "generated_result_sha256": parent["sha256"],
        "prepared_receipt_sha256": receipt["sha256"], "base": pointer, "status": "needs-review", "gaussians": 3,
        "native_identity": parent["artifacts"]["master-splat.ply"], "sdk_export_sources": dict(zip(gaussians.SDK_FILES, gaussians.SDK_HASHES)),
        "all_rows_preserved": True, "appearance_parameters_byte_exact": True, "placed_export_byte_exact": True,
        "native_to_world": matrix.tolist(), "gaussian_to_mesh": gaussians.GAUSSIAN_TO_MESH.tolist(),
        "frame_tolerances": gaussians.FRAME_TOLERANCES, "mesh_color": color,
        "views": [{"image_id": camera["image_id"], "split": camera["split"], "opaque_pixels": 10,
                   "frame": {key: 0. for key in gaussians.FRAME_TOLERANCES}} for camera in receipt["cameras"]],
        "scope": "synthetic transaction fixture; no GPU render or model inference",
    }, [output / "placed-splat.ply", output / "appearance-delivery.glb"])
    return job, pointer, receipt, result


def operation_for(receipt, result):
    return {"kind": "place", "slug": "native-box", "generated_object_id": receipt["generated_object_id"], "gaussians_id": result["gaussians_id"]}


def reseal(job, receipt, result):
    output = gaussians.directory(job, receipt["generated_object_id"], result["gaussians_id"])
    return reviews.seal(job, output, "result.json", {key: value for key, value in result.items() if key not in {"sha256", "artifacts"}},
                        [output / "placed-splat.ply", output / "appearance-delivery.glb"])


def test_native_appearance_mesh_collision_and_background_share_revision_remove_and_undo(gaussian_scene):
    job, pointer, receipt, result = gaussian_scene
    before = scenes.read_revision(job, pointer["revision_id"])
    clearance = collision.prepare(job, receipt["selection_review_id"], 0)
    seal_collision(job, clearance)
    operation = {**operation_for(receipt, result), "kind": "replace", "selected_slug": "box",
                 "selection_collision_id": clearance["collision_id"], "background": {"slug": "background", "recovery_id": receipt["recovery_id"]}}
    proposal = scenes.propose(job, operation, "Review synthetic native appearance with mesh collision", 0)
    preview = scenes.read_revision(job, proposal["preview_revision"])
    entry = next(entry for entry in preview["state"]["viewer"]["elements"] if entry["slug"] == "native-box")
    assert preview["state"]["hidden_capture_slugs"] == ["box"]
    assert entry["files"] == {"glb": "artifact:_world/elements/native-box.glb", "splat": "artifact:_world/elements/native-box.ply"}
    assert entry["gaussian_appearance"]["rows"] == 3
    assert entry["gaussian_appearance"]["frame"] == "world-y-up-metres"
    assert preview["artifacts"]["_world/elements/native-box.glb"] == result["artifacts"]["appearance-delivery.glb"]
    assert preview["artifacts"]["_world/elements/native-box.ply"] == result["artifacts"]["placed-splat.ply"]
    assert f"_studio/generated/{receipt['generated_object_id']}/master-splat.ply" in preview["artifacts"]
    for name in ("native-box.ply", "native-box.glb"):
        with pytest.raises(GenerativeInputRefused):
            assert_not_generative(scenes.artifact(job, proposal["preview_revision"], "_world/elements/" + name), "survey")
    assert scenes.active(job) == pointer
    scenes.activate(job, proposal["proposal_id"], 0)
    removal = scenes.propose(job, {"kind": "remove", "selected_slug": "native-box"}, "Remove paired layers", 1)
    removed = scenes.read_revision(job, removal["preview_revision"])
    assert not any(entry["slug"] == "native-box" for entry in removed["state"]["viewer"]["elements"])
    assert removed["state"]["semantics"]["background"]["active"]
    assert removed["state"]["hidden_capture_slugs"] == ["box"]
    scenes.activate(job, removal["proposal_id"], 1)
    restored = scenes.restore(job, proposal["preview_revision"], 2)
    restored_revision = scenes.read_revision(job, restored["revision_id"])
    assert restored_revision["state"] == preview["state"] and restored_revision["artifacts"] == preview["artifacts"]
    baseline = scenes.restore(job, pointer["revision_id"], 3)
    actual = scenes.read_revision(job, baseline["revision_id"])
    assert actual["state"] == before["state"] and actual["artifacts"] == before["artifacts"]


@pytest.mark.parametrize("defect", ["failed", "base", "parent", "sdk", "matrix", "missing-matrix", "views", "duplicate-view", "limits", "frame", "opacity", "rows", "color", "mesh-geometry", "no-object", "remove"])
def test_unverified_or_unpaired_native_proposals_refuse_without_changing_scene(gaussian_scene, defect):
    job, pointer, receipt, result = gaussian_scene
    operation = operation_for(receipt, result)
    output = gaussians.directory(job, receipt["generated_object_id"], result["gaussians_id"])
    if defect == "failed":
        result["status"] = "failed"
    elif defect == "base":
        result["base"] = {**pointer, "generation": 7}
    elif defect == "parent":
        result["generated_result_sha256"] = "0" * 64
    elif defect == "sdk":
        result["sdk_export_sources"] = {}
    elif defect == "matrix":
        result["native_to_world"][0][3] += 1
    elif defect == "missing-matrix":
        result.pop("native_to_world")
    elif defect == "views":
        result["views"] = []
    elif defect == "duplicate-view":
        result["views"].append(deepcopy(result["views"][0]))
    elif defect == "limits":
        result["frame_tolerances"] = {**gaussians.FRAME_TOLERANCES, "alpha_max": 1}
    elif defect == "frame":
        result["views"][0]["frame"]["rgb_mae"] = .1
    elif defect == "opacity":
        path = output / "placed-splat.ply"
        rows = gaussians.read_native(path)
        rows[0, 9] += 1
        path.write_bytes(path.read_bytes().split(b"end_header\n")[0] + b"end_header\n" + rows.astype("<f4").tobytes())
    elif defect == "rows":
        result["gaussians"] += 1
    elif defect == "color":
        result["mesh_color"] = {}
    elif defect == "mesh-geometry":
        path = output / "appearance-delivery.glb"
        document, binary = glb_transform._chunks(path.read_bytes())
        document["nodes"][0]["translation"] = [10, 0, 0]
        path.write_bytes(glb_transform._assemble(document, binary))
    elif defect == "no-object":
        operation.pop("generated_object_id")
    elif defect == "remove":
        operation.update(kind="remove", selected_slug="box")
    reseal(job, receipt, result)
    with pytest.raises(ValueError):
        scenes.propose(job, operation, "Must refuse synthetic defect", 0)
    assert scenes.active(job) == pointer


def test_corrupt_native_blob_cannot_activate_an_existing_preview(gaussian_scene):
    job, pointer, receipt, result = gaussian_scene
    proposal = scenes.propose(job, operation_for(receipt, result), "Retain immutable evidence", 0)
    path = gaussians.artifact(job, receipt["generated_object_id"], result["gaussians_id"], "placed-splat.ply")
    path.chmod(0o644)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == pointer


def test_api_pins_both_native_and_mesh_urls_and_rejects_stale_history(gaussian_scene):
    job, pointer, receipt, result = gaussian_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    base = f"/jobs/{job.name}/studio"
    with TestClient(app) as client:
        response = client.post(base + "/proposals", json={"expected_generation": 0, "instruction": "Preview both representations", "operation": operation_for(receipt, result)})
        assert response.status_code == 200, response.text
        proposal = response.json()
        preview = client.get(base + "/revisions/" + proposal["preview_revision"]).json()
        entry = next(entry for entry in preview["viewer"]["elements"] if entry["slug"] == "native-box")
        assert all(proposal["preview_revision"] in entry["files"][key] for key in ("splat", "glb"))
        manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
        response = client.post(base + "/proposals", json={"expected_generation": 1, "instruction": "Do not reuse stale evidence", "operation": operation_for(receipt, result)})
        assert response.status_code == 409 and "stale" in response.text
