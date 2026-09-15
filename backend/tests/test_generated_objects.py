from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import background_recovery as recovery
import generated_objects as objects
import scene_revisions as scenes
import scene_studio_route
import selection_reviews as reviews
import selection_collision as collision
import support_surfaces as support
from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY, assert_not_generative, GenerativeInputRefused
from test_place_route import _authored_glb, _glb_from_doc
from test_selection_collision import collision_scene as stage_collision_scene, seal_collision
from test_support_surfaces import support_scene, anchors_from_view
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene


@pytest.fixture
def generated_scene(support_scene):
    job, pointer, study = stage_collision_scene.__wrapped__(support_scene)
    view = support.inspect(job, "box", 0, 1)
    original = recovery.build(job, "box", 0, 64, anchors_from_view(view))
    receipt = objects.prepare(job, study["review_id"], original["recovery_id"], 0)
    return job, pointer, receipt


def seal_generated(job, receipt, status="needs-review", tagged=True):
    output = objects.directory(job, receipt["generated_object_id"])
    document = json.loads(_authored_glb()[20:])
    if tagged:
        document["asset"]["extras"] = {GLTF_EXTRAS_KEY: GENERATIVE_TAG}
    files = []
    for name in ("master.glb", "placed-master.glb", "delivery.glb"):
        path = output / name
        path.write_bytes(_glb_from_doc(document))
        files.append(path)
    return reviews.seal(job, output, "result.json", {"generated_object_id": receipt["generated_object_id"],
        "prepared_receipt_sha256": receipt["sha256"], "status": status, "placement_resolved": status == "needs-review", "render_vr_only": True,
        "model": {"name": "Synthetic unit fixture; no model inference"}, "placement": {"method": "unit fixture"}}, files)


def test_prepared_model_input_is_exact_masked_crop_and_never_a_holdout(generated_scene):
    job, pointer, receipt = generated_scene
    chosen = next(camera for camera in receipt["cameras"] if camera["image_id"] == receipt["input_image_id"])
    assert chosen["split"] == "fit"
    with Image.open(objects.artifact(job, receipt["generated_object_id"], chosen["photo"])) as opened:
        expected = opened.convert("RGBA")
    with Image.open(objects.artifact(job, receipt["generated_object_id"], chosen["mask"])) as opened:
        expected.putalpha(opened.convert("L"))
    with Image.open(objects.artifact(job, receipt["generated_object_id"], "input.png")) as opened:
        np.testing.assert_array_equal(np.asarray(opened), np.asarray(expected.crop(receipt["input_crop"])))
    assert not receipt["recipe"]["holdouts_condition_generation"]
    assert not receipt["recipe"]["holdouts_fit_pose"]
    assert not (objects.directory(job, receipt["generated_object_id"]) / "model-run").exists()
    assert scenes.active(job) == pointer


def test_explicit_holdout_cannot_condition_a_generated_object(generated_scene):
    job, pointer, receipt = generated_scene
    check = next(camera for camera in receipt["cameras"] if camera["split"] == "check")
    with pytest.raises(ValueError, match="held-out"):
        objects.prepare(job, receipt["selection_review_id"], receipt["recovery_id"], 0, check["image_id"])
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("defect", ["edge", "tiny", "dtype", "dimensions"])
def test_unsafe_source_mask_crops_refuse(defect):
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    if defect == "edge":
        mask[0, 25] = True
    elif defect == "tiny":
        mask[:] = False
    elif defect == "dtype":
        mask = mask.astype(float)
    else:
        mask = mask[..., None]
    with pytest.raises(ValueError):
        objects.crop_box(mask)


def test_similarity_camera_is_normalized_without_moving_its_center():
    matrix = np.eye(4)
    matrix[:3, :3] *= 4.7612
    matrix[:3, 3] = np.array([-1, 2, -3]) * 4.7612
    rigid = objects.rigid_camera({"world_to_camera": matrix})
    np.testing.assert_allclose(rigid[:3, :3], np.eye(3))
    np.testing.assert_allclose(np.linalg.inv(rigid)[:3, 3], [1, -2, 3])
    np.testing.assert_allclose(np.linalg.inv(matrix)[:3, 3], np.linalg.inv(rigid)[:3, 3])


@pytest.mark.parametrize("defect", ["anisotropic", "reflected", "shear"])
def test_invalid_camera_frames_cannot_produce_silent_empty_renders(defect):
    matrix = np.eye(4)
    if defect == "anisotropic":
        matrix[0, 0] = 2
    elif defect == "reflected":
        matrix[0, 0] = -1
    else:
        matrix[0, 1] = .2
    with pytest.raises(ValueError):
        objects.rigid_camera({"world_to_camera": matrix})


def test_generated_master_and_delivery_enter_compound_transaction_and_undo(generated_scene):
    job, pointer, receipt = generated_scene
    result = seal_generated(job, receipt)
    clearance = collision.prepare(job, receipt["selection_review_id"], 0)
    seal_collision(job, clearance)
    operation = {"kind": "replace", "selected_slug": "box", "slug": "generated-box", "generated_object_id": receipt["generated_object_id"],
                 "selection_collision_id": clearance["collision_id"], "background": {"slug": "background", "recovery_id": receipt["recovery_id"]}}
    before = scenes.read_revision(job, pointer["revision_id"])
    proposal = scenes.propose(job, operation, "Review synthetic contract fixture", 0)
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["state"]["hidden_capture_slugs"] == ["box"]
    assert preview["state"]["semantics"]["generated-box"]["provenance"] == "generated"
    assert preview["state"]["semantics"]["background"]["active"]
    entry = next(item for item in preview["state"]["viewer"]["elements"] if item["slug"] == "generated-box")
    assert entry["generated"]["master"] == result["artifacts"]["master.glb"]
    assert preview["artifacts"][f"_studio/generated/{receipt['generated_object_id']}/master.glb"] == result["artifacts"]["master.glb"]
    with pytest.raises(GenerativeInputRefused):
        assert_not_generative(scenes.artifact(job, proposal["preview_revision"], "_world/elements/generated-box.glb"), "survey")
    assert scenes.active(job) == pointer
    scenes.activate(job, proposal["proposal_id"], 0)
    removed = scenes.propose(job, {"kind": "remove", "selected_slug": "generated-box"}, "Remove generated content without a captured-row lookup", 1)
    assert not scenes.read_revision(job, removed["preview_revision"])["state"]["semantics"]["generated-box"]["active"]
    restored = scenes.restore(job, pointer["revision_id"], 1)
    actual = scenes.read_revision(job, restored["revision_id"])
    assert actual["state"] == before["state"]
    assert actual["artifacts"] == before["artifacts"]


@pytest.mark.parametrize("defect", ["unbuilt", "failed", "unplaced", "untagged", "wrong-object", "ambiguous", "remove", "stale", "corrupt-master"])
def test_invalid_generated_candidates_never_activate(generated_scene, defect):
    job, pointer, receipt = generated_scene
    identifier = receipt["generated_object_id"]
    if defect != "unbuilt":
        seal_generated(job, receipt, status=defect if defect in {"failed", "unplaced"} else "needs-review", tagged=defect != "untagged")
    operation = {"kind": "place", "slug": "generated-box", "generated_object_id": identifier}
    if defect == "wrong-object":
        with pytest.raises(ValueError, match="different scene or captured"):
            objects.proposal_asset(job, identifier, pointer, "other")
        return
    if defect == "ambiguous":
        operation["recovery_id"] = receipt["recovery_id"]
    elif defect == "remove":
        operation.update(kind="remove", selected_slug="box")
    elif defect == "stale":
        manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
    elif defect == "corrupt-master":
        path = objects.artifact(job, identifier, "master.glb", True)
        path.chmod(0o644)
        path.write_bytes(b"bad")
    with pytest.raises(ValueError):
        scenes.propose(job, operation, "Must refuse", 0)
    assert scenes.active(job)["revision_id"] == pointer["revision_id"]


def test_generated_worker_is_exclusive(generated_scene):
    job, _, receipt = generated_scene
    with objects.worker_slot(job, receipt["generated_object_id"]):
        with pytest.raises(ValueError, match="active worker"):
            with objects.worker_slot(job, receipt["generated_object_id"]):
                pytest.fail("Two workers must not overlap")


def test_rederivation_preserves_model_inputs_and_refuses_unverified_stage(generated_scene):
    job, pointer, receipt = generated_scene
    result = seal_generated(job, receipt, status="unplaced")
    with pytest.raises(ValueError, match="completed model stage"):
        objects.rederive(job, receipt["generated_object_id"])
    output = objects.directory(job, receipt["generated_object_id"])
    (output / "master-splat.ply").write_bytes(b"unit fixture, not a Gaussian model")
    result = reviews.seal(job, output, "result.json", {key: value for key, value in result.items() if key not in {"sha256", "artifacts", "model"}} |
                          {"model": {"name": "synthetic test stage", "worker": {"ok": True}}},
                          [output / name for name in ("master.glb", "placed-master.glb", "delivery.glb", "master-splat.ply")])
    child = objects.rederive(job, receipt["generated_object_id"])
    assert child["base"] == pointer
    assert child["artifacts"] == receipt["artifacts"]
    assert child["reuse_result_sha256"] == result["sha256"]
    assert child["generated_object_id"] != receipt["generated_object_id"]
    assert objects.read(job, receipt["generated_object_id"], True)["sha256"] == result["sha256"]
    assert not (objects.directory(job, child["generated_object_id"]) / "model-run").exists()
    assert scenes.active(job) == pointer


def test_generated_api_exposes_registered_inputs_and_rejects_paths(generated_scene):
    job, pointer, receipt = generated_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    endpoint = f"/jobs/{job.name}/studio/generated-objects"
    with TestClient(app) as client:
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.json()["objects"][0]["result"] is None
        assert client.get(endpoint + f"/{receipt['generated_object_id']}").json()["sha256"] == receipt["sha256"]
        assert client.get(endpoint + f"/{receipt['generated_object_id']}", params={"result": True}).status_code == 409
        result = seal_generated(job, receipt)
        assert client.get(endpoint + f"/{receipt['generated_object_id']}", params={"result": True}).json()["sha256"] == result["sha256"]
        assert client.get(endpoint + f"/{receipt['generated_object_id']}/artifact", params={"name": "input.png"}).status_code == 200
        assert client.get(endpoint + f"/{receipt['generated_object_id']}/artifact", params={"name": "../../meta.json"}).status_code == 409
        assert client.post(endpoint, json={"expected_generation": 0, "selection_review_id": receipt["selection_review_id"],
                                        "recovery_id": receipt["recovery_id"], "model_path": "/etc/passwd"}).status_code == 422
    assert scenes.active(job) == pointer


def test_fresh_preparation_can_reuse_only_identical_native_conditioning(generated_scene):
    job, pointer, receipt = generated_scene
    result = seal_generated(job, receipt)
    output = objects.directory(job, receipt["generated_object_id"])
    (output / "master-splat.ply").write_bytes(b"synthetic retained model fixture")
    parent = reviews.seal(job, output, "result.json", {key: value for key, value in result.items() if key not in {"sha256", "artifacts", "model"}} |
                         {"model": {"name": "synthetic fixture", "worker": {"ok": True}}},
                         [output / name for name in ("master.glb", "placed-master.glb", "delivery.glb", "master-splat.ply")])
    child = objects.prepare(job, receipt["selection_review_id"], receipt["recovery_id"], 0, receipt["input_image_id"],
                            reuse_generated_object_id=receipt["generated_object_id"])
    assert child["generated_object_id"] != receipt["generated_object_id"]
    assert child["base"] == pointer
    assert child["artifacts"]["input.png"] == receipt["artifacts"]["input.png"]
    assert objects.verify_reuse(job, child)["sha256"] == parent["sha256"]
    assert not (objects.directory(job, child["generated_object_id"]) / "result.json").exists()
    for key, value in (("seed", receipt["seed"] + 1), ("recipe", {"model": "different"}), ("artifacts", {"input.png": {"sha256": "0" * 64}})):
        with pytest.raises(ValueError, match="byte-identical"):
            objects.verify_reuse(job, {**child, key: value})
    with pytest.raises(ValueError, match="receipt changed"):
        objects.verify_reuse(job, {**child, "reuse_result_sha256": "0" * 64})
    with pytest.raises(ValueError, match="byte-identical"):
        objects.prepare(job, receipt["selection_review_id"], receipt["recovery_id"], 0, seed=receipt["seed"] + 1,
                        reuse_generated_object_id=receipt["generated_object_id"])
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("contained", [False, True])
def test_worker_requires_compute_containment_and_policy_before_imports(monkeypatch, contained):
    specification = importlib.util.spec_from_file_location("generated_object_worker", Path(__file__).resolve().parents[2] / "tools/generated-object.py")
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0 if contained else 1)

    monkeypatch.setattr(worker.subprocess, "run", run)
    if contained:
        worker.require_gate()
        assert [command[-1] for command, _kwargs in calls] == ["--is-contained", "--check"]
        assert calls[-1][1]["check"] is True
    else:
        with pytest.raises(RuntimeError, match="splatlab-compute-gate"):
            worker.build(Path("/not-a-job"), "not-an-id")
        assert len(calls) == 1
