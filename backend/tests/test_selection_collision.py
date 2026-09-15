from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import selection_collision as collision
import selection_reviews as reviews
from test_place_route import _authored_glb
from test_selection_reviews import observed_scene, selected_scene, stage_models, stage_candidate


@pytest.fixture
def collision_scene(selected_scene):
    job, pointer = selected_scene
    report = {"verdict": "PASS", "geometry_frame": {"axis": "y-up", "units": "scene-units"},
              "probe": {"floor_level_y": 0, "top_level_y": 2}}
    manifests.atomic_write_json(job / "_world/collision_shell.json", report)
    (job / "_world/collision_shell.glb").write_bytes(_authored_glb())
    (job / "_mesh").mkdir()
    (job / "_mesh/mesh.ply").write_bytes((job / "_preview/splat.ply").read_bytes())
    revision = scenes.read_revision(job, pointer["revision_id"])
    revision.pop("revision_id")
    revision.pop("content_sha256")
    revision["source_fingerprint"] = scenes.source_fingerprint(job)
    revision["state"]["viewer"]["collision_shell"] = {"scale_to_world": 1}
    revision["state"]["viewer"]["elements"][0]["role"] = "prop"
    revision["state"]["semantics"] = {"box": {"label": "box", "active": True, "provenance": "observed"}}
    revision["state"]["captured_collision_current"] = True
    revision["state"]["selections"]["elements"]["other"]["coordinate_verified"] = True
    for name in ("collision_shell.glb", "collision_shell.json"):
        revision["artifacts"]["_world/" + name] = scenes.store_file(job, job / "_world" / name)
    revision = scenes.write_revision(job, revision)
    pointer = {**pointer, "revision_id": revision["revision_id"]}
    manifests.atomic_write_json(scenes.root(job) / "active.json", pointer)
    receipt = reviews.prepare(job, "box", 0)
    stage_models(job, receipt)
    candidate = reviews.refine(job, receipt["review_id"])
    stage_candidate(job, receipt, candidate)
    reviews.finalize(job, receipt["review_id"])
    return job, pointer, receipt


def seal_collision(job, prepared, verdict="PASS_LOCAL_EDIT", gates=None):
    output = collision.directory(job, prepared["collision_id"])
    stage = output / "candidate"
    stage.mkdir()
    (stage / "collision.glb").write_bytes(_authored_glb(aabb_min=(-1, -1, -1), aabb_max=(1, 0, 1)))
    manifests.atomic_write_json(stage / "report.json", {"verdict": verdict, "params": {"seed_yup": [0, 1, 0]}})
    manifests.atomic_write_json(stage / "navmesh.json", {"scope": "synthetic unit fixture"})
    return reviews.seal(job, output, "result.json", {"method": collision.METHOD, "verdict": verdict,
        "local_gates": gates if gates is not None else {key: verdict == "PASS_LOCAL_EDIT" for key in collision.REQUIRED_GATES},
        "prepared_receipt_sha256": prepared["sha256"],
        "candidate": {"gates": {"floor_continuity": 1}, "triangles": 12}},
        [stage / name for name in ("collision.glb", "report.json", "navmesh.json")])


def operation(prepared):
    return {"kind": "remove", "selected_slug": "box", "selection_collision_id": prepared["collision_id"]}


def test_preparation_has_exact_complements_and_preserves_active(collision_scene):
    job, pointer, study = collision_scene
    prepared = collision.prepare(job, study["review_id"], 0)
    assert scenes.active(job) == pointer
    assert prepared["counts"] == {"scene": 7, "core": 4, "candidate": 6, "current_background": 2, "candidate_background": 0}
    with np.load(collision.artifact(job, prepared["collision_id"], "partition.npz"), allow_pickle=False) as rows:
        assert rows["current_background"].tolist() == [4, 6]
        assert rows["candidate"].tolist() == [0, 1, 2, 3, 4, 6]


@pytest.mark.parametrize("defect", ["overlap", "duplicate", "unverified", "negative", "float", "bool"])
def test_invalid_or_ambiguous_memberships_refuse(defect):
    selections = {"n_rows": 5, "elements": {"box": {"rows": [0, 1], "coordinate_verified": True},
                                            "neighbor": {"rows": [3], "coordinate_verified": True}}}
    if defect == "overlap":
        selections["elements"]["neighbor"]["rows"] = [1]
    elif defect == "unverified":
        selections["elements"]["box"]["coordinate_verified"] = False
    else:
        selections["elements"]["box"]["rows"] = {"duplicate": [0, 0], "negative": [-1], "float": [1.5], "bool": [True]}[defect]
    with pytest.raises(evidence.EvidenceError):
        collision.membership(selections)


def test_reviewed_removal_updates_rows_collider_semantics_and_exact_undo(collision_scene):
    job, pointer, study = collision_scene
    original = scenes.read_revision(job, pointer["revision_id"])
    prepared = collision.prepare(job, study["review_id"], 0)
    result = seal_collision(job, prepared)
    proposal = scenes.propose(job, operation(prepared), "Review the synthetic transaction", 0)
    assert scenes.active(job) == pointer
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["state"]["selections"]["elements"]["box"]["rows"] == [0, 1, 2, 3, 4, 6]
    assert preview["state"]["hidden_capture_slugs"] == ["box"]
    assert preview["state"]["semantics"]["box"]["active"] is False
    assert preview["artifacts"]["_world/collision_shell.glb"] == result["artifacts"]["candidate/collision.glb"]
    assert scenes.read_artifact_json(job, preview, "_world/pluck.json") == preview["state"]["selections"]
    assert preview["state"]["selections"]["elements"]["other"] == original["state"]["selections"]["elements"]["other"]
    scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, pointer["revision_id"], 1)
    restored_revision = scenes.read_revision(job, restored["revision_id"])
    assert restored_revision["state"] == original["state"]
    assert restored_revision["artifacts"] == original["artifacts"]


@pytest.mark.parametrize("defect", ["unbuilt", "failed", "wrong-instance", "place", "changed-bound"])
def test_unverified_or_mismatched_collider_cannot_enter_proposal(collision_scene, defect):
    job, pointer, study = collision_scene
    prepared = collision.prepare(job, study["review_id"], 0)
    edit = operation(prepared)
    if defect != "unbuilt":
        seal_collision(job, prepared, "FAILED" if defect == "failed" else "PASS_LOCAL_EDIT")
    if defect == "wrong-instance":
        edit["selected_slug"] = "other"
    elif defect == "place":
        edit["kind"] = "place"
    elif defect == "changed-bound":
        with (job / "_mesh/mesh.ply").open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(scenes.SceneRevisionError):
        scenes.propose(job, edit, "Must refuse", 0)
    assert scenes.active(job) == pointer


def test_mutated_membership_or_collision_invalidates_clearance(collision_scene):
    job, _, study = collision_scene
    prepared = collision.prepare(job, study["review_id"], 0)
    seal_collision(job, prepared)
    proposal = scenes.propose(job, operation(prepared), "Bound evidence", 0)
    revision = scenes.read_revision(job, proposal["preview_revision"])
    collision.verify_partition(revision, "box")
    with pytest.raises(evidence.EvidenceError, match="selection-aware"):
        collision.verify_partition(revision, "other")
    changed = deepcopy(revision)
    changed["state"]["selections"]["elements"]["box"]["rows"].pop()
    with pytest.raises(evidence.EvidenceError):
        collision.verify_partition(changed)
    changed = deepcopy(revision)
    changed["artifacts"]["_world/collision_shell.glb"]["sha256"] = "0" * 64
    with pytest.raises(evidence.EvidenceError):
        collision.verify_partition(changed)


@pytest.mark.parametrize("defect", ["missing", "false", "truthy", "empty"])
def test_incomplete_clearance_gates_refuse(collision_scene, defect):
    job, pointer, study = collision_scene
    prepared = collision.prepare(job, study["review_id"], 0)
    gates = {key: True for key in collision.REQUIRED_GATES}
    if defect == "missing":
        gates.pop("selected_centers_clear")
    elif defect == "empty":
        gates = {}
    else:
        gates["selected_centers_clear"] = False if defect == "false" else 1
    seal_collision(job, prepared, gates=gates)
    with pytest.raises(scenes.SceneRevisionError, match="clearance and preservation"):
        scenes.propose(job, operation(prepared), "Must require every local gate", 0)
    assert scenes.active(job) == pointer


def test_saved_legacy_removal_cannot_bypass_clearance_on_activation(collision_scene):
    job, pointer, _ = collision_scene
    revision = scenes.read_revision(job, pointer["revision_id"])
    revision.pop("revision_id")
    revision.pop("content_sha256")
    revision["state"]["hidden_capture_slugs"] = ["box"]
    revision["operation"] = {"kind": "remove", "selected_slug": "box"}
    proposal = scenes.write_proposal(job, pointer, revision, "Saved before the clearance guard")
    with pytest.raises(scenes.SceneRevisionError, match="selection-aware"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == pointer


def test_collision_api_requires_review_and_stale_models_refuse(collision_scene):
    job, pointer, study = collision_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio"
        response = client.post(base + "/selection-collisions", json={"selection_review_id": study["review_id"], "expected_generation": 0})
        assert response.status_code == 200
        prepared = response.json()
        seal_collision(job, prepared)
        assert client.get(base + "/selection-collisions").json()["collisions"][0]["stale"] is False
        proposal = client.post(base + "/proposals", json={"operation": operation(prepared), "instruction": "Private synthetic review", "expected_generation": 0}).json()
        endpoint = base + f"/proposals/{proposal['proposal_id']}/apply"
        assert client.post(endpoint, json={"expected_generation": 0}).status_code == 422
        manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
        assert client.post(endpoint, json={"expected_generation": 0, "reviewed": True}).status_code == 409
        assert client.get(base + "/selection-collisions").json()["collisions"][0]["stale"] is True


def test_painted_geometry_is_not_silently_discarded(collision_scene):
    job, _, study = collision_scene
    patch = job / "_scene/surfaces/patch_wall.ply"
    patch.parent.mkdir(parents=True)
    patch.write_bytes(b"painted authority")
    with pytest.raises(evidence.EvidenceError):
        collision.prepare(job, study["review_id"], 0)
