from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import background_recovery as recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import support_surfaces as support
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene
from test_place_route import _authored_glb


@pytest.fixture
def support_scene(selected_scene):
    job, pointer = selected_scene
    path = job / "_world/elements/box.glb"
    path.parent.mkdir(parents=True)
    path.write_bytes(_authored_glb(aabb_min=(-.2, 0, -.2), aabb_max=(.2, .5, .2)))
    revision = scenes.read_revision(job, pointer["revision_id"])
    revision.pop("revision_id")
    revision.pop("content_sha256")
    revision["source_fingerprint"] = scenes.source_fingerprint(job)
    revision["artifacts"]["_world/elements/box.glb"] = scenes.store_file(job, path)
    revision["state"]["viewer"]["elements"][0]["files"] = {"glb": "artifact:_world/elements/box.glb"}
    revision = scenes.write_revision(job, revision)
    pointer = {**pointer, "revision_id": revision["revision_id"]}
    manifests.atomic_write_json(scenes.root(job) / "active.json", pointer)
    return job, pointer


def anchors_from_view(view, targets=None):
    targets = targets or [[-.7, 0, -.7], [.7, 0, -.7], [-.7, 0, .7]]
    features = [min(view["features"], key=lambda feature: np.linalg.norm(np.array(feature["world"]) - target)) for target in targets]
    return {"evidence_sha256": view["evidence_sha256"], "points": [{"point_id": feature["point_id"],
            "image_id": view["image_id"], "photo_sha256": view["photo_sha256"]} for feature in features]}


def test_missing_captured_mesh_refuses_instead_of_guessing_bounds(selected_scene):
    job, pointer = selected_scene
    with pytest.raises(evidence.EvidenceError, match="baked, revision-pinned"):
        support.inspect(job, "box", 0)
    assert scenes.active(job) == pointer


def test_inspection_returns_verified_observations_and_preserves_scene(support_scene):
    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    assert 3 <= len(view["features"]) <= 6000
    assert all(isinstance(feature["point_id"], str) and feature["reprojection_px"] < 1e-10 for feature in view["features"])
    assert view["width"] == view["height"] == 64
    assert support.photo(job, "box", 0, 1, view["photo_sha256"]).is_file()
    assert scenes.active(job) == pointer


def test_anchor_plane_retains_exact_picks_and_not_an_automatic_refit(support_scene):
    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    picks = anchors_from_view(view)
    _, bounds = support.context(job, "box", 0)
    plane = support.fit(evidence.load(job, retain_image_observations=True), bounds, picks, pointer)
    assert plane["normal"].tolist() == [0, 1, 0]
    assert abs(plane["center"][1]) < 1e-10
    assert plane["selection"]["plane_refit"] is False
    assert plane["selection"]["semantic_class"] == "unclassified support surface"
    assert [anchor["point_id"] for anchor in plane["selection"]["anchors"]] == [point["point_id"] for point in picks["points"]]
    assert len(plane["support_indices"]) >= 60


@pytest.mark.parametrize("defect", ["duplicate", "unknown", "stale", "photo", "unlinked", "collinear", "too-few"])
def test_ambiguous_or_stale_anchors_refuse(support_scene, defect):
    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    picks = anchors_from_view(view, [[-.7, 0, -.7], [0, 0, -.7], [.7, 0, -.7]] if defect == "collinear" else None)
    if defect == "duplicate":
        picks["points"][1] = deepcopy(picks["points"][0])
    elif defect == "unknown":
        picks["points"][0]["point_id"] = "9999999999999999999"
    elif defect == "stale":
        picks["evidence_sha256"] = "0" * 64
    elif defect == "photo":
        picks["points"][0]["photo_sha256"] = "0" * 64
    elif defect == "unlinked":
        picks["points"][0]["image_id"] = 999
    elif defect == "too-few":
        picks["points"].pop()
    _, bounds = support.context(job, "box", 0)
    with pytest.raises(evidence.EvidenceError):
        support.fit(evidence.load(job, retain_image_observations=True), bounds, picks, pointer)
    assert scenes.active(job) == pointer


def test_each_chosen_observation_is_checked_not_just_loader_samples(support_scene):
    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    picks = anchors_from_view(view)
    source = evidence.load(job, retain_image_observations=True)
    record = next(record for record in source.records if str(record["id"]) == picks["points"][0]["point_id"])
    point_index = record["track"][record["track"][:, 0] == 1][0, 1]
    changed = source.images[1]["observations"].copy()
    changed[point_index]["x"] += 20
    source.images[1]["observations"] = changed
    _, bounds = support.context(job, "box", 0)
    with pytest.raises(evidence.EvidenceError, match="reprojection"):
        support.fit(source, bounds, picks, pointer)


def test_masked_support_anchors_cannot_enter_a_fit(support_scene):
    from PIL import Image

    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    picks = anchors_from_view(view)
    source = evidence.load(job, retain_image_observations=True)
    Image.new("L", (64, 64), 0).save(job / "processed/mask.png")
    source.cameras[1]["mask_key"] = "processed/mask.png"
    _, bounds = support.context(job, "box", 0)
    with pytest.raises(evidence.EvidenceError, match="observation mask"):
        support.fit(source, bounds, picks, pointer)


@pytest.mark.parametrize("action", ["inspect", "build"])
def test_support_work_waits_for_brief_worker_contention(support_scene, monkeypatch, action):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    job, _ = support_scene
    if action == "build":
        view = support.inspect(job, "box", 0, 1)
        execute = lambda: recovery.build(job, "box", 0, 64, anchors_from_view(view))
    else:
        execute = lambda: support.inspect(job, "box", 0, 1)
    waiting = Event()
    original_sleep = recovery.time.sleep

    def observed_sleep(seconds):
        waiting.set()
        original_sleep(seconds)

    monkeypatch.setattr(recovery.time, "sleep", observed_sleep)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with recovery.worker_slot():
            future = pool.submit(execute)
            assert waiting.wait(timeout=2)
        assert future.result(timeout=5)["features" if action == "inspect" else "recovery_id"]


@pytest.mark.parametrize("seconds", [-1, 4, float("nan"), float("inf")])
def test_support_read_wait_cannot_exceed_its_bound(seconds):
    with pytest.raises(evidence.EvidenceError, match="wait budget"):
        with recovery.worker_slot(wait_seconds=seconds):
            pytest.fail("Unbounded worker wait was admitted")


def test_anchored_recovery_pins_sources_and_uses_shared_proposal_undo(support_scene):
    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    picks = anchors_from_view(view)
    receipt = recovery.build(job, "box", 0, 64, picks)
    assert receipt["support_anchors"] == picks
    assert receipt["report"]["plane"]["selection"]["method"] == "photo-linked-sfm-anchors"
    assert "support_surfaces_sha256" in receipt["implementation"]
    assert receipt["cameras"]["1"]["model"] == "PINHOLE"
    assert 0 < receipt["report"]["supported_fraction"] < 1
    original = scenes.read_revision(job, pointer["revision_id"])
    proposal = scenes.propose(job, {"kind": "place", "slug": "anchored-surface", "recovery_id": receipt["recovery_id"]}, "Review anchored observations", 0)
    assert scenes.active(job) == pointer
    scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, pointer["revision_id"], 1)
    revision = scenes.read_revision(job, restored["revision_id"])
    assert revision["state"] == original["state"]
    assert revision["artifacts"] == original["artifacts"]


def test_support_api_refuses_unknown_generation_photo_and_incomplete_anchors(support_scene):
    job, pointer = support_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio"
        query = {"selected_slug": "box", "expected_generation": 0, "image_id": 1}
        response = client.get(base + "/recovery-support", params=query)
        assert response.status_code == 200
        view = response.json()
        assert client.get(base + "/recovery-support", params={**query, "expected_generation": 1}).status_code == 409
        assert client.get(base + "/recovery-support", params={**query, "image_id": 999}).status_code == 409
        photo = {"selected_slug": "box", "expected_generation": 0, "photo_sha256": view["photo_sha256"]}
        assert client.get(base + "/recovery-support/1/photo", params=photo).status_code == 200
        assert client.get(base + "/recovery-support/1/photo", params=photo).headers["cache-control"] == "private, max-age=31536000, immutable"
        assert client.get(base + "/recovery-support/1/photo", params={**photo, "photo_sha256": "0" * 64}).status_code == 409
        picks = anchors_from_view(view)
        picks["points"].pop()
        assert client.post(base + "/recoveries", json={"selected_slug": "box", "expected_generation": 0, "support_anchors": picks}).status_code == 422
    assert scenes.active(job) == pointer
