from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import scene_revisions as scenes
import scene_studio_route
import selection_collision
from artifact_dependencies import scale_revision
import splat_route
from test_place_route import _authored_glb


@pytest.fixture
def scene(tmp_path, monkeypatch):
    job = tmp_path / "splat_a11cee"
    world = job / "_world"
    world.mkdir(parents=True)
    (job / "meta.json").write_text(json.dumps({"job_id": job.name, "output_dir": str(job), "status": "completed", "meters_per_unit": 1, "scale_generation": 1}))
    (world / "world.json").write_text(json.dumps({"elements": [], "units": "meters", "shell": {"built": True}}))
    (world / "world_manifest.json").write_text(json.dumps({"elements": [], "shell": {"slug": "shell", "role": "static"}}))
    (world / "shell.glb").write_bytes(_authored_glb())
    (world / "interactions.json").write_text(json.dumps({"elements": [], "schema": "fixture"}))
    export = job / "_blender" / "exports" / "scene-v0001-architecture-room.glb"
    export.parent.mkdir(parents=True)
    export.write_bytes(_authored_glb(aabb_min=(0, 0, 0), aabb_max=(4, 3, 4)))
    manifests.atomic_write_json(export.with_suffix(".json"), {"output": manifests.file_identity(export)})
    viewer = {"units": "meters", "meters_per_unit": 1, "elements": [], "shell": {"slug": "shell", "role": "static", "files": {"glb": f"/api/splat/jobs/{job.name}/world/file?name=shell.glb"}}}
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", tmp_path)
    pointer = scenes.initialize(job, viewer, scenes.source_fingerprint(job))
    return job, pointer


def placement(slug="extension"):
    return {"kind": "place", "slug": slug, "blender_export": "scene-v0001-architecture-room.glb", "label": "Authored extension"}


def test_preview_changes_nothing_until_atomic_apply_and_full_undo(scene):
    job, initial = scene
    original = scenes.read_revision(job, initial["revision_id"])
    legacy_before = (job / "_world" / "world_manifest.json").read_bytes()
    proposal = scenes.propose(job, placement(), "Preview the extension", 0)
    assert scenes.active(job) == initial
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["state"]["viewer"]["elements"][0]["collision"]["strategy"] == "complex_as_simple"
    assert preview["state"]["semantics"]["extension"]["active"]
    assert "_world/elements/extension.glb" in preview["artifacts"]
    frozen_world = scenes.read_artifact_json(job, preview, "_world/world.json")
    assert frozen_world["elements"][0]["slug"] == "extension"
    applied = scenes.activate(job, proposal["proposal_id"], 0)
    assert applied["revision_id"] == proposal["preview_revision"]
    assert (job / "_world" / "world_manifest.json").read_bytes() == legacy_before
    restored = scenes.restore(job, initial["revision_id"], 1)
    restored_scene = scenes.read_revision(job, restored["revision_id"])
    assert restored["generation"] == 2
    assert restored_scene["artifacts"] == original["artifacts"]
    assert restored_scene["state"] == original["state"]
    assert scenes.artifact(job, applied["revision_id"], "_world/elements/extension.glb").is_file()


def test_two_simultaneous_applies_have_exactly_one_winner(scene):
    job, initial = scene
    first = scenes.propose(job, placement("first"), "First candidate", 0)
    second = scenes.propose(job, placement("second"), "Second candidate", 0)

    def apply(proposal):
        try:
            return scenes.activate(job, proposal["proposal_id"], 0)
        except scenes.SceneRevisionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(apply, (first, second)))
    assert sum(result is not None for result in results) == 1
    assert scenes.active(job)["generation"] == 1


def test_changed_scale_or_selection_refuses_without_touching_active(scene):
    job, initial = scene
    proposal = scenes.propose(job, placement(), "Candidate", 0)
    metadata = json.loads((job / "meta.json").read_text())
    metadata["scale_generation"] = 2
    (job / "meta.json").write_text(json.dumps(metadata))
    with pytest.raises(scenes.SceneRevisionError, match="calibration changed"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == initial


def test_captured_removal_refuses_without_row_and_collision_evidence(scene):
    job, initial = scene
    base = scenes.read_revision(job, initial["revision_id"])
    base["state"]["viewer"]["elements"] = [{"slug": "captured-chair", "role": "prop"}]
    base["state"]["semantics"]["captured-chair"] = {"active": True}
    with pytest.raises(scenes.SceneRevisionError, match="fresh splat-row"):
        scenes._remove(base, "captured-chair")
    base["state"]["captured_collision_current"] = True
    base["state"]["selections"] = {"elements": {"captured-chair": {"rows": [1, 2]}}}
    with pytest.raises(scenes.SceneRevisionError, match="fresh splat-row"):
        scenes._remove(base, "captured-chair")
    base["state"]["viewer"]["calibration"] = {"stale": True}
    with pytest.raises(scenes.SceneRevisionError, match="calibrated world"):
        scenes._remove(base, "captured-chair")


def test_revision_serves_frozen_capture_scale_for_raw_backdrop(scene):
    job, initial = scene
    document = scenes.read_revision(job, initial["revision_id"])
    document.pop("revision_id")
    document.pop("content_sha256")
    document["state"]["calibration"]["meters_per_unit"] = 0.94975
    revision = scenes.write_revision(job, document)
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        response = client.get(f"/jobs/{job.name}/studio/revisions/{revision['revision_id']}")
    assert response.status_code == 200
    assert response.json()["viewer"]["meters_per_unit"] == 1
    assert response.json()["backdrop_meters_per_unit"] == 0.94975
    document["state"]["viewer"]["calibration"] = {"meters_per_unit": 1.25, "stale": True}
    revision = scenes.write_revision(job, document)
    with TestClient(app) as client:
        response = client.get(f"/jobs/{job.name}/studio/revisions/{revision['revision_id']}")
    assert response.json()["backdrop_meters_per_unit"] == 1.25


def test_remove_authored_updates_visual_collision_registry_and_semantics(scene):
    job, initial = scene
    added = scenes.propose(job, placement(), "Add room", 0)
    scenes.activate(job, added["proposal_id"], 0)
    removed = scenes.propose(job, {"kind": "remove", "selected_slug": "extension"}, "Remove room", 1)
    scene_doc = scenes.read_revision(job, removed["preview_revision"])
    assert scene_doc["state"]["viewer"]["elements"] == []
    assert scene_doc["state"]["semantics"]["extension"]["active"] is False
    assert scenes.read_artifact_json(job, scene_doc, "_world/placed.json")["elements"] == []
    assert scenes.read_artifact_json(job, scene_doc, "_world/world_manifest.json")["elements"] == []


def test_failed_publication_leaves_old_head_and_preview_intact(scene, monkeypatch):
    job, initial = scene
    proposal = scenes.propose(job, placement(), "Candidate", 0)
    original_write = manifests.atomic_write_json

    def fail_pointer(path, document):
        if path.name == "active.json":
            raise OSError("Simulated disk failure")
        original_write(path, document)

    monkeypatch.setattr(manifests, "atomic_write_json", fail_pointer)
    with pytest.raises(OSError, match="disk failure"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == initial
    assert scenes.read_revision(job, proposal["preview_revision"])


def test_corrupt_asset_refuses_activation(scene):
    job, initial = scene
    proposal = scenes.propose(job, placement(), "Candidate", 0)
    artifact = scenes.artifact(job, proposal["preview_revision"], "_world/elements/extension.glb")
    artifact.chmod(0o644)
    artifact.write_bytes(b"corrupt")
    with pytest.raises(scenes.SceneRevisionError, match="corrupt"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == initial


def test_undo_does_not_bypass_review_or_stale_generation(scene):
    job, initial = scene
    proposal = scenes.propose(job, placement(), "Unreviewed candidate", 0)
    with pytest.raises(scenes.SceneRevisionError, match="unreviewed proposal"):
        scenes.restore(job, proposal["preview_revision"], 0)
    with pytest.raises(scenes.SceneRevisionError, match="Active scene changed"):
        scenes.restore(job, initial["revision_id"], 1)


@pytest.mark.parametrize("key", ["/etc/passwd", "../meta.json", "_world/../../meta.json", "_world//shell.glb", "_world\\shell.glb"])
def test_artifact_paths_cannot_escape_bundle(scene, key):
    job, initial = scene
    with pytest.raises(scenes.SceneRevisionError):
        scenes.artifact(job, initial["revision_id"], key)


def test_api_review_flag_and_revision_pinned_urls(scene):
    job, initial = scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio"
        response = client.post(base + "/proposals", json={"expected_generation": 0, "instruction": "Preview room", "operation": placement()})
        assert response.status_code == 200, response.text
        proposal = response.json()
        revision = client.get(base + "/revisions/" + proposal["preview_revision"]).json()
        url = revision["viewer"]["elements"][0]["files"]["glb"]
        assert proposal["preview_revision"] in url
        assert client.post(base + f"/proposals/{proposal['proposal_id']}/apply", json={"expected_generation": 0}).status_code == 422
        assert client.post(base + f"/proposals/{proposal['proposal_id']}/apply", json={"expected_generation": 0, "reviewed": True}).status_code == 200


def test_refresh_baseline_is_reviewable_replaces_edits_and_preserves_undo(scene):
    job, initial = scene
    captured = scenes.read_revision(job, initial["revision_id"])
    added = scenes.propose(job, placement(), "Keep this version available", 0)
    authored = scenes.activate(job, added["proposal_id"], 0)
    authored_revision = scenes.read_revision(job, authored["revision_id"])
    (job / "_world" / "interactions.json").write_text('{"elements": [], "schema": "updated"}')
    refreshed = scenes.refresh_proposal(job, captured["state"]["viewer"], scenes.source_fingerprint(job), 1)
    assert scenes.active(job) == authored
    preview = scenes.read_revision(job, refreshed["preview_revision"])
    assert preview["state"]["viewer"]["elements"] == []
    assert refreshed["operation"]["kind"] == "refresh-baseline"
    applied = scenes.activate(job, refreshed["proposal_id"], 1)
    assert applied["generation"] == 2
    assert scenes.propose(job, placement("new-room"), "Edit refreshed capture", 2)
    undone = scenes.restore(job, authored["revision_id"], 2)
    undo_revision = scenes.read_revision(job, undone["revision_id"])
    assert undo_revision["state"] == authored_revision["state"]
    assert undo_revision["artifacts"] == authored_revision["artifacts"]
    with pytest.raises(scenes.SceneRevisionError, match="Legacy capture"):
        scenes.propose(job, placement("wrong-frame"), "Do not mix frames", 3)


def test_refresh_refuses_stale_generation_and_viewer_snapshot(scene):
    job, initial = scene
    viewer = scenes.read_revision(job, initial["revision_id"])["state"]["viewer"]
    before = scenes.source_fingerprint(job)
    with pytest.raises(scenes.SceneRevisionError, match="Active scene changed"):
        scenes.refresh_proposal(job, viewer, before, 1)
    (job / "_world" / "interactions.json").write_text('{"elements": []}')
    with pytest.raises(scenes.SceneRevisionError, match="World changed"):
        scenes.refresh_proposal(job, viewer, before, 0)
    assert scenes.active(job) == initial


@pytest.mark.parametrize("dependency", ["_scene/isolated/background.ply", "_scene/ground/floor.npz", "_langfield/STALE"])
def test_collision_or_language_evidence_change_invalidates_pending_edit(scene, dependency):
    job, initial = scene
    proposal = scenes.propose(job, placement(), "Bind evidence", 0)
    path = job / dependency
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"new evidence")
    with pytest.raises(scenes.SceneRevisionError, match="calibration changed"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == initial


@pytest.mark.parametrize("generation", [None, -1, True, "1"])
def test_corrupt_active_generation_is_not_overwritten(scene, generation):
    job, initial = scene
    manifests.atomic_write_json(scenes.root(job) / "active.json", {**initial, "generation": generation})
    with pytest.raises(scenes.SceneRevisionError, match="pointer is damaged"):
        scenes.initialize(job, {}, scenes.source_fingerprint(job))


def test_missing_viewer_asset_refuses_baseline_refresh(scene):
    job, initial = scene
    with pytest.raises(scenes.SceneRevisionError, match="asset missing"):
        scenes.refresh_proposal(job, {"shell": {"files": {"glb": "artifact:_world/missing.glb"}}}, scenes.source_fingerprint(job), 0)
    assert scenes.active(job) == initial


def test_captured_removal_keeps_rows_bound_to_snapshot_and_can_undo(scene):
    job, initial = scene
    captured = scenes.read_revision(job, initial["revision_id"])
    captured.pop("revision_id")
    captured.pop("content_sha256")
    state = captured["state"]
    state["viewer"]["elements"] = [{"slug": "chair", "role": "prop"}]
    state["semantics"] = {"chair": {"active": True, "provenance": "observed"}}
    state["selections"] = {"n_rows": 4, "elements": {"chair": {"rows": [1, 2], "coordinate_verified": True}}}
    state["captured_collision_current"] = True
    with pytest.raises(scenes.SceneRevisionError, match="selection-aware collision"):
        scenes._remove(captured, "chair")
    captured["artifacts"]["_world/collision_shell.glb"] = captured["artifacts"]["_world/shell.glb"]
    state["capture_partition"] = {"method": selection_collision.METHOD,
        "membership_sha256": selection_collision.membership(state["selections"])[0], "cleared_slugs": ["chair"],
        "collision_artifact": captured["artifacts"]["_world/collision_shell.glb"], "sources": {}, "calibration": scale_revision(job)}
    baseline = scenes.write_revision(job, captured)
    manifests.atomic_write_json(scenes.root(job) / "active.json", {**initial, "revision_id": baseline["revision_id"]})
    proposal = scenes.propose(job, {"kind": "remove", "selected_slug": "chair"}, "Review removed splat rows", 0)
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["state"]["hidden_capture_slugs"] == ["chair"]
    assert preview["state"]["viewer"]["elements"] == []
    assert preview["state"]["selections"] == state["selections"]
    scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, baseline["revision_id"], 1)
    assert scenes.read_revision(job, restored["revision_id"])["state"] == state


def test_saved_proposals_survive_reload_and_report_staleness(scene):
    job, initial = scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio"
        proposal = scenes.propose(job, placement(), "Resume this after reload", 0)
        saved = client.get(base).json()["proposals"]
        assert saved[0]["proposal_id"] == proposal["proposal_id"]
        assert saved[0]["stale"] is False
        (job / "_world" / "interactions.json").write_text('{"elements": []}')
        saved = client.get(base).json()["proposals"]
        assert saved[0]["stale"] is True
        assert scenes.active(job) == initial


def test_invalid_receipted_export_returns_conflict_not_server_error(scene):
    job, initial = scene
    export = job / "_blender" / "exports" / placement()["blender_export"]
    export.write_bytes(b"invalid GLB")
    manifests.atomic_write_json(export.with_suffix(".json"), {"output": manifests.file_identity(export)})
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        response = client.post(f"/jobs/{job.name}/studio/proposals", json={"expected_generation": 0, "instruction": "Refuse invalid mesh", "operation": placement()})
        assert response.status_code == 409, response.text
        assert scenes.active(job) == initial
