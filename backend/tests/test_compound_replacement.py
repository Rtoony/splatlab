from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import background_completion as completion
import background_recovery as recovery
import scene_revisions as scenes
import scene_studio_route
import selection_collision as collision
import support_surfaces as support
from test_background_completion import material_image
from test_place_route import _authored_glb
from test_selection_collision import collision_scene as stage_collision_scene, seal_collision
from test_support_surfaces import support_scene, anchors_from_view
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene


@pytest.fixture
def compound_scene(support_scene):
    job, pointer, study = stage_collision_scene.__wrapped__(support_scene)
    clearance = collision.prepare(job, study["review_id"], 0)
    seal_collision(job, clearance)
    view = support.inspect(job, "box", 0, 1)
    original = recovery.build(job, "box", 0, 64, anchors_from_view(view))
    export = job / "_blender/exports/scene-v0001-replacement.glb"
    export.parent.mkdir(parents=True, exist_ok=True)
    export.write_bytes(_authored_glb(aabb_min=(0, 0, 0), aabb_max=(.2, .3, .2)))
    manifests.atomic_write_json(export.with_suffix(".json"), {"output": manifests.file_identity(export)})
    operation = {"kind": "replace", "selected_slug": "box", "slug": "replacement", "blender_export": export.name,
                 "selection_collision_id": clearance["collision_id"],
                 "background": {"slug": "background", "recovery_id": original["recovery_id"]}}
    return job, pointer, operation


@pytest.mark.parametrize("generated", [False, True])
def test_two_assets_one_preview_apply_and_exact_undo(compound_scene, generated):
    job, pointer, operation = compound_scene
    baseline = scenes.read_revision(job, pointer["revision_id"])
    if generated:
        receipt = completion.prepare(job, operation["background"]["recovery_id"], 0, "Unit fixture only")
        completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "no model")
        operation["background"] = {"slug": "background", "completion_id": receipt["completion_id"]}
    proposal = scenes.propose(job, operation, "Review all parts together", 0)
    assert scenes.active(job) == pointer
    preview = scenes.read_revision(job, proposal["preview_revision"])
    assert preview["state"]["hidden_capture_slugs"] == ["box"]
    assert preview["state"]["selections"]["elements"]["box"]["rows"] == [0, 1, 2, 3, 4, 6]
    elements = {entry["slug"]: entry for entry in preview["state"]["viewer"]["elements"]}
    assert {"replacement", "background"} <= elements.keys()
    assert "box" not in elements
    assert elements["background"]["appearance_source"] == ("captured-photos-and-generated-material" if generated else "captured-photos")
    assert not preview["state"]["semantics"]["box"]["active"]
    for slug in ("replacement", "background"):
        assert preview["state"]["semantics"][slug]["active"]
        assert elements[slug]["collision"]["strategy"] == "complex_as_simple"
        assert f"_world/elements/{slug}.glb" in preview["artifacts"]
    for key in ("_world/world.json", "_world/world_manifest.json", "_world/placed.json"):
        entries = scenes.read_artifact_json(job, preview, key)["elements"]
        assert {"replacement", "background"} <= {entry["slug"] for entry in entries}
        assert not any(entry["slug"] == "box" for entry in entries)
    assert "background" in preview["state"]["recovery_dependencies"]
    assert scenes.activate(job, proposal["proposal_id"], 0)["generation"] == 1
    restored = scenes.restore(job, pointer["revision_id"], 1)
    actual = scenes.read_revision(job, restored["revision_id"])
    assert actual["state"] == baseline["state"]
    assert actual["artifacts"] == baseline["artifacts"]


@pytest.mark.parametrize("defect", ["place", "remove", "no-object", "ambiguous-object", "no-background-source", "two-background-sources",
                                    "duplicate-name", "historic-name", "invalid-name", "nested-export", "recursive", "missing-clearance"])
def test_invalid_compound_edit_never_publishes_partial_proposal(compound_scene, defect):
    job, pointer, operation = compound_scene
    if defect in {"place", "remove"}:
        operation["kind"] = defect
    elif defect == "no-object":
        operation.pop("blender_export")
    elif defect == "ambiguous-object":
        operation["recovery_id"] = operation["background"]["recovery_id"]
    elif defect == "no-background-source":
        operation["background"].pop("recovery_id")
    elif defect == "two-background-sources":
        operation["background"]["completion_id"] = "completion_" + "a" * 24
    elif defect == "duplicate-name":
        operation["background"]["slug"] = "replacement"
    elif defect == "historic-name":
        operation["background"]["slug"] = "box"
    elif defect == "invalid-name":
        operation["background"]["slug"] = "../escape"
    elif defect == "nested-export":
        operation["background"]["blender_export"] = operation["blender_export"]
    elif defect == "recursive":
        operation["background"]["background"] = deepcopy(operation["background"])
    else:
        operation.pop("selection_collision_id")
    before = set((scenes.root(job) / "proposals").glob("*.json"))
    with pytest.raises(scenes.SceneRevisionError):
        scenes.propose(job, operation, "Must fail atomically", 0)
    assert scenes.active(job) == pointer
    assert set((scenes.root(job) / "proposals").glob("*.json")) == before


@pytest.mark.parametrize("slug", ["replacement", "background"])
def test_either_corrupt_asset_blocks_the_entire_activation(compound_scene, slug):
    job, pointer, operation = compound_scene
    proposal = scenes.propose(job, operation, "Review both", 0)
    path = scenes.artifact(job, proposal["preview_revision"], f"_world/elements/{slug}.glb")
    path.chmod(0o644)
    path.write_bytes(b"corrupt")
    with pytest.raises(scenes.SceneRevisionError, match="corrupt"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == pointer


def test_api_accepts_compound_but_still_requires_review(compound_scene):
    job, pointer, operation = compound_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    base = f"/jobs/{job.name}/studio"
    with TestClient(app) as client:
        body = {"expected_generation": 0, "instruction": "Unit fixture compound preview", "operation": operation}
        response = client.post(base + "/proposals", json=body)
        assert response.status_code == 200, response.text
        identifier = response.json()["proposal_id"]
        assert client.post(base + f"/proposals/{identifier}/apply", json={"expected_generation": 0}).status_code == 422
        body["operation"]["background"]["filesystem_path"] = "/etc/passwd"
        assert client.post(base + "/proposals", json=body).status_code == 422
    assert scenes.active(job) == pointer
