from io import BytesIO
import json
import struct

import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import background_completion as completion
import background_recovery as recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import selection_collision as collision
import support_surfaces as support
from mesh import provenance
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene
from test_selection_collision import collision_scene as stage_collision_scene, seal_collision
from test_support_surfaces import support_scene, anchors_from_view


def material_image(size=(128, 128), alpha=255, image_format="PNG"):
    output = BytesIO()
    picture = Image.new("RGBA", size, (165, 90, 50, alpha))
    if image_format == "JPEG":
        picture = picture.convert("RGB")
    picture.save(output, format=image_format)
    return output.getvalue()


def unpack_glb(content):
    length = struct.unpack_from("<I", content, 12)[0]
    return json.loads(content[20:20 + length]), content[28 + length:]


def accessor(document, binary, index):
    record = document["accessors"][index]
    view = document["bufferViews"][record["bufferView"]]
    components = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[record["type"]]
    result = np.frombuffer(binary, dtype={5126: "<f4", 5125: "<u4"}[record["componentType"]],
                           count=record["count"] * components, offset=view.get("byteOffset", 0) + record.get("byteOffset", 0))
    return result if components == 1 else result.reshape(-1, components)


@pytest.fixture
def completion_scene(support_scene):
    job, pointer = support_scene
    view = support.inspect(job, "box", 0, 1)
    original = recovery.build(job, "box", 0, 64, anchors_from_view(view))
    receipt = completion.prepare(job, original["recovery_id"], 0, "Synthetic test material, not a model-quality evaluation")
    return job, pointer, original, receipt


def test_preparation_preserves_inputs_without_generating_or_activating(completion_scene):
    job, pointer, original, receipt = completion_scene
    for name in ("atlas.png", "support.png", "observations.npz"):
        assert receipt["artifacts"][name] == original["artifacts"][name]
    assert receipt["observed_cells"] + receipt["generated_cells"] == 4096
    assert not (completion.directory(job, receipt["completion_id"]) / "result.json").exists()
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_layered_mesh_preserves_exact_observed_triangles_and_texture(completion_scene, image_format):
    job, pointer, original, receipt = completion_scene
    content = material_image(image_format=image_format)
    result = completion.import_image(job, receipt["completion_id"], content, "unit fixture", "solid pixels, not inference")
    mesh = completion.artifact(job, receipt["completion_id"], "candidate.glb", True)
    document, binary = unpack_glb(mesh.read_bytes())
    old_document, old_binary = unpack_glb(recovery.artifact(job, original["recovery_id"], "candidate.glb").read_bytes())
    for index, expected in enumerate((recovery.artifact(job, original["recovery_id"], "atlas.png").read_bytes(), content)):
        view = document["bufferViews"][document["images"][index]["bufferView"]]
        assert binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]] == expected
    observed_primitive, generated_primitive = document["meshes"][0]["primitives"]
    positions = accessor(document, binary, observed_primitive["attributes"]["POSITION"])
    observed_indices = accessor(document, binary, observed_primitive["indices"])
    generated_indices = accessor(document, binary, generated_primitive["indices"])
    old_primitive = old_document["meshes"][0]["primitives"][0]
    old_positions = accessor(old_document, old_binary, old_primitive["attributes"]["POSITION"])
    old_indices = accessor(old_document, old_binary, old_primitive["indices"])
    np.testing.assert_array_equal(positions[observed_indices], old_positions[old_indices])
    texture_coordinates = accessor(document, binary, observed_primitive["attributes"]["TEXCOORD_0"])
    old_texture_coordinates = accessor(old_document, old_binary, old_primitive["attributes"]["TEXCOORD_0"])
    np.testing.assert_array_equal(texture_coordinates[observed_indices], old_texture_coordinates[old_indices])
    observed_faces = set(map(tuple, observed_indices.reshape(-1, 3)))
    generated_faces = set(map(tuple, generated_indices.reshape(-1, 3)))
    assert observed_faces.isdisjoint(generated_faces)
    assert len(observed_faces) == receipt["observed_cells"] * 2
    assert len(generated_faces) == receipt["generated_cells"] * 2
    assert observed_primitive["extras"]["appearance_source"] == "captured-photos"
    assert generated_primitive["extras"]["geometry_source"] == "unobserved-plane-extrapolation"
    with np.load(completion.artifact(job, receipt["completion_id"], "cell-provenance.npz", True), allow_pickle=False) as masks:
        np.testing.assert_array_equal(masks["generated"], ~masks["observed"])
    assert result["status"] == "needs-review"
    assert result["triangles"] == 8192
    assert result["material_diagnostics"]["boundary_pairs"] > 0
    assert result["material_diagnostics"]["automatic_acceptance"] is False
    with pytest.raises(provenance.GenerativeInputRefused):
        provenance.assert_not_generative(mesh, "survey")
    assert scenes.active(job) == pointer


def test_material_diagnostics_separate_uv_anchor_disagreement_from_natural_edges():
    accepted = np.zeros((64, 64), dtype=bool)
    accepted[:, :32] = True
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 32:] = 100
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    content = buffer.getvalue()
    report = completion.material_diagnostics(accepted, content, content)
    assert report["observed_reference_mae_rgb8"] == 0
    assert report["boundary_anchor_mae_rgb8"] == 0
    assert report["boundary_pairs"] == 64
    assert report["boundary_jump_mae_rgb8"] == report["generated_boundary_jump_mae_rgb8"] == 100
    shifted = BytesIO()
    Image.fromarray(pixels + 20).save(shifted, format="PNG")
    report = completion.material_diagnostics(accepted, content, shifted.getvalue())
    assert report["observed_reference_mae_rgb8"] == report["boundary_anchor_mae_rgb8"] == 20
    assert report["boundary_jump_mae_rgb8"] == 120
    assert report["generated_boundary_jump_mae_rgb8"] == 100
    assert report["automatic_acceptance"] is False


@pytest.mark.parametrize("content", [b"not an image", material_image(alpha=0), material_image(size=(128, 64)),
                                    material_image(size=(32, 32)), material_image(image_format="WEBP")])
def test_invalid_images_refuse_before_outputs(completion_scene, content):
    job, pointer, _, receipt = completion_scene
    with pytest.raises(evidence.EvidenceError):
        completion.import_image(job, receipt["completion_id"], content, "fixture", "fixture")
    assert not (completion.directory(job, receipt["completion_id"]) / "candidate.glb").exists()
    assert scenes.active(job) == pointer


def test_automatic_plane_cannot_be_extended_into_unknown_cells(support_scene):
    job, pointer = support_scene
    original = recovery.build(job, "box", 0, 64)
    with pytest.raises(evidence.EvidenceError, match="photo-linked"):
        completion.prepare(job, original["recovery_id"], 0, "Unseen material")
    assert scenes.active(job) == pointer


def test_completion_uses_shared_review_apply_and_exact_undo(completion_scene):
    job, pointer, _, receipt = completion_scene
    completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "fixture")
    before = scenes.read_revision(job, pointer["revision_id"])
    proposal = scenes.propose(job, {"kind": "place", "slug": "completed-support", "completion_id": receipt["completion_id"]}, "Review completion", 0)
    assert scenes.active(job) == pointer
    revision = scenes.read_revision(job, proposal["preview_revision"])
    entry = next(entry for entry in revision["state"]["viewer"]["elements"] if entry["slug"] == "completed-support")
    assert entry["appearance_source"] == "captured-photos-and-generated-material"
    assert entry["completion"]["recovery_id"] == receipt["recovery_id"]
    assert f"_studio/completions/{receipt['completion_id']}/result.json" in revision["artifacts"]
    scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, pointer["revision_id"], 1)
    revision = scenes.read_revision(job, restored["revision_id"])
    assert revision["state"] == before["state"]
    assert revision["artifacts"] == before["artifacts"]


@pytest.mark.parametrize("defect", ["no-result", "extra-asset", "wrong-instance", "generation", "remove"])
def test_unbuilt_ambiguous_or_stale_completion_refuses(completion_scene, defect):
    job, pointer, _, receipt = completion_scene
    if defect != "no-result":
        completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "fixture")
    edit = {"kind": "place", "slug": "completed-support", "completion_id": receipt["completion_id"]}
    if defect == "extra-asset":
        edit["recovery_id"] = receipt["recovery_id"]
    elif defect == "wrong-instance":
        with pytest.raises(evidence.EvidenceError, match="different scene or captured object"):
            completion.proposal_asset(job, receipt["completion_id"], pointer, "not-box")
        return
    elif defect == "generation":
        manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
    elif defect == "remove":
        edit.update(kind="remove", selected_slug="box")
    with pytest.raises((evidence.EvidenceError, scenes.SceneRevisionError)):
        scenes.propose(job, edit, "Must refuse", 0)


def test_changed_photo_blocks_apply_and_completed_study_cannot_be_overwritten(completion_scene):
    job, pointer, _, receipt = completion_scene
    completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "fixture")
    with pytest.raises(evidence.EvidenceError, match="already exists"):
        completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "fixture")
    proposal = scenes.propose(job, {"kind": "place", "slug": "completed-support", "completion_id": receipt["completion_id"]}, "Review completion", 0)
    (job / "processed/images/photo-1.png").write_bytes(material_image())
    with pytest.raises(scenes.SceneRevisionError):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == pointer


def test_completion_api_lists_uploads_serves_and_requires_apply_ack(completion_scene):
    job, pointer, _, receipt = completion_scene
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio"
        entry = client.get(base + "/completions").json()["completions"][0]
        assert entry["result"] is None and not entry["stale"]
        endpoint = base + "/completions/" + receipt["completion_id"]
        response = client.post(endpoint + "/image", files={"file": ("fixture.png", material_image(), "image/png")}, data={"provider": "fixture", "model": "fixture"})
        assert response.status_code == 200, response.text
        assert client.get(endpoint + "/artifact", params={"name": "candidate.glb", "result": True}).status_code == 200
        assert client.get(endpoint + "/artifact", params={"name": "../../meta.json"}).status_code == 409
        proposal = client.post(base + "/proposals", json={"expected_generation": 0, "instruction": "Preview test fixture",
                              "operation": {"kind": "place", "slug": "completed-support", "completion_id": receipt["completion_id"]}})
        assert proposal.status_code == 200, proposal.text
        identifier = proposal.json()["proposal_id"]
        assert client.post(base + f"/proposals/{identifier}/apply", json={"expected_generation": 0}).status_code == 422
    assert scenes.active(job) == pointer


def test_completion_replacement_transacts_with_refined_rows_and_local_clearance(support_scene):
    job, pointer, study = stage_collision_scene.__wrapped__(support_scene)
    before = scenes.read_revision(job, pointer["revision_id"])
    clearance = collision.prepare(job, study["review_id"], 0)
    collision_result = seal_collision(job, clearance)
    view = support.inspect(job, "box", 0, 1)
    original = recovery.build(job, "box", 0, 64, anchors_from_view(view))
    receipt = completion.prepare(job, original["recovery_id"], 0, "Synthetic removal and completion contract test")
    completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "fixture")
    edit = {"kind": "replace", "selected_slug": "box", "slug": "completed-support", "completion_id": receipt["completion_id"]}
    with pytest.raises(scenes.SceneRevisionError, match="selection-aware"):
        scenes.propose(job, edit, "Cannot bypass local clearance", 0)
    proposal = scenes.propose(job, {**edit, "selection_collision_id": clearance["collision_id"]}, "Review synthetic combined edit", 0)
    revision = scenes.read_revision(job, proposal["preview_revision"])
    assert revision["state"]["hidden_capture_slugs"] == ["box"]
    assert revision["state"]["selections"]["elements"]["box"]["rows"] == [0, 1, 2, 3, 4, 6]
    assert revision["state"]["semantics"]["box"]["active"] is False
    assert revision["state"]["semantics"]["completed-support"]["active"] is True
    assert revision["artifacts"]["_world/collision_shell.glb"] == collision_result["artifacts"]["candidate/collision.glb"]
    assert scenes.active(job) == pointer
    scenes.activate(job, proposal["proposal_id"], 0)
    restored = scenes.restore(job, pointer["revision_id"], 1)
    revision = scenes.read_revision(job, restored["revision_id"])
    assert revision["state"] == before["state"]
    assert revision["artifacts"] == before["artifacts"]


def test_simultaneous_imports_have_exactly_one_immutable_result(completion_scene):
    from concurrent.futures import ThreadPoolExecutor

    job, pointer, _, receipt = completion_scene

    def import_once(_attempt):
        try:
            return completion.import_image(job, receipt["completion_id"], material_image(), "fixture", "fixture")
        except evidence.EvidenceError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(import_once, range(2)))
    assert sum(result is not None for result in results) == 1
    assert completion.read(job, receipt["completion_id"], True)["status"] == "needs-review"
    assert scenes.active(job) == pointer
