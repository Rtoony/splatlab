import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as selections
import structural_surfaces as surfaces
import scene_studio_route
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene


@pytest.fixture
def wall_scene(selected_scene):
    job, pointer = selected_scene
    transform = np.eye(4)
    transform[:3, :3] = evidence.Y_UP.T
    manifests.atomic_write_json(job / "processed/splatfacto/run/dataparser_transforms.json", {"transform": transform.tolist(), "scale": 1})
    revision = scenes.read_revision(job, pointer["revision_id"])
    revision.pop("revision_id")
    revision.pop("content_sha256")
    revision["source_fingerprint"] = scenes.source_fingerprint(job)
    revision = scenes.write_revision(job, revision)
    pointer = {**pointer, "revision_id": revision["revision_id"]}
    manifests.atomic_write_json(scenes.root(job) / "active.json", pointer)
    receipt = surfaces.prepare(job, 0, [1, 2, 3, 4, 5, 6])
    return job, pointer, receipt


def stage_masks(job, receipt, checks_only=False, invert_checks=False, defect=None):
    output = surfaces.directory(job, receipt["structure_id"])
    (output / "masks/wall").mkdir(parents=True)
    manifests.atomic_write_json(output / "sam3_manifest.json", {"wall": {"slug": "wall", "cams": [camera["image_id"] for camera in receipt["cameras"]]}})
    files = [output / "sam3_manifest.json"]
    for camera in receipt["cameras"]:
        positive = camera["split"] == "check" if checks_only else not (invert_checks and camera["split"] == "check")
        masks = np.full((1, camera["height"], camera["width"]), positive, dtype=bool)
        scores = np.array([.9])
        if defect == "dimensions":
            masks = masks[:, :-1]
        elif defect == "scores":
            scores[0] = np.nan
        elif defect == "type":
            masks = masks.astype(np.uint8)
        path = output / f"masks/wall/cam_{camera['image_id']:03d}.npz"
        np.savez_compressed(path, masks=masks, scores=scores, prompt="floor" if defect == "prompt" else "wall")
        files.append(path)
    if defect == "missing":
        files.pop()
    return selections.seal(job, output, "masks.json", {"prepared_sha256": "0" * 64 if defect == "lineage" else receipt["sha256"],
        "method": "synthetic test masks, not real model output"}, files)


def test_real_track_fitting_retains_sources_and_leaves_active_unchanged(wall_scene):
    job, pointer, receipt = wall_scene
    stage_masks(job, receipt)
    result = surfaces.fit(job, receipt["structure_id"])
    assert result["status"] == "needs-review"
    assert len(result["planes"]) == 1
    assert result["accepted_points"] > 1000
    assert result["planes"][0]["rms_m"] < 1e-10
    assert result["planes"][0]["normal_toward_cameras"] == pytest.approx([0, 0, 1])
    assert result["planes"][0]["appearance_checks"][0]["observed_points"] > 1000
    with np.load(surfaces.artifact(job, receipt["structure_id"], "membership.npz", "result"), allow_pickle=False) as arrays:
        assert arrays["observed"].shape == arrays["positive"].shape == (6, 1225)
        assert np.array_equal(arrays["point_ids"], np.arange(1, 1226))
        assert len(arrays["plane_0_indices"]) == result["planes"][0]["support_count"]
    assert scenes.active(job) == pointer
    with pytest.raises(evidence.EvidenceError, match="already exists"):
        surfaces.fit(job, receipt["structure_id"])


@pytest.mark.parametrize("defect", ["points", "centers", "splits"])
def test_invalid_vote_geometry_refuses(defect):
    points = np.array([[0., 0., 0.]])
    cameras = [{"center": [0., 0., 1.], "split": "fit", "group": "first"},
               {"center": [1., 0., 1.], "split": "fit", "group": "second"}]
    if defect == "points":
        points[0, 0] = np.nan
    elif defect == "centers":
        cameras[0]["center"][0] = np.inf
    else:
        cameras[1]["split"] = "test"
    with pytest.raises(evidence.EvidenceError, match="finite calibrated"):
        surfaces.membership(points, cameras, np.ones((2, 1), bool), np.ones((2, 1), bool))


def test_point_at_camera_center_cannot_supply_angular_support():
    cameras = [{"center": [0., 0., 0.], "split": "fit", "group": "first"},
               {"center": [1., 0., 1.], "split": "fit", "group": "second"}]
    accepted, positives, _ = surfaces.membership(np.zeros((1, 3)), cameras, np.ones((2, 1), bool), np.ones((2, 1), bool))
    assert positives.tolist() == [2]
    assert not accepted.any()


def depth_fixture():
    return ({"center": [0., 0., 2.], "normal_toward_cameras": [0., 0., -1.],
             "right": [1., 0., 0.], "up": [0., 1., 0.], "lower_uv": [-1., -1.], "upper_uv": [1., 1.]},
            {"center": [0., 0., 0.], "world_to_camera": np.diag([2., 2., 2., 1.])})


def test_projected_depth_checks_intersections_not_prior_point_footprints():
    plane, camera = depth_fixture()
    points = np.array([[1.5, 0., 4.], [.75, 0., 1.], [0., 0., 2.02], [0., 0., 1.], [0., 0., -1.], [1., 0., 0.]])
    result = surfaces.projected_depth_comparison(points, plane, camera)
    assert result["indices"].tolist() == [0, 2, 3]
    assert result["signed_optical_depth_error_m"].tolist() == pytest.approx([-2, -.02, 1])
    assert result["within_5cm"] == 1 and result["plane_behind_prior_count"] == result["plane_in_front_of_prior_count"] == 1
    assert result["median_absolute_error_m"] == pytest.approx(1)


def test_empty_projected_depth_check_is_missing_evidence_not_zero_error():
    plane, camera = depth_fixture()
    result = surfaces.projected_depth_comparison(np.empty((0, 3)), plane, camera)
    assert result["projected_footprint_samples"] == 0 and result["median_absolute_error_m"] is None


@pytest.mark.parametrize("defect", ["camera", "basis", "bounds", "points"])
def test_projected_depth_refuses_malformed_geometry(defect):
    plane, camera = depth_fixture()
    points = np.ones((1, 3))
    if defect == "camera":
        camera["center"][0] = 1
    elif defect == "basis":
        plane["up"] = plane["right"]
    elif defect == "bounds":
        plane["upper_uv"] = plane["lower_uv"]
    else:
        points[0, 0] = np.inf
    with pytest.raises(evidence.EvidenceError, match="Projected depth"):
        surfaces.projected_depth_comparison(points, plane, camera)


def test_invalid_plane_orientation_reference_refuses():
    with pytest.raises(evidence.EvidenceError, match="bounded structural"):
        surfaces.fit_planes(np.zeros((60, 3)), np.arange(60), [np.nan, 0, 0])


@pytest.mark.parametrize("defect", ["missing-key", "prompt-array"])
def test_malformed_mask_contract_refuses(wall_scene, defect):
    job, _, receipt = wall_scene
    camera = receipt["cameras"][0]
    output = surfaces.directory(job, receipt["structure_id"])
    path = output / f"masks/wall/cam_{camera['image_id']:03d}.npz"
    path.parent.mkdir(parents=True)
    arrays = {"masks": np.ones((1, camera["height"], camera["width"]), bool), "scores": np.array([.9])}
    if defect == "prompt-array":
        arrays["prompt"] = np.array(["wall", "wall"])
    np.savez_compressed(path, **arrays)
    selections.seal(job, output, "masks.json", {"prepared_sha256": receipt["sha256"]}, [path])
    with pytest.raises(evidence.EvidenceError, match="missing its prompt/photo"):
        surfaces.semantic_mask(job, receipt, camera)


def test_check_only_semantics_never_create_structural_fit(wall_scene):
    job, pointer, receipt = wall_scene
    stage_masks(job, receipt, checks_only=True)
    result = surfaces.fit(job, receipt["structure_id"])
    assert result["accepted_points"] == 0 and result["planes"] == []
    assert result["status"] == "no-supported-wall"
    assert scenes.active(job) == pointer


def test_changed_check_masks_cannot_change_fitted_planes(wall_scene):
    job, _, receipt = wall_scene
    stage_masks(job, receipt)
    first = surfaces.fit(job, receipt["structure_id"])
    other = surfaces.prepare(job, 0, [1, 2, 3, 4, 5, 6])
    stage_masks(job, other, invert_checks=True)
    second = surfaces.fit(job, other["structure_id"])
    assert first["accepted_points"] == second["accepted_points"]
    assert [{key: value for key, value in plane.items() if key != "appearance_checks"} for plane in first["planes"]] == [
        {key: value for key, value in plane.items() if key != "appearance_checks"} for plane in second["planes"]]
    assert first["planes"][0]["appearance_checks"] != second["planes"][0]["appearance_checks"]


@pytest.mark.parametrize("defect", ["dimensions", "scores", "type", "prompt", "missing", "lineage"])
def test_invalid_mask_evidence_refuses_without_active_change(wall_scene, defect):
    job, pointer, receipt = wall_scene
    stage_masks(job, receipt, defect=defect)
    with pytest.raises(evidence.EvidenceError):
        surfaces.fit(job, receipt["structure_id"])
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("identifiers", [[1, 2], [1, 2, 3, 3], [1, 2, 3, 999], [True, 2, 3, 4]])
def test_invalid_or_unknown_structural_photo_ids_refuse(selected_scene, identifiers):
    job, pointer = selected_scene
    with pytest.raises(evidence.EvidenceError):
        surfaces.prepare(job, 0, identifiers)
    assert scenes.active(job) == pointer


def test_stale_input_and_corrupt_source_or_prepared_photo_refuse(wall_scene):
    job, pointer, receipt = wall_scene
    with pytest.raises(evidence.EvidenceError, match="Active scene changed"):
        surfaces.prepare(job, 1, [1, 2, 3, 4])
    path = surfaces.directory(job, receipt["structure_id"]) / receipt["cameras"][0]["photo"]
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(evidence.EvidenceError, match="Prepared structural photo changed"):
        surfaces.verify(job, receipt, prepared_files=True)
    manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
    with pytest.raises(evidence.EvidenceError, match="older active revision"):
        surfaces.verify(job, receipt)


def test_parallel_or_duplicate_views_do_not_establish_structural_support():
    points = np.array([[0., 0., 0.]])
    cameras = [{"group": str(index), "split": "fit", "center": [0, 0, 2]} for index in range(3)]
    observed = np.ones((3, 1), dtype=bool)
    accepted, _, _ = surfaces.membership(points, cameras, observed, observed)
    assert not accepted.any()
    cameras[1]["center"] = [1, 0, 2]
    assert surfaces.membership(points, cameras, observed, observed)[0].all()
    cameras[1]["group"] = cameras[0]["group"]
    with pytest.raises(evidence.EvidenceError, match="Duplicate timestamp"):
        surfaces.membership(points, cameras, observed, observed)


def test_horizontal_or_line_like_semantic_points_are_not_walls():
    points = np.array([[horizontal, 0., vertical] for horizontal in np.linspace(-2, 2, 25) for vertical in np.linspace(-2, 2, 25)])
    assert surfaces.fit_planes(points, np.arange(len(points)), [0, 2, 0]) == []
    points[:, 2] = 0
    assert surfaces.fit_planes(points, np.arange(len(points)), [0, 2, 0]) == []


def test_structural_observations_require_bidirectional_track_link(wall_scene):
    job, _, receipt = wall_scene
    source = evidence.load(job, retain_image_observations=True)
    camera = receipt["cameras"][0]
    mask = np.ones((camera["height"], camera["width"]), bool)
    original = surfaces.track_observations(source, camera, mask)[0]
    assert 0 in original
    source.records[0]["track"] = source.records[0]["track"].copy()
    source.records[0]["track"][0, 1] += 1
    assert 0 not in surfaces.track_observations(source, camera, mask)[0]


def test_structural_review_api_returns_only_retained_artifacts(wall_scene):
    job, pointer, receipt = wall_scene
    stage_masks(job, receipt)
    result = surfaces.fit(job, receipt["structure_id"])
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio/structural-surfaces"
        listing = client.get(base)
        assert listing.status_code == 200
        study = listing.json()["studies"][0]
        assert study["result"]["sha256"] == result["sha256"] and not study["stale"]
        artifact = base + f"/{receipt['structure_id']}/artifact"
        image = client.get(artifact, params={"name": receipt["cameras"][0]["photo"]})
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert client.get(artifact, params={"name": "membership.npz", "stage": "result"}).status_code == 200
        assert client.get(artifact, params={"name": "../../meta.json"}).status_code == 409
        assert client.get(artifact, params={"name": "membership.npz", "stage": "unknown"}).status_code == 422
    assert scenes.active(job) == pointer
