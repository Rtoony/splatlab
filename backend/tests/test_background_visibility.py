from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import background_recovery as recovery
import background_visibility as visibility
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import selection_reviews as selections
import support_surfaces as support
from test_background_recovery import observed_scene
from test_selection_reviews import selected_scene
from test_support_surfaces import support_scene, anchors_from_view


@pytest.fixture
def visibility_scene(support_scene):
    job, pointer = support_scene
    original = recovery.build(job, "box", 0, 64, anchors_from_view(support.inspect(job, "box", 0, 1)))
    receipt = visibility.prepare(job, original["recovery_id"], 0)
    return job, pointer, original, receipt


def stage_visibility(job, receipt, only_checks=False, defect=None):
    output = visibility.directory(job, receipt["visibility_id"])
    size = receipt["texture_size"]
    files = []
    for camera in receipt["cameras"]:
        error = np.zeros((size, size), dtype=np.float32)
        if only_checks and camera["split"] == "fit":
            error[:] = 1
        elif not only_checks:
            error[:, :size // 2] = 1
        alpha = np.ones(size * size, dtype=np.float32)
        plane_depth = np.full(size * size, 3., dtype=np.float64)
        in_frame = np.ones(size * size, dtype=bool)
        qualified = visibility.qualify_samples(alpha, error.reshape(-1), plane_depth, in_frame)
        if defect == "decisions":
            qualified[:] = True
        path = output / f"camera-{camera['image_id']}.npz"
        np.savez_compressed(path, minimum_alpha=alpha, maximum_depth_error_m=error.reshape(-1),
            plane_depth_m=plane_depth, in_frame=in_frame, qualified=qualified)
        files.append(path)
    if defect == "missing-camera":
        files.pop()
    return selections.seal(job, output, "result.json", {
        "prepared_sha256": "0" * 64 if defect == "lineage" else receipt["sha256"],
        "method": "unverified" if defect == "method" else visibility.METHOD,
        "scope": "synthetic unit-test visibility, not model evidence",
    }, files)


def test_four_neighbors_reject_edges_occlusion_and_low_alpha():
    alpha = np.ones((4, 4), dtype=np.float32)
    depth = np.full((4, 4), 3., dtype=np.float32)
    depth[0, 1] = 2
    alpha[3, 3] = .5
    pixels = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 2.], [np.nan, 1.], [1., 1.]])
    plane_depth = np.array([3., 3., 3., 3., 3., -1.])
    arrays = visibility.sample_render(alpha, depth, pixels, plane_depth)
    assert arrays["qualified"].tolist() == [False, True, False, False, False, False]
    assert arrays["maximum_depth_error_m"][0] == 1
    assert arrays["minimum_alpha"][2] == .5
    assert not arrays["in_frame"][3:].any()


@pytest.mark.parametrize("defect", ["shape", "integer", "alpha-range", "nan", "infinite-depth", "negative-error", "invalid-frame"])
def test_malformed_per_cell_evidence_refuses(defect):
    alpha, error, depth, in_frame = np.ones(3), np.zeros(3), np.ones(3), np.ones(3, dtype=bool)
    if defect == "shape":
        alpha = alpha[:, None]
    elif defect == "integer":
        error = error.astype(int)
    elif defect == "alpha-range":
        alpha[0] = 2
    elif defect == "nan":
        error[0] = np.nan
    elif defect == "infinite-depth":
        depth[0] = np.inf
    elif defect == "negative-error":
        error[0] = -1
    else:
        in_frame = in_frame.astype(int)
    with pytest.raises(evidence.EvidenceError):
        visibility.qualify_samples(alpha, error, depth, in_frame)


def test_unknown_out_of_frame_depth_is_not_positive_visibility():
    values = visibility.qualify_samples(np.array([1., 1.]), np.array([np.inf, .03]), np.array([-1., 2.]), np.array([False, True]))
    assert values.tolist() == [False, True]


def test_preparation_and_derivation_preserve_original_and_never_activate(visibility_scene):
    job, pointer, original, receipt = visibility_scene
    assert receipt["base"] == pointer
    assert receipt["recovery_sha256"] == original["sha256"]
    with np.load(visibility.artifact(job, receipt["visibility_id"], "grid.npz"), allow_pickle=False) as grid:
        assert grid["points_world"].shape == (4096, 3)
    stage_visibility(job, receipt)
    qualified = visibility.derive(job, receipt["visibility_id"], 0)
    assert scenes.active(job) == pointer
    assert recovery.read(job, original["recovery_id"]) == original
    assert qualified["report"]["recipe"]["version"] == 3
    assert qualified["report"]["supported_fraction"] < original["report"]["supported_fraction"]
    assert qualified["report"]["plane"] == original["report"]["plane"]
    assert qualified["visibility_review"]["parent_recovery_sha256"] == original["sha256"]
    comparison = qualified["report"]["visibility_comparison"]
    assert comparison["rejected_parent_cells"] > 0 and comparison["newly_supported_cells"] == 0
    assert comparison["retained_parent_cells"] == comparison["qualified_cells"]
    assert all(check["parent_mean_rgb_absolute_error"] == check["qualified_mean_rgb_absolute_error"] for check in comparison["paired_appearance_checks"])
    assert "visibility-receipt.json" in qualified["artifacts"]
    assert "processed/splatfacto/run/config.yml" in qualified["sources"]
    with np.load(recovery.artifact(job, qualified["recovery_id"], "observations.npz"), allow_pickle=False) as rows:
        assert not rows["accepted"][:, :32].any()
        assert (rows["source_image_ids"][:, ~rows["accepted"].reshape(-1)] == -1).all()
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        listed = client.get(f"/jobs/{job.name}/studio/recoveries").json()["recoveries"]
        derived = next(item for item in listed if item["recovery_id"] == qualified["recovery_id"])
        assert derived["report"]["visibility_qualification"]["method"] == visibility.METHOD
        assert client.get(f"/jobs/{job.name}/studio/recoveries/{qualified['recovery_id']}/artifacts/visibility-result.json").status_code == 200
    with pytest.raises(evidence.EvidenceError, match="repeatedly filtering"):
        visibility.prepare(job, qualified["recovery_id"], 0)


def test_check_views_cannot_rescue_missing_fit_visibility(visibility_scene):
    job, pointer, original, receipt = visibility_scene
    stage_visibility(job, receipt, only_checks=True)
    with pytest.raises(evidence.EvidenceError, match="No multi-view-supported"):
        visibility.derive(job, receipt["visibility_id"], 0)
    assert scenes.active(job) == pointer
    assert recovery.read(job, original["recovery_id"]) == original


@pytest.mark.parametrize("defect", ["decisions", "missing-camera", "lineage", "method"])
def test_unbound_visibility_results_refuse_without_mutation(visibility_scene, defect):
    job, pointer, _, receipt = visibility_scene
    stage_visibility(job, receipt, defect=defect)
    with pytest.raises(evidence.EvidenceError):
        visibility.derive(job, receipt["visibility_id"], 0)
    assert scenes.active(job) == pointer


def test_stale_or_unbuilt_visibility_refuses(visibility_scene):
    job, pointer, original, receipt = visibility_scene
    with pytest.raises(evidence.EvidenceError, match="missing or corrupt"):
        visibility.derive(job, receipt["visibility_id"], 0)
    with pytest.raises(evidence.EvidenceError, match="current recovery"):
        visibility.prepare(job, original["recovery_id"], 1)
    manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
    with pytest.raises(evidence.EvidenceError, match="older recovery"):
        visibility.verify(job, receipt)


def test_changed_plane_or_same_size_evidence_damage_refuses(visibility_scene):
    job, pointer, original, receipt = visibility_scene
    result = stage_visibility(job, receipt)
    plane = deepcopy(original["report"]["plane"])
    plane["center"][0] += .01
    with pytest.raises(evidence.EvidenceError, match="different fitted"):
        visibility.qualification(job, receipt["visibility_id"], pointer, "box", 64, plane)
    path = scenes.blob_path(job, next(iter(result["artifacts"].values()))["sha256"])
    content = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(evidence.EvidenceError, match="corrupt"):
        visibility.derive(job, receipt["visibility_id"], 0)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("name", ["atlas.png", "observations.npz"])
def test_changed_parent_artifacts_refuse_after_preparation(visibility_scene, name):
    job, pointer, original, receipt = visibility_scene
    stage_visibility(job, receipt)
    path = recovery.artifact(job, original["recovery_id"], name)
    content = path.read_bytes()
    path.chmod(0o600)
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(evidence.EvidenceError, match="Original recovery artifact changed"):
        visibility.derive(job, receipt["visibility_id"], 0)
    assert scenes.active(job) == pointer


def test_sealed_grid_must_address_actual_texture_cell_centers(visibility_scene):
    job, _, _, receipt = visibility_scene
    output = visibility.directory(job, receipt["visibility_id"])
    points = visibility.verified_grid(job, receipt)
    points[0, 0] += .01
    np.savez_compressed(output / "grid.npz", points_world=points)
    modified = selections.seal(job, output, "receipt.json", {key: value for key, value in receipt.items() if key not in {"sha256", "artifacts"}}, [output / "grid.npz"])
    with pytest.raises(evidence.EvidenceError, match="texture cell centers"):
        visibility.verified_grid(job, modified)


@pytest.mark.parametrize("defect", ["missing-view", "extra-view", "shape", "nonboolean"])
def test_recovery_refuses_incomplete_or_malformed_visibility_contract(observed_scene, defect):
    source = evidence.load(observed_scene)
    bounds = [{"min": [-.2, 0, -.2], "max": [.2, .5, .2]}]
    _, _, _, report, _ = recovery.recover(source, bounds, 64)
    masks = {image_id: np.ones(4096, dtype=bool) for image_id in report["source_image_ids"] + report["appearance_check_image_ids"]}
    first = next(iter(masks))
    if defect == "missing-view":
        masks.pop(first)
    elif defect == "extra-view":
        masks[999] = masks[first]
    elif defect == "shape":
        masks[first] = masks[first].reshape(64, 64)
    else:
        masks[first] = masks[first].astype(int)
    with pytest.raises(evidence.EvidenceError, match="Visibility masks"):
        recovery.recover(source, bounds, 64, visibility_masks=masks)
