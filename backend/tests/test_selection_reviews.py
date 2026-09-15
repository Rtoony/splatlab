import numpy as np
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import artifact_manifest as manifests
import background_recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import selection_reviews as reviews
import splat_route
from test_background_recovery import observed_scene


@pytest.fixture
def selected_scene(observed_scene, monkeypatch, tmp_path):
    job = observed_scene
    monkeypatch.setattr(background_recovery, "WORKER_LOCK", tmp_path / "recovery.lock")
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", job.parent)
    manifests.atomic_write_json(job / "meta.json", {**manifests.read_json(job / "meta.json"), "output_dir": str(job)})
    (job / "processed/splatfacto/run/config.yml").write_text("pipeline:\n  model:\n    camera_optimizer:\n      mode: 'off'\n")
    positions = np.array([[-.2, -.2, .5], [.2, -.2, .5], [-.2, .2, .5], [.2, .2, .5],
                          [0, 0, .5], [.1, 0, .5], [0, .1, .5]], dtype="<f4")
    preview = job / "_preview"
    preview.mkdir()
    manifests.atomic_write_json(job / "_world/world_manifest.json", {"elements": []})
    (preview / "splat.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 7\nproperty float x\nproperty float y\nproperty float z\nend_header\n" + positions.tobytes())
    pointer = scenes.initialize(job, {"elements": [], "units": "meters", "meters_per_unit": 1}, scenes.source_fingerprint(job))
    payload = scenes.read_revision(job, pointer["revision_id"])
    payload.pop("revision_id")
    payload.pop("content_sha256")
    payload["state"]["viewer"]["elements"] = [{"slug": "box", "label": "cardboard box", "provenance": "observed"}]
    payload["state"]["selections"] = {"n_rows": 7, "elements": {
        "box": {"rows": [0, 1, 2, 3], "coordinate_verified": True}, "other": {"rows": [5]}}}
    revision = scenes.write_revision(job, payload)
    pointer = {**pointer, "revision_id": revision["revision_id"]}
    manifests.atomic_write_json(scenes.root(job) / "active.json", pointer)
    return job, pointer


def stage_models(job, receipt):
    output = reviews.directory(job, receipt["review_id"])
    (output / "visibility").mkdir()
    (output / "masks/cardboard-box").mkdir(parents=True)
    manifest = output / "sam3_manifest.json"
    manifests.atomic_write_json(manifest, {"cardboard box": {"slug": "cardboard-box"}})
    masks, visible_files = [manifest], []
    for camera in receipt["cameras"]:
        image_id = camera["image_id"]
        contribution = np.zeros((64, 64), dtype=np.float32)
        contribution[25:30, 25:30] = 1
        mask = np.zeros((1, 64, 64), dtype=bool)
        mask[:, 24:32, 24:32] = True
        mask_file = output / f"masks/cardboard-box/cam_{image_id:03d}.npz"
        np.savez_compressed(mask_file, masks=mask, scores=np.array([.9]))
        masks.append(mask_file)
        visible_file = output / f"visibility/cam_{image_id:03d}.npz"
        np.savez_compressed(visible_file, core=contribution, gaussian_ids=np.arange(7), pixels=np.tile([26, 26], (7, 1)))
        overlay = output / f"visibility/overlay-{image_id}.png"
        Image.new("RGB", (64, 64)).save(overlay)
        visible_files += [visible_file, overlay]
    reviews.record_run(job, receipt, "mask-run.json", {"method": "synthetic test masks, not model evidence"}, masks)
    reviews.record_run(job, receipt, "visibility/run.json", {"input_rows_sha256": receipt["artifacts"]["rows.npz"]["sha256"]}, visible_files)


def stage_candidate(job, receipt, candidate):
    output = reviews.directory(job, receipt["review_id"])
    (output / "candidate-visibility").mkdir()
    files = []
    for camera in receipt["cameras"]:
        image_id = camera["image_id"]
        contribution = np.zeros((64, 64), dtype=np.float32)
        contribution[24:32, 24:32] = 1
        visible = output / f"candidate-visibility/cam_{image_id:03d}.npz"
        np.savez_compressed(visible, core=contribution)
        overlay = output / f"candidate-visibility/overlay-{image_id}.png"
        Image.new("RGB", (64, 64)).save(overlay)
        files += [visible, overlay]
    reviews.record_run(job, receipt, "candidate-visibility/run.json", {"input_rows_sha256": candidate["artifacts"]["candidate-rows.npz"]["sha256"]}, files)


def stage_contribution(job, receipt):
    output = reviews.directory(job, receipt["review_id"])
    stage = output / "contribution"
    stage.mkdir()
    files, views = [], []
    for camera in receipt["cameras"]:
        inside = np.array([1, 1, 1, 1, .9, 1, .5], dtype=np.float32)
        if camera["split"] == "check":
            inside[6] = 1
        path = stage / f"cam_{camera['image_id']:03d}.npz"
        np.savez_compressed(path, inside=inside, outside=1 - inside, total=np.ones(7, dtype=np.float32))
        files.append(path)
        views.append({"image_id": camera["image_id"], "split": camera["split"], "selected_mask": 0, "reason": None})
    return reviews.record_run(job, receipt, "contribution/run.json", {
        "method": reviews.CONTRIBUTION_METHOD, "views": views,
        "mask_run_sha256": reviews.read_record(job, receipt["review_id"], "mask-run.json")["sha256"],
        "visibility_run_sha256": reviews.read_record(job, receipt["review_id"], "visibility/run.json")["sha256"],
    }, files)


@pytest.mark.parametrize("metres", [1, 2.5])
def test_real_camera_projection_matches_normalized_renderer_frame(observed_scene, metres):
    manifests.atomic_write_json(observed_scene / "meta.json", {"meters_per_unit": metres, "scale_generation": 1})
    source = evidence.load(observed_scene)
    camera = source.cameras[2]
    raw_points = np.array([[.2, .1, .5], [-.1, .2, .6]])
    transform, parameters = reviews.raw_camera(camera, metres, 32, 32)
    rigid = np.asarray(transform)
    np.testing.assert_allclose(rigid[:3, :3] @ rigid[:3, :3].T, np.eye(3))
    coordinates = raw_points @ rigid[:3, :3].T + rigid[:3, 3]
    pixels = coordinates[:, :2] / coordinates[:, 2:] * parameters[:2] + parameters[2:]
    expected, _ = evidence.project(raw_points @ evidence.Y_UP.T * metres, camera)
    np.testing.assert_allclose(pixels, expected / 2)


@pytest.mark.parametrize("defect", ["distortion", "anisotropic"])
def test_unsupported_camera_frames_refuse(observed_scene, defect):
    camera = evidence.load(observed_scene).cameras[1]
    if defect == "distortion":
        camera["parameters"] = np.append(camera["parameters"], .1)
    else:
        camera["world_to_camera"][0, :3] *= 2
    with pytest.raises(evidence.EvidenceError):
        reviews.raw_camera(camera, 1, 64, 64)


@pytest.mark.parametrize("mode", ["SO3xR3", "SE3", "on"])
def test_optimized_camera_deltas_cannot_be_silently_ignored(selected_scene, mode):
    job, _ = selected_scene
    path = job / "processed/splatfacto/run/config.yml"
    path.write_text(path.read_text().replace("'off'", mode))
    with pytest.raises(evidence.EvidenceError, match="camera deltas"):
        reviews.require_fixed_cameras(job, {})


def test_prepare_is_read_only_for_scene_with_distinct_fit_and_check_groups(selected_scene):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    assert scenes.active(job) == pointer
    assert receipt["core_count"] == 4
    assert 4 <= len(receipt["cameras"]) <= 8
    assert len({camera["group"] for camera in receipt["cameras"]}) == len(receipt["cameras"])
    assert any(camera["split"] == "check" for camera in receipt["cameras"])
    reviews.verify(job, receipt)
    reviews.verify_prepared_files(job, receipt)
    with pytest.raises(evidence.EvidenceError, match="does not belong"):
        reviews.artifact(job, receipt["review_id"], "../../meta.json")


def test_camera_group_reuse_does_not_create_independent_evidence(observed_scene):
    cameras = evidence.load(observed_scene).cameras
    for camera in cameras.values():
        camera["group"] = "same-timestamp"
    points = np.array([[-.2, .5, -.2], [.2, .5, -.2], [-.2, .5, .2], [.2, .5, .2]])
    with pytest.raises(evidence.EvidenceError, match="four distinct"):
        reviews.choose_views(cameras, points)


def test_visible_core_association_and_ambiguous_masks():
    contribution = np.ones((20, 20))
    masks = np.zeros((2, 20, 20), dtype=bool)
    masks[0, :12] = True
    masks[1, 8:] = True
    assert "ambiguous" in reviews.match_mask(masks, np.array([.9, .9]), contribution)[1]
    assert reviews.match_mask(masks, np.array([.9, .2]), contribution)[0] == 0
    assert reviews.match_mask(masks, np.array([.9, .9]), contribution * .001)[0] is None


@pytest.mark.parametrize("defect", ["nan", "negative", "shape", "score", "mask"])
def test_invalid_model_arrays_refuse(defect):
    contribution = np.ones((4, 4))
    masks = np.ones((1, 4, 4), dtype=bool)
    scores = np.array([.9])
    if defect == "nan":
        contribution[0, 0] = np.nan
    elif defect == "negative":
        contribution[0, 0] = -1
    elif defect == "shape":
        scores = scores[:, None]
    elif defect == "score":
        scores[0] = 2
    else:
        masks = masks.astype(float)
    with pytest.raises(evidence.EvidenceError):
        reviews.match_mask(masks, scores, contribution)


def test_group_votes_preserve_core_exclude_claimed_and_never_use_checks():
    def view(group, positive, negative=(), split="fit"):
        return {"group": group, "split": split, "positive": np.array(positive, dtype=int), "negative": np.array(negative, dtype=int)}
    observations = [view("first", [1, 2, 3, 4], [0]), view("first", [5], [4]),
                    view("second", [1, 2, 3, 4, 5], [0]), view("check", [6], split="check")]
    spatial = np.ones(7, dtype=bool)
    spatial[3] = False
    candidate, positive, negative, conflicts = reviews.vote_rows(7, observations, np.array([0]), np.array([2]), spatial)
    assert candidate.tolist() == [0, 1, 5]
    assert positive[4] == 1 and negative[4] == 1
    assert positive[6] == 0 and positive[1] == 2 and conflicts == 1


def test_contribution_votes_abstain_for_occluded_tiny_and_mixed_footprints():
    inside = np.array([0, .09, .4, 8, 2, 5], dtype=np.float32)
    total = np.array([0, .1, .5, 10, 10, 10], dtype=np.float32)
    positive, negative = reviews.contribution_rows(inside, total - inside, total, 6)
    assert positive.tolist() == [2, 3]
    assert negative.tolist() == [4]


@pytest.mark.parametrize("defect", ["shape", "integer", "nan", "infinite", "negative", "partition"])
@pytest.mark.parametrize("field", ["inside", "outside", "total"])
def test_invalid_footprint_weights_refuse(defect, field):
    weights = {"inside": np.full(3, .9), "outside": np.full(3, .1), "total": np.ones(3)}
    if defect == "shape":
        weights[field] = weights[field][:, None]
    elif defect == "integer":
        weights[field] = weights[field].astype(np.int64)
    else:
        weights[field][0] = {"nan": np.nan, "infinite": np.inf, "negative": -1, "partition": 9}[defect]
    with pytest.raises(evidence.EvidenceError, match="Contribution weights"):
        reviews.contribution_rows(weights["inside"], weights["outside"], weights["total"], 3)


def test_weighted_refinement_retains_lineage_and_never_uses_check_votes(selected_scene):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    identifier = receipt["review_id"]
    stage_models(job, receipt)
    contribution = stage_contribution(job, receipt)
    candidate = reviews.refine(job, identifier, contribution_weighted=True, spatial_margin_m=.2)
    assert candidate["candidate_count"] == 5
    assert candidate["other_instance_conflicts_excluded"] == 1
    assert candidate["contribution_run_sha256"] == contribution["sha256"]
    assert candidate["recipe"]["spatial_margin_m"] == .2
    assert candidate["recipe"]["mixed_footprints_abstain"] is True
    stage_candidate(job, receipt, candidate)
    result = reviews.finalize(job, identifier)
    assert reviews.artifact(job, identifier, "contribution/run.json").is_file()
    assert all(view["candidate_inside_alpha_mass_fraction"] < 1 for view in result["views"])
    assert all(view["candidate_outside_alpha_mass_fraction"] == pytest.approx(.02) for view in result["views"])
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("margin", [-.01, .21, np.nan, np.inf, True, "0.05"])
def test_invalid_selection_margin_refuses_before_workers(selected_scene, margin):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    with pytest.raises(evidence.EvidenceError, match="spatial margin"):
        reviews.refine(job, receipt["review_id"], contribution_weighted=True, spatial_margin_m=margin)
    assert scenes.active(job) == pointer


@pytest.mark.parametrize("defect", ["missing", "method", "mask-run", "visibility-run", "order", "split", "association", "files", "damaged-array"])
def test_weighted_refinement_refuses_unbound_contribution(selected_scene, defect):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    identifier = receipt["review_id"]
    output = reviews.directory(job, identifier)
    stage_models(job, receipt)
    if defect != "missing":
        run = stage_contribution(job, receipt)
        if defect == "damaged-array":
            path = output / next(iter(run["artifacts"]))
            path.write_bytes(path.read_bytes() + b"changed")
        else:
            files = [output / name for name in run["artifacts"]]
            if defect == "method":
                run["method"] = "unsupported"
            elif defect in ("mask-run", "visibility-run"):
                run[defect.replace("-", "_") + "_sha256"] = "0" * 64
            elif defect == "order":
                run["views"].reverse()
            elif defect == "split":
                run["views"][0]["split"] = "check"
            elif defect == "association":
                run["views"][0]["selected_mask"] = 1
            elif defect == "files":
                files.pop()
            reviews.seal(job, output, "contribution/run.json", {key: value for key, value in run.items() if key not in ("sha256", "artifacts")}, files)
    with pytest.raises(evidence.EvidenceError):
        reviews.refine(job, identifier, contribution_weighted=True)
    assert scenes.active(job) == pointer
    assert not (output / "candidate-rows.npz").exists()


def test_weighted_candidate_rejects_changed_contribution_before_finalization(selected_scene):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    identifier = receipt["review_id"]
    output = reviews.directory(job, identifier)
    stage_models(job, receipt)
    run = stage_contribution(job, receipt)
    candidate = reviews.refine(job, identifier, contribution_weighted=True)
    assert candidate["recipe"]["spatial_margin_m"] == .05
    stage_candidate(job, receipt, candidate)
    run["new_worker"] = True
    reviews.seal(job, output, "contribution/run.json", {key: value for key, value in run.items() if key not in ("sha256", "artifacts")}, [output / name for name in run["artifacts"]])
    with pytest.raises(evidence.EvidenceError, match="contribution lineage"):
        reviews.finalize(job, identifier)
    assert scenes.active(job) == pointer


def test_full_candidate_seal_and_api_never_activate_rows(selected_scene):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    identifier = receipt["review_id"]
    stage_models(job, receipt)
    candidate = reviews.refine(job, identifier)
    assert candidate["candidate_count"] == 6 and candidate["other_instance_conflicts_excluded"] == 1
    stage_candidate(job, receipt, candidate)
    result = reviews.finalize(job, identifier)
    assert all(view["candidate_mask_coverage"] == 1 for view in result["views"])
    assert scenes.active(job) == pointer
    with pytest.raises(evidence.EvidenceError, match="already sealed"):
        reviews.finalize(job, identifier)
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    with TestClient(app) as client:
        base = f"/jobs/{job.name}/studio/selection-reviews"
        assert client.get(base).json()["reviews"][0]["stale"] is False
        assert client.get(f"{base}/{identifier}/rows?candidate=true").json()["rows"] == [0, 1, 2, 3, 4, 6]
        assert client.get(f"{base}/{identifier}/artifact", params={"name": "mask-overlay-1.png"}).status_code == 200
        manifests.atomic_write_json(scenes.root(job) / "active.json", {**pointer, "generation": 1})
        assert client.get(base).json()["reviews"][0]["stale"] is True
        assert client.get(f"{base}/{identifier}/rows").status_code == 409


@pytest.mark.parametrize("changed", ["prepared", "model", "candidate"])
def test_changed_stage_inputs_are_refused_without_scene_mutation(selected_scene, changed):
    job, pointer = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    identifier = receipt["review_id"]
    output = reviews.directory(job, identifier)
    stage_models(job, receipt)
    if changed == "candidate":
        candidate = reviews.refine(job, identifier)
        stage_candidate(job, receipt, candidate)
        target = output / "candidate-rows.npz"
        action = reviews.finalize
    else:
        target = output / ("rows.npz" if changed == "prepared" else "sam3_manifest.json")
        action = reviews.refine
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(evidence.EvidenceError, match="changed"):
        action(job, identifier)
    assert scenes.active(job) == pointer


def test_unrecorded_workers_and_concurrent_writers_refuse(selected_scene):
    job, _ = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    with pytest.raises(evidence.EvidenceError, match="Missing sealed"):
        reviews.refine(job, receipt["review_id"])
    with reviews.worker_slot(job, receipt["review_id"]):
        with pytest.raises(evidence.EvidenceError, match="active worker"):
            reviews.refine(job, receipt["review_id"])


def test_changed_capture_photo_and_stale_generation_refuse(selected_scene):
    job, _ = selected_scene
    with pytest.raises(evidence.EvidenceError, match="Active scene changed"):
        reviews.prepare(job, "box", 1)
    receipt = reviews.prepare(job, "box", 0)
    photo = job / receipt["cameras"][0]["image_key"]
    photo.write_bytes(photo.read_bytes() + b"changed")
    with pytest.raises(evidence.EvidenceError, match="changed"):
        reviews.verify(job, receipt)


def test_same_size_blob_damage_and_cross_review_result_refuse(selected_scene):
    job, _ = selected_scene
    receipt = reviews.prepare(job, "box", 0)
    identifier = receipt["review_id"]
    rows = reviews.artifact(job, identifier, "rows.npz")
    original = rows.read_bytes()
    rows.chmod(0o600)
    rows.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(evidence.EvidenceError, match="integrity check"):
        reviews.artifact(job, identifier, "rows.npz")
    output = reviews.directory(job, identifier)
    reviews.seal(job, output, "result.json", {"prepared_receipt_sha256": "0" * 64}, [])
    with pytest.raises(evidence.EvidenceError, match="different prepared"):
        reviews.read(job, identifier, result=True)
