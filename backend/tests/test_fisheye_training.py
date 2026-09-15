from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import artifact_manifest as manifests
from fisheye_reference import build_bundle
from fisheye_training import heldout_frames, virtual_pose, prepare, raw_mask_inputs
from fisheye_evaluation import image_scores, summarize_scores
from test_fisheye_reference import frozen_inputs


@pytest.fixture
def inputs():
    cameras = {"frames": [{"source_image": "train.jpg", "split": "train", "source_group": "time-1", "physical_lens": "lens-0"},
                          {"source_image": "val.jpg", "split": "val", "source_group": "time-2", "physical_lens": "lens-0"},
                          {"source_image": "test.jpg", "split": "test", "source_group": "time-3", "physical_lens": "lens-1"}]}
    reference = {"source_hashes": {"model/cameras.txt": "a" * 64}}
    localization = {"schema": "dev.splatlab.frozen-holdout-localization/v1", "status": "localized-needs-review",
                    "geometry_refined": False, "intrinsics_refined": False, "source_hashes": deepcopy(reference["source_hashes"]),
                    "poses": [{"image": frame["source_image"], "split": frame["split"], "group_id": frame["source_group"],
                               "physical_lens": frame["physical_lens"], "status": "localized-needs-review",
                               "supported_cone_inliers": 40, "supported_inlier_fraction": .9, "p95_inlier_reprojection_px": 2,
                               "camera_to_world_opencv": np.eye(4).tolist()} for frame in cameras["frames"][1:]]}
    return cameras, reference, localization


def test_only_exact_source_bound_heldouts_enter_training_bundle(inputs):
    assert set(heldout_frames(*inputs)) == {"val.jpg", "test.jpg"}


@pytest.mark.parametrize("change", ["hash", "missing", "duplicate", "leaked", "refined"])
def test_invalid_localization_is_rejected(inputs, change):
    localization = inputs[2]
    if change == "hash":
        localization["source_hashes"]["model/cameras.txt"] = "b" * 64
    elif change == "missing":
        localization["poses"].pop()
    elif change == "duplicate":
        localization["poses"].append(deepcopy(localization["poses"][0]))
    elif change == "leaked":
        localization["poses"][0]["image"] = "train.jpg"
    else:
        localization["geometry_refined"] = True
    with pytest.raises(ValueError):
        heldout_frames(*inputs)


@pytest.mark.parametrize("key,value", [("split", "test"), ("group_id", "time-3"), ("physical_lens", "lens-1"),
    ("supported_cone_inliers", 19), ("supported_cone_inliers", float("nan")), ("supported_inlier_fraction", .1), ("supported_inlier_fraction", float("nan")),
    ("p95_inlier_reprojection_px", 4.1), ("p95_inlier_reprojection_px", float("inf")),
    ("camera_to_world_opencv", np.zeros((4, 4)).tolist())])
def test_bad_identity_quality_or_pose_is_rejected(inputs, key, value):
    inputs[2]["poses"][0][key] = value
    with pytest.raises(ValueError):
        heldout_frames(*inputs)


def test_virtual_pose_preserves_center_and_rotates_camera_axes(inputs):
    record = inputs[2]["poses"][0]
    record["camera_to_world_opencv"][0][3] = 2
    actual = np.array(virtual_pose(record, 0))
    np.testing.assert_array_equal(actual[:3, 3], [2, 0, 0])
    np.testing.assert_array_equal(actual[:3, :3], np.diag([1, -1, -1]))
    turned = np.array(virtual_pose(record, 35))
    np.testing.assert_array_equal(turned[:3, 3], actual[:3, 3])
    np.testing.assert_allclose(turned[:3, :3].T @ turned[:3, :3], np.eye(3), atol=1e-12)
    assert not np.allclose(actual[:3, :3], turned[:3, :3])


def test_timestamp_leakage_and_inconsistent_virtual_source_are_rejected(inputs):
    original = deepcopy(inputs)
    inputs[0]["frames"][1]["source_group"] = "time-1"
    with pytest.raises(ValueError, match="timestamp"):
        heldout_frames(*inputs)
    duplicate = deepcopy(original[0]["frames"][1])
    duplicate["physical_lens"] = "lens-1"
    original[0]["frames"].append(duplicate)
    with pytest.raises(ValueError, match="physical source"):
        heldout_frames(*original)


def test_masked_scores_do_not_reward_black_excluded_pixels():
    target = np.zeros((2, 2, 3), dtype=float)
    rendered = target.copy()
    rendered[0, 0] = .5
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    scores = image_scores(target, rendered, mask)
    assert scores["full_mse"] == .0625 and scores["masked_mse"] == .25
    assert scores["masked_psnr_db"] == pytest.approx(6.020599913)
    assert summarize_scores([scores, scores])["masked_psnr_db"] == scores["masked_psnr_db"]


def test_invalid_and_perfect_metrics_are_explicit():
    target = np.zeros((2, 2, 3), dtype=float)
    mask = np.ones((2, 2), dtype=bool)
    scores = image_scores(target, target, mask)
    assert scores["perfect_match"] is True and scores["full_psnr_db"] is None
    with pytest.raises(ValueError):
        image_scores(target, target, ~mask)
    with pytest.raises(ValueError):
        image_scores(target, target + np.nan, mask)


@pytest.mark.parametrize("with_raw_masks", [False, True])
def test_real_preparation_preserves_source_groups_poses_and_hashes(frozen_inputs, tmp_path, with_raw_masks):
    pilot_root, sfm_root, pilot = frozen_inputs
    reference_root = tmp_path / "reference"
    reference_receipt = build_bundle(pilot_root, sfm_root, reference_root, 64)
    localization = {"schema": "dev.splatlab.frozen-holdout-localization/v1", "status": "localized-needs-review",
                    "geometry_refined": False, "intrinsics_refined": False, "source_hashes": reference_receipt["source_hashes"],
                    "poses": [{"image": view["image"], "split": view["split"], "group_id": view["group_id"],
                               "physical_lens": view["lens_id"], "status": "localized-needs-review",
                               "supported_cone_inliers": 40, "supported_inlier_fraction": .9, "p95_inlier_reprojection_px": 2,
                               "camera_to_world_opencv": np.eye(4).tolist()} for view in pilot["views"] if view["split"] != "train"]}
    localization_root = tmp_path / "localization"
    localization_root.mkdir()
    localization_path = localization_root / "receipt.json"
    manifests.atomic_write_json(localization_path, localization)
    mask_review = None
    if with_raw_masks:
        mask_root = tmp_path / "raw-masks"
        mask_root.mkdir()
        records = []
        for view in pilot["views"]:
            mask = np.full((128, 128), 255, dtype=np.uint8)
            mask[55:73, 55:73] = 0
            filename = view["image"].replace("/", "-") + ".png"
            Image.fromarray(mask).save(mask_root / filename)
            records.append({"image": view["image"], "split": view["split"], "source_sha256": view["sha256"],
                            "mask": filename, "mask_sha256": manifests.sha256_file(mask_root / filename)})
        mask_review = mask_root / "receipt.json"
        manifests.atomic_write_json(mask_review, {"schema": "dev.splatlab.raw-lens-exclusions/v1",
            "status": "prepared-needs-visual-review", "semantic_ground_truth": False, "width": 128,
            "pilot_sha256": reference_receipt["source_hashes"]["pilot"], "views": records})
    sources = {str(path): manifests.sha256_file(path) for root in (pilot_root, sfm_root, reference_root, localization_root)
               for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "training"
    receipt = prepare(reference_root, localization_path, sfm_root, output, mask_review)
    document = json.loads((output / "transforms.json").read_text())
    originals = json.loads((reference_root / "camera-set.json").read_text())["frames"]
    actual = {frame["file_path"]: frame for frame in document["frames"]}
    assert receipt["splits"] == {"train": 18, "val": 6, "test": 6}
    assert receipt["registration"] is None and receipt["mask_review"] == "visual-review-required"
    assert document["splatlab_evaluation"]["semantic_masks_during_training"] is with_raw_masks
    if with_raw_masks:
        assert receipt["raw_mask_source_hashes"][str(mask_review)] == manifests.sha256_file(mask_review)
        centered = next(frame for frame in originals if frame["virtual_yaw_deg"] == 0)
        mask = np.asarray(Image.open(output / centered["mask_path"]))
        assert not mask[28:36, 28:36].any() and mask.any()
    for frame in originals:
        assert frame["file_path"] in document[frame["split"] + "_filenames"]
        assert actual[frame["file_path"]]["transform_matrix"] is not None
        if frame["split"] == "train":
            assert actual[frame["file_path"]]["transform_matrix"] == frame["transform_matrix"]
    assert all(manifests.sha256_file(output / name) == digest for name, digest in receipt["files"].items())
    assert sources == {name: manifests.sha256_file(Path(name)) for name in sources}
    with pytest.raises(ValueError, match="new training-input"):
        prepare(reference_root, localization_path, sfm_root, output)


@pytest.mark.parametrize("change", [None, "pilot", "source", "split", "duplicate", "missing", "mask-bytes", "path", "nonbinary"])
def test_raw_exclusions_are_bound_to_exact_sources_and_mask_bytes(tmp_path, change):
    image_path = tmp_path / "mask.png"
    Image.new("L", (8, 8), 255).save(image_path)
    record = {"image": "lens-0/frame-000000.jpg", "split": "train", "source_sha256": "a" * 64,
              "mask": "mask.png", "mask_sha256": manifests.sha256_file(image_path)}
    receipt = {"schema": "dev.splatlab.raw-lens-exclusions/v1", "status": "prepared-needs-visual-review",
               "semantic_ground_truth": False, "width": 8, "pilot_sha256": "b" * 64, "views": [record]}
    cameras = {"frames": [{"source_image": record["image"], "split": "train"}]}
    source = {"source_hashes": {"pilot": "b" * 64, "image/" + record["image"]: "a" * 64}}
    if change == "pilot":
        receipt["pilot_sha256"] = "c" * 64
    elif change == "source":
        record["source_sha256"] = "c" * 64
    elif change == "split":
        record["split"] = "test"
    elif change == "duplicate":
        receipt["views"].append(deepcopy(record))
    elif change == "missing":
        receipt["views"] = []
    elif change == "mask-bytes":
        Image.new("L", (8, 8), 0).save(image_path)
    elif change == "path":
        record["mask"] = "../mask.png"
    elif change == "nonbinary":
        Image.new("L", (8, 8), 127).save(image_path)
        record["mask_sha256"] = manifests.sha256_file(image_path)
    review_path = tmp_path / "receipt.json"
    manifests.atomic_write_json(review_path, receipt)
    if change is not None:
        with pytest.raises(ValueError):
            raw_mask_inputs(review_path, cameras, source)
    else:
        masks, snapshot = raw_mask_inputs(review_path, cameras, source)
        assert masks[record["image"]].all()
        assert snapshot[image_path] == record["mask_sha256"]
