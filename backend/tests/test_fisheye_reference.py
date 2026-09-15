import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import artifact_manifest as manifests
import fisheye_reference as reference


def test_projection_has_known_equidistant_axis_and_diagonal_values():
    params = [100, 200, 256, 256, 0, 0, 0, 0]
    pixels, valid = reference.project_fisheye(np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]), params)
    np.testing.assert_allclose(pixels[:3], [[256, 256], [256 + 25 * math.pi, 256], [256, 256 + 50 * math.pi]])
    angle = math.atan(math.sqrt(2))
    np.testing.assert_allclose(pixels[3], [256 + 100 * angle / math.sqrt(2), 256 + 200 * angle / math.sqrt(2)])
    assert valid.all()


def test_projection_includes_all_four_radial_coefficients_and_refuses_behind_camera():
    params = [100, 100, 256, 256, .1, .02, .003, .0004]
    pixels, valid = reference.project_fisheye(np.array([[1, 0, 1], [0, 0, -1], [0, 0, 0], [1, 0, .01]]), params)
    angle = math.pi / 4
    expected = angle * (1 + .1 * angle**2 + .02 * angle**4 + .003 * angle**6 + .0004 * angle**8)
    assert pixels[0, 0] == pytest.approx(256 + 100 * expected)
    assert valid.tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="finite"):
        reference.project_fisheye(np.array([[np.nan, 0, 1]]), params)


def test_remap_uses_pixel_centers_and_keeps_pose_rays_consistent():
    camera = {"width": 512, "height": 512, "params": [140, 140, 256, 256, 0, 0, 0, 0]}
    samples, valid, virtual_rotation, focal = reference.remap_grid(camera, 64, 100, 35)
    pose = {"rotation": reference.yaw_rotation(17), "centre": np.array([1, 2, 3])}
    matrix = np.array(reference.camera_to_world(pose, virtual_rotation))
    np.testing.assert_array_equal(matrix[:3, 3], pose["centre"])
    np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1)
    for row, column in ((0, 0), (31, 31), (32, 32), (63, 63)):
        blender_ray = np.array([(column + .5 - 32) / focal, -(row + .5 - 32) / focal, -1])
        world_ray = matrix[:3, :3] @ blender_ray
        original_ray = pose["rotation"] @ world_ray
        projected, in_cone = reference.project_fisheye(original_ray, camera["params"])
        np.testing.assert_allclose(samples[row, column], projected - .5, atol=1e-10)
        assert bool(valid[row, column]) == bool(in_cone)


def test_virtual_views_keep_separate_physical_optical_centers():
    for yaw in (-35, 0, 35):
        for centre in ([0, 0, 0], [.04, 0, 0]):
            pose = {"rotation": np.eye(3), "centre": np.array(centre)}
            np.testing.assert_array_equal(np.array(reference.camera_to_world(pose, reference.yaw_rotation(yaw)))[:3, 3], centre)


def test_bilinear_resampling_has_no_wrap_and_blacks_invalid_pixels():
    values = np.zeros((4, 4, 3), dtype=np.uint8)
    values[0, 0], values[0, 1], values[1, 0], values[1, 1] = 0, 40, 80, 120
    samples = np.array([[[.5, .5], [0, 0]], [[3, 3], [-50, -50]]])
    image = reference.resample(Image.fromarray(values), samples, np.array([[True, True], [True, False]]))
    np.testing.assert_array_equal(np.array(image)[0, 0], [60, 60, 60])
    np.testing.assert_array_equal(np.array(image)[1, 1], [0, 0, 0])


@pytest.mark.parametrize("width,fov,yaw", [(True, 100, 0), (2048, 100, 0), (64, 180, 0), (64, 100, 90), (64, float("nan"), 0)])
def test_unbounded_views_are_rejected(width, fov, yaw):
    with pytest.raises(ValueError, match="bounded"):
        reference.remap_grid({}, width, fov, yaw)


@pytest.fixture
def frozen_inputs(tmp_path):
    pilot_root, sfm_root = tmp_path / "pilot", tmp_path / "sfm"
    model_root = sfm_root / "sparse/0"
    model_root.mkdir(parents=True)
    views, groups, poses = [], [], []
    for index, split in enumerate(("train", "train", "test", "val", "train")):
        group = f"frame-{index:06d}"
        groups.append({"group_id": group, "split": split, "source_decoded_frame_index": index})
        for lens in range(2):
            name = f"lens-{lens}/{group}.jpg"
            source = pilot_root / "images" / name
            source.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (128, 128), (40 * index, 40 * lens, 20)).save(source)
            views.append({"image": name, "split": split, "group_id": group, "lens_id": f"lens-{lens}",
                          "source_decoded_frame_index": index, "pts": index * 1001, "time_base": "1/30000",
                          "sha256": manifests.sha256_file(source)})
            if split == "train":
                poses.append(f"{index * 2 + lens + 1} 1 0 0 0 {-index - lens * .03} 0 0 {lens + 1} {name}\n\n")
    pilot = {"schema": "dev.splatlab.dual-fisheye-pilot/v1", "status": "decoded-needs-camera-and-mask-review",
             "projection": "raw-fisheye-resized-no-warp", "clock": {"paired_container_pts_match": True},
             "source": {"sha256": "a" * 64}, "groups": groups, "views": views}
    manifests.atomic_write_json(pilot_root / "pilot.json", pilot)
    manifests.atomic_write_json(sfm_root / "receipt.json", {
        "schema": "dev.splatlab.dual-fisheye-sfm-probe/v1", "status": "estimated-needs-rig-and-heldout-review",
        "heldout_used_in_mapping": False, "pilot_sha256": manifests.sha256_file(pilot_root / "pilot.json"),
        "train_images": [view["image"] for view in views if view["split"] == "train"]})
    (model_root / "cameras.txt").write_text("1 OPENCV_FISHEYE 128 128 35 35 64 64 0 0 0 0\n"
                                             "2 OPENCV_FISHEYE 128 128 36 36 64 64 0 0 0 0\n")
    (model_root / "images.txt").write_text("".join(poses))
    (model_root / "points3D.txt").write_text("1 0 0 5 128 128 128 .5 1 0 2 0\n")
    (model_root / "points.ply").write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 5\n")
    return pilot_root, sfm_root, pilot


def test_bundle_preserves_splits_sources_units_and_independent_handoff(frozen_inputs, tmp_path):
    pilot_root, sfm_root, pilot = frozen_inputs
    output = tmp_path / "reference"
    before = {str(path): manifests.sha256_file(path) for root in (pilot_root, sfm_root) for path in root.rglob("*") if path.is_file()}
    receipt = reference.build_bundle(pilot_root, sfm_root, output, 64)
    assert receipt["splits"] == {"train": 18, "test": 6, "val": 6}
    assert receipt["posed_views"] == 18 and receipt["training_ready"] is False
    cameras = json.loads((output / "camera-set.json").read_text())
    assert all((frame["transform_matrix"] is None) == (frame["split"] != "train") for frame in cameras["frames"])
    training = json.loads((output / "training-transforms.json").read_text())
    assert all(frame["file_path"].startswith("train/") for frame in training["frames"])
    assert training["val_filenames"] == training["test_filenames"] == []
    assert not (output / "transforms.json").exists()
    handoff = json.loads((output / "external-references.json").read_text())
    assert all(asset["units"] == "arbitrary" and asset["registration"] is None for asset in handoff["assets"])
    assert (output / "sparse.ply").read_bytes() == (sfm_root / "sparse/0/points.ply").read_bytes()
    assert before == {name: manifests.sha256_file(Path(name)) for name in before}
    for name, checksum in receipt["artifacts"].items():
        assert manifests.sha256_file(output / name) == checksum
    for frame in cameras["frames"]:
        mask = np.asarray(Image.open(output / frame["mask_path"]))
        assert set(np.unique(mask)) <= {0, 255}
    with pytest.raises(ValueError, match="new output"):
        reference.build_bundle(pilot_root, sfm_root, output, 64)


def test_changed_heldout_and_training_leakage_fail_before_output(frozen_inputs, tmp_path):
    pilot_root, sfm_root, pilot = frozen_inputs
    heldout = next(view for view in pilot["views"] if view["split"] == "test")
    path = pilot_root / "images" / heldout["image"]
    original = path.read_bytes()
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        reference.build_bundle(pilot_root, sfm_root, tmp_path / "bad", 64)
    assert not (tmp_path / "bad").exists()
    path.write_bytes(original)
    poses = sfm_root / "sparse/0/images.txt"
    with poses.open("a") as handle:
        handle.write(f"99 1 0 0 0 0 0 0 1 {heldout['image']}\n\n")
    with pytest.raises(ValueError, match="leaked"):
        reference.build_bundle(pilot_root, sfm_root, tmp_path / "leaked", 64)


@pytest.mark.parametrize("row", ["1 PINHOLE 128 128 35 35 64 64", "1 OPENCV_FISHEYE 128 128 35 35 64 64 -10 0 0 0",
                                 "1 OPENCV_FISHEYE 128 128 nan 35 64 64 0 0 0 0"])
def test_camera_model_mismatch_fold_and_nonfinite_rejected(tmp_path, row):
    path = tmp_path / "cameras.txt"
    path.write_text(row + "\n2 OPENCV_FISHEYE 128 128 35 35 64 64 0 0 0 0\n")
    with pytest.raises(ValueError):
        reference.read_cameras(path)
