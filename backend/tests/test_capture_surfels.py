import json

import numpy as np
import pytest
from PIL import Image

import capture_surfels as capture


def test_solver_similarity_preserves_camera_projection_and_depth():
    pose = np.eye(4)
    pose[:3, 3] = [3, 4, 5]
    frame = {"transform_matrix": pose.tolist()}
    center, scale = np.array([-2, 7, 1]), 4.7
    original = np.array([[2., 5., -10.], [7., 2., -5.]])
    normalized = (original - center) / scale
    converted = (capture.normalized_camera(frame, center, scale) @ np.column_stack((normalized, np.ones(2))).T).T[:, :3]
    native = (np.linalg.inv(pose @ np.diag([1, -1, -1, 1])) @ np.column_stack((original, np.ones(2))).T).T[:, :3]
    np.testing.assert_allclose(converted / converted[:, 2:3], native / native[:, 2:3], atol=1e-12)
    np.testing.assert_allclose(converted[:, 2] * scale, native[:, 2], atol=1e-12)
    np.testing.assert_allclose(normalized * scale + center, original, atol=1e-12)


def test_changed_frozen_input_is_rejected(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("original")
    snapshot = {path: capture.manifests.sha256_file(path)}
    capture.verify_snapshot(snapshot)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Frozen"):
        capture.verify_snapshot(snapshot)


@pytest.fixture
def sealed_input(tmp_path):
    document = {"w": 8, "h": 8, "fl_x": 5., "fl_y": 5., "cx": 4., "cy": 4., "frames": [],
                "train_filenames": ["train-0.png", "train-1.png"], "val_filenames": ["val.png"], "test_filenames": ["test.png"]}
    (tmp_path / "sparse.ply").write_bytes(b"sealed point fixture; parsing is tested independently")
    for ordinal, name in enumerate(document["train_filenames"] + document["val_filenames"] + document["test_filenames"]):
        pose = np.eye(4)
        pose[0, 3] = ordinal
        mask_name = "mask-" + name
        Image.new("RGB", (8, 8), (120, 60, 30)).save(tmp_path / name)
        Image.new("L", (8, 8), 255).save(tmp_path / mask_name)
        document["frames"].append({"file_path": name, "mask_path": mask_name, "transform_matrix": pose.tolist()})

    def seal():
        (tmp_path / "transforms.json").write_text(json.dumps(document))
        receipt = {"schema": "dev.splatlab.fisheye-training-input/v1", "status": "prepared-needs-visual-mask-review",
                   "files": {path.name: capture.manifests.sha256_file(path) for path in tmp_path.iterdir() if path.name != "receipt.json"}}
        (tmp_path / "receipt.json").write_text(json.dumps(receipt))

    seal()
    return tmp_path, document, seal


def test_normalization_uses_only_training_centers_and_keeps_photos(sealed_input):
    directory, document, seal = sealed_input
    document["frames"][-1]["transform_matrix"][0][3] = 100000
    seal()
    loaded, frames, splits, _snapshot, center, scale = capture.frozen_dataset(directory)
    np.testing.assert_array_equal(center, [.5, 0, 0])
    assert scale == .5 and len(splits["train"]) == 2 and loaded == document
    pixels, mask = capture.load_photo(directory, frames["train-0.png"], (8, 8))
    assert mask.all() and pixels[0, 0].tolist() == [120, 60, 30]


@pytest.mark.parametrize("problem", ["overlap", "duplicated", "reflected", "empty", "distortion", "intrinsics", "unsealed"])
def test_malformed_or_leaking_camera_inputs_are_rejected(sealed_input, problem):
    directory, document, seal = sealed_input
    if problem == "overlap":
        document["val_filenames"] = document["train_filenames"][:1]
    elif problem == "duplicated":
        document["frames"].append(document["frames"][0])
    elif problem == "reflected":
        document["frames"][0]["transform_matrix"][0][0] = -1
    elif problem == "empty":
        document["test_filenames"] = []
    elif problem == "distortion":
        document["k1"] = .1
    elif problem == "intrinsics":
        document["fl_x"] = 0
    else:
        document["frames"][0]["mask_path"] = "unsealed.png"
        Image.new("L", (8, 8), 255).save(directory / "unsealed.png")
    seal()
    if problem == "unsealed":
        receipt = json.loads((directory / "receipt.json").read_text())
        del receipt["files"]["unsealed.png"]
        (directory / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        capture.frozen_dataset(directory)


@pytest.mark.parametrize("name", ["transforms.json", "sparse.ply"])
def test_required_initialization_artifacts_cannot_escape_the_seal(sealed_input, name):
    directory, _document, _seal = sealed_input
    path = directory / "receipt.json"
    receipt = json.loads(path.read_text())
    del receipt["files"][name]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="both be sealed"):
        capture.frozen_dataset(directory)


@pytest.mark.parametrize("key,value", [("fl_x", 6.), ("cx", 4.5), ("k1", .1)])
def test_frame_specific_intrinsics_cannot_silently_use_global_projection(sealed_input, key, value):
    directory, document, seal = sealed_input
    document["frames"][0][key] = value
    seal()
    with pytest.raises(ValueError, match="Per-frame"):
        capture.frozen_dataset(directory)


def test_surface_preview_keeps_tangent_scale_orientation_and_native_parameters(tmp_path):
    from reconstruction_review import export_splats

    saved = {"means": np.array([[1., 2., 3.]], dtype=np.float32), "scales": np.log(np.array([[.2, .4, .8]], dtype=np.float32)),
             "quats": np.array([[.70710677, .70710677, 0, 0]], dtype=np.float32), "opacities": np.array([.4], dtype=np.float32),
             "sh0": np.zeros((1, 1, 3), dtype=np.float32), "shN": np.zeros((1, 3, 3), dtype=np.float32)}
    original = {name: value.copy() for name, value in saved.items()}
    center, scale = np.array([8., -3., 10.]), 6.13240211618254
    converted = capture.preview_parameters(saved, center, scale)
    assert all(value.dtype == np.float32 for value in converted.values())
    np.testing.assert_allclose(converted["means"], saved["means"] * scale + center, rtol=1e-6)
    np.testing.assert_allclose(np.exp(converted["scales"][:, :2]), [[.2 * scale, .4 * scale]], rtol=1e-6)
    np.testing.assert_allclose(np.exp(converted["scales"][:, 2]), [scale * 1e-6], rtol=1e-6)
    np.testing.assert_array_equal(converted["quats"], original["quats"])
    for name, value in saved.items():
        np.testing.assert_array_equal(value, original[name])
    receipt = export_splats(tmp_path / "thin-preview.ply", converted)
    assert receipt["rows"] == 1
    saved["means"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite native"):
        capture.preview_parameters(saved, center, scale)


def test_depth_anchors_require_real_tracks_and_reserve_check_point_ids():
    points = np.array([[0., 0., -3.], [0., 0., -4.], [0., 0., -5.], [0., 0., -6.], [0., 0., -7.]])
    identifiers = np.array([1, 5, 7, 9, 11])
    observed = np.array([True, True, False, True, True])
    supported = np.array([True, True, True, False, True])
    frame = {"transform_matrix": np.eye(4).tolist()}
    intrinsic = {"w": 8, "h": 8, "fl_x": 5., "fl_y": 5., "cx": 4., "cy": 4.}
    mask = np.ones((8, 8), dtype=bool)
    anchors = capture.select_depth_anchors(points, identifiers, observed, supported, mask, frame, intrinsic)
    assert anchors["point_ids"].tolist() == [1, 11]
    np.testing.assert_array_equal(anchors["depth"], [3, 7])
    np.testing.assert_array_equal(anchors["pixels"], [[4, 4], [4, 4]])
    mask[4, 4] = False
    assert not len(capture.select_depth_anchors(points, identifiers, observed, supported, mask, frame, intrinsic)["depth"])
