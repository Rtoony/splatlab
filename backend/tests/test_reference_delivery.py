import json
import math
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

import artifact_manifest as manifests
import reference_delivery as delivery


def sparse_ply(path, positions, colors):
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex " + str(len(positions))
              + "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    path.write_bytes(header.encode() + b"".join(struct.pack("<fffBBB", *position, *color) for position, color in zip(positions, colors)))


def read_glb(data):
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))
    json_length, json_kind = struct.unpack_from("<I4s", data, 12)
    assert json_kind == b"JSON" and json_length % 4 == 0
    document = json.loads(data[20:20 + json_length])
    offset = 20 + json_length
    binary_length, binary_kind = struct.unpack_from("<I4s", data, offset)
    assert binary_kind == b"BIN\x00" and binary_length % 4 == 0
    assert offset + 8 + binary_length == length
    return document, data[offset + 8:]


@pytest.fixture
def reference(tmp_path):
    root = tmp_path / "reference"
    root.mkdir()
    positions = np.array([[0, 0, -4], [1, 0, -4], [0, 1, -4], [0, 0, -5]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [128, 128, 128], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    sparse_ply(root / "sparse.ply", positions, colors)
    frames = []
    for index, split in enumerate(("train", "train", "val", "test")):
        image_path, mask_path = f"{split}/frame-{index}.png", f"masks/frame-{index}.png"
        for name in (image_path, mask_path):
            (root / name).parent.mkdir(exist_ok=True)
        Image.new("RGB", (64, 64), (30 * index, 100, 200)).save(root / image_path)
        Image.new("L", (64, 64), 255).save(root / mask_path)
        matrix = np.eye(4)
        matrix[0, 3] = index * .03
        frames.append({"file_path": image_path, "mask_path": mask_path, "sha256": manifests.sha256_file(root / image_path),
                       "mask_sha256": manifests.sha256_file(root / mask_path), "source_image": f"lens-{index % 2}/frame-{index:06d}.jpg",
                       "source_sha256": "b" * 64, "source_group": f"frame-{index:06d}", "physical_lens": f"lens-{index % 2}",
                       "virtual_yaw_deg": 0, "split": split, "pts": index, "time_base": "1/30",
                       "transform_matrix": matrix.tolist() if split == "train" else None,
                       "pose_basis": "frozen-training-sfm" if split == "train" else "unlocalized-heldout"})
    cameras = {"schema": "dev.splatlab.fisheye-reference/v1", "units": "arbitrary", "registration": None,
               "camera_model": "OPENCV", "w": 64, "h": 64, "fl_x": 50, "fl_y": 50, "cx": 32, "cy": 32,
               "k1": 0, "k2": 0, "k3": 0, "k4": 0, "p1": 0, "p2": 0, "frames": frames}
    manifests.atomic_write_json(root / "camera-set.json", cameras)
    receipt = {"schema": cameras["schema"], "status": "prepared-needs-heldout-localization-and-mask-review",
               "registration": None, "metric_scale": "unknown", "artifacts": {
                   str(path.relative_to(root)): manifests.sha256_file(path) for path in root.rglob("*") if path.is_file()}}
    manifests.atomic_write_json(root / "receipt.json", receipt)
    return root, positions, colors, cameras


def test_glb_preserves_xyz_camera_matrix_and_linear_vertex_color(reference):
    _, positions, colors, cameras = reference
    angle = .3
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    camera_matrix = np.eye(4)
    camera_matrix[:3, :3] = rotation
    camera_matrix[:3, 3] = [1, 2, 3]
    cameras["frames"][0]["transform_matrix"] = camera_matrix.tolist()
    document, binary = read_glb(delivery.make_glb(positions, colors, cameras, "a" * 64))
    assert document["meshes"][0]["primitives"][0]["mode"] == 0
    np.testing.assert_array_equal(np.frombuffer(binary, dtype="<f4", count=positions.size).reshape(-1, 3), positions)
    np.testing.assert_array_equal(np.array(document["nodes"][3]["matrix"]).reshape(4, 4).T, camera_matrix)
    assert len(document["cameras"]) == 2
    assert document["cameras"][0]["perspective"]["yfov"] == pytest.approx(2 * math.atan(.64))
    offset = document["bufferViews"][1]["byteOffset"]
    rgba = np.frombuffer(binary, dtype="<f4", offset=offset).reshape(-1, 4)
    np.testing.assert_allclose(rgba[1], [.2158605, .2158605, .2158605, 1], atol=1e-7)
    assert document["asset"]["extras"]["registration"] is None
    assert all(view["byteOffset"] % 4 == 0 for view in document["bufferViews"])


def test_delivery_is_source_bound_complete_and_preserves_unlocalized_exclusion(reference, tmp_path):
    root, positions, colors, _ = reference
    original = {str(path): manifests.sha256_file(path) for path in root.rglob("*") if path.is_file()}
    output = tmp_path / "delivery"
    receipt = delivery.build_delivery(root, output)
    assert receipt["status"] == "reference-package-ready-not-reconstruction-complete"
    assert receipt["source_points"] == 4 and receipt["posed_virtual_views"] == 2
    assert receipt["unlocalized_views_omitted"] == 2
    assert receipt["new_reconstruction"] is False and receipt["blender_import_verified"] is False
    viewer = json.loads((output / "viewer-data.json").read_text())
    np.testing.assert_array_equal(viewer["points"], positions)
    np.testing.assert_array_equal(viewer["colors"], colors)
    assert all(frame["split"] == "train" for frame in viewer["cameras"]["frames"])
    assert all((output / frame["file_path"]).is_file() for frame in viewer["cameras"]["frames"])
    assert (output / "camera-set.json").read_text() == json.dumps(viewer["cameras"], indent=2, sort_keys=True) + "\n"
    assert all(manifests.sha256_file(output / name) == digest for name, digest in receipt["files"].items())
    assert original == {name: manifests.sha256_file(Path(name)) for name in original}
    assert "__DATA__" not in (output / "index.html").read_text()
    with pytest.raises(ValueError, match="new delivery"):
        delivery.build_delivery(root, output)


@pytest.mark.parametrize("name", ["/etc/passwd", "../secret", "a/../receipt.json", "./receipt.json", "a\\b", None])
def test_artifact_paths_cannot_escape_reference(reference, name):
    with pytest.raises(ValueError, match="relative"):
        delivery.artifact_path(reference[0], name)


def test_artifact_symlink_and_modified_photo_are_rejected(reference, tmp_path):
    root, *_ = reference
    (root / "linked").symlink_to(root / "receipt.json")
    with pytest.raises(ValueError, match="escapes"):
        delivery.artifact_path(root, "linked")
    (root / "train/frame-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        delivery.build_delivery(root, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("matrix", [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.diag([2, 2, 2, 1]), np.full((4, 4), float("nan"))])
def test_invalid_or_reflected_camera_transforms_are_refused(matrix):
    with pytest.raises(ValueError, match="rigid"):
        delivery.validate_camera({"transform_matrix": matrix})


def test_sparse_cloud_rejects_truncation_and_nonfinite(reference):
    root, positions, colors, _ = reference
    path = root / "sparse.ply"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="byte count"):
        delivery.read_sparse_points(path)
    positions[0, 0] = np.nan
    sparse_ply(path, positions, colors)
    with pytest.raises(ValueError, match="non-finite"):
        delivery.read_sparse_points(path)
