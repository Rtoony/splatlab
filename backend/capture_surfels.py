"""Frozen camera preparation for a separate, unit-normalized 2DGS experiment."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

import artifact_manifest as manifests
from reference_delivery import artifact_path


def frozen_dataset(directory):
    directory = Path(directory).resolve()
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "dev.splatlab.fisheye-training-input/v1"
            or receipt.get("status") != "prepared-needs-visual-mask-review"):
        raise ValueError("Require a prepared source-bound fisheye dataset")
    snapshot = {artifact_path(directory, name): digest for name, digest in receipt["files"].items()}
    snapshot[receipt_path] = manifests.sha256_file(receipt_path)
    if any(directory / name not in snapshot for name in ("transforms.json", "sparse.ply")):
        raise ValueError("Camera transforms and sparse initialization must both be sealed")
    verify_snapshot(snapshot)
    document = json.loads((directory / "transforms.json").read_text())
    intrinsic = np.asarray([document[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")], dtype=float)
    if (not np.isfinite(intrinsic).all() or np.any(intrinsic[:4] <= 0)
            or any(document[key] != int(document[key]) or document[key] > 2048 for key in ("w", "h"))):
        raise ValueError("Require finite bounded pinhole intrinsics")
    if any(document.get(key, 0) != 0 for key in ("k1", "k2", "k3", "k4", "p1", "p2")):
        raise ValueError("Surface splats require rectified, not distorted photos")
    splits = {name: document[name + "_filenames"] for name in ("train", "val", "test")}
    names = [name for records in splits.values() for name in records]
    frames = {frame["file_path"]: frame for frame in document["frames"]}
    if (not all(splits.values()) or len(names) != len(set(names)) or set(names) != set(frames)
            or len(frames) != len(document["frames"])):
        raise ValueError("Require disjoint, nonempty and complete frozen splits")
    for frame in frames.values():
        if any(key in frame and frame[key] != document[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")):
            raise ValueError("Per-frame intrinsics differ from the fixed-camera renderer")
        if any(frame.get(key, 0) != 0 for key in ("k1", "k2", "k3", "k4", "p1", "p2")):
            raise ValueError("Per-frame distortion must be rectified before rendering")
        pose = np.asarray(frame["transform_matrix"], dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)):
            raise ValueError("Source camera is not a finite rigid transform")
        for field in ("file_path", "mask_path"):
            if artifact_path(directory, frame[field]) not in snapshot:
                raise ValueError("Camera image or mask is not sealed")
    poses = np.asarray([frames[name]["transform_matrix"] for name in splits["train"]])
    center = poses[:, :3, 3].mean(axis=0)
    scale = float(np.linalg.norm(poses[:, :3, 3] - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("Training cameras have no spatial baseline")
    return document, frames, splits, snapshot, center, scale


def verify_snapshot(snapshot):
    if any(manifests.sha256_file(path) != digest for path, digest in snapshot.items()):
        raise ValueError("Frozen input or implementation changed")


def normalized_camera(frame, center, scale):
    pose = np.asarray(frame["transform_matrix"], dtype=np.float64).copy()
    pose[:3, 3] = (pose[:3, 3] - center) / scale
    return np.linalg.inv(pose @ np.diag([1., -1., -1., 1.]))


def load_photo(directory, frame, shape):
    with Image.open(artifact_path(directory, frame["file_path"])) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(artifact_path(directory, frame["mask_path"])) as source:
        mask = np.asarray(source.convert("L")) > 127
    if rgb.shape != (*shape, 3) or mask.shape != shape or not mask.any():
        raise ValueError("Frozen photograph and mask have inconsistent dimensions")
    return rgb, mask


def preview_parameters(saved, center, scale):
    if not np.isfinite(scale) or scale <= 0 or np.asarray(center).shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Invalid solver normalization")
    count = len(saved["means"])
    shapes = {"means": (count, 3), "scales": (count, 3), "quats": (count, 4), "opacities": (count,),
              "sh0": (count, 1, 3), "shN": (count, 3, 3)}
    if count == 0 or any(saved[name].shape != shape or saved[name].dtype != np.float32 or not np.isfinite(saved[name]).all()
                         for name, shape in shapes.items()):
        raise ValueError("Require finite native float32 SH1 surface parameters")
    restored = (saved["means"].astype(np.float64) * scale + center).astype(np.float32)
    flat_scales = (saved["scales"].astype(np.float64) + np.log(scale)).astype(np.float32)
    flat_scales[:, 2] = np.log(scale * 1e-6)
    return {"means": restored, "scales": flat_scales, "quats": saved["quats"], "opacities": saved["opacities"][:, None],
            "features_dc": saved["sh0"][:, 0], "features_rest": saved["shN"]}


def select_depth_anchors(points, identifiers, observed, supported, mask, frame, intrinsics):
    from capture_structure import project_points

    if (points.shape != (len(identifiers), 3) or observed.shape != identifiers.shape or supported.shape != identifiers.shape
            or observed.dtype != bool or supported.dtype != bool or mask.dtype != bool
            or mask.shape != (intrinsics["h"], intrinsics["w"])):
        raise ValueError("Invalid source-tracked anchor dimensions")
    pixels, valid = project_points(points, frame, intrinsics)
    selected = np.flatnonzero(valid & observed & supported & (identifiers % 5 != 0))
    indices = np.floor(pixels[selected]).astype(int)
    selected = selected[mask[indices[:, 1], indices[:, 0]]]
    pose = np.asarray(frame["transform_matrix"], dtype=float)
    depth = -((points[selected] - pose[:3, 3]) @ pose[:3, :3])[:, 2]
    return {"point_ids": identifiers[selected].copy(), "pixels": pixels[selected].copy(), "depth": depth}


def source_depth_anchors(structure, dataset, document, frames):
    structure, dataset = Path(structure).resolve(), Path(dataset).resolve()
    receipt = json.loads((structure / "receipt.json").read_text())
    analysis = json.loads((structure / "analysis.json").read_text())
    input_receipt = json.loads((dataset / "receipt.json").read_text())
    references = [digest for name, digest in receipt["source_hashes"].items() if Path(name).name == "receipt.json"]
    if (receipt.get("schema") != "dev.splatlab.capture-structure/v1" or analysis.get("status") != "analyzed-needs-visual-review"
            or references != [input_receipt["source_reference_sha256"]]
            or analysis["source_receipt_sha256"] != manifests.sha256_file(structure / "receipt.json")):
        raise ValueError("Depth anchors are not bound to this frozen capture")
    snapshot = {structure / name: manifests.sha256_file(structure / name) for name in ("receipt.json", "analysis.json")}
    for name in ["membership.npz", *(f"classified-masks/cam_{view['ordinal']:03d}.npz" for view in receipt["views"])]:
        snapshot[artifact_path(structure, name)] = analysis["files"][name]
    verify_snapshot(snapshot)
    anchors = {}
    with np.load(structure / "membership.npz", allow_pickle=False) as membership:
        points, identifiers = membership["points"], membership["point_ids"]
        supported = membership["facade_supported"] | membership["pavement_supported"]
        for view in receipt["views"]:
            name = view["file_path"]
            if (name not in document["train_filenames"] or name in anchors or view["split"] != "train" or view["virtual_yaw_deg"] != 0
                    or not np.allclose(view["transform_matrix"], frames[name]["transform_matrix"], atol=1e-7, rtol=0)):
                raise ValueError("Anchor selection must retain unique central training cameras")
            with np.load(structure / f"classified-masks/cam_{view['ordinal']:03d}.npz", allow_pickle=False) as masks:
                anchors[name] = select_depth_anchors(points, identifiers, membership["observed"][view["ordinal"]], supported,
                                                     masks["static_surface"], frames[name], document)
    if not anchors or sum(len(record["depth"]) for record in anchors.values()) < 50:
        raise ValueError("Insufficient source-tracked depth anchors")
    metadata = {"observations": sum(len(record["depth"]) for record in anchors.values()),
                "point_ids": sorted({int(identity) for record in anchors.values() for identity in record["point_ids"]}),
                "withheld_from_depth_loss": "Every original SfM point ID divisible by five; still part of original SfM initialization, not survey ground truth",
                "scope": "Only tracked, temporally supported static training features; no validation/test photographs"}
    return anchors, metadata, snapshot
