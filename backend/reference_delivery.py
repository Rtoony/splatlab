"""Serialize existing reconstructed points and cameras for independent DCC review."""

from __future__ import annotations

import json
import math
import re
import shutil
import struct
from pathlib import Path, PurePosixPath

import numpy as np

import artifact_manifest as manifests


SCHEMA = "dev.splatlab.reference-delivery/v1"
VIEWER = Path(__file__).resolve().parents[1] / "tools/reference-viewer"
LIMITATIONS = [
    "Existing sparse triangulation only: no new dense mesh or trained Gaussian splat.",
    "Three virtual directions share each source optical center; they are not additional captures.",
    "Only training cameras have poses. Held-out localization and appearance evaluation are still pending.",
    "Projected dots do not prove visibility through occluding surfaces or identify architectural features.",
    "Coordinates and camera spacing are arbitrary-scale estimates, not measured meters or verified world up.",
    "No alignment to, correction of, or owner acceptance of the condo model is included.",
]


def artifact_path(root: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("Expected a normalized relative artifact path")
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or "\\" in name or str(relative) != name):
        raise ValueError("Expected a normalized relative artifact path")
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact is missing or escapes its reference root")
    return path


def validate_camera(frame: dict) -> np.ndarray:
    matrix = np.asarray(frame["transform_matrix"], dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(np.linalg.det(matrix[:3, :3]), 1, abs_tol=1e-8)):
        raise ValueError("Expected a rigid right-handed camera-to-world matrix")
    return matrix


def verify_reference(root: Path) -> tuple[dict, dict, dict]:
    receipt_path = artifact_path(root, "receipt.json")
    receipt_hash = manifests.sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != "dev.splatlab.fisheye-reference/v1"
            or receipt.get("status") != "prepared-needs-heldout-localization-and-mask-review"
            or receipt.get("registration") is not None or receipt.get("metric_scale") != "unknown"):
        raise ValueError("Expected a completed independent fisheye reference bundle")
    hashes = receipt["artifacts"]
    if not isinstance(hashes, dict) or not 5 <= len(hashes) <= 1500:
        raise ValueError("Invalid or unbounded reference artifact list")
    total_bytes = 0
    for name, digest in hashes.items():
        path = artifact_path(root, name)
        total_bytes += path.stat().st_size
        if total_bytes > 512 * 1024 * 1024 or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("Invalid hash or reference exceeds 512 MiB")
        if manifests.sha256_file(path) != digest:
            raise ValueError("Source reference artifact hash changed")
    if not {"camera-set.json", "sparse.ply"} <= hashes.keys():
        raise ValueError("Missing source-bound cameras or sparse cloud")
    cameras = json.loads((root / "camera-set.json").read_text())
    if (cameras.get("schema") != receipt["schema"] or cameras.get("units") != "arbitrary"
            or cameras.get("registration") is not None or cameras.get("camera_model") != "OPENCV"):
        raise ValueError("Unexpected camera coordinate or projection contract")
    for key in ("k1", "k2", "k3", "k4", "p1", "p2"):
        if cameras.get(key) != 0:
            raise ValueError("Only already-rectified pinhole references can be delivered")
    intrinsics = np.array([cameras[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")], dtype=float)
    if (not np.isfinite(intrinsics).all() or np.any(intrinsics[:4] <= 0)
            or not math.isclose(cameras["cx"], cameras["w"] / 2)
            or not math.isclose(cameras["cy"], cameras["h"] / 2)
            or not math.isclose(cameras["fl_x"], cameras["fl_y"])):
        raise ValueError("GLTF delivery currently requires a centered symmetric pinhole camera")
    frames = cameras["frames"]
    if not 1 <= len(frames) <= 576 or len({frame["file_path"] for frame in frames}) != len(frames):
        raise ValueError("Invalid or unbounded camera list")
    for frame in frames:
        for path_key, hash_key in (("file_path", "sha256"), ("mask_path", "mask_sha256")):
            if hashes.get(frame[path_key]) != frame[hash_key]:
                raise ValueError("Frame lineage disagrees with sealed reference artifacts")
        if frame["split"] == "train":
            if frame.get("pose_basis") != "frozen-training-sfm":
                raise ValueError("Unknown training camera provenance")
            validate_camera(frame)
        elif frame["split"] not in {"val", "test"} or frame.get("transform_matrix") is not None:
            raise ValueError("Unreviewed held-out pose cannot enter a training reference package")
    hashes = {**hashes, "receipt.json": receipt_hash}
    if manifests.sha256_file(receipt_path) != receipt_hash:
        raise ValueError("Reference receipt changed while reading")
    return cameras, receipt, hashes


def read_sparse_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header = manifests.read_ply_header(path)
    expected = (("x", "float"), ("y", "float"), ("z", "float"),
                ("red", "uchar"), ("green", "uchar"), ("blue", "uchar"))
    if (header.vertex_properties != expected or header.encoding != "binary_little_endian"
            or not 1 <= header.vertex_count <= 250000):
        raise ValueError("Select a bounded COLMAP binary XYZ/RGB sparse PLY")
    dtype = np.dtype([(name, "<f4" if scalar == "float" else "u1") for name, scalar in expected])
    if path.stat().st_size != header.header_bytes + header.vertex_count * dtype.itemsize:
        raise ValueError("Sparse PLY byte count disagrees with vertex count")
    values = np.fromfile(path, dtype=dtype, count=header.vertex_count, offset=header.header_bytes)
    positions = np.column_stack([values[name] for name in ("x", "y", "z")])
    colors = np.column_stack([values[name] for name in ("red", "green", "blue")])
    if not np.isfinite(positions).all():
        raise ValueError("Sparse cloud contains non-finite positions")
    return positions, colors


def linear_colors(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors, dtype=float) / 255
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def make_glb(positions: np.ndarray, colors: np.ndarray, cameras: dict, source_hash: str) -> bytes:
    positions = np.asarray(positions, dtype="<f4")
    colors = np.asarray(colors)
    if (positions.ndim != 2 or positions.shape[1] != 3 or not 1 <= len(positions) <= 250000
            or colors.shape != positions.shape or not np.isfinite(positions).all()
            or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 255)):
        raise ValueError("Expected bounded finite XYZ and RGB arrays")
    position_bytes = positions.tobytes()
    rgba = np.column_stack((linear_colors(colors), np.ones(len(colors)))).astype("<f4")
    color_bytes = rgba.tobytes()
    body = position_bytes + color_bytes
    frames = [frame for frame in cameras["frames"] if frame["split"] == "train"]
    metadata = {"schema": SCHEMA, "sourceReceiptSha256": source_hash, "units": "arbitrary",
                "registration": None, "worldUpVerified": False, "denseSurface": False,
                "sourceToGLTF": np.eye(4).ravel().tolist(), "limitations": LIMITATIONS}
    nodes = [{"name": "SplatLab captured reference — unregistered", "children": [1, 2], "extras": metadata},
             {"name": "Reconstructed sparse points — NOT a surface", "mesh": 0},
             {"name": "Estimated source cameras", "children": list(range(3, 3 + len(frames)))}]
    for index, frame in enumerate(frames):
        nodes.append({"name": frame["file_path"], "camera": index,
                      "matrix": validate_camera(frame).T.ravel().tolist(),
                      "extras": {key: frame[key] for key in ("source_image", "source_group", "source_sha256",
                                                             "physical_lens", "virtual_yaw_deg", "split")}})
    perspective = {"yfov": 2 * math.atan(cameras["h"] / (2 * cameras["fl_y"])),
                   "aspectRatio": cameras["w"] / cameras["h"], "znear": .001}
    document = {"asset": {"version": "2.0", "generator": "SplatLab existing-reference serializer", "extras": metadata},
                "scene": 0, "scenes": [{"name": "Independent reconstruction reference", "nodes": [0]}],
                "nodes": nodes, "cameras": [{"type": "perspective", "perspective": perspective,
                                               "name": frame["file_path"]} for frame in frames],
                "meshes": [{"primitives": [{"mode": 0, "attributes": {"POSITION": 0, "COLOR_0": 1}}]}],
                "buffers": [{"byteLength": len(body)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
                                {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(color_bytes), "target": 34962}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
                               "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
                              {"bufferView": 1, "componentType": 5126, "count": len(colors), "type": "VEC4"}]}
    encoded = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    body += b"\x00" * (-len(body) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(body))
            + struct.pack("<I4s", len(encoded), b"JSON") + encoded
            + struct.pack("<I4s", len(body), b"BIN\x00") + body)


def build_delivery(reference_root: Path, output: Path) -> dict:
    reference_root, output = reference_root.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(reference_root):
        raise ValueError("Choose a new delivery directory outside the frozen reference")
    cameras, reference_receipt, snapshot = verify_reference(reference_root)
    positions, colors = read_sparse_points(reference_root / "sparse.ply")
    posed = [frame for frame in cameras["frames"] if frame["split"] == "train"]
    if not posed:
        raise ValueError("No estimated source cameras are available")
    templates = {name: (VIEWER / name).read_text() for name in ("index.html", "viewer.js", "math.js", "style.css")}
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": SCHEMA, "status": "building-not-ready", "started_at": manifests.utc_now(),
               "source_reference_receipt_sha256": snapshot["receipt.json"], "source_hashes": snapshot,
               "serializer_sha256": manifests.sha256_file(Path(__file__)),
               "viewer_source_hashes": {name: manifests.sha256_file(VIEWER / name) for name in templates},
               "source_points": len(positions), "posed_virtual_views": len(posed),
               "physical_source_images": len({frame["source_image"] for frame in posed}),
               "unlocalized_views_omitted": len(cameras["frames"]) - len(posed),
               "blender_import_verified": False, "new_reconstruction": False, "limitations": LIMITATIONS}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        (output / "reference.glb").write_bytes(make_glb(positions, colors, cameras, snapshot["receipt.json"]))
        for frame in posed:
            for name in (frame["file_path"], frame["mask_path"]):
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(artifact_path(reference_root, name), destination)
        shutil.copyfile(reference_root / "sparse.ply", output / "sparse.ply")
        delivered_cameras = {**cameras, "frames": posed,
                             "source_camera_set_sha256": snapshot["camera-set.json"],
                             "unlocalized_views_omitted": receipt["unlocalized_views_omitted"]}
        manifests.atomic_write_json(output / "camera-set.json", delivered_cameras)
        data = {"schema": SCHEMA, "points": positions.tolist(), "colors": colors.tolist(),
                "cameras": delivered_cameras, "limitations": LIMITATIONS,
                "source_receipt_sha256": snapshot["receipt.json"],
                "physical_source_images": receipt["physical_source_images"],
                "unlocalized_views_omitted": receipt["unlocalized_views_omitted"]}
        manifests.atomic_write_json(output / "viewer-data.json", data)
        embedded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
        page = templates["index.html"].replace("__STYLE__", templates["style.css"])
        page = page.replace("__MATH__", templates["math.js"]).replace("__VIEWER__", templates["viewer.js"])
        page = page.replace("__DATA__", embedded)
        (output / "index.html").write_text(page)
        asset = {"id": "splatlab-sparse-camera-reference", "kind": "point-cloud", "file": "reference.glb",
                 "sha256": manifests.sha256_file(output / "reference.glb"),
                 "producer": {"tool": "SplatLab reference delivery", "runId": output.name},
                 "sourceHashes": [snapshot["receipt.json"], snapshot["sparse.ply"], snapshot["camera-set.json"]],
                 "units": "arbitrary", "coordinates": "Unchanged COLMAP world numbers; OpenGL cameras; GLTF column-major matrices. World up and metric scale unverified.",
                 "quality": {"sourcePoints": len(positions), "sourceCameras": len(posed),
                             "newReconstruction": False, "limitations": LIMITATIONS}, "registration": None}
        manifests.atomic_write_json(output / "external-references.json", {"schema": "chanate-external-references/v1", "assets": [asset]})
        if snapshot != {name: manifests.sha256_file(artifact_path(reference_root, name)) for name in snapshot}:
            raise ValueError("Frozen source bundle changed during serialization")
        receipt.update(status="reference-package-ready-not-reconstruction-complete", files={
            str(path.relative_to(output)): manifests.sha256_file(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "receipt.json"})
    except Exception as error:
        receipt.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        receipt["finished_at"] = manifests.utc_now()
        manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt
