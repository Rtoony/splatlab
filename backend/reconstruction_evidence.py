"""CPU-only COLMAP observation evidence in the calibrated studio frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

import artifact_manifest as manifests
from artifact_dependencies import scale_revision
from scene_revisions import key_checked

MAX_FILE_BYTES = 512 * 1024 ** 2
MAX_POINTS = 500_000
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 4: ("OPENCV", 8)}
Y_UP = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
GL_TO_CV = np.diag([1., -1., -1., 1.])


class EvidenceError(ValueError):
    pass


def unpack(handle, layout):
    size = struct.calcsize("<" + layout)
    data = handle.read(size)
    if len(data) != size:
        raise EvidenceError("Truncated COLMAP record")
    return struct.unpack("<" + layout, data)


def count(handle, maximum):
    value = unpack(handle, "Q")[0]
    if value > maximum:
        raise EvidenceError("COLMAP input exceeds the bounded observation budget")
    return value


def contained_file(job: Path, relative: str) -> Path:
    path = (job / key_checked(relative)).resolve()
    if not path.is_relative_to(job.resolve()) or not path.is_file():
        raise EvidenceError("Missing evidence file or a link escaping the registered job")
    return path


def input_path(job: Path, relative: str, sources: dict) -> Path:
    path = contained_file(job, relative)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise EvidenceError("An evidence input exceeds 512 MiB")
    sources[relative] = manifests.file_identity(path)
    return path


def rotation(quaternion):
    scalar, axis_x, axis_y, axis_z = np.asarray(quaternion, dtype=np.float64)
    if not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-5:
        raise EvidenceError("Invalid COLMAP rotation quaternion")
    return np.array([
        [1 - 2 * (axis_y ** 2 + axis_z ** 2), 2 * (axis_x * axis_y - scalar * axis_z), 2 * (axis_x * axis_z + scalar * axis_y)],
        [2 * (axis_x * axis_y + scalar * axis_z), 1 - 2 * (axis_x ** 2 + axis_z ** 2), 2 * (axis_y * axis_z - scalar * axis_x)],
        [2 * (axis_x * axis_z - scalar * axis_y), 2 * (axis_y * axis_z + scalar * axis_x), 1 - 2 * (axis_x ** 2 + axis_y ** 2)],
    ])


def matrix4(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise EvidenceError("Invalid reconstruction transform")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(matrix[:3, :3]) < 0:
        raise EvidenceError("Reconstruction transform must preserve a rigid right-handed frame")
    return matrix


def read_cameras(path):
    cameras = {}
    with path.open("rb") as handle:
        for _ in range(count(handle, 10000)):
            camera_id, model_id, width, height = unpack(handle, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise EvidenceError(f"Camera model {model_id} is unsupported; recovery needs pinhole/OpenCV evidence")
            model, parameters = CAMERA_MODELS[model_id]
            values = unpack(handle, "d" * parameters)
            if camera_id in cameras or not (0 < width <= 32768 and 0 < height <= 32768) or width * height > 40_000_000:
                raise EvidenceError("Duplicate camera or invalid image dimensions")
            if not np.isfinite(values).all() or values[0] <= 0:
                raise EvidenceError("Invalid camera intrinsics")
            if model == "SIMPLE_PINHOLE":
                values = (values[0], values[0], values[1], values[2])
            if values[1] <= 0:
                raise EvidenceError("Invalid vertical focal length")
            cameras[camera_id] = {"model": model, "width": width, "height": height, "parameters": values}
        if handle.read(1):
            raise EvidenceError("Unexpected trailing camera data")
    return cameras


def read_images(path):
    images = {}
    with path.open("rb") as handle:
        for _ in range(count(handle, 10000)):
            record = unpack(handle, "idddddddi")
            name = bytearray()
            while len(name) <= 4096:
                byte = handle.read(1)
                if not byte:
                    raise EvidenceError("Truncated COLMAP image name")
                if byte == b"\0":
                    break
                name.extend(byte)
            else:
                raise EvidenceError("COLMAP image name exceeds the path budget")
            size = count(handle, 2_000_000)
            packed = handle.read(size * 24)
            if len(packed) != size * 24:
                raise EvidenceError("Truncated image observations")
            observations = np.frombuffer(packed, dtype=[("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")])
            world_to_camera = np.eye(4)
            world_to_camera[:3, :3] = rotation(record[1:5])
            world_to_camera[:3, 3] = record[5:8]
            if record[0] in images or not np.isfinite(world_to_camera).all():
                raise EvidenceError("Duplicate image or invalid pose")
            images[record[0]] = {"name": name.decode("utf-8"), "camera_id": record[8],
                                  "raw_world_to_camera": world_to_camera, "observations": observations}
        if handle.read(1):
            raise EvidenceError("Unexpected trailing image data")
    return images


def read_points(path):
    points = []
    identifiers = set()
    with path.open("rb") as handle:
        for _ in range(count(handle, MAX_POINTS)):
            record = unpack(handle, "QdddBBBd")
            size = count(handle, 10000)
            packed = handle.read(size * 8)
            if len(packed) != size * 8:
                raise EvidenceError("Truncated point track")
            if record[0] in identifiers or not np.isfinite(record[1:4]).all() or not np.isfinite(record[7]) or record[7] < 0:
                raise EvidenceError("Duplicate point or nonfinite SfM evidence")
            identifiers.add(record[0])
            points.append({"id": record[0], "xyz": record[1:4], "error": record[7],
                           "track": np.frombuffer(packed, dtype="<i4").reshape(-1, 2)})
        if handle.read(1):
            raise EvidenceError("Unexpected trailing point data")
    return points


def project(points, camera):
    transform = camera["world_to_camera"]
    local = np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]
    depth = local[:, 2]
    normalized = local[:, :2] / np.where(np.abs(depth[:, None]) > 1e-12, depth[:, None], 1e-12)
    horizontal, vertical = normalized[:, 0], normalized[:, 1]
    focal_x, focal_y, center_x, center_y, *distortion = camera["parameters"]
    if distortion:
        radial_1, radial_2, tangent_1, tangent_2 = distortion
        radius = horizontal ** 2 + vertical ** 2
        radial = 1 + radial_1 * radius + radial_2 * radius ** 2
        horizontal, vertical = (
            horizontal * radial + 2 * tangent_1 * horizontal * vertical + tangent_2 * (radius + 2 * horizontal ** 2),
            vertical * radial + tangent_1 * (radius + 2 * vertical ** 2) + 2 * tangent_2 * horizontal * vertical,
        )
    pixels = np.stack([horizontal * focal_x + center_x, vertical * focal_y + center_y], axis=1)
    return pixels, depth


@dataclass
class ReconstructionEvidence:
    job: Path
    points: np.ndarray
    records: list
    cameras: dict
    sources: dict
    calibration: dict
    report: dict
    images: dict | None = None


def load(job: Path, retain_image_observations=False) -> ReconstructionEvidence:
    sources = {}
    transforms = manifests.read_json(input_path(job, "processed/transforms.json", sources)) or {}
    candidates = sorted((job / "processed").glob("splatfacto*/**/dataparser_transforms.json"))
    if len(candidates) != 1:
        raise EvidenceError("Recovery requires one unambiguous retained dataparser transform")
    parser = manifests.read_json(input_path(job, str(candidates[0].relative_to(job)), sources)) or {}
    applied = matrix4(transforms.get("applied_transform", np.eye(4)))
    if transforms.get("applied_scale", 1) != 1:
        raise EvidenceError("A separately scaled saved dataset needs an explicit frame adapter")
    calibrated = scale_revision(job)
    metres = calibrated["meters_per_unit"]
    if not isinstance(metres, (float, int)) or not np.isfinite(metres) or metres <= 0:
        raise EvidenceError("Calibrate the capture before metre-based background recovery")
    scale = float(parser.get("scale", 0))
    if not np.isfinite(scale) or scale <= 0:
        raise EvidenceError("Dataparser scale is missing or invalid")
    raw_to_world = matrix4(parser.get("transform"))
    raw_to_world[:3] = Y_UP @ raw_to_world[:3] * (scale * metres)
    sparse_paths = [job / "processed" / "sparse" / "0", job / "processed" / "colmap" / "sparse" / "0"]
    sparse = [path for path in sparse_paths if (path / "points3D.bin").is_file()]
    if len(sparse) != 1:
        raise EvidenceError("Recovery needs one retained COLMAP binary sparse model")
    prefix = str(sparse[0].relative_to(job))
    intrinsics = read_cameras(input_path(job, prefix + "/cameras.bin", sources))
    images = read_images(input_path(job, prefix + "/images.bin", sources))
    records = read_points(input_path(job, prefix + "/points3D.bin", sources))
    cameras = {}
    names = set()
    for frame in transforms.get("frames", []):
        identifier = frame.get("colmap_im_id")
        if identifier not in images or identifier in cameras:
            raise EvidenceError("Saved frame is not uniquely linked to a COLMAP image")
        record = images[identifier]
        if record["camera_id"] not in intrinsics or record["name"] in names:
            raise EvidenceError("COLMAP image camera is missing or a photo path is duplicated")
        names.add(record["name"])
        expected_pose = applied @ np.linalg.inv(record["raw_world_to_camera"]) @ GL_TO_CV
        if not np.allclose(matrix4(frame.get("transform_matrix")), expected_pose, atol=5e-4):
            raise EvidenceError("Saved camera disagrees with COLMAP; refusing guessed frame alignment")
        filename = str(frame.get("file_path", ""))
        if filename.removeprefix("./") != "images/" + record["name"]:
            raise EvidenceError("Saved frame and COLMAP image filenames disagree")
        camera = {**intrinsics[record["camera_id"]], "name": record["name"], "image_key": "processed/" + filename.removeprefix("./"),
                  "world_to_camera": record["raw_world_to_camera"] @ np.linalg.inv(raw_to_world)}
        camera["group"] = str(frame.get("timestamp_group", Path(record["name"]).stem))
        saved_model = frame.get("camera_model", transforms.get("camera_model", camera["model"]))
        if saved_model not in {"SIMPLE_PINHOLE", "PINHOLE", "OPENCV"}:
            raise EvidenceError("Saved photo projection model needs a separately verified adapter")
        width, height = frame.get("w", transforms.get("w")), frame.get("h", transforms.get("h"))
        if not isinstance(width, int) or not isinstance(height, int) or not 0 < width <= 32768 or not 0 < height <= 32768:
            raise EvidenceError("Saved dataset image dimensions are missing or invalid")
        ratio = np.array([width / camera["width"], height / camera["height"]])
        parameters = list(camera["parameters"])
        expected_intrinsics = np.asarray(parameters[:4]) * np.tile(ratio, 2)
        saved_intrinsics = [frame.get(key, transforms.get(key)) for key in ("fl_x", "fl_y", "cx", "cy")]
        if any(value is None for value in saved_intrinsics) or not np.allclose(expected_intrinsics, saved_intrinsics, rtol=1e-6, atol=1e-5):
            raise EvidenceError("Saved photo intrinsics are not a verified resize of the COLMAP camera")
        distortion = [frame.get(key, transforms.get(key, 0)) for key in ("k1", "k2", "p1", "p2")]
        if not np.allclose(distortion, parameters[4:] or [0, 0, 0, 0], atol=1e-8) or frame.get("k3", transforms.get("k3", 0)) != 0:
            raise EvidenceError("Saved photo distortion differs from the retained COLMAP camera")
        camera.update(width=width, height=height, parameters=list(expected_intrinsics) + parameters[4:], observation_scale=ratio)
        if frame.get("mask_path"):
            camera["mask_key"] = "processed/" + str(frame["mask_path"]).removeprefix("./")
        camera["center"] = np.linalg.inv(camera["world_to_camera"])[:3, 3]
        cameras[identifier] = camera
    if len(cameras) < 3 or not records:
        raise EvidenceError("Recovery needs at least three linked cameras and sparse observations")
    points = np.asarray([record["xyz"] for record in records]) @ raw_to_world[:3, :3].T + raw_to_world[:3, 3]
    residuals = []
    for index in np.linspace(0, len(records) - 1, min(len(records), 256)).astype(int):
        record = records[index]
        for image_id, point_index in record["track"]:
            if image_id not in cameras:
                continue
            observations = images[image_id]["observations"]
            if point_index < 0 or point_index >= len(observations) or observations[point_index]["point_id"] != record["id"]:
                raise EvidenceError("COLMAP track and image observations disagree")
            pixels, depth = project(points[index:index + 1], cameras[image_id])
            if depth[0] <= 0:
                raise EvidenceError("Linked SfM observation projects behind its camera")
            observed = observations[point_index]
            residuals.append(float(np.linalg.norm(pixels[0] - np.array([observed["x"], observed["y"]]) * cameras[image_id]["observation_scale"])))
    if not residuals or np.median(residuals) > 3:
        raise EvidenceError("Camera reprojection evidence is missing or inconsistent")
    report = {"frame": "studio-y-up-metres", "points": len(points), "cameras": len(cameras),
              "colmap_to_studio": raw_to_world.tolist(),
              "reprojection_samples": len(residuals), "median_reprojection_px": float(np.median(residuals)),
              "p95_reprojection_px": float(np.quantile(residuals, .95)),
              "pixel_frame": "source photo resolution; recorded intrinsics verified against COLMAP resize",
              "geometry_evidence": "triangulated SfM observations, not Gaussian centres or independent survey truth"}
    verify_sources(job, sources, calibrated)
    return ReconstructionEvidence(job, points, records, cameras, sources, calibrated, report,
                                  images if retain_image_observations else None)


def verify_sources(job: Path, sources: dict, calibration: dict):
    if scale_revision(job) != calibration:
        raise EvidenceError("Capture calibration changed during recovery")
    for relative, identity in sources.items():
        if not manifests.same_file_identity(contained_file(job, relative), identity):
            raise EvidenceError("Reconstruction evidence changed; rebuild this recovery")
