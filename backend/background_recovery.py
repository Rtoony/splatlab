"""Evidence-supported planar background proposals, never invented hidden pixels."""

from __future__ import annotations

from io import BytesIO
from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path
import re
import struct
import time
import uuid

import numpy as np
from PIL import Image

import artifact_manifest as manifests
import glb_check
import reconstruction_evidence as evidence
import scene_revisions as scenes
from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY

RECOVERY_RE = re.compile(r"recovery_[a-f0-9]{24}\Z")
WORKER_LOCK = Path(__file__).resolve().parents[1] / "data/spatial/background-recovery.lock"


@contextmanager
def worker_slot(wait_seconds=0):
    if type(wait_seconds) not in (int, float) or not 0 <= wait_seconds <= 3:
        raise evidence.EvidenceError("Worker wait budget must be between zero and three seconds")
    WORKER_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with WORKER_LOCK.open("a") as handle:
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise evidence.EvidenceError("A background recovery is already running; retry after it finishes") from exc
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def floor_plane(points, records, bounds):
    lower, upper = np.array(bounds["min"]), np.array(bounds["max"])
    if np.max((upper - lower)[[0, 2]]) > 4:
        raise evidence.EvidenceError("Select an object smaller than four metres for bounded planar recovery")
    quality = np.array([record["error"] <= 2 and len(record["track"]) >= 3 for record in records])
    nearby = ((points[:, [0, 2]] >= lower[[0, 2]] - .8) & (points[:, [0, 2]] <= upper[[0, 2]] + .8)).all(axis=1)
    nearby &= (points[:, 1] >= lower[1] - .65) & (points[:, 1] <= lower[1] + .08)
    inside_object = ((points >= lower - .02) & (points <= upper + .02)).all(axis=1)
    indices = np.flatnonzero(quality & nearby & ~inside_object)
    if len(indices) < 60:
        raise evidence.EvidenceError("Too few tracked background features near the object's support surface")
    sample = points[indices[::max(1, len(indices) // 8000)]]
    random = np.random.default_rng(1234)
    best = None
    for _ in range(384):
        first, second, third = sample[random.choice(len(sample), 3, replace=False)]
        normal = np.cross(second - first, third - first)
        length = np.linalg.norm(normal)
        if length < 1e-9 or abs(normal[1] / length) < .94:
            continue
        normal /= length
        distances = np.abs((sample - first) @ normal)
        inliers = distances < .025
        if best is None or int(inliers.sum()) > int(best.sum()):
            best = inliers
    if best is None or best.sum() < 60 or best.mean() < .15:
        raise evidence.EvidenceError("No sufficiently supported approximately horizontal plane was found")
    fitted = sample[best]
    center = fitted.mean(axis=0)
    _, singular, axes = np.linalg.svd(fitted - center, full_matrices=False)
    if singular[1] < singular[0] * .05:
        raise evidence.EvidenceError("Background feature support is effectively collinear")
    normal = axes[-1]
    if normal[1] < 0:
        normal = -normal
    distances = np.abs((points[indices] - center) @ normal)
    support = indices[distances < .025]
    rms = float(np.sqrt(np.mean(distances[distances < .025] ** 2)))
    if len(support) < 60 or normal[1] < .94 or rms > .02:
        raise evidence.EvidenceError("The refined support surface failed its plane-quality checks")
    right = np.cross(normal, [0., 0., 1.])
    right /= np.linalg.norm(right)
    forward = np.cross(normal, right)
    target_center = (lower + upper) / 2
    center = target_center - normal * np.dot(target_center - center, normal)
    corners = np.array([[axis_x, axis_y, axis_z] for axis_x in (lower[0], upper[0])
                        for axis_y in (lower[1], upper[1]) for axis_z in (lower[2], upper[2])])
    planar = np.stack([(corners - center) @ right, (corners - center) @ forward], axis=1)
    return {"center": center, "normal": normal, "right": right, "forward": forward,
            "lower_uv": planar.min(axis=0) - .2, "upper_uv": planar.max(axis=0) + .2,
            "support_indices": support, "rms_m": rms, "regional_features": len(indices)}


def ray_box_occluded(center, points, bounds):
    directions = points - center
    lower, upper = np.asarray(bounds["min"]) - .015, np.asarray(bounds["max"]) + .015
    parallel = np.abs(directions) < 1e-12
    safe = np.where(parallel, 1., directions)
    first = (lower - center) / safe
    second = (upper - center) / safe
    entry = np.where(parallel, -np.inf, np.minimum(first, second))
    leave = np.where(parallel, np.inf, np.maximum(first, second))
    misses_parallel = (parallel & ((center < lower) | (center > upper))).any(axis=1)
    entry = entry.max(axis=1)
    leave = leave.min(axis=1)
    return ~misses_parallel & (entry <= leave) & (leave > 1e-5) & (entry < .995)


def support_grid(features_uv, lower_uv, span, size, radius_m=.08):
    supported = np.zeros((size, size), dtype=bool)
    if not len(features_uv):
        return supported
    cells = np.floor((features_uv - lower_uv) / span * size).astype(int)
    radii = np.minimum(np.ceil(radius_m / span * size).astype(int) + 1, size)
    for offset_x in range(-radii[0], radii[0] + 1):
        for offset_y in range(-radii[1], radii[1] + 1):
            positions = cells + [offset_x, offset_y]
            centers = lower_uv + (positions + .5) / size * span
            keep = ((positions >= 0) & (positions < size)).all(axis=1)
            keep &= np.linalg.norm(centers - features_uv, axis=1) <= radius_m
            supported[positions[keep, 1], positions[keep, 0]] = True
    return supported


def sample_view(source, image_id, plane, grid_points, size, bounds):
    camera = source.cameras[image_id]
    support = plane["support_indices"]
    visible = [index for index in support if image_id in source.records[index]["track"][:, 0]]
    relative = source.points[visible] - plane["center"]
    support_uv = np.stack([relative @ plane["right"], relative @ plane["forward"]], axis=1)
    valid = support_grid(support_uv, plane["lower_uv"], plane["upper_uv"] - plane["lower_uv"], size).reshape(-1)
    pixels, depths = evidence.project(grid_points, camera)
    valid &= (depths > 0) & np.isfinite(pixels).all(axis=1)
    valid &= ((pixels >= 0) & (pixels < np.array([camera["width"] - 1, camera["height"] - 1]))).all(axis=1)
    for box in bounds:
        valid &= ~ray_box_occluded(camera["center"], grid_points, box)
    normal_view = camera["center"] - plane["center"]
    if normal_view @ plane["normal"] / np.linalg.norm(normal_view) < .15:
        valid[:] = False
    if camera.get("mask_key"):
        mask_path = evidence.input_path(source.job, camera["mask_key"], source.sources)
        with Image.open(mask_path) as opened:
            if opened.size != (camera["width"], camera["height"]):
                raise evidence.EvidenceError("Observation mask dimensions differ from the source photo")
            mask = np.asarray(opened.convert("L"))
        indices = np.flatnonzero(valid)
        nearest = np.rint(pixels[indices]).astype(int)
        valid[indices] &= mask[nearest[:, 1], nearest[:, 0]] >= 128
    image_path = evidence.input_path(source.job, camera["image_key"], source.sources)
    with Image.open(image_path) as opened:
        if opened.size != (camera["width"], camera["height"]):
            raise evidence.EvidenceError("Image dimensions differ from its solved camera; no silent resizing")
        photo = np.asarray(opened.convert("RGB"))
    colors = np.zeros((len(grid_points), 3), dtype=np.float32)
    indices = np.flatnonzero(valid)
    nearest = np.floor(pixels[indices]).astype(int)
    fraction = pixels[indices] - nearest
    horizontal, vertical = nearest[:, 0], nearest[:, 1]
    weight_x, weight_y = fraction[:, 0:1], fraction[:, 1:2]
    colors[indices] = ((1 - weight_x) * (1 - weight_y) * photo[vertical, horizontal]
                      + weight_x * (1 - weight_y) * photo[vertical, horizontal + 1]
                      + (1 - weight_x) * weight_y * photo[vertical + 1, horizontal]
                      + weight_x * weight_y * photo[vertical + 1, horizontal + 1]) / 255
    return colors, valid


def consensus(colors, valid, centers, points, minimum_angle_degrees=3., groups=None):
    sample_count = valid.sum(axis=0)
    ordered = np.sort(np.where(valid[..., None], colors, np.inf), axis=0)
    median_indices = np.maximum(0, (sample_count - 1) // 2)
    median = ordered[median_indices, np.arange(len(points))]
    median[~np.isfinite(median)] = 0
    differences = np.max(np.abs(colors - median), axis=2)
    agrees = valid & (differences <= .12)
    angular = np.zeros(len(points), dtype=bool)
    for first in range(len(centers)):
        direction_first = centers[first] - points
        direction_first /= np.maximum(np.linalg.norm(direction_first, axis=1, keepdims=True), 1e-9)
        for second in range(first + 1, len(centers)):
            if groups is not None and groups[first] == groups[second]:
                continue
            direction_second = centers[second] - points
            direction_second /= np.maximum(np.linalg.norm(direction_second, axis=1, keepdims=True), 1e-9)
            cosine = np.sum(direction_first * direction_second, axis=1)
            angular |= agrees[first] & agrees[second] & (cosine <= np.cos(np.deg2rad(minimum_angle_degrees)))
    accepted = (agrees.sum(axis=0) >= 2) & angular
    texture = np.sum(colors * agrees[..., None], axis=0) / np.maximum(agrees.sum(axis=0)[:, None], 1)
    texture[~accepted] = 0
    return texture, accepted, agrees, sample_count


def textured_grid_glb(plane, accepted, texture_png: bytes, size: int) -> bytes:
    coordinates = np.linspace(0, 1, size + 1)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    uv = np.stack([horizontal, vertical], axis=-1).reshape(-1, 2)
    plane_uv = plane["lower_uv"] + uv * (plane["upper_uv"] - plane["lower_uv"])
    vertices = (plane["center"] + plane_uv[:, 0:1] * plane["right"] + plane_uv[:, 1:2] * plane["forward"]).astype("<f4")
    rows, columns = np.nonzero(accepted.reshape(size, size))
    starts = rows * (size + 1) + columns
    faces = np.stack([starts, starts + 1, starts + size + 2, starts, starts + size + 2, starts + size + 1], axis=1).astype("<u4").reshape(-1)
    if not len(faces):
        raise evidence.EvidenceError("No multi-view-supported cells survived; no mesh can be proposed")
    used, remapped = np.unique(faces, return_inverse=True)
    vertices, uv = vertices[used], uv[used].astype("<f4")
    payloads = [vertices.tobytes(), uv.tobytes(), remapped.astype("<u4").tobytes(), texture_png]
    binary = bytearray()
    views = []
    for payload in payloads:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)})
        binary.extend(payload)
        binary.extend(b"\0" * (-len(binary) % 4))
    document = {"asset": {"version": "2.0", "generator": "SplatLab observed-background recovery"},
        "extensionsUsed": ["KHR_materials_unlit"], "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3", "min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
                      {"bufferView": 1, "componentType": 5126, "count": len(uv), "type": "VEC2"},
                      {"bufferView": 2, "componentType": 5125, "count": len(remapped), "type": "SCALAR"}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}], "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"source": 0, "sampler": 0}],
        "materials": [{"extensions": {"KHR_materials_unlit": {}}, "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": .5,
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "extras": {GLTF_EXTRAS_KEY: GENERATIVE_TAG, "geometry_source": "sfm-plane-fit", "appearance_source": "captured-photos", "unknown_cells_omitted": True}}
    document["asset"]["extras"] = {GLTF_EXTRAS_KEY: GENERATIVE_TAG}
    for item in document["nodes"] + document["meshes"]:
        item["extras"] = {GLTF_EXTRAS_KEY: GENERATIVE_TAG, "geometry_source": "sfm-plane-fit", "appearance_source": "captured-photos"}
    encoded = scenes.canonical_bytes(document)
    encoded += b" " * (-len(encoded) % 4)
    return struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary)) + struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\0") + binary


def recover(source, bounds, size=128, max_views=24, plane=None, visibility_masks=None):
    if size not in {64, 128, 256} or not 3 <= max_views <= 24:
        raise evidence.EvidenceError("Recovery budget requires 64/128/256 texels and 3–24 views")
    plane = floor_plane(source.points, source.records, bounds[0]) if plane is None else plane
    coordinates = (np.arange(size) + .5) / size
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    uv = plane["lower_uv"] + np.stack([horizontal, vertical], axis=-1).reshape(-1, 2) * (plane["upper_uv"] - plane["lower_uv"])
    grid_points = plane["center"] + uv[:, 0:1] * plane["right"] + uv[:, 1:2] * plane["forward"]
    votes = {identifier: 0 for identifier in source.cameras}
    relative = source.points[plane["support_indices"]] - plane["center"]
    feature_uv = np.stack([relative @ plane["right"], relative @ plane["forward"]], axis=1)
    near_texture = ((feature_uv >= plane["lower_uv"] - .08) & (feature_uv <= plane["upper_uv"] + .08)).all(axis=1)
    texture_support = plane["support_indices"][near_texture]
    for index in texture_support:
        for identifier in np.unique(source.records[index]["track"][:, 0]):
            if identifier in votes:
                votes[identifier] += 1
    eligible = sorted((identifier for identifier, votes in votes.items() if votes >= 12), key=lambda identifier: (-votes[identifier], identifier))
    holdout_groups = set(sorted({camera["group"] for camera in source.cameras.values()})[::5])
    holdout_ids = {identifier for identifier, camera in source.cameras.items() if camera["group"] in holdout_groups}
    selected = [identifier for identifier in eligible if identifier not in holdout_ids][:max_views]
    holdouts = [identifier for identifier in eligible if identifier in holdout_ids][:4]
    if len(selected) < 2 or not holdouts:
        raise evidence.EvidenceError("Not enough tracked viewpoints for recovery plus a separate appearance check")
    if visibility_masks is not None:
        if (set(visibility_masks) != set(selected + holdouts)
                or any(mask.shape != (size * size,) or mask.dtype != bool for mask in visibility_masks.values())):
            raise evidence.EvidenceError("Visibility masks must address every fit/check view and texture cell")
    samples = [sample_view(source, identifier, plane, grid_points, size, bounds) for identifier in selected]
    colors, masks = np.stack([sample[0] for sample in samples]), np.stack([sample[1] for sample in samples])
    if visibility_masks is not None:
        masks &= np.stack([visibility_masks[identifier] for identifier in selected])
    texture, accepted, agreement, sample_count = consensus(colors, masks, np.array([source.cameras[identifier]["center"] for identifier in selected]), grid_points,
                                                         groups=[source.cameras[identifier]["group"] for identifier in selected])
    errors = []
    for identifier in holdouts:
        held_colors, held_valid = sample_view(source, identifier, plane, grid_points, size, bounds)
        if visibility_masks is not None:
            held_valid &= visibility_masks[identifier]
        overlap = held_valid & accepted
        if overlap.any():
            errors.append({"image_id": identifier, "pixels": int(overlap.sum()), "mean_rgb_absolute_error": float(np.abs(texture[overlap] - held_colors[overlap]).mean())})
    rgba = np.concatenate([np.rint(texture * 255).astype(np.uint8), accepted[:, None].astype(np.uint8) * 255], axis=1).reshape(size, size, 4)
    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    diagnostics = np.zeros((size * size, 3), dtype=np.uint8)
    diagnostics[:] = [210, 40, 160]
    diagnostics[accepted] = [25, 210, 190]
    report = {"plane": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in plane.items() if key != "support_indices"},
              "support_points": len(plane["support_indices"]), "texture_support_points": len(texture_support),
              "view_selection": "tracked support inside the texture footprint plus the 8 cm sampling radius",
              "texture_size": size, "supported_fraction": float(accepted.mean()),
              "unknown_fraction": float((~accepted).mean()), "source_image_ids": selected, "appearance_check_image_ids": holdouts,
              "source_groups": sorted({source.cameras[identifier]["group"] for identifier in selected}),
              "appearance_check_groups": sorted({source.cameras[identifier]["group"] for identifier in holdouts}),
              "split_unit": "explicit timestamp_group, otherwise shared image basename (SplatLab rig convention)",
              "appearance_checks": errors, "geometry": "plane fitted to tracked SfM points; unsupported cells omitted",
              "recipe": {"version": 2, "plane_inlier_m": .025, "max_plane_rms_m": .02, "feature_support_radius_m": .08,
                         "max_rgb_channel_disagreement": .12, "minimum_angle_degrees": 3., "minimum_distinct_groups": 2,
                         "object_occlusion_padding_m": .015, "max_source_views": max_views},
              "appearance": "bilinear source-photo samples with multi-view color agreement and at least 3 degrees baseline",
              "visibility": "local tracked-feature support plus conservative object bounds; not dense-depth-verified",
              "unknown_policy": "transparent and absent geometry; no inpainting, nearest-color fill or invented pixels",
              "evaluation_scope": "appearance checks exclude texture source images; SfM still used all images, so these are not independent geometry measurements"}
    mesh = textured_grid_glb(plane, accepted, buffer.getvalue(), size)
    per_pixel_sources = np.where(agreement & accepted[None, :], np.array(selected)[:, None], -1).astype(np.int32)
    return mesh, buffer.getvalue(), diagnostics.reshape(size, size, 3), report, {
        "source_image_ids": per_pixel_sources, "support_counts": sample_count.reshape(size, size), "accepted": accepted.reshape(size, size),
        "support_point_ids": np.array([source.records[index]["id"] for index in plane["support_indices"]], dtype=np.uint64),
        "support_points_world": source.points[plane["support_indices"]]}


def build(job: Path, selected_slug: str, expected_generation: int, size=128, support_anchors=None, visibility_id=None):
    with worker_slot(wait_seconds=3), scenes.write_lock(job):
        pointer = scenes.active(job)
        if not pointer or pointer["generation"] != expected_generation:
            raise evidence.EvidenceError("Active scene changed before background recovery")
        revision = scenes.read_revision(job, pointer["revision_id"])
        if revision["source_fingerprint"] != scenes.source_fingerprint(job):
            raise evidence.EvidenceError("Refresh the changed capture baseline before recovery")
        viewer = revision["state"]["viewer"]
        if viewer.get("units") != "meters" or viewer.get("calibration", {}).get("stale"):
            raise evidence.EvidenceError("Rebuild the calibrated metre world before fitting a support surface")
        candidates = [entry for entry in revision["state"]["viewer"]["elements"] if entry.get("provenance") != "authored"]
        selected = next((entry for entry in candidates if entry["slug"] == selected_slug), None)
        if not selected:
            raise evidence.EvidenceError("Select an observed captured element, not an authored replacement")
        bounds = []
        for entry in [selected] + [item for item in candidates if item != selected]:
            key = entry.get("files", {}).get("glb", "").removeprefix("artifact:")
            geometry = glb_check.position_bounds(scenes.artifact(job, pointer["revision_id"], key))
            if not geometry["identity_transforms"]:
                raise evidence.EvidenceError("Background recovery requires baked metre-world transforms")
            bounds.append(geometry["aabb"])
        source = evidence.load(job, retain_image_observations=support_anchors is not None)
        plane = None
        if support_anchors is not None:
            import support_surfaces

            plane = support_surfaces.fit(source, bounds, support_anchors, pointer)
        visibility_masks, visibility_metadata, visibility_artifacts = None, None, {}
        if visibility_id is not None:
            import background_visibility

            plane = floor_plane(source.points, source.records, bounds[0]) if plane is None else plane
            visibility_masks, visibility_metadata, visibility_artifacts = background_visibility.qualification(job, visibility_id, pointer, selected_slug, size, plane)
            source.sources.update(background_visibility.read(job, visibility_id)["sources"])
        mesh, texture, diagnostic, report, pixels = recover(source, bounds, size, plane=plane, visibility_masks=visibility_masks)
        if visibility_metadata:
            report["visibility"] = visibility_metadata["scope"]
            report["visibility_qualification"] = visibility_metadata
            report["visibility_comparison"] = background_visibility.compare_recovery(job, visibility_id, source, bounds, plane, visibility_masks, texture, pixels)
            report["recipe"]["version"] = 3
        evidence.verify_sources(job, source.sources, source.calibration)
        if revision["source_fingerprint"] != scenes.source_fingerprint(job):
            raise evidence.EvidenceError("Capture changed during recovery; proposal discarded")
        identifier = "recovery_" + uuid.uuid4().hex[:24]
        directory = scenes.root(job) / "recoveries" / identifier
        directory.mkdir(parents=True)
        (directory / "candidate.glb").write_bytes(mesh)
        (directory / "atlas.png").write_bytes(texture)
        Image.fromarray(diagnostic).save(directory / "support.png")
        np.savez_compressed(directory / "observations.npz", **pixels)
        glb_check.validate_glb(directory / "candidate.glb")
        camera_ids = set(report["source_image_ids"] + report["appearance_check_image_ids"])
        if support_anchors is not None:
            camera_ids.update(pick["image_id"] for pick in support_anchors["points"])
        receipt = {"schema": "dev.splatlab.background-recovery/v1", "recovery_id": identifier, "base": pointer,
                   "render_vr_only": True,
                   "implementation": {"recovery_sha256": manifests.sha256_file(Path(__file__)), "evidence_sha256": manifests.sha256_file(Path(evidence.__file__))},
                   "selected_slug": selected_slug, "created_at": manifests.utc_now(), "evidence": source.report,
                   "sources": source.sources, "calibration": source.calibration, "report": report,
                   "cameras": {str(identifier): {"image_key": camera["image_key"], "model": camera["model"], "parameters": [float(value) for value in camera["parameters"]],
                                "width": camera["width"], "height": camera["height"], "group": camera["group"], "world_to_camera": camera["world_to_camera"].tolist()}
                               for identifier, camera in source.cameras.items() if identifier in camera_ids},
                   "artifacts": {path.name: scenes.store_file(job, path) for path in directory.iterdir() if path.is_file()}}
        if visibility_metadata:
            receipt["visibility_review"] = visibility_metadata
            receipt["artifacts"].update(visibility_artifacts)
            receipt["implementation"]["background_visibility_sha256"] = manifests.sha256_file(Path(background_visibility.__file__))
        if support_anchors is not None:
            receipt["implementation"]["support_surfaces_sha256"] = manifests.sha256_file(Path(support_surfaces.__file__))
            receipt["support_anchors"] = support_anchors
        receipt["sha256"] = hashlib.sha256(scenes.canonical_bytes(receipt)).hexdigest()
        scenes.durable_json(job, directory / "receipt.json", receipt)
        return receipt


def read(job: Path, identifier: str):
    if not RECOVERY_RE.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid background recovery identifier")
    receipt = manifests.read_json(scenes.root(job) / "recoveries" / identifier / "receipt.json")
    if not receipt:
        raise evidence.EvidenceError("Background recovery not found")
    payload = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != hashlib.sha256(scenes.canonical_bytes(payload)).hexdigest():
        raise evidence.EvidenceError("Background recovery receipt failed its integrity check")
    return receipt


def verify_receipt_sources(job: Path, receipt: dict, checksums=False):
    evidence.verify_sources(job, receipt["sources"], receipt["calibration"])
    if checksums:
        for relative, identity in receipt["sources"].items():
            if manifests.sha256_file(evidence.contained_file(job, relative)) != identity["sha256"]:
                raise evidence.EvidenceError("Recovery source checksum changed; proposal is stale")


def artifact(job: Path, identifier: str, name: str):
    receipt = read(job, identifier)
    record = receipt["artifacts"].get(name)
    if not record:
        raise evidence.EvidenceError("This artifact does not belong to the recovery")
    path = scenes.blob_path(job, record["sha256"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"]:
        raise evidence.EvidenceError("Recovery artifact is missing")
    return path
