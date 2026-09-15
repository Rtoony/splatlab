"""Photo-semantic, multi-view tracked structural surfaces for reviewed architectural editing."""

from contextlib import contextmanager
import fcntl
import hashlib
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageDraw

import artifact_manifest as manifests
import background_recovery as recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as selections

IDENTIFIER = re.compile(r"structure_[a-f0-9]{24}\Z")
METHOD = "photo-semantic-wall-tracks/v1"


def directory(job, identifier):
    if not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid structural surface identifier")
    output = scenes.root(job) / "structural-surfaces" / identifier
    if output.resolve() != output.absolute():
        raise evidence.EvidenceError("Structural surface directory cannot be symlinked")
    return output


def read(job, identifier, stage="receipt"):
    if stage not in {"receipt", "masks", "result"}:
        raise evidence.EvidenceError("Unknown structural surface stage")
    value = manifests.read_json(directory(job, identifier) / (stage + ".json"))
    if not value or value.get("sha256") != hashlib.sha256(scenes.canonical_bytes({key: item for key, item in value.items() if key != "sha256"})).hexdigest():
        raise evidence.EvidenceError("Structural surface receipt is missing or corrupt")
    if stage != "receipt" and value.get("prepared_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Structural result belongs to different preparation")
    if stage == "result" and value.get("masks_sha256") != read(job, identifier, "masks")["sha256"]:
        raise evidence.EvidenceError("Structural fit belongs to different masks")
    return value


def artifact(job, identifier, name, stage="receipt"):
    record = read(job, identifier, stage)["artifacts"].get(name)
    if not record:
        raise evidence.EvidenceError("Artifact does not belong to this structural study")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Structural surface artifact is missing or corrupt")
    return path


def verify(job, receipt, prepared_files=False):
    pointer = scenes.active(job)
    if pointer != receipt["base"]:
        raise evidence.EvidenceError("Structural surface study belongs to an older active revision")
    revision = scenes.read_revision(job, pointer["revision_id"])
    if revision["source_fingerprint"] != scenes.source_fingerprint(job):
        raise evidence.EvidenceError("Capture changed before structural fitting")
    recovery.verify_receipt_sources(job, receipt, checksums=True)
    for name, record in receipt["artifacts"].items():
        artifact(job, receipt["structure_id"], name)
        if prepared_files:
            path = evidence.contained_file(job, str((directory(job, receipt["structure_id"]) / name).relative_to(job)))
            if manifests.sha256_file(path) != record["sha256"]:
                raise evidence.EvidenceError("Prepared structural photo changed")


@contextmanager
def worker_slot(job, identifier):
    with (directory(job, identifier) / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise evidence.EvidenceError("Structural study already has an active worker") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(job, expected_generation, image_ids):
    if (not isinstance(image_ids, list) or not 4 <= len(image_ids) <= 12
            or any(type(image_id) is not int or image_id <= 0 for image_id in image_ids) or len(set(image_ids)) != len(image_ids)):
        raise evidence.EvidenceError("Choose four through twelve distinct captured photo IDs")
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        pointer = scenes.active(job)
        if not pointer or pointer["generation"] != expected_generation:
            raise evidence.EvidenceError("Active scene changed before structural preparation")
        revision = scenes.read_revision(job, pointer["revision_id"])
        if revision["source_fingerprint"] != scenes.source_fingerprint(job):
            raise evidence.EvidenceError("Refresh the changed captured baseline")
        if revision["state"]["viewer"].get("units") != "meters" or revision["state"]["viewer"].get("calibration", {}).get("stale"):
            raise evidence.EvidenceError("Structural fitting requires current metre calibration")
        source = evidence.load(job)
        if any(image_id not in source.cameras for image_id in image_ids):
            raise evidence.EvidenceError("Structural photo does not belong to the solved capture")
        if len({source.cameras[image_id]["group"] for image_id in image_ids}) != len(image_ids):
            raise evidence.EvidenceError("Structural photos must use distinct timestamp groups")
        identifier = "structure_" + uuid.uuid4().hex[:24]
        output = directory(job, identifier)
        (output / "frames").mkdir(parents=True)
        files, cameras = [], []
        for ordinal, image_id in enumerate(image_ids):
            camera = source.cameras[image_id]
            path = evidence.input_path(job, camera["image_key"], source.sources)
            with Image.open(path) as opened:
                if opened.size != (camera["width"], camera["height"]):
                    raise evidence.EvidenceError("Structural photo differs from its calibrated intrinsics")
                scale = min(1., 960 / max(opened.size))
                size = tuple(round(dimension * scale) for dimension in opened.size)
                photo = opened.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            name = f"frames/cam_{image_id:03d}.png"
            photo.save(output / name)
            files.append(output / name)
            if camera.get("mask_key"):
                evidence.input_path(job, camera["mask_key"], source.sources)
            cameras.append({"image_id": image_id, "image_key": camera["image_key"], "group": camera["group"], "photo": name,
                "width": size[0], "height": size[1], "source_width": camera["width"], "source_height": camera["height"],
                "world_to_camera": camera["world_to_camera"].tolist(), "split": "check" if ordinal % 4 == 3 else "fit"})
        manifests.atomic_write_json(output / "views.json", {"cam_indices": image_ids})
        manifests.atomic_write_json(output / "things.json", ["wall"])
        files += [output / "views.json", output / "things.json"]
        evidence.verify_sources(job, source.sources, source.calibration)
        receipt = selections.seal(job, output, "receipt.json", {"schema": "dev.splatlab.structural-surfaces/v1", "structure_id": identifier,
            "method": METHOD, "base": pointer, "calibration": source.calibration, "sources": source.sources, "cameras": cameras,
            "created_at": manifests.utc_now(), "prompt": "wall", "evidence": source.report,
            "recipe": {"semantic_threshold": .5, "point_reprojection_limit_px": 3., "minimum_fit_votes": 2,
                       "minimum_fit_angle_degrees": 3., "positive_to_negative_ratio": 2, "plane_inlier_m": .03},
            "scope": "Semantic wall candidates anchored to captured SfM tracks; not measured structure, automatic approval or a doorway cut"}, files)
        verify(job, receipt)
        return receipt


def semantic_mask(job, receipt, camera):
    name = f"masks/wall/cam_{camera['image_id']:03d}.npz"
    with np.load(artifact(job, receipt["structure_id"], name, "masks"), allow_pickle=False) as stored:
        if not {"masks", "scores", "prompt"}.issubset(stored.files) or stored["prompt"].shape != ():
            raise evidence.EvidenceError("Structural mask is missing its prompt/photo contract")
        masks, scores = stored["masks"], stored["scores"]
        if (masks.ndim != 3 or masks.dtype != bool or masks.shape[1:] != (camera["height"], camera["width"])
                or scores.shape != (len(masks),) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
                or len(masks) > 128 or stored["prompt"].item() != receipt["prompt"]):
            raise evidence.EvidenceError("Structural mask does not match its prompt/photo contract")
        return masks[scores >= receipt["recipe"]["semantic_threshold"]].any(axis=0)


def track_observations(source, camera, semantic):
    original = source.cameras[camera["image_id"]]
    observations = source.images[camera["image_id"]]["observations"]
    lookup = {record["id"]: index for index, record in enumerate(source.records)}
    pairs = [(ordinal, lookup[int(observed["point_id"])]) for ordinal, observed in enumerate(observations) if observed["point_id"] in lookup]
    indices, pixels = [], []
    for ordinal, index in pairs:
        record = source.records[index]
        linked = record["track"][record["track"][:, 0] == camera["image_id"]]
        if record["error"] > 2 or len(record["track"]) < 3 or len(linked) != 1 or linked[0, 1] != ordinal:
            continue
        observed = observations[ordinal]
        indices.append(index)
        pixels.append(np.array([observed["x"], observed["y"]]) * original["observation_scale"])
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        return indices, np.empty(0, bool), np.empty((0, 2)), np.empty(0)
    pixels = np.asarray(pixels)
    projected, depth = evidence.project(source.points[indices], original)
    residuals = np.linalg.norm(projected - pixels, axis=1)
    valid = (depth > 0) & np.isfinite(pixels).all(axis=1) & (residuals <= 3)
    valid &= (pixels >= 0).all(axis=1) & (pixels < [original["width"], original["height"]]).all(axis=1)
    if original.get("mask_key"):
        with Image.open(evidence.input_path(source.job, original["mask_key"], source.sources)) as opened:
            if opened.size != (original["width"], original["height"]):
                raise evidence.EvidenceError("Structural source observation mask has different dimensions")
            observed_mask = np.asarray(opened.convert("L"))
        safe = pixels[valid].astype(int)
        valid[np.flatnonzero(valid)] &= observed_mask[safe[:, 1], safe[:, 0]] >= 128
    indices, pixels, residuals = indices[valid], pixels[valid], residuals[valid]
    pixels *= [camera["width"] / original["width"], camera["height"] / original["height"]]
    nearest = np.floor(pixels).astype(int)
    positive = semantic[nearest[:, 1], nearest[:, 0]]
    return indices, positive, pixels, residuals


def membership(points, cameras, observed, positive):
    centers = np.asarray([camera["center"] for camera in cameras], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or centers.shape != (len(cameras), 3) or not np.isfinite(centers).all()
            or any(camera["split"] not in {"fit", "check"} for camera in cameras)):
        raise evidence.EvidenceError("Structural votes require finite calibrated points/cameras and declared splits")
    if observed.shape != positive.shape or observed.shape != (len(cameras), len(points)) or observed.dtype != bool or positive.dtype != bool or np.any(positive & ~observed):
        raise evidence.EvidenceError("Structural votes must address every point and view")
    groups = [camera["group"] for camera in cameras]
    if len(set(groups)) != len(groups):
        raise evidence.EvidenceError("Duplicate timestamp groups cannot establish structural support")
    fit = np.array([camera["split"] == "fit" for camera in cameras])
    fit_positive = positive[fit].sum(axis=0)
    fit_negative = (observed[fit] & ~positive[fit]).sum(axis=0)
    supported = (fit_positive >= 2) & (fit_positive >= 2 * fit_negative)
    angular = np.zeros(len(points), dtype=bool)
    for first in np.flatnonzero(fit):
        first_direction = centers[first] - points
        first_length = np.linalg.norm(first_direction, axis=1, keepdims=True)
        first_direction /= np.maximum(first_length, 1e-12)
        for second in np.flatnonzero(fit):
            if second <= first:
                continue
            second_direction = centers[second] - points
            second_length = np.linalg.norm(second_direction, axis=1, keepdims=True)
            second_direction /= np.maximum(second_length, 1e-12)
            angular |= (positive[first] & positive[second] & (first_length[:, 0] > 1e-12) & (second_length[:, 0] > 1e-12)
                        & (np.sum(first_direction * second_direction, axis=1) <= np.cos(np.deg2rad(3))))
    return supported & angular, fit_positive, fit_negative


def fit_planes(points, indices, camera_center, maximum=6):
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or indices.ndim != 1
            or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(points)) or len(np.unique(indices)) != len(indices)
            or type(maximum) is not int or not 1 <= maximum <= 6
            or np.asarray(camera_center).shape != (3,) or not np.isfinite(camera_center).all()):
        raise evidence.EvidenceError("Invalid bounded structural fitting input")
    random = np.random.default_rng(20260907)
    remaining = indices.copy()
    planes = []
    for _ in range(maximum):
        if len(remaining) < 60:
            break
        sample = points[random.choice(remaining, min(len(remaining), 8000), replace=False)]
        best = None
        for _ in range(768):
            first, second, third = sample[random.choice(len(sample), 3, replace=False)]
            normal = np.cross(second - first, third - first)
            length = np.linalg.norm(normal)
            if length < .02 or abs(normal[1] / length) > .15:
                continue
            normal /= length
            inliers = np.abs((sample - first) @ normal) < .03
            if best is None or inliers.sum() > best[0]:
                best = int(inliers.sum()), first, normal
        if best is None or best[0] < 40:
            break
        selected = remaining[np.abs((points[remaining] - best[1]) @ best[2]) < .03]
        center = np.mean(points[selected], axis=0)
        _, singular, axes = np.linalg.svd(points[selected] - center, full_matrices=False)
        normal = axes[-1]
        if abs(normal[1]) > .15 or singular[1] < singular[0] * .1:
            remaining = remaining[~np.isin(remaining, selected)]
            continue
        if np.dot(np.asarray(camera_center) - center, normal) < 0:
            normal *= -1
        right = np.cross([0., 1., 0.], normal)
        right /= np.linalg.norm(right)
        up = np.cross(normal, right)
        residuals = np.abs((points[selected] - center) @ normal)
        selected = selected[residuals < .03]
        remaining = remaining[~np.isin(remaining, selected)]
        if len(selected) < 60:
            continue
        uv = np.stack([(points[selected] - center) @ right, (points[selected] - center) @ up], axis=1)
        lower, upper = np.quantile(uv, [.02, .98], axis=0)
        rms = float(np.sqrt(np.mean(((points[selected] - center) @ normal) ** 2)))
        if np.any(upper - lower < [.8, .6]) or rms > .02:
            continue
        cells = np.floor((uv - lower) / .15).astype(int)
        inside = ((uv >= lower) & (uv <= upper)).all(axis=1)
        shape = np.maximum(1, np.ceil((upper - lower) / .15).astype(int))
        coverage = len(np.unique(cells[inside], axis=0)) / int(np.prod(shape))
        planes.append({"center": center.tolist(), "normal_toward_cameras": normal.tolist(), "right": right.tolist(), "up": up.tolist(),
            "lower_uv": lower.tolist(), "upper_uv": upper.tolist(), "support_count": len(selected), "rms_m": rms,
            "support_grid_cell_m": .15, "support_grid_coverage": coverage, "point_indices": selected,
            "scope": "Fitted visible wall support; rectangular extent interpolates/extrapolates gaps and provides no wall thickness"})
    return planes


def projected_depth_comparison(points, plane, camera):
    center = np.asarray(camera["center"], dtype=float)
    plane_center = np.asarray(plane["center"], dtype=float)
    normal = np.asarray(plane["normal_toward_cameras"], dtype=float)
    right, up = np.asarray(plane["right"], dtype=float), np.asarray(plane["up"], dtype=float)
    lower, upper = np.asarray(plane["lower_uv"], dtype=float), np.asarray(plane["upper_uv"], dtype=float)
    transform = np.asarray(camera["world_to_camera"], dtype=float)
    vectors = (center, plane_center, normal, right, up)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or any(value.shape != (3,) or not np.isfinite(value).all() for value in vectors)
            or transform.shape != (4, 4) or not np.isfinite(transform).all()
            or lower.shape != (2,) or upper.shape != (2,) or not np.isfinite([lower, upper]).all() or np.any(upper <= lower)):
        raise evidence.EvidenceError("Projected depth comparison requires finite calibrated geometry")
    basis = np.stack([right, up, normal])
    optical_axis = transform[2, :3]
    length = np.linalg.norm(optical_axis)
    if (not np.allclose(basis @ basis.T, np.eye(3), atol=1e-6) or length < 1e-12
            or not np.allclose(transform[:3, :3] @ center + transform[:3, 3], 0, atol=1e-6)):
        raise evidence.EvidenceError("Projected depth comparison requires a consistent camera and orthonormal plane")
    rays = points - center
    denominator = rays @ normal
    fractions = np.zeros(len(points))
    np.divide(np.dot(plane_center - center, normal), denominator, out=fractions, where=np.abs(denominator) > 1e-12)
    intersections = center + rays * fractions[:, None]
    relative = intersections - plane_center
    uv = np.stack([relative @ right, relative @ up], axis=1)
    depths_m = rays @ (optical_axis / length)
    valid = (fractions > 0) & (depths_m > 0) & (np.abs(denominator) > 1e-12) & ((uv >= lower) & (uv <= upper)).all(axis=1)
    indices = np.flatnonzero(valid)
    errors = (fractions[indices] - 1) * depths_m[indices]
    return {"indices": indices, "signed_optical_depth_error_m": errors,
            "sample_count": len(points), "projected_footprint_samples": len(indices),
            "median_absolute_error_m": float(np.median(np.abs(errors))) if len(errors) else None,
            "p90_absolute_error_m": float(np.quantile(np.abs(errors), .9)) if len(errors) else None,
            "within_5cm": int((np.abs(errors) <= .05).sum()),
            "plane_behind_prior_count": int((errors > .05).sum()), "plane_in_front_of_prior_count": int((errors < -.05).sum()),
            "scope": "Same-ray inferred-depth consistency; positive error can mean occlusion. Neither prior is independent wall geometry."}


def fit(job, identifier):
    with recovery.worker_slot(wait_seconds=3), worker_slot(job, identifier):
        receipt = read(job, identifier)
        verify(job, receipt)
        masks_record = read(job, identifier, "masks")
        expected = {"sam3_manifest.json"} | {f"masks/wall/cam_{camera['image_id']:03d}.npz" for camera in receipt["cameras"]}
        if set(masks_record["artifacts"]) != expected:
            raise evidence.EvidenceError("Structural masks do not address the exact prepared camera set")
        output = directory(job, identifier)
        if (output / "result.json").exists() or (output / "membership.npz").exists():
            raise evidence.EvidenceError("Structural fit already exists; prepare another study")
        source = evidence.load(job, retain_image_observations=True)
        observed = np.zeros((len(receipt["cameras"]), len(source.points)), dtype=bool)
        positive = np.zeros_like(observed)
        cameras, view_reports, files = [], [], []
        for ordinal, camera in enumerate(receipt["cameras"]):
            semantic = semantic_mask(job, receipt, camera)
            indices, accepted, pixels, residuals = track_observations(source, camera, semantic)
            observed[ordinal, indices] = True
            positive[ordinal, indices] = accepted
            cameras.append({**camera, "center": source.cameras[camera["image_id"]]["center"].tolist()})
            with Image.open(artifact(job, identifier, camera["photo"])) as opened:
                overlay = opened.convert("RGBA")
            tint = np.zeros((*semantic.shape, 4), dtype=np.uint8)
            tint[semantic] = [20, 210, 160, 80]
            overlay = Image.alpha_composite(overlay, Image.fromarray(tint)).convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for pixel in pixels[accepted][::max(1, int(accepted.sum()) // 1500)]:
                horizontal, vertical = pixel
                draw.ellipse((horizontal - 1, vertical - 1, horizontal + 1, vertical + 1), fill=(255, 215, 30))
            path = output / f"wall-tracks-{camera['image_id']}.png"
            overlay.save(path)
            files.append(path)
            view_reports.append({"image_id": camera["image_id"], "split": camera["split"], "observed_points": len(indices),
                "positive_points": int(accepted.sum()), "semantic_pixels": int(semantic.sum()),
                "maximum_reprojection_px": float(residuals.max()) if len(residuals) else None, "overlay": path.name})
        accepted, positives, negatives = membership(source.points, cameras, observed, positive)
        planes = fit_planes(source.points, np.flatnonzero(accepted), np.mean([camera["center"] for camera in cameras if camera["split"] == "fit"], axis=0))
        arrays = {"point_ids": np.array([record["id"] for record in source.records], dtype=np.uint64), "points_world": source.points,
            "observed": observed, "positive": positive, "accepted": accepted, "fit_positive_votes": positives, "fit_negative_votes": negatives}
        for ordinal, plane in enumerate(planes):
            indices = plane.pop("point_indices")
            arrays[f"plane_{ordinal}_indices"] = indices
            plane["plane_id"] = f"wall-{ordinal}"
            plane["appearance_checks"] = [{"image_id": camera["image_id"], "observed_points": int(observed[offset, indices].sum()),
                "semantic_agreeing_points": int(positive[offset, indices].sum())} for offset, camera in enumerate(cameras) if camera["split"] == "check"]
        path = output / "membership.npz"
        np.savez_compressed(path, **arrays)
        files.append(path)
        verify(job, receipt)
        return selections.seal(job, output, "result.json", {"prepared_sha256": receipt["sha256"], "masks_sha256": masks_record["sha256"],
            "structure_id": identifier, "method": METHOD, "created_at": manifests.utc_now(), "accepted_points": int(accepted.sum()),
            "views": view_reports, "planes": planes, "code_sha256": manifests.sha256_file(Path(__file__)),
            "status": "needs-review" if planes else "no-supported-wall", "scope": receipt["scope"],
            "check_scope": "Check semantic masks never enter point membership or plane fitting; shared SfM is not independent ground truth"}, files)
