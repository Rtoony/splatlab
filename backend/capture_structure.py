"""Source-tracked structural evidence in an explicitly unregistered reconstruction frame."""

from collections import defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

import artifact_manifest as manifests
from fisheye_reference import project_fisheye, read_cameras
from heldout_localization import read_tracks
from reference_delivery import artifact_path, verify_reference
from rig_diagnostics import read_poses
from mesh.slugify import slug


PROMPTS = ("building exterior wall", "pavement", "sky", "vegetation", "vehicle", "window", "garage door")
SCHEMA = "dev.splatlab.capture-structure/v1"


def image_identifiers(path):
    identifiers = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.split(maxsplit=9)
            if len(values) != 10 or values[9].strip() in identifiers:
                raise ValueError("Malformed or duplicate training image")
            identifiers[values[9].strip()] = int(values[0])
            if next(handle, None) is None:
                raise ValueError("Missing training observations")
    if len(set(identifiers.values())) != len(identifiers):
        raise ValueError("Duplicate training image identifier")
    return identifiers


def read_points(path):
    records = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.split()
            if len(values) < 8 or (len(values) - 8) % 2:
                raise ValueError("Malformed sparse point track")
            track = np.asarray([int(value) for value in values[8:]], dtype=np.int64).reshape(-1, 2)
            record = {"id": int(values[0]), "xyz": [float(value) for value in values[1:4]],
                      "rgb": [int(value) for value in values[4:7]], "error": float(values[7]), "track": track}
            if (not np.isfinite([*record["xyz"], record["error"]]).all() or record["error"] < 0
                    or any(not 0 <= value <= 255 for value in record["rgb"])
                    or np.any(track < 0)):
                raise ValueError("Invalid sparse point evidence")
            record["unambiguous_track"] = len(set(track[:, 0])) == len(track)
            records.append(record)
    if not 1 <= len(records) <= 100000 or len({record["id"] for record in records}) != len(records):
        raise ValueError("Invalid or unbounded sparse point set")
    return sorted(records, key=lambda record: record["id"])


def project_points(points, frame, intrinsics):
    matrix = np.asarray(frame["transform_matrix"], dtype=float)
    local = (points - matrix[:3, 3]) @ matrix[:3, :3]
    depth = -local[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-12, depth, 1.)
    pixels = np.stack([intrinsics["fl_x"] * local[:, 0] / safe_depth + intrinsics["cx"],
                       -intrinsics["fl_y"] * local[:, 1] / safe_depth + intrinsics["cy"]], axis=-1)
    valid = (depth > 0) & np.isfinite(pixels).all(axis=-1)
    valid &= (pixels >= 0).all(axis=-1) & (pixels < [intrinsics["w"], intrinsics["h"]]).all(axis=-1)
    return pixels, valid


def prepare(reference, sfm, output):
    reference, sfm, output = reference.resolve(), sfm.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(reference) or output.is_relative_to(sfm):
        raise ValueError("Choose a new structural study outside frozen sources")
    cameras, source, reference_snapshot = verify_reference(reference)
    snapshot = {reference / name: digest for name, digest in reference_snapshot.items()}
    model = sfm / "sparse/0"
    for name in ("images.txt", "points3D.txt", "cameras.txt"):
        snapshot[model / name] = source["source_hashes"]["model/" + name]
        if manifests.sha256_file(model / name) != snapshot[model / name]:
            raise ValueError("Frozen source model changed")
    records = read_points(model / "points3D.txt")
    points = np.asarray([record["xyz"] for record in records])
    identifiers = image_identifiers(model / "images.txt")
    tracks, poses, raw_cameras = read_tracks(model / "images.txt"), read_poses(model / "images.txt"), read_cameras(model / "cameras.txt")
    selected = [frame for frame in cameras["frames"] if frame["split"] == "train" and frame["virtual_yaw_deg"] == 0]
    if not 4 <= len(selected) <= 48 or len({frame["source_image"] for frame in selected}) != len(selected):
        raise ValueError("Choose one central crop per distinct physical training image")
    lookup = {record["id"]: index for index, record in enumerate(records)}
    observed = np.zeros((len(selected), len(records)), dtype=bool)
    projected = np.zeros((len(selected), len(records), 2))
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    views = []
    for ordinal, frame in enumerate(selected):
        name = frame["source_image"]
        image_id, pose, track = identifiers[name], poses[name], tracks[name]
        raw = raw_cameras[pose["camera_id"]]
        raw_pixels, in_cone = project_fisheye(points @ pose["rotation"].T + pose["translation"], raw["params"])
        projected[ordinal], in_image = project_points(points, frame, cameras)
        residuals = []
        for feature_index, point_id in enumerate(track["point_ids"]):
            index = lookup.get(int(point_id))
            if index is None:
                continue
            record = records[index]
            if not record["unambiguous_track"]:
                continue
            linked = record["track"][record["track"][:, 0] == image_id]
            if len(linked) != 1 or linked[0, 1] != feature_index:
                raise ValueError("Image/point feature-track identity disagrees")
            residual = float(np.linalg.norm(raw_pixels[index] - track["pixels"][feature_index]))
            if record["error"] <= 2 and len(record["track"]) >= 3 and in_cone[index] and in_image[index] and residual <= 3:
                observed[ordinal, index] = True
                residuals.append(residual)
        photo = f"frames/cam_{ordinal:03d}.png"
        shutil.copyfile(reference / frame["file_path"], output / photo)
        views.append({**frame, "image_id": image_id, "ordinal": ordinal, "photo": photo,
                      "tracked_supported_points": int(observed[ordinal].sum()),
                      "maximum_raw_reprojection_px": max(residuals, default=None)})
    np.savez_compressed(output / "tracks.npz", points=points,
                        colors=np.asarray([record["rgb"] for record in records], dtype=np.uint8),
                        point_ids=np.asarray([record["id"] for record in records], dtype=np.int64), observed=observed, pixels=projected)
    manifests.atomic_write_json(output / "views.json", {"cam_indices": list(range(len(views)))})
    manifests.atomic_write_json(output / "things.json", list(PROMPTS))
    for path, digest in snapshot.items():
        if manifests.sha256_file(path) != digest:
            raise ValueError("Source changed during structural preparation")
    receipt = {"schema": SCHEMA, "status": "prepared-needs-masks", "source_hashes": {str(path): digest for path, digest in snapshot.items()},
               "started_at": manifests.utc_now(), "views": views, "prompts": list(PROMPTS),
               "intrinsics": {key: cameras[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")},
               "sparse_points": len(records), "source_timestamps": len({frame["source_group"] for frame in selected}),
               "excluded_ambiguous_track_point_ids": [record["id"] for record in records if not record["unambiguous_track"]],
               "geometry_refined": False, "metric_scale": "unknown", "registration": None, "owner_accepted": False,
               "recipe": {"semantic_threshold": .5, "semantic_border_erosion_px": 3,
                          "minimum_positive_timestamps": 2, "minimum_angle_degrees": 3,
                          "point_holdout_rule": "point ID modulo 5 equals zero reserved from plane fitting",
                          "plane_tolerance_camera_span_fraction": .005},
               "scope": "Semantic predictions on original training photos anchored to verified SfM tracks; not semantic or structural ground truth",
               "files": {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.rglob("*") if path.is_file()}}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt


def verify_study(output):
    receipt = json.loads((output / "receipt.json").read_text())
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "prepared-needs-masks" or receipt.get("registration") is not None:
        raise ValueError("Expected an independent prepared structural study")
    for name, digest in receipt["files"].items():
        if manifests.sha256_file(artifact_path(output, name)) != digest:
            raise ValueError("Prepared structural evidence changed")
    return receipt


def semantic_union(path, prompt, shape):
    with np.load(path, allow_pickle=False) as stored:
        masks, scores = stored["masks"], stored["scores"]
        if (masks.ndim != 3 or masks.dtype != bool or masks.shape[1:] != tuple(shape) or len(masks) > 128
                or scores.shape != (len(masks),) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1)
                or stored["prompt"].shape != () or stored["prompt"].item() != prompt):
            raise ValueError("Semantic masks do not match their photo and prompt")
        return masks[scores >= .5].any(axis=0)


def erode_mask(mask, radius=3):
    if mask.ndim != 2 or mask.dtype != bool or type(radius) is not int or not 0 <= radius <= 8:
        raise ValueError("Need a boolean image mask and bounded integer erosion")
    padded = np.pad(mask, radius, constant_values=False)
    return np.lib.stride_tricks.sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1)).all(axis=(-1, -2))


def temporal_membership(points, views, observed, positive):
    if (observed.shape != positive.shape or observed.shape != (len(views), len(points))
            or observed.dtype != bool or positive.dtype != bool or np.any(positive & ~observed)):
        raise ValueError("Structural votes must belong to actual tracked observations")
    groups = defaultdict(list)
    for index, view in enumerate(views):
        groups[view["source_group"]].append(index)
    positives, negatives = [], []
    for members in groups.values():
        has_positive = positive[members].any(axis=0)
        has_negative = (observed[members] & ~positive[members]).any(axis=0)
        positives.append(has_positive & ~has_negative)
        negatives.append(has_negative)
    positive_count, negative_count = np.sum(positives, axis=0), np.sum(negatives, axis=0)
    angular = np.zeros(len(points), dtype=bool)
    centers = np.array([np.asarray(view["transform_matrix"])[:3, 3] for view in views])
    for first, first_view in enumerate(views):
        for second in range(first + 1, len(views)):
            if first_view["source_group"] == views[second]["source_group"]:
                continue
            first_direction, second_direction = centers[first] - points, centers[second] - points
            norms = np.linalg.norm(first_direction, axis=1) * np.linalg.norm(second_direction, axis=1)
            cosine = np.sum(first_direction * second_direction, axis=1) / np.maximum(norms, 1e-12)
            angular |= positive[first] & positive[second] & (norms > 1e-12) & (cosine <= np.cos(np.deg2rad(3)))
    return (positive_count >= 2) & (positive_count >= 2 * negative_count) & angular, positive_count, negative_count


def fit_planes(points, fit_indices, check_indices, camera_span, maximum=6):
    points = np.asarray(points, dtype=float)
    fit_indices, check_indices = np.asarray(fit_indices), np.asarray(check_indices)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not np.isfinite(camera_span) or camera_span <= 0 or type(maximum) is not int or not 1 <= maximum <= 6):
        raise ValueError("Need finite points and a positive observed camera span")
    for indices in (fit_indices, check_indices):
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(np.unique(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(points))):
            raise ValueError("Invalid structural point membership")
    if np.intersect1d(fit_indices, check_indices).size:
        raise ValueError("Plane fitting and point checks must not overlap")
    tolerance = camera_span * .005
    random = np.random.default_rng(20260909)
    remaining, planes = fit_indices.copy(), []
    for _attempt in range(maximum):
        if len(remaining) < 24:
            break
        sample = points[remaining]
        best = np.empty(0, dtype=int)
        for _trial in range(512):
            first, second, third = sample[random.choice(len(sample), 3, replace=False)]
            normal = np.cross(second - first, third - first)
            length = np.linalg.norm(normal)
            if length < 4 * tolerance ** 2:
                continue
            normal /= length
            selected = np.flatnonzero(np.abs((sample - first) @ normal) <= tolerance)
            if len(selected) > len(best):
                best = selected
        if len(best) < 24:
            break
        selected = remaining[best]
        center = points[selected].mean(axis=0)
        _, singular, axes = np.linalg.svd(points[selected] - center, full_matrices=False)
        normal = axes[-1]
        if np.dot(normal, np.ones(3)) < 0:
            normal = -normal
        right = axes[0]
        up = np.cross(normal, right)
        relative = points[selected] - center
        residuals = np.abs(relative @ normal)
        selected = selected[residuals <= tolerance]
        remaining = remaining[~np.isin(remaining, remaining[best])]
        if len(selected) < 24 or singular[1] < singular[0] * .08:
            continue
        relative = points[selected] - center
        rms = float(np.sqrt(np.mean((relative @ normal) ** 2)))
        uv = np.column_stack((relative @ right, relative @ up))
        lower, upper = uv.min(axis=0), uv.max(axis=0)
        if np.any(upper - lower < tolerance * 8) or rms > tolerance * .75:
            continue
        check_relative = points[check_indices] - center
        check_uv = np.column_stack((check_relative @ right, check_relative @ up))
        footprint = ((check_uv >= lower) & (check_uv <= upper)).all(axis=1)
        distances = np.abs(check_relative[footprint] @ normal)
        near = distances <= tolerance * 1.5
        planes.append({"center": center.tolist(), "normal": normal.tolist(), "right": right.tolist(), "up": up.tolist(),
                       "lower_uv": lower.tolist(), "upper_uv": upper.tolist(), "point_indices": selected.tolist(),
                       "fit_points": len(selected), "fit_rms_arbitrary_units": rms, "tolerance_arbitrary_units": tolerance,
                       "check_points_in_projected_footprint": int(footprint.sum()), "check_points_near_plane": int(near.sum()),
                       "check_near_fraction": float(near.mean()) if len(near) else None,
                       "check_median_distance_arbitrary_units": float(np.median(distances)) if len(distances) else None,
                       "check_p90_distance_arbitrary_units": float(np.quantile(distances, .9)) if len(distances) else None,
                       "status": "source-supported-plane-candidate" if near.sum() >= 5 and near.mean() >= .75 else "insufficient-independent-point-support",
                       "scope": "Plane fitted to semantic source tracks; bounds span gaps, do not imply a wall solid, measured up or visibility"})
    return planes


def analyze(output):
    output = output.resolve()
    receipt = verify_study(output)
    masks_receipt = json.loads((output / "masks.json").read_text())
    if masks_receipt.get("status") != "inferred-needs-visual-review" or (output / "analysis.json").exists():
        raise ValueError("Require complete inferred masks and a new analysis")
    snapshot = {output / "receipt.json": manifests.sha256_file(output / "receipt.json"),
                output / "masks.json": manifests.sha256_file(output / "masks.json")}
    for name, digest in masks_receipt["files"].items():
        path = artifact_path(output, name)
        if manifests.sha256_file(path) != digest:
            raise ValueError("Semantic inference artifacts changed")
        snapshot[path] = digest
    started = time.monotonic()
    with np.load(output / "tracks.npz", allow_pickle=False) as stored:
        points, colors, identifiers, observed, pixels = (stored[name] for name in ("points", "colors", "point_ids", "observed", "pixels"))
    positive = {label: np.zeros_like(observed) for label in ("facade", "pavement")}
    (output / "classified-masks").mkdir()
    (output / "overlays").mkdir()
    views, tiles = [], []
    shape = receipt["intrinsics"]["h"], receipt["intrinsics"]["w"]
    for ordinal, view in enumerate(receipt["views"]):
        predictions = {prompt: semantic_union(output / "masks" / slug(prompt) / f"cam_{ordinal:03d}.npz", prompt, shape)
                       for prompt in receipt["prompts"]}
        nuisance = predictions["sky"] | predictions["vegetation"] | predictions["vehicle"]
        wall = predictions["building exterior wall"] | predictions.get("garage door", np.zeros(shape, dtype=bool))
        pavement = predictions["pavement"] & ~nuisance
        facade = (wall | predictions["window"]) & ~nuisance
        masks = {"facade": erode_mask(facade), "pavement": erode_mask(pavement)}
        surface = erode_mask((wall | pavement) & ~nuisance & ~predictions["window"])
        np.savez_compressed(output / "classified-masks" / f"cam_{ordinal:03d}.npz", static_surface=surface,
                            facade_landmarks=masks["facade"], pavement=masks["pavement"], excluded=nuisance | predictions["window"])
        indices = np.flatnonzero(observed[ordinal])
        sample = np.floor(pixels[ordinal, indices]).astype(int)
        for label, mask in masks.items():
            positive[label][ordinal, indices] = mask[sample[:, 1], sample[:, 0]]
        with Image.open(output / view["photo"]) as original:
            rgb = np.asarray(original.convert("RGB"))
        overlay = np.rint(rgb * .35).astype(np.uint8)
        overlay[surface] = np.rint(rgb[surface] * .7 + np.array([0, 160, 110]) * .3).astype(np.uint8)
        image = Image.fromarray(overlay)
        draw = ImageDraw.Draw(image)
        supported_here = positive["facade"][ordinal] | positive["pavement"][ordinal]
        for pixel in pixels[ordinal, supported_here]:
            horizontal, vertical = pixel
            draw.ellipse((horizontal - 2, vertical - 2, horizontal + 2, vertical + 2), fill="#ffd04b")
        image.save(output / "overlays" / f"cam_{ordinal:03d}.png")
        tiles.append(image.resize((256, 256)))
        views.append({"ordinal": ordinal, "source_image": view["source_image"], "group": view["source_group"],
                      "static_surface_pixels": int(surface.sum()), "fraction": float(surface.mean()),
                      "facade_tracked_points": int(positive["facade"][ordinal].sum()),
                      "pavement_tracked_points": int(positive["pavement"][ordinal].sum()),
                      "predicted_pixels": {prompt: int(mask.sum()) for prompt, mask in predictions.items()}})
    sheet = Image.new("RGB", (6 * 256, int(np.ceil(len(tiles) / 6)) * 282), "#101923")
    draw = ImageDraw.Draw(sheet)
    for index, tile in enumerate(tiles):
        left, top = index % 6 * 256, index // 6 * 282
        sheet.paste(tile, (left, top))
        draw.text((left + 3, top + 258), f"{index:02d} {receipt['views'][index]['source_image']}", fill="white")
    sheet.save(output / "mask-contact.png")
    centers = np.array([np.asarray(view["transform_matrix"])[:3, 3] for view in receipt["views"]])
    camera_span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
    reserved = identifiers % 5 == 0
    classes = {}
    membership = {"point_ids": identifiers, "points": points, "colors": colors, "observed": observed, "plane_check_reserved": reserved}
    for label in positive:
        supported, counts, negatives = temporal_membership(points, receipt["views"], observed, positive[label])
        fit = np.flatnonzero(supported & ~reserved)
        check = np.flatnonzero(supported & reserved)
        classes[label] = {"supported_points": int(supported.sum()), "fit_points": len(fit), "check_points": len(check),
                          "planes": fit_planes(points, fit, check, camera_span)}
        membership.update({label + "_supported": supported, label + "_positive": positive[label],
                           label + "_positive_timestamps": counts, label + "_negative_timestamps": negatives})
    np.savez_compressed(output / "membership.npz", **membership)
    for path, digest in snapshot.items():
        if manifests.sha256_file(path) != digest:
            raise ValueError("Frozen mask or source receipt changed during analysis")
    generated = [output / "mask-contact.png", output / "membership.npz", *sorted((output / "classified-masks").iterdir()), *sorted((output / "overlays").iterdir())]
    result = {"schema": SCHEMA, "status": "analyzed-needs-visual-review", "source_receipt_sha256": snapshot[output / "receipt.json"],
              "mask_receipt_sha256": snapshot[output / "masks.json"], "source_hashes": {str(path): digest for path, digest in snapshot.items()},
              "camera_span_arbitrary_units": camera_span, "classes": classes, "views": views,
              "excluded_ambiguous_tracks": len(receipt["excluded_ambiguous_track_point_ids"]),
              "semantic_policy": "Facade landmarks include windows and any explicitly inferred garage doors; surface excludes windows, sky, vegetation and vehicles with a three-pixel erosion",
              "metric_scale": "unknown", "registration": None, "owner_accepted": False,
              "point_check_scope": "Withheld from plane fitting, but reconstructed in the same original training SfM; not independent survey truth",
              "new_solid_geometry": False, "elapsed_seconds": time.monotonic() - started,
              "files": {str(path.relative_to(output)): manifests.sha256_file(path) for path in generated}}
    manifests.atomic_write_json(output / "analysis.json", result)
    return result
