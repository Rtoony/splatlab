"""Photo-linked SfM anchors for a deliberately selected, fitted support surface."""

import hashlib

import numpy as np
from PIL import Image

import artifact_manifest as manifests
import background_recovery as recovery
import glb_check
import reconstruction_evidence as evidence
import scene_revisions as scenes


def context(job, selected_slug, generation):
    pointer = scenes.active(job)
    if not pointer or pointer["generation"] != generation:
        raise evidence.EvidenceError("Active scene changed before support-surface inspection")
    revision = scenes.read_revision(job, pointer["revision_id"])
    if revision["source_fingerprint"] != scenes.source_fingerprint(job):
        raise evidence.EvidenceError("Refresh the changed capture baseline before support inspection")
    viewer = revision["state"]["viewer"]
    if viewer.get("units") != "meters" or viewer.get("calibration", {}).get("stale"):
        raise evidence.EvidenceError("Support inspection requires a currently calibrated metre world")
    entries = [entry for entry in viewer["elements"] if entry.get("provenance") != "authored"]
    selected = next((entry for entry in entries if entry["slug"] == selected_slug), None)
    if not selected:
        raise evidence.EvidenceError("Choose a visible captured object for support inspection")
    bounds = []
    for entry in [selected] + [entry for entry in entries if entry != selected]:
        reference = entry.get("files", {}).get("glb", "")
        if not isinstance(reference, str) or not reference.startswith("artifact:"):
            raise evidence.EvidenceError("Support inspection needs baked, revision-pinned captured meshes")
        geometry = glb_check.position_bounds(scenes.artifact(job, pointer["revision_id"], reference.removeprefix("artifact:")))
        if not geometry["identity_transforms"]:
            raise evidence.EvidenceError("Support inspection requires baked metre-world transforms")
        bounds.append(geometry["aabb"])
    return pointer, bounds


def fingerprint(source, pointer):
    payload = {"base": pointer, "calibration": source.calibration,
               "sources": {key: identity["sha256"] for key, identity in source.sources.items()}}
    return hashlib.sha256(scenes.canonical_bytes(payload)).hexdigest()


def linked_pixel(source, index, image_id):
    record = source.records[index]
    camera = source.cameras.get(image_id)
    linked = record["track"][record["track"][:, 0] == image_id]
    if not camera or len(linked) != 1 or record["error"] > 2 or len(record["track"]) < 3:
        raise evidence.EvidenceError("Anchor needs a unique photo observation and at least three low-error tracks")
    point_index = int(linked[0, 1])
    if not source.images:
        raise evidence.EvidenceError("Reload retained image observations before fitting anchors")
    observations = source.images[image_id]["observations"]
    if not 0 <= point_index < len(observations) or observations[point_index]["point_id"] != record["id"]:
        raise evidence.EvidenceError("Anchor track and source photograph disagree")
    observed = observations[point_index]
    pixel = np.array([observed["x"], observed["y"]]) * camera["observation_scale"]
    projected, depths = evidence.project(source.points[index:index + 1], camera)
    residual = float(np.linalg.norm(projected[0] - pixel))
    if (depths[0] <= 0 or not np.isfinite(pixel).all() or residual > 3
            or not ((pixel >= 0) & (pixel < [camera["width"], camera["height"]])).all()):
        raise evidence.EvidenceError("Anchor fails source-photo reprojection verification")
    return pixel, residual


def regional_indices(source, bounds):
    lower, upper = np.asarray(bounds[0]["min"]), np.asarray(bounds[0]["max"])
    if np.max((upper - lower)[[0, 2]]) > 4:
        raise evidence.EvidenceError("Select an object smaller than four metres for support recovery")
    quality = np.array([record["error"] <= 2 and len(record["track"]) >= 3 for record in source.records])
    nearby = ((source.points[:, [0, 2]] >= lower[[0, 2]] - 2.5)
              & (source.points[:, [0, 2]] <= upper[[0, 2]] + 2.5)).all(axis=1)
    nearby &= (source.points[:, 1] >= lower[1] - .8) & (source.points[:, 1] <= lower[1] + .1)
    for box in bounds:
        nearby &= ~((source.points >= np.asarray(box["min"]) - .02) & (source.points <= np.asarray(box["max"]) + .02)).all(axis=1)
    return np.flatnonzero(quality & nearby)


def observation_mask(source, camera):
    if not camera.get("mask_key"):
        return None
    path = evidence.input_path(source.job, camera["mask_key"], source.sources)
    with Image.open(path) as opened:
        if opened.size != (camera["width"], camera["height"]):
            raise evidence.EvidenceError("Anchor mask dimensions differ from the solved photo")
        return np.asarray(opened.convert("L"))


def inspect(job, selected_slug, generation, image_id=None):
    with recovery.worker_slot(wait_seconds=3):
        pointer, bounds = context(job, selected_slug, generation)
        source = evidence.load(job, retain_image_observations=True)
        digest = fingerprint(source, pointer)
        indices = regional_indices(source, bounds)
        counts = {identifier: 0 for identifier in source.cameras}
        for index in indices:
            for identifier in np.unique(source.records[index]["track"][:, 0]):
                if identifier in counts:
                    counts[identifier] += 1
        if image_id is None:
            image_id = max(counts, key=counts.get)
        if image_id not in source.cameras:
            raise evidence.EvidenceError("Photograph is not part of this solved capture")
        camera = source.cameras[image_id]
        photo = evidence.input_path(job, camera["image_key"], source.sources)
        with Image.open(photo) as opened:
            if opened.size != (camera["width"], camera["height"]):
                raise evidence.EvidenceError("Source photo dimensions do not match solved intrinsics")
        mask = observation_mask(source, camera)
        features = []
        for index in indices:
            if image_id not in source.records[index]["track"][:, 0]:
                continue
            try:
                pixel, residual = linked_pixel(source, index, image_id)
            except evidence.EvidenceError:
                continue
            if mask is not None and mask[int(pixel[1]), int(pixel[0])] < 128:
                continue
            features.append({"point_id": str(source.records[index]["id"]), "pixel": pixel.tolist(),
                             "world": source.points[index].tolist(), "reprojection_px": residual})
        if len(features) > 6000:
            features = [features[index] for index in np.linspace(0, len(features) - 1, 6000).astype(int)]
        evidence.verify_sources(job, source.sources, source.calibration)
        if scenes.active(job) != pointer:
            raise evidence.EvidenceError("Active scene changed during support inspection")
        return {"base": pointer, "evidence_sha256": digest, "image_id": image_id,
                "photo_sha256": source.sources[camera["image_key"]]["sha256"],
                "width": camera["width"], "height": camera["height"], "features": features,
                "cameras": [{"image_id": identifier, "name": value["name"], "regional_features": counts[identifier]}
                            for identifier, value in sorted(source.cameras.items())],
                "scope": "tracked source-photo observations, not semantic floor labels or independent measurements"}


def photo(job, selected_slug, generation, image_id, photo_sha256):
    with recovery.worker_slot(wait_seconds=3):
        pointer, _ = context(job, selected_slug, generation)
        source = evidence.load(job)
        if image_id not in source.cameras:
            raise evidence.EvidenceError("Photograph is not part of this solved capture")
        path = evidence.input_path(job, source.cameras[image_id]["image_key"], source.sources)
        if manifests.sha256_file(path) != photo_sha256 or scenes.active(job) != pointer:
            raise evidence.EvidenceError("Support photo changed; reload the anchor observations")
        evidence.verify_sources(job, source.sources, source.calibration)
        return path


def fit(source, bounds, anchors, pointer):
    if anchors.get("evidence_sha256") != fingerprint(source, pointer):
        raise evidence.EvidenceError("Support observations are stale; inspect the current capture again")
    picks = anchors.get("points", [])
    if not 3 <= len(picks) <= 6 or len({pick["point_id"] for pick in picks}) != len(picks):
        raise evidence.EvidenceError("Choose three to six distinct tracked support points")
    indices = regional_indices(source, bounds)
    eligible = {str(source.records[index]["id"]): index for index in indices}
    selected, verified = [], []
    for pick in picks:
        index = eligible.get(pick["point_id"])
        if index is None:
            raise evidence.EvidenceError("Support anchor is outside the eligible region or belongs to a captured object")
        image_id = pick["image_id"]
        pixel, residual = linked_pixel(source, index, image_id)
        camera = source.cameras[image_id]
        mask = observation_mask(source, camera)
        if mask is not None and mask[int(pixel[1]), int(pixel[0])] < 128:
            raise evidence.EvidenceError("Anchor is excluded by the source observation mask")
        path = evidence.input_path(source.job, camera["image_key"], source.sources)
        if manifests.sha256_file(path) != pick["photo_sha256"]:
            raise evidence.EvidenceError("Anchor photograph changed after inspection")
        selected.append(source.points[index])
        verified.append({**pick, "pixel": pixel.tolist(), "world": source.points[index].tolist(), "reprojection_px": residual})
    selected = np.asarray(selected)
    separations = np.linalg.norm(selected[:, None] - selected[None, :], axis=2)
    if np.any(separations[np.triu_indices(len(selected), k=1)] < .02):
        raise evidence.EvidenceError("Support anchors must be separate features at least two centimetres apart")
    center = selected.mean(0)
    _, singular, axes = np.linalg.svd(selected - center, full_matrices=False)
    if singular[1] < .05 or singular[1] < singular[0] * .08:
        raise evidence.EvidenceError("Spread support anchors across a non-collinear surface region")
    normal = axes[-1]
    if normal[1] < 0:
        normal *= -1
    anchor_residual = np.abs((selected - center) @ normal)
    if normal[1] < .94 or anchor_residual.max() > .015:
        raise evidence.EvidenceError("Selected anchors do not agree on an approximately horizontal support plane")
    distances = np.abs((source.points[indices] - center) @ normal)
    support = indices[distances < .025]
    if len(support) < 60:
        raise evidence.EvidenceError("The chosen support plane lacks sixty nearby tracked features")
    rms = float(np.sqrt(np.mean(distances[distances < .025] ** 2)))
    if rms > .02:
        raise evidence.EvidenceError("The chosen support plane exceeds the two-centimetre fit budget")
    right = np.cross(normal, [0., 0., 1.])
    right /= np.linalg.norm(right)
    forward = np.cross(normal, right)
    lower, upper = np.asarray(bounds[0]["min"]), np.asarray(bounds[0]["max"])
    target = (lower + upper) / 2
    target_center = target - normal * np.dot(target - center, normal)
    corners = np.array([[axis_x, axis_y, axis_z] for axis_x in (lower[0], upper[0])
                        for axis_y in (lower[1], upper[1]) for axis_z in (lower[2], upper[2])])
    planar = np.stack([(corners - target_center) @ right, (corners - target_center) @ forward], axis=1)
    return {"center": target_center, "normal": normal, "right": right, "forward": forward,
            "lower_uv": planar.min(0) - .2, "upper_uv": planar.max(0) + .2, "support_indices": support,
            "rms_m": rms, "regional_features": len(indices),
            "selection": {"method": "photo-linked-sfm-anchors", "anchors": verified,
                          "maximum_anchor_residual_m": float(anchor_residual.max()),
                          "anchor_to_target_distance_m": float(np.linalg.norm(target_center - center)),
                          "semantic_class": "unclassified support surface", "plane_refit": False,
                          "scope": "anchor-defined extrapolation; photo choice does not independently verify hidden geometry"}}
