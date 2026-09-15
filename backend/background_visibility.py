"""Revision-bound inferred-depth qualification for photo-backed support recovery."""

import hashlib
from io import BytesIO
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image

import artifact_manifest as manifests
import background_recovery as recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as selections

IDENTIFIER = re.compile(r"visibility_[a-f0-9]{24}\Z")
METHOD = "captured-gaussian-plane-depth-consistency/v1"
POLICY = {"minimum_alpha": .85, "absolute_depth_tolerance_m": .025, "relative_depth_tolerance": .01,
          "pixel_neighbors": 4, "maximum_render_dimension": 960}


def directory(job, identifier):
    if not IDENTIFIER.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid background visibility identifier")
    output = scenes.root(job) / "background-visibility" / identifier
    if output.resolve() != output.absolute():
        raise evidence.EvidenceError("Background visibility directory cannot be a symlink")
    return output


def read(job, identifier, result=False):
    value = manifests.read_json(directory(job, identifier) / ("result.json" if result else "receipt.json"))
    if not value or value.get("sha256") != hashlib.sha256(scenes.canonical_bytes({key: item for key, item in value.items() if key != "sha256"})).hexdigest():
        raise evidence.EvidenceError("Background visibility receipt is missing or corrupt")
    if result and value.get("prepared_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Background visibility result belongs to different inputs")
    return value


def artifact(job, identifier, name, result=False):
    record = read(job, identifier, result)["artifacts"].get(name)
    if not record:
        raise evidence.EvidenceError("Artifact does not belong to the visibility study")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Background visibility artifact is missing or corrupt")
    return path


def verify(job, receipt):
    original = recovery.read(job, receipt["recovery_id"])
    if original["sha256"] != receipt["recovery_sha256"] or scenes.active(job) != receipt["base"] or original["base"] != receipt["base"]:
        raise evidence.EvidenceError("Background visibility belongs to an older recovery or active revision")
    recovery.verify_receipt_sources(job, receipt, checksums=True)
    revision = scenes.read_revision(job, receipt["base"]["revision_id"])
    if revision["source_fingerprint"] != scenes.source_fingerprint(job) or revision["artifacts"].get("_preview/splat.ply") != receipt["splat_artifact"]:
        raise evidence.EvidenceError("Captured scene changed before visibility qualification")
    for name, record in original["artifacts"].items():
        if manifests.sha256_file(recovery.artifact(job, receipt["recovery_id"], name)) != record["sha256"]:
            raise evidence.EvidenceError("Original recovery artifact changed")
    return original


def grid_points(plane, size):
    horizontal, vertical = np.meshgrid((np.arange(size) + .5) / size, (np.arange(size) + .5) / size)
    uv = np.asarray(plane["lower_uv"]) + np.stack([horizontal, vertical], axis=-1).reshape(-1, 2) * (np.asarray(plane["upper_uv"]) - plane["lower_uv"])
    return np.asarray(plane["center"]) + uv[:, :1] * plane["right"] + uv[:, 1:] * plane["forward"]


def verified_grid(job, receipt):
    with np.load(artifact(job, receipt["visibility_id"], "grid.npz"), allow_pickle=False) as grid:
        points = grid["points_world"]
    expected = grid_points(receipt["plane"], receipt["texture_size"])
    if not np.isfinite(points).all() or not np.array_equal(points, expected):
        raise evidence.EvidenceError("Visibility grid differs from the fitted texture cell centers")
    return points


def qualify_samples(alpha, depth_error, plane_depth, in_frame):
    if (alpha.ndim != 1 or any(value.shape != alpha.shape for value in (depth_error, plane_depth, in_frame))
            or in_frame.dtype != bool or any(value.dtype.kind != "f" for value in (alpha, depth_error, plane_depth))
            or not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1.001)
            or not np.isfinite(plane_depth).all() or np.any(depth_error < 0)
            or np.isnan(depth_error).any() or not np.isfinite(depth_error[in_frame]).all()
            or np.any(plane_depth[in_frame] <= 0)):
        raise evidence.EvidenceError("Visibility samples disagree with their finite per-cell depth contract")
    tolerance = POLICY["absolute_depth_tolerance_m"] + POLICY["relative_depth_tolerance"] * plane_depth
    return in_frame & (alpha >= POLICY["minimum_alpha"]) & (depth_error <= tolerance)


def sample_render(alpha, depth_m, pixels, plane_depth):
    if (alpha.ndim != 2 or plane_depth.ndim != 1 or depth_m.shape != alpha.shape or pixels.shape != (len(plane_depth), 2)
            or not np.isfinite(alpha).all() or not np.isfinite(depth_m).all() or np.any(depth_m < 0)
            or np.any(alpha < 0) or np.any(alpha > 1.001) or not np.isfinite(plane_depth).all()):
        raise evidence.EvidenceError("Rendered visibility depth or alpha is invalid")
    height, width = alpha.shape
    in_frame = (np.isfinite(pixels).all(axis=1) & (plane_depth > 0)
                & (pixels >= 0).all(axis=1) & (pixels < [width - 1, height - 1]).all(axis=1))
    indices = np.flatnonzero(in_frame)
    nearest = np.floor(pixels[indices]).astype(np.int64)
    minimum_alpha = np.zeros(len(plane_depth), dtype=np.float32)
    maximum_error = np.full(len(plane_depth), np.inf, dtype=np.float32)
    alpha_samples, errors = [], []
    for offset in ([0, 0], [1, 0], [0, 1], [1, 1]):
        pixel = nearest + offset
        alpha_samples.append(alpha[pixel[:, 1], pixel[:, 0]])
        errors.append(np.abs(depth_m[pixel[:, 1], pixel[:, 0]] - plane_depth[indices]))
    minimum_alpha[indices] = np.min(alpha_samples, axis=0)
    maximum_error[indices] = np.max(errors, axis=0)
    return {"minimum_alpha": minimum_alpha, "maximum_depth_error_m": maximum_error,
            "plane_depth_m": plane_depth, "in_frame": in_frame,
            "qualified": qualify_samples(minimum_alpha, maximum_error, plane_depth, in_frame)}


def prepare(job, recovery_id, expected_generation):
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        original = recovery.read(job, recovery_id)
        pointer = scenes.active(job)
        if original["base"] != pointer or pointer["generation"] != expected_generation:
            raise evidence.EvidenceError("Prepare visibility from a current recovery")
        if original.get("visibility_review"):
            raise evidence.EvidenceError("Prepare another original recovery instead of repeatedly filtering a qualified result")
        recovery.verify_receipt_sources(job, original, checksums=True)
        source = evidence.load(job)
        sources = dict(original["sources"])
        selections.require_fixed_cameras(job, sources)
        revision = scenes.read_revision(job, pointer["revision_id"])
        splat = scenes.artifact(job, pointer["revision_id"], "_preview/splat.ply")
        if splat.stat().st_size > evidence.MAX_FILE_BYTES:
            raise evidence.EvidenceError("Visibility splat exceeds the 512 MiB input budget")
        positions = selections.read_ply_xyz(splat)
        if not 0 < len(positions) <= selections.MAX_ROWS or not np.isfinite(positions).all():
            raise evidence.EvidenceError("Visibility requires a bounded finite captured splat")
        for name, record in original["artifacts"].items():
            if manifests.sha256_file(recovery.artifact(job, recovery_id, name)) != record["sha256"]:
                raise evidence.EvidenceError("Original recovery artifact changed")
        cameras = []
        for split, identifiers in (("fit", original["report"]["source_image_ids"]), ("check", original["report"]["appearance_check_image_ids"])):
            for image_id in identifiers:
                camera = source.cameras[image_id]
                scale = min(1, POLICY["maximum_render_dimension"] / max(camera["width"], camera["height"]))
                width, height = round(camera["width"] * scale), round(camera["height"] * scale)
                transform, parameters = selections.raw_camera(camera, original["calibration"]["meters_per_unit"], width, height)
                cameras.append({"image_id": image_id, "group": camera["group"], "split": split,
                    "width": width, "height": height, "raw_to_camera": transform, "parameters": parameters})
        if not 3 <= len(cameras) <= 28 or len({camera["image_id"] for camera in cameras}) != len(cameras):
            raise evidence.EvidenceError("Visibility requires unique bounded fit/check views")
        identifier = "visibility_" + uuid.uuid4().hex[:24]
        output = directory(job, identifier)
        output.mkdir(parents=True)
        np.savez_compressed(output / "grid.npz", points_world=grid_points(original["report"]["plane"], original["report"]["texture_size"]))
        value = {"schema": "dev.splatlab.background-visibility/v1", "visibility_id": identifier, "method": METHOD,
            "base": pointer, "recovery_id": recovery_id, "recovery_sha256": original["sha256"],
            "selected_slug": original["selected_slug"], "sources": sources, "calibration": original["calibration"],
            "splat_artifact": revision["artifacts"]["_preview/splat.ply"], "n_rows": len(positions),
            "texture_size": original["report"]["texture_size"], "plane": original["report"]["plane"],
            "cameras": cameras, "policy": dict(POLICY), "created_at": manifests.utc_now(),
            "scope": "Inferred captured-Gaussian depth consistency, not measured visibility or semantic floor recognition"}
        return selections.seal(job, output, "receipt.json", value, [output / "grid.npz"])


def qualification(job, identifier, pointer, selected_slug, size, plane):
    receipt = read(job, identifier)
    original = verify(job, receipt)
    verified_grid(job, receipt)
    result = read(job, identifier, True)
    if (receipt["base"] != pointer or receipt["selected_slug"] != selected_slug or receipt["texture_size"] != size
            or receipt["method"] != METHOD or receipt["policy"] != POLICY or result.get("method") != METHOD):
        raise evidence.EvidenceError("Visibility qualification does not match this recovery request")
    description = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in plane.items() if key != "support_indices"}
    if description != receipt["plane"]:
        raise evidence.EvidenceError("Visibility qualification belongs to a different fitted support plane")
    masks = {}
    expected = {f"camera-{camera['image_id']}.npz" for camera in receipt["cameras"]}
    if set(result["artifacts"]) != expected:
        raise evidence.EvidenceError("Visibility result is missing or duplicates camera evidence")
    for camera in receipt["cameras"]:
        with np.load(artifact(job, identifier, f"camera-{camera['image_id']}.npz", True), allow_pickle=False) as arrays:
            qualified = qualify_samples(arrays["minimum_alpha"], arrays["maximum_depth_error_m"], arrays["plane_depth_m"], arrays["in_frame"])
            if qualified.shape != (size * size,) or arrays["qualified"].dtype != bool or not np.array_equal(arrays["qualified"], qualified):
                raise evidence.EvidenceError("Stored visibility decisions disagree with their depth evidence")
            masks[camera["image_id"]] = qualified
    metadata = {"visibility_id": identifier, "prepared_sha256": receipt["sha256"], "result_sha256": result["sha256"],
        "parent_recovery_id": receipt["recovery_id"], "parent_recovery_sha256": original["sha256"],
        "method": METHOD, "policy": dict(POLICY), "parent_supported_fraction": original["report"]["supported_fraction"],
        "scope": receipt["scope"]}
    records = {f"visibility-{name}": record for name, record in result["artifacts"].items()}
    records["visibility-receipt.json"] = scenes.store_json(job, receipt)
    records["visibility-result.json"] = scenes.store_json(job, result)
    records["visibility-grid.npz"] = receipt["artifacts"]["grid.npz"]
    return masks, metadata, records


def derive(job, identifier, expected_generation):
    receipt = read(job, identifier)
    original = verify(job, receipt)
    return recovery.build(job, receipt["selected_slug"], expected_generation, receipt["texture_size"],
                          original.get("support_anchors"), visibility_id=identifier)


def compare_recovery(job, identifier, source, bounds, plane, masks, texture, pixels):
    receipt = read(job, identifier)
    original = verify(job, receipt)
    with np.load(recovery.artifact(job, receipt["recovery_id"], "observations.npz"), allow_pickle=False) as observations:
        previous = observations["accepted"].reshape(-1)
    with Image.open(recovery.artifact(job, receipt["recovery_id"], "atlas.png")) as opened:
        original_colors = np.asarray(opened.convert("RGBA"))[:, :, :3].reshape(-1, 3) / 255.
    with Image.open(BytesIO(texture)) as opened:
        new_colors = np.asarray(opened.convert("RGBA"))[:, :, :3].reshape(-1, 3) / 255.
    accepted = pixels["accepted"].reshape(-1)
    points = grid_points(plane, receipt["texture_size"])
    comparisons = []
    for image_id in original["report"]["appearance_check_image_ids"]:
        colors, valid = recovery.sample_view(source, image_id, plane, points, receipt["texture_size"], bounds)
        shared = previous & accepted & valid & masks[image_id]
        comparisons.append({"image_id": image_id, "shared_pixels": int(shared.sum()),
            "parent_mean_rgb_absolute_error": float(np.abs(original_colors[shared] - colors[shared]).mean()) if shared.any() else None,
            "qualified_mean_rgb_absolute_error": float(np.abs(new_colors[shared] - colors[shared]).mean()) if shared.any() else None})
    return {"parent_cells": int(previous.sum()), "qualified_cells": int(accepted.sum()),
        "retained_parent_cells": int(np.count_nonzero(previous & accepted)),
        "rejected_parent_cells": int(np.count_nonzero(previous & ~accepted)),
        "newly_supported_cells": int(np.count_nonzero(~previous & accepted)), "paired_appearance_checks": comparisons,
        "scope": "Same-pixel delivered-atlas comparison on check views passing the inferred-depth filter; changed coverage is reported separately, not an independent visibility or geometry benchmark"}
