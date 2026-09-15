#!/usr/bin/env python3
"""CPU-only review of retained placement leads; no geometry build or scene mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import architectural_edits as architecture
import architectural_volumes as volumes
import artifact_manifest as manifests
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as selections


def memberships(points: np.ndarray, bounds: list, padding: float = 0) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Preservation review requires finite three-dimensional points")
    inside = np.zeros(len(points), dtype=bool)
    for lower, upper in bounds:
        inside |= ((points >= np.asarray(lower) - padding) & (points <= np.asarray(upper) + padding)).all(axis=1)
    return inside


def review(spec: dict, receipt: dict, points: np.ndarray, document: dict, guards: np.ndarray) -> dict:
    coordinate = architecture.frame(spec)
    clip = volumes.contract(spec, coordinate, volumes.method(receipt["clip"]))
    protected = [item["slug"] for item in receipt["protected_elements"]]
    replaced = [item["slug"] for item in receipt["replaced_elements"]]
    local = architecture.local_coordinates(points, spec)
    conflicts = volumes.protected_row_conflicts(local, document, protected, clip)
    impact, rows = volumes.capture_impact(local, document, protected, replaced, clip)
    reserved = architecture.protected_volume_bounds(spec, architecture.geometry_method(receipt))
    reserved_members = memberships(local, reserved, .01)
    reserved_conflicts = {}
    for name in protected:
        selected = volumes._selection_rows(document, name, len(points))
        count = int(reserved_members[selected].sum())
        if count:
            reserved_conflicts[name] = count
    guard_local = architecture.local_coordinates(guards, spec)
    guarded = memberships(guard_local, [*volumes.regions(clip), *reserved], .1)
    return {"spec": spec, "named_clipped_center_conflicts": conflicts,
            "named_reserved_volume_center_conflicts": reserved_conflicts,
            "source_guard_conflicts": int(guarded.sum()), "capture_impact": impact,
            "unassigned_example_rows": rows["unassigned"][:12].tolist(),
            "passes_center_and_source_point_screen": not conflicts and not reserved_conflicts and not guarded.any(),
            "full_preservation_proven": False,
            "unverified": ["unnamed object boundaries outside the retained guard", "Gaussian spatial footprints",
                           "physical floor identity at the proposed start", "full geometry and fixed-body route",
                           "new candidate GPU walking and apply/reload/undo"]}


def source_views(source, start: list[float]) -> list[dict]:
    start = np.asarray(start, dtype=float)
    probes = np.stack([start, start + [.32, 0, 0], start + [0, 0, .32], start + [0, 2.02, 0]])
    candidates = []
    for identifier, camera in source.cameras.items():
        pixels, depth = evidence.project(probes, camera)
        bounds = np.array([camera["width"], camera["height"]])
        if not (depth[:3] > 0).all() or not np.isfinite(pixels[:3]).all() or not ((pixels[:3] >= 8) & (pixels[:3] < bounds - 8)).all():
            continue
        span = float(np.max(np.linalg.norm((pixels[1:3] - pixels[0]) / bounds, axis=1)))
        if span < .015:
            continue
        direction = np.asarray(camera["center"]) - start
        length = np.linalg.norm(direction)
        if length <= .01:
            continue
        candidates.append({"image_id": identifier, "source_key": camera["image_key"],
                           "floor_pixel": pixels[0].tolist(),
                           "head_pixel": pixels[3].tolist() if depth[3] > 0 and np.isfinite(pixels[3]).all() else None,
                           "head_in_frame": bool(depth[3] > 0 and np.isfinite(pixels[3]).all() and ((pixels[3] >= 8) & (pixels[3] < bounds - 8)).all()),
                           "image_dimensions": bounds.tolist(), "floor_radius_span_fraction": span,
                           "view_direction": (direction / length).tolist()})
    candidates.sort(key=lambda item: item["floor_radius_span_fraction"], reverse=True)
    selected = []
    for candidate in candidates:
        if any(np.dot(candidate["view_direction"], previous["view_direction"]) > .95 for previous in selected):
            continue
        image = evidence.input_path(source.job, candidate["source_key"], source.sources)
        candidate["source_sha256"] = manifests.sha256_file(image)
        selected.append(candidate)
        if len(selected) == 3:
            break
    return selected


def nearby_source_samples(source, start: list[float]) -> dict:
    start = np.asarray(start, dtype=float)
    offsets = source.points - start
    horizontal = np.linalg.norm(offsets[:, [0, 2]], axis=1)
    quality = np.array([record["error"] <= 2 and len(record["track"]) >= 3 for record in source.records])
    nearby = np.flatnonzero(quality & (horizontal <= .35) & (np.abs(offsets[:, 1]) <= .35))
    nearest = nearby[np.argsort(np.linalg.norm(offsets[nearby], axis=1))][:12]
    return {"radius_m": .35, "vertical_window_m": .35, "quality_points": len(nearby),
            "points_within_8cm_height": int((np.abs(offsets[nearby, 1]) <= .08).sum()),
            "height_offset_range_m": [float(offsets[nearby, 1].min()), float(offsets[nearby, 1].max())] if len(nearby) else None,
            "nearest_points": [{"point_id": source.records[index]["id"], "world": source.points[index].tolist(),
                                "horizontal_distance_m": float(horizontal[index]), "height_offset_m": float(offsets[index, 1]),
                                "track_length": len(source.records[index]["track"]), "error_px": source.records[index]["error"]}
                               for index in nearest],
            "physical_floor_proven": False,
            "scope": "Triangulated nearby observations, not a semantic floor mask or a complete capsule support test."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--guard-evidence", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-views", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output so existing review evidence stays intact")
    search = manifests.read_json(args.search)
    retained = manifests.read_json(args.guard_evidence)
    pointer = scenes.active(args.job)
    if not search or not retained or pointer != search.get("base") or pointer != retained.get("base"):
        raise ValueError("The retained candidate search and source guards must address the active scene exactly")
    if not 1 <= len(search["candidates"]) <= 40:
        raise ValueError("Review one through forty retained candidates")
    if manifests.sha256_file(args.guard_evidence) != search["source_evidence_sha256"]:
        raise ValueError("Retained source evidence changed")
    receipt = architecture.read(args.job, args.reference)
    architecture.verify(args.job, receipt)
    if receipt["sha256"] != search["reference_receipt_sha256"]:
        raise ValueError("Reference recipe differs from the candidate search")
    evidence.verify_sources(args.job, retained["sources"], retained["calibration"])
    revision = scenes.read_revision(args.job, pointer["revision_id"])
    record = revision["artifacts"]["_preview/langweb.ply"]
    source = scenes.artifact(args.job, pointer["revision_id"], "_preview/langweb.ply")
    if manifests.sha256_file(source) != record["sha256"]:
        raise ValueError("Pinned captured Gaussian artifact changed")
    points = selections.read_ply_xyz(source) @ evidence.Y_UP.T * receipt["calibration"]["meters_per_unit"]
    guards = np.asarray([point["world"] for point in retained["studies"]["existing-cut"]["source_points"]])
    document = revision["state"]["selections"]
    results = []
    for index, candidate in enumerate(search["candidates"]):
        result = review(candidate["spec"], receipt, points, document, guards)
        result.update(index=index, approach_start=candidate["approach_start"],
                      sampled_clearance_m=candidate["default_body_sample_clearance_m"])
        results.append(result)
    starts = []
    if args.source_views:
        reconstruction = evidence.load(args.job)
        for candidate in search["candidates"]:
            if any(np.allclose(candidate["approach_start"], start["world"], atol=1e-8, rtol=0) for start in starts):
                continue
            starts.append({"world": candidate["approach_start"], "body_height_m": 2.02,
                           "views": source_views(reconstruction, candidate["approach_start"]),
                           "nearby_source_samples": nearby_source_samples(reconstruction, candidate["approach_start"]),
                           "scope": "Projection-only source-photo inspection coordinates; no occlusion or physical floor classification is inferred."})
        evidence.verify_sources(args.job, reconstruction.sources, reconstruction.calibration)
    evidence.verify_sources(args.job, retained["sources"], retained["calibration"])
    if scenes.active(args.job) != pointer or manifests.sha256_file(source) != record["sha256"]:
        raise ValueError("Active scene or Gaussian source changed during review")
    passed = [result["index"] for result in results if result["passes_center_and_source_point_screen"]]
    output = {"schema": "dev.splatlab.architectural-candidate-preservation-review/v1",
              "base": pointer, "search_sha256": manifests.sha256_file(args.search),
              "guard_evidence_sha256": manifests.sha256_file(args.guard_evidence),
              "gaussian_sha256": record["sha256"], "code_sha256": manifests.sha256_file(Path(__file__)),
              "reference_receipt_sha256": receipt["sha256"], "source_guard_count": len(guards),
              "reviewed_candidates": len(results), "point_screen_survivors": passed,
              "results": results, "active_unchanged": True, "scene_activated": False,
              "starting_surface_review": starts,
              "scope": "CPU-only evidence review; no Open3D, mesh build, CUDA, model mutation or acceptance. Center-based checks cannot certify preservation of Gaussian footprints or unnamed objects."}
    manifests.atomic_write_json(args.output, output)
    print(json.dumps({key: output[key] for key in ("reviewed_candidates", "point_screen_survivors", "source_guard_count", "active_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
