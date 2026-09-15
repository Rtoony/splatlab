#!/usr/bin/env python3
"""Cut selection-bound local collision under the compute gate without resampling the capture mesh."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from plyfile import PlyData
import trimesh
import open3d as o3d
from scipy.spatial import ConvexHull, HalfspaceIntersection

import artifact_manifest as manifests
import scene_revisions as scenes
import selection_collision as collision
import selection_reviews as reviews
from mesh import world_shell as shell
from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY


def occupancy(mesh, points):
    ray_scene = o3d.t.geometry.RaycastingScene()
    geometry = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(geometry))
    return ray_scene.compute_occupancy(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def distance(mesh, points):
    ray_scene = o3d.t.geometry.RaycastingScene()
    geometry = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces))
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(geometry))
    return ray_scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def local_cut(job, receipt, output, positions, rows, probe):
    metres = receipt["calibration"]["meters_per_unit"]
    recipe = receipt["recipe"]
    base_path = scenes.blob_path(job, receipt["base_collision_artifact"]["sha256"])
    if manifests.sha256_file(base_path) != receipt["base_collision_artifact"]["sha256"]:
        raise RuntimeError("Base collision changed")
    before = trimesh.load(base_path, file_type="glb", force="mesh", process=False)
    if not before.is_watertight or not before.is_volume:
        raise RuntimeError("Local collision subtraction requires a closed, consistently oriented base solid")
    selected = shell.to_yup(positions[rows["candidate"]])
    lower = selected.min(0) - recipe["cut_margin_m"] / metres
    upper = selected.max(0) + recipe["cut_margin_m"] / metres
    floor = receipt["base_collision_report"]["probe"]["floor_level_y"]
    lower[1] = floor + recipe["floor_clearance_m"] / metres
    if np.any(upper <= lower) or not np.all(selected[:, 1] > lower[1]):
        raise RuntimeError("Selection intersects the protected floor; choose a support-specific edit volume")
    prior_selection = scenes.read_revision(job, receipt["base"]["revision_id"])["state"]["selections"]
    other_rows = np.array([row for slug, entry in prior_selection["elements"].items()
                          if slug != receipt["selected_slug"] for row in entry["rows"]], dtype=np.int64)
    other_points = shell.to_yup(positions[other_rows])
    projected = selected.copy()
    projected[:, 1] = lower[1]
    hull = ConvexHull(np.concatenate([selected, projected]))
    planes = hull.equations.copy()
    separation = (other_points @ planes[:, :3].T + planes[:, 3]).max(1)
    if np.any(separation <= 0):
        raise RuntimeError("Selection hull overlaps another named instance; explicit joint selection is required")
    margin = min(recipe["cut_margin_m"] / metres, float(separation.min()) / 4) if len(separation) else recipe["cut_margin_m"] / metres
    if margin * metres < .001:
        raise RuntimeError("Named instances are too close for a verified one-millimetre Boolean clearance")
    floor_faces = planes[:, 1] < -.999999
    planes[~floor_faces, 3] -= margin
    interior = np.concatenate([selected, projected]).mean(0)
    expanded = HalfspaceIntersection(planes, interior).intersections
    cutter = trimesh.convex.convex_hull(expanded)
    conflicts = ((other_points @ planes[:, :3].T + planes[:, 3]) <= 0).all(1)
    if conflicts.any():
        raise RuntimeError("Padded selection hull reaches another named instance")
    if cutter.volume * metres ** 3 > recipe["maximum_removed_volume_m3"]:
        raise RuntimeError("Local removal volume exceeds the one-cubic-metre safety budget")
    lower, upper = cutter.bounds
    after = trimesh.boolean.difference([before, cutter], engine="manifold", check_volume=True)
    after.metadata[GLTF_EXTRAS_KEY] = GENERATIVE_TAG
    after.metadata["geometry_source"] = "selection-aware local subtraction with fitted protected floor"
    stage = output / "candidate"
    stage.mkdir()
    path = stage / "collision.glb"
    after.export(path)
    after = trimesh.load(path, force="mesh", process=False)
    cutter.metadata[GLTF_EXTRAS_KEY] = GENERATIVE_TAG
    cutter.export(output / "cut-volume.glb")
    grid_axes = [np.linspace(lower[axis] + .005 / metres, upper[axis] - .005 / metres, 12) for axis in range(3)]
    grid = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    residual = occupancy(after, selected)
    grid = grid[((grid @ planes[:, :3].T + planes[:, 3]) < -.001 / metres).all(1)]
    grid_before, grid_after = occupancy(before, grid), occupancy(after, grid)
    vertices = np.asarray(before.vertices)
    outside = ((vertices < lower - .002 / metres) | (vertices > upper + .002 / metres)).any(1)
    preserved = vertices[outside][::max(1, int(outside.sum()) // 50000)]
    outside_distance = distance(after, preserved) * metres
    inside_vertices = ((np.asarray(after.vertices) @ planes[:, :3].T + planes[:, 3]) < -.002 / metres).all(1)
    baseline_gates = shell.evaluate(before, probe, recipe["player_radius_m"] / metres, recipe["player_height_m"] / metres)
    candidate_gates = shell.evaluate(after, probe, recipe["player_radius_m"] / metres, recipe["player_height_m"] / metres)
    nav = candidate_gates.pop("_walkable")
    baseline_gates.pop("_walkable")
    removed_volume = (before.volume - after.volume) * metres ** 3
    checks = {"watertight": bool(after.is_watertight and after.is_volume),
        "selected_centers_clear": bool(np.count_nonzero(residual) == 0), "edit_volume_clear": bool(len(grid) >= 100 and np.count_nonzero(grid_after) == 0),
        "no_faces_inside_edit_volume": bool(not inside_vertices.any()),
        "outside_surface_preserved": bool(len(preserved) > 0 and outside_distance.max() < .0001),
        "bounded_positive_volume_removed": bool(0 < removed_volume <= recipe["maximum_removed_volume_m3"]),
        "floor_coverage_not_reduced": candidate_gates["floor_continuity"] >= baseline_gates["floor_continuity"],
        "maximum_hole_not_increased": candidate_gates["max_hole_span"] <= baseline_gates["max_hole_span"]}
    verdict = "PASS_LOCAL_EDIT" if all(checks.values()) else "FAILED"
    report = {"verdict": verdict, "method": collision.METHOD, "triangles": len(after.faces), "gates": {key: candidate_gates[key] for key in
        ("components", "largest_component_frac", "watertight", "floor_continuity", "max_hole_span")},
        "gates_detail": candidate_gates, "local_gates": checks, "geometry_frame": {"axis": "y-up", "units": "scene-units", "meters_per_unit": metres},
        "params": {"seed_yup": probe.seed.tolist(), "player_height": recipe["player_height_m"] / metres,
                   "player_radius": recipe["player_radius_m"] / metres},
        "probe": {"floor_level_y": probe.floor_level, "top_level_y": probe.top_level, "grid_res": probe.cell},
        "scope": "local edit invariants, not whole-scene navigation acceptance"}
    manifests.atomic_write_json(stage / "report.json", report)
    manifests.atomic_write_json(stage / "navmesh.json", {"v": 1, "frame": "y-up", **nav,
        "scope": "background probe only; remaining prop collision is not baked into this grid"})
    np.savez_compressed(output / "membership-probes.npz", selected_raw_yup=selected, selected_occupied_after=residual,
                        cut_grid_raw_yup=grid, cut_grid_before=grid_before, cut_grid_after=grid_after,
                        outside_base_vertices_raw_yup=preserved, outside_surface_distance_m=outside_distance)
    return {"verdict": verdict, "local_gates": checks, "candidate": report,
        "current": {"triangles": len(before.faces), "gates_detail": baseline_gates},
        "cut_volume": {"min_raw_yup": lower.tolist(), "max_raw_yup": upper.tolist(), "removed_volume_m3": removed_volume,
                       "shape": "floor-extruded convex selection hull", "actual_margin_m": margin * metres,
                       "floor_is_fitted": True, "other_named_instance_conflicts": int(conflicts.sum())},
        "local_occupancy": {"selected_centers": len(selected), "selected_centers_occupied_before": int(np.count_nonzero(occupancy(before, selected))),
                            "selected_centers_occupied_after": int(np.count_nonzero(residual)), "grid_points": len(grid),
                            "grid_occupied_before": int(np.count_nonzero(grid_before)), "grid_occupied_after": int(np.count_nonzero(grid_after)),
                            "outside_surface_samples": len(preserved), "maximum_outside_distance_m": float(outside_distance.max())}}


def run(job, identifier):
    receipt = collision.read(job, identifier)
    collision.verify(job, receipt)
    output = collision.directory(job, identifier)
    if any((output / name).exists() for name in ("result.json", "current", "candidate", "cut-volume.glb")):
        raise RuntimeError("Collision outputs already exist; prepare a new build")
    started = time.monotonic()
    recipe = receipt["recipe"]
    if recipe["method"] != collision.METHOD or recipe.get("cut_shape") != "floor-extruded convex selection hull":
        raise RuntimeError("This diagnostic recipe is retired; prepare a current local clearance build")
    metres = receipt["calibration"]["meters_per_unit"]
    splat = scenes.blob_path(job, receipt["splat_artifact"]["sha256"])
    if manifests.sha256_file(splat) != receipt["splat_artifact"]["sha256"]:
        raise RuntimeError("Captured splat changed")
    vertices = PlyData.read(splat)["vertex"]
    positions = np.stack([vertices[field] for field in ("x", "y", "z")], axis=1).astype(np.float64)
    solid = np.asarray(vertices["opacity"]) > 0
    if len(positions) != receipt["counts"]["scene"] or not np.isfinite(positions).all():
        raise RuntimeError("Captured coordinates do not match the partition")
    with np.load(collision.artifact(job, identifier, "partition.npz"), allow_pickle=False) as stored:
        rows = {key: stored[key] for key in stored.files}
    bounds_path = scenes.blob_path(job, receipt["bounds_artifact"]["sha256"])
    if manifests.sha256_file(bounds_path) != receipt["bounds_artifact"]["sha256"]:
        raise RuntimeError("Room-bound reference changed")
    bounds_mesh = trimesh.load(bounds_path, file_type="ply", process=False)
    lower = np.percentile(bounds_mesh.vertices, .1, axis=0) - .25 / metres
    upper = np.percentile(bounds_mesh.vertices, 99.9, axis=0) + .25 / metres
    corners = np.array([[axis_x, axis_y, axis_z] for axis_x in (lower[0], upper[0])
                        for axis_y in (lower[1], upper[1]) for axis_z in (lower[2], upper[2])])
    corners = shell.to_yup(corners)
    bounds = (corners.min(0), corners.max(0))
    dimensions = np.ceil((bounds[1] - bounds[0]) / (recipe["voxel_m"] / metres))
    if np.prod(dimensions) > 32_000_000:
        raise RuntimeError("Collision exceeds the 32-million-voxel work budget")
    inside = solid & ((positions >= lower) & (positions <= upper)).all(axis=1)
    current_points = shell.to_yup(positions[rows["current_background"]][inside[rows["current_background"]]])
    probe = shell.build_probe(current_points, recipe["grid_m"] / metres,
                              recipe["player_radius_m"] / metres, recipe["player_height_m"] / metres)
    result = local_cut(job, receipt, output, positions, rows, probe)
    collision.verify(job, receipt)
    files = [output / "membership-probes.npz", output / "cut-volume.glb"]
    files += [output / "candidate" / name for name in ("collision.glb", "report.json", "navmesh.json")]
    result.update(collision_id=identifier, method=collision.METHOD, prepared_receipt_sha256=receipt["sha256"],
                  seconds=time.monotonic() - started, created_at=manifests.utc_now(),
                  code_sha256=manifests.sha256_file(Path(__file__)), shell_code_sha256=manifests.sha256_file(Path(shell.__file__)),
                  scope="selection-aware local volume edit; fitted floor; full navigation and appearance acceptance remain separate")
    sealed = reviews.seal(job, output, "result.json", result, files)
    print(json.dumps({key: sealed[key] for key in ("collision_id", "verdict", "local_gates", "local_occupancy", "seconds")}, indent=2))
    return 0 if sealed["verdict"] == "PASS_LOCAL_EDIT" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("collision_id")
    args = parser.parse_args()
    job = args.job.resolve()
    with (collision.directory(job, args.collision_id) / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(job, args.collision_id)


if __name__ == "__main__":
    raise SystemExit(main())
