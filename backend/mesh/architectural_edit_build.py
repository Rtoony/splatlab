#!/usr/bin/env python3
"""Build a typed captured portal and authored extension under the bounded compute gate."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import open3d as o3d
import trimesh

import architectural_edits as architecture
import architectural_navigation as navigation
import architectural_volumes as volumes
import artifact_manifest as manifests
import scene_revisions as scenes
import selection_reviews as selections


def ray_scene(mesh):
    result = o3d.t.geometry.RaycastingScene()
    geometry = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)), o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
    result.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(geometry))
    return result


def distances(scene, points):
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def occupancy(scene, points):
    return scene.compute_occupancy(o3d.core.Tensor(np.asarray(points, dtype=np.float32))).numpy()


def floor_hits(scene, points):
    origins = points + [0, .3, 0]
    rays = np.concatenate([origins, np.tile([0., -1., 0.], (len(points), 1))], axis=1)
    distance = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
    return origins[:, 1] - distance


def overlaps(local, lower, upper):
    return bool(np.all(local.max(axis=0) > lower) and np.all(local.min(axis=0) < upper))


def build(job, receipt, output):
    started = time.monotonic()
    spec, recipe = receipt["spec"], receipt["recipe"]
    selected_geometry = architecture.geometry_method(receipt)
    frame = architecture.frame(spec)
    scale = receipt["calibration"]["meters_per_unit"]
    base_path = scenes.blob_path(job, receipt["base_artifacts"]["_world/collision_shell.glb"]["sha256"])
    before = trimesh.load(base_path, file_type="glb", force="mesh", process=False)
    before.apply_scale(scale)
    if len(before.faces) > 2_000_000 or not before.is_watertight or not before.is_volume:
        raise ValueError("Architectural subtraction requires a bounded closed, consistently oriented base solid")
    transform = np.eye(4)
    transform[:3, :3] = frame["rotation"]
    transform[:3, 3] = frame["origin"] + frame["rotation"] @ ((frame["lower"] + frame["upper"]) / 2)
    cut = trimesh.creation.box(extents=frame["upper"] - frame["lower"], transform=transform)
    after = trimesh.boolean.difference([before, cut], engine="manifold", check_volume=True)
    if not isinstance(after, trimesh.Trimesh) or not len(after.faces):
        raise ValueError("Portal subtraction did not produce a retained captured solid")
    with np.load(architecture.artifact(job, receipt["architecture_id"], "authored-room.npz"), allow_pickle=False) as stored:
        room = trimesh.Trimesh(vertices=stored["vertices"], faces=stored["faces"], process=False)
    room.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[185, 203, 217, 255], metallicFactor=0, roughnessFactor=.85, doubleSided=True))
    room.metadata.update(architecture.geometry_metadata(receipt["architecture_id"], "world-y-up-metres",
        "Authored/inferred connected room, not measured architecture"))
    room.metadata["splatlab_architecture"].update(geometry_method=selected_geometry,
        floor_finish_m=architecture.floor_finish_m(spec, selected_geometry))
    cut.metadata.update(architecture.geometry_metadata(receipt["architecture_id"], "world-y-up-metres",
        "Authored/inferred cutting volume, not measured structure"))
    protected, conflicts = [], []
    appearance_method = volumes.method(receipt["clip"])
    protected_gaussian_conflicts = {}
    capture_impact = None
    appearance_rows = {}
    replacement_outside = {}
    with np.load(architecture.artifact(job, receipt["architecture_id"], "replacement-membership.npz"), allow_pickle=False) as members:
        for item in receipt["replaced_elements"]:
            mesh = trimesh.load(scenes.blob_path(job, receipt["base_artifacts"][item["key"]]["sha256"]), file_type="glb", force="mesh", process=False)
            points = np.concatenate([np.asarray(mesh.vertices), members[item["slug"] + "-positions-world"]])
            local = architecture.local_coordinates(points, spec)
            replacement_outside[item["slug"]] = int(np.any((local < frame["lower"]) | (local > frame["upper"]), axis=1).sum())
    protected_regions = architecture.protected_volume_bounds(spec, selected_geometry)
    for item in receipt["protected_elements"]:
        mesh = trimesh.load(scenes.blob_path(job, receipt["base_artifacts"][item["key"]]["sha256"]), file_type="glb", force="mesh", process=False)
        mesh.apply_scale(item["scale"])
        local = architecture.local_coordinates(np.asarray(mesh.vertices), spec)
        if overlaps(local, frame["lower"] - .01, frame["upper"] + .01) or any(overlaps(local, lower, upper) for lower, upper in protected_regions):
            conflicts.append(item["slug"])
        if item["role"] == "prop" or item["provenance"] in {"authored", "generated"} and item["role"] in {"static", "environment"}:
            protected.append(mesh)
    room_intrusion = None
    if appearance_method == volumes.ROOM_METHOD:
        revision = scenes.read_revision(job, receipt["base"]["revision_id"])
        positions = selections.read_ply_xyz(scenes.blob_path(job, receipt["base_artifacts"]["_preview/langweb.ply"]["sha256"]))
        world_positions = positions[:, [0, 2, 1]] * np.array([1, 1, -1]) * scale
        local_positions = architecture.local_coordinates(world_positions, spec)
        protected_names = [item["slug"] for item in receipt["protected_elements"]]
        protected_gaussian_conflicts = volumes.protected_row_conflicts(local_positions,
            revision["state"]["selections"], protected_names, receipt["clip"])
        capture_impact, appearance_rows = volumes.capture_impact(local_positions, revision["state"]["selections"],
            protected_names, [item["slug"] for item in receipt["replaced_elements"]], receipt["clip"])
        conflicts = sorted(set(conflicts) | set(protected_gaussian_conflicts))
        interior_lower, interior_upper = volumes.room_interior(spec)
        interior_transform = np.eye(4)
        interior_transform[:3, :3] = frame["rotation"]
        interior_transform[:3, 3] = frame["origin"] + frame["rotation"] @ ((interior_lower + interior_upper) / 2)
        interior = trimesh.creation.box(extents=interior_upper - interior_lower, transform=interior_transform)
        room_intrusion = trimesh.boolean.intersection([after, interior], engine="manifold", check_volume=True)
    combined = trimesh.util.concatenate([after, room, *protected])
    after_scene, combined_scene = ray_scene(after), ray_scene(combined)
    interior_axes = [np.linspace(lower + .015, upper - .015, max(3, int(np.ceil((upper - lower) / .075))))
                     for lower, upper in zip(frame["lower"], frame["upper"])]
    local_grid = np.stack(np.meshgrid(*interior_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = local_grid @ frame["rotation"].T + frame["origin"]
    inside_after = occupancy(after_scene, grid)
    centers = np.concatenate([np.asarray(before.vertices), np.asarray(before.triangles_center)])
    local = architecture.local_coordinates(centers, spec)
    outside = np.any((local < frame["lower"] - .02) | (local > frame["upper"] + .02), axis=1)
    preserved = centers[outside][::max(1, int(outside.sum()) // 25000)]
    outside_distance = distances(after_scene, preserved)
    navigation_method = navigation.method(recipe)
    route_local, lane_count = navigation.local_routes(spec, recipe)
    route = route_local @ frame["rotation"].T + frame["origin"]
    if navigation_method == navigation.ENTRY_METHOD:
        center_floors, floors = navigation.floor_support(route, lambda points: floor_hits(combined_scene, points), recipe["capsule_radius_m"])
    else:
        center_floors = floors = floor_hits(combined_scene, route)
    finite_floors = np.isfinite(floors) & np.isfinite(center_floors)
    approach = route_local[:, 2] < -spec["cut_depth"] / 2
    floor_errors = np.maximum(np.abs(floors - frame["origin"][1]), np.abs(center_floors - frame["origin"][1]))
    maximum_floor_error = float(floor_errors[finite_floors].max()) if finite_floors.any() else None
    continuous = bool(finite_floors.all() and max(np.max(np.abs(np.diff(values.reshape(3, lane_count), axis=1))) for values in (floors, center_floors)) <= .04)
    floor_ok = finite_floors & (floor_errors <= recipe["floor_tolerance_m"])
    safe_floors = np.where(finite_floors, floors, frame["origin"][1])
    heights = np.linspace(recipe["capsule_radius_m"] + .04, recipe["capsule_height_m"] - recipe["capsule_radius_m"], 32)
    body = np.repeat(route, len(heights), axis=0)
    body[:, 1] = (safe_floors[:, None] + heights).reshape(-1)
    clearance = distances(combined_scene, body)
    occupied_body = occupancy(after_scene, body)
    removed = float(before.volume - after.volume)
    gates = {"base_watertight": bool(before.is_watertight and before.is_volume), "cut_watertight": bool(after.is_watertight and after.is_volume),
        "positive_bounded_removal": bool(1e-5 < removed <= cut.volume + 1e-5), "portal_volume_clear": bool(not inside_after.any()),
        "outside_surface_preserved": bool(len(preserved) and outside_distance.max() <= 2e-5),
        "approach_floor_continuous": bool(approach.any() and floor_ok[approach].all() and continuous),
        "room_floor_continuous": bool((~approach).any() and floor_ok[~approach].all() and continuous),
        "capsule_route_clear": bool(not occupied_body.any() and clearance.min() >= recipe["capsule_radius_m"] + .03),
        "protected_elements_unchanged": not conflicts, "replacement_region_contained": not any(replacement_outside.values())}
    if room_intrusion is not None:
        gates["authored_room_interior_clear"] = len(room_intrusion.faces) == 0
    if selected_geometry == architecture.JOINED_GEOMETRY:
        gates["authored_room_watertight"] = bool(room.is_watertight and room.is_volume)
    verdict = "PASS_ARCHITECTURAL_EDIT" if all(gates.values()) else "FAIL_ARCHITECTURAL_EDIT"
    after.metadata.update(architecture.geometry_metadata(receipt["architecture_id"], "capture-y-up-scene-units",
        "Authored local portal edit, not measured geometry"))
    after.apply_scale(1 / scale)
    after.export(output / "collision_shell.glb")
    visual_room = room.copy()
    if selected_geometry == architecture.JOINED_GEOMETRY:
        visual_room.unmerge_vertices()
        visual_room.metadata["splatlab_architecture"]["surface_normals"] = "flat-per-triangle"
    visual_room.export(output / "room.glb")
    cut.export(output / "cut-volume.glb")
    np.savez_compressed(output / "geometry-probes.npz", route_world=route, floor_world_y=floors, center_floor_world_y=center_floors, capsule_centers_world=body,
        capsule_clearance_m=clearance, capsule_base_occupancy=occupied_body, portal_grid_world=grid, portal_occupancy_after=inside_after,
        preserved_base_world=preserved, outside_surface_distance_m=outside_distance,
        room_intrusion_vertices=np.asarray(room_intrusion.vertices) if room_intrusion is not None else np.empty((0, 3)),
        room_intrusion_faces=np.asarray(room_intrusion.faces) if room_intrusion is not None else np.empty((0, 3), dtype=np.int64),
        **{"appearance_" + name + "_rows": rows for name, rows in appearance_rows.items()})
    collision_report = deepcopy(receipt["base_collision_report"])
    collision_report.update(verdict=verdict, method=architecture.METHOD, architecture_id=receipt["architecture_id"],
        architectural_gates=gates, geometry_frame={"axis": "y-up", "units": "scene-units", "meters_per_unit": scale},
        navigation_method=navigation_method, appearance_method=appearance_method, geometry_method=selected_geometry,
        scope="Captured local Boolean only; room/protected collision is merged by Studio. Local route probes are not whole-scene or full-size browser acceptance.")
    manifests.atomic_write_json(output / "collision_shell.json", collision_report)
    nav = {"v": 2 if navigation_method == navigation.ENTRY_METHOD else 1, "navigation_method": navigation_method,
        "frame": "world-y-up-metres", "architecture_id": receipt["architecture_id"], "route": route.tolist(),
        "floor_y": [float(value) if np.isfinite(value) else None for value in floors], "capsule_radius_m": recipe["capsule_radius_m"],
        "center_floor_y": [float(value) if np.isfinite(value) else None for value in center_floors],
        "capsule_height_m": recipe["capsule_height_m"],
        "scope": "Shared centered approach fans into three interior lanes for the entry method; the legacy method uses three parallel approaches. Floor footprint support is retained separately from center rays. These are sampled worker-profile probes, not whole-scene or default-body navigation acceptance.", "gates": gates}
    if verdict == "PASS_ARCHITECTURAL_EDIT":
        navigation.verify_document(spec, recipe, frame, nav, receipt["architecture_id"], gates)
    manifests.atomic_write_json(output / "navmesh.json", nav)
    architecture.verify(job, receipt)
    names = ("collision_shell.glb", "room.glb", "cut-volume.glb", "geometry-probes.npz", "collision_shell.json", "navmesh.json")
    result = selections.seal(job, output, "result.json", {"architecture_id": receipt["architecture_id"], "prepared_sha256": receipt["sha256"],
        "method": architecture.METHOD, "navigation_method": navigation_method, "appearance_method": appearance_method,
        "geometry_method": selected_geometry,
        "verdict": verdict, "gates": gates, "created_at": manifests.utc_now(), "seconds": time.monotonic() - started,
        "code_sha256": manifests.sha256_file(Path(__file__)), "contract_code_sha256": manifests.sha256_file(Path(architecture.__file__)),
        "appearance_code_sha256": manifests.sha256_file(Path(volumes.__file__)),
        "solid_boundary_code_sha256": manifests.sha256_file(Path(architecture.solids.__file__)),
        "metrics": {"before_triangles": len(before.faces), "after_triangles": len(after.faces), "room_triangles": len(room.faces),
            "removed_volume_m3": removed, "cut_volume_m3": float(cut.volume), "portal_samples": len(grid), "route_samples": len(route),
            "capsule_samples": len(body), "minimum_capsule_clearance_m": float(clearance.min()), "maximum_floor_error_m": maximum_floor_error,
            "maximum_footprint_support_adjustment_m": float(np.max(floors[finite_floors] - center_floors[finite_floors])) if finite_floors.any() else None,
            "outside_samples": len(preserved), "maximum_outside_distance_m": float(outside_distance.max()), "protected_conflicts": conflicts,
            "replacement_samples_outside_portal": replacement_outside, "protected_gaussian_conflicts": protected_gaussian_conflicts,
            "capture_impact": capture_impact, "floor_finish_m": architecture.floor_finish_m(spec, selected_geometry),
            "room_intrusion_triangles": len(room_intrusion.faces) if room_intrusion is not None else None,
            "room_intrusion_volume_m3": float(abs(room_intrusion.volume)) if room_intrusion is not None else None},
        "scope": "Authored/inferred frame with paired captured-volume cut and connected room; visual clipping and browser traversal must still be reviewed"}, [output / name for name in names])
    print(json.dumps({key: result[key] for key in ("architecture_id", "verdict", "gates", "metrics", "seconds")}))
    return 0 if verdict == "PASS_ARCHITECTURAL_EDIT" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("architecture_id")
    args = parser.parse_args()
    gate = Path(__file__).resolve().parents[2] / "tools/splatlab-compute-gate.sh"
    subprocess.run([str(gate), "--is-contained"], check=True)
    with architecture.worker_slot(args.job, args.architecture_id):
        receipt = architecture.read(args.job, args.architecture_id)
        architecture.verify(args.job, receipt)
        output = architecture.directory(args.job, args.architecture_id)
        if any((output / name).exists() for name in ("result.json", "collision_shell.glb", "room.glb")):
            raise ValueError("Architectural build already exists; prepare a new edit")
        return build(args.job, receipt, output)


if __name__ == "__main__":
    raise SystemExit(main())
