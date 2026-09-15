#!/usr/bin/env python3
"""Prepare a local authored-chest plus observed-background experiment on the private fixture."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import artifact_manifest as manifests
import background_recovery as recovery
from dcc import blender_workflow as blender
import scene_revisions as scenes
import selection_collision as collisions
import support_surfaces as support


def mesh_geometry(path):
    loaded = trimesh.load(path, force="scene", process=False)
    return loaded.to_geometry()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-generation", type=int, required=True)
    parser.add_argument("--collision-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    job = root / "data/spatial/scene-proof/outputs/splat_c0ffee"
    pointer = scenes.active(job)
    if not pointer or pointer["generation"] != args.expected_generation:
        raise RuntimeError("Inspect the private generation before preparing this experiment")
    collision = collisions.read(job, args.collision_id)
    collisions.verify(job, collision)
    collision_result = collisions.read(job, args.collision_id, True)
    if collision["selected_slug"] != "cardboard-box-2" or collision_result["verdict"] != "PASS_LOCAL_EDIT":
        raise RuntimeError("This experiment requires current cardboard-box-2 clearance")
    identifier = "compound-" + uuid.uuid4().hex[:24]
    output = root / "data/spatial" / identifier
    output.mkdir()
    started = time.monotonic()
    receipt = {"schema": "dev.splatlab.compound-replacement-proof/v1", "base": pointer, "job_id": job.name,
               "collision_id": args.collision_id, "selection_review_id": collision["selection_review_id"],
               "status": "failed", "scope": "private authored asset replacement with observed-only recovery; not generative inference or navigation acceptance"}
    try:
        views = {image_id: support.inspect(job, "cardboard-box-2", pointer["generation"], image_id) for image_id in (47, 280)}
        anchor_ids = [(47, "9279"), (280, "121451"), (280, "121358"), (280, "9542")]
        anchors = {"evidence_sha256": views[47]["evidence_sha256"], "points": [
            {"image_id": image_id, "point_id": point_id, "photo_sha256": views[image_id]["photo_sha256"]}
            for image_id, point_id in anchor_ids]}
        recovered = recovery.build(job, "cardboard-box-2", pointer["generation"], 128, anchors)
        receipt["recovery_id"] = recovered["recovery_id"]
        receipt["recovery_sha256"] = recovered["sha256"]
        plane = recovered["report"]["plane"]
        asset = root / "assets/library/dungeon-chest.glb"
        receipt["library_asset"] = manifests.file_identity(asset)
        receipt["catalog"] = (manifests.read_json(asset.parent / "catalog.json") or {})["assets"][asset.stem]
        blender.OUTPUT_ROOT = job.parent
        imported = blender.run_action(job.name, "import_asset", {"name": asset.stem, "slug": identifier}, note=receipt["scope"])
        object_name = imported["result"]["object"]
        original = blender.export_glb(job.name, base_version=imported["version"], object_name=object_name, bake_world_transform=True)
        original_path = scenes.local_file(job, original["output"]["path"])
        receipt["master"] = manifests.file_identity(original_path)
        receipt["master_filename"] = original_path.name
        master_mesh = mesh_geometry(original_path)
        vertices = np.asarray(master_mesh.vertices, dtype=np.float64)
        lower, upper = vertices.min(0), vertices.max(0)
        if not np.isfinite(vertices).all() or np.any(upper - lower <= 0):
            raise RuntimeError("Library master has invalid spatial bounds")
        scale = .45 / max((upper - lower)[[0, 2]])
        bottom = np.array([(lower[0] + upper[0]) / 2, lower[1], (lower[2] + upper[2]) / 2])
        to_blender = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
        target = to_blender @ np.asarray(plane["center"])
        normal = to_blender @ np.asarray(plane["normal"])
        axis = np.cross([0, 0, 1], normal)
        angle = np.arctan2(np.linalg.norm(axis), normal[2])
        delta = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix() if np.linalg.norm(axis) > 1e-9 else np.eye(3)
        inspected = blender.run_action(job.name, "inspect", {}, base_version=imported["version"])
        object_state = next(item for item in inspected["result"]["objects"] if item["name"] == object_name)
        original_rotation = Rotation.from_euler("xyz", object_state["rotation_degrees"], degrees=True).as_matrix()
        placement = {"object": object_name,
                     "location": (target + scale * delta @ (np.asarray(object_state["location"]) - to_blender @ bottom)).tolist(),
                     "rotation_degrees": Rotation.from_matrix(delta @ original_rotation).as_euler("xyz", degrees=True).tolist(),
                     "scale": (np.asarray(object_state["scale"]) * scale).tolist()}
        placed = blender.run_action(job.name, "transform_object", placement, base_version=imported["version"], note="Uniform-scale master; base aligned to inferred support, no automatic acceptance")
        exported = blender.export_glb(job.name, base_version=placed["version"], object_name=object_name, bake_world_transform=True)
        export_path = scenes.local_file(job, exported["output"]["path"])
        actual_mesh = mesh_geometry(export_path)
        actual = np.asarray(actual_mesh.vertices, dtype=np.float64)
        expected = (target + scale * ((vertices - bottom) @ to_blender.T) @ delta.T) @ to_blender
        unique, inverse = np.unique(expected, axis=0, return_inverse=True)
        distances, addresses = cKDTree(unique).query(actual)
        expected_faces = np.sort(inverse[master_mesh.faces], axis=1)
        actual_faces = np.sort(addresses[actual_mesh.faces], axis=1)
        same_faces = (actual_faces.shape == expected_faces.shape and np.array_equal(
            expected_faces[np.lexsort(expected_faces.T)], actual_faces[np.lexsort(actual_faces.T)]))
        receipt.update(export=export_path.name, placed_asset=manifests.file_identity(export_path), placement=placement,
                       master_vertices=len(vertices), placed_vertices=len(actual), master_triangles=len(master_mesh.faces),
                       placed_triangles=len(actual_mesh.faces), same_transformed_faces=same_faces,
                       vertex_transform_max_error_m=float(distances.max()))
        if not same_faces or distances.max() > 2e-5:
            raise RuntimeError("Typed Blender export differs from the intended supported master transform")
        signed = (actual - np.asarray(plane["center"])) @ np.asarray(plane["normal"])
        if abs(float(signed.min())) > 2e-5:
            raise RuntimeError("Replacement base does not meet the inferred plane")
        if not manifests.same_file_identity(asset, receipt["library_asset"]) or scenes.active(job) != pointer:
            raise RuntimeError("Scene or library source changed during preparation")
        receipt.update(status="needs-review", object_slug="chest-" + identifier[-8:], uniform_scale=scale,
                       support_contact_error_m=abs(float(signed.min())), supported_fraction=recovered["report"]["supported_fraction"],
                       unknown_fraction=recovered["report"]["unknown_fraction"], vertex_reduction=False, material_replacement=False)
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["seconds"] = time.monotonic() - started
        receipt["sha256"] = hashlib.sha256(scenes.canonical_bytes(receipt)).hexdigest()
        manifests.atomic_write_json(output / "receipt.json", receipt)
        print(json.dumps({"receipt": str(output / "receipt.json"), "status": receipt["status"], "error": receipt.get("error")}), flush=True)


if __name__ == "__main__":
    main()
