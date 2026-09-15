#!/usr/bin/env python3
"""Create a private real-capture fixture and an adjacent authored room; never alter its source job."""

import asyncio
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests
from dcc import blender_workflow
import glb_check
import scene_revisions
import splat_route


def main():
    source = Path("/home/rtoony/projects/splatcli/outputs/3d/splat_aea04ab3")
    outputs = Path(__file__).resolve().parents[1] / "data" / "spatial" / "scene-proof" / "outputs"
    job = outputs / "splat_c0ffee"
    if job.exists():
        raise SystemExit("Proof fixture already exists; inspect its state rather than overwrite it")
    fingerprint = scene_revisions.source_fingerprint(source)
    for key in fingerprint["files"]:
        destination = job / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / key, destination)
    (job / "_regen").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "_regen" / "scene.blend", job / "_regen" / "scene.blend")
    if fingerprint != scene_revisions.source_fingerprint(source):
        raise SystemExit("Source changed during fixture creation; do not use this fixture")
    metadata = manifests.read_json(source / "meta.json")
    metadata.update(job_id=job.name, output_dir=str(job), display_name="Private Bonsai-derived creative revision proof", status="completed")
    manifests.atomic_write_json(job / "meta.json", metadata)
    splat_route.DEFAULT_3D_ROOT = outputs
    blender_workflow.OUTPUT_ROOT = outputs.resolve()
    bounds = glb_check.position_bounds(job / "_world" / "shell.glb")
    floor = (manifests.read_json(job / "_world" / "collision_shell.json") or {}).get("probe", {}).get("floor_level_y", bounds["aabb"]["min"][1])
    front_z = bounds["aabb"]["min"][2] - 0.15
    origin = [sum((bounds["aabb"]["min"][0], bounds["aabb"]["max"][0])) / 2, -front_z, floor]
    room = blender_workflow.run_action(job.name, "create_room", {"name": "studio-room", "origin": origin,
        "width": 3, "depth": 3, "height": 2.8, "door_width": 1.1, "door_height": 2.2}, note="Adjacent candidate for private scene transaction proof; not a captured-wall opening claim")
    material = blender_workflow.run_action(job.name, "assign_material", {"object": room["result"]["object"], "color": [0.4, 0.65, 0.75]}, base_version=room["version"])
    exported = blender_workflow.export_glb(job.name, base_version=material["version"], object_name=room["result"]["object"], bake_world_transform=True)
    before = scene_revisions.source_fingerprint(job)
    viewer = asyncio.run(splat_route.get_splat_world_manifest(job.name))
    pointer = scene_revisions.initialize(job, viewer, before)
    report = {"source_job": source.name, "job_id": job.name, "outputs_root": str(outputs),
              "active": pointer, "export": Path(exported["output"]["path"]).name,
              "room_origin_blender_z_up_m": origin, "source_unchanged": fingerprint == scene_revisions.source_fingerprint(source),
              "acceptance": "private transaction fixture; no user-scene application or connected-doorway acceptance"}
    manifests.atomic_write_json(job.parent.parent / "fixture-receipt.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
