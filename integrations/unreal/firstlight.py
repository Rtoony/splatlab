"""Unattended UE first light: import the staged hydrant, assemble the actor
contract, and record a NUMERIC axis/scale witness — no GUI session needed.

MODE MATTERS (measured 2026-07-26): under `-run=pythonscript -nullrhi` the
IMPORTS work but any actor spawn EXCEPTION_ACCESS_VIOLATIONs inside
EditorFramework — commandlet mode has no level-editor frame. Run the FULL
editor headless instead:

  UnrealEditor-Cmd.exe <uproject> -ExecutePythonScript="firstlight.py"
      -RenderOffscreen -unattended -nosplash -stdout

The receipt JSON is the deliverable; the beauty screenshot is the operator's
30-second RDP peek at the already-assembled FirstLight level.
"""

import json
import os
import sys

import unreal

STAGED = sys.argv[1] if len(sys.argv) > 1 else (
    r"G:\splatlab-ue\SplatLabUE56\SplatLabImports\splat_513e89171d\50a210ca9afe0019")
JOB_ID = sys.argv[2] if len(sys.argv) > 2 else "splat_513e89171d"
DEST = f"/Game/SplatLab/Jobs/{JOB_ID}"
RECEIPT_DIR = r"G:\splatlab-ue\receipts"

receipt = {"job_id": JOB_ID, "staged": STAGED, "imports": {}, "actors": {},
           "witness": {}, "errors": []}


def log(msg):
    unreal.log(f"SPLATLAB-FIRSTLIGHT: {msg}")


def import_file(path, dest, label):
    task = unreal.AssetImportTask()
    task.filename = path
    task.destination_path = dest
    task.automated = True
    task.save = True
    task.replace_existing = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = [str(p) for p in (task.imported_object_paths or [])]
    receipt["imports"][label] = paths
    log(f"import {label}: {paths}")
    return paths


def main():
    ply = os.path.join(STAGED, "Gaussian", "scene.ply")
    if not os.path.isfile(ply):
        raise RuntimeError(f"staged PLY missing: {ply}")

    splat_paths = import_file(ply, DEST, "gaussian-ply")
    if not splat_paths:
        raise RuntimeError(
            "NanoGS factory produced no asset from scene.ply — its importer "
            "likely needs an interactive session; fall back to the runbook")

    winner = os.path.join(STAGED, "Objects", "fire-hydrant",
                          "winner-textured.glb")
    winner_paths = []
    if os.path.isfile(winner):
        try:
            winner_paths = import_file(winner, DEST + "/Winner", "winner-glb")
        except Exception as exc:  # noqa: BLE001 - the splat is the point
            receipt["errors"].append(f"winner import: {exc}")

    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    level_path = f"{DEST}/FirstLight"
    # A crashed earlier run may have left the map asset: load it and rebuild
    # rather than failing on "already exists".
    if unreal.EditorAssetLibrary.does_asset_exist(level_path):
        if not les.load_level(level_path):
            raise RuntimeError(f"could not load existing level {level_path}")
        for actor in list(eas.get_all_level_actors()):
            if actor.get_actor_label() in (
                    "SplatLabSceneRoot", "GaussianRender", "ConventionalGeometry"):
                eas.destroy_actor(actor)
    elif not les.new_level(level_path):
        raise RuntimeError(f"could not create level {level_path}")

    # StaticMeshActor (no mesh) instead of a bare Actor: a Python-spawned
    # unreal.Actor has NO root component, so children cannot attach to it.
    root = eas.spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
    root.set_actor_label("SplatLabSceneRoot")

    keep = unreal.AttachmentRule.KEEP_WORLD
    splat_asset = unreal.load_asset(splat_paths[0])
    splat_actor = eas.spawn_actor_from_object(
        splat_asset, unreal.Vector(0, 0, 0))
    if splat_actor is None:
        raise RuntimeError(
            "spawn_actor_from_object returned None for the splat asset")
    splat_actor.set_actor_label("GaussianRender")
    splat_actor.attach_to_actor(root, "", keep, keep, keep, False)

    mesh_assets = [p for p in winner_paths if "/Winner" in p]
    if mesh_assets:
        try:
            mesh_asset = unreal.load_asset(mesh_assets[0])
            mesh_actor = eas.spawn_actor_from_object(
                mesh_asset, unreal.Vector(200, 0, 0))
            if mesh_actor:
                mesh_actor.set_actor_label("ConventionalGeometry")
                mesh_actor.attach_to_actor(root, "", keep, keep, keep, False)
        except Exception as exc:  # noqa: BLE001
            receipt["errors"].append(f"winner spawn: {exc}")

    # NUMERIC witness — sturdier than an eyeball: a hydrant is TALL, so the
    # up-axis is whichever world axis dominates its bounds; expected height is
    # ~0.85 m * 192.26 cm/unit calibration if the importer honoured units.
    origin, extent = splat_actor.get_actor_bounds(False)
    dims_cm = [2 * extent.x, 2 * extent.y, 2 * extent.z]
    axis_names = ["X", "Y", "Z"]
    up_guess = axis_names[dims_cm.index(max(dims_cm))]
    receipt["witness"] = {
        "bounds_origin_cm": [origin.x, origin.y, origin.z],
        "dims_cm": [round(v, 1) for v in dims_cm],
        "dominant_axis": up_guess,
        "upright_in_ue": up_guess == "Z",
        "expected_height_cm": 85.0 * 1.9226,
        "note": ("UE is +Z up. dominant_axis != Z means the importer changed "
                 "the up-axis: correct by rotating SplatLabSceneRoot only."),
    }
    receipt["actors"] = {
        "root": root.get_actor_label(),
        "children": [a.get_actor_label() for a in
                     eas.get_all_level_actors()
                     if a.get_attach_parent_actor() == root],
    }

    if not les.save_current_level():
        receipt["errors"].append("save_current_level returned False")
    unreal.EditorAssetLibrary.save_directory("/Game/SplatLab")
    log(f"witness: {json.dumps(receipt['witness'])}")


try:
    main()
    receipt["ok"] = not receipt["errors"]
except Exception as exc:  # noqa: BLE001 - the receipt must always land
    receipt["ok"] = False
    receipt["errors"].append(f"{type(exc).__name__}: {exc}")
    log(f"FAILED: {exc}")
finally:
    os.makedirs(RECEIPT_DIR, exist_ok=True)
    out = os.path.join(RECEIPT_DIR, f"firstlight-{JOB_ID}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2)
    log(f"receipt: {out} ok={receipt.get('ok')}")
    # -ExecutePythonScript leaves the editor running on some paths; a stuck
    # headless editor on a remote box is worse than a hard quit.
    unreal.SystemLibrary.quit_editor()
