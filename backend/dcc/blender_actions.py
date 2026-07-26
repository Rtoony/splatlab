"""Blender-embedded implementation of SplatLab's whitelisted DCC actions."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bpy


ALLOWED_ACTIONS = {
    "inspect",
    "snapshot",
    "toggle_collection",
    "transform_object",
    "export_glb",
}


def _inspect() -> dict:
    collections = [
        {
            "name": collection.name,
            "visible_viewport": not collection.hide_viewport,
            "visible_render": not collection.hide_render,
            "objects": len(collection.objects),
        }
        for collection in sorted(bpy.data.collections, key=lambda item: item.name)
    ]
    objects = []
    for obj in sorted(bpy.data.objects, key=lambda item: item.name)[:1000]:
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "location": list(obj.location),
                "rotation_degrees": [
                    math.degrees(value) for value in obj.rotation_euler
                ],
                "scale": list(obj.scale),
                "visible_viewport": not obj.hide_viewport,
                "visible_render": not obj.hide_render,
                "collections": [collection.name for collection in obj.users_collection],
            }
        )
    scene = bpy.context.scene
    return {
        "file": bpy.data.filepath,
        "scene": scene.name,
        "units": {
            "system": scene.unit_settings.system,
            "scale_length": scene.unit_settings.scale_length,
            "length_unit": scene.unit_settings.length_unit,
        },
        "collections": collections,
        "objects": objects,
        "object_count": len(bpy.data.objects),
        "collection_count": len(bpy.data.collections),
    }


def _execute(request: dict) -> dict:
    action = request.get("action")
    params = request.get("params") or {}
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"action is not allowed: {action!r}")

    if action == "toggle_collection":
        collection = bpy.data.collections.get(params["collection"])
        if collection is None:
            raise ValueError(f"collection not found: {params['collection']!r}")
        visible = bool(params["visible"])
        collection.hide_viewport = not visible
        collection.hide_render = not visible
        result = {"collection": collection.name, "visible": visible}
    elif action == "transform_object":
        obj = bpy.data.objects.get(params["object"])
        if obj is None:
            raise ValueError(f"object not found: {params['object']!r}")
        if "location" in params:
            obj.location = params["location"]
        if "rotation_degrees" in params:
            obj.rotation_euler = [
                math.radians(value) for value in params["rotation_degrees"]
            ]
        if "scale" in params:
            obj.scale = params["scale"]
        result = {
            "object": obj.name,
            "location": list(obj.location),
            "rotation_degrees": [math.degrees(value) for value in obj.rotation_euler],
            "scale": list(obj.scale),
        }
    elif action == "inspect":
        return _inspect()
    elif action == "export_glb":
        # Read-only with respect to the .blend: exports the scene as binary
        # glTF and returns without saving a new version. Zero free-form
        # params — the output path comes from the host-side request only.
        output = request.get("output_glb")
        if not isinstance(output, str) or not output.endswith(".glb"):
            raise ValueError("export_glb requires a .glb output path")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Same export contract as blender_assemble.py: apply modifiers (the
        # assembled scene's splat geometry lives behind Geometry Nodes —
        # without export_apply the glTF has zero meshes, caught by the
        # readback gate on the first real run), keep provenance extras, Y-up.
        bpy.ops.export_scene.gltf(
            filepath=str(output_path),
            export_format="GLB",
            export_apply=True,
            export_extras=True,
            export_yup=True,
        )
        return {
            "exported": True,
            "objects": len(bpy.data.objects),
            "meshes": len(bpy.data.meshes),
            "bytes": output_path.stat().st_size,
        }
    else:
        result = {"snapshot": True}

    output = request.get("output_blend")
    if not isinstance(output, str) or not output.endswith(".blend"):
        raise ValueError("mutating actions require a .blend output path")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene["splatlab_job_id"] = request.get("job_id", "")
    bpy.context.scene["splatlab_last_action"] = action
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path), check_existing=False)
    return result


def main() -> int:
    try:
        separator = sys.argv.index("--")
        request_path = Path(sys.argv[separator + 1])
        response_path = Path(sys.argv[separator + 2])
        request = json.loads(request_path.read_text())
        result = _execute(request)
        response = {
            "status": "ok",
            "result": result,
            "blender": {
                "version": bpy.app.version_string,
                "background": bpy.app.background,
            },
        }
        response_path.write_text(json.dumps(response, indent=2))
        return 0
    except Exception as exc:
        traceback.print_exc()
        if "response_path" in locals():
            response_path.write_text(
                json.dumps({"status": "error", "error": str(exc)}, indent=2)
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
