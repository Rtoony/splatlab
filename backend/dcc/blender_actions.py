"""Blender-embedded implementation of SplatLab's whitelisted DCC actions."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import architecture


ALLOWED_ACTIONS = {
    "inspect",
    "snapshot",
    "toggle_collection",
    "transform_object",
    "import_world_element",
    "import_asset",
    "cleanup_mesh",
    "export_glb",
    "create_wall",
    "cut_opening",
    "create_room",
    "assign_material",
}


def _drop_small_islands(bm: bmesh.types.BMesh, min_frac: float) -> int:
    """Delete connected components below min_frac of total faces.

    The largest island is always kept, so a degenerate threshold can never
    empty the mesh."""
    total_faces = len(bm.faces)
    if total_faces == 0:
        return 0
    visited: set[int] = set()
    islands: list[list[bmesh.types.BMVert]] = []
    bm.verts.index_update()
    for seed in bm.verts:
        if seed.index in visited:
            continue
        stack, component = [seed], []
        visited.add(seed.index)
        while stack:
            vert = stack.pop()
            component.append(vert)
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if other.index not in visited:
                    visited.add(other.index)
                    stack.append(other)
        islands.append(component)
    face_counts = [
        len({face.index for vert in component for face in vert.link_faces})
        for component in islands
    ]
    largest = max(range(len(islands)), key=lambda i: face_counts[i])
    doomed_verts: list[bmesh.types.BMVert] = []
    removed = 0
    for index, component in enumerate(islands):
        if index == largest:
            continue
        if face_counts[index] < min_frac * total_faces:
            doomed_verts.extend(component)
            removed += 1
    if doomed_verts:
        bmesh.ops.delete(bm, geom=doomed_verts, context="VERTS")
    return removed


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
                    math.degrees(value) for value in obj.matrix_basis.to_euler("XYZ")
                ],
                "rotation_mode": obj.rotation_mode,
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

    if action in {"create_wall", "create_room"}:
        name = f"architecture_{params['name']}"
        if bpy.data.objects.get(name) is not None:
            raise ValueError(f"Architectural object already exists: {name}")
        if abs(bpy.context.scene.unit_settings.scale_length - 1.0) > 1e-6:
            raise ValueError("Architecture requires a metre-scaled Blender scene (scale_length=1)")
        fields = ("width", "height", "thickness") if action == "create_wall" else ("width", "depth", "height", "thickness", "door_width", "door_height", "ceiling")
        dimensions = {key: params[key] for key in fields}
        boxes = architecture.wall_boxes(**dimensions) if action == "create_wall" else architecture.room_boxes(**dimensions)
        vertices, faces = architecture.box_mesh(boxes)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        collection = bpy.data.collections.get("Architecture")
        if collection is None:
            collection = bpy.data.collections.new("Architecture")
            bpy.context.scene.collection.children.link(collection)
        collection.objects.link(obj)
        obj.location = params["origin"]
        obj.rotation_euler.z = math.radians(params["rotation_degrees"])
        obj["splatlab_provenance"] = "authored"
        obj["splatlab_architecture"] = json.dumps({"kind": action, **dimensions})
        result = {"object": name, "dimensions_m": dimensions, "provenance": "authored", "frame": "blender-z-up-metres", "origin_contract": "front-wall bottom centre; room extends along local +Y"}
    elif action == "cut_opening":
        obj = bpy.data.objects.get(params["object"])
        if obj is None or obj.type != "MESH" or not obj.get("splatlab_architecture"):
            raise ValueError("Opening requires an authored structural wall; captured collision is not visible architecture")
        structure = json.loads(obj["splatlab_architecture"])
        if structure["kind"] != "create_wall":
            raise ValueError("Select an authored wall, not a room or captured mesh")
        if any(abs(value - 1) > 1e-6 for value in obj.scale):
            raise ValueError("Opening dimensions require an unscaled structural wall")
        opening = {key: params[key] for key in ("width", "height", "offset", "sill")}
        boxes = architecture.wall_boxes(structure["width"], structure["height"], structure["thickness"], opening)
        vertices, faces = architecture.box_mesh(boxes)
        mesh = bpy.data.meshes.new(obj.name + "-opening")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        for material in obj.data.materials:
            mesh.materials.append(material)
        obj.data = mesh
        obj["splatlab_architecture"] = json.dumps({**structure, "opening": opening})
        result = {"object": obj.name, "opening_m": opening, "provenance": "authored"}
    elif action == "assign_material":
        obj = bpy.data.objects.get(params["object"])
        if obj is None or obj.type != "MESH":
            raise ValueError("Material assignment requires a mesh object")
        material = bpy.data.materials.new(obj.name + "-material")
        material.diffuse_color = (*params["color"], 1)
        material.use_nodes = True
        shader = material.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*params["color"], 1)
        shader.inputs["Roughness"].default_value = params["roughness"]
        shader.inputs["Metallic"].default_value = params["metallic"]
        obj.data = obj.data.copy()
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0
        result = {"object": obj.name, "material": material.name, "appearance": "authored-pbr-not-relighting"}
    elif action == "toggle_collection":
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
            obj.rotation_mode = "XYZ"
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
    elif action == "import_world_element":
        # The path is host-resolved and containment-checked; params carry only
        # the sanitized slug used for naming.
        import_path = request.get("import_glb")
        if not isinstance(import_path, str) or not import_path.endswith(".glb"):
            raise ValueError("import_world_element requires a host-resolved GLB")
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=import_path)
        imported = [obj for obj in bpy.data.objects if obj not in before]
        meshes = [obj for obj in imported if obj.type == "MESH"]
        if not meshes:
            raise ValueError("imported GLB contained no mesh objects")
        primary = max(meshes, key=lambda obj: len(obj.data.polygons))
        canonical = f"polish_{params['slug']}"
        superseded = None
        existing = bpy.data.objects.get(canonical)
        if existing is not None and existing not in imported:
            # Blender uniquifies the ASSIGNEE on name collision (.001), which
            # would hand the documented name to the STALE import from an
            # earlier version — rename the old object aside so the fresh
            # import owns the contract name.
            existing.name = f"{canonical}.superseded"
            superseded = existing.name
        primary.name = canonical
        polish_collection = bpy.data.collections.get("Polish")
        if polish_collection is None:
            polish_collection = bpy.data.collections.new("Polish")
            bpy.context.scene.collection.children.link(polish_collection)
        for obj in imported:
            for collection in list(obj.users_collection):
                collection.objects.unlink(obj)
            polish_collection.objects.link(obj)
        result = {
            "object": primary.name,
            "imported": sorted(obj.name for obj in imported),
            "faces": len(primary.data.polygons),
            "materials": len(primary.data.materials),
            "superseded": superseded,
        }
    elif action == "import_asset":
        # Same host-resolved-path contract as import_world_element, but the
        # source is the curated library and the library contract is ONE mesh.
        import_path = request.get("import_glb")
        if not isinstance(import_path, str) or not import_path.endswith(".glb"):
            raise ValueError("import_asset requires a host-resolved GLB")
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=import_path)
        imported = [obj for obj in bpy.data.objects if obj not in before]
        meshes = [obj for obj in imported if obj.type == "MESH"]
        if len(meshes) != 1:
            raise ValueError(
                f"library asset must be ONE joined mesh, got {len(meshes)} — "
                "regenerate it with assets/library/generate_starter_assets.py"
            )
        primary = meshes[0]
        canonical = f"asset_{params['slug']}"
        superseded = None
        existing = bpy.data.objects.get(canonical)
        if existing is not None and existing not in imported:
            # Same collision dance as import_world_element: the fresh import
            # must own the contract name, never a .001 alias.
            existing.name = f"{canonical}.superseded"
            superseded = existing.name
        primary.name = canonical
        placed_collection = bpy.data.collections.get("Placed")
        if placed_collection is None:
            placed_collection = bpy.data.collections.new("Placed")
            bpy.context.scene.collection.children.link(placed_collection)
        for obj in imported:
            for collection in list(obj.users_collection):
                collection.objects.unlink(obj)
            placed_collection.objects.link(obj)
        result = {
            "object": primary.name,
            "faces": len(primary.data.polygons),
            "materials": len(primary.data.materials),
            # Blender units (metres) — what the operator scales honestly
            # against inspect_job's coordinate record before placing.
            "dimensions": [round(value, 4) for value in primary.dimensions],
            "superseded": superseded,
        }
    elif action == "cleanup_mesh":
        obj = bpy.data.objects.get(params["object"])
        if obj is None or obj.type != "MESH":
            raise ValueError(f"mesh object not found: {params['object']!r}")
        mesh = obj.data
        faces_before = len(mesh.polygons)
        verts_before = len(mesh.vertices)
        components_removed = 0
        if "merge_distance" in params or "min_component_frac" in params:
            bm = bmesh.new()
            bm.from_mesh(mesh)
            if "merge_distance" in params:
                bmesh.ops.remove_doubles(
                    bm, verts=list(bm.verts), dist=params["merge_distance"]
                )
            if "min_component_frac" in params:
                components_removed = _drop_small_islands(
                    bm, params["min_component_frac"]
                )
            bm.to_mesh(mesh)
            bm.free()
        if "decimate_ratio" in params and params["decimate_ratio"] < 1.0:
            modifier = obj.modifiers.new(name="splatlab_decimate", type="DECIMATE")
            modifier.ratio = params["decimate_ratio"]
            with bpy.context.temp_override(
                object=obj, active_object=obj, selected_objects=[obj]
            ):
                bpy.ops.object.modifier_apply(modifier=modifier.name)
        if params.get("shade_smooth"):
            mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.update()
        result = {
            "object": obj.name,
            "faces_before": faces_before,
            "faces_after": len(mesh.polygons),
            "verts_before": verts_before,
            "verts_after": len(mesh.vertices),
            "components_removed": components_removed,
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
        export_kwargs = {
            "filepath": str(output_path),
            "export_format": "GLB",
            "export_apply": True,
            "export_extras": True,
            "export_yup": True,
        }
        selected_name = params.get("object")
        if selected_name:
            target = bpy.data.objects.get(selected_name)
            if target is None:
                raise ValueError(f"object not found: {selected_name!r}")
            if params.get("bake_world_transform"):
                # The shared-frame contract: world-baked vertices under an
                # identity node. In-memory only — the export path never saves
                # the .blend, so the authored version keeps its transform.
                if target.parent is not None:
                    raise ValueError(
                        "bake_world_transform refuses parented objects — the "
                        "parent's transform would be silently half-baked"
                    )
                matrix = target.matrix_world.copy()
                target.data.transform(matrix)
                target.matrix_world.identity()
            for obj in bpy.context.view_layer.objects:
                obj.select_set(False)
            target.select_set(True)
            export_kwargs["use_selection"] = True
        bpy.ops.export_scene.gltf(**export_kwargs)
        return {
            "exported": True,
            "objects": 1 if selected_name else len(bpy.data.objects),
            "meshes": len(bpy.data.meshes),
            "world_transform_baked": bool(params.get("bake_world_transform")),
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
