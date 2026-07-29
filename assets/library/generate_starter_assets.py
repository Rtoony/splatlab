"""Generate the starter fantasy asset library — deterministic, self-validating.

Run (once, or after editing) under the pinned Blender:

  ~/tools/blender-4.5.11-linux-x64/blender --background \
      --python assets/library/generate_starter_assets.py

Outputs land beside this script and are COMMITTED — the demo ships with zero
external downloads. Contract per asset (what import_asset and the create-door
rely on):

  - metre scale, Y-up in the exported GLB (export_yup)
  - origin at bottom-centre: a `transform_object` location IS the floor point
  - ONE joined mesh object (multi-material is fine; multi-object is not)
  - identity node transforms (all object transforms applied before export)
  - flat baseColorFactor materials, no textures — small files, restyle-able
  - self-validated with backend/glb_check (structure, single mesh, identity,
    plausible height) before the script reports success
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
import glb_check  # noqa: E402


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _material(name: str, rgba: tuple[float, float, float, float],
              rough: float = 0.8, metal: float = 0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return material


def _bake_transform() -> None:
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def _assign(obj, material) -> None:
    obj.data.materials.append(material)


def _cylinder(radius: float, depth: float, location, material, *,
              rotation=(0.0, 0.0, 0.0), vertices: int = 24):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                        vertices=vertices, location=location,
                                        rotation=rotation)
    obj = bpy.context.active_object
    _assign(obj, material)
    _bake_transform()
    return obj


def _cone(r1: float, r2: float, depth: float, location, material, *,
          vertices: int = 24):
    bpy.ops.mesh.primitive_cone_add(radius1=r1, radius2=r2, depth=depth,
                                    vertices=vertices, location=location)
    obj = bpy.context.active_object
    _assign(obj, material)
    _bake_transform()
    return obj


def _box(size, location, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.scale = (size[0], size[1], size[2])
    _assign(obj, material)
    _bake_transform()
    return obj


def _join_export(name: str, parts) -> Path:
    bpy.ops.object.select_all(action="DESELECT")
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = name
    _bake_transform()  # belt and braces: the export must carry identity nodes
    out = HERE / f"{name}.glb"
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB",
                              export_yup=True, export_apply=True)
    return out


def torch_sconce() -> Path:
    """A wall/post torch, ~0.45 m tall. Toggle-tint it to 'light' it."""
    _reset()
    wood = _material("torch-wood", (0.23, 0.14, 0.07, 1.0), rough=0.9)
    iron = _material("torch-iron", (0.15, 0.15, 0.17, 1.0), rough=0.5, metal=0.8)
    flame = _material("torch-head", (0.85, 0.45, 0.12, 1.0), rough=0.7)
    handle = _cylinder(0.022, 0.34, (0.0, 0.0, 0.17), wood)
    ring = _cylinder(0.034, 0.02, (0.0, 0.0, 0.30), iron)
    head = _cone(0.055, 0.028, 0.10, (0.0, 0.0, 0.39), flame)
    return _join_export("torch-sconce", [handle, ring, head])


def treasure_chest() -> Path:
    """A lidded chest, ~0.6 x 0.4 x 0.45 m. Toggle open/closed via states."""
    _reset()
    wood = _material("chest-wood", (0.35, 0.22, 0.11, 1.0), rough=0.85)
    dark = _material("chest-band", (0.16, 0.10, 0.06, 1.0), rough=0.8)
    gold = _material("chest-lock", (0.72, 0.55, 0.18, 1.0), rough=0.35,
                     metal=0.9)
    body = _box((0.60, 0.40, 0.28), (0.0, 0.0, 0.14), wood)
    lid = _cylinder(0.20, 0.60, (0.0, 0.0, 0.28), dark,
                    rotation=(0.0, 1.5707963, 0.0))
    lock = _box((0.08, 0.03, 0.10), (0.0, -0.21, 0.26), gold)
    return _join_export("treasure-chest", [body, lid, lock])


def wooden_signpost() -> Path:
    """A post with a plank, ~1.2 m tall. Inspect-text carrier."""
    _reset()
    post_wood = _material("signpost-post", (0.28, 0.18, 0.10, 1.0), rough=0.9)
    plank_wood = _material("signpost-plank", (0.45, 0.32, 0.18, 1.0),
                           rough=0.85)
    post = _cylinder(0.035, 1.10, (0.0, 0.0, 0.55), post_wood, vertices=16)
    plank = _box((0.55, 0.05, 0.18), (0.10, 0.0, 0.95), plank_wood)
    return _join_export("wooden-signpost", [post, plank])


EXPECTED_HEIGHTS_M = {  # (min, max) sanity band for the Y (up) extent
    "torch-sconce": (0.40, 0.50),
    "treasure-chest": (0.40, 0.55),
    "wooden-signpost": (1.05, 1.25),
}


def main() -> int:
    results = []
    for build in (torch_sconce, treasure_chest, wooden_signpost):
        out = build()
        summary = glb_check.validate_glb(out)
        if summary["meshes"] != 1:
            raise SystemExit(f"{out.name}: {summary['meshes']} meshes — "
                             "the library contract is ONE joined mesh")
        bounds = glb_check.position_bounds(out)
        if not bounds["identity_transforms"]:
            raise SystemExit(f"{out.name}: node transforms survived export "
                             f"({bounds['transformed_nodes']})")
        lo, hi = EXPECTED_HEIGHTS_M[out.stem]
        height = bounds["extent"][1]  # Y is up in the exported GLB
        if not lo <= height <= hi:
            raise SystemExit(f"{out.stem}: height {height} m outside "
                             f"[{lo}, {hi}]")
        floor = bounds["aabb"]["min"][1]
        if abs(floor) > 0.005:
            raise SystemExit(f"{out.stem}: bottom at y={floor} — origin must "
                             "be bottom-centre")
        results.append((out.name, out.stat().st_size, height))
    for name, size, height in results:
        print(f"OK {name}: {size} bytes, {height:.3f} m tall", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
