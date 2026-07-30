"""Import CC0 asset-pack pieces into the library — normalized, self-validating.

Run under the pinned Blender, one pack at a time:

  ~/tools/blender-4.5.11-linux-x64/blender --background \
      --python assets/library/intake_pack.py -- \
      --pack kaykit-dungeon --prefix dungeon \
      --source-dir ~/Downloads/kaykit_dungeon/gltf \
      --license CC0-1.0 --license-url https://creativecommons.org/publicdomain/zero/1.0/ \
      --source-url https://kaylousberg.itch.io/kaykit-dungeon \
      --pieces wall,wall_corner,floor_tile_large [--scale 1.0]

Each piece is imported (glTF/GLB), stripped to geometry (no cameras, lights,
empties; a piece with an ARMATURE is refused — this intake is for STATIC
environment pieces, rigged actors wait for the actor lane), joined to ONE
mesh, transforms applied, re-origined to bottom-centre, exported Y-up to
`assets/library/{prefix}-{kebab}.glb`, and validated against the exact
starter contract (backend/glb_check: structure, single mesh, identity
transforms, bottom within 5 mm of y=0, extent inside a sanity band). A
failed piece aborts the run loudly — nothing half-lands.

Textures ARE allowed (unlike the flat-material starters — that line in
generate_starter_assets.py is that script's aesthetic, not a validator
contract): Kenney/KayKit palette atlases are tiny and ride inside the GLB.

Every exported piece is recorded in `assets/library/catalog.json`
(dev.splatlab.asset-catalog/v1) with its pack, source URL and license —
the license bookkeeping the flat directory itself cannot carry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "backend"))
import glb_check  # noqa: E402

CATALOG_PATH = HERE / "catalog.json"
CATALOG_SCHEMA = "dev.splatlab.asset-catalog/v1"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
EXTENT_MIN_M = 0.05
EXTENT_MAX_M = 30.0


def _kebab(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _reset() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _piece_stem(path: Path) -> str:
    """`wall.gltf.glb` (KayKit's double suffix) and `wall.glb` both -> wall."""
    stem = path.stem.lower()
    return stem[:-5] if stem.endswith(".gltf") else stem


def _find_source(source_dir: Path, piece: str) -> Path:
    """The piece's glTF/GLB file, found case-insensitively by stem."""
    want = piece.lower()
    hits = [p for p in source_dir.rglob("*")
            if p.suffix.lower() in (".glb", ".gltf") and _piece_stem(p) == want]
    if not hits:
        near = sorted({_piece_stem(p) for p in source_dir.rglob("*")
                       if p.suffix.lower() in (".glb", ".gltf")})
        raise SystemExit(f"piece '{piece}' not found under {source_dir} — "
                         f"stems available: {near[:40]}")
    if len(hits) > 1:
        # Prefer .glb (self-contained) over .gltf when both ship.
        glbs = [p for p in hits if p.suffix.lower() == ".glb"]
        hits = glbs or hits
    return hits[0]


def _intake_piece(src: Path, out: Path, scale: float) -> dict:
    _reset()
    bpy.ops.import_scene.gltf(filepath=str(src))

    if any(o.type == "ARMATURE" for o in bpy.data.objects):
        raise SystemExit(f"{src.name}: carries an ARMATURE — rigged pieces "
                         "wait for the actor lane; this intake is static "
                         "environment only")

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"{src.name}: no mesh objects after import")

    # Flatten the hierarchy: keep world transforms, drop non-mesh carriers.
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
    for obj in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if scale != 1.0:
        joined.data.transform(Matrix.Scale(scale, 4))

    # Re-origin: bottom-centre. Blender frame is Z-up here; export_yup makes
    # the exported floor y=0 exactly as the starter contract demands.
    xs = [v.co.x for v in joined.data.vertices]
    ys = [v.co.y for v in joined.data.vertices]
    zs = [v.co.z for v in joined.data.vertices]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    joined.data.transform(Matrix.Translation((-cx, -cy, -min(zs))))
    joined.location = (0.0, 0.0, 0.0)
    joined.rotation_euler = (0.0, 0.0, 0.0)
    joined.scale = (1.0, 1.0, 1.0)

    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB",
                              export_yup=True, export_apply=True)

    # The starter contract, verbatim.
    glb_check.validate_glb(out)
    summary = glb_check.validate_glb(out)
    if summary["meshes"] != 1:
        raise SystemExit(f"{out.name}: {summary['meshes']} meshes — "
                         "the library contract is ONE joined mesh")
    bounds = glb_check.position_bounds(out)
    if not bounds["identity_transforms"]:
        raise SystemExit(f"{out.name}: node transforms survived export "
                         f"({bounds['transformed_nodes']})")
    if abs(bounds["aabb"]["min"][1]) > 0.005:
        raise SystemExit(f"{out.name}: bottom at y={bounds['aabb']['min'][1]} "
                         "— origin must be bottom-centre")
    extent = bounds["extent"]
    if not (EXTENT_MIN_M <= max(extent) <= EXTENT_MAX_M):
        raise SystemExit(f"{out.name}: extent {extent} outside the "
                         f"[{EXTENT_MIN_M}, {EXTENT_MAX_M}] m sanity band — "
                         "wrong --scale?")
    return {"bytes": out.stat().st_size, "extent": extent}


def _merge_catalog(entries: dict[str, dict]) -> None:
    catalog = {"schema": CATALOG_SCHEMA, "assets": {}}
    if CATALOG_PATH.is_file():
        catalog = json.loads(CATALOG_PATH.read_text())
        if catalog.get("schema") != CATALOG_SCHEMA:
            raise SystemExit(f"catalog schema is {catalog.get('schema')!r}, "
                             f"expected {CATALOG_SCHEMA!r} — refusing to merge")
    catalog.setdefault("assets", {}).update(entries)
    staged = CATALOG_PATH.with_suffix(".json.staging")
    staged.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    staged.replace(CATALOG_PATH)


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True, help="pack id for the catalog")
    ap.add_argument("--prefix", required=True,
                    help="library filename prefix ({prefix}-{piece}.glb)")
    ap.add_argument("--source-dir", required=True, type=Path)
    ap.add_argument("--license", required=True)
    ap.add_argument("--license-url", required=True)
    ap.add_argument("--source-url", required=True)
    ap.add_argument("--pieces", required=True,
                    help="comma-separated source stems to import")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(argv)

    source_dir = args.source_dir.expanduser()
    if not source_dir.is_dir():
        raise SystemExit(f"--source-dir {source_dir} is not a directory")

    stamped = datetime.now(timezone.utc).isoformat(timespec="seconds")
    catalog_entries: dict[str, dict] = {}
    for piece in [p.strip() for p in args.pieces.split(",") if p.strip()]:
        name = f"{args.prefix}-{_kebab(piece)}"
        if not NAME_RE.match(name):
            raise SystemExit(f"'{name}' violates the library naming contract")
        src = _find_source(source_dir, piece)
        out = HERE / f"{name}.glb"
        receipt = _intake_piece(src, out, args.scale)
        catalog_entries[name] = {
            "pack": args.pack,
            "source_url": args.source_url,
            "license": args.license,
            "license_url": args.license_url,
            "imported_at": stamped,
        }
        extent = "x".join(f"{v:.2f}" for v in receipt["extent"])
        print(f"OK {out.name}: {receipt['bytes']} bytes, {extent} m")

    _merge_catalog(catalog_entries)
    print(f"CATALOG: {len(catalog_entries)} entr{'y' if len(catalog_entries) == 1 else 'ies'} "
          f"merged into {CATALOG_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
