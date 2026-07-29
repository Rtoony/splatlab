#!/usr/bin/env python3
"""Bake a restyle permanently into element atlases — the W3 fantasy lane.

The walker's restyle preview (frontend/src/lib/world-restyle.ts) shades each
restyled element with a taxonomy-class tile projected triplanarly in world
space, optionally multiplied by a tint. This module writes that exact look
into the element's OWN atlas so it survives export to Blender, the UE bundle,
and a fresh walker load with no overlay document at all.

Frame contract (deliberate, and different from the semantic-lane composite):
the shipped `_world/*.glb` is already the frame the preview shades in — the
GLB's own Y-up coordinates, metres on calibrated worlds. So this bake loads
the CURRENT GLB (which may be a Blender polish with its own UV charts),
rasterizes texel -> 3D in that frame, and samples tiles right there. No
capture-frame inversion exists here on purpose; `--units-per-metre` is the
walker's one scale dial, converting the document's metres-per-tile into GLB
units exactly like `triplanarMaterial(tile, metresPerTile * unitsPerMetre)`.

What is baked: `tint` (multiplied in LINEAR space, matching three.js) and
`material`(+`material_scale`). `lighting` is NEVER baked — the atlas already
contains the capture's light, and a preset is a three.js rig, not surface
data. The route owns that refusal; this script only sees element entries.

All-or-nothing: every requested element must bake into --staging-dir or the
run exits nonzero and the caller discards staging. Nothing here ever touches
a live file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from class_textures import sample_world_triplanar, tile_for  # noqa: E402

DEFAULT_TILE_SIZE = 512


# ── pure helpers (numpy only — unit-tested in the app env) ─────────────────────


def srgb_to_linear(u8: np.ndarray) -> np.ndarray:
    """sRGB bytes -> linear float32 0..1 (the decode three.js applies to maps
    and hex colours before multiplying them)."""
    c = np.asarray(u8, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(f: np.ndarray) -> np.ndarray:
    """Linear float 0..1 -> sRGB bytes, rounded."""
    f = np.clip(np.asarray(f, np.float32), 0.0, 1.0)
    s = np.where(f <= 0.0031308, f * 12.92,
                 1.055 * np.power(f, 1.0 / 2.4) - 0.055)
    return (np.clip(s, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def hex_to_linear_rgb(tint: str) -> np.ndarray:
    """#rrggbb -> linear float32 [3], the way `new THREE.Color(hex)` decodes."""
    h = tint.lstrip("#")
    rgb = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
                   dtype=np.uint8)
    return srgb_to_linear(rgb)


def texels_per_glb_unit(material_scale_m: float, units_per_metre: float,
                        tile_px: int) -> float:
    """One tile repeat spans material_scale_m × units_per_metre GLB units —
    the walker's `unitsPerTile`, with the shader's own 1e-4 floor."""
    units_per_tile = max(1e-4, float(material_scale_m) * float(units_per_metre))
    return float(tile_px) / units_per_tile


def tint_atlas(rgb_u8: np.ndarray, tint: str) -> np.ndarray:
    """Multiply a tint into sRGB texels in LINEAR space — what the preview's
    `material.color` multiply actually does on screen."""
    return linear_to_srgb(srgb_to_linear(rgb_u8) * hex_to_linear_rgb(tint))


def floor_clamp_painted(rgb_u8: np.ndarray, painted: np.ndarray) -> np.ndarray:
    """world_gate's texture_coverage counts exactly-black texels as
    background, so a painted texel must never land on (0,0,0) — a near-black
    tint or class (obsidian) would silently degrade the gate. 1/255 is
    invisible; genuinely empty texels stay 0."""
    out = np.asarray(rgb_u8).copy()
    out[np.asarray(painted, bool) & (out.max(axis=-1) == 0)] = 1
    return out


def bake_plan(restyle_doc: dict | None, only: set[str] | None = None) -> list[dict]:
    """Which elements to bake, from a validated restyle document. Lighting is
    not this script's business. Raises ValueError when nothing qualifies —
    an empty bake claiming success would be a lie."""
    entries = (restyle_doc or {}).get("elements") or {}
    plan: list[dict] = []
    for slug in sorted(entries):
        entry = entries[slug] or {}
        if only is not None and slug not in only:
            continue
        tint = entry.get("tint")
        material = entry.get("material")
        if not tint and not material:
            continue
        plan.append({
            "slug": slug,
            "tint": tint,
            "material": material,
            # The preview's default: metresPerTile = material_scale ?? 1.
            "material_scale": float(entry.get("material_scale", 1.0)),
        })
    if not plan:
        raise ValueError(
            "nothing to bake — no element entry carries a tint or material"
            + (f" (after --slugs filter {sorted(only)})" if only else ""))
    return plan


# ── the bake (mesh env: trimesh + PIL) ─────────────────────────────────────────


def _load_single_mesh(path: Path):
    """The current element GLB as ONE world-frame mesh, geometry untouched.

    force="mesh" applies node transforms (a Blender polish may carry a TRS)
    and flattens — re-exporting flat with identity nodes preserves the
    on-screen geometry and conforms to the shared-frame contract. A GLB with
    more than one mesh is refused: flattening would scramble which atlas
    belongs to which primitive.
    """
    import trimesh

    raw = trimesh.load(str(path), process=False)
    if isinstance(raw, trimesh.Scene) and len(raw.geometry) != 1:
        raise ValueError(
            f"{path.name} holds {len(raw.geometry)} meshes — bake needs a "
            "single-mesh GLB (re-export the polish as one object)")
    return trimesh.load(str(path), force="mesh", process=False)


def _material_factor(material, name: str, default: float) -> float:
    value = getattr(material, name, None)
    return default if value is None else float(value)


def bake_element(world_dir: Path, item: dict, staging_dir: Path, *,
                 units_per_metre: float, tile_size: int,
                 taxonomy_by_id: dict) -> dict:
    """Bake one element into staging. Returns its report entry; raises on any
    refusal (missing GLB, no UVs, no atlas, multi-mesh, no texels)."""
    import object_texture as ot
    from PIL import Image
    import trimesh

    t0 = time.time()
    slug = item["slug"]
    rel = "shell.glb" if slug == "shell" else f"elements/{slug}.glb"
    target = world_dir / rel
    if not target.is_file():
        raise ValueError(f"{slug}: no built GLB at _world/{rel} — solidify first")

    mesh = _load_single_mesh(target)
    uvs = getattr(mesh.visual, "uv", None)
    if uvs is None or len(uvs) != len(mesh.vertices):
        raise ValueError(
            f"{slug}: GLB has no UV charts (vertex-colour fallback build?) — "
            "re-solidify this element with texturing before baking a restyle")
    material_obj = getattr(mesh.visual, "material", None)
    src_img = getattr(material_obj, "baseColorTexture", None)
    if src_img is None:
        raise ValueError(f"{slug}: GLB has no baseColorTexture to restyle")
    atlas = np.asarray(src_img.convert("RGB"), dtype=np.uint8)
    if atlas.ndim != 3 or atlas.shape[0] != atlas.shape[1]:
        raise ValueError(f"{slug}: non-square atlas {atlas.shape} — unsupported")
    size = int(atlas.shape[0])

    verbs: list[str] = []
    texels_per_unit = None
    if item["material"]:
        cls = taxonomy_by_id.get(item["material"])
        if cls is None:
            raise ValueError(f"{slug}: unknown material class {item['material']!r}")
        verbs.append("material")
        texels_per_unit = texels_per_glb_unit(
            item["material_scale"], units_per_metre, tile_size)

        verts = np.asarray(mesh.vertices, np.float32)
        faces = np.asarray(mesh.faces, np.int64)
        vn = np.asarray(mesh.vertex_normals, np.float32)
        px, py, pos, nrm = ot._rasterize_atlas(
            verts, faces, np.asarray(uvs, np.float32) * (size - 1), size, vn)
        if pos is None:
            raise ValueError(f"{slug}: atlas rasterization covered no texels")
        # Last-writer-wins dedup, identical to _composite_class_texels — two
        # faces sharing a texel must resolve the way every bake resolves them.
        winner = np.full(size * size, -1, dtype=np.int64)
        winner[py.astype(np.int64) * size + px] = np.arange(len(px))
        keep = winner[winner >= 0]
        px, py, pos, nrm = px[keep], py[keep], pos[keep], nrm[keep]

        tile = tile_for(cls, tile_size)
        rgb = sample_world_triplanar(tile, pos, nrm, texels_per_unit)
        if item["tint"]:
            verbs.append("tint")
            rgb_u8 = tint_atlas(np.clip(rgb + 0.5, 0, 255).astype(np.uint8),
                                item["tint"])
        else:
            rgb_u8 = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)

        # A FRESH atlas: chart gutters must carry the new surface, not stale
        # capture rims bleeding back in at the seams.
        new = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=bool)
        new[py, px] = rgb_u8
        mask[py, px] = True
        new, shipped = ot.dilate_atlas(new, mask, passes=max(16, size // 48))
        new = floor_clamp_painted(new, shipped)
        covered_frac = float(shipped.mean())
        gen = cls.get("generative") or {}
        # The preview material's own defaults (triplanarMaterial).
        out_rough = float(gen.get("roughness", 0.85))
        out_metal = float(gen.get("metallic", 0.0))
    else:
        verbs.append("tint")
        painted = atlas.max(axis=2) > 0
        new = floor_clamp_painted(tint_atlas(atlas, item["tint"]), painted)
        covered_frac = float(painted.mean())
        # Tint-only keeps the source material's own response.
        out_rough = _material_factor(material_obj, "roughnessFactor", 0.75)
        out_metal = _material_factor(material_obj, "metallicFactor", 0.0)

    image = Image.fromarray(new, mode="RGB")
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                          faces=np.asarray(mesh.faces), process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(uvs),
        image=image,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=image,
            metallicFactor=out_metal,
            roughnessFactor=out_rough,
        ),
    )
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_glb = staging_dir / f"{slug}.glb"
    out.export(str(staged_glb))

    back = trimesh.load(str(staged_glb), force="mesh")
    if len(back.faces) != len(mesh.faces):
        staged_glb.unlink(missing_ok=True)
        raise ValueError(
            f"{slug}: GLB readback mismatch ({len(back.faces)} != "
            f"{len(mesh.faces)}) — geometry must survive a texture bake")
    has_uv = getattr(getattr(back.visual, "uv", None), "shape", None) is not None
    texture_back = getattr(
        getattr(back.visual, "material", None), "baseColorTexture", None)
    if not has_uv or texture_back is None:
        staged_glb.unlink(missing_ok=True)
        raise ValueError(f"{slug}: GLB readback lost UVs or texture")
    # Sidecar atlas only after the readback gate (generated_texture rule) —
    # a failed bake must not leave an orphan PNG for the gate to grade.
    staged_atlas = staging_dir / f"{slug}_atlas.png"
    image.save(staged_atlas)

    return {
        "slug": slug,
        "ok": True,
        "verbs": verbs,
        "material": item["material"],
        "material_scale": item["material_scale"] if item["material"] else None,
        "tint": item["tint"],
        "texels_per_unit": (round(texels_per_unit, 4)
                            if texels_per_unit is not None else None),
        "atlas_size": size,
        "covered_frac": round(covered_frac, 4),
        "faces": int(len(mesh.faces)),
        "roughness": out_rough,
        "metallic": out_metal,
        "staged_glb": staged_glb.name,
        "staged_atlas": staged_atlas.name,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--staging-dir", type=Path, required=True,
                    help="where staged GLBs/atlases land; caller owns cleanup")
    ap.add_argument("--units-per-metre", type=float, required=True,
                    help="the walker's scale dial (1.0 on calibrated worlds)")
    ap.add_argument("--slugs", default=None,
                    help="comma-separated subset of restyled elements")
    ap.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    args = ap.parse_args()

    world_dir = args.job_dir / "_world"
    restyle_path = world_dir / "restyle.json"
    if not restyle_path.is_file():
        print(json.dumps({"ok": False, "error": "no _world/restyle.json"}))
        return 2
    doc = json.loads(restyle_path.read_text())

    import class_taxonomy
    taxonomy_by_id = class_taxonomy.class_by_id(
        class_taxonomy.load_taxonomy(args.job_dir))

    only = ({s.strip() for s in args.slugs.split(",") if s.strip()}
            if args.slugs else None)
    try:
        plan = bake_plan(doc, only)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2

    elements: list[dict] = []
    all_ok = True
    for item in plan:
        try:
            elements.append(bake_element(
                world_dir, item, args.staging_dir,
                units_per_metre=args.units_per_metre,
                tile_size=args.tile_size,
                taxonomy_by_id=taxonomy_by_id))
        except Exception as exc:  # noqa: BLE001 — every element gets reported
            all_ok = False
            elements.append({"slug": item["slug"], "ok": False,
                             "error": str(exc)})

    print(json.dumps({
        "ok": all_ok,
        "units_per_metre": args.units_per_metre,
        "tile_size": args.tile_size,
        "elements": elements,
    }))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
