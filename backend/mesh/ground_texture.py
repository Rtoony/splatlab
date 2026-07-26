#!/usr/bin/env python3
"""Class-textured ground GLB — the first generative consumer of painted labels.

A ground TIN is a height field, so planar XY projection IS a perfect UV unwrap
(no xatlas, no charts, no seams). Per texel: XY -> nearest classed cell ->
class tile sampled in WORLD space (constant physical texel density) with a
noise-jittered boundary so class edges don't run grid-straight. Elevations
stay REAL (the capture's TIN); only the surfacing is "representative but
faked" — provenance says exactly that.

Runs in the dn-splatter-probe env (numpy/trimesh/scipy/PIL).
twin_finish's capture-coloured ground_mesh.glb is untouched — this writes the
SIBLING ground_classed.glb + ground_atlas.png + ground_texture_report.json.

Usage: ground_texture.py <ground_mesh_raw.ply> <ground_class_cells.npz>
       <taxonomy.json> <out.glb> [--meters-per-unit MPU] [--texture-size 2048]
       [--texels-per-unit 64]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from class_textures import sample_world, tile_for  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_mesh")
    ap.add_argument("class_cells")
    ap.add_argument("taxonomy")
    ap.add_argument("out_glb")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    ap.add_argument("--texture-size", type=int, default=2048)
    ap.add_argument("--texels-per-unit", type=float, default=64.0,
                    help="world-space texel density for tile sampling")
    args = ap.parse_args()
    t0 = time.time()
    out_glb = Path(args.out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(args.ground_mesh, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        print("FATAL: ground mesh has no faces", file=sys.stderr)
        return 1

    cells = np.load(args.class_cells)
    cell_ij = cells["cell_ij"].astype(np.int64)
    cell_units = float(cells["cell_units"])
    cell_class = cells["class_idx"].astype(np.int64)
    class_ids = [str(x) for x in cells["class_ids"]]
    taxonomy = json.loads(Path(args.taxonomy).read_text())
    by_id = {c["id"]: c for c in taxonomy.get("classes", [])}
    tiles = {i: tile_for(by_id.get(cid, {"id": cid}), 512)
             for i, cid in enumerate(class_ids)}
    fallback_rgb = np.array([128, 128, 128], np.uint8)

    classed = cell_class >= 0
    if not bool(classed.any()):
        print("FATAL: no classed cells — nothing to texture", file=sys.stderr)
        return 1
    cell_xy = (cell_ij[classed] + 0.5) * cell_units
    cell_cls = cell_class[classed]
    tree = cKDTree(cell_xy)

    # ── planar-XY UV + atlas bake ────────────────────────────────────────────
    lo = verts[:, :2].min(axis=0)
    span = np.maximum(verts[:, :2].max(axis=0) - lo, 1e-9)
    uvs = ((verts[:, :2] - lo) / span).astype(np.float32)

    size = args.texture_size
    px = np.arange(size, dtype=np.float64) + 0.5
    wx = lo[0] + (px / size) * span[0]
    wy = lo[1] + (px / size) * span[1]
    gx, gy = np.meshgrid(wx, wy)  # [size,size] world XY per texel

    # Boundary jitter: perturb the LOOKUP position with seeded noise so class
    # edges wobble at cell scale instead of tracing the grid.
    rng = np.random.default_rng(1234)
    jitter = (rng.random((size, size, 2), dtype=np.float32) - 0.5) * cell_units
    q = np.stack([gx + jitter[..., 0], gy + jitter[..., 1]], axis=-1).reshape(-1, 2)
    dist, nearest = tree.query(q, k=1, workers=-1)
    texel_cls = cell_cls[nearest].reshape(size, size)
    # texels far from any classed cell (outside the hull) fall back to the
    # nearest class anyway — the mesh doesn't extend there, so it never shows.

    tex = np.empty((size, size, 3), np.uint8)
    tex[:] = fallback_rgb
    fractions = {}
    for i, cid in enumerate(class_ids):
        mask = texel_cls == i
        if not bool(mask.any()):
            continue
        fractions[cid] = round(float(mask.mean()), 4)
        tex[mask] = sample_world(tiles[i], gx[mask], gy[mask], args.texels_per_unit)

    atlas_path = out_glb.with_name("ground_atlas.png")
    Image.fromarray(tex, mode="RGB").save(atlas_path)

    # ── export: metres, Y-up (twin_finish convention), PBR w/ atlas ─────────
    scale = args.meters_per_unit or 1.0
    v = verts * scale
    yup = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1).astype(np.float32)
    # Y-up flips triangle winding (Z -> -Y): reverse faces so normals point up.
    out = trimesh.Trimesh(vertices=yup, faces=faces[:, ::-1], process=False)
    majority = max(fractions, key=fractions.get) if fractions else None
    gen = (by_id.get(majority) or {}).get("generative") or {}
    tex_img = Image.fromarray(tex, mode="RGB")
    out.visual = trimesh.visual.TextureVisuals(
        uv=uvs, image=tex_img,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img,
            metallicFactor=float(gen.get("metallic", 0.0)),
            roughnessFactor=float(gen.get("roughness", 0.85)),
        ),
    )
    out.export(str(out_glb))

    # MANDATORY readback (house rule: never trust an exporter's exit code).
    back = trimesh.load(str(out_glb), force="mesh")
    has_uv = getattr(getattr(back.visual, "uv", None), "shape", None) is not None
    img_back = getattr(getattr(back.visual, "material", None), "baseColorTexture", None)
    if len(back.faces) != len(faces) or not has_uv or img_back is None:
        out_glb.unlink(missing_ok=True)
        print("FATAL: GLB readback lost faces, UVs or texture", file=sys.stderr)
        return 1

    report = {
        "v": 1,
        "provenance": "captured-geometry/class-textured",
        "note": "elevations are the capture's TIN; surfacing is generated "
                "from class labels (representative, not photographic)",
        "verts": int(len(yup)), "faces": int(len(faces)),
        "texture_size": size,
        "texels_per_unit": args.texels_per_unit,
        "class_texel_fractions": fractions,
        "classed_cells": int(classed.sum()),
        "meters_per_unit": args.meters_per_unit,
        "artifacts": {"glb": out_glb.name, "atlas": atlas_path.name},
        "seconds": round(time.time() - t0, 1),
    }
    report_path = out_glb.with_name("ground_texture_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
