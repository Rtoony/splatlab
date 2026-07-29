#!/usr/bin/env python3
"""surface_patches: painted structure-class regions become planar fill (PW-A7).

The user paints a sparse wall with a structure-category class; this script
turns each painted region into a clean planar patch mesh — the geometric
"fill in the gaps" the splat could not supply. Paint is authority AND scope:
patches are clustered from the paint, plane-fitted with numeric refusals
(curved region -> "paint flatter segments separately"), and footprint-bounded
by the paint's own extent — generation closes interior window-scale gaps and
never invents surface beyond them.

Runs in the dn-splatter-probe env (trimesh/PIL). Reads DISK paint records
only (class_xyz snapshots are exported-ply truth and realign keeps them
current) — no torch, no checkpoint, no langfield worker, so re-running never
503s on a service.

Outputs (all staged writes):
  <out_dir>/patch_<nn>.ply          — provenance-tagged patch geometry
  <out_dir>/surfaces_preview.glb    — capture-coloured concatenated preview
  <out_dir>/surfaces_atlas.png      — its atlas (written after readback)
  <out_dir>/surface_patches.json    — the report/receipt

Provenance: every artifact carries object_generate's GENERATIVE_TAG (survey
lanes refuse it mechanically); the report says "paint-synthesized". Class
tiles are NOT applied in v1 — world-XY tiling smears on vertical planes; the
future fix is a plane-basis sample_world_triplanar variant.

Usage: surface_patches.py <lfdir> <splat_ply> <taxonomy_json> <out_dir>
       [--plane-dist-units 0.05] [--min-cluster-points 200]
       [--cluster-cell-units 0.10] [--patch-cell-units 0.05]
       [--close-cells 2] [--pad-cells 1] [--max-patches 16]
       [--max-rms-frac 0.6] [--texture-size 1024] [--meters-per-unit MPU]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import surface_patch_core as core  # noqa: E402


def structure_class_ids(taxonomy_path: Path) -> set[str]:
    doc = json.loads(Path(taxonomy_path).read_text())
    return {c["id"] for c in doc.get("classes", [])
            if isinstance(c, dict) and c.get("category") == "structure"}


def load_structure_paint(lfdir: Path, structure_ids: set[str]):
    """(records, refusals) — a record carries its positions source; a record
    with neither snapshot nor rows is refused BY NAME, never guessed."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import class_labels as cl  # noqa: PLC0415 — stdlib+numpy module

    usable, refused = [], []
    for record in cl.load_manifest(Path(lfdir)):
        if record.get("invalid_reason"):
            continue
        if record.get("class_id") not in structure_ids:
            continue
        rid = str(record.get("id", ""))
        xyz_path = Path(lfdir) / f"class_xyz_{rid}.npy"
        idx_path = Path(lfdir) / f"class_idx_{rid}.npy"
        if xyz_path.is_file():
            usable.append({"id": rid, "class_id": record["class_id"],
                           "xyz_path": xyz_path})
        elif idx_path.is_file():
            usable.append({"id": rid, "class_id": record["class_id"],
                           "idx_path": idx_path})
        else:
            refused.append({"record": rid,
                            "refused": "no positions (xyz snapshot and idx "
                                       "rows both missing)"})
    return usable, refused


def _record_positions(record: dict, splat_ply: Path) -> np.ndarray:
    if "xyz_path" in record:
        return np.load(record["xyz_path"]).astype(np.float64)
    from plyfile import PlyData  # noqa: PLC0415 — mesh env
    vertex = PlyData.read(str(splat_ply))["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    rows = np.load(record["idx_path"]).astype(np.int64)
    rows = rows[(rows >= 0) & (rows < len(xyz))]
    return xyz[rows].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lfdir", type=Path)
    ap.add_argument("splat_ply", type=Path)
    ap.add_argument("taxonomy", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--plane-dist-units", type=float,
                    default=core.DEFAULT_PLANE_DIST_UNITS)
    ap.add_argument("--min-cluster-points", type=int,
                    default=core.DEFAULT_MIN_CLUSTER_POINTS)
    ap.add_argument("--cluster-cell-units", type=float,
                    default=core.DEFAULT_CLUSTER_CELL_UNITS)
    ap.add_argument("--patch-cell-units", type=float,
                    default=core.DEFAULT_PATCH_CELL_UNITS)
    ap.add_argument("--close-cells", type=int, default=core.DEFAULT_CLOSE_CELLS)
    ap.add_argument("--pad-cells", type=int, default=core.DEFAULT_PAD_CELLS)
    ap.add_argument("--max-patches", type=int, default=16)
    ap.add_argument("--max-rms-frac", type=float,
                    default=core.DEFAULT_MAX_PLANE_RMS_FRAC)
    ap.add_argument("--min-inlier-frac", type=float, default=0.7,
                    help="THE curvature refusal: the fitted plane must "
                         "explain at least this fraction of the painted "
                         "cluster (rms alone cannot catch a curve)")
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--meters-per-unit", type=float, default=None)
    args = ap.parse_args()
    t0 = time.time()

    structure_ids = structure_class_ids(args.taxonomy)
    if not structure_ids:
        print("FATAL: taxonomy has no structure-category classes", file=sys.stderr)
        return 1
    records, refused_records = load_structure_paint(args.lfdir, structure_ids)
    if not records:
        print("FATAL: no structure-category paint records — paint a wall "
              f"(classes: {', '.join(sorted(structure_ids))}) first",
              file=sys.stderr)
        return 1
    if not args.splat_ply.is_file():
        print(f"FATAL: no splat at {args.splat_ply}", file=sys.stderr)
        return 1

    import trimesh  # noqa: PLC0415 — mesh env from here down
    from PIL import Image  # noqa: PLC0415
    import object_texture as ot  # noqa: PLC0415
    import object_generate  # noqa: PLC0415
    import twin_finish  # noqa: PLC0415

    max_rms_units = args.max_rms_frac * args.plane_dist_units
    candidates: list[dict] = []
    refused_clusters: list[dict] = []
    for record in records:
        xyz = _record_positions(record, args.splat_ply)
        clusters = core.grid_clusters(xyz, args.cluster_cell_units)
        for cluster in clusters:
            if len(cluster) < args.min_cluster_points:
                refused_clusters.append({
                    "record": record["id"], "points": int(len(cluster)),
                    "refused": f"cluster too small ({len(cluster)} < "
                               f"{args.min_cluster_points})"})
                continue
            pts = xyz[cluster]
            try:
                fit = core.fit_plane_ransac(pts, args.plane_dist_units,
                                            max_rms_units=max_rms_units,
                                            min_inlier_frac=args.min_inlier_frac)
            except core.SurfacePatchError as exc:
                refused_clusters.append({"record": record["id"],
                                         "points": int(len(cluster)),
                                         "refused": str(exc)})
                continue
            inlier_pts = pts[fit["inliers"]]
            normal = fit["normal"]
            centroid = inlier_pts.mean(axis=0)
            plane_point = centroid - (normal @ centroid + fit["d"]) * normal
            u, v = core.plane_basis(normal)
            uv = np.stack([(inlier_pts - plane_point) @ u,
                           (inlier_pts - plane_point) @ v], axis=1)
            try:
                grid, origin = core.footprint_cells(
                    uv, args.patch_cell_units, args.close_cells, args.pad_cells)
                verts, faces = core.cells_to_mesh(
                    grid, origin, args.patch_cell_units, plane_point, u, v)
            except core.SurfacePatchError as exc:
                refused_clusters.append({"record": record["id"],
                                         "points": int(len(cluster)),
                                         "refused": str(exc)})
                continue
            candidates.append({
                "record": record["id"], "class_id": record["class_id"],
                "fit": fit, "verts": verts, "faces": faces,
                "plane_point": plane_point, "u": u, "v": v,
                "cells": int(grid.sum()),
                "area_units2": float(grid.sum()) * args.patch_cell_units ** 2,
            })

    candidates.sort(key=lambda c: c["area_units2"], reverse=True)
    overflow = candidates[args.max_patches:]
    for extra in overflow:
        refused_clusters.append({
            "record": extra["record"], "points": None,
            "refused": f"over --max-patches cap ({args.max_patches}); "
                       f"area {extra['area_units2']:.3f}"})
    candidates = candidates[:args.max_patches]
    if not candidates:
        loudest = refused_clusters[-1]["refused"] if refused_clusters else \
            "no viable painted clusters"
        print(f"FATAL: no patch survived — {loudest}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-patch geometry, staged + provenance-tagged ─────────────────────
    patches_report = []
    for index, cand in enumerate(candidates):
        name = f"patch_{index:02d}.ply"
        staged = args.out_dir / f".building-{name}"
        mesh = trimesh.Trimesh(vertices=cand["verts"], faces=cand["faces"],
                               process=False)
        mesh.export(str(staged))
        os.replace(staged, args.out_dir / name)
        object_generate.tag_ply_generative(args.out_dir / name)
        object_generate.verify_ply(args.out_dir / name)  # tag + counts, from disk
        fit = cand["fit"]
        patches_report.append({
            "id": f"patch_{index:02d}", "record": cand["record"],
            "class_id": cand["class_id"],
            "plane": {"normal": [round(float(x), 6) for x in fit["normal"]],
                      "d": round(float(fit["d"]), 6),
                      "rms_units": round(fit["rms_units"], 5),
                      "inlier_frac": round(fit["inlier_frac"], 4),
                      "inliers": int(len(fit["inliers"]))},
            "cells": cand["cells"],
            "area_units2": round(cand["area_units2"], 4),
            "verts": int(len(cand["verts"])), "faces": int(len(cand["faces"])),
            "ply": name,
        })

    # ── capture-coloured preview GLB (metres, Y-up, readback) ──────────────
    uv_charts = [core.planar_uvs(c["verts"], c["plane_point"], c["u"], c["v"])
                 for c in candidates]
    packed = core.pack_tiles(uv_charts)
    all_verts = np.vstack([c["verts"] for c in candidates])
    all_uvs = np.vstack(packed)
    offset = 0
    face_blocks = []
    for cand in candidates:
        face_blocks.append(cand["faces"] + offset)
        offset += len(cand["verts"])
    all_faces = np.vstack(face_blocks)

    splat_xyz, splat_rgb = twin_finish.load_solid_gaussians(args.splat_ply)
    texture = {"baked": False}
    tex, mask = ot.bake_texture(all_verts.astype(np.float32),
                                all_faces.astype(np.int64),
                                all_uvs.astype(np.float32),
                                splat_xyz.astype(np.float64), splat_rgb,
                                args.texture_size)
    if tex is None:
        print("FATAL: atlas rasterization covered no texels", file=sys.stderr)
        return 1
    coverage_rasterized = float(mask.mean())
    tex, shipped = ot.dilate_atlas(tex, mask,
                                   passes=max(16, args.texture_size // 48))
    image = Image.fromarray(tex, mode="RGB")

    scale = args.meters_per_unit or 1.0
    scaled = all_verts * scale
    yup = np.stack([scaled[:, 0], scaled[:, 2], -scaled[:, 1]],
                   axis=1).astype(np.float32)
    # Y-up flips winding (Z -> -Y): reverse faces, the ground_texture rule.
    out = trimesh.Trimesh(vertices=yup, faces=all_faces[:, ::-1], process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=all_uvs, image=image,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.85))
    glb_path = args.out_dir / "surfaces_preview.glb"
    staged_glb = args.out_dir / f".building-{glb_path.name}"
    out.export(str(staged_glb))
    back = trimesh.load(str(staged_glb), force="mesh")
    has_uv = getattr(getattr(back.visual, "uv", None), "shape", None) is not None
    img_back = getattr(getattr(back.visual, "material", None),
                       "baseColorTexture", None)
    if len(back.faces) != len(all_faces) or not has_uv or img_back is None:
        staged_glb.unlink(missing_ok=True)
        print("FATAL: preview GLB readback lost faces, UVs or texture",
              file=sys.stderr)
        return 1
    os.replace(staged_glb, glb_path)
    object_generate.tag_glb_generative(glb_path)
    # Atlas lands only AFTER the readback gate (generated_texture rule).
    atlas_path = args.out_dir / "surfaces_atlas.png"
    image.save(atlas_path)
    texture = {"baked": True, "size": args.texture_size,
               "coverage": round(float(shipped.mean()), 4),
               "coverage_rasterized": round(coverage_rasterized, 4)}

    report = {
        "v": 1,
        "provenance": "paint-synthesized",
        "note": "geometry generated from user structure-paint; plausible "
                "fill bounded by the painted extent, not capture",
        "params": {"plane_dist_units": args.plane_dist_units,
                   "min_cluster_points": args.min_cluster_points,
                   "cluster_cell_units": args.cluster_cell_units,
                   "patch_cell_units": args.patch_cell_units,
                   "close_cells": args.close_cells, "pad_cells": args.pad_cells,
                   "max_patches": args.max_patches,
                   "max_rms_frac": args.max_rms_frac,
                   "min_inlier_frac": args.min_inlier_frac,
                   "seed": core.RANSAC_SEED},
        "records_considered": len(records),
        "records_refused": refused_records,
        "patches": patches_report,
        "refused_clusters": refused_clusters,
        "class_tiles": "not-applied-v1",
        "texture": texture,
        "meters_per_unit": args.meters_per_unit,
        "artifacts": {"preview_glb": glb_path.name, "atlas": atlas_path.name},
        "seconds": round(time.time() - t0, 1),
    }
    report_path = args.out_dir / "surface_patches.json"
    staged_report = args.out_dir / f".building-{report_path.name}"
    staged_report.write_text(json.dumps(report, indent=2))
    os.replace(staged_report, report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
