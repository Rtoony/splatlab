#!/usr/bin/env python3
"""R3.2 plane-scaffold architecture layer for a captured building (CPU, splatops env).

Reads a native 2DGS surfel package (train-capture-surfels.py), the structure study's
SAM3 masks + frozen view cameras (capture_structure.prepare), optionally the Condo Lab
alignment receipt (source -> canonical metres similarity) and building.json, and writes
an explicit, reviewable layer: façade/pavement planes, garage-door / window / gap
rectangles, an up vector + Manhattan frame, reserved-track residuals, per-view overlays
on the source photos, a plane PLY and a receipt. Nothing is promoted: status stays
inferred-needs-review, owner_accepted false, registration provisional or null.

  architecture-scaffold.py --evaluation data/spatial/condo-surfel-anchored-evaluation-2026-09-09-01 \\
      --structure data/spatial/condo-capture-structure-2026-09-09-02 --output data/spatial/condo-architecture-scaffold-<date>-01 \\
      [--alignment <alignment-evidence receipt.json> --alignment-case "New four-view study"] [--building <building.json>]
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests                    # noqa: E402
from architecture import scaffold_core as sc            # noqa: E402
from mesh.slugify import slug                            # noqa: E402

CLASSES = sc.FACADE_CLASSES + sc.NUISANCE_CLASSES + (sc.PAVEMENT_CLASS,)
WALL, GARAGE, WINDOW, PAVEMENT = 0, 1, 2, len(CLASSES) - 1
SCHEMA = "dev.splatlab.architecture-scaffold/v1"
PALETTE = [(230, 60, 60), (60, 160, 230), (60, 200, 90), (240, 180, 40), (180, 80, 220), (40, 210, 210), (250, 120, 30), (120, 120, 250)]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load_views(structure: Path, evaluation: Path, receipt_eval: dict, mask_score: float):
    rec = json.loads((structure / "receipt.json").read_text())
    intr = rec["intrinsics"]
    depth_by_image = {r["image"]: r for r in receipt_eval.get("training_depth", [])}
    views = []
    for view in rec["views"]:
        c2w = np.asarray(view["transform_matrix"], dtype=np.float64)
        masks, instances = {}, {}
        for prompt in rec["prompts"]:
            p = structure / "masks" / slug(prompt) / f"cam_{view['ordinal']:03d}.npz"
            if p.is_file():
                z = np.load(p); m = np.asarray(z["masks"], dtype=bool); s = np.asarray(z["scores"], dtype=np.float64)
                masks[prompt] = m[s >= mask_score].any(axis=0) if len(m) else np.zeros((intr["h"], intr["w"]), bool)
                instances[prompt] = [(m[k], float(s[k])) for k in range(len(m)) if s[k] >= mask_score]
        depth = None; drec = depth_by_image.get(view["file_path"])
        if drec is not None:
            z = np.load(evaluation / drec["file"]); depth = np.asarray(z["depth"], dtype=np.float64).copy()
            depth[np.asarray(z["alpha"]) < 0.5] = np.nan
        views.append({"ordinal": view["ordinal"], "file_path": view["file_path"], "photo": structure / view["photo"], "c2w": c2w,
                      "fx": intr["fl_x"], "fy": intr["fl_y"], "cx": intr["cx"], "cy": intr["cy"], "w": intr["w"], "h": intr["h"],
                      "depth": depth, "masks": masks, "instances": instances, "has_depth": depth is not None})
    return rec, views


def patch_record(patch, points, tol, up=None, similarity=None):
    corners = sc.rect_corners_3d(patch, patch["lower_uv"], patch["upper_uv"])
    rec = {"centre": patch["centre"].tolist(), "normal": patch["normal"].tolist(), "basis": patch["basis"].tolist(),
           "lower_uv": np.asarray(patch["lower_uv"]).tolist(), "upper_uv": np.asarray(patch["upper_uv"]).tolist(),
           "extent_units": patch["extent"], "surfels": int(len(patch["ids"])), "fit_rms_units": float(patch["rms"]), "tolerance_units": tol,
           "corners_units": corners.tolist()}
    if up is not None:
        rec["tilt_deg"] = sc.tilt_deg(patch["normal"], up)
    if similarity is not None:
        R, s, _ = sc.similarity_parts(similarity)
        rec["corners_m"] = sc.apply_similarity(corners, similarity).tolist(); rec["normal_canonical"] = (R @ patch["normal"]).tolist()
        rec["extent_m"] = (np.asarray(patch["extent"]) * s).tolist(); rec["fit_rms_m"] = float(patch["rms"] * s)
    return rec


def rect_record(patch, r, similarity=None):
    corners = sc.rect_corners_3d(patch, r["lower_uv"], r["upper_uv"])
    out = {**r, "corners_units": corners.tolist()}
    if similarity is not None:
        _, s, _ = sc.similarity_parts(similarity)
        out["width_m"], out["height_m"] = r["width"] * s, r["height"] * s; out["corners_m"] = sc.apply_similarity(corners, similarity).tolist()
    return out


def ray_openings(patch, views, prompt, min_frac=0.6, join_frac=0.15):
    """Depth-free openings: every SAM3 instance of `prompt`, in every view where it
    does not touch the image border, is cast onto the wall plane; instances whose hits
    fall mostly inside this wall's footprint give one rectangle per view; rectangles
    are then clustered across views (median bounds, view count, spread)."""
    rects = []
    for view in views:
        for mask, score in view.get("instances", {}).get(prompt, []):
            if sc.touches_image_border(mask) or mask.sum() < 50:
                continue
            uv, ok = sc.ray_plane_uv(np.nonzero(mask), view["c2w"], view["fx"], view["fy"], view["cx"], view["cy"], patch)
            if not ok.any():
                continue
            inside = ((uv >= patch["lower_uv"]) & (uv <= patch["upper_uv"])).all(axis=1)
            if inside.mean() < min_frac:
                continue
            lo, hi = np.percentile(uv[inside], 1, axis=0), np.percentile(uv[inside], 99, axis=0)
            rects.append({"lower_uv": lo.tolist(), "upper_uv": hi.tolist(), "view": int(view["ordinal"]), "score": score})
    extent = float(np.max(np.asarray(patch["upper_uv"]) - np.asarray(patch["lower_uv"])))
    return [r for r in sc.cluster_rectangles(rects, join_dist=join_frac * extent) if r["n_views"] >= 1]


def draw_overlay(view, points, patches, openings, out_path, font):
    im = Image.open(view["photo"]).convert("RGB"); dr = ImageDraw.Draw(im, "RGBA")
    rng = np.random.default_rng(0)
    for pi, patch in enumerate(patches):
        col = PALETTE[pi % len(PALETTE)]
        ids = patch["ids"] if len(patch["ids"]) <= 3000 else rng.choice(patch["ids"], 3000, replace=False)
        u, v, _, ok = sc.project(points[ids], view["c2w"], view["fx"], view["fy"], view["cx"], view["cy"], view["w"], view["h"])
        for x, y in zip(u[ok], v[ok]):
            dr.point((x, y), fill=col + (150,))
        corners = sc.rect_corners_3d(patch, patch["lower_uv"], patch["upper_uv"])
        u, v, _, ok = sc.project(corners, view["c2w"], view["fx"], view["fy"], view["cx"], view["cy"], view["w"] * 4, view["h"] * 4)
        if ok.sum() == 4:
            dr.polygon([(x, y) for x, y in zip(u, v)], outline=col + (255,), width=3)
            dr.text((float(u.min()) + 4, float(v.min()) + 4), patch["name"], fill=col + (255,), font=font, stroke_width=2, stroke_fill=(0, 0, 0, 255))
        for kind, rects in openings.get(patch["name"], {}).items():
            ocol = {"garage_door": (255, 140, 0), "window": (0, 230, 255), "gap": (255, 0, 255)}[kind]
            for r in rects:
                c = sc.rect_corners_3d(patch, r["lower_uv"], r["upper_uv"])
                u, v, _, ok = sc.project(c, view["c2w"], view["fx"], view["fy"], view["cx"], view["cy"], view["w"] * 4, view["h"] * 4)
                if ok.sum() == 4:
                    dr.polygon([(x, y) for x, y in zip(u, v)], outline=ocol + (255,), width=2)
    dr.text((6, 6), f"cam {view['ordinal']:03d} {'depth-checked' if view['has_depth'] else 'no depth map'}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    im.save(out_path)
    return im


def write_ply(path, quads):
    """quads: list of (4x3 corners, rgb)."""
    verts, faces, cols = [], [], []
    for corners, rgb in quads:
        i = len(verts); verts.extend(np.asarray(corners).tolist()); cols.extend([rgb] * 4); faces.append((i, i + 1, i + 2)); faces.append((i, i + 2, i + 3))
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\ncomment splatlab architecture scaffold: inferred reference planes, not a solid\nelement vertex {len(verts)}\n"
                "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
                f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for (x, y, z), (r, g, b) in zip(verts, cols):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evaluation", type=Path, required=True); ap.add_argument("--structure", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alignment", type=Path); ap.add_argument("--alignment-case", default="New four-view study")
    ap.add_argument("--building", type=Path)
    ap.add_argument("--opacity", type=float, default=0.3); ap.add_argument("--mask-score", type=float, default=0.3)
    ap.add_argument("--depth-tol", type=float, default=0.06); ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--tol-frac", type=float, default=0.01, help="plane tolerance as a fraction of the camera span")
    ap.add_argument("--angle-tol", type=float, default=12.0); ap.add_argument("--min-inliers", type=int, default=300)
    ap.add_argument("--min-cells", type=int, default=40); ap.add_argument("--max-planes", type=int, default=8)
    ap.add_argument("--max-tilt", type=float, default=15.0, help="degrees from vertical for a façade patch to count as a wall")
    ap.add_argument("--source", choices=["depth", "surfels", "moge"], default="depth", help="oriented points from the rendered depth maps (default), the raw surfels, or MoGe-2 depth (--moge)")
    ap.add_argument("--moge", type=Path, help="condo-moge-depth.py output directory (for --source moge)")
    ap.add_argument("--step", type=int, default=4, help="depth-map pixel stride"); ap.add_argument("--edge-rel", type=float, default=0.04)
    ap.add_argument("--wall-tol-frac", type=float, default=0.01, help="façade plane tolerance (fraction of span)")
    ap.add_argument("--opening-cell-frac", type=float, default=0.005, help="in-plane cell for opening/gap rectangles (fraction of span)")
    ap.add_argument("--merge-angle", type=float, default=6.0); ap.add_argument("--min-opening-frac", type=float, default=0.6)
    ap.add_argument("--anchor-min-tracks", type=int, default=8); ap.add_argument("--anchor-dilate", type=float, default=0.1)
    ap.add_argument("--max-shift-frac", type=float, default=0.1, help="largest allowed track-anchoring shift, fraction of span (2DGS depth bias reached 0.62 m on the condo)")
    ap.add_argument("--band-frac", type=float, default=0.05, help="tracks further than this (fraction of span) from a plane belong to another surface")
    ap.add_argument("--no-anchor", action="store_true", help="keep the dense-depth plane position instead of the sparse-track median")
    a = ap.parse_args(); t0 = time.time()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    a.output.mkdir(parents=True); (a.output / "overlays").mkdir()
    receipt_eval = json.loads((a.evaluation / "receipt.json").read_text())
    rec_struct, views = load_views(a.structure, a.evaluation, receipt_eval, a.mask_score)
    cams = np.array([v["c2w"][:3, 3] for v in views]); span = float(np.max(np.linalg.norm(cams[:, None] - cams[None], axis=-1)))
    tol = a.tol_frac * span
    if a.source == "surfels":
        # the surfels themselves, normals from their quaternions (measured 2026-09-15 on the condo: those
        # normals are ~random after 3,000 steps — median 48 deg off the pavement plane — keep for diagnostics)
        z = np.load(a.evaluation / "surfels.npz")
        means = sc.to_original_frame(z["means"], z["solver_center"], float(z["solver_scale"]))
        opac = sc.sigmoid(z["opacities"]); keep = opac >= a.opacity
        points, normals, weights = means[keep], sc.quat_to_normal(z["quats"][keep]), opac[keep]
        votes = sc.label_votes(points, views, CLASSES, depth_tol=a.depth_tol)
        cls = sc.classify(votes, CLASSES, min_votes=a.min_votes)
        print(f"[scaffold] {len(points)} of {len(means)} surfels with opacity >= {a.opacity}", flush=True)
    else:
        # per-view depth maps -> oriented points with per-pixel labels. "depth" = the 2DGS model's own
        # rendered depth; "moge" = MoGe-2 metric depth scaled into SfM units by each view's sparse tracks
        chunks = []
        for view in views:
            depth = view["depth"]
            if a.source == "moge":
                p = a.moge / f"cam_{view['ordinal']:03d}.npz"
                if not p.is_file():
                    continue
                z = np.load(p); s_v = float(z["scale_u_per_m"])
                if not np.isfinite(s_v):
                    continue
                depth = np.asarray(z["depth_m"], dtype=np.float64) * s_v; depth[~np.asarray(z["mask"], dtype=bool)] = np.nan
            if depth is None:
                continue
            pts, nrm, rows, cols = sc.backproject_depth(depth, view["c2w"], view["fx"], view["fy"], view["cx"], view["cy"], step=a.step, edge_rel=a.edge_rel)
            lab = np.full(len(pts), -1, dtype=np.int64)
            for k, name in reversed(list(enumerate(CLASSES))):             # earlier classes win ties (wall > garage > window)
                m = view["masks"].get(name)
                if m is not None:
                    lab[np.asarray(m, dtype=bool)[rows, cols]] = k
            nuisance = np.zeros(len(pts), dtype=bool)
            for name in sc.NUISANCE_CLASSES:
                m = view["masks"].get(name)
                if m is not None:
                    nuisance |= np.asarray(m, dtype=bool)[rows, cols]
            lab[nuisance] = -1
            chunks.append((pts, nrm, lab, np.full(len(pts), view["ordinal"])))
        points = np.concatenate([c[0] for c in chunks]); normals = np.concatenate([c[1] for c in chunks])
        cls = np.concatenate([c[2] for c in chunks]); weights = np.ones(len(points))
        print(f"[scaffold] depth cloud: {len(points)} oriented points from {len(chunks)} views (step {a.step})", flush=True)
    counts = {name: int((cls == k).sum()) for k, name in enumerate(CLASSES)}
    print(f"[scaffold] views {len(views)} ({sum(v['has_depth'] for v in views)} with depth), span {span:.3f} u, tol {tol:.4f} u, labels {counts}", flush=True)
    n_or = sc.orient_towards(normals, points, cams.mean(axis=0))
    fac_ids = np.flatnonzero(np.isin(cls, [WALL, GARAGE, WINDOW])); pav_ids = np.flatnonzero(cls == PAVEMENT)
    wall_tol = (a.wall_tol_frac or a.tol_frac) * span
    fac = sc.extract_planes(points[fac_ids], n_or[fac_ids], weights[fac_ids], wall_tol, a.angle_tol, a.min_inliers, a.min_cells, a.max_planes)
    pav = sc.extract_planes(points[pav_ids], n_or[pav_ids], weights[pav_ids], tol, a.angle_tol, a.min_inliers, a.min_cells, 3)
    for p in fac: p["ids"] = fac_ids[p["ids"]]
    for p in pav: p["ids"] = pav_ids[p["ids"]]
    up = sc.up_vector(pav[0], cams) if pav else None
    mem = np.load(a.structure / "membership.npz") if (a.structure / "membership.npz").is_file() else None
    walls, anchors = [], {}
    fit_tracks = mem["points"][mem["facade_supported"].astype(bool) & ~mem["plane_check_reserved"].astype(bool)] if mem is not None else np.zeros((0, 3))
    max_shift, band = a.max_shift_frac * span, a.band_frac * span

    def anchor(p):
        if a.no_anchor or not len(fit_tracks):
            return p, {"applied": False, "reason": "anchoring disabled or no tracks"}
        q, info = sc.anchor_to_tracks(p, fit_tracks, min_tracks=a.anchor_min_tracks, max_shift=max_shift, dilate=a.anchor_dilate, band=band)
        return (sc.rebase_patch(q, points, q["basis"]) if info["applied"] else q), info

    for i, p in enumerate(fac):
        p["name"] = f"facade-{i}"; p["tilt"] = sc.tilt_deg(p["normal"], up) if up is not None else None; p["kind"] = "facade-tilted"
        if up is None or p["tilt"] <= a.max_tilt:
            p = sc.rebase_patch(p, points, sc.gravity_basis(p["normal"], up)); p["kind"] = "wall"       # plumb rectangles
            p, _ = anchor(p)                                                                           # position from the sparse SfM tracks, then merge
            walls.append(p)
        fac[i] = p
    for i, p in enumerate(pav):
        p["name"] = f"pavement-{i}"; p["kind"] = "pavement"
    n_before = len(walls)
    walls = sc.merge_coplanar(walls, points, angle_deg=a.merge_angle, offset_tol=1.5 * wall_tol)
    for i, p in enumerate(walls):                                                                     # anchor the merged walls
        p["name"] = f"wall-{i}"; p["fragments"] = int(p.get("merged", 1))
        p, anchors[p["name"]] = anchor(p); walls[i] = p
    print(f"[scaffold] facade patches {len(fac)} -> walls {n_before} (tilt <= {a.max_tilt} deg) -> merged {len(walls)}; pavement patches {len(pav)}; anchors "
          + ", ".join(f"{k}:{v.get('tracks_used')}t/{v.get('shift_units', 0):+.3f}u" for k, v in anchors.items() if v.get("applied")), flush=True)
    frame = sc.manhattan_frame(up, walls[0]["normal"]) if (up is not None and walls) else None
    # alignment (optional): source -> canonical metres
    similarity, align_info = None, None
    if a.alignment:
        ar = json.loads(a.alignment.read_text())
        case = next((c for c in ar.get("cases", []) if c.get("name") == a.alignment_case), None)
        if case is None:
            raise SystemExit(f"alignment case {a.alignment_case!r} not in {[c.get('name') for c in ar.get('cases', [])]}")
        similarity = np.asarray(case["fit"]["matrix"], dtype=np.float64).reshape(4, 4)
        align_info = {"file": str(a.alignment), "case": a.alignment_case, "scale": case["fit"].get("scale"), "convention": case["fit"].get("convention"),
                      "fit_rms_m": case["fit"].get("fitRmsMeters", case.get("fitRms")), "residuals": case["fit"].get("residuals"),
                      "scope": "provisional similarity from the Condo Lab alignment candidate; its scale comes from the modelled garage opening, so door dimensions measured through it are circular"}
    # openings + checks
    cell = a.opening_cell_frac * span
    openings, patch_recs, checks = {}, [], {}
    for p in walls + pav:
        rec = patch_record(p, points, wall_tol if p["kind"] == "wall" else tol, up, similarity); rec["name"] = p["name"]; rec["kind"] = p["kind"]
        if p["name"] in anchors:
            rec["anchor"] = anchors[p["name"]]
            if similarity is not None:
                rec["anchor"]["shift_m"] = anchors[p["name"]]["shift_units"] * sc.similarity_parts(similarity)[1]
        if p["kind"] == "wall":
            edge = sc.ground_edge(p, up) if up is not None else ()
            o = {"garage_door": ray_openings(p, views, "garage door", a.min_opening_frac), "window": ray_openings(p, views, "window", a.min_opening_frac),
                 "gap": sc.gap_rectangles(p, points, cell, closed_edges=(edge,) if edge else ())}
            openings[p["name"]] = o
            rec["openings"] = {k: [rect_record(p, r, similarity) for r in v] for k, v in o.items()}
        if mem is not None:
            key = "facade_supported" if p["kind"] == "wall" else "pavement_supported"
            chk_pts = mem["points"][mem["plane_check_reserved"].astype(bool) & mem[key].astype(bool)]
            rec["reserved_track_check"] = sc.reserved_track_check(p, chk_pts, wall_tol if p["kind"] == "wall" else tol, band=band)
        patch_recs.append(rec)
    # measurements vs building.json
    measurements = []
    if a.building and similarity is not None and walls and up is not None:
        b = json.loads(a.building.read_text()); R, s, _ = sc.similarity_parts(similarity)
        lvl = b["levels"][0]; garage_wall = next((w for w in lvl["walls"] if any(o.get("kind") == "garage" for o in w.get("openings", []))), None)
        if garage_wall:
            length = float(np.linalg.norm(np.subtract(garage_wall["b"], garage_wall["a"])))
            door = next(o for o in garage_wall["openings"] if o["kind"] == "garage")
            w0 = patch_recs[0]
            measurements.append({"item": "street wall plane extent (all coplanar units)", "model_m": length, "model_note": "model value = 2286 garage wall only", "model_source": garage_wall.get("source"), "model_confidence": garage_wall.get("confidence"),
                                 "capture_m": w0["extent_m"][0], "circular": False, "note": "largest wall plane along its own axis; coplanar neighbouring units are included, so this is NOT the 2286 wall length"})
            doors = w0["openings"]["garage_door"]
            if doors:
                measurements.append({"item": "garage door width", "model_m": door["width"], "model_confidence": door.get("confidence"), "capture_m": doors[0]["width_m"], "circular": True,
                                     "n_views": doors[0]["n_views"], "spread_m": doors[0]["width_spread"] * s, "note": "the alignment scale was fitted to this very opening"})
                measurements.append({"item": "garage door height", "model_m": door["height"], "model_confidence": door.get("confidence"), "capture_m": doors[0]["height_m"], "circular": True,
                                     "n_views": doors[0]["n_views"], "spread_m": doors[0]["height_spread"] * s})
            for k, g in enumerate(w0["openings"]["gap"][:3]):
                measurements.append({"item": f"unobserved gap {k}", "model_m": None, "capture_m": [g["width_m"], g["height_m"]], "circular": False})
            for k, g in enumerate(w0["openings"]["window"][:4]):
                measurements.append({"item": f"window {k}", "model_m": None, "capture_m": [g["width_m"], g["height_m"]], "circular": False})
            if a.moge and (a.moge / "receipt.json").is_file():
                mr = json.loads((a.moge / "receipt.json").read_text()); mpu = mr.get("global_meters_per_unit")
                if mpu:
                    ratio = mpu / s
                    measurements.append({"item": "SfM-frame scale: MoGe-2 (tracks) vs alignment (garage opening)", "model_m": s, "model_note": "alignment metres per unit, from the modelled 4.8768 x 2.1336 m opening",
                                         "capture_m": mpu, "circular": False, "ratio": ratio, "n_views": mr.get("n_views_scaled"), "spread_m": mr.get("per_view_scale_mad_u_per_m"),
                                         "note": "independent monocular metric depth scaled by each view's own sparse tracks; agreement means the modelled opening size is consistent with MoGe-2 to this ratio"})
                    if doors:
                        measurements.append({"item": "garage door width at MoGe-2 scale", "model_m": door["width"], "capture_m": doors[0]["width_m"] * ratio, "circular": False, "n_views": doors[0]["n_views"],
                                             "note": "SAM3 door-leaf mask cast onto the anchored wall plane, metres from MoGe-2 (not from the door itself)"})
                        measurements.append({"item": "garage door height at MoGe-2 scale", "model_m": door["height"], "capture_m": doors[0]["height_m"] * ratio, "circular": False, "n_views": doors[0]["n_views"]})
                        measurements.append({"item": "garage door aspect (width / height), scale-free", "model_m": door["width"] / door["height"], "capture_m": doors[0]["width"] / doors[0]["height"], "circular": False})
            measurements.append({"item": "wall normal vs model -Y (street side)", "model_m": None,
                                 "capture_deg": float(math.degrees(math.acos(np.clip(-np.asarray(w0["normal_canonical"])[1], -1, 1)))), "circular": False})
            measurements.append({"item": "up vector vs model +Z", "capture_deg": float(math.degrees(math.acos(np.clip((R @ np.asarray(up, dtype=np.float64))[2], -1, 1)))), "circular": False})
    # overlays
    font = ImageFont.truetype(FONT, 18); tiles = []
    for view in views:
        tiles.append(draw_overlay(view, points, walls + pav, openings, a.output / "overlays" / f"cam_{view['ordinal']:03d}.png", font))
    if tiles:
        tw, th = tiles[0].size; cols = 6; rows = math.ceil(len(tiles) / cols); sheet = Image.new("RGB", (cols * tw // 2, rows * th // 2))
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((tw // 2, th // 2)), ((i % cols) * tw // 2, (i // cols) * th // 2))
        sheet.save(a.output / "contact.png")
    # plane PLY (patches + openings as quads, inferred reference only)
    quads = []
    for p in walls + pav:
        quads.append((sc.rect_corners_3d(p, p["lower_uv"], p["upper_uv"]), (200, 200, 200) if p["kind"] == "wall" else (120, 120, 120)))
        for kind, rects in openings.get(p["name"], {}).items():
            for r in rects:
                quads.append((sc.rect_corners_3d(p, r["lower_uv"], r["upper_uv"]) + 0.01 * span * p["normal"], {"garage_door": (255, 140, 0), "window": (0, 200, 255), "gap": (255, 0, 255)}[kind]))
    write_ply(a.output / "scaffold.ply", quads)
    scaffold = {"schema": SCHEMA, "status": "inferred-needs-review", "owner_accepted": False, "new_solid_geometry": False,
                "registration": "provisional-similarity-from-alignment-candidate" if similarity is not None else None,
                "frame": {"units": "arbitrary SfM units (original frame)", "up": up.tolist() if up is not None else None,
                          "manhattan_rows_right_out_up": frame.tolist() if frame is not None else None,
                          "pavement_patches": len(pav), "facade_patches": len(fac), "wall_patches": len(walls)},
                "labels": counts, "tolerance_units": tol, "wall_tolerance_units": wall_tol, "opening_cell_units": cell, "camera_span_units": span, "patches": patch_recs, "measurements": measurements,
                "alignment": align_info, "scope": "Planes are fitted to inferred 2DGS surfels labelled by SAM3 masks; they are reference evidence for review, not accepted walls, not a solid, not a survey."}
    manifests.atomic_write_json(a.output / "scaffold.json", scaffold)
    receipt = {"schema": SCHEMA, "started_at": manifests.utc_now(), "elapsed_seconds": round(time.time() - t0, 1), "status": "inferred-needs-review",
               "inputs": {"evaluation": str(a.evaluation), "structure": str(a.structure), "moge": str(a.moge) if a.moge else None, "alignment": str(a.alignment) if a.alignment else None, "building": str(a.building) if a.building else None},
               "source_hashes": {"surfels.npz": manifests.sha256_file(a.evaluation / "surfels.npz"), "evaluation receipt": manifests.sha256_file(a.evaluation / "receipt.json"), "structure receipt": manifests.sha256_file(a.structure / "receipt.json")},
               "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               "summary": {"surfels_used": int(len(points)), "labels": counts, "wall_patches": [{"name": p["name"], "surfels": int(len(p["ids"])), "rms_units": float(p["rms"]), "tilt_deg": p["tilt"], "extent_units": p["extent"], "fragments": p.get("fragments", 1), "anchor": anchors.get(p["name"])} for p in walls],
                           "pavement_patches": [{"name": p["name"], "surfels": int(len(p["ids"])), "rms_units": float(p["rms"])} for p in pav],
                           "openings": {n: {k: len(v) for k, v in o.items()} for n, o in openings.items()}, "measurements": measurements}}
    manifests.atomic_write_json(a.output / "receipt.json", receipt)
    print(json.dumps(receipt["summary"], indent=1)); print(f"[scaffold] {a.output} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
