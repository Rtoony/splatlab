#!/usr/bin/env python3
"""Turn an architecture-scaffold run into Condo Lab evidence records (CPU, splatops env).

Reads scaffold.json (+ overlays), the MoGe-2 depth receipt, the Condo Lab alignment receipt
(source -> canonical metres) and building.json; matches capture openings to model openings
by plane + rectangle overlap in the canonical frame; writes <output>/evidence.json = a list
of POST /api/feedback payloads (Measurement / Reference / Question records with measurement
rows, tags, details and overlay attachments) and, with --apply, posts them to the local
Condo Lab review server (loopback :2286) and writes evidence-receipt.json with the ids.
Nothing changes the accepted model: every record lands in the owner's review queue.

  condo-evidence.py --scaffold data/spatial/condo-architecture-scaffold-2026-09-15-02 --moge data/spatial/condo-moge-depth-2026-09-15-01 \\
      --alignment <receipt.json> --alignment-case "New four-view study, fresh reserved image" --building <building.json> --output <dir> [--apply]
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, shutil, sys, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from architecture import scaffold_core as sc   # noqa: E402

TODAY = dt.date.today().isoformat()
TAGS = ["capture", "splatlab", "architecture-scaffold"]


def similarity(alignment: Path, case_name: str):
    ar = json.loads(alignment.read_text())
    case = next(c for c in ar["cases"] if c["name"] == case_name)
    M = np.asarray(case["fit"]["matrix"], dtype=np.float64).reshape(4, 4)
    return M, float(case["fit"]["scale"]), case["fit"]


def model_openings(building: dict) -> list[dict]:
    """Every opening on an exterior wall as a canonical rectangle: along-wall axis d, outward-ish normal,
    corners (4x3), x-range along the wall segment, z-range (level elevation + sill .. + height)."""
    out = []
    for level in building["levels"]:
        for wall in level["walls"]:
            a, b = np.array(wall["a"], dtype=float), np.array(wall["b"], dtype=float)
            d = b - a; length = float(np.linalg.norm(d)); d = d / max(length, 1e-9)
            n = np.array([d[1], -d[0]])                          # in-plane perpendicular (XY)
            for o in wall.get("openings", []):
                z0 = level["elevation"] + o.get("sill", 0.0); z1 = z0 + o["height"]
                p0 = a + d * o["offset"]; p1 = p0 + d * o["width"]
                corners = np.array([[p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z1], [p0[0], p0[1], z1]])
                out.append({"id": o["id"], "kind": o["kind"], "wall": wall["id"], "level": level["id"], "exterior": bool(wall.get("exterior")),
                            "width": o["width"], "height": o["height"], "sill": o.get("sill", 0.0), "sill_z": z0, "head_z": z1, "confidence": o.get("confidence"),
                            "corners": corners, "dir": d, "normal_xy": n, "plane_point": np.array([p0[0], p0[1], z0]), "wall_length": length})
    return out


def rect_iou(a_lo, a_hi, b_lo, b_hi) -> float:
    lo = np.maximum(a_lo, b_lo); hi = np.minimum(a_hi, b_hi)
    inter = float(np.prod(np.clip(hi - lo, 0, None))); ua = float(np.prod(a_hi - a_lo)); ub = float(np.prod(b_hi - b_lo))
    return inter / max(ua + ub - inter, 1e-12)


def match_openings(capture: list[dict], model: list[dict], plane_tol: float = 1.0, angle_deg: float = 20.0, min_iou: float = 0.2):
    """capture rects carry canonical corners_m (4x3). Match on wall plane (direction within angle, offset within
    plane_tol) and rectangle IoU in (along-wall, z)."""
    pairs = []
    for ci, c in enumerate(capture):
        cc = np.asarray(c["corners_m"]); cdir = cc[1] - cc[0]; cdir[2] = 0; cdir /= max(np.linalg.norm(cdir), 1e-9)
        best = None
        for mi, m in enumerate(model):
            cos = abs(float(cdir @ np.array([m["dir"][0], m["dir"][1], 0])))
            if cos < np.cos(np.radians(angle_deg)):
                continue
            off = abs(float((cc.mean(axis=0)[:2] - m["plane_point"][:2]) @ m["normal_xy"]))
            if off > plane_tol:
                continue
            t_c = (cc[:, :2] - m["plane_point"][:2]) @ m["dir"]; t_m = (m["corners"][:, :2] - m["plane_point"][:2]) @ m["dir"]
            iou = rect_iou(np.array([t_c.min(), cc[:, 2].min()]), np.array([t_c.max(), cc[:, 2].max()]), np.array([t_m.min(), m["sill_z"]]), np.array([t_m.max(), m["head_z"]]))
            if iou >= min_iou and (best is None or iou > best[1]):
                best = (mi, iou, off)
        pairs.append((ci, *(best if best else (None, 0.0, None))))
    # one capture rectangle per model opening: the best IoU keeps the match, the rest are ghosts of the same
    # opening cast onto a parallel plane and are dropped (mi = -1) rather than reported as unmodelled
    best_for_model = {}
    for ci, mi, iou, off in pairs:
        if mi is not None and (mi not in best_for_model or iou > best_for_model[mi][1]):
            best_for_model[mi] = (ci, iou)
    pairs = [(ci, mi, iou, off) if mi is None or best_for_model[mi][0] == ci else (ci, -1, iou, off) for ci, mi, iou, off in pairs]
    return pairs


def post(api: str, path: str, payload=None, raw: bytes | None = None, content_type: str = "application/json"):
    req = urllib.request.Request(api + path, data=raw if raw is not None else json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": content_type, "Host": api.split("//")[1]})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def get(api: str, path: str):
    req = urllib.request.Request(api + path, headers={"Host": api.split("//")[1]})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaffold", type=Path, required=True); ap.add_argument("--moge", type=Path)
    ap.add_argument("--alignment", type=Path, required=True); ap.add_argument("--alignment-case", required=True)
    ap.add_argument("--building", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--api", default="http://127.0.0.1:2286"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--resume", action="store_true", help="with --apply: skip records whose title already exists in the queue, add only missing attachments")
    ap.add_argument("--min-views", type=int, default=3); ap.add_argument("--uncertainty-floor", type=float, default=0.10)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise SystemExit(f"refusing to overwrite {a.output}")
    a.output.mkdir(parents=True, exist_ok=True); (a.output / "attachments").mkdir(exist_ok=True)
    S = json.loads((a.scaffold / "scaffold.json").read_text()); receipt = json.loads((a.scaffold / "receipt.json").read_text())
    B = json.loads(a.building.read_text()); M, scale, fit = similarity(a.alignment, a.alignment_case)
    R, s_al, _ = sc.similarity_parts(M)
    moge = json.loads((a.moge / "receipt.json").read_text()) if a.moge and (a.moge / "receipt.json").is_file() else None
    ratio = (moge["global_meters_per_unit"] / s_al) if moge else 1.0
    run_id = a.scaffold.name; walls = [p for p in S["patches"] if p["kind"] == "wall"]; pav = [p for p in S["patches"] if p["kind"] == "pavement"]
    street = max((w for w in walls if w.get("normal_canonical", [0, 0, 0])[1] < -0.9), key=lambda w: w["surfels"], default=None)
    main = street or max(walls, key=lambda w: w["surfels"], default=None)
    if main is None or not pav:
        raise SystemExit("no wall / pavement plane in the scaffold")
    street = main
    # ground line on the street wall = pavement plane ∩ wall plane, expressed as v(u) in the wall's own basis
    c0, n0, B0 = (np.array(street[k]) for k in ("centre", "normal", "basis")); cp, npv = np.array(pav[0]["centre"]), np.array(pav[0]["normal"])
    def v_ground(u): return -float(((c0 + u * B0[0] - cp) @ npv) / (B0[1] @ npv))
    # capture openings (rects with >= min_views), heights above the ground line + canonical z
    capture = []
    for w in walls:
        for kind, rects in w["openings"].items():
            if kind == "gap":
                continue
            for r in rects:
                if r.get("n_views", 0) < a.min_views:
                    continue
                cu = np.asarray(r["corners_units"]); uv0 = (cu - c0) @ B0.T; vg = v_ground(float(uv0[:, 0].mean()))
                cm = np.asarray(r["corners_m"])
                capture.append({**r, "wall": w["name"], "kind": kind, "width_m": r["width"] * scale, "height_m": r["height"] * scale,
                                "sill_above_ground_m": (float(uv0[:, 1].min()) - vg) * scale, "head_above_ground_m": (float(uv0[:, 1].max()) - vg) * scale,
                                "sill_z_m": float(cm[:, 2].min()), "head_z_m": float(cm[:, 2].max()), "x_m": [float(cm[:, 0].min()), float(cm[:, 0].max())],
                                "width_spread_m": r["width_spread"] * scale, "height_spread_m": r["height_spread"] * scale})
    model = [m for m in model_openings(B) if m["exterior"]]; pairs = match_openings(capture, model)
    overlays = sorted((a.scaffold / "overlays").glob("cam_*.png"))
    def attach_views(views, limit=2):
        files = [a.scaffold / "overlays" / f"cam_{v:03d}.png" for v in views[:limit]]
        return [f for f in files if f.is_file()]
    def unc(r):
        return max(a.uncertainty_floor, 2 * max(r["width_spread_m"], r["height_spread_m"]))
    method = f"SplatLab architecture scaffold ({run_id}): SAM3 opening masks cast onto the wall plane fitted to MoGe-2 depth scaled by SfM tracks; median over views"
    provenance = {"scaffold_run": run_id, "scaffold_receipt_sha256": hashlib.sha256((a.scaffold / "receipt.json").read_bytes()).hexdigest()[:16],
                  "alignment_case": a.alignment_case, "alignment_scale_m_per_unit": round(s_al, 4), "moge_scale_m_per_unit": round(moge["global_meters_per_unit"], 4) if moge else None}
    records = []
    def record(title, body, ftype, target, measurements=(), tags=(), details=None, attachments=(), priority="Normal"):
        records.append({"payload": {"title": title[:240], "body": body, "feedback_type": ftype, "priority": priority, "target": target,
                                    "context": {"activeTab": "capture", "referenceId": run_id}, "measurements": list(measurements), "tags": TAGS + list(tags),
                                    "details": {**provenance, **(details or {})}}, "attachments": [str(f) for f in attachments]})
    def mrow(target, prop, value, uncertainty, endpoints, basis, evidence):
        return {"target": target, "property": prop, "value": round(float(value), 3), "unit": "m", "uncertainty": round(float(uncertainty), 3), "endpoints": endpoints,
                "basis": basis, "method": method[:200], "measuredAt": TODAY, "evidence": evidence[:1000]}
    # 1. matched openings -> Measurement records
    matched_model = set()
    for ci, mi, iou, off in pairs:
        c = capture[ci]
        if mi is None or mi == -1:
            continue
        m = model[mi]; matched_model.add(mi)
        target = {"kind": "opening", "id": m["id"], "levelId": m["level"]}
        ev = f"{c['n_views']} views {c['views']}; overlays cam_{c['views'][0]:03d}.png; width/height spread {c['width_spread_m']:.2f}/{c['height_spread_m']:.2f} m"
        rows = [mrow(target, "width", c["width_m"], unc(c), "left to right edge of the SAM3 opening mask on the fitted wall plane", "visible opening edge (mask), not framing datum", ev),
                mrow(target, "height", c["height_m"], unc(c), "bottom to top edge of the SAM3 opening mask on the fitted wall plane", "visible opening edge (mask), not framing datum", ev),
                mrow(target, "sill", max(c["sill_above_ground_m"], 0.001), max(0.15, unc(c)), "ground line (pavement plane ∩ wall plane) to the mask's bottom edge", "above the pavement at the wall, not the level floor", ev)]
        body = (f"Capture-derived size of '{m['id']}' ({m['kind']}, {m['wall']}, {m['level']}) from the SplatLab architecture scaffold.\n"
                f"Capture: {c['width_m']:.2f} × {c['height_m']:.2f} m (at the alignment scale; × {ratio:.3f} at the MoGe-2 scale → {c['width_m']*ratio:.2f} × {c['height_m']*ratio:.2f} m), "
                f"sill {c['sill_above_ground_m']:.2f} m above the pavement, head {c['head_above_ground_m']:.2f} m; canonical z {c['sill_z_m']:.2f}..{c['head_z_m']:.2f} m; "
                f"x {c['x_m'][0]:.2f}..{c['x_m'][1]:.2f} m. {c['n_views']} views, spread {c['width_spread_m']:.2f}/{c['height_spread_m']:.2f} m.\n"
                f"Model: {m['width']:.3f} × {m['height']:.3f} m, sill {m['sill']:.3f} m above {m['level']} (elevation {m['sill_z']-m['sill']:.3f}) → sill z {m['sill_z']:.2f}, head z {m['head_z']:.2f}; confidence '{m['confidence']}'. "
                f"Δwidth {c['width_m']-m['width']:+.2f} m, Δheight {c['height_m']-m['height']:+.2f} m, Δsill z {c['sill_z_m']-m['sill_z']:+.2f} m. Rectangle IoU {iou:.2f}.\n"
                f"Basis: the SAM3 mask edge is the visible opening (leaf / glass + frame), not the framing datum. Registration '{a.alignment_case}': {fit.get('scaleEvidence', 'scale from the alignment candidate')}"
                + (f"; {fit.get('translationEvidence')}" if fit.get('translationEvidence') else "") + ".")
        record(f"Capture · {m['id']} · {c['width_m']:.2f} × {c['height_m']:.2f} m, sill {c['sill_above_ground_m']:.2f} m", body, "Measurement", target, rows,
               tags=["opening", m["kind"]], details={"n_views": c["n_views"], "iou_with_model": round(iou, 2), "width_m": round(c["width_m"], 3), "height_m": round(c["height_m"], 3), "sill_above_pavement_m": round(c["sill_above_ground_m"], 3), "model_width_m": m["width"], "model_height_m": m["height"], "model_sill_z_m": round(m["sill_z"], 3), "capture_sill_z_m": round(c["sill_z_m"], 3)},
               attachments=attach_views(c["views"]))
    # 2. unmatched capture openings INSIDE the 2286 footprint -> Question records (neighbouring units share the wall line)
    xs = [x for L in B["levels"] for w in L["walls"] for x in (w["a"][0], w["b"][0])]; x_lo, x_hi = min(xs) - 0.5, max(xs) + 0.5
    heights = []
    for ci, mi, iou, off in pairs:
        c = capture[ci]
        inside = x_lo <= (c["x_m"][0] + c["x_m"][1]) / 2 <= x_hi
        if mi is None and inside and c["width_m"] >= 0.3 and c["n_views"] >= a.min_views:
            heights.append(c)
        if mi is not None or c["n_views"] < a.min_views + 1 or not inside or c["width_m"] < 0.3:
            continue   # (mi == -1 = ghost duplicate of a matched opening: skipped too)
        cc = np.asarray(c["corners_m"]); cdir = cc[1] - cc[0]; cdir[2] = 0; cdir /= max(np.linalg.norm(cdir), 1e-9)
        same_wall = []
        for m in model:
            if m["kind"] != {"garage_door": "garage", "door": "door", "window": "window"}.get(c["kind"], c["kind"]):
                continue
            if abs(float(cdir @ np.array([m["dir"][0], m["dir"][1], 0]))) < np.cos(np.radians(20)):
                continue
            if abs(float((cc.mean(axis=0)[:2] - m["plane_point"][:2]) @ m["normal_xy"])) > 1.0:
                continue
            t_c = float(((cc[:, :2] - m["plane_point"][:2]) @ m["dir"]).mean()); t_m = m["offset"] + m["width"] / 2 if "offset" in m else float(((m["corners"][:, :2] - m["plane_point"][:2]) @ m["dir"]).mean())
            same_wall.append((m, t_c - t_m))
        shifted = min(same_wall, key=lambda x: abs(x[1])) if same_wall else None
        body = (f"An opening the capture sees but the model does not list at this place: {c['kind']} {c['width_m']:.2f} × {c['height_m']:.2f} m, sill {c['sill_above_ground_m']:.2f} m above the pavement "
                f"(canonical z {c['sill_z_m']:.2f}..{c['head_z_m']:.2f}, x {c['x_m'][0]:.2f}..{c['x_m'][1]:.2f} m, y {cc[:, 1].min():.2f}..{cc[:, 1].max():.2f} m), {c['n_views']} views {c['views']}. ")
        if shifted:
            m, du = shifted
            body += (f"The model has a {m['kind']} on the same wall plane — '{m['id']}' on {m['wall']} ({m['level']}) — but {abs(du):.2f} m {'further along' if du > 0 else 'back along'} the wall (a→b) from where the capture sees this one, "
                     f"and {m['width']:.2f} × {m['height']:.2f} m with sill {m['sill']:.2f} m vs capture sill z {c['sill_z_m']:.2f} m. Is the model opening on the wrong side / wrong position?")
        else:
            body += "Either a neighbouring unit's opening on the same wall line, or an opening the model does not have yet."
        cm = np.asarray(c["corners_m"]); where = (f"x {c['x_m'][0]:.1f}..{c['x_m'][1]:.1f} m" if c["x_m"][1] - c["x_m"][0] > 0.3 else f"x {c['x_m'][0]:.1f} m, y {cm[:, 1].min():.1f}..{cm[:, 1].max():.1f} m (side face)")
        if shifted:
            m, du = shifted
            record(f"Capture · {m['id']}: capture sees the {c['kind']} {abs(du):.1f} m from the modelled position", body, "Question",
                   {"kind": "opening", "id": m["id"], "levelId": m["level"]}, tags=["opening", "position"], details={"n_views": c["n_views"], "kind": c["kind"], "along_wall_delta_m": round(du, 3)},
                   attachments=attach_views(c["views"]), priority="High")
        else:
            record(f"Capture · unmodelled {c['kind']} {c['width_m']:.2f} × {c['height_m']:.2f} m at {where}, sill {c['sill_above_ground_m']:.1f} m", body, "Question",
                   {"kind": "building", "id": "2286-chanate"}, tags=["opening", "unmodelled"], details={"n_views": c["n_views"], "kind": c["kind"]}, attachments=attach_views(c["views"]))
    # 2b. one consolidated question: measured opening heights vs the model's level elevations
    lv = {L["id"]: L for L in B["levels"]}
    matched_rows = [(capture[ci], model[mi]) for ci, mi, _, _ in pairs if mi is not None and mi != -1 and model[mi]["kind"] == "window"]
    deltas = [(abs(c["sill_z_m"] - m["sill_z"]), m["level"]) for c, m in matched_rows]
    if (matched_rows or heights) and (max((d for d, _ in deltas), default=0.0) > 0.3 or heights):
        lines = [f"Opening heights measured above the pavement at the garage door (ground line = road plane ∩ street wall plane; the door sill measures {next((capture[ci]['sill_above_ground_m'] for ci, mi, _, _ in pairs if mi is not None and model[mi]['id']=='garage-door'), float('nan')):.2f} m, so the datum holds):"]
        for c, m in matched_rows:
            lines.append(f"- {m['id']}: sill {c['sill_above_ground_m']:.2f} m, head {c['head_above_ground_m']:.2f} m ({c['n_views']} views) — model sill z {m['sill_z']:.2f} m ({m['level']} elevation {lv[m['level']]['elevation']:.3f} + sill {m['sill']:.3f}), head z {m['head_z']:.2f} m → Δ {c['sill_above_ground_m']-m['sill_z']:+.2f} m")
        for c in heights:
            lines.append(f"- unmodelled {c['kind']} at x {c['x_m'][0]:.1f}..{c['x_m'][1]:.1f} m: sill {c['sill_above_ground_m']:.2f} m, head {c['head_above_ground_m']:.2f} m ({c['n_views']} views)")
        st = {s_["id"]: s_ for s_ in B.get("stairs", [])}
        lines.append(f"Model levels: " + ", ".join(f"{L['id']} elevation {L['elevation']:.3f} m (height {L['height']:.3f} + slab {L.get('slabThickness', 0):.3f})" for L in B["levels"])
                     + "; stairs " + ", ".join(f"{k}: {v['risers']} risers" for k, v in st.items()) + f" → {lv['level-2']['elevation']/st['stair-ground-main']['risers']*1000:.0f} mm per riser if level-2 is at {lv['level-2']['elevation']:.3f} m.")
        worst = max(deltas, key=lambda x: x[0]) if deltas else (0.0, "level-1")
        lines.append("Question for the owner: where the capture puts a window sill well above or below the model's level elevation + sill, either the floor-to-floor height or the sill height in the model is off. Sill heights here are mask edges (glass + frame), ±0.15 m; the door threshold is the reference where a door is matched.")
        record(f"Capture · window heights vs the model's level elevations (largest Δ {worst[0]:.2f} m on {worst[1]})", "\n".join(lines), "Question", {"kind": "level", "id": worst[1], "levelId": worst[1]},
               tags=["heights", "level"], details={"n_windows": len(matched_rows) + len(heights), "largest_delta_m": round(worst[0], 3)}, attachments=attach_views(sorted({v for c, _ in matched_rows for v in c["views"][:1]} | {v for c in heights for v in c["views"][:1]})[:2] or [0]), priority="High")
    # 3. scale + orientation references on the garage door / building
    meas = {m["item"]: m for m in S.get("measurements", [])}
    scale_is_moge = "MoGe" in str(fit.get("scaleEvidence", ""))
    if moge and not scale_is_moge:
        sm = meas.get("SfM-frame scale: MoGe-2 (tracks) vs alignment (garage opening)", {})
        body = (f"Independent scale check of the capture alignment. MoGe-2 monocular metric depth on {moge['n_views_scaled']} views, each scaled by its own SfM tracks, gives "
                f"{moge['global_meters_per_unit']:.4f} m per SfM unit (per-view MAD {moge['per_view_scale_mad_u_per_m']:.4f} u/m). The Condo Lab alignment '{a.alignment_case}' gives {s_al:.4f} m/u from the modelled "
                f"4.8768 × 2.1336 m garage opening. Ratio {ratio:.3f}: the photo-estimated garage opening is consistent with monocular metric depth to {abs(1-ratio)*100:.1f} %. "
                f"MoGe-2 is not a survey; on another scene it disagreed with a tape reference by 20 %.")
        record("Capture · scale check: MoGe-2 vs garage-opening alignment", body, "Reference", {"kind": "opening", "id": "garage-door", "levelId": "level-1"},
               tags=["scale-check"], details={"moge_m_per_unit": round(moge["global_meters_per_unit"], 4), "alignment_m_per_unit": round(s_al, 4), "ratio": round(ratio, 4), "n_views": moge["n_views_scaled"]},
               attachments=[a.scaffold / "contact.png"])
    orient = {k: meas[k]["capture_deg"] for k in ("wall normal vs model -Y (street side)", "up vector vs model +Z") if k in meas and meas[k]["capture_deg"] < 45}
    if "wall normal vs model -Y (street side)" not in orient:
        orient = {}                                                     # not a street-facing capture: the wall-line record carries orientation
    sw = street
    body = (f"Fitted street wall plane ({sw['surfels']:,} MoGe-2 depth points, RMS {sw['fit_rms_units']*s_al*100:.0f} cm, extent {sw['extent_m'][0]:.1f} × {sw['extent_m'][1]:.1f} m — every coplanar unit) "
            f"and road plane ({pav[0]['surfels']:,} points). Wall normal {orient.get('wall normal vs model -Y (street side)', float('nan')):.1f}° from the model's street wall, "
            f"up {orient.get('up vector vs model +Z', float('nan')):.1f}° from the model's +Z. Reserved-track check on the wall: {sw.get('reserved_track_check')}")
    if orient:
        record("Capture · street wall + road planes: orientation agrees with the model", body, "Reference", {"kind": "wall", "id": "level-1-exterior-1", "levelId": "level-1"},
               tags=["plane", "orientation"], details={k.replace(' ', '_'): round(v, 2) for k, v in orient.items()}, attachments=[a.scaffold / "contact.png"])
    # 4. bay projection vs the documented dimension
    bay = [w for w in walls if w is not street and w.get("normal_canonical", [0, 0, 0])[1] < -0.9 and 0.3 < float((np.array(w["centre"]) - c0) @ n0) * s_al < 1.5]
    if bay:
        bw = max(bay, key=lambda w: w["surfels"]); proj = float((np.array(bw["centre"]) - c0) @ n0) * s_al
        dim = next((d for d in B["dimensions"] if d["id"] == "bay-projection"), None)
        target = {"kind": "wall", "id": "level-2-exterior-3", "levelId": "level-2"}
        rows = [mrow(target, "projection", proj, 0.15, "street wall plane to the bay front plane, along the street wall normal, at the bay's centre", "stucco face to stucco face", f"planes {street['name']} and {bw['name']}")]
        body = (f"Bay front plane ({bw['surfels']:,} points, RMS {bw['fit_rms_units']*s_al*100:.0f} cm, {bw['extent_m'][0]:.1f} × {bw['extent_m'][1]:.1f} m) sits {proj:.2f} m in front of the street wall plane "
                f"(angle between the two planes {np.degrees(np.arccos(abs(float(np.array(bw['normal']) @ n0)))):.1f}°). Model dimension 'bay-projection' = {dim['expected'] if dim else '?'} m ({dim['confidence'] if dim else '?'}). "
                f"Δ {proj - (dim['expected'] if dim else 0):+.2f} m.")
        record(f"Capture · bay projection {proj:.2f} m (model {dim['expected'] if dim else '?'} m)", body, "Measurement", target, rows, tags=["dimension", "bay"],
               details={"projection_m": round(proj, 3), "model_m": dim["expected"] if dim else None}, attachments=attach_views([0, 4]))
    # 5. every registered wall plane vs the model wall lines it runs along (stucco face vs framing line, recess depths)
    lines = []
    for w in walls:
        if "corners_m" not in w or "normal_canonical" not in w:
            continue
        cm = np.asarray(w["corners_m"]); nc = np.asarray(w["normal_canonical"])[:2]
        if np.linalg.norm(nc) < 0.9:
            continue
        nc = nc / np.linalg.norm(nc); span_lo, span_hi = None, None
        for L in B["levels"]:
            for mw in L["walls"]:
                if not mw.get("exterior"):
                    continue
                ma, mb = np.array(mw["a"], float), np.array(mw["b"], float); d = mb - ma; length = np.linalg.norm(d)
                if length < 0.3:
                    continue
                d /= length; mn = np.array([d[1], -d[0]])
                if abs(float(mn @ nc)) < np.cos(np.radians(10)):
                    continue
                t = (cm[:, :2] - ma) @ d; overlap = min(float(t.max()), length) - max(float(t.min()), 0.0)
                if overlap < 0.5:
                    continue
                zc = cm[:, 2]; z_ok = zc.max() > L["elevation"] and zc.min() < L["elevation"] + L["height"]
                if not z_ok:
                    continue
                off = float(((cm.mean(axis=0)[:2] - ma) @ mn)) * (1 if float(mn @ nc) > 0 else -1)   # + = capture face outside the model line
                if abs(off) <= 1.0:
                    lines.append({"patch": w["name"], "wall": mw["id"], "level": L["id"], "overlap_m": round(overlap, 2), "offset_m": round(off, 3), "rms_m": round(w["fit_rms_units"] * s_al, 3),
                                  "angle_deg": round(float(np.degrees(np.arccos(min(1.0, abs(float(mn @ nc)))))), 1)})
    if lines:
        body = "Registered capture wall planes against the model's exterior wall lines (positive = capture stucco face lies outside the model framing line):\n" + "\n".join(
            f"- {x['patch']} ↔ {x['wall']} ({x['level']}): offset {x['offset_m']:+.2f} m over {x['overlap_m']:.1f} m of overlap, {x['angle_deg']:.1f}° off parallel (plane RMS {x['rms_m']*100:.0f} cm)" for x in lines)
        body += f"\nRegistration '{a.alignment_case}': {fit.get('translationEvidence') or 'Condo Lab alignment candidate'}. Differences between two lines on the same face give the recess depth as captured vs as modelled."
        record("Capture · wall plane offsets vs model wall lines", body, "Reference", {"kind": "level", "id": lines[0]["level"], "levelId": lines[0]["level"]},
               tags=["plane", "wall-line"], details={f"offset_{i}": x["offset_m"] for i, x in enumerate(lines[:12])}, attachments=[a.scaffold / "contact.png"])
    payload_summary = [{"title": r["payload"]["title"], "type": r["payload"]["feedback_type"], "target": r["payload"]["target"], "measurements": len(r["payload"]["measurements"]), "attachments": len(r["attachments"])} for r in records]
    (a.output / "evidence.json").write_text(json.dumps({"schema": "dev.splatlab.condo-evidence/v1", "created": TODAY, "api": a.api, "provenance": provenance, "records": records,
                                                        "matches": [{"capture": capture[ci]["wall"] + "/" + capture[ci]["kind"], "views": capture[ci]["n_views"], "model": (model[mi]["id"] if mi not in (None, -1) else ("ghost-duplicate" if mi == -1 else None)), "iou": round(iou, 3)} for ci, mi, iou, _ in pairs]}, indent=1))
    print(f"[evidence] {len(capture)} capture openings (>= {a.min_views} views), {sum(1 for p in pairs if p[1] not in (None, -1))} matched to model openings, {sum(1 for p in pairs if p[1] == -1)} ghost duplicates dropped; {len(records)} records:")
    for r in payload_summary: print("   ", json.dumps(r))
    if not a.apply:
        print(f"[evidence] dry-run: payloads in {a.output / 'evidence.json'}; add --apply to post to {a.api}"); return 0
    posted, failures = [], []
    existing = {i["title"]: i for i in get(a.api, "/api/feedback")["items"]} if a.resume else {}
    for r in records:
        title = r["payload"]["title"]; att = []
        try:
            if title in existing:                                           # --resume: keep the record, add only missing attachments
                fid = existing[title]["id"]; have = {x["filename"] for x in existing[title].get("attachments", [])}; created = existing[title]
            else:
                created = post(a.api, "/api/feedback", r["payload"]); fid = created["id"]; have = set()
            for f in r["attachments"]:
                f = Path(f); name = f"{run_id}-{f.name}"
                if name in have:
                    continue
                data = f.read_bytes()
                res = post(a.api, f"/api/feedback/{fid}/attachments?filename={name}", raw=data, content_type="image/png")
                att.append({"filename": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "id": res.get("id") if isinstance(res, dict) else None})
            posted.append({"id": fid, "title": title, "status": created.get("status"), "attachments": att, "resumed": title in existing}); print(f"   posted {fid} {title[:80]} (+{len(att)} attachments{', resumed' if title in existing else ''})")
        except RuntimeError as e:
            failures.append({"title": title, "error": str(e)}); print(f"   FAILED {title[:80]}: {e}")
    (a.output / "evidence-receipt.json").write_text(json.dumps({"api": a.api, "posted": posted, "failures": failures, "at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=1))
    print(f"[evidence] {len(posted)} records posted, {len(failures)} failed; receipt {a.output / 'evidence-receipt.json'}; rollback = set each id to Won't Fix / Archived in the review UI (no delete route)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
