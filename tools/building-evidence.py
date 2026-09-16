#!/usr/bin/env python3
"""Turn an architecture-scaffold run into review-queue evidence for a building model (CPU, splatops env).

Building-agnostic: the model (`building.json` in the Condo Lab / reality-regenerator shape), the review
API, the building target id, the floor datums and the tags are inputs. Reads scaffold.json (+ overlays),
the MoGe-2 depth receipt, a registration receipt (source → canonical metres similarity; a Condo Lab
alignment receipt or register-by-openings.py output) and building.json; matches capture openings to
model openings by wall plane + rectangle IoU (backend/architecture/model_match.py); writes
<output>/evidence.json = a list of POST /api/feedback payloads and, with --apply, posts them to the
review server, writing evidence-receipt.json. Nothing changes the accepted model.

Records: Measurement per matched opening (width / height / sill rows, basis = visible mask edge),
High Question where the model has the same kind of opening elsewhere on that wall, Question per
unmodelled opening inside the footprint, one heights Question (sills vs level elevations, against
named datums), Reference for the scale check (when the registration scale is not MoGe itself),
Reference for wall-plane offsets vs model lines, Measurement per captured projection/recess between
two parallel walls (cited against a matching model `dimensions` entry when one exists).

  building-evidence.py --scaffold DIR --moge DIR --alignment receipt.json --alignment-case NAME --building building.json \\
      --building-target building:<id> --output DIR [--floor-datums "slab=0,FF=0.3048"] [--api http://127.0.0.1:2286] [--apply] [--resume]
"""
from __future__ import annotations

import argparse, datetime as dt, hashlib, json, sys, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from architecture import model_match as mm     # noqa: E402
from architecture import scaffold_core as sc   # noqa: E402

TODAY = dt.date.today().isoformat()


def similarity(alignment: Path, case_name: str):
    ar = json.loads(alignment.read_text())
    case = next(c for c in ar["cases"] if c["name"] == case_name)
    return np.asarray(case["fit"]["matrix"], dtype=np.float64).reshape(4, 4), float(case["fit"]["scale"]), case["fit"]


def target_of(spec: str) -> dict:
    kind, _, ident = spec.partition(":")
    return {"kind": kind, "id": ident}


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
    ap.add_argument("--building-target", required=True, help="review target for building-wide records, kind:id (e.g. building:2286-chanate)")
    ap.add_argument("--scale-check-target", default=None, help="review target for the scale-check record, kind:id (default: the building target)")
    ap.add_argument("--api", default="http://127.0.0.1:2286"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--resume", action="store_true", help="with --apply: skip records whose title already exists in the queue, add only missing attachments")
    ap.add_argument("--min-views", type=int, default=3); ap.add_argument("--uncertainty-floor", type=float, default=0.10)
    ap.add_argument("--floor-datums", default="", help="named heights above the model's z=0 (m), e.g. 'garage slab=0,studio FF=0.3048'")
    ap.add_argument("--tags", default="capture,splatlab,architecture-scaffold"); ap.add_argument("--footprint-margin", type=float, default=0.5)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise SystemExit(f"refusing to overwrite {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    S = json.loads((a.scaffold / "scaffold.json").read_text())
    B = json.loads(a.building.read_text()); M, scale, fit = similarity(a.alignment, a.alignment_case)
    R, s_al, _ = sc.similarity_parts(M)
    moge = json.loads((a.moge / "receipt.json").read_text()) if a.moge and (a.moge / "receipt.json").is_file() else None
    ratio = (moge["global_meters_per_unit"] / s_al) if moge else 1.0
    TAGS = [t.strip() for t in a.tags.split(",") if t.strip()]
    run_id = a.scaffold.name; walls = [p for p in S["patches"] if p["kind"] == "wall"]; pav = [p for p in S["patches"] if p["kind"] == "pavement"]
    datums = []
    for item in (a.floor_datums or "").split(","):
        if "=" in item:
            name, value = item.rsplit("=", 1); datums.append((name.strip(), float(value)))
    def against_datums(z: float) -> str:
        return "; ".join(f"{z - d:+.2f} m above {name}" for name, d in datums)
    model = mm.model_openings(B); mwalls = mm.model_walls(B)
    lines = mm.wall_line_matches(walls, mwalls)
    if not walls or not pav:
        raise SystemExit("no wall / pavement plane in the scaffold")
    # reference wall for the ground line: the largest wall that sits on a model line, else the largest wall
    matched_names = {x["patch"] for x in lines}
    main_wall = max([w for w in walls if w["name"] in matched_names] or walls, key=lambda w: w["surfels"])
    c0, n0, B0 = (np.array(main_wall[k]) for k in ("centre", "normal", "basis")); cp, npv = np.array(pav[0]["centre"]), np.array(pav[0]["normal"])
    def v_ground(u): return -float(((c0 + u * B0[0] - cp) @ npv) / (B0[1] @ npv))
    capture = []
    for w in walls:
        for kind, rects in w["openings"].items():
            if kind == "gap":
                continue
            for r in rects:
                if r.get("n_views", 0) < a.min_views or "corners_m" not in r:
                    continue
                cu = np.asarray(r["corners_units"]); uv0 = (cu - c0) @ B0.T; vg = v_ground(float(uv0[:, 0].mean()))
                cm = np.asarray(r["corners_m"])
                capture.append({**r, "wall": w["name"], "kind": kind, "width_m": r["width"] * scale, "height_m": r["height"] * scale,
                                "sill_above_ground_m": (float(uv0[:, 1].min()) - vg) * scale, "head_above_ground_m": (float(uv0[:, 1].max()) - vg) * scale,
                                "sill_z_m": float(cm[:, 2].min()), "head_z_m": float(cm[:, 2].max()), "x_m": [float(cm[:, 0].min()), float(cm[:, 0].max())],
                                "centre_xy": cm.mean(axis=0)[:2].tolist(), "width_spread_m": r["width_spread"] * scale, "height_spread_m": r["height_spread"] * scale})
    pairs = mm.match_openings(capture, model)
    def attach_views(views, limit=2):
        return [f for f in (a.scaffold / "overlays" / f"cam_{v:03d}.png" for v in views[:limit]) if f.is_file()]
    def unc(r):
        return max(a.uncertainty_floor, 2 * max(r["width_spread_m"], r["height_spread_m"]))
    method = f"SplatLab architecture scaffold ({run_id}): SAM3 opening masks cast onto the wall plane fitted to MoGe-2 depth scaled by SfM tracks; median over views"
    provenance = {"scaffold_run": run_id, "scaffold_receipt_sha256": hashlib.sha256((a.scaffold / "receipt.json").read_bytes()).hexdigest()[:16],
                  "alignment_case": a.alignment_case, "alignment_scale_m_per_unit": round(s_al, 4), "moge_scale_m_per_unit": round(moge["global_meters_per_unit"], 4) if moge else None}
    building_target = target_of(a.building_target)
    records = []
    def record(title, body, ftype, target, measurements=(), tags=(), details=None, attachments=(), priority="Normal"):
        records.append({"payload": {"title": title[:240], "body": body, "feedback_type": ftype, "priority": priority, "target": target,
                                    "context": {"activeTab": "capture", "referenceId": run_id}, "measurements": list(measurements), "tags": TAGS + list(tags),
                                    "details": {**provenance, **(details or {})}}, "attachments": [str(f) for f in attachments]})
    def mrow(target, prop, value, uncertainty, endpoints, basis, evidence):
        return {"target": target, "property": prop, "value": round(float(value), 3), "unit": "m", "uncertainty": round(float(uncertainty), 3), "endpoints": endpoints,
                "basis": basis, "method": method[:200], "measuredAt": TODAY, "evidence": evidence[:1000]}
    reg_note = f"Registration '{a.alignment_case}': {fit.get('scaleEvidence', 'scale from the alignment candidate')}" + (f"; {fit.get('translationEvidence')}" if fit.get("translationEvidence") else "")
    # 1. matched openings -> Measurement records
    for ci, mi, iou, off in pairs:
        c = capture[ci]
        if mi is None or mi == -1:
            continue
        m = model[mi]
        target = {"kind": "opening", "id": m["id"], "levelId": m["level"]}
        ev = f"{c['n_views']} views {c['views']}; overlays cam_{c['views'][0]:03d}.png; width/height spread {c['width_spread_m']:.2f}/{c['height_spread_m']:.2f} m"
        rows = [mrow(target, "width", c["width_m"], unc(c), "left to right edge of the SAM3 opening mask on the fitted wall plane", "visible opening edge (mask), not framing datum", ev),
                mrow(target, "height", c["height_m"], unc(c), "bottom to top edge of the SAM3 opening mask on the fitted wall plane", "visible opening edge (mask), not framing datum", ev),
                mrow(target, "sill", max(c["sill_above_ground_m"], 0.001), max(0.15, unc(c)), "ground line (pavement plane ∩ wall plane) to the mask's bottom edge", "above the pavement at the wall, not the level floor", ev)]
        body = (f"Capture-derived size of '{m['id']}' ({m['kind']}, {m['wall']}, {m['level']}) from the SplatLab architecture scaffold.\n"
                f"Capture: {c['width_m']:.2f} × {c['height_m']:.2f} m (at the registration scale; × {ratio:.3f} at the MoGe-2 scale → {c['width_m']*ratio:.2f} × {c['height_m']*ratio:.2f} m), "
                f"sill {c['sill_above_ground_m']:.2f} m above the pavement, head {c['head_above_ground_m']:.2f} m; canonical z {c['sill_z_m']:.2f}..{c['head_z_m']:.2f} m"
                + (f" (sill {against_datums(c['sill_z_m'])}; head {against_datums(c['head_z_m'])})" if datums else "") + "; "
                f"x {c['x_m'][0]:.2f}..{c['x_m'][1]:.2f} m. {c['n_views']} views, spread {c['width_spread_m']:.2f}/{c['height_spread_m']:.2f} m.\n"
                f"Model: {m['width']:.3f} × {m['height']:.3f} m, sill {m['sill']:.3f} m above {m['level']} (elevation {m['sill_z']-m['sill']:.3f}) → sill z {m['sill_z']:.2f}, head z {m['head_z']:.2f}; confidence '{m['confidence']}'. "
                f"Δwidth {c['width_m']-m['width']:+.2f} m, Δheight {c['height_m']-m['height']:+.2f} m, Δsill z {c['sill_z_m']-m['sill_z']:+.2f} m. Rectangle IoU {iou:.2f}.\n"
                f"Basis: the SAM3 mask edge is the visible opening (leaf / glass + frame), not the framing datum. {reg_note}.")
        record(f"Capture · {m['id']} · {c['width_m']:.2f} × {c['height_m']:.2f} m, sill {c['sill_above_ground_m']:.2f} m", body, "Measurement", target, rows,
               tags=["opening", m["kind"]], details={"n_views": c["n_views"], "iou_with_model": round(iou, 2), "width_m": round(c["width_m"], 3), "height_m": round(c["height_m"], 3), "sill_above_pavement_m": round(c["sill_above_ground_m"], 3), "model_width_m": m["width"], "model_height_m": m["height"], "model_sill_z_m": round(m["sill_z"], 3), "capture_sill_z_m": round(c["sill_z_m"], 3)},
               attachments=attach_views(c["views"]))
    # 2. unmatched capture openings inside the footprint -> position Questions (same kind exists on that wall) or unmodelled Questions
    heights = []
    for ci, mi, iou, off in pairs:
        c = capture[ci]
        inside = mm.inside_footprint(c["centre_xy"], B, a.footprint_margin)
        if mi is None and inside and c["width_m"] >= 0.3 and c["n_views"] >= a.min_views:
            heights.append(c)
        if mi is not None or c["n_views"] < a.min_views + 1 or not inside or c["width_m"] < 0.3:
            continue
        cc = np.asarray(c["corners_m"]); shifted = mm.same_wall_shift(c, model)
        body = (f"An opening the capture sees but the model does not list at this place: {c['kind']} {c['width_m']:.2f} × {c['height_m']:.2f} m, sill {c['sill_above_ground_m']:.2f} m above the pavement "
                f"(canonical z {c['sill_z_m']:.2f}..{c['head_z_m']:.2f}, x {c['x_m'][0]:.2f}..{c['x_m'][1]:.2f} m, y {cc[:, 1].min():.2f}..{cc[:, 1].max():.2f} m), {c['n_views']} views {c['views']}. ")
        if shifted:
            m, du = shifted
            body += (f"The model has a {m['kind']} on the same wall plane — '{m['id']}' on {m['wall']} ({m['level']}) — but {abs(du):.2f} m {'further along' if du > 0 else 'back along'} the wall (a→b) from where the capture sees this one, "
                     f"and {m['width']:.2f} × {m['height']:.2f} m with sill {m['sill']:.2f} m vs capture sill z {c['sill_z_m']:.2f} m. Is the model opening on the wrong side / wrong position?")
            record(f"Capture · {m['id']}: capture sees the {c['kind']} {abs(du):.1f} m from the modelled position", body, "Question",
                   {"kind": "opening", "id": m["id"], "levelId": m["level"]}, tags=["opening", "position"], details={"n_views": c["n_views"], "kind": c["kind"], "along_wall_delta_m": round(du, 3)},
                   attachments=attach_views(c["views"]), priority="High")
        else:
            body += "Either a neighbouring building's opening on the same wall line, or an opening the model does not have yet."
            where = (f"x {c['x_m'][0]:.1f}..{c['x_m'][1]:.1f} m" if c["x_m"][1] - c["x_m"][0] > 0.3 else f"x {c['x_m'][0]:.1f} m, y {cc[:, 1].min():.1f}..{cc[:, 1].max():.1f} m (side face)")
            record(f"Capture · unmodelled {c['kind']} {c['width_m']:.2f} × {c['height_m']:.2f} m at {where}, sill {c['sill_above_ground_m']:.1f} m", body, "Question",
                   building_target, tags=["opening", "unmodelled"], details={"n_views": c["n_views"], "kind": c["kind"]}, attachments=attach_views(c["views"]))
    # 3. one heights Question when any matched window sill disagrees with its level elevation + sill by > 0.3 m
    lv = {L["id"]: L for L in B["levels"]}
    matched_rows = [(capture[ci], model[mi]) for ci, mi, _, _ in pairs if mi not in (None, -1) and model[mi]["kind"] == "window"]
    deltas = [(abs(c["sill_z_m"] - m["sill_z"]), m["level"]) for c, m in matched_rows]
    if (matched_rows or heights) and (max((d for d, _ in deltas), default=0.0) > 0.3 or heights):
        ground_ref = next((capture[ci]["sill_above_ground_m"] for ci, mi, _, _ in pairs if mi not in (None, -1) and model[mi]["kind"] in ("garage", "door")), None)
        lines_txt = ["Opening heights measured above the pavement at the reference wall (ground line = pavement plane ∩ wall plane)" + (f"; a matched door sill measures {ground_ref:.2f} m, so the datum holds" if ground_ref is not None else "") + ":"]
        for c, m in matched_rows:
            lines_txt.append(f"- {m['id']}: sill {c['sill_above_ground_m']:.2f} m, head {c['head_above_ground_m']:.2f} m ({c['n_views']} views)" + (f" [{against_datums(c['sill_z_m'])}]" if datums else "") +
                             f" — model sill z {m['sill_z']:.2f} m ({m['level']} elevation {lv[m['level']]['elevation']:.3f} + sill {m['sill']:.3f}), head z {m['head_z']:.2f} m → Δ {c['sill_z_m']-m['sill_z']:+.2f} m")
        for c in heights:
            lines_txt.append(f"- unmodelled {c['kind']} at x {c['x_m'][0]:.1f}..{c['x_m'][1]:.1f} m: sill {c['sill_above_ground_m']:.2f} m, head {c['head_above_ground_m']:.2f} m ({c['n_views']} views)")
        lines_txt.append("Model levels: " + ", ".join(f"{L['id']} elevation {L['elevation']:.3f} m (height {L['height']:.3f} + slab {L.get('slabThickness', 0):.3f})" for L in B["levels"]) + ".")
        lines_txt.append("Question for the owner: where the capture puts a window sill well above or below the model's level elevation + sill, either the floor-to-floor height or the sill height in the model is off. Sill heights here are mask edges (glass + frame), ±0.15 m; a matched door threshold is the reference where one exists.")
        worst = max(deltas, key=lambda x: x[0]) if deltas else (0.0, B["levels"][0]["id"])
        record(f"Capture · window heights vs the model's level elevations (largest Δ {worst[0]:.2f} m on {worst[1]})", "\n".join(lines_txt), "Question", {"kind": "level", "id": worst[1], "levelId": worst[1]},
               tags=["heights", "level"], details={"n_windows": len(matched_rows) + len(heights), "largest_delta_m": round(worst[0], 3)},
               attachments=attach_views(sorted({v for c, _ in matched_rows for v in c["views"][:1]} | {v for c in heights for v in c["views"][:1]})[:2] or [0]), priority="High")
    # 4. scale check (only when the registration's scale is something other than MoGe-2 itself)
    if moge and "MoGe" not in str(fit.get("scaleEvidence", "")):
        body = (f"Independent scale check of the registration. MoGe-2 monocular metric depth on {moge['n_views_scaled']} views, each scaled by its own SfM tracks, gives "
                f"{moge['global_meters_per_unit']:.4f} m per SfM unit (per-view MAD {moge['per_view_scale_mad_u_per_m']:.4f} u/m). Registration '{a.alignment_case}' gives {s_al:.4f} m/u "
                f"({fit.get('scaleEvidence', 'scale evidence not recorded')}). Ratio {ratio:.3f}. MoGe-2 is not a survey; on another scene it disagreed with a tape reference by 20 %.")
        record("Capture · scale check: MoGe-2 vs the registration scale", body, "Reference", target_of(a.scale_check_target) if a.scale_check_target else building_target,
               tags=["scale-check"], details={"moge_m_per_unit": round(moge["global_meters_per_unit"], 4), "alignment_m_per_unit": round(s_al, 4), "ratio": round(ratio, 4), "n_views": moge["n_views_scaled"]},
               attachments=[a.scaffold / "contact.png"])
    # 5. wall planes vs model wall lines (offset + orientation), and captured projections/recesses between parallel walls
    if lines:
        body = "Registered capture wall planes against the model's exterior wall lines (positive = capture stucco face lies outside the model framing line):\n" + "\n".join(
            f"- {x['patch']} ↔ {x['wall']} ({x['level']}): offset {x['offset_m']:+.2f} m over {x['overlap_m']:.1f} m of overlap, {x['angle_deg']:.1f}° off parallel" for x in lines)
        body += f"\n{reg_note}. Differences between two lines on the same face give the recess depth as captured vs as modelled."
        record("Capture · wall plane offsets vs model wall lines", body, "Reference", {"kind": "level", "id": lines[0]["level"], "levelId": lines[0]["level"]},
               tags=["plane", "wall-line"], details={f"offset_{i}": x["offset_m"] for i, x in enumerate(lines[:12])} | {f"angle_{i}": x["angle_deg"] for i, x in enumerate(lines[:12])}, attachments=[a.scaffold / "contact.png"])
    for pr in mm.parallel_offsets(walls, lines, mwalls, dimensions=B.get("dimensions", [])):
        dim = next((d for d in B.get("dimensions", []) if d["id"] == pr.get("dimension_id")), None)
        target = {"kind": "wall", "id": pr["wall_b"], "levelId": next(x["level"] for x in lines if x["wall"] == pr["wall_b"])}
        rows = [mrow(target, "projection", abs(pr["capture_m"]), 0.15, f"plane {pr['patch_a']} ({pr['wall_a']}) to plane {pr['patch_b']} ({pr['wall_b']}) along the first wall's normal", "stucco face to stucco face", f"parallel wall planes in {run_id}")]
        body = (f"Captured distance between the parallel wall planes {pr['patch_a']} (on {pr['wall_a']}) and {pr['patch_b']} (on {pr['wall_b']}): {abs(pr['capture_m']):.2f} m. "
                f"The model's lines are {abs(pr['model_m']):.3f} m apart" + (f" (dimension '{dim['id']}' = {dim['expected']} m, {dim.get('confidence')})" if dim else "") + f". Δ {pr['delta_m']:+.2f} m. {reg_note}.")
        record(f"Capture · {pr['wall_b']} sits {abs(pr['capture_m']):.2f} m from {pr['wall_a']} (model {abs(pr['model_m']):.2f} m)", body, "Measurement", target, rows, tags=["dimension", "projection"],
               details={"capture_m": pr["capture_m"], "model_m": pr["model_m"], "dimension_id": dim["id"] if dim else None}, attachments=[a.scaffold / "contact.png"])
    summary = [{"title": r["payload"]["title"], "type": r["payload"]["feedback_type"], "target": r["payload"]["target"], "measurements": len(r["payload"]["measurements"]), "attachments": len(r["attachments"])} for r in records]
    (a.output / "evidence.json").write_text(json.dumps({"schema": "dev.splatlab.building-evidence/v1", "created": TODAY, "api": a.api, "building": str(a.building), "provenance": provenance, "records": records,
                                                        "matches": [{"capture": capture[ci]["wall"] + "/" + capture[ci]["kind"], "views": capture[ci]["n_views"], "model": (model[mi]["id"] if mi not in (None, -1) else ("ghost-duplicate" if mi == -1 else None)), "iou": round(iou, 3)} for ci, mi, iou, _ in pairs]}, indent=1))
    print(f"[evidence] {len(capture)} capture openings (>= {a.min_views} views), {sum(1 for p in pairs if p[1] not in (None, -1))} matched to model openings, {sum(1 for p in pairs if p[1] == -1)} ghost duplicates dropped; {len(records)} records:")
    for r in summary: print("   ", json.dumps(r, ensure_ascii=False))
    if not a.apply:
        print(f"[evidence] dry-run: payloads in {a.output / 'evidence.json'}; add --apply to post to {a.api}"); return 0
    posted, failures = [], []
    existing = {i["title"]: i for i in get(a.api, "/api/feedback")["items"]} if a.resume else {}
    for r in records:
        title = r["payload"]["title"]; att = []
        try:
            if title in existing:
                fid = existing[title]["id"]; have = {x["filename"] for x in existing[title].get("attachments", [])}; created = existing[title]
            else:
                created = post(a.api, "/api/feedback", r["payload"]); fid = created["id"]; have = set()
            for f in r["attachments"]:
                f = Path(f); name = f"{run_id}-{f.name}"
                if name in have:
                    continue
                data = f.read_bytes(); res = post(a.api, f"/api/feedback/{fid}/attachments?filename={name}", raw=data, content_type="image/png")
                att.append({"filename": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "id": res.get("id") if isinstance(res, dict) else None})
            posted.append({"id": fid, "title": title, "status": created.get("status"), "attachments": att, "resumed": title in existing}); print(f"   posted {fid} {title[:80]} (+{len(att)} attachments{', resumed' if title in existing else ''})")
        except RuntimeError as e:
            failures.append({"title": title, "error": str(e)}); print(f"   FAILED {title[:80]}: {e}")
    (a.output / "evidence-receipt.json").write_text(json.dumps({"api": a.api, "posted": posted, "failures": failures, "at": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=1))
    print(f"[evidence] {len(posted)} records posted, {len(failures)} failed; receipt {a.output / 'evidence-receipt.json'}; rollback = set each id to Won't Fix / Archived in the review UI (no delete route)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
