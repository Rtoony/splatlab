#!/usr/bin/env python3
"""Capture-derived registration for a scaffold run that has no manual alignment (CPU, splatops env).

Rotation from the fitted planes (wall normal -> the model wall's outward normal, pavement up -> +Z),
scale from the MoGe-2 per-view track scale, translation by placing the capture's door rectangle
(bottom-centre, on the wall plane) onto the model door's bottom-centre on the wall line. Writes a
registration file in the same shape as the Condo Lab alignment receipt ({cases:[{name, fit:{matrix,
scale, convention, residuals}}]}) so architecture-scaffold.py / condo-evidence.py can take it as
--alignment. The result is provisional: sizes are MoGe-scaled (independent of the model), positions
along the wall are relative to the door, the stucco face is placed ON the framing line.

  condo-register-by-openings.py --scaffold DIR --moge DIR --building building.json --wall level-1-exterior-5 \\
      --patch wall-1 --door office-garden-door --output DIR/registration.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from architecture import scaffold_core as sc   # noqa: E402


def outward_normal(level: dict, wall: dict) -> np.ndarray:
    a, b = np.array(wall["a"], float), np.array(wall["b"], float); d = (b - a) / np.linalg.norm(b - a)
    n = np.array([d[1], -d[0]]); centroid = np.mean(np.array(level["outline"], float), axis=0)
    mid = (a + b) / 2
    return n if np.linalg.norm(mid + n - centroid) > np.linalg.norm(mid - n - centroid) else -n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaffold", type=Path, required=True); ap.add_argument("--moge", type=Path, required=True)
    ap.add_argument("--building", type=Path, required=True); ap.add_argument("--wall", required=True); ap.add_argument("--patch", required=True)
    ap.add_argument("--door", required=True, help="model opening id on --wall that the capture's best door rectangle is placed on")
    ap.add_argument("--anchor", choices=["door", "window-pair"], default="door", help="translation anchor: the door, or the midpoint of the two best windows (checks the door instead)")
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--name", default="capture-openings")
    a = ap.parse_args()
    S = json.loads((a.scaffold / "scaffold.json").read_text()); B = json.loads(a.building.read_text())
    mpu = json.loads((a.moge / "receipt.json").read_text())["global_meters_per_unit"]
    patch = next(p for p in S["patches"] if p["name"] == a.patch)
    level, wall = next((L, w) for L in B["levels"] for w in L["walls"] if w["id"] == a.wall)
    door = next(o for o in wall["openings"] if o["id"] == a.door)
    doors = [r for r in patch["openings"].get("door", []) if r.get("n_views", 0) >= 2]
    if not doors:
        raise SystemExit(f"{a.patch} has no multi-view door rectangle")
    cd = doors[0]
    # frames: capture columns [right, normal, up] -> canonical [Z x n_model, n_model, Z]
    n = np.array(patch["normal"], float); up = np.array(S["frame"]["up"], float); up = up - (up @ n) * n; up /= np.linalg.norm(up)
    r = np.cross(up, n); r /= np.linalg.norm(r)
    nm2 = outward_normal(level, wall); nm = np.array([nm2[0], nm2[1], 0.0]); Z = np.array([0, 0, 1.0]); rm = np.cross(Z, nm)
    A = np.stack([r, n, up], axis=1); Bm = np.stack([rm, nm, Z], axis=1); R = Bm @ A.T
    if abs(np.linalg.det(R) - 1) > 1e-6:
        raise SystemExit("frame is not a proper rotation")
    s = float(mpu)
    basis = np.array(patch["basis"], float); c = np.array(patch["centre"], float)
    wa, wb = np.array(wall["a"], float), np.array(wall["b"], float); d = (wb - wa) / np.linalg.norm(wb - wa)
    def uv_of(rect): return (np.array(rect["corners_units"], float) - c) @ basis.T
    if a.anchor == "door":
        # capture door bottom-centre on the wall plane -> model door bottom-centre on the wall line
        uv = uv_of(cd); u_mid = float(uv[:, 0].mean()); v_bot = float(uv[:, 1].min())
        target_xy = wa + d * (door["offset"] + door["width"] / 2); z_ref = level["elevation"] + door.get("sill", 0.0)
        anchor_note = f"door {a.door} bottom-centre"
    else:
        wins = sorted([x for x in patch["openings"].get("window", []) if x.get("n_views", 0) >= 3], key=lambda x: -x["n_views"])[:2]
        mwins = [o for o in wall["openings"] if o["kind"] == "window"]
        if len(wins) < 2 or len(mwins) < 2:
            raise SystemExit("window-pair anchor needs two multi-view capture windows and two model windows on the wall")
        uvs = [uv_of(w) for w in wins]; u_mid = float(np.mean([u[:, 0].mean() for u in uvs])); v_bot = float(np.mean([u[:, 1].min() for u in uvs]))
        target_xy = wa + d * float(np.mean([o["offset"] + o["width"] / 2 for o in mwins])); z_ref = level["elevation"] + float(np.mean([o.get("sill", 0.0) for o in mwins]))
        anchor_note = f"midpoint of the two best capture windows ({wins[0]['n_views']}/{wins[1]['n_views']} views) on the midpoint of {mwins[0]['id']}/{mwins[1]['id']}, sill = mean model sill"
    p_ref = c + u_mid * basis[0] + v_bot * basis[1]
    target = np.array([target_xy[0], target_xy[1], z_ref])
    t = target - s * (R @ p_ref)
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
    # residuals: every multi-view rectangle on this patch vs the model openings on this wall (same kind), by along-wall centre
    def to_canon(pts): return (np.asarray(pts, float) @ (s * R).T) + t
    residuals = []
    for kind_c, kind_m in (("door", "door"), ("window", "window")):
        for rect in [x for x in patch["openings"].get(kind_c, []) if x.get("n_views", 0) >= 2]:
            cc = to_canon(rect["corners_units"]); cu = float(((cc[:, :2] - wa) @ d).mean()); cz = float(cc[:, 2].min())
            best = None
            for o in wall["openings"]:
                if o["kind"] != kind_m:
                    continue
                mu = o["offset"] + o["width"] / 2; dz = level["elevation"] + o.get("sill", 0.0)
                du = cu - mu
                if best is None or abs(du) < abs(best["along_wall_m"]):
                    best = {"id": o["id"], "role": ("fit" if (a.anchor == "door" and o["id"] == a.door) or (a.anchor == "window-pair" and o["kind"] == "window") else "check"), "along_wall_m": du, "sill_m": cz - dz, "capture_w_h_m": [rect["width"] * s, rect["height"] * s], "model_w_h_m": [o["width"], o["height"]], "n_views": rect["n_views"]}
            if best:
                residuals.append(best)
    out = {"schema": "dev.splatlab.capture-registration/v1", "scaffold": str(a.scaffold), "wall": a.wall, "patch": a.patch, "door": a.door,
           "cases": [{"name": a.name, "fit": {"matrix": M.reshape(-1).tolist(), "scale": s, "convention": "row-major source-to-canonical meters Z-up; column vectors",
                                             "scaleEvidence": f"MoGe-2 monocular metric depth scaled by each view's SfM tracks: {mpu:.4f} m per SfM unit (not from any modelled dimension)",
                                             "rotationEvidence": f"capture wall plane {a.patch} normal -> outward normal of model wall {a.wall}; pavement up -> +Z",
                                             "translationEvidence": f"anchor = {anchor_note}; stucco face placed ON the framing line",
                                             "residuals": residuals, "fitRmsMeters": None}}]}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=1))
    print(f"[register] {a.patch} -> {a.wall} via {a.door}: scale {s:.4f} m/u, wall dir {d.round(3).tolist()}, outward {nm2.round(3).tolist()}")
    for x in residuals: print("   ", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
