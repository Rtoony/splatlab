"""model_match: match capture-derived planes and openings to a building model.

The building model is the Condo Lab / reality-regenerator `building.json` shape
(levels with elevation + height, walls a→b in plan metres with openings given by
offset / width / sill / height, an outline polygon per level). Everything here is
numpy only and building-agnostic: the model path, target ids and datums are inputs.

Used by tools/building-evidence.py; tested in backend/tests/test_model_match.py.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

KIND_MAP = {"garage_door": "garage", "door": "door", "window": "window"}


def model_openings(building: dict, exterior_only: bool = True) -> list[dict[str, Any]]:
    """Every opening as a canonical rectangle: along-wall direction d (XY unit), in-plane
    normal, 4x3 corners, sill_z / head_z (level elevation + sill [+ height])."""
    out = []
    for level in building["levels"]:
        for wall in level["walls"]:
            if exterior_only and not wall.get("exterior"):
                continue
            a, b = np.array(wall["a"], dtype=float), np.array(wall["b"], dtype=float)
            d = b - a; length = float(np.linalg.norm(d)); d = d / max(length, 1e-9)
            n = np.array([d[1], -d[0]])
            for o in wall.get("openings", []):
                if o.get("presence", {}).get("state") == "not-present":
                    continue
                z0 = level["elevation"] + o.get("sill", 0.0); z1 = z0 + o["height"]
                p0 = a + d * o["offset"]; p1 = p0 + d * o["width"]
                out.append({"id": o["id"], "kind": o["kind"], "wall": wall["id"], "level": level["id"], "exterior": bool(wall.get("exterior")),
                            "offset": o["offset"], "width": o["width"], "height": o["height"], "sill": o.get("sill", 0.0), "sill_z": z0, "head_z": z1,
                            "confidence": o.get("confidence"), "corners": np.array([[p0[0], p0[1], z0], [p1[0], p1[1], z0], [p1[0], p1[1], z1], [p0[0], p0[1], z1]]),
                            "dir": d, "normal_xy": n, "plane_point": np.array([p0[0], p0[1], z0]), "wall_a": a, "wall_length": length})
    return out


def model_walls(building: dict, exterior_only: bool = True) -> list[dict[str, Any]]:
    out = []
    for level in building["levels"]:
        for wall in level["walls"]:
            if exterior_only and not wall.get("exterior"):
                continue
            if wall.get("presence", {}).get("state") == "not-present":
                continue
            a, b = np.array(wall["a"], dtype=float), np.array(wall["b"], dtype=float); d = b - a; length = float(np.linalg.norm(d))
            if length < 1e-6:
                continue
            d /= length
            out.append({"id": wall["id"], "level": level["id"], "a": a, "b": b, "dir": d, "normal_xy": np.array([d[1], -d[0]]), "length": length,
                        "z0": level["elevation"], "z1": level["elevation"] + level["height"]})
    return out


def rect_iou(a_lo, a_hi, b_lo, b_hi) -> float:
    lo = np.maximum(a_lo, b_lo); hi = np.minimum(a_hi, b_hi)
    inter = float(np.prod(np.clip(hi - lo, 0, None))); ua = float(np.prod(a_hi - a_lo)); ub = float(np.prod(b_hi - b_lo))
    return inter / max(ua + ub - inter, 1e-12)


def _along(corners_xy: np.ndarray, m: dict[str, Any]) -> np.ndarray:
    return (corners_xy - m["plane_point"][:2]) @ m["dir"]


def match_openings(capture: list[dict[str, Any]], model: list[dict[str, Any]], plane_tol: float = 1.0, angle_deg: float = 20.0,
                   min_iou: float = 0.2) -> list[tuple[int, int | None, float, float | None]]:
    """capture rects carry canonical `corners_m` (4x3). A capture rect matches the model
    opening on a parallel wall plane (within angle_deg, plane offset ≤ plane_tol) with the
    best rectangle IoU in (along-wall, z) ≥ min_iou. Each model opening keeps only its
    best capture rect; the others get mi = -1 (ghost duplicates)."""
    pairs: list[tuple[int, int | None, float, float | None]] = []
    cos_tol = math.cos(math.radians(angle_deg))
    for ci, c in enumerate(capture):
        cc = np.asarray(c["corners_m"], dtype=float); cdir = cc[1] - cc[0]; cdir[2] = 0; cdir /= max(np.linalg.norm(cdir), 1e-9)
        best = None
        for mi, m in enumerate(model):
            if abs(float(cdir[:2] @ m["dir"])) < cos_tol:
                continue
            off = abs(float((cc.mean(axis=0)[:2] - m["plane_point"][:2]) @ m["normal_xy"]))
            if off > plane_tol:
                continue
            t_c = _along(cc[:, :2], m); t_m = _along(m["corners"][:, :2], m)
            iou = rect_iou(np.array([t_c.min(), cc[:, 2].min()]), np.array([t_c.max(), cc[:, 2].max()]), np.array([t_m.min(), m["sill_z"]]), np.array([t_m.max(), m["head_z"]]))
            if iou >= min_iou and (best is None or iou > best[1]):
                best = (mi, iou, off)
        pairs.append((ci, *(best if best else (None, 0.0, None))))
    best_for_model: dict[int, tuple[int, float]] = {}
    for ci, mi, iou, off in pairs:
        if mi is not None and (mi not in best_for_model or iou > best_for_model[mi][1]):
            best_for_model[mi] = (ci, iou)
    return [(ci, mi, iou, off) if mi is None or best_for_model[mi][0] == ci else (ci, -1, iou, off) for ci, mi, iou, off in pairs]


def same_wall_shift(c: dict[str, Any], model: list[dict[str, Any]], angle_deg: float = 20.0, plane_tol: float = 1.0) -> tuple[dict[str, Any], float] | None:
    """For an unmatched capture rect: the nearest model opening of the same kind on the same
    wall plane and how far along the wall (a→b) the capture sees it from the modelled place."""
    cc = np.asarray(c["corners_m"], dtype=float); cdir = cc[1] - cc[0]; cdir[2] = 0; cdir /= max(np.linalg.norm(cdir), 1e-9)
    kind = KIND_MAP.get(c["kind"], c["kind"]); cos_tol = math.cos(math.radians(angle_deg)); found = []
    for m in model:
        if m["kind"] != kind or abs(float(cdir[:2] @ m["dir"])) < cos_tol:
            continue
        if abs(float((cc.mean(axis=0)[:2] - m["plane_point"][:2]) @ m["normal_xy"])) > plane_tol:
            continue
        t_c = float(((cc[:, :2] - m["wall_a"]) @ m["dir"]).mean()); t_m = m["offset"] + m["width"] / 2      # both from the wall's a
        found.append((m, t_c - t_m))
    return min(found, key=lambda x: abs(x[1])) if found else None


def wall_line_matches(walls: list[dict[str, Any]], model: list[dict[str, Any]], angle_deg: float = 10.0, min_overlap: float = 0.5,
                      max_offset: float = 1.0) -> list[dict[str, Any]]:
    """Registered capture wall patches (corners_m, normal_canonical) against model wall
    lines: parallel within angle_deg, overlapping along the line, within max_offset.
    offset_m > 0 = the capture face lies outside the model line (along the model normal
    that faces the capture)."""
    out = []
    cos_tol = math.cos(math.radians(angle_deg))
    for w in walls:
        if "corners_m" not in w or "normal_canonical" not in w:
            continue
        cm = np.asarray(w["corners_m"], dtype=float); nc = np.asarray(w["normal_canonical"], dtype=float)[:2]
        if np.linalg.norm(nc) < 0.9:
            continue
        nc = nc / np.linalg.norm(nc); candidates = []
        for mw in model:
            if abs(float(mw["normal_xy"] @ nc)) < cos_tol:
                continue
            t = (cm[:, :2] - mw["a"]) @ mw["dir"]; overlap = min(float(t.max()), mw["length"]) - max(float(t.min()), 0.0)
            if overlap < min_overlap:
                continue
            zc = cm[:, 2]
            if not (zc.max() > mw["z0"] and zc.min() < mw["z1"]):
                continue
            sign = 1.0 if float(mw["normal_xy"] @ nc) > 0 else -1.0
            off = float(((cm.mean(axis=0)[:2] - mw["a"]) @ mw["normal_xy"])) * sign
            if abs(off) <= max_offset:
                candidates.append({"patch": w["name"], "wall": mw["id"], "level": mw["level"], "overlap_m": round(overlap, 2), "offset_m": round(off, 3),
                                   "angle_deg": round(float(math.degrees(math.acos(min(1.0, abs(float(mw["normal_xy"] @ nc)))))), 1)})
        if candidates:                                                     # one model line per capture wall: the nearest
            best = min(candidates, key=lambda c: abs(c["offset_m"])); best["alternatives"] = len(candidates) - 1; out.append(best)
    return out


def parallel_offsets(walls: list[dict[str, Any]], lines: list[dict[str, Any]], model: list[dict[str, Any]], angle_deg: float = 8.0,
                     dimensions: list[dict[str, Any]] | None = None, same_level_only: bool = True) -> list[dict[str, Any]]:
    """Pairs of parallel capture walls that each matched a different model wall line:
    the captured plane-to-plane distance vs the modelled line-to-line distance (a bay
    projection, a recess), in metres, signed along the first wall's outward normal.
    One capture patch per model wall (the largest); pairs across levels only when a
    model `dimensions` entry documents that distance (a bay projection, a set-back)."""
    by_patch = {x["patch"]: x for x in lines}; mw = {m["id"]: m for m in model}; out = []
    best_patch: dict[str, dict[str, Any]] = {}
    for w in walls:
        if w["name"] in by_patch:
            key = by_patch[w["name"]]["wall"]
            if key not in best_patch or w.get("surfels", 0) > best_patch[key].get("surfels", 0):
                best_patch[key] = w
    ws = sorted(best_patch.values(), key=lambda w: w["name"])
    expected = [d["expected"] for d in (dimensions or []) if isinstance(d.get("expected"), (int, float))]
    for i, a in enumerate(ws):
        for b in ws[i + 1:]:
            la, lb = by_patch[a["name"]], by_patch[b["name"]]
            if la["wall"] == lb["wall"]:
                continue
            na, nb = np.asarray(a["normal_canonical"], dtype=float), np.asarray(b["normal_canonical"], dtype=float)
            if abs(float(na @ nb)) < math.cos(math.radians(angle_deg)):
                continue
            ca, cb = np.asarray(a["corners_m"], dtype=float).mean(axis=0), np.asarray(b["corners_m"], dtype=float).mean(axis=0)
            capture_d = float((cb - ca) @ na)
            ma, mb = mw[la["wall"]], mw[lb["wall"]]
            model_d = float(((mb["a"] - ma["a"]) @ ma["normal_xy"])) * (1.0 if float(ma["normal_xy"] @ na[:2]) > 0 else -1.0)
            documented = next((d for d in (dimensions or []) if isinstance(d.get("expected"), (int, float)) and abs(d["expected"] - abs(model_d)) < 0.02), None)
            if same_level_only and la["level"] != lb["level"] and documented is None:
                continue
            out.append({"patch_a": a["name"], "patch_b": b["name"], "wall_a": la["wall"], "wall_b": lb["wall"], "capture_m": round(capture_d, 3), "model_m": round(model_d, 3),
                        "delta_m": round(capture_d - model_d, 3), "dimension_id": documented["id"] if documented else None})
    return out


def inside_footprint(point_xy, building: dict, margin: float = 0.5) -> bool:
    """Is a plan point inside any level outline's bounding box grown by `margin`?"""
    x, y = float(point_xy[0]), float(point_xy[1])
    for level in building["levels"]:
        pts = np.asarray(level["outline"], dtype=float)
        if pts[:, 0].min() - margin <= x <= pts[:, 0].max() + margin and pts[:, 1].min() - margin <= y <= pts[:, 1].max() + margin:
            return True
    return False
