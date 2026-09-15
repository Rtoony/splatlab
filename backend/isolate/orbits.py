"""CPU geometry for isolate-by-reference (numpy only, unit-tested).

Frame conventions: nerfstudio's normalised world (the exported PLY's frame,
Z up after `orientation_method='up'`), cameras as OpenGL camera-to-world 3x4
(x right, y up, z backward), image v growing downward — identical to
backend/scale_estimate.py and backend/health/render_views.py.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


# ── seed lift ────────────────────────────────────────────────────────────────

def backproject(mask: np.ndarray, depth: np.ndarray, c2w: np.ndarray, fx: float, fy: float,
                cx: float, cy: float, max_points: int = 20000, seed: int = 0) -> np.ndarray:
    """World points for the mask pixels at the rendered expected depth."""
    mask = np.asarray(mask, dtype=bool); depth = np.asarray(depth, dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError(f"mask {mask.shape} vs depth {depth.shape}")
    v, u = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if len(u) == 0:
        return np.zeros((0, 3))
    if len(u) > max_points:
        pick = np.random.default_rng(seed).choice(len(u), size=max_points, replace=False)
        u, v = u[pick], v[pick]
    z = depth[v, u]
    x = (u + 0.5 - cx) / fx * z
    y = -(v + 0.5 - cy) / fy * z          # image v down -> camera y up
    cam = np.stack([x, y, -z], axis=1)     # OpenGL: forward is -z
    c2w = np.asarray(c2w, dtype=np.float64)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def voxel_select(means: np.ndarray, points: np.ndarray, voxel: float, dilate: int = 1) -> np.ndarray:
    """Indices of gaussians whose centre falls in a voxel (± dilate) occupied by a
    lifted point. O(N) via a hashed voxel set — no KD-tree dependency."""
    means = np.asarray(means, dtype=np.float64); points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or voxel <= 0:
        return np.zeros(0, dtype=np.int64)
    origin = points.min(axis=0)
    occupied = set(map(tuple, np.floor((points - origin) / voxel).astype(np.int64)))
    if dilate > 0:
        rng = range(-dilate, dilate + 1)
        occupied = {(a + i, b + j, c + k) for (a, b, c) in list(occupied) for i in rng for j in rng for k in rng}
    cells = np.floor((means - origin) / voxel).astype(np.int64)
    hit = np.fromiter((tuple(c) in occupied for c in cells), dtype=bool, count=len(cells))
    return np.nonzero(hit)[0]


def project(points: np.ndarray, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float,
            w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, depth) for points; depth <= 0 or outside the image -> NaN u/v."""
    c2w = np.asarray(c2w, dtype=np.float64)
    cam = (np.asarray(points, dtype=np.float64) - c2w[:3, 3]) @ c2w[:3, :3]
    depth = -cam[:, 2]
    ok = depth > 1e-6
    u = np.where(ok, fx * cam[:, 0] / np.where(ok, depth, 1.0) + cx, np.nan)
    v = np.where(ok, -fy * cam[:, 1] / np.where(ok, depth, 1.0) + cy, np.nan)
    inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return np.where(inside, u, np.nan), np.where(inside, v, np.nan), depth


def seed_stats(means: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(means)[idx]
    if len(pts) == 0:
        return {"n": 0, "centroid": None, "radius": None}
    c = pts.mean(axis=0); dist = np.linalg.norm(pts - c, axis=1)
    return {"n": int(len(pts)), "centroid": c.tolist(), "radius": float(np.percentile(dist, 95)),
            "radius_median": float(np.median(dist))}


# ── virtual orbit ────────────────────────────────────────────────────────────

def look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray = UP) -> np.ndarray:
    """OpenGL camera-to-world 3x4 at `position` looking at `target`."""
    position = np.asarray(position, dtype=np.float64); target = np.asarray(target, dtype=np.float64)
    fwd = target - position; fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(fwd, up / np.linalg.norm(up)))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    c2w = np.zeros((3, 4)); c2w[:, 0] = right; c2w[:, 1] = true_up; c2w[:, 2] = -fwd; c2w[:, 3] = position
    return c2w


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(angle) + np.cross(axis, v) * math.sin(angle)
            + axis * np.dot(axis, v) * (1.0 - math.cos(angle)))


def orbit_cameras(ref_c2w: np.ndarray, centroid: np.ndarray, azimuths_deg: list[float],
                  elevations_deg: list[float], up: np.ndarray = UP,
                  distance_scales: list[float] | None = None) -> list[dict[str, Any]]:
    """Candidate cameras on spheres around `centroid`: rotate the reference offset
    about `up` by each azimuth, tilt by each elevation about the local horizontal
    axis, and optionally dolly by each distance scale (a closer ring helps objects
    against walls, where the wide reference-distance views sit inside geometry).
    Index 0 is the reference itself."""
    ref_c2w = np.asarray(ref_c2w, dtype=np.float64); centroid = np.asarray(centroid, dtype=np.float64)
    offset = ref_c2w[:3, 3] - centroid
    cams = [{"tag": "ref", "azimuth_deg": 0.0, "elevation_deg": 0.0, "distance_scale": 1.0, "c2w": ref_c2w[:3, :4].tolist()}]
    for ds in (distance_scales or [1.0]):
        for el in elevations_deg:
            for az in azimuths_deg:
                if az == 0.0 and el == 0.0 and ds == 1.0:
                    continue
                v = _rotate_about(offset, up, math.radians(az))
                horiz = np.cross(up, v)
                if np.linalg.norm(horiz) > 1e-9:
                    v = _rotate_about(v, horiz, math.radians(el))
                v = v * ds
                cams.append({"tag": f"az{az:+.0f}_el{el:+.0f}_d{ds:.2f}", "azimuth_deg": float(az), "elevation_deg": float(el),
                             "distance_scale": float(ds), "c2w": look_at(centroid + v, centroid, up).tolist()})
    return cams


def select_views(cams: list[dict[str, Any]], min_visible: float, min_views: int, max_views: int,
                 floor_visible: float = 0.1) -> list[dict[str, Any]]:
    """Reference first, then every candidate at/above `min_visible` by visibility;
    top up to `min_views` from the best remaining above `floor_visible`; cap at `max_views`."""
    ref = [c for c in cams if c.get("tag") == "ref"]
    rest = sorted((c for c in cams if c.get("tag") != "ref"), key=lambda c: -float(c.get("visible_fraction", 0.0)))
    good = [c for c in rest if float(c.get("visible_fraction", 0.0)) >= min_visible]
    chosen = good[: max(0, max_views - len(ref))]
    if len(ref) + len(chosen) < min_views:
        extra = [c for c in rest if c not in chosen and float(c.get("visible_fraction", 0.0)) >= floor_visible]
        chosen += extra[: min_views - len(ref) - len(chosen)]
    return ref + chosen


# ── visibility ───────────────────────────────────────────────────────────────

def visible_fraction(seed_alpha: np.ndarray, seed_depth: np.ndarray, full_depth: np.ndarray,
                     tol: float, alpha_min: float = 0.5) -> float:
    """Share of seed-covered pixels where the seed is the front surface
    (its depth is within `tol` of the full scene's expected depth)."""
    cover = np.asarray(seed_alpha) > alpha_min
    n = int(cover.sum())
    if n == 0:
        return 0.0
    front = np.abs(np.asarray(seed_depth)[cover] - np.asarray(full_depth)[cover]) <= tol
    return float(front.mean())


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    v, u = np.nonzero(np.asarray(mask, dtype=bool))
    if len(u) == 0:
        return None
    return [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


# ── reference grounding pick (SAM3 npz layout from backend/mesh/scene_sam3_masks.py) ─

MAX_MASK_FRAC = 0.6     # a "detection" covering most of the frame is background, not an object
MIN_MASK_FRAC = 0.002


def pick_reference(views: list[dict[str, Any]], load_npz, slug: str) -> dict[str, Any] | None:
    """Highest-scoring SAM3 instance across the staged views whose mask covers a
    plausible fraction of the frame. `load_npz(cam) -> {masks, scores} | None`."""
    best = None
    for row in views:
        d = load_npz(int(row["cam"]))
        if d is None:
            continue
        masks, scores = np.asarray(d["masks"]), np.asarray(d["scores"], dtype=np.float64)
        for k in range(masks.shape[0]):
            frac = float(masks[k].mean())
            if not (MIN_MASK_FRAC <= frac <= MAX_MASK_FRAC):
                continue
            q = reference_quality(float(scores[k]), masks[k])
            if best is None or q > best["quality"]:
                best = {"row": row, "mask": masks[k].astype(bool), "score": float(scores[k]), "quality": q, "instance": int(k), "frac": frac}
    return best


# ── prompt points for the tracker (numpy-only morphology) ────────────────────

def _shift_or(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r = np.roll(mask, (dy, dx), axis=(0, 1))
        if dy == 1: r[0, :] = False
        if dy == -1: r[-1, :] = False
        if dx == 1: r[:, 0] = False
        if dx == -1: r[:, -1] = False
        out |= r
    return out


def dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    for _ in range(max(0, iterations)):
        m = _shift_or(m)
    return m


def erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    return ~dilate(~np.asarray(mask, dtype=bool), iterations)


def erosion_depth(mask: np.ndarray, max_iter: int = 64) -> np.ndarray:
    """Per-pixel number of 4-neighbour erosions survived (a cheap distance transform)."""
    m = np.asarray(mask, dtype=bool); depth = np.zeros(m.shape, dtype=np.int32); cur = m
    for _ in range(max_iter):
        if not cur.any():
            break
        depth[cur] += 1
        cur = erode(cur, 1)
    return depth


def prompt_points(mask: np.ndarray, n_pos: int = 6, n_neg: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Positives: the deepest interior pixels, spread apart; negatives: pixels in a
    ring just outside the mask. Returns (points [K,2] as (x, y), labels [K])."""
    m = np.asarray(mask, dtype=bool); h, w = m.shape
    depth = erosion_depth(m).astype(np.float64)
    pos = []; d = depth.copy(); radius = max(6.0, float(depth.max()) * 0.8)
    yy, xx = np.ogrid[:h, :w]
    for _ in range(n_pos):
        if d.max() <= 0:
            break
        y, x = np.unravel_index(int(d.argmax()), d.shape); pos.append((int(x), int(y)))
        d[(yy - y) ** 2 + (xx - x) ** 2 < radius ** 2] = 0
    ring = dilate(m, 25) & ~dilate(m, 10)
    ry, rx = np.nonzero(ring); neg = []
    if len(rx):
        for i in np.linspace(0, len(rx) - 1, n_neg).astype(int):
            neg.append((int(rx[i]), int(ry[i])))
    pts = np.array(pos + neg, dtype=np.float32).reshape(-1, 2)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    return pts, labels


# ── frame ordering for the tracker + reference quality ───────────────────────

def path_order(cams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedy nearest-camera path starting at the reference: SAM3's tracker
    follows an object frame to frame, so consecutive orbit frames must be
    neighbours in space, not sorted by visibility."""
    if not cams:
        return []
    ref = next((c for c in cams if c.get("tag") == "ref"), cams[0])
    rest = [c for c in cams if c is not ref]
    order = [ref]; cur = np.asarray(ref["c2w"])[:, 3]
    while rest:
        d = [np.linalg.norm(np.asarray(c["c2w"])[:, 3] - cur) for c in rest]
        k = int(np.argmin(d)); cur = np.asarray(rest[k]["c2w"])[:, 3]; order.append(rest.pop(k))
    return order


def border_fraction(mask: np.ndarray, margin: int = 2) -> float:
    """Share of mask pixels within `margin` px of the image border — a mask
    that touches the edge is a partial object, a poor reference."""
    m = np.asarray(mask, dtype=bool); n = int(m.sum())
    if n == 0:
        return 0.0
    edge = np.zeros_like(m); edge[:margin, :] = True; edge[-margin:, :] = True; edge[:, :margin] = True; edge[:, -margin:] = True
    return float((m & edge).sum() / n)


def touches_border(mask: np.ndarray, margin: int = 2) -> bool:
    """True when the mask's bounding box reaches the image edge — the object is cut off."""
    bb = mask_bbox(mask)
    if bb is None:
        return False
    h, w = np.asarray(mask).shape
    return bb[0] <= margin or bb[1] <= margin or bb[2] >= w - margin or bb[3] >= h - margin


def reference_quality(score: float, mask: np.ndarray) -> float:
    """Detector score discounted for partial objects: x0.3 when the mask's box touches
    the image edge, further reduced by the share of mask pixels on the edge."""
    q = float(score) * (1.0 - min(1.0, 5.0 * border_fraction(mask)))
    return q * 0.3 if touches_border(mask) else q


# ── pulled-back re-grounding cameras ─────────────────────────────────────────

def pullback_cameras(ref_c2w: np.ndarray, centroid: np.ndarray, radius: float, fov_y_deg: float,
                     aspect: float, margin: float = 1.6, azimuths_deg: list[float] | None = None,
                     elevations_deg: list[float] | None = None, extra_scales: list[float] | None = None,
                     up: np.ndarray = UP) -> list[dict[str, Any]]:
    """Candidate cameras that frame the WHOLE seed: along the reference direction
    (and a few azimuths/elevations), at the distance where a sphere of `radius`
    fills 1/margin of the narrower field of view, plus optional further scales."""
    ref_c2w = np.asarray(ref_c2w, dtype=np.float64); centroid = np.asarray(centroid, dtype=np.float64)
    offset = ref_c2w[:3, 3] - centroid
    d_ref = float(np.linalg.norm(offset))
    half = math.radians(fov_y_deg) / 2.0
    half_narrow = half if aspect >= 1.0 else math.atan(math.tan(half) * aspect)
    d_fit = margin * float(radius) / max(math.tan(half_narrow), 1e-6)
    d = max(d_ref, d_fit)
    cams = []
    for sc in [1.0] + list(extra_scales or []):
        for el in (elevations_deg or [0.0]):
            for az in (azimuths_deg or [0.0]):
                v = _rotate_about(offset / d_ref, up, math.radians(az))
                horiz = np.cross(up, v)
                if np.linalg.norm(horiz) > 1e-9:
                    v = _rotate_about(v, horiz, math.radians(el))
                cams.append({"tag": f"pull_az{az:+.0f}_el{el:+.0f}_s{sc:.2f}", "distance": d * sc, "azimuth_deg": float(az),
                             "elevation_deg": float(el), "c2w": look_at(centroid + v * d * sc, centroid, up).tolist()})
    return cams


def fits_in_frame(bbox: list[int] | None, w: int, h: int, margin_frac: float = 0.04) -> bool:
    """True when the box sits inside the frame with a margin on every side."""
    if bbox is None:
        return False
    mx, my = margin_frac * w, margin_frac * h
    x0, y0, x1, y1 = bbox
    return x0 >= mx and y0 >= my and x1 <= w - mx and y1 <= h - my


def pick_reground(views: list[dict[str, Any]], load_npz, load_seed, min_containment: float = 0.5,
                  margin_frac: float = 0.02, min_score: float = 0.5, max_growth: float = 4.0) -> dict[str, Any] | None:
    """Second grounding on pulled-back renders: the SAM3 instance that CONTAINS the
    seed silhouette (same object) and GROWS it the most (its full extent), preferring
    one that fits inside the frame. quality = score x containment x sqrt(growth)
    x (1 | 0.6 if it touches the frame edge). `load_seed(cam) -> bool silhouette | None`.
    Measured 2026-09-15 (red bicycle): a plain score x containment pick with a hard
    0.3 border penalty chose a rear-triangle mask over the whole bike two rows later."""
    best = None
    for row in views:
        d = load_npz(int(row["cam"])); sil = load_seed(int(row["cam"]))
        if d is None or sil is None:
            continue
        sil = np.asarray(sil, dtype=bool); n_sil = int(sil.sum())
        if n_sil == 0:
            continue
        masks, scores = np.asarray(d["masks"]), np.asarray(d["scores"], dtype=np.float64)
        h, w = sil.shape
        for k in range(masks.shape[0]):
            m = masks[k].astype(bool); frac = float(m.mean())
            if not (MIN_MASK_FRAC <= frac <= MAX_MASK_FRAC) or float(scores[k]) < min_score:
                continue
            contain = float((m & sil).sum() / n_sil)
            if contain < min_containment:
                continue
            fits = fits_in_frame(mask_bbox(m), w, h, margin_frac)
            growth = float(m.sum()) / n_sil
            q = float(scores[k]) * contain * math.sqrt(min(growth, max_growth)) * (1.0 if fits else 0.6)
            if best is None or q > best["quality"]:
                best = {"row": row, "mask": m, "score": float(scores[k]), "quality": q, "instance": int(k), "frac": frac,
                        "containment": round(contain, 4), "fits": bool(fits), "growth": round(growth, 3)}
    return best


def mask_half_extent(bbox: list[int] | None, distance: float, fx: float, fy: float) -> float:
    """World half-size of what a mask's bounding box spans at `distance` from a pinhole
    camera (extent = distance x pixels / focal). Robust to depth outliers in the seed
    (see-through spokes / leaves lift wall gaussians that blow up the seed radius)."""
    if bbox is None:
        return 0.0
    x0, y0, x1, y1 = bbox
    return float(distance) * max((x1 - x0) / (2.0 * fx), (y1 - y0) / (2.0 * fy))


def depth_band_mask(mask: np.ndarray, depth: np.ndarray, rel: float = 0.3) -> np.ndarray:
    """Drop mask pixels whose depth is far from the mask's median depth. A SAM3 mask
    over a see-through object (spokes, leaves) contains pixels whose rendered depth
    is the WALL behind it; lifting those puts wall gaussians in the seed (bonsai:
    p95 seed radius 2.65 units at a 1.2-unit reference distance)."""
    m = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    if not m.any():
        return m
    med = float(np.median(depth[m]))
    return m & (np.abs(depth - med) <= rel * max(med, 1e-6))
