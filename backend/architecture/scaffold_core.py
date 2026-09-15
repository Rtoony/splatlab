"""scaffold_core: the pure math of the plane-scaffold architecture layer (R3.2).

Dense 2DGS surfels (position + normal + opacity) are labelled by projecting them
into semantically masked source views, planes are pulled out of the façade and
pavement sets by normal clustering + offset peaks + in-plane connectivity, the
pavement gives an up vector, façade planes give a Manhattan frame, and openings
(garage door, windows, unobserved gaps) are rectangles in each façade plane's
own coordinates. Everything is in the capture's arbitrary SfM frame unless a
similarity to metres is supplied; nothing here decides acceptance.

numpy + scipy.ndimage only (surface_patch_core tradition): testable in the app
venv, no torch / open3d at import time. The CLI (tools/architecture-scaffold.py)
wires these to the surfel package, the structure study and the alignment receipt.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

FACADE_CLASSES = ("building exterior wall", "garage door", "window", "door")
NUISANCE_CLASSES = ("sky", "vegetation", "vehicle")
PAVEMENT_CLASS = "pavement"


# ── surfels ──────────────────────────────────────────────────────────────────

def quat_to_normal(quats: np.ndarray) -> np.ndarray:
    """gsplat 2DGS: the surfel's local z axis (third column of R(q), q = wxyz) is its normal."""
    q = np.asarray(quats, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def to_original_frame(means: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Undo train-capture-surfels' solver normalisation: original = normalised x scale + center."""
    return np.asarray(means, dtype=np.float64) * float(scale) + np.asarray(center, dtype=np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


# ── projection (OpenGL camera-to-world, pinhole) ─────────────────────────────

def project(points: np.ndarray, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, w: int, h: int):
    """Pixels (u, v), depth (+ in front) and an inside-frame mask; same convention as capture_structure.project_points."""
    m = np.asarray(c2w, dtype=np.float64)
    local = (np.asarray(points, dtype=np.float64) - m[:3, 3]) @ m[:3, :3]
    depth = -local[:, 2]
    safe = np.where(np.abs(depth) > 1e-12, depth, 1.0)
    u = fx * local[:, 0] / safe + cx
    v = -fy * local[:, 1] / safe + cy
    ok = (depth > 0) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < w) & (v < h)
    return u, v, depth, ok


def label_votes(points: np.ndarray, views: list[dict[str, Any]], classes: tuple[str, ...],
                depth_tol: float = 0.06) -> np.ndarray:
    """Votes[n, k]: in how many views surfel n is VISIBLE (its depth within depth_tol
    x rendered depth of the view's own depth map, or in front of it) and lands in a
    pixel the view's SAM3 mask calls class k. Each view: {c2w, fx, fy, cx, cy, w, h,
    depth (HxW or None), masks: {class: bool HxW}}."""
    votes = np.zeros((len(points), len(classes)), dtype=np.int32)
    for view in views:
        u, v, d, ok = project(points, view["c2w"], view["fx"], view["fy"], view["cx"], view["cy"], view["w"], view["h"])
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        ui, vi = u[idx].astype(int), v[idx].astype(int)
        depth_map = view.get("depth")
        if depth_map is not None:
            ref = np.asarray(depth_map, dtype=np.float64)[vi, ui]
            visible = ~np.isfinite(ref) | (ref <= 0) | (d[idx] <= ref * (1.0 + depth_tol))
            idx, ui, vi = idx[visible], ui[visible], vi[visible]
        for k, name in enumerate(classes):
            mask = view["masks"].get(name)
            if mask is None:
                continue
            hit = np.asarray(mask, dtype=bool)[vi, ui]
            votes[idx[hit], k] += 1
    return votes


def classify(votes: np.ndarray, classes: tuple[str, ...], min_votes: int = 2) -> np.ndarray:
    """Per-surfel class index (or -1): the most-voted non-nuisance class, needing
    min_votes and more votes than every nuisance class combined."""
    nuis = np.zeros(len(votes), dtype=np.int64)
    keep = []
    for k, name in enumerate(classes):
        if name in NUISANCE_CLASSES:
            nuis += votes[:, k]
        else:
            keep.append(k)
    keep = np.asarray(keep)
    best = keep[np.argmax(votes[:, keep], axis=1)]
    top = votes[np.arange(len(votes)), best]
    out = np.where((top >= min_votes) & (top > nuis), best, -1)
    return out


# ── planes ───────────────────────────────────────────────────────────────────

def orient_towards(normals: np.ndarray, points: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Flip each normal so it points towards `anchor` (the cameras' centroid)."""
    n = np.asarray(normals, dtype=np.float64).copy()
    flip = np.einsum("ij,ij->i", n, np.asarray(anchor, dtype=np.float64) - points) < 0
    n[flip] *= -1
    return n


def fibonacci_sphere(count: int = 600) -> np.ndarray:
    i = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * i / count); theta = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def dominant_directions(normals: np.ndarray, weights: np.ndarray | None = None, angle_tol_deg: float = 12.0,
                        max_directions: int = 8, min_weight: float = 1.0) -> list[np.ndarray]:
    """Greedy: bin normals on a Fibonacci sphere, take the heaviest bin, average the
    normals within angle_tol of it, remove them, repeat."""
    n = np.asarray(normals, dtype=np.float64)
    w = np.ones(len(n)) if weights is None else np.asarray(weights, dtype=np.float64)
    grid = fibonacci_sphere()
    remaining = np.ones(len(n), dtype=bool)
    cos_tol = math.cos(math.radians(angle_tol_deg))
    out = []
    for _ in range(max_directions):
        if not remaining.any():
            break
        idx = np.flatnonzero(remaining)
        nearest = np.concatenate([np.argmax(n[idx[i:i + 65536]] @ grid.T, axis=1) for i in range(0, len(idx), 65536)])
        heavy = np.bincount(nearest, weights=w[idx], minlength=len(grid))
        b = int(np.argmax(heavy))
        if heavy[b] < min_weight:
            break
        members = idx[(n[idx] @ grid[b]) >= cos_tol]
        d = (n[members] * w[members, None]).sum(axis=0); d /= max(np.linalg.norm(d), 1e-12)
        members = idx[(n[idx] @ d) >= cos_tol]
        out.append(d)
        remaining[members] = False
    return out


def offset_peaks(offsets: np.ndarray, weights: np.ndarray, bin_size: float, min_weight: float, min_separation_bins: int = 3) -> list[float]:
    """Local maxima of the (lightly smoothed) weighted histogram of plane offsets."""
    if not len(offsets):
        return []
    lo, hi = float(offsets.min()), float(offsets.max())
    n_bins = max(3, int(math.ceil((hi - lo) / bin_size)) + 1)
    hist, edges = np.histogram(offsets, bins=n_bins, range=(lo, lo + n_bins * bin_size), weights=weights)
    smooth = np.convolve(hist, [0.25, 0.5, 0.25], mode="same")
    order = np.argsort(-smooth)
    peaks: list[int] = []
    for b in order:
        if smooth[b] < min_weight:
            break
        if all(abs(b - p) >= min_separation_bins for p in peaks):
            peaks.append(int(b))
    return [float(edges[p] + bin_size / 2) for p in peaks]


def fit_plane(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Weighted SVD plane: centre, normal, in-plane basis (right, up) and RMS distance."""
    p = np.asarray(points, dtype=np.float64)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    c = (p * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
    rel = (p - c) * np.sqrt(w)[:, None]
    _, _, axes = np.linalg.svd(rel, full_matrices=False)
    normal = axes[2]; right = axes[0]; up = np.cross(normal, right)
    rms = float(np.sqrt(np.mean(((p - c) @ normal) ** 2)))
    return c, normal, np.stack([right, up]), rms


def plane_components(uv: np.ndarray, cell: float, min_cells: int) -> list[np.ndarray]:
    """8-connected components of the occupied (u, v) cells; returns member index arrays, largest first."""
    ij = np.floor(uv / cell).astype(np.int64)
    ij -= ij.min(axis=0)
    grid = np.zeros(ij.max(axis=0) + 1, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, count = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return []
    member = labels[ij[:, 0], ij[:, 1]]
    sizes = ndimage.sum(grid, labels, index=np.arange(1, count + 1))
    out = []
    for lab in np.argsort(-sizes) + 1:
        if sizes[lab - 1] < min_cells:
            break
        out.append(np.flatnonzero(member == lab))
    return out


def extract_planes(points: np.ndarray, normals: np.ndarray, weights: np.ndarray, tol: float, angle_tol_deg: float = 12.0,
                   min_inliers: int = 300, min_cells: int = 40, max_planes: int = 8, cell_mult: float = 2.0) -> list[dict[str, Any]]:
    """Planar patches: for each dominant normal direction, each offset peak, refine
    by SVD twice, then split the inliers into in-plane connected components.
    Returns patches (largest first) with centre/normal/basis/uv bounds/inlier ids/rms."""
    p = np.asarray(points, dtype=np.float64); n = np.asarray(normals, dtype=np.float64); w = np.asarray(weights, dtype=np.float64)
    cos_tol = math.cos(math.radians(angle_tol_deg))
    free = np.ones(len(p), dtype=bool)
    patches: list[dict[str, Any]] = []
    for d in dominant_directions(n, w, angle_tol_deg, max_directions=max_planes, min_weight=min_inliers * 0.5):
        cand = np.flatnonzero(free & ((n @ d) >= cos_tol))
        if len(cand) < min_inliers:
            continue
        for off in offset_peaks(p[cand] @ d, w[cand], bin_size=tol, min_weight=min_inliers * 0.5):
            normal, centre = d.copy(), None
            inl = cand[np.abs(p[cand] @ d - off) <= tol]
            for _ in range(2):
                if len(inl) < min_inliers:
                    break
                centre, normal, basis, rms = fit_plane(p[inl], w[inl])
                if normal @ d < 0:
                    normal = -normal
                inl = np.flatnonzero(free & (np.abs((p - centre) @ normal) <= tol) & ((n @ normal) >= cos_tol))
            if centre is None or len(inl) < min_inliers:
                continue
            centre, normal, basis, rms = fit_plane(p[inl], w[inl])
            if normal @ d < 0:
                normal = -normal; basis = np.stack([basis[0], -basis[1]])
            uv = (p[inl] - centre) @ basis.T
            for comp in plane_components(uv, cell=cell_mult * tol, min_cells=min_cells):
                ids = inl[comp]
                if len(ids) < min_inliers:
                    continue
                c2, n2, b2, rms2 = fit_plane(p[ids], w[ids])
                if n2 @ normal < 0:
                    n2 = -n2; b2 = np.stack([b2[0], -b2[1]])
                uv2 = (p[ids] - c2) @ b2.T
                lo, hi = uv2.min(axis=0), uv2.max(axis=0)
                patches.append({"centre": c2, "normal": n2, "basis": b2, "lower_uv": lo, "upper_uv": hi, "ids": ids,
                                "rms": rms2, "weight": float(w[ids].sum()), "extent": (hi - lo).tolist()})
                free[ids] = False
                if len(patches) >= max_planes:
                    return sorted(patches, key=lambda q: -len(q["ids"]))
    return sorted(patches, key=lambda q: -len(q["ids"]))


# ── frame ────────────────────────────────────────────────────────────────────

def up_vector(pavement: dict[str, Any], camera_centres: np.ndarray) -> np.ndarray:
    """Pavement normal, signed so the cameras sit above the pavement."""
    n = np.asarray(pavement["normal"], dtype=np.float64)
    if np.mean((np.asarray(camera_centres) - pavement["centre"]) @ n) < 0:
        n = -n
    return n / np.linalg.norm(n)


def manhattan_frame(up: np.ndarray, facade_normal: np.ndarray) -> np.ndarray:
    """Rows: along-façade (right), out of the façade (towards the street/camera), up."""
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    out = np.asarray(facade_normal, dtype=np.float64); out = out - (out @ up) * up; out /= max(np.linalg.norm(out), 1e-12)
    right = np.cross(up, out); right /= max(np.linalg.norm(right), 1e-12)
    return np.stack([right, out, up])


def tilt_deg(normal: np.ndarray, up: np.ndarray) -> float:
    """Angle between a façade normal and the horizontal plane (0 = perfectly vertical wall)."""
    return float(math.degrees(math.asin(min(1.0, abs(float(np.asarray(normal) @ np.asarray(up)))))))


# ── openings ─────────────────────────────────────────────────────────────────

def opening_rectangles(patch: dict[str, Any], points: np.ndarray, labels: np.ndarray, class_index: int, cell: float,
                       min_cells: int = 6) -> list[dict[str, Any]]:
    """Rectangles (plane-local u/v bounds) of connected cells whose surfels carry
    `class_index` inside this patch's footprint, largest first."""
    uv = (np.asarray(points, dtype=np.float64)[patch["ids"]] - patch["centre"]) @ patch["basis"].T
    lab = np.asarray(labels)[patch["ids"]]
    sel = np.flatnonzero(lab == class_index)
    if len(sel) < min_cells:
        return []
    out = []
    for comp in plane_components(uv[sel], cell=cell, min_cells=min_cells):
        cuv = uv[sel][comp]
        lo, hi = cuv.min(axis=0), cuv.max(axis=0)
        out.append({"lower_uv": lo.tolist(), "upper_uv": hi.tolist(), "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]),
                    "surfels": int(len(comp))})
    return out


def gap_rectangles(patch: dict[str, Any], points: np.ndarray, cell: float, min_cells: int = 12, max_rects: int = 6,
                   closed_edges: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Unobserved holes inside the patch: empty cells enclosed by the filled footprint
    (binary_fill_holes), as rectangles — candidate openings the masks did not name.
    `closed_edges` (any of u0, u1, v0, v1) are treated as solid boundary, so a hole
    open to that edge still counts: a garage door reaches the ground line (v0/v1)."""
    uv = (np.asarray(points, dtype=np.float64)[patch["ids"]] - patch["centre"]) @ patch["basis"].T
    origin = uv.min(axis=0)
    ij = np.floor((uv - origin) / cell).astype(np.int64)
    grid = np.zeros(ij.max(axis=0) + 1, dtype=bool); grid[ij[:, 0], ij[:, 1]] = True
    closed = ndimage.binary_closing(grid, structure=np.ones((3, 3), dtype=bool)) | grid
    padded = np.pad(closed, 1)
    if "u0" in closed_edges: padded[0, :] = True
    if "u1" in closed_edges: padded[-1, :] = True
    if "v0" in closed_edges: padded[:, 0] = True
    if "v1" in closed_edges: padded[:, -1] = True
    holes = (ndimage.binary_fill_holes(padded) & ~padded)[1:-1, 1:-1]
    labels, count = ndimage.label(holes, structure=np.ones((3, 3), dtype=int))
    out = []
    if count:
        sizes = ndimage.sum(holes, labels, index=np.arange(1, count + 1))
        for lab in (np.argsort(-sizes) + 1)[:max_rects]:
            if sizes[lab - 1] < min_cells:
                break
            ii, jj = np.nonzero(labels == lab)
            lo = origin + np.array([ii.min(), jj.min()]) * cell; hi = origin + np.array([ii.max() + 1, jj.max() + 1]) * cell
            out.append({"lower_uv": lo.tolist(), "upper_uv": hi.tolist(), "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]),
                        "cells": int(sizes[lab - 1])})
    return out


def rect_corners_3d(patch: dict[str, Any], lower_uv, upper_uv) -> np.ndarray:
    """The four 3D corners (u0v0, u1v0, u1v1, u0v1) of a plane-local rectangle."""
    c, b = np.asarray(patch["centre"]), np.asarray(patch["basis"])
    (u0, v0), (u1, v1) = lower_uv, upper_uv
    return np.array([c + u * b[0] + v * b[1] for u, v in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])


# ── checks and metres ────────────────────────────────────────────────────────

def reserved_track_check(patch: dict[str, Any], check_points: np.ndarray, tol: float, band: float | None = None) -> dict[str, Any]:
    """capture_structure's independent-support test on a patch: reserved points whose
    in-plane projection falls in the footprint, and how many sit near the plane.
    With `band`, points further than that from the plane are counted as `far`
    (another surface seen through the footprint) and excluded from the medians."""
    pts = np.asarray(check_points, dtype=np.float64)
    empty = {"in_footprint": 0, "far": 0, "near": 0, "near_fraction": None, "median_distance": None, "p90_distance": None}
    if not len(pts):
        return empty
    rel = pts - patch["centre"]; uv = rel @ np.asarray(patch["basis"]).T
    inside = ((uv >= patch["lower_uv"]) & (uv <= patch["upper_uv"])).all(axis=1)
    dist = np.abs(rel[inside] @ np.asarray(patch["normal"]))
    far = dist > band if band is not None else np.zeros(len(dist), dtype=bool)
    dist = dist[~far]
    near = dist <= tol * 1.5
    return {"in_footprint": int(inside.sum()), "far": int(far.sum()), "near": int(near.sum()), "near_fraction": float(near.mean()) if len(near) else None,
            "median_distance": float(np.median(dist)) if len(dist) else None, "p90_distance": float(np.quantile(dist, 0.9)) if len(dist) else None}


def similarity_parts(matrix4: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Rotation, scale and translation of a row-major source->target similarity (column vectors)."""
    m = np.asarray(matrix4, dtype=np.float64).reshape(4, 4)
    scale = float(np.cbrt(abs(np.linalg.det(m[:3, :3]))))
    return m[:3, :3] / scale, scale, m[:3, 3]


def apply_similarity(points: np.ndarray, matrix4: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix4, dtype=np.float64).reshape(4, 4)
    p = np.asarray(points, dtype=np.float64)
    return p @ m[:3, :3].T + m[:3, 3]


def ground_edge(patch: dict[str, Any], up: np.ndarray) -> str:
    """Which plane-local edge (v0 or v1) is the ground side of a wall patch."""
    return "v0" if float(np.asarray(patch["basis"])[1] @ np.asarray(up)) > 0 else "v1"


# ── depth maps -> oriented points ────────────────────────────────────────────

def backproject_depth(depth: np.ndarray, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      step: int = 4, edge_rel: float = 0.04):
    """Rendered depth map -> world points with normals from the depth gradient.
    Returns (points, normals, pixel_rows, pixel_cols) for the sampled pixels that
    have finite depth and no depth discontinuity to their 4 neighbours (edges are
    dropped: their normals are meaningless). Normals face the camera."""
    d = np.asarray(depth, dtype=np.float64); h, w = d.shape
    m = np.asarray(c2w, dtype=np.float64); R, t = m[:3, :3], m[:3, 3]
    vv, uu = np.mgrid[0:h, 0:w]
    xc = (uu - cx) / fx * d; yc = -(vv - cy) / fy * d; zc = -d
    P = np.stack([xc, yc, zc], axis=-1) @ R.T + t                      # world points, [h, w, 3]
    ok = np.isfinite(d) & (d > 0)
    ok[1:-1, 1:-1] &= ok[:-2, 1:-1] & ok[2:, 1:-1] & ok[1:-1, :-2] & ok[1:-1, 2:]
    ok[[0, -1], :] = False; ok[:, [0, -1]] = False
    with np.errstate(invalid="ignore"):
        jump = np.zeros_like(ok)
        jump[1:-1, 1:-1] = (np.abs(d[:-2, 1:-1] - d[2:, 1:-1]) > edge_rel * 2 * d[1:-1, 1:-1]) | (np.abs(d[1:-1, :-2] - d[1:-1, 2:]) > edge_rel * 2 * d[1:-1, 1:-1])
    ok &= ~jump
    rows, cols = np.nonzero(ok[::step, ::step]); rows, cols = rows * step, cols * step
    du = P[rows, np.minimum(cols + 1, w - 1)] - P[rows, np.maximum(cols - 1, 0)]
    dv = P[np.minimum(rows + 1, h - 1), cols] - P[np.maximum(rows - 1, 0), cols]
    n = np.cross(du, dv); norm = np.linalg.norm(n, axis=1); good = norm > 1e-12
    n = n[good] / norm[good, None]; pts = P[rows[good], cols[good]]
    flip = np.einsum("ij,ij->i", n, t - pts) < 0; n[flip] *= -1
    return pts, n, rows[good], cols[good]


# ── gravity-aligned walls and sparse-track anchoring ─────────────────────────

def gravity_basis(normal: np.ndarray, up: np.ndarray) -> np.ndarray:
    """In-plane basis rows [along-wall, up-in-plane] so wall rectangles are plumb."""
    n = np.asarray(normal, dtype=np.float64) / np.linalg.norm(normal)
    u = np.asarray(up, dtype=np.float64); u = u - (u @ n) * n; u /= max(np.linalg.norm(u), 1e-12)
    right = np.cross(u, n); right /= max(np.linalg.norm(right), 1e-12)
    return np.stack([right, u])


def rebase_patch(patch: dict[str, Any], points: np.ndarray, basis: np.ndarray, centre: np.ndarray | None = None) -> dict[str, Any]:
    """Same inliers, new in-plane basis (and optionally a new centre): recompute the uv bounds and extent."""
    out = dict(patch); out["basis"] = np.asarray(basis, dtype=np.float64)
    if centre is not None:
        out["centre"] = np.asarray(centre, dtype=np.float64)
    uv = (np.asarray(points, dtype=np.float64)[out["ids"]] - out["centre"]) @ out["basis"].T
    out["lower_uv"], out["upper_uv"] = uv.min(axis=0), uv.max(axis=0); out["extent"] = (out["upper_uv"] - out["lower_uv"]).tolist()
    out["rms"] = float(np.sqrt(np.mean(((np.asarray(points)[out["ids"]] - out["centre"]) @ out["normal"]) ** 2)))
    return out


def anchor_to_tracks(patch: dict[str, Any], track_points: np.ndarray, min_tracks: int = 8, max_shift: float | None = None,
                     dilate: float = 0.0, band: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the dense plane's orientation but move it along its normal to the median
    offset of the sparse SfM tracks inside its footprint (the tracks are the more
    trusted position source; inferred depth carries a bias). `dilate` widens the
    footprint by that fraction of its extent on every side; `band` ignores tracks
    further than that from the plane (another wall seen through the footprint)."""
    pts = np.asarray(track_points, dtype=np.float64)
    info = {"tracks_in_footprint": 0, "tracks_used": 0, "shift_units": 0.0, "applied": False}
    if not len(pts):
        return patch, info
    rel = pts - patch["centre"]; uv = rel @ np.asarray(patch["basis"]).T
    lo, hi = np.asarray(patch["lower_uv"], dtype=np.float64), np.asarray(patch["upper_uv"], dtype=np.float64); pad = dilate * (hi - lo)
    inside = ((uv >= lo - pad) & (uv <= hi + pad)).all(axis=1)
    if band is not None:
        inside &= np.abs(rel @ np.asarray(patch["normal"])) <= band
    info["tracks_in_footprint"] = int(inside.sum())
    if inside.sum() < min_tracks:
        return patch, info
    off = rel[inside] @ patch["normal"]
    # robust: drop tracks further than 2.5 MAD from the median (a window reveal, a car, a bush)
    med = float(np.median(off)); mad = float(np.median(np.abs(off - med))) + 1e-9
    core = np.abs(off - med) <= 2.5 * mad * 1.4826 + 1e-9
    shift = float(np.median(off[core]))
    info.update({"tracks_used": int(core.sum()), "shift_units": shift, "track_spread_units": float(np.std(off[core]))})
    if max_shift is not None and abs(shift) > max_shift:
        info["refused"] = f"shift {shift:.3f} exceeds {max_shift:.3f}"
        return patch, info
    out = dict(patch); out["centre"] = np.asarray(patch["centre"]) + shift * np.asarray(patch["normal"]); info["applied"] = True
    return out, info


# ── coplanar merge and ray-based openings ────────────────────────────────────

def merge_coplanar(patches: list[dict[str, Any]], points: np.ndarray, angle_deg: float = 6.0, offset_tol: float = 0.1) -> list[dict[str, Any]]:
    """Union patches whose normals agree within angle_deg and whose planes sit within
    offset_tol of each other (fragments of one wall seen from different views)."""
    cos_tol = math.cos(math.radians(angle_deg)); out: list[dict[str, Any]] = []
    for p in sorted(patches, key=lambda q: -len(q["ids"])):
        for q in out:
            if float(np.asarray(q["normal"]) @ np.asarray(p["normal"])) >= cos_tol and abs(float((np.asarray(p["centre"]) - q["centre"]) @ q["normal"])) <= offset_tol:
                q["ids"] = np.concatenate([q["ids"], p["ids"]]); q["merged"] = q.get("merged", 1) + 1
                q.update(rebase_patch(q, points, q["basis"]))
                break
        else:
            out.append(dict(p))
    return out


def ray_plane_uv(pixels_rc: np.ndarray, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, patch: dict[str, Any], return_distance: bool = False):
    """Intersect the rays through pixels (rows, cols) with the patch's plane; returns
    plane-local (u, v) for rays that hit in front of the camera (depth-free). With
    return_distance, also the hit's camera depth (rays are built with unit -z, so the
    ray parameter equals the pinhole depth)."""
    m = np.asarray(c2w, dtype=np.float64); R, t = m[:3, :3], m[:3, 3]
    rows, cols = np.asarray(pixels_rc[0], dtype=np.float64), np.asarray(pixels_rc[1], dtype=np.float64)
    d_cam = np.stack([(cols + 0.5 - cx) / fx, -(rows + 0.5 - cy) / fy, -np.ones_like(rows)], axis=1)
    d = d_cam @ R.T
    n, c = np.asarray(patch["normal"], dtype=np.float64), np.asarray(patch["centre"], dtype=np.float64)
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((c - t) @ n) / denom
    ok = np.isfinite(s) & (s > 0) & (np.abs(denom) > 1e-9)
    hit = t + d[ok] * s[ok, None]
    uv = (hit - c) @ np.asarray(patch["basis"]).T
    return (uv, ok, s[ok]) if return_distance else (uv, ok)


def touches_image_border(mask: np.ndarray, margin: int = 2) -> bool:
    m = np.asarray(mask, dtype=bool)
    return bool(m[:margin].any() or m[-margin:].any() or m[:, :margin].any() or m[:, -margin:].any())


def cluster_rectangles(rects: list[dict[str, Any]], join_dist: float) -> list[dict[str, Any]]:
    """Group per-view rectangles (plane-local) whose centres lie within join_dist;
    each group becomes one rectangle with median bounds, a view count and spread."""
    groups: list[list[dict[str, Any]]] = []
    for r in rects:
        c = (np.asarray(r["lower_uv"]) + np.asarray(r["upper_uv"])) / 2
        for g in groups:
            gc = np.median([(np.asarray(x["lower_uv"]) + np.asarray(x["upper_uv"])) / 2 for x in g], axis=0)
            if np.linalg.norm(c - gc) <= join_dist and not any(x["view"] == r["view"] for x in g):
                g.append(r); break
        else:
            groups.append([r])
    out = []
    for g in groups:
        lo = np.median([x["lower_uv"] for x in g], axis=0); hi = np.median([x["upper_uv"] for x in g], axis=0)
        widths = [x["upper_uv"][0] - x["lower_uv"][0] for x in g]; heights = [x["upper_uv"][1] - x["lower_uv"][1] for x in g]
        out.append({"lower_uv": lo.tolist(), "upper_uv": hi.tolist(), "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]),
                    "views": sorted(x["view"] for x in g), "n_views": len(g), "width_spread": float(np.std(widths)), "height_spread": float(np.std(heights)),
                    "score": float(np.mean([x.get("score", 1.0) for x in g]))})
    return sorted(out, key=lambda r: (-r["n_views"], -r["width"] * r["height"]))
