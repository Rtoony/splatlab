"""Pure scene→world math for the survey export (no pyproj/open3d imports, so the
repo test suite can exercise it directly).

The scene→ENU formula is the documented anchor contract from geo_route.py:
heading_deg = compass bearing (deg CW from true north) of the scene's +Y axis,
anchor_scene = [ax, ay] scene ground point that sits at (lat, lon),
meters_per_unit = survey-calibrated scale.

ENU→grid uses calibration numbers the CALLER derives from the target CRS by
probing (project the anchor plus geodesic points 100 m true-north/true-east of
it, measure the grid frame) — never from a convergence-sign convention.
"""
from __future__ import annotations

import math

import numpy as np


def scene_to_enu(
    xyz: np.ndarray,
    meters_per_unit: float,
    heading_deg: float,
    anchor_scene: tuple[float, float],
) -> np.ndarray:
    """(N,3) scene-unit points -> (N,3) ENU meters about the anchor ground point."""
    th = math.radians(heading_deg)
    ax, ay = float(anchor_scene[0]), float(anchor_scene[1])
    s = float(meters_per_unit)
    x = xyz[:, 0] - ax
    y = xyz[:, 1] - ay
    east = s * (x * math.cos(th) + y * math.sin(th))
    north = s * (-x * math.sin(th) + y * math.cos(th))
    up = s * xyz[:, 2]
    return np.stack([east, north, up], axis=1)


def enu_to_grid(
    enu_m: np.ndarray,
    e0: float,
    n0: float,
    unit_factor_m: float,
    grid_rot_deg: float = 0.0,
    scale_factor: float = 1.0,
    elev0_units: float = 0.0,
) -> np.ndarray:
    """ENU meters -> projected-CRS coordinates in the CRS's native linear unit.

    e0/n0: projected anchor (CRS units). unit_factor_m: meters per CRS unit
    (1.0 metric, 1200/3937 US survey foot). grid_rot_deg: measured grid azimuth
    of TRUE NORTH at the anchor (from the caller's probe — grid north vs true
    north). scale_factor: measured projection point scale k at the anchor, so
    horizontal distances land as grid distances. Elevations get the unit
    conversion only — never the grid scale.
    """
    g = math.radians(grid_rot_deg)
    e, n = enu_m[:, 0], enu_m[:, 1]
    # Rotate the true-north ENU frame into the grid frame: a vector at true
    # azimuth a sits at grid azimuth a + grid_rot_deg.
    ge = e * math.cos(g) + n * math.sin(g)
    gn = -e * math.sin(g) + n * math.cos(g)
    h = float(scale_factor) / float(unit_factor_m)
    out = np.empty_like(enu_m)
    out[:, 0] = e0 + ge * h
    out[:, 1] = n0 + gn * h
    out[:, 2] = float(elev0_units) + enu_m[:, 2] / float(unit_factor_m)
    return out


def grid_to_scene(
    enz: np.ndarray,
    e0: float,
    n0: float,
    unit_factor_m: float,
    grid_rot_deg: float,
    scale_factor: float,
    elev0_units: float,
    meters_per_unit: float,
    heading_deg: float,
    anchor_scene: tuple[float, float],
) -> np.ndarray:
    """(N,3) projected-CRS [E, N, Z] -> scene-unit points: the exact inverse of
    scene_to_enu + enu_to_grid (round-trip unit-tested). Used to place survey
    products back into the scene frame for receipts/overlays."""
    h = float(scale_factor) / float(unit_factor_m)
    ge = (enz[:, 0] - e0) / h
    gn = (enz[:, 1] - n0) / h
    g = math.radians(grid_rot_deg)
    c, s = math.cos(g), math.sin(g)
    east = ge * c - gn * s
    north = ge * s + gn * c
    up = (enz[:, 2] - float(elev0_units)) * float(unit_factor_m)
    th = math.radians(heading_deg)
    ct, st = math.cos(th), math.sin(th)
    sc = float(meters_per_unit)
    out = np.empty_like(enz)
    out[:, 0] = (east * ct - north * st) / sc + float(anchor_scene[0])
    out[:, 1] = (east * st + north * ct) / sc + float(anchor_scene[1])
    out[:, 2] = up / sc
    return out


def xy_convex_hull(points_xy: np.ndarray) -> np.ndarray:
    """Monotone-chain 2D convex hull, (M,2) CCW without repeated endpoint."""
    pts = np.unique(points_xy[:, :2], axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(seq):
        chain: list[np.ndarray] = []
        for p in seq:
            while len(chain) >= 2 and np.cross(chain[-1] - chain[-2], p - chain[-2]) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])
