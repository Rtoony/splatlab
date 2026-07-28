#!/usr/bin/env python3
"""THE definition of floor support — shared by world_gate and world_shell.

Until 2026-07-28 the two graders measured "floor" differently and disagreed
on the same mesh (bonsai: world_shell 1.0 PASS vs world_gate 0.8968 FAIL).
The diagnosis: world_gate demanded the LOWEST surface of every interior
column to sit in a narrow band at ground level, which counts standable
furniture as missing floor — the failing columns hugged the room perimeter
with desk/shelf-height surfaces (0.6-1.3 m) exactly where cameras never saw
the occluded floor beneath. A player cannot fall through a desk.

The reconciled measurement is a HEAD-HEIGHT SUPPORT CAST, the physics
world_shell already used: from a probe height above ground, cast straight
down in every interior column; a hit means something standable/collidable
supports the player there. A genuine fall-through hole lets the ray sail
past ground level into void. This has no ceiling blind spot (rays start
below the ceiling, so a ceiling can never fake a floor — the failure mode
world_gate's lowest-surface design existed to catch) and it matches the
operator's lived walk of the world, which per the metric-trust doctrine is
the grade a metric must agree with before it drives work.

The interior footprint and ground level are still derived from an up-cast
from beneath the model (tessellation-independent, cannot be faked by a
canopy), and the old strict ground-band number is kept as a DIAGNOSTIC
(`ground_band_coverage`) — it is a fine measure of reconstruction quality
at ground level; it just is not a fall-through test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

GRID_CELLS = 128
INTERIOR_ERODE = 2
# Probe height as a fraction of the vertical span: unit-free, so the same
# measurement works on calibrated (metres) and uncalibrated (scene-unit)
# captures. 0.6 of a room's height is chest/head level for a single-storey
# capture — below any ceiling, above any furniture.
PROBE_HEIGHT_FRAC = 0.6
GROUND_BAND_FRAC = 0.15  # diagnostic band only


def _axis_indices(up_axis: str) -> tuple[int, list[int]]:
    up = {"X": 0, "Y": 1, "Z": 2}[up_axis.upper()]
    return up, [a for a in (0, 1, 2) if a != up]


def measure_floor_support(
    vertices: np.ndarray,
    faces: np.ndarray,
    up_axis: str = "Y",
    *,
    grid_cells: int = GRID_CELLS,
    erode: int = INTERIOR_ERODE,
    probe_height_frac: float = PROBE_HEIGHT_FRAC,
) -> dict[str, Any]:
    """Support coverage + largest-void analysis for one mesh.

    Raises ValueError when the mesh is too degenerate to lay out a grid or
    the interior footprint collapses — callers decide whether that is a
    gate failure or a skip.
    """
    import open3d as o3d
    from scipy import ndimage

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if len(faces) == 0:
        raise ValueError("empty mesh: no faces to measure")
    up, hax = _axis_indices(up_axis)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    span = hi - lo
    if span[hax[0]] <= 0 or span[hax[1]] <= 0 or span[up] <= 0:
        raise ValueError("mesh has zero extent on an axis; cannot lay out a floor grid")

    cell = max(span[hax[0]], span[hax[1]]) / grid_cells
    na = max(int(np.ceil(span[hax[0]] / cell)), 2)
    nb = max(int(np.ceil(span[hax[1]] / cell)), 2)
    ca = lo[hax[0]] + (np.arange(na) + 0.5) * cell
    cb = lo[hax[1]] + (np.arange(nb) + 0.5) * cell
    A, B = np.meshgrid(ca, cb, indexing="ij")
    pa, pb = A.ravel(), B.ravel()

    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(vertices),
                o3d.utility.Vector3iVector(faces),
            )
        )
    )
    pad = 0.01 * span[up] + 1e-3

    def _cast(level: float, direction: float) -> np.ndarray:
        rays = np.zeros((len(pa), 6), dtype=np.float32)
        rays[:, hax[0]] = pa
        rays[:, hax[1]] = pb
        rays[:, up] = level
        rays[:, 3 + up] = direction
        return rs.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    # Footprint + ground level from beneath the model: the first up-hit is
    # the lowest surface per column, independent of tessellation and blind
    # to canopies.
    t_up = _cast(float(lo[up] - pad), +1.0)
    hit_up = np.isfinite(t_up)
    hit_grid = hit_up.reshape(na, nb)
    footprint = ndimage.binary_fill_holes(hit_grid)
    interior = (
        ndimage.binary_erosion(footprint, iterations=erode) if erode else footprint
    )
    n_interior = int(interior.sum())
    if n_interior < 16:
        raise ValueError(
            f"interior footprint collapsed to {n_interior} cells; mesh is too "
            "sparse or too thin to test for a floor"
        )
    lowest_all = np.where(hit_up, (lo[up] - pad) + t_up, np.nan).reshape(na, nb)
    lowest_interior = lowest_all[interior]
    if not np.isfinite(lowest_interior).any():
        raise ValueError("no interior column contains any surface; mesh is empty")
    ground = float(np.nanmedian(lowest_interior))

    # THE measurement: support below a head-height probe.
    probe_level = ground + probe_height_frac * float(span[up])
    t_down = _cast(probe_level, -1.0)
    supported = np.isfinite(t_down).reshape(na, nb)
    coverage = float(supported[interior].mean())

    holes = interior & ~supported
    labels, n_regions = ndimage.label(holes)  # 4-connected: under-reports
    gap_cells = int(np.bincount(labels.ravel())[1:].max()) if n_regions else 0
    gap_frac = gap_cells / n_interior

    band = GROUND_BAND_FRAC * float(span[up])
    ground_band = (np.nan_to_num(lowest_all, nan=np.inf) <= ground + band) & hit_grid
    ground_band_coverage = float(ground_band[interior].mean())

    return {
        "coverage": round(coverage, 4),
        "largest_gap_frac": round(gap_frac, 4),
        "largest_gap_cells": gap_cells,
        "largest_gap_span": round(float(np.sqrt(gap_cells) * cell), 4),
        "void_regions": int(n_regions),
        "unsupported_cells": int(holes.sum()),
        "ground_level": round(ground, 4),
        "probe_level": round(float(probe_level), 4),
        "probe_height_frac": probe_height_frac,
        "grid": {
            "cells": [na, nb],
            "cell_size": round(float(cell), 5),
            "footprint_cells": int(footprint.sum()),
            "interior_cells": n_interior,
            "interior_frac_of_bbox": round(n_interior / (na * nb), 4),
        },
        "rays_cast": int(2 * len(pa)),
        "ground_band_coverage": round(ground_band_coverage, 4),
        "ground_band_note": (
            "diagnostic only: columns whose LOWEST surface sits within "
            f"{GROUND_BAND_FRAC:.2f}*span of ground level. Punishes standable "
            "furniture over occluded floor, which is why it is not the gated "
            "value (reconciliation, 2026-07-28)."
        ),
        "method": (
            "head-height support cast: down-rays from "
            f"ground + {probe_height_frac:.2f}*span over the interior footprint; "
            "a hit is standable/collidable support, a miss is a fall-through. "
            "Shared verbatim by world_gate and world_shell."
        ),
    }
