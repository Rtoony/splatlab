"""Ground-cell binning: the numerical core of ground_mesh_build.py.

Lifted out of `ground_mesh_build.main()` unchanged so it can be tested. The
algorithm is ground_extract.py's, and each of its three stages exists because of
a specific observed failure:

  1. **Cell-bin with a 15th-percentile z.** A mean or a min would ride on the
     lowest stray gaussian in the cell; the low-percentile picks the ground
     surface while tolerating a few points below it.
  2. **Spike rejection.** A cell whose height disagrees with the median of its
     neighbours is a stray, not terrain. This is the "stray far-field ground
     gaussians make the TIN interpolate a mountain" fix (2026-07-21) — without
     it, Delaunay happily bridges a rogue cell to the real ground and produces
     a spike across the whole scene.
  3. **Largest connected component.** A detached island of cells across the yard
     is not part of this ground surface; keeping it would make the TIN span the
     gap between them.

Only requires numpy — ground_mesh_build itself imports open3d and
scipy.spatial, which is exactly why this had no tests before.
"""

from __future__ import annotations

import numpy as np

# Kept here so the CLI defaults and the tested defaults cannot drift apart.
DEFAULT_CELL_UNITS = 0.03
DEFAULT_MIN_PTS_CELL = 3
DEFAULT_SPIKE_TOL_UNITS = 0.15

# The percentile used for a cell's height, and the neighbour count below which
# spike rejection abstains (too few neighbours to have an opinion).
GROUND_PERCENTILE = 15
MIN_NEIGHBOURS_FOR_SPIKE_TEST = 3


def bin_cells(points: np.ndarray, cell_units: float = DEFAULT_CELL_UNITS,
              min_pts_cell: int = DEFAULT_MIN_PTS_CELL,
              class_relevancy: np.ndarray | None = None,
              painted_mask: np.ndarray | None = None):
    """Bin XY into square cells; a cell's height is its 15th-percentile z.

    Returns (cells, votes): `cells` maps (i, j) -> height, `votes` maps
    (i, j) -> summed per-class evidence (empty when class_relevancy is None).
    Cells holding fewer than `min_pts_cell` points are dropped entirely —
    UNLESS the cell contains a painted-ground point (painted_mask): paint is
    explicit human evidence that this is floor, so the evidence-quantity
    floor relaxes to a single point there. The height percentile and every
    consistency check stay identical for painted cells.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return {}, {}

    ij = np.floor(points[:, :2] / cell_units).astype(np.int64)
    order = np.lexsort((ij[:, 1], ij[:, 0]))
    ij_sorted, z_sorted = ij[order], points[order, 2]
    keys, starts = np.unique(ij_sorted, axis=0, return_index=True)
    rel_sorted = class_relevancy[order] if class_relevancy is not None else None
    painted_sorted = (np.asarray(painted_mask, dtype=bool)[order]
                      if painted_mask is not None else None)

    cells: dict[tuple[int, int], float] = {}
    votes: dict[tuple[int, int], np.ndarray] = {}
    for k, (s, e) in enumerate(zip(starts, list(starts[1:]) + [len(z_sorted)])):
        has_paint = bool(painted_sorted[s:e].any()) if painted_sorted is not None else False
        if e - s >= min_pts_cell or has_paint:
            key = tuple(keys[k])
            cells[key] = float(np.percentile(z_sorted[s:e], GROUND_PERCENTILE))
            if rel_sorted is not None:
                # Summed-score vote: every member gaussian's class evidence
                # counts, not just its argmax — robust at class boundaries.
                votes[key] = rel_sorted[s:e].sum(axis=0)
    return cells, votes


def reject_spikes(cells: dict, spike_tol_units: float = DEFAULT_SPIKE_TOL_UNITS):
    """Drop cells that disagree with the median of their 8 neighbours.

    A cell with fewer than 3 neighbours is kept: an edge cell has too little
    context to be called a spike, and dropping it would erode the boundary.
    """
    kept: dict[tuple[int, int], float] = {}
    rejected = 0
    for (i, j), z in cells.items():
        neighbours = [cells[(i + di, j + dj)]
                      for di in (-1, 0, 1) for dj in (-1, 0, 1)
                      if (di or dj) and (i + di, j + dj) in cells]
        if (len(neighbours) >= MIN_NEIGHBOURS_FOR_SPIKE_TEST
                and abs(z - float(np.median(neighbours))) > spike_tol_units):
            rejected += 1
            continue
        kept[(i, j)] = z
    return kept, rejected


def largest_component(cells: dict, protected: frozenset = frozenset()):
    """Keep only the largest 8-connected group of cells — plus any PROTECTED
    (painted) cells outside it. A detached island the user explicitly painted
    is ground on human authority, not a stray; an unpainted island still
    drops. `dropped` counts every cell that did not survive."""
    if not cells:
        return {}, 0
    unvisited = set(cells)
    best: set[tuple[int, int]] = set()
    while unvisited:
        seed = unvisited.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            ci, cj = frontier.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    neighbour = (ci + di, cj + dj)
                    if neighbour in unvisited:
                        unvisited.remove(neighbour)
                        component.add(neighbour)
                        frontier.append(neighbour)
        if len(component) > len(best):
            best = component
    keep = best | (protected & set(cells))
    return {k: cells[k] for k in keep}, len(cells) - len(keep)


def painted_cell_keys(points: np.ndarray, painted_mask: np.ndarray,
                      cell_units: float) -> set[tuple[int, int]]:
    """The (i, j) cell keys that contain at least one painted point."""
    points = np.asarray(points, dtype=np.float64)
    mask = np.asarray(painted_mask, dtype=bool)
    if not mask.any():
        return set()
    ij = np.floor(points[mask, :2] / cell_units).astype(np.int64)
    return {tuple(row) for row in np.unique(ij, axis=0)}


def cell_centres(cells: dict, cell_keys: list, cell_units: float) -> np.ndarray:
    """(N, 3) cell-centre points in scene units — the TIN's vertices."""
    return np.array([((i + 0.5) * cell_units, (j + 0.5) * cell_units, cells[(i, j)])
                     for (i, j) in cell_keys])


def build_ground_cells(points: np.ndarray, *, cell_units: float = DEFAULT_CELL_UNITS,
                       min_pts_cell: int = DEFAULT_MIN_PTS_CELL,
                       spike_tol_units: float = DEFAULT_SPIKE_TOL_UNITS,
                       class_relevancy: np.ndarray | None = None,
                       painted_mask: np.ndarray | None = None) -> dict:
    """All three stages in order, with the counts each one dropped.

    Paint authority comes with a ledger: painted cells relax the sparse-cell
    floor and survive the component drop, and every such rescue is COUNTED
    (`painted_rescued_sparse` / `painted_rescued_component`) so a receipt
    reader can see exactly how much ground exists on human authority. Spike
    rejection is deliberately NOT exempted — it is a consistency check, and a
    stroke that clipped a hedge must not put a 3-unit spike into the TIN; a
    rejected painted cell shows up in `spikes_rejected` where the
    disagreement is visible."""
    has_paint = painted_mask is not None and np.asarray(painted_mask, bool).any()
    cells, votes = bin_cells(points, cell_units, min_pts_cell, class_relevancy,
                             painted_mask if has_paint else None)
    binned = len(cells)

    painted_keys: frozenset = frozenset()
    rescued_sparse = 0
    if has_paint:
        painted_keys = frozenset(painted_cell_keys(points, painted_mask, cell_units))
        # Exact rescue count: cells that exist now but would not without paint.
        strict, _ = bin_cells(points, cell_units, min_pts_cell, None)
        rescued_sparse = sum(1 for k in cells if k in painted_keys and k not in strict)

    cells, rejected = reject_spikes(cells, spike_tol_units)

    rescued_component = 0
    if has_paint:
        strict_kept, _ = largest_component(dict(cells))
        cells, disconnected = largest_component(cells, protected=painted_keys)
        rescued_component = len(set(cells) - set(strict_kept))
    else:
        cells, disconnected = largest_component(cells)

    return {
        "cells": cells,
        "votes": votes,
        "binned": binned,
        "spikes_rejected": rejected,
        "disconnected_dropped": disconnected,
        "painted_cells": len(painted_keys & set(cells)),
        "painted_rescued_sparse": rescued_sparse,
        "painted_rescued_component": rescued_component,
    }
