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
              class_relevancy: np.ndarray | None = None):
    """Bin XY into square cells; a cell's height is its 15th-percentile z.

    Returns (cells, votes): `cells` maps (i, j) -> height, `votes` maps
    (i, j) -> summed per-class evidence (empty when class_relevancy is None).
    Cells holding fewer than `min_pts_cell` points are dropped entirely.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return {}, {}

    ij = np.floor(points[:, :2] / cell_units).astype(np.int64)
    order = np.lexsort((ij[:, 1], ij[:, 0]))
    ij_sorted, z_sorted = ij[order], points[order, 2]
    keys, starts = np.unique(ij_sorted, axis=0, return_index=True)
    rel_sorted = class_relevancy[order] if class_relevancy is not None else None

    cells: dict[tuple[int, int], float] = {}
    votes: dict[tuple[int, int], np.ndarray] = {}
    for k, (s, e) in enumerate(zip(starts, list(starts[1:]) + [len(z_sorted)])):
        if e - s >= min_pts_cell:
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


def largest_component(cells: dict):
    """Keep only the largest 8-connected group of cells."""
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
    return {k: cells[k] for k in best}, len(cells) - len(best)


def cell_centres(cells: dict, cell_keys: list, cell_units: float) -> np.ndarray:
    """(N, 3) cell-centre points in scene units — the TIN's vertices."""
    return np.array([((i + 0.5) * cell_units, (j + 0.5) * cell_units, cells[(i, j)])
                     for (i, j) in cell_keys])


def build_ground_cells(points: np.ndarray, *, cell_units: float = DEFAULT_CELL_UNITS,
                       min_pts_cell: int = DEFAULT_MIN_PTS_CELL,
                       spike_tol_units: float = DEFAULT_SPIKE_TOL_UNITS,
                       class_relevancy: np.ndarray | None = None) -> dict:
    """All three stages in order, with the counts each one dropped."""
    cells, votes = bin_cells(points, cell_units, min_pts_cell, class_relevancy)
    binned = len(cells)
    cells, rejected = reject_spikes(cells, spike_tol_units)
    cells, disconnected = largest_component(cells)
    return {
        "cells": cells,
        "votes": votes,
        "binned": binned,
        "spikes_rejected": rejected,
        "disconnected_dropped": disconnected,
    }
