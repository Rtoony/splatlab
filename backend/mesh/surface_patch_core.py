"""surface_patch_core: the pure math of paint-to-wall fill (PW-A6).

The user paints a sparse wall with a structure-category class; these
functions turn those painted points into a clean planar patch: cluster the
paint, fit a plane (seeded RANSAC — deterministic, receipt-friendly, and
refusing curved or degenerate regions BY NUMBER), rasterize the painted
footprint on the plane (closing bridges window-scale gaps; the footprint
never grows beyond the paint's bbox + pad — paint is authority AND scope),
and emit a shared-vertex quad-grid mesh wound to the plane normal.

numpy + scipy.ndimage only, in the ground_binning extraction tradition:
testable in the app venv, no open3d/trimesh at import time. The mesh-env CLI
(surface_patches.py) wires these to paint records, the capture-colour bake,
and provenance tags.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

import ground_binning

DEFAULT_CLUSTER_CELL_UNITS = 0.10   # 3D grid clustering (26-connected)
DEFAULT_PLANE_DIST_UNITS = 0.05     # RANSAC inlier distance
DEFAULT_PATCH_CELL_UNITS = 0.05     # 2D footprint grid on the plane
DEFAULT_MIN_CLUSTER_POINTS = 200
DEFAULT_MAX_PLANE_RMS_FRAC = 0.6    # rms <= frac * plane_dist_units
DEFAULT_CLOSE_CELLS = 2             # binary_closing radius (window-scale gaps)
DEFAULT_PAD_CELLS = 1
RANSAC_ITERS = 256
RANSAC_SEED = 1234


class SurfacePatchError(ValueError):
    """A painted region that must not become a patch — with the numbers."""


def grid_clusters(points: np.ndarray,
                  cell_units: float = DEFAULT_CLUSTER_CELL_UNITS) -> list[np.ndarray]:
    """Index arrays of 26-connected voxel components, largest first.

    A coarse-grid flood fill stands in for DBSCAN: same behaviour at wall
    scale, deterministic, and dependency-free."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return []
    keys = np.floor(points / cell_units).astype(np.int64)
    cell_members: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, keys)):
        cell_members.setdefault(key, []).append(index)

    unvisited = set(cell_members)
    clusters: list[np.ndarray] = []
    while unvisited:
        seed = unvisited.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            ci, cj, ck = frontier.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        if not (di or dj or dk):
                            continue
                        neighbour = (ci + di, cj + dj, ck + dk)
                        if neighbour in unvisited:
                            unvisited.remove(neighbour)
                            component.append(neighbour)
                            frontier.append(neighbour)
        indices = np.concatenate([np.asarray(cell_members[c], dtype=np.int64)
                                  for c in component])
        clusters.append(np.sort(indices))
    clusters.sort(key=len, reverse=True)
    return clusters


def fit_plane_ransac(points: np.ndarray,
                     dist_units: float = DEFAULT_PLANE_DIST_UNITS,
                     iters: int = RANSAC_ITERS,
                     seed: int = RANSAC_SEED,
                     max_rms_units: float | None = None,
                     min_inlier_frac: float | None = None) -> dict:
    """Seeded RANSAC + least-squares refinement. Returns
    {"normal", "d", "inliers", "inlier_frac", "rms_units"} with n·p + d = 0.

    Refuses — with the numbers in the message — when the cluster is too
    small or degenerate/collinear, and (the curvature check) when the plane
    explains too little of the paint: min_inlier_frac is THE curved-region
    detector. rms-over-inliers is kept in the result but is mathematically
    capped at ~0.577×dist_units (a uniform band), so it can never catch a
    curve on its own — measured by the half-cylinder test. 'Paint flatter
    segments separately' is the operator's remedy."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise SurfacePatchError(
            f"only {len(points)} points — a plane needs at least 3")

    rng = np.random.default_rng(seed)
    best_count = 0
    best_mask: np.ndarray | None = None
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        dist = np.abs((points - a) @ normal)
        mask = dist <= dist_units
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < 3:
        raise SurfacePatchError(
            "no plane hypothesis survived — the painted cluster is "
            "degenerate (collinear or coincident points)")

    # Least-squares refinement on the consensus set, then a final inlier pass.
    inlier_pts = points[best_mask]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    scale = float(s[0]) if s[0] > 0 else 1.0
    if s[1] / scale < 1e-9:
        raise SurfacePatchError(
            "painted points are collinear — no plane orientation exists")
    normal = vt[2]
    # Deterministic sign: the largest-|component| axis points positive.
    lead = int(np.argmax(np.abs(normal)))
    if normal[lead] < 0:
        normal = -normal
    d = -float(normal @ centroid)
    dist = np.abs(points @ normal + d)
    final_mask = dist <= dist_units
    inliers = np.flatnonzero(final_mask)
    rms = float(np.sqrt(np.mean(dist[final_mask] ** 2))) if len(inliers) else float("inf")
    frac = float(len(inliers)) / float(len(points))

    if min_inlier_frac is not None and frac < min_inlier_frac:
        raise SurfacePatchError(
            f"plane explains only {frac:.0%} of the painted points "
            f"(floor {min_inlier_frac:.0%}) — the region is not planar; "
            "paint flatter segments separately")
    if max_rms_units is not None and rms > max_rms_units:
        raise SurfacePatchError(
            f"plane rms {rms:.4f} > {max_rms_units:.4f} units — the painted "
            "region is not planar; paint flatter segments separately")
    return {"normal": normal, "d": d, "inliers": inliers,
            "inlier_frac": frac, "rms_units": rms}


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic orthonormal tangent frame with cross(u, v) == normal —
    which is exactly what makes cells_to_mesh's winding face the normal."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    u = np.cross(normal, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def footprint_cells(uv: np.ndarray, cell_units: float = DEFAULT_PATCH_CELL_UNITS,
                    close_cells: int = DEFAULT_CLOSE_CELLS,
                    pad_cells: int = DEFAULT_PAD_CELLS,
                    ) -> tuple[np.ndarray, tuple[int, int]]:
    """(occupied bool grid, origin (i0, j0)) — the painted footprint on the
    plane. binary_closing bridges interior window-scale gaps; the grid is
    bounded by the paint's bbox + pad, so fill never invents surface beyond
    what was painted. Only the largest 8-connected region survives (exact
    reuse of ground_binning.largest_component)."""
    uv = np.asarray(uv, dtype=np.float64)
    ij = np.floor(uv / cell_units).astype(np.int64)
    i0, j0 = ij.min(axis=0) - pad_cells
    i1, j1 = ij.max(axis=0) + pad_cells
    grid = np.zeros((int(i1 - i0 + 1), int(j1 - j0 + 1)), dtype=bool)
    grid[ij[:, 0] - i0, ij[:, 1] - j0] = True

    if close_cells > 0:
        structure = np.ones((2 * close_cells + 1, 2 * close_cells + 1), bool)
        grid = ndimage.binary_closing(grid, structure=structure)

    cells = {(int(i) + int(i0), int(j) + int(j0)): 1.0
             for i, j in zip(*np.nonzero(grid))}
    kept, _dropped = ground_binning.largest_component(cells)
    out = np.zeros_like(grid)
    for (i, j) in kept:
        out[i - i0, j - j0] = True
    return out, (int(i0), int(j0))


def cells_to_mesh(grid: np.ndarray, origin_ij: tuple[int, int],
                  cell_units: float, plane_point: np.ndarray,
                  u: np.ndarray, v: np.ndarray,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Shared-vertex quad grid over the occupied cells: two triangles per
    cell, wound so face normals equal cross(u, v) — the plane normal when
    (u, v) came from plane_basis."""
    i0, j0 = origin_ij
    corner_index: dict[tuple[int, int], int] = {}
    verts: list[np.ndarray] = []

    def corner(ci: int, cj: int) -> int:
        key = (ci, cj)
        if key not in corner_index:
            corner_index[key] = len(verts)
            offset_u = (i0 + ci) * cell_units
            offset_v = (j0 + cj) * cell_units
            verts.append(plane_point + u * offset_u + v * offset_v)
        return corner_index[key]

    faces: list[tuple[int, int, int]] = []
    for gi, gj in zip(*np.nonzero(grid)):
        c00 = corner(gi, gj)
        c10 = corner(gi + 1, gj)
        c11 = corner(gi + 1, gj + 1)
        c01 = corner(gi, gj + 1)
        faces.append((c00, c10, c11))
        faces.append((c00, c11, c01))
    if not faces:
        raise SurfacePatchError("empty footprint — nothing to mesh")
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int64))


def planar_uvs(verts: np.ndarray, plane_point: np.ndarray,
               u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """[0,1]-normalized planar UVs — the ground_texture 'planar projection is
    a perfect unwrap' trick, on this patch's own plane basis."""
    rel = np.asarray(verts, dtype=np.float64) - np.asarray(plane_point)
    raw = np.stack([rel @ u, rel @ v], axis=1)
    lo = raw.min(axis=0)
    span = raw.max(axis=0) - lo
    span[span <= 0] = 1.0
    return (raw - lo) / span


def pack_tiles(uv_list: list[np.ndarray], inset: float = 0.01) -> list[np.ndarray]:
    """Pack per-patch [0,1] UV charts into disjoint tiles of one atlas —
    the sqrt-grid layout without xatlas."""
    count = len(uv_list)
    if count == 0:
        return []
    grid = int(np.ceil(np.sqrt(count)))
    tile = 1.0 / grid
    packed = []
    for index, uv in enumerate(uv_list):
        gi, gj = divmod(index, grid)
        scale = tile - 2 * inset
        offset = np.array([gj * tile + inset, gi * tile + inset])
        packed.append(np.asarray(uv) * scale + offset)
    return packed
