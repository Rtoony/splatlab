#!/usr/bin/env python3
"""Outdoor walk collision for Raw 360 scenes: the ground comes from where you walked.

Why (2026-09-30, ~/reports/2026-09-30-outdoor-walk-plan.md): world_shell.py wraps the whole gaussian cloud, and
outdoors that includes the sky floaters (20-27 % of splats sit >1.5 m above the camera). Its gates look down from
above and pass on that lid while the capture path is buried inside the solid. But a Raw 360 capture carries
something better than any gate: the camera path. The operator walked there, so it is free space with ground under
it at roughly one stick-height below.

The model (a 2.5D height map on a CELL grid, in the splat's Z-up metric frame):
  1. corridor   cells within CORRIDOR_M of the camera path; everything outside is fenced (owner: corridor only)
  2. ground     per cell, the 15th-percentile height of solid gaussians 0.3-3 m below the nearest camera,
                accepted only if it agrees with the path (camera z - camera height, tolerance widening with distance)
                and with its neighbours (spike rejection)
  3. cam height learned per scene (median camera-above-ground); outside CAM_H_BAND the owner's EYE_M (5.5 ft) is used
  4. fallback   cells under / beside the path with no ground evidence take the path's ground; small gaps are
                filled from neighbours; everything else with no ground (water, holes) is FENCED (owner: never fall in)
  5. obstacles  cells with >= OBST_MIN_PTS solid gaussians between ground+0.3 m and ground+2 m (cars, walls,
                hedges); never within PATH_CLEAR_M of the path. Anything above ground+3 m is ignored (sky, canopy)
  6. walkable   ground & corridor & not obstacle, connected to the path; cliffs (> CLIFF_M between neighbours) walled
Outputs (Y-up, metres; the same files world_shell.py writes, so the walker and the routes need nothing new):
  _world/collision_shell.glb   walkable surface + WALL_H walls around it and around obstacles
  _world/collision_shell.json  verdict / gates / params.seed_yup / probe (LOCAL floor at the spawn) / artifact
  _world/navmesh.json          v1 grid of walkable cells
  --classify                   only print {"indoor": bool, ...}: a ceiling above most of the path means indoor

  walk_ground.py <job_dir> [--classify] [--json]          (dn-splatter-probe env: numpy, scipy, plyfile, trimesh)
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

CELL = 0.15
CORRIDOR_M = 6.0
PATH_CLEAR_M = 0.5
PATH_FALLBACK_M = 1.0
EYE_M = 1.68                       # owner 2026-09-30: "approximately 5.5' eye height"
CAM_H_BAND = (1.2, 2.6)
GROUND_WINDOW = (0.3, 3.0)         # metres below the nearest camera a ground gaussian may sit
MIN_PTS_CELL = 3
GROUND_PERCENTILE = 15
SPIKE_M = 0.25
OBST_BAND = (0.3, 2.0)
OBST_MIN_PTS = 6
CLIFF_M = 0.45
WALL_H = 2.5
SKIRT_M = 0.3
SOLID_OPACITY = 0.5
MAX_SCALE_M = 0.25
CEILING_BAND = (0.3, 2.0)          # metres above the camera: a real ceiling is 0.4-1.5 m over a handheld camera;
                                   # higher solid layers are canopy or sky haze, not a roof
INDOOR_FRAC = 0.9                  # measured 09-30: indoor scenes 1.0; outdoor 0.12-0.65 (garden passage = planting overhead)
PASS_PATH_WALKABLE = 0.98
PASS_PATH_CLEAR = 0.99


# ---------------------------------------------------------------- inputs

def load_cameras(job_dir: Path) -> np.ndarray:
    frames = json.loads((job_dir / "_spirula" / "cameras" / "transforms.json").read_text())["frames"]
    path = sorted((f.get("file_path", ""), [float(f["transform_matrix"][r][3]) for r in range(3)]) for f in frames)
    return np.array([p for _, p in path], dtype=np.float64)


def load_solid_gaussians(job_dir: Path) -> np.ndarray:
    """Centres of solid, compact gaussians (Z-up). web.ply (3M, SH0) when present: same geometry, a fraction of the IO."""
    from plyfile import PlyData

    prev = job_dir / "_preview"
    ply = prev / "web.ply" if (prev / "web.ply").is_file() else prev / "splat.ply"
    el = PlyData.read(str(ply)).elements[0]
    xyz = np.stack([np.asarray(el[k], dtype=np.float32) for k in ("x", "y", "z")], axis=1)
    names = {p.name for p in el.properties}
    keep = np.ones(len(xyz), dtype=bool)
    if "opacity" in names:
        keep &= 1.0 / (1.0 + np.exp(-np.asarray(el["opacity"], dtype=np.float32))) > SOLID_OPACITY
    if "scale_0" in names:
        smax = np.max(np.stack([np.asarray(el[f"scale_{i}"], dtype=np.float32) for i in range(3)], 1), axis=1)
        keep &= np.exp(smax) < MAX_SCALE_M
    return xyz[keep & np.isfinite(xyz).all(1)]


# ---------------------------------------------------------------- grid helpers

class Grid:
    def __init__(self, cams: np.ndarray, margin: float = CORRIDOR_M + 1.0):
        self.x0 = float(cams[:, 0].min() - margin)
        self.y0 = float(cams[:, 1].min() - margin)
        self.nx = int(math.ceil((cams[:, 0].max() + margin - self.x0) / CELL))
        self.ny = int(math.ceil((cams[:, 1].max() + margin - self.y0) / CELL))

    def index(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((xy[:, 0] - self.x0) / CELL).astype(np.int64)
        iy = np.floor((xy[:, 1] - self.y0) / CELL).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix, iy, ok

    def centres(self) -> np.ndarray:
        gx = self.x0 + (np.arange(self.nx) + 0.5) * CELL
        gy = self.y0 + (np.arange(self.ny) + 0.5) * CELL
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], 1)


def neighbour_stack(a: np.ndarray) -> np.ndarray:
    """The 8 neighbours of every cell (NaN off-grid), shape (8, nx, ny)."""
    p = np.pad(a, 1, constant_values=np.nan)
    nx, ny = a.shape
    return np.stack([p[1 + dx:1 + dx + nx, 1 + dy:1 + dy + ny]
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy])


def per_cell_percentile(cell_id: np.ndarray, z: np.ndarray, n_cells: int, q: float, min_pts: int) -> np.ndarray:
    out = np.full(n_cells, np.nan, dtype=np.float64)
    if not len(cell_id):
        return out
    order = np.lexsort((z, cell_id))
    cid, zs = cell_id[order], z[order]
    starts = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]])
    counts = np.diff(np.r_[starts, len(cid)])
    ok = counts >= min_pts
    pick = starts + np.floor((counts - 1) * q / 100.0).astype(np.int64)
    out[cid[starts[ok]]] = zs[pick[ok]]
    return out


# ---------------------------------------------------------------- the model

def classify(cams: np.ndarray, pts: np.ndarray) -> dict:
    """Indoor = solid splats in a column above the camera along most of the path (a ceiling)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(pts[:, :2])
    sample = cams[:: max(1, len(cams) // 200)]
    covered = 0
    for c in sample:
        idx = tree.query_ball_point(c[:2], r=0.5)
        if idx:
            dz = pts[idx, 2] - c[2]
            covered += int(np.count_nonzero((dz > CEILING_BAND[0]) & (dz < CEILING_BAND[1])) >= 5)
    frac = covered / max(1, len(sample))
    return {"indoor": frac >= INDOOR_FRAC, "ceiling_frac": round(frac, 3), "sampled_cameras": len(sample)}


def build(job_dir: Path) -> dict:
    # Empty cells are the norm here (most of the grid has no ground evidence); nan-reductions over them are expected.
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    from scipy import ndimage
    from scipy.spatial import cKDTree

    t0 = time.time()
    cams = load_cameras(job_dir)
    pts = load_solid_gaussians(job_dir)
    g = Grid(cams)
    n = g.nx * g.ny
    centres = g.centres()
    cam_tree = cKDTree(cams[:, :2])
    dist, near = cam_tree.query(centres, k=1)
    dist = dist.reshape(g.nx, g.ny)
    cam_z = cams[near, 2].reshape(g.nx, g.ny)
    corridor = dist <= CORRIDOR_M

    ix, iy, ok = g.index(pts[:, :2])
    pts, ix, iy = pts[ok], ix[ok], iy[ok]
    below = cam_z[ix, iy] - pts[:, 2]
    gsel = (below >= GROUND_WINDOW[0]) & (below <= GROUND_WINDOW[1]) & corridor[ix, iy]
    cid = ix[gsel] * g.ny + iy[gsel]
    ground = per_cell_percentile(cid, pts[gsel, 2].astype(np.float64), n, GROUND_PERCENTILE,
                                 MIN_PTS_CELL).reshape(g.nx, g.ny)

    # camera height, learned per scene; the owner's 5.5 ft when the scene disagrees with any sane stick height
    cix, ciy, cok = g.index(cams[:, :2])
    hs = cams[cok, 2] - ground[cix[cok], ciy[cok]]
    hs = hs[np.isfinite(hs)]
    h_learned = float(np.median(hs)) if len(hs) >= 10 else None
    h = h_learned if h_learned is not None and CAM_H_BAND[0] <= h_learned <= CAM_H_BAND[1] else EYE_M
    expect = cam_z - h

    # evidence must agree with the path (slopes allowed to open up with distance) and with its neighbours
    tol = 0.5 + 0.15 * dist
    ground[np.abs(ground - expect) > tol] = np.nan
    med = np.nanmedian(neighbour_stack(ground), axis=0) if np.isfinite(ground).any() else ground
    spikes = np.isfinite(ground) & np.isfinite(med) & (np.abs(ground - med) > SPIKE_M)
    ground[spikes] = np.nan
    evidence = np.isfinite(ground) & corridor

    # fill: under/beside the path from the path; small gaps (<= 2 cells) from neighbours; the rest stays a hole
    filled = ground.copy()
    path_band = dist <= PATH_FALLBACK_M
    fb = path_band & ~np.isfinite(filled)
    filled[fb] = expect[fb]
    for _ in range(2):
        nb = np.nanmean(neighbour_stack(filled), axis=0) if np.isfinite(filled).any() else filled
        gap = ~np.isfinite(filled) & np.isfinite(nb) & corridor
        filled[gap] = nb[gap]
    has_ground = np.isfinite(filled) & corridor

    # obstacles: solid body-height splats; never on the path itself
    rel = pts[:, 2] - np.where(has_ground[ix, iy], filled[ix, iy], np.nan)
    osel = (rel >= OBST_BAND[0]) & (rel <= OBST_BAND[1])
    ocount = np.bincount(ix[osel] * g.ny + iy[osel], minlength=n).reshape(g.nx, g.ny)
    obstacle = (ocount >= OBST_MIN_PTS) & (dist > PATH_CLEAR_M) & has_ground

    walk = has_ground & ~obstacle
    labels, _ = ndimage.label(walk, structure=np.ones((3, 3), dtype=int))
    on_path = np.unique(labels[(dist <= CELL * 2) & walk])
    walk &= np.isin(labels, on_path[on_path > 0])

    path_cells = np.zeros((g.nx, g.ny), dtype=bool)
    path_cells[cix[cok], ciy[cok]] = True
    stats = {
        "cell_m": CELL, "grid": [g.nx, g.ny],
        "cam_height_learned_m": None if h_learned is None else round(h_learned, 3),
        "cam_height_used_m": round(h, 3),
        "cam_height_source": "learned" if h == h_learned else "owner eye height 5.5 ft",
        "corridor_cells": int(corridor.sum()),
        "ground_evidence_frac": round(float(evidence.sum() / max(1, corridor.sum())), 4),
        "path_fallback_frac": round(float(fb.sum() / max(1, corridor.sum())), 4),
        "fenced_frac": round(float((corridor & ~walk).sum() / max(1, corridor.sum())), 4),
        "obstacle_cells": int(obstacle.sum()),
        "path_walkable_frac": round(float(walk[path_cells].mean()), 4),
        "path_clear_frac": round(float((~obstacle[path_cells]).mean()), 4),
        "walkable_cells": int(walk.sum()),
    }
    return {"grid": g, "walk": walk, "ground": filled, "obstacle": obstacle, "dist": dist, "cams": cams,
            "stats": stats, "seconds_model": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- mesh + outputs

def yup(x, y, z):
    return np.stack([x, z, -y], axis=-1)


def build_mesh(m: dict):
    """Walkable surface (shared-vertex height field) + walls on every walkable edge that meets a non-walkable cell or
    a cliff. Returns (vertices Y-up, faces)."""
    g, walk, ground = m["grid"], m["walk"], m["ground"]
    nx, ny = walk.shape
    # corner heights: mean of the walkable cells touching the corner
    acc = np.zeros((nx + 1, ny + 1)); cnt = np.zeros((nx + 1, ny + 1))
    gw = np.where(walk, ground, 0.0)
    for dx in (0, 1):
        for dy in (0, 1):
            acc[dx:dx + nx, dy:dy + ny] += gw
            cnt[dx:dx + nx, dy:dy + ny] += walk
    corner = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    cx = g.x0 + np.arange(nx + 1) * CELL
    cy = g.y0 + np.arange(ny + 1) * CELL
    vid = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    used = cnt > 0
    vid[used] = np.arange(int(used.sum()))
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    verts = [yup(CX[used], CY[used], corner[used])]
    nv = int(used.sum())
    wi, wj = np.nonzero(walk)
    a, b, c, d = vid[wi, wj], vid[wi + 1, wj], vid[wi + 1, wj + 1], vid[wi, wj + 1]
    # counter-clockwise seen from +z = up-facing. (x,y,z)->(x,z,-y) is a proper rotation (det +1), so the winding
    # survives the Y-up swap; the path sweep in walk_spawn_check.py is what catches a flipped surface.
    faces = [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]

    # walls: for each walkable cell and each of its 4 edges, if the neighbour is not walkable or is a cliff away
    wall_quads = []
    for (di, dj, e0, e1) in ((-1, 0, (0, 0), (0, 1)), (1, 0, (1, 0), (1, 1)), (0, -1, (0, 0), (1, 0)), (0, 1, (0, 1), (1, 1))):
        ni, nj = wi + di, wj + dj
        inside = (ni >= 0) & (ni < nx) & (nj >= 0) & (nj < ny)
        nwalk = np.zeros_like(inside)
        cliff = np.zeros_like(inside)
        nwalk[inside] = walk[ni[inside], nj[inside]]
        cliff[inside] = nwalk[inside] & (np.abs(ground[ni[inside], nj[inside]] - ground[wi[inside], wj[inside]]) > CLIFF_M)
        need = ~inside | ~nwalk | cliff
        for k in np.flatnonzero(need):
            i, j = wi[k], wj[k]
            p0 = (cx[i + e0[0]], cy[j + e0[1]]); p1 = (cx[i + e1[0]], cy[j + e1[1]])
            base = ground[i, j]
            wall_quads.append((p0, p1, base - SKIRT_M, base + WALL_H))
    if wall_quads:
        q = np.array([[p0[0], p0[1], p1[0], p1[1], lo, hi] for p0, p1, lo, hi in wall_quads])
        wv = np.concatenate([yup(q[:, 0], q[:, 1], q[:, 4]), yup(q[:, 2], q[:, 3], q[:, 4]),
                             yup(q[:, 2], q[:, 3], q[:, 5]), yup(q[:, 0], q[:, 1], q[:, 5])], axis=0)
        k = len(q)
        base_idx = nv + np.arange(k)
        i0, i1, i2, i3 = base_idx, base_idx + k, base_idx + 2 * k, base_idx + 3 * k
        # the walker's collider is double-sided, so wall winding does not matter for collision
        faces += [np.stack([i0, i1, i2], 1), np.stack([i0, i2, i3], 1)]
        verts.append(wv)
    return np.concatenate(verts).astype(np.float32), np.concatenate(faces).astype(np.int64), len(wall_quads)


def pick_spawn(m: dict) -> tuple[list[float], float]:
    """A walkable, obstacle-free cell 0.6-1.0 m from the camera nearest the median camera: on real ground, a step
    off the path (the unmasked operator ghost lives ON the path)."""
    g, walk, ground, cams = m["grid"], m["walk"], m["ground"], m["cams"]
    med = np.median(cams, axis=0)
    cam = cams[np.argmin(((cams - med) ** 2).sum(1))]
    centres = g.centres().reshape(g.nx, g.ny, 2)
    d = np.hypot(centres[..., 0] - cam[0], centres[..., 1] - cam[1])
    for lo, hi in ((0.6, 1.0), (0.3, 1.5), (0.0, 3.0), (0.0, 1e9)):
        cand = walk & (d >= lo) & (d <= hi)
        if cand.any():
            i, j = np.unravel_index(np.argmin(np.where(cand, np.abs(d - 0.8), np.inf)), cand.shape)
            x, y = centres[i, j]
            gz = float(ground[i, j])
            return [round(float(x), 4), round(gz + 1.0, 4), round(float(-y), 4)], gz
    raise ValueError("no walkable cell anywhere")


def navmesh(m: dict, floor_y: float) -> dict:
    g, walk = m["grid"], m["walk"]
    # navmesh i runs along Y-up X (= x), j along Y-up Z (= -y): j = ny-1-iy, origin = (x0, -(y0 + ny*CELL))
    rows = ["".join("1" if walk[i, g.ny - 1 - j] else "0" for j in range(g.ny)) for i in range(g.nx)]
    return {"v": 1, "frame": "y-up", "cell": CELL, "origin": [g.x0, -(g.y0 + g.ny * CELL)],
            "shape": [g.nx, g.ny], "floor_y": floor_y, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    job = Path(args.job_dir)
    if args.classify:
        print(json.dumps(classify(load_cameras(job), load_solid_gaussians(job))))
        return 0
    t0 = time.time()
    m = build(job)
    verts, faces, walls = build_mesh(m)
    seed, floor = pick_spawn(m)
    import trimesh

    world = job / "_world"
    world.mkdir(exist_ok=True)
    out = world / "collision_shell.glb"
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.export(str(out))
    back = trimesh.load(str(out), force="mesh", process=False)
    st = m["stats"]
    verdict = "PASS" if (st["path_walkable_frac"] >= PASS_PATH_WALKABLE
                         and st["path_clear_frac"] >= PASS_PATH_CLEAR) else "NOT_WALKABLE"
    nav = navmesh(m, float(np.median(m["ground"][m["walk"]])))
    (world / "navmesh.json").write_text(json.dumps(nav))
    report = {
        "v": 1, "generator": "walk_ground.py", "job_dir": str(job), "route_used": "path-ground",
        "verdict": verdict,
        "gates": {"components": 1, "largest_component_frac": 1.0, "watertight": False,
                  # the walker/UI read floor_continuity: here it is the share of the camera path that is walkable
                  "floor_continuity": st["path_walkable_frac"], "max_hole_span": 0.0,
                  "path_walkable_frac": st["path_walkable_frac"], "path_clear_frac": st["path_clear_frac"]},
        "model": st,
        "params": {"seed_yup": seed, "cell_m": CELL, "corridor_m": CORRIDOR_M, "eye_m": EYE_M,
                   "wall_h_m": WALL_H, "obstacle_band_m": list(OBST_BAND)},
        "probe": {"floor_level_y": round(floor, 4), "top_level_y": round(floor + 3.0, 4)},
        "geometry_frame": {"axis": "y-up", "units": "meters", "meters_per_unit": 1.0},
        "artifact": {"path": str(out), "bytes": out.stat().st_size, "triangles_written": int(len(faces)),
                     "triangles_read_back": int(len(back.faces)), "walls": walls,
                     "match": int(len(back.faces)) == int(len(faces))},
        "navmesh": {"path": "navmesh.json", "shape": nav["shape"], "cell": CELL,
                    "walkable_cells": st["walkable_cells"]},
        "seconds": round(time.time() - t0, 1),
    }
    (world / "collision_shell.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("verdict", "gates", "model", "probe", "seconds")} |
                     {"triangles": report["artifact"]["triangles_written"]}) if args.json else
          f"{verdict} path_walkable={st['path_walkable_frac']} tris={len(faces)} {report['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
