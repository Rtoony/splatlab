#!/usr/bin/env python3
"""Extract the GROUND surface from a splat mesh -> PNEZD survey points in a
projected CRS, ready for cdt survey_to_surface (TIN + contours).

Productionized 2026-07-21 from ~/tools/solidify-probe/probe_ground.py (proven
on garden: 2,123 pts -> 194 contours). Runs in the dn-splatter-probe env
(open3d/numpy/pyproj); shares geo_transform + the probe-derived CRS calibration
with geo_export.py.

Method (all measured, fail-loud):
  1. Upward faces only: face normal z >= cos(max_slope_deg) in ENU meters space.
  2. XY grid (cell_m): per-cell ground z = low percentile of upward-face
     centroid z (rejects tables/hedge tops sitting above the true ground).
  3. Spike rejection: drop cells deviating > spike_tol_m from the 3x3
     neighborhood median.
  4. Cell centers -> grid coordinates (probe-derived calibration) -> PNEZD CSV
     (Point,Northing,Easting,Z,Desc) + ground.json report.

Usage: probe_ground.py <mesh.ply> <out_dir> --params-json '<json>'
       params = geo_export params (+ optional cell_m, max_slope_deg,
                spike_tol_m, min_pts_cell)
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_export import _crs_calibration  # noqa: E402
from geo_transform import enu_to_grid, scene_to_enu  # noqa: E402
from provenance import GenerativeInputRefused, assert_not_generative  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--ground-gaussians", default=None,
                    help="ground_gaussians.npz from semantic_ground.py — when given, "
                         "ground samples come from SEMANTICALLY-ground gaussians "
                         "(richer + hole-free vs the TSDF mesh, and mesh-optional)")
    args = ap.parse_args()
    # Survey lane: refuse generative geometry (path OR in-file tag), fail-loud.
    try:
        assert_not_generative(args.mesh, lane="survey")
        if args.ground_gaussians:
            assert_not_generative(args.ground_gaussians, lane="survey")
    except GenerativeInputRefused as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1
    p = json.loads(args.params_json)
    geo, mpu, epsg = p["geo"], float(p["meters_per_unit"]), int(p.get("epsg", 2226))
    cell_m = float(p.get("cell_m", 0.25))
    max_slope_deg = float(p.get("max_slope_deg", 40.0))
    spike_tol_m = float(p.get("spike_tol_m", 0.5))
    min_pts_cell = int(p.get("min_pts_cell", 3))
    anchor_scene = geo.get("anchor_scene") or [0.0, 0.0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faces_total = faces_upward = None
    if args.ground_gaussians:
        # SEMANTIC source: gaussians the language field reads as ground — the
        # A/B insight (2026-07-21): raw gaussian centers carry far more usable
        # structure than the TSDF mesh, and semantics replace the slope filter.
        g = np.load(args.ground_gaussians)
        thresh = float(p.get("semantic_thresh", 0.5))
        keep = g["rel"] >= thresh
        pts_scene = g["xyz"][keep]
        if len(pts_scene) < 500:
            print(f"FATAL: only {len(pts_scene)} ground gaussians at rel>={thresh}", file=sys.stderr)
            return 1
        up_pts = scene_to_enu(pts_scene, mpu, geo.get("heading_deg", 0.0), anchor_scene)
        source = "semantic-gaussians"
        source_stats = {"gaussians_total": int(len(g["rel"])),
                        "gaussians_ground": int(keep.sum()),
                        "semantic_thresh": thresh}
    else:
        mesh = o3d.io.read_triangle_mesh(args.mesh)
        if len(mesh.triangles) == 0:
            print("FATAL: empty mesh", file=sys.stderr)
            return 1
        mesh.compute_triangle_normals()

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        # Everything in ENU meters: slopes and cell sizes mean what they say.
        enu = scene_to_enu(verts, mpu, geo.get("heading_deg", 0.0), anchor_scene)
        centroids = enu[tris].mean(axis=1)
        # Scene->ENU is a rotation about z + uniform scale, so normal z survives.
        normals_z = np.asarray(mesh.triangle_normals)[:, 2]
        upward = np.abs(normals_z) >= math.cos(math.radians(max_slope_deg))
        up_pts = centroids[upward]
        faces_total, faces_upward = int(len(tris)), int(upward.sum())
        source = "mesh-slope"
        source_stats = {}
    if len(up_pts) < 100:
        print(f"FATAL: only {len(up_pts)} ground samples", file=sys.stderr)
        return 1

    # Per-cell low-percentile z.
    ij = np.floor(up_pts[:, :2] / cell_m).astype(np.int64)
    order = np.lexsort((ij[:, 1], ij[:, 0]))
    ij_sorted, z_sorted = ij[order], up_pts[order, 2]
    keys, starts = np.unique(ij_sorted, axis=0, return_index=True)
    cells: dict[tuple[int, int], float] = {}
    for k, (s, e) in enumerate(zip(starts, list(starts[1:]) + [len(z_sorted)])):
        if e - s >= min_pts_cell:
            cells[tuple(keys[k])] = float(np.percentile(z_sorted[s:e], 15))

    # 3x3 neighborhood median spike rejection.
    kept_cells: dict[tuple[int, int], float] = {}
    rejected = 0
    for (i, j), z in cells.items():
        neigh = [cells[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di or dj) and (i + di, j + dj) in cells]
        if len(neigh) >= 3 and abs(z - float(np.median(neigh))) > spike_tol_m:
            rejected += 1
            continue
        kept_cells[(i, j)] = z

    # Semantic gaussians include stray far-field false positives with no spatial
    # coherence (found on garden: scattered background cells made the TIN
    # interpolate a mountain across the hull). Keep only the largest
    # 8-connected cell component — the site itself. Mesh-slope input already
    # has surface coherence, so its graded behavior stays untouched.
    disconnected_dropped = 0
    if args.ground_gaussians and kept_cells:
        unvisited = set(kept_cells)
        best_comp: set[tuple[int, int]] = set()
        while unvisited:
            seed = unvisited.pop()
            comp = {seed}
            frontier = [seed]
            while frontier:
                ci, cj = frontier.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        nb = (ci + di, cj + dj)
                        if nb in unvisited:
                            unvisited.remove(nb)
                            comp.add(nb)
                            frontier.append(nb)
            if len(comp) > len(best_comp):
                best_comp = comp
        disconnected_dropped = len(kept_cells) - len(best_comp)
        kept_cells = {k: kept_cells[k] for k in best_comp}

    ground = [((i + 0.5) * cell_m, (j + 0.5) * cell_m, z) for (i, j), z in kept_cells.items()]
    if len(ground) < 50:
        print(f"FATAL: only {len(ground)} ground cells after filtering", file=sys.stderr)
        return 1

    g_enu = np.array(ground)
    cal = _crs_calibration(epsg, geo["lat"], geo["lon"])
    cal.pop("transformer")
    elev0_units = (geo.get("alt_m") or 0.0) / cal["unit_factor_m"]
    grid = enu_to_grid(g_enu, cal["e0"], cal["n0"], cal["unit_factor_m"],
                       cal["grid_rot_deg"], cal["scale_factor"], elev0_units)

    pnezd = out_dir / "ground_points.txt"
    with pnezd.open("w") as f:
        for n, (e, no, z) in enumerate(grid, start=1):
            f.write(f"{n},{no:.4f},{e:.4f},{z:.4f},SPLAT-GRND\n")

    report = {
        "v": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epsg": epsg,
        "cell_m": cell_m,
        "max_slope_deg": max_slope_deg,
        "source": source,
        **source_stats,
        "faces_total": faces_total,
        "faces_upward": faces_upward,
        "cells_with_data": len(cells),
        "cells_spike_rejected": rejected,
        "cells_disconnected_dropped": disconnected_dropped,
        "ground_points": len(ground),
        "coverage_m2": round(len(ground) * cell_m * cell_m, 2),
        "ground_z_range_m": [round(float(g_enu[:, 2].min()), 3), round(float(g_enu[:, 2].max()), 3)],
        "grid_z_range_units": [round(float(grid[:, 2].min()), 3), round(float(grid[:, 2].max()), 3)],
        "artifacts": {"pnezd": "ground_points.txt"},
    }
    (out_dir / "ground.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
