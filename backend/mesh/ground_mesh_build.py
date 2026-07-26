#!/usr/bin/env python3
"""P6e: build a scene-unit ground TIN from semantic_ground.py's ground
gaussians, for the render/VR lane (twin_finish.py colors it into
ground_mesh.glb next) -- NOT the survey lane (that's ground_extract.py,
which requires a real-world CRS + geo anchor this scene may not have).

Reuses ground_extract.py's proven cell-binning + 15th-percentile-z +
spike-rejection + largest-connected-component logic verbatim (the "stray
far-field ground gaussians make the TIN interpolate a mountain" fix,
2026-07-21), just staying in scene units instead of transforming to ENU/CRS.

Runs in the dn-splatter-probe env (open3d/scipy/numpy).

Usage: ground_mesh_build.py <ground_gaussians.npz> <out_dir>
       [--semantic-thresh 0.5] [--cell-units 0.03] [--spike-tol-units 0.15]
       [--min-pts-cell 3]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_gaussians", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--semantic-thresh", type=float, default=0.5)
    ap.add_argument("--cell-units", type=float, default=0.03)
    ap.add_argument("--spike-tol-units", type=float, default=0.15)
    ap.add_argument("--min-pts-cell", type=int, default=3)
    ap.add_argument("--min-ground-points", type=int, default=50,
                    help="coverage floor -- loud fail below this, per the P6e gate")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    g = np.load(args.ground_gaussians)
    keep = g["rel"] >= args.semantic_thresh
    pts = g["xyz"][keep].astype(np.float64)
    if len(pts) < 500:
        print(f"FATAL: only {len(pts)} ground gaussians at rel>={args.semantic_thresh}",
              file=sys.stderr)
        return 1
    # Per-class scores (semantic_ground keeps the class axis since 2026-07-26);
    # absent on legacy npz files -> classless build, same output as before.
    class_ids = [str(x) for x in g["class_ids"]] if "class_ids" in g.files else None
    class_rel_kept = (
        g["class_rel"].astype(np.float32)[keep] if "class_rel" in g.files else None
    )

    # ── cell-bin (verbatim ground_extract.py algorithm, scene-unit XY) ───────
    ij = np.floor(pts[:, :2] / args.cell_units).astype(np.int64)
    order = np.lexsort((ij[:, 1], ij[:, 0]))
    ij_sorted, z_sorted = ij[order], pts[order, 2]
    keys, starts = np.unique(ij_sorted, axis=0, return_index=True)
    cells: dict[tuple[int, int], float] = {}
    cell_votes: dict[tuple[int, int], np.ndarray] = {}
    rel_sorted = class_rel_kept[order] if class_rel_kept is not None else None
    for k, (s, e) in enumerate(zip(starts, list(starts[1:]) + [len(z_sorted)])):
        if e - s >= args.min_pts_cell:
            key = tuple(keys[k])
            cells[key] = float(np.percentile(z_sorted[s:e], 15))
            if rel_sorted is not None:
                # summed-score vote: every member gaussian's class evidence
                # counts, not just its argmax — robust at class boundaries.
                cell_votes[key] = rel_sorted[s:e].sum(axis=0)

    kept_cells: dict[tuple[int, int], float] = {}
    rejected = 0
    for (i, j), z in cells.items():
        neigh = [cells[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di or dj) and (i + di, j + dj) in cells]
        if len(neigh) >= 3 and abs(z - float(np.median(neigh))) > args.spike_tol_units:
            rejected += 1
            continue
        kept_cells[(i, j)] = z

    disconnected_dropped = 0
    if kept_cells:
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

    cell_keys = list(kept_cells.keys())
    ground = np.array([((i + 0.5) * args.cell_units, (j + 0.5) * args.cell_units,
                        kept_cells[(i, j)]) for (i, j) in cell_keys])
    if len(ground) < args.min_ground_points:
        print(f"FATAL: only {len(ground)} ground cells after filtering "
              f"(floor: {args.min_ground_points})", file=sys.stderr)
        return 1

    # Per-cell class (TIN vertices are cell centers 1:1, so this array is also
    # the per-VERTEX class). -1 = no class evidence in the cell.
    class_counts: dict[str, int] = {}
    if class_ids is not None:
        votes = np.array([
            cell_votes.get(k, np.zeros(len(class_ids), np.float32))
            for k in cell_keys
        ])
        totals = votes.sum(axis=1)
        cell_class = np.where(totals > 0, votes.argmax(axis=1), -1).astype(np.int16)
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_conf = np.where(totals > 0, votes.max(axis=1) / np.maximum(totals, 1e-9), 0.0)
        np.savez_compressed(
            args.out_dir / "ground_class_cells.npz",
            cell_ij=np.array(cell_keys, dtype=np.int64),
            cell_units=np.float64(args.cell_units),
            class_idx=cell_class,
            conf=cell_conf.astype(np.float32),
            class_ids=np.array(class_ids),
        )
        class_counts = {
            cid: int((cell_class == i).sum()) for i, cid in enumerate(class_ids)
            if int((cell_class == i).sum())
        }

    # ── Delaunay TIN in XY, Z carried per vertex ─────────────────────────────
    tri = Delaunay(ground[:, :2])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(ground)
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    mesh.compute_vertex_normals()
    mesh_path = args.out_dir / "ground_mesh_raw.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    report = {
        "v": 1, "provenance": "ground-derived",
        "semantic_thresh": args.semantic_thresh, "cell_units": args.cell_units,
        "gaussians_total": int(len(g["rel"])), "gaussians_ground": int(keep.sum()),
        "cells_with_data": len(cells), "cells_spike_rejected": rejected,
        "cells_disconnected_dropped": disconnected_dropped,
        "ground_points": int(len(ground)), "triangles": int(len(tri.simplices)),
        "ground_z_range_units": [round(float(ground[:, 2].min()), 4),
                                 round(float(ground[:, 2].max()), 4)],
        "class_cells": class_counts or None,
        "artifacts": {"mesh": "ground_mesh_raw.ply",
                      **({"class_cells": "ground_class_cells.npz"}
                         if class_counts else {})},
    }
    (args.out_dir / "ground_mesh_build.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
