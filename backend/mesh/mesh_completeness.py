#!/usr/bin/env python3
"""WS2 mesh-completeness metric (adopted from the Blender-lab report, 2026-07-21):
"what did the mesh drop, in 3D?" — nearest-surface distance from every SOLID
gaussian (sigmoid(raw opacity) > 0.5, read from the job's _preview/splat.ply)
inside the mesh bbox (+25 cm margin) to the mesh surface, via open3d
RaycastingScene.compute_distance. CPU-only, seconds on a 1.3M-gaussian scene.

Distances are RAW SCENE UNITS unless --meters-per-unit is given; without it the
*_cm fields treat 1 unit = 1 m and the report says so — pass the job's
calibrated meters_per_unit (garden: 2.3537) for true centimeters.

Lab reference baseline (Blender BVH path, garden o3dtsdf-vanilla, mpu 2.3537):
70.4% within 5 cm / 20.1% beyond 10 cm / median 2.1 cm / p90 27.4 cm. The
>10 cm mass IS the dropped geometry (deep hedge, bushes, thin structures).
Compare within-implementation over time, not across implementations.

Usage: mesh_completeness.py <mesh.ply> <splat.ply> <out_json> [--meters-per-unit X]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData

BBOX_MARGIN_M = 0.25  # lab-report margin: solid gaussians within bbox +25 cm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("splat")
    ap.add_argument("out_json")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    args = ap.parse_args()
    t0 = time.time()

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.triangles) == 0:
        print("FATAL: empty mesh", file=sys.stderr)
        return 1

    v = PlyData.read(args.splat)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    solid = xyz[opac > 0.5]
    if len(solid) < 1000:
        print(f"FATAL: only {len(solid)} solid gaussians in {args.splat}", file=sys.stderr)
        return 1

    # cm thresholds and the 25 cm margin are defined in METERS; work in scene
    # units and convert. Uncalibrated jobs fall back to 1 unit = 1 m, flagged.
    mpu = args.meters_per_unit
    cm_per_unit = (mpu or 1.0) * 100.0
    margin_units = BBOX_MARGIN_M / (mpu or 1.0)

    bmin = np.asarray(mesh.get_min_bound()) - margin_units
    bmax = np.asarray(mesh.get_max_bound()) + margin_units
    in_bbox = np.all((solid >= bmin) & (solid <= bmax), axis=1)
    pts = solid[in_bbox]
    if len(pts) == 0:
        print("FATAL: no solid gaussians inside mesh bbox (+margin) — "
              "mesh and splat are not in the same space", file=sys.stderr)
        return 1

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    dist_cm = scene.compute_distance(
        o3d.core.Tensor(pts.astype(np.float32))
    ).numpy() * cm_per_unit

    report = {
        "v": 1,
        "solid_total": int(len(solid)),
        "solid_in_bbox": int(len(pts)),
        "pct_within_5cm": round(float((dist_cm <= 5.0).mean() * 100.0), 1),
        "pct_beyond_10cm": round(float((dist_cm > 10.0).mean() * 100.0), 1),
        "median_cm": round(float(np.median(dist_cm)), 2),
        "p90_cm": round(float(np.percentile(dist_cm, 90)), 2),
        "meters_per_unit": mpu,
        "units": "cm" if mpu else "scene-units treated as meters (UNCALIBRATED — pass --meters-per-unit)",
        "bbox_margin_m": BBOX_MARGIN_M,
        "convention": "o3d RaycastingScene.compute_distance, solid = sigmoid(opacity)>0.5",
        "lab_reference_blender": {
            "pct_within_5cm": 70.4, "pct_beyond_10cm": 20.1,
            "median_cm": 2.1, "p90_cm": 27.4,
        },
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
