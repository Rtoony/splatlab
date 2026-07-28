#!/usr/bin/env python3
"""Typed, attribute-preserving edits on a 3DGS splat PLY.

The engine half of the audited splat-edit lane: PLY in, PLY out, JSON report
on stdout. No lane management here — versioning, receipts, locking and
promotion live in the route layer; this script's whole contract is that the
output is the input with ONE typed operation applied and every gaussian
attribute preserved (schema, dtypes and property order byte-compatible).

Ops:
  crop_box   keep gaussians inside an axis-aligned box (PLY-native frame)
  clean      drop debris: opacity below a floor (real 0-1 opacity — the
             stored value is a logit), any axis scale above a ceiling
             (stored values are log-scale), and optional far-field outliers
             beyond a distance from the centroid
  transform  translate and/or uniformly scale (positions scaled AND log
             scales shifted by ln(s) — the lesson from the KIRI validation:
             gaussian scale lives in log space)

Runs in the dn-splatter-probe env (numpy + plyfile), like every mesh script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def _vec3(text: str, name: str) -> np.ndarray:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3 or not all(np.isfinite(parts)):
        raise SystemExit(f"FATAL: {name} must be three finite comma-separated numbers")
    return np.asarray(parts, dtype=np.float64)


def _positions(vertex) -> np.ndarray:
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def crop_box_mask(vertex, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if not np.all(hi > lo):
        raise SystemExit("FATAL: crop box max must exceed min on every axis")
    pos = _positions(vertex)
    return np.all((pos >= lo) & (pos <= hi), axis=1)


def clean_mask(
    vertex,
    *,
    min_opacity: float | None,
    max_scale: float | None,
    max_dist: float | None,
) -> np.ndarray:
    keep = np.ones(vertex.count, dtype=bool)
    if min_opacity is not None:
        if not 0.0 < min_opacity < 1.0:
            raise SystemExit("FATAL: --min-opacity must be in (0, 1)")
        # Stored opacity is the pre-sigmoid logit.
        logit = float(np.log(min_opacity / (1.0 - min_opacity)))
        keep &= np.asarray(vertex["opacity"], dtype=np.float64) >= logit
    if max_scale is not None:
        if max_scale <= 0:
            raise SystemExit("FATAL: --max-scale must be positive")
        # Stored scales are logs; a gaussian is a floater when ANY axis
        # exceeds the ceiling.
        log_ceiling = float(np.log(max_scale))
        scales = np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=1)
        keep &= np.asarray(scales, dtype=np.float64).max(axis=1) <= log_ceiling
    if max_dist is not None:
        if max_dist <= 0:
            raise SystemExit("FATAL: --max-dist must be positive")
        pos = _positions(vertex)
        centroid = np.median(pos, axis=0)  # median: robust to the very
        keep &= np.linalg.norm(pos - centroid, axis=1) <= max_dist  # outliers
    return keep


def apply_transform(data, *, translate: np.ndarray | None, scale: float | None):
    if scale is not None:
        if scale <= 0 or not np.isfinite(scale):
            raise SystemExit("FATAL: --scale must be a positive finite factor")
        for axis in ("x", "y", "z"):
            data[axis] = (np.asarray(data[axis], np.float64) * scale).astype(
                data[axis].dtype
            )
        shift = float(np.log(scale))
        for i in range(3):
            name = f"scale_{i}"
            data[name] = (np.asarray(data[name], np.float64) + shift).astype(
                data[name].dtype
            )
    if translate is not None:
        for axis, delta in zip(("x", "y", "z"), translate):
            data[axis] = (np.asarray(data[axis], np.float64) + delta).astype(
                data[axis].dtype
            )
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_ply")
    ap.add_argument("out_ply")
    ap.add_argument("--op", required=True, choices=("crop_box", "clean", "transform"))
    ap.add_argument("--min", dest="box_min", help="crop_box: x,y,z")
    ap.add_argument("--max", dest="box_max", help="crop_box: x,y,z")
    ap.add_argument("--min-opacity", type=float, default=None)
    ap.add_argument("--max-scale", type=float, default=None)
    ap.add_argument("--max-dist", type=float, default=None)
    ap.add_argument("--translate", default=None, help="transform: x,y,z")
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    in_ply, out_ply = Path(args.in_ply), Path(args.out_ply)
    ply = PlyData.read(str(in_ply))
    vertex = ply["vertex"]
    in_count = int(vertex.count)
    properties = [p.name for p in vertex.properties]
    for required in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"):
        if required not in properties:
            print(f"FATAL: not a 3DGS splat PLY (missing {required})",
                  file=sys.stderr)
            return 1

    data = vertex.data.copy()
    if args.op == "crop_box":
        if not args.box_min or not args.box_max:
            print("FATAL: crop_box needs --min and --max", file=sys.stderr)
            return 1
        mask = crop_box_mask(vertex, _vec3(args.box_min, "--min"),
                             _vec3(args.box_max, "--max"))
        data = data[mask]
    elif args.op == "clean":
        if args.min_opacity is None and args.max_scale is None and args.max_dist is None:
            print("FATAL: clean needs at least one of --min-opacity / "
                  "--max-scale / --max-dist", file=sys.stderr)
            return 1
        mask = clean_mask(vertex, min_opacity=args.min_opacity,
                          max_scale=args.max_scale, max_dist=args.max_dist)
        data = data[mask]
    else:
        if args.translate is None and args.scale is None:
            print("FATAL: transform needs --translate and/or --scale",
                  file=sys.stderr)
            return 1
        translate = _vec3(args.translate, "--translate") if args.translate else None
        data = apply_transform(data, translate=translate, scale=args.scale)

    out_count = int(len(data))
    if out_count == 0:
        print("FATAL: edit removed every gaussian — refusing to write an "
              "empty splat", file=sys.stderr)
        return 1
    for axis in ("x", "y", "z"):
        if not np.isfinite(np.asarray(data[axis], np.float64)).all():
            print(f"FATAL: non-finite {axis} after edit", file=sys.stderr)
            return 1

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(data, "vertex")
    PlyData([element], text=False).write(str(out_ply))

    back = PlyData.read(str(out_ply))["vertex"]
    back_props = [p.name for p in back.properties]
    if back.count != out_count or back_props != properties:
        out_ply.unlink(missing_ok=True)
        print("FATAL: readback mismatch (count or property schema changed)",
              file=sys.stderr)
        return 1

    print(json.dumps({
        "op": args.op,
        "in_count": in_count,
        "out_count": out_count,
        "removed": in_count - out_count,
        "properties": len(properties),
        "params": {
            "box_min": args.box_min, "box_max": args.box_max,
            "min_opacity": args.min_opacity, "max_scale": args.max_scale,
            "max_dist": args.max_dist,
            "translate": args.translate, "scale": args.scale,
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
