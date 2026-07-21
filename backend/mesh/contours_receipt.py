#!/usr/bin/env python3
"""Render a plan-view receipt PNG of a contour DXF (majors emphasized).

Best-effort by contract: the route treats a receipt failure as a note, never a
build failure. Runs in the dn-splatter-probe env (ezdxf + matplotlib).

Usage: contours_receipt.py <contours.dxf> <out.png> [--title "..."]
"""
import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf")
    ap.add_argument("out_png")
    ap.add_argument("--title", default="Splat ground contours")
    args = ap.parse_args()

    import ezdxf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doc = ezdxf.readfile(args.dxf)
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#15171c")
    ax.set_facecolor("#15171c")
    drawn = 0
    for e in doc.modelspace():
        if e.dxftype() != "LWPOLYLINE":
            continue
        layer = e.dxf.layer
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        if "MJR" in layer:
            ax.plot(xs, ys, color="#a78bfa", lw=1.6)
        elif "MNR" in layer:
            ax.plot(xs, ys, color="#8b93a3", lw=0.7)
        else:  # TIN-face review linework, if drawn
            ax.plot(xs, ys, color="#3a3f49", lw=0.3)
        drawn += 1
    if drawn == 0:
        print("FATAL: no polylines to draw", file=sys.stderr)
        return 1
    ax.set_aspect("equal")
    ax.tick_params(colors="#9aa0ac", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#2e323a")
    ax.set_title(args.title, color="#e8e9ed", fontsize=10)
    fig.savefig(args.out_png, dpi=140, bbox_inches="tight")
    print(f"receipt: {drawn} polylines -> {args.out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
