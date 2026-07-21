#!/usr/bin/env python3
"""Surface receipts: cross-SECTIONS + isometric TIN view for a survey build.

RToony's requested standard views (2026-07-21): "the iso view and section view
are tools I want in general." Runs in the dn-splatter-probe env, matplotlib
only (no offscreen GL — immune to the two-renderer segfault).

- sections.png: two auto swaths (principal axis of the ground set + its
  perpendicular, through the centroid) with the scene's colored mesh points as
  the reality reference (when a mesh exists) and the TIN line overlaid.
- surface_iso.png: shaded elevation-colored TIN from two view angles.

Usage: surface_receipts.py <ground_points.txt> <out_dir> --params-json '<json>'
       [--mesh mesh.ply]
Exit 0 = both receipts written (prints JSON artifact list).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_export import _crs_calibration  # noqa: E402
from geo_transform import grid_to_scene  # noqa: E402

BG, PANEL, TEXT, MUTED, LINE = "#15171c", "#1d2026", "#e8e9ed", "#9aa0ac", "#2e323a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out_dir")
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--mesh", default=None)
    args = ap.parse_args()
    p = json.loads(args.params_json)
    geo, mpu, epsg = p["geo"], float(p["meters_per_unit"]), int(p.get("epsg", 2226))
    anchor_scene = geo.get("anchor_scene") or [0.0, 0.0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.loadtxt(args.points, delimiter=",", usecols=(1, 2, 3))  # N,E,Z
    if d.ndim == 1:
        d = d[None, :]
    grid_pts = np.stack([d[:, 1], d[:, 0], d[:, 2]], axis=1)      # E,N,Z (ft)

    cal = _crs_calibration(epsg, geo["lat"], geo["lon"])
    cal.pop("transformer")
    elev0 = (geo.get("alt_m") or 0.0) / cal["unit_factor_m"]
    scene_pts = grid_to_scene(
        grid_pts, cal["e0"], cal["n0"], cal["unit_factor_m"], cal["grid_rot_deg"],
        cal["scale_factor"], elev0, mpu, geo.get("heading_deg", 0.0), anchor_scene,
    )
    ft = mpu * 3.28084  # scene unit -> feet, for human axes

    mesh_v = mesh_c = None
    if args.mesh and Path(args.mesh).is_file():
        import open3d as o3d  # io only — no GL

        m = o3d.io.read_triangle_mesh(args.mesh)
        if len(m.vertices):
            mesh_v = np.asarray(m.vertices)
            mesh_c = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None

    # ── sections: principal axis + perpendicular through the centroid ────────
    xy = scene_pts[:, :2]
    centroid = xy.mean(axis=0)
    evec = np.linalg.eigh(np.cov((xy - centroid).T))[1][:, -1]
    axes_dirs = [
        (f"Section A–A (principal axis)", np.array([evec[0], evec[1]])),
        (f"Section B–B (perpendicular)", np.array([-evec[1], evec[0]])),
    ]
    interp = LinearNDInterpolator(xy, scene_pts[:, 2])
    span = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    swath = 0.15

    fig, axs = plt.subplots(2, 1, figsize=(15, 8.5), facecolor=BG)
    for ax, (title, u) in zip(axs, axes_dirs):
        ax.set_facecolor(PANEL)
        perp = np.array([-u[1], u[0]])
        if mesh_v is not None:
            rel = mesh_v[:, :2] - centroid
            off = rel @ perp
            sel = np.abs(off) < swath
            sta = (rel[sel] @ u)
            colors = mesh_c[sel] if mesh_c is not None else None
            order = np.argsort(sta)
            ax.scatter(sta[order] * ft, mesh_v[sel][order][:, 2] * ft, s=1.1,
                       c=colors[order] if colors is not None else MUTED,
                       alpha=0.8, linewidths=0, label="scene (true colors)")
        t = np.linspace(-span * 0.6, span * 0.6, 400)
        line = centroid + t[:, None] * u[None, :]
        z = interp(line)
        ax.plot(t * ft, z * ft, color="#a78bfa", lw=2.2, label="ground TIN")
        ax.set_title(title, color=TEXT, fontsize=11)
        ax.set_xlabel("station (ft)", color=MUTED)
        ax.set_ylabel("elev (ft)", color=MUTED)
        ax.tick_params(colors=MUTED)
        ax.grid(color=LINE, lw=0.5)
        ax.set_aspect(2.0)
        ax.legend(loc="upper right", fontsize=8, facecolor=PANEL,
                  edgecolor=LINE, labelcolor=TEXT)
    fig.suptitle(f"{p.get('job_id', '')} ground sections · 2× vert. exag. · EPSG:{epsg}",
                 color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "sections.png", dpi=130, facecolor=BG)
    plt.close(fig)

    # ── isometric TIN ────────────────────────────────────────────────────────
    tri = Delaunay(grid_pts[:, :2])
    E, N, Z = grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2]
    fig = plt.figure(figsize=(15, 7), facecolor=BG)
    for i, (elev_a, azim) in enumerate(((32, -55), (28, 20))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_facecolor(BG)
        ax.plot_trisurf(E - E.min(), N - N.min(), Z, triangles=tri.simplices,
                        cmap=cm.viridis, linewidth=0.05,
                        edgecolor=(1, 1, 1, 0.06), antialiased=True)
        ax.set_box_aspect((E.max() - E.min(), N.max() - N.min(), (Z.max() - Z.min()) * 2))
        ax.view_init(elev=elev_a, azim=azim)
        ax.tick_params(colors=MUTED, labelsize=6)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor(PANEL)
            pane.pane.set_edgecolor(LINE)
    fig.suptitle(f"{p.get('job_id', '')} ground TIN · grid ft · 2× vert. exag.",
                 color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "surface_iso.png", dpi=130, facecolor=BG)
    plt.close(fig)

    print(json.dumps({"artifacts": ["sections.png", "surface_iso.png"],
                      "scene_reference": mesh_v is not None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
