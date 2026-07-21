#!/usr/bin/env python3
"""Twin finishing stage (adopted from the Blender-lab WS3 findings, 2026-07-21):
splat->mesh COLOR TRANSFER + decimate + real-world-scale Y-up GLB.

The TSDF mesh's own colors are pale/washed; the splat's solid gaussians carry
the real material colors. This transfers them (6-NN inverse-distance blend),
scales to METERS when meters_per_unit is known, decimates with color
preservation (pymeshlab quadric), and writes a Blender-ready vertex-colored
GLB (Y-up) with a mandatory readback check. ~6 s on the garden twin.

Judged by SSIM + eyeball, not PSNR (lab finding: PSNR rewards TSDF blur).
xatlas UVs deliberately skipped — chart generation does not converge on
fragmented TSDF meshes (lab finding, >1 h runs killed).

Usage: twin_finish.py <mesh.ply> <splat.ply> <out.glb>
       [--meters-per-unit MPU] [--target-faces 400000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from plyfile import PlyData
from scipy.spatial import cKDTree

C0 = 0.28209479177387814  # SH DC -> RGB


def load_solid_gaussians(splat_ply: Path):
    v = PlyData.read(str(splat_ply))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    rgb = np.clip(0.5 + C0 * np.stack(
        [v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float32), 0, 1)
    solid = opac > 0.5
    return xyz[solid], rgb[solid]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("splat")
    ap.add_argument("out_glb")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    ap.add_argument("--target-faces", type=int, default=400_000)
    args = ap.parse_args()
    t0 = time.time()
    out_glb = Path(args.out_glb)

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if len(mesh.faces) == 0:
        print("FATAL: empty mesh", file=sys.stderr)
        return 1
    gx, gc = load_solid_gaussians(Path(args.splat))
    if len(gx) < 1000:
        print(f"FATAL: only {len(gx)} solid gaussians", file=sys.stderr)
        return 1

    # 6-NN inverse-distance color blend from solid gaussians (lab WS3 recipe).
    dist, idx = cKDTree(gx).query(np.asarray(mesh.vertices, dtype=np.float32), k=6, workers=-1)
    w = 1.0 / np.maximum(dist, 1e-6)
    w /= w.sum(axis=1, keepdims=True)
    vcols = np.einsum("nk,nkc->nc", w, gc[idx])

    scale = args.meters_per_unit or 1.0
    ms = pymeshlab.MeshSet()
    src = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64) * scale,
        face_matrix=np.asarray(mesh.faces),
        v_color_matrix=np.concatenate([vcols, np.ones((len(vcols), 1))], axis=1),
    )
    ms.add_mesh(src)          # id 0: full-res colored source
    ms.add_mesh(src)          # id 1: working copy to decimate
    ms.set_current_mesh(1)
    if ms.current_mesh().face_number() > args.target_faces:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=args.target_faces, preservenormal=True,
            preserveboundary=True, autoclean=True,
        )
        ms.transfer_attributes_per_vertex(sourcemesh=0, targetmesh=1, colortransfer=True)
    m = ms.current_mesh()
    verts = m.vertex_matrix().astype(np.float32)
    faces = m.face_matrix().astype(np.int64)
    cols = (np.clip(m.vertex_color_matrix(), 0, 1) * 255).astype(np.uint8)

    yup = np.stack([verts[:, 0], verts[:, 2], -verts[:, 1]], axis=1)
    out = trimesh.Trimesh(vertices=yup, faces=faces, process=False)
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=cols)
    out.export(str(out_glb))
    back = trimesh.load(str(out_glb), force="mesh")
    if len(back.faces) != len(faces):
        out_glb.unlink(missing_ok=True)
        print(f"FATAL: GLB readback mismatch ({len(back.faces)} != {len(faces)})", file=sys.stderr)
        return 1

    report = {
        "verts": int(len(yup)), "faces": int(len(faces)),
        "solid_gaussians": int(len(gx)),
        "units": "meters" if args.meters_per_unit else "scene-units (uncalibrated)",
        "extent": [round(float(x), 2) for x in back.extents],
        "glb_bytes": out_glb.stat().st_size,
        "seconds": round(time.time() - t0, 1),
    }
    (out_glb.parent / "twin_finish.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
