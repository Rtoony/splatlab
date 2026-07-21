#!/usr/bin/env python3
"""WS1 mesh-fidelity gate (adopted from the Blender-lab report, 2026-07-21):
render the mesh flat through evenly-spaced TRAIN cameras and score it against
the real photos. The honest number for "does this mesh look like the site?".

Cameras come from the DATAPARSER (the lab's verified fact — hand-composing
transforms.json chains is subtly wrong). Rendering is Open3D offscreen
(defaultUnlit vertex colors; ONE renderer per process — two segfault).
Convention: nerfstudio c2w is OpenGL; OpenCV extrinsic = inv(c2w @ diag(1,-1,-1,1)).

Lab reference baseline (Blender path, garden o3dtsdf-vanilla): PSNR 17.64 dB /
SSIM 0.234 / coverage 74.6%. This gate records its OWN baseline per metric
convention — compare within-gate over time, not across implementations.

Usage: mesh_gate.py <mesh.ply> <config.yml> <out_dir> [--cams 6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_orig_load = torch.load


def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _patched_load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--cams", type=int, default=6)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if len(mesh.triangles) == 0:
        print("FATAL: empty mesh", file=sys.stderr)
        return 1
    mesh.compute_vertex_normals()

    cfg_path = Path(args.config).resolve()
    # nerfstudio's own !!python/object config graph, self-produced on this box.
    config = yaml.load(cfg_path.read_text(), Loader=yaml.Loader)
    dp_config = config.pipeline.datamanager.dataparser
    dp_config.data = cfg_path.parents[2]  # <job>/processed — configs carry stale paths
    outputs = dp_config.setup().get_dataparser_outputs(split="train")
    cams = outputs.cameras
    n = int(cams.camera_to_worlds.shape[0])
    sel = sorted({int(round(i)) for i in np.linspace(0, n - 1, args.cams)})

    # ONE renderer per process (two segfault), so mixed-size camera sets
    # (ETH3D DSLR: four calibrations) are normalized: every cam's intrinsics
    # scale to the first cam's frame; photos resize to match. Sub-0.1% aspect
    # distortion on ETH3D — fine for a fidelity gate.
    W, H = int(cams.width[sel[0]]), int(cams.height[sel[0]])
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    per_cam = []
    for ci in sel:
        kx, ky = W / int(cams.width[ci]), H / int(cams.height[ci])
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(cams.fx[ci]) * kx, float(cams.fy[ci]) * ky,
            float(cams.cx[ci]) * kx, float(cams.cy[ci]) * ky,
        )
        c2w = np.eye(4)
        c2w[:3, :] = cams.camera_to_worlds[ci].numpy()
        extrinsic = np.linalg.inv(c2w @ flip)
        renderer.setup_camera(intr, extrinsic)
        render = np.asarray(renderer.render_to_image()).astype(np.float32) / 255.0
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        covered = np.isfinite(depth)

        photo = np.asarray(
            Image.open(outputs.image_filenames[ci]).convert("RGB").resize((W, H))
        ).astype(np.float32) / 255.0

        coverage = float(covered.mean())
        entry = {"cam": int(ci), "coverage": round(coverage, 3)}
        if coverage > 0.05:
            entry["psnr_covered"] = round(float(
                peak_signal_noise_ratio(photo[covered], render[covered], data_range=1.0)
            ), 2)
            entry["ssim_fullframe"] = round(float(
                structural_similarity(photo, render, channel_axis=2, data_range=1.0)
            ), 3)
        per_cam.append(entry)

        strip = np.concatenate([photo, render], axis=1)
        Image.fromarray((strip * 255).astype(np.uint8)).save(
            out_dir / f"gate_cam{ci:03d}.jpg", quality=88
        )

    scored = [e for e in per_cam if "psnr_covered" in e]
    report = {
        "v": 1,
        "cams": len(per_cam),
        "per_cam": per_cam,
        "median_coverage": round(float(np.median([e["coverage"] for e in per_cam])), 3),
        "median_psnr": round(float(np.median([e["psnr_covered"] for e in scored])), 2) if scored else None,
        "median_ssim": round(float(np.median([e["ssim_fullframe"] for e in scored])), 3) if scored else None,
        "convention": "o3d-unlit, psnr on covered px, ssim full-frame vs black-backed render",
        "lab_reference_blender": {"psnr": 17.64, "ssim": 0.234, "coverage": 0.746},
        "artifacts": [f"gate_cam{e['cam']:03d}.jpg" for e in per_cam],
    }
    (out_dir / "mesh_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_cam"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
