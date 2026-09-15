#!/usr/bin/env python3
"""Render occlusion-weighted instance contribution and depth for a prepared selection study.

Manual invocation must use tools/splatlab-compute-gate.sh --run and a bounded timeout.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from gsplat import rasterization

import artifact_manifest as manifests
import scene_revisions as scenes
import selection_reviews as reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("review_id")
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    with reviews.worker_slot(args.job.resolve(), args.review_id):
        run(args)


def run(args):
    job = args.job.resolve()
    receipt = reviews.read(job, args.review_id)
    reviews.verify(job, receipt)
    reviews.verify_prepared_files(job, receipt)
    output = reviews.directory(job, args.review_id)
    if args.candidate:
        candidate = reviews.read_record(job, args.review_id, "candidate.json")
        row_hash = candidate["artifacts"]["candidate-rows.npz"]["sha256"]
    else:
        row_hash = receipt["artifacts"]["rows.npz"]["sha256"]
    rows = np.load(output / ("candidate-rows.npz" if args.candidate else "rows.npz"), allow_pickle=False)
    core = rows["candidate"] if args.candidate else rows["core"]
    with np.load(output / "rows.npz", allow_pickle=False) as membership:
        other = membership["other"]
    ply_path = scenes.blob_path(job, receipt["splat_artifact"]["sha256"])
    if manifests.sha256_file(ply_path) != receipt["splat_artifact"]["sha256"]:
        raise RuntimeError("Captured splat checksum changed")
    vertices = PlyData.read(str(ply_path))["vertex"]
    if len(vertices) != receipt["n_rows"] or np.any(core < 0) or np.any(core >= len(vertices)):
        raise RuntimeError("Visibility rows do not address this captured splat")
    def values(fields):
        return torch.as_tensor(np.stack([vertices[field] for field in fields], axis=1).astype(np.float32), device="cuda")
    means = values(["x", "y", "z"])
    scales = values(["scale_0", "scale_1", "scale_2"]).exp()
    quats = values(["rot_0", "rot_1", "rot_2", "rot_3"])
    opacities = values(["opacity"]).sigmoid().squeeze(-1)
    colors = torch.zeros((len(vertices), 3), device="cuda")
    colors[torch.as_tensor(core, device="cuda"), 0] = 1
    colors[torch.as_tensor(other, device="cuda"), 1] = 1
    colors[:, 2] = 1
    stage = output / ("candidate-visibility" if args.candidate else "visibility")
    stage.mkdir(exist_ok=False)
    metres = receipt["calibration"]["meters_per_unit"]
    with torch.inference_mode():
        for camera in receipt["cameras"]:
            image_id = camera["image_id"]
            width, height = camera["width"], camera["height"]
            focal_x, focal_y, center_x, center_y = camera["parameters"]
            intrinsic = torch.tensor([[[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]]], device="cuda")
            viewmat = torch.tensor([camera["raw_to_camera"]], device="cuda", dtype=torch.float32)
            rendered, alpha, info = rasterization(means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmat, Ks=intrinsic, width=width, height=height, packed=True,
                near_plane=.01, far_plane=10000., render_mode="RGB+ED", sh_degree=None, rasterize_mode="antialiased")
            image = rendered[0].cpu().numpy()
            coverage = alpha[0, ..., 0].cpu().numpy()
            pixels = info["means2d"].round().long()
            valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
            pixels = pixels[valid].cpu().numpy()
            identifiers = info["gaussian_ids"][valid].cpu().numpy()
            depths = info["depths"][valid].cpu().numpy() * metres
            front = image[pixels[:, 1], pixels[:, 0], 3] * metres
            confident = coverage[pixels[:, 1], pixels[:, 0]] >= .85
            visible = confident & (np.abs(depths - front) <= .025 + .02 * front)
            np.savez_compressed(stage / f"cam_{image_id:03d}.npz", gaussian_ids=identifiers[visible], pixels=pixels[visible],
                                core=image[..., 0], other=image[..., 1], alpha=coverage, depth_m=image[..., 3] * metres)
            with Image.open(output / camera["photo"]) as opened:
                photo = np.array(opened.convert("RGB"), dtype=np.float32)
            weight = np.clip(image[..., 0], 0, 1)[..., None] * .65
            overlay = photo * (1 - weight) + np.array([255, 190, 0]) * weight
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(stage / f"overlay-{image_id}.png")
            print(f"view {image_id}: {int(visible.sum())} depth-consistent projected rows", flush=True)
    if manifests.sha256_file(output / ("candidate-rows.npz" if args.candidate else "rows.npz")) != row_hash:
        raise RuntimeError("Selection rows changed during visibility rendering")
    reviews.record_run(job, receipt, str((stage / "run.json").relative_to(output)), {"input_rows_sha256": row_hash, "candidate": args.candidate,
        "renderer": "gsplat", "torch": torch.__version__, "code_sha256": manifests.sha256_file(Path(__file__)),
        "visibility": "captured-Gaussian expected depth and alpha, not measured depth; 25 mm + 2% band, alpha >= 0.85"},
        sorted(stage.glob("*.npz")) + sorted(stage.glob("*.png")))


if __name__ == "__main__":
    main()
