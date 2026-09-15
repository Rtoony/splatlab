#!/usr/bin/env python3
"""Render inferred-depth evidence for a prepared background study through the bounded compute gate."""

import argparse
import fcntl
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData
import torch
import gsplat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import artifact_manifest as manifests
import background_visibility as visibility
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as selections


def run(job, identifier):
    receipt = visibility.read(job, identifier)
    visibility.verify(job, receipt)
    output = visibility.directory(job, identifier)
    if (output / "result.json").exists() or list(output.glob("camera-*.npz")):
        raise RuntimeError("Visibility outputs already exist; prepare another study instead of overwriting")
    if receipt["method"] != visibility.METHOD or receipt["policy"] != visibility.POLICY:
        raise RuntimeError("Prepared visibility policy is not supported by this worker")
    source = scenes.blob_path(job, receipt["splat_artifact"]["sha256"])
    if manifests.sha256_file(source) != receipt["splat_artifact"]["sha256"]:
        raise RuntimeError("Captured splat changed")
    vertices = PlyData.read(str(source))["vertex"]
    if len(vertices) != receipt["n_rows"]:
        raise RuntimeError("Visibility row space differs from its preparation")
    metres = receipt["calibration"]["meters_per_unit"]
    points = visibility.verified_grid(job, receipt)
    raw_points = points @ evidence.Y_UP / metres
    def values(fields):
        return torch.as_tensor(np.stack([vertices[field] for field in fields], axis=1).astype(np.float32), device="cuda")
    means = values(["x", "y", "z"])
    scales = values(["scale_0", "scale_1", "scale_2"]).exp()
    quaternions = values(["rot_0", "rot_1", "rot_2", "rot_3"])
    opacity = values(["opacity"]).sigmoid().squeeze(-1)
    features = torch.ones((len(vertices), 3), device="cuda")
    reports = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for camera in receipt["cameras"]:
            image_id = camera["image_id"]
            focal_x, focal_y, center_x, center_y = camera["parameters"]
            intrinsic = torch.tensor([[[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]]], device="cuda")
            transform = np.asarray(camera["raw_to_camera"])
            rendered, alpha, _ = gsplat.rasterization(means=means, quats=quaternions, scales=scales, opacities=opacity,
                colors=features, viewmats=torch.as_tensor(transform[None], device="cuda", dtype=torch.float32), Ks=intrinsic,
                width=camera["width"], height=camera["height"], packed=True, near_plane=.01, far_plane=10000.,
                render_mode="RGB+ED", sh_degree=None, rasterize_mode="antialiased")
            coordinates = raw_points @ transform[:3, :3].T + transform[:3, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                pixels = coordinates[:, :2] / coordinates[:, 2:] * camera["parameters"][:2] + camera["parameters"][2:]
            arrays = visibility.sample_render(alpha[0, ..., 0].cpu().numpy(), rendered[0, ..., 3].cpu().numpy() * metres,
                                             pixels, coordinates[:, 2] * metres)
            np.savez_compressed(output / f"camera-{image_id}.npz", **arrays)
            report = {"image_id": image_id, "split": camera["split"], "in_frame": int(arrays["in_frame"].sum()),
                      "qualified_cells": int(arrays["qualified"].sum())}
            reports.append(report)
            print(report, flush=True)
    visibility.verify(job, receipt)
    report = {"prepared_sha256": receipt["sha256"], "method": visibility.METHOD, "views": reports,
        "seconds": time.monotonic() - started, "created_at": manifests.utc_now(), "gsplat": gsplat.__version__, "torch": torch.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "code_sha256": manifests.sha256_file(Path(__file__)), "qualification_code_sha256": manifests.sha256_file(Path(visibility.__file__)),
        "scope": receipt["scope"]}
    result = selections.seal(job, output, "result.json", report, sorted(output.glob("camera-*.npz")))
    print({key: result[key] for key in ("method", "seconds", "sha256")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("visibility_id")
    args = parser.parse_args()
    job = args.job.resolve()
    with (visibility.directory(job, args.visibility_id) / "worker.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(job, args.visibility_id)


if __name__ == "__main__":
    main()
