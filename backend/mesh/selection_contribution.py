#!/usr/bin/env python3
"""Measure per-Gaussian mask contribution; invoke only through the bounded compute gate."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from plyfile import PlyData
import torch
from gsplat import rasterization

import artifact_manifest as manifests
import scene_revisions as scenes
import selection_reviews as reviews


def run(job, identifier):
    receipt = reviews.read(job, identifier)
    reviews.verify(job, receipt)
    reviews.verify_prepared_files(job, receipt)
    mask_run = reviews.read_record(job, identifier, "mask-run.json")
    visibility = reviews.read_record(job, identifier, "visibility/run.json")
    if visibility.get("input_rows_sha256") != receipt["artifacts"]["rows.npz"]["sha256"]:
        raise RuntimeError("Contribution requires visibility from the prepared selection")
    output = reviews.directory(job, identifier)
    noun = manifests.read_json(output / "sam3_manifest.json")[receipt["label"]]["slug"]
    source = scenes.blob_path(job, receipt["splat_artifact"]["sha256"])
    if manifests.sha256_file(source) != receipt["splat_artifact"]["sha256"]:
        raise RuntimeError("Captured splat identity changed")
    vertices = PlyData.read(str(source))["vertex"]
    if len(vertices) != receipt["n_rows"]:
        raise RuntimeError("Contribution rows must address the prepared capture")
    def values(names):
        return torch.as_tensor(np.stack([vertices[name] for name in names], axis=1).astype(np.float32), device="cuda")
    means = values(["x", "y", "z"])
    scales = values(["scale_0", "scale_1", "scale_2"]).exp()
    rotations = values(["rot_0", "rot_1", "rot_2", "rot_3"])
    opacity = values(["opacity"]).sigmoid().squeeze(-1)
    stage = output / "contribution"
    stage.mkdir(exist_ok=False)
    started = time.monotonic()
    reports = []
    torch.cuda.reset_peak_memory_stats()
    for camera in receipt["cameras"]:
        image_id = camera["image_id"]
        with np.load(output / "visibility" / f"cam_{image_id:03d}.npz", allow_pickle=False) as visible:
            core = visible["core"]
        with np.load(output / "masks" / noun / f"cam_{image_id:03d}.npz", allow_pickle=False) as predicted:
            selected, reason = reviews.match_mask(predicted["masks"], predicted["scores"], core)
            mask = predicted["masks"][selected] if selected is not None else None
        record = {"image_id": image_id, "split": camera["split"], "selected_mask": selected, "reason": reason}
        if mask is None:
            reports.append(record)
            continue
        width, height = camera["width"], camera["height"]
        focal_x, focal_y, center_x, center_y = camera["parameters"]
        intrinsic = torch.tensor([[[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]]], device="cuda")
        transform = torch.tensor([camera["raw_to_camera"]], device="cuda", dtype=torch.float32)
        features = torch.zeros((len(vertices), 3), device="cuda", requires_grad=True)
        image, alpha, _ = rasterization(means=means, quats=rotations, scales=scales, opacities=opacity,
            colors=features, viewmats=transform, Ks=intrinsic, width=width, height=height, packed=True,
            near_plane=.01, far_plane=10000., render_mode="RGB", sh_degree=None, rasterize_mode="antialiased")
        pixels = torch.as_tensor(np.stack([mask, ~mask, np.ones_like(mask)], axis=-1), dtype=torch.float32, device="cuda")
        weights = torch.autograd.grad((image[0] * pixels).sum(), features)[0].detach().cpu().numpy()
        coverage = alpha[0, ..., 0].detach().cpu().numpy()
        reviews.contribution_rows(weights[:, 0], weights[:, 1], weights[:, 2], len(vertices))
        actual = weights.sum(axis=0, dtype=np.float64)
        expected = np.array([coverage[mask].sum(dtype=np.float64), coverage[~mask].sum(dtype=np.float64), coverage.sum(dtype=np.float64)])
        if not np.allclose(actual, expected, rtol=5e-5, atol=.5):
            raise RuntimeError("Per-Gaussian contribution does not conserve rendered alpha mass")
        np.savez_compressed(stage / f"cam_{image_id:03d}.npz", inside=weights[:, 0], outside=weights[:, 1], total=weights[:, 2])
        record.update(rows_with_visible_mass=int(np.count_nonzero(weights[:, 2] >= .5)),
                      inside_alpha_mass=float(actual[0]), conservation_error=float(np.abs(actual - expected).max()),
                      maximum_row_partition_error=float(np.abs(weights[:, 0] + weights[:, 1] - weights[:, 2]).max()))
        reports.append(record)
        print(record, flush=True)
    torch.cuda.synchronize()
    report = {"method": reviews.CONTRIBUTION_METHOD, "mask_run_sha256": mask_run["sha256"],
              "visibility_run_sha256": visibility["sha256"], "views": reports, "seconds": time.monotonic() - started,
              "code_sha256": manifests.sha256_file(Path(__file__)), "torch": torch.__version__,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "scope": "Integrated visible footprint weights, not point-center expected-depth filtering, trained semantics or measured geometry; no selection is activated"}
    reviews.read_record(job, identifier, "mask-run.json")
    reviews.read_record(job, identifier, "visibility/run.json")
    return reviews.record_run(job, receipt, "contribution/run.json", report, sorted(stage.glob("*.npz")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("review_id")
    args = parser.parse_args()
    with reviews.worker_slot(args.job.resolve(), args.review_id):
        result = run(args.job.resolve(), args.review_id)
    print({key: result[key] for key in ("method", "seconds", "peak_allocated_bytes", "sha256")})


if __name__ == "__main__":
    main()
