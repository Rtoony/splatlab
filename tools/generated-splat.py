#!/usr/bin/env python3
"""Place a retained native Gaussian master and verify frame-invariant real rasterization."""

import argparse
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import generated_gaussians as gaussians
import generated_color
import generated_objects as objects
import scene_revisions as scenes
import selection_reviews as reviews


def require_gate():
    gate = str(ROOT / "tools/splatlab-compute-gate.sh")
    if subprocess.run([gate, "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Use tools/splatlab-compute-gate.sh --run")
    subprocess.run([gate, "--check"], check=True)


def render(rows, camera, parameters, size, torch, rasterization):
    values = torch.as_tensor(rows, dtype=torch.float32, device="cuda")
    focal_x, focal_y, center_x, center_y = parameters
    intrinsic = torch.tensor([[[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]]], dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        rendered, alpha, _info = rasterization(
            means=values[:, :3].contiguous(), scales=values[:, 10:13].exp().contiguous(),
            quats=torch.nn.functional.normalize(values[:, 13:17], dim=1).contiguous(), opacities=values[:, 9].sigmoid().contiguous(),
            colors=values[:, 6:9].unsqueeze(1).contiguous(), sh_degree=0,
            viewmats=torch.tensor(camera[None], dtype=torch.float32, device="cuda"), Ks=intrinsic,
            width=size[0], height=size[1], near_plane=.001, far_plane=10000., packed=True, render_mode="RGB+ED", rasterize_mode="antialiased")
        torch.cuda.synchronize()
        return rendered[0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


def run(job, object_id):
    require_gate()
    import gsplat
    import torch
    from gsplat import rasterization

    torch.set_num_threads(4)
    with objects.worker_slot(job, object_id):
        receipt, parent = objects.read(job, object_id), objects.read(job, object_id, True)
        for completed, record in ((False, receipt), (True, parent)):
            for name in record["artifacts"]:
                objects.artifact(job, object_id, name, completed)
        sdk = gaussians.verify_sdk()
        matrix = gaussians.world_matrix(receipt, parent)
        identifier = "gaussians_" + uuid.uuid4().hex[:24]
        output = gaussians.directory(job, object_id, identifier)
        output.mkdir(parents=True)
        active = scenes.active(job)
        result = {"schema": "dev.splatlab.generated-gaussians/v1", "gaussians_id": identifier, "generated_object_id": object_id,
                  "generated_result_sha256": parent["sha256"], "prepared_receipt_sha256": receipt["sha256"],
                  "created_at": manifests.utc_now(), "base": receipt["base"], "active_at_review": active,
                  "status": "failed", "views": [], "sdk_export_sources": sdk,
                  "sdk_trace_scope": "verified installed sources at derivation; not retrospectively frozen at the original model run",
                  "gaussian_to_mesh": gaussians.GAUSSIAN_TO_MESH.tolist(), "native_to_world": matrix.tolist(),
                  "world_frame": "Y-up metres; transform baked into positions, log scales and WXYZ rotations",
                  "scope": "all-row native Gaussian master placement and renderer check; historical review only, no scene activation, no measured surface claim",
                  "frame_tolerances": gaussians.FRAME_TOLERANCES,
                  "independence": "check masks excluded from pose optimization; support/SfM can share camera evidence",
                  "implementation": {name: manifests.sha256_file(ROOT / name) for name in ("tools/generated-splat.py", "backend/generated_gaussians.py", "backend/generated_objects.py", "backend/generated_color.py")}}
        started = time.monotonic()

        def expire(_signal, _frame):
            raise TimeoutError("Native Gaussian review exceeded its 180-second budget")

        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(180)
        try:
            torch.cuda.reset_peak_memory_stats()
            native = gaussians.read_native(objects.artifact(job, object_id, "master-splat.ply", True))
            placed = gaussians.transform_rows(native, matrix)
            gaussians.write_placed(output / "placed-splat.ply", placed)
            exported = gaussians.read_native(output / "placed-splat.ply")
            np.testing.assert_array_equal(exported, placed)
            np.testing.assert_array_equal(exported[:, 3:10], native[:, 3:10])
            scale, _rotation = gaussians.similarity(matrix)
            result.update(gaussians=len(native), native_identity=parent["artifacts"]["master-splat.ply"],
                          all_rows_preserved=True, appearance_parameters_byte_exact=True, placed_export_byte_exact=True,
                          uniform_scale=scale, opacity_range=[float(native[:, 9].min()), float(native[:, 9].max())],
                          native_linear_scale_range=np.exp([float(native[:, 10:13].min()), float(native[:, 10:13].max())]).tolist())
            source_delivery = objects.artifact(job, object_id, "delivery.glb", True)
            if parent.get("mesh_color", {}).get("policy") == generated_color.COLOR_POLICY:
                if generated_color.decoded_rgba(source_delivery) is None:
                    raise RuntimeError("Generated mesh receipt and linear-color export disagree")
                (output / "appearance-delivery.glb").write_bytes(source_delivery.read_bytes())
                result["mesh_color"] = {**parent["mesh_color"], "already_encoded_delivery_retained": True}
            else:
                result["mesh_color"] = generated_color.derive(source_delivery, output / "appearance-delivery.glb")
            review = output / "review"
            review.mkdir()
            for camera in receipt["cameras"]:
                image_id = camera["image_id"]
                size = (480, round(camera["height"] * 480 / camera["width"]))
                parameters = np.asarray(camera["parameters"]) * np.tile(np.array(size) / [camera["width"], camera["height"]], 2)
                world_camera = objects.rigid_camera(camera)
                native_camera = world_camera @ matrix
                native_camera[:3] /= scale
                original, original_alpha = render(native, native_camera, parameters, size, torch, rasterization)
                current, current_alpha = render(exported, world_camera, parameters, size, torch, rasterization)
                common = (original_alpha >= .9) & (current_alpha >= .9)
                if not common.any():
                    raise RuntimeError("Cannot verify a view without opaque object coverage")
                difference = np.abs(original[..., :3] - current[..., :3])
                metrics = {"rgb_mae": float(difference.mean()), "rgb_p99": float(np.percentile(difference, 99)),
                           "alpha_max": float(np.abs(original_alpha - current_alpha).max()),
                           "depth_mae_m": float(np.abs(original[..., 3][common] * scale - current[..., 3][common]).mean())}
                with Image.open(objects.artifact(job, object_id, camera["mask"])) as opened:
                    mask = np.asarray(opened.resize(size, Image.Resampling.NEAREST)) >= 128
                with Image.open(objects.artifact(job, object_id, camera["photo"])) as opened:
                    photo = np.asarray(opened.convert("RGB").resize(size, Image.Resampling.LANCZOS))
                with Image.open(objects.artifact(job, object_id, f"review/master-{image_id}.png", True)) as opened:
                    opened.save(review / f"mesh-{image_id}.png")
                with np.load(objects.artifact(job, object_id, f"review/depth-{image_id}.npz", True), allow_pickle=False) as depths:
                    mesh_mask = np.isfinite(depths["master"])
                    depth_overlap = mesh_mask & (current_alpha >= .9)
                    mesh_depth_error = float(np.abs(depths["master"][depth_overlap] - current[..., 3][depth_overlap]).mean()) if depth_overlap.any() else None
                splat_mask = current_alpha >= .5
                record = {"image_id": image_id, "split": camera["split"], "frame": metrics,
                          "opaque_pixels": int(common.sum()), "splat_mask_iou": float(np.count_nonzero(splat_mask & mask) / max(1, np.count_nonzero(splat_mask | mask))),
                          "mesh_splat_silhouette_iou": float(np.count_nonzero(splat_mask & mesh_mask) / max(1, np.count_nonzero(splat_mask | mesh_mask))),
                          "mesh_splat_depth_mae_m": mesh_depth_error}
                result["views"].append(record)
                for label, picture in (("native", original[..., :3]), ("placed", current[..., :3]), ("alpha", current_alpha)):
                    Image.fromarray(np.rint(picture.clip(0, 1) * 255).astype(np.uint8)).save(review / f"{label}-{image_id}.png")
                Image.fromarray(photo).save(review / f"reference-{image_id}.png")
                np.savez_compressed(review / f"maps-{image_id}.npz", native_depth_m=original[..., 3] * scale,
                                    placed_depth_m=current[..., 3], native_alpha=original_alpha, placed_alpha=current_alpha)
                if any(metrics[name] > tolerance for name, tolerance in result["frame_tolerances"].items()):
                    raise RuntimeError("Native/placed rasterization exceeds the frozen frame-invariance tolerance")
                print({"image_id": image_id, "mask_iou": record["splat_mask_iou"], "frame": metrics}, flush=True)
            if gaussians.verify_sdk() != sdk:
                raise RuntimeError("SDK frame evidence changed")
            for name, digest in result["implementation"].items():
                if manifests.sha256_file(ROOT / name) != digest:
                    raise RuntimeError("Gaussian review implementation changed during the run")
            for completed, record in ((False, receipt), (True, parent)):
                if objects.read(job, object_id, completed)["sha256"] != record["sha256"]:
                    raise RuntimeError("Parent model evidence changed")
                for name in record["artifacts"]:
                    objects.artifact(job, object_id, name, completed)
            if scenes.active(job) != active:
                raise RuntimeError("Active scene changed during this read-only review")
            result["status"] = "needs-review"
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            result.update(seconds=time.monotonic() - started, torch=torch.__version__, gsplat=gsplat.__version__,
                          gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(), active_unchanged=scenes.active(job) == active)
            result = reviews.seal(job, output, "result.json", result, [path for path in output.rglob("*") if path.is_file() and path.name != "result.json"])
            print({key: result.get(key) for key in ("gaussians_id", "status", "seconds", "error", "sha256")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--generated-object-id", required=True)
    args = parser.parse_args()
    run(args.job.resolve(), args.generated_object_id)


if __name__ == "__main__":
    main()
