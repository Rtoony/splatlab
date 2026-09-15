#!/usr/bin/env python3
"""Fuse paired model depths into private, explicitly inferred surface meshes."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import reconstruction_comparison as comparison
import reconstruction_surfaces as surfaces


def worker_module():
    specification = importlib.util.spec_from_file_location("comparison_worker", ROOT / "tools/reconstruction-comparison.py")
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    return worker


def implementation_sources():
    return {str(path): manifests.file_identity(path) for path in (Path(__file__), Path(surfaces.__file__), Path(comparison.__file__), ROOT / "tools/reconstruction-comparison.py")}


def verify(output):
    receipt = comparison.read(output)
    result = comparison.summarize(output.parent, receipt["arms"])
    if receipt["comparison_sha256"] != result["sha256"] or receipt["run_sha256"] != result["run_sha256"]:
        raise ValueError("Surface study no longer matches the registered comparison")
    for path, identity in receipt["implementation_sources"].items():
        if manifests.sha256_file(Path(path)) != identity["sha256"]:
            raise ValueError("Surface-study implementation changed; retain this study and prepare a new one")
    return receipt


def camera_record(camera):
    return surfaces.camera_record(camera.camera_to_worlds[0].detach().cpu().numpy(), int(camera.width.item()), int(camera.height.item()),
                                   float(camera.fx.item()), float(camera.fy.item()), float(camera.cx.item()), float(camera.cy.item()))


def integrate(volume, camera, depth, rgb, open3d):
    import numpy as np

    pose = np.eye(4)
    pose[:3] = camera["camera_to_world_opengl"]
    pose = pose @ np.diag([1., -1., -1., 1.])
    intrinsic = open3d.camera.PinholeCameraIntrinsic(camera["width"], camera["height"], camera["fx"], camera["fy"], camera["cx"], camera["cy"])
    image = open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
        open3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)), depth_scale=1., depth_trunc=4., convert_rgb_to_intensity=False)
    volume.integrate(image, intrinsic, np.linalg.inv(pose))


def cast_mesh(scene, mesh, camera, open3d):
    import numpy as np

    ray_values = surfaces.rays(camera)
    hits = scene.cast_rays(open3d.core.Tensor(ray_values), nthreads=4)
    depth = hits["t_hit"].numpy()
    valid = np.isfinite(depth) & (depth > 0)
    rgb = np.zeros((*depth.shape, 3), dtype=np.float32)
    normals = np.zeros_like(rgb)
    if valid.any():
        triangles = np.asarray(mesh.triangles)[hits["primitive_ids"].numpy()[valid]]
        uv = hits["primitive_uvs"].numpy()[valid]
        weights = np.column_stack([1 - uv.sum(axis=1), uv])
        rgb[valid] = (np.asarray(mesh.vertex_colors)[triangles] * weights[..., None]).sum(axis=1)
        normals[valid] = hits["primitive_normals"].numpy()[valid] * .5 + .5
    return depth, rgb, normals, ray_values


def self_test():
    import numpy as np
    import open3d as open3d

    camera = surfaces.camera_record(np.column_stack([np.eye(3), [0., 0., 1.]]), 64, 64, 64., 64., 32., 32.)
    volume = open3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.01, sdf_trunc=.04, color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    depth = np.ones((64, 64), dtype=np.float32)
    rgb = np.broadcast_to(np.array([60, 120, 180], dtype=np.uint8), (64, 64, 3)).copy()
    for _iteration in range(4):
        integrate(volume, camera, depth, rgb, open3d)
    mesh = volume.extract_triangle_mesh()
    scene = open3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rendered, colors, _normals, _rays = cast_mesh(scene, mesh, camera, open3d)
    center = rendered[16:48, 16:48]
    if not np.isfinite(center).all() or np.max(np.abs(center - 1)) > .02 or np.max(np.abs(colors[16:48, 16:48] * 255 - rgb[16:48, 16:48])) > 1:
        raise ValueError("Synthetic plane failed TSDF/depth-axis/color parity")
    print({"status": "passed", "scope": "controlled plane; not a captured-scene result", "triangles": len(mesh.triangles), "maximum_depth_error": float(np.max(np.abs(center - 1))), "open3d": open3d.__version__}, flush=True)


def load_pipeline(study, arm, worker):
    import numpy as np
    import torch
    import yaml

    run = comparison.read(study / arm, "run.json")
    receipt = comparison.read(study)
    sources = run["implementation_sources"]
    changed_local = {str(ROOT / "tools/reconstruction-comparison.py"), str(ROOT / "backend/reconstruction_comparison.py")}
    for path, identity in sources.items():
        if path not in changed_local and manifests.sha256_file(Path(path)) != identity["sha256"]:
            raise ValueError("Trained research model/runtime source changed")
    if any(importlib.metadata.version(name) != version for name, version in run["packages"].items()):
        raise ValueError("Trained model package versions changed")
    configs = [name for name in run["artifacts"] if name.endswith("config.yml")]
    checkpoints = [name for name in run["artifacts"] if name.endswith(".ckpt")]
    if len(configs) != 1 or len(checkpoints) != 1:
        raise ValueError("Require exactly one registered config and final checkpoint")
    config = worker.configuration(study, receipt, arm, study / arm)
    if yaml.dump(config) != (study / arm / configs[0]).read_text():
        raise ValueError("Reconstructed runtime config differs from the frozen training YAML")
    torch.manual_seed(42)
    pipeline = config.pipeline.setup(device="cuda", test_mode=receipt["evaluation_split"])
    seed = pipeline.datamanager.train_dataparser_outputs.metadata["points3D_xyz"].cpu().numpy()
    if hashlib.sha256(seed.tobytes()).hexdigest() != run["initial_positions_sha256"]:
        raise ValueError("Runtime seed coordinates differ from the trained normalization")
    expected_seeds, _normalization = surfaces.normalized_seeds(study, run)
    if not np.allclose(seed, expected_seeds, atol=1e-5, rtol=1e-5):
        raise ValueError("Prepared surface crop uses a different coordinate normalization")
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
        state = torch.load(study / arm / checkpoints[0], map_location="cpu", weights_only=True)
    if state["step"] + 1 != run["iterations"]:
        raise ValueError("Checkpoint is not the completed training step")
    pipeline.model.update_to_step(state["step"])
    pipeline.load_state_dict(state["pipeline"], strict=True)
    if len(pipeline.model.means) != run["gaussians"]:
        raise ValueError("Checkpoint Gaussian count changed during loading")
    pipeline.eval()
    return pipeline, run


def build(output, arm, worker):
    import numpy as np
    import open3d as open3d
    import torch
    from PIL import Image

    receipt = verify(output)
    if arm not in receipt["arms"]:
        raise ValueError("Arm is not registered in this surface study")
    if shutil.disk_usage(output).free < 5 * 1024 ** 3:
        raise ValueError("Surface retention requires five GiB of free space")
    destination = output / arm
    destination.mkdir()
    parameters = receipt["parameters"]
    started = time.monotonic()
    result = {"status": "running", "arm": arm, "surface_receipt_sha256": receipt["sha256"], "trained_run_sha256": receipt["run_sha256"][arm],
              "created_at": manifests.utc_now(), "parameters": parameters, "promoted": False, "provenance": receipt["provenance"]}
    comparison.seal(destination, "run.json", result)

    def expired(_signal, _frame):
        raise TimeoutError("Surface arm exceeded its registered wall-clock ceiling")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, parameters["seconds_per_arm"])
    try:
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        pipeline, trained_run = load_pipeline(output.parent, arm, worker)
        manager = pipeline.datamanager
        actual = [str(path.relative_to(output.parent / "dataset")) for path in manager.train_dataset.image_filenames]
        if len(actual) != len(receipt["training_images"]) or set(actual) != set(receipt["training_images"]) or set(actual) & set(receipt["evaluation_images"]):
            raise ValueError("Fusion must use all and only registered training images")
        batches = manager.cached_train
        if len(batches) != len(actual):
            raise ValueError("Incomplete training-image cache")
        volume = open3d.pipelines.integration.ScalableTSDFVolume(voxel_length=parameters["voxel_length"], sdf_trunc=parameters["sdf_truncation"], color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        records, camera_records = [], []
        samples = destination / "training-samples"
        samples.mkdir()
        result["initialization_seconds"] = time.monotonic() - started
        fusion_started = time.monotonic()
        with torch.no_grad():
            for index, filename in enumerate(actual):
                camera = manager.train_dataset.cameras[index:index + 1].to("cuda")
                record = camera_record(camera)
                camera_records.append({"image": filename, **record})
                rendered = pipeline.model.get_outputs_for_camera(camera)
                depth = rendered["depth"].squeeze(-1).cpu().numpy()
                opacity = rendered["accumulation"].squeeze(-1).cpu().numpy()
                batch = batches[index]
                rgb = np.rint(batch["image"].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                valid = batch["mask"].cpu().numpy().reshape(depth.shape) if "mask" in batch else np.ones_like(depth, dtype=bool)
                accepted_depth, accepted = surfaces.filtered_depth(depth, opacity, valid, surfaces.rays(record), receipt["bounds"], parameters)
                if not accepted.any():
                    raise ValueError(f"No supported crop pixels in training image {filename}")
                integrate(volume, record, accepted_depth, rgb, open3d)
                stem = Path(filename).stem
                np.savez_compressed(samples / (stem + ".npz"), depth=accepted_depth)
                Image.fromarray(rgb).save(samples / (stem + ".png"))
                records.append({"image": filename, "accepted_pixels": int(accepted.sum()), "pixels": accepted.size})
                if index % 24 == 0 or index + 1 == len(actual):
                    print(f"{arm}: fused {index + 1}/{len(actual)} training views", flush=True)
        result["fusion_seconds"] = time.monotonic() - fusion_started
        result["training_camera_sha256"] = surfaces.camera_digest(camera_records)
        result["training_views"] = records
        manifests.atomic_write_json(destination / "training-cameras.json", camera_records)
        comparison.seal(destination, "fusion.json", {"surface_receipt_sha256": receipt["sha256"], "trained_run_sha256": receipt["run_sha256"][arm],
            "training_views": records, "training_camera_sha256": result["training_camera_sha256"], "fusion_seconds": result["fusion_seconds"],
            "artifacts": {str(path.relative_to(destination)): manifests.file_identity(path) for path in sorted(samples.iterdir())}})
        mesh = volume.extract_triangle_mesh()
        vertices, triangles, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        result["extracted_mesh_counts"] = {"vertices": len(vertices), "triangles": len(triangles)}
        if not 1 <= len(triangles) <= parameters["maximum_triangles"] or not 1 <= len(vertices) <= parameters["maximum_vertices"]:
            raise ValueError(f"Extracted {len(vertices)} vertices / {len(triangles)} triangles; outside the registered nonempty artifact limits")
        surfaces.write_mesh(destination / "surface.ply", vertices, triangles, colors)
        clusters, cluster_counts, cluster_areas = mesh.cluster_connected_triangles()
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        result["mesh"] = {"vertices": len(vertices), "triangles": len(triangles), "components": len(cluster_counts),
                          "largest_component_triangle_fraction": float(max(cluster_counts) / len(triangles)),
                          "surface_area_training_units_squared": float(sum(cluster_areas)), "boundary_or_nonmanifold_edges": len(edges),
                          "minimum": vertices.min(axis=0).tolist(), "maximum": vertices.max(axis=0).tolist(),
                          "postprocessing": "none; all extracted components retained; no filling, smoothing, simplification or collider promotion"}
        del volume, clusters
        scene = open3d.t.geometry.RaycastingScene(nthreads=4)
        scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh))
        rendering = destination / "renders"
        rendering.mkdir()
        evaluation, eval_cameras = [], []
        with torch.no_grad():
            for filename, camera, batch in comparison.held_out_frames(manager, output.parent / "dataset", receipt["evaluation_images"]):
                camera = camera.to("cuda")
                record = camera_record(camera)
                eval_cameras.append({"image": filename, **record})
                model = pipeline.model.get_outputs_for_camera(camera)
                predicted = model["depth"].squeeze(-1).cpu().numpy()
                opacity = model["accumulation"].squeeze(-1).cpu().numpy()
                mesh_depth, rgb, normals, ray_values = cast_mesh(scene, mesh, record, open3d)
                target = batch["image"].cpu().numpy()
                valid = batch["mask"].cpu().numpy().reshape(predicted.shape) if "mask" in batch else np.ones_like(predicted, dtype=bool)
                roi = surfaces.crop_mask(ray_values, receipt["bounds"]) & valid
                _, model_support = surfaces.filtered_depth(predicted, opacity, valid, ray_values, receipt["bounds"], parameters)
                if not roi.any():
                    raise ValueError("Held-out view has no rays intersecting the registered crop")
                stem = Path(filename).stem
                old_render = np.asarray(Image.open(output.parent / arm / "renders" / (stem + ".png")))
                current_render = np.rint(model["rgb"].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                parity = float(np.abs(current_render.astype(float) - old_render).mean())
                if parity > .1:
                    raise ValueError("Reloaded checkpoint no longer reproduces retained RGB renders")
                squared_error = (rgb - target) ** 2
                mesh_hits = np.isfinite(mesh_depth) & (mesh_depth > 0)
                metrics = surfaces.depth_agreement(mesh_depth, predicted, model_support, parameters["voxel_length"])
                metrics.update(image=filename, roi_ray_pixels=int(roi.sum()), mesh_hit_fraction_on_roi_rays=float(mesh_hits[roi].mean()),
                               source_photo_psnr_on_roi_rays=float(-10 * np.log10(max(float(squared_error[roi].mean()), 1e-12))),
                               checkpoint_rgb_mean_absolute_byte_difference=parity)
                evaluation.append(metrics)
                Image.fromarray(np.rint(rgb.clip(0, 1) * 255).astype(np.uint8)).save(rendering / (stem + ".png"))
                Image.fromarray(np.rint(normals.clip(0, 1) * 255).astype(np.uint8)).save(rendering / (stem + "-normals.png"))
                Image.fromarray(np.rint(target.clip(0, 1) * 255).astype(np.uint8)).save(rendering / (stem + "-reference.png"))
                np.savez_compressed(rendering / (stem + ".npz"), mesh_depth=mesh_depth, model_depth=predicted, model_opacity=opacity, roi_rays=roi, model_support=model_support)
        result.update(evaluation=evaluation, training_views=records, evaluation_camera_sha256=surfaces.camera_digest(eval_cameras),
                      status="completed", geometric_reference_error=None, geometric_reference_reason="No independent dense reference; self-consistency is not accuracy",
                      max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                      open3d_version=open3d.__version__, torch_version=torch.__version__)
        manifests.atomic_write_json(destination / "evaluation-cameras.json", eval_cameras)
        verify(output)
        for path, identity in trained_run["implementation_sources"].items():
            if path not in {str(ROOT / "tools/reconstruction-comparison.py"), str(ROOT / "backend/reconstruction_comparison.py")} and manifests.sha256_file(Path(path)) != identity["sha256"]:
                raise ValueError("Research model/runtime source changed during surface extraction")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(total_seconds=time.monotonic() - started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        result["artifacts"] = {str(path.relative_to(destination)): manifests.file_identity(path) for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "run.json"}
        comparison.seal(destination, "run.json", result)
        print({key: value for key, value in result.items() if key not in {"artifacts", "evaluation", "training_views"}}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--comparison", type=Path, required=True)
    prepare.add_argument("--runs", nargs="+", required=True)
    prepare.add_argument("--voxel", type=float, default=.01)
    prepare.add_argument("--seconds", type=int, default=600)
    run = commands.add_parser("build")
    run.add_argument("--surface-study", type=Path, required=True)
    run.add_argument("--arm", choices=("splatfacto", "dn-splatter", "ags-mesh"), required=True)
    commands.add_parser("self-test")
    summarize = commands.add_parser("compare")
    summarize.add_argument("--surface-study", type=Path, required=True)
    args = parser.parse_args()
    worker = worker_module()
    worker.require_gate()
    if args.action == "prepare":
        output, receipt = surfaces.prepare(args.comparison.resolve(), args.runs, args.voxel, args.seconds)
        comparison.seal(output, "receipt.json", {**receipt, "implementation_sources": implementation_sources()})
        print(f"Surface study: {output}", flush=True)
    elif args.action == "compare":
        verify(args.surface_study.resolve())
        result, runs = surfaces.summarize(args.surface_study.resolve())
        filename = args.surface_study / "index.html"
        if not filename.exists():
            filename.write_text(surfaces.review_html(result, runs))
        print(f"Review: {filename}\nReceipt SHA-256: {result['sha256']}", flush=True)
    else:
        worker.runtime_environment()
        if args.action == "self-test":
            self_test()
        else:
            build(args.surface_study.resolve(), args.arm, worker)


if __name__ == "__main__":
    main()
