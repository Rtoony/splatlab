#!/usr/bin/env python3
"""Prepare and run isolated, budget-matched RGB reconstruction experiments."""

import argparse
from copy import deepcopy
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import reconstruction_comparison as comparison


def require_gate():
    gate = ROOT / "tools/splatlab-compute-gate.sh"
    if subprocess.run([str(gate), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Launch this runner through tools/splatlab-compute-gate.sh --run")
    subprocess.run([str(gate), "--check"], check=True)


def runtime_environment():
    environment = Path("/home/rtoony/miniconda3/envs/dn-splatter-probe")
    if Path(sys.prefix).resolve() != environment.resolve():
        raise RuntimeError("Training requires the existing isolated dn-splatter-probe environment")
    for name in ("CUDA_HOME", "LIBRARY_PATH"):
        os.environ.pop(name, None)
    os.environ.update(CPATH=str(environment / "targets/x86_64-linux/include"),
                      TORCH_EXTENSIONS_DIR="/home/rtoony/tools/dn-splatter-probe/.torch_ext_finetune",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    os.environ["PATH"] = str(environment / "bin") + os.pathsep + os.environ.get("PATH", "")


def configuration(output, receipt, arm, destination):
    from dn_splatter.data.normal_nerfstudio import NormalNerfstudioConfig
    from dn_splatter.dn_config import ags_mesh, dn_splatter
    from dn_splatter.dn_datamanager import DNSplatterManagerConfig
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig

    config = deepcopy(method_configs["splatfacto"] if arm == "splatfacto" else ags_mesh.config if arm == "ags-mesh" else dn_splatter.config)
    budget = receipt["budget"]
    config.data = output / "dataset"
    config.output_dir = destination
    config.experiment_name = "paired"
    config.timestamp = "run"
    config.machine.seed = budget["seed"]
    config.max_num_iterations = budget["iterations_per_arm"]
    config.steps_per_eval_image = config.steps_per_eval_batch = config.steps_per_eval_all_images = 0
    config.steps_per_save = 1000
    config.load_checkpoint = config.load_dir = config.load_config = None
    config.vis = "tensorboard"
    config.viewer.quit_on_train_completion = True
    config.logging.local_writer.enable = False
    config.logging.profiler = "none"
    config.gradient_accumulation_steps = {}
    parser = NormalNerfstudioConfig(data=config.data, load_depths=False, load_normals=False, load_pcd_normals=False, downscale_factor=1)
    manager = DNSplatterManagerConfig(dataparser=parser, cache_images="cpu", cache_images_type="float32", max_thread_workers=4)
    model = config.pipeline.model
    model.camera_optimizer.mode = "off"
    model.num_downscales = 0
    model.background_color = "black"
    model.sh_degree = 3
    model.use_bilateral_grid = False
    model.output_depth_during_training = True
    if arm != "splatfacto":
        model.normal_supervision = "depth"
        model.use_depth_loss = False
    config.pipeline = VanillaPipelineConfig(datamanager=manager, model=model)
    config.optimizers = deepcopy(method_configs["splatfacto"].optimizers)
    if arm != "splatfacto":
        config.optimizers["normals"] = deepcopy(ags_mesh.config.optimizers["normals"])
    return config


def evaluate(pipeline, output, receipt):
    import numpy as np
    import torch
    from PIL import Image
    from torchmetrics.functional.image import structural_similarity_index_measure

    manager = pipeline.datamanager
    expected = receipt["prepared_splits"][receipt["evaluation_split"]]
    pipeline.eval()
    records = []
    rendering = output / "renders"
    rendering.mkdir()
    with torch.no_grad():
        for filename, camera, batch in comparison.held_out_frames(manager, output.parent / "dataset", expected):
            camera = camera.to("cuda")
            target = batch["image"].to("cuda").float()
            if batch["image"].dtype == torch.uint8:
                target /= 255
            torch.cuda.synchronize()
            started = time.monotonic()
            rendered = pipeline.model.get_outputs_for_camera(camera)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            rgb = rendered["rgb"].clamp(0, 1)
            if rgb.shape != target.shape or not torch.isfinite(rgb).all():
                raise RuntimeError("Evaluation render is nonfinite or has mismatched dimensions")
            valid = batch.get("mask", torch.ones_like(target[..., :1], dtype=torch.bool)).to("cuda").bool()
            if valid.ndim == 2:
                valid = valid[..., None]
            if not valid.any():
                raise RuntimeError("Held-out view has no valid evaluation pixels")
            difference = (rgb - target).square()[valid.expand_as(target)]
            mse = float(difference.mean())
            psnr = float(-10 * np.log10(max(mse, 1e-12)))
            ssim = float(structural_similarity_index_measure(rgb.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1)) if bool(valid.all()) else None
            opacity = rendered.get("accumulation")
            covered = float((opacity[valid] >= .5).float().mean()) if opacity is not None and opacity.shape == valid.shape else None
            image_name = Path(filename).stem + ".png"
            Image.fromarray((rgb.cpu().numpy() * 255).round().astype(np.uint8)).save(rendering / image_name)
            Image.fromarray((target.cpu().numpy() * 255).round().astype(np.uint8)).save(rendering / (Path(filename).stem + "-reference.png"))
            depth = rendered.get("depth")
            if depth is not None:
                np.save(rendering / (Path(filename).stem + "-depth.npy"), depth.cpu().numpy(), allow_pickle=False)
            records.append({"image": filename, "psnr_db": psnr, "ssim": ssim, "covered_fraction_alpha_0_5": covered,
                            "render_seconds": elapsed, "valid_pixels": int(valid.sum()), "width": rgb.shape[1], "height": rgb.shape[0]})
    return {"split": receipt["evaluation_split"], "views": records,
            "mean_psnr_db": float(np.mean([record["psnr_db"] for record in records])),
            "mean_ssim": float(np.mean([record["ssim"] for record in records])) if all(record["ssim"] is not None for record in records) else None,
            "geometry_error": None, "geometry_error_reason": "No independent dense reference is supplied",
            "coverage_scope": "rendered opacity, not independently verified geometric coverage", "ssim_scope": "full unmasked image only; omitted when masks are present"}


def train(output, arm, attempt=1):
    require_gate()
    receipt = comparison.verify(output)
    if not 1 <= attempt <= 16:
        raise ValueError("Use a numbered attempt from 1 to 16; never overwrite old runs")
    destination = output / (arm if attempt == 1 else f"{arm}-attempt-{attempt:02d}")
    destination.mkdir()
    runtime_environment()
    started = time.monotonic()
    result = {"arm": arm, "attempt": attempt, "prepared_receipt_sha256": receipt["sha256"], "created_at": manifests.utc_now(),
              "status": "running", "scope": receipt["scope"], "budget": receipt["budget"]}
    comparison.seal(destination, "run.json", result)

    def expired(_signal, _frame):
        raise TimeoutError("Comparison arm exceeded its registered wall-clock budget")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, receipt["budget"]["seconds_per_arm"])
    try:
        import numpy as np
        import torch
        from gsplat import rendering
        from nerfstudio.models.splatfacto import SplatfactoModel
        from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
        from nerfstudio.engine.trainer import Trainer

        torch.set_num_threads(4)
        random.seed(receipt["budget"]["seed"])
        np.random.seed(receipt["budget"]["seed"])
        torch.manual_seed(receipt["budget"]["seed"])
        torch.cuda.manual_seed_all(receipt["budget"]["seed"])
        torch.cuda.reset_peak_memory_stats()
        config = configuration(output, receipt, arm, destination)
        implementation_paths = [Path(inspect.getfile(config.pipeline.model._target)),
                                Path(inspect.getfile(config.pipeline.datamanager._target)),
                                Path(inspect.getfile(config.pipeline.datamanager.dataparser._target)),
                                Path(inspect.getfile(SplatfactoModel)), Path(inspect.getfile(FullImageDatamanager)),
                                Path(inspect.getfile(Trainer)), Path(inspect.getfile(rendering)),
                                Path(comparison.__file__), Path(__file__)]
        if arm != "splatfacto":
            implementation_paths.append(Path("/home/rtoony/tools/dn-splatter-probe/dn-splatter/dn_splatter/regularization_strategy.py"))
        implementations = {str(path): manifests.file_identity(path) for path in implementation_paths}
        result["implementation_sources"] = implementations
        config.save_config()
        trainer = config.setup(local_rank=0, world_size=1)
        trainer.setup(test_mode=receipt["evaluation_split"])
        actual_train = [str(path.relative_to(output / "dataset")) for path in trainer.pipeline.datamanager.train_dataset.image_filenames]
        if set(actual_train) != set(receipt["prepared_splits"]["train"]) or len(actual_train) != len(set(actual_train)):
            raise RuntimeError("Training datamanager does not honor the registered split")
        seed = trainer.pipeline.model.means.detach().cpu().numpy()
        initial_hash = __import__("hashlib").sha256(seed.tobytes()).hexdigest()
        if len(seed) != receipt["seed_points"]:
            raise RuntimeError("Training did not initialize from the shared seed point set")
        warmup_started = time.monotonic()
        trainer.pipeline.datamanager.cached_train
        trainer.pipeline.eval()
        with torch.no_grad():
            trainer.pipeline.model.get_outputs_for_camera(trainer.pipeline.datamanager.train_dataset.cameras[:1].to("cuda"))
        torch.cuda.synchronize()
        trainer.pipeline.train()
        result["cache_and_kernel_warmup_seconds"] = time.monotonic() - warmup_started
        result["initialization_seconds"] = time.monotonic() - started
        random.seed(receipt["budget"]["seed"])
        np.random.seed(receipt["budget"]["seed"])
        torch.manual_seed(receipt["budget"]["seed"])
        torch.cuda.manual_seed_all(receipt["budget"]["seed"])
        training_started = time.monotonic()
        trainer.train()
        torch.cuda.synchronize()
        training_seconds = time.monotonic() - training_started
        if trainer.step + 1 != receipt["budget"]["iterations_per_arm"]:
            raise RuntimeError("Training ended before its equal iteration budget")
        result.update(evaluation=evaluate(trainer.pipeline, destination, receipt), training_seconds=training_seconds,
                      iterations=trainer.step + 1, gaussians=int(len(trainer.pipeline.model.means)), initial_positions_sha256=initial_hash,
                      initial_gaussians=len(seed), max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                      packages={name: importlib.metadata.version(name) for name in ("torch", "nerfstudio", "gsplat", "torchmetrics", "numpy", "Pillow")},
                      variant="RGB-only Splatfacto baseline with shared sequential camera order" if arm == "splatfacto" else "depth-gradient normal regularization; no external monocular or sensor depth/normal priors")
        comparison.verify(output)
        if any(manifests.sha256_file(Path(path)) != identity["sha256"] for path, identity in implementations.items()):
            raise RuntimeError("Model implementation changed during the comparison arm")
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(total_seconds=time.monotonic() - started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      implementation_sha256=manifests.sha256_file(Path(__file__)))
        result["artifacts"] = {str(path.relative_to(destination)): manifests.file_identity(path) for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "run.json" and path.suffix in {".ckpt", ".yml", ".json", ".png", ".npy"}}
        comparison.seal(destination, "run.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"artifacts", "evaluation"}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--job", type=Path, required=True)
    prepare.add_argument("--iterations", type=int, default=3000)
    prepare.add_argument("--seconds", type=int, default=1200)
    prepare.add_argument("--downscale", type=int, default=2)
    run = commands.add_parser("train")
    run.add_argument("--comparison", type=Path, required=True)
    run.add_argument("--arm", choices=("splatfacto", "dn-splatter", "ags-mesh"), required=True)
    run.add_argument("--attempt", type=int, default=1)
    summarize = commands.add_parser("compare")
    summarize.add_argument("--comparison", type=Path, required=True)
    summarize.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    require_gate()
    if args.action == "prepare":
        output, receipt = comparison.prepare(args.job, args.iterations, args.seconds, args.downscale)
        print(json.dumps({"output": str(output), "receipt_sha256": receipt["sha256"], "seed_points": receipt["seed_points"], "split_sizes": {split: len(names) for split, names in receipt["splits"].items()}}), flush=True)
    elif args.action == "train":
        train(args.comparison.resolve(), args.arm, args.attempt)
    else:
        print(json.dumps(comparison.summarize(args.comparison.resolve(), args.runs), indent=2), flush=True)


if __name__ == "__main__":
    main()
