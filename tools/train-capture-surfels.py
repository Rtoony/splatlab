#!/usr/bin/env python3
"""Bounded local 2D surface-splat pilot; no invented metric scale or camera refinement."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import capture_surfels as capture
from fisheye_evaluation import image_scores, summarize_scores, review_html
from reference_delivery import read_sparse_points
from reconstruction_review import export_splats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--sparse-depth-weight", type=float, default=0.)
    args = parser.parse_args()
    if subprocess.run([str(ROOT / "tools/splatlab-compute-gate.sh"), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Use the existing SplatLab compute gate")
    lease = json.loads(args.lease.read_text())
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat(lease["no_new_starts_after"]) or not 30 <= args.iterations <= 6000:
        raise ValueError("Outside the compute start window or bounded iteration range")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("Use a new output outside the frozen source dataset")
    document, frames, splits, snapshot, center, scale = capture.frozen_dataset(dataset)
    if not 0 <= args.sparse_depth_weight <= .2 or bool(args.structure) != (args.sparse_depth_weight > 0):
        raise ValueError("Source anchors require a structural study and explicit weight in (0, .2]")
    anchors, anchor_metadata = {}, None
    if args.structure:
        anchors, anchor_metadata, anchor_snapshot = capture.source_depth_anchors(args.structure, dataset, document, frames)
        snapshot.update(anchor_snapshot)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
    import numpy as np
    from PIL import Image
    import torch
    from gsplat import rendering
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree
    from torchmetrics.functional.image import structural_similarity_index_measure

    for implementation in (Path(__file__), Path(capture.__file__), Path(inspect.getfile(rendering)), Path(inspect.getfile(DefaultStrategy))):
        snapshot[implementation] = manifests.sha256_file(implementation)
    torch.set_num_threads(4)
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    random = np.random.default_rng(20260909)
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"schema": "dev.splatlab.fisheye-baseline-evaluation/v1", "status": "running", "method": "gsplat-2dgs",
               "started_at": manifests.utc_now(), "metric_scale": "unknown", "registration": None, "owner_accepted": False,
               "source_hashes": {str(path): digest for path, digest in snapshot.items()}, "test_split_evaluated": False,
               "solver_normalization": {"center": center.tolist(), "scale": scale, "rotation": "identity",
                                        "export": "exact inverse similarity to original SfM frame; not metric registration"},
               "experiment": "geometry-specific pilot, not an equal-budget or equal-optimizer comparison with Splatfacto",
               "settings": {"seed": 20260909, "iterations": args.iterations, "normal_weight": .05, "distortion_weight": .01,
                            "normal_start": 700, "distortion_start": 300, "refine_stop": 1500, "camera_optimization": False,
                            "external_depth_or_normals": False, "semantic_masks_during_training": document.get("splatlab_evaluation", {}).get("semantic_masks_during_training", False),
                            "source_sparse_depth_weight": args.sparse_depth_weight}, "source_depth_anchors": anchor_metadata,
               "validation": [], "training_depth": [], "progress": []}

    def expired(_signal, _frame):
        raise TimeoutError("The 2DGS pilot exceeded its 11-minute budget")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 660)
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        points, colors = read_sparse_points(dataset / "sparse.ply")
        if len(points) < 4:
            raise ValueError("Surface initialization requires at least four sparse neighbors")
        normalized_points = (points - center) / scale
        distances = cKDTree(normalized_points).query(normalized_points, k=4, workers=1)[0][:, 1:]
        radii = np.maximum(np.sqrt(np.mean(distances ** 2, axis=1)), 1e-6)
        count = len(points)
        initial = {"means": torch.tensor(normalized_points, dtype=torch.float32),
                   "scales": torch.tensor(np.log(radii), dtype=torch.float32)[:, None].repeat(1, 3),
                   "quats": torch.rand(count, 4), "opacities": torch.full((count,), float(torch.logit(torch.tensor(.1)))),
                   "sh0": torch.tensor((colors / 255. - .5) / .28209479177387814, dtype=torch.float32)[:, None],
                   "shN": torch.zeros(count, 3, 3)}
        parameters = torch.nn.ParameterDict({name: torch.nn.Parameter(value.cuda()) for name, value in initial.items()})
        rates = {"means": 1.6e-4, "scales": .005, "quats": .001, "opacities": .05, "sh0": .0025, "shN": .0025 / 20}
        optimizers = {name: torch.optim.Adam([parameters[name]], lr=rate, eps=1e-15) for name, rate in rates.items()}
        strategy = DefaultStrategy(refine_start_iter=100, refine_stop_iter=1500, refine_every=50, reset_every=1000,
                                   key_for_gradient="gradient_2dgs")
        strategy.check_sanity(parameters, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=1.)
        intrinsic = torch.tensor([[document["fl_x"], 0, document["cx"]], [0, document["fl_y"], document["cy"]], [0, 0, 1]],
                                 dtype=torch.float32, device="cuda")[None]
        shape = (document["h"], document["w"])
        cameras = {name: torch.tensor(capture.normalized_camera(frames[name], center, scale), dtype=torch.float32, device="cuda")[None]
                   for name in splits["train"] + splits["val"]}
        photos = {name: capture.load_photo(dataset, frames[name], shape) for name in splits["train"]}

        def render(name, degree):
            return rendering.rasterization_2dgs(means=parameters["means"], quats=parameters["quats"],
                       scales=parameters["scales"].exp(), opacities=parameters["opacities"].sigmoid(),
                       colors=torch.cat((parameters["sh0"], parameters["shN"]), dim=1), viewmats=cameras[name], Ks=intrinsic,
                       width=shape[1], height=shape[0], near_plane=.01 / scale, far_plane=120. / scale,
                       sh_degree=degree, render_mode="RGB+ED", distloss=True, depth_mode="expected")

        order = []
        for step in range(args.iterations):
            if not order:
                order = random.permutation(splits["train"]).tolist()
            name = order.pop()
            pixels, mask = photos[name]
            target = torch.from_numpy(pixels).cuda().float() / 255.
            valid = torch.from_numpy(mask).cuda()
            rendered, alpha, normals, surface_normals, distortion, _median, info = render(name, min(step // 1000, 1))
            strategy.step_pre_backward(parameters, optimizers, strategy_state, step, info)
            predicted = rendered[0, ..., :3].clamp(0, 1)
            image_loss = .8 * torch.abs(predicted[valid] - target[valid]).mean()
            image_loss += .2 * (1 - structural_similarity_index_measure((predicted * valid[..., None]).permute(2, 0, 1)[None],
                                       (target * valid[..., None]).permute(2, 0, 1)[None], data_range=1.))
            normal_error = 1 - (normals[0] * surface_normals * alpha[0].detach()).sum(dim=-1)
            normal_loss = normal_error[valid].mean() * (.05 if step >= 700 else 0.)
            distortion_loss = distortion[0, ..., 0][valid].mean() * (.01 if step >= 300 else 0.)
            loss = image_loss + normal_loss + distortion_loss
            sparse_depth_loss = loss.new_zeros(())
            if name in anchors and len(anchors[name]["depth"]) and step >= 300:
                sample = anchors[name]
                grid = torch.tensor(sample["pixels"] / [shape[1], shape[0]] * 2 - 1, dtype=torch.float32, device="cuda")[None, :, None]
                predicted_depth = torch.nn.functional.grid_sample(rendered[..., 3:4].permute(0, 3, 1, 2), grid,
                                      mode="bilinear", padding_mode="zeros", align_corners=False).flatten()
                target_depth = torch.tensor(sample["depth"] / scale, dtype=torch.float32, device="cuda")
                sparse_depth_loss = torch.abs(predicted_depth.clamp_min(1e-6).log() - target_depth.log()).mean() * args.sparse_depth_weight
                loss = loss + sparse_depth_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite 2DGS loss; no silent replacement")
            loss.backward()
            strategy.step_post_backward(parameters, optimizers, strategy_state, step, info, packed=False)
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            optimizers["means"].param_groups[0]["lr"] = rates["means"] * (.01 ** ((step + 1) / args.iterations))
            if not 1 <= len(parameters["means"]) <= 800000:
                raise ValueError("Surface-splat count exceeded the bounded pilot")
            if step % 250 == 0 or step + 1 == args.iterations:
                progress = {"step": step + 1, "surfels": len(parameters["means"]), "image_loss": float(image_loss.detach()),
                            "normal_loss": float(normal_loss.detach()), "distortion_loss": float(distortion_loss.detach()),
                            "source_depth_loss": float(sparse_depth_loss.detach()),
                            "elapsed_seconds": time.monotonic() - started}
                receipt["progress"].append(progress)
                manifests.atomic_write_json(output / "receipt.json", receipt)
                print(json.dumps(progress), flush=True)
        torch.cuda.synchronize()
        receipt["training_seconds"] = time.monotonic() - started
        saved = {name: value.detach().cpu().numpy() for name, value in parameters.items()}
        np.savez_compressed(output / "surfels.npz", **saved, solver_center=center, solver_scale=scale)
        receipt["gaussians"] = export_splats(output / "splat.ply", capture.preview_parameters(saved, center, scale))
        receipt["gaussians"]["scope"] = "thin 3D preview approximation only; NPZ retains exact 2DGS parameters; metrics use native 2DGS"
        (output / "renders").mkdir()
        (output / "training-depth").mkdir()
        with torch.no_grad():
            for split in ("val", "train"):
                for ordinal, name in enumerate(splits[split]):
                    pixels, mask = photos[name] if split == "train" else capture.load_photo(dataset, frames[name], shape)
                    rendered, alpha, *_auxiliary = render(name, 1)
                    rgb = rendered[0, ..., :3].clamp(0, 1).cpu().numpy()
                    depth = rendered[0, ..., 3].cpu().numpy() * scale
                    opacity = alpha[0, ..., 0].cpu().numpy()
                    record = {"image": name, "width": shape[1], "height": shape[0], "fx": document["fl_x"], "fy": document["fl_y"],
                              "cx": document["cx"], "cy": document["cy"], "camera_to_world_opengl": frames[name]["transform_matrix"][:3]}
                    if split == "val":
                        target = pixels.astype(np.float32) / 255.
                        record.update(image_scores(target, rgb, mask))
                        for kind, values in (("reference", target), ("render", rgb), ("error", np.abs(target - rgb) * 3)):
                            Image.fromarray(np.rint(values * 255).clip(0, 255).astype(np.uint8)).save(output / f"renders/{ordinal:03d}-{kind}.png")
                        receipt["validation"].append(record)
                    else:
                        record["file"] = f"training-depth/{ordinal:03d}.npz"
                        np.savez_compressed(output / record["file"], depth=depth, alpha=opacity, rgb=pixels, mask=mask)
                        receipt["training_depth"].append(record)
        capture.verify_snapshot(snapshot)
        receipt.update(status="evaluated-needs-review", training_iterations=args.iterations, summary=summarize_scores(receipt["validation"]),
                       max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), initial_sfm_points=count,
                       packages={name: importlib.metadata.version(name) for name in ("torch", "gsplat", "numpy")},
                       geometry_accuracy="not measured; native 2DGS expected depth is inferred, not surveyed")
        page = review_html(receipt).replace("Download full Gaussian splat", "Download thin 3D preview approximation")
        page = page.replace("<h1>Your condo, reconstructed from raw video.</h1>",
                            "<h1>Condo 2D surface-splat experiment</h1><p>Native 2DGS renders with depth-distortion and normal consistency. Separate pilot, not a matched-budget benchmark. Downloaded PLY approximates the 2D surfaces for a conventional 3D splat viewer.</p>")
        (output / "index.html").write_text(page)
        receipt["files"] = {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.rglob("*")
                            if path.is_file() and path.name != "receipt.json"}
    except BaseException as error:
        receipt.update(status="failed-not-ready", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt.update(finished_at=manifests.utc_now(), elapsed_seconds=time.monotonic() - started)
        manifests.atomic_write_json(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "summary", "training_seconds", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
