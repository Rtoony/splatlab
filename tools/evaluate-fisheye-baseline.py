#!/usr/bin/env python3
"""Export a locally trusted checkpoint and evaluate only its registered validation split."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
from reference_delivery import artifact_path
from fisheye_evaluation import image_scores, summarize_scores, review_html
from reconstruction_review import export_splats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    args = parser.parse_args()
    gate = ROOT / "tools/splatlab-compute-gate.sh"
    if subprocess.run([str(gate), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Use tools/splatlab-compute-gate.sh --run")
    lease = json.loads(args.lease.read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(lease["no_new_starts_after"]):
        raise RuntimeError("No new starts in the compute window expiry buffer")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(dataset):
        raise ValueError("Choose a new output outside the frozen dataset")
    source = json.loads((dataset / "receipt.json").read_text())
    snapshot = {dataset / name: digest for name, digest in source["files"].items()}
    for name in source["files"]:
        artifact_path(dataset, name)
    snapshot.update({dataset / "receipt.json": manifests.sha256_file(dataset / "receipt.json"),
                     args.config.resolve(): args.config_sha256, args.checkpoint.resolve(): args.checkpoint_sha256})

    def verify():
        if any(manifests.sha256_file(path) != digest for path, digest in snapshot.items()):
            raise ValueError("Frozen training inputs, local config or checkpoint changed")

    verify()
    document = json.loads((dataset / "transforms.json").read_text())
    splits = {name: document[name + "_filenames"] for name in ("train", "val", "test")}
    flattened = [name for names in splits.values() for name in names]
    if len(flattened) != len(set(flattened)) or set(flattened) != {frame["file_path"] for frame in document["frames"]}:
        raise ValueError("Explicit training, validation and test splits must be disjoint and complete")
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    from PIL import Image
    import torch
    from nerfstudio.utils.eval_utils import eval_setup

    torch.set_num_threads(4)
    started = time.monotonic()
    output.mkdir(parents=True)
    receipt = {"schema": "dev.splatlab.fisheye-baseline-evaluation/v1", "status": "running",
               "started_at": manifests.utc_now(), "metric_scale": "unknown", "registration": None,
               "source_hashes": {str(path): digest for path, digest in snapshot.items()},
               "checkpoint_load_policy": "weights_only=True, scoped NumPy scalar/dtype allowlist; no unrestricted fallback",
               "config_policy": "explicitly hash-pinned local training YAML; never use untrusted downloaded YAML",
               "test_split_evaluated": False, "owner_accepted": False}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        def configure(config):
            if (Path(config.data).resolve() != dataset or config.pipeline.model.camera_optimizer.mode != "off"
                    or config.pipeline.datamanager.dataparser.auto_scale_poses
                    or config.pipeline.datamanager.dataparser.center_method != "none"
                    or config.pipeline.datamanager.dataparser.orientation_method != "none"
                    or (config.get_checkpoint_dir() / args.checkpoint.name).resolve() != args.checkpoint.resolve()):
                raise ValueError("Evaluation must retain the exact fixed-camera dataset and coordinate frame")
            config.load_step = int(args.checkpoint.stem.split("-")[1])
            config.pipeline.datamanager.cache_images = "cpu"
            config.pipeline.datamanager.max_thread_workers = 4
            return config

        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
            config, pipeline, checkpoint, step = eval_setup(args.config, test_mode="val", update_config_callback=configure)
        if checkpoint.resolve() != args.checkpoint.resolve() or step + 1 != config.max_num_iterations:
            raise ValueError("Require the completed local training checkpoint")
        manager = pipeline.datamanager
        actual = [str(path.resolve().relative_to(dataset)) for path in manager.eval_dataset.image_filenames]
        actual_train = [str(path.resolve().relative_to(dataset)) for path in manager.train_dataset.image_filenames]
        if sorted(actual) != sorted(splits["val"]) or sorted(actual_train) != sorted(splits["train"]):
            raise ValueError("Installed dataparser does not preserve the explicit split")
        if not np.allclose(manager.train_dataparser_outputs.dataparser_transform.cpu().numpy(), np.eye(4)[:3], atol=1e-7) or manager.train_dataparser_outputs.dataparser_scale != 1:
            raise ValueError("Unexpected world reorientation or scale")
        parameters = {name: value.detach().cpu().numpy() for name, value in pipeline.model.gauss_params.items()}
        receipt["gaussians"] = export_splats(output / "splat.ply", parameters)
        receipt["gaussians"]["coordinate_scope"] = "unchanged raw training SfM frame; arbitrary units and unverified up"
        receipt.update(training_iterations=step + 1, validation=[], training_depth=[])
        rendering = output / "renders"
        rendering.mkdir()
        training = output / "training-depth"
        training.mkdir()
        with torch.no_grad():
            for split, filenames, selected_dataset, batches in (
                ("val", actual, manager.eval_dataset, manager.cached_eval),
                ("train", actual_train, manager.train_dataset, manager.cached_train),
            ):
                for index, filename in enumerate(filenames):
                    camera = selected_dataset.cameras[index:index + 1].to("cuda")
                    rendered = pipeline.model.get_outputs_for_camera(camera)
                    rgb = rendered["rgb"].detach().cpu().numpy()
                    depth = rendered["depth"].detach().cpu().numpy().squeeze(-1)
                    alpha = rendered["accumulation"].detach().cpu().numpy().squeeze(-1)
                    batch = batches[index]
                    target = batch["image"].cpu().numpy()
                    if target.dtype == np.uint8:
                        target = target.astype(np.float32) / 255
                    mask = batch["mask"].cpu().numpy().reshape(depth.shape).astype(bool)
                    if not np.isfinite(depth).all() or not np.isfinite(alpha).all():
                        raise ValueError("Nonfinite depth or opacity")
                    record = {"image": filename, "camera_to_world_opengl": camera.camera_to_worlds[0].cpu().numpy().tolist(),
                              "width": int(camera.width.item()), "height": int(camera.height.item()),
                              "fx": float(camera.fx.item()), "fy": float(camera.fy.item()),
                              "cx": float(camera.cx.item()), "cy": float(camera.cy.item())}
                    if split == "val":
                        record.update(image_scores(target, rgb, mask))
                        for kind, pixels in (("reference", target), ("render", rgb), ("error", np.abs(target - rgb) * 3)):
                            Image.fromarray(np.rint(pixels * 255).clip(0, 255).astype(np.uint8)).save(rendering / f"{index:03d}-{kind}.png")
                        receipt["validation"].append(record)
                    else:
                        np.savez_compressed(training / f"{index:03d}.npz", depth=depth, alpha=alpha, mask=mask,
                                            rgb=np.rint(target * 255).clip(0, 255).astype(np.uint8))
                        record["file"] = f"training-depth/{index:03d}.npz"
                        receipt["training_depth"].append(record)
                    if index % 12 == 0 or index + 1 == len(filenames):
                        print(f"{split}: rendered {index + 1}/{len(filenames)}", flush=True)
        verify()
        receipt.update(status="evaluated-needs-review", summary=summarize_scores(receipt["validation"]),
                       elapsed_seconds=time.monotonic() - started,
                       geometry_accuracy="not measured; expected splat depth is an inferred surface hint only")
        (output / "index.html").write_text(review_html(receipt))
        receipt["files"] = {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.rglob("*")
                            if path.is_file() and path.name != "receipt.json"}
    except BaseException as error:
        receipt.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        receipt["finished_at"] = manifests.utc_now()
        manifests.atomic_write_json(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "summary", "training_iterations", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
