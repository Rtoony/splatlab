#!/usr/bin/env python3
"""Reopen native 2DGS parameters, verify saved views, and render a source-camera tour."""

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests
import capture_surfels as capture
from reference_delivery import artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "dataset", "output", "lease"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if subprocess.run([str(ROOT / "tools/splatlab-compute-gate.sh"), "--is-contained"], capture_output=True).returncode:
        raise ValueError("Use the shared compute gate")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(json.loads(args.lease.read_text())["no_new_starts_after"]):
        raise ValueError("No new starts in the window's expiry buffer")
    evaluation, dataset, output = args.evaluation.resolve(), args.dataset.resolve(), args.output.resolve()
    if output.exists() or any(output.is_relative_to(path) for path in (evaluation, dataset)):
        raise ValueError("Choose a new tour directory outside its sources")
    receipt = json.loads((evaluation / "receipt.json").read_text())
    if receipt.get("method") != "gsplat-2dgs" or receipt.get("status") != "evaluated-needs-review":
        raise ValueError("Require a completed native 2DGS model")
    document, frames, splits, snapshot, center, scale = capture.frozen_dataset(dataset)
    if any(receipt["source_hashes"].get(str(path)) != digest for path, digest in snapshot.items()):
        raise ValueError("Tour dataset differs from the trained source")
    snapshot[evaluation / "receipt.json"] = manifests.sha256_file(evaluation / "receipt.json")
    for name, digest in receipt["files"].items():
        snapshot[artifact_path(evaluation, name)] = digest
    capture.verify_snapshot(snapshot)
    import numpy as np
    from PIL import Image
    from scipy.spatial.transform import Rotation, Slerp
    import torch
    from gsplat import rendering

    implementation = Path(inspect.getfile(rendering))
    if manifests.sha256_file(implementation) != receipt["source_hashes"].get(str(implementation)):
        raise ValueError("Native rendering implementation differs from training")
    snapshot[implementation] = manifests.sha256_file(implementation)
    snapshot[Path(__file__)] = manifests.sha256_file(Path(__file__))
    torch.set_num_threads(4)
    with np.load(evaluation / "surfels.npz", allow_pickle=False) as stored:
        if not np.array_equal(stored["solver_center"], center) or float(stored["solver_scale"]) != scale:
            raise ValueError("Saved native solver coordinates changed")
        saved = {name: stored[name].copy() for name in ("means", "scales", "quats", "opacities", "sh0", "shN")}
    capture.preview_parameters(saved, center, scale)
    parameters = {name: torch.tensor(value, device="cuda") for name, value in saved.items()}
    intrinsic = torch.tensor([[document["fl_x"], 0, document["cx"]], [0, document["fl_y"], document["cy"]], [0, 0, 1]],
                             dtype=torch.float32, device="cuda")[None]
    width, height = document["w"], document["h"]

    @torch.no_grad()
    def render(pose):
        matrix = capture.normalized_camera({"transform_matrix": pose}, center, scale)
        values, *_others = rendering.rasterization_2dgs(means=parameters["means"], quats=parameters["quats"],
            scales=parameters["scales"].exp(), opacities=parameters["opacities"].sigmoid(),
            colors=torch.cat((parameters["sh0"], parameters["shN"]), dim=1),
            viewmats=torch.tensor(matrix, dtype=torch.float32, device="cuda")[None], Ks=intrinsic,
            width=width, height=height, near_plane=.01 / scale, far_plane=120. / scale,
            sh_degree=1, render_mode="RGB+ED", distloss=True, depth_mode="expected")
        return np.rint(values[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    output.mkdir(parents=True)
    started = time.monotonic()
    result = {"status": "running", "schema": "dev.splatlab.native-surfel-tour/v1", "registration": None,
              "source_hashes": {str(path): digest for path, digest in snapshot.items()}, "replayed_views": [],
              "scope": "Native model renders from reopened NPZ; no original video/photo pixels used to construct the tour",
              "owner_accepted": False, "test_images_read": False}
    manifests.atomic_write_json(output / "receipt.json", result)
    encoder = None
    try:
        for ordinal, record in enumerate(receipt["validation"]):
            pixels = render(frames[record["image"]]["transform_matrix"])
            with Image.open(evaluation / f"renders/{ordinal:03d}-render.png") as source:
                previous = np.asarray(source.convert("RGB"))
            difference = np.abs(pixels.astype(np.int16) - previous.astype(np.int16))
            if difference.max() > 1 or difference.mean() > .1:
                raise ValueError("Native saved-parameter replay does not reproduce evaluated views")
            result["replayed_views"].append({"image": record["image"], "maximum_channel_error_255": int(difference.max()),
                                             "mean_channel_error_255": float(difference.mean())})
        selected = sorted((name for name in splits["train"] if re.fullmatch(r"train/lens-0-frame-\d+-centre\.png", name)),
                          key=lambda name: int(re.search(r"frame-(\d+)", name)[1]))
        if len(selected) < 4:
            raise ValueError("Insufficient same-physical-lens training centers for a tour")
        times = np.array([int(re.search(r"frame-(\d+)", name)[1]) for name in selected], dtype=float)
        poses = np.array([frames[name]["transform_matrix"] for name in selected])
        interpolation = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
        samples = np.linspace(times[0], times[-1], 120)
        rotations = interpolation(samples).as_matrix()
        positions = np.column_stack([np.interp(samples, times, poses[:, coordinate, 3]) for coordinate in range(3)])
        result["tour"] = {"frames": 120, "fps": 15, "seconds": 8, "width": width, "height": height,
                          "source_training_cameras": selected, "interpolation": "linear centers and quaternion SLERP; training-pose path, no world-up or metric assumption"}
        with (output / "ffmpeg.log").open("wb") as log:
            encoder = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", "15", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "4", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output / "native-camera-tour.mp4")], stdin=subprocess.PIPE, stdout=log, stderr=log)
            for ordinal, (rotation, position) in enumerate(zip(rotations, positions)):
                pose = np.eye(4)
                pose[:3, :3], pose[:3, 3] = rotation, position
                pixels = render(pose)
                encoder.stdin.write(pixels.tobytes())
                if ordinal in (0, 30, 60, 90, 119):
                    Image.fromarray(pixels).save(output / f"tour-{ordinal:03d}.png")
            encoder.stdin.close()
            if encoder.wait(timeout=60):
                raise ValueError("Native tour video encoding failed")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(output / "native-camera-tour.mp4"), "-an",
                            "-c:v", "libvpx-vp9", "-threads", "4", "-deadline", "realtime", "-cpu-used", "6",
                            "-b:v", "0", "-crf", "28", str(output / "native-camera-tour.webm")],
                           check=True, stdout=log, stderr=log, timeout=60)
        capture.verify_snapshot(snapshot)
        result["status"] = "reopened-and-rendered-needs-review"
        result["files"] = {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.iterdir() if path.name != "receipt.json"}
    except BaseException as error:
        result.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait(timeout=10)
        result.update(elapsed_seconds=time.monotonic() - started, finished_at=manifests.utc_now())
        manifests.atomic_write_json(output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "replayed_views": len(result["replayed_views"]), "tour": result["tour"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
