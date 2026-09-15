#!/usr/bin/env python3
"""Estimate a small training-only fisheye model; not a calibrated rig or final reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests


def verified_inputs(pilot_root: Path, mask_root: Path) -> dict:
    manifest_path = pilot_root / "pilot.json"
    document = json.loads(manifest_path.read_text())
    if (document.get("schema") != "dev.splatlab.dual-fisheye-pilot/v1"
            or document.get("status") != "decoded-needs-camera-and-mask-review"
            or document.get("clock", {}).get("paired_container_pts_match") is not True):
        raise ValueError("Select a completed, paired raw-lens pilot")
    groups = document["groups"]
    if not 5 <= len(groups) <= 96 or len({group["group_id"] for group in groups}) != len(groups):
        raise ValueError("Invalid bounded timestamp groups")
    groups_by_id = {group["group_id"]: group for group in groups}
    if {group["split"] for group in groups} != {"train", "val", "test"}:
        raise ValueError("Keep nonempty train, validation and test timestamp groups")
    views = document["views"]
    if len(views) != len(groups) * 2 or len({view["image"] for view in views}) != len(views):
        raise ValueError("Each timestamp group must have exactly two unique views")
    masks = []
    for view in views:
        name = view["image"]
        if not re.fullmatch(r"lens-[01]/frame-\d{6}\.jpg", name):
            raise ValueError("Invalid pilot image path")
        group = groups_by_id.get(view["group_id"])
        if (not group or view["split"] != group["split"] or view["lens_id"] != name.split("/")[0]
                or Path(name).stem != group["group_id"]
                or view["source_decoded_frame_index"] != group["source_decoded_frame_index"]):
            raise ValueError("Image lineage or paired split changed")
        path = pilot_root / "images" / name
        if path.is_symlink() or manifests.sha256_file(path) != view["sha256"]:
            raise ValueError("Pilot image changed")
        with Image.open(path) as picture:
            size = picture.size
            if size != (document["output_width"], document["output_width"]):
                raise ValueError("Pilot image dimensions changed")
        if view["split"] == "train":
            mask_path = mask_root / (name + ".png")
            with Image.open(mask_path) as mask:
                if mask.mode != "L" or mask.size != size or mask.getextrema()[1] == 0:
                    raise ValueError("Each training image needs a nonempty grayscale mask of matching dimensions")
            masks.append({"image": name, "mask": name + ".png", **manifests.file_identity(mask_path)})
    for group in groups:
        pair = [view for view in views if view["group_id"] == group["group_id"]]
        if len(pair) != 2 or {view["lens_id"] for view in pair} != {"lens-0", "lens-1"}:
            raise ValueError("Missing physical lens in timestamp group")
        if pair[0]["pts"] * Fraction(pair[0]["time_base"]) != pair[1]["pts"] * Fraction(pair[1]["time_base"]):
            raise ValueError("Paired presentation timestamps changed")
    train_names = [view["image"] for view in views if view["split"] == "train"]
    return {"pilot_sha256": manifests.sha256_file(manifest_path), "document": document,
            "train_names": train_names, "masks": masks}


def model_summary(reconstruction, views: list[dict]) -> dict:
    names = sorted(reconstruction.images[image_id].name for image_id in reconstruction.reg_image_ids())
    training = {view["image"]: view for view in views if view["split"] == "train"}
    if not set(names) <= training.keys():
        raise ValueError("Held-out images unexpectedly entered the training reconstruction")
    paired = Counter(training[name]["group_id"] for name in names)
    return {"registered_images": len(names), "image_names": names,
            "registered_by_lens": dict(Counter(training[name]["lens_id"] for name in names)),
            "same_timestamp_both_lenses_registered": sum(count == 2 for count in paired.values()),
            "points3d": reconstruction.num_points3D(),
            "mean_track_length": reconstruction.compute_mean_track_length(),
            "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
            "cameras": [{"camera_id": camera_id, "model": camera.model.name,
                         "width": camera.width, "height": camera.height, "params": camera.params.tolist()}
                        for camera_id, camera in reconstruction.cameras.items()]}


def run(pilot_root: Path, mask_root: Path, output: Path, focal_ratio: float) -> dict:
    if not math.isfinite(focal_ratio) or not 0.2 <= focal_ratio <= 0.5:
        raise ValueError("Focal initializer ratio must be 0.2–0.5; it is not measured calibration")
    pilot_root, mask_root, output = pilot_root.resolve(), mask_root.resolve(), output.resolve()
    if output.exists() or output == pilot_root or pilot_root in output.parents or mask_root in output.parents:
        raise ValueError("Choose a new output directory outside the frozen input roots")
    inputs = verified_inputs(pilot_root, mask_root)
    import pycolmap

    if not pycolmap.__version__.startswith("4.1."):
        raise ValueError("This probe targets installed pycolmap 4.1; review API compatibility before another version")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    width = inputs["document"]["output_width"]
    focal = width * focal_ratio
    reader = pycolmap.ImageReaderOptions(camera_model="OPENCV_FISHEYE",
        camera_params=f"{focal},{focal},{width / 2},{width / 2},0,0,0,0", mask_path=mask_root)
    extraction = pycolmap.FeatureExtractionOptions(num_threads=4, use_gpu=False, max_image_size=width)
    extraction.sift.max_num_features = 4096
    extraction.sift.first_octave = 0
    matching = pycolmap.FeatureMatchingOptions(num_threads=4, use_gpu=False)
    mapping = pycolmap.IncrementalPipelineOptions(num_threads=4, ba_use_gpu=False, random_seed=0,
        max_num_models=4, min_model_size=3, init_num_trials=50, image_names=inputs["train_names"])
    receipt = {"schema": "dev.splatlab.dual-fisheye-sfm-probe/v1", "status": "running",
               "started_at": manifests.utc_now(), "pycolmap_version": pycolmap.__version__,
               "tool_sha256": manifests.sha256_file(Path(__file__)), "python_version": sys.version,
               "pilot_sha256": inputs["pilot_sha256"], "train_images": inputs["train_names"],
               "masks": inputs["masks"], "focal_initializer_ratio": focal_ratio,
               "camera_model": "OPENCV_FISHEYE", "intrinsics_shared": "per-physical-lens-folder",
               "rig_constraints": "none-initial-unconstrained-sfm", "metric_scale": "unknown",
               "calibration": "estimated-not-vendor-calibration", "heldout_used_in_mapping": False,
               "heldout_evaluation": "not-run", "models": [],
               "warnings": ["Training reprojection error is not a held-out appearance or metric-accuracy score.",
                            "Separate components cannot be merged using assumed lens poses or coarse GPS.",
                            "Mask files suppress features only; they are not approved splat-training masks.",
                            "No novel geometry, full Gaussian training or indoor Goal acceptance is claimed."]}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        database = output / "database.db"
        pycolmap.extract_features(database, pilot_root / "images", image_names=inputs["train_names"],
            camera_mode=pycolmap.CameraMode.PER_FOLDER, reader_options=reader,
            extraction_options=extraction, device=pycolmap.Device.cpu)
        pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
        sparse = output / "sparse"
        sparse.mkdir()
        models = pycolmap.incremental_mapping(database, pilot_root / "images", sparse, options=mapping)
        for model_id, reconstruction in models.items():
            summary = model_summary(reconstruction, inputs["document"]["views"])
            model_path = sparse / str(model_id)
            model_path.mkdir(exist_ok=True)
            reconstruction.write(model_path)
            reconstruction.write_text(model_path)
            reconstruction.export_PLY(model_path / "points.ply")
            receipt["models"].append({"model_id": model_id, **summary})
        current = verified_inputs(pilot_root, mask_root)
        if (current["pilot_sha256"] != inputs["pilot_sha256"] or current["masks"] != inputs["masks"]):
            raise ValueError("Pilot inputs changed during camera estimation")
        receipt["status"] = "estimated-needs-rig-and-heldout-review" if models else "no-training-model"
    except Exception as error:
        receipt["status"] = "failed-not-accepted"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["seconds"] = round(time.monotonic() - started, 3)
        receipt["finished_at"] = manifests.utc_now()
        manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--focal-ratio", type=float, default=0.3)
    args = parser.parse_args()
    result = run(args.pilot, args.mask_root, args.output, args.focal_ratio)
    print(json.dumps({"status": result["status"], "models": result["models"],
                      "seconds": result["seconds"]}, indent=2))
    return 0 if result["models"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
