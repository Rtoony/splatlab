"""Join source-bound held-out poses to rectified images for a bounded training experiment."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import artifact_manifest as manifests
from fisheye_reference import read_cameras, remap_grid, yaw_rotation
from reference_delivery import artifact_path, validate_camera, verify_reference


def heldout_frames(cameras: dict, reference_receipt: dict, localization: dict) -> dict[str, dict]:
    groups, images = {}, {}
    for frame in cameras["frames"]:
        identity = frame["split"], frame["source_group"], frame["physical_lens"]
        if groups.setdefault(frame["source_group"], frame["split"]) != frame["split"]:
            raise ValueError("A source timestamp cannot cross training and held-out splits")
        if images.setdefault(frame["source_image"], identity) != identity:
            raise ValueError("Virtual crops must agree on their physical source identity")
    if (localization.get("schema") != "dev.splatlab.frozen-holdout-localization/v1"
            or localization.get("status") != "localized-needs-review"
            or localization.get("geometry_refined") is not False
            or localization.get("intrinsics_refined") is not False):
        raise ValueError("Need successful pose-only localization against frozen training geometry")
    if any(localization.get("source_hashes", {}).get(name) != digest
           for name, digest in reference_receipt["source_hashes"].items()):
        raise ValueError("Held-out poses belong to a different frozen source/model")
    expected = {frame["source_image"]: frame for frame in cameras["frames"] if frame["split"] != "train"}
    records = {record["image"]: record for record in localization["poses"]}
    if len(records) != len(localization["poses"]) or set(records) != set(expected):
        raise ValueError("Held-out image set is incomplete, duplicated or includes training images")
    converted = {}
    for name, record in records.items():
        frame = expected[name]
        quality = np.array([record.get("supported_inlier_fraction", 0), record.get("p95_inlier_reprojection_px", math.inf)])
        if (record.get("status") != "localized-needs-review" or record["split"] != frame["split"]
                or record["group_id"] != frame["source_group"] or record["physical_lens"] != frame["physical_lens"]
                or type(record.get("supported_cone_inliers")) is not int
                or record["supported_cone_inliers"] < 20 or not np.isfinite(quality).all()
                or not .25 <= quality[0] <= 1 or not 0 <= quality[1] <= 4):
            raise ValueError("Held-out identity or pose-fit evidence is invalid")
        converted[name] = record
        validate_camera({"transform_matrix": record["camera_to_world_opencv"]})
    return converted


def virtual_pose(record: dict, yaw: float) -> list:
    transform = np.array(record["camera_to_world_opencv"], dtype=float)
    transform[:3, :3] = transform[:3, :3] @ yaw_rotation(yaw) @ np.diag([1, -1, -1])
    return transform.tolist()


def raw_mask_inputs(review_path: Path, cameras: dict, source_receipt: dict) -> tuple[dict, dict]:
    review_path = review_path.resolve()
    receipt = json.loads(review_path.read_text())
    if (receipt.get("schema") != "dev.splatlab.raw-lens-exclusions/v1"
            or receipt.get("status") != "prepared-needs-visual-review"
            or receipt.get("pilot_sha256") != source_receipt["source_hashes"]["pilot"]
            or receipt.get("semantic_ground_truth") is not False):
        raise ValueError("Raw exclusions must belong to this exact source pilot")
    records = {record["image"]: record for record in receipt["views"]}
    names = {frame["source_image"] for frame in cameras["frames"]}
    if len(records) != len(receipt["views"]) or set(records) != names:
        raise ValueError("Raw exclusions must cover each physical source exactly once")
    masks, snapshot = {}, {review_path: manifests.sha256_file(review_path)}
    for frame in cameras["frames"]:
        name = frame["source_image"]
        record = records[name]
        if record["split"] != frame["split"] or record["source_sha256"] != source_receipt["source_hashes"]["image/" + name]:
            raise ValueError("Raw exclusion image or split lineage changed")
        if name in masks:
            continue
        filename = artifact_path(review_path.parent, record["mask"])
        if manifests.sha256_file(filename) != record["mask_sha256"]:
            raise ValueError("Raw exclusion mask changed")
        with Image.open(filename) as image:
            values = np.asarray(image)
            if image.mode != "L" or image.size != (receipt["width"], receipt["width"]) or not np.isin(values, [0, 255]).all():
                raise ValueError("Raw exclusion must be a binary grayscale mask")
            masks[name] = values == 255
        snapshot[filename] = record["mask_sha256"]
    return masks, snapshot


def sample_raw_mask(mask: np.ndarray, samples: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if mask.ndim != 2 or mask.dtype != bool or samples.shape != (*valid.shape, 2) or valid.dtype != bool:
        raise ValueError("Expected boolean support and a matching rectification grid")
    horizontal = np.rint(np.clip(samples[..., 0], 0, mask.shape[1] - 1)).astype(int)
    vertical = np.rint(np.clip(samples[..., 1], 0, mask.shape[0] - 1)).astype(int)
    return valid & mask[vertical, horizontal]


def prepare(reference: Path, localization_path: Path, sfm: Path, output: Path, raw_mask_review: Path | None = None) -> dict:
    reference, localization_path, sfm, output = reference.resolve(), localization_path.resolve(), sfm.resolve(), output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in (reference, sfm, localization_path.parent)):
        raise ValueError("Choose a new training-input directory outside frozen sources")
    cameras, source_receipt, snapshot = verify_reference(reference)
    localization_hash = manifests.sha256_file(localization_path)
    localization = json.loads(localization_path.read_text())
    records = heldout_frames(cameras, source_receipt, localization)
    camera_file = sfm / "sparse/0/cameras.txt"
    if manifests.sha256_file(camera_file) != source_receipt["source_hashes"]["model/cameras.txt"]:
        raise ValueError("Raw fisheye intrinsics changed")
    raw_cameras = read_cameras(camera_file)
    raw_masks, mask_snapshot = raw_mask_inputs(raw_mask_review, cameras, source_receipt) if raw_mask_review else ({}, {})
    fov = math.degrees(2 * math.atan(cameras["w"] / (2 * cameras["fl_x"])))
    grids = {}
    output.mkdir(parents=True, exist_ok=False)
    frames, splits, contacts, mask_fractions = [], {"train": [], "val": [], "test": []}, [], []
    receipt = {"schema": "dev.splatlab.fisheye-training-input/v1", "status": "preparing-not-ready",
               "source_reference_sha256": snapshot["receipt.json"], "localization_sha256": localization_hash,
               "started_at": manifests.utc_now(), "tool_sha256": manifests.sha256_file(Path(__file__)),
               "mask_recipe": {"raw_lens_radius_fraction": .43, "raw_y_exclude_at_fraction": .76,
                               "basis": "experimental geometric rim/nadir exclusion, not semantic masking"},
               "mask_review": "visual-review-required", "metric_scale": "unknown", "registration": None,
               "appearance_evaluated": False, "limitations": [
                   "Masking may exclude valid background and retain moving traffic, foliage, reflection or shadow.",
                   "All crops of a timestamp stay in their original split. Validation and test poses are fitted against frozen training tracks.",
                   "Pose-fit residuals are not held-out rendered-image scores. This dataset does not establish metric or architectural accuracy."]}
    if raw_mask_review:
        receipt["raw_mask_source_hashes"] = {str(filename): digest for filename, digest in mask_snapshot.items()}
        receipt["mask_recipe"]["additional_exclusions"] = "source-bound raw person/hand/sky predictions, nearest-pixel rectification; predictions are not ground truth"
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        for frame in cameras["frames"]:
            copied = {key: frame[key] for key in ("file_path", "mask_path", "transform_matrix")}
            if frame["split"] != "train":
                copied["transform_matrix"] = virtual_pose(records[frame["source_image"]], frame["virtual_yaw_deg"])
            validate_camera(copied)
            raw = raw_cameras[frame["estimated_camera_id"]]
            key = frame["estimated_camera_id"], frame["virtual_yaw_deg"]
            if key not in grids:
                samples, valid, _, _ = remap_grid(raw, cameras["w"], fov, frame["virtual_yaw_deg"])
                radial = np.hypot(samples[..., 0] + .5 - raw["width"] / 2, samples[..., 1] + .5 - raw["height"] / 2)
                grids[key] = samples, valid & (radial <= .43 * raw["width"]) & (samples[..., 1] + .5 < .76 * raw["height"])
            samples, mask = grids[key]
            if raw_masks:
                raw_mask = raw_masks[frame["source_image"]]
                if raw_mask.shape != (raw["height"], raw["width"]):
                    raise ValueError("Raw exclusion dimensions do not match the estimated lens camera")
                mask = sample_raw_mask(raw_mask, samples, mask)
            if not mask.any():
                raise ValueError("No supported pixels remain in a rectified view")
            mask_fractions.append(float(mask.mean()))
            image_path, mask_path = output / copied["file_path"], output / copied["mask_path"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(reference / copied["file_path"], image_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            frames.append(copied)
            splits[frame["split"]].append(copied["file_path"])
            if frame["split"] == "train" and frame["virtual_yaw_deg"] == 0:
                with Image.open(image_path) as image:
                    masked = np.asarray(image.convert("RGB")).copy()
                    masked[~mask] = 0
                    tile = Image.fromarray(masked).resize((240, 240))
                contacts.append((tile, frame["source_image"]))
        document = {key: cameras[key] for key in ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2")}
        document.update(frames=frames, **{split + "_filenames": names for split, names in splits.items()},
                        ply_file_path="sparse.ply", orientation_override="none", applied_transform=np.eye(4)[:3].tolist(),
                        splatlab_evaluation={"split_unit": "paired-source-timestamp", "source_reference_sha256": snapshot["receipt.json"],
                                            "localization_sha256": localization_hash, "metric_scale": "unknown",
                                            "semantic_masks_during_training": bool(raw_masks)})
        shutil.copyfile(reference / "sparse.ply", output / "sparse.ply")
        manifests.atomic_write_json(output / "transforms.json", document)
        sheet = Image.new("RGB", (1200, math.ceil(len(contacts) / 5) * 265), "#20252d")
        draw = ImageDraw.Draw(sheet)
        for index, (tile, name) in enumerate(contacts):
            left, top = index % 5 * 240, index // 5 * 265
            sheet.paste(tile, (left, top))
            draw.text((left + 3, top + 242), name, fill="white")
        sheet.save(output / "training-mask-review.png")
        if snapshot != {name: manifests.sha256_file(reference / name) for name in snapshot}:
            raise ValueError("Source bundle changed while preparing training input")
        if manifests.sha256_file(localization_path) != localization_hash:
            raise ValueError("Held-out localization evidence changed")
        if any(manifests.sha256_file(filename) != digest for filename, digest in mask_snapshot.items()):
            raise ValueError("Raw exclusions changed during rectification")
        receipt.update(status="prepared-needs-visual-mask-review", splits={key: len(value) for key, value in splits.items()},
                       min_mask_fraction=min(mask_fractions), max_mask_fraction=max(mask_fractions),
                       files={str(path.relative_to(output)): manifests.sha256_file(path)
                              for path in output.rglob("*") if path.is_file() and path.name != "receipt.json"})
    except Exception as error:
        receipt.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        receipt["finished_at"] = manifests.utc_now()
        manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt
