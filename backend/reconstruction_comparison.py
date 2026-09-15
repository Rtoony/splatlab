"""Frozen RGB reconstruction comparisons with explicit photo/timestamp holdouts."""

from copy import deepcopy
import hashlib
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image

import artifact_manifest as manifests
import background_recovery as recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes

ROOT = Path(__file__).resolve().parents[1] / "data/spatial/reconstruction-comparisons"


def held_out_frames(manager, dataset, expected):
    actual = [str(path.relative_to(dataset)) for path in manager.eval_dataset.image_filenames]
    if set(actual) != set(expected) or len(actual) != len(expected) or not actual:
        raise evidence.EvidenceError("Actual evaluation images do not match the registered held-out split")
    batches = manager.cached_eval
    if len(batches) != len(actual):
        raise evidence.EvidenceError("Cached evaluation does not cover every registered image")
    for index, filename in enumerate(actual):
        yield filename, manager.eval_dataset.cameras[index:index + 1], batches[index]


def partition(frames, specified=None):
    names = [frame["file_path"] for frame in frames]
    if len(names) != len(set(names)) or len(names) < 12:
        raise evidence.EvidenceError("Comparison needs at least twelve uniquely named frames")
    groups = {}
    for frame in sorted(frames, key=lambda frame: frame["file_path"]):
        group = str(frame.get("timestamp_group", frame["file_path"]))
        groups.setdefault(group, []).append(frame["file_path"])
    if specified:
        splits = {split: list(specified.get(f"{split}_filenames", [])) for split in ("train", "val", "test")}
    else:
        if len(groups) < 12:
            raise evidence.EvidenceError("Comparison needs twelve independent photo/timestamp groups")
        splits = {split: [] for split in ("train", "val", "test")}
        for index, members in enumerate(groups.values()):
            split = "test" if index % 12 == 0 else "val" if index % 12 == 6 else "train"
            splits[split].extend(members)
    combined = [name for members in splits.values() for name in members]
    if any(not members for members in splits.values()) or len(combined) != len(set(combined)) or set(combined) != set(names):
        raise evidence.EvidenceError("Train/validation/test lists must be nonempty, disjoint and cover all frames")
    ownership = {name: split for split, members in splits.items() for name in members}
    if any(len({ownership[name] for name in members}) != 1 for members in groups.values()):
        raise evidence.EvidenceError("Virtual views from one timestamp cannot cross splits")
    return splits


def seal(output, filename, payload):
    payload = {key: value for key, value in payload.items() if key != "sha256"}
    payload["sha256"] = hashlib.sha256(scenes.canonical_bytes(payload)).hexdigest()
    manifests.atomic_write_json(output / filename, payload)
    return payload


def read(output, filename="receipt.json"):
    receipt = manifests.read_json(output / filename)
    if not receipt or receipt.get("sha256") != hashlib.sha256(scenes.canonical_bytes({key: value for key, value in receipt.items() if key != "sha256"})).hexdigest():
        raise evidence.EvidenceError("Comparison receipt is missing or corrupt")
    return receipt


def summarize(output, names):
    receipt = verify(output)
    if not 2 <= len(names) <= 3 or len(set(names)) != len(names) or any(not re.fullmatch(r"(?:splatfacto|dn-splatter|ags-mesh)(?:-attempt-[0-9]{2})?", name) for name in names):
        raise evidence.EvidenceError("Choose two or three distinct registered arm directories")
    runs = {name: read(output / name, "run.json") for name in names}
    baseline_names = [name for name, run in runs.items() if run["arm"] == "splatfacto"]
    if len(baseline_names) != 1 or len({run["arm"] for run in runs.values()}) != len(runs):
        raise evidence.EvidenceError("Comparison needs one baseline and distinct candidate methods")
    baseline = runs[baseline_names[0]]
    expected = set(receipt["prepared_splits"][receipt["evaluation_split"]])
    for name, run in runs.items():
        if run.get("status") != "completed" or run.get("prepared_receipt_sha256") != receipt["sha256"] or run.get("budget") != receipt["budget"] or run.get("iterations") != receipt["budget"]["iterations_per_arm"]:
            raise evidence.EvidenceError("Failed, stale or unequal-budget runs cannot support a comparison")
        if any(run.get(key) != baseline.get(key) for key in ("initial_positions_sha256", "initial_gaussians", "implementation_sha256", "packages")):
            raise evidence.EvidenceError("Run initialization, harness or runtime versions differ")
        shared_sources = set(run["implementation_sources"]) & set(baseline["implementation_sources"])
        if any(run["implementation_sources"][path]["sha256"] != baseline["implementation_sources"][path]["sha256"] for path in shared_sources):
            raise evidence.EvidenceError("Shared model/runtime source files changed between arms")
        evaluation = run["evaluation"]
        if evaluation["split"] != receipt["evaluation_split"] or len(evaluation["views"]) != len(expected) or {view["image"] for view in evaluation["views"]} != expected:
            raise evidence.EvidenceError("Held-out image membership differs from the frozen split")
        if any(not np.isfinite(view["psnr_db"]) or view["ssim"] is not None and not np.isfinite(view["ssim"]) for view in evaluation["views"]):
            raise evidence.EvidenceError("Comparison metrics must be finite")
        for relative, identity in run["artifacts"].items():
            path = output / name / relative
            if not path.resolve().is_relative_to((output / name).resolve()) or path.is_symlink() or not path.is_file() or manifests.sha256_file(path) != identity["sha256"]:
                raise evidence.EvidenceError("Run artifacts changed after evaluation")
        for filename in expected:
            reference = "renders/" + Path(filename).stem + "-reference.png"
            if reference not in run["artifacts"] or run["artifacts"][reference]["sha256"] != baseline["artifacts"][reference]["sha256"]:
                raise evidence.EvidenceError("Runs did not evaluate identical reference pixels")
    metrics = {}
    for name, run in runs.items():
        views = run["evaluation"]["views"]
        metrics[name] = {"psnr_db": float(np.mean([view["psnr_db"] for view in views])),
                         "ssim": float(np.mean([view["ssim"] for view in views])) if all(view["ssim"] is not None for view in views) else None,
                         "opacity_coverage": float(np.mean([view["covered_fraction_alpha_0_5"] for view in views])) if all(view.get("covered_fraction_alpha_0_5") is not None for view in views) else None,
                         **{key: run[key] for key in ("gaussians", "training_seconds", "total_seconds", "max_cuda_allocated_bytes", "max_rss_bytes")},
                         "checkpoint_bytes": sum(identity["bytes"] for relative, identity in run["artifacts"].items() if relative.endswith(".ckpt"))}
    baseline_metrics = metrics[baseline_names[0]]
    deltas = {name: {key: value - baseline_metrics[key] if value is not None and baseline_metrics[key] is not None else None for key, value in values.items()} for name, values in metrics.items() if name != baseline_names[0]}
    payload = {"schema": "dev.splatlab.reconstruction-comparison-result/v1", "prepared_receipt_sha256": receipt["sha256"],
               "run_sha256": {name: run["sha256"] for name, run in runs.items()}, "baseline": baseline_names[0],
               "metrics": metrics, "candidate_minus_baseline": deltas, "evaluation_split": receipt["evaluation_split"],
               "held_out_views": len(expected), "budget": receipt["budget"], "status": "needs-review", "promoted": False,
               "scope": "paired held-out RGB comparison; opacity is not geometry accuracy and no model default changes",
               "geometry_error": None, "geometry_error_reason": "No independent dense geometric reference supplied"}
    if (output / "comparison.json").exists():
        existing = read(output, "comparison.json")
        if existing["run_sha256"] != payload["run_sha256"]:
            raise evidence.EvidenceError("A different comparison is already retained; do not overwrite it")
        return existing
    return seal(output, "comparison.json", payload)


def verify(output):
    receipt = read(output)
    source = Path(receipt["source_job"])
    recovery.verify_receipt_sources(source, receipt, checksums=True)
    for name, identity in receipt["artifacts"].items():
        path = output / name
        if not path.resolve().is_relative_to(output.resolve()) or path.is_symlink() or not manifests.same_file_identity(path, identity) or manifests.sha256_file(path) != identity["sha256"]:
            raise evidence.EvidenceError("Prepared comparison input is missing or changed")
    return receipt


def prepare(job, iterations=3000, seconds=1200, downscale=2):
    if type(iterations) is not int or not 32 <= iterations <= 30000 or type(seconds) is not int or not 60 <= seconds <= 7200 or type(downscale) is not int or downscale not in {1, 2, 4}:
        raise evidence.EvidenceError("Use 32–30000 iterations, 60–7200 seconds per arm and downscale 1/2/4")
    job = job.resolve()
    with recovery.worker_slot(wait_seconds=3):
        source = evidence.load(job, retain_image_observations=True)
        original = manifests.read_json(job / "processed/transforms.json")
        frames = original["frames"]
        specified = original if any(f"{split}_filenames" in original for split in ("train", "val", "test")) else None
        splits = partition(frames, specified)
        if any("pano-" in frame["file_path"] and "timestamp_group" not in frame for frame in frames) and not specified:
            raise evidence.EvidenceError("Panorama-derived frames need a registered whole-timestamp split")
        identifier = "comparison-" + uuid.uuid4().hex[:24]
        output = ROOT / identifier
        output.mkdir(parents=True)
        dataset = output / "dataset"
        (dataset / "images").mkdir(parents=True)
        train_ids = {frame["colmap_im_id"] for frame in frames if frame["file_path"] in splits["train"]}
        records = source.records
        colors = np.zeros((len(records), 3), dtype=np.float64)
        color_counts = np.zeros(len(records), dtype=np.int32)
        associations = {image_id: [] for image_id in train_ids}
        ambiguous_training_observations = 0
        for point_index, record in enumerate(records):
            if record["error"] > 2:
                continue
            image_ids, counts = np.unique(record["track"][:, 0], return_counts=True)
            ambiguous = set(image_ids[counts > 1])
            for image_id, observation_index in record["track"]:
                if image_id in associations:
                    if image_id in ambiguous:
                        ambiguous_training_observations += 1
                        continue
                    associations[image_id].append((point_index, int(observation_index)))
        mapped = {}
        prepared_frames = []
        has_masks = any("mask_path" in frame for frame in frames)
        if has_masks and not all("mask_path" in frame for frame in frames):
            raise evidence.EvidenceError("Every comparison frame needs a mask when masking is enabled")
        for frame in frames:
            image_id = frame["colmap_im_id"]
            camera = source.cameras[image_id]
            image_path = evidence.input_path(job, "processed/" + frame["file_path"], source.sources)
            with Image.open(image_path) as opened:
                picture = opened.convert("RGB")
                original_width, original_height = picture.size
                width, height = max(1, original_width // downscale), max(1, original_height // downscale)
                image_array = np.asarray(picture)
                mask_array = None
                if has_masks:
                    mask_path = evidence.input_path(job, "processed/" + frame["mask_path"], source.sources)
                    with Image.open(mask_path) as opened_mask:
                        if opened_mask.size != picture.size:
                            raise evidence.EvidenceError("Comparison masks must match source image dimensions")
                        mask_array = np.asarray(opened_mask.convert("L")) > 0
                        (dataset / "masks").mkdir(exist_ok=True)
                        opened_mask.convert("L").resize((width, height), Image.Resampling.NEAREST).save(dataset / "masks" / f"{image_id:06d}.png")
                if image_id in associations and associations[image_id]:
                    pairs = np.asarray(associations[image_id])
                    observations = source.images[image_id]["observations"]
                    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= len(observations)):
                        raise evidence.EvidenceError("Invalid point-track image index")
                    observed = observations[pairs[:, 1]]
                    identifiers = np.asarray([records[index]["id"] for index in pairs[:, 0]])
                    if not np.array_equal(observed["point_id"], identifiers):
                        raise evidence.EvidenceError("Point tracks do not match their source observations")
                    pixels = np.stack([observed["x"], observed["y"]], axis=1)
                    pixels *= np.array([original_width / camera["width"], original_height / camera["height"]])
                    valid = np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < original_width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < original_height)
                    indices = pairs[valid, 0]
                    pixels = pixels[valid].astype(int)
                    if mask_array is not None:
                        mask_valid = mask_array[pixels[:, 1], pixels[:, 0]]
                        indices, pixels = indices[mask_valid], pixels[mask_valid]
                    np.add.at(colors, indices, image_array[pixels[:, 1], pixels[:, 0]])
                    np.add.at(color_counts, indices, 1)
                filename = f"images/{image_id:06d}.png"
                picture.resize((width, height), Image.Resampling.LANCZOS).save(dataset / filename)
            mapped[frame["file_path"]] = filename
            changed = {key: deepcopy(value) for key, value in frame.items() if key not in {"depth_file_path", "mask_path"}}
            changed.update(file_path=filename, w=width, h=height)
            declared_width = frame.get("w", original.get("w"))
            declared_height = frame.get("h", original.get("h"))
            for name in ("fl_x", "cx", "fl_y", "cy"):
                ratio = width / declared_width if name in {"fl_x", "cx"} else height / declared_height
                changed[name] = frame.get(name, original.get(name)) * ratio
            if has_masks:
                changed["mask_path"] = f"masks/{image_id:06d}.png"
            prepared_frames.append(changed)
        keep = color_counts >= 3
        if keep.sum() < 100:
            raise evidence.EvidenceError("Insufficient points with three valid training-photo color observations")
        points = np.asarray([record["xyz"] for record in records])[keep]
        applied = evidence.matrix4(original.get("applied_transform", np.eye(4)))
        points = points @ applied[:3, :3].T + applied[:3, 3]
        vertex = np.empty(int(keep.sum()), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rgb = np.clip(np.rint(colors[keep] / color_counts[keep, None]), 0, 255).astype(np.uint8)
        for column, name in enumerate(("x", "y", "z")):
            vertex[name] = points[:, column]
        for column, name in enumerate(("red", "green", "blue")):
            vertex[name] = rgb[:, column]
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertex)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        (dataset / "train-seeds.ply").write_bytes(header.encode() + vertex.tobytes())
        np.savez_compressed(dataset / "seed-lineage.npz", point_ids=np.asarray([record["id"] for record in records], dtype=np.uint64)[keep], training_color_observations=color_counts[keep], training_image_ids=np.asarray(sorted(train_ids)))
        transformed = {key: deepcopy(value) for key, value in original.items() if key not in {"frames", "w", "h", "fl_x", "fl_y", "cx", "cy", "ply_file_path", "train_filenames", "val_filenames", "test_filenames"}}
        transformed.update(frames=prepared_frames, ply_file_path="train-seeds.ply")
        transformed.update({f"{split}_filenames": [mapped[name] for name in members] for split, members in splits.items()})
        manifests.atomic_write_json(dataset / "transforms.json", transformed)
        recovery.verify_receipt_sources(job, {"sources": source.sources, "calibration": source.calibration}, checksums=True)
        artifacts = {str(path.relative_to(output)): manifests.file_identity(path) for path in sorted(dataset.rglob("*")) if path.is_file()}
        return output, seal(output, "receipt.json", {"schema": "dev.splatlab.reconstruction-comparison/v1", "comparison_id": identifier,
            "source_job": str(job), "sources": source.sources, "calibration": source.calibration, "created_at": manifests.utc_now(),
            "splits": splits, "prepared_splits": {split: transformed[f"{split}_filenames"] for split in splits},
            "seed_points": len(vertex), "seed_color_source": "only valid training-photo tracked observations; minimum three",
            "ambiguous_training_observations_excluded": ambiguous_training_observations,
            "pose_scope": "shared existing SfM geometry/poses may use all source photos; held out from RGB optimization, not an independent pose-solving benchmark",
            "masking": "registered masks retained" if has_masks else "no input masks supplied; static-scene assumption is not motion verification",
            "budget": {"iterations_per_arm": iterations, "seconds_per_arm": seconds, "downscale": downscale, "seed": 42},
            "evaluation_split": "val" if iterations < 1000 else "test", "warm_start": False,
            "scope": "paired RGB reconstruction experiment; no geometric ground-truth or method promotion implied", "artifacts": artifacts})
