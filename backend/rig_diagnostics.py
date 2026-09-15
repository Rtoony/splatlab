"""Read-only relative-pose diagnostics for independently reconstructed dual lenses."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


def rotation(quaternion: list[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or abs(np.linalg.norm(values) - 1) > 1e-3:
        raise ValueError("Expected a finite unit quaternion in COLMAP w,x,y,z order")
    scalar, horizontal, vertical, depth = values / np.linalg.norm(values)
    return np.array([
        [1 - 2 * (vertical**2 + depth**2), 2 * (horizontal*vertical - scalar*depth), 2 * (horizontal*depth + scalar*vertical)],
        [2 * (horizontal*vertical + scalar*depth), 1 - 2 * (horizontal**2 + depth**2), 2 * (vertical*depth - scalar*horizontal)],
        [2 * (horizontal*depth - scalar*vertical), 2 * (vertical*depth + scalar*horizontal), 1 - 2 * (horizontal**2 + vertical**2)],
    ])


def read_poses(path: Path) -> dict[str, dict]:
    result = {}
    with path.open() as source:
        for line in source:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Malformed COLMAP image pose")
            name = fields[9].strip()
            if name in result:
                raise ValueError("Duplicate COLMAP image name")
            matrix = rotation([float(value) for value in fields[1:5]])
            translation = np.array([float(value) for value in fields[5:8]])
            if not np.isfinite(translation).all():
                raise ValueError("Non-finite camera translation")
            result[name] = {"rotation": matrix, "translation": translation,
                            "camera_id": int(fields[8]), "centre": -matrix.T @ translation}
            if next(source, None) is None:
                raise ValueError("Missing COLMAP observations line")
    return result


def angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    cosine = float((np.trace(first @ second.T) - 1) / 2)
    return math.degrees(math.acos(max(-1, min(1, cosine))))


def summarize(poses: dict[str, dict], training_names: list[str]) -> dict:
    if len(set(training_names)) != len(training_names) or set(poses) != set(training_names):
        raise ValueError("Model names must exactly match the frozen training split")
    pairs = {}
    for name in sorted(poses):
        match = re.fullmatch(r"lens-([01])/(frame-\d+)\.jpg", name)
        if not match:
            raise ValueError("Expected paired raw-lens training image names")
        pairs.setdefault(match[2], {})[int(match[1])] = poses[name]
    if len(pairs) < 3 or any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("Need at least three complete two-lens timestamp pairs")
    records = []
    matrices = []
    translations = []
    for group, pair in sorted(pairs.items()):
        first, second = pair[0], pair[1]
        relative_rotation = second["rotation"] @ first["rotation"].T
        relative_translation = second["translation"] - relative_rotation @ first["translation"]
        matrices.append(relative_rotation)
        translations.append(relative_translation)
        records.append({"group": group, "lens1_from_lens0_rotation": relative_rotation.tolist(),
                        "lens1_from_lens0_translation": relative_translation.tolist(),
                        "baseline_arbitrary_units": float(np.linalg.norm(relative_translation)),
                        "centres": [first["centre"].tolist(), second["centre"].tolist()]})
    distances = np.array([[angular_distance(first, second) for second in matrices] for first in matrices])
    medoid_index = int(np.argmin(distances.sum(axis=1)))
    errors = distances[medoid_index]
    baseline = np.array([record["baseline_arbitrary_units"] for record in records])
    translation_median = np.median(translations, axis=0)
    translation_errors = np.linalg.norm(np.array(translations) - translation_median, axis=1)
    centres = np.array([record["centres"][0] for record in records])
    trajectory_length = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
    for record, error in zip(records, errors):
        record["rotation_from_medoid_deg"] = float(error)
    return {"schema": "dev.splatlab.raw-lens-rig-diagnostic/v1", "status": "diagnostic_only",
            "pairs": records, "training_images": len(poses), "paired_timestamps": len(records),
            "rotation_medoid_group": records[medoid_index]["group"],
            "rotation_error_deg": {"median": float(np.median(errors)), "max": float(errors.max())},
            "baseline_arbitrary_units": {"min": float(baseline.min()), "median": float(np.median(baseline)),
                                          "max": float(baseline.max()), "std": float(baseline.std())},
            "max_translation_deviation_arbitrary_units": float(translation_errors.max()),
            "lens0_trajectory_arbitrary_units": trajectory_length,
            "median_baseline_over_trajectory": float(np.median(baseline)) / trajectory_length if trajectory_length else None,
            "metric_scale": "unknown", "calibration_accepted": False, "heldout_evaluated": False,
            "limitations": ["Training-only diagnostic; small residuals do not establish lens or rig calibration.",
                            "Independently estimated lens poses may absorb distortion and weak overlap errors.",
                            "No known physical baseline, orientation, survey scale or GPS registration is imposed.",
                            "Localize held-out images against frozen geometry and verify projection before Gaussian training."]}
