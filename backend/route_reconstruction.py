"""Evidence-linked reconstruction preparation; never launches training implicitly."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

import artifact_manifest as manifests
import route_builder as routes


class ReconstructionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    selection_window_s: float = Field(default=5, ge=1, le=60)
    min_sharpness: float = Field(default=0, ge=0)
    max_clipped_fraction: float = Field(default=0.5, ge=0, le=1)
    holdout_every: int = Field(default=5, ge=3, le=20)
    section_s: float = Field(default=45, ge=15, le=300)
    overlap_s: float = Field(default=10, ge=0, le=60)
    semantic_interval_s: float = Field(default=15, ge=5, le=120)
    nadir_exclusion_deg: float = Field(default=50, ge=0, le=80)

    @model_validator(mode="after")
    def overlap_fits(self):
        if self.overlap_s >= self.section_s:
            raise ValueError("Overlap must be shorter than each section")
        return self


def prepare(route_id: str, spec: ReconstructionSpec) -> dict:
    route = routes.load_route(route_id)
    if route["status"] == "running":
        raise ValueError("Pause the route before preparing a stable reconstruction input")
    directory = routes.route_dir(route_id)
    candidates, rejected = {}, []
    start = route["spec"]["start_s"]
    for point in route["points"]:
        if point["status"] != "ready":
            continue
        path = directory / f"pano-{point['index']:05d}.jpg"
        if not path.is_file() or manifests.sha256_file(path) != point.get("sha256"):
            raise ValueError(f"Viewpoint {point['index']} changed; resume and verify the route first")
        quality = point.get("quality") or {}
        if quality.get("sharpness", 0) < spec.min_sharpness or quality.get("dark_fraction", 0) + quality.get("bright_fraction", 0) > spec.max_clipped_fraction:
            rejected.append(point["index"])
            continue
        bucket = math.floor((point["time_s"] - start) / spec.selection_window_s)
        prior = candidates.get(bucket)
        if prior is None or quality.get("sharpness", 0) > (prior.get("quality") or {}).get("sharpness", 0):
            if prior is not None:
                rejected.append(prior["index"])
            candidates[bucket] = point
        else:
            rejected.append(point["index"])
    selected = sorted(candidates.values(), key=lambda point: point["time_s"])
    if len(selected) < 3:
        raise ValueError("At least three usable timestamp groups are required; extract more viewpoints")
    identity = {"route_id": route_id, "source_sha256": route["source_sha256"],
                "clock": route["clock"], "route_spec": route["spec"], "spec": spec.model_dump(),
                "points": [{"index": point["index"], "time_s": point["time_s"], "sha256": point["sha256"]} for point in selected]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    plan_dir = directory / "reconstruction" / fingerprint[:16]
    plan_dir.mkdir(parents=True, exist_ok=True)
    heldout = {point["index"] for index, point in enumerate(selected) if index % spec.holdout_every == spec.holdout_every // 2}
    if not heldout:
        heldout.add(selected[-1]["index"])
    validation = {point["index"] for index, point in enumerate(selected) if index % spec.holdout_every == (spec.holdout_every // 2 + 1) % spec.holdout_every and point["index"] not in heldout}
    if not validation:
        validation.add(next(point["index"] for point in reversed(selected) if point["index"] not in heldout))
    groups, semantic_ids = [], []
    next_semantic = -math.inf
    for point in selected:
        mask_path = plan_dir / f"mask-{point['index']:05d}.png"
        with Image.open(directory / f"pano-{point['index']:05d}.jpg") as panorama:
            mask = Image.new("L", panorama.size, 255)
            first_excluded_row = math.ceil(panorama.height * (1 - spec.nadir_exclusion_deg / 180))
            if first_excluded_row < panorama.height:
                ImageDraw.Draw(mask).rectangle((0, first_excluded_row, panorama.width - 1, panorama.height - 1), fill=0)
            staged = mask_path.with_name(mask_path.stem + f".{uuid.uuid4().hex}.building.png")
            mask.save(staged)
            staged.replace(mask_path)
        groups.append({"timestamp_group": f"pano-{point['index']:05d}", "point_index": point["index"],
                       "time_s": point["time_s"], "source_file": f"pano-{point['index']:05d}.jpg",
                       "sha256": point["sha256"], "split": "test" if point["index"] in heldout else "val" if point["index"] in validation else "train",
                       "mask": mask_path.name, "mask_sha256": manifests.sha256_file(mask_path)})
        if point["time_s"] >= next_semantic:
            semantic_ids.append(point["index"])
            next_semantic = point["time_s"] + spec.semantic_interval_s
    sections = []
    section_start = selected[0]["time_s"]
    while section_start <= selected[-1]["time_s"]:
        members = [group for group in groups if section_start <= group["time_s"] < section_start + spec.section_s]
        if not members:
            section_start += spec.section_s - spec.overlap_s
            continue
        sections.append({"section_id": f"section-{len(sections):04d}", "start_s": section_start,
                         "end_s": min(section_start + spec.section_s, route["points"][-1]["time_s"] + route["spec"]["interval_s"]),
                         "point_indices": [group["point_index"] for group in members],
                         "train_groups": sum(group["split"] == "train" for group in members),
                         "validation_groups": sum(group["split"] == "val" for group in members),
                         "test_groups": sum(group["split"] == "test" for group in members),
                         "status": "planned", "job_id": None, "navigation": "panorama-only"})
        if section_start + spec.section_s > selected[-1]["time_s"]:
            break
        section_start += spec.section_s - spec.overlap_s
    result = {"schema": "dev.splatlab.route-reconstruction/v1", "fingerprint": fingerprint,
              "created_at": manifests.utc_now(), **identity, "groups": groups, "sections": sections,
              "rejected_point_indices": sorted(rejected), "semantic_point_indices": semantic_ids,
              "mask_convention": "255=retain, 0=exclude; output-panorama coordinates",
              "status": "prepared-not-trained", "provenance": "observed",
              "warnings": ["Hold out every virtual view of a test timestamp; never split crops independently",
                           "A nadir cap is not an operator/motion segmentation mask; inspect remaining contamination",
                           "Translation, overlap and geometry confidence are not measured before camera solving",
                           "GPS is uncertain support, not a replacement for visual alignment",
                           "Small sections may lack held-out groups; do not report them as evaluated"]}
    manifests.atomic_write_json(plan_dir / "plan.json", result)
    manifests.atomic_write_json(directory / "reconstruction-plan.json", {"fingerprint": fingerprint, "plan": str((plan_dir / "plan.json").relative_to(directory))})
    return result


def apply_group_split(transforms: dict, plan: dict) -> dict:
    splits = {group["timestamp_group"]: group["split"] for group in plan["groups"]}
    filenames = {"train": [], "val": [], "test": []}
    for frame in transforms.get("frames", []):
        filename = frame["file_path"]
        timestamp_group = Path(filename).stem
        if timestamp_group not in splits:
            raise ValueError(f"No timestamp lineage for {filename}; refusing a potentially leaking split")
        filenames[splits[timestamp_group]].append(filename)
    if not all(filenames.values()):
        raise ValueError("Registered cameras must include train, validation and held-out test timestamp groups")
    return {**transforms, "train_filenames": filenames["train"], "val_filenames": filenames["val"],
            "test_filenames": filenames["test"], "splatlab_evaluation": {"plan_fingerprint": plan["fingerprint"], "unit": "complete-panorama-timestamp", "validation": "separate timestamps; reserve test for final comparison"}}
