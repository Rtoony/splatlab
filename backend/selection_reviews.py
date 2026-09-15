"""Evidence-bound, review-only multi-view captured-instance selection studies."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import fcntl
import re
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import yaml

import artifact_manifest as manifests
import background_recovery
from langfield_align import read_ply_xyz
import reconstruction_evidence as evidence
import scene_revisions as scenes

REVIEW_RE = re.compile(r"selection_[a-f0-9]{24}\Z")
MAX_ROWS = 2_000_000
CONTRIBUTION_METHOD = "alpha-compositing-color-gradient/v1"
CONTRIBUTION_MINIMUM_MASS = .5
CONTRIBUTION_MINIMUM_PURITY = .8


def directory(job: Path, identifier: str) -> Path:
    if not REVIEW_RE.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid selection review identifier")
    path = scenes.root(job) / "selection-reviews" / identifier
    if path.resolve() != path.absolute():
        raise evidence.EvidenceError("Selection review directory cannot be a symlink")
    return path


def seal(job: Path, output: Path, name: str, payload: dict, files: list[Path]):
    payload = {**payload, "artifacts": {str(path.relative_to(output)): scenes.store_file(job, path) for path in files}}
    payload["sha256"] = hashlib.sha256(scenes.canonical_bytes(payload)).hexdigest()
    manifests.atomic_write_json(output / name, payload)
    return payload


def read(job: Path, identifier: str, result=False):
    payload = manifests.read_json(directory(job, identifier) / ("result.json" if result else "receipt.json"))
    if not payload:
        raise evidence.EvidenceError("Selection review is not available")
    digest = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in payload.items() if key != "sha256"})).hexdigest()
    if payload.get("sha256") != digest:
        raise evidence.EvidenceError("Selection review failed its integrity check")
    if result and payload.get("prepared_receipt_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Selection result belongs to different prepared inputs")
    return payload


def artifact(job: Path, identifier: str, name: str):
    receipt = read(job, identifier)
    records = dict(receipt["artifacts"])
    if (directory(job, identifier) / "result.json").is_file():
        records.update(read(job, identifier, result=True)["artifacts"])
    record = records.get(name)
    if not record:
        raise evidence.EvidenceError("Artifact does not belong to the selection review")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
        raise evidence.EvidenceError("Selection review artifact is missing")
    if manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Selection review artifact failed its integrity check")
    return path


def verify(job: Path, receipt: dict, checksums=True):
    pointer = scenes.active(job)
    if pointer != receipt["base"]:
        raise evidence.EvidenceError("Selection review belongs to an older active revision")
    revision = scenes.read_revision(job, pointer["revision_id"])
    if revision["source_fingerprint"] != scenes.source_fingerprint(job):
        raise evidence.EvidenceError("Refresh changed capture inputs before selection review")
    background_recovery.verify_receipt_sources(job, receipt, checksums=checksums)


def verify_prepared_files(job: Path, receipt: dict):
    output = directory(job, receipt["review_id"])
    for name, record in receipt["artifacts"].items():
        path = evidence.contained_file(job, str((output / name).relative_to(job)))
        if manifests.sha256_file(path) != record["sha256"]:
            raise evidence.EvidenceError("Prepared selection input changed; create another review")


@contextmanager
def worker_slot(job: Path, identifier: str):
    output = directory(job, identifier)
    with (output / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise evidence.EvidenceError("This selection review already has an active worker") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_record(job: Path, identifier: str, name: str):
    output = directory(job, identifier)
    payload = manifests.read_json(output / name)
    if not payload:
        raise evidence.EvidenceError(f"Missing sealed selection stage: {name}")
    digest = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in payload.items() if key != "sha256"})).hexdigest()
    if payload.get("sha256") != digest:
        raise evidence.EvidenceError("Selection stage failed its integrity check")
    receipt = read(job, identifier)
    if payload.get("prepared_receipt_sha256") != receipt["sha256"]:
        raise evidence.EvidenceError("Selection stage belongs to different prepared inputs")
    for filename, record in payload["artifacts"].items():
        path = evidence.contained_file(job, str((output / filename).relative_to(job)))
        if manifests.sha256_file(path) != record["sha256"]:
            raise evidence.EvidenceError("Selection stage output changed after recording")
    return payload


def record_run(job: Path, receipt: dict, name: str, payload: dict, files: list[Path]):
    verify(job, receipt)
    verify_prepared_files(job, receipt)
    output = directory(job, receipt["review_id"])
    if (output / name).exists():
        raise evidence.EvidenceError("Selection stage is already recorded; prepare another review")
    return seal(job, output, name, {**payload, "prepared_receipt_sha256": receipt["sha256"],
                                  "created_at": manifests.utc_now()}, files)


def require_fixed_cameras(job: Path, sources: dict):
    paths = list((job / "processed").glob("splatfacto*/**/config.yml"))
    if len(paths) != 1 or paths[0].stat().st_size > 2 * 1024 ** 2:
        raise evidence.EvidenceError("Selection projection requires one retained camera-optimizer configuration")
    path = evidence.input_path(job, str(paths[0].relative_to(job)), sources)
    node = yaml.compose(path.read_text(), Loader=yaml.SafeLoader)
    for field in ("pipeline", "model", "camera_optimizer", "mode"):
        if not isinstance(node, yaml.MappingNode):
            raise evidence.EvidenceError("Camera optimizer metadata is missing")
        matches = [value for key, value in node.value if key.value == field]
        if len(matches) != 1:
            raise evidence.EvidenceError("Camera optimizer metadata is ambiguous")
        node = matches[0]
    if not isinstance(node, yaml.ScalarNode) or node.value != "off":
        raise evidence.EvidenceError("Optimized camera deltas need a verified adapter before selection projection")


def raw_camera(camera: dict, metres: float, width: int, height: int):
    to_studio = np.eye(4)
    to_studio[:3, :3] = evidence.Y_UP * metres
    matrix = camera["world_to_camera"] @ to_studio
    row_scales = np.linalg.norm(matrix[:3, :3], axis=1)
    if not np.allclose(row_scales, row_scales[0], rtol=1e-6) or row_scales[0] <= 0:
        raise evidence.EvidenceError("Selection renderer needs a uniformly scaled camera frame")
    matrix[:3] /= row_scales[0]
    evidence.matrix4(matrix)
    if np.any(np.abs(camera["parameters"][4:]) > 1e-8):
        raise evidence.EvidenceError("Selection visibility requires an undistorted camera adapter")
    parameters = np.asarray(camera["parameters"][:4]) * np.tile([width / camera["width"], height / camera["height"]], 2)
    return matrix.tolist(), parameters.tolist()


def choose_views(cameras: dict, points: np.ndarray, maximum=8):
    center = np.median(points, axis=0)
    candidates = []
    for identifier, camera in cameras.items():
        pixels, depth = evidence.project(points, camera)
        valid = (depth > 0) & np.isfinite(pixels).all(axis=1)
        valid &= ((pixels >= 2) & (pixels < np.array([camera["width"], camera["height"]]) - 2)).all(axis=1)
        if valid.sum() < max(3, len(points) * .5):
            continue
        span = np.ptp(pixels[valid], axis=0) / [camera["width"], camera["height"]]
        direction = camera["center"] - center
        distance = np.linalg.norm(direction)
        if distance <= .01 or np.any(span > .8):
            continue
        candidates.append({"id": identifier, "area": float(np.prod(span)), "direction": direction / distance, "group": camera["group"]})
    selected = []
    while candidates and len(selected) < maximum:
        def score(candidate):
            if not selected:
                return candidate["area"]
            novelty = min(1 - float(candidate["direction"] @ previous["direction"]) for previous in selected)
            return candidate["area"] * min(1., novelty / .02)
        chosen = max(candidates, key=score)
        selected.append(chosen)
        candidates = [item for item in candidates if item["group"] != chosen["group"]
                      and item["direction"] @ chosen["direction"] <= np.cos(np.deg2rad(3))]
    if len(selected) < 4:
        raise evidence.EvidenceError("Selection review needs four distinct camera groups, separated by three degrees, with the instance in frame")
    return [item["id"] for item in selected]


def prepare(job: Path, selected_slug: str, expected_generation: int, maximum=8):
    if not 4 <= maximum <= 8:
        raise evidence.EvidenceError("Selection review is bounded to four through eight views")
    with background_recovery.worker_slot(), scenes.write_lock(job):
        pointer = scenes.active(job)
        if not pointer or pointer["generation"] != expected_generation:
            raise evidence.EvidenceError("Active scene changed before selection review")
        revision = scenes.read_revision(job, pointer["revision_id"])
        if revision["source_fingerprint"] != scenes.source_fingerprint(job):
            raise evidence.EvidenceError("Refresh the changed capture baseline before selection review")
        state = revision["state"]
        entry = next((item for item in state["viewer"]["elements"] if item["slug"] == selected_slug), None)
        selections = state.get("selections") or {}
        selected = selections.get("elements", {}).get(selected_slug, {})
        if not entry or entry.get("provenance") == "authored" or not selected.get("coordinate_verified"):
            raise evidence.EvidenceError("Select a captured instance with coordinate-verified row evidence")
        if selected_slug in state["hidden_capture_slugs"] or state["viewer"].get("calibration", {}).get("stale"):
            raise evidence.EvidenceError("Selection inspection requires a visible, currently calibrated capture")
        splat = scenes.artifact(job, pointer["revision_id"], "_preview/splat.ply")
        if splat.stat().st_size > evidence.MAX_FILE_BYTES:
            raise evidence.EvidenceError("Selection splat exceeds the 512 MiB inspection budget")
        positions = read_ply_xyz(splat)
        if len(positions) > MAX_ROWS or len(positions) != selections["n_rows"] or not np.isfinite(positions).all():
            raise evidence.EvidenceError("Selection splat exceeds the row budget or disagrees with its mapping")
        rows = np.asarray(selected["rows"], dtype=np.int64)
        if not 0 < len(rows) <= 200_000 or np.any(rows < 0) or np.any(rows >= len(positions)):
            raise evidence.EvidenceError("Selection rows escape the captured splat")
        source = evidence.load(job)
        require_fixed_cameras(job, source.sources)
        metres = source.calibration["meters_per_unit"]
        selected_world = positions[rows] @ evidence.Y_UP.T * metres
        identifiers = choose_views(source.cameras, selected_world, maximum)
        identifier = "selection_" + uuid.uuid4().hex[:24]
        output = directory(job, identifier)
        (output / "frames").mkdir(parents=True)
        cameras = []
        files = []
        for offset, image_id in enumerate(identifiers):
            camera = source.cameras[image_id]
            image_path = evidence.input_path(job, camera["image_key"], source.sources)
            with Image.open(image_path) as opened:
                if opened.size != (camera["width"], camera["height"]):
                    raise evidence.EvidenceError("Selection photo dimensions disagree with its solved camera")
                scale = min(1., 960 / max(opened.size))
                size = tuple(max(1, round(value * scale)) for value in opened.size)
                photo = opened.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            filename = f"frames/cam_{image_id:03d}.png"
            photo.save(output / filename)
            transform, parameters = raw_camera(camera, metres, *size)
            pixels, depth = evidence.project(selected_world, camera)
            pixels *= np.array(size) / [camera["width"], camera["height"]]
            overlay = photo.copy()
            draw = ImageDraw.Draw(overlay, "RGBA")
            for pixel in pixels[(depth > 0) & np.isfinite(pixels).all(axis=1)][::max(1, len(rows) // 3000)]:
                horizontal, vertical = pixel
                draw.ellipse((horizontal - 1, vertical - 1, horizontal + 1, vertical + 1), fill=(255, 210, 0, 170))
            overlay_name = f"centers-{image_id}.png"
            overlay.save(output / overlay_name)
            cameras.append({"image_id": image_id, "group": camera["group"], "image_key": camera["image_key"],
                            "photo": filename, "centers_overlay": overlay_name, "width": size[0], "height": size[1],
                            "world_to_camera": camera["world_to_camera"].tolist(), "raw_to_camera": transform,
                            "parameters": parameters, "split": "check" if offset % 4 == 3 else "fit"})
            files += [output / filename, output / overlay_name]
        other_rows = sorted({row for slug, item in selections["elements"].items() if slug != selected_slug for row in item["rows"]})
        np.savez_compressed(output / "rows.npz", core=rows, other=np.asarray(other_rows, dtype=np.int64))
        manifests.atomic_write_json(output / "views.json", {"cam_indices": identifiers})
        manifests.atomic_write_json(output / "things.json", [entry.get("label") or selected_slug.replace("-", " ")])
        files += [output / name for name in ("rows.npz", "views.json", "things.json")]
        payload = {"schema": "dev.splatlab.selection-review/v1", "review_id": identifier, "base": pointer,
                   "created_at": manifests.utc_now(), "selected_slug": selected_slug,
                   "label": entry.get("label") or selected_slug, "sources": source.sources, "calibration": source.calibration,
                   "cameras": cameras, "n_rows": len(positions), "core_count": len(rows),
                   "bounds_raw": {"min": positions[rows].min(0).tolist(), "max": positions[rows].max(0).tolist()},
                   "splat_artifact": revision["artifacts"]["_preview/splat.ply"], "evidence": source.report,
                   "recipe": {"minimum_view_angle_degrees": 3, "max_views": maximum, "camera_optimizer": "off"},
                   "scope": "review only; yellow marks are projected Gaussian centers, not visibility or complete-object masks"}
        verify(job, payload)
        return seal(job, output, "receipt.json", payload, files)


def match_mask(masks: np.ndarray, scores: np.ndarray, contribution: np.ndarray):
    if masks.ndim != 3 or masks.shape[1:] != contribution.shape or scores.shape != (len(masks),):
        raise evidence.EvidenceError("Mask dimensions disagree with the recorded review view")
    if (not np.isfinite(contribution).all() or not np.isfinite(scores).all()
            or np.any(contribution < 0) or np.any(contribution > 1.001)
            or np.any(scores < 0) or np.any(scores > 1) or masks.dtype != bool):
        raise evidence.EvidenceError("Invalid mask confidence or contribution values")
    mass = float(contribution.sum())
    if mass < 10:
        return None, "insufficient visible core contribution"
    overlap = (masks * contribution[None]).sum(axis=(1, 2)) / mass
    eligible = np.flatnonzero((overlap >= .5) & (scores >= .7))
    if not len(eligible):
        return None, "no predicted mask covers half the visible core"
    ranking = overlap[eligible] / np.sqrt(np.maximum(masks[eligible].sum(axis=(1, 2)), 1))
    order = eligible[np.argsort(-ranking)]
    if len(order) > 1:
        first, second = masks[order[0]], masks[order[1]]
        union = np.count_nonzero(first | second)
        if overlap[order[1]] > .8 * overlap[order[0]] and np.count_nonzero(first & second) / max(union, 1) < .5:
            return None, "ambiguous instance association; explicit mask choice required"
    return int(order[0]), None


def vote_rows(n_rows: int, observations: list[dict], core: np.ndarray, other: np.ndarray, spatial: np.ndarray):
    positive = np.zeros(n_rows, dtype=np.uint16)
    negative = np.zeros(n_rows, dtype=np.uint16)
    groups = {}
    for view in observations:
        if view["split"] != "fit":
            continue
        group = groups.setdefault(view["group"], {"positive": set(), "negative": set()})
        group["positive"].update(view["positive"].tolist())
        group["negative"].update(view["negative"].tolist())
    for group in groups.values():
        positive[np.asarray(sorted(group["positive"] - group["negative"]), dtype=np.int64)] += 1
        negative[np.asarray(sorted(group["negative"]), dtype=np.int64)] += 1
    eligible = spatial & (positive >= 2) & (positive >= 3 * negative)
    conflicts = int(np.count_nonzero(eligible[other]))
    eligible[other] = False
    eligible[core] = True
    return np.flatnonzero(eligible), positive, negative, conflicts


def contribution_rows(inside: np.ndarray, outside: np.ndarray, total: np.ndarray, n_rows: int):
    for weights in (inside, outside, total):
        if (weights.shape != (n_rows,) or weights.dtype.kind != "f"
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise evidence.EvidenceError("Contribution weights must be finite nonnegative floating-point arrays for every captured row")
    if not np.allclose(inside + outside, total, rtol=5e-4, atol=.01):
        raise evidence.EvidenceError("Contribution weights do not partition their rendered alpha mass")
    purity = np.divide(inside, total, out=np.zeros_like(inside), where=total > 0)
    visible = total >= CONTRIBUTION_MINIMUM_MASS
    positive = np.flatnonzero(visible & (purity >= CONTRIBUTION_MINIMUM_PURITY))
    negative = np.flatnonzero(visible & (purity <= 1 - CONTRIBUTION_MINIMUM_PURITY))
    return positive, negative


def read_contribution_run(job: Path, receipt: dict, mask_run: dict, visibility_run: dict):
    run = read_record(job, receipt["review_id"], "contribution/run.json")
    if (run.get("method") != CONTRIBUTION_METHOD
            or run.get("mask_run_sha256") != mask_run["sha256"]
            or run.get("visibility_run_sha256") != visibility_run["sha256"]):
        raise evidence.EvidenceError("Contribution lineage disagrees with its recorded mask or visibility run")
    cameras = receipt["cameras"]
    views = run.get("views", [])
    if ([view.get("image_id") for view in views] != [camera["image_id"] for camera in cameras]
            or [view.get("split") for view in views] != [camera["split"] for camera in cameras]):
        raise evidence.EvidenceError("Contribution views disagree with the prepared camera order or split")
    expected = {f"contribution/cam_{view['image_id']:03d}.npz" for view in views if view.get("selected_mask") is not None}
    if set(run["artifacts"]) != expected:
        raise evidence.EvidenceError("Contribution files disagree with associated camera views")
    return run


def refine(job: Path, identifier: str, contribution_weighted=False, spatial_margin_m=.05):
    with worker_slot(job, identifier):
        return _refine(job, identifier, contribution_weighted, spatial_margin_m)


def _refine(job: Path, identifier: str, contribution_weighted=False, spatial_margin_m=.05):
    if (type(spatial_margin_m) not in (int, float) or not np.isfinite(spatial_margin_m)
            or not 0 <= spatial_margin_m <= .2):
        raise evidence.EvidenceError("Selection spatial margin must be finite and between zero and 20 cm")
    receipt = read(job, identifier)
    verify(job, receipt)
    verify_prepared_files(job, receipt)
    output = directory(job, identifier)
    if (output / "result.json").exists() or (output / "candidate-rows.npz").exists():
        raise evidence.EvidenceError("This review already has a candidate; prepare another review instead of overwriting")
    mask_run = read_record(job, identifier, "mask-run.json")
    visibility_run = read_record(job, identifier, "visibility/run.json")
    if visibility_run.get("input_rows_sha256") != receipt["artifacts"]["rows.npz"]["sha256"]:
        raise evidence.EvidenceError("Visibility was not rendered from the prepared selection")
    contribution_run = read_contribution_run(job, receipt, mask_run, visibility_run) if contribution_weighted else None
    manifest = manifests.read_json(output / "sam3_manifest.json") or {}
    noun = manifest.get(receipt["label"])
    if not noun or not re.fullmatch(r"[a-z0-9-]+", noun.get("slug", "")):
        raise evidence.EvidenceError("A matching retained SAM mask run is required")
    with np.load(artifact(job, identifier, "rows.npz"), allow_pickle=False) as data:
        core, other = data["core"], data["other"]
    positions = read_ply_xyz(scenes.blob_path(job, receipt["splat_artifact"]["sha256"]))
    bounds = receipt["bounds_raw"]
    margin_m = spatial_margin_m
    margin = margin_m / receipt["calibration"]["meters_per_unit"]
    spatial = ((positions >= np.array(bounds["min"]) - margin) & (positions <= np.array(bounds["max"]) + margin)).all(axis=1)
    observations, views = [], []
    for camera in receipt["cameras"]:
        image_id = camera["image_id"]
        with np.load(output / "visibility" / f"cam_{image_id:03d}.npz", allow_pickle=False) as visible:
            contribution = visible["core"]
            row_ids, pixels = visible["gaussian_ids"], visible["pixels"]
            if (contribution.shape != (camera["height"], camera["width"])
                    or row_ids.ndim != 1 or row_ids.dtype.kind not in "iu"
                    or pixels.shape != (len(row_ids), 2) or pixels.dtype.kind not in "iu"
                    or np.any(row_ids < 0) or np.any(row_ids >= receipt["n_rows"])
                    or np.any(pixels < 0) or np.any(pixels >= [camera["width"], camera["height"]])):
                raise evidence.EvidenceError("Visibility rows or pixels escape their recorded scene view")
            with np.load(output / "masks" / noun["slug"] / f"cam_{image_id:03d}.npz", allow_pickle=False) as predicted:
                masks, scores = predicted["masks"], predicted["scores"]
            selected, reason = match_mask(masks, scores, contribution)
            view = {"image_id": image_id, "split": camera["split"], "selected_mask": selected, "reason": reason}
            if contribution_run:
                recorded = next(item for item in contribution_run["views"] if item["image_id"] == image_id)
                if recorded.get("selected_mask") != selected or recorded.get("reason") != reason:
                    raise evidence.EvidenceError("Contribution mask association changed from the recorded worker")
            if selected is not None:
                mask = masks[selected]
                if contribution_run:
                    with np.load(output / "contribution" / f"cam_{image_id:03d}.npz", allow_pickle=False) as weights:
                        positive_rows, negative_rows = contribution_rows(weights["inside"], weights["outside"], weights["total"], receipt["n_rows"])
                else:
                    hits = mask[pixels[:, 1], pixels[:, 0]]
                    positive_rows, negative_rows = row_ids[hits], row_ids[~hits]
                observations.append({"split": camera["split"], "group": camera["group"],
                                     "positive": positive_rows, "negative": negative_rows})
                mask_area = int(mask.sum())
                view.update(score=float(scores[selected]), mask_pixels=mask_area,
                            core_mask_coverage=float(np.count_nonzero(mask & (contribution >= .1)) / max(mask_area, 1)))
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"mask-{image_id}.png")
                with Image.open(output / camera["photo"]) as opened:
                    photo = np.asarray(opened.convert("RGB"), dtype=np.float32)
                overlay = np.where(mask[..., None], photo * .55 + np.array([25, 210, 190]) * .45, photo)
                Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(output / f"mask-overlay-{image_id}.png")
            views.append(view)
    if sum(view["split"] == "fit" for view in observations) < 2:
        raise evidence.EvidenceError("Fewer than two fit views associate unambiguously with this instance")
    candidate, positive, negative, conflicts = vote_rows(receipt["n_rows"], observations, core, other, spatial)
    np.savez_compressed(output / "candidate-rows.npz", candidate=candidate, positive_votes=positive[candidate], negative_votes=negative[candidate])
    report = {"review_id": identifier, "core_count": len(core), "candidate_count": len(candidate), "added_count": len(candidate) - len(core),
              "implementation_sha256": manifests.sha256_file(Path(__file__)),
              "mask_run_sha256": mask_run["sha256"], "visibility_run_sha256": visibility_run["sha256"],
              "core_rows_with_more_negative_votes": int(np.count_nonzero(negative[core] > positive[core])),
              "other_instance_conflicts_excluded": conflicts, "views": views,
              "recipe": {"visibility_method": CONTRIBUTION_METHOD if contribution_weighted else "projected-center-expected-depth/v1",
                         "minimum_fit_groups": 2, "minimum_positive_fraction": .75, "spatial_margin_m": margin_m,
                         "preserve_core": True, "exclude_other_instances": True, "holdouts_vote": False},
              "scope": "review-only candidate; inferred masks/depth are not ground truth; collision is not rebuilt"}
    if contribution_run:
        report["contribution_run_sha256"] = contribution_run["sha256"]
        report["recipe"].update(minimum_visible_alpha_mass_pixels=CONTRIBUTION_MINIMUM_MASS,
                                minimum_inside_alpha_fraction=CONTRIBUTION_MINIMUM_PURITY,
                                maximum_negative_inside_alpha_fraction=1 - CONTRIBUTION_MINIMUM_PURITY,
                                mixed_footprints_abstain=True)
    files = [output / "candidate-rows.npz", *output.glob("mask-*.png")]
    return record_run(job, receipt, "candidate.json", report, files)


def finalize(job: Path, identifier: str):
    with worker_slot(job, identifier):
        return _finalize(job, identifier)


def _finalize(job: Path, identifier: str):
    receipt = read(job, identifier)
    verify(job, receipt)
    verify_prepared_files(job, receipt)
    output = directory(job, identifier)
    if (output / "result.json").exists():
        raise evidence.EvidenceError("Selection result is already sealed")
    candidate = read_record(job, identifier, "candidate.json")
    mask_run = read_record(job, identifier, "mask-run.json")
    visibility_run = read_record(job, identifier, "visibility/run.json")
    rendered = read_record(job, identifier, "candidate-visibility/run.json")
    if (candidate["mask_run_sha256"] != mask_run["sha256"]
            or candidate["visibility_run_sha256"] != visibility_run["sha256"]
            or rendered.get("input_rows_sha256") != candidate["artifacts"]["candidate-rows.npz"]["sha256"]):
        raise evidence.EvidenceError("Candidate lineage disagrees with its recorded model runs")
    report = {key: value for key, value in candidate.items() if key not in ("sha256", "artifacts")}
    files = [output / name for name in ("candidate-rows.npz", "candidate.json", "sam3_manifest.json", "mask-run.json")]
    contribution_run = None
    if candidate.get("contribution_run_sha256"):
        contribution_run = read_contribution_run(job, receipt, mask_run, visibility_run)
        if contribution_run["sha256"] != candidate["contribution_run_sha256"]:
            raise evidence.EvidenceError("Candidate contribution lineage changed before finalization")
        files += [output / "contribution/run.json", *(output / name for name in contribution_run["artifacts"])]
        with np.load(output / "candidate-rows.npz", allow_pickle=False) as rows:
            candidate_rows = rows["candidate"]
    for camera in receipt["cameras"]:
        image_id = camera["image_id"]
        view = next(item for item in report["views"] if item["image_id"] == image_id)
        files += [output / "visibility" / f"overlay-{image_id}.png", output / "candidate-visibility" / f"overlay-{image_id}.png"]
        if view["selected_mask"] is not None:
            with Image.open(output / f"mask-{image_id}.png") as opened:
                mask = np.array(opened) >= 128
            with np.load(output / "candidate-visibility" / f"cam_{image_id:03d}.npz", allow_pickle=False) as data:
                contribution = data["core"]
            if contribution.shape != mask.shape or not np.isfinite(contribution).all():
                raise evidence.EvidenceError("Candidate contribution disagrees with the recorded mask")
            view["candidate_mask_coverage"] = float(np.count_nonzero(mask & (contribution >= .1)) / max(mask.sum(), 1))
            view["candidate_outside_mask_fraction"] = float(np.count_nonzero(~mask & (contribution >= .1)) / max(np.count_nonzero(contribution >= .1), 1))
            if contribution_run:
                with np.load(output / "contribution" / f"cam_{image_id:03d}.npz", allow_pickle=False) as weights:
                    inside, outside, total = weights["inside"], weights["outside"], weights["total"]
                    contribution_rows(inside, outside, total, receipt["n_rows"])
                    view["candidate_inside_alpha_mass_fraction"] = float(inside[candidate_rows].sum(dtype=np.float64) / max(inside.sum(dtype=np.float64), 1e-8))
                    view["candidate_outside_alpha_mass_fraction"] = float(outside[candidate_rows].sum(dtype=np.float64) / max(total[candidate_rows].sum(dtype=np.float64), 1e-8))
            files += [output / f"mask-{image_id}.png", output / f"mask-overlay-{image_id}.png"]
    files += list((output / "visibility").glob("*.npz")) + list((output / "candidate-visibility").glob("*.npz"))
    files += list((output / "masks").glob("*/*.npz"))
    files += [output / "visibility/run.json", output / "candidate-visibility/run.json"]
    report.update(base=receipt["base"], prepared_receipt_sha256=receipt["sha256"], created_at=manifests.utc_now(),
                  finalization_implementation_sha256=manifests.sha256_file(Path(__file__)))
    verify(job, receipt)
    return seal(job, output, "result.json", report, files)
