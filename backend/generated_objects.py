"""Revision-bound local generated-object inputs and immutable reviewed assets."""

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image

import artifact_manifest as manifests
import background_completion
import background_recovery as recovery
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as reviews
from langfield_align import read_ply_xyz
from mesh.provenance import glb_is_generative

OBJECT_RE = re.compile(r"generated_[a-f0-9]{24}\Z")
IMPLEMENTATION = ("backend/generated_objects.py", "backend/mesh/object_generate.py", "tools/generated-object.py", "backend/generated_color.py")


def directory(job, identifier):
    if not OBJECT_RE.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid generated-object identifier")
    output = scenes.root(job) / "generated-objects" / identifier
    if output.resolve() != output.absolute():
        raise evidence.EvidenceError("Generated-object paths cannot be symlinked")
    return output


def read(job, identifier, result=False):
    record = manifests.read_json(directory(job, identifier) / ("result.json" if result else "receipt.json"))
    if not record:
        raise evidence.EvidenceError("Generated-object evidence is missing")
    digest = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in record.items() if key != "sha256"})).hexdigest()
    if record.get("sha256") != digest:
        raise evidence.EvidenceError("Generated-object integrity check failed")
    if result and record.get("prepared_receipt_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Generated result belongs to different inputs")
    return record


def artifact(job, identifier, name, result=False):
    identity = read(job, identifier, result).get("artifacts", {}).get(name)
    if not identity:
        raise evidence.EvidenceError("Artifact does not belong to this generated object")
    path = scenes.blob_path(job, identity["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != identity["bytes"] or manifests.sha256_file(path) != identity["sha256"]:
        raise evidence.EvidenceError("Generated-object artifact is missing or corrupt")
    return path


@contextmanager
def worker_slot(job, identifier):
    with (directory(job, identifier) / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise evidence.EvidenceError("This generated object already has an active worker") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify(job, receipt):
    if scenes.active(job) != receipt["base"]:
        raise evidence.EvidenceError("Generated object is stale; the active scene changed")
    revision = scenes.read_revision(job, receipt["base"]["revision_id"])
    if revision["source_fingerprint"] != scenes.source_fingerprint(job):
        raise evidence.EvidenceError("Capture, scale or selection changed before generation")
    recovery.verify_receipt_sources(job, receipt, checksums=True)
    if reviews.read(job, receipt["selection_review_id"], True)["sha256"] != receipt["selection_result_sha256"]:
        raise evidence.EvidenceError("Generated-object selection evidence changed")
    if recovery.read(job, receipt["recovery_id"])["sha256"] != receipt["recovery_sha256"]:
        raise evidence.EvidenceError("Generated-object support evidence changed")


def crop_box(mask):
    if mask.ndim != 2 or mask.dtype.kind != "b" or mask.sum() < 64 or max(mask.shape) > 960:
        raise evidence.EvidenceError("Choose a bounded source mask with at least 64 foreground pixels")
    rows, columns = np.nonzero(mask)
    if min(rows.min(), columns.min(), mask.shape[0] - 1 - rows.max(), mask.shape[1] - 1 - columns.max()) < 2:
        raise evidence.EvidenceError("The selected object touches the image edge; choose an untruncated view")
    padding = max(4, int(.15 * max(np.ptp(rows) + 1, np.ptp(columns) + 1)))
    return [max(0, int(columns.min()) - padding), max(0, int(rows.min()) - padding),
            min(mask.shape[1], int(columns.max()) + 1 + padding), min(mask.shape[0], int(rows.max()) + 1 + padding)]


def rigid_camera(camera):
    matrix = np.asarray(camera["world_to_camera"], dtype=np.float64).copy()
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise evidence.EvidenceError("Invalid raycasting camera transform")
    scales = np.linalg.norm(matrix[:3, :3], axis=1)
    if scales.min() <= 0 or not np.allclose(scales, scales[0], rtol=1e-6):
        raise evidence.EvidenceError("Raycasting requires a uniformly scaled camera frame")
    matrix[:3] /= scales[0]
    return evidence.matrix4(matrix)


def rederive(job, identifier):
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        original = read(job, identifier)
        verify(job, original)
        result = read(job, identifier, True)
        if result.get("model", {}).get("worker", {}).get("ok") is not True:
            raise evidence.EvidenceError("Only a completed model stage can supply a retained master")
        for name in ("master.glb", "master-splat.ply"):
            artifact(job, identifier, name, True)
        child = "generated_" + uuid.uuid4().hex[:24]
        output = directory(job, child)
        output.mkdir(parents=True)
        files = []
        for name in original["artifacts"]:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifact(job, identifier, name).read_bytes())
            files.append(path)
        payload = {key: deepcopy(value) for key, value in original.items() if key not in {"artifacts", "sha256"}}
        payload.update(generated_object_id=child, created_at=manifests.utc_now(), reuse_generated_object_id=identifier,
                       reuse_result_sha256=result["sha256"], implementation={name: manifests.sha256_file(Path(__file__).resolve().parents[1] / name)
                           for name in IMPLEMENTATION})
        return reviews.seal(job, output, "receipt.json", payload, files)


def verify_reuse(job, receipt):
    identifier = receipt["reuse_generated_object_id"]
    original = read(job, identifier)
    result = read(job, identifier, True)
    if (result["sha256"] != receipt["reuse_result_sha256"] or result.get("model", {}).get("worker", {}).get("ok") is not True):
        raise evidence.EvidenceError("Retained model-stage receipt changed or has no completed model stage")
    if (original["seed"] != receipt["seed"] or original["recipe"]["model"] != receipt["recipe"]["model"]
            or original["artifacts"]["input.png"] != receipt["artifacts"]["input.png"]):
        raise evidence.EvidenceError("Retained inference requires byte-identical conditioning, seed and model recipe; refit uses fresh evidence")
    artifact(job, identifier, "input.png")
    for name in ("master.glb", "master-splat.ply"):
        artifact(job, identifier, name, True)
    return result


def prepare(job, review_id, recovery_id, expected_generation, image_id=None, seed=42, reuse_generated_object_id=None):
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise evidence.EvidenceError("Use a bounded integer generation seed")
    with recovery.worker_slot(wait_seconds=3), scenes.write_lock(job):
        selection = reviews.read(job, review_id)
        reviews.verify(job, selection)
        result = reviews.read(job, review_id, True)
        support = recovery.read(job, recovery_id)
        pointer = scenes.active(job)
        if pointer["generation"] != expected_generation or support["base"] != pointer or support["selected_slug"] != selection["selected_slug"]:
            raise evidence.EvidenceError("Generation requires current selection and support for the same object")
        if support["report"]["plane"].get("selection", {}).get("method") != "photo-linked-sfm-anchors":
            raise evidence.EvidenceError("Choose photo-linked support before fitting a generated replacement")
        background_completion.plane_arrays(support["report"]["plane"])
        views = []
        for camera in selection["cameras"]:
            record = next(item for item in result["views"] if item["image_id"] == camera["image_id"])
            if record.get("selected_mask") is None:
                continue
            mask_name = f"mask-{camera['image_id']}.png"
            with Image.open(reviews.artifact(job, review_id, mask_name)) as opened:
                mask = np.asarray(opened.convert("L")) >= 128
            if mask.shape != (camera["height"], camera["width"]):
                raise evidence.EvidenceError("Registered mask and camera dimensions disagree")
            try:
                bounds = crop_box(mask)
                input_reason = None
            except evidence.EvidenceError as exc:
                bounds, input_reason = None, str(exc)
            views.append({**camera, "mask": mask_name, "crop": bounds, "input_refusal": input_reason,
                          "selection_coverage": record["candidate_mask_coverage"]})
        eligible = [view for view in views if view["split"] == "fit" and view["crop"] is not None
                    and (image_id is None or view["image_id"] == image_id)]
        if len([view for view in views if view["split"] == "fit"]) < 2 or not any(view["split"] == "check" for view in views):
            raise evidence.EvidenceError("Retain multiple fit views and independent check views for generation")
        if not eligible:
            raise evidence.EvidenceError("Choose an untruncated fit view; held-out cameras cannot condition generation")
        chosen = max(eligible, key=lambda view: view["selection_coverage"])
        identifier = "generated_" + uuid.uuid4().hex[:24]
        output = directory(job, identifier)
        output.mkdir(parents=True)
        files = []
        for view in views:
            for name in (view["photo"], view["mask"]):
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(reviews.artifact(job, review_id, name).read_bytes())
                files.append(path)
        with Image.open(output / chosen["photo"]) as opened:
            picture = opened.convert("RGBA")
        with Image.open(output / chosen["mask"]) as opened:
            picture.putalpha(opened.convert("L"))
        picture.crop(chosen["crop"]).save(output / "input.png")
        with np.load(reviews.artifact(job, review_id, "candidate-rows.npz"), allow_pickle=False) as rows:
            positions = read_ply_xyz(scenes.blob_path(job, selection["splat_artifact"]["sha256"]))[rows["candidate"]]
        np.savez_compressed(output / "target-points.npz", positions=positions)
        files += [output / "input.png", output / "target-points.npz"]
        payload = {"schema": "dev.splatlab.generated-object/v1", "generated_object_id": identifier, "base": pointer,
                   "selected_slug": selection["selected_slug"], "selection_review_id": review_id, "selection_result_sha256": result["sha256"],
                   "recovery_id": recovery_id, "recovery_sha256": support["sha256"], "sources": selection["sources"], "calibration": selection["calibration"],
                   "support_plane": support["report"]["plane"], "cameras": views, "input_image_id": chosen["image_id"], "input_crop": chosen["crop"],
                   "seed": seed, "created_at": manifests.utc_now(), "render_vr_only": True,
                   "recipe": {"model": "local SAM-3D Objects", "delivery_faces": 32000, "seconds": 600,
                              "placement": "multi-fit-view silhouette search with inferred-plane contact", "fit_samples": 30000,
                              "minimum_fit_iou": .5, "holdouts_condition_generation": False, "holdouts_fit_pose": False},
                   "implementation": {name: manifests.sha256_file(Path(__file__).resolve().parents[1] / name)
                                      for name in IMPLEMENTATION},
                   "scope": "single-reference generated appearance/geometry, fitted placement; not measured reconstruction or navigation acceptance"}
        verify(job, payload)
        if reuse_generated_object_id:
            parent = read(job, reuse_generated_object_id, True)
            payload.update(reuse_generated_object_id=reuse_generated_object_id, reuse_result_sha256=parent["sha256"],
                           reuse_scope="native inference only from identical conditioning; current selection, support and pose are recomputed")
            verify_reuse(job, {**payload, "artifacts": {"input.png": scenes.store_file(job, output / "input.png")}})
        return reviews.seal(job, output, "receipt.json", payload, files)


def proposal_asset(job, identifier, pointer, selected_slug=None):
    receipt = read(job, identifier)
    verify(job, receipt)
    result = read(job, identifier, True)
    if receipt["base"] != pointer or selected_slug and receipt["selected_slug"] != selected_slug:
        raise evidence.EvidenceError("Generated object belongs to a different scene or captured instance")
    if result.get("status") != "needs-review" or result.get("render_vr_only") is not True or result.get("placement_resolved") is not True:
        raise evidence.EvidenceError("A complete, placed generated candidate is required")
    for name in ("master.glb", "placed-master.glb", "delivery.glb"):
        if not glb_is_generative(artifact(job, identifier, name, True)):
            raise evidence.EvidenceError("Generated master and delivery need portable generated provenance")
    return artifact(job, identifier, "delivery.glb", True), result["artifacts"]["delivery.glb"]["sha256"]


def attach(job, revision, entry, slug, identifier):
    receipt = read(job, identifier)
    result = read(job, identifier, True)
    entry.update(provenance="generated", geometry_source="sam3d-generated", appearance_source="model-generated",
                 classification=["local single-reference generative model; unseen surfaces invented; fitted placement, not measured geometry"],
                 generated={"generated_object_id": identifier, "model": result["model"], "placement": result["placement"],
                            "master": result["artifacts"]["master.glb"], "delivery": result["artifacts"]["delivery.glb"]})
    revision["state"].setdefault("recovery_dependencies", {})[slug] = {key: deepcopy(receipt[key]) for key in ("sources", "calibration")}
    for filename, record in (("receipt.json", receipt), ("result.json", result)):
        revision["artifacts"][f"_studio/generated/{identifier}/{filename}"] = scenes.store_json(job, record)
        for name, identity in record["artifacts"].items():
            artifact(job, identifier, name, filename == "result.json")
            revision["artifacts"][f"_studio/generated/{identifier}/{name}"] = identity
