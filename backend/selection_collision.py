"""Selection-partitioned collision candidates for atomic captured removals."""

from copy import deepcopy
import hashlib
from pathlib import Path
import re
import uuid

import numpy as np

import artifact_manifest as manifests
import background_recovery
import glb_check
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as reviews

COLLISION_RE = re.compile(r"collision_[a-f0-9]{24}\Z")
METHOD = "selection-aware-local-volume-cut"
REQUIRED_GATES = frozenset({"watertight", "selected_centers_clear", "edit_volume_clear", "no_faces_inside_edit_volume",
    "outside_surface_preserved", "bounded_positive_volume_removed", "floor_coverage_not_reduced", "maximum_hole_not_increased"})


def directory(job: Path, identifier: str):
    if not COLLISION_RE.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid selection collision identifier")
    output = scenes.root(job) / "selection-collisions" / identifier
    if output.resolve() != output.absolute():
        raise evidence.EvidenceError("Selection collision paths cannot be symlinked")
    return output


def read(job: Path, identifier: str, result=False):
    document = manifests.read_json(directory(job, identifier) / ("result.json" if result else "receipt.json"))
    if not document:
        raise evidence.EvidenceError("Selection collision evidence is missing")
    digest = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in document.items() if key != "sha256"})).hexdigest()
    if document.get("sha256") != digest:
        raise evidence.EvidenceError("Selection collision integrity check failed")
    if result and document.get("prepared_receipt_sha256") != read(job, identifier)["sha256"]:
        raise evidence.EvidenceError("Collision result belongs to different inputs")
    return document


def membership(selections: dict):
    count = selections.get("n_rows")
    if type(count) is not int or not 0 < count <= reviews.MAX_ROWS:
        raise evidence.EvidenceError("Invalid captured row space for collision")
    owner = np.zeros(count, dtype=bool)
    normalized = {}
    for slug, entry in sorted(selections.get("elements", {}).items()):
        rows = np.asarray(entry.get("rows", []))
        if (entry.get("coordinate_verified") is not True or rows.ndim != 1 or rows.dtype.kind not in "iu"
                or not 0 < len(rows) <= 200_000 or np.any(rows < 0) or np.any(rows >= count)
                or len(np.unique(rows)) != len(rows) or owner[rows].any()):
            raise evidence.EvidenceError("Collision partition needs disjoint, coordinate-verified served rows")
        owner[rows] = True
        normalized[slug] = sorted(rows.tolist())
    if not normalized:
        raise evidence.EvidenceError("No captured membership is available")
    digest = hashlib.sha256(scenes.canonical_bytes({"n_rows": count, "elements": normalized})).hexdigest()
    return digest, np.flatnonzero(~owner)


def artifact(job: Path, identifier: str, name: str, result=False):
    record = read(job, identifier, result)["artifacts"].get(name)
    if not record:
        raise evidence.EvidenceError("Artifact does not belong to this collision build")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Collision artifact is missing or corrupt")
    return path


def verify(job: Path, receipt: dict):
    study = reviews.read(job, receipt["selection_review_id"])
    reviews.verify(job, study)
    result = reviews.read(job, receipt["selection_review_id"], result=True)
    if study["base"] != receipt["base"] or result["sha256"] != receipt["selection_result_sha256"]:
        raise evidence.EvidenceError("Collision build belongs to a different selection result")
    background_recovery.verify_receipt_sources(job, receipt, checksums=True)


def prepare(job: Path, review_id: str, expected_generation: int, voxel_m=.05):
    if not isinstance(voxel_m, (int, float)) or isinstance(voxel_m, bool) or not .03 <= voxel_m <= .1:
        raise evidence.EvidenceError("Collision voxel size must be between 3 and 10 cm")
    with background_recovery.worker_slot(), scenes.write_lock(job):
        study = reviews.read(job, review_id)
        reviews.verify(job, study)
        if study["base"]["generation"] != expected_generation:
            raise evidence.EvidenceError("Active scene changed before collision preparation")
        result = reviews.read(job, review_id, result=True)
        revision = scenes.read_revision(job, study["base"]["revision_id"])
        state = revision["state"]
        if state["viewer"].get("units") != "meters" or state["viewer"].get("calibration", {}).get("stale"):
            raise evidence.EvidenceError("Selection collision requires a currently calibrated metre world")
        if list((job / "_scene/surfaces").glob("patch_*.ply")) or list((job / "_scene/ground").glob("*.npz")):
            raise evidence.EvidenceError("Painted/measured surface authority needs a partition adapter before this collider can replace it")
        selections = deepcopy(state["selections"])
        original_hash, original_background = membership(selections)
        with np.load(reviews.artifact(job, review_id, "candidate-rows.npz"), allow_pickle=False) as stored:
            candidate = stored["candidate"]
        selected = study["selected_slug"]
        if selected not in selections["elements"] or len(candidate) != result["candidate_count"]:
            raise evidence.EvidenceError("Candidate does not match the selected captured instance")
        core = np.asarray(selections["elements"][selected]["rows"], dtype=np.int64)
        if not np.isin(core, candidate).all():
            raise evidence.EvidenceError("Refined candidate must retain the original captured core")
        selections["elements"][selected] = {"rows": candidate.tolist(), "count": len(candidate), "coordinate_verified": True,
            "row_space": "served-splat", "selection_review_id": review_id,
            "coordinate_sources": {"splat": study["splat_artifact"], "rows": result["artifacts"]["candidate-rows.npz"]}}
        selections.get("coordinate_sources", {}).pop(selected, None)
        selections["method"] = "revision-pinned multi-view selection with retained served-row evidence"
        refined_hash, refined_background = membership(selections)
        prior_report = scenes.read_artifact_json(job, revision, "_world/collision_shell.json")
        frame = prior_report.get("geometry_frame") or {}
        if (prior_report.get("verdict") not in {"PASS", "PASS_LOCAL_EDIT"} or frame.get("axis") != "y-up"
                or frame.get("units") != "scene-units" or not prior_report.get("probe")
                or state["viewer"].get("collision_shell", {}).get("scale_to_world") != study["calibration"]["meters_per_unit"]):
            raise evidence.EvidenceError("A graded, frame-verified base collider is required for a local edit")
        sources = deepcopy(study["sources"])
        mesh = evidence.input_path(job, "_mesh/mesh.ply", sources)
        identifier = "collision_" + uuid.uuid4().hex[:24]
        output = directory(job, identifier)
        output.mkdir(parents=True)
        np.savez_compressed(output / "partition.npz", current_background=original_background,
                            candidate_background=refined_background, core=core, candidate=candidate)
        payload = {"schema": "dev.splatlab.selection-collision/v1", "collision_id": identifier,
            "selection_review_id": review_id, "selection_result_sha256": result["sha256"], "selected_slug": selected,
            "base": study["base"], "calibration": study["calibration"], "sources": sources,
            "splat_artifact": study["splat_artifact"], "bounds_artifact": scenes.store_file(job, mesh),
            "base_collision_artifact": revision["artifacts"]["_world/collision_shell.glb"],
            "base_collision_report": prior_report,
            "selections": selections, "current_membership_sha256": original_hash, "membership_sha256": refined_hash,
            "counts": {"scene": selections["n_rows"], "core": len(core), "candidate": len(candidate),
                       "current_background": len(original_background), "candidate_background": len(refined_background)},
            "recipe": {"method": METHOD, "voxel_m": float(voxel_m), "grid_m": .05, "close_radius": 2,
                       "player_height_m": 1.7, "player_radius_m": .25, "shared_baseline_probe": True,
                       "mesh_role": "room bounds only; never sampled into occupancy", "floor_mode": "plane",
                       "cut_shape": "floor-extruded convex selection hull", "cut_margin_m": .03,
                       "floor_clearance_m": .03, "maximum_removed_volume_m3": 1.0},
            "created_at": manifests.utc_now(), "scope": "regularized collision candidate, not measured geometry or navigation acceptance"}
        verify(job, payload)
        return reviews.seal(job, output, "receipt.json", payload, [output / "partition.npz"])


def verify_partition(revision: dict, selected_slug=None):
    partition = revision["state"].get("capture_partition") or {}
    if (partition.get("method") != METHOD or partition.get("membership_sha256") != membership(revision["state"]["selections"])[0]
            or partition.get("collision_artifact") != revision["artifacts"].get("_world/collision_shell.glb")
            or selected_slug and selected_slug not in partition.get("cleared_slugs", [])):
        raise evidence.EvidenceError("Captured removal requires selection-aware collision clearance; rebuild before removing captured rows")


def install(job: Path, revision: dict, identifier: str, pointer: dict, selected_slug: str):
    receipt = read(job, identifier)
    verify(job, receipt)
    result = read(job, identifier, result=True)
    if receipt["base"] != pointer or selected_slug != receipt["selected_slug"]:
        raise evidence.EvidenceError("Collision candidate belongs to a different scene or selection")
    gates = result.get("local_gates")
    if (result.get("verdict") != "PASS_LOCAL_EDIT" or result.get("method") != METHOD or not isinstance(gates, dict)
            or any(gates.get(key) is not True for key in REQUIRED_GATES) or any(value is not True for value in gates.values())):
        raise evidence.EvidenceError("Selection collision did not pass its local clearance and preservation gates")
    if receipt["current_membership_sha256"] != membership(revision["state"]["selections"])[0]:
        raise evidence.EvidenceError("Collision base membership changed")
    path = artifact(job, identifier, "candidate/collision.glb", result=True)
    glb_check.validate_glb(path)
    if not glb_check.position_bounds(path)["identity_transforms"]:
        raise evidence.EvidenceError("Selection collision requires baked Y-up transforms")
    collision_artifact = result["artifacts"]["candidate/collision.glb"]
    for destination, name in (("_world/collision_shell.glb", "candidate/collision.glb"),
                              ("_world/collision_shell.json", "candidate/report.json"), ("_world/navmesh.json", "candidate/navmesh.json")):
        artifact(job, identifier, name, result=True)
        revision["artifacts"][destination] = result["artifacts"][name]
    state = revision["state"]
    cleared = sorted(set(state.get("capture_partition", {}).get("cleared_slugs", [])) | {selected_slug})
    state["selections"] = deepcopy(receipt["selections"])
    state["selections"].get("coordinate_sources", {}).pop(selected_slug, None)
    state["selections"]["method"] = "revision-pinned multi-view selection with retained served-row evidence"
    state["captured_collision_current"] = True
    state["capture_partition"] = {"collision_id": identifier, "method": METHOD,
        "membership_sha256": receipt["membership_sha256"], "collision_artifact": collision_artifact,
        "sources": receipt["sources"], "calibration": receipt["calibration"], "result_sha256": result["sha256"]}
    state["capture_partition"]["cleared_slugs"] = cleared
    state["viewer"]["collision_shell"] = {"glb": "artifact:_world/collision_shell.glb", "report": "artifact:_world/collision_shell.json",
        "scale_to_world": receipt["calibration"]["meters_per_unit"], "gates": result["candidate"]["gates"]}
    revision["artifacts"][f"_studio/selection-collisions/{identifier}/receipt.json"] = scenes.store_json(job, receipt)
    revision["artifacts"][f"_studio/selection-collisions/{identifier}/result.json"] = scenes.store_json(job, result)
    for record in (receipt, result):
        for name, identity in record["artifacts"].items():
            revision["artifacts"][f"_studio/selection-collisions/{identifier}/{name}"] = identity
    revision["artifacts"]["_world/pluck.json"] = scenes.store_json(job, state["selections"])
    verify_partition(revision, selected_slug)
