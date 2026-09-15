"""Immutable creative scene bundles with compare-and-swap activation."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import uuid
from urllib.parse import parse_qs, urlparse

import artifact_manifest as manifests
from artifact_dependencies import scale_revision, shell_dependencies, shell_is_current
import glb_check
import world_pluck
import world_placed

SCHEMA = "dev.splatlab.scene-revision/v1"
MAX_SNAPSHOT_BYTES = 8 * 1024 ** 3
REVISION_RE = re.compile(r"scene_[a-f0-9]{24}\Z")
PROPOSAL_RE = re.compile(r"edit_[a-f0-9]{24}\Z")
EXPORT_RE = re.compile(r"scene-v[0-9]{4}(?:-[a-z0-9-]+)?\.glb\Z")
SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,39}\Z")
PATTERNS = (
    "_world/*.json", "_world/*.glb", "_world/*.png",
    "_world/elements/*.json", "_world/elements/*.glb", "_world/elements/*.png",
    "_world/collision/*.glb", "_preview/splat.ply", "_preview/langweb.ply",
    "_scene/inventory.json", "_scene/isolated/batch_isolate.json",
    "_scene/isolated/*/object_indices.npz", "_scene/isolated/*/object.ply", "_scene/surfaces/patch_*.ply",
    "_langfield/class_labels.json", "_langfield/ply_index_map.npy",
)


class SceneRevisionError(ValueError):
    pass


def root(job_dir: Path) -> Path:
    return job_dir / "_studio"


@contextmanager
def write_lock(job_dir: Path):
    directory = root(job_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def key_checked(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SceneRevisionError("Invalid artifact key")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise SceneRevisionError("Artifact paths must be relative and contained")
    return str(path)


def local_file(job_dir: Path, key: str) -> Path:
    path = job_dir / key_checked(key)
    if path.resolve() != path.absolute() or not path.is_file():
        raise SceneRevisionError(f"Missing or symlinked artifact: {key}")
    return path


def source_fingerprint(job_dir: Path) -> dict:
    files = sorted({path for pattern in PATTERNS for path in job_dir.glob(pattern) if path.is_file()})
    metadata = manifests.read_json(job_dir / "meta.json") or {}
    return {"collision_dependencies": shell_dependencies(job_dir),
            "language_field_stale": (job_dir / "_langfield" / "STALE").exists(),
            "calibration": {**scale_revision(job_dir), "geo": metadata.get("geo"),
                            "scale_calibration": metadata.get("scale_calibration")},
            "files": {str(path.relative_to(job_dir)): manifests.file_identity(local_file(job_dir, str(path.relative_to(job_dir))), include_sha256=False)
                      for path in files}}


def blob_path(job_dir: Path, digest: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise SceneRevisionError("Invalid blob digest")
    return root(job_dir) / "blobs" / digest[:2] / digest


def sync_directories(job_dir: Path, directory: Path) -> None:
    while directory.is_relative_to(job_dir):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = directory.parent


def durable_json(job_dir: Path, path: Path, document: dict) -> None:
    manifests.atomic_write_json(path, document)
    sync_directories(job_dir, path.parent)


def store_file(job_dir: Path, source: Path) -> dict:
    before = manifests.file_identity(source, include_sha256=False)
    staged_dir = root(job_dir) / "staging"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / uuid.uuid4().hex
    try:
        shutil.copyfile(source, staged)
        digest = manifests.sha256_file(staged)
        if not manifests.same_file_identity(source, before):
            raise SceneRevisionError("Source changed while snapshotting; retry after the writer finishes")
        target = blob_path(job_dir, digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != target.absolute() or manifests.sha256_file(target) != digest:
                raise SceneRevisionError("Stored scene blob failed its integrity check")
        else:
            staged.chmod(0o444)
            with staged.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(staged, target)
            sync_directories(job_dir, target.parent)
        return {"sha256": digest, "bytes": before["bytes"]}
    finally:
        staged.unlink(missing_ok=True)


def store_json(job_dir: Path, document: dict) -> dict:
    staged_dir = root(job_dir) / "staging"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / uuid.uuid4().hex
    try:
        staged.write_bytes(canonical_bytes(document))
        return store_file(job_dir, staged)
    finally:
        staged.unlink(missing_ok=True)


def read_artifact_json(job_dir: Path, revision: dict, key: str) -> dict:
    record = revision["artifacts"].get(key)
    if not record:
        return {}
    document = manifests.read_json(blob_path(job_dir, record["sha256"]))
    if not isinstance(document, dict):
        raise SceneRevisionError(f"Snapshot JSON is unreadable: {key}")
    return document


def canonical_bytes(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_revision(job_dir: Path, payload: dict) -> dict:
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    revision_id = "scene_" + digest[:24]
    document = {**payload, "revision_id": revision_id, "content_sha256": digest}
    path = root(job_dir) / "revisions" / f"{revision_id}.json"
    if path.exists() and manifests.read_json(path) != document:
        raise SceneRevisionError("Revision identity collision")
    if not path.exists():
        durable_json(job_dir, path, document)
    return document


def read_revision(job_dir: Path, revision_id: str) -> dict:
    if not isinstance(revision_id, str) or not REVISION_RE.fullmatch(revision_id):
        raise SceneRevisionError("Invalid scene revision")
    document = manifests.read_json(root(job_dir) / "revisions" / f"{revision_id}.json")
    if not document or document.get("schema") != SCHEMA:
        raise SceneRevisionError("Scene revision not found")
    payload = {key: value for key, value in document.items() if key not in {"revision_id", "content_sha256"}}
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    if document.get("revision_id") != revision_id or document.get("content_sha256") != digest or revision_id != "scene_" + digest[:24]:
        raise SceneRevisionError("Scene revision integrity check failed")
    return document


def active(job_dir: Path) -> dict | None:
    path = root(job_dir) / "active.json"
    pointer = manifests.read_json(path)
    if path.exists() and (not pointer or type(pointer.get("generation")) is not int or pointer["generation"] < 0):
        raise SceneRevisionError("Active scene pointer is damaged; refusing to overwrite it")
    if pointer:
        read_revision(job_dir, pointer.get("revision_id", ""))
    return pointer


def ancestry(job_dir: Path, revision_id: str) -> set[str]:
    result = set()
    while revision_id:
        if revision_id in result or len(result) >= 10000:
            raise SceneRevisionError("Revision ancestry is cyclic or exceeds its safety limit")
        result.add(revision_id)
        revision_id = read_revision(job_dir, revision_id)["parent"]
    return result


def _freeze_urls(value, job_id: str):
    if isinstance(value, dict):
        return {key: _freeze_urls(item, job_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_freeze_urls(item, job_id) for item in value]
    if isinstance(value, str) and value.startswith(f"/api/splat/jobs/{job_id}/world/file?"):
        query = parse_qs(urlparse(value).query)
        if len(query.get("name", [])) != 1:
            raise SceneRevisionError("World artifact URL has no unambiguous filename")
        return "artifact:" + key_checked("_world/" + key_checked(query["name"][0]))
    return value


def verify_references(value, artifacts: dict) -> None:
    if isinstance(value, dict):
        for item in value.values():
            verify_references(item, artifacts)
    elif isinstance(value, list):
        for item in value:
            verify_references(item, artifacts)
    elif isinstance(value, str) and value.startswith("artifact:"):
        if key_checked(value[len("artifact:"):]) not in artifacts:
            raise SceneRevisionError("Viewer references an asset missing from the scene snapshot")


def snapshot(job_dir: Path, viewer: dict, expected_sources: dict) -> dict:
    before = source_fingerprint(job_dir)
    if before != expected_sources:
        raise SceneRevisionError("World changed while resolving the viewer; retry initialization")
    if "_world/world_manifest.json" not in before["files"]:
        raise SceneRevisionError("A graded world is required before opening creative revisions")
    if sum(record["bytes"] for record in before["files"].values()) > MAX_SNAPSHOT_BYTES:
        raise SceneRevisionError("Scene snapshot exceeds the 8 GiB budget")
    frozen_viewer = _freeze_urls(viewer, job_dir.name)
    verify_references(frozen_viewer, before["files"])
    artifacts = {key: store_file(job_dir, local_file(job_dir, key)) for key in before["files"]}
    pluck, stale, reasons = world_pluck.read_pluck(job_dir / "_world", job_dir)
    state = {"viewer": frozen_viewer,
             "calibration": before["calibration"], "hidden_capture_slugs": [],
             "selections": pluck if pluck and not stale else None,
             "selection_warnings": reasons if stale else [],
             "captured_collision_current": shell_is_current(job_dir),
             "semantics": {entry["slug"]: {"label": entry.get("label") or entry["slug"],
                                           "provenance": entry.get("provenance") or "observed", "active": True}
                           for entry in viewer.get("elements", [])}}
    if before != source_fingerprint(job_dir):
        raise SceneRevisionError("Scene changed while snapshotting; retry when the writer finishes")
    return {"schema": SCHEMA, "job_id": job_dir.name, "created_at": manifests.utc_now(),
            "parent": None, "operation": {"kind": "capture-baseline"},
            "source_fingerprint": before, "artifacts": artifacts, "state": state}


def initialize(job_dir: Path, viewer: dict, expected_sources: dict) -> dict:
    with write_lock(job_dir):
        pointer = active(job_dir)
        if pointer:
            return pointer
        revision = write_revision(job_dir, snapshot(job_dir, viewer, expected_sources))
        pointer = {"revision_id": revision["revision_id"], "generation": 0, "proposal_id": None}
        durable_json(job_dir, root(job_dir) / "active.json", pointer)
        return pointer


def refresh_proposal(job_dir: Path, viewer: dict, expected_sources: dict, expected_generation: int) -> dict:
    with write_lock(job_dir):
        pointer = active(job_dir)
        if not pointer or pointer["generation"] != expected_generation:
            raise SceneRevisionError("Active scene changed; refresh before replacing the baseline")
        revision = snapshot(job_dir, viewer, expected_sources)
        revision.update(parent=pointer["revision_id"], operation={"kind": "refresh-baseline"})
        return write_proposal(job_dir, pointer, revision,
                              "Replace the studio state with the latest captured baseline. Studio edits are not carried over; prior revisions remain undoable.")


def read_proposal(job_dir: Path, proposal_id: str) -> dict:
    if not PROPOSAL_RE.fullmatch(proposal_id):
        raise SceneRevisionError("Invalid edit proposal")
    document = manifests.read_json(root(job_dir) / "proposals" / f"{proposal_id}.json")
    if not document:
        raise SceneRevisionError("Edit proposal not found")
    payload = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != hashlib.sha256(canonical_bytes(payload)).hexdigest():
        raise SceneRevisionError("Edit proposal integrity check failed")
    return document


def _sync_scene_documents(job_dir: Path, revision: dict) -> None:
    viewer = revision["state"]["viewer"]
    retained = {entry["slug"] for entry in viewer.get("elements", [])}
    world = read_artifact_json(job_dir, revision, "_world/world.json")
    grade = read_artifact_json(job_dir, revision, "_world/world_manifest.json")
    registry = read_artifact_json(job_dir, revision, "_world/placed.json") or world_placed.new_placed(job_dir.name)
    for document in (world, grade, registry):
        document["elements"] = [entry for entry in document.get("elements", []) if entry.get("slug") in retained]
    known_world = {entry["slug"] for entry in world["elements"]}
    known_grade = {entry["slug"] for entry in grade["elements"]}
    known_placed = {entry["slug"] for entry in registry["elements"]}
    for entry in viewer.get("elements", []):
        if entry["slug"] in known_world:
            continue
        key = "_world/elements/" + entry["slug"] + ".glb"
        bounds = glb_check.position_bounds(blob_path(job_dir, revision["artifacts"][key]["sha256"]))
        raw = {key: value for key, value in entry.items() if key != "files"}
        raw.update(glb=entry["slug"] + ".glb", built=True, geometry_source=entry.get("geometry_source", "authored"))
        world["elements"].append(raw)
        if entry["slug"] not in known_grade:
            grade["elements"].append(raw)
        if entry["slug"] not in known_placed:
            registry["elements"].append({"slug": entry["slug"], "label": entry["label"], "role": "environment",
                "aabb": bounds["aabb"], "extent": bounds["extent"], "placed_at": manifests.utc_now(),
                "source_filename": "revisioned-blender-export", "file": revision["artifacts"][key]})
    registry = world_placed.validate_placed(registry)
    for key, document in (("_world/world.json", world), ("_world/world_manifest.json", grade), ("_world/placed.json", registry),
                          ("_studio/semantic-state.json", {"elements": revision["state"]["semantics"]}),
                          ("_studio/splat-visibility.json", {"hidden_capture_slugs": revision["state"]["hidden_capture_slugs"]}
                           | ({"architectural_portals": revision["state"]["architectural_portals"]} if "architectural_portals" in revision["state"] else {}))):
        revision["artifacts"][key] = store_json(job_dir, document)


def _remove(revision: dict, slug: str) -> None:
    state = revision["state"]
    elements = state["viewer"].get("elements", [])
    entry = next((item for item in elements if item["slug"] == slug), None)
    if not entry:
        raise SceneRevisionError("Selected element does not exist in this revision")
    if entry.get("architecture_id"):
        raise SceneRevisionError("Restore the paired architectural revision to remove its room and close its captured opening together")
    if entry.get("provenance") not in {"authored", "generated"}:
        if state["viewer"].get("calibration", {}).get("stale"):
            raise SceneRevisionError("Rebuild the calibrated world before captured removal")
        selections = state.get("selections") or {}
        if not (selections.get("elements", {}).get(slug, {}).get("coordinate_verified") is True and state["captured_collision_current"] and entry.get("role") == "prop"):
            raise SceneRevisionError("Captured removal requires fresh splat-row selection with verified coordinates and walkable background collision evidence; re-isolate, rebuild pluck/collision, and preview a fresh baseline")
        import selection_collision
        from reconstruction_evidence import EvidenceError

        try:
            selection_collision.verify_partition(revision, slug)
        except EvidenceError as exc:
            raise SceneRevisionError(str(exc)) from exc
        state["hidden_capture_slugs"] = sorted(set(state["hidden_capture_slugs"]) | {slug})
    state["viewer"]["elements"] = [item for item in elements if item["slug"] != slug]
    state["semantics"][slug]["active"] = False
    if "recovery_dependencies" in state:
        state["recovery_dependencies"].pop(slug, None)


def proposal_asset(job_dir: Path, operation: dict, pointer: dict):
    generated_id = operation.get("generated_object_id")
    if generated_id:
        import generated_objects
        from reconstruction_evidence import EvidenceError

        if any(operation.get(name) for name in ("recovery_id", "completion_id", "blender_export")):
            raise SceneRevisionError("Choose one generated object, completion, recovery or Blender export")
        try:
            selected = operation.get("selected_slug") if operation["kind"] == "replace" else None
            if operation.get("gaussians_id"):
                import generated_gaussians

                path, digest = generated_gaussians.proposal_asset(job_dir, generated_id, operation["gaussians_id"], pointer, selected)
            else:
                path, digest = generated_objects.proposal_asset(job_dir, generated_id, pointer, selected)
        except EvidenceError as exc:
            raise SceneRevisionError(str(exc)) from exc
        return path, digest, None
    recovery_id = operation.get("recovery_id")
    filename = operation.get("blender_export", "")
    completion_id = operation.get("completion_id")
    if completion_id:
        import background_completion
        from reconstruction_evidence import EvidenceError

        if recovery_id or filename:
            raise SceneRevisionError("Choose one completion, recovery or Blender export")
        try:
            path, digest, _ = background_completion.proposal_asset(job_dir, completion_id, pointer,
                operation.get("selected_slug") if operation["kind"] == "replace" else None)
        except EvidenceError as exc:
            raise SceneRevisionError(str(exc)) from exc
        return path, digest, None
    if recovery_id:
        import background_recovery
        from reconstruction_evidence import EvidenceError

        if filename:
            raise SceneRevisionError("Choose a recovery or Blender export, not both")
        try:
            recovery = background_recovery.read(job_dir, recovery_id)
            if recovery.get("render_vr_only") is not True:
                raise SceneRevisionError("Rebuild this recovery to include its portable render-only provenance")
            if recovery["base"] != pointer:
                raise SceneRevisionError("Recovery belongs to an older scene; rebuild against the active revision")
            if operation["kind"] == "replace" and recovery["selected_slug"] != operation.get("selected_slug"):
                raise SceneRevisionError("Recovery was fitted for a different captured element")
            background_recovery.verify_receipt_sources(job_dir, recovery, checksums=True)
            path = background_recovery.artifact(job_dir, recovery_id, "candidate.glb")
        except EvidenceError as exc:
            raise SceneRevisionError(str(exc)) from exc
        return path, recovery["artifacts"]["candidate.glb"]["sha256"], recovery
    if not EXPORT_RE.fullmatch(filename):
        raise SceneRevisionError("Choose a registered Blender export filename, not a filesystem path")
    export = local_file(job_dir, "_blender/exports/" + filename)
    receipt = manifests.read_json(export.with_suffix(".json")) or {}
    if not receipt or not manifests.same_file_identity(export, receipt.get("output")):
        raise SceneRevisionError("Blender export receipt is missing or stale; export again")
    return export, receipt["output"].get("sha256"), None


def verify_recovery_dependencies(job_dir: Path, revision: dict):
    import background_recovery
    from reconstruction_evidence import EvidenceError

    try:
        import architectural_edits

        architectural_removals = architectural_edits.verify_revision(job_dir, revision)
        for dependency in revision["state"].get("recovery_dependencies", {}).values():
            background_recovery.verify_receipt_sources(job_dir, dependency, checksums=True)
        partition = revision["state"].get("capture_partition")
        if partition:
            import selection_collision

            selection_collision.verify_partition(revision)
            background_recovery.verify_receipt_sources(job_dir, partition, checksums=True)
        for slug in revision["state"].get("hidden_capture_slugs", []):
            if slug in architectural_removals:
                continue
            import selection_collision

            selection_collision.verify_partition(revision, slug)
    except EvidenceError as exc:
        raise SceneRevisionError(str(exc)) from exc


def background_operation(operation: dict):
    background = operation.get("background")
    if background is None:
        return None
    if (operation.get("kind") != "replace" or bool(operation.get("blender_export")) == bool(operation.get("generated_object_id"))
            or operation.get("recovery_id") or operation.get("completion_id")):
        raise SceneRevisionError("A companion background requires one registered Blender or generated replacement object")
    if not isinstance(background, dict) or set(background) - {"slug", "label", "recovery_id", "completion_id"}:
        raise SceneRevisionError("Background accepts only a named recovery or completion")
    if bool(background.get("recovery_id")) == bool(background.get("completion_id")):
        raise SceneRevisionError("Choose exactly one recovery or completion for the background")
    if background.get("slug") == operation.get("slug"):
        raise SceneRevisionError("Replacement and background need distinct element names")
    return {**background, "kind": "replace", "selected_slug": operation.get("selected_slug")}


def place_asset(job_dir: Path, revision: dict, operation: dict, pointer: dict):
    if revision["state"].get("architectural_portals") and operation.get("gaussians_id"):
        raise SceneRevisionError("Native Gaussian placement after an architectural cut needs verified footprint/portal clearance")
    slug = operation.get("slug") or ""
    if not isinstance(slug, str) or not SLUG_RE.fullmatch(slug) or slug == "shell":
        raise SceneRevisionError("New element slug must be lowercase kebab-case and cannot be shell")
    if slug in revision["state"]["semantics"]:
        raise SceneRevisionError("Element slug already exists in this scene's history")
    export, expected_digest, recovery = proposal_asset(job_dir, operation, pointer)
    if revision["state"]["viewer"].get("units") != "meters" or revision["state"]["viewer"].get("calibration", {}).get("stale"):
        raise SceneRevisionError("Metre-authored assets require a calibrated metre world")
    try:
        glb_check.validate_glb(export)
        bounds = glb_check.position_bounds(export)
    except ValueError as exc:
        raise SceneRevisionError(f"Blender export is not a valid self-contained mesh: {exc}") from exc
    if not bounds["identity_transforms"]:
        raise SceneRevisionError("Blender export must bake world transforms before placement")
    key = f"_world/elements/{slug}.glb"
    asset_identity = store_file(job_dir, export)
    if expected_digest != asset_identity["sha256"]:
        raise SceneRevisionError("Asset checksum does not match its receipt")
    revision["artifacts"][key] = asset_identity
    label = str(operation.get("label") or slug)[:80]
    entry = {"slug": slug, "label": label, "role": "environment", "provenance": "authored",
             "files": {"glb": "artifact:" + key}, "extent": bounds["extent"],
             "collision": {"ok": True, "strategy": "complex_as_simple", "hulls": 0},
             "classification": ["revisioned authored architecture; visual and collision share geometry"]}
    if recovery:
        entry.update(geometry_source="sfm-plane-fit", appearance_source="captured-photos",
                     classification=["tracked SfM planar fit; observed-image texture; unknown cells omitted"],
                     recovery={"recovery_id": recovery["recovery_id"], "supported_fraction": recovery["report"]["supported_fraction"]})
        revision["state"].setdefault("recovery_dependencies", {})[slug] = {key: recovery[key] for key in ("sources", "calibration")}
        revision["artifacts"][f"_studio/{recovery['recovery_id']}.json"] = store_json(job_dir, recovery)
        for name, identity in recovery["artifacts"].items():
            revision["artifacts"][f"_studio/recoveries/{recovery['recovery_id']}/{name}"] = identity
    if operation.get("completion_id"):
        import background_completion
        from reconstruction_evidence import EvidenceError

        try:
            background_completion.attach(job_dir, revision, entry, slug, operation["completion_id"])
        except EvidenceError as exc:
            raise SceneRevisionError(str(exc)) from exc
    if operation.get("generated_object_id"):
        import generated_objects
        from reconstruction_evidence import EvidenceError

        try:
            if operation["kind"] == "replace":
                import selection_collision

                generated = generated_objects.read(job_dir, operation["generated_object_id"])
                partition = revision["state"].get("capture_partition", {})
                clearance = selection_collision.read(job_dir, partition["collision_id"]) if partition.get("collision_id") else {}
                if clearance.get("selection_review_id") != generated["selection_review_id"]:
                    raise SceneRevisionError("Generated replacement requires clearance from the same refined selection")
            generated_objects.attach(job_dir, revision, entry, slug, operation["generated_object_id"])
            if operation.get("gaussians_id"):
                import generated_gaussians

                generated_gaussians.attach(job_dir, revision, entry, slug, operation["generated_object_id"], operation["gaussians_id"])
        except EvidenceError as exc:
            raise SceneRevisionError(str(exc)) from exc
    portals = revision["state"].get("architectural_portals", [])
    if portals:
        import numpy as np

        corners = np.array([[horizontal, vertical, depth] for horizontal in (bounds["aabb"]["min"][0], bounds["aabb"]["max"][0])
                            for vertical in (bounds["aabb"]["min"][1], bounds["aabb"]["max"][1])
                            for depth in (bounds["aabb"]["min"][2], bounds["aabb"]["max"][2])])
        for portal in portals:
            transform = np.asarray(portal["world_to_portal"])
            local = corners @ transform[:3, :3].T + transform[:3, 3]
            if np.all(local.max(axis=0) > np.asarray(portal["lower"]) - .01) and np.all(local.min(axis=0) < np.asarray(portal["upper"]) + .01):
                raise SceneRevisionError("Placed geometry overlaps a retained doorway; keep its paired visual/collision route clear")
    revision["state"]["viewer"].setdefault("elements", []).append(entry)
    revision["state"]["semantics"][slug] = {"label": label, "provenance": entry["provenance"], "active": True}
    return asset_identity


def propose(job_dir: Path, operation: dict, instruction: str, expected_generation: int) -> dict:
    with write_lock(job_dir):
        pointer = active(job_dir)
        if not pointer or pointer["generation"] != expected_generation:
            raise SceneRevisionError("Active scene changed; refresh before proposing")
        base = read_revision(job_dir, pointer["revision_id"])
        if source_fingerprint(job_dir) != base["source_fingerprint"]:
            raise SceneRevisionError("Legacy capture/scale/selection changed; open a fresh baseline before editing")
        revision = deepcopy(base)
        revision.pop("revision_id")
        revision.pop("content_sha256")
        kind = operation["kind"]
        if kind not in {"place", "remove", "replace", "extend-room"}:
            raise SceneRevisionError("Unsupported scene operation")
        if kind == "extend-room":
            import architectural_edits
            from reconstruction_evidence import EvidenceError

            if any(value is not None and key not in {"kind", "architecture_id", "slug", "label"} for key, value in operation.items()):
                raise SceneRevisionError("A connected-room proposal cannot mix independent placement/removal operations")
            try:
                identity = architectural_edits.install(job_dir, revision, operation.get("architecture_id"), pointer, operation.get("slug"), operation.get("label"))
            except (EvidenceError, ValueError) as exc:
                raise SceneRevisionError(str(exc)) from exc
            _sync_scene_documents(job_dir, revision)
            revision.update(parent=base["revision_id"], created_at=manifests.utc_now(), operation=operation)
            return write_proposal(job_dir, pointer, revision, instruction, identity)
        if operation.get("architecture_id"):
            raise SceneRevisionError("Architectural evidence belongs to a paired connected-room operation")
        if operation.get("completion_id") and kind not in {"place", "replace"}:
            raise SceneRevisionError("A completion belongs to a placement or replacement proposal")
        if operation.get("generated_object_id") and kind not in {"place", "replace"}:
            raise SceneRevisionError("A generated object belongs to a placement or replacement proposal")
        if operation.get("gaussians_id") and (not operation.get("generated_object_id") or kind not in {"place", "replace"}):
            raise SceneRevisionError("Native Gaussian appearance requires its generated object placement or replacement")
        background = background_operation(operation)
        selected = operation.get("selected_slug")
        collision_id = operation.get("selection_collision_id")
        if collision_id:
            import selection_collision
            from reconstruction_evidence import EvidenceError

            if kind not in {"remove", "replace"}:
                raise SceneRevisionError("Selection-aware collision can only accompany removal or replacement")
            try:
                selection_collision.install(job_dir, revision, collision_id, pointer, selected)
            except (EvidenceError, ValueError) as exc:
                raise SceneRevisionError(str(exc)) from exc
        if kind in {"remove", "replace"}:
            _remove(revision, selected)
        asset_identity = None
        if kind in {"place", "replace"}:
            asset_identity = place_asset(job_dir, revision, operation, pointer)
            if background:
                place_asset(job_dir, revision, background, pointer)
        _sync_scene_documents(job_dir, revision)
        revision.update(parent=base["revision_id"], created_at=manifests.utc_now(), operation=operation)
        return write_proposal(job_dir, pointer, revision, instruction, asset_identity)


def write_proposal(job_dir: Path, pointer: dict, revision: dict, instruction: str, asset_identity: dict | None = None) -> dict:
    proposal_id = "edit_" + uuid.uuid4().hex[:24]
    verify_references(revision["state"]["viewer"], revision["artifacts"])
    preview = write_revision(job_dir, revision)
    document = {"schema": "dev.splatlab.edit-proposal/v1", "proposal_id": proposal_id,
                "base": pointer, "preview_revision": preview["revision_id"], "operation": revision["operation"],
                "instruction": instruction[:2000], "asset": asset_identity, "created_at": manifests.utc_now(),
                "review": {"status": "needs-review", "visual_collision_same_revision": True,
                           "multi_view_appearance_accepted": False, "navigation_accepted": False}}
    document["sha256"] = hashlib.sha256(canonical_bytes(document)).hexdigest()
    durable_json(job_dir, root(job_dir) / "proposals" / f"{proposal_id}.json", document)
    return document


def verify_artifacts(job_dir: Path, revision: dict) -> None:
    for record in revision["artifacts"].values():
        path = blob_path(job_dir, record["sha256"])
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"] or manifests.sha256_file(path) != record["sha256"]:
            raise SceneRevisionError("Scene artifact is missing or corrupt; activation refused")


def activate(job_dir: Path, proposal_id: str, expected_generation: int) -> dict:
    with write_lock(job_dir):
        pointer = active(job_dir)
        proposal = read_proposal(job_dir, proposal_id)
        if not pointer or pointer != proposal["base"] or pointer["generation"] != expected_generation:
            raise SceneRevisionError("Proposal is stale; active scene changed")
        revision = read_revision(job_dir, proposal["preview_revision"])
        if source_fingerprint(job_dir) != revision["source_fingerprint"]:
            raise SceneRevisionError("Capture, selection or calibration changed; proposal cannot apply")
        verify_artifacts(job_dir, revision)
        verify_recovery_dependencies(job_dir, revision)
        if source_fingerprint(job_dir) != revision["source_fingerprint"]:
            raise SceneRevisionError("Scene sources changed during verification; activation refused")
        next_pointer = {"revision_id": revision["revision_id"], "generation": pointer["generation"] + 1, "proposal_id": proposal_id}
        durable_json(job_dir, root(job_dir) / "active.json", next_pointer)
        return next_pointer


def restore(job_dir: Path, revision_id: str, expected_generation: int) -> dict:
    with write_lock(job_dir):
        pointer = active(job_dir)
        if not pointer or pointer["generation"] != expected_generation:
            raise SceneRevisionError("Active scene changed; refresh before undo")
        if revision_id not in ancestry(job_dir, pointer["revision_id"]):
            raise SceneRevisionError("Undo can only restore a previously activated scene, not an unreviewed proposal")
        target = read_revision(job_dir, revision_id)
        verify_artifacts(job_dir, target)
        payload = {key: value for key, value in target.items() if key not in {"revision_id", "content_sha256"}}
        payload.update(parent=pointer["revision_id"], created_at=manifests.utc_now(), operation={"kind": "restore", "revision_id": revision_id})
        restored = write_revision(job_dir, payload)
        next_pointer = {"revision_id": restored["revision_id"], "generation": pointer["generation"] + 1, "proposal_id": None}
        durable_json(job_dir, root(job_dir) / "active.json", next_pointer)
        return next_pointer


def artifact(job_dir: Path, revision_id: str, key: str) -> Path:
    document = read_revision(job_dir, revision_id)
    record = document["artifacts"].get(key_checked(key))
    if not record:
        raise SceneRevisionError("Artifact is not part of this scene revision")
    path = blob_path(job_dir, record["sha256"])
    if not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"]:
        raise SceneRevisionError("Scene blob is unavailable")
    return path
