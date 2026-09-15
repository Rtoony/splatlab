"""Authenticated entry points for revision-pinned creative scenes."""

import asyncio
import mimetypes
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

import artifact_manifest as manifests
import background_recovery
import background_completion
import generated_objects
import generated_gaussians
from reconstruction_evidence import EvidenceError
import polish_route
import scene_revisions as revisions
import selection_reviews
import selection_collision
import support_surfaces
import structural_surfaces
import architectural_edits
import splat_route

router = APIRouter()


def require_job(job_id: str) -> Path:
    _, job_dir = polish_route._require_job(job_id)
    return job_dir.resolve()


async def execute(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except (revisions.SceneRevisionError, EvidenceError) as exc:
        raise HTTPException(409, str(exc)) from exc


class BackgroundPlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
    label: str = Field(default="", max_length=80)
    recovery_id: str | None = Field(default=None, pattern=r"^recovery_[a-f0-9]{24}$")
    completion_id: str | None = Field(default=None, pattern=r"^completion_[a-f0-9]{24}$")


class EditOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["place", "remove", "replace", "extend-room"]
    architecture_id: str | None = Field(default=None, pattern=r"^architecture_[a-f0-9]{24}$")
    selected_slug: str | None = Field(default=None, min_length=1, max_length=40)
    slug: str | None = Field(default=None, min_length=1, max_length=40)
    blender_export: str | None = Field(default=None, max_length=120)
    recovery_id: str | None = Field(default=None, pattern=r"^recovery_[a-f0-9]{24}$")
    completion_id: str | None = Field(default=None, pattern=r"^completion_[a-f0-9]{24}$")
    selection_collision_id: str | None = Field(default=None, pattern=r"^collision_[a-f0-9]{24}$")
    generated_object_id: str | None = Field(default=None, pattern=r"^generated_[a-f0-9]{24}$")
    gaussians_id: str | None = Field(default=None, pattern=r"^gaussians_[a-f0-9]{24}$")
    background: BackgroundPlacement | None = None
    label: str = Field(default="", max_length=80)


class ProposalBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_generation: int = Field(ge=0)
    instruction: str = Field(min_length=1, max_length=2000)
    operation: EditOperation


class ActivateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_generation: int = Field(ge=0)
    reviewed: Literal[True]


class GenerationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_generation: int = Field(ge=0)


class ArchitecturalBody(GenerationBody):
    instruction: str = Field(min_length=1, max_length=2000)
    spec: dict
    replaced_slugs: list[str] = Field(default_factory=list, max_length=3)
    geometry_method: Literal["nominal-floor/v1", "raised-floor-finish/v1", "joined-floor-envelope/v1"] = "nominal-floor/v1"


class RestoreBody(GenerationBody):
    revision_id: str


class SupportAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    point_id: str = Field(pattern=r"^[0-9]{1,20}$")
    image_id: int = Field(gt=0, strict=True)
    photo_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class SupportAnchors(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    points: list[SupportAnchor] = Field(min_length=3, max_length=6)


class RecoveryBody(GenerationBody):
    selected_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
    texture_size: Literal[64, 128, 256] = 128
    support_anchors: SupportAnchors | None = None


class CompletionBody(GenerationBody):
    recovery_id: str = Field(pattern=r"^recovery_[a-f0-9]{24}$")
    prompt: str = Field(min_length=1, max_length=2000)


class GeneratedObjectBody(GenerationBody):
    selection_review_id: str = Field(pattern=r"^selection_[a-f0-9]{24}$")
    recovery_id: str = Field(pattern=r"^recovery_[a-f0-9]{24}$")
    image_id: int | None = Field(default=None, gt=0)
    seed: int = Field(default=42, ge=0, lt=2 ** 31, strict=True)


class SelectionReviewBody(GenerationBody):
    selected_slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")


class SelectionCollisionBody(GenerationBody):
    selection_review_id: str = Field(pattern=r"^selection_[a-f0-9]{24}$")


def rewrite_urls(value, job_id: str, revision_id: str):
    if isinstance(value, dict):
        return {key: rewrite_urls(item, job_id, revision_id) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_urls(item, job_id, revision_id) for item in value]
    if isinstance(value, str) and value.startswith("artifact:"):
        return artifact_url(job_id, revision_id, value[len("artifact:"):])
    return value


def artifact_url(job_id: str, revision_id: str, key: str) -> str:
    return f"/api/splat/jobs/{job_id}/studio/revisions/{revision_id}/artifact?key={quote(key, safe='')}"


@router.post("/jobs/{job_id}/studio/initialize")
async def initialize_studio(job_id: str):
    job_dir = require_job(job_id)
    before = await execute(revisions.source_fingerprint, job_dir)
    viewer = await splat_route.get_splat_world_manifest(job_id)
    return await execute(revisions.initialize, job_dir, viewer, before)


@router.get("/jobs/{job_id}/studio")
async def get_studio(job_id: str):
    job_dir = require_job(job_id)
    pointer = await execute(revisions.active, job_dir)
    revision = await execute(revisions.read_revision, job_dir, pointer["revision_id"]) if pointer else None
    ancestors = await execute(revisions.ancestry, job_dir, pointer["revision_id"]) if pointer else set()
    history = []
    sources = await execute(revisions.source_fingerprint, job_dir)
    documents = {}
    for path in sorted((revisions.root(job_dir) / "revisions").glob("scene_*.json")):
        document = await execute(revisions.read_revision, job_dir, path.stem)
        documents[document["revision_id"]] = document
        history.append({**{key: document[key] for key in ("revision_id", "parent", "created_at", "operation")}, "activated": document["revision_id"] in ancestors})
    proposals = []
    for path in sorted((revisions.root(job_dir) / "proposals").glob("edit_*.json")):
        proposal = await execute(revisions.read_proposal, job_dir, path.stem)
        if proposal["preview_revision"] not in ancestors:
            preview = documents.get(proposal["preview_revision"])
            proposals.append({**proposal, "stale": proposal["base"] != pointer or not preview or preview["source_fingerprint"] != sources})
    exports = []
    for path in sorted((job_dir / "_blender" / "exports").glob("scene-v*.glb")):
        if revisions.EXPORT_RE.fullmatch(path.name):
            receipt = manifests.read_json(path.with_suffix(".json")) or {}
            if manifests.same_file_identity(path, receipt.get("output")):
                exports.append({"filename": path.name, "bytes": path.stat().st_size})
    return {"active": pointer, "history": sorted(history, key=lambda item: item["created_at"]),
            "proposals": sorted(proposals, key=lambda item: item["created_at"]),
            "blender_exports": exports, "legacy_sources_changed": bool(revision and revision["source_fingerprint"] != sources),
            "state": revision["state"] if revision else None}


@router.post("/jobs/{job_id}/studio/refresh-proposal")
async def refresh_baseline(job_id: str, body: GenerationBody):
    job_dir = require_job(job_id)
    before = await execute(revisions.source_fingerprint, job_dir)
    viewer = await splat_route.get_splat_world_manifest(job_id)
    return await execute(revisions.refresh_proposal, job_dir, viewer, before, body.expected_generation)


@router.get("/jobs/{job_id}/studio/recoveries")
async def list_recoveries(job_id: str):
    job_dir = require_job(job_id)
    pointer = await execute(revisions.active, job_dir)
    items = []
    for path in (revisions.root(job_dir) / "recoveries").glob("recovery_*/receipt.json"):
        receipt = await execute(background_recovery.read, job_dir, path.parent.name)
        stale = receipt["base"] != pointer or receipt.get("render_vr_only") is not True
        if not stale:
            try:
                await execute(background_recovery.verify_receipt_sources, job_dir, receipt)
            except HTTPException:
                stale = True
        items.append({"recovery_id": receipt["recovery_id"], "selected_slug": receipt["selected_slug"],
                      "created_at": receipt["created_at"], "report": receipt["report"], "evidence": receipt["evidence"], "stale": stale})
    return {"recoveries": sorted(items, key=lambda item: item["created_at"])}


@router.post("/jobs/{job_id}/studio/architectural-edits")
async def prepare_architectural_edit(job_id: str, body: ArchitecturalBody):
    return await execute(architectural_edits.prepare, require_job(job_id), body.spec, body.expected_generation, body.instruction,
                         body.replaced_slugs, selected_geometry=body.geometry_method)


@router.get("/jobs/{job_id}/studio/architectural-edits")
async def list_architectural_edits(job_id: str):
    job = require_job(job_id)
    items = []
    for path in (revisions.root(job) / "architectural-edits").glob("architecture_*/receipt.json"):
        receipt = await execute(architectural_edits.read, job, path.parent.name)
        stale = False
        try:
            await execute(architectural_edits.verify, job, receipt)
        except HTTPException:
            stale = True
        result = await execute(architectural_edits.read, job, path.parent.name, True) if (path.parent / "result.json").is_file() else None
        items.append({key: receipt[key] for key in ("architecture_id", "base", "created_at", "spec", "clip", "instruction", "scope", "replaced_elements")}
                     | {"geometry_method": receipt.get("geometry_method", architectural_edits.LEGACY_GEOMETRY),
                        "stale": stale, "result": {key: value for key, value in result.items() if key != "artifacts"} if result else None})
    return {"edits": sorted(items, key=lambda item: item["created_at"])}


@router.get("/jobs/{job_id}/studio/architectural-edits/{architecture_id}/artifact")
async def architectural_artifact(job_id: str, architecture_id: str, name: str, result: bool = False):
    path = await execute(architectural_edits.artifact, require_job(job_id), architecture_id, name, result)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        filename=Path(name).name, content_disposition_type="inline")


@router.get("/jobs/{job_id}/studio/structural-surfaces")
async def list_structural_surfaces(job_id: str):
    job = require_job(job_id)
    items = []
    for path in (revisions.root(job) / "structural-surfaces").glob("structure_*/receipt.json"):
        receipt = await execute(structural_surfaces.read, job, path.parent.name)
        stale = False
        try:
            await execute(structural_surfaces.verify, job, receipt)
        except HTTPException:
            stale = True
        result = await execute(structural_surfaces.read, job, path.parent.name, "result") if (path.parent / "result.json").is_file() else None
        items.append({key: receipt[key] for key in ("structure_id", "base", "created_at", "prompt", "cameras", "scope")}
                     | {"stale": stale, "result": {key: value for key, value in result.items() if key != "artifacts"} if result else None})
    return {"studies": sorted(items, key=lambda item: item["created_at"])}


@router.get("/jobs/{job_id}/studio/structural-surfaces/{structure_id}/artifact")
async def structural_surface_artifact(job_id: str, structure_id: str, name: str,
                                      stage: Literal["receipt", "masks", "result"] = "receipt"):
    path = await execute(structural_surfaces.artifact, require_job(job_id), structure_id, name, stage)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        filename=Path(name).name, content_disposition_type="inline")


@router.get("/jobs/{job_id}/studio/generated-objects")
async def list_generated_objects(job_id: str):
    job = require_job(job_id)
    items = []
    for path in (revisions.root(job) / "generated-objects").glob("generated_*/receipt.json"):
        receipt = await execute(generated_objects.read, job, path.parent.name)
        stale = False
        try:
            await execute(generated_objects.verify, job, receipt)
        except HTTPException:
            stale = True
        result = await execute(generated_objects.read, job, path.parent.name, True) if (path.parent / "result.json").is_file() else None
        items.append({key: receipt[key] for key in ("generated_object_id", "base", "selected_slug", "created_at", "input_image_id", "recovery_id", "selection_review_id", "scope")}
                     | {"stale": stale, "result": {key: value for key, value in result.items() if key not in {"artifacts", "model"}} if result else None})
    return {"objects": sorted(items, key=lambda item: item["created_at"])}


@router.post("/jobs/{job_id}/studio/generated-objects")
async def prepare_generated_object(job_id: str, body: GeneratedObjectBody):
    return await execute(generated_objects.prepare, require_job(job_id), body.selection_review_id, body.recovery_id,
                         body.expected_generation, body.image_id, body.seed)


@router.get("/jobs/{job_id}/studio/generated-objects/{identifier}")
async def generated_object_receipt(job_id: str, identifier: str, result: bool = False):
    return await execute(generated_objects.read, require_job(job_id), identifier, result)


@router.get("/jobs/{job_id}/studio/generated-objects/{identifier}/artifact")
async def generated_object_artifact(job_id: str, identifier: str, name: str, result: bool = False):
    path = await execute(generated_objects.artifact, require_job(job_id), identifier, name, result)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.get("/jobs/{job_id}/studio/generated-objects/{identifier}/gaussians")
async def list_generated_gaussians(job_id: str, identifier: str):
    job = require_job(job_id)
    await execute(generated_objects.read, job, identifier)
    items = []
    for path in (generated_objects.directory(job, identifier) / "gaussians").glob("gaussians_*/result.json"):
        result = await execute(generated_gaussians.read, job, identifier, path.parent.name)
        items.append({key: value for key, value in result.items() if key != "artifacts"})
    return {"reviews": sorted(items, key=lambda item: item["created_at"])}


@router.get("/jobs/{job_id}/studio/generated-objects/{identifier}/gaussians/{review_id}")
async def generated_gaussian_receipt(job_id: str, identifier: str, review_id: str):
    return await execute(generated_gaussians.read, require_job(job_id), identifier, review_id)


@router.get("/jobs/{job_id}/studio/generated-objects/{identifier}/gaussians/{review_id}/artifact")
async def generated_gaussian_artifact(job_id: str, identifier: str, review_id: str, name: str):
    path = await execute(generated_gaussians.artifact, require_job(job_id), identifier, review_id, name)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.post("/jobs/{job_id}/studio/recoveries")
async def build_recovery(job_id: str, body: RecoveryBody):
    anchors = body.support_anchors.model_dump() if body.support_anchors else None
    return await execute(background_recovery.build, require_job(job_id), body.selected_slug, body.expected_generation, body.texture_size, anchors)


@router.get("/jobs/{job_id}/studio/recovery-support")
async def support_observations(job_id: str, selected_slug: str = Query(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$"),
                               expected_generation: int = Query(ge=0), image_id: int | None = Query(default=None, gt=0)):
    return await execute(support_surfaces.inspect, require_job(job_id), selected_slug, expected_generation, image_id)


@router.get("/jobs/{job_id}/studio/recovery-support/{image_id}/photo")
async def support_photo(job_id: str, image_id: int, selected_slug: str = Query(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$"),
                        expected_generation: int = Query(ge=0), photo_sha256: str = Query(pattern=r"^[a-f0-9]{64}$")):
    path = await execute(support_surfaces.photo, require_job(job_id), selected_slug, expected_generation, image_id, photo_sha256)
    return FileResponse(path, headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.get("/jobs/{job_id}/studio/completions")
async def list_completions(job_id: str):
    job = require_job(job_id)
    items = []
    for path in (revisions.root(job) / "completions").glob("completion_*/receipt.json"):
        receipt = await execute(background_completion.read, job, path.parent.name)
        stale = False
        try:
            await execute(background_completion.verify, job, receipt)
        except HTTPException:
            stale = True
        result = await execute(background_completion.read, job, path.parent.name, True) if (path.parent / "result.json").is_file() else None
        items.append({key: receipt[key] for key in ("completion_id", "recovery_id", "selected_slug", "created_at", "prompt", "observed_fraction", "generated_fraction", "scope")}
                     | {"stale": stale, "result": {key: value for key, value in result.items() if key != "artifacts"} if result else None})
    return {"completions": sorted(items, key=lambda item: item["created_at"])}


@router.post("/jobs/{job_id}/studio/completions")
async def prepare_completion(job_id: str, body: CompletionBody):
    return await execute(background_completion.prepare, require_job(job_id), body.recovery_id, body.expected_generation, body.prompt)


@router.post("/jobs/{job_id}/studio/completions/{completion_id}/image")
async def import_completion_image(job_id: str, completion_id: str, file: UploadFile = File(...),
                                  provider: str = Form(min_length=1, max_length=120), model: str = Form(min_length=1, max_length=120)):
    content = await file.read(background_completion.MAX_IMAGE_BYTES + 1)
    if len(content) > background_completion.MAX_IMAGE_BYTES:
        raise HTTPException(413, "Generated material exceeds 16 MiB")
    return await execute(background_completion.import_image, require_job(job_id), completion_id, content, provider, model)


@router.get("/jobs/{job_id}/studio/completions/{completion_id}/artifact")
async def completion_artifact(job_id: str, completion_id: str, name: str, result: bool = False):
    path = await execute(background_completion.artifact, require_job(job_id), completion_id, name, result)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.get("/jobs/{job_id}/studio/selection-reviews")
async def list_selection_reviews(job_id: str):
    job = require_job(job_id)
    items = []
    for path in (revisions.root(job) / "selection-reviews").glob("selection_*/receipt.json"):
        receipt = await execute(selection_reviews.read, job, path.parent.name)
        stale = False
        try:
            await execute(selection_reviews.verify, job, receipt, False)
        except HTTPException:
            stale = True
        result = await execute(selection_reviews.read, job, path.parent.name, True) if (path.parent / "result.json").is_file() else None
        items.append({key: receipt[key] for key in ("review_id", "base", "label", "selected_slug", "created_at", "core_count", "cameras", "scope")}
                     | {"stale": stale, "result": {key: value for key, value in result.items() if key != "artifacts"} if result else None})
    return {"reviews": sorted(items, key=lambda item: item["created_at"])}


@router.post("/jobs/{job_id}/studio/selection-reviews")
async def prepare_selection_review(job_id: str, body: SelectionReviewBody):
    return await execute(selection_reviews.prepare, require_job(job_id), body.selected_slug, body.expected_generation)


@router.get("/jobs/{job_id}/studio/selection-collisions")
async def list_selection_collisions(job_id: str):
    job = require_job(job_id)
    items = []
    for path in (revisions.root(job) / "selection-collisions").glob("collision_*/receipt.json"):
        receipt = await execute(selection_collision.read, job, path.parent.name)
        stale = False
        try:
            await execute(selection_collision.verify, job, receipt)
        except HTTPException:
            stale = True
        result = await execute(selection_collision.read, job, path.parent.name, True) if (path.parent / "result.json").is_file() else None
        items.append({key: receipt[key] for key in ("collision_id", "selection_review_id", "selected_slug", "created_at", "counts", "recipe")}
                     | {"stale": stale, "result": {key: value for key, value in result.items() if key != "artifacts"} if result else None})
    return {"collisions": sorted(items, key=lambda item: item["created_at"])}


@router.post("/jobs/{job_id}/studio/selection-collisions")
async def prepare_selection_collision(job_id: str, body: SelectionCollisionBody):
    return await execute(selection_collision.prepare, require_job(job_id), body.selection_review_id, body.expected_generation)


@router.get("/jobs/{job_id}/studio/selection-collisions/{collision_id}/artifact")
async def selection_collision_artifact(job_id: str, collision_id: str, name: str):
    path = await execute(selection_collision.artifact, require_job(job_id), collision_id, name, True)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.get("/jobs/{job_id}/studio/selection-reviews/{review_id}/artifact")
async def selection_review_artifact(job_id: str, review_id: str, name: str):
    path = await execute(selection_reviews.artifact, require_job(job_id), review_id, name)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.get("/jobs/{job_id}/studio/selection-reviews/{review_id}/rows")
async def selection_review_rows(job_id: str, review_id: str, candidate: bool = False):
    import numpy as np

    job = require_job(job_id)
    receipt = await execute(selection_reviews.read, job, review_id)
    await execute(selection_reviews.verify, job, receipt)
    name = "candidate-rows.npz" if candidate else "rows.npz"
    path = await execute(selection_reviews.artifact, job, review_id, name)
    with np.load(path, allow_pickle=False) as data:
        rows = data["candidate" if candidate else "core"]
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not 0 < len(rows) <= 200_000 or np.any(rows < 0) or np.any(rows >= receipt["n_rows"]):
        raise HTTPException(409, "Selection rows exceed the bounded preview contract")
    return {"review_id": review_id, "n_rows": receipt["n_rows"], "rows": rows.tolist(), "base": receipt["base"], "selected_slug": receipt["selected_slug"]}


@router.get("/jobs/{job_id}/studio/recoveries/{recovery_id}/artifacts/{name}")
async def recovery_artifact(job_id: str, recovery_id: str, name: str):
    path = await execute(background_recovery.artifact, require_job(job_id), recovery_id, name)
    return FileResponse(path, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})


@router.post("/jobs/{job_id}/studio/proposals")
async def propose_edit(job_id: str, body: ProposalBody):
    return await execute(revisions.propose, require_job(job_id), body.operation.model_dump(exclude_none=True), body.instruction, body.expected_generation)


@router.get("/jobs/{job_id}/studio/proposals/{proposal_id}")
async def get_proposal(job_id: str, proposal_id: str):
    return await execute(revisions.read_proposal, require_job(job_id), proposal_id)


@router.post("/jobs/{job_id}/studio/proposals/{proposal_id}/apply")
async def apply_edit(job_id: str, proposal_id: str, body: ActivateBody):
    return await execute(revisions.activate, require_job(job_id), proposal_id, body.expected_generation)


@router.post("/jobs/{job_id}/studio/restore")
async def restore_scene(job_id: str, body: RestoreBody):
    return await execute(revisions.restore, require_job(job_id), body.revision_id, body.expected_generation)


@router.get("/jobs/{job_id}/studio/revisions/{revision_id}")
async def get_revision(job_id: str, revision_id: str):
    document = await execute(revisions.read_revision, require_job(job_id), revision_id)
    state = document["state"]
    backdrop = next((key for key in ("_preview/langweb.ply", "_preview/splat.ply") if key in document["artifacts"]), None)
    return {"revision_id": revision_id, "parent": document["parent"], "state": state,
            "viewer": rewrite_urls(state["viewer"], job_id, revision_id),
            "backdrop_meters_per_unit": state["viewer"].get("calibration", {}).get("meters_per_unit", state["calibration"]["meters_per_unit"]) if state["viewer"].get("units") == "meters" else None,
            "backdrop_url": artifact_url(job_id, revision_id, backdrop) if backdrop else None}


@router.get("/jobs/{job_id}/studio/revisions/{revision_id}/artifact")
async def get_artifact(job_id: str, revision_id: str, key: str):
    path = await execute(revisions.artifact, require_job(job_id), revision_id, key)
    return FileResponse(path, media_type=mimetypes.guess_type(key)[0] or "application/octet-stream",
                        headers={"Cache-Control": "private, max-age=31536000, immutable"})
