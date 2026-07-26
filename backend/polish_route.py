"""The polish round-trip: externally-edited GLBs land back in the app.

The doctrine is "Blender/UE are polish stations, three.js stays the runtime" —
but until this route existed the pipeline was one-way: assets could be exported
(UE bundle, Blender workflow) and never returned. These two routes are the one
audited ingest door back in:

  POST /jobs/{id}/objects/{slug}/polish        -> _objects/<slug>/mesh/polished.glb
  POST /jobs/{id}/world/elements/{slug}/polish -> _world/elements/<slug>.glb
                                                  (slug "shell" -> _world/shell.glb)

Discipline mirrors edit_ops.upload_edited_ply exactly: stream to a .building-*
tmp, validate BEFORE touching anything live (glb_check — stdlib, cannot crash
the process the way Open3D can), then under the per-job mesh lock: version the
prior file, os.replace atomically, write a provenance receipt. Every rejection
(400/404/409/413) leaves the tree untouched.

World replacements need no walker change — world-walker fetches elements by
relative name, so the replaced file serves on the next load; a `.polish.json`
marker beside the element records provenance and that the old sidecar atlas no
longer describes the file (a polished GLB embeds its own textures).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, UploadFile

import artifact_manifest as manifests
import glb_check
import splat_route
from operator_audit import audit_operator_event

router = APIRouter()

POLISH_SCHEMA = "dev.splatlab.polish-receipt/v1"


def _require_job(job_id: str) -> tuple[dict[str, Any], Path]:
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = splat_route._read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return meta, Path(meta["output_dir"])


async def _stream_glb_to_tmp(file: UploadFile, tmp: Path) -> int:
    filename = (file.filename or "").strip() or "upload.glb"
    if not filename.lower().endswith(".glb"):
        raise HTTPException(
            status_code=400,
            detail=f"'{filename}' is not a .glb file — export a binary glTF "
            "(GLB) from your DCC tool and upload that",
        )
    written = 0
    try:
        with tmp.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > splat_route.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Upload exceeds 2 GB."
                    )
                handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - client disconnects / disk errors
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    if written == 0:
        raise HTTPException(
            status_code=400, detail=f"'{filename}' is empty — nothing was uploaded"
        )
    return written


def _validate_glb_or_400(tmp: Path, filename: str) -> dict[str, Any]:
    try:
        return glb_check.validate_glb(tmp)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"'{filename}' is not a usable GLB: {exc}"
        ) from exc


def _next_version_path(versions_dir: Path, stem: str) -> Path:
    versions_dir.mkdir(parents=True, exist_ok=True)
    seq = 1
    for existing in versions_dir.glob(f"{stem}-v*.glb"):
        tail = existing.stem.rsplit("-v", 1)
        if len(tail) == 2 and tail[1].isdigit():
            seq = max(seq, int(tail[1]) + 1)
    return versions_dir / f"{stem}-v{seq:04d}.glb"


def _land_polish(
    *,
    tmp: Path,
    target: Path,
    versions_dir: Path,
    version_stem: str,
    receipt_path: Path,
    filename: str,
    written: int,
    summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    supersedes = (
        manifests.file_identity(target) if target.is_file() else None
    )
    if target.is_file():
        os.replace(target, _next_version_path(versions_dir, version_stem))
    os.replace(tmp, target)
    receipt = {
        "schema": POLISH_SCHEMA,
        "provenance": "polished",
        "source_filename": filename,
        "uploaded_bytes": written,
        "gltf": summary,
        "uploaded_at": manifests.utc_now(),
        "supersedes": supersedes,
        **(extra or {}),
        **manifests.file_identity(target),
    }
    manifests.atomic_write_json(receipt_path, receipt)
    return receipt


@router.post("/jobs/{job_id}/objects/{slug}/polish")
async def upload_polished_object(
    job_id: str, slug: str, file: UploadFile, request: Request
) -> dict[str, Any]:
    """Land an externally-polished GLB as this object's `polished` asset.

    Non-destructive: the captured/textured artifacts stay untouched; a prior
    polished.glb is versioned to mesh/versions/ before the new one lands."""
    meta, job_dir = _require_job(job_id)
    if splat_route._object_slug(slug) != slug:
        raise HTTPException(status_code=404, detail="No such object")
    obj_dir = job_dir / "_objects" / slug
    if not (obj_dir / "object.json").is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No object '{slug}' for this scene — isolate it first "
            "(POST /objects), then polish the result",
        )
    mesh_dir = obj_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    filename = (file.filename or "").strip() or "upload.glb"
    tmp = mesh_dir / f".building-polish-{uuid.uuid4().hex}.glb"
    try:
        written = await _stream_glb_to_tmp(file, tmp)
        summary = _validate_glb_or_400(tmp, filename)
        lock = splat_route._mesh_export_lock(job_id)
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="A mesh build or export is running for this scene — retry "
                "when it finishes",
            )
        async with lock:
            receipt = _land_polish(
                tmp=tmp,
                target=mesh_dir / "polished.glb",
                versions_dir=mesh_dir / "versions",
                version_stem="polished",
                receipt_path=mesh_dir / "polished.json",
                filename=filename,
                written=written,
                summary=summary,
            )
    finally:
        tmp.unlink(missing_ok=True)

    await audit_operator_event(
        request=request,
        title="Polished object asset uploaded",
        description=f"{job_id}/{slug}: {filename} ({written} bytes)",
        variant="success",
        action="splat.polish_object",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug, "bytes": written},
    )
    return {
        "ok": True,
        "slug": slug,
        "url": f"/api/splat/jobs/{job_id}/objects/{slug}/file?fmt=polished",
        "receipt": receipt,
    }


@router.post("/jobs/{job_id}/world/elements/{slug}/polish")
async def upload_polished_world_element(
    job_id: str, slug: str, file: UploadFile, request: Request
) -> dict[str, Any]:
    """Replace a built world element (or `shell`) with a polished GLB.

    Polish REPLACES a built artifact — it never invents one, so a slug that
    was not solidified 404s. The prior file is versioned to _world/versions/;
    the walker picks the new file up by name on its next load."""
    meta, job_dir = _require_job(job_id)
    world_dir = job_dir / splat_route.WORLD_DIRNAME
    if slug == "shell":
        target = world_dir / "shell.glb"
        marker = world_dir / "shell.polish.json"
    else:
        if splat_route._object_slug(slug) != slug:
            raise HTTPException(status_code=404, detail="No such world element")
        target = world_dir / "elements" / f"{slug}.glb"
        marker = world_dir / "elements" / f"{slug}.polish.json"
    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No built world element '{slug}' — polish replaces a "
            "solidified artifact, it does not create one",
        )
    filename = (file.filename or "").strip() or "upload.glb"
    tmp = world_dir / f".building-polish-{uuid.uuid4().hex}.glb"
    try:
        written = await _stream_glb_to_tmp(file, tmp)
        summary = _validate_glb_or_400(tmp, filename)
        lock = splat_route._mesh_export_lock(job_id)
        if lock.locked():
            raise HTTPException(
                status_code=409,
                detail="A mesh build or export is running for this scene — retry "
                "when it finishes",
            )
        async with lock:
            receipt = _land_polish(
                tmp=tmp,
                target=target,
                versions_dir=world_dir / "versions",
                version_stem=slug,
                receipt_path=marker,
                filename=filename,
                written=written,
                summary=summary,
                # The old sidecar atlas described the previous bake; a polished
                # GLB embeds its own textures, so consumers must not pair the
                # replaced file with the stale PNG.
                extra={"atlas_superseded": True},
            )
    finally:
        tmp.unlink(missing_ok=True)

    await audit_operator_event(
        request=request,
        title="Polished world element uploaded",
        description=f"{job_id}/{slug}: {filename} ({written} bytes)",
        variant="success",
        action="splat.polish_world_element",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "slug": slug, "bytes": written},
    )
    relative = "shell.glb" if slug == "shell" else f"elements/{slug}.glb"
    return {
        "ok": True,
        "slug": slug,
        "url": f"/api/splat/jobs/{job_id}/world/file?name={relative}",
        "receipt": receipt,
    }
