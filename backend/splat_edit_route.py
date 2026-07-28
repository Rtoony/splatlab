"""The audited splat-edit lane: versioned edits on the capture's gaussians.

Meshes got their round-trip discipline in R3 (the polish route); this gives
the SPLAT itself the same treatment. Edits never touch the live asset:
each POST /edit runs mesh/splat_edit.py (typed engine, attribute-preserving,
dn-splatter env) against the latest version — or the pristine
_preview/splat.ply — and lands an immutable _splat/versions/splat-vNNNN.ply
with a receipt. Promotion to the LIVE viewer asset is a separate, guarded
step: the pristine original is preserved on first promote, and index-keyed
consumers built against the original gaussian ORDER (the language field's
gauss_emb.npz, _scene isolated claims) make promotion refuse without an
explicit force acknowledgment — editing the splat under them silently
desynchronises every per-gaussian index they hold.

Discipline mirrors polish_route: stage to .building-*, validate BEFORE
touching anything live, land under the per-job mesh lock, receipts with
supersedes identities, operator-audit rows. Every rejection leaves the tree
untouched.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import artifact_manifest as manifests
import splat_route
from operator_audit import audit_operator_event

router = APIRouter()

EDIT_SCHEMA = "dev.splatlab.splat-edit-receipt/v1"
PROMOTE_SCHEMA = "dev.splatlab.splat-promote-receipt/v1"
VERSION_RE = re.compile(r"^splat-v(?P<version>\d{4})\.ply$")
ENGINE = Path(__file__).with_name("mesh") / "splat_edit.py"
ENGINE_PY = Path(
    os.environ.get(
        "SPLATLAB_MESH_PY",
        str(Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"),
    )
)
ENGINE_TIMEOUT_S = 10 * 60
OPS = ("crop_box", "clean", "transform")


class SplatEditBody(BaseModel):
    op: str
    params: dict[str, Any] = {}
    base_version: int | None = None
    note: str = ""


class SplatPromoteBody(BaseModel):
    version: int
    force: bool = False
    note: str = ""


def _require_job(job_id: str) -> tuple[dict[str, Any], Path]:
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = splat_route._read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return meta, Path(meta["output_dir"])


def _lane(job_dir: Path) -> Path:
    return job_dir / "_splat"


def _versions_dir(job_dir: Path) -> Path:
    return _lane(job_dir) / "versions"


def _list_versions(job_dir: Path) -> list[dict[str, Any]]:
    versions = []
    for path in sorted(_versions_dir(job_dir).glob("splat-v*.ply")):
        match = VERSION_RE.match(path.name)
        if not match:
            continue
        number = int(match.group("version"))
        receipt = manifests.read_json(
            _lane(job_dir) / "receipts" / f"splat-v{number:04d}.json"
        )
        versions.append({
            "version": number,
            "path": f"_splat/versions/{path.name}",
            "bytes": path.stat().st_size,
            "receipt": receipt,
        })
    return versions


def _source_ply(job_dir: Path, base_version: int | None) -> tuple[Path, int | None]:
    if base_version is not None:
        if not 1 <= base_version <= 9999:
            raise HTTPException(status_code=400,
                                detail="base_version must be 1-9999")
        path = _versions_dir(job_dir) / f"splat-v{base_version:04d}.ply"
        if not path.is_file():
            raise HTTPException(status_code=404,
                                detail=f"splat version {base_version} does not exist")
        return path, base_version
    versions = _list_versions(job_dir)
    if versions:
        latest = versions[-1]
        return job_dir / latest["path"], latest["version"]
    live = job_dir / "_preview" / "splat.ply"
    if not live.is_file():
        raise HTTPException(status_code=404,
                            detail="job has no _preview/splat.ply to edit")
    return live, None


def _number(params: dict[str, Any], key: str, low: float, high: float,
            *, required: bool = False) -> float | None:
    value = params.get(key)
    if value is None:
        if required:
            raise HTTPException(status_code=400, detail=f"{key} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=400, detail=f"{key} must be a number")
    if not low <= float(value) <= high:
        raise HTTPException(status_code=400,
                            detail=f"{key} must be within [{low}, {high}]")
    return float(value)


def _triple(params: dict[str, Any], key: str, *, required: bool = False) -> str | None:
    value = params.get(key)
    if value is None:
        if required:
            raise HTTPException(status_code=400, detail=f"{key} is required")
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, (int, float))
                   for item in value)):
        raise HTTPException(status_code=400,
                            detail=f"{key} must be three numbers")
    return ",".join(repr(float(item)) for item in value)


def _engine_args(op: str, params: dict[str, Any]) -> list[str]:
    """Server-side sanitization: only typed, bounded values reach the engine.

    Values that can start with '-' use --flag=value form (the argparse trap)."""
    args = ["--op", op]
    if op == "crop_box":
        args += [f"--min={_triple(params, 'min', required=True)}",
                 f"--max={_triple(params, 'max', required=True)}"]
    elif op == "clean":
        opacity = _number(params, "min_opacity", 1e-6, 0.999)
        scale = _number(params, "max_scale", 1e-6, 1e6)
        dist = _number(params, "max_dist", 1e-6, 1e9)
        if opacity is None and scale is None and dist is None:
            raise HTTPException(status_code=400,
                                detail="clean needs min_opacity, max_scale "
                                       "and/or max_dist")
        if opacity is not None:
            args += [f"--min-opacity={opacity}"]
        if scale is not None:
            args += [f"--max-scale={scale}"]
        if dist is not None:
            args += [f"--max-dist={dist}"]
    elif op == "transform":
        translate = _triple(params, "translate")
        scale = _number(params, "scale", 1e-6, 1e6)
        if translate is None and scale is None:
            raise HTTPException(status_code=400,
                                detail="transform needs translate and/or scale")
        if translate is not None:
            args += [f"--translate={translate}"]
        if scale is not None:
            args += [f"--scale={scale}"]
    else:
        raise HTTPException(status_code=400, detail=f"unknown op: {op}")
    return args


def _run_engine(source: Path, staged: Path, extra: list[str]) -> dict[str, Any]:
    if not ENGINE_PY.is_file():
        raise HTTPException(status_code=503,
                            detail="mesh env python is unavailable")
    command = [str(ENGINE_PY), str(ENGINE), str(source), str(staged)] + extra
    proc = subprocess.run(command, capture_output=True, text=True,
                          timeout=ENGINE_TIMEOUT_S)
    if proc.returncode != 0 or not staged.is_file():
        staged.unlink(missing_ok=True)
        tail = "\n".join((proc.stderr or "").splitlines()[-4:])
        raise HTTPException(status_code=422,
                            detail=f"splat edit failed: {tail[:400]}")
    try:
        import json as _json
        return _json.loads(proc.stdout.splitlines()[-1])
    except (ValueError, IndexError) as exc:
        staged.unlink(missing_ok=True)
        raise HTTPException(status_code=500,
                            detail="engine produced no report") from exc


@router.get("/jobs/{job_id}/splat/versions")
async def list_splat_versions(job_id: str) -> dict[str, Any]:
    _meta, job_dir = _require_job(job_id)
    original = _lane(job_dir) / "original.ply"
    return {
        "versions": _list_versions(job_dir),
        "original_preserved": original.is_file(),
        "promote": manifests.read_json(_lane(job_dir) / "promote.json"),
    }


@router.post("/jobs/{job_id}/splat/edit")
async def edit_splat(job_id: str, body: SplatEditBody, request: Request) -> dict[str, Any]:
    if body.op not in OPS:
        raise HTTPException(status_code=400,
                            detail=f"op must be one of {', '.join(OPS)}")
    meta, job_dir = _require_job(job_id)
    extra = _engine_args(body.op, body.params)

    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        raise HTTPException(status_code=409,
                            detail="A mesh build or export is running for this "
                                   "scene — retry when it finishes")
    async with lock:
        source, base = _source_ply(job_dir, body.base_version)
        versions_dir = _versions_dir(job_dir)
        versions_dir.mkdir(parents=True, exist_ok=True)
        existing = [v["version"] for v in _list_versions(job_dir)]
        number = (max(existing) + 1) if existing else 1
        final = versions_dir / f"splat-v{number:04d}.ply"
        staged = versions_dir / f".building-{uuid.uuid4().hex}.ply"
        try:
            report = await asyncio.to_thread(_run_engine, source, staged, extra)
            os.replace(staged, final)
        finally:
            staged.unlink(missing_ok=True)
        receipt = {
            "schema": EDIT_SCHEMA,
            "job_id": job_id,
            "version": number,
            "op": body.op,
            "note": body.note[:500],
            "engine": report,
            "base": {
                "version": base,
                "path": manifests.relative_job_path(source, job_dir),
                **manifests.file_identity(source),
            },
            "output": {
                "path": manifests.relative_job_path(final, job_dir),
                **manifests.file_identity(final),
            },
            "created_at": manifests.utc_now(),
        }
        receipts_dir = _lane(job_dir) / "receipts"
        receipts_dir.mkdir(parents=True, exist_ok=True)
        manifests.atomic_write_json(
            receipts_dir / f"splat-v{number:04d}.json", receipt
        )

    await audit_operator_event(
        request=request,
        title="Splat edit version created",
        description=f"{job_id}: {body.op} -> v{number:04d} "
        f"({report.get('out_count')} of {report.get('in_count')} gaussians kept)",
        variant="success",
        action="splat.edit",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "op": body.op, "version": number},
    )
    return {"ok": True, "version": number, "receipt": receipt}


def _dependent_index_consumers(job_dir: Path) -> list[str]:
    consumers = []
    langfield = job_dir / "_langfield"
    if langfield.is_dir() and any(langfield.iterdir()):
        consumers.append("_langfield (gauss_emb.npz is keyed by gaussian index)")
    if (job_dir / "gauss_emb.npz").is_file():
        consumers.append("gauss_emb.npz (keyed by gaussian index)")
    isolated = job_dir / "_scene" / "isolated"
    if isolated.is_dir() and any(isolated.iterdir()):
        consumers.append("_scene/isolated (per-object gaussian claims)")
    return consumers


@router.post("/jobs/{job_id}/splat/promote")
async def promote_splat_version(
    job_id: str, body: SplatPromoteBody, request: Request
) -> dict[str, Any]:
    """Replace the LIVE _preview/splat.ply with an edited version.

    The pristine original is preserved to _splat/original.ply on the first
    promote, so the pre-edit state is always one copy away. Index-keyed
    consumers refuse promotion without force: their per-gaussian indices
    were built against the original ordering and silently desynchronise."""
    meta, job_dir = _require_job(job_id)
    source = _versions_dir(job_dir) / f"splat-v{body.version:04d}.ply"
    if not source.is_file():
        raise HTTPException(status_code=404,
                            detail=f"splat version {body.version} does not exist")
    live = job_dir / "_preview" / "splat.ply"
    if not live.is_file():
        raise HTTPException(status_code=404,
                            detail="job has no live _preview/splat.ply")
    consumers = _dependent_index_consumers(job_dir)
    if consumers and not body.force:
        raise HTTPException(
            status_code=409,
            detail="index-keyed consumers exist and would silently "
                   f"desynchronise: {'; '.join(consumers)}. Re-promote with "
                   "force=true to acknowledge, then rebuild them.",
        )

    lock = splat_route._mesh_export_lock(job_id)
    if lock.locked():
        raise HTTPException(status_code=409,
                            detail="A mesh build or export is running for this "
                                   "scene — retry when it finishes")
    async with lock:
        original = _lane(job_dir) / "original.ply"
        preserved_now = False
        if not original.is_file():
            staged_orig = original.with_name(f".building-{original.name}")
            await asyncio.to_thread(__import__("shutil").copy2, live, staged_orig)
            os.replace(staged_orig, original)
            preserved_now = True
        supersedes = manifests.file_identity(live)
        staged = live.with_name(f".building-promote-{uuid.uuid4().hex}.ply")
        await asyncio.to_thread(__import__("shutil").copy2, source, staged)
        os.replace(staged, live)
        receipt = {
            "schema": PROMOTE_SCHEMA,
            "job_id": job_id,
            "promoted_version": body.version,
            "note": body.note[:500],
            "forced_over_consumers": consumers if body.force else [],
            "original_preserved_now": preserved_now,
            "supersedes": supersedes,
            "created_at": manifests.utc_now(),
            **manifests.file_identity(live),
        }
        manifests.atomic_write_json(_lane(job_dir) / "promote.json", receipt)

    await audit_operator_event(
        request=request,
        title="Splat version promoted to live",
        description=f"{job_id}: v{body.version:04d} -> _preview/splat.ply"
        + (f" (FORCED over: {'; '.join(consumers)})" if body.force and consumers else ""),
        variant="warning" if consumers else "success",
        action="splat.promote",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "version": body.version,
                  "forced": bool(body.force and consumers)},
    )
    return {"ok": True, "receipt": receipt}
