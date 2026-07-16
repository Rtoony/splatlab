"""Splat Lab — scene-editing backend (Wave 1, Lane 3 v1).

Wraps the installed `splat-transform` CLI (PlayCanvas, MIT) plus a pure-stdlib PLY
row-mask rewriter to let an operator clean up a trained splat (crop / filter /
decimate / floater-removal / cluster-isolation), merge several scenes into one, and
select-by-text (delete / isolate / extract) using the existing language-field
embeddings. Every destructive op snapshots the current `_preview/` artifacts into
`versions/` first, so `edit/revert` is real undo.

OWNERSHIP: this file + tests/test_edit_ops.py only. `main.py` mounts `router` under
/api/splat (auth applied there, same as splat_route.router) — this module never
gates its own routes and never touches main.py.

Runtime constraint (verified, not assumed): this backend's own venv
(splatlab/.venv, the one uvicorn actually runs under — confirmed via `ps aux`) has
NEITHER numpy NOR plyfile installed, and backend/requirements.txt is out of this
file's ownership so it can't be added. Every PLY read/write here is therefore pure
stdlib (struct + pathlib), header-driven and property-order-agnostic — it never
assumes which columns exist, only the total per-row byte width. Heavy work (SigLIP
encode, per-gaussian relevancy, floater/cluster GPU voxelization) stays fully
subprocessed/HTTP-delegated, matching splat_route.py's existing MASt3R/langfield
pattern: nothing heavy is ever imported into this process.

GEOMETRY GOTCHA (probe receipt, 2026-07-04): splat-transform's `-r x,y,z` does NOT
use the textbook right-handed Rx/Ry/Rz convention on every axis. Empirically
verified with a synthetic 1-vertex PLY (see PLAN appendix / STATUS.md): `-r 90,0,0`
and `-r 0,90,0` reproduce the STANDARD math rotation only with the angle NEGATED;
`-r 0,0,90` matches the standard convention directly (no sign flip). This file's
matrix-decomposition path (`_axis_rotation_flag`) bakes in that verified per-axis
correction — see the docstring there before touching the signs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import secrets
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

import gpu_arbiter
import splat_route
from operator_audit import audit_operator_event

router = APIRouter()

# ── tunables (env-overridable, matching splat_route.py's style) ────────────────
VERSIONS_DIRNAME = "versions"
MAX_VERSIONS = max(1, int(os.environ.get("SPLAT_EDIT_MAX_VERSIONS", "5") or "5"))
# Voxelization actions (-G/-D) are cheap relative to training/MASt3R/langfield, but
# they DO touch the GPU and must serialize on the shared 5090 like every other heavy
# lane. Small reserve; overridable if a huge scene needs more headroom.
EDIT_GPU_VRAM_MB = max(256, int(os.environ.get("SPLAT_EDIT_GPU_VRAM_MB", "2000") or "2000"))
# langweb.ply IS a preview artifact (client-side language heatmap splat) — it must
# ride along in snapshots/reverts or fmt=langweb serves a different generation than
# the reverted splat.ply.
SNAPSHOT_ARTIFACT_NAMES = ("splat.ply", "splat.spz", "web.ply", "langweb.ply", "thumb.webp")
_VERSION_DIR_RE = re.compile(r"^v(\d+)-\d{8}T\d{6}Z$")
_EDIT_TMP_SUFFIX = ".edit-tmp"


def _utc_stamp() -> str:
    """Compact UTC timestamp for version dir names: YYYYMMDDTHHMMSSZ."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _edit_tmp_path(target: Path) -> Path:
    """Unique per-invocation tmp sibling for an atomic tmp->Path.replace() write.
    The random token means two operations (even a leaked lock / crashed run) can
    never interleave on a shared fixed tmp name."""
    return target.with_name(f"{target.name}.{secrets.token_hex(4)}{_EDIT_TMP_SUFFIX}")


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


# ── per-job edit serialization ─────────────────────────────────────────────────
# One asyncio.Lock per job_id, held across EVERY mutating endpoint (apply /
# semantic / revert; merge holds the locks of all its source jobs). Acquisition is
# NON-BLOCKING: a second concurrent edit gets an immediate 409 rather than queueing
# behind a possibly-minutes-long transform. Single event loop -> plain dict is safe.
_EDIT_LOCKS: dict[str, asyncio.Lock] = {}


def _edit_lock(job_id: str) -> asyncio.Lock:
    lock = _EDIT_LOCKS.get(job_id)
    if lock is None:
        lock = _EDIT_LOCKS[job_id] = asyncio.Lock()
    return lock


@contextlib.asynccontextmanager
async def _hold_edit_locks(job_ids: list[str]):
    """Acquire the edit lock of every listed job or 409 immediately if ANY is busy.
    The locked()-check → acquire() sequence has no await between check and grab on
    an uncontended asyncio.Lock, so it cannot interleave on a single event loop."""
    acquired: list[asyncio.Lock] = []
    try:
        for jid in dict.fromkeys(job_ids):  # de-dupe, preserve order
            lock = _edit_lock(jid)
            if lock.locked():
                raise HTTPException(
                    status_code=409,
                    detail=f"an edit is already in progress for {jid}; retry when it finishes",
                )
            await lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in acquired:
            lock.release()


# =============================================================================
# Job/path safety — reuse the app's OWN id pattern + helpers rather than a second,
# possibly-diverging validation rule.
# =============================================================================


def _require_editable_job(job_id: str) -> tuple[dict[str, Any], Path]:
    """Resolve + validate a job for a destructive edit. Raises HTTPException on any
    problem (bad id, missing job, running job, no finished preview to edit)."""
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="not found")
    job_dir = splat_route._job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="job not found")
    meta = splat_route._read_meta(job_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="job metadata not found")
    if meta.get("status") in {"starting", "running"}:
        raise HTTPException(status_code=409, detail="job is actively running; wait for it to finish before editing")
    preview = splat_route._preview_file_path(job_dir)
    if not preview.is_file():
        raise HTTPException(status_code=409, detail="scene has no finished preview to edit yet")
    return meta, job_dir


def _new_job_id() -> str:
    return f"splat_{secrets.token_hex(5)}"


def _create_derived_job_dir() -> tuple[str, Path]:
    job_id = _new_job_id()
    job_dir = splat_route._job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_id, job_dir


def _write_derived_meta(
    job_id: str,
    job_dir: Path,
    *,
    source_type: str,
    parents: list[str],
    name: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a meta.json for a job dir that was never a training pipeline run (a
    merge or a semantic extract). Mirrors _new_meta's key set so every existing
    splat_route consumer (gallery listing, delete, stop, _job_payload) can read it
    without a special case."""
    now = splat_route._utc_now()
    meta: dict[str, Any] = {
        "job_id": job_id,
        "mode": "3d",
        "status": "completed",
        "stage": None,
        "stages_planned": [],
        "stages_completed": [source_type],
        "input_path": name.replace("/", "-")[:200],
        "output_dir": str(job_dir),
        "capture_format": "standard",
        "max_num_iterations": 0,
        "capture_mode": "standard",
        "source_type": source_type,
        "parents": parents,
        "command": [],
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "pid": None,
        "exit_code": 0,
        "error_message": None,
        "stop_requested": False,
        "pinned": False,
    }
    if extra:
        meta.update(extra)
    splat_route._write_meta(job_id, meta)


# =============================================================================
# 1. SNAPSHOT VERSIONING
# =============================================================================


def _versions_dir(job_dir: Path) -> Path:
    return job_dir / VERSIONS_DIRNAME


def _list_version_dirs(job_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return (dir, manifest) pairs, newest (highest seq) first. Corrupt/foreign
    entries are skipped rather than raised — a bad version dir must never break the
    versions list or block new edits."""
    vdir = _versions_dir(job_dir)
    if not vdir.is_dir():
        return []
    out: list[tuple[Path, dict[str, Any]]] = []
    for d in vdir.iterdir():
        if not d.is_dir() or not _VERSION_DIR_RE.match(d.name):
            continue
        try:
            manifest = json.loads((d / "manifest.json").read_text())
        except Exception:  # noqa: BLE001 - corrupt/missing manifest: skip, don't crash listing
            continue
        if not isinstance(manifest, dict) or "seq" not in manifest:
            continue
        out.append((d, manifest))
    out.sort(key=lambda t: t[1].get("seq", 0), reverse=True)
    return out


def _next_seq(job_dir: Path) -> int:
    existing = _list_version_dirs(job_dir)
    return (max((m.get("seq", 0) for _d, m in existing), default=0)) + 1


def _prune_versions(job_dir: Path) -> None:
    """Pinned-newest pruning: keep the newest MAX_VERSIONS, delete anything older."""
    versions = _list_version_dirs(job_dir)  # newest first
    for d, _m in versions[MAX_VERSIONS:]:
        shutil.rmtree(d, ignore_errors=True)


def _file_stat(path: Path) -> list[int] | None:
    """[size, mtime_ns] identity for cheap 'content unchanged?' snapshot dedupe
    (shutil.copy2 preserves mtime, so a restored/unchanged file compares equal)."""
    try:
        st = path.stat()
        return [st.st_size, st.st_mtime_ns]
    except OSError:
        return None


def _snapshot_preview(job_dir: Path, op: str, params: dict[str, Any]) -> tuple[Path, dict[str, Any], bool]:
    """Copy the CURRENT _preview/ artifacts into versions/v<seq>-<utc>/ before any
    destructive edit. Returns (version_dir, manifest, created).

    - DEDUPE: if the current splat.ply is byte-identical (size+mtime_ns) to the one
      saved in the NEWEST version, no new snapshot is made — the existing newest
      version is returned with created=False (so a caller must only discard a
      snapshot it actually created).
    - splat.ply itself is MANDATORY: if it exists but cannot be copied, the partial
      version dir is removed and the whole op aborts (HTTPException 500) — silently
      proceeding would leave a destructive edit with no undo. Secondary artifacts
      (.spz / web.ply / langweb.ply / thumb) stay best-effort.
    - Pruning is NOT done here: callers prune only AFTER the mutation succeeds, so
      a failing op can never push a good old version out of the cap.
    """
    preview_dir = splat_route._preview_dir_path(job_dir)
    cur_stat = _file_stat(preview_dir / "splat.ply")

    existing = _list_version_dirs(job_dir)  # newest first
    if existing and cur_stat is not None:
        newest_dir, newest_manifest = existing[0]
        if (
            "splat.ply" in newest_manifest.get("files", [])
            and newest_manifest.get("splat_ply_stat") == cur_stat
        ):
            return newest_dir, newest_manifest, False

    seq = _next_seq(job_dir)
    vdir = _versions_dir(job_dir) / f"v{seq}-{_utc_stamp()}"
    vdir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for name in SNAPSHOT_ARTIFACT_NAMES:
        src = preview_dir / name
        if src.is_file():
            try:
                shutil.copy2(src, vdir / name)
                saved.append(name)
            except OSError as exc:
                if name == "splat.ply":
                    # No backup of the file we're about to destroy => abort the op.
                    shutil.rmtree(vdir, ignore_errors=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"could not snapshot splat.ply before editing ({exc}); op aborted",
                    ) from exc
                # secondary artifacts stay best-effort
    manifest = {
        "seq": seq,
        "ts": splat_route._utc_now(),
        "op": op,
        "params": params,
        "files": saved,
        "splat_ply_stat": cur_stat,
    }
    (vdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return vdir, manifest, True


def _discard_snapshot(vdir: Path, created: bool) -> None:
    """Remove a snapshot made for an op that FAILED before mutating splat.ply —
    otherwise repeated failed attempts churn identical snapshots through the
    MAX_VERSIONS cap and evict the only good old version."""
    if created:
        shutil.rmtree(vdir, ignore_errors=True)


def _find_version(job_dir: Path, seq: int) -> tuple[Path, dict[str, Any]] | None:
    for d, m in _list_version_dirs(job_dir):
        if m.get("seq") == seq:
            return d, m
    return None


class RevertRequest(BaseModel):
    version: int = Field(gt=0)


@router.get("/jobs/{job_id}/edit/versions")
async def list_versions(job_id: str) -> dict[str, Any]:
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="not found")
    job_dir = splat_route._job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="job not found")
    versions = [m for _d, m in _list_version_dirs(job_dir)]
    return {"job_id": job_id, "versions": versions, "max_versions": MAX_VERSIONS}


@router.post("/jobs/{job_id}/edit/revert")
async def revert_version(job_id: str, req: RevertRequest, request: Request) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    meta, job_dir = _require_editable_job(job_id)
    found = _find_version(job_dir, req.version)
    if found is None:
        raise HTTPException(status_code=404, detail=f"version {req.version} not found")
    src_dir, _manifest = found

    async with _hold_edit_locks([job_id]):
        async def transaction() -> list[str]:
            # Snapshot and every live artifact replacement share one backup lease.
            snap_dir, _snap_manifest, snap_created = _snapshot_preview(
                job_dir, op="revert", params={"reverted_to": req.version}
            )
            preview_dir = splat_route._preview_dir_path(job_dir)
            preview_dir.mkdir(parents=True, exist_ok=True)
            restored: list[str] = []
            for name in SNAPSHOT_ARTIFACT_NAMES:
                src = src_dir / name
                dst = preview_dir / name
                if src.is_file():
                    tmp = _edit_tmp_path(dst)
                    try:
                        shutil.copy2(src, tmp)
                        tmp.replace(dst)
                    except OSError as exc:
                        tmp.unlink(missing_ok=True)
                        if name == "splat.ply":
                            _discard_snapshot(snap_dir, snap_created)
                            raise HTTPException(
                                status_code=500,
                                detail=f"failed to restore splat.ply from v{req.version}: {exc}",
                            ) from exc
                        continue
                    restored.append(name)
                elif dst.is_file():
                    with contextlib.suppress(OSError):
                        dst.unlink()

            _prune_versions(job_dir)
            _mark_langfield_stale(job_dir)
            splat_route._patch_meta(job_id, stats=None)
            return restored

        try:
            restored = await _run_edit_transaction(
                needs_gpu=False,
                lane_id=job_id,
                operation=transaction,
            )
        except gpu_arbiter.GPUArbiterUnavailable as exc:
            raise HTTPException(status_code=503, detail=f"revert coordination failed: {exc}") from exc

    await audit_operator_event(
        request=request,
        title="Reverted Splat edit",
        description=f"{job_id} -> v{req.version}",
        variant="default",
        action="splat.edit_revert",
        target="3d",
    )
    updated_meta = splat_route._read_meta(job_id) or meta
    return {
        "ok": True,
        "reverted_to": req.version,
        "restored_files": restored,
        "job": splat_route._job_payload(updated_meta),
    }


def _mark_langfield_stale(job_dir: Path) -> None:
    """After a row-level or geometry-mutating edit, gauss_emb.npz no longer lines up
    with splat.ply. Write a marker rather than silently serving wrong search
    results. No-op if the scene never had a language field."""
    lf_dir = job_dir / splat_route.LANGFIELD_DIRNAME
    if lf_dir.is_dir():
        try:
            (lf_dir / "STALE").write_text(splat_route._utc_now() + "\n")
        except OSError:
            pass


def _invalidate_previews(job_dir: Path) -> None:
    """Delete cached thumb/hero so the gallery re-renders from the NEW geometry."""
    preview_dir = splat_route._preview_dir_path(job_dir)
    thumb = preview_dir / "thumb.webp"
    if thumb.is_file():
        try:
            thumb.unlink()
        except OSError:
            pass
    hero = job_dir / splat_route.LANGFIELD_DIRNAME / "hero.webp"
    if hero.is_file():
        try:
            hero.unlink()
        except OSError:
            pass


# =============================================================================
# 2. UTILITY OPS (splat-transform)
# =============================================================================

_FILTER_VALUE_NAMES = {"opacity", "scale_0", "scale_1", "scale_2", "x", "y", "z"}
_COORD_BOUND = 1.0e5


def _check_vec3(v: tuple[float, float, float], field_name: str) -> tuple[float, float, float]:
    if any(abs(c) > _COORD_BOUND for c in v):
        raise ValueError(f"{field_name} components must be within +/-{_COORD_BOUND:g}")
    return v


class CropBoxOp(BaseModel):
    type: Literal["crop_box"]
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @model_validator(mode="after")
    def _check(self) -> "CropBoxOp":
        _check_vec3(self.min, "min")
        _check_vec3(self.max, "max")
        if any(mn > mx for mn, mx in zip(self.min, self.max)):
            raise ValueError("min must be <= max on every axis")
        return self


class CropSphereOp(BaseModel):
    type: Literal["crop_sphere"]
    center: tuple[float, float, float]
    radius: float = Field(gt=0, le=_COORD_BOUND)

    @model_validator(mode="after")
    def _check(self) -> "CropSphereOp":
        _check_vec3(self.center, "center")
        return self


class FilterValueOp(BaseModel):
    """Value-band filter: covers both 'filter by opacity' and 'filter by scale
    band' with one generic op mapped straight onto splat-transform's -V flag."""

    type: Literal["filter_value"]
    name: str
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check(self) -> "FilterValueOp":
        if self.name not in _FILTER_VALUE_NAMES:
            raise ValueError(f"name must be one of {sorted(_FILTER_VALUE_NAMES)}")
        if self.min is None and self.max is None:
            raise ValueError("filter_value needs at least one of min/max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min must be <= max")
        return self


class FilterFloatersOp(BaseModel):
    type: Literal["filter_floaters"]
    size: float | None = Field(default=None, gt=0)
    op: float | None = Field(default=None, gt=0, le=1)
    min: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> "FilterFloatersOp":
        given = [v is not None for v in (self.size, self.op, self.min)]
        if any(given) and not all(given):
            raise ValueError("filter_floaters needs all of size/op/min together, or none (CLI default)")
        return self


class FilterClusterOp(BaseModel):
    type: Literal["filter_cluster"]
    res: float | None = Field(default=None, gt=0)
    op: float | None = Field(default=None, gt=0, le=1)
    min: float | None = Field(default=None, ge=0)
    seed_pos: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> "FilterClusterOp":
        given = [v is not None for v in (self.res, self.op, self.min)]
        if any(given) and not all(given):
            raise ValueError("filter_cluster needs all of res/op/min together, or none (CLI default)")
        if self.seed_pos is not None:
            _check_vec3(self.seed_pos, "seed_pos")
        return self


class DecimateOp(BaseModel):
    type: Literal["decimate"]
    n: int | None = Field(default=None, gt=0, le=200_000_000)
    pct: float | None = Field(default=None, gt=0, le=100)

    @model_validator(mode="after")
    def _check(self) -> "DecimateOp":
        if (self.n is None) == (self.pct is None):
            raise ValueError("decimate needs exactly one of n or pct")
        return self


class TranslateOp(BaseModel):
    type: Literal["translate"]
    x: float = Field(ge=-_COORD_BOUND, le=_COORD_BOUND)
    y: float = Field(ge=-_COORD_BOUND, le=_COORD_BOUND)
    z: float = Field(ge=-_COORD_BOUND, le=_COORD_BOUND)


class RotateOp(BaseModel):
    type: Literal["rotate"]
    x: float = Field(ge=-360.0, le=360.0)
    y: float = Field(ge=-360.0, le=360.0)
    z: float = Field(ge=-360.0, le=360.0)


class ScaleOp(BaseModel):
    type: Literal["scale"]
    factor: float = Field(gt=0, le=1_000.0)


EditOp = Annotated[
    Union[
        CropBoxOp,
        CropSphereOp,
        FilterValueOp,
        FilterFloatersOp,
        FilterClusterOp,
        DecimateOp,
        TranslateOp,
        RotateOp,
        ScaleOp,
    ],
    Field(discriminator="type"),
]


class ApplyOpsRequest(BaseModel):
    ops: list[EditOp] = Field(min_length=1, max_length=32)


def _fnum(v: float) -> str:
    """Format a float for a splat-transform CLI arg without scientific notation or
    a trailing '.0' that could confuse its numeric parser."""
    if float(v).is_integer():
        return str(int(v))
    return repr(float(v))


def _op_to_argv(op: BaseModel) -> list[str]:
    """Translate one validated op into its splat-transform argv fragment. Pure
    function — no subprocess, no I/O — so tests can assert exact argv."""
    if isinstance(op, CropBoxOp):
        return ["-B", ",".join(_fnum(c) for c in (*op.min, *op.max))]
    if isinstance(op, CropSphereOp):
        return ["-S", ",".join(_fnum(c) for c in (*op.center, op.radius))]
    if isinstance(op, FilterValueOp):
        argv: list[str] = []
        if op.min is not None:
            argv += ["-V", f"{op.name},gte,{_fnum(op.min)}"]
        if op.max is not None:
            argv += ["-V", f"{op.name},lte,{_fnum(op.max)}"]
        return argv
    if isinstance(op, FilterFloatersOp):
        if op.size is None:
            return ["-G"]
        return ["-G", f"{_fnum(op.size)},{_fnum(op.op)},{_fnum(op.min)}"]
    if isinstance(op, FilterClusterOp):
        argv = ["-D"] if op.res is None else ["-D", f"{_fnum(op.res)},{_fnum(op.op)},{_fnum(op.min)}"]
        if op.seed_pos is not None:
            argv += ["--seed-pos", ",".join(_fnum(c) for c in op.seed_pos)]
        return argv
    if isinstance(op, DecimateOp):
        return ["-F", _fnum(op.n) if op.n is not None else f"{_fnum(op.pct)}%"]
    if isinstance(op, TranslateOp):
        return ["-t", ",".join(_fnum(c) for c in (op.x, op.y, op.z))]
    if isinstance(op, RotateOp):
        return ["-r", ",".join(_fnum(c) for c in (op.x, op.y, op.z))]
    if isinstance(op, ScaleOp):
        return ["-s", _fnum(op.factor)]
    raise TypeError(f"unhandled op type: {type(op)!r}")  # pragma: no cover - discriminated union is exhaustive


def _build_apply_argv(transform_bin: str, src: Path, ops: list[BaseModel], dst: Path) -> list[str]:
    """Pure argv builder for POST .../edit/apply — no subprocess execution. Ops are
    NOT reordered: the caller's list order IS the pipeline order, per splat-transform's
    own 'ACTIONS executed in order' contract."""
    argv = [transform_bin, "-w", str(src)]
    for op in ops:
        argv.extend(_op_to_argv(op))
    argv.append(str(dst))
    return argv


def _ops_need_gpu(ops: list[BaseModel]) -> bool:
    return any(isinstance(op, (FilterFloatersOp, FilterClusterOp)) for op in ops)


def _ops_change_topology(ops: list[BaseModel]) -> bool:
    """Any op that can change gaussian COUNT or ORDER invalidates a language field's
    row-indexed embeddings (rigid translate/rotate/scale alone do not)."""
    return any(
        isinstance(op, (CropBoxOp, CropSphereOp, FilterValueOp, FilterFloatersOp, FilterClusterOp, DecimateOp))
        for op in ops
    )


async def _run_splat_transform(argv: list[str], needs_gpu: bool, lane_id: str) -> tuple[int, str]:
    """Safely coordinate one standalone splat-transform invocation."""

    splat_route.require_heavy_work_admitted()

    try:
        return await _run_edit_transaction(
            needs_gpu=needs_gpu,
            lane_id=lane_id,
            operation=lambda: _execute_splat_transform(argv),
        )
    except gpu_arbiter.GPUArbiterUnavailable as exc:
        return -1, f"GPU admission failed: {exc}"


async def _execute_splat_transform(argv: list[str]) -> tuple[int, str]:
    """Raw subprocess execution; caller must already own a transaction lease."""
    return_code, out, err = await splat_route._run_capture_subprocess(argv)
    log = (out.decode("utf-8", "replace") + err.decode("utf-8", "replace")).strip()
    return return_code, log


async def _run_edit_transaction(*, needs_gpu: bool, lane_id: str, operation):
    """Select exactly one outer coordination lease for a complete edit."""
    if needs_gpu:
        return await gpu_arbiter.run_gpu_operation(
            lane="splat-edit",
            operation_id=lane_id,
            vram_mb=EDIT_GPU_VRAM_MB,
            operation=operation,
        )
    return await gpu_arbiter.run_host_operation(operation=operation)


async def _run_transform_in_transaction(
    argv: list[str],
    *,
    needs_gpu: bool,
    lane_id: str,
    coordinated: bool,
) -> tuple[int, str]:
    if coordinated:
        return await _execute_splat_transform(argv)
    return await _run_splat_transform(argv, needs_gpu, lane_id)


async def _regen_compress(job_dir: Path, *, coordinated: bool = False) -> bool:
    """Rebuild the .spz from the CURRENT splat.ply (same command shape as
    splat_route.py's `compress` pipeline stage; unique tmp + replace for atomicity).
    On ANY failure the OLD splat.spz is UNLINKED: splat_route's fmt=spz fallback
    only triggers when the file is missing, so leaving a stale .spz would keep
    serving PRE-EDIT geometry while claiming success."""
    spz = splat_route._preview_spz_path(job_dir)
    transform = splat_route._splat_transform_path()
    ply = splat_route._preview_file_path(job_dir)
    if not (transform and ply.is_file()):
        spz.unlink(missing_ok=True)
        return False
    tmp = _edit_tmp_path(spz)
    rc, _log = await _run_transform_in_transaction(
        [transform, "-w", str(ply), str(tmp)],
        needs_gpu=False,
        lane_id="compress",
        coordinated=coordinated,
    )
    if rc == 0 and tmp.is_file():
        tmp.replace(spz)
        return True
    tmp.unlink(missing_ok=True)
    spz.unlink(missing_ok=True)
    return False


async def _regen_webopt(job_dir: Path, *, coordinated: bool = False) -> bool:
    """Rebuild the web-viewer .ply (same command shape as splat_route.py's `webopt`
    pipeline stage). On ANY failure the OLD web.ply is UNLINKED so fmt=web really
    does fall back to the raw .ply instead of serving pre-edit geometry."""
    web = splat_route._preview_web_path(job_dir)
    transform = splat_route._splat_transform_path()
    ply = splat_route._preview_file_path(job_dir)
    if not (transform and ply.is_file()):
        web.unlink(missing_ok=True)
        return False
    tmp = _edit_tmp_path(web)
    command = [
        transform,
        "-w",
        str(ply),
        "--filter-harmonics",
        "0",
        "--decimate",
        str(splat_route.WEB_DECIMATE_TARGET),
        str(tmp),
    ]
    rc, _log = await _run_transform_in_transaction(
        command,
        needs_gpu=False,
        lane_id="webopt",
        coordinated=coordinated,
    )
    if rc == 0 and tmp.is_file():
        tmp.replace(web)
        return True
    tmp.unlink(missing_ok=True)
    web.unlink(missing_ok=True)
    return False


async def _regen_langweb(job_dir: Path, *, coordinated: bool = False) -> bool:
    """Rebuild langweb.ply after a geometry edit, mirroring splat_route's
    _langweb_command action flags EXACTLY: --filter-harmonics 0 and NO decimation
    (probe-verified to preserve row order bit-exactly, so langweb row i == splat.ply
    row i — adding --decimate would break that; see _langweb_command's docstring).

    Returns True when the artifact is consistent afterwards (regenerated, or the
    scene has no language field so any stray copy was dropped); False when regen
    FAILED — in which case the stale langweb.ply is unlinked so fmt=langweb's
    missing-file fallback to the raw .ply is real."""
    langweb = splat_route._preview_langweb_path(job_dir)
    gauss_emb = job_dir / splat_route.LANGFIELD_DIRNAME / "gauss_emb.npz"
    if not gauss_emb.is_file():
        # No language field -> nothing consumes langweb; drop any stray stale copy.
        langweb.unlink(missing_ok=True)
        return True
    transform = splat_route._splat_transform_path()
    ply = splat_route._preview_file_path(job_dir)
    if not (transform and ply.is_file()):
        langweb.unlink(missing_ok=True)
        return False
    tmp = _edit_tmp_path(langweb)
    command = [transform, "-w", str(ply), "--filter-harmonics", "0", str(tmp)]
    rc, _log = await _run_transform_in_transaction(
        command,
        needs_gpu=False,
        lane_id="langweb",
        coordinated=coordinated,
    )
    if rc == 0 and tmp.is_file():
        tmp.replace(langweb)
        return True
    tmp.unlink(missing_ok=True)
    langweb.unlink(missing_ok=True)
    return False


async def _regen_derived_artifacts(job_dir: Path, *, coordinated: bool = False) -> list[str]:
    """Rebuild every derived preview artifact from the freshly-edited splat.ply and
    return accurate warnings. Each failure UNLINKS its stale artifact (see the
    individual _regen_* docstrings), so the fallbacks the warnings promise are real."""
    warnings: list[str] = []
    if not await _regen_compress(job_dir, coordinated=coordinated):
        warnings.append("compress (.spz) regeneration failed; stale splat.spz removed (no .spz until a later edit or re-export succeeds)")
    if not await _regen_webopt(job_dir, coordinated=coordinated):
        warnings.append("web-optimized preview regeneration failed; stale web.ply removed — viewer falls back to the raw .ply")
    if not await _regen_langweb(job_dir, coordinated=coordinated):
        warnings.append("langweb regeneration failed; stale langweb.ply removed — client heatmap falls back to the raw .ply")
    return warnings


@router.post("/jobs/{job_id}/edit/apply")
async def apply_edit_ops(job_id: str, req: ApplyOpsRequest, request: Request) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    meta, job_dir = _require_editable_job(job_id)
    transform_bin = splat_route._splat_transform_path()
    if not transform_bin:
        raise HTTPException(status_code=503, detail="splat-transform binary not available on this host")

    src_ply = splat_route._preview_file_path(job_dir)
    needs_gpu = _ops_need_gpu(req.ops)
    changes_topology = _ops_change_topology(req.ops)

    async with _hold_edit_locks([job_id]):
        async def transaction() -> tuple[dict[str, Any], list[str]]:
            snap_dir, manifest, snap_created = _snapshot_preview(
                job_dir, op="apply", params={"ops": [op.model_dump() for op in req.ops]}
            )
            tmp_ply = _edit_tmp_path(src_ply)
            landed = False
            try:
                argv = _build_apply_argv(transform_bin, src_ply, list(req.ops), tmp_ply)
                rc, log = await _execute_splat_transform(argv)
                if rc != 0:
                    raise HTTPException(
                        status_code=500,
                        detail=f"splat-transform failed (exit {rc}): {log[-2000:]}",
                    )
                if not tmp_ply.is_file():
                    raise HTTPException(
                        status_code=500,
                        detail="splat-transform reported success but produced no output file",
                    )
                tmp_ply.replace(src_ply)
                landed = True
                _prune_versions(job_dir)
                warnings = await _regen_derived_artifacts(job_dir, coordinated=True)
                _invalidate_previews(job_dir)
                if changes_topology:
                    _mark_langfield_stale(job_dir)
                splat_route._patch_meta(job_id, stats=None)
                return manifest, warnings
            except BaseException:
                tmp_ply.unlink(missing_ok=True)
                if not landed:
                    _discard_snapshot(snap_dir, snap_created)
                raise

        try:
            manifest, warnings = await _run_edit_transaction(
                needs_gpu=needs_gpu,
                lane_id=job_id,
                operation=transaction,
            )
        except gpu_arbiter.GPUArbiterUnavailable as exc:
            raise HTTPException(status_code=503, detail=f"edit coordination failed: {exc}") from exc

    await audit_operator_event(
        request=request,
        title="Applied Splat edit ops",
        description=f"{job_id}: {[op.type for op in req.ops]}",
        variant="default",
        action="splat.edit_apply",
        target="3d",
    )
    updated_meta = splat_route._read_meta(job_id) or meta
    return {
        "ok": True,
        "version_before": manifest["seq"],
        "warnings": warnings,
        "job": splat_route._job_payload(updated_meta),
    }


# =============================================================================
# 3. TEXT-SELECT EDITS (semantic)
# =============================================================================


def _decode_relevancy_bytes(data: bytes, n: int, rel_min: float, rel_max: float) -> list[float]:
    """Dequantize the worker's uint8 relevancy payload using the X-Min/X-Max params
    it ships alongside. Contract (langfield_worker.quantize_relevancy, read on disk):
    the worker quantizes over the vector's OWN [min, max], so a byte q dequantizes to
    rel = rel_min + q/255 * (rel_max - rel_min). Thresholding the raw q/255 instead
    would compare RANGE-NORMALIZED values against an absolute threshold — mode=delete
    could destructively remove an arbitrary half of the scene.

    Degenerate case: a constant vector arrives as all-zero bytes with min == max and
    must dequantize back to the constant (span <= 0 -> every score is rel_min).

    The worker only ever sends uint8 (one byte per gaussian) — any other length is a
    hard error, because guessing a dtype would silently corrupt every downstream mask.
    """
    if len(data) != n:
        raise HTTPException(
            status_code=502,
            detail=f"relevancy vector length mismatch: got {len(data)} bytes for {n} gaussians "
            f"(contract is exactly one uint8 per gaussian)",
        )
    span = rel_max - rel_min
    if span <= 0.0:
        return [rel_min] * n
    return [rel_min + (b / 255.0) * span for b in data]


async def _worker_has_relevancy_endpoint() -> bool:
    """Probe the warm langfield worker's own OpenAPI schema for a /relevancy route
    rather than guessing a payload shape via trial POST — cheap, side-effect-free,
    and tells us precisely whether the parallel Lane-2 work has landed yet."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=1.5)) as client:
            resp = await client.get(f"{splat_route.LANGFIELD_WORKER_URL}/openapi.json")
        if resp.status_code != 200:
            return False
        paths = resp.json().get("paths", {})
        return "/relevancy" in paths
    except Exception:  # noqa: BLE001 - unreachable worker => not available
        return False


async def _post_worker_relevancy(config_path: str, lfdir: str, clean_text: str) -> httpx.Response:
    """The raw HTTP hop to the worker's /relevancy (config+lfdir+text, mirroring the
    existing /query and /inventory request shape). Factored out so tests can inject
    a canned httpx.Response with real X-Min/X-Max headers."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=1.5)) as client:
        return await client.post(
            f"{splat_route.LANGFIELD_WORKER_URL}/relevancy",
            json={"config": config_path, "lfdir": lfdir, "text": clean_text},
        )


async def _worker_relevancy_scores(config_path: str, lfdir: str, clean_text: str, n: int) -> list[float]:
    """Fetch + dequantize per-gaussian relevancy scores. The worker's /relevancy
    returns a uint8 body plus X-Count/X-Min/X-Max dequantization headers (rel =
    min + q/255*(max-min)); splat_route's proxy forwards those same headers. Any
    mismatch is a hard 5xx — a language-search worker that might be scoring the
    wrong thing is worse than an honest 'unavailable'."""
    try:
        resp = await _post_worker_relevancy(config_path, lfdir, clean_text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"language-field relevancy worker unreachable: {exc}") from exc
    if resp.status_code == 404:
        raise HTTPException(status_code=503, detail="worker does not implement /relevancy yet")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"relevancy worker returned {resp.status_code}: {resp.text[:300]}"
        )
    try:
        rel_min = float(resp.headers["x-min"])
        rel_max = float(resp.headers["x-max"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="relevancy response is missing/invalid X-Min/X-Max dequantization headers; "
            "refusing to threshold range-normalized values",
        ) from exc
    count_hdr = resp.headers.get("x-count")
    if count_hdr is not None:
        try:
            count = int(count_hdr)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"invalid X-Count header: {count_hdr!r}") from exc
        if count != n:
            raise HTTPException(
                status_code=502,
                detail=f"relevancy X-Count {count} != splat.ply vertex count {n}; the language field "
                "no longer lines up with the scene rows",
            )
    return _decode_relevancy_bytes(resp.content, n, rel_min, rel_max)


# ── header-driven, property-order-agnostic PLY row rewrite (pure stdlib) ───────

_PLY_TYPE_SIZE = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


@dataclass
class _PlyHeader:
    header_text: str
    header_len: int
    vertex_count: int
    row_size: int


def _parse_ply_header(path: Path) -> _PlyHeader:
    """Read + validate a binary_little_endian vertex-only PLY header. Fails loud
    (ValueError) on anything this fixed-stride rewriter can't safely handle: a
    non-binary/big-endian format, more than one element, or any 'list' property
    (variable-width rows — e.g. face indices)."""
    with path.open("rb") as f:
        raw = b""
        while b"end_header\n" not in raw:
            chunk = f.read(4096)
            if not chunk:
                raise ValueError(f"{path}: truncated PLY header (no end_header)")
            raw += chunk
            if len(raw) > 2_000_000:
                raise ValueError(f"{path}: PLY header implausibly large (>2MB); refusing to parse")
    header_len = raw.index(b"end_header\n") + len(b"end_header\n")
    text = raw[:header_len].decode("latin1")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file (missing 'ply' magic)")
    format_line = next((line for line in lines if line.startswith("format ")), "")
    if "binary_little_endian" not in format_line:
        raise ValueError(f"{path}: only binary_little_endian PLY is supported, got: {format_line!r}")

    element_lines = [line for line in lines if line.startswith("element ")]
    if len(element_lines) != 1 or not element_lines[0].startswith("element vertex "):
        raise ValueError(
            f"{path}: expected exactly one 'element vertex' element (got {element_lines!r}); "
            "non-vertex elements (e.g. faces) aren't supported by this row-mask rewriter"
        )
    vertex_count = int(element_lines[0].split()[2])

    if any(line.startswith("property list") for line in lines):
        raise ValueError(
            f"{path}: 'property list' entries (variable-width rows) aren't supported by this "
            "fixed-stride rewriter"
        )

    prop_types = [line.split()[1] for line in lines if line.startswith("property ")]
    unknown = sorted({t for t in prop_types if t not in _PLY_TYPE_SIZE})
    if unknown:
        raise ValueError(f"{path}: unknown PLY property type(s): {unknown!r}")
    row_size = sum(_PLY_TYPE_SIZE[t] for t in prop_types)
    if row_size <= 0:
        raise ValueError(f"{path}: PLY header declares no vertex properties")

    return _PlyHeader(header_text=text, header_len=header_len, vertex_count=vertex_count, row_size=row_size)


def _rewrite_ply_masked(src: Path, dst: Path, keep_mask: list[bool]) -> int:
    """Header-driven boolean row-mask rewrite: streams rows through unmodified
    (whole-row byte copy, no per-field unpack — property-order-agnostic and fast).
    Returns the number of rows kept."""
    hdr = _parse_ply_header(src)
    if len(keep_mask) != hdr.vertex_count:
        raise ValueError(f"mask length {len(keep_mask)} != vertex count {hdr.vertex_count} for {src}")
    kept = sum(1 for k in keep_mask if k)
    new_header = hdr.header_text.replace(f"element vertex {hdr.vertex_count}", f"element vertex {kept}", 1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(new_header.encode("latin1"))
        fin.seek(hdr.header_len)
        for i in range(hdr.vertex_count):
            row = fin.read(hdr.row_size)
            if len(row) < hdr.row_size:
                raise ValueError(f"{src}: truncated at row {i} of {hdr.vertex_count}")
            if keep_mask[i]:
                fout.write(row)
    return kept


class SemanticEditRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    mode: Literal["delete", "isolate", "extract"]
    name: str | None = Field(default=None, max_length=80)
    # -G floater cleanup after the row-mask cut. Default on (kills boundary halos),
    # but opt-OUT matters: isolate/extract of a legitimately thin/wispy object
    # (wires, foliage, smoke) is sparse by nature and -G would gut it.
    cleanup: bool = True


async def _chain_floater_cleanup(
    ply_path: Path,
    lane_id: str,
    *,
    coordinated: bool = False,
) -> Path:
    """Chain a -G floater cleanup after a row-mask edit (kills boundary halos left
    by the cut). Callers skip this entirely when the request sets cleanup=false.
    Best-effort: on any failure the un-cleaned masked result is kept rather than
    failing the whole edit — the mask itself already succeeded."""
    transform_bin = splat_route._splat_transform_path()
    if not transform_bin:
        return ply_path
    cleaned = ply_path.with_name(ply_path.name + ".float-clean")
    rc, _log = await _run_transform_in_transaction(
        [transform_bin, "-w", str(ply_path), "-G", str(cleaned)],
        needs_gpu=True,
        lane_id=lane_id,
        coordinated=coordinated,
    )
    if rc == 0 and cleaned.is_file():
        ply_path.unlink(missing_ok=True)
        return cleaned
    cleaned.unlink(missing_ok=True)
    return ply_path


@router.post("/jobs/{job_id}/edit/semantic")
async def semantic_edit(job_id: str, req: SemanticEditRequest, request: Request) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    meta, job_dir = _require_editable_job(job_id)
    lf_dir = job_dir / splat_route.LANGFIELD_DIRNAME
    gauss_emb = lf_dir / "gauss_emb.npz"
    if not gauss_emb.is_file():
        raise HTTPException(
            status_code=422,
            detail="this scene has no language field (gauss_emb.npz missing) — enable language_field on a new job",
        )
    # STALE marker FIRST: a previous geometry edit invalidated the row<->embedding
    # correspondence. Thresholding embeddings that no longer match rows would mask
    # arbitrary gaussians — refuse loudly instead.
    if (lf_dir / "STALE").is_file():
        raise HTTPException(
            status_code=409,
            detail="language field is stale — re-run langfield for this scene before semantic edits "
            "(a previous edit changed splat.ply rows, so the embeddings no longer line up)",
        )
    config_path = splat_route._find_latest_config(job_dir)
    if config_path is None:
        raise HTTPException(status_code=422, detail="no training config found for this scene")

    if not await _worker_has_relevancy_endpoint():
        raise HTTPException(
            status_code=503,
            detail="the language-search worker doesn't expose /relevancy yet; semantic scene edits need it",
        )

    clean_text = splat_route._langfield_clean_text(req.text)
    src_ply = splat_route._preview_file_path(job_dir)

    async with _hold_edit_locks([job_id]):
        n = splat_route._ply_vertex_count(src_ply)
        if n is None:
            raise HTTPException(status_code=500, detail="could not read splat.ply vertex count")
        source_identity = _file_identity(src_ply)
        # The worker acquires the canonical GPU lease itself. Do not hold our host
        # transaction while waiting or the two processes would deadlock.
        scores = await _worker_relevancy_scores(str(config_path), str(lf_dir), clean_text, n)
        match_mask = [s >= req.threshold for s in scores]
        n_matched = sum(match_mask)
        if n_matched == 0:
            raise HTTPException(
                status_code=422,
                detail=f"no gaussians matched '{req.text}' at threshold {req.threshold}; try a lower threshold or different wording",
            )

        keep_mask = match_mask if req.mode in ("isolate", "extract") else [not m for m in match_mask]

        async def transaction() -> dict[str, Any]:
            if _file_identity(src_ply) != source_identity or (lf_dir / "STALE").is_file():
                raise HTTPException(
                    status_code=409,
                    detail="scene changed while semantic relevancy was computed; retry the edit",
                )

            if req.mode in ("delete", "isolate"):
                snap_dir, manifest, snap_created = _snapshot_preview(
                    job_dir,
                    op="semantic",
                    params={"text": req.text, "threshold": req.threshold, "mode": req.mode},
                )
                tmp_masked = _edit_tmp_path(src_ply)
                cleaned = tmp_masked
                landed = False
                try:
                    kept = _rewrite_ply_masked(src_ply, tmp_masked, keep_mask)
                    if req.cleanup:
                        cleaned = await _chain_floater_cleanup(
                            tmp_masked,
                            job_id,
                            coordinated=True,
                        )
                    cleaned.replace(src_ply)
                    landed = True
                    _prune_versions(job_dir)
                    rows_after = splat_route._ply_vertex_count(src_ply)
                    warnings = await _regen_derived_artifacts(job_dir, coordinated=True)
                    _invalidate_previews(job_dir)
                    _mark_langfield_stale(job_dir)
                    splat_route._patch_meta(job_id, stats=None)
                    updated_meta = splat_route._read_meta(job_id) or meta
                    return {
                        "ok": True,
                        "mode": req.mode,
                        "matched": n_matched,
                        "kept": kept,
                        "cleanup": req.cleanup,
                        "rows_before_cleanup": kept,
                        "rows_after_cleanup": rows_after,
                        "version_before": manifest["seq"],
                        "warnings": warnings,
                        "language_field_stale": True,
                        "job": splat_route._job_payload(updated_meta),
                    }
                except ValueError as exc:
                    raise HTTPException(status_code=500, detail=f"row-mask rewrite failed: {exc}") from exc
                finally:
                    tmp_masked.unlink(missing_ok=True)
                    if cleaned != tmp_masked:
                        cleaned.unlink(missing_ok=True)
                    if not landed:
                        _discard_snapshot(snap_dir, snap_created)

            new_job_id, new_job_dir = _create_derived_job_dir()
            committed = False
            try:
                new_preview = splat_route._preview_dir_path(new_job_dir)
                new_preview.mkdir(parents=True, exist_ok=True)
                final_ply = new_preview / "splat.ply"
                tmp_masked = _edit_tmp_path(final_ply)
                kept = _rewrite_ply_masked(src_ply, tmp_masked, keep_mask)
                cleaned = tmp_masked
                if req.cleanup:
                    cleaned = await _chain_floater_cleanup(
                        tmp_masked,
                        new_job_id,
                        coordinated=True,
                    )
                cleaned.replace(final_ply)
                rows_after = splat_route._ply_vertex_count(final_ply)
                await _regen_compress(new_job_dir, coordinated=True)
                await _regen_webopt(new_job_dir, coordinated=True)
                name = (req.name or f"{req.text} (from {job_id})")[:120]
                _write_derived_meta(
                    new_job_id,
                    new_job_dir,
                    source_type="extracted",
                    parents=[job_id],
                    name=name,
                    extra={"extract_query": req.text, "extract_threshold": req.threshold},
                )
                committed = True
                return {
                    "ok": True,
                    "mode": "extract",
                    "matched": n_matched,
                    "kept": kept,
                    "cleanup": req.cleanup,
                    "rows_before_cleanup": kept,
                    "rows_after_cleanup": rows_after,
                    "new_job_id": new_job_id,
                    "job": splat_route._job_payload(splat_route._read_meta(new_job_id)),
                }
            except HTTPException:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - convert filesystem/PLY errors
                raise HTTPException(status_code=500, detail=f"semantic extract failed: {exc}") from exc
            finally:
                if not committed:
                    shutil.rmtree(new_job_dir, ignore_errors=True)

        try:
            result = await _run_edit_transaction(
                needs_gpu=req.cleanup,
                lane_id=job_id,
                operation=transaction,
            )
        except gpu_arbiter.GPUArbiterUnavailable as exc:
            raise HTTPException(status_code=503, detail=f"semantic edit coordination failed: {exc}") from exc

    extracted = result["mode"] == "extract"
    await audit_operator_event(
        request=request,
        title="Extracted Splat scene" if extracted else "Semantic Splat edit",
        description=(
            f"{job_id} -> {result['new_job_id']} ('{req.text}')"
            if extracted
            else f"{job_id}: {req.mode} '{req.text}'"
        ),
        variant="success" if extracted else "default",
        action="splat.edit_semantic",
        target="3d",
    )
    return result


# =============================================================================
# 4. MERGE
# =============================================================================


class MergeTransformSpec(BaseModel):
    """Either a 4x4 similarity matrix OR explicit translate/rotate/scale — never
    both. `rotate` is Euler XYZ degrees passed straight through to splat-transform's
    own -r (whatever its internal convention is; the caller supplied it directly,
    so there's no invented-convention risk here — only the matrix path needs
    decomposition)."""

    matrix: list[list[float]] | None = None
    translate: tuple[float, float, float] | None = None
    rotate: tuple[float, float, float] | None = None
    scale: float | None = Field(default=None, gt=0, le=1_000.0)

    @model_validator(mode="after")
    def _check(self) -> "MergeTransformSpec":
        has_matrix = self.matrix is not None
        has_trs = self.translate is not None or self.rotate is not None or self.scale is not None
        if has_matrix and has_trs:
            raise ValueError("provide either 'matrix' or translate/rotate/scale, not both")
        if has_matrix:
            if len(self.matrix) != 4 or any(len(row) != 4 for row in self.matrix):
                raise ValueError("matrix must be 4x4")
        return self


class MergeRequest(BaseModel):
    job_ids: list[str] = Field(min_length=2, max_length=16)
    name: str = Field(min_length=1, max_length=120)
    transforms: dict[str, MergeTransformSpec] | None = None


def _vec3_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _axis_rotation_flag(axis: str, degrees: float) -> tuple[float, float, float]:
    """Map a STANDARD (right-handed, textbook Rx/Ry/Rz) single-axis rotation onto
    the (x,y,z) triple to hand splat-transform's -r.

    VERIFIED EMPIRICALLY (2026-07-04, synthetic 1-vertex PLY probe — see module
    docstring): splat-transform's -r applies Rx(-angle) and Ry(-angle) for the X
    and Y components but Rz(+angle) for Z (not a uniform convention across axes).
    Concretely: `-r 0,90,0` sent (1,0,0) -> (0,0,1), which is standard Ry(-90), not
    Ry(+90); `-r 90,0,0` sent (0,1,0) -> (0,0,-1), standard Rx(-90); `-r 0,0,90`
    sent (1,0,0) -> (0,1,0), standard Rz(+90) (no flip). This function bakes in
    that correction so a decomposed 4x4 rotation reproduces exactly on the wire.
    """
    if axis == "x":
        return (-degrees, 0.0, 0.0)
    if axis == "y":
        return (0.0, -degrees, 0.0)
    if axis == "z":
        return (0.0, 0.0, degrees)
    raise ValueError(f"unknown axis {axis!r}")  # pragma: no cover - internal callers only pass x/y/z


def _single_axis_angle_deg(rot: list[list[float]], axis_idx: int, tol: float = 1e-4) -> float | None:
    """If `rot` (a 3x3 orthonormal, det=+1 matrix) is a rotation about ONLY the
    given axis (standard Rx/Ry/Rz form), return the angle in degrees; else None.
    Single-axis rotations are convention-independent (the other two Euler angles
    are zero, so composition order can't matter) — this is what lets us safely
    decompose a 4x4 matrix without knowing splat-transform's multi-axis Euler
    order."""
    if axis_idx == 0:  # Rx: [[1,0,0],[0,c,-s],[0,s,c]]
        off = (rot[0][1], rot[0][2], rot[1][0], rot[2][0])
        if abs(rot[0][0] - 1) > tol or any(abs(v) > tol for v in off):
            return None
        c, s = rot[1][1], rot[2][1]
    elif axis_idx == 1:  # Ry: [[c,0,s],[0,1,0],[-s,0,c]]
        off = (rot[0][1], rot[2][1], rot[1][0], rot[1][2])
        if abs(rot[1][1] - 1) > tol or any(abs(v) > tol for v in off):
            return None
        c, s = rot[0][0], -rot[2][0]
    elif axis_idx == 2:  # Rz: [[c,-s,0],[s,c,0],[0,0,1]]
        off = (rot[0][2], rot[1][2], rot[2][0], rot[2][1])
        if abs(rot[2][2] - 1) > tol or any(abs(v) > tol for v in off):
            return None
        c, s = rot[0][0], rot[1][0]
    else:  # pragma: no cover - internal callers only pass 0/1/2
        raise ValueError(f"unknown axis index {axis_idx!r}")
    if abs(c * c + s * s - 1) > tol:
        return None
    return math.degrees(math.atan2(s, c))


_Decomposed = tuple[tuple[float, float, float], tuple[float, float, float], "float | None"]


def _decompose_similarity_matrix(m: list[list[float]]) -> "_Decomposed | None":
    """Decompose a 4x4 similarity matrix (uniform scale + rotation + translation,
    no shear/reflection/non-uniform scale) into (translate_xyz, rotate_xyz_deg,
    scale) — the pieces splat-transform CAN represent. Returns None only for a true
    identity (nothing to apply). Raises HTTPException(422) for anything that isn't
    a pure single-axis rotation — see the module docstring for why multi-axis
    rotations are rejected rather than guessed."""
    if len(m) != 4 or any(len(row) != 4 for row in m):
        raise HTTPException(status_code=422, detail="matrix must be 4x4")

    translate = (m[0][3], m[1][3], m[2][3])
    linear = [[m[r][c] for c in range(3)] for r in range(3)]
    cols = [[linear[r][c] for r in range(3)] for c in range(3)]
    norms = [_vec3_norm(c) for c in cols]
    if any(n < 1e-9 for n in norms):
        raise HTTPException(status_code=422, detail="matrix has a degenerate (zero-length) axis; cannot decompose")
    avg = sum(norms) / 3
    if any(abs(n - avg) > 1e-4 * max(1.0, avg) for n in norms):
        raise HTTPException(
            status_code=422,
            detail="matrix has non-uniform axis scaling; splat-transform only supports a single uniform --scale factor",
        )
    scale = avg
    rot = [[linear[r][c] / scale for c in range(3)] for r in range(3)]

    rt_r = [[sum(rot[k][i] * rot[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            if abs(rt_r[i][j] - expected) > 1e-4:
                raise HTTPException(
                    status_code=422, detail="matrix has shear and cannot be decomposed into translate/rotate/scale"
                )

    det = (
        rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
        - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
        + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0])
    )
    if det < 0:
        raise HTTPException(status_code=422, detail="matrix includes a reflection, which splat-transform can't represent")

    identity = all(abs(rot[i][j] - (1.0 if i == j else 0.0)) < 1e-6 for i in range(3) for j in range(3))
    no_translate = _vec3_norm(list(translate)) < 1e-9
    no_scale = abs(scale - 1.0) < 1e-9
    if identity and no_translate and no_scale:
        return None

    rotate_deg = (0.0, 0.0, 0.0)
    if not identity:
        for axis_idx, axis in enumerate(("x", "y", "z")):
            angle = _single_axis_angle_deg(rot, axis_idx)
            if angle is not None:
                rotate_deg = _axis_rotation_flag(axis, angle)
                break
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    "matrix is a general multi-axis rotation; splat-transform's multi-axis Euler "
                    "composition order isn't verified — decompose to a single-axis rotation, or "
                    "provide translate/rotate/scale directly"
                ),
            )

    translate_out = (
        translate[0] if not no_translate else 0.0,
        translate[1] if not no_translate else 0.0,
        translate[2] if not no_translate else 0.0,
    )
    return translate_out, rotate_deg, (None if no_scale else scale)


def _merge_transform_to_argv(spec: MergeTransformSpec) -> list[str]:
    """Emit ONE transform's CLI fragment in the standard TRS composition order
    (scale, then rotate, then translate) so `translate/rotate/scale` given together
    reads as p' = T + R*(S*p) — the conventional interpretation of a combined
    rigid+scale transform. splat-transform applies actions strictly in the order
    given, so scale MUST come first here (see module docstring)."""
    if spec.matrix is not None:
        decomposed = _decompose_similarity_matrix(spec.matrix)
        if decomposed is None:
            return []
        translate, rotate, scale = decomposed
        argv: list[str] = []
        if scale is not None:
            argv += ["-s", _fnum(scale)]
        if rotate != (0.0, 0.0, 0.0):
            argv += ["-r", ",".join(_fnum(c) for c in rotate)]
        if _vec3_norm(list(translate)) > 1e-9:
            argv += ["-t", ",".join(_fnum(c) for c in translate)]
        return argv

    argv = []
    if spec.scale is not None:
        argv += ["-s", _fnum(spec.scale)]
    if spec.rotate is not None:
        argv += ["-r", ",".join(_fnum(c) for c in spec.rotate)]
    if spec.translate is not None:
        argv += ["-t", ",".join(_fnum(c) for c in spec.translate)]
    return argv


def _build_merge_argv(
    transform_bin: str, job_plys: list[tuple[str, Path]], transforms: dict[str, MergeTransformSpec], out_ply: Path
) -> list[str]:
    """Pure argv builder for POST /edit/merge — no subprocess execution."""
    argv = [transform_bin, "-w"]
    for job_id, ply in job_plys:
        argv.append(str(ply))
        spec = transforms.get(job_id)
        if spec is not None:
            argv.extend(_merge_transform_to_argv(spec))
    argv.append(str(out_ply))
    return argv


@router.post("/edit/merge")
async def merge_scenes(req: MergeRequest, request: Request) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    transform_bin = splat_route._splat_transform_path()
    if not transform_bin:
        raise HTTPException(status_code=503, detail="splat-transform binary not available on this host")

    seen: set[str] = set()
    job_plys: list[tuple[str, Path]] = []
    for jid in req.job_ids:
        if jid in seen:
            raise HTTPException(status_code=422, detail=f"duplicate job_id {jid} in merge request")
        seen.add(jid)
        if not splat_route._safe_job_id(jid):
            raise HTTPException(status_code=404, detail=f"invalid job id {jid}")
        jdir = splat_route._job_dir(jid)
        jmeta = splat_route._read_meta(jid)
        if jmeta is None or not jdir.is_dir():
            raise HTTPException(status_code=404, detail=f"job {jid} not found")
        if jmeta.get("status") in {"starting", "running"}:
            raise HTTPException(status_code=409, detail=f"job {jid} is still running")
        ply = splat_route._preview_file_path(jdir)
        if not ply.is_file():
            raise HTTPException(status_code=409, detail=f"job {jid} has no finished preview to merge")
        job_plys.append((jid, ply))

    transforms = req.transforms or {}
    unknown_keys = sorted(set(transforms) - seen)
    if unknown_keys:
        raise HTTPException(status_code=422, detail=f"transforms references job(s) not in job_ids: {unknown_keys}")

    # Hold EVERY source job's edit lock (409 if any is mid-edit): merging a scene
    # while another endpoint is rewriting its splat.ply would read a torn source.
    async with _hold_edit_locks(list(req.job_ids)):
        async def transaction() -> tuple[str, Path]:
            new_job_id, new_job_dir = _create_derived_job_dir()
            committed = False
            try:
                # Revalidate after winning the host flock. No source may change
                # between this point and the merged destination commit.
                for source_job_id, ply in job_plys:
                    source_meta = splat_route._read_meta(source_job_id)
                    if source_meta is None or source_meta.get("status") in {"starting", "running"}:
                        raise HTTPException(status_code=409, detail=f"job {source_job_id} changed before merge")
                    if not ply.is_file():
                        raise HTTPException(status_code=409, detail=f"job {source_job_id} preview disappeared")

                new_preview = splat_route._preview_dir_path(new_job_dir)
                new_preview.mkdir(parents=True, exist_ok=True)
                out_ply = new_preview / "splat.ply"
                argv = _build_merge_argv(transform_bin, job_plys, transforms, out_ply)
                rc, log = await _execute_splat_transform(argv)
                if rc != 0 or not out_ply.is_file():
                    raise HTTPException(
                        status_code=500,
                        detail=f"splat-transform merge failed (exit {rc}): {log[-2000:]}",
                    )

                await _regen_compress(new_job_dir, coordinated=True)
                await _regen_webopt(new_job_dir, coordinated=True)
                _write_derived_meta(
                    new_job_id,
                    new_job_dir,
                    source_type="merged",
                    parents=list(req.job_ids),
                    name=req.name,
                    extra={
                        "merge_transforms": {
                            key: value.model_dump(exclude_none=True)
                            for key, value in transforms.items()
                        }
                    },
                )
                committed = True
                return new_job_id, new_job_dir
            finally:
                if not committed:
                    shutil.rmtree(new_job_dir, ignore_errors=True)

        try:
            new_job_id, _new_job_dir = await _run_edit_transaction(
                needs_gpu=False,
                lane_id="merge",
                operation=transaction,
            )
        except gpu_arbiter.GPUArbiterUnavailable as exc:
            raise HTTPException(status_code=503, detail=f"merge coordination failed: {exc}") from exc

    await audit_operator_event(
        request=request,
        title="Merged Splat scenes",
        description=f"{req.job_ids} -> {new_job_id}",
        variant="success",
        action="splat.edit_merge",
        target="3d",
    )
    return {"ok": True, "new_job_id": new_job_id, "job": splat_route._job_payload(splat_route._read_meta(new_job_id))}
