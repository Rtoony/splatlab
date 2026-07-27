"""Read-only live-activity snapshot: what is busy RIGHT NOW.

GET /activity reports two things, both from state the process already keeps:

- ``gpu.holder`` — the cross-process GPU lease holder, via the existing
  ``gpu_arbiter.holder_info()`` reader (Redis hash with in-process fallback).
  Read-only: this module never acquires, extends, or releases anything.
- ``jobs`` — which jobs currently hold one of the per-job in-process
  ``asyncio.Lock``s (scene edit, preview export, mesh/object build, portable
  export), inspected via ``.locked()``. No task registry, no persistence, no
  progress percentages — just "an operation of this kind is in flight".

- ``operations`` — the persistent registry (``opregistry``), which adds what the
  locks cannot express: a stable id per operation, its current step, and its
  outcome after the fact. ``GET /ops`` and ``GET /ops/{op_id}`` make a blocking
  POST pollable and give ``/activity`` history across restarts.

Truthfulness rail: the in-process locks die with a backend restart, but a
restart also kills the operation each lock guarded, so the signal stays
truthful — a fresh process correctly reports nothing in flight. The registry
keeps that property deliberately: rows are stamped with a per-process runtime
id, and a ``running`` row from a dead process is reported as ``abandoned``,
never as running.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query

import edit_ops
import export_route
import geo_route
import gpu_arbiter
import opregistry
import splat_route

router = APIRouter()


def _lock_sources() -> list[tuple[str, dict[str, asyncio.Lock]]]:
    """(flag name, per-job lock map) pairs — resolved at call time so tests can
    monkeypatch the underlying dicts. Every map is keyed by job_id."""
    return [
        # Mutating scene edits (apply/semantic/merge/restore) — edit_ops.
        ("editing", edit_ops._EDIT_LOCKS),
        # ns-export preview regeneration — splat_route.
        ("preview_exporting", splat_route._PREVIEW_EXPORT_LOCKS),
        # Heavy mesh / object-isolate / scene-lane builds share one per-job
        # lock in splat_route, so one flag covers them all.
        ("meshing", splat_route._MESH_EXPORT_LOCKS),
        # Portable SPZ/SOG/glTF + collision + Unreal bundle — export_route.
        ("exporting", export_route._export_locks),
        # Survey exports (contours / DXF / sections) — geo_route.
        ("surveying", geo_route._GEO_EXPORT_LOCKS),
    ]


@router.get("/activity")
async def get_activity() -> dict[str, Any]:
    info = gpu_arbiter.holder_info()
    holder = (
        {"lane": info.get("lane"), "job_id": info.get("job_id"), "since": info.get("since")}
        if info.get("lane")
        else None
    )
    jobs: dict[str, dict[str, bool]] = {}
    for flag, locks in _lock_sources():
        for job_id, lock in list(locks.items()):
            if lock.locked():
                jobs.setdefault(job_id, {})[flag] = True
    # Stepped per-job edit progress (apply/upload/semantic) — additive so the
    # boolean `editing` flag above keeps its contract for existing consumers.
    # Same restart-truthfulness rail: the dict dies with the process AND the op.
    return {
        "gpu": {"holder": holder},
        "jobs": jobs,
        "edit_progress": {job_id: dict(entry) for job_id, entry in edit_ops.EDIT_PROGRESS.items()},
        "operations": opregistry.active_by_job(),
    }


@router.get("/ops")
async def list_operations(
    job_id: str | None = Query(default=None),
    kind: str | None = Query(default=None),
    active_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Recent operations, newest first — the history /activity cannot carry."""
    return {
        "operations": opregistry.list_ops(
            job_id=job_id, kind=kind, active_only=active_only, limit=limit)
    }


@router.get("/ops/{op_id}")
async def get_operation(op_id: str) -> dict[str, Any]:
    """Poll one operation. This is what turns a blocking multi-minute POST into
    something a reconnecting client can ask about."""
    record = opregistry.get(op_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown operation")
    return record
