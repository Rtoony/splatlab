"""Read-only live-activity snapshot: what is busy RIGHT NOW.

GET /activity reports two things, both from state the process already keeps:

- ``gpu.holder`` — the cross-process GPU lease holder, via the existing
  ``gpu_arbiter.holder_info()`` reader (Redis hash with in-process fallback).
  Read-only: this module never acquires, extends, or releases anything.
- ``jobs`` — which jobs currently hold one of the per-job in-process
  ``asyncio.Lock``s (scene edit, preview export, mesh/object build, portable
  export), inspected via ``.locked()``. No task registry, no persistence, no
  progress percentages — just "an operation of this kind is in flight".

Truthfulness rail: the in-process locks die with a backend restart, but a
restart also kills the operation each lock guarded, so the signal stays
truthful — a fresh process correctly reports nothing in flight.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter

import edit_ops
import export_route
import geo_route
import gpu_arbiter
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
    return {"gpu": {"holder": holder}, "jobs": jobs}
