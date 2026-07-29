"""Pluck-doc routes: build the per-prop splat-row artifact, serve it with a
staleness verdict. Pure numpy/metadata work (the heaviest read is splat.ply's
xyz in the no-index-map fallback) — no GPU lease, no subprocess."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

import splat_route
from operator_audit import audit_operator_event
from world_interactions_route import _require_world

router = APIRouter()


@router.post("/jobs/{job_id}/world/pluck")
async def build_world_pluck(job_id: str, request: Request) -> dict[str, Any]:
    """Build (or rebuild) `_world/pluck.json` from the isolate claims and the
    exported-splat row map. Refusals carry their remedy verbatim."""
    job_dir, world_dir, _manifest = _require_world(job_id)
    import world_pluck  # noqa: PLC0415 — keep the import chain lazy

    try:
        doc = world_pluck.build_pluck(job_dir)
        world_pluck.write_pluck(world_dir, doc)
    except world_pluck.PluckError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    meta = splat_route._read_meta(job_id) or {}
    await audit_operator_event(
        request=request,
        title="Pluck data built",
        description=f"{job_id}: {len(doc['elements'])} prop(s), "
                    f"method {doc['method']}",
        variant="success",
        action="splat.world_pluck",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "props": len(doc["elements"])},
    )
    return {"ok": True, "pluck": doc, "stale": False, "reasons": []}


@router.get("/jobs/{job_id}/world/pluck")
async def get_world_pluck(job_id: str) -> dict[str, Any]:
    """The stored doc + its staleness verdict. 404 when never built; a stale
    or corrupt doc returns 200 with reasons so the page can warn loudly."""
    job_dir, world_dir, _manifest = _require_world(job_id)
    import world_pluck  # noqa: PLC0415

    doc, stale, reasons = world_pluck.read_pluck(world_dir, job_dir)
    if doc is None and not stale:
        raise HTTPException(status_code=404,
                            detail="Pluck data not built for this world — "
                                   "POST /world/pluck")
    return {"ok": doc is not None, "pluck": doc, "stale": stale,
            "reasons": reasons}
