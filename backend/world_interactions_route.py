"""Authoring and player state for walkable-world interactions.

Its own router for the same reason `dimensions_route.py` is: this is pure
metadata CRUD with no GPU and no subprocess, and `splat_route.py` is already
~7k lines and the standing top refactor target. Nothing here belongs in it.

Deliberately NOT behind `require_heavy_work_admitted()` — same policy as
`dimensions_route.py` and the geo anchor routes. Authoring an affordance is
metadata, and it must stay available while the GPU is held or under maintenance.

Layout, both beside the world they describe:
  <job>/_world/interactions.json  — authored affordances (see world_interactions)
  <job>/_world/state.json         — the player's diff against them

NOTE for future edits: helpers stay ABOVE the `@router.*` decorators. Inserting a
function between a decorator and its handler registers the HELPER as the route
and 422s everything — that is recorded in STATUS.md as having cost 12 tests once
already.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import artifact_manifest as manifests
import splat_route
import world_interactions as wi

router = APIRouter()


def _require_world(job_id: str) -> tuple[Path, Path, dict[str, Any] | None]:
    """(job_dir, world_dir, world_manifest) or the right 404/409.

    A world that was solidified but never graded has no `world_manifest.json`;
    that is a legitimate state, so the manifest may be None and the presence
    check simply does not run.
    """
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    job_dir = splat_route._job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Splat job not found")

    world_dir = job_dir / splat_route.WORLD_DIRNAME
    if not (world_dir / "world.json").is_file():
        raise HTTPException(
            status_code=409,
            detail="No walkable world for this scene yet — run the world solidify "
                   "stage first.")
    return job_dir, world_dir, manifests.read_json(world_dir / "world_manifest.json")


def _world_identity(world_dir: Path) -> dict[str, Any] | None:
    manifest = world_dir / "world_manifest.json"
    return manifests.file_identity(manifest) if manifest.is_file() else None


def _payload(job_id: str, world_dir: Path,
             world_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Everything the walker needs in one round trip."""
    interactions = wi.read_interactions(world_dir)
    try:
        state = wi.read_state(world_dir, job_id)
        state_error = ""
    except wi.InteractionsError as exc:
        # A save copied in by job duplication. Report it rather than applying it
        # or crashing the world load.
        state, state_error = None, str(exc)

    resolved = wi.resolve_state(interactions, state, world_manifest,
                                _world_identity(world_dir))
    return {
        "job_id": job_id,
        "interactions": interactions,
        "state": state,
        "resolved": resolved,
        "state_error": state_error,
        "known_slugs": sorted(wi.world_slugs(world_manifest)),
    }


class InteractionsBody(BaseModel):
    """The whole authored document — PUT replaces, it does not merge."""
    elements: list[dict[str, Any]] = Field(default_factory=list)


class StateBody(BaseModel):
    slug: str = Field(..., min_length=1, max_length=64)
    state: str = Field(..., min_length=1, max_length=32)


@router.get("/jobs/{job_id}/world/interactions")
async def get_world_interactions(job_id: str) -> dict[str, Any]:
    """Authored affordances plus the resolved player state, in one call."""
    _job_dir, world_dir, world_manifest = _require_world(job_id)
    return _payload(job_id, world_dir, world_manifest)


@router.put("/jobs/{job_id}/world/interactions")
async def put_world_interactions(job_id: str, body: InteractionsBody) -> dict[str, Any]:
    """Author or replace the affordances for this world.

    Every slug must exist in the world manifest — authoring an interaction for
    an element that is not there would only ever surface as a mystery in the
    walker, so it is refused with the list of slugs that do exist.
    """
    _job_dir, world_dir, world_manifest = _require_world(job_id)
    document = wi.new_interactions(job_id)
    document["elements"] = body.elements

    known = wi.world_slugs(world_manifest) if world_manifest else None
    try:
        wi.write_interactions(world_dir, document, known)
    except wi.InteractionsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _payload(job_id, world_dir, world_manifest)


@router.post("/jobs/{job_id}/world/state")
async def set_world_element_state(job_id: str, body: StateBody) -> dict[str, Any]:
    """Record one element's new state. This is what an interaction persists."""
    _job_dir, world_dir, world_manifest = _require_world(job_id)

    interactions = wi.read_interactions(world_dir)
    records = {r["slug"]: r for r in ((interactions or {}).get("elements") or [])}
    record = records.get(body.slug)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"no interaction is authored for {body.slug!r}")
    if body.state not in record["states"]:
        raise HTTPException(
            status_code=400,
            detail=f"{body.state!r} is not one of {record['states']} for {body.slug!r}")

    try:
        state = wi.read_state(world_dir, job_id)
    except wi.InteractionsError:
        state = None            # a foreign save is replaced, not merged into
    document = state or wi.new_state(job_id, _world_identity(world_dir))
    document["world"] = _world_identity(world_dir)
    document["elements"][body.slug] = body.state
    wi.write_state(world_dir, document)

    return _payload(job_id, world_dir, world_manifest)


@router.delete("/jobs/{job_id}/world/state")
async def reset_world_state(job_id: str) -> dict[str, Any]:
    """Drop the save; the world returns to its authored initials."""
    _job_dir, world_dir, world_manifest = _require_world(job_id)
    wi.state_path(world_dir).unlink(missing_ok=True)
    return _payload(job_id, world_dir, world_manifest)
