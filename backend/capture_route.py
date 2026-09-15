"""Authenticated capture and route operations for the spatial workspace."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

import artifact_manifest as manifests
import capture_records as captures
import capture_atlas
import opregistry
import route_builder as routes
import route_reconstruction
import splat_route

router = APIRouter()
TASKS: dict[str, asyncio.Task] = {}


class CaptureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_path: str


class CaptureClock(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video_start_utc_s: float = Field(gt=0)
    offset_s: float = Field(default=0, ge=-86400, le=86400)
    uncertainty_s: float = Field(ge=0, le=86400)


def accepted_input(raw: str) -> Path:
    candidate = Path(raw).resolve()
    approved = {Path(entry["path"]).resolve() for entry in splat_route._transfers_entries()}
    approved.update(Path(meta["input_path"]).resolve() for meta in splat_route._all_metas()
                    if meta.get("input_path") and Path(meta["input_path"]).is_file())
    approved.update(Path(record["source"]["path"]).resolve() for record in captures.list_captures())
    if candidate not in approved:
        raise HTTPException(400, "Choose a Transfers input or register this local source with capture-route.py")
    return candidate


def require_capture(capture_id: str) -> dict:
    try:
        return captures.load_capture(capture_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


def require_route(route_id: str) -> dict:
    try:
        return routes.load_route(route_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


async def _inspect(source: Path, operation_id: str):
    try:
        result = await asyncio.to_thread(captures.inspect_capture, source)
        opregistry.finish(operation_id, result={"capture_id": result["capture_id"]})
    except Exception as exc:
        opregistry.finish(operation_id, status="failed", error=str(exc))
    finally:
        TASKS.pop(operation_id, None)


async def _build(route_id: str, operation_id: str):
    try:
        def report(document):
            ready = sum(point["status"] == "ready" for point in document["points"])
            opregistry.update(operation_id, progress=ready / max(1, len(document["points"])),
                              detail=f"{ready}/{len(document['points'])} viewpoints")

        document = await asyncio.to_thread(routes.build_route, route_id, report)
        opregistry.finish(operation_id,
                          status="succeeded" if document["status"] == "completed" else "cancelled" if document["status"] == "paused" else "failed",
                          result={"route_id": route_id, "route_status": document["status"]},
                          error=document.get("error") or "")
    except Exception as exc:
        document = routes.load_route(route_id)
        document.update(status="failed", error=str(exc))
        routes.save_route(document)
        opregistry.finish(operation_id, status="failed", error=str(exc))
    finally:
        TASKS.pop(route_id, None)


def start_build(route_id: str) -> str:
    if route_id in TASKS:
        raise HTTPException(409, "This route is already processing")
    (routes.route_dir(route_id) / "stop-requested").unlink(missing_ok=True)
    operation_id = opregistry.start("route.build", detail=route_id)
    TASKS[route_id] = asyncio.create_task(_build(route_id, operation_id))
    return operation_id


@router.get("/captures")
def list_captures():
    return {"captures": captures.list_captures()}


@router.get("/capture-atlases")
def list_capture_atlases():
    return {"atlases": capture_atlas.list_atlases()}


@router.get("/capture-atlases/{atlas_id}/{filename}")
def get_capture_atlas_file(atlas_id: str, filename: str):
    try:
        path = capture_atlas.atlas_file(atlas_id, filename)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path, headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
                                       "Referrer-Policy": "no-referrer"})


@router.post("/captures/inspect", status_code=202)
async def inspect_capture(body: CaptureInput):
    source = accepted_input(body.input_path)
    operation_id = opregistry.start("capture.inspect", detail=source.name)
    TASKS[operation_id] = asyncio.create_task(_inspect(source, operation_id))
    return {"operation_id": operation_id}


@router.get("/captures/{capture_id}")
def get_capture(capture_id: str):
    return require_capture(capture_id)


@router.put("/captures/{capture_id}/clock")
def update_clock(capture_id: str, body: CaptureClock):
    document = require_capture(capture_id)
    document["clock"] = {**body.model_dump(), "source": "operator_verified", "verified": True}
    manifests.atomic_write_json(captures.capture_path(capture_id), document)
    return document["clock"]


@router.get("/routes")
def list_routes():
    return {"routes": routes.list_routes()}


@router.post("/routes", status_code=202)
async def build_route(body: routes.RouteSpec):
    require_capture(body.capture_id)
    try:
        document = routes.create_route(body)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"route_id": document["route_id"], "operation_id": start_build(document["route_id"])}


@router.get("/routes/{route_id}")
def get_route(route_id: str):
    document = require_route(route_id)
    return {**document, "reconstruction_prepared": (routes.route_dir(route_id) / "reconstruction-plan.json").is_file()}


@router.post("/routes/{route_id}/resume", status_code=202)
async def resume_route(route_id: str):
    require_route(route_id)
    return {"route_id": route_id, "operation_id": start_build(route_id)}


@router.post("/routes/{route_id}/stop")
def stop_route(route_id: str):
    require_route(route_id)
    (routes.route_dir(route_id) / "stop-requested").touch()
    return {"route_id": route_id, "stop_requested": True}


@router.get("/routes/{route_id}/panoramas/{index}")
def get_panorama(route_id: str, index: int):
    document = require_route(route_id)
    if index < 0 or index >= len(document["points"]) or document["points"][index]["status"] != "ready":
        raise HTTPException(404, "Viewpoint is not ready")
    path = routes.route_dir(route_id) / f"pano-{index:05d}.jpg"
    if not path.is_file():
        raise HTTPException(404, "Viewpoint file is missing; resume the route")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.post("/routes/{route_id}/reconstruction/prepare")
def prepare_reconstruction(route_id: str, body: route_reconstruction.ReconstructionSpec):
    require_route(route_id)
    try:
        return route_reconstruction.prepare(route_id, body)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/routes/{route_id}/reconstruction")
def get_reconstruction(route_id: str):
    require_route(route_id)
    directory = routes.route_dir(route_id)
    pointer = manifests.read_json(directory / "reconstruction-plan.json") or {}
    fingerprint = pointer.get("fingerprint", "")
    if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
        raise HTTPException(404, "Prepare reconstruction inputs first")
    document = manifests.read_json(directory / "reconstruction" / fingerprint[:16] / "plan.json")
    if not document:
        raise HTTPException(404, "Reconstruction plan is missing")
    return document


def reconcile_routes() -> None:
    for entry in routes.list_routes():
        if entry["status"] == "running":
            document = routes.load_route(entry["route_id"])
            document.update(status="paused", error="Service restarted; resume to continue")
            routes.save_route(document)


async def shutdown_tasks() -> None:
    active = list(TASKS.items())
    for task_id, task in active:
        if task_id.startswith("route_") and not task.done():
            (routes.route_dir(task_id) / "stop-requested").touch()
    if active:
        await asyncio.wait([task for _, task in active], timeout=10)
