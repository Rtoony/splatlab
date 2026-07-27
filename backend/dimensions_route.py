"""Survey dimensions: server-persisted two-point measurements for a scene.

Pure stdlib, CPU-only, no GPU/ML — same "small feature, own router" shape as
geo_route.py, not langfield_overrides.py (that module runs in the GPU-worker
process and its id scheme is numpy/index-file-dependent; dimensions have no
per-record sidecar file, so none of that applies here).

Previously these lived only in the browser's sessionStorage (spark-scene-
viewer.tsx) and were lost across sessions/devices. This is the same idea,
moved server-side, plus a CSV export — RToony's stated goal for this tool is
"generate files I can share," which sessionStorage-only measurements can't do.

Layout: <job_dir>/dimensions.json — a flat list of records:
  [{id, a: [x,y,z], b: [x,y,z], label, created_at}]

`id` is CLIENT-supplied (the frontend already generates one, Date.now()) and
POST is an idempotent upsert keyed on it — dimensions have no filename-safety
reason to need a server-minted id, and upsert-by-id lets drag-to-move reuse
this same route instead of needing a fourth (PATCH) one.

Deliberately NOT behind require_heavy_work_admitted — pure metadata, same
admission policy as geo_route.py's anchor/scale-adjacent routes (stays
available during hardware maintenance).
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import splat_route

router = APIRouter()

MANIFEST_NAME = "dimensions.json"


def _manifest_path(job_dir: Path) -> Path:
    return job_dir / MANIFEST_NAME


def load_dimensions(job_dir: Path) -> list[dict]:
    p = _manifest_path(job_dir)
    if not p.is_file():
        return []
    try:
        items = json.loads(p.read_text())
        return items if isinstance(items, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_dimensions(job_dir: Path, items: list[dict]) -> None:
    _manifest_path(job_dir).write_text(json.dumps(items, indent=1))


def upsert_dimension(job_dir: Path, *, dim_id: str, a: list[float], b: list[float], label: str) -> dict:
    items = load_dimensions(job_dir)
    record = {"id": dim_id, "a": a, "b": b, "label": label, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for i, rec in enumerate(items):
        if rec.get("id") == dim_id:
            record["created_at"] = rec.get("created_at", record["created_at"])
            items[i] = record
            break
    else:
        items.append(record)
    _save_dimensions(job_dir, items)
    return record


def delete_dimension(job_dir: Path, dim_id: str) -> bool:
    items = load_dimensions(job_dir)
    kept = [r for r in items if r.get("id") != dim_id]
    if len(kept) == len(items):
        return False
    _save_dimensions(job_dir, kept)
    return True


def _require_job_dir(job_id: str) -> Path:
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="not found")
    job_dir = splat_route._job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="job not found")
    return job_dir


class DimensionBody(BaseModel):
    id: str = Field(..., min_length=1, max_length=40)
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    label: str = Field(default="", max_length=80)


@router.get("/jobs/{job_id}/dimensions")
async def list_dimensions(job_id: str) -> dict[str, Any]:
    job_dir = _require_job_dir(job_id)
    return {"job_id": job_id, "dimensions": load_dimensions(job_dir)}


@router.post("/jobs/{job_id}/dimensions")
async def add_dimension(job_id: str, body: DimensionBody) -> dict[str, Any]:
    job_dir = _require_job_dir(job_id)
    # pydantic accepts NaN/Infinity for a bare float, and a non-finite endpoint
    # silently poisons every length derived from it — including a scale
    # calibration that cites this measurement as its evidence. Checked here
    # rather than in a field_validator for the same reason geo_route does it
    # this way: FastAPI's 422 body echoes the offending input, and NaN cannot
    # be encoded into that response at all.
    if not all(math.isfinite(c) for c in (*body.a, *body.b)):
        raise HTTPException(status_code=400, detail="measurement endpoints must be finite")
    record = upsert_dimension(job_dir, dim_id=body.id, a=list(body.a), b=list(body.b), label=body.label)
    return {"job_id": job_id, "dimension": record}


@router.delete("/jobs/{job_id}/dimensions/{dim_id}")
async def remove_dimension(job_id: str, dim_id: str) -> dict[str, Any]:
    job_dir = _require_job_dir(job_id)
    ok = delete_dimension(job_dir, dim_id)
    if not ok:
        raise HTTPException(status_code=404, detail="dimension not found")
    return {"job_id": job_id, "deleted": dim_id}


@router.get("/jobs/{job_id}/dimensions/export")
async def export_dimensions(job_id: str, fmt: str = "csv") -> Response:
    job_dir = _require_job_dir(job_id)
    items = load_dimensions(job_dir)
    if fmt == "json":
        return Response(
            content=json.dumps({"job_id": job_id, "dimensions": items}, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{job_id}-dimensions.json"'},
        )
    if fmt != "csv":
        raise HTTPException(status_code=400, detail="fmt must be csv or json")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "label", "ax", "ay", "az", "bx", "by", "bz", "length_units", "created_at"])
    for rec in items:
        a, b = rec.get("a", [0, 0, 0]), rec.get("b", [0, 0, 0])
        length = sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
        writer.writerow([rec.get("id", ""), rec.get("label", ""), *a, *b, round(length, 4), rec.get("created_at", "")])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{job_id}-dimensions.csv"'},
    )
