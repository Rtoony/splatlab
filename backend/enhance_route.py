"""Enhance this view: one viewer frame -> an AI-enhanced still (NVIDIA Difix), owner go 2026-10-01.

The realism lab (~/reports/2026-10-01-realism-lab/) measured Difix guided by the NEAREST REAL CAPTURE PHOTO as the
clear winner for making weak splat views look real (blur, holes, floater haze), and also its failure mode: a
reference that doesn't resemble the view pastes in the wrong content. So the reference is used only when a real
camera stood near this spot facing the same way; otherwise plain Difix runs.

The browser sends what it is showing (a PNG of the canvas) and where the camera is IN THE SPLAT'S FRAME. Nothing about
the scene changes: every shot is a new folder under <job>/_enhance/shots/<id>/ holding input.png (what you saw),
output.png (AI-enhanced), ref.png (the real photo it leaned on, if any) and shot.json. Generated detail is labelled
as such everywhere (route payload, PNG metadata, UI badge) — it never masquerades as capture.

Runs out of process in the isolated `difix` conda env, under the GPU arbiter. Reference views for Raw 360 scenes are
the upright pinhole cuts of the fisheye frames (langfield/spirula_views.py), built once per scene and cached.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import splat_route

router = APIRouter()

ENHANCE_DIRNAME = "_enhance"
DIFIX_PYTHON = Path.home() / "miniconda3" / "envs" / "difix" / "bin" / "python"
DIFIX_SRC = Path.home() / "tools" / "difix3d" / "src"
RUNNER = Path(__file__).resolve().parent / "enhance" / "difix_enhance.py"
VIEWS_SCRIPT = Path(__file__).resolve().parent / "langfield" / "spirula_views.py"
ENHANCE_VRAM_MB = 14_000              # difix_ref at 1024 long side peaked at 12.4 GB (realism lab)
REF_MAX_DIST_M = 3.0                   # a reference farther than this, or facing away, misleads more than it helps
REF_MIN_FACING = 0.5                   # cos(60 deg)
MAX_IMAGE_BYTES = 24 * 1024 * 1024
_SHOT_RE = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{6}$")
_VIEW_LOCKS: dict[str, asyncio.Lock] = {}


class EnhanceCamera(BaseModel):
    position: list[float] = Field(..., min_length=3, max_length=3)
    forward: list[float] = Field(..., min_length=3, max_length=3)


class EnhanceBody(BaseModel):
    image: str = Field(..., description="data:image/png;base64,... of the frame as shown")
    camera: EnhanceCamera | None = None


def available() -> bool:
    return DIFIX_PYTHON.is_file() and (DIFIX_SRC / "pipeline_difix.py").is_file() and RUNNER.is_file()


def nearest_view(views: list[dict], position: list[float], forward: list[float]) -> tuple[int | None, float, float]:
    """Index of the real view closest in position while facing the same way, with its distance (m) and facing
    (cosine). views[i]["c2w"] is a 3x4 OpenGL camera-to-world in the splat frame (forward = -z column)."""
    fl = math.sqrt(sum(f * f for f in forward)) or 1.0
    fw = [f / fl for f in forward]
    best, best_score, best_d, best_c = None, float("inf"), float("inf"), -1.0
    for i, v in enumerate(views):
        m = v["c2w"]
        p = [m[0][3], m[1][3], m[2][3]]
        vf = [-m[0][2], -m[1][2], -m[2][2]]
        d = math.dist(p, position)
        c = sum(a * b for a, b in zip(vf, fw))
        score = d - 2.0 * c
        if score < best_score:
            best, best_score, best_d, best_c = i, score, d, c
    return best, best_d, best_c


def _decode_image(data_url: str) -> bytes:
    m = re.match(r"^data:image/(png|jpeg|webp);base64,(.+)$", data_url, re.S)
    if not m:
        raise HTTPException(status_code=400, detail="image must be a data:image/png|jpeg|webp;base64 URL")
    raw = base64.b64decode(m.group(2), validate=False)
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image is empty or too large")
    return raw


async def _ensure_views(job_dir: Path) -> Path | None:
    """Upright pinhole reference views for a Raw 360 scene (cached in _enhance/views). None if not Raw 360."""
    if not (job_dir / "_spirula" / "data" / "sparse" / "0" / "images.bin").is_file():
        return None
    vd = job_dir / ENHANCE_DIRNAME / "views"
    if (vd / "cameras.json").is_file():
        return vd
    lock = _VIEW_LOCKS.setdefault(str(job_dir), asyncio.Lock())
    async with lock:
        if (vd / "cameras.json").is_file():
            return vd
        tmp = vd.with_name("views.tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        rc, _out, err = await splat_route._run_capture_subprocess(
            ["env", "PYTHONNOUSERSITE=1", str(splat_route.LANGFIELD_ENV_PYTHON), str(VIEWS_SCRIPT), str(job_dir), str(tmp)])
        if rc != 0 or not (tmp / "cameras.json").is_file():
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(status_code=500, detail="could not build reference views: "
                                + err.decode("utf-8", "replace")[-300:])
        shutil.rmtree(vd, ignore_errors=True)
        tmp.rename(vd)
    return vd


@router.post("/jobs/{job_id}/enhance")
async def enhance_view(job_id: str, body: EnhanceBody) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = splat_route._read_meta(job_id)
    if not meta or meta.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Enhance needs a completed scene")
    if not available():
        raise HTTPException(status_code=503, detail="Enhance toolchain (difix env / Difix3D) is not installed")
    job_dir = Path(meta["output_dir"])
    raw = _decode_image(body.image)
    t0 = time.time()
    shot_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
    sd = job_dir / ENHANCE_DIRNAME / "shots" / shot_id
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "input.png").write_bytes(raw)

    ref = None
    ref_info: dict[str, Any] = {"used": False}
    if body.camera is not None:
        vd = await _ensure_views(job_dir)
        if vd is not None:
            views = json.loads((vd / "cameras.json").read_text())["views"]
            i, d, c = nearest_view(views, body.camera.position, body.camera.forward)
            ref_info = {"used": False, "view": i, "distance_m": round(d, 2), "facing": round(c, 2),
                        "source": views[i]["source"] if i is not None else None}
            if i is not None and d <= REF_MAX_DIST_M and c >= REF_MIN_FACING:
                shutil.copy(vd / "frames" / f"frame_{i:03d}.png", sd / "ref.png")
                ref = sd / "ref.png"
                ref_info["used"] = True
            else:
                ref_info["why_not"] = ("no real photo was taken near here facing this way — a mismatched "
                                       "reference pastes in the wrong things, so plain Difix ran")

    cmd = ["env", "PYTHONNOUSERSITE=1", str(DIFIX_PYTHON), str(RUNNER), str(sd / "input.png"),
           str(ref) if ref else "-", str(sd / "output.png")]
    try:
        rc, out, err = await splat_route.gpu_arbiter.run_gpu_operation(
            lane="splat-enhance", operation_id=f"{job_id}:{shot_id}", vram_mb=ENHANCE_VRAM_MB,
            operation=lambda: splat_route._run_capture_subprocess(cmd))
    except splat_route.gpu_arbiter.GPUArbiterUnavailable as exc:
        shutil.rmtree(sd, ignore_errors=True)
        raise HTTPException(status_code=503, detail=f"Enhance blocked: {exc}") from exc
    if rc != 0 or not (sd / "output.png").is_file():
        tail = err.decode("utf-8", "replace").splitlines()[-4:]
        shutil.rmtree(sd, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Enhance failed: " + " | ".join(tail)[-400:])
    try:
        run = json.loads(out.decode().strip().splitlines()[-1])
    except (ValueError, IndexError):
        run = {}
    base = f"/api/splat/jobs/{job_id}/enhance/{shot_id}"
    doc = {"job_id": job_id, "shot_id": shot_id, "label": "AI-enhanced",
           "note": "Generated detail (NVIDIA Difix), not a capture.", "model": run.get("model"),
           "model_ms": run.get("ms"), "seconds": round(time.time() - t0, 1), "camera": body.camera.model_dump() if body.camera else None,
           "reference": ref_info, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (sd / "shot.json").write_text(json.dumps(doc, indent=1))
    return {**doc, "before_url": f"{base}/input", "after_url": f"{base}/output",
            "ref_url": f"{base}/ref" if ref else None}


@router.get("/jobs/{job_id}/enhance/{shot_id}/{name}")
async def enhance_file(job_id: str, shot_id: str, name: str):
    if not splat_route._safe_job_id(job_id) or not _SHOT_RE.match(shot_id) or name not in ("input", "output", "ref"):
        raise HTTPException(status_code=404, detail="Not found")
    p = splat_route._job_dir(job_id) / ENHANCE_DIRNAME / "shots" / shot_id / f"{name}.png"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    fname = f"{job_id}-{shot_id}-{'ai-enhanced' if name == 'output' else name}.png"
    return FileResponse(str(p), media_type="image/png", filename=fname, headers={"Cache-Control": "no-cache"})
