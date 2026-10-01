"""Record walk: the walker's canvas as a video, plus an optional "realism finish" (owner go 2026-10-01).

The browser records exactly what the 3D canvas draws (MediaRecorder on canvas.captureStream; the HUD is DOM, so it is
not in the clip) and posts the WebM here. The server keeps it untouched and makes a plain MP4 (H.264, 24 fps, at most
1280 wide), which downloads and plays everywhere and is what nexus-film can take.

Realism finish = nexus-film's `film finish --preset realism` (SeedVR2 7B sharp at the clip's own size): steadier and
crisper, invents nothing (realism lab, ~/reports/2026-10-01-realism-lab/). It runs in the background under the GPU
arbiter; ComfyUI is started for it when it is down and stopped again afterwards, so it never sits on VRAM unasked.

Every walk is a folder <job>/_walks/<id>/: recording.webm (as recorded), walk.mp4, realism.mp4 (+ .json sidecar),
walk.json. Nothing about the scene changes, and nothing is ever deleted here.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

import splat_route

router = APIRouter()

WALKS_DIRNAME = "_walks"
MAX_UPLOAD_BYTES = 400 * 1024 * 1024
MAX_WIDTH = 1280
FPS = 24
FILM_ROOT = Path.home() / "projects" / "nexus-film"
FILM_PYTHON = "/usr/bin/python3"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_UNIT = "comfyui.service"
FINISH_VRAM_MB = 24_000
# Measured: 120 frames at 1024x576 in 85 s (2026-10-01) -> seconds of compute per (second of clip x 1024x576 pixels).
FINISH_S_PER_CLIP_S = 17.0
_WALK_RE = re.compile(r"^[0-9]{8}T[0-9]{6}-[0-9a-f]{6}$")
_FINISH_LOCK = asyncio.Lock()
_FINISHING: dict[str, asyncio.Task] = {}


def finish_available() -> bool:
    return (FILM_ROOT / "bin" / "film").is_file()


def estimate_finish_seconds(seconds: float, width: int, height: int) -> int:
    return int(round(FINISH_S_PER_CLIP_S * seconds * (width * height) / (1024 * 576)))


def _walk_dir(job_id: str, walk_id: str) -> Path:
    if not splat_route._safe_job_id(job_id) or not _WALK_RE.match(walk_id):
        raise HTTPException(status_code=404, detail="Not found")
    return splat_route._job_dir(job_id) / WALKS_DIRNAME / walk_id


def _read(wd: Path) -> dict[str, Any]:
    try:
        return json.loads((wd / "walk.json").read_text())
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="Walk not found") from None


def _write(wd: Path, doc: dict[str, Any]) -> None:
    tmp = wd / "walk.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(wd / "walk.json")


def _public(job_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    """walk.json as the browser sees it: URLs, and a finish that died with a restart reported as interrupted."""
    d = dict(doc)
    key = f"{job_id}:{d['walk_id']}"
    if d.get("finish", {}).get("status") == "running" and key not in _FINISHING:
        d["finish"] = {**d["finish"], "status": "interrupted", "error": "the service restarted mid-finish; run it again"}
    base = f"/api/splat/jobs/{job_id}/walks/{d['walk_id']}"
    d["video_url"] = f"{base}/walk"
    d["realism_url"] = f"{base}/realism" if d.get("finish", {}).get("status") == "done" else None
    return d


async def _probe(path: Path) -> tuple[int, int, int]:
    rc, out, _ = await splat_route._run_capture_subprocess(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,nb_read_frames", "-of", "csv=p=0", str(path)])
    try:
        w, h, n = (int(x) for x in out.decode().strip().split(",")[:3])
    except ValueError:
        raise HTTPException(status_code=500, detail="could not read the converted video") from None
    return w, h, n


@router.post("/jobs/{job_id}/walks")
async def record_walk(job_id: str, request: Request) -> dict[str, Any]:
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = splat_route._read_meta(job_id)
    if not meta or meta.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Record walk needs a completed scene")
    raw = await request.body()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="the recording is empty or too large")
    if raw[:4] != b"\x1a\x45\xdf\xa3":
        raise HTTPException(status_code=400, detail="expected a WebM recording")
    walk_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
    wd = Path(meta["output_dir"]) / WALKS_DIRNAME / walk_id
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "recording.webm").write_bytes(raw)
    # Constant 24 fps (MediaRecorder writes variable frame timing), even sizes for H.264, never upscaled.
    vf = f"fps={FPS},scale='trunc(min({MAX_WIDTH},iw)/2)*2':-2"
    rc, _out, err = await splat_route._run_capture_subprocess(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wd / "recording.webm"), "-vf", vf, "-an",
         "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(wd / "walk.mp4")])
    if rc != 0 or not (wd / "walk.mp4").is_file():
        raise HTTPException(status_code=500, detail="could not convert the recording: "
                            + err.decode("utf-8", "replace")[-300:])
    w, h, n = await _probe(wd / "walk.mp4")
    seconds = round(n / FPS, 1)
    doc = {"job_id": job_id, "walk_id": walk_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "width": w, "height": h, "frames": n, "seconds": seconds, "fps": FPS,
           "recording_bytes": len(raw), "finish_available": finish_available(),
           "finish_estimate_s": estimate_finish_seconds(seconds, w, h), "finish": {"status": "none"}}
    _write(wd, doc)
    return _public(job_id, doc)


@router.get("/jobs/{job_id}/walks/{walk_id}")
async def walk_status(job_id: str, walk_id: str) -> dict[str, Any]:
    return _public(job_id, _read(_walk_dir(job_id, walk_id)))


def _comfy_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


async def _systemctl(*args: str) -> int:
    p = await asyncio.create_subprocess_exec("systemctl", "--user", *args, env=os.environ.copy(),
                                             stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    return await p.wait()


async def _ensure_comfy() -> bool:
    """True when WE started ComfyUI (so we stop it again afterwards)."""
    if await asyncio.to_thread(_comfy_alive):
        return False
    await _systemctl("start", COMFY_UNIT)
    for _ in range(90):
        if await asyncio.to_thread(_comfy_alive):
            return True
        await asyncio.sleep(2)
    raise RuntimeError("ComfyUI did not come up within 3 minutes (systemctl --user status comfyui.service)")


async def _finish(job_id: str, wd: Path) -> None:
    doc = _read(wd)
    started_comfy = False
    try:
        async with _FINISH_LOCK:                        # one SeedVR2 finish at a time
            async def op():
                nonlocal started_comfy
                started_comfy = await _ensure_comfy()
                return await splat_route._run_capture_subprocess(
                    [FILM_PYTHON, str(FILM_ROOT / "bin" / "film"), "finish", str(wd / "walk.mp4"),
                     "--preset", "realism", "--out", str(wd / "realism.mp4")])
            rc, _out, err = await splat_route.gpu_arbiter.run_gpu_operation(
                lane="splat-walk-finish", operation_id=f"{job_id}:{wd.name}", vram_mb=FINISH_VRAM_MB, operation=op)
        if rc != 0 or not (wd / "realism.mp4").is_file():
            raise RuntimeError(" | ".join(err.decode("utf-8", "replace").splitlines()[-3:])[-400:] or f"exit {rc}")
        doc["finish"] = {**doc["finish"], "status": "done", "seconds": round(time.time() - doc["finish"]["started"], 1)}
    except Exception as exc:  # noqa: BLE001 - reported to the UI, never swallowed
        doc["finish"] = {**doc["finish"], "status": "failed", "error": str(exc)[-400:]}
    finally:
        if started_comfy:
            await _systemctl("stop", COMFY_UNIT)
        _write(wd, doc)
        _FINISHING.pop(f"{job_id}:{wd.name}", None)


@router.post("/jobs/{job_id}/walks/{walk_id}/finish")
async def finish_walk(job_id: str, walk_id: str) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    wd = _walk_dir(job_id, walk_id)
    doc = _read(wd)
    if not finish_available():
        raise HTTPException(status_code=503, detail="nexus-film is not installed at ~/projects/nexus-film")
    key = f"{job_id}:{walk_id}"
    if key in _FINISHING or doc["finish"].get("status") == "done":
        return _public(job_id, doc)
    doc["finish"] = {"status": "running", "preset": "realism", "started": time.time(),
                     "estimate_s": doc.get("finish_estimate_s")}
    _write(wd, doc)
    _FINISHING[key] = asyncio.create_task(_finish(job_id, wd))
    return _public(job_id, doc)


@router.get("/jobs/{job_id}/walks/{walk_id}/{name}")
async def walk_file(job_id: str, walk_id: str, name: str):
    if name not in ("walk", "realism"):
        raise HTTPException(status_code=404, detail="Not found")
    p = _walk_dir(job_id, walk_id) / f"{name}.mp4"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    fname = f"{job_id}-walk-{walk_id}{'-realism-finish' if name == 'realism' else ''}.mp4"
    return FileResponse(str(p), media_type="video/mp4", filename=fname, headers={"Cache-Control": "no-cache"})
