"""Checkpointed local panorama routes, with explicit source and GPS lineage."""

from __future__ import annotations

import fcntl
import math
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

import artifact_manifest as manifests
import capture_records as captures

ROUTE_SCHEMA = "dev.splatlab.route/v1"


class RouteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    capture_id: str
    start_s: float = Field(default=0, ge=0)
    end_s: float | None = Field(default=None, gt=0)
    interval_s: float = Field(default=5, ge=1, le=60)
    width: int = Field(default=2048, ge=1024, le=4096, multiple_of=2)
    fisheye_fov: float = Field(default=200, ge=180, le=220)
    pitch_deg: float = Field(default=0, ge=-180, le=180)
    roll_deg: float = Field(default=0, ge=-180, le=180)
    yaw_deg: float = Field(default=0, ge=-180, le=180)
    budget_s: int = Field(default=3600, ge=30, le=28800)

    @model_validator(mode="after")
    def ordered(self):
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("End time must follow start time")
        return self


def route_dir(route_id: str) -> Path:
    if not re.fullmatch(r"route_[a-f0-9]{16}", route_id):
        raise ValueError("Invalid route ID")
    return captures.DATA_ROOT / "routes" / route_id


def load_route(route_id: str) -> dict:
    document = manifests.read_json(route_dir(route_id) / "route-manifest.json")
    if not document or document.get("schema") != ROUTE_SCHEMA:
        raise ValueError("Route not found")
    return document


def save_route(document: dict) -> None:
    manifests.atomic_write_json(route_dir(document["route_id"]) / "route-manifest.json", document)


def create_route(spec: RouteSpec) -> dict:
    capture = captures.load_capture(spec.capture_id)
    end = min(spec.end_s if spec.end_s is not None else capture["duration_s"], capture["duration_s"])
    if spec.start_s >= end:
        raise ValueError("Selected segment is outside the video")
    count = math.ceil((end - spec.start_s) / spec.interval_s)
    if count > 5000:
        raise ValueError("Route exceeds 5,000 viewpoints; select a shorter segment or larger interval")
    route_id = "route_" + uuid.uuid4().hex[:16]
    points = [{"index": index, "time_s": round(spec.start_s + index * spec.interval_s, 6),
               "status": "pending", "timestamp_kind": "requested-video-seek",
               "exact_frame_pts_verified": False} for index in range(count)]
    document = {"schema": ROUTE_SCHEMA, "route_id": route_id, "capture_id": spec.capture_id,
                "name": capture["source"]["name"], "created_at": manifests.utc_now(),
                "source_sha256": capture["source"]["sha256"], "clock": capture["clock"],
                "spec": spec.model_dump(), "status": "pending", "points": points,
                "sections": [], "provenance": "observed", "seconds": 0,
                "warnings": ["Generic lens reprojection; calibration and stabilization are not verified",
                             "Viewpoint times are requested seeks, not verified decoded presentation timestamps",
                             "GPS placement is approximate until clock alignment is verified"]}
    save_route(document)
    return document


def panorama_command(source: Path, streams: list[dict], spec: RouteSpec,
                     timestamp: float, destination: Path) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is unavailable")
    square = len(streams) == 2 and all(stream.get("width") == stream.get("height") for stream in streams)
    equirect = len(streams) == 1 and streams[0].get("width") == 2 * streams[0].get("height", 0)
    if not square and not equirect:
        raise ValueError("Expected synchronized dual fisheyes or a 2:1 panorama video")
    projection = (f"v360=input={'dfisheye' if square else 'e'}:output=e"
                  f":w={spec.width}:h={spec.width // 2}:interp=lanczos"
                  f":yaw={spec.yaw_deg}:pitch={spec.pitch_deg}:roll={spec.roll_deg}")
    if square:
        projection += f":ih_fov={spec.fisheye_fov}:iv_fov={spec.fisheye_fov}"
        graph = f"[0:{streams[0]['index']}][0:{streams[1]['index']}]hstack=inputs=2[lenses];[lenses]{projection}[pano]"
    else:
        graph = f"[0:{streams[0]['index']}]{projection}[pano]"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "none", "-threads", "2",
               "-ss", f"{timestamp:.6f}", "-i", str(source), "-filter_complex_threads", "1",
               "-filter_complex", graph, "-map", "[pano]", "-frames:v", "1", "-an",
               "-c:v", "mjpeg", "-q:v", "3", "-threads", "2", "-y", str(destination)]
    if shutil.which("taskset") and hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        selected = [cpu for cpu in available if 8 <= cpu <= 15] or available[:4]
        command = ["taskset", "-c", ",".join(map(str, selected)), *command]
    return command


def frame_quality(path: Path) -> dict:
    with Image.open(path) as original:
        original.verify()
    with Image.open(path) as original:
        preview = original.convert("L").resize((512, 256))
        band = preview.crop((0, 64, 512, 192))
        edge = band.filter(ImageFilter.FIND_EDGES)
        histogram = band.histogram()
        count = sum(histogram)
        return {"sharpness": round(ImageStat.Stat(edge).var[0], 3),
                "dark_fraction": round(sum(histogram[:8]) / count, 4),
                "bright_fraction": round(sum(histogram[248:]) / count, 4),
                "width": original.width, "height": original.height}


class FrameError(ValueError):
    pass


def extract_point(source: Path, streams: list[dict], spec: RouteSpec, point: dict,
                  directory: Path, deadline: float) -> dict:
    destination = directory / f"pano-{point['index']:05d}.jpg"
    staged = destination.with_name(destination.stem + ".building.jpg")
    stop_path = directory / "stop-requested"
    command = panorama_command(source, streams, spec, point["time_s"], staged)
    try:
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log,
                                       env=captures.worker_env(), start_new_session=True)
            try:
                while process.poll() is None:
                    if stop_path.exists() or time.monotonic() > deadline:
                        raise TimeoutError("Stopped or frame budget exceeded; route can resume")
                    time.sleep(0.2)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            if process.returncode:
                raise FrameError(f"Panorama extraction failed at {point['time_s']}s; inspect worker.log")
        try:
            quality = frame_quality(staged)
        except (UnidentifiedImageError, ValueError) as exc:
            raise FrameError(f"Invalid panorama at {point['time_s']}s") from exc
        staged.replace(destination)
        return {"status": "ready", "file": destination.name, "quality": quality,
                "sha256": manifests.sha256_file(destination), "error": None}
    finally:
        staged.unlink(missing_ok=True)


def build_route(route_id: str, progress: Callable[[dict], None] | None = None) -> dict:
    directory = route_dir(route_id)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = captures.DATA_ROOT / "route-worker.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another route is processing; resume after it finishes") from exc
        document = load_route(route_id)
        spec = RouteSpec(**document["spec"])
        capture = captures.load_capture(spec.capture_id)
        source = Path(capture["source"]["path"])
        if not manifests.same_file_identity(source, capture["source"]):
            raise ValueError("Capture source changed or is disconnected; inspect it again")
        if document["clock"] != capture["clock"]:
            raise ValueError("Capture clock changed; create a new route to preserve lineage")
        started = time.monotonic()
        prior_seconds = document.get("seconds", 0)
        stop_path = directory / "stop-requested"
        document.update(status="running", error=None)
        save_route(document)
        try:
            for point in document["points"]:
                if stop_path.exists() or time.monotonic() - started >= spec.budget_s:
                    document["status"] = "paused"
                    break
                destination = directory / f"pano-{point['index']:05d}.jpg"
                if point["status"] == "ready" and destination.is_file():
                    if manifests.sha256_file(destination) == point.get("sha256"):
                        continue
                import gpu_arbiter

                gpu_arbiter.require_backup_idle()
                try:
                    point.update(extract_point(source, capture["streams"], spec, point, directory,
                                               min(started + spec.budget_s, time.monotonic() + 120)))
                    point["location"] = captures.location_at(capture, point["time_s"])
                except FrameError as exc:
                    point.update(status="failed", error=str(exc))
                save_route(document)
                if progress:
                    progress(document)
            else:
                failed = sum(point["status"] == "failed" for point in document["points"])
                document["status"] = "partial" if failed else "completed"
                if failed:
                    document["error"] = f"{failed} viewpoints failed; completed viewpoints are retained. Resume to retry."
        except Exception as exc:
            document.update(status="paused" if isinstance(exc, TimeoutError) else "failed",
                            error=str(exc))
        finally:
            if not manifests.same_file_identity(source, capture["source"]):
                document.update(status="failed", error="Source changed during extraction; inspect it again")
                for point in document["points"]:
                    point["status"] = "stale"
            document["seconds"] = round(prior_seconds + time.monotonic() - started, 3)
            save_route(document)
        return document


def list_routes() -> list[dict]:
    result = []
    for path in sorted((captures.DATA_ROOT / "routes").glob("route_*/route-manifest.json")):
        document = manifests.read_json(path)
        if document and document.get("schema") == ROUTE_SCHEMA:
            result.append({"route_id": document["route_id"], "name": document["name"],
                           "status": document["status"], "points": len(document["points"]),
                           "ready": sum(point["status"] == "ready" for point in document["points"])})
    return result
