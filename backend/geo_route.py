"""Locate-in-the-world lane: pin a splat scene to real WGS84 coordinates.

A scene is "located" by a single geo anchor stored on job meta (survey-lane
sibling of meters_per_unit, same _patch_meta idiom, reaches every client via
the **meta spread in _job_payload):

    meta["geo"] = {
        "v": 1,
        "lat": <WGS84 degrees>, "lon": <WGS84 degrees>,
        "alt_m": <anchor altitude, meters, optional>,
        "heading_deg": <compass bearing, deg CW from true north, of the scene's
                        +Y ground axis — i.e. the footprint image's "up">,
        "anchor_scene": [x, y],  # scene-unit ground point that sits at (lat, lon)
        "source": "map" | "exif" | "manual",
        "set_at": <iso8601 utc>,
    }

Together with meters_per_unit this fully determines the scene->world transform:
ENU offset of scene ground point (x, y) from the anchor, with th = radians(heading):
    east  = s * ( (x-ax)*cos(th) + (y-ay)*sin(th) )
    north = s * (-(x-ax)*sin(th) + (y-ay)*cos(th) )
    up    = s * z  (+ alt_m)

Everything here is metadata / CPU-only imaging — deliberately NOT behind
require_heavy_work_admitted (same admission policy as /scale and /unpin), so
locating scenes keeps working during GPU hardware maintenance.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import geo_footprint
import splat_route

router = APIRouter()

GEO_SOURCES = {"map", "exif", "manual"}
SUGGEST_MAX_IMAGES = 32
SUGGEST_TOOL_TIMEOUT_S = 60


# ── stored-anchor validation ─────────────────────────────────────────────────
class GeoAnchorIn(BaseModel):
    lat: float
    lon: float
    alt_m: float | None = None
    heading_deg: float = 0.0
    anchor_scene: list[float] | None = None
    source: str = "map"


class GeoBody(BaseModel):
    geo: GeoAnchorIn | None


def _validated_geo(g: GeoAnchorIn) -> dict[str, Any]:
    """Normalize + range-check a client anchor into the stored meta["geo"] shape."""
    # json.loads accepts bare NaN/Infinity literals, so finiteness is on us.
    if not (math.isfinite(g.lat) and -90.0 <= g.lat <= 90.0):
        raise HTTPException(status_code=400, detail="lat must be finite and within [-90, 90]")
    if not (math.isfinite(g.lon) and -180.0 <= g.lon <= 180.0):
        raise HTTPException(status_code=400, detail="lon must be finite and within [-180, 180]")
    if not math.isfinite(g.heading_deg):
        raise HTTPException(status_code=400, detail="heading_deg must be finite")
    if g.alt_m is not None and not (math.isfinite(g.alt_m) and abs(g.alt_m) < 20000.0):
        raise HTTPException(status_code=400, detail="alt_m must be finite and within +/-20000 m")
    anchor: list[float] | None = None
    if g.anchor_scene is not None:
        if len(g.anchor_scene) != 2 or not all(math.isfinite(a) and abs(a) < 1e7 for a in g.anchor_scene):
            raise HTTPException(status_code=400, detail="anchor_scene must be [x, y] finite scene units")
        anchor = [float(g.anchor_scene[0]), float(g.anchor_scene[1])]
    if g.source not in GEO_SOURCES:
        raise HTTPException(status_code=400, detail=f"source must be one of {sorted(GEO_SOURCES)}")
    return {
        "v": 1,
        "lat": float(g.lat),
        "lon": float(g.lon),
        "alt_m": float(g.alt_m) if g.alt_m is not None else None,
        "heading_deg": float(g.heading_deg) % 360.0,
        "anchor_scene": anchor,
        "source": g.source,
        "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _require_job_meta(job_id: str) -> dict[str, Any]:
    meta = splat_route._read_meta(job_id) if splat_route._safe_job_id(job_id) else None
    if meta is None:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return meta


def _preview_dir(meta: dict[str, Any], job_id: str) -> Path:
    output_dir = (
        Path(meta["output_dir"]) if meta.get("output_dir") else splat_route.DEFAULT_3D_ROOT / job_id
    )
    return output_dir / splat_route.PREVIEW_DIRNAME


@router.post("/jobs/{job_id}/geo")
async def set_splat_geo(job_id: str, body: GeoBody):
    """Set (body {"geo": {...}}) or clear (body {"geo": null}) a scene's world anchor."""
    _require_job_meta(job_id)
    stored = _validated_geo(body.geo) if body.geo is not None else None
    meta = splat_route._patch_meta(job_id, geo=stored)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return {"ok": True, "job_id": job_id, "geo": meta.get("geo")}


# ── top-down footprint (map overlay) ─────────────────────────────────────────
@router.get("/jobs/{job_id}/geo/footprint")
async def splat_geo_footprint(job_id: str):
    """Bounds + bootstrap for the map-alignment overlay (image itself is the .webp route)."""
    meta = _require_job_meta(job_id)
    made = await asyncio.to_thread(geo_footprint.get_or_make, _preview_dir(meta, job_id))
    payload: dict[str, Any] = {
        "job_id": job_id,
        "available": made is not None,
        "meters_per_unit": meta.get("meters_per_unit"),
        "geo": meta.get("geo"),
    }
    if made is None:
        payload["reason"] = "no exported .ply for this scene yet"
        return payload
    _, fp_meta = made
    payload.update(fp_meta)
    payload["url"] = f"/api/splat/jobs/{job_id}/geo/footprint.webp"
    return payload


@router.get("/jobs/{job_id}/geo/footprint.webp")
async def splat_geo_footprint_image(job_id: str):
    meta = _require_job_meta(job_id)
    made = await asyncio.to_thread(geo_footprint.get_or_make, _preview_dir(meta, job_id))
    if made is None:
        raise HTTPException(status_code=404, detail="footprint unavailable")
    image, _ = made
    return FileResponse(str(image), media_type="image/webp")


# ── GPS suggestion from the capture source ───────────────────────────────────
def _dms_to_deg(triplet: Any, ref: Any) -> float | None:
    """EXIF (deg, min, sec) rationals + N/S/E/W ref -> signed decimal degrees."""
    try:
        d, m, s = (float(v) for v in triplet)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(v) for v in (d, m, s)):
        return None
    deg = d + m / 60.0 + s / 3600.0
    ref_s = ref.decode() if isinstance(ref, bytes) else str(ref or "")
    if ref_s.upper().startswith(("S", "W")):
        deg = -deg
    return deg


def _gps_from_exif(gps: dict[int, Any]) -> tuple[float, float, float | None] | None:
    """Pillow GPS IFD dict -> (lat, lon, alt_m|None); None when not present/parseable."""
    lat = _dms_to_deg(gps.get(2), gps.get(1)) if 2 in gps else None
    lon = _dms_to_deg(gps.get(4), gps.get(3)) if 4 in gps else None
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    if abs(lat) < 1e-9 and abs(lon) < 1e-9:
        return None  # (0, 0) is the classic "GPS chip never locked" placeholder
    alt: float | None = None
    if 6 in gps:
        try:
            alt = float(gps[6])
            ref = gps.get(5, 0)
            ref_i = ref[0] if isinstance(ref, bytes) and ref else int(ref or 0)
            if ref_i == 1:
                alt = -alt
        except (TypeError, ValueError, ZeroDivisionError):
            alt = None
    return lat, lon, alt


_ISO6709 = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)(?:([+-]\d+(?:\.\d+)?))?")


def _parse_iso6709(value: str) -> tuple[float, float, float | None] | None:
    """"+34.0522-118.2437+089.0/" (QuickTime/MP4 location tag) -> (lat, lon, alt)."""
    m = _ISO6709.match(value.strip())
    if not m:
        return None
    try:
        lat, lon = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    alt = float(m.group(3)) if m.group(3) else None
    return lat, lon, alt


def _median_fix(points: list[tuple[float, float, float | None]]) -> tuple[float, float, float | None]:
    lat = statistics.median(p[0] for p in points)
    lon = statistics.median(p[1] for p in points)
    alts = [p[2] for p in points if p[2] is not None]
    return lat, lon, statistics.median(alts) if alts else None


def _suggest_from_image_dir(path: Path) -> dict[str, Any] | None:
    from PIL import Image  # deferred: only this rung needs it

    images = sorted(
        p for p in path.iterdir() if p.suffix.lower() in splat_route.IMAGE_EXTENSIONS
    )[:SUGGEST_MAX_IMAGES]
    points: list[tuple[float, float, float | None]] = []
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                gps = img.getexif().get_ifd(0x8825)
        except Exception:
            continue
        fix = _gps_from_exif(dict(gps)) if gps else None
        if fix:
            points.append(fix)
    if not points:
        return None
    lat, lon, alt = _median_fix(points)
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "source": "exif",
        "detail": f"GPS EXIF median of {len(points)}/{len(images)} photos",
    }


def _suggest_from_video_tags(path: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_format", str(path)],
            capture_output=True, text=True, timeout=SUGGEST_TOOL_TIMEOUT_S,
        )
        tags = {k.lower(): v for k, v in json.loads(out.stdout)["format"].get("tags", {}).items()}
    except Exception:
        return None
    for key in ("location", "location-eng", "com.apple.quicktime.location.iso6709"):
        raw = tags.get(key)
        fix = _parse_iso6709(raw) if raw else None
        if fix:
            return {
                "lat": fix[0], "lon": fix[1], "alt_m": fix[2],
                "source": "exif", "detail": f"container location tag ({key})",
            }
    return None


def _suggest_from_exiftool(path: Path) -> dict[str, Any] | None:
    """Embedded GPS track (Insta360 trailer, GoPro, phone MP4s) via exiftool -ee."""
    exiftool = shutil.which("exiftool")
    if not exiftool:
        return None
    try:
        out = subprocess.run(
            [exiftool, "-n", "-q", "-ee", "-f", "-p", "$gpslatitude,$gpslongitude,$gpsaltitude", str(path)],
            capture_output=True, text=True, timeout=SUGGEST_TOOL_TIMEOUT_S,
        )
    except Exception:
        return None
    points: list[tuple[float, float, float | None]] = []
    for line in out.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            lat, lon = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (abs(lat) < 1e-9 and abs(lon) < 1e-9):
            continue
        alt: float | None = None
        if len(parts) > 2:
            try:
                alt = float(parts[2])
            except ValueError:
                alt = None
        points.append((lat, lon, alt))
    if not points:
        return None
    lat, lon, alt = _median_fix(points)
    return {
        "lat": lat, "lon": lon, "alt_m": alt,
        "source": "exif", "detail": f"embedded GPS track median of {len(points)} fixes (exiftool)",
    }


def _collect_suggestions(input_path: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    suffix = input_path.suffix.lower()
    if input_path.is_dir():
        fix = _suggest_from_image_dir(input_path)
        if fix:
            candidates.append(fix)
    elif input_path.is_file():
        if suffix in splat_route.VIDEO_EXTENSIONS:
            for rung in (_suggest_from_video_tags, _suggest_from_exiftool):
                fix = rung(input_path)
                if fix:
                    candidates.append(fix)
                    break
        elif suffix in splat_route.INSV_EXTENSIONS:
            fix = _suggest_from_exiftool(input_path)
            if fix:
                candidates.append(fix)
        elif suffix in splat_route.IMAGE_EXTENSIONS:
            fix = _suggest_from_image_dir(input_path.parent)
            if fix:
                candidates.append(fix)
    return candidates


@router.get("/jobs/{job_id}/geo/suggest")
async def suggest_splat_geo(job_id: str):
    """Best-effort GPS from the capture source (photo EXIF / container tags /
    embedded track). Empty candidates is a normal answer, never an error."""
    meta = _require_job_meta(job_id)
    raw = meta.get("input_path")
    if not raw:
        return {"job_id": job_id, "candidates": []}
    try:
        candidates = await asyncio.to_thread(_collect_suggestions, Path(raw))
    except Exception:
        candidates = []
    return {"job_id": job_id, "candidates": candidates}


# ── exports ──────────────────────────────────────────────────────────────────
def _geojson_doc(job_id: str, meta: dict[str, Any]) -> str:
    geo = meta["geo"]
    coords = [geo["lon"], geo["lat"]] + ([geo["alt_m"]] if geo.get("alt_m") is not None else [])
    doc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": coords},
                "properties": {
                    "name": f"SplatLab scene {job_id}",
                    "job_id": job_id,
                    "heading_deg": geo.get("heading_deg"),
                    "anchor_scene": geo.get("anchor_scene"),
                    "meters_per_unit": meta.get("meters_per_unit"),
                    "source": geo.get("source"),
                    "set_at": geo.get("set_at"),
                },
            }
        ],
    }
    return json.dumps(doc, indent=2)


def _kml_doc(job_id: str, meta: dict[str, Any]) -> str:
    geo = meta["geo"]
    alt = geo.get("alt_m") or 0
    heading = geo.get("heading_deg", 0)
    mpu = meta.get("meters_per_unit")
    desc = f"SplatLab geo anchor — heading {heading}° true, " + (
        f"scale {mpu} m/unit" if mpu else "scale uncalibrated"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <name>SplatLab scene {job_id}</name>
    <description>{desc}</description>
    <Point>
      <altitudeMode>relativeToGround</altitudeMode>
      <coordinates>{geo["lon"]},{geo["lat"]},{alt}</coordinates>
    </Point>
  </Placemark>
</kml>
"""


@router.get("/jobs/{job_id}/geo/export")
async def export_splat_geo(job_id: str, fmt: str = "geojson"):
    meta = _require_job_meta(job_id)
    if not meta.get("geo"):
        raise HTTPException(status_code=404, detail="scene has no geo anchor yet")
    if fmt == "geojson":
        body, media, ext = _geojson_doc(job_id, meta), "application/geo+json", "geojson"
    elif fmt == "kml":
        body, media, ext = _kml_doc(job_id, meta), "application/vnd.google-earth.kml+xml", "kml"
    else:
        raise HTTPException(status_code=400, detail="fmt must be geojson or kml")
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{job_id}-geo.{ext}"'},
    )
