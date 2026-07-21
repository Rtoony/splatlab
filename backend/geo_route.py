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

# ── survey export (Digital Twin kernel P2) ───────────────────────────────────
# Places the job's _mesh/mesh.ply into a projected CRS (scale + anchor +
# heading composed with a probe-derived grid calibration) and writes site.dxf /
# surface.xml (LandXML TIN) / site.geojson into _mesh/geo/. CPU-only, a few
# seconds — same admission policy as the rest of this lane (no heavy-work
# gate); runs in the dn-splatter-probe env (open3d + pyproj + ezdxf).
GEO_EXPORT_SCRIPT = Path(__file__).resolve().parent / "mesh" / "geo_export.py"
GEO_EXPORT_SUBDIR = "geo"  # under MESH_DIRNAME
_GEO_EXPORT_LOCKS: dict[str, asyncio.Lock] = {}


def _geo_export_lock(job_id: str) -> asyncio.Lock:
    lock = _GEO_EXPORT_LOCKS.get(job_id)
    if lock is None:
        lock = _GEO_EXPORT_LOCKS[job_id] = asyncio.Lock()
    return lock


class GeoExportBody(BaseModel):
    epsg: int = 2226  # NAD83 / California zone 2, US survey foot — RToony's area
    max_faces: int = 50_000


# ── ground contours (Digital Twin kernel P1, productionized 2026-07-21) ──────
# Three fail-loud steps sharing _mesh/geo/: ground extraction (dn-splatter-probe
# env) → cdt survey_to_surface (cdt venv — real office contour layers) →
# best-effort receipt PNG. Proven on garden: 2,123 pts → 194 contours.
GROUND_EXTRACT_SCRIPT = Path(__file__).resolve().parent / "mesh" / "ground_extract.py"
CONTOURS_BUILD_SCRIPT = Path(__file__).resolve().parent / "mesh" / "contours_build.py"
CONTOURS_RECEIPT_SCRIPT = Path(__file__).resolve().parent / "mesh" / "contours_receipt.py"


class ContoursBody(BaseModel):
    epsg: int = 2226
    cell_m: float = 0.25          # ground-sampling grid (meters); 0.5-1.0 for big sites
    max_slope_deg: float = 40.0   # steeper faces are not "ground"
    spike_tol_m: float = 0.5      # 3x3 neighborhood median rejection
    minor_ft: float = 0.5
    major_ft: float = 2.5
    tin_faces: bool = False       # also draw the TIN as review linework


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


def _survey_export_dir(meta: dict[str, Any], job_id: str) -> Path:
    output_dir = (
        Path(meta["output_dir"]) if meta.get("output_dir") else splat_route.DEFAULT_3D_ROOT / job_id
    )
    return output_dir / splat_route.MESH_DIRNAME / GEO_EXPORT_SUBDIR


@router.post("/jobs/{job_id}/geo/export")
async def build_survey_export(job_id: str, body: GeoExportBody | None = None):
    """Build the survey export (site.dxf / LandXML surface / site.geojson) from
    the job's mesh + scale + anchor. Every missing prerequisite is a loud 409
    with the exact next step — never a silently unscaled or unplaced file."""
    body = body or GeoExportBody()
    meta = _require_job_meta(job_id)
    if not (1000 <= body.epsg <= 999_999):
        raise HTTPException(status_code=400, detail="epsg must be a plausible EPSG code")
    if not (100 <= body.max_faces <= 2_000_000):
        raise HTTPException(status_code=400, detail="max_faces must be within [100, 2000000]")
    output_dir = (
        Path(meta["output_dir"]) if meta.get("output_dir") else splat_route.DEFAULT_3D_ROOT / job_id
    )
    mesh_file = output_dir / splat_route.MESH_DIRNAME / "mesh.ply"
    if not mesh_file.is_file():
        raise HTTPException(
            status_code=409,
            detail=f"No mesh for this scene yet — run mesh export first (POST /api/splat/jobs/{job_id}/mesh).",
        )
    mpu = meta.get("meters_per_unit")
    if not mpu:
        raise HTTPException(
            status_code=409,
            detail="Scene scale is uncalibrated — measure a known distance in the viewer "
            f"and set it (POST /api/splat/jobs/{job_id}/scale) first.",
        )
    if not meta.get("geo"):
        raise HTTPException(
            status_code=409,
            detail="Scene has no geo anchor — set one with Locate (map pin or GPS suggest) first.",
        )
    if not GEO_EXPORT_SCRIPT.is_file() or not splat_route.MESH_ENV_PYTHON.is_file():
        raise HTTPException(status_code=400, detail="Survey-export toolchain unavailable.")

    export_dir = _survey_export_dir(meta, job_id)
    export_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "job_id": job_id,
        "meters_per_unit": mpu,
        "geo": meta["geo"],
        "epsg": body.epsg,
        "max_faces": body.max_faces,
    }
    command = [
        str(splat_route.MESH_ENV_PYTHON),
        str(GEO_EXPORT_SCRIPT),
        str(mesh_file),
        str(export_dir),
        "--params-json",
        json.dumps(params),
    ]
    lock = _geo_export_lock(job_id)
    if lock.locked():
        raise HTTPException(status_code=409, detail=f"A survey export is already running for {job_id}.")
    async with lock:
        return_code, _stdout, stderr = await splat_route._run_capture_subprocess(command)
        report_path = export_dir / "geo_export.json"
        if return_code != 0 or not report_path.is_file():
            tail = "\n".join(stderr.decode("utf-8", errors="replace").splitlines()[-8:])
            raise HTTPException(status_code=500, detail=f"Survey export failed (exit {return_code}): {tail}")
        try:
            report = json.loads(report_path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Survey export report unreadable: {exc}") from exc
        splat_route._patch_meta(job_id, geo_export=report)
    return {
        "job_id": job_id,
        "geo_export": report,
        "dxf_url": f"/api/splat/jobs/{job_id}/geo/export?fmt=dxf",
        "landxml_url": f"/api/splat/jobs/{job_id}/geo/export?fmt=landxml",
        "site_geojson_url": f"/api/splat/jobs/{job_id}/geo/export?fmt=site",
    }


@router.post("/jobs/{job_id}/geo/contours")
async def build_ground_contours(job_id: str, body: ContoursBody | None = None):
    """Splat mesh -> ground PNEZD points -> cdt TIN + contour DXF on office
    layers. Same prerequisites and loud-409 contract as /geo/export."""
    body = body or ContoursBody()
    meta = _require_job_meta(job_id)
    if not (1000 <= body.epsg <= 999_999):
        raise HTTPException(status_code=400, detail="epsg must be a plausible EPSG code")
    if not (0.05 <= body.cell_m <= 5.0):
        raise HTTPException(status_code=400, detail="cell_m must be within [0.05, 5.0] meters")
    if not (0.0 < body.minor_ft <= body.major_ft <= 100.0):
        raise HTTPException(status_code=400, detail="need 0 < minor_ft <= major_ft <= 100")
    output_dir = (
        Path(meta["output_dir"]) if meta.get("output_dir") else splat_route.DEFAULT_3D_ROOT / job_id
    )
    mesh_file = output_dir / splat_route.MESH_DIRNAME / "mesh.ply"
    if not mesh_file.is_file():
        raise HTTPException(
            status_code=409,
            detail=f"No mesh for this scene yet — run mesh export first (POST /api/splat/jobs/{job_id}/mesh).",
        )
    mpu = meta.get("meters_per_unit")
    if not mpu:
        raise HTTPException(
            status_code=409,
            detail="Scene scale is uncalibrated — measure a known distance in the viewer "
            f"and set it (POST /api/splat/jobs/{job_id}/scale) first.",
        )
    if not meta.get("geo"):
        raise HTTPException(
            status_code=409,
            detail="Scene has no geo anchor — set one with Locate (map pin or GPS suggest) first.",
        )
    toolchain = (
        GROUND_EXTRACT_SCRIPT.is_file()
        and CONTOURS_BUILD_SCRIPT.is_file()
        and splat_route.MESH_ENV_PYTHON.is_file()
        and splat_route.CDT_VENV_PYTHON.is_file()
    )
    if not toolchain:
        raise HTTPException(status_code=400, detail="Contours toolchain (mesh env + cdt venv) unavailable.")

    geo_dir = _survey_export_dir(meta, job_id)
    geo_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "job_id": job_id,
        "meters_per_unit": mpu,
        "geo": meta["geo"],
        "epsg": body.epsg,
        "cell_m": body.cell_m,
        "max_slope_deg": body.max_slope_deg,
        "spike_tol_m": body.spike_tol_m,
    }
    pnezd = geo_dir / "ground_points.txt"
    dxf = geo_dir / "contours.dxf"
    receipt = geo_dir / "contours_receipt.png"

    lock = _geo_export_lock(job_id)  # shared with /geo/export: both write _mesh/geo/
    if lock.locked():
        raise HTTPException(status_code=409, detail=f"A survey build is already running for {job_id}.")
    async with lock:
        rc, _out, stderr = await splat_route._run_capture_subprocess([
            str(splat_route.MESH_ENV_PYTHON), str(GROUND_EXTRACT_SCRIPT),
            str(mesh_file), str(geo_dir), "--params-json", json.dumps(params),
        ])
        if rc != 0 or not pnezd.is_file():
            tail = "\n".join(stderr.decode("utf-8", errors="replace").splitlines()[-8:])
            raise HTTPException(status_code=500, detail=f"Ground extraction failed (exit {rc}): {tail}")

        cdt_cmd = [
            str(splat_route.CDT_VENV_PYTHON), str(CONTOURS_BUILD_SCRIPT),
            str(pnezd), str(dxf),
            "--epsg", str(body.epsg), "--minor", str(body.minor_ft), "--major", str(body.major_ft),
        ]
        if body.tin_faces:
            cdt_cmd.append("--tin-faces")
        rc, _out, stderr = await splat_route._run_capture_subprocess(cdt_cmd)
        if rc != 0 or not dxf.is_file():
            tail = "\n".join(stderr.decode("utf-8", errors="replace").splitlines()[-8:])
            raise HTTPException(status_code=500, detail=f"Contour build failed (exit {rc}): {tail}")

        report: dict[str, Any] = {
            "v": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": {k: v for k, v in params.items() if k != "geo"} | {
                "minor_ft": body.minor_ft, "major_ft": body.major_ft, "tin_faces": body.tin_faces,
            },
        }
        for name, key in (("ground.json", "ground"), ("contours_result.json", "contours")):
            try:
                report[key] = json.loads((geo_dir / name).read_text())
            except Exception as exc:  # a written DXF with an unreadable report is still a failure
                raise HTTPException(status_code=500, detail=f"Contours report {name} unreadable: {exc}") from exc

        # Receipt is best-effort: a render failure becomes a note, never a 500.
        rc, _out, stderr = await splat_route._run_capture_subprocess([
            str(splat_route.MESH_ENV_PYTHON), str(CONTOURS_RECEIPT_SCRIPT), str(dxf), str(receipt),
            "--title", f"{job_id} ground contours ({body.minor_ft}/{body.major_ft} ft, EPSG:{body.epsg})",
        ])
        if rc == 0 and receipt.is_file():
            report["receipt"] = "contours_receipt.png"
        else:
            report["receipt_error"] = f"receipt render exit {rc}"
        (geo_dir / "contours.json").write_text(json.dumps(report, indent=2))
        splat_route._patch_meta(job_id, contours=report)

    return {
        "job_id": job_id,
        "contours": report,
        "contours_dxf_url": f"/api/splat/jobs/{job_id}/geo/export?fmt=contours",
        "ground_points_url": f"/api/splat/jobs/{job_id}/geo/export?fmt=ground",
        "receipt_url": (
            f"/api/splat/jobs/{job_id}/geo/export?fmt=contours-receipt"
            if report.get("receipt") else None
        ),
    }


_SURVEY_FILES = {
    "dxf": ("site.dxf", "application/dxf", "site.dxf"),
    "landxml": ("surface.xml", "application/xml", "surface.landxml.xml"),
    "site": ("site.geojson", "application/geo+json", "site.geojson"),
    "contours": ("contours.dxf", "application/dxf", "contours.dxf"),
    "ground": ("ground_points.txt", "text/plain", "ground_points.pnezd.txt"),
    "contours-receipt": ("contours_receipt.png", "image/png", "contours_receipt.png"),
}


@router.get("/jobs/{job_id}/geo/export")
async def export_splat_geo(job_id: str, fmt: str = "geojson"):
    meta = _require_job_meta(job_id)
    if fmt in _SURVEY_FILES:
        name, media, download = _SURVEY_FILES[fmt]
        path = _survey_export_dir(meta, job_id) / name
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Survey export not built yet — POST /geo/export with mesh + scale + anchor in place.",
            )
        return FileResponse(
            str(path), media_type=media, filename=f"{job_id}-{download}"
        )
    if not meta.get("geo"):
        raise HTTPException(status_code=404, detail="scene has no geo anchor yet")
    if fmt == "geojson":
        body, media, ext = _geojson_doc(job_id, meta), "application/geo+json", "geojson"
    elif fmt == "kml":
        body, media, ext = _kml_doc(job_id, meta), "application/vnd.google-earth.kml+xml", "kml"
    else:
        raise HTTPException(status_code=400, detail="fmt must be geojson, kml, dxf, landxml, or site")
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{job_id}-geo.{ext}"'},
    )
