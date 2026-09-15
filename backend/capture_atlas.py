"""Portable, source-linked exterior capture context without model registration."""

from __future__ import annotations

import base64
import html
import json
import math
import re
import shutil
import uuid
from pathlib import Path

from PIL import Image

import artifact_manifest as manifests
import capture_records as captures

SCHEMA = "dev.splatlab.capture-atlas/v1"
ASSETS = Path(__file__).resolve().parents[1] / "tools" / "capture-atlas-viewer"
LIMITATIONS = [
    "Review panoramas are provisional generic stitches, not calibrated reconstruction inputs.",
    "GPS is a coarse route prior, not camera orientation, survey control or parcel boundaries.",
    "Unverified video clocks produce tentative image locations; GPS gaps are not bridged.",
    "Local east/north metres describe a geographic display frame, not architectural model scale.",
    "Altitude is retained as telemetry only; its datum is unknown and no terrain is inferred.",
    "Sparse review views do not establish surface coverage, reconstruction quality or unseen geometry.",
    "This private bundle contains home-location data and photographs. Do not publish it automatically.",
]


def atlas_dir(atlas_id: str) -> Path:
    if not re.fullmatch(r"atlas_[a-f0-9]{16}", atlas_id):
        raise ValueError("Invalid capture atlas ID")
    return captures.DATA_ROOT / "capture-atlases" / atlas_id


def load_atlas(atlas_id: str) -> dict:
    document = manifests.read_json(atlas_dir(atlas_id) / "atlas.json")
    if not document or document.get("schema") != SCHEMA or document.get("atlas_id") != atlas_id:
        raise ValueError("Capture atlas not found")
    return document


def list_atlases() -> list[dict]:
    result = []
    for directory in sorted((captures.DATA_ROOT / "capture-atlases").glob("atlas_*")):
        try:
            document = load_atlas(directory.name)
        except ValueError:
            continue
        result.append({key: document[key] for key in ("atlas_id", "title", "created_at", "summary")})
    return result


def contained_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Expected a bundle-relative file")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Bundle path escapes its root")
    return candidate


def local_position(longitude: float, latitude: float, origin: dict) -> dict:
    def ecef(lon: float, lat: float) -> tuple[float, float, float]:
        longitude_rad, latitude_rad = math.radians(lon), math.radians(lat)
        eccentricity_squared = 6.69437999014e-3
        radius = 6378137 / math.sqrt(1 - eccentricity_squared * math.sin(latitude_rad) ** 2)
        return (radius * math.cos(latitude_rad) * math.cos(longitude_rad),
                radius * math.cos(latitude_rad) * math.sin(longitude_rad),
                radius * (1 - eccentricity_squared) * math.sin(latitude_rad))

    delta = [value - reference for value, reference in
             zip(ecef(longitude, latitude), ecef(origin["longitude"], origin["latitude"]))]
    longitude_rad, latitude_rad = math.radians(origin["longitude"]), math.radians(origin["latitude"])
    east = -math.sin(longitude_rad) * delta[0] + math.cos(longitude_rad) * delta[1]
    north = (-math.sin(latitude_rad) * math.cos(longitude_rad) * delta[0]
             - math.sin(latitude_rad) * math.sin(longitude_rad) * delta[1]
             + math.cos(latitude_rad) * delta[2])
    return {"east_m": round(east, 3), "north_m": round(north, 3)}


def valid_fix(sample: dict) -> bool:
    values = [captures.finite(sample.get(key)) for key in ("longitude", "latitude", "utc_s")]
    longitude, latitude, timestamp = values
    return (None not in values and -180 <= longitude <= 180 and -90 <= latitude <= 90
            and timestamp > 0 and (longitude != 0 or latitude != 0))


def track_segments(samples: list[dict], origin: dict, max_gap_s: float = 2,
                   max_speed_m_s: float = 25) -> tuple[list[list[dict]], list[dict]]:
    segments: list[list[dict]] = []
    breaks = []
    previous = None
    for sample in samples:
        if not valid_fix(sample):
            breaks.append({"reason": "invalid_fix", "utc_s": captures.finite(sample.get("utc_s"))})
            previous = None
            continue
        point = {**sample, **local_position(sample["longitude"], sample["latitude"], origin)}
        reason = None
        if previous is not None:
            elapsed = point["utc_s"] - previous["utc_s"]
            distance = math.hypot(point["east_m"] - previous["east_m"], point["north_m"] - previous["north_m"])
            if elapsed <= 0:
                reason = "non_increasing_timestamp"
            elif elapsed > max_gap_s:
                reason = "telemetry_gap"
            elif distance / elapsed > max_speed_m_s:
                reason = "position_jump"
            if reason:
                breaks.append({"reason": reason, "after_utc_s": previous["utc_s"],
                               "utc_s": point["utc_s"], "duration_s": round(elapsed, 6)})
        if previous is None or reason:
            segments.append([])
        segments[-1].append(point)
        previous = point
    return segments, breaks


def viewpoint_location(capture: dict, time_s: float, segments: list[list[dict]], origin: dict) -> dict | None:
    clock = capture["clock"]
    start = clock.get("video_start_utc_s")
    if start is None:
        return None
    timestamp = start + clock.get("offset_s", 0) + time_s
    if not any(segment[0]["utc_s"] <= timestamp <= segment[-1]["utc_s"] for segment in segments):
        return None
    location = captures.location_at(capture, time_s)
    if not location or not valid_fix(location):
        return None
    return {**location, **local_position(location["longitude"], location["latitude"], origin),
            "status": "clock_verified_gps_approximate" if clock.get("verified") else "tentative_clock_unverified"}


def clock_diagnostic(capture: dict, segments: list[list[dict]]) -> dict:
    start = capture["clock"].get("video_start_utc_s")
    if not segments:
        return {"status": "no_gps", "overlap_s": 0}
    if start is None:
        return {"status": "unknown_video_clock", "overlap_s": 0}
    aligned_start = start + capture["clock"].get("offset_s", 0)
    overlap = sum(max(0, min(aligned_start + capture["duration_s"], segment[-1]["utc_s"])
                      - max(aligned_start, segment[0]["utc_s"])) for segment in segments)
    return {"status": "no_time_overlap" if overlap == 0 else "overlap_not_alignment_proof",
            "overlap_s": round(overlap, 3),
            "first_fix_minus_container_start_s": round(segments[0][0]["utc_s"] - start, 3),
            "offset_applied_s": capture["clock"].get("offset_s", 0),
            "automatic_clock_correction": False}


def inspect_sources(handoff: Path, raw_root: Path) -> list[dict]:
    source_manifest = json.loads((handoff / "metadata/manifest.json").read_text())
    if source_manifest.get("schema") != "dev.roonytoony.condo-360-handoff/v1":
        raise ValueError("Unsupported capture handoff schema")
    results = []
    for record in source_manifest["raw_files"]:
        if not record["filename"].lower().endswith(".insv"):
            continue
        source = contained_file(raw_root, record["filename"])
        if not source.is_file() or source.stat().st_size != record["bytes"]:
            raise ValueError(f"Raw source missing or size changed: {source.name}")
        capture_id = "capture_" + record["sha256"][:24]
        try:
            capture = captures.load_capture(capture_id)
        except ValueError:
            capture = None
        if not capture or Path(capture["source"]["path"]) != source or not manifests.same_file_identity(source, capture["source"]):
            capture = captures.inspect_capture(source)
        if capture["source"]["sha256"] != record["sha256"]:
            raise ValueError(f"Raw source hash differs from handoff: {source.name}")
        results.append(capture)
    if not results:
        raise ValueError("No original INSV captures in this handoff")
    return results


def build_document(handoff: Path, documents: list[dict], title: str) -> tuple[dict, list[tuple[Path, str]]]:
    manifest_path = handoff / "metadata/manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("schema") != "dev.roonytoony.condo-360-handoff/v1":
        raise ValueError("Unsupported capture handoff schema")
    records = {record["filename"]: record for record in source_manifest["raw_files"]
               if record["filename"].lower().endswith(".insv")}
    if len(documents) != len(records) or {item["source"]["name"] for item in documents} != set(records):
        raise ValueError("Inspect every original INSV before building the atlas")
    for capture in documents:
        if capture["source"]["sha256"] != records[capture["source"]["name"]]["sha256"]:
            raise ValueError("Capture/handoff source hash mismatch")
    all_samples = [sample for capture in documents for sample in capture["gps"]["samples"] if valid_fix(sample)]
    if not all_samples:
        raise ValueError("No timestamped GPS fixes; inspect embedded telemetry first")
    origin = {key: all_samples[0][key] for key in ("longitude", "latitude")}
    document = {"schema": SCHEMA, "title": title, "created_at": manifests.utc_now(),
                "handoff_manifest_sha256": manifests.sha256_file(manifest_path),
                "provenance": "observed_capture_context", "registration": {"status": "unregistered", "model_transform": None},
                "coordinate_frame": {"origin_wgs84": origin, "axes": "east,north", "units": "metres",
                                     "altitude_used": False, "basis": "WGS84 surface projected to local tangent plane"},
                "track_policy": {"max_gap_s": 2, "max_speed_m_s": 25, "meaning": "display segmentation, not accuracy certification"},
                "limitations": list(LIMITATIONS), "clips": [], "views": [], "missing_derivatives": []}
    copies = []
    for capture in documents:
        segments, breaks = track_segments(capture["gps"]["samples"], origin)
        source = capture["source"]
        clip = {"capture_id": capture["capture_id"], "source_name": source["name"],
                "source_sha256": source["sha256"], "duration_s": capture["duration_s"],
                "clock": capture["clock"], "gps_count": capture["gps"]["count"],
                "accuracy_known": capture["gps"].get("accuracy_known", False),
                "streams": capture["streams"], "segments": segments, "breaks": breaks,
                "clock_diagnostic": clock_diagnostic(capture, segments),
                "gps_rejected_records": capture["gps"].get("rejected_records", 0),
                "gps_duplicate_timestamps": capture["gps"].get("duplicate_timestamps", 0),
                "warnings": capture.get("warnings", []), "view_ids": []}
        for derivative in source_manifest["derivatives"]:
            if derivative.get("source") != source["name"]:
                continue
            path = contained_file(handoff, derivative["path"])
            if not path.is_file():
                document["missing_derivatives"].append({"source": source["name"], "path": derivative["path"]})
                continue
            if derivative.get("kind") != "review_frame":
                continue
            timestamp = captures.finite(derivative.get("approximate_source_time_s"))
            if timestamp is None or timestamp < 0 or timestamp >= capture["duration_s"]:
                raise ValueError("Review frame time is outside its source recording")
            with Image.open(path) as picture:
                if picture.format != "JPEG" or picture.width != 2 * picture.height:
                    raise ValueError("Expected a 2:1 review JPEG")
                dimensions = [picture.width, picture.height]
                picture.verify()
            index = len(document["views"])
            view_id = f"view-{index:04d}"
            filename = view_id + ".jpg"
            document["views"].append({"view_id": view_id, "capture_id": capture["capture_id"],
                                      "file": filename, "sha256": manifests.sha256_file(path),
                                      "source_time_s": timestamp, "source_derivative": derivative["path"],
                                      "time_basis": "approximate_requested_seek", "dimensions": dimensions,
                                      "projection": derivative.get("projection", "unknown"),
                                      "quality_flags": derivative.get("quality_flags", []),
                                      "location": viewpoint_location(capture, timestamp, segments, origin)})
            clip["view_ids"].append(view_id)
            copies.append((path, filename))
        document["clips"].append(clip)
    document["summary"] = {"clips": len(documents), "review_views": len(document["views"]),
                           "gps_fixes": sum(clip["gps_count"] for clip in document["clips"]),
                           "missing_derivatives": len(document["missing_derivatives"]),
                           "located_views": sum(view["location"] is not None for view in document["views"]),
                           "duration_s": round(sum(clip["duration_s"] for clip in document["clips"]), 3),
                           "track_breaks": sum(len(clip["breaks"]) for clip in document["clips"])}
    return document, copies


def geojson(document: dict) -> dict:
    features = []
    for clip in document["clips"]:
        for index, segment in enumerate(clip["segments"]):
            coordinates = [[point["longitude"], point["latitude"]] for point in segment]
            features.append({"type": "Feature", "properties": {"capture_id": clip["capture_id"],
                             "source_name": clip["source_name"], "source_sha256": clip["source_sha256"],
                             "segment": index, "utc_seconds": [point["utc_s"] for point in segment],
                             "evidence": "coarse GPS track; no model registration"},
                             "geometry": {"type": "Point" if len(coordinates) == 1 else "LineString",
                                          "coordinates": coordinates[0] if len(coordinates) == 1 else coordinates}})
    return {"type": "FeatureCollection", "features": features}


def write_atlas(handoff: Path, documents: list[dict], title: str) -> dict:
    document, copies = build_document(handoff, documents, title)
    atlas_id = "atlas_" + uuid.uuid4().hex[:16]
    document["atlas_id"] = atlas_id
    destination = atlas_dir(atlas_id)
    staging = destination.with_name(".building-" + atlas_id)
    staging.mkdir(parents=True, exist_ok=False)
    for source, filename in copies:
        shutil.copyfile(source, staging / filename)
        view = next(item for item in document["views"] if item["file"] == filename)
        if manifests.sha256_file(staging / filename) != view["sha256"]:
            raise ValueError("Review source changed while copying")
    manifests.atomic_write_json(staging / "atlas.json", document)
    manifests.atomic_write_json(staging / "tracks.geojson", geojson(document))
    payload = json.dumps(document, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = (ASSETS / "index.html").read_text()
    page = template.replace("__TITLE__", html.escape(title)).replace("__STYLE__", (ASSETS / "style.css").read_text())
    page = page.replace("__MATH__", (ASSETS / "math.js").read_text()).replace("__VIEWER__", (ASSETS / "viewer.js").read_text())
    page = page.replace("__ATLAS_DATA__", payload)
    embedded_images = {filename: "data:image/jpeg;base64," + base64.b64encode((staging / filename).read_bytes()).decode("ascii")
                       for _, filename in copies}
    page = page.replace("__ATLAS_IMAGES__", json.dumps(embedded_images))
    (staging / "index.html").write_text(page)
    receipt = {"schema": "dev.splatlab.capture-atlas-receipt/v1", "atlas_id": atlas_id,
               "files": {path.name: manifests.sha256_file(path) for path in sorted(staging.iterdir())}}
    manifests.atomic_write_json(staging / "receipt.json", receipt)
    staging.rename(destination)
    return document


def atlas_file(atlas_id: str, filename: str) -> Path:
    directory = atlas_dir(atlas_id)
    receipt = manifests.read_json(directory / "receipt.json") or {}
    allowed = filename in {"index.html", "atlas.json", "tracks.geojson"} or re.fullmatch(r"view-\d{4}\.jpg", filename)
    if not allowed or receipt.get("atlas_id") != atlas_id or filename not in receipt.get("files", {}):
        raise ValueError("Capture atlas file not found")
    path = contained_file(directory, filename)
    if not path.is_file() or manifests.sha256_file(path) != receipt["files"][filename]:
        raise ValueError("Capture atlas file missing or changed; build a new snapshot")
    return path
