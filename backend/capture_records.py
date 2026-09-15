"""Source-preserving video inspection and timestamped geographic observations."""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import artifact_manifest as manifests

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "spatial"
CAPTURE_SCHEMA = "dev.splatlab.capture/v1"
ID_PATTERN = re.compile(r"capture_[a-f0-9]{24}\Z")


def worker_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items()
            if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}}


def run_json(command: list[str], timeout: int = 120) -> Any:
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout, env=worker_env())
    if result.returncode:
        raise ValueError(result.stderr[-1200:] or "Metadata inspection failed")
    return json.loads(result.stdout)


def utc_seconds(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", value)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except ValueError:
        return None


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def gps_records(metadata: dict) -> dict:
    grouped: dict[str, dict] = {}
    for name, value in metadata.items():
        group, separator, tag = name.rpartition(":")
        if separator and tag.startswith("GPS"):
            grouped.setdefault(group, {})[tag] = value
    samples = []
    rejected = 0
    for group, record in grouped.items():
        latitude = finite(record.get("GPSLatitude"))
        longitude = finite(record.get("GPSLongitude"))
        timestamp = utc_seconds(record.get("GPSDateTime"))
        if (latitude is None or longitude is None or timestamp is None
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180
                or (latitude == 0 and longitude == 0)):
            rejected += 1
            continue
        samples.append({"utc_s": timestamp, "latitude": latitude,
                        "longitude": longitude, "altitude_m": finite(record.get("GPSAltitude")),
                        "horizontal_accuracy_m": finite(record.get("GPSHPositioningError")),
                        "source_record": group})
    samples.sort(key=lambda sample: sample["utc_s"])
    unique = {sample["utc_s"]: sample for sample in samples}
    values = list(unique.values())
    gaps = [{"after_utc_s": first["utc_s"], "duration_s": second["utc_s"] - first["utc_s"]}
            for first, second in zip(values, values[1:])
            if second["utc_s"] - first["utc_s"] > 2]
    return {"samples": values, "count": len(values), "rejected_records": rejected,
            "duplicate_timestamps": len(samples) - len(values), "gaps": gaps,
            "accuracy_known": bool(values) and all(sample["horizontal_accuracy_m"] is not None
                                                       for sample in values)}


def capture_path(capture_id: str) -> Path:
    if not ID_PATTERN.fullmatch(capture_id):
        raise ValueError("Invalid capture ID")
    return DATA_ROOT / "captures" / capture_id / "capture-manifest.json"


def load_capture(capture_id: str) -> dict:
    document = manifests.read_json(capture_path(capture_id))
    if not document or document.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("Capture not found")
    return document


def inspect_capture(source: Path) -> dict:
    source = source.resolve(strict=True)
    if source.suffix.lower() not in {".insv", ".mp4", ".mov", ".mkv"} or not source.is_file():
        raise ValueError("Select an INSV, MP4, MOV or MKV video")
    ffprobe = shutil.which("ffprobe")
    exiftool = shutil.which("exiftool")
    if not ffprobe:
        raise ValueError("ffprobe is unavailable")
    before = manifests.file_identity(source, include_sha256=False)
    probe = run_json([ffprobe, "-v", "error", "-show_entries",
                      "format=duration,size:format_tags=creation_time:stream=index,codec_type,codec_name,width,height,r_frame_rate,duration",
                      "-of", "json", str(source)])
    metadata = {}
    warnings = []
    if exiftool:
        metadata = run_json([exiftool, "-api", "LargeFileSupport=1", "-ee", "-a", "-n", "-G3", "-j",
                             "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", "-GPSDateTime",
                             "-GPSHPositioningError", "-CreateDate", "-Model", "-Warning", str(source)])[0]
        warnings.extend(str(value) for name, value in metadata.items() if name.endswith(":Warning"))
    else:
        warnings.append("ExifTool is unavailable; embedded GPS has not been inspected")
    duration = finite(probe.get("format", {}).get("duration"))
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"]
    if not duration or duration <= 0 or not streams:
        raise ValueError("Video duration or video streams are missing")
    digest = manifests.sha256_file(source)
    if not manifests.same_file_identity(source, before):
        raise ValueError("Source changed during inspection; inspect it again")
    capture_id = "capture_" + digest[:24]
    existing = manifests.read_json(capture_path(capture_id))
    start_raw = probe.get("format", {}).get("tags", {}).get("creation_time")
    start_raw = start_raw or metadata.get("Main:CreateDate")
    clock = {"video_start_utc_s": utc_seconds(start_raw), "offset_s": 0.0,
             "source": "container_creation_time" if start_raw else "unknown",
             "verified": False, "uncertainty_s": None}
    if existing and existing.get("clock", {}).get("verified"):
        clock = existing["clock"]
    document = {"schema": CAPTURE_SCHEMA, "capture_id": capture_id,
                "created_at": existing.get("created_at") if existing else manifests.utc_now(),
                "source": {"path": str(source), "name": source.name, **before, "sha256": digest},
                "duration_s": duration, "streams": streams,
                "camera_model": next((value for name, value in metadata.items() if name.endswith(":Model")), None),
                "clock": clock, "gps": gps_records(metadata), "warnings": warnings,
                "provenance": "observed", "imu": {"status": "not_extracted"}}
    manifests.atomic_write_json(capture_path(capture_id), document)
    return document


def location_at(capture: dict, video_time_s: float, max_gap_s: float = 2.0) -> dict | None:
    clock = capture["clock"]
    start = clock.get("video_start_utc_s")
    samples = capture["gps"]["samples"]
    if start is None or not samples:
        return None
    timestamp = start + clock.get("offset_s", 0) + video_time_s
    position = bisect.bisect_left([sample["utc_s"] for sample in samples], timestamp)
    if position < len(samples) and abs(samples[position]["utc_s"] - timestamp) < 1e-6:
        result = dict(samples[position])
    else:
        if position == 0 or position == len(samples):
            return None
        first, second = samples[position - 1], samples[position]
        interval = second["utc_s"] - first["utc_s"]
        if interval <= 0 or interval > max_gap_s:
            return None
        fraction = (timestamp - first["utc_s"]) / interval
        longitude_delta = (second["longitude"] - first["longitude"] + 180) % 360 - 180
        result = {"utc_s": timestamp,
                  "latitude": first["latitude"] + fraction * (second["latitude"] - first["latitude"]),
                  "longitude": (first["longitude"] + fraction * longitude_delta + 180) % 360 - 180,
                  "altitude_m": None, "horizontal_accuracy_m": None}
        if first["altitude_m"] is not None and second["altitude_m"] is not None:
            result["altitude_m"] = first["altitude_m"] + fraction * (second["altitude_m"] - first["altitude_m"])
    return {**result, "alignment_verified": clock["verified"]}


def list_captures() -> list[dict]:
    records = []
    for path in sorted((DATA_ROOT / "captures").glob("capture_*/capture-manifest.json")):
        document = manifests.read_json(path)
        if document and document.get("schema") == CAPTURE_SCHEMA:
            records.append({key: value for key, value in document.items() if key != "gps"}
                           | {"gps_count": document["gps"]["count"]})
    return records
