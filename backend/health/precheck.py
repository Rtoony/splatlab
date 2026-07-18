"""Capture Coach Phase 2 — Tier-0 upload-time heuristics (ADVISORY-ONLY).

Cheap, CPU-only screening the moment an input lands, BEFORE any dispatch:
pure-Pillow blur / exposure / static-frame checks over ~8 sampled frames, plus
duration-vs-density advisories for video. No numpy, no cv2, no GPU — the
backend venv has Pillow only (thumb.py precedent).

Contract per the metric-trust doctrine: output is a list of human advisory
strings + raw metrics. It NEVER blocks the Create button and never gates a
job — the user always outranks the heuristic.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

SAMPLE_FRAMES = int(os.environ.get("HEALTH_PRECHECK_FRAMES", "8"))
ANALYZE_WIDTH = int(os.environ.get("HEALTH_PRECHECK_WIDTH", "640"))
BLUR_EDGE_MIN = float(os.environ.get("HEALTH_PRECHECK_BLUR_MIN", "4.0"))
CLIP_FRAC_MAX = float(os.environ.get("HEALTH_PRECHECK_CLIP_MAX", "0.25"))
STATIC_DIFF_MAX = float(os.environ.get("HEALTH_PRECHECK_STATIC_DIFF", "2.0"))
STATIC_RATIO_MAX = float(os.environ.get("HEALTH_PRECHECK_STATIC_RATIO", "0.5"))
LONG_CLIP_S = float(os.environ.get("HEALTH_PRECHECK_LONG_CLIP_S", "180"))
FFMPEG_TIMEOUT_S = int(os.environ.get("HEALTH_PRECHECK_FFMPEG_TIMEOUT_S", "20"))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".bmp"}


def _prep(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    if gray.width > ANALYZE_WIDTH:
        gray = gray.resize(
            (ANALYZE_WIDTH, max(1, round(gray.height * ANALYZE_WIDTH / gray.width)))
        )
    return gray


def _mean(image: Image.Image) -> float:
    histogram = image.histogram()
    total = sum(histogram) or 1
    return sum(value * count for value, count in enumerate(histogram)) / total


def analyze_frames(images: list[Image.Image]) -> dict[str, Any]:
    """Pure-Pillow metrics over sampled frames. Deterministic and testable."""
    grays = [_prep(img) for img in images]
    edge_scores: list[float] = []
    dark_fracs: list[float] = []
    bright_fracs: list[float] = []
    for gray in grays:
        edge_scores.append(_mean(gray.filter(ImageFilter.FIND_EDGES)))
        histogram = gray.histogram()
        total = sum(histogram) or 1
        dark_fracs.append(sum(histogram[:5]) / total)
        bright_fracs.append(sum(histogram[250:]) / total)
    static_pairs = 0
    for previous, current in zip(grays, grays[1:]):
        if current.size != previous.size:
            current = current.resize(previous.size)
        if _mean(ImageChops.difference(previous, current)) < STATIC_DIFF_MAX:
            static_pairs += 1

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    return {
        "n_frames": len(images),
        "median_edge_energy": round(median(edge_scores), 2),
        "median_dark_frac": round(median(dark_fracs), 3),
        "median_bright_frac": round(median(bright_fracs), 3),
        "static_pair_ratio": (
            round(static_pairs / (len(grays) - 1), 2) if len(grays) > 1 else 0.0
        ),
    }


def advisories_from_metrics(
    metrics: dict[str, Any],
    duration_s: float | None = None,
    images_per_equirect: int | None = None,
) -> list[str]:
    advisories: list[str] = []
    if metrics.get("n_frames", 0) == 0:
        return advisories
    if metrics["median_edge_energy"] < BLUR_EDGE_MIN:
        advisories.append(
            "Frames look soft or blurred — hold steadier, or shoot in better "
            "light so the camera can use a faster shutter."
        )
    if metrics["median_dark_frac"] > CLIP_FRAC_MAX:
        advisories.append(
            "Large underexposed areas — dark captures cost SfM features. "
            "Add light or shoot at a brighter time."
        )
    if metrics["median_bright_frac"] > CLIP_FRAC_MAX:
        advisories.append(
            "Large blown-out areas — overexposed surfaces carry no texture. "
            "Avoid shooting straight into bright light."
        )
    if metrics["static_pair_ratio"] > STATIC_RATIO_MAX:
        advisories.append(
            "Much of the sample holds still — parallax builds 3D. "
            "Keep moving through the space while recording."
        )
    if duration_s is not None:
        if images_per_equirect:
            cap = 4000 // images_per_equirect
            if duration_s * 3.0 > cap:
                advisories.append(
                    f"Clip is long ({duration_s:.0f}s): the frame budget caps at "
                    f"{cap} views, below the proven 3 fps density for the full "
                    "clip. Consider a Test Flight window first."
                )
        elif duration_s > LONG_CLIP_S:
            advisories.append(
                f"Long clip ({duration_s:.0f}s) — expect a long build; a "
                "Test Flight-style short window proves the capture faster."
            )
    return advisories


def _sample_video_frames(
    input_path: Path, ffmpeg: str, duration_s: float | None, workdir: Path
) -> list[Image.Image]:
    """Extract SAMPLE_FRAMES stills spread across the clip (scaled, cheap)."""
    images: list[Image.Image] = []
    span = duration_s if duration_s and duration_s > 0 else 10.0
    for index in range(SAMPLE_FRAMES):
        timestamp = span * (index + 0.5) / SAMPLE_FRAMES
        out = workdir / f"precheck_{index:02d}.jpg"
        cmd = [
            ffmpeg, "-y", "-ss", f"{timestamp:.2f}", "-i", str(input_path),
            "-frames:v", "1", "-vf", f"scale={ANALYZE_WIDTH}:-2", "-q:v", "5",
            str(out),
        ]
        try:
            subprocess.run(
                cmd, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if out.is_file() and out.stat().st_size > 0:
            try:
                with Image.open(out) as img:
                    images.append(img.convert("RGB"))
            except OSError:
                continue
    return images


def _sample_dir_images(input_path: Path) -> list[Image.Image]:
    files = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        return []
    step = max(1, len(files) // SAMPLE_FRAMES)
    images = []
    for path in files[::step][:SAMPLE_FRAMES]:
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except OSError:
            continue
    return images


def _sample_zip_images(input_path: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    try:
        with zipfile.ZipFile(input_path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if Path(name).suffix.lower() in IMAGE_SUFFIXES
                and not name.startswith("__MACOSX")
            )
            step = max(1, len(members) // SAMPLE_FRAMES) if members else 1
            for name in members[::step][:SAMPLE_FRAMES]:
                try:
                    with archive.open(name) as fh, Image.open(fh) as img:
                        images.append(img.convert("RGB"))
                except (OSError, zipfile.BadZipFile):
                    continue
    except (OSError, zipfile.BadZipFile):
        return []
    return images


def precheck_input(
    input_path: Path,
    ffmpeg: str | None = None,
    duration_s: float | None = None,
    images_per_equirect: int | None = None,
) -> dict[str, Any]:
    """Advisory-only Tier-0 screen of any splat input. Never raises."""
    try:
        if input_path.is_dir():
            capture_type = "photo-folder"
            images = _sample_dir_images(input_path)
        elif input_path.suffix.lower() == ".zip":
            capture_type = "photo-zip"
            images = _sample_zip_images(input_path)
        elif ffmpeg:
            capture_type = "video"
            with tempfile.TemporaryDirectory(prefix="splat-precheck-") as tmp:
                images = _sample_video_frames(
                    input_path, ffmpeg, duration_s, Path(tmp)
                )
        else:
            return {"v": 1, "capture_type": "unknown", "advisories": [],
                    "metrics": {}, "note": "no ffmpeg available for video sampling"}
        metrics = analyze_frames(images)
        return {
            "v": 1,
            "capture_type": capture_type,
            "metrics": metrics,
            "advisories": advisories_from_metrics(
                metrics, duration_s=duration_s,
                images_per_equirect=images_per_equirect,
            ),
        }
    except Exception as exc:  # noqa: BLE001 — advisory-only: never break upload
        return {"v": 1, "capture_type": "unknown", "advisories": [],
                "metrics": {}, "note": f"precheck failed: {exc}"}
