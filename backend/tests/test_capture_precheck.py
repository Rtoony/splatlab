"""Capture Coach Phase 2 Tier-0 precheck: pure-Pillow analyzers must separate
sharp/blurred, exposed/clipped, moving/static fixtures; the driver handles
photo dirs and zips without ffmpeg; everything is advisory-only and
never raises. CPU-only."""

from __future__ import annotations

import random
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from health.precheck import (  # noqa: E402
    advisories_from_metrics,
    analyze_frames,
    precheck_input,
)


def _textured(seed: int = 0, size: int = 320) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("L", (size, size))
    img.putdata([rng.randrange(256) for _ in range(size * size)])
    return img.convert("RGB")


def test_sharp_vs_blurred_separates():
    sharp = [_textured(i) for i in range(4)]
    blurred = [img.filter(ImageFilter.GaussianBlur(8)) for img in sharp]
    sharp_metrics = analyze_frames(sharp)
    blurred_metrics = analyze_frames(blurred)
    assert sharp_metrics["median_edge_energy"] > blurred_metrics["median_edge_energy"]
    assert not any("blurred" in a for a in advisories_from_metrics(sharp_metrics))
    assert any("blurred" in a for a in advisories_from_metrics(blurred_metrics))


def test_exposure_clipping_flagged():
    dark = [Image.new("RGB", (64, 64), (0, 0, 0)) for _ in range(3)]
    bright = [Image.new("RGB", (64, 64), (255, 255, 255)) for _ in range(3)]
    dark_advice = advisories_from_metrics(analyze_frames(dark))
    bright_advice = advisories_from_metrics(analyze_frames(bright))
    assert any("underexposed" in a.lower() for a in dark_advice)
    assert any("blown-out" in a.lower() for a in bright_advice)


def test_static_sequence_flagged():
    frame = _textured(1)
    static_metrics = analyze_frames([frame] * 5)
    moving_metrics = analyze_frames([_textured(i) for i in range(5)])
    assert static_metrics["static_pair_ratio"] == 1.0
    assert any("holds still" in a for a in advisories_from_metrics(static_metrics))
    assert not any("holds still" in a for a in advisories_from_metrics(moving_metrics))


def test_density_cap_advisory_for_long_insv():
    metrics = analyze_frames([_textured(0)])
    advice = advisories_from_metrics(metrics, duration_s=300.0, images_per_equirect=8)
    assert any("frame budget caps" in a for a in advice)
    short = advisories_from_metrics(metrics, duration_s=30.0, images_per_equirect=8)
    assert not any("frame budget caps" in a for a in short)


def test_photo_dir_driver(tmp_path):
    folder = tmp_path / "orbit"
    folder.mkdir()
    for i in range(6):
        _textured(i, size=128).save(folder / f"img_{i:03d}.jpg", quality=90)
    result = precheck_input(folder)
    assert result["capture_type"] == "photo-folder"
    assert result["metrics"]["n_frames"] == 6
    assert isinstance(result["advisories"], list)


def test_photo_zip_driver(tmp_path):
    archive_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for i in range(4):
            frame_path = tmp_path / f"z_{i}.jpg"
            _textured(i, size=128).save(frame_path, quality=90)
            archive.write(frame_path, f"photos/z_{i}.jpg")
    result = precheck_input(archive_path)
    assert result["capture_type"] == "photo-zip"
    assert result["metrics"]["n_frames"] == 4


def test_driver_never_raises_on_garbage(tmp_path):
    bogus = tmp_path / "clip.mp4"
    bogus.write_bytes(b"not a video")
    result = precheck_input(bogus, ffmpeg=None)
    assert result["advisories"] == []
    assert "note" in result


def test_precheck_route_available_in_safe_browse_mode(tmp_path, monkeypatch):
    """The endpoint is deliberately NOT behind require_compute_enabled —
    advice must work while the hardware gate blocks GPU work."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import splat_route  # noqa: PLC0415

    folder = tmp_path / "orbit"
    folder.mkdir()
    _textured(0, size=128).save(folder / "img_000.jpg", quality=90)
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", "gate active")
    monkeypatch.setattr(splat_route, "_resolve_input_path", lambda raw: folder)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    client = TestClient(app)

    response = client.post("/api/splat/precheck", json={"input_path": str(folder)})
    assert response.status_code == 200
    assert response.json()["capture_type"] == "photo-folder"

    missing = tmp_path / "missing"
    monkeypatch.setattr(splat_route, "_resolve_input_path", lambda raw: missing)
    assert client.post(
        "/api/splat/precheck", json={"input_path": str(missing)}
    ).status_code == 404
