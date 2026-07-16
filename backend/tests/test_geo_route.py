"""Locate-in-the-world lane: POST /jobs/{id}/geo + footprint/suggest/export.

The geo anchor is the survey-lane sibling of meters_per_unit: it must
round-trip through meta.json, reach the status payload via the **meta spread,
reject garbage coordinates, and clear with {"geo": null}. The footprint
endpoints must render a transparent top-down projection whose bounds map
edge-to-edge, and suggest must stay fail-soft (empty list, never 500).
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geo_footprint  # noqa: E402
import geo_route  # noqa: E402
import splat_route  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(geo_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_9e0aabc", **meta_extra) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"job_id": job_id, "output_dir": str(job_dir), "status": "completed", **meta_extra})
    )
    return job_dir


GEO = {"lat": 34.0522, "lon": -118.2437, "alt_m": 89.0, "heading_deg": 132.5, "anchor_scene": [1.5, -0.25]}


# ── anchor set / clear / validation ──────────────────────────────────────────
def test_geo_set_persists_and_flows_to_payload(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": GEO})
    assert r.status_code == 200
    stored = r.json()["geo"]
    assert stored["lat"] == 34.0522 and stored["lon"] == -118.2437
    assert stored["heading_deg"] == 132.5 and stored["anchor_scene"] == [1.5, -0.25]
    assert stored["v"] == 1 and stored["set_at"]
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["geo"]["lat"] == 34.0522
    # reaches the status payload via the **meta spread — zero extra plumbing
    assert splat_route._job_payload(meta)["geo"]["lon"] == -118.2437


def test_geo_null_clears(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": GEO})
    r = http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": None})
    assert r.status_code == 200 and r.json()["geo"] is None
    assert json.loads((job_dir / "meta.json").read_text())["geo"] is None


def test_geo_heading_normalized(client):
    http, outputs = client
    _mk_job(outputs)
    for sent, want in ((450.0, 90.0), (-90.0, 270.0), (360.0, 0.0)):
        r = http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": {**GEO, "heading_deg": sent}})
        assert r.status_code == 200 and r.json()["geo"]["heading_deg"] == want


@pytest.mark.parametrize(
    "bad",
    [
        {**GEO, "lat": 91.0},
        {**GEO, "lon": -180.5},
        {**GEO, "alt_m": 25000.0},
        {**GEO, "anchor_scene": [1.0]},
        {**GEO, "anchor_scene": [1.0, 2.0, 3.0]},
        {**GEO, "source": "vibes"},
    ],
)
def test_geo_rejects_garbage(client, bad):
    http, outputs = client
    _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": bad})
    assert r.status_code in (400, 422)


def test_geo_rejects_nan_lat(client):
    http, outputs = client
    _mk_job(outputs)
    # json.loads happily parses a bare NaN literal — the endpoint must not.
    r = http.post(
        "/api/splat/jobs/splat_9e0aabc/geo",
        content=b'{"geo": {"lat": NaN, "lon": 10.0}}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code in (400, 422)


def test_geo_missing_and_unsafe_job(client):
    http, _ = client
    assert http.post("/api/splat/jobs/splat_nope000/geo", json={"geo": GEO}).status_code == 404
    assert http.post("/api/splat/jobs/..%2Fetc/geo", json={"geo": GEO}).status_code == 404


# ── pure parsers ─────────────────────────────────────────────────────────────
def test_parse_iso6709():
    assert geo_route._parse_iso6709("+34.0522-118.2437+089.0/") == (34.0522, -118.2437, 89.0)
    assert geo_route._parse_iso6709("+48.8577+002.295/") == (48.8577, 2.295, None)
    assert geo_route._parse_iso6709("not a location") is None
    assert geo_route._parse_iso6709("+99.0+002.0/") is None  # lat out of range


def test_gps_from_exif():
    gps = {1: "N", 2: (34.0, 3.0, 7.2), 3: "W", 4: (118.0, 14.0, 30.0), 5: 0, 6: 89.5}
    fix = geo_route._gps_from_exif(gps)
    assert fix is not None
    lat, lon, alt = fix
    assert math.isclose(lat, 34.052, abs_tol=1e-3)
    assert math.isclose(lon, -118.2417, abs_tol=1e-3)
    assert alt == 89.5
    # below-sea-level altitude ref flips the sign
    below = geo_route._gps_from_exif({**gps, 5: 1})
    assert below is not None and below[2] == -89.5
    # the (0, 0) never-locked placeholder is rejected
    assert geo_route._gps_from_exif({1: "N", 2: (0.0, 0.0, 0.0), 3: "E", 4: (0.0, 0.0, 0.0)}) is None
    assert geo_route._gps_from_exif({}) is None


# ── footprint ────────────────────────────────────────────────────────────────
_PLY_PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]


def _write_ply(path: Path, points: list[tuple[float, ...]]) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        + "".join(f"property float {name}\n" for name in _PLY_PROPS)
        + "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        for p in points:
            f.write(struct.pack("<7f", *p))


def _grid_points() -> list[tuple[float, ...]]:
    # A dense 4x2-unit slab centered on (10, 20): footprint should be ~2:1 landscape.
    pts = []
    for i in range(80):
        for j in range(40):
            x = 8.0 + 4.0 * i / 79
            y = 19.0 + 2.0 * j / 39
            pts.append((x, y, 0.5, 1.0, 0.5, 0.0, 5.0))
    return pts


def test_footprint_renders_with_bounds(tmp_path: Path):
    preview = tmp_path / "_preview"
    preview.mkdir()
    _write_ply(preview / "web.ply", _grid_points())
    made = geo_footprint.get_or_make(preview)
    assert made is not None
    image, meta = made
    assert image.is_file() and (preview / "footprint.json").is_file()
    assert meta["x0"] < meta["x1"] and meta["y0"] < meta["y1"]
    assert 8.0 <= meta["x0"] <= 8.3 and 20.7 <= meta["y1"] <= 21.0
    # landscape slab -> landscape image, bounds map edge-to-edge
    assert meta["width"] > meta["height"]
    assert math.isclose(meta["units_per_px"], (meta["x1"] - meta["x0"]) / meta["width"], rel_tol=1e-6)
    # transparent background outside the cloud
    from PIL import Image

    with Image.open(image) as img:
        assert img.mode == "RGBA"
    # second call returns the cache, not a re-render
    again = geo_footprint.get_or_make(preview)
    assert again is not None and again[0] == image


def test_footprint_endpoints(client):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=0.5)
    preview = job_dir / "_preview"
    preview.mkdir()
    _write_ply(preview / "web.ply", _grid_points())
    r = http.get("/api/splat/jobs/splat_9e0aabc/geo/footprint")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["meters_per_unit"] == 0.5
    assert body["url"].endswith("/geo/footprint.webp")
    img = http.get("/api/splat/jobs/splat_9e0aabc/geo/footprint.webp")
    assert img.status_code == 200 and img.headers["content-type"] == "image/webp"
    assert img.content[:4] == b"RIFF"


def test_footprint_unavailable_is_soft(client):
    http, outputs = client
    _mk_job(outputs)  # no _preview ply at all
    r = http.get("/api/splat/jobs/splat_9e0aabc/geo/footprint")
    assert r.status_code == 200 and r.json()["available"] is False
    assert http.get("/api/splat/jobs/splat_9e0aabc/geo/footprint.webp").status_code == 404


# ── suggest ──────────────────────────────────────────────────────────────────
def test_suggest_empty_without_input(client):
    http, outputs = client
    _mk_job(outputs)  # no input_path key
    r = http.get("/api/splat/jobs/splat_9e0aabc/geo/suggest")
    assert r.status_code == 200 and r.json()["candidates"] == []


def test_suggest_photos_without_gps_is_empty(client, tmp_path: Path):
    from PIL import Image

    http, outputs = client
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (8, 8)).save(photos / "a.jpg")
    _mk_job(outputs, input_path=str(photos))
    r = http.get("/api/splat/jobs/splat_9e0aabc/geo/suggest")
    assert r.status_code == 200 and r.json()["candidates"] == []


def test_suggest_photos_with_gps_exif(client, tmp_path: Path):
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    http, outputs = client
    photos = tmp_path / "photos"
    photos.mkdir()
    img = Image.new("RGB", (8, 8))
    exif = Image.Exif()
    exif[0x8825] = {
        1: "N",
        2: (IFDRational(34, 1), IFDRational(3, 1), IFDRational(36, 5)),
        3: "W",
        4: (IFDRational(118, 1), IFDRational(14, 1), IFDRational(30, 1)),
        5: 0,
        6: IFDRational(179, 2),
    }
    img.save(photos / "a.jpg", exif=exif)
    # Roundtrip guard: if this Pillow can't write a GPS IFD, skip rather than lie.
    with Image.open(photos / "a.jpg") as check:
        if not check.getexif().get_ifd(0x8825):
            pytest.skip("Pillow here does not roundtrip GPS EXIF")
    _mk_job(outputs, input_path=str(photos))
    r = http.get("/api/splat/jobs/splat_9e0aabc/geo/suggest")
    assert r.status_code == 200
    cands = r.json()["candidates"]
    assert len(cands) == 1
    assert math.isclose(cands[0]["lat"], 34.052, abs_tol=1e-3)
    assert math.isclose(cands[0]["lon"], -118.2417, abs_tol=1e-3)
    assert cands[0]["alt_m"] == 89.5 and cands[0]["source"] == "exif"


# ── export ───────────────────────────────────────────────────────────────────
def test_export_requires_anchor_and_valid_fmt(client):
    http, outputs = client
    _mk_job(outputs)
    assert http.get("/api/splat/jobs/splat_9e0aabc/geo/export").status_code == 404
    http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": GEO})
    assert http.get("/api/splat/jobs/splat_9e0aabc/geo/export?fmt=shapefile").status_code == 400


def test_export_geojson_and_kml(client):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=0.5)
    http.post("/api/splat/jobs/splat_9e0aabc/geo", json={"geo": GEO})
    gj = http.get("/api/splat/jobs/splat_9e0aabc/geo/export?fmt=geojson")
    assert gj.status_code == 200
    doc = json.loads(gj.content)
    feat = doc["features"][0]
    assert feat["geometry"]["coordinates"] == [-118.2437, 34.0522, 89.0]  # lon, lat, alt
    assert feat["properties"]["heading_deg"] == 132.5
    assert feat["properties"]["meters_per_unit"] == 0.5
    kml = http.get("/api/splat/jobs/splat_9e0aabc/geo/export?fmt=kml")
    assert kml.status_code == 200
    assert "-118.2437,34.0522,89.0" in kml.text
    assert 'filename="splat_9e0aabc-geo.kml"' in kml.headers["content-disposition"]
