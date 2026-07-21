"""Survey export (Digital Twin kernel P2): the pure scene→grid math must match
the documented anchor contract, and the POST route must refuse loudly — never
emit a silently unscaled/unplaced file — when mesh, scale, or anchor is missing.

CPU-only: the exporter subprocess (open3d/pyproj/ezdxf in the probe env) is
monkeypatched; its real behavior is proven by the executable first-light run.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import geo_route  # noqa: E402
import geo_transform  # noqa: E402
import splat_route  # noqa: E402

GEO = {"lat": 38.44, "lon": -122.71, "alt_m": 30.0, "heading_deg": 0.0, "anchor_scene": [0.0, 0.0]}
SF = 1200.0 / 3937.0  # meters per US survey foot


# ── scene→ENU (the documented geo_route contract) ────────────────────────────
def test_scene_to_enu_heading_zero_identity():
    xyz = np.array([[1.0, 2.0, 3.0]])
    enu = geo_transform.scene_to_enu(xyz, 2.0, 0.0, (0.0, 0.0))
    assert np.allclose(enu, [[2.0, 4.0, 6.0]])


def test_scene_to_enu_heading_90_maps_plus_y_to_east():
    # heading 90 = the scene's +Y axis points compass east.
    xyz = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    enu = geo_transform.scene_to_enu(xyz, 1.0, 90.0, (0.0, 0.0))
    assert np.allclose(enu[0], [1.0, 0.0, 0.0], atol=1e-12)   # +Y -> east
    assert np.allclose(enu[1], [0.0, -1.0, 0.0], atol=1e-12)  # +X -> south


def test_scene_to_enu_anchor_offset():
    xyz = np.array([[5.0, 5.0, 0.0]])
    enu = geo_transform.scene_to_enu(xyz, 1.0, 0.0, (5.0, 5.0))
    assert np.allclose(enu, [[0.0, 0.0, 0.0]])


# ── ENU→grid (probe-derived calibration semantics) ───────────────────────────
def test_enu_to_grid_units_and_offset():
    enu = np.array([[SF, 2 * SF, 3 * SF]])  # meters
    grid = geo_transform.enu_to_grid(enu, 1000.0, 2000.0, SF, 0.0, 1.0, 100.0)
    assert np.allclose(grid, [[1001.0, 2002.0, 103.0]])  # exact survey-foot counts


def test_enu_to_grid_rotation_moves_true_north_to_grid_azimuth():
    # grid_rot_deg = measured grid azimuth of true north. A pure-north ENU
    # vector must land at exactly that azimuth in the grid frame.
    rot = 1.5
    enu = np.array([[0.0, 100.0, 0.0]])
    grid = geo_transform.enu_to_grid(enu, 0.0, 0.0, 1.0, rot, 1.0, 0.0)
    az = math.degrees(math.atan2(grid[0, 0], grid[0, 1]))
    assert abs(az - rot) < 1e-9
    assert abs(np.hypot(grid[0, 0], grid[0, 1]) - 100.0) < 1e-9


def test_enu_to_grid_scale_factor_scales_horizontal_only():
    enu = np.array([[3.0, 4.0, 7.0]])
    grid = geo_transform.enu_to_grid(enu, 0.0, 0.0, 1.0, 0.0, 1.0001, 0.0)
    assert abs(np.hypot(grid[0, 0], grid[0, 1]) - 5.0005) < 1e-9
    assert grid[0, 2] == 7.0  # elevations never get grid scale


def test_xy_convex_hull_square():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.8]], dtype=float)
    hull = geo_transform.xy_convex_hull(pts)
    assert len(hull) == 4
    assert {tuple(p) for p in hull} == {(0, 0), (1, 0), (1, 1), (0, 1)}


# ── POST /geo/export guards ──────────────────────────────────────────────────
@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(geo_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_9e0001", **meta_extra) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"job_id": job_id, "output_dir": str(job_dir), "status": "completed", **meta_extra})
    )
    return job_dir


def _add_mesh(job_dir: Path) -> None:
    mdir = job_dir / splat_route.MESH_DIRNAME
    mdir.mkdir(parents=True)
    (mdir / "mesh.ply").write_bytes(b"ply")


def test_export_requires_mesh_first(client):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=0.5, geo=GEO)
    r = http.post("/api/splat/jobs/splat_9e0001/geo/export", json={})
    assert r.status_code == 409
    assert "mesh export first" in r.json()["detail"]


def test_export_requires_scale(client):
    http, outputs = client
    job_dir = _mk_job(outputs, geo=GEO)
    _add_mesh(job_dir)
    r = http.post("/api/splat/jobs/splat_9e0001/geo/export", json={})
    assert r.status_code == 409
    assert "uncalibrated" in r.json()["detail"]


def test_export_requires_anchor(client):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=0.5)
    _add_mesh(job_dir)
    r = http.post("/api/splat/jobs/splat_9e0001/geo/export", json={})
    assert r.status_code == 409
    assert "anchor" in r.json()["detail"]


def test_export_rejects_bad_epsg(client):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=0.5, geo=GEO)
    _add_mesh(job_dir)
    r = http.post("/api/splat/jobs/splat_9e0001/geo/export", json={"epsg": 3})
    assert r.status_code == 400


def test_export_success_patches_meta_and_serves_files(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=0.5, geo=GEO)
    _add_mesh(job_dir)
    fake_report = {"v": 1, "epsg": 2226, "tris_exported": 42, "audits": {}}

    async def fake_subprocess(command):
        export_dir = Path(command[3])
        (export_dir / "geo_export.json").write_text(json.dumps(fake_report))
        (export_dir / "site.dxf").write_text("dxf")
        (export_dir / "surface.xml").write_text("<LandXML/>")
        (export_dir / "site.geojson").write_text("{}")
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_subprocess)

    r = http.post("/api/splat/jobs/splat_9e0001/geo/export", json={"epsg": 2226})
    assert r.status_code == 200
    assert r.json()["geo_export"]["tris_exported"] == 42

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["geo_export"]["epsg"] == 2226

    for fmt in ("dxf", "landxml", "site"):
        got = http.get(f"/api/splat/jobs/splat_9e0001/geo/export?fmt={fmt}")
        assert got.status_code == 200, fmt

    payload = splat_route._job_payload(meta)
    assert payload["survey_dxf_url"] == "/api/splat/jobs/splat_9e0001/geo/export?fmt=dxf"
    assert payload["survey_landxml_url"] == "/api/splat/jobs/splat_9e0001/geo/export?fmt=landxml"


def test_export_subprocess_failure_is_a_500_not_a_silent_success(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=0.5, geo=GEO)
    _add_mesh(job_dir)

    async def fake_subprocess(command):
        return 1, b"", b"FATAL: known-distance audit failed"

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_subprocess)
    r = http.post("/api/splat/jobs/splat_9e0001/geo/export", json={})
    assert r.status_code == 500
    assert "known-distance" in r.json()["detail"]
    meta = json.loads((job_dir / "meta.json").read_text())
    assert "geo_export" not in meta


def test_survey_file_404_before_build(client):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=0.5, geo=GEO)
    r = http.get("/api/splat/jobs/splat_9e0001/geo/export?fmt=dxf")
    assert r.status_code == 404
    assert "not built yet" in r.json()["detail"]


def test_grid_to_scene_round_trips_the_forward_transform():
    """grid_to_scene must exactly invert scene_to_enu + enu_to_grid."""
    pts = np.array([[1.0, 2.0, 0.5], [-3.0, 1.5, -0.2], [0.0, 0.0, 0.0]])
    kw = dict(e0=6357000.0, n0=1923000.0, unit_factor_m=SF,
              grid_rot_deg=0.45, scale_factor=0.999977, elev0_units=98.4)
    fwd = geo_transform.enu_to_grid(
        geo_transform.scene_to_enu(pts, 2.3537, 35.0, (0.0, 0.0)), **kw)
    back = geo_transform.grid_to_scene(fwd, meters_per_unit=2.3537, heading_deg=35.0,
                                       anchor_scene=(0.0, 0.0), **kw)
    assert np.abs(back - pts).max() < 1e-9
