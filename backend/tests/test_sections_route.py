"""Site-sections route (POST /geo/sections): the lighter "just give me a
picture" tool, distinct from the full /geo/contours CAD/DXF pipeline. Only
needs ground-extract + surface_receipts.py — no CDT venv, no DXF authoring —
and unlike /geo/contours, a surface_receipts.py failure here is loud (it's
the entire deliverable, not a bonus on top of a DXF).

CPU-only: subprocesses are monkeypatched and dispatched by script name,
mirroring test_contours_route.py's exact fixture style.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import geo_route  # noqa: E402
import splat_route  # noqa: E402

GEO = {"lat": 38.44, "lon": -122.71, "alt_m": 30.0, "heading_deg": 0.0, "anchor_scene": [0.0, 0.0]}
GROUND_REPORT = {"v": 1, "ground_points": 2123, "coverage_m2": 132.7}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(geo_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_5ec701", with_mesh: bool = True, **meta_extra) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"job_id": job_id, "output_dir": str(job_dir), "status": "completed", **meta_extra})
    )
    if with_mesh:
        mdir = job_dir / splat_route.MESH_DIRNAME
        mdir.mkdir(parents=True)
        (mdir / "mesh.ply").write_bytes(b"ply")
    return job_dir


def _fake_subprocess(fail_step: str | None = None, log: list | None = None):
    async def run(command):
        if log is not None:
            log.append(command)
        script = command[1]
        if "semantic_ground" in script:
            if fail_step == "semantic":
                return 1, b"", b"FATAL: row mismatch"
            Path(command[4]).write_bytes(b"npz")
            return 0, b"", b""
        if "ground_extract" in script:
            if fail_step == "ground":
                return 1, b"", b"FATAL: only 3 ground cells after filtering"
            out = Path(command[3])
            (out / "ground_points.txt").write_text("1,100.0,200.0,50.0,SPLAT-GRND\n")
            (out / "ground.json").write_text(json.dumps(GROUND_REPORT))
            return 0, b"", b""
        if "surface_receipts" in script:
            if fail_step == "sections":
                return 1, b"", b"FATAL: fewer than 3 ground points"
            out = Path(command[3])
            (out / "sections.png").write_bytes(b"png")
            (out / "surface_iso.png").write_bytes(b"png")
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_sections_requires_mesh(client):
    http, outputs = client
    _mk_job(outputs, with_mesh=False, meters_per_unit=2.35, geo=GEO)
    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 409
    assert "mesh export first" in r.json()["detail"]


def test_sections_requires_scale(client):
    http, outputs = client
    _mk_job(outputs, geo=GEO)
    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 409 and "uncalibrated" in r.json()["detail"]


def test_sections_requires_anchor(client):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=2.35)
    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 409 and "geo anchor" in r.json()["detail"]


def test_sections_does_not_require_cdt_venv(client, monkeypatch):
    """The whole point of the narrower toolchain: this route must 200 even
    when CDT_VENV_PYTHON (contours-only) doesn't exist on this host."""
    http, outputs = client
    _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "CDT_VENV_PYTHON", Path("/nonexistent/cdt/python"))
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess())
    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 200


def test_sections_happy_path(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess())

    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["site_sections"]["ground"]["ground_points"] == 2123
    assert body["sections_url"] and body["surface_iso_url"] and body["ground_points_url"]

    geo_dir = job_dir / splat_route.MESH_DIRNAME / "geo"
    assert (geo_dir / "ground_points.txt").is_file()
    assert (geo_dir / "sections.png").is_file()
    assert (geo_dir / "surface_iso.png").is_file()
    assert (geo_dir / "site_sections.json").is_file()

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["site_sections"]["ground"]["ground_points"] == 2123

    for fmt in ("sections", "surface-iso", "ground"):
        assert http.get(f"/api/splat/jobs/splat_5ec701/geo/export?fmt={fmt}").status_code == 200, fmt


def test_sections_ground_failure_is_500(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(fail_step="ground"))
    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 500 and "Ground extraction failed" in r.json()["detail"]
    assert "site_sections" not in json.loads((job_dir / "meta.json").read_text())


def test_sections_surface_receipts_failure_is_loud_500(client, monkeypatch):
    """Unlike /geo/contours (where surface_receipts is a best-effort bonus),
    a failure here IS the failure — this route's entire deliverable."""
    http, outputs = client
    _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(fail_step="sections"))
    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 500
    assert "Site sections failed" in r.json()["detail"]


def test_sections_after_contours_preserves_contours_artifacts(client, monkeypatch):
    """Running /geo/sections after a prior /geo/contours build must not
    disturb contours' own exclusive files (contours.dxf etc) — sections only
    promotes the files it just produced."""
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    geo_dir = job_dir / splat_route.MESH_DIRNAME / "geo"
    geo_dir.mkdir(parents=True)
    (geo_dir / "contours.dxf").write_text("PRIOR CONTOURS DXF")
    (geo_dir / "contours_result.json").write_text(json.dumps({"contours_drawn": 194}))
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess())

    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 200
    assert (geo_dir / "contours.dxf").read_text() == "PRIOR CONTOURS DXF"
    assert json.loads((geo_dir / "contours_result.json").read_text())["contours_drawn"] == 194
    # and sections' own files landed alongside, untouched by the prior contours build
    assert (geo_dir / "sections.png").is_file()


def test_sections_semantic_auto(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, with_mesh=False, meters_per_unit=0.5, geo=GEO)
    lf = job_dir / splat_route.LANGFIELD_DIRNAME
    lf.mkdir(parents=True)
    (lf / "gauss_emb.npz").write_bytes(b"npz")
    cfg = job_dir / "processed" / "splatfacto" / "2026-07-01_000000"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(log=calls))

    r = http.post("/api/splat/jobs/splat_5ec701/geo/sections", json={})
    assert r.status_code == 200
    assert r.json()["site_sections"]["params"]["semantic"] is True
    assert Path(calls[0][1]).name == "semantic_ground.py"
