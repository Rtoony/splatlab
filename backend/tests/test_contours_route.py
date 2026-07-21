"""Ground-contours route (Digital Twin kernel P1): the three-step build must
refuse loudly on missing prerequisites, fail honestly when a step dies, treat
the receipt as best-effort only, and land its merged report in meta["contours"].

CPU-only: subprocesses are monkeypatched and dispatched by script name; the
real pipeline is proven by the solidify-probe first-light + standalone runs.
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
CDT_RESULT = {"points_imported": 2123, "contours_drawn": 194, "watermarked": True, "warnings": []}


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


def _mk_job(outputs: Path, job_id: str = "splat_c0a701", with_mesh: bool = True, **meta_extra) -> Path:
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


def _fake_subprocess(fail_step: str | None = None, receipt_ok: bool = True, log: list | None = None):
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
        if "contours_build" in script:
            if fail_step == "cdt":
                return 1, b"", b"RecipeError: contour set is EMPTY"
            dxf = Path(command[3])
            dxf.write_text("dxf")
            (dxf.parent / "contours_result.json").write_text(json.dumps(CDT_RESULT))
            return 0, b"", b""
        if "contours_receipt" in script:
            if not receipt_ok:
                return 1, b"", b"FATAL: no polylines to draw"
            Path(command[3]).write_bytes(b"png")
            return 0, b"", b""
        if "surface_receipts" in script:
            out = Path(command[3])
            (out / "sections.png").write_bytes(b"png")
            (out / "surface_iso.png").write_bytes(b"png")
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_contours_requires_mesh(client):
    http, outputs = client
    _mk_job(outputs, with_mesh=False, meters_per_unit=2.35, geo=GEO)
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 409
    assert "mesh export first" in r.json()["detail"]


def test_contours_requires_scale_and_anchor(client):
    http, outputs = client
    _mk_job(outputs, geo=GEO)
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 409 and "uncalibrated" in r.json()["detail"]


def test_contours_rejects_bad_intervals(client):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={"minor_ft": 5.0, "major_ft": 1.0})
    assert r.status_code == 400


def test_contours_success_merges_reports_and_patches_meta(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess())

    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={"minor_ft": 0.5, "major_ft": 2.5})
    assert r.status_code == 200
    body = r.json()
    assert body["contours"]["ground"]["ground_points"] == 2123
    assert body["contours"]["contours"]["contours_drawn"] == 194
    assert body["contours"]["receipt"] == "contours_receipt.png"
    assert body["receipt_url"]

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["contours"]["contours"]["contours_drawn"] == 194

    for fmt in ("contours", "ground", "contours-receipt"):
        assert http.get(f"/api/splat/jobs/splat_c0a701/geo/export?fmt={fmt}").status_code == 200, fmt

    payload = splat_route._job_payload(meta)
    assert payload["contours_dxf_url"] == "/api/splat/jobs/splat_c0a701/geo/export?fmt=contours"
    assert payload["ground_points_url"] == "/api/splat/jobs/splat_c0a701/geo/export?fmt=ground"


def test_contours_ground_failure_is_500_no_meta(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(fail_step="ground"))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 500 and "Ground extraction failed" in r.json()["detail"]
    assert "contours" not in json.loads((job_dir / "meta.json").read_text())


def test_contours_cdt_failure_is_500_with_recipe_error(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(fail_step="cdt"))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 500 and "EMPTY" in r.json()["detail"]


def test_contours_receipt_failure_is_a_note_not_a_failure(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(receipt_ok=False))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["contours"].get("receipt") is None
    assert "receipt_error" in body["contours"]
    assert body["receipt_url"] is None
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["contours"]["contours"]["contours_drawn"] == 194


# ── semantic ground (P5a) ────────────────────────────────────────────────────
def _add_langfield(job_dir: Path) -> None:
    lf = job_dir / splat_route.LANGFIELD_DIRNAME
    lf.mkdir(parents=True)
    (lf / "gauss_emb.npz").write_bytes(b"npz")
    cfg = job_dir / "processed" / "splatfacto" / "2026-07-01_000000"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")


def test_semantic_requires_langfield(client):
    http, outputs = client
    _mk_job(outputs, meters_per_unit=0.5, geo=GEO)  # mesh present, no langfield
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={"semantic": True})
    assert r.status_code == 409
    assert "language field" in r.json()["detail"]


def test_semantic_needs_no_mesh_and_passes_gaussians(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, with_mesh=False, meters_per_unit=0.5, geo=GEO)
    _add_langfield(job_dir)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(log=calls))

    r = http.post(
        "/api/splat/jobs/splat_c0a701/geo/contours",
        json={"semantic": True, "semantic_thresh": 0.5},
    )
    assert r.status_code == 200
    assert r.json()["contours"]["contours"]["contours_drawn"] == 194

    scripts = [Path(c[1]).name for c in calls]
    assert scripts[0] == "semantic_ground.py"
    ground_call = calls[1]
    assert "--ground-gaussians" in ground_call
    params = json.loads(ground_call[ground_call.index("--params-json") + 1])
    assert params["semantic_thresh"] == 0.5


def test_semantic_step_failure_is_500(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, with_mesh=False, meters_per_unit=0.5, geo=GEO)
    _add_langfield(job_dir)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(fail_step="semantic"))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={"semantic": True})
    assert r.status_code == 500
    assert "Semantic ground failed" in r.json()["detail"]


def test_auto_semantic_when_langfield_exists(client, monkeypatch):
    """Default body (semantic unset) must pick the semantic path on scenes with
    a language field — RToony's graded default."""
    http, outputs = client
    job_dir = _mk_job(outputs, with_mesh=False, meters_per_unit=0.5, geo=GEO)
    _add_langfield(job_dir)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(log=calls))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 200
    assert Path(calls[0][1]).name == "semantic_ground.py"
    assert r.json()["contours"]["params"]["semantic"] is True


def test_surface_receipts_in_report_and_served(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess())
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 200
    assert r.json()["contours"]["surface_receipts"] == ["sections.png", "surface_iso.png"]
    assert r.json()["sections_url"]
    for fmt in ("sections", "surface-iso"):
        assert http.get(f"/api/splat/jobs/splat_c0a701/geo/export?fmt={fmt}").status_code == 200, fmt


def test_auto_semantic_falls_back_to_mesh_when_no_checkpoint(client, monkeypatch):
    """Review fix: AUTO mode with a language field but pruned processed/ must
    fall back to the mesh-slope path, not 409."""
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)  # mesh present
    lf = job_dir / splat_route.LANGFIELD_DIRNAME
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"npz")  # langfield but NO processed/
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(log=calls))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 200
    assert r.json()["contours"]["params"]["semantic"] is False
    assert all("semantic_ground" not in " ".join(map(str, c)) for c in calls)


def test_contours_failure_preserves_prior_artifacts(client, monkeypatch):
    """Review fix: a mid-chain failure must never leave mixed-generation
    artifacts — the prior build survives untouched."""
    http, outputs = client
    job_dir = _mk_job(outputs, meters_per_unit=2.35, geo=GEO)
    geo_dir = job_dir / splat_route.MESH_DIRNAME / "geo"
    geo_dir.mkdir(parents=True)
    (geo_dir / "contours.dxf").write_text("OLD GOOD DXF")
    (geo_dir / "ground_points.txt").write_text("OLD POINTS")
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(fail_step="cdt"))
    r = http.post("/api/splat/jobs/splat_c0a701/geo/contours", json={})
    assert r.status_code == 500
    assert (geo_dir / "contours.dxf").read_text() == "OLD GOOD DXF"
    assert (geo_dir / "ground_points.txt").read_text() == "OLD POINTS"
