"""Ground + environment route (P6e): requires a built language field AND an
exported preview splat.ply (color source), the semantic-ground -> TIN ->
twin_finish chain, loud failure on a toolchain crash, and slug-safe file
serving. Subprocesses mocked; ground_extract.py's cell-binning/spike-reject/
connected-component algorithm (reused verbatim in ground_mesh_build.py) is
already proven on real data (P1, 2026-07-21).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

BUILD_REPORT = {
    "v": 1, "provenance": "ground-derived", "semantic_thresh": 0.5, "cell_units": 0.03,
    "gaussians_total": 500_000, "gaussians_ground": 12_000,
    "cells_with_data": 900, "cells_spike_rejected": 5, "cells_disconnected_dropped": 20,
    "ground_points": 875, "triangles": 1600, "ground_z_range_units": [-0.5, -0.48],
    "artifacts": {"mesh": "ground_mesh_raw.ply"},
}
FINISH_REPORT = {"verts": 875, "faces": 1600, "solid_gaussians": 12_000,
                 "units": "meters", "extent": [4.0, 4.0, 0.1]}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0b0005", langfield: bool = True,
           preview_splat: bool = True) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d", "meters_per_unit": 2.35,
    }))
    cfg = job_dir / "processed" / "splatfacto" / "ts"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")
    if langfield:
        lf = job_dir / splat_route.LANGFIELD_DIRNAME
        lf.mkdir()
        (lf / "gauss_emb.npz").write_bytes(b"npz")
    if preview_splat:
        prev = job_dir / "_preview"
        prev.mkdir()
        (prev / "splat.ply").write_bytes(b"ply")
    return job_dir


def _fake_subprocess(calls: list, build_ok: bool = True, finish_ok: bool = True):
    async def run(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "semantic_ground" in joined:
            Path(command[4]).write_bytes(b"npz")
            return 0, b"", b""
        if "ground_mesh_build" in joined:
            if not build_ok:
                return 1, b"", b"FATAL: only 10 ground cells after filtering (floor: 50)"
            out_dir = Path(command[3])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "ground_mesh_raw.ply").write_bytes(b"ply")
            (out_dir / "ground_mesh_build.json").write_text(json.dumps(BUILD_REPORT))
            return 0, b"", b""
        if "twin_finish" in joined:
            if not finish_ok:
                return 1, b"", b"FATAL: empty mesh"
            out_glb = Path(command[4])
            out_glb.write_bytes(b"glb")
            out_glb.with_name("twin_finish.json").write_text(json.dumps(FINISH_REPORT))
            return 0, b"", b""
        if "ground_mesh_receipt" in joined:
            out_dir = Path(command[3])
            (out_dir / "receipt_top.png").write_bytes(b"png")
            (out_dir / "receipt_oblique.png").write_bytes(b"png")
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_scene_ground_requires_langfield(client):
    http, outputs = client
    _mk_job(outputs, langfield=False)
    r = http.post("/api/splat/jobs/splat_0b0005/scene/ground", json={})
    assert r.status_code == 409
    assert "language field" in r.json()["detail"].lower()


def test_scene_ground_requires_preview_splat(client):
    http, outputs = client
    _mk_job(outputs, preview_splat=False)
    r = http.post("/api/splat/jobs/splat_0b0005/scene/ground", json={})
    assert r.status_code == 409
    assert "splat.ply" in r.json()["detail"]


def test_scene_ground_full_build(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0005/scene/ground",
                  json={"semantic_thresh": 0.6, "cell_units": 0.02})
    assert r.status_code == 200
    body = r.json()
    assert body["ground_points"] == 875
    assert body["provenance"] == "ground-derived"
    assert body["finish"]["faces"] == 1600

    # meters_per_unit passed through to twin_finish
    finish_call = next(c for c in calls if "twin_finish" in " ".join(str(x) for x in c))
    assert "--meters-per-unit" in finish_call
    assert "2.35" in [str(x) for x in finish_call]

    # Review finding 2026-07-23: the posted tuning knobs must actually reach
    # ground_mesh_build.py's argv, not silently fall back to defaults.
    build_call = [str(x) for x in next(
        c for c in calls if "ground_mesh_build" in " ".join(str(x) for x in c))]
    assert "--semantic-thresh" in build_call and "0.6" in build_call
    assert "--cell-units" in build_call and "0.02" in build_call

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["ground"]["ground_points"] == 875

    for fmt in ("report", "glb", "top", "oblique"):
        resp = http.get("/api/splat/jobs/splat_0b0005/scene/ground/file", params={"fmt": fmt})
        assert resp.status_code == 200, fmt


def test_scene_ground_coverage_floor_fails_loud(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(calls, build_ok=False))
    r = http.post("/api/splat/jobs/splat_0b0005/scene/ground", json={})
    assert r.status_code == 500
    assert "floor" in r.json()["detail"]


def test_scene_ground_file_404_before_build(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0005/scene/ground/file", params={"fmt": "glb"})
    assert r.status_code == 404
