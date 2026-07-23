"""Batch proxy route (P6d): requires a completed batch isolation, loops
crop -> TripoSplat -> ICP-register under ONE shared GPU lease, per-instance
SKIPPED:<reason> never aborts the batch, and slug-safe file serving. The
underlying crop/generate/register chain is unchanged from the proven P5c
single-object flow (test_objects_route.py); this only tests the NEW batching
orchestration and failure isolation.
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

ISOLATE_REPORT = {
    "n_gaussians": 500_000,
    "instances": [
        {"slug": "fire-hydrant", "label": "fire hydrant", "n_members_final": 28627, "status": "built"},
        {"slug": "tiny-pebble", "label": "tiny pebble", "status": "SKIPPED:too-few-members-after-dedup"},
    ],
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_triposplat_availability",
                        lambda: {"triposplat_available": True, "triposplat_runner": "/fake/run.sh"})

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job_with_isolation(outputs: Path, job_id: str = "splat_0b0004") -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    cfg = job_dir / "processed" / "splatfacto" / "ts"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")
    isolated = job_dir / splat_route.SCENE_DIRNAME / "isolated"
    (isolated / "fire-hydrant").mkdir(parents=True)
    (isolated / "fire-hydrant" / "object.ply").write_bytes(b"ply")
    (isolated / "fire-hydrant" / "object.json").write_text(json.dumps(
        {"bbox_tight": {"min": [-1, -1, -1], "max": [1, 1, 1]}}))
    (isolated / "batch_isolate.json").write_text(json.dumps(ISOLATE_REPORT))
    return job_dir


def _fake_subprocess(calls: list, register_ok: bool = True, generate_ok: bool = True):
    async def run(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "object_crop" in joined:
            Path(command[4]).write_bytes(b"png")
            Path(command[4]).with_suffix(".json").write_text(json.dumps({"cam": 3, "box": [0, 0, 10, 10]}))
            return 0, b"", b""
        if "/fake/run.sh" in joined:
            if not generate_ok:
                return 1, b"", b"FATAL: triposplat crashed"
            raw = Path(command[3])
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "splat.ply").write_bytes(b"ply")
            (raw / "thumb.webp").write_bytes(b"webp")
            return 0, b"", b""
        if "proxy_register" in joined:
            if not register_ok:
                return 1, b"", b"FATAL: icp failed"
            proxy_ply = Path(command[4])
            proxy_ply.write_bytes(b"ply")
            proxy_ply.with_suffix(".json").write_text(json.dumps(
                {"icp_fitness": 0.95, "icp_rmse": 0.01, "total_scale": 0.8,
                 "transform_4x4": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}))
            return 0, b"", b""
        if "proxy_triptych" in joined:
            Path(command[-1]).write_bytes(b"png")
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_scene_proxy_requires_isolation(client):
    http, outputs = client
    job_dir = outputs / "splat_0b0004"
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": "splat_0b0004", "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    r = http.post("/api/splat/jobs/splat_0b0004/scene/proxy", json={})
    assert r.status_code == 409
    assert "isolation" in r.json()["detail"].lower()


def test_scene_proxy_full_build(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0004/scene/proxy", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["n_built"] == 1
    built = [i for i in body["instances"] if i["status"] == "built"][0]
    assert built["slug"] == "fire-hydrant"
    assert built["icp_fitness"] == 0.95
    assert built["transform_4x4"]

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["proxy"]["n_built"] == 1

    for fmt, extra in (("report", {}), ("proxy", {"name": "fire-hydrant"}),
                       ("preview", {"name": "fire-hydrant"}), ("triptych", {"name": "fire-hydrant"})):
        resp = http.get("/api/splat/jobs/splat_0b0004/scene/proxy/file", params={"fmt": fmt, **extra})
        assert resp.status_code == 200, fmt

    # proxy_raw scratch cleaned
    assert not (job_dir / splat_route.SCENE_DIRNAME / "proxied" / "fire-hydrant" / "proxy_raw").exists()


def test_scene_proxy_generation_failure_skips_never_aborts(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(calls, generate_ok=False))
    r = http.post("/api/splat/jobs/splat_0b0004/scene/proxy", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["n_built"] == 0
    assert body["instances"][0]["status"] == "SKIPPED:generation-failed"


def test_scene_proxy_registration_failure_skips_never_aborts(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(calls, register_ok=False))
    r = http.post("/api/splat/jobs/splat_0b0004/scene/proxy", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["n_built"] == 0
    assert body["instances"][0]["status"] == "SKIPPED:registration-failed"


def test_scene_proxy_triposplat_unavailable_400(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    monkeypatch.setattr(splat_route, "_triposplat_availability",
                        lambda: {"triposplat_available": False, "triposplat_runner": ""})

    async def fake_sub(command):
        raise AssertionError("no subprocess may run")

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_0b0004/scene/proxy", json={})
    assert r.status_code == 400
