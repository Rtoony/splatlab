"""Language search for Raw 360 (Spirula) scenes: no checkpoint, so the field's scene.json stands in for one."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_route  # noqa: E402

JOB = "splat_1f0001"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(splat_route, "_langfield_available", lambda: True)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk(outputs: Path, spirula: bool = True) -> Path:
    job = outputs / JOB
    (job / "_preview").mkdir(parents=True)
    (job / "_preview" / "web.ply").write_bytes(b"ply")
    meta = {"job_id": JOB, "output_dir": str(job), "status": "completed", "mode": "3d"}
    if spirula:
        (job / "_spirula").mkdir()
        meta["trainer_resolved"] = "spirula"
    (job / "meta.json").write_text(json.dumps(meta))
    return job


def test_scene_json_stands_in_for_the_checkpoint(tmp_path):
    job = tmp_path / "j"
    (job / "_langfield").mkdir(parents=True)
    assert splat_route._langfield_config(job) is None
    (job / "_langfield" / "scene.json").write_text("{}")
    assert splat_route._langfield_config(job) == job / "_langfield" / "scene.json"
    cfg = job / "processed" / "x" / "config.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("")
    assert splat_route._langfield_config(job) == cfg          # a real checkpoint always wins


def test_build_starts_in_the_background_and_is_a_noop_once_built(client, monkeypatch):
    tc, outputs = client
    job = _mk(outputs)
    started = []

    async def fake_gpu_op(**kw):
        started.append(kw["lane"])
        lf = job / "_langfield"
        (lf / "gauss_emb.npz").write_bytes(b"npz")
        (lf / "scene.json").write_text("{}")
        return 0, b"", b""

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu_op)
    r = tc.post(f"/api/splat/jobs/{JOB}/langfield/build", json={})
    assert r.status_code == 200 and r.json()["status"] == "started" and r.json()["op_id"]
    # TestClient runs the event loop per request; the task ran by the time the next request is served
    tc.get("/api/splat/viewer-context")
    assert started == ["splat-langfield"]
    assert tc.post(f"/api/splat/jobs/{JOB}/langfield/build", json={}).json()["status"] == "already built"


def test_build_is_for_raw_360_scenes(client):
    tc, outputs = client
    _mk(outputs, spirula=False)
    r = tc.post(f"/api/splat/jobs/{JOB}/langfield/build", json={})
    assert r.status_code == 409 and "Raw 360" in r.json()["detail"]


def test_queries_on_a_raw_360_scene_never_take_the_checkpoint_only_cold_path(client, monkeypatch):
    tc, outputs = client
    job = _mk(outputs)
    lf = job / "_langfield"
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"npz")
    (lf / "scene.json").write_text("{}")

    async def worker_down(*a, **k):
        return None

    async def cold(*a, **k):
        raise AssertionError("cold path loads checkpoints only")

    monkeypatch.setattr(splat_route, "_langfield_worker_query", worker_down)
    monkeypatch.setattr(splat_route, "_langfield_query_cold", cold)
    r = tc.post(f"/api/splat/jobs/{JOB}/langfield/query", json={"text": "toolbox"})
    assert r.status_code == 503 and "worker" in r.json()["detail"]
