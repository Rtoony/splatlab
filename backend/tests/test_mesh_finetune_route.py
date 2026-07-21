"""finetune=true on POST /jobs/{id}/mesh: must bypass the idempotency cache,
prefix the runner env with MESH_FINETUNE=1, and reserve the training-class
VRAM — while the default path stays byte-identical (cached, plain command).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request  # noqa: F401
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


RESERVED: list[int] = []


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: True)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda d: d / "config.yml")

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    RESERVED.clear()

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        RESERVED.append(vram_mb)
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_completed_job(outputs: Path, job_id: str = "splat_f70001", with_mesh: bool = True) -> Path:
    job_dir = outputs / job_id
    (job_dir / splat_route.MESH_DIRNAME).mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    if with_mesh:
        (job_dir / splat_route.MESH_DIRNAME / "mesh.ply").write_bytes(b"ply")
        (job_dir / splat_route.MESH_DIRNAME / "mesh.json").write_text(
            json.dumps({"v": 1, "tris": 1, "recipe": {"checkpoint": "vanilla"}})
        )
    return job_dir


class SubprocessSpy:
    def __init__(self, job_dir: Path):
        self.commands: list[list[str]] = []
        self.job_dir = job_dir

    async def __call__(self, command):
        self.commands.append(command)
        mdir = self.job_dir / splat_route.MESH_DIRNAME
        (mdir / "mesh.ply").write_bytes(b"ply2")
        (mdir / "mesh.json").write_text(
            json.dumps({"v": 1, "tris": 2, "recipe": {"checkpoint": "dn-finetune-3000"}})
        )
        return 0, b"", b""


def test_default_post_stays_cached_when_mesh_exists(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_completed_job(outputs)
    spy = SubprocessSpy(job_dir)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", spy)

    r = http.post("/api/splat/jobs/splat_f70001/mesh")
    assert r.status_code == 200
    assert r.json()["cached"] is True
    assert spy.commands == []  # never ran


def test_finetune_bypasses_cache_and_prefixes_env(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_completed_job(outputs)
    spy = SubprocessSpy(job_dir)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", spy)

    r = http.post("/api/splat/jobs/splat_f70001/mesh", json={"finetune": True})
    assert r.status_code == 200
    assert r.json()["cached"] is False
    assert r.json()["mesh"]["recipe"]["checkpoint"] == "dn-finetune-3000"
    assert len(spy.commands) == 1
    assert spy.commands[0][:2] == ["env", "MESH_FINETUNE=1"]
    assert RESERVED == [splat_route.MESH_FT_VRAM_MB]

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["mesh"]["recipe"]["checkpoint"] == "dn-finetune-3000"


def test_finish_builds_twin_glb(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_completed_job(outputs)
    (job_dir / splat_route.PREVIEW_DIRNAME).mkdir()
    (job_dir / splat_route.PREVIEW_DIRNAME / "splat.ply").write_bytes(b"ply")
    calls: list = []

    async def fake_sub(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        assert "twin_finish" in joined
        mdir = job_dir / splat_route.MESH_DIRNAME
        (mdir / "twin.glb").write_bytes(b"glb")
        (mdir / "twin_finish.json").write_text(json.dumps({"faces": 399999, "units": "meters"}))
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_f70001/mesh", json={"finish": True})
    assert r.status_code == 200
    assert r.json()["cached"] is True  # mesh itself cached; finish still ran
    assert len(calls) == 1
    assert r.json()["twin_glb_url"]

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["mesh"]["twin"]["faces"] == 399999
    assert http.get("/api/splat/jobs/splat_f70001/mesh/file?fmt=twin").status_code == 200


def test_finish_requires_preview_splat(client, monkeypatch):
    http, outputs = client
    _mk_completed_job(outputs)

    async def fake_sub(command):
        raise AssertionError("should not run")

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_f70001/mesh", json={"finish": True})
    assert r.status_code == 409
    assert "splat.ply" in r.json()["detail"]


def test_gate_scores_mesh_and_patches_meta(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_completed_job(outputs)
    cfg = job_dir / "processed" / "splatfacto" / "ts"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")
    calls: list = []

    async def fake_sub(command):
        calls.append(command)
        assert "mesh_gate" in " ".join(str(c) for c in command)
        (job_dir / splat_route.MESH_DIRNAME / "mesh_gate.json").write_text(json.dumps({
            "v": 1, "median_coverage": 0.745, "median_psnr": 11.86,
            "median_ssim": 0.216, "convention": "o3d-unlit", "per_cam": [],
        }))
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_f70001/mesh", json={"gate": True})
    assert r.status_code == 200
    assert len(calls) == 1
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["mesh"]["gate"]["median_psnr"] == 11.86
    assert "per_cam" not in meta["mesh"]["gate"]  # meta stays lean
