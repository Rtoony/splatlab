"""The one-command world pipeline: ordering, skipping, resuming, honesty."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_route  # noqa: E402

JOB = "splat_9e0001"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path) -> Path:
    job = outputs / JOB
    (job / "_langfield").mkdir(parents=True)
    (job / "_langfield" / "gauss_emb.npz").write_bytes(b"npz")
    (job / "meta.json").write_text(json.dumps({
        "job_id": JOB, "output_dir": str(job), "status": "completed",
        "mode": "3d"}))
    return job


def _mock_stages(monkeypatch: pytest.MonkeyPatch, job: Path,
                 fail: str = "", ground_partial: bool = False):
    """Each mocked stage writes the artifact its real counterpart would."""
    calls: list[str] = []

    async def mesh(request, job_id, body=None):
        calls.append("mesh")
        (job / "_mesh").mkdir(exist_ok=True)
        (job / "_mesh" / "mesh.ply").write_bytes(b"ply")

    async def inventory(request, job_id, body):
        calls.append("inventory")
        if fail == "inventory":
            raise HTTPException(status_code=503, detail="arbiter says no")
        (job / "_scene").mkdir(exist_ok=True)
        (job / "_scene" / "inventory.json").write_text("{}")

    async def isolate(request, job_id, body):
        calls.append("isolate")
        d = job / "_scene" / "isolated"
        d.mkdir(parents=True, exist_ok=True)
        (d / "batch_isolate.json").write_text("{}")

    async def ground(request, job_id, body):
        calls.append("ground")
        d = job / "_scene" / "ground"
        d.mkdir(parents=True, exist_ok=True)
        (d / "ground_mesh_raw.ply").write_bytes(b"ply")
        if ground_partial:
            raise HTTPException(status_code=500, detail="class textures blew up")

    async def solidify(job_id, body):
        calls.append("solidify")
        w = job / "_world"
        w.mkdir(exist_ok=True)
        (w / "world.json").write_text("{}")
        (w / "navmesh.json").write_text("{}")
        return {"gate": {"passed": True}}

    async def affordances(job_id, body):
        calls.append("affordances")
        return {"applied": True}

    monkeypatch.setattr(splat_route, "generate_splat_mesh", mesh)
    monkeypatch.setattr(splat_route, "scene_inventory", inventory)
    monkeypatch.setattr(splat_route, "scene_batch_isolate", isolate)
    monkeypatch.setattr(splat_route, "scene_ground", ground)
    monkeypatch.setattr(splat_route, "world_solidify", solidify)
    import world_interactions_route
    monkeypatch.setattr(world_interactions_route,
                        "propose_world_affordances", affordances)
    return calls


def test_prepare_runs_all_stages_in_order_then_skips_on_rerun(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job)

    first = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert first.status_code == 200, first.text
    body = first.json()
    assert calls == ["mesh", "inventory", "isolate", "ground",
                     "solidify", "affordances"]
    assert body["stages"]["solidify"] == "done"
    assert body["stages"]["scenario"] == "done"
    assert body["gate"] == {"passed": True}
    assert json.loads((job / "_world" / "scenario.json").read_text())["waves"]

    calls.clear()
    second = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert second.status_code == 200
    stages = second.json()["stages"]
    # Everything artifact-backed skips; affordances stays cheap-safe (no-clobber).
    assert calls == ["affordances"]
    for stage in ("mesh", "inventory", "isolate", "ground", "solidify", "scenario"):
        assert stages[stage] == "skipped", stage


def test_prepare_failure_names_the_stage_and_keeps_progress(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job, fail="inventory")

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 503
    assert "stage 'inventory' failed" in r.json()["detail"]
    assert "re-POST to resume" in r.json()["detail"]
    assert (job / "_mesh" / "mesh.ply").is_file()  # completed stage kept
    assert calls == ["mesh", "inventory"]  # halted at the failure


def test_prepare_tolerates_partial_ground(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job, ground_partial=True)

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200, r.text
    assert r.json()["stages"]["ground"] == "partial"
    assert any("partially" in w for w in r.json()["warnings"])
    assert r.json()["stages"]["solidify"] == "done"  # the chain went on


def test_prepare_requires_language_field(client, monkeypatch):
    tc, outputs = client
    job = outputs / JOB
    job.mkdir(parents=True)
    (job / "meta.json").write_text(json.dumps({
        "job_id": JOB, "output_dir": str(job), "status": "completed"}))
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 409
    assert "language field" in r.json()["detail"].lower()
