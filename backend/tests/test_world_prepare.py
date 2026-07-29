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


def test_force_redoes_the_named_stage_and_rejects_typos(client, monkeypatch):
    """The advertised redo mechanism was entirely unpinned (review finding),
    and an unknown name used to no-op in silence."""
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job)
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={}).status_code == 200

    calls.clear()
    forced = tc.post(f"/api/splat/jobs/{JOB}/world/prepare",
                     json={"force": ["ground", "solidify"]})
    assert forced.status_code == 200, forced.text
    assert calls == ["ground", "solidify", "affordances"]  # only the forced ones re-ran
    stages = forced.json()["stages"]
    assert stages["ground"] == "done" and stages["solidify"] == "done"
    assert stages["mesh"] == "skipped"

    typo = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={"force": ["gound"]})
    assert typo.status_code == 400
    assert "gound" in typo.json()["detail"]


def test_stale_artifact_never_launders_a_pre_work_failure_into_partial(client, monkeypatch):
    """A 503/409 raised BEFORE ground does any work must not read as partial
    success just because an older run left a ply behind (review finding)."""
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job)
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={}).status_code == 200
    assert (job / "_scene" / "ground" / "ground_mesh_raw.ply").is_file()

    async def blocked(request, job_id, body):  # raises without touching the ply
        raise HTTPException(status_code=503, detail="Semantic ground blocked")

    monkeypatch.setattr(splat_route, "scene_ground", blocked)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={"force": ["ground"]})
    assert r.status_code == 503, r.text
    assert "stage 'ground' failed" in r.json()["detail"]


def test_failure_diagnostics_survive_in_the_operation_registry(client, monkeypatch):
    """The op row must still say WHICH stage failed and WHY — a second
    unconditional finish() used to erase both (review finding)."""
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job, fail="inventory")
    import opregistry

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 503
    op = opregistry.list_ops(job_id=JOB, kind="world_prepare", limit=5)[0]
    assert op["status"] == opregistry.FAILED
    assert op["step"] == "inventory"
    assert op["result"]["stage"] == "inventory"
    assert "arbiter says no" in op["result"]["detail"]
    assert "arbiter says no" in op["error"]


def test_solidify_without_a_navmesh_is_partial_not_done(client, monkeypatch):
    """scene_solidify swallows a failed shell build, so world.json can land
    with no navmesh — an unwalkable world must not report 'done'."""
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job)

    async def shell_less(job_id, body):
        w = job / "_world"
        w.mkdir(exist_ok=True)
        (w / "world.json").write_text("{}")
        return {"gate": None}

    monkeypatch.setattr(splat_route, "world_solidify", shell_less)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200, r.text
    assert r.json()["stages"]["solidify"] == "partial"
    assert any("navmesh" in w for w in r.json()["warnings"])


def test_install_scenario_false_reports_skipped_and_writes_nothing(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare",
                json={"install_scenario": False})
    assert r.status_code == 200, r.text
    assert r.json()["stages"]["scenario"] == "skipped"
    assert not (job / "_world" / "scenario.json").exists()


def test_resume_after_a_failure_redoes_only_the_failed_stage(client, monkeypatch):
    """The headline promise, asserted behaviourally rather than by substring."""
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job, fail="inventory")
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={}).status_code == 503

    calls = _mock_stages(monkeypatch, job)  # the transient failure clears
    ok = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert ok.status_code == 200, ok.text
    assert calls == ["inventory", "isolate", "ground", "solidify", "affordances"]
    assert ok.json()["stages"]["mesh"] == "skipped"


def test_a_second_concurrent_prepare_is_refused_with_the_running_op(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job)
    lock = splat_route._world_prepare_lock(JOB)
    splat_route._WORLD_PREPARE_OPS[JOB] = "op-already-running"

    async def hold():
        async with lock:
            return tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})

    import asyncio
    r = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(hold())
    splat_route._WORLD_PREPARE_OPS.pop(JOB, None)
    assert r.status_code == 409
    assert "op-already-running" in r.json()["detail"]


def test_pluck_stage_runs_last_and_only_warns_on_refusal(client, monkeypatch):
    """The default fixture has no _preview/splat.ply, so the REAL pluck
    builder 409s — the ladder must record 'partial' + a warning, never fail.
    With a mocked builder the stage lands 'done' LAST and skips on re-POST."""
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job)

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200
    assert r.json()["stages"]["pluck"] == "partial"
    assert any("pluck data not built" in w for w in r.json()["warnings"])

    import world_pluck_route

    async def fake_pluck(job_id, request):
        calls.append("pluck")
        (job / "_world" / "pluck.json").write_text("{}")
        return {"ok": True}

    monkeypatch.setattr(world_pluck_route, "build_world_pluck", fake_pluck)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200
    assert r.json()["stages"]["pluck"] == "done"
    assert calls[-1] == "pluck"

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.json()["stages"]["pluck"] == "skipped"
    assert calls.count("pluck") == 1


def _seed_structure_paint(job: Path) -> None:
    (job / "_langfield" / "class_labels.json").write_text(json.dumps([
        {"id": "ab12cd34", "class_id": "structure", "count": 500}]))


def _mock_surfaces(monkeypatch, job: Path, calls: list):
    async def surfaces(request, job_id, body):
        calls.append("surfaces")
        d = job / "_scene" / "surfaces"
        d.mkdir(parents=True, exist_ok=True)
        (d / "surface_patches.json").write_text("{}")
        return {}
    monkeypatch.setattr(splat_route, "scene_surfaces", surfaces)


def test_surfaces_stage_runs_between_ground_and_solidify_when_painted(
    client, monkeypatch
):
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job)
    _mock_surfaces(monkeypatch, job, calls)
    _seed_structure_paint(job)

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200
    assert r.json()["stages"]["surfaces"] == "done"
    assert calls.index("ground") < calls.index("surfaces") < calls.index("solidify")


def test_surfaces_stage_skips_without_paint(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job)
    _mock_surfaces(monkeypatch, job, calls)

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200
    assert r.json()["stages"]["surfaces"] == "skipped"
    assert "surfaces" not in calls


def test_force_surfaces_redoes_it(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    calls = _mock_stages(monkeypatch, job)
    _mock_surfaces(monkeypatch, job, calls)
    _seed_structure_paint(job)

    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare",
                   json={}).status_code == 200
    first = calls.count("surfaces")
    assert first == 1
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare",
                json={"force": ["surfaces"]})
    assert r.status_code == 200
    assert calls.count("surfaces") == 2
    assert r.json()["stages"]["surfaces"] == "done"


def test_prepare_passes_prefer_generated_through(client, monkeypatch):
    """The ladder used to HARDCODE prefer_generated=True into its solidify
    stage; the body flag now carries it (default preserves history)."""
    tc, outputs = client
    job = _mk_job(outputs)
    _mock_stages(monkeypatch, job)
    seen: list = []

    async def capture(job_id, body):
        seen.append(body)
        w = job / "_world"
        w.mkdir(exist_ok=True)
        (w / "world.json").write_text("{}")
        (w / "navmesh.json").write_text("{}")
        return {"gate": {"passed": True}}

    monkeypatch.setattr(splat_route, "world_solidify", capture)

    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare",
                   json={}).status_code == 200
    assert seen[0].prefer_generated is True  # historical default

    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare",
                json={"force": ["solidify"], "prefer_generated": False})
    assert r.status_code == 200
    assert seen[1].prefer_generated is False
