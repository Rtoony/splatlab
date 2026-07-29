"""POST /jobs/{id}/world/solidify — the walkable world, buildable from the app.

scene_solidify.py, world_collision.py and world_gate.py were CLI-only: a browser
could read a world but never build one, so the artifact the whole interactive
lane rests on had no admission control, no lock, no progress and no receipt.

Subprocesses are stubbed (the real ones need the mesh conda env, CoACD and
minutes of CPU); what is tested here is the contract around them — gating,
locking, argument construction, the honest reporting of a world that builds but
fails its gates, and the operation-registry bookkeeping.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opregistry  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_0d1e5a"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(splat_route, "MESH_ENV_PYTHON", tmp_path / "python")
    (tmp_path / "python").write_text("#!/bin/sh\n")
    monkeypatch.setattr(splat_route, "_MESH_EXPORT_LOCKS", {})
    opregistry.configure_storage(tmp_path / "ops")
    opregistry.init_db()
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    try:
        yield TestClient(app), outputs
    finally:
        opregistry.configure_storage(Path(opregistry.PROJECT_ROOT) / "data" / "ops")


def _mk_job(outputs: Path, *, status="completed", isolated=True, mpu=0.5,
            generation=3) -> Path:
    job_dir = outputs / JOB
    job_dir.mkdir(parents=True)
    meta = {"job_id": JOB, "output_dir": str(job_dir), "status": status}
    if mpu is not None:
        meta["meters_per_unit"] = mpu
    if generation is not None:
        meta["scale_generation"] = generation
    (job_dir / "meta.json").write_text(json.dumps(meta))
    if isolated:
        (job_dir / "_scene" / "isolated").mkdir(parents=True)
    return job_dir


def _stub_runs(monkeypatch, job_dir: Path, *, solidify_rc=0, collision_rc=0,
               gate_rc=0, write_world=True, gate_report=None):
    """Record every command and fake the three stage outcomes."""
    calls: list[list[str]] = []

    async def fake_run(command):
        calls.append(list(command))
        script = Path(command[1]).name
        if script == "scene_solidify.py":
            if write_world:
                world = job_dir / "_world"
                world.mkdir(parents=True, exist_ok=True)
                (world / "world.json").write_text(json.dumps({"elements": ["chair"]}))
            return solidify_rc, b"", b"solidify stderr tail"
        if script == "world_collision.py":
            return collision_rc, b"", b"collision stderr tail"
        if script == "world_gate.py":
            if gate_report is not None:
                (job_dir / "_world" / "world_gate.json").write_text(json.dumps(gate_report))
            return gate_rc, b"", b"gate stderr tail"
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_run)
    # The route checks each script exists before running it, and the stub above
    # dispatches on the script's filename, so these must be real files with the
    # real names.
    scripts = job_dir.parent.parent / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for attr, filename in (("SCENE_SOLIDIFY_SCRIPT", "scene_solidify.py"),
                           ("WORLD_COLLISION_SCRIPT", "world_collision.py"),
                           ("WORLD_GATE_SCRIPT", "world_gate.py")):
        path = scripts / filename
        path.write_text("# stub\n")
        monkeypatch.setattr(splat_route, attr, path)
    return calls


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def test_unknown_job_404s(client):
    http, _ = client
    assert http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={}).status_code == 404


def test_incomplete_job_409s(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, status="running")
    _stub_runs(monkeypatch, job_dir)
    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})
    assert r.status_code == 409 and "completed" in r.json()["detail"]


def test_missing_isolated_elements_409s(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, isolated=False)
    _stub_runs(monkeypatch, job_dir)
    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})
    assert r.status_code == 409 and "isolate" in r.json()["detail"]


def test_heavy_work_gate_is_enforced(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir)

    def refuse():
        raise splat_route.HTTPException(status_code=503, detail="maintenance")
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", refuse)

    assert http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={}).status_code == 503


def test_a_running_build_for_the_same_job_409s(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir)

    held = asyncio.Lock()
    asyncio.run(held.acquire())
    monkeypatch.setattr(splat_route, "_MESH_EXPORT_LOCKS", {JOB: held})

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})
    assert r.status_code == 409 and "already running" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------

def test_all_three_stages_run_in_order(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    assert r.status_code == 200, r.text
    assert [Path(c[1]).name for c in calls] == [
        "scene_solidify.py", "world_collision.py", "world_gate.py"]


def test_calibration_is_passed_through_when_present(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs, mpu=0.5)
    calls = _stub_runs(monkeypatch, job_dir)

    http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    assert "--meters-per-unit" in calls[0]
    assert calls[0][calls[0].index("--meters-per-unit") + 1] == "0.5"


def test_an_uncalibrated_world_still_builds_and_says_so(client, monkeypatch):
    """Refusing would be worse: the gates report scale_sanity as SKIPPED."""
    http, outputs = client
    job_dir = _mk_job(outputs, mpu=None)
    calls = _stub_runs(monkeypatch, job_dir)

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    assert r.status_code == 200
    assert "--meters-per-unit" not in calls[0]
    assert r.json()["uncalibrated"] is True


def test_body_options_reach_the_command(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)
    # --only is a patch of an EXISTING world; seed one so the guard passes.
    world = job_dir / "_world"
    world.mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({"elements": ["chair"]}))

    http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={
        "prop_faces": 4000, "shell_faces": 90000, "shell_source": "voxel",
        "only": "chair,table", "skip_shell": True, "prefer_generated": True})

    cmd = calls[0]
    assert cmd[cmd.index("--prop-faces") + 1] == "4000"
    assert cmd[cmd.index("--shell-faces") + 1] == "90000"
    assert cmd[cmd.index("--shell-source") + 1] == "voxel"
    assert cmd[cmd.index("--only") + 1] == "chair,table"
    # The review footgun: bare --only over HTTP rewrote world.json to just the
    # filtered slugs. The route now ALWAYS patches.
    assert "--patch-elements" in cmd
    assert "--skip-shell" in cmd and "--prefer-generated" in cmd


def test_only_without_an_existing_world_409s(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)
    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify",
                  json={"only": "chair"})
    assert r.status_code == 409
    assert "full solidify" in r.json()["detail"]
    assert calls == []  # refused before any subprocess ran


def test_only_plus_shell_only_400s(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)
    world = job_dir / "_world"
    world.mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({"elements": ["chair"]}))
    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify",
                  json={"only": "chair", "shell_only": True})
    assert r.status_code == 400
    assert "conflict" in r.json()["detail"]
    assert calls == []


def test_overrides_reach_the_command_as_sorted_json(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)
    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={
        "overrides": {"chair": "captured", "bench": "generated"}})
    assert r.status_code == 200
    cmd = calls[0]
    assert cmd[cmd.index("--element-source") + 1] == json.dumps(
        {"bench": "generated", "chair": "captured"}, sort_keys=True)


def test_override_values_are_a_closed_vocabulary(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify",
                  json={"overrides": {"chair": "proxy"}})
    assert r.status_code == 422  # Literal["captured","generated"] refuses


def test_override_keys_are_slug_safe(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)
    for key in ("UPPER", "../x", "a b"):
        r = http.post(f"/api/splat/jobs/{JOB}/world/solidify",
                      json={"overrides": {key: "captured"}})
        assert r.status_code == 400, key
        assert "bad override slug" in r.json()["detail"]
    assert calls == []  # refused before any subprocess ran


def test_follow_on_stages_can_be_skipped(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls = _stub_runs(monkeypatch, job_dir)

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify",
                  json={"run_collision": False, "run_gate": False})

    assert [Path(c[1]).name for c in calls] == ["scene_solidify.py"]
    assert r.json()["collision"] is None and r.json()["gate"] is None


@pytest.mark.parametrize("bad", [{"prop_faces": 0}, {"texture_size": 99999},
                                 {"shell_source": "marble"}])
def test_out_of_range_options_are_rejected(client, monkeypatch, bad):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir)
    assert http.post(f"/api/splat/jobs/{JOB}/world/solidify", json=bad).status_code == 422


# ---------------------------------------------------------------------------
# Failure and honesty
# ---------------------------------------------------------------------------

def test_solidify_failure_is_a_500_with_the_log_tail(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir, solidify_rc=1, write_world=False)

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    assert r.status_code == 500
    assert "solidify stderr tail" in r.json()["detail"]


def test_a_zero_exit_that_wrote_no_world_is_still_a_failure(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir, solidify_rc=0, write_world=False)

    assert http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={}).status_code == 500


def test_a_world_that_fails_its_gates_is_returned_not_hidden(client, monkeypatch):
    """world_gate is the acceptance suite, not a precondition. Suppressing a
    built-but-failing world would hide exactly what the gates exist to show."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir, gate_rc=1,
               gate_report={"floor_continuity": "FAIL", "shell_connectivity": "PASS"})

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    assert r.status_code == 200
    assert r.json()["gate"]["passed"] is False
    assert r.json()["gate"]["report"]["floor_continuity"] == "FAIL"
    assert r.json()["world"]["elements"] == ["chair"]


def test_collision_failure_does_not_sink_the_build(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir, collision_rc=1)

    r = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    assert r.status_code == 200
    assert r.json()["collision"]["exit_code"] == 1
    assert "collision stderr tail" in r.json()["collision"]["error"]


# ---------------------------------------------------------------------------
# Operation registry bookkeeping
# ---------------------------------------------------------------------------

def test_a_successful_build_leaves_a_finished_operation(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir)

    op_id = http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={}).json()["op_id"]

    record = opregistry.get(op_id)
    assert record["kind"] == "world_solidify"
    assert record["job_id"] == JOB
    assert record["status"] == opregistry.SUCCEEDED
    assert record["result"]["gate_passed"] is True
    assert record["progress"] == pytest.approx(1.0)


def test_a_failed_build_leaves_a_durable_record(client, monkeypatch):
    """The point of the registry: a dropped connection is no longer the only
    trace of a multi-minute build that failed."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    _stub_runs(monkeypatch, job_dir, solidify_rc=1, write_world=False)

    http.post(f"/api/splat/jobs/{JOB}/world/solidify", json={})

    records = opregistry.list_ops(job_id=JOB, kind="world_solidify")
    assert len(records) == 1
    assert records[0]["status"] == opregistry.FAILED
    assert "solidify stderr tail" in records[0]["error"]


def test_the_build_records_the_scale_generation_it_used(client, monkeypatch):
    """A later /scale bump makes this comparison the staleness signal."""
    http, outputs = client
    job_dir = _mk_job(outputs, generation=7)
    _stub_runs(monkeypatch, job_dir)

    assert http.post(f"/api/splat/jobs/{JOB}/world/solidify",
                     json={}).json()["scale_generation"] == 7
