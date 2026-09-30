"""Raw 360 (Spirula) walk: /world/prepare builds a checkpoint-free world from the splat alone."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spirula_lane  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_5e0001"


def _shell_report(verdict: str = "PASS") -> dict:
    return {"verdict": verdict, "route_used": "voxel",
            "gates": {"components": 1, "watertight": True, "floor_continuity": 0.987},
            "probe": {"floor_level_y": -1.72, "top_level_y": 1.13},
            "params": {"seed_yup": [0.1, -0.3, 0.0]}, "artifact": {"triangles_written": 1234},
            "geometry_frame": {"axis": "y-up", "units": "scene-units", "meters_per_unit": None}}


def _fake_world_shell(job: Path, verdict: str = "PASS"):
    def run(cmd, **kw):
        w = job / "_world"
        (w / "collision_shell.glb").write_bytes(b"glTF")
        (w / "collision_shell.json").write_text(json.dumps(_shell_report(verdict)))
        (w / "navmesh.json").write_text("{}")
        run.cmd = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run


def _mk_job(outputs: Path, spirula: bool = True) -> Path:
    job = outputs / JOB
    (job / "_preview").mkdir(parents=True)
    (job / "_preview" / "splat.ply").write_bytes(b"ply")
    cams = job / "_spirula" / "cameras"
    cams.mkdir(parents=True)
    # Z-up metric cameras at heights 1.2..1.6 around (1, 2): median -> Y-up (1, 1.4, -2).
    frames = [{"transform_matrix": [[1, 0, 0, 1.0], [0, 1, 0, 2.0], [0, 0, 1, z], [0, 0, 0, 1]]}
              for z in (1.2, 1.4, 1.6)]
    (cams / "transforms.json").write_text(json.dumps({"frames": frames}))
    meta = {"job_id": JOB, "output_dir": str(job), "status": "completed", "mode": "3d",
            "meters_per_unit": 1.0}
    if spirula:
        meta["trainer_resolved"] = "spirula"
    (job / "meta.json").write_text(json.dumps(meta))
    return job


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(splat_route, "MESH_ENV_PYTHON", Path(sys.executable))
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def test_seed_is_the_median_camera_in_y_up(tmp_path):
    job = _mk_job(tmp_path)
    assert spirula_lane.walk_seed_yup(job) == [1.0, 1.4, -2.0]
    assert spirula_lane.walk_seed_yup(tmp_path / "nothing") is None


def test_shell_command_uses_metric_player_and_seed(tmp_path):
    cmd = spirula_lane.walk_shell_command("py", Path("world_shell.py"), tmp_path, [1.0, 1.4, -2.0])
    assert cmd[cmd.index("--player-height") + 1] == "1.7"
    assert cmd[cmd.index("--seed") + 1] == "1,1.4,-2"
    assert cmd[cmd.index("--source") + 1] == str(tmp_path / "_preview" / "splat.ply")
    assert "--seed" not in spirula_lane.walk_shell_command("py", Path("w.py"), tmp_path, None)


def test_prepare_builds_a_metric_walk_the_manifest_route_serves(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    fake = _fake_world_shell(job)
    monkeypatch.setattr(splat_route.subprocess, "run", fake)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stages"] == {"walk": "done"} and body["walk"]["verdict"] == "PASS"
    assert fake.cmd[fake.cmd.index("--seed") + 1] == "1,1.4,-2"
    world = json.loads((job / "_world" / "world.json").read_text())
    assert world["units"] == "meters" and world["meters_per_unit"] == 1.0
    assert (job / "_world" / "shell.glb").read_bytes() == b"glTF"

    m = tc.get(f"/api/splat/jobs/{JOB}/world/manifest").json()
    assert m["shell"]["files"]["glb"].endswith("shell.glb")
    assert m["collision_shell"]["scale_to_world"] == 1.0
    assert m["collision_shell"]["spawn"]["seed_yup"] == [0.1, -0.3, 0.0]
    assert m["meters_per_unit"] == 1.0 and m["uncalibrated"] is False and m["elements"] == []

    # Built -> a second POST is a no-op; force=["walk"] rebuilds; other stage names are refused.
    fake.cmd = None
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={}).json()["stages"] == {"walk": "skipped"}
    assert fake.cmd is None
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={"force": ["walk"]}).status_code == 200
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={"force": ["mesh"]}).status_code == 400


def test_unwalkable_shell_fails_loudly_and_writes_no_world(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    monkeypatch.setattr(splat_route.subprocess, "run", _fake_world_shell(job, verdict="FAIL"))
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 422 and "not walkable" in r.json()["detail"]
    assert not (job / "_world" / "world.json").exists()
    import opregistry
    op = opregistry.list_ops(job_id=JOB, kind="world_prepare", limit=5)[0]
    assert op["status"] == opregistry.FAILED


def test_rig_lane_jobs_still_need_the_language_field(client):
    tc, outputs = client
    _mk_job(outputs, spirula=False)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 409 and "language field" in r.json()["detail"].lower()
