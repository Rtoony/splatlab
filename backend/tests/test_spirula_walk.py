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


def _shell_report(verdict: str = "PASS", tris: int = 1234) -> dict:
    return {"verdict": verdict, "route_used": "voxel",
            "gates": {"components": 1, "watertight": True, "floor_continuity": 0.987},
            "probe": {"floor_level_y": -1.72, "top_level_y": 1.13},
            "params": {"seed_yup": [0.1, -0.3, 0.0]}, "artifact": {"triangles_written": tris},
            "geometry_frame": {"axis": "y-up", "units": "scene-units", "meters_per_unit": None}}


def _fake_world_shell(job: Path, verdict: str = "PASS", by_route: dict | None = None,
                      tris_by_voxel: dict | None = None):
    def run(cmd, **kw):
        w = job / "_world"
        if cmd[1].endswith("walk_visual_shell.py"):
            run.visual.append(cmd[cmd.index("--max-faces") + 1])
            (w / "shell.glb").write_bytes(b"glTF-light")
            return subprocess.CompletedProcess(cmd, 0, "{}", "")
        route = cmd[cmd.index("--route") + 1]
        voxel = cmd[cmd.index("--voxel-size") + 1]
        run.voxels.append(voxel)
        (w / "collision_shell.glb").write_bytes(b"glTF")
        (w / "collision_shell.json").write_text(json.dumps(_shell_report(
            (by_route or {}).get(route, verdict), (tris_by_voxel or {}).get(voxel, 1234))))
        (w / "navmesh.json").write_text("{}")
        run.cmd = cmd
        run.routes.append(route)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    run.routes = []
    run.visual = []
    run.voxels = []
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
    assert "--seed=1,1.4,-2" in cmd and cmd[cmd.index("--route") + 1] == "auto"
    assert cmd[cmd.index("--source") + 1] == str(tmp_path / "_preview" / "splat.ply")
    assert not any(a.startswith("--seed") for a in spirula_lane.walk_shell_command("py", Path("w.py"), tmp_path, None))


def test_negative_seed_parses_as_a_value_not_an_option(tmp_path):
    """Condo frontage etc. failed with argparse exit 2: `--seed -1.2,...` reads as an option."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed")
    cmd = spirula_lane.walk_shell_command("py", Path("w.py"), tmp_path, [-1.2, 0.5, -3.0])
    seed_arg = next(a for a in cmd if a.startswith("--seed"))
    assert ap.parse_args([seed_arg]).seed == "-1.2,0.5,-3"


def test_prepare_builds_a_metric_walk_the_manifest_route_serves(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    fake = _fake_world_shell(job)
    monkeypatch.setattr(splat_route.subprocess, "run", fake)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stages"] == {"walk": "done"} and body["walk"]["verdict"] == "PASS"
    assert "--seed=1,1.4,-2" in fake.cmd and fake.routes == ["auto"]
    world = json.loads((job / "_world" / "world.json").read_text())
    assert world["units"] == "meters" and world["meters_per_unit"] == 1.0
    # The visual stand-in is the decimated one, never a second full-size download of the collision solid.
    assert fake.visual == ["200000"] and (job / "_world" / "shell.glb").read_bytes() == b"glTF-light"

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


def test_auto_pick_under_the_floor_gate_retries_the_splat_transform_route(client, monkeypatch):
    tc, outputs = client
    job = _mk_job(outputs)
    fake = _fake_world_shell(job, by_route={"auto": "NOT_WALKABLE", "splat-transform": "WALKABLE_NOT_WATERTIGHT"})
    monkeypatch.setattr(splat_route.subprocess, "run", fake)
    r = tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={})
    assert r.status_code == 200, r.text
    assert fake.routes == ["auto", "splat-transform"]
    assert r.json()["walk"]["verdict"] == "WALKABLE_NOT_WATERTIGHT"


def test_a_solid_too_heavy_for_the_browser_is_rebuilt_once_coarser(client, monkeypatch):
    """Pool long take: 16.6M triangles at the 0.07 voxel never loaded in the walker."""
    tc, outputs = client
    job = _mk_job(outputs)
    fake = _fake_world_shell(job, tris_by_voxel={"0.07": 16_581_712, "0.165": 3_050_000})
    monkeypatch.setattr(splat_route.subprocess, "run", fake)
    assert tc.post(f"/api/splat/jobs/{JOB}/world/prepare", json={}).status_code == 200
    assert fake.voxels == ["0.07", "0.165"]
    assert json.loads((job / "_world" / "collision_shell.json").read_text())["artifact"]["triangles_written"] == 3_050_000


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
