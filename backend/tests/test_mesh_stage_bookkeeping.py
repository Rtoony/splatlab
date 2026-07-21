"""Splat→mesh stage bookkeeping: mesh is a best-effort OPT-IN stage — its failure
must NEVER flip the overall job to "failed", its measured report must land
durably in meta["mesh"], and the opt-in/toolchain guards must control whether it
is planned at all.

CPU-only: no GPU, no real subprocess. _run_locked_stage is monkeypatched
(pattern copied from test_health_stage_bookkeeping.py).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

FAKE_REPORT = {
    "v": 1,
    "generated_at": "2026-07-21T00:00:00+00:00",
    "verts": 120_000,
    "tris": 240_000,
    "watertight": False,
    "clusters": 42,
    "lcc_pct": 78.8,
    "bbox_extent_m": [9.9, 10.3, 3.2],
    "bbox_extent_robust_m": [8.9, 9.6, 2.9],
    "recipe": {"method": "o3dtsdf", "voxel_size": 0.015, "sdf_truc": 0.045},
    "artifacts": {"ply": "mesh.ply", "receipts": ["view_top.png"]},
}


def _mk_job_dir(outputs: Path, job_id: str, stages: list[str]) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    meta = splat_route._new_meta(
        job_id,
        splat_route.SplatTrainRequest(mode="3d", input_path="clip.mp4", output_dir="outputs/3d"),
        Path("/in/clip.mp4"),
        job_dir,
        stages,
    )
    (job_dir / "meta.json").write_text(json.dumps(meta))
    return job_dir


@pytest.fixture()
def job_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda job_dir: job_dir / "config.yml")
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: True)
    return outputs


def _run(job_id: str, job_dir: Path) -> None:
    job = splat_route.SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path="/in/clip.mp4",
        stages_planned=["mesh"],
        stage_commands={},
    )
    asyncio.run(splat_route._run_pipeline(job))


def test_mesh_nonzero_exit_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_me00001"
    job_dir = _mk_job_dir(outputs, job_id, ["mesh"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        return 1  # simulated gs-mesh crash

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["error_message"] is None
    assert meta["stages_completed"] == []
    assert meta["stages_failed"] == [{"stage": "mesh", "reason": "exit code 1"}]
    assert meta.get("mesh") is None


def test_mesh_zero_exit_without_report_is_a_failure(job_env, monkeypatch):
    """Exit 0 but no mesh.json = no receipt = failed export, never success."""
    outputs = job_env
    job_id = "splat_me00002"
    job_dir = _mk_job_dir(outputs, job_id, ["mesh"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        return 0  # liar: exits clean, wrote nothing

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == []
    assert meta["stages_failed"] == [{"stage": "mesh", "reason": "exit code 0"}]
    assert meta.get("mesh") is None


def test_mesh_success_persists_report_in_meta(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_me00003"
    job_dir = _mk_job_dir(outputs, job_id, ["mesh"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        # The real runner writes mesh.json + mesh.ply into _mesh/ before exit 0;
        # the runner branch created the dir before invoking us.
        mdir = job_dir / splat_route.MESH_DIRNAME
        (mdir / "mesh.json").write_text(json.dumps(FAKE_REPORT))
        (mdir / "mesh.ply").write_bytes(b"ply")
        return 0

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["mesh"]
    assert meta["stages_failed"] == []
    assert meta["mesh"]["lcc_pct"] == 78.8
    assert meta["mesh"]["recipe"]["method"] == "o3dtsdf"

    # Visible end-to-end through the **meta spread + URL the API/frontend read.
    payload = splat_route._job_payload(meta)
    assert payload["mesh"]["tris"] == 240_000
    assert payload["mesh_file_url"] == f"/api/splat/jobs/{job_id}/mesh/file"
    assert payload["mesh_glb_url"] is None  # no mesh.glb written


def test_mesh_exception_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_me00004"
    job_dir = _mk_job_dir(outputs, job_id, ["mesh"])

    async def raising_run_locked_stage(job, stage, command, vram_mb):
        raise OSError("disk full")

    monkeypatch.setattr(splat_route, "_run_locked_stage", raising_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == []
    assert len(meta["stages_failed"]) == 1
    assert meta["stages_failed"][0]["stage"] == "mesh"
    assert "disk full" in meta["stages_failed"][0]["reason"]


def test_mesh_skipped_no_config_is_not_recorded_as_failure(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda job_dir: None)
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: False)

    job_id = "splat_me00005"
    job_dir = _mk_job_dir(outputs, job_id, ["mesh"])
    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["mesh"]
    assert meta["stages_failed"] == []


# ── plan guard (_append_mesh_stage) ──────────────────────────────────────────────


def _req(**kwargs) -> splat_route.SplatTrainRequest:
    return splat_route.SplatTrainRequest(
        mode="3d", input_path="clip.mp4", output_dir="outputs/3d", **kwargs
    )


def test_plan_guard_appends_when_opted_in(monkeypatch):
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: True)
    stages = ["process", "train", "export", "health"]
    splat_route._append_mesh_stage(stages, _req(mesh_export=True))
    assert stages == ["process", "train", "export", "health", "mesh"]


def test_plan_guard_default_off(monkeypatch):
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: True)
    stages = ["train", "export"]
    splat_route._append_mesh_stage(stages, _req())
    assert "mesh" not in stages


def test_plan_guard_toolchain_unavailable(monkeypatch):
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: False)
    stages = ["train", "export"]
    splat_route._append_mesh_stage(stages, _req(mesh_export=True))
    assert "mesh" not in stages


def test_mesh_export_survives_meta_roundtrip():
    """Re-run/restart rebuilds the request from meta — mesh_export must persist."""
    req = _req(mesh_export=True)
    meta = splat_route._new_meta(
        "splat_me00006", req, Path("/in/clip.mp4"), Path("/out/splat_me00006"), ["train"]
    )
    assert meta["mesh_export"] is True
    rebuilt = splat_route._req_from_meta(meta)
    assert rebuilt is not None
    assert rebuilt.mesh_export is True
