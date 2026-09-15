"""Held-out eval stage bookkeeping (2026-09-14): `eval` is a best-effort
REPORT-ONLY stage like health — its failure must never flip the job to failed,
its numbers must land durably in meta["health"]["eval"], and planning must obey
the toolchain guard + kill-switch. CPU-only: _run_locked_stage is monkeypatched
(pattern from test_health_stage_bookkeeping.py)."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

FAKE_EVAL = {"experiment_name": "processed", "method_name": "splatfacto", "checkpoint": "x.ckpt",
             "results": {"psnr": 31.55, "psnr_std": 2.1, "ssim": 0.936, "ssim_std": 0.02,
                         "lpips": 0.143, "lpips_std": 0.03, "num_rays_per_sec": 2.6e8, "fps": 163.0}}


def _mk_job_dir(outputs: Path, job_id: str, stages: list[str]) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    meta = splat_route._new_meta(
        job_id,
        splat_route.SplatTrainRequest(mode="3d", input_path="clip.mp4", output_dir="outputs/3d"),
        Path("/in/clip.mp4"), job_dir, stages)
    (job_dir / "meta.json").write_text(json.dumps(meta))
    return job_dir


@pytest.fixture()
def job_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda job_dir: job_dir / "config.yml")
    monkeypatch.setattr(splat_route, "_tool_path",
                        lambda binary, env_var: "/fake/bin/ns-eval" if binary == "ns-eval" else None)
    return outputs


def _run(job_id: str, job_dir: Path) -> None:
    job = splat_route.SplatJob(job_id=job_id, output_dir=str(job_dir), input_path="/in/clip.mp4",
                               stages_planned=["eval"], stage_commands={})
    asyncio.run(splat_route._run_pipeline(job))


def test_eval_success_persists_numbers_in_meta_health(job_env, monkeypatch):
    job_id = "splat_ev000001"
    job_dir = _mk_job_dir(job_env, job_id, ["eval"])
    seen = {}

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        seen["command"] = command; seen["vram"] = vram_mb
        Path(command[command.index("--output-path") + 1]).write_text(json.dumps(FAKE_EVAL))
        return 0

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)
    _run(job_id, job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed" and meta["stages_completed"] == ["eval"] and meta["stages_failed"] == []
    ev = meta["health"]["eval"]
    assert ev["psnr"] == 31.55 and ev["ssim"] == 0.936 and ev["lpips"] == 0.143 and ev["enforced"] is False
    assert ev["split"] == "eval" and ev["checkpoint"].endswith("config.yml")
    assert seen["command"][0] == "/fake/bin/ns-eval" and "--load-config" in seen["command"]
    assert seen["vram"] == splat_route.EVAL_VRAM_MB
    assert Path(seen["command"][-1]) == job_dir / splat_route.HEALTH_DIRNAME / "eval.json"
    assert splat_route._job_payload(meta)["health"]["eval"]["psnr"] == 31.55


def test_eval_keeps_existing_health_fog_record(job_env, monkeypatch):
    job_id = "splat_ev000002"
    job_dir = _mk_job_dir(job_env, job_id, ["eval"])
    splat_route._patch_meta(job_id, health={"v": 1, "fog": {"verdict": "HEALTHY"}})

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        Path(command[-1]).write_text(json.dumps(FAKE_EVAL)); return 0

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)
    _run(job_id, job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["health"]["fog"]["verdict"] == "HEALTHY" and meta["health"]["eval"]["psnr"] == 31.55


def test_eval_nonzero_exit_does_not_fail_job(job_env, monkeypatch):
    job_id = "splat_ev000003"
    job_dir = _mk_job_dir(job_env, job_id, ["eval"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        return 1

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)
    _run(job_id, job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed" and meta["error_message"] is None
    assert meta["stages_completed"] == [] and meta["stages_failed"] == [{"stage": "eval", "reason": "exit code 1"}]
    assert "eval" not in (meta.get("health") or {})


def test_eval_malformed_output_is_a_recorded_failure(job_env, monkeypatch):
    job_id = "splat_ev000004"
    job_dir = _mk_job_dir(job_env, job_id, ["eval"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        Path(command[-1]).write_text(json.dumps({"results": {}})); return 0

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)
    _run(job_id, job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed" and meta["stages_failed"][0]["stage"] == "eval"


def test_eval_exception_does_not_fail_job(job_env, monkeypatch):
    job_id = "splat_ev000005"
    job_dir = _mk_job_dir(job_env, job_id, ["eval"])

    async def raising(job, stage, command, vram_mb):
        raise OSError("disk full")

    monkeypatch.setattr(splat_route, "_run_locked_stage", raising)
    _run(job_id, job_dir)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed" and "disk full" in meta["stages_failed"][0]["reason"]


def test_eval_planned_only_with_toolchain_and_without_kill_switch(monkeypatch):
    monkeypatch.setattr(splat_route, "_health_available", lambda: False)
    monkeypatch.setattr(splat_route, "_tool_path", lambda b, e: "/fake/bin/ns-eval" if b == "ns-eval" else None)
    monkeypatch.delenv("SPLAT_EVAL_GATE", raising=False)
    stages: list[str] = []
    splat_route._append_health_stage(stages)
    assert stages == ["eval"]
    monkeypatch.setenv("SPLAT_EVAL_GATE", "0")
    stages = []
    splat_route._append_health_stage(stages)
    assert stages == []
    monkeypatch.delenv("SPLAT_EVAL_GATE")
    monkeypatch.setattr(splat_route, "_tool_path", lambda b, e: None)
    stages = []
    splat_route._append_health_stage(stages)
    assert stages == []
