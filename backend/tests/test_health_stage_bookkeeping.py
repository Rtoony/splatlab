"""Capture-health stage bookkeeping: health is a best-effort REPORT-ONLY stage —
its failure must NEVER flip the overall job to "failed", its verdict must land
durably in meta["health"], and the kill-switch/toolchain guards must control
whether it is planned at all.

CPU-only: no GPU, no real subprocess. _run_locked_stage is monkeypatched
(pattern copied from test_langfield_stage_bookkeeping.py).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

FAKE_FOG = {
    "v": 1,
    "verdict": "FOG",
    "checked_at": "2026-07-11T00:00:00+00:00",
    "runtime_s": 4.0,
    "cameras": [{"cam": 0, "counted": True, "shell_frac": 0.99, "fog": True, "healthy": False}],
    "summary": {"n_cams": 1, "n_counted": 1, "n_fog": 1, "n_healthy": 0,
                "median_shell_frac": 0.99, "median_spread": 1.0,
                "median_p50": 0.01, "median_acc": 1.0},
    "thresholds": {"shell_d": 0.03},
    "receipts": ["fog_cam000_spread1.0_p50-0.010.webp"],
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
    monkeypatch.setattr(splat_route, "_health_available", lambda: True)
    return outputs


def _run(job_id: str, job_dir: Path) -> None:
    job = splat_route.SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path="/in/clip.mp4",
        stages_planned=["health"],
        stage_commands={},
    )
    asyncio.run(splat_route._run_pipeline(job))


def test_health_nonzero_exit_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_hc00001"
    job_dir = _mk_job_dir(outputs, job_id, ["health"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        return 1  # simulated fog-gate crash

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["error_message"] is None
    assert meta["stages_completed"] == ["health"]
    assert meta["stages_failed"] == [{"stage": "health", "reason": "exit code 1"}]
    assert "health" not in meta or meta.get("health") is None or "fog" not in (meta.get("health") or {})


def test_health_success_persists_verdict_in_meta(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_hc00002"
    job_dir = _mk_job_dir(outputs, job_id, ["health"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        # The real gate writes fog.json into _health/ before exiting 0; the
        # runner branch created the dir before invoking us.
        (job_dir / splat_route.HEALTH_DIRNAME / "fog.json").write_text(json.dumps(FAKE_FOG))
        return 0

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["health"]
    assert meta["stages_failed"] == []
    fog = meta["health"]["fog"]
    assert fog["verdict"] == "FOG"
    assert fog["enforced"] is False  # report-only contract
    assert fog["receipts"] == FAKE_FOG["receipts"]

    # Visible end-to-end through the **meta spread the API + frontend read.
    payload = splat_route._job_payload(meta)
    assert payload["health"]["fog"]["verdict"] == "FOG"


def test_health_exception_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_hc00003"
    job_dir = _mk_job_dir(outputs, job_id, ["health"])

    async def raising_run_locked_stage(job, stage, command, vram_mb):
        raise OSError("disk full")

    monkeypatch.setattr(splat_route, "_run_locked_stage", raising_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["health"]
    assert len(meta["stages_failed"]) == 1
    assert meta["stages_failed"][0]["stage"] == "health"
    assert "disk full" in meta["stages_failed"][0]["reason"]


def test_health_skipped_no_config_is_not_recorded_as_failure(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda job_dir: None)
    monkeypatch.setattr(splat_route, "_health_available", lambda: False)

    job_id = "splat_hc00004"
    job_dir = _mk_job_dir(outputs, job_id, ["health"])
    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["health"]
    assert meta["stages_failed"] == []


# ── plan guard (_append_health_stage) ────────────────────────────────────────────


def test_plan_guard_appends_after_export(monkeypatch):
    monkeypatch.setattr(splat_route, "_health_available", lambda: True)
    monkeypatch.delenv("SPLAT_HEALTH_GATE", raising=False)
    stages = ["stitch", "process", "train", "export"]
    splat_route._append_health_stage(stages)
    assert stages == ["stitch", "process", "train", "export", "health"]
    assert stages.count("health") == 1


def test_plan_guard_kill_switch(monkeypatch):
    monkeypatch.setattr(splat_route, "_health_available", lambda: True)
    monkeypatch.setenv("SPLAT_HEALTH_GATE", "0")
    stages = ["train", "export"]
    splat_route._append_health_stage(stages)
    assert "health" not in stages


def test_plan_guard_toolchain_unavailable(monkeypatch):
    monkeypatch.setattr(splat_route, "_health_available", lambda: False)
    monkeypatch.delenv("SPLAT_HEALTH_GATE", raising=False)
    stages = ["train", "export"]
    splat_route._append_health_stage(stages)
    assert "health" not in stages
