"""Optional-stage bookkeeping: langfield is a best-effort OPT-IN stage — its
failure must NEVER flip the overall job to "failed" (the splat itself already
succeeded independent of langfield). But the failure must still be visible
somewhere durable in job meta (stages_failed), not silently swallowed behind
an unconditional stages_completed append that makes it look identical to a
success.

CPU-only: no GPU, no real subprocess. _run_locked_stage is monkeypatched to
return a fake non-zero exit code (or raise), so only the _run_pipeline
bookkeeping around the langfield branch is exercised.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


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
    monkeypatch.setattr(splat_route, "_langfield_available", lambda: True)
    return outputs


def _run(job_id: str, job_dir: Path) -> None:
    job = splat_route.SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path="/in/clip.mp4",
        stages_planned=["langfield"],
        stage_commands={},
    )
    asyncio.run(splat_route._run_pipeline(job))


def test_langfield_nonzero_exit_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_lf00001"
    job_dir = _mk_job_dir(outputs, job_id, ["langfield"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        return 1  # simulated non-zero exit from the langfield subprocess

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    # Job semantics unchanged: the splat is done regardless of langfield.
    assert meta["status"] == "completed"
    assert meta["error_message"] is None
    # 2026-07-18: a failed optional stage no longer ALSO claims completion —
    # stages_failed is its record; the stage rail shows it untraversed.
    assert meta["stages_completed"] == []
    # The failure is durably visible.
    assert meta["stages_failed"] == [{"stage": "langfield", "reason": "exit code 1"}]

    # Visible end-to-end: _job_payload (the same **meta spread every API
    # response and the frontend's SplatJob type read) carries it too.
    payload = splat_route._job_payload(meta)
    assert payload["status"] == "completed"
    assert payload["stages_failed"] == [{"stage": "langfield", "reason": "exit code 1"}]


def test_langfield_exception_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_lf00002"
    job_dir = _mk_job_dir(outputs, job_id, ["langfield"])

    async def raising_run_locked_stage(job, stage, command, vram_mb):
        raise OSError("disk full")

    monkeypatch.setattr(splat_route, "_run_locked_stage", raising_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == []
    assert len(meta["stages_failed"]) == 1
    assert meta["stages_failed"][0]["stage"] == "langfield"
    assert "disk full" in meta["stages_failed"][0]["reason"]


def test_langfield_success_leaves_stages_failed_empty(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_lf00003"
    job_dir = _mk_job_dir(outputs, job_id, ["langfield"])

    async def fake_run_locked_stage(job, stage, command, vram_mb):
        return 0

    monkeypatch.setattr(splat_route, "_run_locked_stage", fake_run_locked_stage)

    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["langfield"]
    assert meta["stages_failed"] == []


def test_langfield_skipped_no_config_is_not_recorded_as_failure(tmp_path, monkeypatch):
    """A skip (no config / toolchain unavailable) is a normal no-op, not a
    failure — must NOT be recorded in stages_failed."""
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda job_dir: None)
    monkeypatch.setattr(splat_route, "_langfield_available", lambda: False)

    job_id = "splat_lf00004"
    job_dir = _mk_job_dir(outputs, job_id, ["langfield"])
    _run(job_id, job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["langfield"]
    assert meta["stages_failed"] == []
