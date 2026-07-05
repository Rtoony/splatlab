"""Optional-stage bookkeeping for compress/webopt: both are best-effort
post-processing stages — a failure must NEVER flip the overall job to
"failed" (the splat itself already succeeded independent of them). Mirrors
test_langfield_stage_bookkeeping.py, which fixed the identical bug for
langfield and left compress/webopt explicitly out of scope at the time.

CPU-only: no GPU, no real subprocess. _run_stage is monkeypatched to return
a fake exit code, so only the _run_pipeline bookkeeping around the
compress/webopt branches is exercised.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


def _mk_job_dir(outputs: Path, job_id: str, stages: list[str], with_ply: bool = True) -> Path:
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
    if with_ply:
        ply = splat_route._preview_file_path(job_dir)
        ply.parent.mkdir(parents=True, exist_ok=True)
        ply.write_bytes(b"fake ply")
    return job_dir


@pytest.fixture()
def job_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: "/bin/splat-transform")
    return outputs


def _run(job_id: str, job_dir: Path, stages: list[str]) -> None:
    job = splat_route.SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path="/in/clip.mp4",
        stages_planned=stages,
        stage_commands={},
    )
    asyncio.run(splat_route._run_pipeline(job))


def test_compress_nonzero_exit_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_cw00001"
    job_dir = _mk_job_dir(outputs, job_id, ["compress"])

    async def fake_run_stage(job, stage, command):
        return 1

    monkeypatch.setattr(splat_route, "_run_stage", fake_run_stage)
    _run(job_id, job_dir, ["compress"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["error_message"] is None
    assert meta["stages_completed"] == ["compress"]
    assert meta["stages_failed"] == [{"stage": "compress", "reason": "exit code 1"}]

    payload = splat_route._job_payload(meta)
    assert payload["stages_failed"] == [{"stage": "compress", "reason": "exit code 1"}]


def test_compress_success_leaves_stages_failed_empty(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_cw00002"
    job_dir = _mk_job_dir(outputs, job_id, ["compress"])

    async def fake_run_stage(job, stage, command):
        return 0

    monkeypatch.setattr(splat_route, "_run_stage", fake_run_stage)
    _run(job_id, job_dir, ["compress"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["stages_completed"] == ["compress"]
    assert meta["stages_failed"] == []


def test_compress_skipped_no_ply_is_not_recorded_as_failure(tmp_path, monkeypatch):
    """A skip (tool or .ply unavailable) is a normal no-op, not a failure."""
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: None)

    job_id = "splat_cw00003"
    job_dir = _mk_job_dir(outputs, job_id, ["compress"], with_ply=False)
    _run(job_id, job_dir, ["compress"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["stages_completed"] == ["compress"]
    assert meta["stages_failed"] == []


def test_webopt_nonzero_exit_does_not_fail_job(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_cw00004"
    job_dir = _mk_job_dir(outputs, job_id, ["webopt"])

    async def fake_run_stage(job, stage, command):
        return 1

    monkeypatch.setattr(splat_route, "_run_stage", fake_run_stage)
    _run(job_id, job_dir, ["webopt"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["stages_completed"] == ["webopt"]
    assert meta["stages_failed"] == [{"stage": "webopt", "reason": "exit code 1"}]


def test_webopt_success_leaves_stages_failed_empty(job_env, monkeypatch):
    outputs = job_env
    job_id = "splat_cw00005"
    job_dir = _mk_job_dir(outputs, job_id, ["webopt"])

    async def fake_run_stage(job, stage, command):
        return 0

    monkeypatch.setattr(splat_route, "_run_stage", fake_run_stage)
    _run(job_id, job_dir, ["webopt"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["stages_completed"] == ["webopt"]
    assert meta["stages_failed"] == []


def test_webopt_skipped_no_ply_is_not_recorded_as_failure(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: None)

    job_id = "splat_cw00006"
    job_dir = _mk_job_dir(outputs, job_id, ["webopt"], with_ply=False)
    _run(job_id, job_dir, ["webopt"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["stages_completed"] == ["webopt"]
    assert meta["stages_failed"] == []


def test_webopt_langweb_failure_recorded_distinctly(job_env, monkeypatch):
    """webopt succeeds but its langweb sub-run (built only when langfield is
    planned) fails independently — must be recorded under its own stage name,
    not conflated with the main webopt receipt."""
    outputs = job_env
    job_id = "splat_cw00007"
    job_dir = _mk_job_dir(outputs, job_id, ["webopt", "langfield"])

    async def fake_run_stage(job, stage, command):
        return 1 if stage == "webopt-langweb" else 0

    monkeypatch.setattr(splat_route, "_run_stage", fake_run_stage)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda job_dir: None)
    monkeypatch.setattr(splat_route, "_langfield_available", lambda: False)
    _run(job_id, job_dir, ["webopt", "langfield"])

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["stages_completed"] == ["webopt", "langfield"]
    assert meta["stages_failed"] == [{"stage": "webopt-langweb", "reason": "exit code 1"}]
