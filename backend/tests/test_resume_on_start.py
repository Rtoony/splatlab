"""Resume-on-start: a service restart must AUTO-RESTART the newest orphaned
in-flight job (stages are self-cleaning) instead of just marking it failed —
but only when fresh, under the restart cap, with its input still on disk, and
never more than one (single-job GPU). Everything else keeps the honest failed
marker. CPU-only: _run_pipeline and planning are monkeypatched."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


def _mk_orphan(outputs: Path, job_id: str, *, status="running", age_hours=0.5,
               restart_count=0, input_path: Path | None = None) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path=str(input_path or "/in/clip.mp4"), output_dir="outputs/3d")
    meta = splat_route._new_meta(job_id, req, Path(req.input_path), job_dir, ["train"])
    born = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    meta.update({"status": status, "started_at": born, "created_at": born})
    if restart_count:
        meta["restart_count"] = restart_count
    (job_dir / "meta.json").write_text(json.dumps(meta))
    return job_dir


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    ran: list[str] = []

    async def fake_pipeline(job):
        ran.append(job.job_id)

    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "JOBS", {})
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: {})
    monkeypatch.setattr(splat_route, "_plan_3d_job",
                        lambda req, availability, job_dir, input_path: (["train"], {"train": ["true"]}, None))
    monkeypatch.setattr(splat_route, "_run_pipeline", fake_pipeline)
    monkeypatch.delenv("SPLAT_RESUME_ON_START", raising=False)
    return outputs, clip, ran


async def _resume_and_settle() -> int:
    n = await splat_route.resume_orphan_jobs()
    for job in splat_route.JOBS.values():
        if job.runner_task:
            await job.runner_task
    return n


def test_newest_orphan_restarts_older_marked_failed(env):
    outputs, clip, ran = env
    _mk_orphan(outputs, "splat_ee000001", age_hours=3.0, input_path=clip)
    _mk_orphan(outputs, "splat_ee000002", age_hours=0.2, input_path=clip)  # newest

    assert asyncio.run(_resume_and_settle()) == 1
    assert ran == ["splat_ee000002"]

    newest = json.loads((outputs / "splat_ee000002" / "meta.json").read_text())
    assert newest["status"] == "starting"
    assert newest["restart_count"] == 1
    assert newest["restarted_at"]
    older = json.loads((outputs / "splat_ee000001" / "meta.json").read_text())
    assert older["status"] == "failed"
    assert "restarted while job was active" in older["error_message"]


def test_kill_switch_restores_mark_failed_only(env, monkeypatch):
    outputs, clip, ran = env
    _mk_orphan(outputs, "splat_ee000003", input_path=clip)
    monkeypatch.setenv("SPLAT_RESUME_ON_START", "0")

    assert asyncio.run(_resume_and_settle()) == 0
    assert ran == []
    meta = json.loads((outputs / "splat_ee000003" / "meta.json").read_text())
    assert meta["status"] == "failed"


def test_restart_cap_prevents_crash_loops(env):
    outputs, clip, ran = env
    _mk_orphan(outputs, "splat_ee000004", input_path=clip,
               restart_count=splat_route.RESUME_MAX_RESTARTS)

    assert asyncio.run(_resume_and_settle()) == 0
    assert ran == []
    meta = json.loads((outputs / "splat_ee000004" / "meta.json").read_text())
    assert meta["status"] == "failed"


def test_stale_orphan_not_restarted(env):
    outputs, clip, ran = env
    _mk_orphan(outputs, "splat_ee000005", input_path=clip,
               age_hours=splat_route.RESUME_MAX_AGE_HOURS + 1)

    assert asyncio.run(_resume_and_settle()) == 0
    meta = json.loads((outputs / "splat_ee000005" / "meta.json").read_text())
    assert meta["status"] == "failed"


def test_missing_input_not_restarted(env, tmp_path):
    outputs, clip, ran = env
    _mk_orphan(outputs, "splat_ee000006", input_path=tmp_path / "gone.mp4")

    assert asyncio.run(_resume_and_settle()) == 0
    meta = json.loads((outputs / "splat_ee000006" / "meta.json").read_text())
    assert meta["status"] == "failed"


def test_completed_jobs_untouched(env):
    outputs, clip, ran = env
    _mk_orphan(outputs, "splat_ee000007", status="completed", input_path=clip)

    assert asyncio.run(_resume_and_settle()) == 0
    meta = json.loads((outputs / "splat_ee000007" / "meta.json").read_text())
    assert meta["status"] == "completed"