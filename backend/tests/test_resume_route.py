"""Tests for POST /jobs/{id}/resume (born 2026-07-31: a job whose training
SUCCEEDED was killed at export by a backup collision; the only remedy was a
full 2.5h re-run)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_route  # noqa: E402


def _attempt5_corpse(tmp_path: Path) -> Path:
    """Replicate the exact artifact layout of splat_3e9cabeeac at death:
    stitch/rig_sfm/process/train all completed and on disk, export dead."""
    job = tmp_path / "splat_corpse"
    (job / "stitched").mkdir(parents=True)
    (job / "stitched" / "equirect.mp4").write_bytes(b"x" * 128)
    (job / "rig" / "sparse" / "0").mkdir(parents=True)
    (job / "rig" / "sparse" / "0" / "cameras.bin").write_bytes(b"x")
    (job / "processed").mkdir()
    (job / "processed" / "transforms.json").write_text("{}")
    # Real nerfstudio layout: output lands under processed/<experiment>/...
    run = job / "processed" / "splatfacto" / "2026-07-31_000000"
    (run / "nerfstudio_models").mkdir(parents=True)
    (run / "nerfstudio_models" / "step-000029999.ckpt").write_bytes(b"x")
    (run / "config.yml").write_text("x: 1")
    return job


PLANNED = ["stitch", "rig_sfm", "process", "train", "export", "health", "compress"]
COMPLETED = ["stitch", "rig_sfm", "process", "train"]


def test_prefix_skips_through_train_on_full_corpse(tmp_path: Path) -> None:
    job = _attempt5_corpse(tmp_path)
    prefix = splat_route._resume_verified_prefix(job, PLANNED, COMPLETED)
    assert prefix == ["stitch", "rig_sfm", "process", "train"]


def test_prefix_stops_at_first_missing_artifact(tmp_path: Path) -> None:
    """meta claims process completed but transforms.json is gone: the walk
    must stop there AND ignore the intact train artifact after the gap."""
    job = _attempt5_corpse(tmp_path)
    (job / "processed" / "transforms.json").unlink()
    prefix = splat_route._resume_verified_prefix(job, PLANNED, COMPLETED)
    assert prefix == ["stitch", "rig_sfm"]


def test_prefix_requires_meta_completion_not_just_artifact(tmp_path: Path) -> None:
    """An artifact on disk from a run whose meta does NOT record the stage
    as completed is not trusted (half-written outputs)."""
    job = _attempt5_corpse(tmp_path)
    prefix = splat_route._resume_verified_prefix(job, PLANNED, ["stitch"])
    assert prefix == ["stitch"]


def test_prefix_empty_when_nothing_verifies(tmp_path: Path) -> None:
    job = tmp_path / "empty_job"
    job.mkdir()
    assert splat_route._resume_verified_prefix(job, PLANNED, COMPLETED) == []


def test_zero_byte_stitch_not_trusted(tmp_path: Path) -> None:
    job = _attempt5_corpse(tmp_path)
    (job / "stitched" / "equirect.mp4").write_bytes(b"")
    assert splat_route._resume_verified_prefix(job, PLANNED, COMPLETED) == []


def test_unknown_stage_never_resumable(tmp_path: Path) -> None:
    job = _attempt5_corpse(tmp_path)
    prefix = splat_route._resume_verified_prefix(
        job, ["export", "train"], ["export", "train"]
    )
    assert prefix == []  # export is not artifact-verifiable; walk stops at once


def test_train_needs_both_checkpoint_and_config(tmp_path: Path) -> None:
    job = _attempt5_corpse(tmp_path)
    ckpt = (
        job / "processed" / "splatfacto" / "2026-07-31_000000"
        / "nerfstudio_models" / "step-000029999.ckpt"
    )
    ckpt.unlink()  # config without checkpoint = interrupted training, not trusted
    prefix = splat_route._resume_verified_prefix(job, PLANNED, COMPLETED)
    assert prefix == ["stitch", "rig_sfm", "process"]


def test_restart_job_resume_threads_skips_into_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_restart_job(resume=True) computes the prefix and pre-loads the
    SplatJob's resume_completed set; the runner is not actually started."""
    job_dir = _attempt5_corpse(tmp_path)
    job_id = "splat_corpse"
    meta = {
        "job_id": job_id,
        "stages_completed": COMPLETED,
        "created_at": "2026-07-31T00:00:00Z",
    }
    req = splat_route.SplatTrainRequest(mode="3d", input_path="/in/clip.insv")

    monkeypatch.setattr(splat_route, "_engine_availability", lambda: {"ns_train_path": "/bin/ns-train"})
    monkeypatch.setattr(splat_route, "_resolve_input_path", lambda p: Path(p))
    monkeypatch.setattr(splat_route, "_job_dir", lambda _jid: job_dir)
    monkeypatch.setattr(
        splat_route,
        "_plan_3d_job",
        lambda *_a, **_k: (list(PLANNED), {s: ["cmd"] for s in PLANNED}, None),
    )
    monkeypatch.setattr(splat_route, "_new_meta", lambda *a, **k: {"created_at": "x"})
    written: dict = {}
    monkeypatch.setattr(splat_route, "_write_meta", lambda _jid, fresh: written.update(fresh))

    class _FakeTask:
        def done(self) -> bool:
            return False

    monkeypatch.setattr(
        splat_route.asyncio, "create_task", lambda _coro: (_coro.close(), _FakeTask())[1]
    )

    skipped = splat_route._restart_job(meta, req, resume=True)
    assert skipped == COMPLETED
    assert splat_route.JOBS[job_id].resume_completed == set(COMPLETED)
    assert written["resumed_stages"] == COMPLETED
    del splat_route.JOBS[job_id]
