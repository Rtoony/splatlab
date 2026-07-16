"""Every CUDA-capable SplatLab path must use the canonical runner."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_ops  # noqa: E402
import splat_route  # noqa: E402


@pytest.mark.parametrize(
    ("stage", "expected_vram"),
    [
        ("process", splat_route.PROCESS_VRAM_MB),
        ("reprocess1", splat_route.PROCESS_VRAM_MB),
        ("rig_sfm", splat_route.SFM_VRAM_MB),
        ("glomap_sfm", splat_route.SFM_VRAM_MB),
        ("export", splat_route.EXPORT_VRAM_MB),
    ],
)
def test_pipeline_dispatches_every_cuda_capable_stage_to_locked_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    expected_vram: int,
) -> None:
    job_id = "splat_deadbeef"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    metadata: dict = {
        "job_id": job_id,
        "status": "starting",
        "stages_completed": [],
        "stages_failed": [],
    }
    calls: list[tuple[str, int]] = []

    if stage.startswith("process") or stage.startswith("reprocess"):
        processed = job_dir / "processed"
        images = processed / "images"
        images.mkdir(parents=True)
        (images / "frame.jpg").write_bytes(b"image")
        (processed / "transforms.json").write_text(json.dumps({"frames": [{}]}))
    if stage == "export":
        config = job_dir / "config.yml"
        config.write_text("config")
        monkeypatch.setattr(splat_route, "_find_latest_config", lambda _path: config)
        monkeypatch.setattr(
            splat_route,
            "_engine_availability",
            lambda: {"ns_export_path": "/fake/ns-export"},
        )
        monkeypatch.setattr(
            splat_route,
            "_export_command",
            lambda *_args: ["/fake/ns-export"],
        )

    def patch_meta(_job_id: str, **fields):
        metadata.update(fields)
        return dict(metadata)

    async def locked(_job, locked_stage, _command, vram_mb):
        calls.append((locked_stage, vram_mb))
        return 0

    async def unlocked(*_args, **_kwargs):
        pytest.fail(f"{stage} bypassed the canonical GPU runner")

    async def audit(**_kwargs):
        return None

    monkeypatch.setattr(splat_route, "_patch_meta", patch_meta)
    monkeypatch.setattr(splat_route, "_read_meta", lambda _job_id: dict(metadata))
    monkeypatch.setattr(splat_route, "_run_locked_stage", locked)
    monkeypatch.setattr(splat_route, "_run_stage", unlocked)
    monkeypatch.setattr(splat_route, "_flush_log", lambda _job: None)
    monkeypatch.setattr(splat_route, "_prune_old_jobs", lambda: 0)
    monkeypatch.setattr(splat_route, "audit_operator_event", audit)

    job = splat_route.SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path=str(tmp_path / "input.mp4"),
        stages_planned=[stage],
        stage_commands={stage: ["/fake/command"]},
    )
    asyncio.run(splat_route._run_pipeline(job))

    assert calls == [(stage, expected_vram)]
    assert metadata["status"] == "completed"


def test_manual_preview_rejects_duplicate_inflight_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job_id = "splat_deadbeef"
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    config = job_dir / "config.yml"
    config.write_text("config")
    meta = {
        "job_id": job_id,
        "status": "completed",
        "mode": "3d",
        "output_dir": str(job_dir),
        "input_path": str(tmp_path / "input.mp4"),
    }
    started = asyncio.Event()
    release = asyncio.Event()

    async def captured(_command):
        started.set()
        await release.wait()
        preview = splat_route._preview_file_path(job_dir)
        preview.write_bytes(b"preview")
        return 0, b"", b""

    async def gpu_runner(**kwargs):
        return await kwargs["operation"]()

    async def audit(**_kwargs):
        return None

    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(
        splat_route,
        "_engine_availability",
        lambda: {"ns_export_available": True, "ns_export_path": "/fake/ns-export"},
    )
    monkeypatch.setattr(splat_route, "_read_meta", lambda _job_id: meta)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda _path: config)
    monkeypatch.setattr(splat_route, "_export_command", lambda *_args: ["/fake/ns-export"])
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", captured)
    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", gpu_runner)
    monkeypatch.setattr(splat_route, "audit_operator_event", audit)
    monkeypatch.setattr(splat_route, "_PREVIEW_EXPORT_LOCKS", {})

    async def scenario() -> None:
        first = asyncio.create_task(
            splat_route.generate_splat_preview(SimpleNamespace(), job_id)
        )
        await started.wait()
        with pytest.raises(HTTPException) as exc:
            await splat_route.generate_splat_preview(SimpleNamespace(), job_id)
        assert exc.value.status_code == 409
        assert "already running" in exc.value.detail
        release.set()
        result = await first
        assert result["job_id"] == job_id

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/splat/train", {"mode": "3d", "input_path": "/unused"}),
        ("post", "/api/splat/jobs/splat_deadbeef/preview", None),
        ("post", "/api/splat/jobs/splat_deadbeef/langfield/query", {"text": "chair"}),
        ("post", "/api/splat/jobs/splat_deadbeef/langfield/relevancy", {"text": "chair"}),
        (
            "post",
            "/api/splat/jobs/splat_deadbeef/langfield/select/sphere",
            {"center": [0, 0, 0], "radius": 1},
        ),
        ("post", "/api/splat/jobs/splat_deadbeef/langfield/overrides", {}),
        ("delete", "/api/splat/jobs/splat_deadbeef/langfield/overrides/deadbeef", None),
        ("get", "/api/splat/jobs/splat_deadbeef/langfield/inventory", None),
        ("post", "/api/splat/jobs/splat_deadbeef/edit/revert", {"version": 1}),
        (
            "post",
            "/api/splat/jobs/splat_deadbeef/edit/apply",
            {"ops": [{"type": "translate", "x": 1, "y": 0, "z": 0}]},
        ),
        (
            "post",
            "/api/splat/jobs/splat_deadbeef/edit/semantic",
            {"text": "chair", "mode": "delete", "cleanup": False},
        ),
        (
            "post",
            "/api/splat/edit/merge",
            {"job_ids": ["splat_deadbeef", "splat_feedface"], "name": "blocked"},
        ),
    ],
)
def test_every_heavy_route_rejects_active_backup(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    payload: dict | None,
) -> None:
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", "")
    monkeypatch.setattr(
        splat_route,
        "_backup_interlock_busy",
        lambda: (True, "restic-backup-core.service", "active"),
    )
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(edit_ops.router, prefix="/api/splat")
    response = TestClient(app).request(method, path, json=payload)
    assert response.status_code == 409, response.text
    assert "restic-backup-core.service is active" in response.json()["detail"]
