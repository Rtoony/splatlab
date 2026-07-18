"""Hardware maintenance must leave browsing online while all GPU work fails closed."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import edit_ops  # noqa: E402
import gpu_arbiter  # noqa: E402
import langfield_worker  # noqa: E402
import maintenance_gate  # noqa: E402
import main as splat_main  # noqa: E402
import splat_route  # noqa: E402


REASON = "Persistent RTX 5090 PCIe AER requires physical remediation."
JOB_ID = "splat_deadbeef"


def _write_supervised_unlock(path: Path, **overrides) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = {
        "schema": maintenance_gate.UNLOCK_SCHEMA,
        "enabled": True,
        "mode": "supervised",
        "reason": "supervised test window",
        "operator": "pytest",
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "expires_at": (now + timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
        "max_active_jobs": 1,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


def _write_watcher_status(path: Path, **overrides) -> None:
    payload = {
        "run_success": True,
        "finished_at_epoch": time.time(),
        "fault_counts": {
            "aer_current": 0,
            "aer_previous": 1,
            "aer_severe": 0,
            "gpu_unreadable": 0,
            "platform_fatal": 0,
            "xid": 0,
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload))


@pytest.fixture()
def gated_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", REASON)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(edit_ops.router, prefix="/api/splat")
    return TestClient(app)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/splat/train", {"mode": "3d", "input_path": "/nonexistent/input.mp4"}),
        ("post", f"/api/splat/jobs/{JOB_ID}/preview", None),
        ("post", f"/api/splat/jobs/{JOB_ID}/langfield/query", {"text": "chair"}),
        ("post", f"/api/splat/jobs/{JOB_ID}/langfield/relevancy", {"text": "chair"}),
        (
            "post",
            f"/api/splat/jobs/{JOB_ID}/langfield/select/sphere",
            {"center": [0, 0, 0], "radius": 1},
        ),
        ("post", f"/api/splat/jobs/{JOB_ID}/langfield/overrides", {}),
        ("delete", f"/api/splat/jobs/{JOB_ID}/langfield/overrides/deadbeef", None),
        ("get", f"/api/splat/jobs/{JOB_ID}/langfield/inventory", None),
        ("post", f"/api/splat/jobs/{JOB_ID}/edit/revert", {"version": 1}),
        (
            "post",
            f"/api/splat/jobs/{JOB_ID}/edit/apply",
            {"ops": [{"type": "translate", "x": 1, "y": 0, "z": 0}]},
        ),
        (
            "post",
            f"/api/splat/jobs/{JOB_ID}/edit/semantic",
            {"text": "chair", "mode": "delete", "cleanup": False},
        ),
        (
            "post",
            "/api/splat/edit/merge",
            {"job_ids": [JOB_ID, "splat_feedface"], "name": "blocked merge"},
        ),
    ],
)
def test_compute_and_gpu_mutation_routes_are_gated(
    gated_client: TestClient,
    method: str,
    path: str,
    payload: dict | None,
) -> None:
    response = gated_client.request(method, path, json=payload)
    assert response.status_code == 409, response.text
    assert REASON in response.json()["detail"]


@pytest.mark.parametrize(
    ("helper", "args"),
    [
        (splat_route._langfield_worker_query, ("/config", "/field", "chair")),
        (splat_route._langfield_worker_relevancy, ("/config", "/field", "chair")),
        (splat_route._langfield_worker_json, ("/select_sphere", {})),
        (splat_route._langfield_worker_inventory, ("/config", "/field")),
        (splat_route._langfield_query_cold, (JOB_ID, "/config", "/field", "chair")),
        (edit_ops._run_splat_transform, (["must-not-run"], False, JOB_ID)),
    ],
)
def test_internal_compute_helpers_are_gated(
    monkeypatch: pytest.MonkeyPatch,
    helper,
    args: tuple,
) -> None:
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", REASON)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(helper(*args))
    assert exc.value.status_code == 409
    assert REASON in exc.value.detail


def test_read_only_routes_and_cpu_thumbnail_fallback_remain_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gated_client: TestClient,
) -> None:
    outputs = tmp_path / "outputs"
    job_dir = outputs / JOB_ID
    preview = job_dir / "_preview"
    preview.mkdir(parents=True)
    (preview / "splat.ply").write_bytes(b"safe-read-only-preview")
    (job_dir / "_langfield").mkdir()
    (job_dir / "_langfield" / "gauss_emb.npz").write_bytes(b"field")

    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: {})
    monkeypatch.setattr(splat_route, "_sample_media_entries", lambda: [])
    monkeypatch.setattr(gpu_arbiter, "holder_info", lambda: {"locked": False})

    status = gated_client.get("/api/splat/status")
    assert status.status_code == 200
    assert status.json()["compute"]["enabled"] is False
    assert REASON in status.json()["compute"]["reason"]
    assert "New splat generation" in status.json()["compute"]["blocked_capabilities"]
    assert gated_client.get(f"/api/splat/jobs/{JOB_ID}/preview/file").status_code == 200
    assert gated_client.get(f"/api/splat/jobs/{JOB_ID}/langfield/overrides").status_code == 200
    assert gated_client.get(f"/api/splat/jobs/{JOB_ID}/edit/versions").status_code == 200
    assert asyncio.run(splat_main.healthz())["ok"] is True

    assert asyncio.run(splat_route.ensure_hero_thumb(job_dir)) is None


def test_supervised_unlock_admits_compute_with_marker_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = maintenance_gate.MAINTENANCE_FILE
    unlock = maintenance_gate.SUPERVISED_UNLOCK_FILE
    marker.write_text(f"{maintenance_gate.REASON_KEY}={REASON}\n")
    _write_supervised_unlock(unlock)
    _write_watcher_status(maintenance_gate.WATCHER_STATUS_FILE)
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", REASON)

    splat_route.require_compute_enabled()
    payload = splat_route._compute_status_payload()
    assert payload["enabled"] is True
    assert payload["mode"] == "supervised"
    assert payload["supervised_unlock"]["active"] is True
    assert payload["supervised_unlock"]["max_active_jobs"] == 1
    assert payload["supervised_unlock"]["watcher"]["ok"] is True
    assert "Run LangField search queries" in payload["safe_capabilities"]
    assert "Run bounded mesh/autoresearch trials" in payload["safe_capabilities"]
    assert "Background mesh autoresearch" not in payload["blocked_capabilities"]
    assert "Concurrent splat jobs" in payload["blocked_capabilities"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"mode": "normal"},
        {"max_active_jobs": 2},
        {"expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()},
        {"expires_at": (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()},
    ],
)
def test_invalid_supervised_unlock_keeps_marker_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
) -> None:
    maintenance_gate.MAINTENANCE_FILE.write_text(f"{maintenance_gate.REASON_KEY}={REASON}\n")
    _write_supervised_unlock(maintenance_gate.SUPERVISED_UNLOCK_FILE, **overrides)
    _write_watcher_status(maintenance_gate.WATCHER_STATUS_FILE)
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", REASON)

    with pytest.raises(HTTPException) as exc:
        splat_route.require_compute_enabled()
    assert exc.value.status_code == 409
    assert REASON in exc.value.detail


@pytest.mark.parametrize(
    "watcher",
    [
        {"run_success": False},
        {"finished_at_epoch": time.time() - maintenance_gate.WATCHER_MAX_AGE_SECONDS - 1},
        {"fault_counts": {"aer_current": 1}},
        {"fault_counts": {"xid": 1}},
    ],
)
def test_supervised_unlock_requires_fresh_clean_watcher(
    monkeypatch: pytest.MonkeyPatch,
    watcher: dict,
) -> None:
    maintenance_gate.MAINTENANCE_FILE.write_text(f"{maintenance_gate.REASON_KEY}={REASON}\n")
    _write_supervised_unlock(maintenance_gate.SUPERVISED_UNLOCK_FILE)
    _write_watcher_status(maintenance_gate.WATCHER_STATUS_FILE, **watcher)
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", REASON)

    with pytest.raises(HTTPException) as exc:
        splat_route.require_compute_enabled()
    assert exc.value.status_code == 409
    assert REASON in exc.value.detail


@pytest.mark.parametrize(("reason", "expected"), [("", 0), (REASON, 75)])
def test_worker_execstart_guard(tmp_path: Path, reason: str, expected: int) -> None:
    script = Path(__file__).resolve().parents[2] / "tools" / "splatlab-compute-gate.sh"
    marker = tmp_path / "maintenance.conf"
    if reason:
        marker.write_text(f"SPLAT_TRAINING_DISABLED_REASON={reason}\n")
    result = subprocess.run(
        [
            "/usr/bin/bash",
            "-c",
            'source "$1"; check_gate "$2"',
            "splatlab-compute-gate-test",
            str(script),
            str(marker),
        ],
        env={
            **os.environ,
            "SPLAT_TRAINING_DISABLED_REASON": "",
            "SPLAT_COMPUTE_UNLOCK_FILE": str(tmp_path / "absent-unlock.json"),
            "SPLAT_GPU_WATCHER_STATUS_FILE": str(tmp_path / "absent-watcher.json"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    if reason:
        assert reason in result.stderr


def test_worker_application_startup_is_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langfield_worker, "WORKER_MAINTENANCE_REASON", REASON)
    monkeypatch.setattr(
        langfield_worker.STATE,
        "load_siglip",
        lambda: pytest.fail("SigLIP must not load during maintenance"),
    )
    with pytest.raises(RuntimeError, match="hardware maintenance"):
        asyncio.run(langfield_worker._startup())


@pytest.mark.parametrize("status", [None, {"vram_free_mb": 32_000, "services": []}])
def test_gpu_acquire_refuses_maintenance_even_with_arbiter_state(
    monkeypatch: pytest.MonkeyPatch,
    status: dict | None,
) -> None:
    async def fake_status() -> dict | None:
        return status

    monkeypatch.setattr(gpu_arbiter, "GPU_MAINTENANCE_REASON", REASON)
    monkeypatch.setattr(gpu_arbiter, "gpu_status", fake_status)
    ok, detail = asyncio.run(gpu_arbiter.acquire_gpu(1))
    assert ok is False
    assert REASON in detail


@pytest.mark.parametrize("reason", [REASON, ""])
def test_redis_outage_is_always_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    lock = gpu_arbiter._CrossProcessLock()
    monkeypatch.setenv(gpu_arbiter.HOST_LOCK_ENV, str(tmp_path / "heavy.lock"))
    monkeypatch.setattr(gpu_arbiter, "GPU_MAINTENANCE_REASON", reason)
    monkeypatch.setattr(gpu_arbiter, "_redis", lambda: None)

    async def acquire() -> None:
        async with lock:
            return None

    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable):
        asyncio.run(acquire())
    assert lock.locked() is False


def test_marker_activation_after_import_blocks_backend_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", "")
    monkeypatch.delenv(maintenance_gate.REASON_KEY, raising=False)
    marker = maintenance_gate.MAINTENANCE_FILE

    splat_route.require_compute_enabled()
    marker.write_text(f'{maintenance_gate.REASON_KEY}="dynamic hardware hold"\n')

    with pytest.raises(HTTPException) as exc:
        splat_route.require_compute_enabled()
    assert exc.value.status_code == 409
    assert "dynamic hardware hold" in exc.value.detail


@pytest.mark.parametrize("kind", ["malformed", "dangling"])
def test_malformed_or_dangling_dynamic_marker_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    marker = tmp_path / "maintenance.conf"
    monkeypatch.setattr(maintenance_gate, "MAINTENANCE_FILE", marker)
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", "")
    if kind == "malformed":
        marker.write_text("# missing reason assignment\n")
    else:
        marker.symlink_to(tmp_path / "missing-target")

    with pytest.raises(HTTPException) as exc:
        splat_route.require_compute_enabled()
    assert exc.value.status_code == 409
    assert maintenance_gate.DEFAULT_REASON in exc.value.detail


def test_marker_activation_between_pipeline_stages_blocks_next_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = maintenance_gate.MAINTENANCE_FILE
    monkeypatch.setattr(splat_route, "TRAINING_DISABLED_REASON", "")
    metadata: dict = {"status": "starting", "stages_completed": []}
    calls: list[str] = []

    def patch_meta(_job_id: str, **fields):
        metadata.update(fields)
        return dict(metadata)

    async def run_stage(_job, stage: str, _command):
        calls.append(stage)
        marker.write_text(f"{maintenance_gate.REASON_KEY}=activated between stages\n")
        return 0

    async def audit(**_kwargs):
        return None

    monkeypatch.setattr(splat_route, "_patch_meta", patch_meta)
    monkeypatch.setattr(splat_route, "_read_meta", lambda _job_id: dict(metadata))
    monkeypatch.setattr(splat_route, "_run_stage", run_stage)
    monkeypatch.setattr(splat_route, "_flush_log", lambda _job: None)
    monkeypatch.setattr(splat_route, "_prune_old_jobs", lambda: 0)
    monkeypatch.setattr(splat_route, "audit_operator_event", audit)
    job = splat_route.SplatJob(
        job_id=JOB_ID,
        output_dir="/tmp/unused",
        input_path="/tmp/unused.mp4",
        stages_planned=["cpu-one", "cpu-two"],
        stage_commands={"cpu-one": ["true"], "cpu-two": ["true"]},
    )

    asyncio.run(splat_route._run_pipeline(job))

    assert calls == ["cpu-one"]
    assert metadata["status"] == "failed"
    assert "activated between stages" in metadata["error_message"]


def test_gpu_lock_reads_marker_dynamically_before_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_arbiter, "GPU_MAINTENANCE_REASON", "")
    maintenance_gate.MAINTENANCE_FILE.write_text(
        f"{maintenance_gate.REASON_KEY}=late GPU hold\n"
    )
    monkeypatch.setattr(
        gpu_arbiter,
        "_redis",
        lambda: pytest.fail("maintenance must reject before Redis"),
    )

    async def acquire() -> None:
        async with gpu_arbiter._CrossProcessLock():
            pytest.fail("dynamic marker must block lock admission")

    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="late GPU hold"):
        asyncio.run(acquire())


def test_manual_gate_routes_run_through_python_coordinator() -> None:
    gate = Path(__file__).resolve().parents[2] / "tools" / "splatlab-compute-gate.sh"
    source = gate.read_text()
    assert "backend/gpu_command_runner.py" in source
    assert '"$COORDINATOR" --vram-mb "$MANUAL_VRAM_MB" -- "$@"' in source
