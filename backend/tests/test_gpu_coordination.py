"""Host-wide GPU serialization, failure policy, and cancellation safety."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_arbiter  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_coordination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(gpu_arbiter.HOST_LOCK_ENV, str(tmp_path / "heavy.lock"))
    monkeypatch.setattr(gpu_arbiter, "GPU_MAINTENANCE_REASON", "")
    monkeypatch.setattr(gpu_arbiter, "HOST_WORK_LOCK", gpu_arbiter._HostWorkLock())
    monkeypatch.setattr(gpu_arbiter, "_notify_degraded", lambda _detail: None)
    for name in (
        gpu_arbiter.EMERGENCY_OVERRIDE_ENV,
        gpu_arbiter.EMERGENCY_REASON_ENV,
        gpu_arbiter.EMERGENCY_ACTOR_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_redis_outage_fails_closed_and_releases_host_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = gpu_arbiter._CrossProcessLock()
    monkeypatch.setattr(gpu_arbiter, "_redis", lambda: None)

    async def acquire() -> None:
        async with lock:
            pytest.fail("Redis outage must not admit work")

    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="Redis"):
        asyncio.run(acquire())
    assert lock._local.locked() is False
    assert gpu_arbiter.HOST_WORK_LOCK._fd is None


def test_emergency_override_requires_reason_and_actor_and_is_audited(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(gpu_arbiter, "_redis", lambda: None)
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_OVERRIDE_ENV, "1")

    async def acquire(lock: gpu_arbiter._CrossProcessLock) -> None:
        async with lock:
            return None

    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable):
        asyncio.run(acquire(gpu_arbiter._CrossProcessLock()))

    monkeypatch.setenv(gpu_arbiter.EMERGENCY_ACTOR_ENV, "test-operator")
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_REASON_ENV, "test-only Redis outage")
    caplog.set_level(logging.CRITICAL)
    asyncio.run(acquire(gpu_arbiter._CrossProcessLock()))
    assert "GPU_EMERGENCY_OVERRIDE" in caplog.text
    assert "test-only Redis outage" in caplog.text


def test_orchestrator_outage_fails_closed_without_override(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def unavailable():
        return None

    monkeypatch.setattr(gpu_arbiter, "gpu_status", unavailable)
    ok, detail = asyncio.run(gpu_arbiter.acquire_gpu(4000))
    assert ok is False
    assert "refusing" in detail

    monkeypatch.setenv(gpu_arbiter.EMERGENCY_OVERRIDE_ENV, "1")
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_ACTOR_ENV, "test-operator")
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_REASON_ENV, "orchestrator recovery test")
    caplog.set_level(logging.CRITICAL)
    ok, detail = asyncio.run(gpu_arbiter.acquire_gpu(4000))
    assert ok is True
    assert "emergency override" in detail
    assert "orchestrator-unavailable" in caplog.text


def test_host_file_lock_serializes_independent_process_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_arbiter, "_redis", lambda: None)
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_OVERRIDE_ENV, "1")
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_ACTOR_ENV, "test-operator")
    monkeypatch.setenv(gpu_arbiter.EMERGENCY_REASON_ENV, "host-lock concurrency test")
    first = gpu_arbiter._CrossProcessLock()
    second = gpu_arbiter._CrossProcessLock()

    async def scenario() -> None:
        second_entered = asyncio.Event()

        async def take_second() -> None:
            async with second:
                second_entered.set()

        async with first:
            task = asyncio.create_task(take_second())
            await asyncio.sleep(0.05)
            assert second_entered.is_set() is False
        await asyncio.wait_for(task, timeout=1)
        assert second_entered.is_set() is True

    asyncio.run(scenario())


class _TrackingLock:
    def __init__(self) -> None:
        self.held = False
        self.lost = asyncio.Event()

    async def __aenter__(self):
        self.held = True
        return self

    async def __aexit__(self, *_exc):
        self.held = False

    async def wait_coordination_lost(self) -> None:
        await self.lost.wait()

    def locked(self) -> bool:
        return self.held


def _stub_runner_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    lock: _TrackingLock,
) -> None:
    async def admitted(_needed_mb: int):
        return True, "admitted"

    monkeypatch.setattr(gpu_arbiter, "HEAVY_GPU_LOCK", lock)
    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", lambda: None)
    monkeypatch.setattr(gpu_arbiter, "acquire_gpu", admitted)
    monkeypatch.setattr(gpu_arbiter, "set_holder", lambda *_args: None)
    monkeypatch.setattr(gpu_arbiter, "clear_holder", lambda: None)


def test_to_thread_cancellation_holds_lease_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _TrackingLock()
    _stub_runner_dependencies(monkeypatch, lock)
    started = threading.Event()
    release = threading.Event()

    def blocking_cuda_call() -> str:
        started.set()
        assert release.wait(timeout=2)
        assert lock.held is True
        return "finished"

    async def scenario() -> None:
        task = asyncio.create_task(
            gpu_arbiter.run_sync_gpu_operation(
                lane="test",
                operation_id="thread-cancel",
                vram_mb=1,
                func=blocking_cuda_call,
            )
        )
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0.05)
        assert task.done() is False
        assert lock.held is True
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lock.held is False

    asyncio.run(scenario())


def test_redis_heartbeat_loss_drains_operation_before_releasing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _TrackingLock()
    _stub_runner_dependencies(monkeypatch, lock)

    async def scenario() -> None:
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()

        async def operation() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await cleanup_release.wait()

        task = asyncio.create_task(
            gpu_arbiter.run_gpu_operation(
                lane="test",
                operation_id="heartbeat-loss",
                vram_mb=1,
                operation=operation,
            )
        )
        await started.wait()
        lock.lost.set()
        await cleanup_started.wait()
        assert lock.held is True
        assert task.done() is False
        cleanup_release.set()
        with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="coordination was lost"):
            await task
        assert lock.held is False

    asyncio.run(scenario())


def test_backup_admission_rejects_before_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran = False

    def blocked() -> None:
        raise gpu_arbiter.GPUArbiterUnavailable("backup is active")

    async def operation() -> None:
        nonlocal ran
        ran = True

    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", blocked)
    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="backup"):
        asyncio.run(
            gpu_arbiter.run_gpu_operation(
                lane="test",
                operation_id="backup",
                vram_mb=1,
                operation=operation,
            )
        )
    assert ran is False


def test_host_runner_waits_for_backup_flock_and_holds_it_through_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", lambda: None)
    path = tmp_path / "heavy.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    async def scenario() -> None:
        started = asyncio.Event()

        async def operation() -> None:
            assert gpu_arbiter.HOST_WORK_LOCK.locked() is True
            started.set()

        task = asyncio.create_task(gpu_arbiter.run_host_operation(operation=operation))
        await asyncio.sleep(0.05)
        assert started.is_set() is False
        fcntl.flock(fd, fcntl.LOCK_UN)
        await asyncio.wait_for(task, timeout=1)
        assert started.is_set() is True
        assert gpu_arbiter.HOST_WORK_LOCK._fd is None

    try:
        asyncio.run(scenario())
    finally:
        os.close(fd)


def test_host_runner_rechecks_backup_after_winning_flock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = 0
    ran = False

    def backup_check() -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise gpu_arbiter.GPUArbiterUnavailable("backup started")

    async def operation() -> None:
        nonlocal ran
        ran = True

    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", backup_check)
    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="backup started"):
        asyncio.run(gpu_arbiter.run_host_operation(operation=operation))
    assert checks == 2
    assert ran is False
    assert gpu_arbiter.HOST_WORK_LOCK._fd is None


def test_host_runner_cancellation_releases_flock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", lambda: None)

    async def scenario() -> None:
        started = asyncio.Event()

        async def operation() -> None:
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(gpu_arbiter.run_host_operation(operation=operation))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert gpu_arbiter.HOST_WORK_LOCK._fd is None

    asyncio.run(scenario())


def test_nested_host_lease_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu_arbiter, "require_backup_idle", lambda: None)

    async def scenario() -> None:
        async with gpu_arbiter.HOST_WORK_LOCK:
            with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="nested"):
                await gpu_arbiter.run_host_operation(operation=lambda: asyncio.sleep(0))

    asyncio.run(scenario())
