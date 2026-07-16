from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_arbiter  # noqa: E402
import gpu_command_runner  # noqa: E402


def test_manual_command_uses_full_gpu_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    held = False
    calls: list[tuple[str, str, int]] = []

    async def command(argv: list[str]) -> int:
        assert held is True
        assert argv == ["/bin/true", "arg"]
        return 17

    async def runner(**kwargs):
        nonlocal held
        calls.append((kwargs["lane"], kwargs["operation_id"], kwargs["vram_mb"]))
        held = True
        try:
            return await kwargs["operation"]()
        finally:
            held = False

    monkeypatch.setattr(gpu_command_runner, "_run_command", command)
    monkeypatch.setattr(gpu_command_runner.gpu_arbiter, "run_gpu_operation", runner)
    result = asyncio.run(gpu_command_runner.run_coordinated(["/bin/true", "arg"], 9000))

    assert result == 17
    assert calls == [("splat-manual", f"manual:{os.getpid()}:true", 9000)]


def test_coordination_failure_never_spawns_command(monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked(**_kwargs):
        raise gpu_arbiter.GPUArbiterUnavailable("Redis unavailable")

    async def command(_argv: list[str]) -> int:
        pytest.fail("command must not start without coordination")

    monkeypatch.setattr(gpu_command_runner, "_run_command", command)
    monkeypatch.setattr(gpu_command_runner.gpu_arbiter, "run_gpu_operation", blocked)
    with pytest.raises(gpu_arbiter.GPUArbiterUnavailable, match="Redis"):
        asyncio.run(gpu_command_runner.run_coordinated(["/bin/true"], 9000))


def test_cli_maps_coordination_outage_to_fail_closed_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def blocked(_command: list[str], _vram_mb: int) -> int:
        raise gpu_arbiter.GPUArbiterUnavailable("orchestrator unavailable")

    monkeypatch.setattr(gpu_command_runner, "run_coordinated", blocked)
    assert gpu_command_runner.main(["--", "/bin/true"]) == 75


def test_cancelled_manual_command_is_reaped(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"

    async def scenario() -> int:
        task = asyncio.create_task(
            gpu_command_runner._run_command(
                ["/bin/sh", "-c", f"printf '%s' $$ > {pid_file}; exec sleep 30"]
            )
        )
        for _ in range(100):
            if pid_file.is_file():
                break
            await asyncio.sleep(0.01)
        assert pid_file.is_file()
        pid = int(pid_file.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(scenario())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(("argv", "expected"), [([], 64), (["--vram-mb", "1", "--", "true"], 64)])
def test_cli_rejects_missing_command_and_unsafe_vram(argv: list[str], expected: int) -> None:
    assert gpu_command_runner.main(argv) == expected
