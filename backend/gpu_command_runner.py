"""Fail-closed CLI wrapper for sanctioned manual SplatLab GPU commands."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import sys
from pathlib import Path

import gpu_arbiter

DEFAULT_VRAM_MB = 24_000
STOP_GRACE_SECONDS = 10.0


async def _drain_task(task: asyncio.Task[object]) -> None:
    """Finish cleanup even if the parent task receives repeated cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
    task.result()


async def _terminate_subprocess(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    wait_task = asyncio.create_task(process.wait())
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=STOP_GRACE_SECONDS)
        return
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await asyncio.shield(wait_task)


async def _run_command(command: list[str]) -> int:
    process = await asyncio.create_subprocess_exec(*command, start_new_session=True)
    try:
        return await process.wait()
    except asyncio.CancelledError:
        cleanup = asyncio.create_task(_terminate_subprocess(process))
        await _drain_task(cleanup)
        raise


async def run_coordinated(command: list[str], vram_mb: int) -> int:
    async def operation() -> int:
        return await _run_command(command)

    return await gpu_arbiter.run_gpu_operation(
        lane="splat-manual",
        operation_id=f"manual:{os.getpid()}:{Path(command[0]).name}",
        vram_mb=vram_mb,
        operation=operation,
        status_callback=lambda detail: print(
            f"SplatLab GPU coordinator: {detail}", file=sys.stderr
        ),
    )


async def _run_with_signal_forwarding(command: list[str], vram_mb: int) -> int:
    loop = asyncio.get_running_loop()
    current = asyncio.current_task()
    received_signal: list[int] = []

    def cancel_for_signal(signum: int) -> None:
        if not received_signal:
            received_signal.append(signum)
        if current is not None:
            current.cancel()

    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, cancel_for_signal, int(signum))
            installed.append(signum)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        return await run_coordinated(command, vram_mb)
    except asyncio.CancelledError:
        if received_signal:
            return 128 + received_signal[0]
        raise
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vram-mb", type=int, default=DEFAULT_VRAM_MB)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("gpu_command_runner: command is required", file=sys.stderr)
        return 64
    if not 256 <= args.vram_mb <= 30_000:
        print(
            "gpu_command_runner: --vram-mb must be between 256 and 30000",
            file=sys.stderr,
        )
        return 64
    try:
        return_code = asyncio.run(_run_with_signal_forwarding(command, args.vram_mb))
        return 128 - return_code if return_code < 0 else return_code
    except gpu_arbiter.GPUArbiterUnavailable as exc:
        print(f"gpu_command_runner: coordination blocked: {exc}", file=sys.stderr)
        return 75
    except FileNotFoundError as exc:
        print(f"gpu_command_runner: command not found: {exc}", file=sys.stderr)
        return 127
    except PermissionError as exc:
        print(f"gpu_command_runner: command is not executable: {exc}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
