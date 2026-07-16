"""Warm LangField CUDA work stays serialized for its complete operation."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import langfield_worker  # noqa: E402


def test_worker_startup_loads_siglip_through_gpu_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []
    loaded = False

    def load_siglip() -> None:
        nonlocal loaded
        loaded = True

    async def run_sync_gpu_operation(**kwargs):
        calls.append((kwargs["lane"], kwargs["operation_id"], kwargs["vram_mb"]))
        return kwargs["func"](*kwargs.get("args", ()))

    async def scenario() -> None:
        monkeypatch.setattr(langfield_worker, "WORKER_MAINTENANCE_REASON", "")
        monkeypatch.setattr(langfield_worker, "_HEAVY_OK", True)
        monkeypatch.setattr(langfield_worker.STATE, "load_siglip", load_siglip)
        monkeypatch.setattr(
            langfield_worker.gpu_arbiter,
            "run_sync_gpu_operation",
            run_sync_gpu_operation,
        )
        await langfield_worker._startup()
        await langfield_worker._shutdown()
        await asyncio.sleep(0)

    asyncio.run(scenario())
    assert loaded is True
    assert calls == [
        ("langfield-startup", "siglip", langfield_worker.QUERY_VRAM_MB)
    ]


def test_scene_load_and_override_composition_share_one_gpu_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    runner_calls: list[str] = []

    class FakeScene:
        def __init__(self, config_path: str, lfdir: str):
            events.append(f"scene:{config_path}:{lfdir}")

    def apply_overrides(_state, _scene, lfdir: str) -> None:
        events.append(f"overrides:{lfdir}")

    async def run_cuda(operation_id: str, func, *args):
        runner_calls.append(operation_id)
        return func(*args)

    monkeypatch.setattr(langfield_worker, "Scene", FakeScene)
    monkeypatch.setattr(langfield_worker, "_apply_paint_overrides", apply_overrides)
    monkeypatch.setattr(langfield_worker, "_run_cuda_sync", run_cuda)
    result = asyncio.run(
        langfield_worker._build_scene_locked("/config", "/field", "scene-1")
    )

    assert isinstance(result, FakeScene)
    assert runner_calls == ["scene-1:scene-load"]
    assert events == ["scene:/config:/field", "overrides:/field"]


def test_query_relevancy_and_render_share_one_gpu_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    cuda_operations: list[str] = []
    scene = object()

    def compute(_state, actual_scene, text: str):
        assert actual_scene is scene
        calls.append(f"compute:{text}")
        return "relevancy"

    def focus(actual_scene, relevancy):
        assert actual_scene is scene
        assert relevancy == "relevancy"
        calls.append("focus")
        return {"matches": ["match"]}

    def render(actual_scene, relevancy, matches, text: str):
        assert actual_scene is scene
        assert relevancy == "relevancy"
        assert matches == ["match"]
        calls.append(f"render:{text}")
        return ["heatmap.png"]

    async def run_cuda(operation_id: str, func, *args):
        cuda_operations.append(operation_id)
        return func(*args)

    monkeypatch.setattr(langfield_worker, "_compute_relevancy", compute)
    monkeypatch.setattr(langfield_worker, "_relevancy_focus", focus)
    monkeypatch.setattr(langfield_worker, "_render_match_thumbs_locked", render)
    monkeypatch.setattr(langfield_worker, "_run_cuda_sync", run_cuda)
    result = asyncio.run(langfield_worker._render_locked(scene, "chair", "scene-1"))

    assert result == ("heatmap.png", {"matches": ["match"]})
    assert cuda_operations == ["scene-1:query"]
    assert calls == ["compute:chair", "focus", "render:chair"]


def test_worker_contains_no_raw_to_thread_cuda_escape_hatch() -> None:
    source = inspect.getsource(langfield_worker)
    assert "asyncio.to_thread" not in source
