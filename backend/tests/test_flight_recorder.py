"""Tests for the GPU-coordination flight recorder (2026-07-31)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flight_recorder as fr  # noqa: E402


@pytest.fixture(autouse=True)
def clean_ring_and_spool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(fr.SPOOL_ENV, str(tmp_path / "spool"))
    with fr._beats_lock:
        fr._beats.clear()
    yield
    with fr._beats_lock:
        fr._beats.clear()


def test_heartbeat_ring_records_and_caps() -> None:
    for i in range(fr.HEARTBEAT_RING + 10):
        fr.record_heartbeat("tokentail", gap_secs=15.0 + i, refresh_secs=0.01)
    with fr._beats_lock:
        beats = list(fr._beats)
    assert len(beats) == fr.HEARTBEAT_RING  # ring, not unbounded
    assert beats[-1]["gap_secs"] == 15.0 + fr.HEARTBEAT_RING + 9
    assert beats[0]["token"] == "tokentail"


def test_snapshot_has_all_sections_even_when_probes_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fr, "_nvidia_snapshot", lambda: None)
    monkeypatch.setattr(fr, "_redis_ping_ms", lambda: None)
    snap = fr.snapshot("test-reason", "op-123")
    for key in (
        "at",
        "reason",
        "operation_id",
        "heartbeats",
        "psi",
        "cgroup",
        "loadavg",
        "gpu_compute_apps",
        "redis_ping_ms",
    ):
        assert key in snap
    assert snap["reason"] == "test-reason"
    assert snap["operation_id"] == "op-123"


def test_write_record_produces_readable_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fr, "_nvidia_snapshot", lambda: "pid, name, 123 MiB")
    fr.record_heartbeat("cadence1", 15.0, 0.02)
    path = fr.write_record("redis-coordination-lost", "splat_deadbeef")
    assert path is not None
    data = json.loads(Path(path).read_text())
    assert data["reason"] == "redis-coordination-lost"
    assert data["heartbeats"][0]["token"] == "cadence1"
    assert "splat_deadbeef" in Path(path).name


def test_write_record_never_raises_on_unwritable_spool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fr.SPOOL_ENV, "/proc/definitely/not/writable")
    assert fr.write_record("reason", "op") is None


def test_spool_pruned_to_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fr, "_nvidia_snapshot", lambda: None)
    spool = fr.spool_dir()
    spool.mkdir(parents=True, exist_ok=True)
    for i in range(fr.KEEP_RECORDS + 5):
        stale = spool / f"stale-{i:03d}.json"
        stale.write_text("{}")
        mtime = time.time() - 10_000 + i
        import os

        os.utime(stale, (mtime, mtime))
    path = fr.write_record("reason", "op-prune")
    assert path is not None
    remaining = list(spool.glob("*.json"))
    assert len(remaining) == fr.KEEP_RECORDS
    assert Path(path) in remaining  # newest survives


def test_operation_id_sanitized_for_filename() -> None:
    path = fr.write_record("reason", "../../etc/passwd !!")
    assert path is not None
    assert "/etc/" not in Path(path).name
    assert Path(path).parent == fr.spool_dir()
