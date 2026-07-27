"""The operator audit trail, which used to be a no-op.

25 call sites across 8 modules awaited `audit_operator_event` with rich payloads
— job completions, deletions, edits, exports, scale changes, polish ingests —
and every one was assembled and discarded. These tests cover the trail that now
receives them, and in particular the two properties that make it safe to leave
in the hot path: it never raises, and it never records a secret.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import operator_audit  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_trail(tmp_path):
    operator_audit.configure_storage(tmp_path / "audit")
    yield
    operator_audit.configure_storage(
        Path(operator_audit.PROJECT_ROOT) / "data" / "audit")


def _emit(**kwargs):
    asyncio.run(operator_audit.audit_operator_event(**kwargs))


def _lines() -> list[dict]:
    path = operator_audit.log_path()
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_an_event_is_appended_as_one_json_line():
    _emit(title="Splat 3D job completed", description="garden -> splat_abc",
          variant="success", action="splat.train", target="3d",
          metadata={"job_id": "splat_abc", "status": "completed"})

    records = _lines()
    assert len(records) == 1
    assert records[0]["action"] == "splat.train"
    assert records[0]["target"] == "3d"
    assert records[0]["variant"] == "success"
    assert records[0]["metadata"]["job_id"] == "splat_abc"
    assert records[0]["at"].endswith("Z")


def test_events_append_rather_than_overwrite():
    for index in range(5):
        _emit(action="splat.delete", metadata={"job_id": f"splat_{index}"})
    assert len(_lines()) == 5


def test_the_file_is_rotated_by_day():
    _emit(action="splat.train")
    assert operator_audit.log_path().name.endswith(".jsonl")
    assert datetime.now(timezone.utc).strftime("%Y-%m-%d") in operator_audit.log_path().name


def test_the_stubs_call_signature_still_works():
    """Every existing call site passes request=None plus keywords."""
    _emit(request=None, title="t", description="d", variant="default",
          action="splat.scale", target="3d", metadata={"job_id": "splat_abc"})
    assert _lines()[0]["action"] == "splat.scale"


def test_unexpected_keywords_are_kept_not_dropped():
    _emit(action="splat.train", surprise="value")
    assert _lines()[0]["extra"]["surprise"] == "value"


# ---------------------------------------------------------------------------
# Never raises, never leaks
# ---------------------------------------------------------------------------

def test_an_unserialisable_payload_does_not_raise():
    """An audit trail that can kill the operation it records is worse than none."""
    _emit(action="splat.train", metadata={"handle": object()})

    records = _lines()
    assert len(records) == 1
    assert records[0]["action"] == "splat.train"


def test_an_unwritable_directory_does_not_raise(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    operator_audit.configure_storage(blocker / "audit")

    _emit(action="splat.train")           # must not raise

    assert operator_audit.read_events() == []


def test_an_oversized_event_is_replaced_by_a_marker_not_written_whole():
    _emit(action="splat.train", metadata={"blob": "x" * 200_000})

    records = _lines()
    assert len(records) == 1
    assert "error" in records[0]
    assert len(json.dumps(records[0])) < operator_audit.MAX_LINE_BYTES


def test_long_fields_are_clipped():
    _emit(action="splat.train", description="y" * 50_000)
    assert records_description_length() <= operator_audit.MAX_FIELD_CHARS + 20


def records_description_length() -> int:
    return len(_lines()[0]["description"])


class _FakeRequest:
    """Shaped like the Starlette request the call sites pass."""

    class _URL:
        path = "/api/splat/jobs/splat_abc/scale"

    class _Client:
        host = "127.0.0.1"

    url = _URL()
    client = _Client()
    headers = {"cookie": "splatlab_session=SECRET", "authorization": "Bearer SECRET"}
    cookies = {"splatlab_session": "SECRET"}


def test_a_request_contributes_only_path_and_client_never_credentials():
    """Same rule as the feedback module: no cookies, headers or bodies."""
    _emit(request=_FakeRequest(), action="splat.scale")

    raw = operator_audit.log_path().read_text()
    assert "SECRET" not in raw
    assert "cookie" not in raw.lower()
    record = _lines()[0]
    assert record["request"] == {"path": "/api/splat/jobs/splat_abc/scale",
                                 "client": "127.0.0.1"}


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------

def test_events_read_back_newest_first():
    for index in range(3):
        _emit(action="splat.train", metadata={"job_id": f"splat_{index}"})

    records = operator_audit.read_events()
    assert [r["metadata"]["job_id"] for r in records] == ["splat_2", "splat_1", "splat_0"]


def test_reads_filter_by_action_and_job():
    _emit(action="splat.train", metadata={"job_id": "splat_a"})
    _emit(action="splat.delete", metadata={"job_id": "splat_a"})
    _emit(action="splat.train", metadata={"job_id": "splat_b"})

    assert len(operator_audit.read_events(action="splat.train")) == 2
    assert len(operator_audit.read_events(job_id="splat_a")) == 2
    assert len(operator_audit.read_events(action="splat.train", job_id="splat_a")) == 1


def test_reads_respect_the_limit():
    for index in range(10):
        _emit(action="splat.train", metadata={"job_id": str(index)})
    assert len(operator_audit.read_events(limit=4)) == 4


def test_a_corrupt_line_is_skipped_not_fatal():
    _emit(action="splat.train", metadata={"job_id": "good"})
    with open(operator_audit.log_path(), "a") as handle:
        handle.write("{ this is not json\n")
    _emit(action="splat.train", metadata={"job_id": "also-good"})

    records = operator_audit.read_events()
    assert [r["metadata"]["job_id"] for r in records] == ["also-good", "good"]


def test_reading_an_absent_trail_is_empty_not_an_error():
    assert operator_audit.read_events() == []


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_prune_removes_only_files_past_the_window():
    _emit(action="splat.train")
    fresh = operator_audit.log_path()
    stale = operator_audit.DATA_DIR / "2020-01-01.jsonl"
    stale.write_text('{"action": "old"}\n')
    old_time = (datetime.now(timezone.utc) - timedelta(days=800)).timestamp()
    import os
    os.utime(stale, (old_time, old_time))

    removed = operator_audit.prune(retention_days=400)

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()
