"""Tests for the persistent heavy-operation registry.

The contract that matters most here is not "can it store a row" — it is that
persistence must not cost the truthfulness `activity_route` used to get for
free. In-process locks died with the backend, and a restart also killed the
work they guarded, so a fresh process correctly reported nothing in flight. A
SQLite row does not die, so a `running` row from a dead process must never be
reported as running. Most of this file is about that.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opregistry  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path):
    opregistry.configure_storage(tmp_path / "ops")
    opregistry.init_db()
    yield
    opregistry.configure_storage(
        Path(opregistry.PROJECT_ROOT) / "data" / "ops")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_records_a_running_operation():
    op_id = opregistry.start("mesh", "splat_abc", step="poisson", detail="8k faces")
    record = opregistry.get(op_id)

    assert record["kind"] == "mesh"
    assert record["job_id"] == "splat_abc"
    assert record["status"] == opregistry.RUNNING
    assert record["active"] is True
    assert record["step"] == "poisson"
    assert record["detail"] == "8k faces"
    assert record["finished_at"] is None
    assert record["started_at"] and record["updated_at"]


def test_update_advances_step_and_progress():
    op_id = opregistry.start("world_solidify", "splat_abc")
    opregistry.update(op_id, step="shell", progress=0.4, detail="voxelising")
    record = opregistry.get(op_id)

    assert record["step"] == "shell"
    assert record["progress"] == pytest.approx(0.4)
    assert record["detail"] == "voxelising"


def test_progress_is_clamped_to_the_unit_interval():
    op_id = opregistry.start("mesh", "j")
    opregistry.update(op_id, progress=3.5)
    assert opregistry.get(op_id)["progress"] == pytest.approx(1.0)
    opregistry.update(op_id, progress=-2.0)
    assert opregistry.get(op_id)["progress"] == pytest.approx(0.0)


def test_update_of_an_unknown_id_is_a_no_op():
    """A progress callback must never be able to kill the work it reports on."""
    opregistry.update("does-not-exist", step="x", progress=0.5)


def test_update_cannot_reopen_a_finished_operation():
    op_id = opregistry.start("mesh", "j")
    opregistry.finish(op_id)
    opregistry.update(op_id, step="sneaky")

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.SUCCEEDED
    assert record["step"] != "sneaky"


def test_finish_records_result_and_completes_progress():
    op_id = opregistry.start("mesh", "splat_abc")
    opregistry.finish(op_id, result={"faces": 8000, "path": "_mesh/mesh.glb"})

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.SUCCEEDED
    assert record["active"] is False
    assert record["result"] == {"faces": 8000, "path": "_mesh/mesh.glb"}
    assert record["progress"] == pytest.approx(1.0)
    assert record["finished_at"]


def test_finish_records_failure_without_forcing_progress():
    op_id = opregistry.start("mesh", "splat_abc")
    opregistry.update(op_id, progress=0.3)
    opregistry.finish(op_id, status=opregistry.FAILED, error="xatlas did not converge")

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.FAILED
    assert "xatlas" in record["error"]
    assert record["progress"] == pytest.approx(0.3), "a failure must not read as complete"


def test_finish_rejects_a_non_terminal_status():
    op_id = opregistry.start("mesh", "j")
    with pytest.raises(ValueError, match="not a terminal"):
        opregistry.finish(op_id, status="running")


def test_unserialisable_result_does_not_break_finish():
    op_id = opregistry.start("mesh", "j")
    opregistry.finish(op_id, result={"path": object()})

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.SUCCEEDED
    assert record["result"] is None


# ---------------------------------------------------------------------------
# operation() context manager
# ---------------------------------------------------------------------------

def test_operation_marks_success_on_a_clean_exit():
    with opregistry.operation("export", "splat_abc") as op_id:
        opregistry.update(op_id, step="spz")
    assert opregistry.get(op_id)["status"] == opregistry.SUCCEEDED


def test_operation_records_failure_and_reraises():
    with pytest.raises(RuntimeError, match="boom"):
        with opregistry.operation("export", "splat_abc") as op_id:
            raise RuntimeError("boom")

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.FAILED
    assert "RuntimeError: boom" in record["error"]


def test_operation_does_not_swallow_cancellation():
    with pytest.raises(KeyboardInterrupt):
        with opregistry.operation("export", "j") as op_id:
            raise KeyboardInterrupt
    assert opregistry.get(op_id)["status"] == opregistry.FAILED


def test_operation_respects_an_explicit_finish_inside_the_block():
    with opregistry.operation("mesh", "j") as op_id:
        opregistry.finish(op_id, status=opregistry.CANCELLED, error="user stopped it")
    assert opregistry.get(op_id)["status"] == opregistry.CANCELLED


# ---------------------------------------------------------------------------
# The truthfulness rail
# ---------------------------------------------------------------------------

def _simulate_restart(monkeypatch):
    """A new backend process = a new runtime id."""
    monkeypatch.setattr(opregistry, "RUNTIME_ID", "restarted-process-runtime-id")


def test_a_running_row_from_a_dead_process_reads_as_abandoned(monkeypatch):
    op_id = opregistry.start("mesh", "splat_abc")
    _simulate_restart(monkeypatch)

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.ABANDONED
    assert record["active"] is False
    assert "exited before it finished" in record["error"]


def test_abandoned_work_never_appears_as_active(monkeypatch):
    opregistry.start("mesh", "splat_abc")
    _simulate_restart(monkeypatch)

    assert opregistry.active_by_job() == {}
    assert opregistry.list_ops(active_only=True) == []


def test_reconcile_orphans_corrects_the_stored_rows(monkeypatch):
    first = opregistry.start("mesh", "splat_abc")
    second = opregistry.start("export", "splat_def")
    done = opregistry.start("mesh", "splat_ghi")
    opregistry.finish(done)

    _simulate_restart(monkeypatch)
    corrected = opregistry.reconcile_orphans()

    assert corrected == 2
    assert opregistry.get(first)["status"] == opregistry.ABANDONED
    assert opregistry.get(second)["status"] == opregistry.ABANDONED
    assert opregistry.get(done)["status"] == opregistry.SUCCEEDED, "finished work is history"
    assert opregistry.get(first)["finished_at"]


def test_reconcile_is_idempotent_and_leaves_live_work_alone(monkeypatch):
    _simulate_restart(monkeypatch)
    live = opregistry.start("mesh", "splat_live")     # started by THIS runtime

    assert opregistry.reconcile_orphans() == 0
    assert opregistry.reconcile_orphans() == 0
    assert opregistry.get(live)["status"] == opregistry.RUNNING


def test_terminal_outcomes_survive_a_restart(monkeypatch):
    """History is the reason to persist at all: a failure recorded before a
    restart must still explain itself afterwards."""
    op_id = opregistry.start("world_solidify", "splat_abc")
    opregistry.finish(op_id, status=opregistry.FAILED, error="zero props built")

    _simulate_restart(monkeypatch)
    opregistry.reconcile_orphans()

    record = opregistry.get(op_id)
    assert record["status"] == opregistry.FAILED
    assert record["error"] == "zero props built"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def test_get_of_an_unknown_id_is_none():
    assert opregistry.get("nope") is None


def test_list_filters_by_job_and_kind():
    opregistry.start("mesh", "job_a")
    opregistry.start("export", "job_a")
    opregistry.start("mesh", "job_b")

    assert len(opregistry.list_ops(job_id="job_a")) == 2
    assert len(opregistry.list_ops(kind="mesh")) == 2
    assert len(opregistry.list_ops(job_id="job_a", kind="mesh")) == 1


def test_list_is_newest_first_and_limited():
    for index in range(5):
        opregistry.start("mesh", f"job_{index}")

    listed = opregistry.list_ops(limit=3)
    assert len(listed) == 3
    assert listed[0]["job_id"] == "job_4"


def test_limit_is_bounded():
    opregistry.start("mesh", "j")
    assert len(opregistry.list_ops(limit=100_000)) == 1


def test_active_by_job_groups_live_work():
    first = opregistry.start("mesh", "job_a")
    opregistry.update(first, step="poisson", progress=0.5)
    opregistry.start("export", "job_a")
    finished = opregistry.start("mesh", "job_b")
    opregistry.finish(finished)

    grouped = opregistry.active_by_job()

    assert set(grouped) == {"job_a"}
    assert {entry["kind"] for entry in grouped["job_a"]} == {"mesh", "export"}
    mesh = next(e for e in grouped["job_a"] if e["kind"] == "mesh")
    assert mesh["step"] == "poisson"
    assert mesh["progress"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_prune_drops_old_finished_rows_only():
    old = opregistry.start("mesh", "old_job")
    opregistry.finish(old)
    fresh = opregistry.start("mesh", "fresh_job")
    opregistry.finish(fresh)
    running = opregistry.start("mesh", "running_job")

    long_ago = (datetime.now(timezone.utc) - timedelta(days=90)
                ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with opregistry._connect() as conn:
        conn.execute("update operations set finished_at = ? where id = ?", (long_ago, old))

    removed = opregistry.prune(retention_days=30)

    assert removed == 1
    assert opregistry.get(old) is None
    assert opregistry.get(fresh) is not None
    assert opregistry.get(running) is not None, "in-flight work is never pruned"
