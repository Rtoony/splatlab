"""Persistent registry of heavy operations.

Why this exists
---------------
Every long-running SplatLab operation — mesh builds, object isolation, scene
ground/assemble, world solidify, portable exports, survey exports — was tracked
only by an in-process `asyncio.Lock` and a dict. That made three things
impossible:

  * a blocking multi-minute POST could not be polled; a client that lost its
    connection had no way to ask what happened;
  * `/activity` reported nothing after a backend restart, even about work that
    had genuinely finished;
  * there was no history — "did the world build last night, and why did it
    fail?" had to be answered by grepping job.log.

The truthfulness rail
---------------------
`activity_route` earned its honesty for free: in-process locks die with the
backend, and a restart also kills the operation each lock guarded, so a fresh
process correctly reported nothing in flight. Persistence breaks that gift — a
row saying `running` outlives the process that wrote it, and a stale `running`
row is exactly the "claimed something is happening when it is not" failure this
project refuses to ship.

So every row is stamped with `runtime_id`, a UUID minted once per backend
process. A `running` row whose `runtime_id` is not the current one belongs to a
dead process and is reported as **abandoned**, never as running. Reconciliation
happens at startup (`reconcile_orphans`) and defensively on every read, so the
registry cannot claim in-flight work it is not actually running.

Deliberately stdlib-only and self-contained, mirroring `feedback.py`: one
connection per call, WAL for concurrent readers, no ORM.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPLATLAB_OPS_DATA_DIR", PROJECT_ROOT / "data" / "ops"))
DB_PATH = DATA_DIR / "ops.sqlite3"

# Minted once per backend process. Rows carrying any other value were written by
# a process that no longer exists.
RUNTIME_ID = uuid.uuid4().hex

RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
ABANDONED = "abandoned"
TERMINAL = (SUCCEEDED, FAILED, CANCELLED, ABANDONED)

_INIT_LOCK = threading.Lock()
_INITIALIZED_DB: Path | None = None

DEFAULT_RETENTION_DAYS = 30


def configure_storage(data_dir: Path) -> None:
    """Test hook for redirecting SQLite to a temp directory."""
    global DATA_DIR, DB_PATH, _INITIALIZED_DB

    DATA_DIR = Path(data_dir)
    DB_PATH = DATA_DIR / "ops.sqlite3"
    _INITIALIZED_DB = None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_db() -> None:
    global _INITIALIZED_DB

    if _INITIALIZED_DB == DB_PATH:
        return
    with _INIT_LOCK:
        if _INITIALIZED_DB == DB_PATH:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                create table if not exists operations (
                  id text primary key,
                  kind text not null,
                  job_id text not null default '',
                  status text not null,
                  step text not null default '',
                  progress real,
                  detail text not null default '',
                  error text not null default '',
                  result_json text not null default '',
                  runtime_id text not null,
                  pid integer not null,
                  started_at text not null,
                  updated_at text not null,
                  finished_at text
                );
                create index if not exists operations_job_idx on operations(job_id, started_at desc);
                create index if not exists operations_status_idx on operations(status, started_at desc);
                """
            )
            conn.commit()
        finally:
            conn.close()
        _INITIALIZED_DB = DB_PATH


def init_db() -> None:
    _ensure_db()


def _connect() -> sqlite3.Connection:
    _ensure_db()
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    record = dict(row)
    raw = record.pop("result_json", "") or ""
    try:
        record["result"] = json.loads(raw) if raw else None
    except ValueError:
        record["result"] = None
    # Defence in depth: a `running` row from a dead process is never reported as
    # running, even if reconciliation has not run yet in this process.
    if record["status"] == RUNNING and record.get("runtime_id") != RUNTIME_ID:
        record["status"] = ABANDONED
        record["error"] = record["error"] or (
            "the backend process running this operation exited before it finished"
        )
    record["active"] = record["status"] == RUNNING
    return record


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start(kind: str, job_id: str = "", *, step: str = "", detail: str = "") -> str:
    """Record an operation as running. Returns its id."""
    op_id = uuid.uuid4().hex
    now = _now()
    with _connect() as conn:
        conn.execute(
            "insert into operations (id, kind, job_id, status, step, detail, "
            "runtime_id, pid, started_at, updated_at) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (op_id, str(kind), str(job_id or ""), RUNNING, str(step), str(detail),
             RUNTIME_ID, os.getpid(), now, now),
        )
    return op_id


def update(op_id: str, *, step: str | None = None, progress: float | None = None,
           detail: str | None = None) -> None:
    """Advance a running operation. Silently ignores unknown/finished ids so a
    progress callback can never take down the work it is reporting on."""
    sets, params = ["updated_at = ?"], [_now()]
    if step is not None:
        sets.append("step = ?")
        params.append(str(step))
    if progress is not None:
        sets.append("progress = ?")
        params.append(max(0.0, min(1.0, float(progress))))
    if detail is not None:
        sets.append("detail = ?")
        params.append(str(detail))
    params.append(op_id)
    with _connect() as conn:
        conn.execute(
            f"update operations set {', '.join(sets)} where id = ? and status = '{RUNNING}'",
            params,
        )


def finish(op_id: str, *, status: str = SUCCEEDED, result: Any = None,
           error: str = "", step: str | None = None) -> None:
    if status not in TERMINAL:
        raise ValueError(f"{status!r} is not a terminal operation status")
    now = _now()
    payload = ""
    if result is not None:
        try:
            payload = json.dumps(result)[:64_000]
        except (TypeError, ValueError):
            payload = ""
    with _connect() as conn:
        conn.execute(
            "update operations set status = ?, result_json = ?, error = ?, "
            "updated_at = ?, finished_at = ?, "
            "progress = case when ? = ? then 1.0 else progress end, "
            "step = coalesce(?, step) where id = ?",
            (status, payload, str(error)[:4000], now, now, status, SUCCEEDED, step, op_id),
        )


@contextmanager
def operation(kind: str, job_id: str = "", *, step: str = "",
              detail: str = "") -> Iterator[str]:
    """Run a block as a tracked operation.

    A raised exception is recorded as `failed` with its message and re-raised —
    the registry never swallows a failure, it only witnesses it.
    """
    op_id = start(kind, job_id, step=step, detail=detail)
    try:
        yield op_id
    except BaseException as exc:            # noqa: BLE001 - recorded then re-raised
        finish(op_id, status=FAILED, error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        current = get(op_id)
        if current and current["status"] == RUNNING:
            finish(op_id, status=SUCCEEDED)


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get(op_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("select * from operations where id = ?", (op_id,)).fetchone()
    return _row_to_dict(row) if row else None


def list_ops(*, job_id: str | None = None, kind: str | None = None,
             active_only: bool = False, limit: int = 50) -> list[dict[str, Any]]:
    where, params = [], []
    if job_id:
        where.append("job_id = ?")
        params.append(job_id)
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if active_only:
        # Only this runtime's rows can legitimately still be running.
        where.append("status = ? and runtime_id = ?")
        params.extend([RUNNING, RUNTIME_ID])
    clause = f"where {' and '.join(where)}" if where else ""
    params.append(max(1, min(int(limit), 500)))
    with _connect() as conn:
        rows = conn.execute(
            f"select * from operations {clause} order by started_at desc, rowid desc limit ?",
            params,
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def active_by_job() -> dict[str, list[dict[str, Any]]]:
    """Genuinely-in-flight operations, grouped by job, for /activity."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in list_ops(active_only=True, limit=500):
        grouped.setdefault(record["job_id"], []).append({
            "id": record["id"], "kind": record["kind"], "step": record["step"],
            "progress": record["progress"], "detail": record["detail"],
            "started_at": record["started_at"], "updated_at": record["updated_at"],
        })
    return grouped


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def reconcile_orphans() -> int:
    """Mark every `running` row from a previous process as abandoned.

    Called at startup. Returns how many rows were corrected, so the caller can
    say plainly what the last shutdown interrupted instead of leaving phantom
    work on the activity view.
    """
    now = _now()
    with _connect() as conn:
        cursor = conn.execute(
            "update operations set status = ?, error = ?, updated_at = ?, finished_at = ? "
            "where status = ? and runtime_id != ?",
            (ABANDONED,
             "the backend process running this operation exited before it finished",
             now, now, RUNNING, RUNTIME_ID),
        )
        return cursor.rowcount or 0


def prune(retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Drop finished rows older than the retention window."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(retention_days)))
              ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    with _connect() as conn:
        cursor = conn.execute(
            "delete from operations where status != ? and finished_at is not null "
            "and finished_at < ?",
            (RUNNING, cutoff),
        )
        return cursor.rowcount or 0
