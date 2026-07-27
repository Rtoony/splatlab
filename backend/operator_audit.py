"""Append-only operator audit trail.

This was a no-op stub inherited from the portal ("splatlab is single-operator"),
while 25 call sites across 8 modules faithfully awaited it with rich payloads —
job completions, deletions, edits, exports, scale changes, polish ingests. Every
one of those events was assembled and thrown away: infrastructure for a thing,
without the thing. Single-operator is a reason not to need *authorisation*, not
a reason not to be able to answer "what happened to this job, and when".

Deliberately small: one JSON object per line under `data/audit/`, rotated by
day, opened in append mode so concurrent writers interleave whole lines rather
than corrupting each other. No index, no query language — `grep` and `jq` are
the interface, which is what an operator actually reaches for.

Never raises. An audit trail that can take down the operation it is recording is
worse than no audit trail, so every failure degrades to a stderr line.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPLATLAB_AUDIT_DATA_DIR", PROJECT_ROOT / "data" / "audit"))

# A single event should never be able to fill a disk; payloads are metadata.
MAX_FIELD_CHARS = 4000
MAX_LINE_BYTES = 64_000
RETENTION_DAYS = 400


def configure_storage(data_dir: Path) -> None:
    """Test hook for redirecting the trail to a temp directory."""
    global DATA_DIR
    DATA_DIR = Path(data_dir)


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def log_path(when: datetime | None = None) -> Path:
    return DATA_DIR / f"{(when or _now()).strftime('%Y-%m-%d')}.jsonl"


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_FIELD_CHARS:
        return value[:MAX_FIELD_CHARS] + "...[clipped]"
    return value


async def audit_operator_event(
    *,
    request: Any = None,
    title: str = "",
    description: str = "",
    variant: str = "default",
    action: str = "",
    target: str = "",
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Record one operator-visible event. Signature unchanged from the stub."""
    record = {
        "at": _now().isoformat().replace("+00:00", "Z"),
        "action": _clip(str(action)),
        "target": _clip(str(target)),
        "variant": _clip(str(variant)),
        "title": _clip(str(title)),
        "description": _clip(str(description)),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }
    if extra:
        record["extra"] = {str(k): _clip(v) for k, v in extra.items()}
    # `request` is a live Starlette object at most call sites; record only the
    # two safe identifying fields and never headers, cookies or bodies — the
    # feedback module's context rules apply here too.
    if request is not None:
        with_client = getattr(request, "client", None)
        record["request"] = {
            "path": str(getattr(getattr(request, "url", None), "path", "") or ""),
            "client": str(getattr(with_client, "host", "") or ""),
        }
    write_event(record)


def write_event(record: dict[str, Any]) -> None:
    """Append one record. Never raises."""
    try:
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError):
            line = json.dumps({"at": record.get("at"), "action": record.get("action"),
                               "error": "payload was not serialisable"})
        if len(line.encode()) > MAX_LINE_BYTES:
            line = json.dumps({"at": record.get("at"), "action": record.get("action"),
                               "target": record.get("target"),
                               "error": f"event exceeded {MAX_LINE_BYTES} bytes and was dropped"})
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Append mode: the OS keeps concurrent writers from interleaving within
        # a single line, which is the whole reason this is JSONL and not JSON.
        with open(log_path(), "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception as exc:                    # noqa: BLE001 - never fatal
        print(f"[audit] failed to record event: {exc}", file=sys.stderr, flush=True)


def read_events(*, limit: int = 200, action: str | None = None,
                job_id: str | None = None, days: int = 7) -> list[dict[str, Any]]:
    """Most recent events first. Reads at most `days` daily files."""
    collected: list[dict[str, Any]] = []
    if not DATA_DIR.is_dir():
        return collected
    for path in sorted(DATA_DIR.glob("*.jsonl"), reverse=True)[:max(1, days)]:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if action and record.get("action") != action:
                continue
            if job_id and (record.get("metadata") or {}).get("job_id") != job_id:
                continue
            collected.append(record)
            if len(collected) >= limit:
                return collected
    return collected


def prune(retention_days: int = RETENTION_DAYS) -> int:
    """Delete daily files older than the retention window. Returns the count."""
    if not DATA_DIR.is_dir():
        return 0
    cutoff = _now().timestamp() - max(1, int(retention_days)) * 86_400
    removed = 0
    for path in DATA_DIR.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed
