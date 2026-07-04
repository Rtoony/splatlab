"""Splatlab-native feedback API.

This module is intentionally self-contained: main.py applies auth when it
includes the router, while this slice owns local SQLite persistence and
filesystem attachment storage under data/feedback/.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

router = APIRouter()

FeedbackType = Literal["Bug", "UX", "Copy", "Data", "Performance", "Request", "Question", "Idea", "Comment"]
FeedbackPriority = Literal["Low", "Medium", "High", "Critical"]
FeedbackStatus = Literal[
    "New",
    "Triaged",
    "Planned",
    "In Progress",
    "Needs Info",
    "Ready to Test",
    "Fixed",
    "Accepted",
    "Closed",
    "Won't Fix",
    "Archived",
]

CODEX_QUEUE = ("New", "Triaged", "Planned", "In Progress")
USER_QUEUE = ("Needs Info", "Ready to Test", "Fixed")
TERMINAL_STATUSES = ("Accepted", "Closed", "Won't Fix", "Archived")
ACTIVE_QUEUE = CODEX_QUEUE + USER_QUEUE
QUEUE_STATUSES = {
    "active": ACTIVE_QUEUE,
    "codex": CODEX_QUEUE,
    "user": USER_QUEUE,
    "verification": USER_QUEUE,
    "ready": USER_QUEUE,
    "terminal": TERMINAL_STATUSES,
    "closed": TERMINAL_STATUSES,
}
ALL_STATUSES = set(ACTIVE_QUEUE + TERMINAL_STATUSES)

MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024
MAX_CONTEXT_DEPTH = 8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("SPLATLAB_FEEDBACK_DATA_DIR", PROJECT_ROOT / "data" / "feedback"))
DB_PATH = DATA_DIR / "feedback.sqlite3"
ATTACHMENTS_DIR = DATA_DIR / "attachments"

_INIT_LOCK = threading.Lock()
_INITIALIZED_DB: Path | None = None

_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SECRET_KEY_RE = re.compile(
    r"(token|secret|password|passwd|pwd|api[_-]?key|apikey|authorization|bearer|cookie|session|jwt|csrf|credential|private[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)
_DROP_KEY_RE = re.compile(
    r"(^body$|request[_-]?body|response[_-]?body|payload|local[_-]?storage|cookies?)",
    re.IGNORECASE,
)


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=240)
    body: str = Field(..., min_length=1, max_length=20_000)
    feedback_type: FeedbackType = "Comment"
    priority: FeedbackPriority = "Medium"
    status: FeedbackStatus = "New"
    page_url: str = ""
    page_path: str = ""
    page_tab: str = ""
    component_label: str = ""
    tags_json: Any = Field(default_factory=list)
    context_json: Any = Field(default_factory=dict)
    resolution_notes: str = ""
    resolution_metadata_json: Any = Field(default_factory=dict)
    created_by: str = Field(default="operator", min_length=1, max_length=120)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "feedback_type" not in data and "type" in data:
            data = {**data, "feedback_type": data.get("type")}
            data.pop("type", None)
        return data

    @field_validator(
        "title",
        "body",
        "page_url",
        "page_path",
        "page_tab",
        "component_label",
        "resolution_notes",
        "created_by",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


class FeedbackPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=240)
    body: str | None = Field(default=None, min_length=1, max_length=20_000)
    feedback_type: FeedbackType | None = None
    priority: FeedbackPriority | None = None
    status: FeedbackStatus | None = None
    page_url: str | None = None
    page_path: str | None = None
    page_tab: str | None = None
    component_label: str | None = None
    tags_json: Any | None = None
    context_json: Any | None = None
    resolution_notes: str | None = None
    resolution_metadata_json: Any | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "feedback_type" not in data and "type" in data:
            data = {**data, "feedback_type": data.get("type")}
            data.pop("type", None)
        return data

    @field_validator(
        "title",
        "body",
        "page_url",
        "page_path",
        "page_tab",
        "component_label",
        "resolution_notes",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: Any) -> str | None:
        return None if value is None else str(value).strip()


class FeedbackCommentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str = Field(..., min_length=1, max_length=20_000)
    created_by: str = Field(default="operator", min_length=1, max_length=120)

    @field_validator("body", "created_by", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


def configure_storage(data_dir: Path) -> None:
    """Test hook for redirecting SQLite and attachments to a temp directory."""
    global DATA_DIR, DB_PATH, ATTACHMENTS_DIR, _INITIALIZED_DB

    DATA_DIR = Path(data_dir)
    DB_PATH = DATA_DIR / "feedback.sqlite3"
    ATTACHMENTS_DIR = DATA_DIR / "attachments"
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
        ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                create table if not exists feedback_items (
                  id integer primary key autoincrement,
                  title text not null,
                  body text not null,
                  feedback_type text not null,
                  priority text not null,
                  status text not null,
                  page_url text not null default '',
                  page_path text not null default '',
                  page_tab text not null default '',
                  component_label text not null default '',
                  tags_json text not null default '[]',
                  context_json text not null default '{}',
                  resolution_notes text not null default '',
                  resolution_metadata_json text not null default '{}',
                  created_by text not null,
                  created_at text not null,
                  updated_at text not null,
                  completed_at text,
                  archived_at text
                );

                create table if not exists feedback_comments (
                  id integer primary key autoincrement,
                  feedback_item_id integer not null references feedback_items(id) on delete cascade,
                  body text not null,
                  created_by text not null,
                  created_at text not null
                );

                create table if not exists feedback_attachments (
                  id integer primary key autoincrement,
                  feedback_item_id integer not null references feedback_items(id) on delete cascade,
                  storage_key text not null,
                  original_name text not null,
                  content_type text not null,
                  size_bytes integer not null,
                  sha256 text not null,
                  created_by text not null,
                  created_at text not null
                );

                create index if not exists idx_feedback_items_status_updated
                  on feedback_items(status, updated_at desc);
                create index if not exists idx_feedback_comments_item
                  on feedback_comments(feedback_item_id, created_at, id);
                create index if not exists idx_feedback_attachments_item
                  on feedback_attachments(feedback_item_id, created_at, id);
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _loads_json(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return fallback


def _json_text(value: Any, *, fallback: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            value = fallback
        else:
            try:
                value = json.loads(stripped)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="invalid JSON field") from exc
    value = _sanitize_json(value)
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise HTTPException(status_code=413, detail="JSON field too large")
    return text


def _tags_text(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            value = []
        else:
            try:
                value = json.loads(stripped)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="tags_json must be valid JSON") from exc
    if not isinstance(value, list):
        raise HTTPException(status_code=422, detail="tags_json must be a JSON array")
    tags: list[str] = []
    seen: set[str] = set()
    for raw in value:
        tag = str(raw).strip()
        if not tag or len(tag) > 64:
            continue
        key = tag.lower()
        if key not in seen:
            tags.append(tag)
            seen.add(key)
    return json.dumps(tags[:25], ensure_ascii=True, separators=(",", ":"))


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_RE.search(key))


def _sanitize_url(value: str) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme and not parts.netloc and "?" not in value:
        return value
    safe_query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_secret_key(key)
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(safe_query, doseq=True), ""))


def _sanitize_json(value: Any, depth: int = 0) -> Any:
    if depth > MAX_CONTEXT_DEPTH:
        return "[truncated]"
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key)
            if _DROP_KEY_RE.search(key) or _is_secret_key(key):
                continue
            if key == "recent_failed_api_calls" and isinstance(raw_val, list):
                safe[key] = [_sanitize_failed_call(item) for item in raw_val[:25] if isinstance(item, dict)]
            else:
                safe[key] = _sanitize_json(raw_val, depth + 1)
        return safe
    if isinstance(value, list):
        return [_sanitize_json(item, depth + 1) for item in value[:100]]
    if isinstance(value, str):
        return _sanitize_url(value.strip()[:5000])
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


def _sanitize_failed_call(item: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    if "method" in item:
        safe["method"] = str(item["method"]).upper()[:12]
    path = item.get("path") or item.get("url")
    if path is not None:
        safe["path"] = _sanitize_url(str(path))[:1000]
    if "status" in item:
        with contextlib_suppress(ValueError, TypeError):
            safe["status"] = int(item["status"])
    duration = item.get("duration_ms")
    if duration is not None:
        with contextlib_suppress(ValueError, TypeError):
            safe["duration_ms"] = int(duration)
    return safe


class contextlib_suppress:
    def __init__(self, *exceptions: type[BaseException]) -> None:
        self._exceptions = exceptions

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: Any) -> bool:
        return exc_type is not None and issubclass(exc_type, self._exceptions)


def _row_to_feedback(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "body": row["body"],
        "feedback_type": row["feedback_type"],
        "type": row["feedback_type"],
        "priority": row["priority"],
        "status": row["status"],
        "page_url": row["page_url"],
        "page_path": row["page_path"],
        "page_tab": row["page_tab"],
        "component_label": row["component_label"],
        "tags_json": _loads_json(row["tags_json"], []),
        "context_json": _loads_json(row["context_json"], {}),
        "resolution_notes": row["resolution_notes"],
        "resolution_metadata_json": _loads_json(row["resolution_metadata_json"], {}),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "completed_at": row["completed_at"],
        "archived_at": row["archived_at"],
    }


def _row_to_comment(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "feedback_item_id": row["feedback_item_id"],
        "body": row["body"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
    }


def _row_to_attachment(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "feedback_item_id": row["feedback_item_id"],
        "original_name": row["original_name"],
        "content_type": row["content_type"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "download_url": f"/api/feedback/{row["feedback_item_id"]}/attachments/{row["id"]}",
        "created_by": row["created_by"],
        "created_at": row["created_at"],
    }


def _get_feedback_row(conn: sqlite3.Connection, item_id: int) -> sqlite3.Row:
    row = conn.execute("select * from feedback_items where id = ?", (item_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="feedback item not found")
    return row


def _get_attachment_row(conn: sqlite3.Connection, item_id: int, attachment_id: int) -> sqlite3.Row:
    row = conn.execute(
        "select * from feedback_attachments where feedback_item_id = ? and id = ?",
        (item_id, attachment_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="attachment not found")
    return row


def _parse_status_filter(status: str) -> list[str]:
    statuses = [part.strip() for part in status.split(",") if part.strip()]
    invalid = [item for item in statuses if item not in ALL_STATUSES]
    if invalid:
        raise HTTPException(status_code=422, detail=f"unknown status: {invalid[0]}")
    return statuses


@router.get("/api/feedback")
def list_feedback(
    queue: str | None = Query(default="active"),
    status: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    where: list[str] = []
    params: list[Any] = []
    if status:
        statuses = _parse_status_filter(status)
        where.append(f"status in ({','.join('?' for _ in statuses)})")
        params.extend(statuses)
    elif queue and queue != "all":
        statuses = QUEUE_STATUSES.get(queue)
        if statuses is None:
            raise HTTPException(status_code=422, detail="unknown feedback queue")
        where.append(f"status in ({','.join('?' for _ in statuses)})")
        params.extend(statuses)
    if search:
        term = f"%{search.strip().lower()}%"
        where.append(
            "(lower(title) like ? or lower(body) like ? or lower(page_path) like ? "
            "or lower(component_label) like ? or lower(tags_json) like ?)"
        )
        params.extend([term, term, term, term, term])
    where_sql = f"where {' and '.join(where)}" if where else ""
    with _connect() as conn:
        total = conn.execute(f"select count(*) from feedback_items {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"select * from feedback_items {where_sql} order by updated_at desc, id desc limit ? offset ?",
            [*params, limit, offset],
        ).fetchall()
    return {"items": [_row_to_feedback(row) for row in rows], "total": total}


@router.post("/api/feedback", status_code=201)
def create_feedback(payload: FeedbackCreate) -> dict[str, Any]:
    now = _now()
    page_url = _sanitize_url(payload.page_url)
    status = payload.status
    completed_at = now if status in TERMINAL_STATUSES else None
    archived_at = now if status == "Archived" else None
    with _connect() as conn:
        cur = conn.execute(
            """
            insert into feedback_items (
              title, body, feedback_type, priority, status, page_url, page_path,
              page_tab, component_label, tags_json, context_json, resolution_notes,
              resolution_metadata_json, created_by, created_at, updated_at,
              completed_at, archived_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.title,
                payload.body,
                payload.feedback_type,
                payload.priority,
                status,
                page_url,
                payload.page_path,
                payload.page_tab,
                payload.component_label,
                _tags_text(payload.tags_json),
                _json_text(payload.context_json, fallback={}),
                payload.resolution_notes,
                _json_text(payload.resolution_metadata_json, fallback={}),
                payload.created_by,
                now,
                now,
                completed_at,
                archived_at,
            ),
        )
        conn.commit()
        row = _get_feedback_row(conn, int(cur.lastrowid))
    return _row_to_feedback(row)


@router.get("/api/feedback/{item_id}")
def get_feedback(item_id: int) -> dict[str, Any]:
    with _connect() as conn:
        item = _row_to_feedback(_get_feedback_row(conn, item_id))
        comments = conn.execute(
            "select * from feedback_comments where feedback_item_id = ? order by created_at, id",
            (item_id,),
        ).fetchall()
        attachments = conn.execute(
            "select * from feedback_attachments where feedback_item_id = ? order by created_at, id",
            (item_id,),
        ).fetchall()
    item["comments"] = [_row_to_comment(row) for row in comments]
    item["attachments"] = [_row_to_attachment(row) for row in attachments]
    return item


@router.patch("/api/feedback/{item_id}")
def update_feedback(item_id: int, payload: FeedbackPatch) -> dict[str, Any]:
    updates: list[str] = []
    params: list[Any] = []
    fields = payload.model_fields_set

    def set_field(column: str, value: Any) -> None:
        updates.append(f"{column} = ?")
        params.append(value)

    if "title" in fields:
        set_field("title", payload.title)
    if "body" in fields:
        set_field("body", payload.body)
    if "feedback_type" in fields:
        set_field("feedback_type", payload.feedback_type)
    if "priority" in fields:
        set_field("priority", payload.priority)
    if "status" in fields:
        set_field("status", payload.status)
        now_for_status = _now()
        set_field("completed_at", now_for_status if payload.status in TERMINAL_STATUSES else None)
        set_field("archived_at", now_for_status if payload.status == "Archived" else None)
    if "page_url" in fields:
        set_field("page_url", _sanitize_url(payload.page_url or ""))
    if "page_path" in fields:
        set_field("page_path", payload.page_path or "")
    if "page_tab" in fields:
        set_field("page_tab", payload.page_tab or "")
    if "component_label" in fields:
        set_field("component_label", payload.component_label or "")
    if "tags_json" in fields:
        set_field("tags_json", _tags_text(payload.tags_json))
    if "context_json" in fields:
        set_field("context_json", _json_text(payload.context_json, fallback={}))
    if "resolution_notes" in fields:
        set_field("resolution_notes", payload.resolution_notes or "")
    if "resolution_metadata_json" in fields:
        set_field("resolution_metadata_json", _json_text(payload.resolution_metadata_json, fallback={}))

    if not updates:
        with _connect() as conn:
            return _row_to_feedback(_get_feedback_row(conn, item_id))

    set_field("updated_at", _now())
    with _connect() as conn:
        _get_feedback_row(conn, item_id)
        conn.execute(f"update feedback_items set {', '.join(updates)} where id = ?", [*params, item_id])
        conn.commit()
        row = _get_feedback_row(conn, item_id)
    return _row_to_feedback(row)


@router.get("/api/feedback/{item_id}/comments")
def list_comments(item_id: int) -> dict[str, Any]:
    with _connect() as conn:
        _get_feedback_row(conn, item_id)
        rows = conn.execute(
            "select * from feedback_comments where feedback_item_id = ? order by created_at, id",
            (item_id,),
        ).fetchall()
    return {"items": [_row_to_comment(row) for row in rows]}


@router.post("/api/feedback/{item_id}/comments", status_code=201)
def create_comment(item_id: int, payload: FeedbackCommentCreate) -> dict[str, Any]:
    now = _now()
    with _connect() as conn:
        _get_feedback_row(conn, item_id)
        cur = conn.execute(
            "insert into feedback_comments (feedback_item_id, body, created_by, created_at) values (?, ?, ?, ?)",
            (item_id, payload.body, payload.created_by, now),
        )
        conn.execute("update feedback_items set updated_at = ? where id = ?", (now, item_id))
        conn.commit()
        row = conn.execute("select * from feedback_comments where id = ?", (cur.lastrowid,)).fetchone()
    return _row_to_comment(row)


@router.get("/api/feedback/{item_id}/attachments")
def list_attachments(item_id: int) -> dict[str, Any]:
    with _connect() as conn:
        _get_feedback_row(conn, item_id)
        rows = conn.execute(
            "select * from feedback_attachments where feedback_item_id = ? order by created_at, id",
            (item_id,),
        ).fetchall()
    return {"items": [_row_to_attachment(row) for row in rows]}


@router.post("/api/feedback/{item_id}/attachments", status_code=201)
async def upload_attachments(
    item_id: int,
    files: list[UploadFile] = File(...),
    created_by: str = Form(default="operator"),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=422, detail="at least one file is required")
    created_by = created_by.strip() or "operator"
    with _connect() as conn:
        _get_feedback_row(conn, item_id)
        attachments = [await _store_upload(conn, item_id, upload, created_by) for upload in files]
        conn.execute("update feedback_items set updated_at = ? where id = ?", (_now(), item_id))
        conn.commit()
    return {"items": attachments}


async def _store_upload(
    conn: sqlite3.Connection,
    item_id: int,
    upload: UploadFile,
    created_by: str,
) -> dict[str, Any]:
    original_name = Path(upload.filename or "attachment").name or "attachment"
    safe_name = _safe_filename(original_name)
    content_type = upload.content_type or mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    now = _now()
    cur = conn.execute(
        """
        insert into feedback_attachments (
          feedback_item_id, storage_key, original_name, content_type,
          size_bytes, sha256, created_by, created_at
        ) values (?, '', ?, ?, 0, '', ?, ?)
        """,
        (item_id, original_name, content_type, created_by, now),
    )
    attachment_id = int(cur.lastrowid)
    relative_key = f"attachments/{item_id}/{attachment_id}-{safe_name}"
    destination = DATA_DIR / relative_key
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_ATTACHMENT_BYTES:
                    raise HTTPException(status_code=413, detail="attachment too large")
                digest.update(chunk)
                out.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        conn.execute("delete from feedback_attachments where id = ?", (attachment_id,))
        raise
    sha256 = digest.hexdigest()
    conn.execute(
        "update feedback_attachments set storage_key = ?, size_bytes = ?, sha256 = ? where id = ?",
        (relative_key, size, sha256, attachment_id),
    )
    row = conn.execute("select * from feedback_attachments where id = ?", (attachment_id,)).fetchone()
    return _row_to_attachment(row)


def _safe_filename(filename: str) -> str:
    stem = _FILENAME_RE.sub("-", filename.strip()).strip(".-")
    return stem[:160] or "attachment"


@router.get("/api/feedback/{item_id}/attachments/{attachment_id}")
def stream_attachment(item_id: int, attachment_id: int) -> FileResponse:
    return _attachment_response(item_id, attachment_id)


@router.get("/api/feedback/{item_id}/attachments/{attachment_id}/stream")
def stream_attachment_alias(item_id: int, attachment_id: int) -> FileResponse:
    return _attachment_response(item_id, attachment_id)


def _attachment_response(item_id: int, attachment_id: int) -> FileResponse:
    with _connect() as conn:
        row = _get_attachment_row(conn, item_id, attachment_id)
        path = (DATA_DIR / row["storage_key"]).resolve()
    data_root = DATA_DIR.resolve()
    if path != data_root and data_root not in path.parents:
        raise HTTPException(status_code=404, detail="attachment not found")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="attachment file not found")
    return FileResponse(path, media_type=row["content_type"], filename=row["original_name"])


@router.get("/api/app-context")
def get_app_context() -> dict[str, Any]:
    return {
        "service": "splatlab",
        "feedback_context_version": 1,
        "version": os.environ.get("SPLATLAB_VERSION", ""),
        "build_id": os.environ.get("SPLATLAB_BUILD_ID", ""),
        "environment": os.environ.get("SPLATLAB_ENV", "local"),
        "git_short_commit": _git_short_commit(),
    }


def _git_short_commit() -> str:
    git_dir = PROJECT_ROOT / ".git"
    head_path = git_dir / "HEAD"
    try:
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            return (git_dir / ref).read_text(encoding="utf-8").strip()[:12]
        return head[:12]
    except OSError:
        return ""
