"""GET /activity: read-only busy-now snapshot — GPU lease holder via the
existing gpu_arbiter.holder_info() reader plus which jobs hold a per-job
in-process op lock (.locked() inspection only, no new bookkeeping). Also
proves the route is mounted in main.py behind require_auth."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import activity_route  # noqa: E402
import edit_ops  # noqa: E402
import export_route  # noqa: E402
import gpu_arbiter  # noqa: E402
import main as splat_main  # noqa: E402
import splat_route  # noqa: E402

_NO_HOLDER = {"lane": None, "job_id": None, "since": None, "locked": False}


def _held_lock() -> asyncio.Lock:
    """A REAL asyncio.Lock left in the held state (py3.10+ binds the lock to a
    loop lazily, so .locked() reads fine from the TestClient's loop)."""
    lock = asyncio.Lock()

    async def hold() -> None:
        await lock.acquire()

    asyncio.run(hold())
    return lock


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Hermetic: empty lock maps, no live Redis/holder bleed-through.
    monkeypatch.setattr(edit_ops, "_EDIT_LOCKS", {})
    monkeypatch.setattr(splat_route, "_PREVIEW_EXPORT_LOCKS", {})
    monkeypatch.setattr(splat_route, "_MESH_EXPORT_LOCKS", {})
    monkeypatch.setattr(export_route, "_export_locks", {})
    monkeypatch.setattr(gpu_arbiter, "holder_info", lambda: dict(_NO_HOLDER))
    app = FastAPI()
    app.include_router(activity_route.router, prefix="/api/splat")
    return TestClient(app)


def test_activity_idle_shape(client: TestClient) -> None:
    r = client.get("/api/splat/activity")
    assert r.status_code == 200
    assert r.json() == {"gpu": {"holder": None}, "jobs": {}, "edit_progress": {}}


def test_activity_reports_gpu_holder(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gpu_arbiter,
        "holder_info",
        lambda: {"lane": "mesh", "job_id": "splat_feed0001", "since": "2026-07-25T00:00:00+00:00", "locked": True},
    )
    r = client.get("/api/splat/activity")
    assert r.status_code == 200
    assert r.json()["gpu"]["holder"] == {
        "lane": "mesh",
        "job_id": "splat_feed0001",
        "since": "2026-07-25T00:00:00+00:00",
    }


def test_activity_reports_held_job_locks(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Held locks show up under their flag; idle locks in the same maps do not."""
    monkeypatch.setattr(edit_ops, "_EDIT_LOCKS", {"splat_aaaa0001": _held_lock(), "splat_idle0001": asyncio.Lock()})
    monkeypatch.setattr(splat_route, "_MESH_EXPORT_LOCKS", {"splat_aaaa0001": _held_lock()})
    monkeypatch.setattr(export_route, "_export_locks", {"splat_bbbb0001": _held_lock()})
    r = client.get("/api/splat/activity")
    assert r.status_code == 200
    assert r.json()["jobs"] == {
        "splat_aaaa0001": {"editing": True, "meshing": True},
        "splat_bbbb0001": {"exporting": True},
    }


def test_activity_preview_export_lock_flag(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(splat_route, "_PREVIEW_EXPORT_LOCKS", {"splat_cccc0001": _held_lock()})
    r = client.get("/api/splat/activity")
    assert r.json()["jobs"] == {"splat_cccc0001": {"preview_exporting": True}}


def test_activity_mounted_in_main_behind_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real app must 401 an anonymous /api/splat/activity (same require_auth
    gate as every other /api/splat router) and serve it once auth passes.
    TestClient is used WITHOUT the context manager on purpose: main's lifespan
    (orphan-job resume against the live outputs root) must not run in tests."""
    monkeypatch.setattr(gpu_arbiter, "holder_info", lambda: dict(_NO_HOLDER))
    http = TestClient(splat_main.app)
    assert http.get("/api/splat/activity").status_code == 401

    splat_main.app.dependency_overrides[splat_main.require_auth] = lambda: None
    try:
        r = http.get("/api/splat/activity")
        assert r.status_code == 200
        body = r.json()
        assert body["gpu"] == {"holder": None}
        assert isinstance(body["jobs"], dict)
    finally:
        splat_main.app.dependency_overrides.pop(splat_main.require_auth, None)


def test_activity_reports_surveying_lock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Contours/DXF/sections builds hold geo_route's per-job lock — a reloaded
    tab must see the running survey build like any other lane."""
    import geo_route

    monkeypatch.setattr(geo_route, "_GEO_EXPORT_LOCKS", {"splat_svy00001": _held_lock()})
    r = client.get("/api/splat/activity")
    assert r.json()["jobs"] == {"splat_svy00001": {"surveying": True}}


def test_activity_exposes_edit_progress(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stepped edit progress rides /activity additively — `editing` stays a
    plain boolean flag for existing consumers."""
    monkeypatch.setattr(
        edit_ops,
        "EDIT_PROGRESS",
        {"splat_prog0001": {"step": "compress", "step_index": 3, "steps": 6,
                            "labels": list(edit_ops.EDIT_STEPS), "started_at": "2026-07-25T22:00:00+00:00"}},
    )
    r = client.get("/api/splat/activity")
    body = r.json()
    assert body["edit_progress"]["splat_prog0001"]["step"] == "compress"
    assert body["edit_progress"]["splat_prog0001"]["step_index"] == 3


def test_activity_edit_progress_empty_by_default(client: TestClient) -> None:
    assert client.get("/api/splat/activity").json()["edit_progress"] == {}
