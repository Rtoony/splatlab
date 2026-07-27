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
import opregistry  # noqa: E402
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
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    # Hermetic: empty lock maps, no live Redis/holder bleed-through, and an
    # operation registry in a temp dir so tests never see the live one.
    monkeypatch.setattr(edit_ops, "_EDIT_LOCKS", {})
    monkeypatch.setattr(splat_route, "_PREVIEW_EXPORT_LOCKS", {})
    monkeypatch.setattr(splat_route, "_MESH_EXPORT_LOCKS", {})
    monkeypatch.setattr(export_route, "_export_locks", {})
    monkeypatch.setattr(gpu_arbiter, "holder_info", lambda: dict(_NO_HOLDER))
    opregistry.configure_storage(tmp_path / "ops")
    opregistry.init_db()
    app = FastAPI()
    app.include_router(activity_route.router, prefix="/api/splat")
    try:
        yield TestClient(app)
    finally:
        opregistry.configure_storage(Path(opregistry.PROJECT_ROOT) / "data" / "ops")


def test_activity_idle_shape(client: TestClient) -> None:
    r = client.get("/api/splat/activity")
    assert r.status_code == 200
    assert r.json() == {
        "gpu": {"holder": None}, "jobs": {}, "edit_progress": {}, "operations": {}}


# ---------------------------------------------------------------------------
# The persistent operation registry, over the wire
# ---------------------------------------------------------------------------

def test_activity_reports_live_registry_operations(client: TestClient) -> None:
    op_id = opregistry.start("world_solidify", "splat_abc", step="shell")
    opregistry.update(op_id, progress=0.25)

    body = client.get("/api/splat/activity").json()

    assert list(body["operations"]) == ["splat_abc"]
    entry = body["operations"]["splat_abc"][0]
    assert entry["id"] == op_id
    assert entry["kind"] == "world_solidify"
    assert entry["step"] == "shell"
    assert entry["progress"] == 0.25


def test_finished_operations_leave_the_activity_view(client: TestClient) -> None:
    op_id = opregistry.start("mesh", "splat_abc")
    opregistry.finish(op_id)
    assert client.get("/api/splat/activity").json()["operations"] == {}


def test_ops_endpoint_polls_a_single_operation(client: TestClient) -> None:
    op_id = opregistry.start("mesh", "splat_abc", step="poisson")
    opregistry.finish(op_id, result={"faces": 8000})

    r = client.get(f"/api/splat/ops/{op_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "succeeded"
    assert body["result"] == {"faces": 8000}
    assert body["active"] is False


def test_ops_endpoint_404s_on_an_unknown_id(client: TestClient) -> None:
    assert client.get("/api/splat/ops/nope").status_code == 404


def test_ops_list_filters_and_orders(client: TestClient) -> None:
    opregistry.start("mesh", "job_a")
    opregistry.start("export", "job_a")
    opregistry.start("mesh", "job_b")

    listed = client.get("/api/splat/ops", params={"job_id": "job_a"}).json()["operations"]
    assert {entry["kind"] for entry in listed} == {"mesh", "export"}

    by_kind = client.get("/api/splat/ops", params={"kind": "mesh"}).json()["operations"]
    assert {entry["job_id"] for entry in by_kind} == {"job_a", "job_b"}


def test_ops_list_keeps_history_a_restart_would_otherwise_lose(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    op_id = opregistry.start("mesh", "splat_abc")
    opregistry.finish(op_id, status=opregistry.FAILED, error="xatlas did not converge")
    monkeypatch.setattr(opregistry, "RUNTIME_ID", "a-new-backend-process")

    listed = client.get("/api/splat/ops", params={"job_id": "splat_abc"}).json()["operations"]

    assert listed[0]["status"] == "failed"
    assert "xatlas" in listed[0]["error"]


def test_a_running_row_from_a_dead_process_is_not_reported_as_busy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rail: persistence must not resurrect phantom in-flight work."""
    opregistry.start("mesh", "splat_abc")
    monkeypatch.setattr(opregistry, "RUNTIME_ID", "a-new-backend-process")

    assert client.get("/api/splat/activity").json()["operations"] == {}
    listed = client.get("/api/splat/ops", params={"job_id": "splat_abc"}).json()["operations"]
    assert listed[0]["status"] == "abandoned"
    assert listed[0]["active"] is False


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
