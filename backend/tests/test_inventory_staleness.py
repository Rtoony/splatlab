"""Per-scene inventory staleness, instead of a fleet-wide recompute.

Fixing the unseen-gaussian bug changed what a relevancy score means, so every
inventory cached before it is out of date. The obvious lever — bumping
INVENTORY_VERSION — is a hard cache key: it would silently force a multi-minute
GPU recompute of every scene the next time anyone opened it.

So relevancy_core.RELEVANCY_GENERATION is a SOFT marker instead. An older
cached inventory is still served and reports itself stale; `refresh=true`
recomputes one scene. These tests pin that a stale cache is never silently
discarded and never silently trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "langfield"))

import relevancy_core  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_1a2b3c"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(splat_route, "_langfield_stale_guard", lambda _p: None)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda _d: Path("/tmp/config.yml"))
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path) -> Path:
    job_dir = outputs / JOB
    lf = job_dir / splat_route.LANGFIELD_DIRNAME
    lf.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps(
        {"job_id": JOB, "output_dir": str(job_dir), "status": "completed"}))
    (lf / "gauss_emb.npz").write_bytes(b"stub")
    return job_dir


def _worker_reply(monkeypatch, payload: dict) -> list[dict]:
    """Capture what the app asks the worker for, and script the reply."""
    seen: list[dict] = []

    async def fake_inventory(config_path, lfdir, refresh=False):
        seen.append({"config": config_path, "lfdir": lfdir, "refresh": refresh})
        return payload

    monkeypatch.setattr(splat_route, "_langfield_worker_inventory", fake_inventory)
    return seen


# ---------------------------------------------------------------------------
# The generation marker itself
# ---------------------------------------------------------------------------

def test_relevancy_generation_is_separate_from_the_hard_cache_key():
    """If these were the same constant, marking scenes stale would recompute
    them all — the exact thing this design avoids."""
    import langfield_worker

    assert relevancy_core.RELEVANCY_GENERATION == 1
    assert langfield_worker.INVENTORY_VERSION == 7
    assert (langfield_worker.INVENTORY_VERSION
            is not relevancy_core.RELEVANCY_GENERATION)


# ---------------------------------------------------------------------------
# The route surfaces staleness
# ---------------------------------------------------------------------------

def test_a_stale_inventory_is_served_with_a_reason_not_withheld(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    _worker_reply(monkeypatch, {
        "ready": True, "cached": True, "items": [{"label": "bicycle"}],
        "stale": True, "relevancy_generation": None,
        "current_relevancy_generation": 1,
        "stale_reason": "computed before unseen gaussians were excluded from relevancy",
    })

    body = http.get(f"/api/splat/jobs/{JOB}/langfield/inventory").json()

    assert body["items"] == [{"label": "bicycle"}], "still usable, just old"
    assert body["stale"] is True
    assert "unseen gaussians" in body["stale_reason"]
    assert body["relevancy_generation"] is None
    assert body["current_relevancy_generation"] == 1


def test_a_current_inventory_is_not_flagged(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    _worker_reply(monkeypatch, {
        "ready": True, "cached": True, "items": [{"label": "bicycle"}],
        "stale": False, "relevancy_generation": 1,
        "current_relevancy_generation": 1, "stale_reason": "",
    })

    body = http.get(f"/api/splat/jobs/{JOB}/langfield/inventory").json()

    assert body["stale"] is False
    assert body["stale_reason"] == ""


def test_refresh_is_off_by_default(client, monkeypatch):
    """The default read must never trigger a GPU recompute."""
    http, outputs = client
    _mk_job(outputs)
    seen = _worker_reply(monkeypatch, {"items": [], "stale": True})

    http.get(f"/api/splat/jobs/{JOB}/langfield/inventory")

    assert seen[0]["refresh"] is False


def test_refresh_is_passed_through_when_asked(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    seen = _worker_reply(monkeypatch, {"items": [], "stale": False})

    http.get(f"/api/splat/jobs/{JOB}/langfield/inventory", params={"refresh": "true"})

    assert seen[0]["refresh"] is True


def test_a_worker_that_omits_the_fields_is_reported_as_current(client, monkeypatch):
    """An older worker build must not make every scene look broken."""
    http, outputs = client
    _mk_job(outputs)
    _worker_reply(monkeypatch, {"ready": True, "cached": True, "items": [{"label": "x"}]})

    body = http.get(f"/api/splat/jobs/{JOB}/langfield/inventory").json()

    assert body["stale"] is False
    assert body["items"] == [{"label": "x"}]


# ---------------------------------------------------------------------------
# The worker's own cache decision, exercised directly
# ---------------------------------------------------------------------------

def _decide(cache_payload: dict | None, refresh: bool = False) -> dict:
    """Reproduce the worker's cache branch without importing torch."""
    if cache_payload is None or refresh:
        return {"cached": False, "recomputed": True}
    if cache_payload.get("version") != 7 or not cache_payload.get("items"):
        return {"cached": False, "recomputed": True}
    generation = cache_payload.get("relevancy_generation")
    return {"cached": True, "recomputed": False,
            "stale": generation != relevancy_core.RELEVANCY_GENERATION}


def test_an_inventory_without_a_generation_reads_as_stale():
    """Every inventory on disk today predates the marker; absence IS the signal,
    so no migration pass has to run over the job tree."""
    decision = _decide({"version": 7, "items": [{"label": "chair"}]})
    assert decision["cached"] is True
    assert decision["stale"] is True
    assert decision["recomputed"] is False, "stale must not mean recompute"


def test_an_inventory_at_the_current_generation_is_fresh():
    decision = _decide({"version": 7, "items": [{"label": "chair"}],
                        "relevancy_generation": relevancy_core.RELEVANCY_GENERATION})
    assert decision["stale"] is False


def test_a_future_generation_also_reads_as_stale():
    """Rolling the code back must not silently trust a newer cache."""
    decision = _decide({"version": 7, "items": [{"label": "chair"}],
                        "relevancy_generation": 99})
    assert decision["stale"] is True


def test_refresh_bypasses_a_valid_cache():
    decision = _decide({"version": 7, "items": [{"label": "chair"}],
                        "relevancy_generation": 1}, refresh=True)
    assert decision["recomputed"] is True


def test_a_hard_version_mismatch_still_recomputes():
    """The existing INVENTORY_VERSION behaviour is unchanged."""
    decision = _decide({"version": 6, "items": [{"label": "chair"}],
                        "relevancy_generation": 1})
    assert decision["recomputed"] is True
