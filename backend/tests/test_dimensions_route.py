"""Survey dimensions (dimensions_route.py): CRUD + CSV/JSON export.

Server-persisted so measurements survive across sessions/devices (previously
sessionStorage-only in spark-scene-viewer.tsx) and can be shared as a real
file. POST is an idempotent upsert keyed on the client-supplied id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dimensions_route  # noqa: E402
import splat_route  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(dimensions_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0d1001") -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({"job_id": job_id, "output_dir": str(job_dir), "status": "completed"}))
    return job_dir


DIM_A = {"id": "1721700000000", "a": [0.1, 0.2, 0.3], "b": [1.1, 0.2, 0.3], "label": "curb"}


def test_dimensions_empty_list(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0d1001/dimensions")
    assert r.status_code == 200
    assert r.json()["dimensions"] == []


def test_dimensions_add_and_list(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0d1001/dimensions", json=DIM_A)
    assert r.status_code == 200
    rec = r.json()["dimension"]
    assert rec["id"] == DIM_A["id"] and rec["a"] == DIM_A["a"] and rec["label"] == "curb"
    assert rec["created_at"]

    r2 = http.get("/api/splat/jobs/splat_0d1001/dimensions")
    assert len(r2.json()["dimensions"]) == 1

    on_disk = json.loads((job_dir / "dimensions.json").read_text())
    assert on_disk[0]["id"] == DIM_A["id"]


def test_dimensions_post_is_upsert_by_id(client):
    """Drag-to-move re-POSTs the same id with updated coordinates — must
    replace, not duplicate."""
    http, outputs = client
    _mk_job(outputs)
    http.post("/api/splat/jobs/splat_0d1001/dimensions", json=DIM_A)
    moved = {**DIM_A, "b": [2.0, 2.0, 2.0]}
    r = http.post("/api/splat/jobs/splat_0d1001/dimensions", json=moved)
    assert r.status_code == 200

    items = http.get("/api/splat/jobs/splat_0d1001/dimensions").json()["dimensions"]
    assert len(items) == 1
    assert items[0]["b"] == [2.0, 2.0, 2.0]
    # created_at preserved across the upsert, not reset
    assert items[0]["created_at"] == r.json()["dimension"]["created_at"]


def test_dimensions_delete(client):
    http, outputs = client
    _mk_job(outputs)
    http.post("/api/splat/jobs/splat_0d1001/dimensions", json=DIM_A)
    r = http.delete(f"/api/splat/jobs/splat_0d1001/dimensions/{DIM_A['id']}")
    assert r.status_code == 200
    assert http.get("/api/splat/jobs/splat_0d1001/dimensions").json()["dimensions"] == []


def test_dimensions_delete_unknown_404(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.delete("/api/splat/jobs/splat_0d1001/dimensions/nope")
    assert r.status_code == 404


def test_dimensions_unknown_job_404(client):
    http, _outputs = client
    r = http.get("/api/splat/jobs/splat_missing/dimensions")
    assert r.status_code == 404


def test_dimensions_export_csv(client):
    http, outputs = client
    _mk_job(outputs)
    http.post("/api/splat/jobs/splat_0d1001/dimensions", json=DIM_A)
    r = http.get("/api/splat/jobs/splat_0d1001/dimensions/export", params={"fmt": "csv"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    body = r.text
    assert "curb" in body and DIM_A["id"] in body
    # length column = |a-b| = 1.0 for DIM_A
    assert "1.0" in body


def test_dimensions_export_json(client):
    http, outputs = client
    _mk_job(outputs)
    http.post("/api/splat/jobs/splat_0d1001/dimensions", json=DIM_A)
    r = http.get("/api/splat/jobs/splat_0d1001/dimensions/export", params={"fmt": "json"})
    assert r.status_code == 200
    assert r.json()["dimensions"][0]["id"] == DIM_A["id"]


def test_dimensions_export_bad_fmt_400(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0d1001/dimensions/export", params={"fmt": "xml"})
    assert r.status_code == 400
