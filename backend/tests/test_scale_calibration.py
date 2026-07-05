"""Survey-lane scale calibration endpoint: POST /jobs/{id}/scale.

nerfstudio scenes are non-metric; meters_per_unit is the single stored bridge
to real-world distances, so it must round-trip through meta.json, reach the
status payload via the **meta spread, reject garbage, and clear with null.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path]:
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_5ca1e0") -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(
        json.dumps({"job_id": job_id, "output_dir": str(job_dir), "status": "completed"})
    )
    return job_dir


def test_scale_set_persists_and_flows_to_payload(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_5ca1e0/scale", json={"meters_per_unit": 0.3125})
    assert r.status_code == 200
    assert r.json()["meters_per_unit"] == 0.3125
    # persisted in meta.json
    assert json.loads((job_dir / "meta.json").read_text())["meters_per_unit"] == 0.3125
    # reaches the status payload via the **meta spread
    payload = splat_route._job_payload(json.loads((job_dir / "meta.json").read_text()))
    assert payload["meters_per_unit"] == 0.3125


def test_scale_null_clears(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    http.post("/api/splat/jobs/splat_5ca1e0/scale", json={"meters_per_unit": 2.0})
    r = http.post("/api/splat/jobs/splat_5ca1e0/scale", json={"meters_per_unit": None})
    assert r.status_code == 200 and r.json()["meters_per_unit"] is None
    assert json.loads((job_dir / "meta.json").read_text())["meters_per_unit"] is None


@pytest.mark.parametrize("bad", ["banana", 0, -1.5, float("nan"), float("inf"), 1e9])
def test_scale_rejects_garbage(client, bad):
    http, outputs = client
    _mk_job(outputs)
    if isinstance(bad, float) and not math.isfinite(bad):
        # httpx's own request encoder refuses to serialize NaN/Infinity
        # (RFC 8259 has no such literals — httpx calls json.dumps(...,
        # allow_nan=False)), so http.post(json={...}) would raise
        # client-side before the request ever reaches the server, never
        # exercising the endpoint's own rejection at all. A real client can
        # still send these as literal JSON tokens (Python's json.loads
        # accepts them as an extension on the decode side, with no
        # allow_nan-equivalent restriction) — send raw bytes to actually
        # test the server's math.isfinite() check.
        body = json.dumps({"meters_per_unit": bad}, allow_nan=True).encode()
        r = http.post(
            "/api/splat/jobs/splat_5ca1e0/scale",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    else:
        r = http.post("/api/splat/jobs/splat_5ca1e0/scale", json={"meters_per_unit": bad})
    assert r.status_code == 400, f"{bad!r} must be rejected"


def test_scale_unknown_job_404(client):
    http, _ = client
    r = http.post("/api/splat/jobs/splat_00dead/scale", json={"meters_per_unit": 1.0})
    assert r.status_code == 404
