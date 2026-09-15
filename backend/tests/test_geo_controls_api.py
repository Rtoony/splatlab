import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import geo_route
import splat_route


@pytest.fixture
def client(tmp_path, monkeypatch):
    job_id = "splat_c011ab"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", tmp_path)
    job_dir = tmp_path / job_id
    job_dir.mkdir()
    (job_dir / "meta.json").write_text(json.dumps({"job_id": job_id, "status": "completed", "output_dir": str(job_dir), "geo": {"alt_m": 31}}))
    app = FastAPI()
    app.include_router(geo_route.router)
    return TestClient(app), job_dir


CONTROLS = [{"scene": [0, 0], "lat": 38, "lon": -122},
            {"scene": [10, 0], "lat": 38, "lon": -121.9999},
            {"scene": [0, 10], "lat": 38.00008, "lon": -122}]


def test_propose_is_read_only_apply_checks_generation_and_keeps_altitude(client):
    http, job_dir = client
    before = (job_dir / "meta.json").read_bytes()
    preview = http.post("/jobs/splat_c011ab/geo/controls/propose", json={"controls": CONTROLS})
    assert preview.status_code == 200
    assert preview.json()["base_scale_generation"] == 0
    assert (job_dir / "meta.json").read_bytes() == before
    applied = http.post("/jobs/splat_c011ab/geo/controls/apply", json={"controls": CONTROLS})
    assert applied.status_code == 200
    assert applied.json()["geo"]["alt_m"] == 31
    assert applied.json()["scale_generation"] == 1
    assert (job_dir / "_geo" / "alignment-before-0.json").is_file()
    assert http.post("/jobs/splat_c011ab/geo/controls/apply", json={"controls": CONTROLS}).status_code == 409


def test_degenerate_and_incomplete_controls_refuse_without_writes(client):
    http, job_dir = client
    for controls in (CONTROLS[:1], [CONTROLS[0], CONTROLS[0]]):
        assert http.post("/jobs/splat_c011ab/geo/controls/apply", json={"controls": controls}).status_code == 422
    assert not (job_dir / "_geo").exists()
