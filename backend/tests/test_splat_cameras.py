from __future__ import annotations

import json
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


def test_camera_endpoint_applies_dataparser_transform_and_sampling(client: tuple[TestClient, Path]):
    http, outputs = client
    job_id = "splat_abcdef"
    job_dir = outputs / job_id
    processed = job_dir / "processed"
    run_dir = processed / "splatfacto" / "2026-07-01_000000"
    run_dir.mkdir(parents=True)

    (job_dir / "meta.json").write_text(json.dumps({"job_id": job_id, "output_dir": str(job_dir)}))
    (processed / "transforms.json").write_text(json.dumps({
        "w": 100,
        "h": 50,
        "frames": [
            {
                "file_path": "images/a.jpg",
                "transform_matrix": [
                    [1, 0, 0, 10],
                    [0, 1, 0, 4],
                    [0, 0, 1, 6],
                    [0, 0, 0, 1],
                ],
            },
            {
                "file_path": "images/b.jpg",
                "transform_matrix": [
                    [1, 0, 0, 12],
                    [0, 1, 0, 4],
                    [0, 0, 1, 6],
                    [0, 0, 0, 1],
                ],
            },
        ],
        "applied_transform": [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
    }))
    (run_dir / "dataparser_transforms.json").write_text(json.dumps({
        # Combined original-data -> output transform. The endpoint must undo
        # applied_transform before using it with saved transform_matrix poses.
        "transform": [
            [1, 0, 0, 1],
            [0, 0, 1, 2],
            [0, 1, 0, 3],
        ],
        "scale": 0.5,
    }))

    response = http.get(f"/api/splat/jobs/{job_id}/cameras", params={"limit": 1})
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["count"] == 1
    assert payload["total"] == 2
    assert payload["sampled"] is True
    assert payload["frame"] == "viewer"
    assert payload["source"] == "dataparser_transforms"
    assert payload["image_size"] == {"width": 100, "height": 50}

    camera = payload["cameras"][0]
    assert camera["image_name"] == "a.jpg"
    assert camera["position"] == [5.5, 3.0, 4.5]
    assert camera["forward"] == [0.0, 0.0, -1.0]
