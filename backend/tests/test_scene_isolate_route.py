"""Batch isolation route (P6c): requires a built scene inventory, loops the
subprocess once (batch_isolate.py claims all instances + background in one
process), meta bookkeeping, and slug-safe file serving. Subprocess mocked;
the claim/sanity mechanism itself is proven by the isolate-probe pattern
(sanity_sum_ok) this script reuses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

BATCH_REPORT = {
    "n_gaussians": 500_000,
    "instances": [
        {"slug": "fire-hydrant", "label": "fire hydrant", "n_members_original": 28627,
         "n_overlap_removed": 0, "n_members_final": 28627, "status": "built"},
        {"slug": "tiny-pebble", "label": "tiny pebble", "n_members_original": 40,
         "n_overlap_removed": 0, "status": "SKIPPED:too-few-members-after-dedup"},
    ],
    "sanity": {"n_gaussians": 500_000, "n_claimed": 28627, "n_background": 471373,
               "sanity_sum_ok": True},
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job_with_inventory(outputs: Path, job_id: str = "splat_0b0003") -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    cfg = job_dir / "processed" / "splatfacto" / "ts"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")
    scene_dir = job_dir / splat_route.SCENE_DIRNAME
    scene_dir.mkdir()
    (scene_dir / "inventory.json").write_text(json.dumps({
        "instances": [{"id": 0, "label": "fire hydrant", "slug": "fire-hydrant",
                       "n_members": 28627}],
    }))
    return job_dir


def _fake_subprocess(calls: list):
    async def run(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "batch_isolate" in joined:
            out_dir = Path(command[4])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "fire-hydrant").mkdir(parents=True, exist_ok=True)
            (out_dir / "fire-hydrant" / "object.ply").write_bytes(b"ply")
            (out_dir / "background.ply").write_bytes(b"ply")
            (out_dir / "receipt_background_cam_000.png").write_bytes(b"png")
            (out_dir / "batch_isolate.json").write_text(json.dumps(BATCH_REPORT))
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_scene_isolate_requires_inventory(client):
    http, outputs = client
    job_dir = outputs / "splat_0b0003"
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": "splat_0b0003", "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    r = http.post("/api/splat/jobs/splat_0b0003/scene/isolate", json={})
    assert r.status_code == 409
    assert "inventory" in r.json()["detail"].lower()


def test_scene_isolate_full_build(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job_with_inventory(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0003/scene/isolate", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["sanity"]["sanity_sum_ok"] is True
    built = [i for i in body["instances"] if i["status"] == "built"]
    skipped = [i for i in body["instances"] if i["status"] != "built"]
    assert len(built) == 1 and len(skipped) == 1

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["isolate"]["n_built"] == 1
    assert meta["scene"]["isolate"]["n_skipped"] == 1

    for fmt, extra in (("report", {}), ("background", {}),
                       ("object", {"name": "fire-hydrant"}), ("receipt", {})):
        resp = http.get("/api/splat/jobs/splat_0b0003/scene/isolate/file",
                        params={"fmt": fmt, **extra})
        assert resp.status_code == 200, fmt


def test_scene_isolate_recall_expand_requires_langfield(client):
    http, outputs = client
    _mk_job_with_inventory(outputs)  # no _langfield dir
    r = http.post("/api/splat/jobs/splat_0b0003/scene/isolate", json={"recall_expand": True})
    assert r.status_code == 409
    assert "language field" in r.json()["detail"].lower()


def test_scene_isolate_recall_expand_passes_flags(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job_with_inventory(outputs)
    lf = job_dir / splat_route.LANGFIELD_DIRNAME
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"npz")
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0003/scene/isolate",
                  json={"recall_expand": True, "dilation_mult": 15.0, "rel_floor": 0.25})
    assert r.status_code == 200
    joined = " ".join(str(c) for c in calls[0])
    assert "--recall-expand" in joined
    assert "--dilation-mult 15.0" in joined
    assert "--rel-floor 0.25" in joined
    assert str(lf / "gauss_emb.npz") in joined


def test_scene_isolate_object_path_traversal_404(client):
    http, outputs = client
    _mk_job_with_inventory(outputs)
    r = http.get("/api/splat/jobs/splat_0b0003/scene/isolate/file",
                 params={"fmt": "object", "name": "../../etc/passwd"})
    assert r.status_code == 404


def test_scene_isolate_file_404_before_build(client):
    http, outputs = client
    _mk_job_with_inventory(outputs)
    r = http.get("/api/splat/jobs/splat_0b0003/scene/isolate/file", params={"fmt": "background"})
    assert r.status_code == 404
