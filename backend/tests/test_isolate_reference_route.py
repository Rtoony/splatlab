"""R5.1 isolate-by-reference route (2026-09-15): completed-job + checkpoint
preconditions, the orchestrator is invoked with --no-gate under the route's own
arbiter lease, meta bookkeeping per slug, and slug-safe file serving. Subprocess
mocked; the mechanism itself is proven by backend/tests/test_isolate_orbits.py and
the bonsai / red-bicycle receipts under ~/reports/2026-09-14-splatlab-research-sweep/.
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

RECEIPT = {"n_object": 8826, "n_seed": 8530, "seed_retained": 0.88, "mask_iou_mean": 0.86, "mask_iou_min": 0.7,
           "reground": {"used": True, "view": "pull_az+0_el+0_s1.00"}, "timing_s": {"fit": 23.4, "total": 61.0}}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    tool = tmp_path / "isolate-by-reference.py"; tool.write_text("# stub")
    py = tmp_path / "python"; py.write_text("# stub")
    monkeypatch.setattr(splat_route, "ISOLATE_REFERENCE_SCRIPT", tool)
    monkeypatch.setattr(splat_route, "LANGFIELD_ENV_PYTHON", py)
    monkeypatch.setattr(splat_route, "SAM3_ENV_PYTHON", py)

    async def fake_audit(**kwargs):
        return None
    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        assert lane == "isolate-reference" and vram_mb == splat_route.ISOLATE_REFERENCE_VRAM_MB
        return await operation()
    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0b0004", status: str = "completed", checkpoint: bool = True) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": status, "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    if checkpoint:
        cfg = job_dir / "processed" / "splatfacto" / "ts"
        cfg.mkdir(parents=True)
        (cfg / "config.yml").write_text("cfg")
    return job_dir


def _fake_subprocess(calls: list):
    async def run(command):
        calls.append(command)
        args = [str(c) for c in command]
        job_dir = Path(args[args.index("--job") + 1]); concept = args[args.index("--concept") + 1]
        work = job_dir / "_isolate" / splat_route._isolate_slug(concept)
        work.mkdir(parents=True, exist_ok=True)
        (work / "receipt.json").write_text(json.dumps({**RECEIPT, "concept": concept}))
        (work / "object.ply").write_bytes(b"ply"); (work / "object_indices.npz").write_bytes(b"npz")
        (work / "receipt_object.png").write_bytes(b"png")
        return 0, b"", b""
    return run


def test_isolate_reference_requires_completed_job_and_checkpoint(client):
    http, outputs = client
    _mk_job(outputs, "splat_0b0005", status="processing")
    r = http.post("/api/splat/jobs/splat_0b0005/isolate/reference", json={"concept": "bonsai"})
    assert r.status_code == 409 and "completed" in r.json()["detail"]
    _mk_job(outputs, "splat_0b0006", checkpoint=False)
    r = http.post("/api/splat/jobs/splat_0b0006/isolate/reference", json={"concept": "bonsai"})
    assert r.status_code == 409 and "checkpoint" in r.json()["detail"]
    r = http.post("/api/splat/jobs/splat_0b0006/isolate/reference", json={})
    assert r.status_code == 422


def test_isolate_reference_runs_orchestrator_under_lease_and_records_meta(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0004/isolate/reference", json={"concept": "Red Bicycle", "iters": 80})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["slug"] == "red-bicycle" and body["n_object"] == 8826 and body["reground"]["used"] is True
    args = [str(c) for c in calls[0]]
    assert args[1] == str(splat_route.ISOLATE_REFERENCE_SCRIPT)
    assert "--no-gate" in args and "--no-reground" not in args          # the route's arbiter lease is the claim
    assert args[args.index("--concept") + 1] == "Red Bicycle" and args[args.index("--iters") + 1] == "80"
    assert args[args.index("--job") + 1] == str(job_dir)

    meta = json.loads((job_dir / "meta.json").read_text())
    rec = meta["isolate_reference"]["red-bicycle"]
    assert rec["n_object"] == 8826 and rec["reground_used"] is True and rec["seconds"] == 61.0 and rec["built_at"]

    for fmt in ("report", "object", "indices", "receipt"):
        resp = http.get("/api/splat/jobs/splat_0b0004/isolate/reference/file", params={"fmt": fmt, "slug": "red-bicycle"})
        assert resp.status_code == 200, fmt


def test_isolate_reference_no_reground_flag_and_second_slug_keeps_first(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))
    assert http.post("/api/splat/jobs/splat_0b0004/isolate/reference", json={"concept": "bonsai"}).status_code == 200
    r = http.post("/api/splat/jobs/splat_0b0004/isolate/reference", json={"concept": "tree stump", "reground": False})
    assert r.status_code == 200
    assert "--no-reground" in [str(c) for c in calls[1]]
    meta = json.loads((job_dir / "meta.json").read_text())
    assert set(meta["isolate_reference"]) == {"bonsai", "tree-stump"}


def test_isolate_reference_failure_surfaces_stderr_tail(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)

    async def failing(command):
        return 1, b"", b"...\nSystemExit: no usable SAM3 detection for 'unicorn'\n"
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", failing)
    r = http.post("/api/splat/jobs/splat_0b0004/isolate/reference", json={"concept": "unicorn"})
    assert r.status_code == 500 and "no usable SAM3 detection" in r.json()["detail"]


def test_isolate_reference_file_slug_traversal_and_missing(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0004/isolate/reference/file", params={"fmt": "object", "slug": "../../etc/passwd"})
    assert r.status_code == 404
    r = http.get("/api/splat/jobs/splat_0b0004/isolate/reference/file", params={"fmt": "object", "slug": "bonsai"})
    assert r.status_code == 404
