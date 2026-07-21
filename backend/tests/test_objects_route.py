"""Object isolation route (P5b): loud guards (language field required), the
three-step build with per-step honesty, subset scratch cleanup, meta bookkeeping,
and slug-safe file serving. Subprocesses mocked; the real chain is proven by the
garden-table probe (LCC 99.4% top / full table at 1.6/0.30).
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

OBJ_REPORT = {
    "query": "round wooden table", "cluster": 0, "clusters_found": 1,
    "pool_members": 1801, "expanded_members": 43984,
    "bbox_scene": {"min": [-0.3, -0.4, -0.9], "max": [0.5, 0.5, -0.2]},
    "artifacts": {"splat": "object.ply", "indices": "object_indices.npz"},
}
MESH_REPORT = {"v": 1, "tris": 109232, "lcc_pct": 99.1,
               "recipe": {"checkpoint": "vanilla"}, "artifacts": {"ply": "mesh.ply"}}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: True)

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0b0001", langfield: bool = True) -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    if langfield:
        lf = job_dir / splat_route.LANGFIELD_DIRNAME
        lf.mkdir()
        (lf / "gauss_emb.npz").write_bytes(b"npz")
        cfg = job_dir / "processed" / "splatfacto" / "ts"
        cfg.mkdir(parents=True)
        (cfg / "config.yml").write_text("cfg")
    return job_dir


def _fake_subprocess(job_dir: Path, calls: list):
    async def run(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "object_isolate" in joined:
            obj_dir = Path(command[5])
            (obj_dir / "object.json").write_text(json.dumps(OBJ_REPORT))
            (obj_dir / "object.ply").write_bytes(b"ply")
            (obj_dir / "object_indices.npz").write_bytes(b"npz")
            return 0, b"", b""
        if "checkpoint_subset" in joined:
            subset = Path(command[3])
            cfg = subset / "processed" / "splatfacto" / "ts"
            cfg.mkdir(parents=True)
            (cfg / "config.yml").write_text("cfg")
            return 0, f"kept\n{cfg / 'config.yml'}\n".encode(), b""
        if "run_mesh.sh" in joined:
            assert "MESH_MIN_COMPONENT_FRAC=0.05" in command
            mesh_dir = Path(command[-1])
            mesh_dir.mkdir(parents=True, exist_ok=True)
            (mesh_dir / "mesh.ply").write_bytes(b"ply")
            (mesh_dir / "mesh.glb").write_bytes(b"glb")
            (mesh_dir / "view_ext0.png").write_bytes(b"png")
            (mesh_dir / "mesh.json").write_text(json.dumps(MESH_REPORT))
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_objects_requires_langfield(client):
    http, outputs = client
    _mk_job(outputs, langfield=False)
    r = http.post("/api/splat/jobs/splat_0b0001/objects", json={"query": "table"})
    assert r.status_code == 409
    assert "language field" in r.json()["detail"]


def test_objects_full_build(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))

    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "Round Wooden Table!"})
    assert r.status_code == 200
    body = r.json()
    assert body["slug"] == "round-wooden-table"
    assert body["object"]["mesh"]["lcc_pct"] == 99.1
    assert body["mesh_glb_url"]

    # object-mesh voxel scaled from the bbox diagonal, within clamps
    mesh_call = " ".join(str(c) for c in calls[2])
    assert "MESH_VOXEL_SIZE=0.0063" in mesh_call

    # subset scratch checkpoint removed after a successful mesh
    assert not (job_dir / splat_route.OBJECTS_DIRNAME / "round-wooden-table" / "subset").exists()

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["objects"]["round-wooden-table"]["expanded_members"] == 43984

    for fmt in ("splat", "ply", "glb", "receipt"):
        assert http.get(f"/api/splat/jobs/splat_0b0001/objects/round-wooden-table/file?fmt={fmt}").status_code == 200, fmt


def test_objects_splat_only(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "mesh": False})
    assert r.status_code == 200
    assert len(calls) == 1  # isolation only
    assert r.json()["mesh_ply_url"] is None


def test_objects_slug_path_traversal_404(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0001/objects/..%2F..%2Fetc/file?fmt=splat")
    assert r.status_code == 404


def test_objects_proxy_chain(client, monkeypatch, tmp_path):
    http, outputs = client
    job_dir = _mk_job(outputs)
    monkeypatch.setattr(splat_route, "_triposplat_availability",
                        lambda: {"triposplat_available": True, "triposplat_runner": "/fake/run.sh"})
    calls: list = []
    base_fake = _fake_subprocess(job_dir, calls)

    async def run(command):
        joined = " ".join(str(c) for c in command)
        if "object_crop" in joined:
            calls.append(command)
            Path(command[4]).write_bytes(b"png")
            return 0, b"", b""
        if "/fake/run.sh" in joined:
            calls.append(command)
            raw = Path(command[3])
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "splat.ply").write_bytes(b"ply")
            (raw / "thumb.webp").write_bytes(b"webp")
            return 0, b"", b""
        if "proxy_register" in joined:
            calls.append(command)
            obj_dir = Path(command[4]).parent
            (obj_dir / "proxy.ply").write_bytes(b"ply")
            (obj_dir / "proxy.json").write_text(json.dumps({"icp_fitness": 0.91, "total_scale": 0.76}))
            return 0, b"", b""
        return await base_fake(command)

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", run)
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "proxy": True})
    assert r.status_code == 200
    body = r.json()
    assert body["proxy_url"]
    assert body["proxy_preview_url"]
    assert body["object"]["proxy"]["icp_fitness"] == 0.91

    obj_dir = job_dir / splat_route.OBJECTS_DIRNAME / "table"
    assert not (obj_dir / "proxy_raw").exists()      # scratch cleaned
    assert (obj_dir / "proxy_preview.webp").is_file()

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["objects"]["table"]["proxy"]["total_scale"] == 0.76

    for fmt in ("proxy", "proxy-preview"):
        assert http.get(f"/api/splat/jobs/splat_0b0001/objects/table/file?fmt={fmt}").status_code == 200, fmt


def test_query_leading_dash_rejected(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0b0001/objects", json={"query": "--help"})
    assert r.status_code == 422


def test_stale_langfield_refused(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    (job_dir / splat_route.LANGFIELD_DIRNAME / "STALE").write_text("edited")
    r = http.post("/api/splat/jobs/splat_0b0001/objects", json={"query": "table"})
    assert r.status_code == 409
    assert "stale" in r.json()["detail"].lower()


def test_proxy_unavailable_is_preflight_400(client, monkeypatch):
    """Review fix: TripoSplat-unavailable must fail BEFORE any GPU work."""
    http, outputs = client
    _mk_job(outputs)
    monkeypatch.setattr(splat_route, "_triposplat_availability",
                        lambda: {"triposplat_available": False, "triposplat_runner": ""})

    async def fake_sub(command):
        raise AssertionError("no subprocess may run")

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_0b0001/objects", json={"query": "table", "proxy": True})
    assert r.status_code == 400


def test_subset_scratch_cleaned_on_mesh_failure(client, monkeypatch):
    """Review fix: a failed object-mesh build must not orphan the ~300MB
    subset checkpoint."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    base_fake = _fake_subprocess(job_dir, calls)

    async def run(command):
        joined = " ".join(str(c) for c in command)
        if "run_mesh.sh" in joined:
            return 1, b"", b"FATAL: gs-mesh crashed"
        return await base_fake(command)

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", run)
    r = http.post("/api/splat/jobs/splat_0b0001/objects", json={"query": "table"})
    assert r.status_code == 500
    assert not (job_dir / splat_route.OBJECTS_DIRNAME / "table" / "subset").exists()
