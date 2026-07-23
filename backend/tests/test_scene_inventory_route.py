"""Scene instance inventory route (P6b): loud guards (language field required,
SAM3 preflight before any GPU work), the four-step build (views -> noun
consolidation -> SAM3 masks -> lift), the all-stuff short-circuit, explicit
`nouns` override bypassing auto-sourcing, and slug-safe file serving.
Subprocesses mocked; the lift mechanism itself is proven by the Step 0
de-risk spike (~/tools/scene-regen-spike/STATUS.md, mechanism PROVEN GO).
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

INSTANCES_REPORT = {
    "n_gaussians": 500_000, "views": [0, 10, 20], "things": ["fire hydrant"],
    "params": {"jaccard": 0.25, "vote_frac": 0.3, "depth_tol": 0.07,
               "min_members": 200, "min_views": 2},
    "instances": [{
        "id": 0, "label": "fire hydrant", "slug": "fire-hydrant",
        "n_members": 28000, "n_views": 3, "views_seen": [0, 10, 20],
        "vote_threshold": 1, "mean_score": 0.91,
        "centroid_scene": [0.0, 0.0, -0.3],
        "bbox_tight_scene": {"min": [-0.1, -0.1, -0.5], "max": [0.1, 0.1, -0.1]},
        "best_view": 10, "best_view_box": [100, 100, 300, 400],
    }],
    "vetoed": [],
    "conservation": {"n_gaussians": 500_000, "n_claimed": 28000,
                      "n_remainder": 472000, "n_overlap": 0, "holds": True},
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


def _mk_job(outputs: Path, job_id: str = "splat_0b0002", langfield: bool = True) -> Path:
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


def _fake_subprocess(calls: list, things=("fire hydrant",), stuff=()):
    async def run(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "sam3_doctor" in joined:
            return 0, b"sam3_doctor: healthy", b""
        if "scene_views" in joined:
            workdir = Path(command[3])
            frames = workdir / "frames"
            frames.mkdir(parents=True, exist_ok=True)
            for i in (0, 10, 20):
                (frames / f"cam_{i:03d}.png").write_bytes(b"png")
            (workdir / "views.json").write_text(json.dumps(
                {"cam_indices": [0, 10, 20], "W": 640, "H": 480, "n_train": 49}))
            return 0, b"", b""
        if "noun_consolidate" in joined:
            nouns_path = Path(command[3])
            nouns_path.write_text(json.dumps({
                "raw_count": len(things) + len(stuff), "cleaned_count": len(things) + len(stuff),
                "dedup_map": {}, "things": list(things), "stuff": list(stuff),
            }))
            return 0, b"", b""
        if "scene_sam3_masks" in joined:
            return 0, b"", b""
        if "instance_lift" in joined:
            scene_dir = Path(command[5])
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "instances.json").write_text(json.dumps(INSTANCES_REPORT))
            (scene_dir / "crop_fire-hydrant.png").write_bytes(b"png")
            (scene_dir / "receipt_overlay_cam_010.png").write_bytes(b"png")
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_scene_inventory_requires_langfield(client):
    http, outputs = client
    _mk_job(outputs, langfield=False)
    r = http.post("/api/splat/jobs/splat_0b0002/scene/inventory", json={"nouns": ["hydrant"]})
    assert r.status_code == 409
    assert "language field" in r.json()["detail"]


def test_scene_inventory_explicit_nouns_full_build(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    async def unreachable(*a, **kw):
        raise AssertionError("auto-sourcing must not run when nouns is explicit")

    monkeypatch.setattr(splat_route, "_qwen_vl_nouns", unreachable)
    monkeypatch.setattr(splat_route, "_langfield_worker_inventory", unreachable)

    r = http.post("/api/splat/jobs/splat_0b0002/scene/inventory",
                  json={"nouns": ["fire hydrant"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["instances"]) == 1
    assert body["instances"][0]["slug"] == "fire-hydrant"
    assert body["conservation"]["holds"] is True

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["inventory"]["n_instances"] == 1

    # scratch workdir cleaned; scene artifacts remain
    assert not (job_dir / splat_route.SCENE_DIRNAME / "_work").exists()
    assert (job_dir / splat_route.SCENE_DIRNAME / "inventory.json").is_file()

    for fmt, extra in (("report", {}), ("crop", {"name": "fire-hydrant"}),
                       ("overlay", {})):
        params = {"fmt": fmt, **extra}
        resp = http.get(f"/api/splat/jobs/splat_0b0002/scene/inventory/file", params=params)
        assert resp.status_code == 200, fmt


def test_scene_inventory_auto_sourcing_merges_vl_and_langfield(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    calls: list = []
    seen_raw = {}

    async def fake_sub(command):
        joined = " ".join(str(c) for c in command)
        if "noun_consolidate" in joined:
            raw_path = Path(command[2])
            seen_raw["nouns"] = json.loads(raw_path.read_text())
        return await _fake_subprocess(calls)(command)

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)

    async def fake_vl(frame_paths):
        assert len(frame_paths) == 3
        return ["fire hydrant"]

    async def fake_lf(config_path, lfdir):
        return {"items": [{"label": "ground disc"}]}

    monkeypatch.setattr(splat_route, "_qwen_vl_nouns", fake_vl)
    monkeypatch.setattr(splat_route, "_langfield_worker_inventory", fake_lf)

    r = http.post("/api/splat/jobs/splat_0b0002/scene/inventory", json={})
    assert r.status_code == 200
    assert seen_raw["nouns"] == ["fire hydrant", "ground disc"]


def test_scene_inventory_all_stuff_short_circuits_before_sam3(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(
        splat_route, "_run_capture_subprocess",
        _fake_subprocess(calls, things=(), stuff=("grass lawn",)),
    )
    r = http.post("/api/splat/jobs/splat_0b0002/scene/inventory",
                  json={"nouns": ["grass lawn"]})
    assert r.status_code == 200
    body = r.json()
    assert body["instances"] == []
    assert any("stuff" in v["reason"] for v in body["vetoed"])
    assert not any("scene_sam3_masks" in " ".join(str(c) for c in call) for call in calls)
    assert not any("instance_lift" in " ".join(str(c) for c in call) for call in calls)

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["inventory"]["n_instances"] == 0


def test_scene_inventory_sam3_unhealthy_is_preflight_400(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)

    async def fake_sub(command):
        joined = " ".join(str(c) for c in command)
        if "sam3_doctor" in joined:
            return 1, b"", b"FAIL bpe tokenizer asset"
        raise AssertionError(f"no other subprocess may run: {command}")

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_0b0002/scene/inventory",
                  json={"nouns": ["fire hydrant"]})
    assert r.status_code == 400


def test_scene_inventory_stale_langfield_refused(client):
    http, outputs = client
    job_dir = _mk_job(outputs)
    (job_dir / splat_route.LANGFIELD_DIRNAME / "STALE").write_text("edited")
    r = http.post("/api/splat/jobs/splat_0b0002/scene/inventory", json={"nouns": ["x"]})
    assert r.status_code == 409
    assert "stale" in r.json()["detail"].lower()


def test_scene_inventory_file_404_before_build(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0002/scene/inventory/file", params={"fmt": "report"})
    assert r.status_code == 404


def test_scene_inventory_crop_path_traversal_404(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0002/scene/inventory/file",
                 params={"fmt": "crop", "name": "../../etc/passwd"})
    assert r.status_code == 404
