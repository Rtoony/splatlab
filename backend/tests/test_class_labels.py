"""Class-label painting: taxonomy validation (stdlib), the numpy store, the
app routes, and realign edit-survival for class records."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import class_labels as cl  # noqa: E402
import class_taxonomy as ct  # noqa: E402
import splat_route  # noqa: E402


# ── taxonomy (stdlib, app-venv importable) ──────────────────────────────────────


def test_builtin_taxonomy_loads_and_is_valid() -> None:
    tax = ct.load_taxonomy()
    ids = [c["id"] for c in tax["classes"]]
    assert {"grass", "dirt", "gravel", "pavement", "concrete",
            "vegetation", "structure", "water"} <= set(ids)
    for entry in tax["classes"]:
        assert entry["category"] in ct.CATEGORIES
        assert entry["generative"]["base_rgb"]


def test_job_extras_extend_but_cannot_redefine(tmp_path: Path) -> None:
    lf = tmp_path / "_langfield"
    lf.mkdir(parents=True)
    (lf / ct.EXTRA_NAME).write_text(json.dumps({"classes": [
        {"id": "flowerbed", "display": "Flower bed", "color": "#ec4899",
         "category": "vegetation",
         "generative": {"base_rgb": [0.8, 0.3, 0.5]}},
    ]}))
    tax = ct.load_taxonomy(tmp_path)
    assert "flowerbed" in {c["id"] for c in tax["classes"]}

    (lf / ct.EXTRA_NAME).write_text(json.dumps({"classes": [
        {"id": "grass", "display": "My grass", "color": "#000000",
         "category": "ground"},
    ]}))
    with pytest.raises(ct.TaxonomyError, match="redefine"):
        ct.load_taxonomy(tmp_path)


def test_malformed_extra_fails_loud(tmp_path: Path) -> None:
    lf = tmp_path / "_langfield"
    lf.mkdir(parents=True)
    (lf / ct.EXTRA_NAME).write_text(json.dumps({"classes": [
        {"id": "BAD ID!", "display": "x", "color": "#123456", "category": "ground"},
    ]}))
    with pytest.raises(ct.TaxonomyError, match="bad class id"):
        ct.load_taxonomy(tmp_path)


# ── store (numpy) ───────────────────────────────────────────────────────────────


def test_add_resolve_precedence_and_delete(tmp_path: Path) -> None:
    order = ["grass", "dirt"]
    xyz = np.random.default_rng(1).random((50, 3)).astype(np.float32)
    r1 = cl.add_class_label(tmp_path, class_id="grass",
                            indices=np.arange(0, 30), xyz=xyz[:30], n_ply=100)
    r2 = cl.add_class_label(tmp_path, class_id="dirt",
                            indices=np.arange(20, 40), xyz=xyz[20:40], n_ply=100)
    assert (tmp_path / f"class_xyz_{r1['id']}.npy").is_file()

    cmap = cl.resolve_class_map(tmp_path, 100, order)
    assert (cmap[:20] == 0).all()          # grass only
    assert (cmap[20:40] == 1).all()        # dirt painted LATER wins the overlap
    assert (cmap[40:] == -1).all()

    s = cl.summary(tmp_path, 100, order)
    assert s["by_class"] == {"grass": 20, "dirt": 20}
    assert s["labeled"] == 40 and s["unlabeled"] == 60

    assert cl.delete_class_label(tmp_path, r2["id"])
    cmap2 = cl.resolve_class_map(tmp_path, 100, order)
    assert (cmap2[:30] == 0).all() and (cmap2[30:] == -1).all()
    assert not (tmp_path / f"class_xyz_{r2['id']}.npy").exists()


def test_guardrails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least"):
        cl.add_class_label(tmp_path, class_id="grass",
                           indices=np.arange(3), xyz=None, n_ply=100)
    with pytest.raises(ValueError, match="force=true"):
        cl.add_class_label(tmp_path, class_id="grass",
                           indices=np.arange(60), xyz=None, n_ply=100)
    # force lifts the fraction cap
    rec = cl.add_class_label(tmp_path, class_id="grass",
                             indices=np.arange(60), xyz=None, n_ply=100, force=True)
    assert rec["count"] == 60


# ── routes (worker stubbed) ─────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(
        splat_route, "_langfield_paint_context",
        lambda job_id: ("cfg.yml", str(outputs / job_id / "_langfield")),
    )
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0c1001") -> Path:
    job_dir = outputs / job_id
    (job_dir / "_langfield").mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "mode": "3d",
    }))
    return job_dir


def _stub_worker(monkeypatch: pytest.MonkeyPatch, responder):
    async def fake(path, body):
        return responder(path, body)
    monkeypatch.setattr(splat_route, "_langfield_worker_json", fake)


def test_taxonomy_route_serves_merged(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    (job_dir / "_langfield" / ct.EXTRA_NAME).write_text(json.dumps({"classes": [
        {"id": "flowerbed", "display": "Flower bed", "color": "#ec4899",
         "category": "vegetation"},
    ]}))
    r = http.get("/api/splat/jobs/splat_0c1001/class-taxonomy")
    assert r.status_code == 200
    assert "flowerbed" in {c["id"] for c in r.json()["classes"]}


def test_class_add_validates_class_id_before_worker(client, monkeypatch) -> None:
    http, outputs = client
    _mk_job(outputs)
    called = {}
    _stub_worker(monkeypatch, lambda path, body: called.setdefault("hit", path))
    payload = base64.b64encode(np.arange(12, dtype="<u4").tobytes()).decode()
    r = http.post("/api/splat/jobs/splat_0c1001/class-labels",
                  json={"class_id": "not-a-class", "indices_b64": payload})
    assert r.status_code == 400
    assert "unknown class" in r.json()["detail"]
    assert "hit" not in called  # the worker was never bothered


def test_class_add_happy_path_proxies_to_worker(client, monkeypatch) -> None:
    http, outputs = client
    _mk_job(outputs)
    seen = {}

    def responder(path, body):
        seen["path"], seen["body"] = path, body
        return SimpleNamespace(status_code=200,
                               json=lambda: {"ok": True, "record": {"id": "ab12"}})
    _stub_worker(monkeypatch, responder)
    payload = base64.b64encode(np.arange(12, dtype="<u4").tobytes()).decode()
    r = http.post("/api/splat/jobs/splat_0c1001/class-labels",
                  json={"class_id": "grass", "indices_b64": payload})
    assert r.status_code == 200, r.text
    assert seen["path"] == "/class_add"
    assert seen["body"]["class_id"] == "grass"


def test_class_map_passthrough_carries_order_header(client, monkeypatch) -> None:
    http, outputs = client
    _mk_job(outputs)
    body_bytes = np.array([-1, 0, 3], dtype="<i2").tobytes()

    def responder(path, body):
        assert path == "/class_map"
        return SimpleNamespace(status_code=200, content=body_bytes,
                               headers={"X-Count": "3"})
    _stub_worker(monkeypatch, responder)
    r = http.get("/api/splat/jobs/splat_0c1001/class-labels/map")
    assert r.status_code == 200
    assert r.content == body_bytes
    order = json.loads(r.headers["X-Class-Order"])
    assert order[0] == "grass" and "water" in order


def test_class_list_reads_disk_without_worker(client) -> None:
    http, outputs = client
    job_dir = _mk_job(outputs)
    (job_dir / "_langfield" / "class_labels.json").write_text(json.dumps([
        {"id": "ab12", "class_id": "grass", "count": 30},
    ]))
    r = http.get("/api/splat/jobs/splat_0c1001/class-labels")
    assert r.status_code == 200
    assert r.json()["records"][0]["class_id"] == "grass"


# ── realign carries class records (same lane as overrides) ──────────────────────


def test_realign_remaps_class_records(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    ckpt = rng.random((80, 3), dtype=np.float32)
    keep = rng.permutation(80)[:70]
    job = tmp_path / "splat_0c1002"
    lf = job / "_langfield"
    lf.mkdir(parents=True)
    (job / "_preview").mkdir()
    np.savez(lf / "gauss_emb.npz", gauss_emb=np.zeros((80, 4), np.float16))
    np.save(lf / "ckpt_xyz.npy", ckpt)
    # write the edited ply
    from test_langfield_realign import _write_splat_ply  # noqa: PLC0415
    _write_splat_ply(job / "_preview" / "splat.ply", ckpt[keep])
    (lf / "STALE").write_text("t\n")

    dropped = np.setdiff1d(np.arange(80), keep)
    painted = np.concatenate([keep[:10], dropped[:3]])
    (lf / "class_labels.json").write_text(json.dumps([
        {"id": "dd44", "class_id": "grass", "count": 13},
    ]))
    np.save(lf / "class_idx_dd44.npy", np.arange(13, dtype=np.uint32))
    np.save(lf / "class_xyz_dd44.npy", ckpt[painted])

    proc = subprocess.run(
        [sys.executable, str(BACKEND / "langfield_realign.py"), str(job)],
        capture_output=True, text=True, cwd=str(BACKEND), timeout=120)
    receipt = json.loads(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 0, receipt
    rec = next(r for r in receipt["records"] if r["id"] == "dd44")
    assert rec["store"] == "class_labels.json"
    assert rec["label"] == "grass"
    assert rec["kept"] == 10 and rec["dropped"] == 3
    manifest = json.loads((lf / "class_labels.json").read_text())
    assert manifest[0]["count"] == 10
