"""POST /jobs/{id}/duplicate — the working copy.

The whole point of the feature is that editing the copy CANNOT touch the
original, so the load-bearing assertion here is that nothing shares an inode.
An earlier draft of this route hardlinked processed/, colmap/ and
_langfield/gauss_emb.npz for speed; that is unsafe because colmap opens
database.db read-write and langfield_v2 writes gauss_emb.npz with a plain
np.savez_compressed (truncate-in-place), either of which would corrupt the
ORIGINAL. test_duplicate_shares_no_inodes_with_source is what keeps that
optimisation from coming back.
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


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_0d0001") -> Path:
    job = outputs / job_id
    (job / "processed" / "splatfacto" / "ts").mkdir(parents=True)
    (job / "processed" / "splatfacto" / "ts" / "config.yml").write_text("cfg")
    (job / "colmap").mkdir()
    (job / "colmap" / "database.db").write_bytes(b"sqlite")
    (job / "_preview").mkdir()
    (job / "_preview" / "splat.ply").write_bytes(b"ply-bytes")
    (job / "_langfield").mkdir()
    (job / "_langfield" / "gauss_emb.npz").write_bytes(b"emb")
    (job / "_langfield" / "overrides.json").write_text("[]")
    (job / "_langfield" / ".building-tmp.npy").write_bytes(b"junk")
    (job / "versions" / "v1-20260726T000000Z").mkdir(parents=True)
    (job / "versions" / "v1-20260726T000000Z" / "splat.ply").write_bytes(b"old" * 100)
    (job / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job), "status": "completed",
        "mode": "3d", "pinned": True, "input_path": "/captures/backyard.mp4",
        "meters_per_unit": 1.5,
    }))
    return job


def test_duplicate_shares_no_inodes_with_source(client) -> None:
    """The safety property. If this fails, editing a copy can destroy the original."""
    http, outputs = client
    src = _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0d0001/duplicate")
    assert r.status_code == 200, r.text
    dst = outputs / r.json()["new_job_id"]

    src_inodes = {p.stat().st_ino for p in src.rglob("*") if p.is_file()}
    dst_inodes = {p.stat().st_ino for p in dst.rglob("*") if p.is_file()}
    assert src_inodes and dst_inodes
    assert src_inodes.isdisjoint(dst_inodes), "duplicate shares an inode with its source"
    # every copied file has exactly one link — nothing was os.link()ed
    assert all(p.stat().st_nlink == 1 for p in dst.rglob("*") if p.is_file())


def test_duplicate_copies_content_and_skips_litter(client) -> None:
    http, outputs = client
    _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0d0001/duplicate")
    assert r.status_code == 200, r.text
    dst = outputs / r.json()["new_job_id"]

    assert (dst / "_preview" / "splat.ply").read_bytes() == b"ply-bytes"
    assert (dst / "colmap" / "database.db").read_bytes() == b"sqlite"
    assert (dst / "processed" / "splatfacto" / "ts" / "config.yml").read_text() == "cfg"
    assert (dst / "_langfield" / "overrides.json").read_text() == "[]"
    # transient build litter is not carried
    assert not (dst / "_langfield" / ".building-tmp.npy").exists()
    # restore points belong to the original's history; the copy starts clean
    assert not (dst / "versions").exists()
    # ...and the reported size excludes them too
    assert r.json()["bytes"] < 300


def test_duplicate_rewrites_identity_but_keeps_scene_truth(client) -> None:
    http, outputs = client
    src = _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0d0001/duplicate")
    new_id = r.json()["new_job_id"]
    dst = outputs / new_id

    meta = json.loads((dst / "meta.json").read_text())
    assert meta["job_id"] == new_id
    assert meta["output_dir"] == str(dst)
    assert meta["duplicated_from"] == "splat_0d0001"
    assert meta["parents"] == ["splat_0d0001"]
    assert meta["pinned"] is False
    assert meta["input_path"] == "backyard.mp4 (copy)"
    # scale calibration and other scene truth survive — it IS the same scene
    assert meta["meters_per_unit"] == 1.5

    src_meta = json.loads((src / "meta.json").read_text())
    assert src_meta["job_id"] == "splat_0d0001"
    assert src_meta["pinned"] is True
    assert src_meta["input_path"] == "/captures/backyard.mp4"


def test_duplicate_requires_completed(client) -> None:
    http, outputs = client
    job = _mk_job(outputs, "splat_0d0002")
    meta = json.loads((job / "meta.json").read_text())
    meta["status"] = "running"
    (job / "meta.json").write_text(json.dumps(meta))
    r = http.post("/api/splat/jobs/splat_0d0002/duplicate")
    assert r.status_code == 409


def test_duplicate_refuses_when_disk_would_run_out(client, monkeypatch) -> None:
    """Fail loud rather than wedge the disk — a full disk takes down every
    service on the box, not just SplatLab."""
    http, outputs = client
    _mk_job(outputs)
    import shutil as _shutil

    real = _shutil.disk_usage

    def tight(path):
        u = real(path)
        return type(u)(u.total, u.used, 1024)  # 1 KB free

    monkeypatch.setattr(splat_route.shutil, "disk_usage", tight)
    r = http.post("/api/splat/jobs/splat_0d0001/duplicate")
    assert r.status_code == 507
    assert "Not enough disk" in r.json()["detail"]
    # nothing half-built left behind
    assert sorted(p.name for p in outputs.iterdir()) == ["splat_0d0001"]


def test_duplicate_404s_on_unknown_and_unsafe_ids(client) -> None:
    http, _ = client
    assert http.post("/api/splat/jobs/splat_0dbeef/duplicate").status_code == 404
    assert http.post("/api/splat/jobs/..%2Fetc/duplicate").status_code in (404, 422)
