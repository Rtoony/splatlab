"""Audited splat-edit lane: versioning, sanitization, and the promote guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_edit_route  # noqa: E402
import splat_route  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_edit_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, job_id: str = "splat_5e0001") -> Path:
    job_dir = outputs / job_id
    (job_dir / "_preview").mkdir(parents=True)
    (job_dir / "_preview" / "splat.ply").write_bytes(b"PLY-LIVE-ORIGINAL")
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir),
        "status": "completed", "mode": "3d",
    }))
    return job_dir


def _fake_engine(monkeypatch: pytest.MonkeyPatch, *, out_count: int = 90):
    calls: list[list[str]] = []

    def fake(source: Path, staged: Path, extra: list[str]) -> dict:
        calls.append(extra)
        staged.write_bytes(b"PLY-EDITED-" + source.read_bytes()[-8:])
        return {"op": extra[1], "in_count": 100, "out_count": out_count,
                "removed": 100 - out_count, "properties": 62, "params": {}}

    monkeypatch.setattr(splat_edit_route, "_run_engine", fake)
    return calls


def test_edit_creates_versions_and_receipts(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    tc, outputs = client
    job_dir = _mk_job(outputs)
    calls = _fake_engine(monkeypatch)

    first = tc.post("/api/splat/jobs/splat_5e0001/splat/edit", json={
        "op": "clean", "params": {"min_opacity": 0.05}, "note": "ghost sweep"})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["version"] == 1
    assert body["receipt"]["base"]["version"] is None  # edited the live ply
    assert body["receipt"]["base"]["path"] == "_preview/splat.ply"
    assert (job_dir / "_splat" / "versions" / "splat-v0001.ply").is_file()
    assert calls[0] == ["--op", "clean", "--min-opacity=0.05"]

    second = tc.post("/api/splat/jobs/splat_5e0001/splat/edit", json={
        "op": "crop_box", "params": {"min": [-1, -1, -1], "max": [1, 1, 1]}})
    assert second.status_code == 200
    assert second.json()["version"] == 2
    # chains off v1, not the live ply
    assert second.json()["receipt"]["base"]["version"] == 1

    listed = tc.get("/api/splat/jobs/splat_5e0001/splat/versions").json()
    assert [v["version"] for v in listed["versions"]] == [1, 2]
    assert listed["original_preserved"] is False
    # the live asset is untouched by edits
    assert (job_dir / "_preview" / "splat.ply").read_bytes() == b"PLY-LIVE-ORIGINAL"


def test_edit_params_are_sanitized(client, monkeypatch: pytest.MonkeyPatch) -> None:
    tc, outputs = client
    _mk_job(outputs)
    _fake_engine(monkeypatch)
    bad = [
        {"op": "melt", "params": {}},
        {"op": "clean", "params": {}},
        {"op": "clean", "params": {"min_opacity": 2.0}},
        {"op": "clean", "params": {"min_opacity": True}},
        {"op": "crop_box", "params": {"min": [0, 0], "max": [1, 1, 1]}},
        {"op": "transform", "params": {}},
        {"op": "transform", "params": {"scale": -2}},
    ]
    for payload in bad:
        response = tc.post("/api/splat/jobs/splat_5e0001/splat/edit", json=payload)
        assert response.status_code == 400, payload


def test_promote_guard_and_original_preservation(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    tc, outputs = client
    job_dir = _mk_job(outputs)
    _fake_engine(monkeypatch)
    assert tc.post("/api/splat/jobs/splat_5e0001/splat/edit", json={
        "op": "clean", "params": {"max_scale": 0.5}}).status_code == 200

    # Index-keyed consumers block promotion without force.
    langfield = job_dir / "_langfield"
    langfield.mkdir()
    (langfield / "gauss_emb.npz").write_bytes(b"npz")
    refused = tc.post("/api/splat/jobs/splat_5e0001/splat/promote",
                      json={"version": 1})
    assert refused.status_code == 409
    assert "desynchronise" in refused.json()["detail"]
    assert (job_dir / "_preview" / "splat.ply").read_bytes() == b"PLY-LIVE-ORIGINAL"

    forced = tc.post("/api/splat/jobs/splat_5e0001/splat/promote",
                     json={"version": 1, "force": True})
    assert forced.status_code == 200, forced.text
    receipt = forced.json()["receipt"]
    assert receipt["original_preserved_now"] is True
    assert receipt["forced_over_consumers"]
    # live asset replaced; pristine original preserved
    assert (job_dir / "_preview" / "splat.ply").read_bytes().startswith(b"PLY-EDITED-")
    assert (job_dir / "_splat" / "original.ply").read_bytes() == b"PLY-LIVE-ORIGINAL"

    listed = tc.get("/api/splat/jobs/splat_5e0001/splat/versions").json()
    assert listed["original_preserved"] is True
    assert listed["promote"]["promoted_version"] == 1

    # A second promote must NOT overwrite the preserved original.
    assert tc.post("/api/splat/jobs/splat_5e0001/splat/edit", json={
        "op": "transform", "params": {"scale": 2.0}}).status_code == 200
    again = tc.post("/api/splat/jobs/splat_5e0001/splat/promote",
                    json={"version": 2, "force": True})
    assert again.status_code == 200
    assert again.json()["receipt"]["original_preserved_now"] is False
    assert (job_dir / "_splat" / "original.ply").read_bytes() == b"PLY-LIVE-ORIGINAL"


def test_promote_missing_version_404s(client, monkeypatch) -> None:
    tc, outputs = client
    _mk_job(outputs)
    response = tc.post("/api/splat/jobs/splat_5e0001/splat/promote",
                       json={"version": 3})
    assert response.status_code == 404
