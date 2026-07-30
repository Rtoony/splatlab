"""Propose-generated: candidate reading, review payloads, promote/revert
refusals — everything that doesn't need the GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_route  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_dec0de"  # hex-only — a readable slug silently 404s every route


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(generate_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path) -> Path:
    job = outputs / JOB
    (job / "_world" / "elements").mkdir(parents=True)
    (job / "meta.json").write_text(json.dumps(
        {"job_id": JOB, "output_dir": str(job), "status": "completed"}))
    return job


def _mk_candidate(job: Path, slug: str = "hydrant", *, placed: bool = True,
                  with_glb: bool = True) -> Path:
    cdir = job / "_regen" / "objects" / slug
    cdir.mkdir(parents=True)
    if with_glb:
        (cdir / "generated_mesh.glb").write_bytes(b"glb-bytes")
    (cdir / "preview.png").write_bytes(b"png")
    placement = {"placement_resolved": placed}
    if placed:
        placement["transform_4x4_generated_to_capture"] = [[1, 0, 0, 0]] * 4
    (cdir / "generate_report.json").write_text(json.dumps({
        "schema": "splatlab.object_generate/1",
        "mask_alignment_gate": {"iou_vs_captured_object": 0.87},
        "capture_frame_placement": {"mesh_glb": placement},
    }))
    return cdir


def test_candidate_payload_placed_and_files(client) -> None:
    http, outputs = client
    job = _mk_job(outputs)
    _mk_candidate(job)
    r = http.get(f"/api/splat/jobs/{JOB}/objects/hydrant/generate/candidate")
    assert r.status_code == 200
    body = r.json()
    assert body["placed"] is True
    assert "mesh" in body["files"] and "preview" in body["files"]
    assert body["report"]["mask_alignment_gate"]["iou_vs_captured_object"] == 0.87

    # An unplaced candidate reports placed=false, still reviewable.
    _mk_candidate(job, "bike", placed=False)
    r = http.get(f"/api/splat/jobs/{JOB}/objects/bike/generate/candidate")
    assert r.status_code == 200 and r.json()["placed"] is False

    # No candidate at all -> 404 with the propose hint.
    r = http.get(f"/api/splat/jobs/{JOB}/objects/ghost/generate/candidate")
    assert r.status_code == 404


def test_generate_file_serves_no_cache_and_contains(client) -> None:
    http, outputs = client
    job = _mk_job(outputs)
    _mk_candidate(job)
    r = http.get(f"/api/splat/jobs/{JOB}/generate/file",
                 params={"slug": "hydrant", "fmt": "preview"})
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"
    assert http.get(f"/api/splat/jobs/{JOB}/generate/file",
                    params={"slug": "hydrant", "fmt": "weird"}).status_code == 400
    assert http.get(f"/api/splat/jobs/{JOB}/generate/file",
                    params={"slug": "ghost", "fmt": "mesh"}).status_code == 404


def test_promote_refusals(client) -> None:
    http, outputs = client
    job = _mk_job(outputs)

    # No element to replace -> 404 (promotion replaces, never creates).
    _mk_candidate(job)
    r = http.post(f"/api/splat/jobs/{JOB}/objects/hydrant/generate/promote")
    assert r.status_code == 404

    # Element exists but candidate is UNPLACED -> 409.
    (job / "_world" / "elements" / "bike.glb").write_bytes(b"captured")
    _mk_candidate(job, "bike", placed=False)
    r = http.post(f"/api/splat/jobs/{JOB}/objects/bike/generate/promote")
    assert r.status_code == 409
    assert "PLACED" in r.json()["detail"]

    # Already promoted -> 409 until reverted.
    (job / "_world" / "elements" / "lamp.glb").write_bytes(b"captured")
    _mk_candidate(job, "lamp")
    (job / "_world" / "elements" / "lamp.generated.json").write_text(
        json.dumps({"schema": generate_route.GENERATED_MARKER_SCHEMA}))
    r = http.post(f"/api/splat/jobs/{JOB}/objects/lamp/generate/promote")
    assert r.status_code == 409
    assert "revert" in r.json()["detail"]


def test_revert_and_discard_refusals(client) -> None:
    http, outputs = client
    job = _mk_job(outputs)

    # Nothing to revert.
    r = http.post(f"/api/splat/jobs/{JOB}/objects/hydrant/generate/revert")
    assert r.status_code == 404

    # Marker names a missing version -> exact revert impossible -> 409.
    (job / "_world" / "elements" / "hydrant.generated.json").write_text(
        json.dumps({"schema": generate_route.GENERATED_MARKER_SCHEMA,
                    "prior_glb_version": "hydrant-v0001.glb"}))
    r = http.post(f"/api/splat/jobs/{JOB}/objects/hydrant/generate/revert")
    assert r.status_code == 409

    # Discard refuses while promoted; works otherwise.
    _mk_candidate(job)
    r = http.delete(f"/api/splat/jobs/{JOB}/objects/hydrant/generate/candidate")
    assert r.status_code == 409  # marker above marks it promoted
    (job / "_world" / "elements" / "hydrant.generated.json").unlink()
    r = http.delete(f"/api/splat/jobs/{JOB}/objects/hydrant/generate/candidate")
    assert r.status_code == 200
    assert not (job / "_regen" / "objects" / "hydrant").exists()
    assert http.delete(
        f"/api/splat/jobs/{JOB}/objects/hydrant/generate/candidate"
    ).status_code == 404
