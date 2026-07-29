"""POST /scene/surfaces: the paint-gated, bounded, lock-respecting sibling of
/scene/ground. Subprocesses mocked at the same seam; refusals must carry the
CLI's own reasons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

JOB = "splat_0f10e5"

SURFACES_REPORT = {
    "v": 1, "provenance": "paint-synthesized",
    "params": {"seed": 1234},
    "records_considered": 1, "records_refused": [],
    "patches": [{"id": "patch_00", "record": "ab12cd34",
                 "class_id": "structure",
                 "plane": {"normal": [0, 1, 0], "d": -0.5,
                           "rms_units": 0.012, "inliers": 800},
                 "cells": 500, "area_units2": 1.25,
                 "verts": 600, "faces": 1000, "ply": "patch_00.ply"}],
    "refused_clusters": [],
    "class_tiles": "not-applied-v1",
    "texture": {"baked": True, "size": 1024, "coverage": 0.8},
    "artifacts": {"preview_glb": "surfaces_preview.glb",
                  "atlas": "surfaces_atlas.png"},
    "seconds": 2.0,
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(splat_route, "MESH_ENV_PYTHON", tmp_path / "python")
    (tmp_path / "python").write_text("#!/bin/sh\n")

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job(outputs: Path, *, paint: bool = True,
            preview_splat: bool = True) -> Path:
    job_dir = outputs / JOB
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": JOB, "output_dir": str(job_dir), "status": "completed",
        "mode": "3d", "meters_per_unit": 2.35,
    }))
    lf = job_dir / splat_route.LANGFIELD_DIRNAME
    lf.mkdir()
    (lf / "gauss_emb.npz").write_bytes(b"npz")
    if paint:
        (lf / "class_labels.json").write_text(json.dumps([
            {"id": "ab12cd34", "class_id": "structure", "count": 900}]))
    if preview_splat:
        prev = job_dir / "_preview"
        prev.mkdir()
        (prev / "splat.ply").write_bytes(b"ply")
    return job_dir


def _fake_subprocess(calls: list, *, build_rc: int = 0,
                     build_stderr: bytes = b""):
    async def run(command):
        calls.append([str(c) for c in command])
        joined = " ".join(str(c) for c in command)
        if "surface_patches" in joined:
            if build_rc != 0:
                return build_rc, b"", build_stderr
            out_dir = Path(command[5])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "patch_00.ply").write_bytes(b"ply")
            (out_dir / "surfaces_preview.glb").write_bytes(b"glb")
            (out_dir / "surfaces_atlas.png").write_bytes(b"png")
            (out_dir / "surface_patches.json").write_text(
                json.dumps(SURFACES_REPORT))
            return 0, b"", b""
        if "ground_mesh_receipt" in joined:
            out_dir = Path(command[3])
            (out_dir / "receipt_top.png").write_bytes(b"png")
            (out_dir / "receipt_oblique.png").write_bytes(b"png")
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_surfaces_requires_structure_paint(client):
    http, outputs = client
    _mk_job(outputs, paint=False)
    r = http.post(f"/api/splat/jobs/{JOB}/scene/surfaces", json={})
    assert r.status_code == 409
    assert "paint" in r.json()["detail"].lower()


def test_surfaces_requires_preview_splat(client):
    http, outputs = client
    _mk_job(outputs, preview_splat=False)
    r = http.post(f"/api/splat/jobs/{JOB}/scene/surfaces", json={})
    assert r.status_code == 409
    assert "splat.ply" in r.json()["detail"]


def test_surfaces_full_build_passes_bounded_params(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(calls))

    r = http.post(f"/api/splat/jobs/{JOB}/scene/surfaces", json={
        "plane_dist_units": 0.08, "min_cluster_points": 120,
        "max_patches": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["provenance"] == "paint-synthesized"
    assert len(body["patches"]) == 1

    build_call = next(c for c in calls if any("surface_patches" in x for x in c))
    assert "--plane-dist-units" in build_call and "0.08" in build_call
    assert "--min-cluster-points" in build_call and "120" in build_call
    assert "--max-patches" in build_call and "4" in build_call
    assert "--meters-per-unit" in build_call and "2.35" in build_call

    # All fmt values serve.
    for fmt, blob in (("report", None), ("glb", b"glb"), ("atlas", b"png"),
                      ("top", b"png"), ("oblique", b"png")):
        got = http.get(f"/api/splat/jobs/{JOB}/scene/surfaces/file",
                       params={"fmt": fmt})
        assert got.status_code == 200, fmt
        if blob is not None:
            assert got.content == blob

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["surfaces"]["patches"] == 1
    assert meta["scene"]["surfaces"]["built_at"]


def test_surfaces_refusal_is_a_500_with_the_reason(client, monkeypatch):
    http, outputs = client
    _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(
                            calls, build_rc=1,
                            build_stderr=b"FATAL: plane rms 0.0410 > 0.0300 "
                                         b"units - paint flatter segments"))
    r = http.post(f"/api/splat/jobs/{JOB}/scene/surfaces", json={})
    assert r.status_code == 500
    assert "plane rms" in r.json()["detail"]


def test_surfaces_params_are_bounded(client):
    http, outputs = client
    _mk_job(outputs)
    for payload in ({"plane_dist_units": 0.6}, {"min_cluster_points": 10},
                    {"max_patches": 65}, {"texture_size": 128},
                    {"max_rms_frac": 0.0}, {"min_inlier_frac": 0.0},
                    {"min_inlier_frac": 1.5}):
        r = http.post(f"/api/splat/jobs/{JOB}/scene/surfaces", json=payload)
        assert r.status_code == 422, payload


def test_surfaces_respects_the_job_lock(client):
    http, outputs = client
    _mk_job(outputs)
    lock = splat_route._mesh_export_lock(JOB)
    lock._locked = True  # noqa: SLF001 - simulate a running build
    try:
        r = http.post(f"/api/splat/jobs/{JOB}/scene/surfaces", json={})
        assert r.status_code == 409
    finally:
        lock._locked = False  # noqa: SLF001
