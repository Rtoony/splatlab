"""The pluck doc: inverse-map correctness, the total-or-skip xyz fallback,
fail-loud refusals, identity staleness, and the routes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402
import world_pluck as wpk  # noqa: E402
import world_pluck_route  # noqa: E402

JOB = "splat_a1c40b"  # hex-only — the trap that 404s every route silently


def _binary_ply(path: Path, xyz: np.ndarray, extra_floats: int = 4) -> None:
    """A binary_little_endian all-float PLY (the 3DGS layout read_ply_xyz
    parses): x,y,z plus `extra_floats` padding properties."""
    n = len(xyz)
    props = ["x", "y", "z"] + [f"f_{i}" for i in range(extra_floats)]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props)
              + "end_header\n").encode("ascii")
    body = np.hstack([np.ascontiguousarray(xyz, dtype="<f4"),
                      np.zeros((n, extra_floats), dtype="<f4")])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body.tobytes())


SPLAT_XYZ = np.array([[0.0, 0.0, 0.0],
                      [1.0, 0.5, -0.25],
                      [2.0, 1.0, 0.75]], dtype=np.float32)


def _mk_job(tmp_path: Path, *, index_map=(3, 0, 2), crate_indices=(0, 2, 4),
            props=("crate",), langweb_rows: int | None = None) -> Path:
    job = tmp_path / JOB
    (job / "meta.json").parent.mkdir(parents=True, exist_ok=True)
    (job / "meta.json").write_text(json.dumps({
        "job_id": JOB, "output_dir": str(job), "status": "completed",
        "mode": "3d"}))
    _binary_ply(job / "_preview" / "splat.ply", SPLAT_XYZ)
    if langweb_rows is not None:
        _binary_ply(job / "_preview" / "langweb.ply",
                    SPLAT_XYZ[:langweb_rows])
    world = job / "_world"
    (world / "elements").mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({
        "v": 1, "job_id": JOB,
        "elements": [{"slug": s, "built": True} for s in props]
                    + [{"slug": "bench", "built": True}]}))
    (world / "world_manifest.json").write_text(json.dumps({
        "v": 1,
        "elements": [{"slug": s, "role": "prop"} for s in props]
                    + [{"slug": "bench", "role": "static"}],
        "shell": {"slug": "shell", "role": "static"}}))
    isolated = job / "_scene" / "isolated"
    (isolated / "crate").mkdir(parents=True)
    (isolated / "batch_isolate.json").write_text("{}")
    np.savez(isolated / "crate" / "object_indices.npz",
             indices=np.asarray(crate_indices, dtype=np.int64))
    if index_map is not None:
        lf = job / "_langfield"
        lf.mkdir(exist_ok=True)
        np.save(lf / "ply_index_map.npy",
                np.asarray(index_map, dtype=np.int64))
    return job


# ── builder ─────────────────────────────────────────────────────────────────────


def test_build_maps_checkpoint_indices_through_the_inverse_map(tmp_path):
    """map[ply_row]=ckpt_row = [3,0,2] (n_ckpt=4): ckpt indices [0,2,4] ->
    served rows [1,2], with the out-of-range 4 counted as dropped."""
    job = _mk_job(tmp_path)
    doc = wpk.build_pluck(job)
    assert doc["method"] == "index-map"
    assert doc["n_rows"] == 3
    crate = doc["elements"]["crate"]
    assert crate["rows"] == [1, 2]
    assert crate["ckpt_count"] == 3
    assert crate["dropped_by_export"] == 1
    # Statics never get pluck entries — the mesh IS their representation.
    assert "bench" not in doc["elements"] and "bench" not in doc["skipped"]


def test_build_refuses_stale_langfield(tmp_path):
    job = _mk_job(tmp_path)
    (job / "_langfield" / "STALE").write_text("t\n")
    with pytest.raises(wpk.PluckError, match="STALE.*rebuild"):
        wpk.build_pluck(job)


def test_build_refuses_langweb_row_mismatch(tmp_path):
    job = _mk_job(tmp_path, langweb_rows=2)
    with pytest.raises(wpk.PluckError, match="row count differs"):
        wpk.build_pluck(job)


def test_xyz_fallback_is_total_or_skip(tmp_path):
    job = _mk_job(tmp_path, index_map=None)
    # Byte-exact subset of the splat -> matches totally.
    _binary_ply(job / "_scene" / "isolated" / "crate" / "object.ply",
                SPLAT_XYZ[[0, 2]])
    doc = wpk.build_pluck(job)
    assert doc["method"] == "xyz-match"
    assert doc["elements"]["crate"]["rows"] == [0, 2]

    # One perturbed vertex -> below the 99% floor -> the SLUG is skipped
    # with a reason, never a partial silent map.
    perturbed = SPLAT_XYZ[[0, 2]].copy()
    perturbed[1] += 0.001
    _binary_ply(job / "_scene" / "isolated" / "crate" / "object.ply", perturbed)
    with pytest.raises(wpk.PluckError, match="xyz-match below"):
        wpk.build_pluck(job)  # crate was the only prop -> loud overall refusal


def test_cross_slug_row_claims_must_be_disjoint(tmp_path):
    job = _mk_job(tmp_path, index_map=(0, 1, 2), crate_indices=(0,),
                  props=("crate", "box"))
    isolated = job / "_scene" / "isolated"
    (isolated / "box").mkdir()
    np.savez(isolated / "box" / "object_indices.npz",
             indices=np.asarray([0], dtype=np.int64))
    with pytest.raises(wpk.PluckError, match="same splat rows"):
        wpk.build_pluck(job)


def test_missing_isolate_indices_is_a_named_skip(tmp_path):
    job = _mk_job(tmp_path, props=("crate", "ghost"))
    doc = wpk.build_pluck(job)
    assert doc["skipped"]["ghost"] == "no-isolate-indices"
    assert "crate" in doc["elements"]


def test_read_pluck_goes_stale_when_an_input_changes(tmp_path):
    job = _mk_job(tmp_path)
    doc = wpk.build_pluck(job)
    wpk.write_pluck(job / "_world", doc)
    fresh, stale, reasons = wpk.read_pluck(job / "_world", job)
    assert fresh is not None and stale is False and reasons == []

    _binary_ply(job / "_preview" / "splat.ply",
                np.vstack([SPLAT_XYZ, [[9.0, 9.0, 9.0]]]))
    _doc, stale, reasons = wpk.read_pluck(job / "_world", job)
    assert stale is True
    assert any("splat_ply changed" in r for r in reasons)


def test_validate_rejects_malformed_docs(tmp_path):
    job = _mk_job(tmp_path)
    doc = wpk.build_pluck(job)
    for mutate in (
        lambda d: d.update(schema="nope"),
        lambda d: d.update(n_rows=0),
        lambda d: d["elements"]["crate"].update(rows=[99]),   # out of range
        lambda d: d["elements"]["crate"].update(count=7),
    ):
        bad = json.loads(json.dumps(doc))
        mutate(bad)
        with pytest.raises(wpk.PluckError):
            wpk.validate_pluck(bad)


# ── routes ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    app.include_router(world_pluck_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def test_pluck_routes_build_and_serve_with_staleness(client):
    http, outputs = client
    job = _mk_job(outputs)

    assert http.get(f"/api/splat/jobs/{JOB}/world/pluck").status_code == 404

    r = http.post(f"/api/splat/jobs/{JOB}/world/pluck")
    assert r.status_code == 200, r.text
    assert r.json()["pluck"]["elements"]["crate"]["rows"] == [1, 2]
    assert (job / "_world" / "pluck.json").is_file()

    r = http.get(f"/api/splat/jobs/{JOB}/world/pluck")
    assert r.status_code == 200
    assert r.json()["stale"] is False

    _binary_ply(job / "_preview" / "splat.ply",
                np.vstack([SPLAT_XYZ, [[9.0, 9.0, 9.0]]]))
    r = http.get(f"/api/splat/jobs/{JOB}/world/pluck")
    assert r.status_code == 200
    assert r.json()["stale"] is True and r.json()["reasons"]


def test_pluck_post_refusal_carries_the_remedy(client):
    http, outputs = client
    job = _mk_job(outputs)
    (job / "_scene" / "isolated" / "batch_isolate.json").unlink()
    r = http.post(f"/api/splat/jobs/{JOB}/world/pluck")
    assert r.status_code == 409
    assert "isolate stage" in r.json()["detail"]
