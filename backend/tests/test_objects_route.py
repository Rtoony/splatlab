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
TWIN_REPORT = {"verts": 6919, "faces": 10000, "solid_gaussians": 19659,
               "units": "scene-units (uncalibrated)", "extent": [0.47, 0.47, 0.47],
               "glb_bytes": 231708, "seconds": 0.7}
TEXTURE_REPORT = {"verts": 6580, "faces": 7999, "solid_gaussians": 19659,
                  "units": "meters", "meters_per_unit": 2.0, "up_axis": "Y",
                  "extent": [0.318, 0.846, 0.278],
                  "simplify": {"faces_in": 89990, "faces_out": 7999,
                               "reconstructed": True,
                               "crop": {"source": "auto (object.json bbox_tight)",
                                        "faces_kept": 47625, "faces_before": 89990}},
                  "texture": {"baked": True, "size": 1024, "coverage": 0.6,
                              "atlas_png": "textured_atlas.png"},
                  "glb_bytes": 1194076, "seconds": 6.5}


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
        if "twin_finish" in joined:
            twin_glb = Path(command[4])
            twin_glb.write_bytes(b"glb")
            (twin_glb.parent / "twin_finish.json").write_text(json.dumps(TWIN_REPORT))
            return 0, b"", b""
        if "ground_mesh_receipt" in joined:
            recv_dir = Path(command[3])
            (recv_dir / "receipt_top.png").write_bytes(b"png")
            (recv_dir / "receipt_oblique.png").write_bytes(b"png")
            return 0, b"", b""
        if "object_texture" in joined:
            out_glb = Path(command[4])
            out_glb.parent.mkdir(parents=True, exist_ok=True)
            out_glb.write_bytes(b"glb")
            out_glb.with_name(out_glb.stem + "_atlas.png").write_bytes(b"png")
            (out_glb.parent / "object_texture.json").write_text(json.dumps(TEXTURE_REPORT))
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


def test_objects_finish_builds_twin_glb(client, monkeypatch):
    """Real hydrant proof (2026-07-23, direct twin_finish.py smoke test):
    89,990 raw tris -> 10,000 decimated + colored. This proves the route
    wiring calls it correctly with the OBJECT'S OWN object.ply, not any
    whole-scene splat path."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))

    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "finish": True})
    assert r.status_code == 200
    body = r.json()
    assert body["twin_glb_url"]
    assert body["object"]["twin"]["faces"] == 10000

    finish_call = [str(c) for c in next(c for c in calls if "twin_finish" in " ".join(str(x) for x in c))]
    assert "--target-faces" in finish_call and "10000" in finish_call
    obj_dir = job_dir / splat_route.OBJECTS_DIRNAME / "table"
    assert str(obj_dir / "object.ply") in finish_call  # scoped to THIS object, not a scene splat
    assert str(obj_dir / "mesh" / "mesh.ply") in finish_call

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["objects"]["table"]["twin"]["faces"] == 10000

    for fmt in ("twin", "twin-top", "twin-oblique"):
        assert http.get(f"/api/splat/jobs/splat_0b0001/objects/table/file?fmt={fmt}").status_code == 200, fmt


def test_objects_finish_requires_mesh(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "mesh": False, "finish": True})
    assert r.status_code == 400


def test_objects_finish_target_faces_override(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "finish": True, "finish_target_faces": 25000})
    assert r.status_code == 200
    finish_call = [str(c) for c in next(c for c in calls if "twin_finish" in " ".join(str(x) for x in c))]
    assert "25000" in finish_call


def test_objects_finish_smooths_when_requested(client, monkeypatch):
    """Wiring-correctness only -- the smoothing algorithm itself (pipeline
    order, curvature-adaptiveness) is live-verified against the real hydrant
    mesh, recorded in STATUS.md, not re-proven here."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "finish": True, "smooth": True,
                        "smooth_iterations": 3, "smooth_feature_deg": 35.0})
    assert r.status_code == 200
    finish_call = [str(c) for c in next(c for c in calls if "twin_finish" in " ".join(str(x) for x in c))]
    assert "--smooth" in finish_call
    assert "--smooth-iterations" in finish_call and "3" in finish_call
    assert "--smooth-feature-deg" in finish_call and "35.0" in finish_call


def test_objects_finish_no_smooth_flags_when_not_requested(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    r = http.post("/api/splat/jobs/splat_0b0001/objects", json={"query": "table", "finish": True})
    assert r.status_code == 200
    finish_call = [str(c) for c in next(c for c in calls if "twin_finish" in " ".join(str(x) for x in c))]
    assert "--smooth" not in finish_call


def test_objects_listing_unknown_job_404(client):
    http, outputs = client
    r = http.get("/api/splat/jobs/splat_00000bad/objects")
    assert r.status_code == 404


def test_objects_listing_empty(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.get("/api/splat/jobs/splat_0b0001/objects")
    assert r.status_code == 200
    assert r.json() == {"job_id": "splat_0b0001", "objects": []}


def test_objects_listing_after_build(client, monkeypatch):
    """Splat-only build -> one entry carrying the receipt summary (no raw
    'artifacts' passthrough) and files{} limited to what exists on disk; the
    listed URL must be servable by the real file route."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    assert http.post("/api/splat/jobs/splat_0b0001/objects",
                     json={"query": "Round Wooden Table!", "mesh": False}).status_code == 200

    r = http.get("/api/splat/jobs/splat_0b0001/objects")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == "splat_0b0001"
    assert len(body["objects"]) == 1
    entry = body["objects"][0]
    assert entry["slug"] == "round-wooden-table"
    assert entry["pool_members"] == 1801           # receipt summary fields surface
    assert "artifacts" in OBJ_REPORT and "artifacts" not in entry
    # splat-only build: object.ply exists, mesh/twin/proxy artifacts do not
    assert set(entry["files"]) == {"splat"}
    assert http.get(entry["files"]["splat"]).status_code == 200

    # a stray dir without an object.json is a half-built isolation, not listable
    (job_dir / splat_route.OBJECTS_DIRNAME / "aborted-thing").mkdir()
    assert len(http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"]) == 1


def test_objects_listing_full_build_file_subset(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    assert http.post("/api/splat/jobs/splat_0b0001/objects",
                     json={"query": "table", "finish": True}).status_code == 200
    entry = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]
    # mesh+finish artifacts present; proxy formats absent (never built)
    assert set(entry["files"]) == {"splat", "ply", "glb", "receipt", "twin", "twin-top", "twin-oblique"}
    assert entry["files"]["twin"].endswith("/objects/table/file?fmt=twin")


def test_objects_texture_bakes_a_uv_mapped_asset(client, monkeypatch):
    """The texture lane produces a simplified GLB plus a standalone atlas, and
    both are served. It is independent of `finish` — asking for texture alone
    must not require or trigger the twin."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    _patch_json(job_dir / "meta.json", meters_per_unit=2.0)
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "texture": True})
    assert r.status_code == 200
    body = r.json()
    assert body["object"]["textured"]["texture"]["baked"] is True
    assert body["textured_glb_url"].endswith("fmt=textured")
    assert body["textured_atlas_url"].endswith("fmt=textured-atlas")

    tex_cmd = next(c for c in calls if "object_texture" in " ".join(str(x) for x in c))
    joined = " ".join(str(x) for x in tex_cmd)
    assert "--meters-per-unit 2.0" in joined      # calibration threaded through
    assert "--no-crop" not in joined              # ground removal on by default
    assert not any("twin_finish" in " ".join(str(x) for x in c) for c in calls)

    entry = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]
    assert {"textured", "textured-atlas"} <= set(entry["files"])
    for fmt in ("textured", "textured-atlas"):
        assert http.get(entry["files"][fmt]).status_code == 200


def test_objects_texture_requires_mesh(client):
    http, outputs = client
    _mk_job(outputs)
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "mesh": False, "texture": True})
    assert r.status_code == 400
    assert "requires mesh" in r.json()["detail"]


def test_objects_texture_honours_crop_and_size_overrides(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    assert http.post("/api/splat/jobs/splat_0b0001/objects",
                     json={"query": "table", "texture": True, "texture_crop": False,
                           "texture_size": 2048, "texture_target_faces": 3000}).status_code == 200
    joined = " ".join(str(x) for c in calls if "object_texture" in " ".join(str(y) for y in c)
                      for x in c)
    assert "--no-crop" in joined
    assert "--texture-size 2048" in joined
    assert "--target-faces 3000" in joined


def test_objects_texture_failure_is_loud_500(client, monkeypatch):
    """A failed bake must not be reported as a successful build."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    inner = _fake_subprocess(job_dir, calls)

    async def run(command):
        if "object_texture" in " ".join(str(c) for c in command):
            return 1, b"", b"xatlas exploded"
        return await inner(command)

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", run)
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "texture": True})
    assert r.status_code == 500
    assert "texture bake failed" in r.json()["detail"]
    # the raw mesh it already built is a valid artifact on its own
    assert (job_dir / splat_route.OBJECTS_DIRNAME / "table" / "mesh" / "mesh.ply").is_file()


def _finished_object(http, monkeypatch, job_dir: Path, slug: str = "table") -> Path:
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    assert http.post("/api/splat/jobs/splat_0b0001/objects",
                     json={"query": slug, "finish": True}).status_code == 200
    return job_dir / splat_route.OBJECTS_DIRNAME / slug


def _patch_json(path: Path, **fields) -> None:
    doc = json.loads(path.read_text())
    for k, v in fields.items():
        if v is None:
            doc.pop(k, None)
        else:
            doc[k] = v
    path.write_text(json.dumps(doc))


def test_objects_listing_flags_calibration_added_after_the_build(client, monkeypatch):
    """Calibration can land AFTER an object is built (the fire-hydrant field
    proof: meshed 13:05:03, calibrated 13:15:16). Nothing re-stamps the twin, so
    the listing must say so rather than serve scene-units as if they were real."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _finished_object(http, monkeypatch, job_dir)
    # TWIN_REPORT carries no meters_per_unit -> built before any calibration
    _patch_json(obj_dir / "object.json",
                bbox_tight={"min": [0.0, 0.0, 0.0], "max": [0.1, 0.2, 0.4]})
    _patch_json(job_dir / "meta.json", meters_per_unit=2.0)

    cal = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]["calibration"]
    assert cal["stale"] is True
    assert cal["job_meters_per_unit"] == 2.0
    assert cal["artifact_meters_per_unit"] is None
    assert "Rebuild the twin" in cal["detail"]
    # semantic bbox * mpu -> real dims; scene z is up
    assert cal["object_dims_m"] == [0.2, 0.4, 0.8]
    assert cal["object_height_m"] == 0.8


def test_objects_listing_calibration_current_is_not_stale(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _finished_object(http, monkeypatch, job_dir)
    _patch_json(obj_dir / "mesh" / "twin_finish.json", meters_per_unit=2.0, units="meters")
    _patch_json(job_dir / "meta.json", meters_per_unit=2.0)

    cal = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]["calibration"]
    assert cal["stale"] is False
    assert cal["artifact_meters_per_unit"] == 2.0
    assert cal["detail"] == "Twin carries the job's current calibration."


def test_objects_listing_recalibration_marks_the_twin_stale(client, monkeypatch):
    """A twin built at one scale must not silently survive a scale change."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _finished_object(http, monkeypatch, job_dir)
    _patch_json(obj_dir / "mesh" / "twin_finish.json", meters_per_unit=2.0, units="meters")
    _patch_json(job_dir / "meta.json", meters_per_unit=3.5)

    cal = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]["calibration"]
    assert cal["stale"] is True
    assert cal["artifact_meters_per_unit"] == 2.0 and cal["job_meters_per_unit"] == 3.5


def test_objects_listing_uncalibrated_job_makes_no_scale_claim(client, monkeypatch):
    """No calibration anywhere is not staleness, and must not invent metres."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    obj_dir = _finished_object(http, monkeypatch, job_dir)
    _patch_json(obj_dir / "object.json",
                bbox_tight={"min": [0.0, 0.0, 0.0], "max": [0.1, 0.2, 0.4]})

    cal = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]["calibration"]
    assert cal["stale"] is False
    assert cal["job_meters_per_unit"] is None
    assert cal["object_dims_m"] is None and cal["object_height_m"] is None
    assert "uncalibrated" in cal["detail"]


def test_objects_listing_calibration_without_a_twin(client, monkeypatch):
    """Splat-only build: no twin on disk -> nothing to call stale."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(job_dir, calls))
    assert http.post("/api/splat/jobs/splat_0b0001/objects",
                     json={"query": "table", "mesh": False}).status_code == 200
    _patch_json(job_dir / "meta.json", meters_per_unit=2.0)

    cal = http.get("/api/splat/jobs/splat_0b0001/objects").json()["objects"][0]["calibration"]
    assert cal["stale"] is False
    assert cal["detail"] == "No twin built for this object yet."


def test_objects_finish_failure_is_loud_500(client, monkeypatch):
    """A failed finish must not roll back the already-succeeded raw mesh
    artifacts -- they were a complete, valid build on their own."""
    http, outputs = client
    job_dir = _mk_job(outputs)
    calls: list = []
    base_fake = _fake_subprocess(job_dir, calls)

    async def run(command):
        joined = " ".join(str(c) for c in command)
        if "twin_finish" in joined:
            return 1, b"", b"FATAL: pymeshlab decimation crashed"
        return await base_fake(command)

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", run)
    r = http.post("/api/splat/jobs/splat_0b0001/objects",
                  json={"query": "table", "finish": True})
    assert r.status_code == 500
    obj_dir = job_dir / splat_route.OBJECTS_DIRNAME / "table"
    assert (obj_dir / "mesh" / "mesh.glb").is_file()
    assert (obj_dir / "mesh" / "mesh.json").is_file()


def test_listing_carries_bakeoff_verdict(client):
    http, outputs = client
    job_dir = _mk_job(outputs, langfield=False)
    obj_dir = job_dir / "_objects" / "crate"
    (obj_dir / "mesh" / "bakeoff").mkdir(parents=True)
    (obj_dir / "object.json").write_text(json.dumps({"slug": "crate"}))
    (obj_dir / "mesh" / "bakeoff" / "bakeoff.json").write_text(json.dumps({
        "verdict": {
            "winner": "textured",
            "reason": "textured scores 13.76 dB",
            "ranked": [
                {"name": "textured", "median_psnr_paired": 13.76,
                 "median_coverage_paired": 0.074},
                {"name": "raw_tsdf", "median_psnr_paired": 13.55,
                 "median_coverage_paired": 0.176},
            ],
        },
    }))
    listing = http.get("/api/splat/jobs/splat_0b0001/objects").json()
    entry = listing["objects"][0]
    assert entry["bakeoff"]["winner"] == "textured"
    assert entry["bakeoff"]["ranked"][0] == {
        "name": "textured", "median_psnr_paired": 13.76}


def test_listing_bakeoff_absent_and_corrupt(client):
    http, outputs = client
    job_dir = _mk_job(outputs, langfield=False)
    for slug, payload in (("plain", None), ("broken", "{not json")):
        obj_dir = job_dir / "_objects" / slug
        (obj_dir / "mesh" / "bakeoff").mkdir(parents=True)
        (obj_dir / "object.json").write_text(json.dumps({"slug": slug}))
        if payload is not None:
            (obj_dir / "mesh" / "bakeoff" / "bakeoff.json").write_text(payload)
    listing = http.get("/api/splat/jobs/splat_0b0001/objects").json()
    by_slug = {e["slug"]: e for e in listing["objects"]}
    assert by_slug["plain"]["bakeoff"] is None
    assert by_slug["broken"]["bakeoff"] == {"error": "unreadable bakeoff.json"}
