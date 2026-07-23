"""Scene assembly route (P6f): requires completed batch isolation, the
two-stage build (fidelity-dial resolution -> headless Blender build+export+
contamination-gate), the mandatory-HITL approve step staying separate from
the build itself, and slug-safe file serving. Subprocesses mocked; the
fidelity-resolution logic itself is proven directly against real garden data
in a live run (see STATUS.md), and the Blender mechanics (glTF export_extras,
the chaos test) are proven by hand against the real binary before this route
existed.
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

ASSEMBLE_REPORT = {
    "n_elements_total": 4, "n_built": 4,
    "built": ["background", "round-wooden-table", "flower-vase", "ground"],
    "n_flagged": 0, "flagged": [],
    "contamination_gate": {"ok": True, "errors": []},
    "seconds": 2.4,
}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    # BLENDER_BIN/SCENE_ASSEMBLE_SCRIPT/BLENDER_ASSEMBLE_SCRIPT/MESH_ENV_PYTHON
    # are left as the REAL paths (all genuinely exist on this machine) -- the
    # route's toolchain-availability check is a plain .is_file() test, and
    # _run_capture_subprocess is mocked below so nothing is actually invoked.

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_job_with_isolation(outputs: Path, job_id: str = "splat_0b0006") -> Path:
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d", "meters_per_unit": 2.35,
    }))
    (job_dir / splat_route.SCENE_DIRNAME / "isolated").mkdir(parents=True)
    (job_dir / splat_route.SCENE_DIRNAME / "isolated" / "batch_isolate.json").write_text("{}")
    return job_dir


def _fake_subprocess(calls: list, resolve_ok: bool = True, build_ok: bool = True,
                     contamination_ok: bool = True):
    async def run(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "scene_assemble.py" in joined:
            if not resolve_ok:
                return 1, b"", b"FATAL: overrides name unknown slug(s): ['bogus']"
            manifest_path = Path(command[4])
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            # scene_assemble.py's REAL new_manifest() writes state="building" --
            # matching that here is what caught the real bug (the route never
            # promoted it to "built") that a hardcoded "built" mock papered over.
            manifest_path.write_text(json.dumps({
                "version": 1, "job_id": "splat_0b0006", "created_at": "2026-07-23T00:00:00",
                "state": "building", "doctrine": "SplatLab-provenance: generative render-vr-only",
                "units": {"mode": "meters", "meters_per_unit": 2.35}, "elements": [],
            }))
            return 0, b"", b""
        if "blender_assemble.py" in joined:
            if not build_ok:
                return 1, b"", b"FATAL: zero elements built"
            out_dir = Path(command[-1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "scene.blend").write_bytes(b"blend")
            (out_dir / "scene.glb").write_bytes(b"glb")
            report = dict(ASSEMBLE_REPORT)
            if not contamination_ok:
                report["contamination_gate"] = {"ok": False, "errors": ["proxy node missing tag"]}
            (out_dir / "assemble_report.json").write_text(json.dumps(report))
            return 0, b"", b""
        raise AssertionError(f"unexpected subprocess: {command}")

    return run


def test_scene_assemble_requires_isolation(client):
    http, outputs = client
    job_dir = outputs / "splat_0b0006"
    job_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text(json.dumps({
        "job_id": "splat_0b0006", "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }))
    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble", json={})
    assert r.status_code == 409
    assert "isolation" in r.json()["detail"].lower()


def test_scene_assemble_default_mode_is_styled(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "styled"
    assert body["assemble"]["n_built"] == 4

    resolve_call = [str(c) for c in calls[0]]
    assert "--mode" in resolve_call and "styled" in resolve_call

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["scene"]["assemble"]["state"] == "built"
    assert meta["scene"]["assemble"]["n_built"] == 4


def test_scene_assemble_faithful_mode_and_overrides_pass_through(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))

    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble",
                  json={"mode": "faithful", "overrides": {"round-wooden-table": "proxy"}})
    assert r.status_code == 200
    resolve_call = [str(c) for c in calls[0]]
    assert "faithful" in resolve_call
    assert '{"round-wooden-table": "proxy"}' in resolve_call


def test_scene_assemble_bad_override_is_422(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(calls, resolve_ok=False))
    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble",
                  json={"overrides": {"bogus": "proxy"}})
    assert r.status_code == 422


def test_scene_assemble_contamination_gate_failure_is_500(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess",
                        _fake_subprocess(calls, contamination_ok=False))
    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble", json={})
    assert r.status_code == 500
    assert "contamination" in r.json()["detail"].lower()


def test_scene_assemble_never_auto_approved(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))
    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble", json={})
    assert r.status_code == 200
    manifest = json.loads((job_dir / splat_route.REGEN_DIRNAME / "scene_manifest.json").read_text())
    assert manifest["state"] == "built"  # NOT "approved" -- HITL is a separate call


def test_scene_assemble_approve_requires_prior_build(client):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble/approve")
    assert r.status_code == 409


def test_scene_assemble_approve_flips_state(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))
    http.post("/api/splat/jobs/splat_0b0006/scene/assemble", json={})

    r = http.post("/api/splat/jobs/splat_0b0006/scene/assemble/approve")
    assert r.status_code == 200
    assert r.json()["state"] == "approved"
    manifest = json.loads((job_dir / splat_route.REGEN_DIRNAME / "scene_manifest.json").read_text())
    assert manifest["state"] == "approved"


def test_scene_assemble_file_serving(client, monkeypatch):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    calls: list = []
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", _fake_subprocess(calls))
    http.post("/api/splat/jobs/splat_0b0006/scene/assemble", json={})

    for fmt in ("report", "glb", "blend", "manifest"):
        resp = http.get("/api/splat/jobs/splat_0b0006/scene/assemble/file", params={"fmt": fmt})
        assert resp.status_code == 200, fmt


def test_scene_assemble_file_404_before_build(client):
    http, outputs = client
    _mk_job_with_isolation(outputs)
    r = http.get("/api/splat/jobs/splat_0b0006/scene/assemble/file", params={"fmt": "glb"})
    assert r.status_code == 404
