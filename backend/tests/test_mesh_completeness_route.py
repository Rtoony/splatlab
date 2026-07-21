"""gate=true on POST /jobs/{id}/mesh must ALSO run the WS2 completeness metric
afterward: same export lock, plain CPU subprocess (no arbiter lane), headline
numbers merged into meta.mesh.gate.completeness, --meters-per-unit forwarded
from meta. A missing _preview/splat.ply skips with a note — never a failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI, Request  # noqa: F401
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


RESERVED: list[int] = []


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "_mesh_available", lambda: True)
    monkeypatch.setattr(splat_route, "_find_latest_config", lambda d: d / "config.yml")

    async def fake_audit(**kwargs):
        return None

    monkeypatch.setattr(splat_route, "audit_operator_event", fake_audit)

    RESERVED.clear()

    async def fake_gpu(lane, operation_id, vram_mb, operation, **kw):
        RESERVED.append(vram_mb)
        return await operation()

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_gate_ready_job(
    outputs: Path,
    job_id: str = "splat_c50001",
    with_splat: bool = True,
    meters_per_unit: float | None = 2.3537,
) -> Path:
    job_dir = outputs / job_id
    (job_dir / splat_route.MESH_DIRNAME).mkdir(parents=True)
    meta = {
        "job_id": job_id, "output_dir": str(job_dir), "status": "completed",
        "input_path": "/in/clip.mp4", "mode": "3d",
    }
    if meters_per_unit is not None:
        meta["meters_per_unit"] = meters_per_unit
    (job_dir / "meta.json").write_text(json.dumps(meta))
    (job_dir / splat_route.MESH_DIRNAME / "mesh.ply").write_bytes(b"ply")
    (job_dir / splat_route.MESH_DIRNAME / "mesh.json").write_text(
        json.dumps({"v": 1, "tris": 1, "recipe": {"checkpoint": "vanilla"}})
    )
    cfg = job_dir / "processed" / "splatfacto" / "ts"
    cfg.mkdir(parents=True)
    (cfg / "config.yml").write_text("cfg")
    if with_splat:
        (job_dir / splat_route.PREVIEW_DIRNAME).mkdir()
        (job_dir / splat_route.PREVIEW_DIRNAME / "splat.ply").write_bytes(b"ply")
    return job_dir


def _write_gate_json(job_dir: Path) -> None:
    (job_dir / splat_route.MESH_DIRNAME / "mesh_gate.json").write_text(json.dumps({
        "v": 1, "median_coverage": 0.745, "median_psnr": 11.86,
        "median_ssim": 0.216, "convention": "o3d-unlit", "per_cam": [],
    }))


def _write_completeness_json(job_dir: Path) -> None:
    (job_dir / splat_route.MESH_DIRNAME / "completeness.json").write_text(json.dumps({
        "v": 1, "solid_total": 500000, "solid_in_bbox": 400000,
        "pct_within_5cm": 70.4, "pct_beyond_10cm": 20.1,
        "median_cm": 2.1, "p90_cm": 27.4, "units": "cm",
        "meters_per_unit": 2.3537, "seconds": 9.0,
        "lab_reference_blender": {"pct_within_5cm": 70.4},
    }))


def test_gate_runs_completeness_and_merges_meta(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_gate_ready_job(outputs)
    calls: list = []

    async def fake_sub(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "mesh_gate" in joined:
            _write_gate_json(job_dir)
        else:
            assert "mesh_completeness" in joined
            _write_completeness_json(job_dir)
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_c50001/mesh", json={"gate": True})
    assert r.status_code == 200
    assert len(calls) == 2  # gate, then completeness

    comp_cmd = [str(c) for c in calls[1]]
    assert str(job_dir / splat_route.PREVIEW_DIRNAME / "splat.ply") in comp_cmd
    assert str(job_dir / splat_route.MESH_DIRNAME / "completeness.json") in comp_cmd
    assert comp_cmd[-2:] == ["--meters-per-unit", "2.3537"]  # forwarded from meta

    # Completeness is CPU/non-arbiter: only mesh + gate lanes ever reserved VRAM.
    assert RESERVED == [splat_route.MESH_VRAM_MB, splat_route.MESH_GATE_VRAM_MB]

    meta = json.loads((job_dir / "meta.json").read_text())
    comp = meta["mesh"]["gate"]["completeness"]
    assert comp["pct_within_5cm"] == 70.4
    assert comp["pct_beyond_10cm"] == 20.1
    assert comp["median_cm"] == 2.1
    assert comp["p90_cm"] == 27.4
    assert comp["solid_in_bbox"] == 400000
    # meta stays lean — full report remains in _mesh/completeness.json only.
    assert "seconds" not in comp
    assert "lab_reference_blender" not in comp


def test_missing_splat_skips_with_note_never_fails(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_gate_ready_job(outputs, with_splat=False)
    calls: list = []

    async def fake_sub(command):
        calls.append(command)
        assert "mesh_completeness" not in " ".join(str(c) for c in command)
        _write_gate_json(job_dir)
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_c50001/mesh", json={"gate": True})
    assert r.status_code == 200  # gate result survives; completeness only noted
    assert len(calls) == 1  # gate only

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["mesh"]["gate"]["median_psnr"] == 11.86  # gate scores intact
    assert "splat.ply" in meta["mesh"]["gate"]["completeness"]["skipped"]


def test_no_meters_per_unit_omits_flag(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_gate_ready_job(outputs, meters_per_unit=None)
    calls: list = []

    async def fake_sub(command):
        calls.append(command)
        joined = " ".join(str(c) for c in command)
        if "mesh_gate" in joined:
            _write_gate_json(job_dir)
        elif "mesh_completeness" in joined:
            _write_completeness_json(job_dir)
        return 0, b"", b""

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_c50001/mesh", json={"gate": True})
    assert r.status_code == 200
    assert len(calls) == 2
    assert "--meters-per-unit" not in [str(c) for c in calls[1]]


def test_completeness_failure_is_loud(client, monkeypatch):
    http, outputs = client
    job_dir = _mk_gate_ready_job(outputs)

    async def fake_sub(command):
        joined = " ".join(str(c) for c in command)
        if "mesh_gate" in joined:
            _write_gate_json(job_dir)
            return 0, b"", b""
        return 1, b"", b"FATAL: empty mesh"

    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_sub)
    r = http.post("/api/splat/jobs/splat_c50001/mesh", json={"gate": True})
    assert r.status_code == 500
    assert "completeness" in r.json()["detail"].lower()
    assert "FATAL: empty mesh" in r.json()["detail"]
