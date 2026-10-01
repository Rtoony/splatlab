"""Enhance this view: a viewer frame -> AI-enhanced still. The reference photo is used only when a real camera stood
near this spot facing the same way (realism lab: a mismatched reference pastes in the wrong content)."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enhance_route  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_e7e001"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()


def c2w(x, y, z, fwd):
    """OpenGL c2w whose -z axis is `fwd` (only the forward column matters to nearest_view)."""
    return [[1, 0, -fwd[0], x], [0, 1, -fwd[1], y], [0, 0, -fwd[2], z]]


def test_nearest_view_prefers_close_and_facing_the_same_way():
    views = [{"c2w": c2w(0, 0, 1.5, [1, 0, 0])},       # here, facing +x
             {"c2w": c2w(0.5, 0, 1.5, [-1, 0, 0])},    # closer to the target but facing away
             {"c2w": c2w(10, 0, 1.5, [1, 0, 0])}]      # same way, far
    i, d, c = enhance_route.nearest_view(views, [0.4, 0, 1.5], [1, 0, 0])
    assert i == 0 and abs(d - 0.4) < 1e-6 and c > 0.99


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(enhance_route, "available", lambda: True)
    job = outputs / JOB
    (job / "_spirula" / "data" / "sparse" / "0").mkdir(parents=True)
    (job / "_spirula" / "data" / "sparse" / "0" / "images.bin").write_bytes(b"x")
    vd = job / "_enhance" / "views"; (vd / "frames").mkdir(parents=True)
    (vd / "frames" / "frame_000.png").write_bytes(b"REALPHOTO")
    (vd / "cameras.json").write_text(json.dumps({"views": [{"c2w": c2w(0, 0, 1.5, [1, 0, 0]), "source": "cam0/00009.jpg"}]}))
    (job / "meta.json").write_text(json.dumps({"job_id": JOB, "output_dir": str(job), "status": "completed", "mode": "3d"}))
    calls = []

    async def fake_gpu(**kw):
        calls.append(kw)  # the subprocess is never run: the runner is emulated below
        return 0, json.dumps({"ok": True, "ms": 300, "model": "nvidia/difix_ref"}).encode(), b""

    async def fake_runner(**kw):
        # write the output where the runner would
        r = await fake_gpu(**kw)
        shots = sorted((job / "_enhance" / "shots").iterdir())
        (shots[-1] / "output.png").write_bytes(b"ENHANCED")
        return r

    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_runner)
    app = FastAPI(); app.include_router(enhance_route.router, prefix="/api/splat")
    return TestClient(app), job, calls


def test_enhance_uses_the_nearby_real_photo_and_keeps_every_file(client):
    tc, job, calls = client
    r = tc.post(f"/api/splat/jobs/{JOB}/enhance", json={"image": f"data:image/png;base64,{PNG}",
                                                       "camera": {"position": [0.5, 0, 1.5], "forward": [1, 0, 0]}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"] == "AI-enhanced" and d["reference"]["used"] and d["ref_url"]
    assert calls[0]["lane"] == "splat-enhance" and calls[0]["vram_mb"] == enhance_route.ENHANCE_VRAM_MB
    sd = job / "_enhance" / "shots" / d["shot_id"]
    assert (sd / "input.png").read_bytes().startswith(b"\x89PNG") and (sd / "ref.png").read_bytes() == b"REALPHOTO"
    assert json.loads((sd / "shot.json").read_text())["note"].startswith("Generated detail")
    assert tc.get(d["after_url"]).content == b"ENHANCED"
    assert "ai-enhanced" in tc.get(d["after_url"]).headers["content-disposition"]


def test_a_far_or_backwards_reference_is_not_used(client):
    tc, job, _ = client
    r = tc.post(f"/api/splat/jobs/{JOB}/enhance", json={"image": f"data:image/png;base64,{PNG}",
                                                       "camera": {"position": [0, 0, 1.5], "forward": [-1, 0, 0]}})
    d = r.json()
    assert r.status_code == 200 and d["reference"]["used"] is False and d["ref_url"] is None
    assert "plain Difix" in d["reference"]["why_not"]


def test_bad_inputs_are_refused(client):
    tc, _, _ = client
    assert tc.post(f"/api/splat/jobs/{JOB}/enhance", json={"image": "not a data url"}).status_code == 400
    assert tc.get(f"/api/splat/jobs/{JOB}/enhance/../../etc/passwd").status_code == 404
    assert tc.get(f"/api/splat/jobs/{JOB}/enhance/20261001T000000-abcdef/meta").status_code == 404
