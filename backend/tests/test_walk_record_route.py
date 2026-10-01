"""Record walk: a WebM of the walker canvas -> a plain MP4 kept beside the scene, and an optional realism finish that
runs in the background (GPU + ComfyUI are faked here; ffmpeg is real)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_route  # noqa: E402
import walk_record_route  # noqa: E402

JOB = "splat_7a1c00"
pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@pytest.fixture(scope="module")
def webm(tmp_path_factory) -> bytes:
    """1 s at 30 fps, 1300x702 (odd-ish, wider than the 1280 cap) - what a browser canvas recording looks like."""
    p = tmp_path_factory.mktemp("rec") / "rec.webm"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=1300x702:rate=30:duration=1",
                    "-c:v", "libvpx", "-b:v", "1M", str(p)], check=True)
    return p.read_bytes()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    job = outputs / JOB
    job.mkdir(parents=True)
    (job / "meta.json").write_text(json.dumps({"job_id": JOB, "output_dir": str(job), "status": "completed", "mode": "3d"}))
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(splat_route, "require_heavy_work_admitted", lambda: None)
    monkeypatch.setattr(walk_record_route, "finish_available", lambda: True)
    calls: list[dict] = []

    async def no_comfy_start():
        return False

    async def fake_gpu(**kw):
        calls.append(kw)
        rc, out, err = await kw["operation"]()
        return rc, out, err

    async def fake_film(cmd):
        if "finish" in cmd:                       # the film CLI: write the --out file like nexus-film would
            Path(cmd[cmd.index("--out") + 1]).write_bytes(b"FINISHED")
            return 0, b"FINISHED: ok\n", b""
        return await real_run(cmd)

    real_run = splat_route._run_capture_subprocess
    monkeypatch.setattr(walk_record_route, "_ensure_comfy", no_comfy_start)
    monkeypatch.setattr(splat_route.gpu_arbiter, "run_gpu_operation", fake_gpu)
    monkeypatch.setattr(splat_route, "_run_capture_subprocess", fake_film)
    app = FastAPI(); app.include_router(walk_record_route.router, prefix="/api/splat")
    with TestClient(app) as tc:                    # one event loop for the whole test: the finish task survives requests
        yield tc, job, calls


def _record(tc, data: bytes):
    return tc.post(f"/api/splat/jobs/{JOB}/walks", content=data, headers={"Content-Type": "video/webm"})


def test_a_recording_becomes_a_plain_mp4_and_the_original_is_kept(client, webm):
    tc, job, _ = client
    r = _record(tc, webm)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["width"] == 1280 and d["height"] % 2 == 0 and d["fps"] == 24 and d["frames"] == 24 and d["seconds"] == 1.0
    wd = job / "_walks" / d["walk_id"]
    assert (wd / "recording.webm").read_bytes() == webm and (wd / "walk.mp4").stat().st_size > 1000
    assert d["finish"]["status"] == "none" and d["realism_url"] is None and d["finish_estimate_s"] > 0
    v = tc.get(d["video_url"])
    assert v.status_code == 200 and v.headers["content-type"] == "video/mp4"


def test_realism_finish_runs_in_the_background_under_the_gpu_arbiter(client, webm):
    tc, job, calls = client
    d = _record(tc, webm).json()
    f = tc.post(f"/api/splat/jobs/{JOB}/walks/{d['walk_id']}/finish").json()
    assert f["finish"]["status"] in ("running", "done")
    for _ in range(50):
        s = tc.get(f"/api/splat/jobs/{JOB}/walks/{d['walk_id']}").json()
        if s["finish"]["status"] != "running":
            break
        time.sleep(0.05)
    assert s["finish"]["status"] == "done", s
    assert calls[0]["lane"] == "splat-walk-finish" and calls[0]["vram_mb"] == walk_record_route.FINISH_VRAM_MB
    assert tc.get(s["realism_url"]).content == b"FINISHED"
    assert "realism-finish" in tc.get(s["realism_url"]).headers["content-disposition"]
    assert (job / "_walks" / d["walk_id"] / "walk.mp4").is_file()              # the plain walk is kept


def test_a_finish_killed_by_a_restart_reads_as_interrupted(client, webm):
    tc, job, _ = client
    d = _record(tc, webm).json()
    wd = job / "_walks" / d["walk_id"]
    doc = json.loads((wd / "walk.json").read_text()); doc["finish"] = {"status": "running", "started": time.time()}
    (wd / "walk.json").write_text(json.dumps(doc))
    s = tc.get(f"/api/splat/jobs/{JOB}/walks/{d['walk_id']}").json()
    assert s["finish"]["status"] == "interrupted" and "again" in s["finish"]["error"]


def test_bad_inputs_are_refused(client):
    tc, _, _ = client
    assert _record(tc, b"").status_code == 413
    assert _record(tc, b"not a webm at all").status_code == 400
    assert tc.get(f"/api/splat/jobs/{JOB}/walks/20261001T000000-abcdef").status_code == 404
    assert tc.get(f"/api/splat/jobs/{JOB}/walks/..%2F..%2Fetc/walk").status_code == 404
    assert tc.get(f"/api/splat/jobs/{JOB}/walks/20261001T000000-abcdef/recording").status_code == 404


def test_finish_estimate_scales_with_length_and_size():
    assert walk_record_route.estimate_finish_seconds(5, 1024, 576) == 85
    assert walk_record_route.estimate_finish_seconds(10, 1280, 720) == 266
