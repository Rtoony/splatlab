"""Viewing-quality tiers: full only on the Nexus PC; compressed web / lite everywhere else (owner 2026-09-30)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quality_tiers  # noqa: E402
import splat_route  # noqa: E402

JOB = "splat_7e0001"


def _ply(path: Path, n: int, body: bytes = b"\0" * 64, extra: str = "") -> None:
    path.write_bytes(f"ply\nformat binary_little_endian 1.0\n{extra}element vertex {n}\n"
                     f"property float x\nend_header\n".encode() + body)


@pytest.fixture()
def job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    d = outputs / JOB
    (d / "_preview").mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"job_id": JOB, "output_dir": str(d), "status": "completed",
                                             "mode": "3d"}))
    app = FastAPI()
    app.include_router(splat_route.router, prefix="/api/splat")
    return TestClient(app), d / "_preview"


def test_only_a_direct_loopback_request_is_local():
    assert quality_tiers.is_local_request("127.0.0.1", {})
    assert quality_tiers.is_local_request("::1", {"host": "127.0.0.1:3416"})
    # The Cloudflare tunnel connects from loopback too — its headers are what give it away.
    assert not quality_tiers.is_local_request("127.0.0.1", {"cf-connecting-ip": "203.0.113.9"})
    assert not quality_tiers.is_local_request("127.0.0.1", {"x-forwarded-for": "10.0.0.2"})
    assert not quality_tiers.is_local_request("192.168.87.20", {})
    assert not quality_tiers.is_local_request(None, {})


def test_vertex_count_reads_plain_and_compressed_headers(tmp_path):
    _ply(tmp_path / "a.ply", 2_994_194)
    _ply(tmp_path / "b.compressed.ply", 1_000_000, extra="element chunk 3907\nproperty float min_x\n")
    assert quality_tiers.ply_vertex_count(tmp_path / "a.ply") == 2_994_194
    assert quality_tiers.ply_vertex_count(tmp_path / "b.compressed.ply") == 1_000_000
    assert quality_tiers.ply_vertex_count(tmp_path / "missing.ply") is None


def test_tiers_list_what_exists_and_web_falls_back_to_the_legacy_copy(tmp_path):
    _ply(tmp_path / "splat.ply", 10)
    _ply(tmp_path / "web.ply", 5)
    tiers = quality_tiers.tiers(tmp_path, JOB)
    assert [t["id"] for t in tiers] == ["full", "web"]
    assert tiers[1]["compressed"] is False and tiers[1]["url"].endswith("fmt=webc")
    _ply(tmp_path / "web.compressed.ply", 5)
    _ply(tmp_path / "lite.compressed.ply", 2)
    tiers = {t["id"]: t for t in quality_tiers.tiers(tmp_path, JOB)}
    assert tiers["web"]["compressed"] and tiers["lite"]["splats"] == 2 and tiers["full"]["splats"] == 10


def test_preview_file_serves_each_tier_and_keeps_fmt_web_meaning_web_ply(job):
    tc, prev = job
    _ply(prev / "splat.ply", 10, b"FULL")
    _ply(prev / "web.ply", 5, b"WEBPLY")
    _ply(prev / "web.compressed.ply", 5, b"WEBC")
    assert tc.get(f"/api/splat/jobs/{JOB}/preview/file?fmt=full").content.endswith(b"FULL")
    assert tc.get(f"/api/splat/jobs/{JOB}/preview/file?fmt=webc").content.endswith(b"WEBC")
    assert tc.get(f"/api/splat/jobs/{JOB}/preview/file?fmt=web").content.endswith(b"WEBPLY")
    r = tc.get(f"/api/splat/jobs/{JOB}/preview/file?fmt=lite")
    assert r.status_code == 404 and "lite" in r.json()["detail"]
    assert f'{JOB}-web.compressed.ply' in tc.get(
        f"/api/splat/jobs/{JOB}/preview/file?fmt=webc").headers["content-disposition"]


def test_viewer_context_says_local_only_without_tunnel_headers(job):
    tc, _ = job   # TestClient's client host is "testclient", i.e. not the PC
    assert tc.get("/api/splat/viewer-context").json() == {"local": False}
    assert quality_tiers.is_local_request("127.0.0.1", {"cf-ray": "8c1d"}) is False


def test_the_compressed_tiers_are_built_from_web_ply(tmp_path):
    cmds = dict(quality_tiers.build_commands("st", tmp_path))
    assert cmds["webopt-webc"] == ["st", str(tmp_path / "web.ply"), str(tmp_path / "web.compressed.ply")]
    assert cmds["webopt-lite"][2:4] == ["--decimate", "1000000"]
    assert cmds["webopt-lite"][-1] == str(tmp_path / "lite.compressed.ply")
