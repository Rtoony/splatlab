"""Escalation state must reach the frontend: _new_meta seeds the resolved
start solver + tried set, _maybe_escalate_sfm persists structured reroute
events, and exhaustion guidance is keyed to the capture type. CPU-only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


def _req(**kwargs) -> splat_route.SplatTrainRequest:
    kwargs.setdefault("input_path", "in")
    kwargs.setdefault("output_dir", "outputs/3d")
    return splat_route.SplatTrainRequest(mode="3d", **kwargs)


def test_new_meta_seeds_escalation_visibility(tmp_path):
    ctx = {"start_solver": "glomap", "processed_dir": "/p", "process_input": "/i",
           "subcommand": "images", "is_equirect": False}
    meta = splat_route._new_meta(
        "splat_test", _req(), Path("/in/orbit"), tmp_path, ["glomap_sfm", "process"], ctx)
    assert meta["sfm_start_solver"] == "glomap"
    assert meta["sfm_tried"] == ["glomap"]
    assert meta["reroute_count"] == 0
    assert meta["sfm_reroutes"] == []


def test_new_meta_without_context_is_honest(tmp_path):
    meta = splat_route._new_meta(
        "splat_test", _req(), Path("/in/data"), tmp_path, ["train"], None)
    assert meta["sfm_start_solver"] is None
    assert meta["sfm_reroutes"] == []


def test_escalation_persists_reroute_event(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    job_id = "splat_reroute"
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    monkeypatch.setattr(
        splat_route, "_engine_availability",
        lambda: {"glomap_available": True, "glomap_path": "/bin/colmap4",
                 "ffmpeg_path": "/bin/ffmpeg", "ns_process_data_path": "/bin/npd"})

    ctx = {"start_solver": "colmap", "processed_dir": str(job_dir / "processed"),
           "process_input": "/in/orbit", "subcommand": "images", "is_equirect": False}
    job = splat_route.SplatJob(
        job_id=job_id, output_dir=str(job_dir), input_path="/in/orbit",
        stages_planned=["process", "train"],
        stage_commands={"process": ["true"], "train": ["true"]},
        sfm_tried={"colmap"}, sfm_context=ctx, sfm_req=_req())
    (job_dir / "meta.json").write_text(json.dumps(
        splat_route._new_meta(job_id, _req(), Path("/in/orbit"), job_dir,
                              ["process", "train"], ctx)))

    assert splat_route._maybe_escalate_sfm(job, 0, 3, 29, "10.3%") is True

    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["sfm_tried"] == ["colmap", "glomap"]
    assert meta["reroute_count"] == 1
    (event,) = meta["sfm_reroutes"]
    assert event["from_solver"] == "colmap"
    assert event["to_solver"] == "glomap"
    assert event["registered"] == 3 and event["extracted"] == 29
    assert event["pct"] == "10.3%"
    assert event["at"]


def test_recapture_guidance_keys_on_capture_type():
    photo = splat_route._recapture_guidance(
        {"subcommand": "images", "is_equirect": False})
    video = splat_route._recapture_guidance(
        {"subcommand": "video", "is_equirect": False})
    equirect = splat_route._recapture_guidance(
        {"subcommand": "video", "is_equirect": True})
    assert "orbit" in photo and "photos" in photo
    assert "sweeps" in video
    assert "OVERHEAD" in equirect
    # No context (pre-processed dataset) falls back to the generic video text.
    assert splat_route._recapture_guidance(None) == video
