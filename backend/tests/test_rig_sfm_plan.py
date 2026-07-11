"""Rig-lane plan construction: the opt-in sfm_backend="rig" must emit the
rig_sfm + process stage pair with the rig-constraint flags, and its
availability must gate on the colmap4 binary + render script + env python.
CPU-only, no subprocess."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


def _availability() -> dict:
    return {
        "glomap_path": "/fake/colmap4/bin/colmap",
        "ffmpeg_path": "/fake/bin/ffmpeg",
        "ns_process_data_path": "/fake/bin/ns-process-data",
    }


def test_rig_solver_emits_rig_sfm_and_process(tmp_path):
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="clip.insv", output_dir="outputs/3d",
        capture_format="equirectangular360", sfm_backend="rig", num_frames_target=90)
    cmds = splat_route._sfm_stage_commands(
        solver="rig", req=req, availability=_availability(),
        job_dir=tmp_path, processed_dir=tmp_path / "processed",
        process_input=tmp_path / "stitched" / "equirect.mp4",
        subcommand="video", is_equirect=True)

    assert list(cmds) == ["rig_sfm", "process"]
    script = cmds["rig_sfm"][2]
    for needle in (
        "render_rig.py",
        "rig_configurator",
        "--FeatureMatching.rig_verification 1",
        "--FeatureMatching.skip_image_pairs_in_same_frame 1",
        "--GlobalMapper.refine_sensor_from_rig 0",
        "--ImageReader.single_camera_per_folder 1",
        "cameras.bin",
    ):
        assert needle in script, f"missing {needle!r} in rig_sfm script"
    process = cmds["process"]
    assert "--skip-colmap" in process
    assert str(tmp_path / "rig" / "images") in process


def test_rig_request_model_accepts_backend():
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="clip.insv", output_dir="outputs/3d", sfm_backend="rig")
    assert req.sfm_backend == "rig"


def test_rig_availability_key_registered():
    assert splat_route.SFM_SOLVER_AVAILABILITY["rig"] == "rig_available"
    # rig must NOT be an auto-escalation rung (opt-in only until default-flip).
    assert "rig" not in splat_route.SFM_ESCALATION
