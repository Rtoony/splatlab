"""DEFAULT-FLIP (2026-07-17): still-photo folders start on glomap when colmap4
is present — the same-subject A/B registered 29/29 (glomap) vs 17/29 (incremental
COLMAP) on a low-light iPhone orbit. Video and equirect lanes must be untouched,
sparse mode must still override, and a missing toolchain must fall back to
colmap. CPU-only, no subprocess."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402

AVAIL = {
    "ns_process_data_available": True, "ns_process_data_path": "/bin/ns-process-data",
    "ns_train_available": True, "ns_train_path": "/bin/ns-train",
    "ns_export_available": True, "ns_export_path": "/bin/ns-export",
    "colmap_available": True, "colmap_path": "/bin/colmap",
    "glomap_available": True, "glomap_path": "/bin/colmap4",
    "mast3r_available": False,
    "ffmpeg_available": True, "ffmpeg_path": "/bin/ffmpeg",
    "insv_stitch_available": True,
    "triposplat_available": False,
    "langfield_available": False,
}


def _plan(req, input_path, monkeypatch, avail=None):
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: None)
    monkeypatch.setattr(splat_route, "_health_available", lambda: False)
    return splat_route._plan_3d_job(
        req, avail or AVAIL, Path("/jobs/splat_snapshot"), Path(input_path)
    )


def _photo_dir(tmp_path: Path) -> Path:
    images = tmp_path / "orbit"
    images.mkdir()
    (images / "img_0001.jpg").write_bytes(b"\xff\xd8")
    return images


def test_default_photo_folder_flips_to_glomap(tmp_path, monkeypatch):
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="orbit", output_dir="outputs/3d")
    stages, commands, ctx = _plan(req, _photo_dir(tmp_path), monkeypatch)
    assert ctx["start_solver"] == "glomap"
    assert stages[0] == "glomap_sfm"
    assert "--skip-colmap" in commands["process"]


def test_photo_folder_stays_colmap_without_toolchain(tmp_path, monkeypatch):
    avail = {**AVAIL, "glomap_available": False}
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="orbit", output_dir="outputs/3d")
    stages, commands, ctx = _plan(req, _photo_dir(tmp_path), monkeypatch, avail=avail)
    assert ctx["start_solver"] == "colmap"
    assert stages[0] == "process"
    assert "--skip-colmap" not in commands["process"]


def test_explicit_glomap_request_unchanged(tmp_path, monkeypatch):
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="orbit", output_dir="outputs/3d", sfm_backend="glomap")
    _, _, ctx = _plan(req, _photo_dir(tmp_path), monkeypatch)
    assert ctx["start_solver"] == "glomap"


def test_standard_video_keeps_colmap_first(monkeypatch):
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="clip.mp4", output_dir="outputs/3d")
    _, _, ctx = _plan(req, "/in/clip.mp4", monkeypatch)
    assert ctx["start_solver"] == "colmap"


def test_equirect_photo_folder_not_flipped(tmp_path, monkeypatch):
    # Equirect stills are excluded — the flip is for flat photo orbits only.
    avail = {**AVAIL, "rig_available": False}
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="orbit", output_dir="outputs/3d",
        capture_format="equirectangular360")
    _, _, ctx = _plan(req, _photo_dir(tmp_path), monkeypatch, avail=avail)
    assert ctx["start_solver"] == "colmap"


def test_sparse_mode_still_overrides_to_mast3r(tmp_path, monkeypatch):
    avail = {
        **AVAIL,
        "mast3r_available": True,
        "mast3r_python": "/fake/env/python",
        "mast3r_runner": "/fake/run_mast3r_sfm.py",
        "mast3r_converter": "/fake/mast3r_to_nerfstudio.py",
        "mast3r_checkpoint": "/fake/ckpt.pth",
    }
    req = splat_route.SplatTrainRequest(
        mode="3d", input_path="orbit", output_dir="outputs/3d", capture_mode="sparse")
    stages, _, ctx = _plan(req, _photo_dir(tmp_path), monkeypatch, avail=avail)
    assert stages[0] == "mast3r_sfm"
    assert ctx is None  # sparse jobs are not escalation-eligible
