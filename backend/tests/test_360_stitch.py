"""360/.insv lane tests: stream-aware stitch planning, the post-stitch sanity
gate, the equirect matcher choice, and the equirect escalation matrix.

CPU-only: no GPU, no ffmpeg execution — every probe/tool lookup is
monkeypatched and only the COMMAND BUILDERS + gate logic are exercised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

AVAIL = {
    "ns_train_available": True, "ns_train_path": "/bin/ns-train",
    "ns_process_data_available": True, "ns_process_data_path": "/bin/ns-process-data",
    "ns_export_available": True, "ns_export_path": "/bin/ns-export",
    "colmap_available": True, "colmap_path": "/bin/colmap",
    "glomap_available": True, "glomap_path": "/bin/colmap4",
    "mast3r_available": False,
    "ffmpeg_available": True, "ffmpeg_path": "/bin/ffmpeg",
    "insv_stitch_available": True,
    "triposplat_available": False,
    "langfield_available": False,
}

# Planner output for a standard (non-360) video job captured BEFORE the 360
# lane changes landed — the default path must stay byte-for-byte identical.
GOLDEN_STANDARD_VIDEO_PLAN = {
    "stages": ["process", "train", "export"],
    "commands": {
        "process": [
            "/bin/ns-process-data", "video",
            "--data", "/in/clip.mp4",
            "--output-dir", "/jobs/splat_snapshot/processed",
            "--num-frames-target", "300",
            "--matching-method", "sequential",
        ],
        "train": [
            "/bin/ns-train", "splatfacto",
            "--data", "/jobs/splat_snapshot/processed",
            "--output-dir", "/jobs/splat_snapshot",
            "--max-num-iterations", "30000",
            "--viewer.quit-on-train-completion", "True",
        ],
    },
    "sfm_context": {
        "start_solver": "colmap",
        "processed_dir": "/jobs/splat_snapshot/processed",
        "process_input": "/in/clip.mp4",
        "subcommand": "video",
        "is_equirect": False,
    },
}


def _plan(req: splat_route.SplatTrainRequest, input_path: str, monkeypatch: pytest.MonkeyPatch,
          probe: dict | None = None):
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: None)
    if probe is not None:
        monkeypatch.setattr(splat_route, "_tool_path", lambda binary, env: f"/bin/{binary}")
        monkeypatch.setattr(splat_route, "_probe_video_streams", lambda ffprobe, src: probe)
    return splat_route._plan_3d_job(req, AVAIL, Path("/jobs/splat_snapshot"), Path(input_path))


def _req(**kwargs) -> splat_route.SplatTrainRequest:
    kwargs.setdefault("input_path", "in")
    return splat_route.SplatTrainRequest(mode="3d", **kwargs)


DUAL_PROBE = {"streams": 2, "width": 3840, "height": 3840,
              "dims": [(3840, 3840), (3840, 3840)]}
SINGLE_PROBE = {"streams": 1, "width": 5760, "height": 2880, "dims": [(5760, 2880)]}


# ---------------------------------------------------------------------------
# R-FIX: stream-count-aware stitch command
# ---------------------------------------------------------------------------


def test_stitch_command_single_stream_is_legacy_form(monkeypatch):
    monkeypatch.setenv("SPLAT_STITCH_CPUS", "0")  # leash off -> byte-for-byte legacy argv
    cmd = splat_route._stitch_command("/bin/ffmpeg", Path("/in/a.insv"), Path("/out/eq.mp4"), 204.0)
    assert cmd == [
        "/bin/ffmpeg", "-y", "-i", "/in/a.insv",
        "-vf",
        "v360=input=dfisheye:output=e:ih_fov=204.0:iv_fov=204.0:w=5760:h=2880:interp=lanczos",
        "-c:v", "libx264", "-crf", "18", "-an", "/out/eq.mp4",
    ]


def test_stitch_command_dual_stream_hstacks_both_lenses(monkeypatch):
    monkeypatch.setenv("SPLAT_STITCH_CPUS", "0")
    cmd = splat_route._stitch_command(
        "/bin/ffmpeg", Path("/in/a.insv"), Path("/out/eq.mp4"), 204.0, dual_stream=True
    )
    assert cmd == [
        "/bin/ffmpeg", "-y", "-i", "/in/a.insv",
        "-filter_complex",
        "[0:v:0][0:v:1]hstack=inputs=2[d];"
        "[d]v360=input=dfisheye:output=e:ih_fov=204.0:iv_fov=204.0:w=5760:h=2880:interp=lanczos[eq]",
        "-map", "[eq]",
        "-c:v", "libx264", "-crf", "18", "-an", "/out/eq.mp4",
    ]


# ---------------------------------------------------------------------------
# CPU leash (2026-07-04 crash guard): the all-core x264 launch is the proven
# trigger of a firmware-fatal hardware reset — the stitch must never again
# slam every core at once.
# ---------------------------------------------------------------------------


def test_stitch_cpu_leash_default_caps_to_half_cores(monkeypatch):
    monkeypatch.delenv("SPLAT_STITCH_CPUS", raising=False)
    cmd = splat_route._stitch_command("/bin/ffmpeg", Path("/in/a.insv"), Path("/out/eq.mp4"), 204.0)
    i = cmd.index("/bin/ffmpeg")
    assert i > 0, "leash prefix missing — stitch would launch all-core"
    prefix = cmd[:i]
    import os as _os
    expected = max(4, (_os.cpu_count() or 8) // 2)
    assert f"0-{expected - 1}" in prefix  # taskset CPU list
    assert "-n" in prefix and "10" in prefix  # nice
    # ffmpeg argv after the prefix is the untouched legacy form
    assert cmd[i:] == [
        "/bin/ffmpeg", "-y", "-i", "/in/a.insv",
        "-vf",
        "v360=input=dfisheye:output=e:ih_fov=204.0:iv_fov=204.0:w=5760:h=2880:interp=lanczos",
        "-c:v", "libx264", "-crf", "18", "-an", "/out/eq.mp4",
    ]


def test_stitch_cpu_leash_env_override(monkeypatch):
    monkeypatch.setenv("SPLAT_STITCH_CPUS", "4")
    prefix = splat_route._stitch_cpu_leash()
    assert "0-3" in prefix


def test_stitch_cpu_leash_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SPLAT_STITCH_CPUS", "banana")
    prefix = splat_route._stitch_cpu_leash()
    assert prefix, "bad env value must not disable the leash"


def test_stitch_cpu_leash_applies_to_dual_stream_too(monkeypatch):
    monkeypatch.setenv("SPLAT_STITCH_CPUS", "4")
    cmd = splat_route._stitch_command(
        "/bin/ffmpeg", Path("/in/a.insv"), Path("/out/eq.mp4"), 204.0, dual_stream=True
    )
    assert cmd.index("/bin/ffmpeg") > 0
    assert "0-3" in cmd[: cmd.index("/bin/ffmpeg")]


@pytest.mark.parametrize(
    "info,expected",
    [
        (SINGLE_PROBE, "single"),
        (DUAL_PROBE, "dual"),
    ],
)
def test_stitch_layout_supported(info, expected):
    layout, err = splat_route._stitch_layout(info)
    assert layout == expected
    assert err is None


@pytest.mark.parametrize(
    "info",
    [
        # 2 streams, mismatched dims
        {"streams": 2, "width": 3840, "height": 3840, "dims": [(3840, 3840), (1920, 1080)]},
        # 2 streams, equal but non-square
        {"streams": 2, "width": 3840, "height": 1920, "dims": [(3840, 1920), (3840, 1920)]},
        # 3+ streams
        {"streams": 3, "width": 3840, "height": 3840, "dims": [(3840, 3840)] * 3},
        # unprobeable
        {"streams": 0, "width": None, "height": None, "dims": []},
    ],
)
def test_stitch_layout_weird_layouts_fail_loud_toward_entry_b(info):
    layout, err = splat_route._stitch_layout(info)
    assert layout == "error"
    assert err is not None and "Insta360 Studio" in err


def test_plan_insv_dual_stream_uses_hstack(monkeypatch):
    stages, commands, ctx = _plan(_req(), "/in/cap.insv", monkeypatch, probe=DUAL_PROBE)
    assert stages[0] == "stitch"
    assert "-filter_complex" in commands["stitch"]
    assert "hstack=inputs=2" in commands["stitch"][commands["stitch"].index("-filter_complex") + 1]
    # .insv forces equirect semantics and is now escalation-eligible (glomap rung)
    assert ctx is not None and ctx["is_equirect"] is True


def test_plan_insv_single_stream_keeps_legacy_command(monkeypatch):
    stages, commands, _ = _plan(_req(), "/in/cap.insv", monkeypatch, probe=SINGLE_PROBE)
    assert "-vf" in commands["stitch"]
    assert "-filter_complex" not in commands["stitch"]


@pytest.mark.parametrize(
    "bad",
    [
        # POSITIVE detections of an unsupported layout — these must still 400.
        {"streams": 3, "width": 3840, "height": 3840, "dims": [(3840, 3840)] * 3},
        {"streams": 2, "width": 3840, "height": 3840, "dims": [(3840, 3840), (1920, 1080)]},
    ],
)
def test_plan_insv_weird_layout_rejected_before_job_starts(bad, monkeypatch):
    with pytest.raises(HTTPException) as exc:
        _plan(_req(), "/in/cap.insv", monkeypatch, probe=bad)
    assert exc.value.status_code == 400
    assert "Insta360 Studio" in exc.value.detail


def test_plan_insv_probe_error_falls_back_to_legacy_single_stream(monkeypatch):
    """A failure of the probe ITSELF (timeout/unreadable -> streams:0) proves
    nothing about the file: planning must warn and fall back to the legacy
    single-stream stitch (pre-probe behavior), never 400 a single-stream-capable
    capture. The post-stitch sanity gate still guards a mis-composed result."""
    unprobeable = {"streams": 0, "width": None, "height": None, "dims": []}
    stages, commands, ctx = _plan(_req(), "/in/cap.insv", monkeypatch, probe=unprobeable)
    assert stages[0] == "stitch"
    assert "-vf" in commands["stitch"]
    assert "-filter_complex" not in commands["stitch"]
    assert ctx is not None and ctx["is_equirect"] is True


def test_plan_insv_missing_ffprobe_falls_back_to_legacy_single_stream(monkeypatch):
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: None)
    monkeypatch.setattr(
        splat_route, "_tool_path",
        lambda binary, env: None if binary == "ffprobe" else f"/bin/{binary}",
    )
    stages, commands, _ = splat_route._plan_3d_job(
        _req(), AVAIL, Path("/jobs/splat_snapshot"), Path("/in/cap.insv")
    )
    assert stages[0] == "stitch"
    assert "-vf" in commands["stitch"]
    assert "-filter_complex" not in commands["stitch"]


# ---------------------------------------------------------------------------
# Default (non-360) planner output must be byte-identical to the pre-change plan
# ---------------------------------------------------------------------------


def test_standard_video_plan_matches_pre_change_snapshot(monkeypatch):
    stages, commands, ctx = _plan(_req(input_path="clip.mp4"), "/in/clip.mp4", monkeypatch)
    got = {"stages": stages, "commands": commands, "sfm_context": ctx}
    assert json.loads(json.dumps(got)) == GOLDEN_STANDARD_VIDEO_PLAN


# ---------------------------------------------------------------------------
# R0: post-stitch sanity gate — pure threshold logic on synthetic images
# ---------------------------------------------------------------------------

W, H = splat_route.STITCH_SANITY_ANALYSIS_SIZE


def _rows(arr: np.ndarray) -> list[list[int]]:
    assert arr.shape == (H, W)
    return arr.astype(int).tolist()


def _textured(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(30, 220, size=(H, W)).astype(np.int64)


def test_sanity_detects_black_wedge_in_upper_half():
    rng = np.random.default_rng(7)
    img = _textured(rng)
    img[: H // 2, int(W * 0.55):] = 2  # 45%-wide dead wedge, upper half only
    verdict = splat_route._equirect_frame_corruption(_rows(img))
    assert verdict is not None and verdict[0] == "wedge"


def test_sanity_detects_hard_vertical_discontinuity():
    img = np.full((H, W), 180, dtype=np.int64)
    img[:, W // 2:] = 20  # full-height hard cut
    verdict = splat_route._equirect_frame_corruption(_rows(img))
    assert verdict is not None
    kind, col = verdict
    assert kind == "seam"
    assert abs(col - W // 2) <= 2


def test_sanity_passes_healthy_textured_pano():
    rng = np.random.default_rng(11)
    assert splat_route._equirect_frame_corruption(_rows(_textured(rng))) is None


def test_sanity_passes_dark_night_pano():
    # Uniformly near-black sky: the wedge rule needs non-dark context, so a
    # legitimately dark capture must never false-fail.
    rng = np.random.default_rng(13)
    img = rng.integers(0, 5, size=(H, W)).astype(np.int64)
    assert splat_route._equirect_frame_corruption(_rows(img)) is None


def test_sanity_passes_smooth_gradient():
    img = np.tile(np.linspace(0, 255, W).astype(np.int64), (H, 1))
    assert splat_route._equirect_frame_corruption(_rows(img)) is None


def test_sanity_gate_skippable_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPLAT_STITCH_SANITY", "0")
    job = SimpleNamespace(log_lines=[])
    assert splat_route._stitch_sanity_check(job, tmp_path / "eq.mp4") is None
    assert any("SPLAT_STITCH_SANITY=0" in line for line in job.log_lines)


def test_sanity_gate_fails_non_2to1_aspect(monkeypatch, tmp_path):
    monkeypatch.delenv("SPLAT_STITCH_SANITY", raising=False)
    monkeypatch.setattr(splat_route, "_tool_path", lambda binary, env: f"/bin/{binary}")
    monkeypatch.setattr(
        splat_route, "_probe_video_streams",
        lambda ffprobe, src: {"streams": 1, "width": 4000, "height": 3000, "dims": [(4000, 3000)]},
    )
    job = SimpleNamespace(log_lines=[])
    err = splat_route._stitch_sanity_check(job, tmp_path / "eq.mp4")
    assert err is not None and "2:1" in err and "stitching problem" in err


def _run_gate(monkeypatch, tmp_path, frames: list[list[list[int]]]):
    """Drive _stitch_sanity_check with a healthy 2:1 probe and the given two
    analysis frames; ffmpeg/ffprobe are faked. Returns (error_or_None, job)."""
    monkeypatch.delenv("SPLAT_STITCH_SANITY", raising=False)
    monkeypatch.setattr(splat_route, "_tool_path", lambda binary, env: f"/bin/{binary}")
    monkeypatch.setattr(
        splat_route, "_probe_video_streams",
        lambda ffprobe, src: {"streams": 1, "width": 5760, "height": 2880, "dims": [(5760, 2880)]},
    )

    def fake_run(cmd, capture_output=True, text=True, timeout=0):
        if "-frames:v" in cmd:
            Path(cmd[-1]).write_bytes(b"png")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="20.0", stderr="")

    monkeypatch.setattr(splat_route.subprocess, "run", fake_run)
    it = iter(frames)
    monkeypatch.setattr(splat_route, "_load_analysis_rows", lambda p: next(it))
    job = SimpleNamespace(log_lines=[])
    return splat_route._stitch_sanity_check(job, tmp_path / "eq.mp4"), job


def test_sanity_gate_requires_both_frames_corrupt(monkeypatch, tmp_path):
    """Corruption is structural (every frame) — one flagged frame alone must not fail."""
    rng = np.random.default_rng(3)
    corrupt = _textured(rng)
    corrupt[: H // 2, int(W * 0.55):] = 2
    healthy = _textured(rng)
    err, _ = _run_gate(monkeypatch, tmp_path, [_rows(corrupt), _rows(healthy)])
    assert err is None


def test_sanity_gate_fails_when_both_frames_show_wedge(monkeypatch, tmp_path):
    """Static + wedge => FAIL. Both frames are IDENTICAL (a static capture) —
    the near-black wedge is the proven corruption mode and stays fatal on its
    own, with no motion corroboration required."""
    rng = np.random.default_rng(5)
    corrupt = _textured(rng)
    corrupt[: H // 2, int(W * 0.55):] = 2
    err, _ = _run_gate(monkeypatch, tmp_path, [_rows(corrupt), _rows(corrupt)])
    assert err is not None
    assert "coherence check" in err
    assert "not a capture-technique problem" in err
    assert "Insta360 Studio" in err


# Non-lens seam column for the cross-frame tests: far from both v360
# lens-boundary positions (0.25*W=128 and 0.75*W=384 at the 512-wide analysis size).
NON_LENS_COL = 300


def test_sanity_gate_static_frames_seam_only_passes(monkeypatch, tmp_path):
    """Static capture false-positive fix: when the two analysis frames are
    near-identical (content never moved), the same-column cross-check has no
    power — a seam finding alone must NOT fail the gate (proved false-fail on a
    static test pattern AND a healthy static dual-stream hstack+v360 stitch)."""
    img = np.full((H, W), 180, dtype=np.int64)
    img[:, NON_LENS_COL:] = 20  # hard full-height cut at a non-lens column
    # Sanity: the per-frame detector DOES see this as a seam...
    verdict = splat_route._equirect_frame_corruption(_rows(img))
    assert verdict is not None and verdict[0] == "seam"
    # ...but identical frames (mean |Δ| = 0 -> static) must pass the gate.
    err, job = _run_gate(monkeypatch, tmp_path, [_rows(img), _rows(img)])
    assert err is None
    assert any("static capture" in line for line in job.log_lines)


def test_sanity_gate_moving_frames_same_column_seam_fails(monkeypatch, tmp_path):
    """Moving capture + a strong non-lens-column discontinuity at the SAME
    column in both frames = position-fixed projection fault -> still fatal."""
    a = _textured(np.random.default_rng(21))
    b = _textured(np.random.default_rng(22))  # different texture = moving content
    a[:, NON_LENS_COL:] = 10
    b[:, NON_LENS_COL:] = 10
    err, _ = _run_gate(monkeypatch, tmp_path, [_rows(a), _rows(b)])
    assert err is not None
    assert "coherence check" in err
    assert "same column" in err


@pytest.mark.parametrize("lens_col", [int(W * 0.25), int(W * 0.75)])
def test_sanity_gate_lens_boundary_seam_passes(monkeypatch, tmp_path, lens_col):
    """A discontinuity at ~0.25/0.75 of the width is the EXPECTED v360
    dfisheye->e lens seam (hstack'd dual fisheye) — never corruption, even on a
    moving capture where the cross-frame check has full power."""
    a = _textured(np.random.default_rng(31))
    b = _textured(np.random.default_rng(32))  # moving content
    a[:, lens_col:] = 20
    b[:, lens_col:] = 20
    err, job = _run_gate(monkeypatch, tmp_path, [_rows(a), _rows(b)])
    assert err is None
    assert any("lens" in line for line in job.log_lines)


def test_frames_mean_abs_delta():
    a = [[10, 20], [30, 40]]
    assert splat_route._frames_mean_abs_delta(a, a) == 0.0
    b = [[12, 20], [30, 44]]
    assert splat_route._frames_mean_abs_delta(a, b) == pytest.approx((2 + 0 + 0 + 4) / 4)
    assert splat_route._frames_mean_abs_delta([], []) == 0.0


def test_is_lens_boundary_column():
    tol = max(2, int(splat_route.STITCH_SANITY_LENS_COL_TOL_FRAC * W))
    for frac in splat_route.STITCH_SANITY_LENS_COL_FRACS:
        center = int(frac * W)
        assert splat_route._is_lens_boundary_column(center, W)
        assert splat_route._is_lens_boundary_column(center - tol, W)
        assert splat_route._is_lens_boundary_column(center + tol, W)
        assert not splat_route._is_lens_boundary_column(center + tol + 2, W)
    assert not splat_route._is_lens_boundary_column(W // 2, W)   # pano center
    assert not splat_route._is_lens_boundary_column(0, W)        # frame edge
    assert not splat_route._is_lens_boundary_column(5, 0)        # degenerate width


# ---------------------------------------------------------------------------
# R1: matcher choice for the equirect fan-out
# ---------------------------------------------------------------------------


def test_equirect_video_8_crops_uses_sequential(monkeypatch):
    req = _req(input_path="pano.mp4", capture_format="equirectangular360",
               images_per_equirect=8, num_frames_target=75)
    _, commands, ctx = _plan(req, "/in/pano.mp4", monkeypatch)
    process = commands["process"]
    assert ["--matching-method", "sequential"] == process[process.index("--matching-method"):process.index("--matching-method") + 2]
    assert ["--camera-type", "equirectangular"] == process[process.index("--camera-type"):process.index("--camera-type") + 2]
    assert ctx is not None and ctx["is_equirect"] is True


def test_equirect_video_14_crops_keeps_vocab_tree_default(monkeypatch):
    # 14 crops/frame puts same-view temporal neighbors 14 apart — beyond
    # COLMAP's default sequential overlap of 10 (nerfstudio exposes no overlap
    # flag), so sequential would miss every temporal pair: keep vocab_tree.
    req = _req(input_path="pano.mp4", capture_format="equirectangular360",
               images_per_equirect=14, num_frames_target=75)
    _, commands, _ = _plan(req, "/in/pano.mp4", monkeypatch)
    assert "--matching-method" not in commands["process"]
    assert "--images-per-equirect" in commands["process"]
    idx = commands["process"].index("--images-per-equirect")
    assert commands["process"][idx + 1] == "14"


# ---------------------------------------------------------------------------
# R2: equirect glomap stage + escalation eligibility matrix
# ---------------------------------------------------------------------------


def test_glomap_equirect_stage_fans_out_and_widens_overlap():
    req = _req(input_path="pano.mp4", capture_format="equirectangular360",
               images_per_equirect=8, crop_bottom=0.15, num_frames_target=75)
    cmds = splat_route._sfm_stage_commands(
        solver="glomap", req=req, availability=AVAIL,
        job_dir=Path("/jobs/j1"), processed_dir=Path("/jobs/j1/processed"),
        process_input=Path("/jobs/j1/stitched/equirect.mp4"),
        subcommand="video", is_equirect=True,
    )
    script = cmds["glomap_sfm"][2]
    assert "generate_planar_projections_from_equirectangular" in script
    assert "crop_factor=(0.0, 0.15, 0.0, 0.0)" in script
    assert "--SequentialMatching.overlap 16" in script  # 2x images_per_equirect
    # crops are perspective: the process stage consumes the COLMAP model plainly
    assert "--skip-colmap" in cmds["process"]
    assert "--camera-type" not in cmds["process"]


def test_glomap_non_equirect_stage_has_no_fanout_or_overlap():
    req = _req(input_path="clip.mp4")
    cmds = splat_route._sfm_stage_commands(
        solver="glomap", req=req, availability=AVAIL,
        job_dir=Path("/jobs/j1"), processed_dir=Path("/jobs/j1/processed"),
        process_input=Path("/in/clip.mp4"), subcommand="video", is_equirect=False,
    )
    script = cmds["glomap_sfm"][2]
    assert "generate_planar_projections_from_equirectangular" not in script
    assert "--SequentialMatching.overlap" not in script


ALL_AVAILABLE = {**AVAIL, "mast3r_available": True}


@pytest.mark.parametrize(
    "tried,is_equirect,expected",
    [
        ({"colmap"}, True, "glomap"),            # equirect climbs to glomap
        ({"colmap", "glomap"}, True, None),      # ...but NEVER to mast3r
        ({"colmap", "glomap"}, False, "mast3r"),  # non-equirect chain unchanged
        ({"colmap"}, False, "glomap"),
        ({"colmap", "glomap", "mast3r"}, False, None),
    ],
)
def test_escalation_matrix(tried, is_equirect, expected):
    assert splat_route._next_sfm_solver(set(tried), ALL_AVAILABLE, is_equirect=is_equirect) == expected


def test_maybe_escalate_reroutes_equirect_to_glomap_only(monkeypatch, tmp_path):
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", tmp_path)
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: ALL_AVAILABLE)
    req = _req(input_path="pano.mp4", capture_format="equirectangular360")
    job = splat_route.SplatJob(
        job_id="splat_abc123", output_dir=str(tmp_path / "splat_abc123"),
        input_path="/in/pano.mp4",
        stages_planned=["process", "train", "export"],
        stage_commands={"process": ["x"], "train": ["y"]},
        sfm_tried={"colmap"},
        sfm_context={
            "start_solver": "colmap",
            "processed_dir": str(tmp_path / "splat_abc123" / "processed"),
            "process_input": "/in/pano.mp4",
            "subcommand": "video",
            "is_equirect": True,
        },
        sfm_req=req,
    )
    # First trip: reroutes to glomap, injecting its stages after `process`.
    assert splat_route._maybe_escalate_sfm(job, 0, 2, 624, "0.3%") is True
    assert job.sfm_tried == {"colmap", "glomap"}
    assert job.stages_planned == ["process", "glomap_sfm", "reprocess1", "train", "export"]
    assert "generate_planar_projections_from_equirectangular" in job.stage_commands["glomap_sfm"][2]
    # Second trip: mast3r is available but equirect-ineligible -> chain exhausted.
    assert splat_route._maybe_escalate_sfm(job, 2, 2, 624, "0.3%") is False


def test_maybe_escalate_non_equirect_still_reaches_mast3r(monkeypatch, tmp_path):
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", tmp_path)
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: ALL_AVAILABLE)
    monkeypatch.setattr(
        splat_route, "_mast3r_availability",
        lambda: {"mast3r_available": True, "mast3r_python": "/bin/py",
                 "mast3r_runner": "/bin/run", "mast3r_converter": "/bin/conv",
                 "mast3r_checkpoint": "/ckpt"},
    )
    req = _req(input_path="clip.mp4")
    job = splat_route.SplatJob(
        job_id="splat_def456", output_dir=str(tmp_path / "splat_def456"),
        input_path="/in/clip.mp4",
        stages_planned=["process", "train", "export"],
        stage_commands={"process": ["x"], "train": ["y"]},
        sfm_tried={"colmap", "glomap"},
        sfm_context={
            "start_solver": "colmap",
            "processed_dir": str(tmp_path / "splat_def456" / "processed"),
            "process_input": "/in/clip.mp4",
            "subcommand": "video",
            "is_equirect": False,
        },
        sfm_req=req,
    )
    availability_with_paths = {**ALL_AVAILABLE,
                               "mast3r_python": "/bin/py", "mast3r_runner": "/bin/run",
                               "mast3r_converter": "/bin/conv", "mast3r_checkpoint": "/ckpt"}
    monkeypatch.setattr(splat_route, "_engine_availability", lambda: availability_with_paths)
    assert splat_route._maybe_escalate_sfm(job, 0, 2, 311, "0.6%") is True
    assert "mast3r" in job.sfm_tried
    assert job.stages_planned == ["process", "mast3r_sfm", "reprocess1", "train", "export"]


# ---------------------------------------------------------------------------
# Item 5: 360 sub-params persisted in job meta for Re-run/Retry
# ---------------------------------------------------------------------------


def test_meta_persists_360_sub_params():
    req = _req(input_path="cap.insv", capture_format="equirectangular360",
               images_per_equirect=14, crop_bottom=0.22, insv_fov=208.0)
    meta = splat_route._new_meta("splat_aaa111", req, Path("/in/cap.insv"),
                                 Path("/jobs/splat_aaa111"), ["stitch", "process"])
    assert meta["images_per_equirect"] == 14
    assert meta["crop_bottom"] == 0.22
    assert meta["insv_fov"] == 208.0
