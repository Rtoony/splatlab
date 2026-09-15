import json
import math
from pathlib import Path

import numpy as np
import pytest

import scale_estimate as se


def _cam(cx=320.0, cy=240.0, fx=500.0, fy=500.0, w=640, h=480):
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h)


def _look_at_origin_from(pos):
    """OpenGL camera at `pos` looking at the origin (camera -z toward origin)."""
    pos = np.asarray(pos, dtype=np.float64)
    fwd = -pos / np.linalg.norm(pos)              # direction camera looks
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -fwd, pos
    return c2w


def test_projection_convention_matches_opengl_camera_and_image_v_down():
    cam = _cam(fx=200.0, fy=200.0)
    c2w = np.eye(4)  # camera at origin looking down -z (OpenGL)
    pts = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]])
    proj = se.project_points(c2w, **cam, points=pts)
    assert proj["index"].tolist() == [0, 1, 2]            # the point behind the camera is dropped
    assert proj["u"].tolist() == [320, 320 + 200, 320]    # +x world -> +u
    assert proj["v"].tolist() == [240, 240, 240 - 200]    # +y world (up) -> smaller v (image y down)
    assert np.allclose(proj["depth"], [1.0, 1.0, 1.0])


def test_frame_scale_recovers_true_factor_with_noise_and_outliers():
    rng = np.random.default_rng(7)
    cam = _cam()
    c2w = _look_at_origin_from([0.0, 0.5, 4.0])
    pts = rng.uniform(-1.0, 1.0, size=(600, 3))
    proj = se.project_points(c2w, **cam, points=pts)
    assert len(proj["depth"]) > 300
    true_scale = 0.9498
    depth_map = np.full((cam["h"], cam["w"]), np.nan)
    noise = rng.normal(1.0, 0.04, size=len(proj["depth"]))
    depth_map[proj["v"], proj["u"]] = proj["depth"] * true_scale * noise
    outliers = rng.choice(len(proj["depth"]), size=len(proj["depth"]) // 10, replace=False)
    depth_map[proj["v"][outliers], proj["u"][outliers]] *= rng.uniform(2.0, 5.0, size=len(outliers))
    result = se.frame_scale(depth_map, None, proj)
    assert result["n_inliers"] >= se.MIN_POINTS_PER_FRAME
    assert abs(result["ratio"] / true_scale - 1.0) < 0.02


def test_frame_scale_refuses_thin_evidence():
    cam = _cam()
    proj = {"u": np.array([1, 2]), "v": np.array([1, 2]), "depth": np.array([1.0, 2.0]), "index": np.array([0, 1])}
    depth_map = np.ones((cam["h"], cam["w"]))
    assert se.frame_scale(depth_map, None, proj)["ratio"] is None
    assert se.frame_scale(depth_map, None, {"u": np.array([], int), "v": np.array([], int),
                                            "depth": np.array([]), "index": np.array([], int)})["n_points"] == 0


def test_valid_mask_excludes_pixels():
    rng = np.random.default_rng(3)
    cam = _cam()
    c2w = _look_at_origin_from([0.0, 0.0, 3.0])
    pts = rng.uniform(-0.5, 0.5, size=(400, 3))
    proj = se.project_points(c2w, **cam, points=pts)
    depth_map = np.full((cam["h"], cam["w"]), np.nan)
    depth_map[proj["v"], proj["u"]] = proj["depth"] * 2.0
    mask = np.zeros((cam["h"], cam["w"]), dtype=bool)
    assert se.frame_scale(depth_map, mask, proj)["ratio"] is None
    mask[:] = True
    assert abs(se.frame_scale(depth_map, mask, proj)["ratio"] - 2.0) < 1e-6


def test_aggregate_weighted_median_and_confidence_labels():
    frames = [{"ratio": 1.0 + d, "n_inliers": 100} for d in (-0.01, 0.0, 0.01, 0.02, -0.02, 0.0)]
    agg = se.aggregate(frames)
    assert abs(agg["meters_per_unit"] - 1.0) <= 0.02 and agg["confidence"] == "high" and agg["frames_used"] == 6
    spread = [{"ratio": r, "n_inliers": 50} for r in (0.7, 1.0, 1.4)]
    assert se.aggregate(spread)["confidence"] == "low"
    assert se.aggregate([{"ratio": None, "n_inliers": 0}])["meters_per_unit"] is None
    heavy = [{"ratio": 0.5, "n_inliers": 1000}, {"ratio": 3.0, "n_inliers": 10}, {"ratio": 3.1, "n_inliers": 10}]
    assert se.aggregate(heavy)["meters_per_unit"] == 0.5


def test_select_keyframes_evenly_spaced_and_bounded():
    assert se.select_keyframes(0, 5) == []
    assert se.select_keyframes(3, 12) == [0, 1, 2]
    picks = se.select_keyframes(292, 12)
    assert picks[0] == 0 and picks[-1] == 291 and len(picks) == 12


def test_build_proposal_is_flagged_and_compares_to_existing_without_applying():
    agg = se.aggregate([{"ratio": 0.93, "n_inliers": 80, "frame": 0, "file_path": "a.jpg"}] * 6)
    rec = se.build_proposal(agg, [{"ratio": 0.93, "n_inliers": 80, "n_points": 90, "frame": 0,
                                   "file_path": "a.jpg", "mad_relative": 0.01}],
                            model="moge-2-vitl-normal", source="tools/moge2-scale.py",
                            job_id="splat_aea04ab3",
                            existing={"meters_per_unit": 0.9497500155762266, "method": "dimension"})
    assert rec["proposed"] is True and rec["applied"] is False and rec["method"] == "moge2-depth"
    assert rec["existing"]["within_5_percent"] is True
    assert abs(rec["existing"]["relative_error"] - (0.93 / 0.9497500155762266 - 1.0)) < 1e-6
    assert rec["references"][0]["meters_per_unit"] == 0.93
    json.dumps(rec)  # serialisable


def test_load_transforms_and_ply_roundtrip(tmp_path: Path):
    frames = [{"file_path": f"images/{i:03d}.jpg", "transform_matrix": np.eye(4).tolist()} for i in (2, 0, 1)]
    (tmp_path / "transforms.json").write_text(json.dumps({"w": 64, "h": 48, "fl_x": 50.0, "fl_y": 50.0,
                                                          "cx": 32.0, "cy": 24.0, "camera_model": "OPENCV",
                                                          "frames": frames}))
    (tmp_path / "sparse_pc.ply").write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
                                           "property float y\nproperty float z\nproperty uchar red\n"
                                           "property uchar green\nproperty uchar blue\nend_header\n"
                                           "1 2 3 255 0 0\n4 5 6 0 255 0\n")
    t = se.load_transforms(tmp_path)
    assert [f["file_path"] for f in t["frames"]] == ["images/000.jpg", "images/001.jpg", "images/002.jpg"]
    xyz = se.load_ply_xyz(tmp_path / "sparse_pc.ply")
    assert xyz.shape == (2, 3) and xyz[1].tolist() == [4.0, 5.0, 6.0]
    binary = tmp_path / "bin.ply"
    hdr = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    binary.write_bytes(hdr + np.array([7.0, 8.0, 9.0], dtype="<f4").tobytes())
    assert se.load_ply_xyz(binary).tolist() == [[7.0, 8.0, 9.0]]
    with pytest.raises(se.ScaleEstimateError):
        se.load_transforms(tmp_path / "nope")


def test_frame_agreement_flags_mismatched_frames():
    rng = np.random.default_rng(1)
    cam = _cam()
    pts = rng.uniform(-1, 1, size=(500, 3))
    good = {**cam, "frames": [{"c2w": _look_at_origin_from([0, 0.2, 4.0]), "file_path": "a"}]}
    assert se.frame_agreement(good, pts)["consistent"] is True
    far = {**cam, "frames": [{"c2w": _look_at_origin_from([0, 0.0, 4.0]) @ np.diag([1, 1, 1, 1]), "file_path": "a"}]}
    far["frames"][0]["c2w"][:3, 3] = [500.0, 500.0, 500.0]  # camera nowhere near the points, looking away
    assert se.frame_agreement(far, pts)["consistent"] is False


def test_fov_helper():
    assert abs(se.fov_x_degrees(1611.35, 1559) - 51.65) < 0.2


def test_load_transforms_accepts_per_frame_intrinsics(tmp_path: Path):
    frames = [{"file_path": f"images/cam{i}.png", "transform_matrix": np.eye(4).tolist(),
               "w": 512, "h": 512, "fl_x": 256.0 + i, "fl_y": 256.0, "cx": 256.0, "cy": 256.0} for i in range(2)]
    (tmp_path / "transforms.json").write_text(json.dumps({"camera_model": "SIMPLE_PINHOLE", "frames": frames}))
    t = se.load_transforms(tmp_path)
    assert t["per_frame_intrinsics"] is True and t["frames"][1]["fx"] == 257.0 and t["w"] == 512
    proj = se.project_frame(t["frames"][1], np.array([[0.0, 0.0, -2.0]]))
    assert proj["u"].tolist() == [256] and proj["v"].tolist() == [256]


def test_build_proposal_converts_into_the_viewer_frame_when_dataparser_scale_is_known():
    agg = se.aggregate([{"ratio": 0.25, "n_inliers": 100}] * 6)
    rec = se.build_proposal(agg, [], model="m", source="s", job_id="j", dataparser_scale=0.25,
                            existing={"meters_per_unit": 1.0, "method": "dimension"})
    assert rec["frame"] == "viewer-normalized" and rec["meters_per_unit_colmap_frame"] == 0.25
    assert abs(rec["meters_per_unit"] - 1.0) < 1e-9 and rec["existing"]["within_5_percent"] is True
    raw = se.build_proposal(agg, [], model="m", source="s", job_id="j")
    assert raw["frame"] == "colmap" and raw["meters_per_unit"] == 0.25


def test_find_dataparser_scale_reads_the_checkpoint_file(tmp_path: Path):
    assert se.find_dataparser_scale(tmp_path) is None
    d = tmp_path / "processed" / "splatfacto" / "2026-07-01_060756"; d.mkdir(parents=True)
    (d / "dataparser_transforms.json").write_text(json.dumps({"transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "scale": 0.2211}))
    got = se.find_dataparser_scale(tmp_path)
    assert got["scale"] == 0.2211 and got["path"].endswith("dataparser_transforms.json")


def test_windowed_minimum_sampling_prefers_the_foreground_at_edges():
    depth = np.full((10, 10), 5.0)          # background
    depth[:, :5] = 1.0                       # foreground object on the left; edge between columns 4 and 5
    u = np.array([5, 5, 0]); v = np.array([3, 8, 0])   # first two land one pixel INTO the background
    nearest = se.sample_depth(depth, u, v, window=1)
    local_min = se.sample_depth(depth, u, v, window=3)
    assert nearest.tolist() == [5.0, 5.0, 1.0] and local_min.tolist() == [1.0, 1.0, 1.0]
    depth[3, 5] = np.nan                     # a hole at the sample pixel is skipped, neighbours still answer
    assert se.sample_depth(depth, np.array([5]), np.array([3]), window=3).tolist() == [1.0]
    assert np.isnan(se.sample_depth(np.full((10, 10), np.nan), np.array([5]), np.array([3]), window=3))[0]
    proj = {"u": u, "v": v, "depth": np.array([1.0, 1.0, 1.0]), "index": np.arange(3)}
    assert se.frame_scale(np.full((10, 10), 2.0), None, proj, window=3)["n_points"] == 3
