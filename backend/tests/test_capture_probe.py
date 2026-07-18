"""Capture Coach Phase 1 probe: verdicts from synthetic SfM fixtures, and the
report-only meta merge (probe and fog must never clobber each other).
CPU-only, pure stdlib fixtures."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import splat_route  # noqa: E402
from health.probe import probe_capture  # noqa: E402


def _write_ply(path: Path, points: list[tuple[float, float, float]]) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode())
        for x, y, z in points:
            fh.write(struct.pack("<fff", x, y, z))


def _orbit_frames(radius: float = 5.0, n: int = 12) -> list[dict]:
    frames = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        cx, cy = radius * math.cos(angle), radius * math.sin(angle)
        if radius > 0:
            forward = (-cx / radius, -cy / radius, 0.0)  # look at the origin
        else:
            forward = (1.0, 0.0, 0.0)  # standing still: fixed gaze
        # Probe reads only col3 (center) and col2 (forward = -col2).
        matrix = [
            [1.0, 0.0, -forward[0], cx],
            [0.0, 1.0, -forward[1], cy],
            [0.0, 0.0, -forward[2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        frames.append({"file_path": f"images/frame_{i:05d}.jpg", "transform_matrix": matrix})
    return frames


def _box_points(half: float, n: int = 20000) -> list[tuple[float, float, float]]:
    pts = []
    side = max(2, round(n ** (1 / 3)))
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append((
                    -half + 2 * half * i / (side - 1),
                    -half + 2 * half * j / (side - 1),
                    -half + 2 * half * k / (side - 1),
                ))
    return pts


def _mk_processed(tmp_path: Path, frames: list[dict], points) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "transforms.json").write_text(json.dumps({"frames": frames}))
    _write_ply(processed / "sparse_pc.ply", points)
    return processed


def test_healthy_orbit_is_good(tmp_path):
    processed = _mk_processed(tmp_path, _orbit_frames(), _box_points(4.0))
    result = probe_capture(processed, registered=29, extracted=29)
    assert result["verdict"] == "GOOD"
    assert result["metrics"]["capture_shape"] == "orbit"
    assert result["metrics"]["traj_cloud_ratio"] < 2.5
    assert result["enforced"] is False
    assert "fog" in result["caveat"]


def test_trajectory_explosion_is_poor(tmp_path):
    # Camera path bbox ~8x the point cloud = the 07-11 scattered-pose fingerprint.
    processed = _mk_processed(tmp_path, _orbit_frames(), _box_points(0.5))
    result = probe_capture(processed)
    assert result["verdict"] == "POOR"
    assert any("trajectory-explosion" in f for f in result["findings"])


def test_thin_map_is_poor(tmp_path):
    processed = _mk_processed(tmp_path, _orbit_frames(), _box_points(4.0, n=27)[:500])
    result = probe_capture(processed)
    assert result["verdict"] == "POOR"
    assert any("nearly empty" in f for f in result["findings"])


def test_standing_still_is_poor(tmp_path):
    frames = _orbit_frames(radius=0.0)
    processed = _mk_processed(tmp_path, frames, _box_points(4.0))
    result = probe_capture(processed)
    assert result["verdict"] == "POOR"
    assert any("standing still" in f for f in result["findings"])


def test_low_registration_is_marginal(tmp_path):
    processed = _mk_processed(tmp_path, _orbit_frames(), _box_points(4.0))
    result = probe_capture(processed, registered=15, extracted=29)
    assert result["verdict"] == "MARGINAL"
    assert result["metrics"]["registration_ratio"] == 0.517


def test_missing_transforms_is_poor(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    result = probe_capture(processed)
    assert result["verdict"] == "POOR"
    assert "no usable transforms.json" in result["findings"][0]


def test_probe_and_fog_merge_without_clobbering(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    job_id = "splat_probe01"
    job_dir = outputs / job_id
    job_dir.mkdir(parents=True)
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    req = splat_route.SplatTrainRequest(mode="3d", input_path="/in/x", output_dir="outputs/3d")
    (job_dir / "meta.json").write_text(json.dumps(
        splat_route._new_meta(job_id, req, Path("/in/x"), job_dir, ["process"])))

    processed = _mk_processed(job_dir, _orbit_frames(), _box_points(4.0))
    splat_route._patch_probe_health(job_id, processed, 29, 29)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["health"]["probe"]["verdict"] == "GOOD"

    # Fog-stage-style merge on top must keep the probe.
    health_meta = (splat_route._read_meta(job_id) or {}).get("health") or {}
    health_meta.update({"v": 1, "fog": {"verdict": "HEALTHY", "enforced": False}})
    splat_route._patch_meta(job_id, health=health_meta)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["health"]["fog"]["verdict"] == "HEALTHY"
    assert meta["health"]["probe"]["verdict"] == "GOOD"

    # And a second probe patch must keep the fog verdict.
    splat_route._patch_probe_health(job_id, processed, 29, 29)
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["health"]["fog"]["verdict"] == "HEALTHY"
    assert meta["health"]["probe"]["verdict"] == "GOOD"
