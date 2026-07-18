"""Capture Coach Phase 1 — pre-train capture probe (REPORT-ONLY).

Scores an SfM result from artifacts that already exist the moment the A1
registration gate runs — no GPU, no new subprocess, pure stdlib:

- ``processed/transforms.json``  → camera centers + forward axes
- ``processed/sparse_pc.ply``    → sampled point-cloud bbox + count

Headline metric: **trajectory/cloud bbox ratio** — the 07-11 root-cause
fingerprint. An unrigged/scattered solve puts the camera path bbox far outside
the point cloud (measured 12x on fog scenes; same-frame centers 5.1 units
apart vs a true step of 0.13). Healthy captures keep the path inside or around
the map (ratio ~<= 2).

Verdicts are GOOD / MARGINAL / POOR with human coaching strings. Per the
metric-trust doctrine this NEVER changes gate outcomes — it lands in
``meta["health"]["probe"]`` for the UI and RToony's grading only.

Honest caveat baked into every result: fog scenes can register HIGH (99.8% on
garbage poses) — registration-adjacent metrics may not separate them; fog
detection stays the post-train fog gate's job.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any

PROBE_VERSION = 1

# Env-tunable thresholds (HEALTH_PROBE_*), same convention as the fog gate.
TRAJ_RATIO_MAX = float(os.environ.get("HEALTH_PROBE_TRAJ_RATIO_MAX", "6.0"))
TRAJ_RATIO_WARN = float(os.environ.get("HEALTH_PROBE_TRAJ_RATIO_WARN", "2.5"))
MIN_POINTS = int(os.environ.get("HEALTH_PROBE_MIN_POINTS", "1000"))
WARN_POINTS = int(os.environ.get("HEALTH_PROBE_WARN_POINTS", "10000"))
MIN_PATH_DIAG = float(os.environ.get("HEALTH_PROBE_MIN_PATH_DIAG", "1e-4"))
ORBIT_INWARD_FRAC = float(os.environ.get("HEALTH_PROBE_ORBIT_INWARD_FRAC", "0.7"))
PLY_SAMPLE_N = int(os.environ.get("HEALTH_PROBE_PLY_SAMPLE_N", "20000"))

CAVEAT = (
    "Registration can be high on garbage poses — fog detection stays the "
    "post-train fog gate's job."
)


def _bbox_diag(points: list[tuple[float, float, float]]) -> float:
    if not points:
        return 0.0
    xs, ys, zs = zip(*points)
    return math.dist((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))


def _camera_geometry(transforms_path: Path) -> dict[str, Any] | None:
    """Camera centers + forward axes from a Nerfstudio transforms.json.

    transform_matrix is camera-to-world; translation column = camera center,
    and the camera looks down -Z (OpenGL convention), so forward = -R[:,2].
    """
    try:
        frames = json.loads(transforms_path.read_text()).get("frames", [])
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    entries: list[tuple[str, list[list[float]]]] = []
    for frame in frames:
        matrix = frame.get("transform_matrix")
        if isinstance(matrix, list) and len(matrix) >= 3:
            entries.append((str(frame.get("file_path", "")), matrix))
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    centers = [(m[0][3], m[1][3], m[2][3]) for _, m in entries]
    forwards = [(-m[0][2], -m[1][2], -m[2][2]) for _, m in entries]
    steps = [math.dist(centers[i], centers[i + 1]) for i in range(len(centers) - 1)]
    return {
        "n_frames": len(centers),
        "centers": centers,
        "forwards": forwards,
        "path_bbox_diag": _bbox_diag(centers),
        "mean_step": (sum(steps) / len(steps)) if steps else 0.0,
    }


def _sample_ply_xyz(ply_path: Path, sample_n: int = PLY_SAMPLE_N) -> dict[str, Any] | None:
    """Sampled xyz bbox + vertex count from a (binary LE or ascii) point ply.

    Same seek-and-sample trick as thumb.py — cheap even on millions of points.
    """
    try:
        with ply_path.open("rb") as fh:
            header = b""
            while b"end_header\n" not in header:
                chunk = fh.read(1)
                if not chunk:
                    return None
                header += chunk
            text = header.decode("latin1")
            count_match = re.search(r"element vertex (\d+)", text)
            if not count_match:
                return None
            vcount = int(count_match.group(1))
            if vcount <= 0:
                return {"n_points": 0, "cloud_bbox_diag": 0.0}
            props = re.findall(r"property (\w+) (\w+)", text)
            fmt_map = {"float": "f", "float32": "f", "double": "d", "float64": "d",
                       "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
                       "short": "h", "ushort": "H", "int": "i", "int32": "i",
                       "uint": "I", "uint32": "I"}
            names = [name for _, name in props]
            try:
                rowfmt = "<" + "".join(fmt_map[kind] for kind, _ in props)
                xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
            except (KeyError, ValueError):
                return None
            if "binary_little_endian" not in text:
                return None  # ascii/BE ply: not produced by our pipeline; skip honestly
            base, rowsz = len(header), struct.calcsize(rowfmt)
            step = max(1, vcount // max(1, sample_n))
            pts: list[tuple[float, float, float]] = []
            for i in range(0, vcount, step):
                fh.seek(base + i * rowsz)
                raw = fh.read(rowsz)
                if len(raw) < rowsz:
                    break
                row = struct.unpack(rowfmt, raw)
                pts.append((row[xi], row[yi], row[zi]))
    except OSError:
        return None
    return {"n_points": vcount, "cloud_bbox_diag": _bbox_diag(pts)}


def probe_capture(
    processed_dir: Path,
    registered: int | None = None,
    extracted: int | None = None,
) -> dict[str, Any]:
    """Score the capture from on-disk SfM artifacts. Never raises."""
    findings: list[str] = []
    coaching: list[str] = []
    metrics: dict[str, Any] = {}
    verdict = "GOOD"

    def worse(new: str) -> None:
        nonlocal verdict
        order = {"GOOD": 0, "MARGINAL": 1, "POOR": 2}
        if order[new] > order[verdict]:
            verdict = new

    cams = _camera_geometry(processed_dir / "transforms.json")
    cloud = _sample_ply_xyz(processed_dir / "sparse_pc.ply")

    if cams is None:
        return {
            "v": PROBE_VERSION, "verdict": "POOR",
            "findings": ["no usable transforms.json — SfM produced no poses"],
            "coaching": ["Recapture: the solver could not pose any cameras."],
            "metrics": {}, "caveat": CAVEAT, "enforced": False,
        }

    metrics["n_posed"] = cams["n_frames"]
    metrics["path_bbox_diag"] = round(cams["path_bbox_diag"], 4)
    metrics["mean_step"] = round(cams["mean_step"], 6)

    if registered is not None and extracted:
        ratio = registered / extracted
        metrics["registration_ratio"] = round(ratio, 3)
        if ratio < 0.6:
            worse("MARGINAL")
            findings.append(f"only {registered}/{extracted} frames registered")
            coaching.append(
                "Coverage is partial — the scene will have holes where frames dropped. "
                "More overlap between shots keeps them."
            )

    if cams["path_bbox_diag"] < MIN_PATH_DIAG:
        worse("POOR")
        findings.append("camera path has ~zero extent (standing still)")
        coaching.append(
            "The camera never moved — parallax is what builds 3D. "
            "Move THROUGH the space, don't pan from one spot."
        )

    if cloud is not None:
        metrics["n_points"] = cloud["n_points"]
        metrics["cloud_bbox_diag"] = round(cloud["cloud_bbox_diag"], 4)
        if cloud["n_points"] < MIN_POINTS:
            worse("POOR")
            findings.append(f"sparse map is nearly empty ({cloud['n_points']} points)")
            coaching.append(
                "Poses solved but the map is almost empty — usually too little "
                "texture or too fast a sweep. Slow down and add overlap."
            )
        elif cloud["n_points"] < WARN_POINTS:
            worse("MARGINAL")
            findings.append(f"sparse map is thin ({cloud['n_points']} points)")
            coaching.append("Map is thin — slow down; give every surface several looks.")
        if cloud["cloud_bbox_diag"] > 0 and cams["path_bbox_diag"] > 0:
            ratio = cams["path_bbox_diag"] / cloud["cloud_bbox_diag"]
            metrics["traj_cloud_ratio"] = round(ratio, 2)
            if ratio > TRAJ_RATIO_MAX:
                worse("POOR")
                findings.append(
                    f"camera path bbox is {ratio:.1f}x the point cloud — "
                    "scattered poses (the trajectory-explosion fingerprint)"
                )
                coaching.append(
                    "The solved trajectory dwarfs the map — poses are likely "
                    "scattered and the render will collapse to fog. This capture "
                    "type needs the rig lane (360) or a recapture."
                )
            elif ratio > TRAJ_RATIO_WARN:
                worse("MARGINAL")
                findings.append(
                    f"camera path bbox is {ratio:.1f}x the point cloud (watch for scatter)"
                )

    # Orbit vs walkthrough characterization (informational, never a downgrade):
    # an orbit ring looks inward at its centroid; a walkthrough doesn't.
    centers, forwards = cams["centers"], cams["forwards"]
    if len(centers) >= 3:
        cx = sum(c[0] for c in centers) / len(centers)
        cy = sum(c[1] for c in centers) / len(centers)
        cz = sum(c[2] for c in centers) / len(centers)
        inward = 0
        for center, forward in zip(centers, forwards):
            to_centroid = (cx - center[0], cy - center[1], cz - center[2])
            norm = math.hypot(*to_centroid) or 1.0
            fnorm = math.hypot(*forward) or 1.0
            dot = sum(t * f for t, f in zip(to_centroid, forward)) / (norm * fnorm)
            if dot > 0.5:
                inward += 1
        inward_frac = inward / len(centers)
        metrics["inward_frac"] = round(inward_frac, 2)
        metrics["capture_shape"] = (
            "orbit" if inward_frac >= ORBIT_INWARD_FRAC else "walkthrough"
        )

    return {
        "v": PROBE_VERSION,
        "verdict": verdict,
        "findings": findings,
        "coaching": coaching,
        "metrics": metrics,
        "caveat": CAVEAT,
        "enforced": False,
    }
