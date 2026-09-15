"""Metric-scale proposal from monocular metric depth (MoGe-2) vs sparse SfM depth.

Why
---
`meters_per_unit` is unknown on most captures (7 of 12 worlds on 2026-09-14).
The only calibration path is a human measuring a dimension in the viewer. This
module turns a *monocular metric depth model* into a second, automatic source of
evidence: for a handful of keyframes, compare the model's metric depth at the
pixels where COLMAP triangulated points project, take a robust ratio, aggregate
across frames, and emit a *proposal* record in the `scale_calibration` shape.

It never writes `meta.json`: a proposal is evidence the owner accepts (the
`geo_route`/dimension flows stay the acceptance path). Pure numpy + stdlib so it
is unit-testable without torch; the GPU half (running MoGe-2) lives in
`tools/moge2-scale.py` and only calls into here.

Conventions (nerfstudio `transforms.json`): `transform_matrix` is camera-to-world
in OpenGL axes (x right, y up, z backward); image v grows downward. Sparse points
in `sparse_pc.ply` share that frame (ns-process-data applies `applied_transform`
to both), which `frame_agreement()` verifies empirically rather than assumes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from scale_calibration import uncertainty

METHOD_MOGE2 = "moge2-depth"
MIN_POINTS_PER_FRAME = 30
INLIER_BAND = 0.25          # ratio within ±25 % of the frame median counts
DEFAULT_KEYFRAMES = 12
CONFIDENCE_HIGH = 0.05      # relative MAD across frames
CONFIDENCE_MEDIUM = 0.12


class ScaleEstimateError(ValueError):
    """Bad input for a scale estimate. Always actionable."""


# ── dataset loading ───────────────────────────────────────────────────────────

def load_transforms(processed_dir: Path) -> dict[str, Any]:
    """Nerfstudio transforms.json. Intrinsics may live at the top level (single
    camera) or per frame (rig lanes write w/h/fl_x/... inside each frame); each
    returned frame carries its own resolved intrinsics either way."""
    path = Path(processed_dir) / "transforms.json"
    if not path.is_file():
        raise ScaleEstimateError(f"missing {path}")
    data = json.loads(path.read_text())
    if "frames" not in data:
        raise ScaleEstimateError("transforms.json lacks 'frames'")
    keys = ("w", "h", "fl_x", "fl_y", "cx", "cy")
    top = {k: data.get(k) for k in keys}
    frames = []
    for f in sorted(data["frames"], key=lambda f: str(f.get("file_path", ""))):
        intr = {k: f.get(k, top[k]) for k in keys}
        if any(intr[k] is None for k in keys):
            raise ScaleEstimateError(f"frame {f.get('file_path')} lacks intrinsics and transforms.json has no top-level ones")
        frames.append({"file_path": str(f["file_path"]),
                       "c2w": np.asarray(f["transform_matrix"], dtype=np.float64)[:4, :4],
                       "w": int(intr["w"]), "h": int(intr["h"]), "fx": float(intr["fl_x"]),
                       "fy": float(intr["fl_y"]), "cx": float(intr["cx"]), "cy": float(intr["cy"])})
    first = frames[0] if frames else {"w": 0, "h": 0, "fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0}
    per_frame = any(k in data["frames"][0] for k in keys) if data["frames"] else False
    return {"w": first["w"], "h": first["h"], "fx": first["fx"], "fy": first["fy"],
            "cx": first["cx"], "cy": first["cy"], "per_frame_intrinsics": per_frame,
            "camera_model": data.get("camera_model") or (data["frames"][0].get("camera_model") if data["frames"] else None),
            "frames": frames, "ply_file_path": data.get("ply_file_path", "sparse_pc.ply")}


def project_frame(frame: dict[str, Any], points: np.ndarray,
                  default: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """project_points with the frame's own intrinsics, falling back to the
    dataset-level ones when a caller built frames without them."""
    src = {k: frame.get(k, (default or {}).get(k)) for k in ("fx", "fy", "cx", "cy", "w", "h")}
    return project_points(frame["c2w"], src["fx"], src["fy"], src["cx"], src["cy"],
                          src["w"], src["h"], points)


def load_ply_xyz(path: Path) -> np.ndarray:
    """Read x,y,z from an ASCII or little-endian binary PLY (only what we need)."""
    path = Path(path)
    if not path.is_file():
        raise ScaleEstimateError(f"missing {path}")
    with path.open("rb") as fh:
        header: list[str] = []
        while True:
            line = fh.readline()
            if not line:
                raise ScaleEstimateError("PLY header never ended")
            header.append(line.decode("ascii", errors="replace").strip())
            if header[-1] == "end_header":
                break
        fmt = next((h.split()[1] for h in header if h.startswith("format ")), "ascii")
        count = next((int(h.split()[2]) for h in header if h.startswith("element vertex")), 0)
        props = [h.split() for h in header if h.startswith("property ") and len(h.split()) == 3]
        names = [p[2] for p in props]
        if not {"x", "y", "z"} <= set(names):
            raise ScaleEstimateError("PLY has no x/y/z vertex properties")
        if fmt == "ascii":
            rows = np.loadtxt(fh, max_rows=count, dtype=np.float64, ndmin=2)
            idx = [names.index(a) for a in ("x", "y", "z")]
            return rows[:, idx]
        types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                 "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                 "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
                 "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
        if fmt != "binary_little_endian":
            raise ScaleEstimateError(f"unsupported PLY format {fmt}")
        dtype = np.dtype([(p[2], types[p[1]]) for p in props])
        arr = np.frombuffer(fh.read(dtype.itemsize * count), dtype=dtype, count=count)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


# ── geometry ──────────────────────────────────────────────────────────────────

def project_points(c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                   w: int, h: int, points: np.ndarray) -> dict[str, np.ndarray]:
    """Project world points through an OpenGL-convention camera.

    Returns integer pixel coords (u, v) and positive depths for the points that
    land inside the image in front of the camera.
    """
    c2w = np.asarray(c2w, dtype=np.float64)
    R, t = c2w[:3, :3], c2w[:3, 3]
    cam = (np.asarray(points, dtype=np.float64) - t) @ R          # R^T (P - t)
    depth = -cam[:, 2]
    ok = depth > 1e-6
    u = fx * cam[:, 0] / np.where(ok, depth, 1.0) + cx
    v = -fy * cam[:, 1] / np.where(ok, depth, 1.0) + cy
    inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return {"u": np.floor(u[inside]).astype(np.int64),
            "v": np.floor(v[inside]).astype(np.int64),
            "depth": depth[inside], "index": np.nonzero(inside)[0]}


def frame_agreement(transforms: dict[str, Any], points: np.ndarray,
                    sample_frames: int = 8) -> dict[str, Any]:
    """Empirical check that points and cameras share a frame: a healthy dataset
    has a large share of sparse points in front of, and inside, the cameras."""
    frames = transforms["frames"]
    picks = select_keyframes(len(frames), min(sample_frames, len(frames)))
    fracs = []
    for i in picks:
        proj = project_frame(frames[i], points, transforms)
        fracs.append(len(proj["depth"]) / max(1, len(points)))
    med = float(np.median(fracs)) if fracs else 0.0
    return {"frames_checked": len(picks), "median_in_view_fraction": round(med, 4),
            "consistent": med >= 0.02}


def select_keyframes(n_frames: int, k: int) -> list[int]:
    if n_frames <= 0 or k <= 0:
        return []
    k = min(k, n_frames)
    return sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, k)})


# ── estimation ────────────────────────────────────────────────────────────────

def sample_depth(depth_map: np.ndarray, u: np.ndarray, v: np.ndarray, window: int = 1) -> np.ndarray:
    """Depth at (v, u). With window > 1, the MINIMUM finite depth in the
    (window x window) neighbourhood: SfM features sit on corners and edges,
    where a nearest-pixel lookup often lands on the *background* behind the
    edge and biases the ratio high. Taking the local minimum picks the
    foreground surface the feature actually belongs to."""
    depth_map = np.asarray(depth_map, dtype=np.float64)
    if window <= 1:
        return depth_map[v, u]
    r = window // 2
    h, w = depth_map.shape
    out = np.full(len(u), np.inf)
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            vv = np.clip(v + dv, 0, h - 1)
            uu = np.clip(u + du, 0, w - 1)
            d = depth_map[vv, uu]
            ok = np.isfinite(d) & (d > 0)
            out = np.where(ok & (d < out), d, out)
    out[~np.isfinite(out)] = np.nan
    return out


def frame_scale(metric_depth: np.ndarray, valid_mask: np.ndarray | None,
                proj: dict[str, np.ndarray], window: int = 1) -> dict[str, Any]:
    """Robust ratio metric_depth / sparse_depth for one frame.

    Two passes: median of all finite ratios, then median of the ratios within
    ±INLIER_BAND of that. Reports how many points survived so the caller can
    weight frames and refuse thin evidence. `window` > 1 samples the local
    minimum depth around each feature (see sample_depth).
    """
    depth_map = np.asarray(metric_depth, dtype=np.float64)
    u, v, sparse = proj["u"], proj["v"], proj["depth"]
    if len(sparse) == 0:
        return {"n_points": 0, "n_inliers": 0, "ratio": None, "mad_relative": None}
    sampled = sample_depth(depth_map, u, v, window)
    good = np.isfinite(sampled) & (sampled > 0) & (sparse > 0)
    if valid_mask is not None:
        good &= np.asarray(valid_mask, dtype=bool)[v, u]
    if int(good.sum()) < MIN_POINTS_PER_FRAME:
        return {"n_points": int(good.sum()), "n_inliers": 0, "ratio": None, "mad_relative": None}
    ratios = sampled[good] / sparse[good]
    med = float(np.median(ratios))
    inl = ratios[np.abs(ratios / med - 1.0) <= INLIER_BAND]
    if len(inl) < MIN_POINTS_PER_FRAME:
        return {"n_points": int(good.sum()), "n_inliers": int(len(inl)), "ratio": None,
                "mad_relative": None}
    med2 = float(np.median(inl))
    mad = float(np.median(np.abs(inl - med2))) / med2 if med2 > 0 else None
    return {"n_points": int(good.sum()), "n_inliers": int(len(inl)),
            "ratio": med2, "mad_relative": round(mad, 6) if mad is not None else None}


def aggregate(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Weighted median across frames + a confidence label from their spread."""
    usable = [r for r in frame_results if r.get("ratio")]
    if not usable:
        return {"meters_per_unit": None, "frames_used": 0, "confidence": "none",
                "relative_mad": None, "uncertainty": uncertainty([])}
    ratios = np.array([r["ratio"] for r in usable], dtype=np.float64)
    weights = np.array([r["n_inliers"] for r in usable], dtype=np.float64)
    order = np.argsort(ratios)
    cum = np.cumsum(weights[order])
    est = float(ratios[order][np.searchsorted(cum, cum[-1] / 2.0)])
    rel_mad = float(np.median(np.abs(ratios - est)) / est) if est > 0 else None
    if len(usable) >= 6 and rel_mad is not None and rel_mad < CONFIDENCE_HIGH:
        conf = "high"
    elif len(usable) >= 3 and rel_mad is not None and rel_mad < CONFIDENCE_MEDIUM:
        conf = "medium"
    else:
        conf = "low"
    return {"meters_per_unit": est, "frames_used": len(usable), "confidence": conf,
            "relative_mad": round(rel_mad, 6) if rel_mad is not None else None,
            "uncertainty": uncertainty([float(r) for r in ratios])}


def build_proposal(agg: dict[str, Any], frame_results: list[dict[str, Any]], *,
                   model: str, source: str, job_id: str,
                   frame_check: dict[str, Any] | None = None,
                   existing: dict[str, Any] | None = None,
                   dataparser_scale: float | None = None) -> dict[str, Any]:
    """A `scale_calibration`-shaped record flagged as a proposal, plus the
    comparison against any existing calibration (never applied here).

    MoGe depth is compared against COLMAP-frame sparse depth, so the raw ratio is
    metres per COLMAP unit. The viewer PLY is in nerfstudio's normalised frame
    (COLMAP units x dataparser_scale), which is where `meters_per_unit` is
    measured and consumed — so when the scale is known the headline number is
    converted into that frame and the COLMAP-frame value is kept alongside."""
    colmap_factor = agg.get("meters_per_unit")
    factor = colmap_factor
    if colmap_factor and dataparser_scale:
        factor = colmap_factor / dataparser_scale
    record: dict[str, Any] = {
        "schema": "dev.splatlab.scale-proposal/v1",
        "job_id": job_id,
        "proposed": True,
        "applied": False,
        "meters_per_unit": factor,
        "frame": "viewer-normalized" if (colmap_factor and dataparser_scale) else "colmap",
        "meters_per_unit_colmap_frame": colmap_factor,
        "dataparser_scale": dataparser_scale,
        "method": METHOD_MOGE2,
        "source": source,
        "model": model,
        "confidence": agg.get("confidence"),
        "relative_mad": agg.get("relative_mad"),
        "frames_used": agg.get("frames_used"),
        "references": [
            {"frame": r.get("frame"), "file_path": r.get("file_path"),
             "n_points": r.get("n_points"), "n_inliers": r.get("n_inliers"),
             "meters_per_unit": r.get("ratio"), "mad_relative": r.get("mad_relative")}
            for r in frame_results
        ],
        "uncertainty": agg.get("uncertainty"),
        "frame_agreement": frame_check,
        "set_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if isinstance(existing, dict) and existing.get("meters_per_unit") and factor:
        have = float(existing["meters_per_unit"])
        record["existing"] = {"meters_per_unit": have, "method": existing.get("method"),
                              "relative_error": round(factor / have - 1.0, 6),
                              "within_5_percent": bool(abs(factor / have - 1.0) <= 0.05)}
    return record


def find_dataparser_scale(job_dir: Path) -> dict[str, Any] | None:
    """nerfstudio normalises poses (orient + centre + scale-to-unit-box) before
    training, and `ns-export gaussian-splat` writes raw `model.means` — so the
    viewer PLY, and every calibration measured in it, lives in that normalised
    frame. dataparser_transforms.json under the checkpoint records the scale."""
    cands = sorted(Path(job_dir).glob("processed/splatfacto/*/dataparser_transforms.json"),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    data = json.loads(cands[-1].read_text())
    try:
        scale = float(data["scale"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (scale > 0):
        return None
    return {"scale": scale, "transform": data.get("transform"), "path": str(cands[-1])}


def fov_x_degrees(fx: float, w: int) -> float:
    return math.degrees(2.0 * math.atan(w / (2.0 * fx)))
