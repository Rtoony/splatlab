"""Viewer-vs-trained agreement (CPU half).

Question (2026-09-15): does a splat trained with gsplat's `antialiased` rasterize
mode look the same in SparkJS — the walker's renderer, which applies no opacity
compensation — as it does in nerfstudio? We answer with numbers, per camera:

  photo    : the real held-out photograph (never trained on)
  ns       : nerfstudio/gsplat render in the mode the arm was trained with
  spark    : the same camera rendered by Spark from the exported PLY

Per arm: PSNR(spark, photo) is what the user sees vs reality; PSNR(spark, ns) is
how faithfully the viewer reproduces what was trained. The verdict compares the
antialiased arm to the baseline arm on both. Pure numpy + Pillow; no GPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Adopt only if the viewer shows the antialiased arm at least this close to the
# baseline arm against real photos, AND the viewer reproduces the antialiased
# arm's own training render no worse than this much below the baseline's.
PHOTO_TOLERANCE_DB = 0.25
FIDELITY_TOLERANCE_DB = 1.0


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != tuple(size):
        im = im.resize(tuple(size), Image.LANCZOS)
    return np.asarray(im, dtype=np.float64) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    # identical images are "100 dB", not inf, so means and deltas stay finite
    return 100.0 if mse == 0 else min(100.0, 10.0 * math.log10(1.0 / mse))


def camera_to_three(row: dict[str, Any]) -> dict[str, Any]:
    """nerfstudio camera-to-world (OpenGL axes) -> the SplatLab viewer's camera
    contract: position = column 3, up = column 1, forward = -column 2, fov_y from
    fy and the render height. Identical to backend/splat_route.py's camera list
    and spark-scene-viewer.tsx's `lookAt(pos + forward)`."""
    c2w = np.asarray(row["c2w"], dtype=np.float64)
    if c2w.shape != (3, 4):
        raise ValueError("c2w must be 3x4")
    pos = c2w[:, 3]
    up = c2w[:, 1] / np.linalg.norm(c2w[:, 1])
    fwd = -c2w[:, 2] / np.linalg.norm(c2w[:, 2])
    fov_y = math.degrees(2.0 * math.atan(float(row["h"]) / (2.0 * float(row["fy"]))))
    return {"position": pos.tolist(), "up": up.tolist(), "forward": fwd.tolist(),
            "fov_y_degrees": fov_y, "aspect": float(row["w"]) / float(row["h"]),
            "w": int(row["w"]), "h": int(row["h"]), "file": row["file"]}


def eval_rows(cams_json: Path) -> list[dict[str, Any]]:
    cams = json.loads(Path(cams_json).read_text())
    return [r for r in cams.get("rendered", []) if r.get("variant") == "eval" and r.get("c2w") is not None]


def compare_arm(arm_dir: Path) -> dict[str, Any]:
    """Expects <arm>/_renders/{cams.json, camXXXX_eval.png} and <arm>/_spark/camXXXX_eval.png."""
    arm_dir = Path(arm_dir)
    renders = arm_dir / "_renders"
    spark = arm_dir / "_spark"
    cams = json.loads((renders / "cams.json").read_text())
    per_cam = []
    for r in eval_rows(renders / "cams.json"):
        size = (int(r["w"]), int(r["h"]))
        ns = load_rgb(renders / r["file"], size)
        sp_path = spark / r["file"]
        if not sp_path.is_file():
            per_cam.append({"cam": r["cam"], "file": r["file"], "missing_spark": True}); continue
        sp = load_rgb(sp_path, size)
        entry = {"cam": r["cam"], "file": r["file"], "psnr_spark_vs_ns": round(psnr(sp, ns), 3),
                 "mean_abs_spark_vs_ns": round(float(np.mean(np.abs(sp - ns))) * 255.0, 3)}
        if r.get("photo") and Path(r["photo"]).is_file():
            ph = load_rgb(Path(r["photo"]), size)
            entry["psnr_spark_vs_photo"] = round(psnr(sp, ph), 3)
            entry["psnr_ns_vs_photo"] = round(psnr(ns, ph), 3)
        per_cam.append(entry)
    def mean(key):
        vals = [e[key] for e in per_cam if key in e and math.isfinite(e[key])]
        return round(float(np.mean(vals)), 3) if vals else None
    return {"arm": arm_dir.name, "rasterize_mode": cams.get("rasterize_mode"),
            "n": sum(1 for e in per_cam if "psnr_spark_vs_ns" in e),
            "psnr_spark_vs_photo": mean("psnr_spark_vs_photo"), "psnr_ns_vs_photo": mean("psnr_ns_vs_photo"),
            "psnr_spark_vs_ns": mean("psnr_spark_vs_ns"), "mean_abs_spark_vs_ns": mean("mean_abs_spark_vs_ns"),
            "per_cam": per_cam}


def verdict(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Report-only decision rule with its inputs spelled out."""
    def d(key):
        a, b = candidate.get(key), baseline.get(key)
        return None if a is None or b is None else round(a - b, 3)
    photo_delta = d("psnr_spark_vs_photo")
    fidelity_delta = d("psnr_spark_vs_ns")
    ok_photo = photo_delta is not None and photo_delta >= -PHOTO_TOLERANCE_DB
    ok_fidelity = fidelity_delta is not None and fidelity_delta >= -FIDELITY_TOLERANCE_DB
    return {"candidate": candidate.get("arm"), "baseline": baseline.get("arm"),
            "spark_vs_photo_delta_db": photo_delta, "spark_vs_ns_delta_db": fidelity_delta,
            "thresholds": {"photo_tolerance_db": PHOTO_TOLERANCE_DB, "fidelity_tolerance_db": FIDELITY_TOLERANCE_DB},
            "viewer_safe": bool(ok_photo and ok_fidelity),
            "reason": ("viewer shows the candidate at least as well against photos and reproduces its training render"
                       if ok_photo and ok_fidelity else
                       "candidate looks worse in the viewer than baseline against photos" if not ok_photo else
                       "viewer does not reproduce the candidate's training render (opacity compensation mismatch)")}


def contact_sheet(arm_dirs: list[Path], out_png: Path, max_cams: int = 4, tile_w: int = 320) -> Path:
    """Rows = cameras; columns = photo | per arm: nerfstudio render, Spark render."""
    arm_dirs = [Path(a) for a in arm_dirs]
    rows = eval_rows(arm_dirs[0] / "_renders" / "cams.json")[:max_cams]
    cols = 1 + 2 * len(arm_dirs)
    tile_h = None; tiles = []
    for r in rows:
        size = (int(r["w"]), int(r["h"])); th = round(tile_w * size[1] / size[0]); tile_h = th
        line = []
        ph = Image.open(r["photo"]).convert("RGB").resize((tile_w, th)) if r.get("photo") and Path(r["photo"]).is_file() else Image.new("RGB", (tile_w, th), (40, 40, 40))
        line.append(("photo", ph))
        for a in arm_dirs:
            for sub in ("_renders", "_spark"):
                p = a / sub / r["file"]
                im = Image.open(p).convert("RGB").resize((tile_w, th)) if p.is_file() else Image.new("RGB", (tile_w, th), (40, 40, 40))
                line.append((f"{a.name} {'nerfstudio' if sub == '_renders' else 'spark'}", im))
        tiles.append((r["cam"], line))
    if not tiles:
        raise ValueError("no eval rows to draw")
    header = 18
    sheet = Image.new("RGB", (cols * tile_w, len(tiles) * (tile_h + header)), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for i, (cam, line) in enumerate(tiles):
        y = i * (tile_h + header)
        for j, (label, im) in enumerate(line):
            sheet.paste(im, (j * tile_w, y + header))
            draw.text((j * tile_w + 4, y + 2), f"cam {cam} · {label}", fill=(255, 255, 255))
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)
    return out_png
