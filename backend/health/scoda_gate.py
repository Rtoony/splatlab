"""SCODA render-agreement gate — CPU orchestration (stdlib + Pillow).

SCODA (arXiv 2609.07346, MIT, cloned at ~/tools/research/scoda) scores a
rendered view against the *distribution* of a scene's clean photographs, no
aligned reference needed. That is the signal the world gates lacked: a world
that "looks wrong but passes every gate" (splat_3aaf8067) should score below one
that looks right (splat_aea04ab3). Report-only until that ranking is proven.

Pieces: pick reference photos → write SCODA's input CSV for the renders →
run SCODA in its own venv → aggregate per-view scores into <job>/_health/scoda.json.
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCODA_ROOT = Path.home() / "tools" / "research" / "scoda"
SCODA_PY = SCODA_ROOT / ".venv" / "bin" / "python"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def pick_reference_views(images_dir: Path, n: int = 40) -> list[Path]:
    files = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files or n <= 0:
        return []
    n = min(n, len(files))
    idx = sorted({round(i * (len(files) - 1) / max(1, n - 1)) for i in range(n)})
    return [files[i] for i in idx]


def stage_reference_views(images_dir: Path, ref_root: Path, scene: str, n: int = 40,
                          max_side: int = 640) -> list[Path]:
    """SCODA wants Ref_views/<scene>/NNN.png. Downscale on the way (it works at 224 px)."""
    from PIL import Image
    out_dir = Path(ref_root) / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, src in enumerate(pick_reference_views(images_dir, n)):
        dst = out_dir / f"{i:03d}.png"
        if not dst.exists():
            im = Image.open(src).convert("RGB")
            im.thumbnail((max_side, max_side))
            im.save(dst)
        written.append(dst)
    return written


def write_input_csv(render_dir: Path, scene: str, csv_path: Path) -> int:
    rows = sorted(p for p in Path(render_dir).iterdir() if p.suffix.lower() == ".png")
    with Path(csv_path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "scene"])
        for p in rows:
            w.writerow([str(p), scene])
    return len(rows)


def scoda_command(input_csv: Path, ref_root: Path, stats_dir: Path, output_csv: Path,
                  device: str = "cpu") -> list[str]:
    return [str(SCODA_PY), str(SCODA_ROOT / "run_scoda.py"), "--input-csv", str(input_csv),
            "--scene-ref-root", str(ref_root), "--stats-dir", str(stats_dir),
            "--disc-ckpt", str(SCODA_ROOT / "models" / "disc_patchgan_l3_attn_ep10_scenes5.pth"),
            "--output-csv", str(output_csv), "--device", device, "--no-use-pca"]


def read_scores(output_csv: Path) -> list[dict[str, Any]]:
    with Path(output_csv).open() as fh:
        return list(csv.DictReader(fh))


def aggregate(rows: list[dict[str, Any]], cams: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mean/min of Q_final overall and split by eval-pose vs perturbed (novel) renders."""
    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    novel_by_file = {}
    for r in (cams or {}).get("rendered", []):
        novel_by_file[r["file"]] = bool(r.get("novel"))
    per_view = []
    for r in rows:
        q = f(r.get("Q_final"))
        if q is None:
            continue
        fn = Path(r.get("image", "")).name
        per_view.append({"file": fn, "Q_F": f(r.get("Q_F")), "Q_R": f(r.get("Q_R")), "Q_final": q,
                         "novel": novel_by_file.get(fn)})
    def stats(vals):
        return {"n": len(vals), "mean": round(statistics.fmean(vals), 4) if vals else None,
                "min": round(min(vals), 4) if vals else None, "max": round(max(vals), 4) if vals else None}
    allq = [v["Q_final"] for v in per_view]
    return {"overall": stats(allq),
            "eval_pose": stats([v["Q_final"] for v in per_view if v["novel"] is False]),
            "novel_pose": stats([v["Q_final"] for v in per_view if v["novel"] is True]),
            "per_view": per_view}


def run_gate(job_dir: Path, *, scene: str | None = None, ref_n: int = 40, device: str = "cpu",
             dry_run: bool = False) -> dict[str, Any]:
    job_dir = Path(job_dir)
    scene = scene or job_dir.name
    render_dir = job_dir / "_health" / "scoda" / "renders"
    work = job_dir / "_health" / "scoda"
    work.mkdir(parents=True, exist_ok=True)
    cams = json.loads((render_dir / "cams.json").read_text()) if (render_dir / "cams.json").is_file() else None
    result: dict[str, Any] = {"schema": "dev.splatlab.scoda-gate/v1", "job_id": job_dir.name,
                              "report_only": True, "renders_present": cams is not None,
                              "set_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")}
    if cams is None:
        result["error"] = f"no renders at {render_dir} — run backend/health/run_render_views.sh first"
        return result
    refs = stage_reference_views(job_dir / "processed" / "images", work / "Ref_views", scene, n=ref_n)
    n_renders = write_input_csv(render_dir, scene, work / "input.csv")
    cmd = scoda_command(work / "input.csv", work / "Ref_views", work / "scene_stats", work / "scores.csv", device)
    result.update({"reference_views": len(refs), "renders": n_renders, "command": cmd})
    if dry_run:
        return result
    proc = subprocess.run(cmd, cwd=str(SCODA_ROOT), capture_output=True, text=True, timeout=1800)
    result["scoda_exit"] = proc.returncode
    result["scoda_stderr_tail"] = proc.stderr[-800:]
    if proc.returncode == 0 and (work / "scores.csv").is_file():
        result.update(aggregate(read_scores(work / "scores.csv"), cams))
    (job_dir / "_health" / "scoda.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def compare(results: dict[str, dict[str, Any]], expected_order: list[str]) -> dict[str, Any]:
    """Does SCODA rank the known-good world above the known-bad one?"""
    means = {j: (r.get("overall") or {}).get("mean") for j, r in results.items()}
    ok = all(means.get(a) is not None and means.get(b) is not None and means[a] > means[b]
             for a, b in zip(expected_order, expected_order[1:]))
    return {"expected_order_best_first": expected_order, "means": means, "ranking_correct": ok}
