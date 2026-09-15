#!/usr/bin/env python3
"""Isolate-by-reference orchestrator (CPU): runs the four GPU steps through the
compute gate in their own envs and prints the receipt.

  isolate-by-reference.py --job splat_aea04ab3 --concept bonsai [--views 6] [--iters 60]

Artifacts land in <job>/_isolate/<slug>/ (object_indices.npz in checkpoint order,
object.ply in the batch_isolate 14-field layout, receipt.json, track/ frames+masks).
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = Path.home() / "projects" / "splatcli" / "outputs" / "3d"
GATE = REPO / "tools" / "splatlab-compute-gate.sh"
LANGFIELD_PY = Path.home() / "miniconda3" / "envs" / "langfield-spike" / "bin" / "python"
SAM3_PY = Path.home() / "miniconda3" / "envs" / "sam3" / "bin" / "python"
ISO = REPO / "backend" / "isolate"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "object"


def gated(cmd: list[str], vram_mb: int, env: dict | None = None, timeout: int = 900) -> None:
    full = ["env", f"SPLAT_GPU_MANUAL_VRAM_MB={vram_mb}", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1", str(GATE), "--run",
            "timeout", "--kill-after=20", str(timeout), *cmd]
    print("   $ " + " ".join(str(c) for c in full[4:]), flush=True)
    subprocess.run(full, check=True, cwd=str(REPO), env={**os.environ, **(env or {})})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--concept", required=True)
    ap.add_argument("--slug", default=None); ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--from-step", choices=["ground", "orbits", "track", "fit"], default="ground")
    a = ap.parse_args()
    job_dir = Path(a.job) if Path(a.job).is_dir() else OUTPUTS / a.job
    cfgs = sorted(job_dir.glob("processed/splatfacto/*/config.yml"), key=lambda p: p.stat().st_mtime)
    if not cfgs:
        sys.exit(f"no splatfacto checkpoint under {job_dir}")
    cfg = cfgs[-1]; slug = a.slug or slugify(a.concept)
    work = job_dir / "_isolate" / slug; work.mkdir(parents=True, exist_ok=True)
    steps = ["ground", "orbits", "track", "fit"]; t0 = time.time(); timing = {}
    sam3_env = {"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/home/rtoony/projects/ml/sam3"}
    for step in steps[steps.index(a.from_step):]:
        ts = time.time(); print(f"-- {step}", flush=True)
        if step == "ground":
            gated([str(LANGFIELD_PY), str(ISO / "prepare_ref.py"), str(cfg), str(work), "--views", str(a.views)], 4000)
            (work / "ref" / "things.json").write_text(json.dumps([a.concept]))
            (work / "concept.json").write_text(json.dumps({"concept": a.concept, "slug": slug}))
            gated([str(SAM3_PY), str(REPO / "backend" / "mesh" / "scene_sam3_masks.py"), str(work / "ref"), str(work / "ref" / "things.json"), "0.3"],
                  8000, env=sam3_env)
        elif step == "orbits":
            gated([str(LANGFIELD_PY), str(ISO / "render_orbits.py"), str(cfg), str(work), "--slug", slug], 6000)
        elif step == "track":
            gated([str(SAM3_PY), str(ISO / "sam3_track.py"), str(work)], 14000, env=sam3_env)  # multiplex model + 1008 px memory bank ~12 GB
        elif step == "fit":
            base = None
            for cand in (job_dir / "_scene" / "isolated" / slug / "object_indices.npz", job_dir / "_scene" / f"instance_{slug}.npz",
                         job_dir / "_objects" / slug / "object_indices.npz"):
                if cand.is_file(): base = cand; break
            cmd = [str(LANGFIELD_PY), str(ISO / "fit_logits.py"), str(cfg), str(work), "--iters", str(a.iters), "--threshold", str(a.threshold)]
            if base: cmd += ["--baseline", str(base)]
            gated(cmd, 8000)
        timing[step] = round(time.time() - ts, 1)
    receipt = json.loads((work / "receipt.json").read_text())
    receipt["timing_s"] = {**timing, "total": round(time.time() - t0, 1)}; receipt["concept"] = a.concept; receipt["job_id"] = job_dir.name
    receipt["orbit"] = json.loads((work / "orbit_receipt.json").read_text())
    (work / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("job_id", "concept", "n_object", "n_seed", "seed_retained", "mask_iou_mean", "mask_iou_min", "timing_s") if k in receipt}, indent=1))
    if "baseline" in receipt: print("baseline:", json.dumps(receipt["baseline"]))
    print(f"receipt: {work / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
