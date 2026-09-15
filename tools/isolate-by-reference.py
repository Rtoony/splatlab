#!/usr/bin/env python3
"""Isolate-by-reference orchestrator (CPU): runs the GPU steps through the compute
gate in their own envs and prints the receipt.

  isolate-by-reference.py --job splat_aea04ab3 --concept bonsai [--views 6] [--iters 60]
                          [--no-reground] [--no-gate] [--from-step ground|seed|reground|orbits|track|fit]

Steps: ground (SAM3 text masks on K training photos) -> seed (photo seed + pulled-back
renders) -> reground (SAM3 on the renders, so the reference covers the WHOLE object
even when no photo does) -> orbits -> track (SAM3 per-frame) -> fit (per-gaussian logit).
--no-gate runs the steps directly (for callers that already hold a GPU lease, e.g. the
backend route). Artifacts land in <job>/_isolate/<slug>/ (object_indices.npz in
checkpoint order, object.ply in the batch_isolate 14-field layout, receipt.json, track/).
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
SAM3_MASKS = REPO / "backend" / "mesh" / "scene_sam3_masks.py"
STEPS = ["ground", "seed", "reground", "orbits", "track", "fit"]
SAM3_ENV = {"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/home/rtoony/projects/ml/sam3"}


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "object"


def run_step(cmd: list[str], vram_mb: int, env: dict | None = None, timeout: int = 900, gate: bool = True) -> None:
    base = ["env", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1"]
    if gate:
        base += [f"SPLAT_GPU_MANUAL_VRAM_MB={vram_mb}", str(GATE), "--run"]
    full = base + ["timeout", "--kill-after=20", str(timeout), *cmd]
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(full, check=True, cwd=str(REPO), env={**os.environ, **(env or {})})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--concept", required=True)
    ap.add_argument("--slug", default=None); ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--from-step", choices=STEPS, default="ground")
    ap.add_argument("--no-reground", action="store_true", help="orbit straight from the photo seed (old one-pass behaviour)")
    ap.add_argument("--no-gate", action="store_true", help="caller already holds the GPU lease: run steps directly")
    a = ap.parse_args()
    job_dir = Path(a.job) if Path(a.job).is_dir() else OUTPUTS / a.job
    cfgs = sorted(job_dir.glob("processed/splatfacto/*/config.yml"), key=lambda p: p.stat().st_mtime)
    if not cfgs:
        sys.exit(f"no splatfacto checkpoint under {job_dir}")
    cfg = cfgs[-1]; slug = a.slug or slugify(a.concept)
    work = job_dir / "_isolate" / slug; work.mkdir(parents=True, exist_ok=True)
    gate = not a.no_gate; t0 = time.time(); timing = {}
    steps = STEPS[STEPS.index(a.from_step):]
    if a.no_reground:
        steps = [s for s in steps if s not in ("seed", "reground")]
    for step in steps:
        ts = time.time(); print(f"-- {step}", flush=True)
        if step == "ground":
            run_step([str(LANGFIELD_PY), str(ISO / "prepare_ref.py"), str(cfg), str(work), "--views", str(a.views)], 4000, gate=gate)
            (work / "ref" / "things.json").write_text(json.dumps([a.concept]))
            (work / "concept.json").write_text(json.dumps({"concept": a.concept, "slug": slug}))
            run_step([str(SAM3_PY), str(SAM3_MASKS), str(work / "ref"), str(work / "ref" / "things.json"), "0.3"], 8000, env=SAM3_ENV, gate=gate)
        elif step == "seed":
            run_step([str(LANGFIELD_PY), str(ISO / "render_orbits.py"), str(cfg), str(work), "--slug", slug, "--stage", "seed"], 6000, gate=gate)
        elif step == "reground":
            if not (work / "reground" / "views.json").is_file():
                print("   (no pulled-back views rendered — skipping)", flush=True); continue
            run_step([str(SAM3_PY), str(SAM3_MASKS), str(work / "reground"), str(work / "ref" / "things.json"), "0.3"], 8000, env=SAM3_ENV, gate=gate)
        elif step == "orbits":
            stage = "full" if a.no_reground else "orbit"
            run_step([str(LANGFIELD_PY), str(ISO / "render_orbits.py"), str(cfg), str(work), "--slug", slug, "--stage", stage], 6000, gate=gate)
        elif step == "track":
            run_step([str(SAM3_PY), str(ISO / "sam3_track.py"), str(work)], 14000, env=SAM3_ENV, gate=gate)  # multiplex model + 1008 px memory bank ~12 GB
        elif step == "fit":
            base = None
            for cand in (job_dir / "_scene" / "isolated" / slug / "object_indices.npz", job_dir / "_scene" / f"instance_{slug}.npz",
                         job_dir / "_objects" / slug / "object_indices.npz"):
                if cand.is_file(): base = cand; break
            cmd = [str(LANGFIELD_PY), str(ISO / "fit_logits.py"), str(cfg), str(work), "--iters", str(a.iters), "--threshold", str(a.threshold)]
            if base: cmd += ["--baseline", str(base)]
            run_step(cmd, 8000, gate=gate)
        timing[step] = round(time.time() - ts, 1)
    receipt = json.loads((work / "receipt.json").read_text())
    orbit = json.loads((work / "orbit_receipt.json").read_text())
    receipt["timing_s"] = {**timing, "total": round(time.time() - t0, 1)}; receipt["concept"] = a.concept; receipt["job_id"] = job_dir.name
    receipt["orbit"] = orbit; receipt["reground"] = orbit.get("reground", {"used": False, "reason": "orbit receipt v1"})
    (work / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({k: receipt[k] for k in ("job_id", "concept", "n_object", "n_seed", "seed_retained", "mask_iou_mean", "mask_iou_min", "reground", "timing_s") if k in receipt}, indent=1))
    if "baseline" in receipt: print("baseline:", json.dumps(receipt["baseline"]))
    print(f"receipt: {work / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
