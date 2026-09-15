#!/usr/bin/env python3
"""SCODA render-agreement gate, CPU side. Scores <job>/_health/scoda/renders/*.png
(made by backend/health/run_render_views.sh) against the job's own photographs and
writes <job>/_health/scoda.json (report-only). `--compare good bad` checks the ranking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from health import scoda_gate as sg  # noqa: E402

OUTPUTS = Path.home() / "projects" / "splatcli" / "outputs" / "3d"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="job ids or dirs; with --compare, best first")
    ap.add_argument("--compare", action="store_true", help="assert jobs[0] scores above jobs[1] above ...")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ref-n", type=int, default=40)
    args = ap.parse_args()
    results = {}
    for j in args.jobs:
        job_dir = Path(j) if Path(j).is_dir() else OUTPUTS / j
        res = sg.run_gate(job_dir, device=args.device, ref_n=args.ref_n, dry_run=args.dry_run)
        results[job_dir.name] = res
        line = {k: res.get(k) for k in ("job_id", "renders_present", "renders", "reference_views", "scoda_exit", "overall", "eval_pose", "novel_pose", "error")}
        print(json.dumps(line))
    if args.compare and len(args.jobs) >= 2:
        cmp = sg.compare(results, [Path(j).name for j in args.jobs])
        print(json.dumps(cmp))
        return 0 if cmp["ranking_correct"] or args.dry_run else 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
