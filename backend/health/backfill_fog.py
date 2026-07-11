"""Backfill the fog-fingerprint gate across existing trained scenes.

Stdlib-only CLI (system python3, no splatlab venv needed). For each job dir it
finds the newest trained config.yml, runs run_health.sh (langfield-spike env)
in a fresh subprocess (VRAM fully releases between scenes), and collects
<job_dir>/_health/fog.json into a human calibration report.

SAFETY: refuses to run while any splat job is active or GPU free VRAM is low
(--force overrides). NEVER touches meta.json unless --write-meta is passed —
and _patch-style writes are only safe because the preflight guarantees the
server isn't mid-job (meta writes are read-modify-write with no cross-process
lock).

Usage:
  python3 backend/health/backfill_fog.py --jobs splat_a,splat_b \
      [--expected splat_a=FOG,splat_b=HEALTHY] [--lenient splat_b] [--strict] \
      [--max-runtime 120] [--report-dir DIR] [--write-meta] [--force] [--all]
Exit codes: 0 = ran (and, with --strict, all expectations met); 1 = expectation
mismatch or scene error under --strict; 2 = preflight refusal / bad args.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_3D = Path("/home/rtoony/projects/splatcli/outputs/3d")
RUNNER = Path(__file__).resolve().parent / "run_health.sh"
JOB_ID_RE = re.compile(r"^splat_[0-9a-f]{6,32}$")
ACTIVE_STATUSES = {"starting", "running"}
MIN_FREE_VRAM_MB = 6000
SCENE_TIMEOUT_S = 900
VERDICTS = {"FOG", "HEALTHY", "UNCERTAIN"}


def preflight(force: bool) -> None:
    active = []
    for meta_path in ROOT_3D.glob("splat_*/meta.json"):
        try:
            status = json.loads(meta_path.read_text()).get("status")
        except (json.JSONDecodeError, OSError):
            continue
        if status in ACTIVE_STATUSES:
            active.append(meta_path.parent.name)
    if active:
        msg = f"active splat job(s): {', '.join(active)}"
        if not force:
            sys.exit(f"REFUSING to run: {msg} (use --force to override)")
        print(f"[backfill] WARNING: {msg} — continuing under --force", flush=True)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip().splitlines()[0]
        used_mb, total_mb = (int(x.strip()) for x in out.split(","))
        free_mb = total_mb - used_mb
        if free_mb < MIN_FREE_VRAM_MB:
            msg = f"only {free_mb}MB GPU free (< {MIN_FREE_VRAM_MB}MB) — something heavy is resident"
            if not force:
                sys.exit(f"REFUSING to run: {msg} (use --force to override)")
            print(f"[backfill] WARNING: {msg} — continuing under --force", flush=True)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        print(f"[backfill] WARNING: could not check GPU ({exc}) — continuing", flush=True)


def find_config(job_dir: Path) -> Path | None:
    candidates = sorted(job_dir.rglob("config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_scene(job_id: str, cams: int, max_width: int) -> dict:
    job_dir = ROOT_3D / job_id
    row = {"job": job_id, "verdict": None, "error": None}
    if not job_dir.is_dir():
        row["error"] = "job dir missing"
        return row
    config = find_config(job_dir)
    if config is None:
        row["error"] = "no trained config.yml"
        return row
    health_dir = job_dir / "_health"
    print(f"\n[backfill] === {job_id} ({config.parent.name}) ===", flush=True)
    try:
        proc = subprocess.run(
            ["bash", str(RUNNER), str(config), str(health_dir),
             "--cams", str(cams), "--max-width", str(max_width)],
            timeout=SCENE_TIMEOUT_S)
        if proc.returncode != 0:
            row["error"] = f"fog gate exited {proc.returncode}"
            return row
    except subprocess.TimeoutExpired:
        row["error"] = f"fog gate timed out after {SCENE_TIMEOUT_S}s"
        return row
    try:
        result = json.loads((health_dir / "fog.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        row["error"] = f"unreadable fog.json ({exc})"
        return row
    row.update({
        "verdict": result["verdict"],
        "median_shell": result["summary"].get("median_shell_frac"),
        "median_spread": result["summary"]["median_spread"],
        "median_p50": result["summary"]["median_p50"],
        "median_acc": result["summary"]["median_acc"],
        "n_fog_cams": f"{result['summary']['n_fog']}/{result['summary']['n_counted']}",
        "runtime_s": result["runtime_s"],
        "n_receipts": len(result["receipts"]),
        "health_dir": str(health_dir),
        "result": result,
    })
    return row


def write_meta(job_id: str, result: dict) -> str:
    """Patch meta['health'] the way the server would (atomic tmp+replace)."""
    meta_path = ROOT_3D / job_id / "meta.json"
    if not meta_path.exists():
        return "no meta.json (skipped)"
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return "corrupt meta.json (skipped)"
    fog = {k: result[k] for k in
           ("verdict", "checked_at", "runtime_s", "cameras", "summary", "thresholds", "receipts")}
    fog["enforced"] = False
    meta["health"] = {"v": 1, "fog": fog}
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(meta_path)
    return "meta patched"


def write_report(report_dir: Path, rows: list[dict], expected: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Capture Coach — fog-gate calibration ({datetime.now(timezone.utc).date()})",
        "",
        "Fingerprint: per-camera SHELL FRACTION = share of opaque pixels within 3× the near plane "
        "(depth ≤ 0.03 scene units). Cocoon/fog camera: shell ≥ 50% at acc ≈ 1. Clean camera: shell ≤ 5% "
        "with median depth ≥ 0.1. Verdict = 2/3 camera majority, else UNCERTAIN. "
        "(p95/p5 spread is still reported but no longer drives the verdict — it breaks on mixed scenes "
        "where a few pixels punch through the cocoon.)",
        "Receipts below are [RGB render | turbo log-depth] side-by-sides at the probed training cameras — "
        "a fog scene reads as a mushy smear next to a flat depth blob.",
        "",
        "| job | expected | verdict | fog cams | median shell | median spread | median p50 | median acc | runtime s | receipts |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        exp = expected.get(r["job"], "—")
        if r["error"]:
            lines.append(f"| {r['job']} | {exp} | ERROR | — | — | — | — | — | {r['error']} |")
            continue
        mark = "" if exp in ("—", r["verdict"]) else " ⚠️"
        lines.append(
            f"| {r['job']} | {exp} | **{r['verdict']}**{mark} | {r['n_fog_cams']} | "
            f"{r['median_shell']} | {r['median_spread']} | {r['median_p50']} | {r['median_acc']} | "
            f"{r['runtime_s']} | {r['n_receipts']} |")
    for r in rows:
        if r["error"]:
            continue
        scene_dir = report_dir / r["job"]
        scene_dir.mkdir(exist_ok=True)
        lines += ["", f"## {r['job']} — {r['verdict']}"
                      f" (expected {expected.get(r['job'], 'unlabeled')})", ""]
        for name in r["result"]["receipts"]:
            src = Path(r["health_dir"]) / name
            if src.exists():
                shutil.copy2(src, scene_dir / name)
                lines.append(f"![{name}]({r['job']}/{name})")
    (report_dir / "index.md").write_text("\n".join(lines) + "\n")
    slim = [{k: v for k, v in r.items() if k != "result"} for r in rows]
    (report_dir / "summary.json").write_text(json.dumps(slim, indent=2))
    print(f"\n[backfill] report -> {report_dir}/index.md", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", default="", help="comma-separated job ids")
    ap.add_argument("--all", action="store_true", help="every splat_* dir with a config.yml")
    ap.add_argument("--expected", default="", help="id=VERDICT,... assertions for --strict")
    ap.add_argument("--lenient", default="", help="ids for which UNCERTAIN also passes")
    ap.add_argument("--strict", action="store_true", help="exit 1 on expectation mismatch/error")
    ap.add_argument("--max-runtime", type=float, default=0, help="per-scene runtime assertion (s)")
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--write-meta", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cams", type=int, default=6)
    ap.add_argument("--max-width", type=int, default=640)
    args = ap.parse_args()

    if args.all:
        jobs = sorted(d.name for d in ROOT_3D.glob("splat_*")
                      if d.is_dir() and find_config(d) is not None)
    else:
        jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    if not jobs:
        sys.exit("no jobs: pass --jobs id,id or --all")
    bad = [j for j in jobs if not JOB_ID_RE.match(j)]
    if bad:
        sys.exit(f"invalid job id(s): {bad}")
    expected = {}
    for pair in filter(None, (p.strip() for p in args.expected.split(","))):
        job, _, verdict = pair.partition("=")
        if verdict not in VERDICTS:
            sys.exit(f"bad --expected verdict in '{pair}' (use {sorted(VERDICTS)})")
        expected[job] = verdict
    lenient = {j.strip() for j in args.lenient.split(",") if j.strip()}

    preflight(args.force)
    rows = [run_scene(j, args.cams, args.max_width) for j in jobs]

    if args.write_meta:
        for r in rows:
            if not r["error"]:
                print(f"[backfill] {r['job']}: {write_meta(r['job'], r['result'])}", flush=True)

    if args.report_dir:
        write_report(args.report_dir, rows, expected)

    failures = []
    for r in rows:
        if r["error"]:
            failures.append(f"{r['job']}: ERROR — {r['error']}")
            continue
        exp = expected.get(r["job"])
        if exp and r["verdict"] != exp and not (r["job"] in lenient and r["verdict"] == "UNCERTAIN"):
            failures.append(f"{r['job']}: verdict {r['verdict']} != expected {exp} "
                            f"(spread {r['median_spread']}, p50 {r['median_p50']})")
        if args.max_runtime and r["runtime_s"] > args.max_runtime:
            failures.append(f"{r['job']}: runtime {r['runtime_s']}s > {args.max_runtime}s")
        on_disk = sum(1 for name in r["result"]["receipts"]
                      if (Path(r["health_dir"]) / name).exists())
        if on_disk == 0 or on_disk < r["n_receipts"]:
            failures.append(f"{r['job']}: only {on_disk}/{r['n_receipts']} receipt images on disk")

    print("\n[backfill] " + " | ".join(
        f"{r['job']}={r['verdict'] or 'ERROR'}" for r in rows), flush=True)
    if failures and args.strict:
        print("\n[backfill] STRICT FAILURES:", flush=True)
        for f in failures:
            print(f"  ✗ {f}", flush=True)
        sys.exit(1)
    if failures:
        print(f"\n[backfill] {len(failures)} expectation mismatch(es) (non-strict):", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)


if __name__ == "__main__":
    main()
