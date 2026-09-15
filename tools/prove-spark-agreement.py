#!/usr/bin/env python3
"""Render each arm's exported PLY in SparkJS at the held-out eval cameras and
score viewer-vs-trained agreement. GPU browser: run through the compute gate.

  prove-spark-agreement.py --arm baseline=<dir> --arm antialiased=<dir> --out <dir>
Each <dir> needs _export/splat.ply and _renders/cams.json (+ PNGs) from
backend/health/run_render_views.sh --variants eval --rasterize-mode auto.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend")); sys.path.insert(0, str(HERE / "spark-agreement"))
from health import spark_agreement as sa  # noqa: E402
import serve  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="name=dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settle-frames", type=int, default=40)
    args = ap.parse_args()
    arms = {k: Path(v) for k, v in (a.split("=", 1) for a in args.arm)}
    ply_map = {f"{k}.ply": v / "_export" / "splat.ply" for k, v in arms.items()}
    for k, p in ply_map.items():
        if not p.is_file(): sys.exit(f"missing {p}")
    srv, port = serve.start(ply_map)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist"])
        for name, arm in arms.items():
            rows = sa.eval_rows(arm / "_renders" / "cams.json")
            if not rows: errors.append(f"{name}: no eval rows"); continue
            w, h = int(rows[0]["w"]), int(rows[0]["h"])
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
            console = []
            page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: console.append(f"pageerror: {e}"))
            page.goto(f"http://127.0.0.1:{port}/?ply=/ply/{name}.ply&w={w}&h={h}")
            try:
                page.wait_for_function("window.__spark && (window.__spark.ready || window.__spark.error)", timeout=180_000)
            except Exception as exc:  # noqa: BLE001 — report what the page said instead of a bare timeout
                page_errors = page.evaluate("window.__pageErrors || []")
                errors.append(f"{name}: page never became ready ({type(exc).__name__}); page errors: {page_errors[:3]}; console: {console[-5:]}")
                page.close(); continue
            err = page.evaluate("window.__spark.error")
            if err: errors.append(f"{name}: {err}; console: {console[-5:]}"); page.close(); continue
            n_splats = page.evaluate("window.__spark.numSplats()")
            out_dir = arm / "_spark"; out_dir.mkdir(exist_ok=True)
            for r in rows:
                cam = sa.camera_to_three(r)
                page.evaluate("c => window.__spark.setCamera(c)", cam)
                f0 = page.evaluate("window.__spark.frames()")
                page.wait_for_function(f"window.__spark.frames() >= {f0 + args.settle_frames}", timeout=60_000)
                page.locator("canvas").screenshot(path=str(out_dir / r["file"]))
            print(f"[spark] {name}: {n_splats} splats, {len(rows)} cameras -> {out_dir}", flush=True)
            page.close()
        browser.close()
    srv.shutdown()
    results = {name: sa.compare_arm(arm) for name, arm in arms.items()}
    names = list(arms)
    report = {"schema": "dev.splatlab.spark-agreement/v1", "arms": results, "errors": errors,
              "runtime_s": round(time.time() - t0, 1)}
    if len(names) >= 2:
        report["verdict"] = sa.verdict(results[names[0]], results[names[1]])
    sheet = sa.contact_sheet(list(arms.values()), args.out / "contact-sheet.png")
    report["contact_sheet"] = str(sheet)
    (args.out / "spark-agreement.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{'arm':<14}{'mode':<13}{'spark~photo':>12}{'ns~photo':>10}{'spark~ns':>10}{'|Δ|/255':>9}")
    for name, r in results.items():
        print(f"{name:<14}{str(r['rasterize_mode']):<13}{str(r['psnr_spark_vs_photo']):>12}{str(r['psnr_ns_vs_photo']):>10}{str(r['psnr_spark_vs_ns']):>10}{str(r['mean_abs_spark_vs_ns']):>9}")
    if "verdict" in report:
        v = report["verdict"]; print(f"verdict: viewer_safe={v['viewer_safe']}  Δ(spark~photo)={v['spark_vs_photo_delta_db']} dB  Δ(spark~ns)={v['spark_vs_ns_delta_db']} dB — {v['reason']}")
    if errors: print("errors:", *errors[:5], sep="\n  ")
    print(f"receipt: {args.out / 'spark-agreement.json'}  sheet: {sheet}")
    if errors and not any(r.get("n") for r in results.values()):
        return 3
    return 0 if not errors else 3


if __name__ == "__main__":
    sys.exit(main())
