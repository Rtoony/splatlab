#!/usr/bin/env python3
"""External-trainer bake-off helpers (research only; see
~/reports/2026-09-29-spirula-bakeoff/). Runs in the `langfield-spike` env, the
one that has nerfstudio 1.1.5, so the held-out split and the metrics are
nerfstudio's own rather than re-implementations.

  split   JOB OUT             dump nerfstudio's eval split for JOB and build a
                              symlink mirror whose names carry the split
  score   ARM_DIR --mirror M  score one arm's held-out renders against the
                              photos on disk (PSNR/SSIM/LPIPS, raw + colour-corrected)
  to-ns-frame PLY --dataparser-transforms F  move a PLY trained on transforms.json into the
                              nerfstudio viewer frame, SH0 (Spark comparison)
  summary SCENE_DIR           results table over every scored arm
  check-renders ARM_DIR       exit 0 iff every held-out photo has exactly one render
  sheet   OUT --mirror M --arms L=score.json..  photo | arm | arm contact sheet

CPU-only except `score` (LPIPS), which must go through
tools/splatlab-compute-gate.sh like every GPU command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from bakeoff import ply_frame  # noqa: E402
from bakeoff import scoring  # noqa: E402
from bakeoff import split as bsplit  # noqa: E402

OUTPUTS = Path.home() / "projects" / "splatcli" / "outputs" / "3d"


def job_dir(job: str) -> Path:
    p = Path(job)
    return p if p.is_dir() else OUTPUTS / job


def latest_config(job: Path) -> Path:
    configs = sorted((job / "processed" / "splatfacto").glob("*/config.yml"))
    if not configs:
        raise SystemExit(f"{job}: no processed/splatfacto/*/config.yml")
    return configs[-1]


def nerfstudio_eval_names(job: Path) -> list[str]:
    """Frame file_paths of nerfstudio's `test` split, in eval-dataloader order,
    from the job's own dataparser config (so any non-default split is kept)."""
    import yaml
    processed = (job / "processed").resolve()
    # The job's own ns-train config (typed !!python/object tags); ns-eval loads it the same way.
    cfg = yaml.load(latest_config(job).read_text(), Loader=yaml.Loader)
    dp_cfg = cfg.pipeline.datamanager.dataparser
    dp_cfg.data = processed
    outs = dp_cfg.setup().get_dataparser_outputs(split="test")
    return [str(Path(p).resolve().relative_to(processed)) for p in outs.image_filenames]


def cmd_split(args) -> int:
    job = job_dir(args.job)
    names = nerfstudio_eval_names(job)
    receipt = bsplit.build_mirror(job / "processed", names, Path(args.out))
    print(json.dumps({k: receipt[k] for k in ("source", "frames", "train", "eval")}, indent=2))
    print("first eval frames:", names[:3])
    return 0


def load_split(mirror: Path) -> dict:
    return json.loads((Path(mirror) / "split-receipt.json").read_text())


def cmd_score(args) -> int:
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    photos = {n: scoring.load_rgb(src / n) for n in receipt["eval_names"]}
    pairs = scoring.ARM_LAYOUTS[args.layout](Path(args.renders))
    result = scoring.score(pairs, photos, scoring.Metrics(args.device))
    result.update({"layout": args.layout, "renders": str(Path(args.renders).resolve()),
                   "mirror": str(Path(args.mirror).resolve())})
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    m = result["mean"]
    print(f"{out}: {result['views']} views  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}"
          f"  | cc PSNR {m['cc_psnr']:.3f}  SSIM {m['cc_ssim']:.4f}  LPIPS {m['cc_lpips']:.4f}"
          f"  (max GT match {result['max_gt_match_mean_abs_255']:.2f}/255)")
    return 0


def cmd_check_renders(args) -> int:
    """Exit 0 iff the renders cover every held-out photo exactly once (CPU only).
    Spirula 2026.9.24's parallel eval writer sometimes writes one view twice and
    another never; the runner re-evaluates from the full checkpoint until this passes."""
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    photos = {n: scoring.load_rgb(src / n) for n in receipt["eval_names"]}
    try:
        scoring.match_to_photos(scoring.ARM_LAYOUTS[args.layout](Path(args.renders)), photos)
    except ValueError as exc:
        print(f"INCOMPLETE: {exc}")
        return 4
    print(f"COMPLETE: {len(photos)} held-out photos, one render each")
    return 0


def cmd_to_ns_frame(args) -> int:
    from isolate.splat_ply import write_splat_ply
    cols = ply_frame.read_vertex_ply(Path(args.ply))
    if args.dataparser_transforms == "identity":  # already in the nerfstudio frame: SH0 strip only
        r, t, s = np.eye(3), np.zeros(3), 1.0
    else:
        r, t, s = ply_frame.load_dataparser_transform(Path(args.dataparser_transforms))
    xyz, f_dc, opacity, scale, rot = ply_frame.to_ns_frame(cols, r, t, s)
    n = write_splat_ply(Path(args.out), xyz, f_dc, opacity, scale, rot,
                        comment="trainer-bakeoff to-ns-frame SH0")
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    report = {"gaussians": n, "centroid": xyz.mean(axis=0).round(4).tolist(),
              "p1": np.percentile(xyz, 1, axis=0).round(4).tolist(),
              "p99": np.percentile(xyz, 99, axis=0).round(4).tolist(),
              "bbox": [lo.round(4).tolist(), hi.round(4).tolist()], "scale": s}
    if args.reference:
        ref = ply_frame.read_vertex_ply(Path(args.reference))
        rxyz = np.stack([ref["x"], ref["y"], ref["z"]], axis=1)
        rp1, rp99 = np.percentile(rxyz, 1, axis=0), np.percentile(rxyz, 99, axis=0)
        span = rp99 - rp1
        c = np.median(xyz, axis=0)
        report["reference_p1_p99"] = [rp1.round(4).tolist(), rp99.round(4).tolist()]
        report["median_inside_reference"] = bool(np.all((c >= rp1) & (c <= rp99)))
        report["median_offset_over_span"] = (np.abs(c - np.median(rxyz, axis=0)) / span).round(4).tolist()
    print(json.dumps(report, indent=2))
    if args.reference and not report["median_inside_reference"]:
        print("FRAME CHECK FAILED: median splat is outside the reference's 1-99% box", file=sys.stderr)
        return 3
    return 0


def cmd_sheet(args) -> int:
    from PIL import Image, ImageDraw
    arms = []
    for spec in args.arms:
        label, score_path = spec.split("=", 1)
        arms.append((label, json.loads(Path(score_path).read_text())))
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    rows0 = {r["photo"]: r for r in arms[0][1]["per_view"]}
    rows1 = {r["photo"]: r for r in arms[-1][1]["per_view"]}
    by_gap = sorted(rows0, key=lambda n: rows1[n]["lpips"] - rows0[n]["lpips"])
    by_base = sorted(rows0, key=lambda n: rows0[n]["lpips"])
    picks = []
    for n in (by_gap[0], by_gap[-1], by_base[len(by_base) // 2], by_base[-1]):
        if n not in picks:
            picks.append(n)
    tile_w = args.tile_width
    tiles = []
    for name in picks:
        row = [("photo " + Path(name).name, scoring.load_rgb(src / name))]
        for label, res in arms:
            r = next(v for v in res["per_view"] if v["photo"] == name)
            pairs = scoring.ARM_LAYOUTS[res["layout"]](Path(res["renders"]))
            pair = next(p for p in pairs if p.key == r["arm_key"])
            row.append((f"{label}  {r['psnr']:.2f} dB  LPIPS {r['lpips']:.3f}", pair.pred))
        tiles.append(row)
    h0, w0 = tiles[0][0][1].shape[:2]
    tile_h = int(round(tile_w * h0 / w0))
    sheet = Image.new("RGB", (tile_w * len(tiles[0]), (tile_h + 22) * len(tiles)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(tiles):
        for j, (label, img) in enumerate(row):
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((tile_w, tile_h), Image.LANCZOS)
            sheet.paste(im, (j * tile_w, i * (tile_h + 22) + 22))
            draw.text((j * tile_w + 6, i * (tile_h + 22) + 5), label, fill="black")
    sheet.save(args.out)
    print(f"{args.out}: {len(tiles)} views x {len(tiles[0])} columns: {picks}")
    return 0


def gaussian_count(arm: Path) -> int | None:
    plys = sorted(arm.glob("step-*.ckpt/splat.ply"))
    if plys:
        head = plys[-1].read_bytes()[:4096].decode("ascii", "replace")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                return int(line.split()[2])
    ckpts = sorted(arm.glob("processed/splatfacto/*/nerfstudio_models/step-*.ckpt"))
    if ckpts:
        import torch
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)["pipeline"]
        return int(state["_model.gauss_params.means"].shape[0])
    return None


def cmd_summary(args) -> int:
    scene = Path(args.scene_dir)
    rows = []
    for arm in sorted(p for p in scene.iterdir() if (p / "score.json").is_file()):
        sc = json.loads((arm / "score.json").read_text())
        tm = json.loads((arm / "timing.json").read_text()) if (arm / "timing.json").is_file() else {}
        m = sc["mean"]
        rows.append({"arm": arm.name, "views": sc["views"], **{k: round(v, 4) for k, v in m.items()},
                     "train_s": tm.get("train_s"), "peak_vram_mb": tm.get("peak_vram_mb"),
                     "gaussians": gaussian_count(arm), "renders": sc["renders"]})
    (scene / "summary.json").write_text(json.dumps(rows, indent=2))
    cols = ["arm", "views", "psnr", "ssim", "lpips", "cc_psnr", "cc_ssim", "cc_lpips", "train_s", "peak_vram_mb",
            "gaussians"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("job")
    p.add_argument("out")
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("score")
    p.add_argument("renders", help="directory holding the arm's held-out renders")
    p.add_argument("--layout", required=True, choices=sorted(scoring.ARM_LAYOUTS))
    p.add_argument("--mirror", required=True, help="split mirror (split-receipt.json names the photos)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("check-renders")
    p.add_argument("renders")
    p.add_argument("--layout", required=True, choices=sorted(scoring.ARM_LAYOUTS))
    p.add_argument("--mirror", required=True)
    p.set_defaults(fn=cmd_check_renders)
    p = sub.add_parser("to-ns-frame")
    p.add_argument("ply")
    p.add_argument("--dataparser-transforms", required=True,
                   help="the job's dataparser_transforms.json, or `identity` to only strip to SH0")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", help="the job's own ns-export splat.ply, for the frame check")
    p.set_defaults(fn=cmd_to_ns_frame)
    p = sub.add_parser("summary")
    p.add_argument("scene_dir")
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("sheet")
    p.add_argument("out")
    p.add_argument("--mirror", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="LABEL=score.json (first = baseline)")
    p.add_argument("--tile-width", type=int, default=560)
    p.set_defaults(fn=cmd_sheet)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
