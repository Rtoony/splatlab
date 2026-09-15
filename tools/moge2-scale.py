#!/usr/bin/env python3
"""Propose `meters_per_unit` for a SplatLab job from MoGe-2 metric depth.

Runs in the `sam3d-body` conda env (moge 2.0.0 + torch cu128). CPU parts
(dataset load, keyframe pick, frame-agreement, splat-vs-sparse extent check) run
in `--dry-run`; the MoGe-2 inference needs CUDA and must be launched through
`tools/splatlab-compute-gate.sh` (see ~/scripts/splatlab-moge2-scale-2026-09-14.sh).

Writes ONLY <job>/_scale/moge2-proposal.json (+ per-frame preview PNGs). Never
touches meta.json — acceptance is the owner's, through the existing calibration
routes. Existing calibration (if any) is compared and reported, not replaced.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
import scale_estimate as se  # noqa: E402

OUTPUTS = Path.home() / "projects" / "splatcli" / "outputs" / "3d"
MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def robust_extent(xyz: np.ndarray) -> float:
    lo, hi = np.percentile(xyz, 5, axis=0), np.percentile(xyz, 95, axis=0)
    return float(np.linalg.norm(hi - lo))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job", help="job id (splat_xxx) or job directory")
    ap.add_argument("--frames", type=int, default=se.DEFAULT_KEYFRAMES)
    ap.add_argument("--num-tokens", type=int, default=1800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="CPU only: dataset, keyframes, frame agreement, extents")
    ap.add_argument("--out", type=Path, default=None, help="default <job>/_scale")
    ap.add_argument("--brief", action="store_true", help="one-line dry-run summary instead of JSON")
    ap.add_argument("--rescore", action="store_true", help="rebuild the proposal without inference: from saved depth_*.npy when present (re-applies --window), else from the receipt's per-frame ratios")
    ap.add_argument("--window", type=int, default=3, help="local-minimum depth sampling window in px (1 = nearest pixel)")
    args = ap.parse_args()

    job_dir = Path(args.job) if Path(args.job).is_dir() else OUTPUTS / args.job
    job_id = job_dir.name
    processed = job_dir / "processed"
    out = args.out or (job_dir / "_scale")
    t0 = time.time()

    transforms = se.load_transforms(processed)
    sparse = se.load_ply_xyz(processed / transforms["ply_file_path"])
    agreement = se.frame_agreement(transforms, sparse)
    meta = json.loads((job_dir / "meta.json").read_text()) if (job_dir / "meta.json").is_file() else {}
    existing = meta.get("scale_calibration") or ({"meters_per_unit": meta["meters_per_unit"], "method": "unknown"}
                                                 if meta.get("meters_per_unit") else None)
    dp = se.find_dataparser_scale(job_dir)
    dp_scale = dp["scale"] if dp else None
    extents = {"sparse_pc": robust_extent(sparse)}
    splat_ply = job_dir / "_preview" / "splat.ply"
    if splat_ply.is_file():
        try:
            extents["splat"] = robust_extent(se.load_ply_xyz(splat_ply))
            extents["splat_over_sparse"] = round(extents["splat"] / extents["sparse_pc"], 4)
        except se.ScaleEstimateError as exc:
            extents["splat_error"] = str(exc)
    picks = se.select_keyframes(len(transforms["frames"]), args.frames)
    plan = []
    for i in picks:
        f = transforms["frames"][i]
        proj = se.project_frame(f, sparse)
        plan.append({"frame": i, "file_path": f["file_path"], "n_projected": int(len(proj["depth"]))})
    summary = {"job_id": job_id, "frames_total": len(transforms["frames"]), "sparse_points": int(len(sparse)),
               "camera_model": transforms["camera_model"], "per_frame_intrinsics": transforms["per_frame_intrinsics"], "fov_x_deg": round(se.fov_x_degrees(transforms["fx"], transforms["w"]), 2),
               "frame_agreement": agreement, "extents": extents, "existing_calibration": existing,
               "dataparser_scale": dp_scale,
               "keyframes": plan}
    if args.brief:
        ex = (existing or {}).get("meters_per_unit")
        print(f"   {job_id}: frames={summary['frames_total']} sparse={summary['sparse_points']} "
              f"fov_x={summary['fov_x_deg']} agreement={agreement['consistent']} "
              f"(in-view {agreement['median_in_view_fraction']}) dataparser_scale={dp_scale} "
              f"existing={ex}")
    else:
        print(json.dumps(summary, indent=1))
    if not agreement["consistent"]:
        print("!! cameras and sparse points do not share a frame — refusing", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    def finish(results, model_id, t_start):
        agg = se.aggregate(results)
        proposal = se.build_proposal(agg, results, model=model_id, source="tools/moge2-scale.py", job_id=job_id,
                                     frame_check=agreement, existing=existing, dataparser_scale=dp_scale)
        proposal["extents"] = extents
        proposal["runtime_s"] = round(time.time() - t_start, 1)
        proposal["num_tokens"] = args.num_tokens
        proposal["sample_window"] = args.window
        out.mkdir(parents=True, exist_ok=True)
        (out / "moge2-proposal.json").write_text(json.dumps(proposal, indent=1) + "\n")
        print(json.dumps({k: proposal[k] for k in ("meters_per_unit", "frame", "meters_per_unit_colmap_frame", "dataparser_scale",
                                                   "confidence", "relative_mad", "frames_used", "existing", "runtime_s") if k in proposal}, indent=1))
        print(f"receipt: {out / 'moge2-proposal.json'}")

    if args.rescore:
        receipt_path = out / "moge2-proposal.json"
        old = json.loads(receipt_path.read_text())
        depths = sorted(out.glob("depth_*.npz"))
        if depths:
            results = []
            for npy in depths:
                fi = int(npy.stem.split("_")[1])
                f = transforms["frames"][fi]
                arr = np.load(npy)
                depth = arr["depth"].astype(np.float64) if hasattr(arr, "files") else arr.astype(np.float64)
                mask = arr["mask"].astype(bool) if hasattr(arr, "files") and "mask" in arr.files else None
                r = se.frame_scale(depth, mask, se.project_frame(f, sparse), window=args.window)
                r.update({"frame": fi, "file_path": f["file_path"]}); results.append(r)
                print(f"  frame {fi:>4} {Path(f['file_path']).name}: n={r['n_points']} inl={r['n_inliers']} ratio={r['ratio']} (window {args.window})", flush=True)
        else:
            print("  (no saved depth maps — reusing the receipt's per-frame ratios; --window has no effect)")
            results = [{"frame": r.get("frame"), "file_path": r.get("file_path"), "n_points": r.get("n_points"),
                        "n_inliers": r.get("n_inliers"), "ratio": r.get("meters_per_unit"), "mad_relative": r.get("mad_relative")}
                       for r in old.get("references", [])]
        keep = out / f"moge2-proposal.v{len(list(out.glob('moge2-proposal.v*.json'))) + 1}.json"
        keep.write_text(json.dumps(old, indent=1) + "\n")
        print(f"rescoring {len(results)} frames from {receipt_path.name} (previous kept as {keep.name})")
        finish(results, old.get("model", MODEL_ID), t0)
        return 0

    import torch  # noqa: E402  (GPU half)
    from PIL import Image  # noqa: E402
    from moge.model.v2 import MoGeModel  # noqa: E402

    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(MODEL_ID).to(device).eval()
    results = []
    for entry in plan:
        f = transforms["frames"][entry["frame"]]
        img_path = processed / f["file_path"]
        img = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        h, w = img.shape[:2]
        assert (w, h) == (f["w"], f["h"]), f"{img_path}: {w}x{h} != transforms {f['w']}x{f['h']}"
        tensor = torch.from_numpy(img).permute(2, 0, 1).to(device)
        with torch.no_grad():
            pred = model.infer(tensor, num_tokens=args.num_tokens, fov_x=se.fov_x_degrees(f["fx"], f["w"]), apply_mask=False)
        depth = pred["depth"].detach().float().cpu().numpy()
        mask = pred["mask"].detach().cpu().numpy().astype(bool) if "mask" in pred else None
        proj = se.project_frame(f, sparse)
        r = se.frame_scale(depth, mask, proj, window=args.window)
        r.update({"frame": entry["frame"], "file_path": f["file_path"]})
        results.append(r)
        # keep the metric depth (float16, ~3 MB/frame) so sampling variants can be rescored on CPU
        np.savez_compressed(out / f"depth_{entry['frame']:04d}.npz", depth=depth.astype(np.float16),
                            mask=(mask if mask is not None else np.ones_like(depth, dtype=bool)))
        print(f"  frame {entry['frame']:>4} {Path(f['file_path']).name}: n={r['n_points']} inl={r['n_inliers']} ratio={r['ratio']}", flush=True)
        # preview: depth as 8-bit PNG for the receipt (not a metric artifact)
        finite = np.isfinite(depth) & (depth > 0)
        if finite.any():
            lo, hi = np.percentile(depth[finite], [2, 98])
            img8 = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            Image.fromarray((img8 * 255).astype(np.uint8)).save(out / f"depth_{entry['frame']:04d}.png")
    finish(results, MODEL_ID, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
