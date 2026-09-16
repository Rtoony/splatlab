#!/usr/bin/env python3
"""MoGe-2 metric depth for a structure study's views (any building), scaled into the capture's SfM
frame through that view's own sparse tracks (sam3d-body env, GPU via the compute gate).

For every view in <structure>/receipt.json: MoGe-2 depth (metres, known FoV) ->
sample it at the pixels of the tracks observed in that view -> per-view scale
s_v = median(track depth [u] / MoGe depth [m]). Writes <output>/cam_%03d.npz
{depth_m, mask, scale_u_per_m} and <output>/receipt.json with every per-view
scale, their spread and the implied metres-per-unit — an independent scale
estimate for the SfM frame (the Condo Lab alignment's scale came from the
modelled garage opening, not from a measurement).

  moge-depth-views.py --structure <capture-structure dir> --output <dir>
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests          # noqa: E402
import scale_estimate as se                    # noqa: E402

MODEL_ID = "Ruicheng/moge-2-vitl-normal"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-tokens", type=int, default=1800); ap.add_argument("--min-tracks", type=int, default=20)
    a = ap.parse_args(); t0 = time.time()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    a.output.mkdir(parents=True)
    rec = json.loads((a.structure / "receipt.json").read_text()); intr = rec["intrinsics"]
    tr = np.load(a.structure / "tracks.npz"); points, observed, pixels = tr["points"], tr["observed"], tr["pixels"]
    import torch
    from moge.model.v2 import MoGeModel
    device = torch.device("cuda")
    model = MoGeModel.from_pretrained(MODEL_ID).to(device).eval()
    fov_x = se.fov_x_degrees(intr["fl_x"], intr["w"])
    views = []
    for view in rec["views"]:
        o = view["ordinal"]
        img = np.asarray(Image.open(a.structure / view["photo"]).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.tensor(img, device=device).permute(2, 0, 1)
        with torch.no_grad():
            pred = model.infer(tensor, num_tokens=a.num_tokens, fov_x=fov_x, apply_mask=False)
        depth = pred["depth"].float().cpu().numpy(); mask = pred["mask"].cpu().numpy().astype(bool)
        c2w = np.asarray(view["transform_matrix"], dtype=np.float64)
        obs = observed[o]; pts = points[obs]; px = pixels[o][obs]
        local = (pts - c2w[:3, 3]) @ c2w[:3, :3]; track_depth = -local[:, 2]
        u, v = np.rint(px[:, 0]).astype(int), np.rint(px[:, 1]).astype(int)
        inside = (u >= 1) & (v >= 1) & (u < intr["w"] - 1) & (v < intr["h"] - 1) & (track_depth > 0)
        md = se.sample_depth(depth, u[inside], v[inside], window=1) if inside.any() else np.zeros(0)
        valid = np.isfinite(md) & (md > 0) & mask[v[inside], u[inside]]
        ratio = track_depth[inside][valid] / md[valid]
        if len(ratio) >= a.min_tracks:
            s = float(np.median(ratio)); mad = float(np.median(np.abs(ratio - s)) * 1.4826)
            core = np.abs(ratio - s) <= 2.5 * mad + 1e-9; s = float(np.median(ratio[core])); spread = float(np.std(ratio[core]) / s)
        else:
            s, mad, spread, core = None, None, None, np.zeros(0, dtype=bool)
        np.savez_compressed(a.output / f"cam_{o:03d}.npz", depth_m=depth.astype(np.float32), mask=mask, scale_u_per_m=np.float64(s if s else np.nan))
        views.append({"ordinal": o, "file_path": view["file_path"], "tracks_observed": int(obs.sum()), "tracks_used": int(core.sum()) if s else 0,
                      "scale_u_per_m": s, "mad_u_per_m": mad, "relative_spread": spread, "moge_depth_median_m": float(np.median(depth[mask])) if mask.any() else None})
        print(f"[moge] cam {o:03d}: {int(obs.sum())} tracks, scale {s if s is None else round(s, 4)} u/m (spread {spread if spread is None else round(spread, 3)})", flush=True)
    scales = np.array([v["scale_u_per_m"] for v in views if v["scale_u_per_m"]])
    global_scale = float(np.median(scales)) if len(scales) else None
    receipt = {"schema": "dev.splatlab.moge-depth-views/v1", "model": MODEL_ID, "fov_x_deg": fov_x, "num_tokens": a.num_tokens,
               "structure": str(a.structure), "structure_receipt_sha256": manifests.sha256_file(a.structure / "receipt.json"),
               "views": views, "global_scale_u_per_m": global_scale, "global_meters_per_unit": (1.0 / global_scale) if global_scale else None,
               "per_view_scale_mad_u_per_m": float(np.median(np.abs(scales - global_scale)) * 1.4826) if len(scales) else None,
               "n_views_scaled": int(len(scales)), "elapsed_seconds": round(time.time() - t0, 1), "status": "inferred-needs-review",
               "scope": "MoGe-2 monocular metric depth per view; scale per view from that view's own SfM tracks. Independent of the Condo Lab garage-opening scale, but not a survey."}
    manifests.atomic_write_json(a.output / "receipt.json", receipt)
    print(f"[moge] global scale {global_scale} u/m -> {receipt['global_meters_per_unit']} m/u from {len(scales)} views in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
