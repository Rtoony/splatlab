"""Fog-fingerprint gate (langfield-spike env): is this reconstruction fog?

Renders expected depth (gsplat render_mode="ED") at a handful of training
cameras and applies the 2026-07-10 fingerprint: a fog-collapsed scene shows
depth as a thin uniform shell around the camera path — per-camera
spread = p95(depth)/p5(depth) < 3, median depth ~0.01-0.02 scene units, and
accumulation ~1.0 everywhere. Healthy scenes spread >= 4.

REPORT-ONLY by doctrine: this script analyzes and writes receipts; it never
gates anything by itself. Exit 0 = analysis completed (ANY verdict, including
FOG). Non-zero exit = execution failure only (bad checkpoint, OOM, ...).

Outputs into <out_dir>:
  fog.json                                   verdict + per-camera numbers
  fog_cam<idx>_spread<..>_p50-<..>.webp      side-by-side [RGB | turbo log-depth]
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup            # noqa: E402
from nerfstudio.models.splatfacto import get_viewmat          # noqa: E402
from gsplat import rasterization                              # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
from matplotlib import cm                                     # noqa: E402
from PIL import Image, ImageDraw                              # noqa: E402

DEV = "cuda"

# Fingerprint thresholds (env-overridable). Calibrated 2026-07-11 against the
# labeled scenes: the original p95/p5 spread breaks on MIXED scenes (a fog
# cocoon with a few punch-through pixels inflates p95 — the selfie scene read
# spread 45 at one camera while every camera had p5 pinned at the near plane).
# The robust per-camera signal is SHELL FRACTION: how much of the view sits
# within a few near-plane widths of the camera. Fog cocoon = most of the view
# in the shell; clean scene = essentially none of it.
SHELL_D = float(os.environ.get("HEALTH_FOG_SHELL_D", "0.03"))          # 3x near_plane
SHELL_FRAC_FOG = float(os.environ.get("HEALTH_FOG_SHELL_FRAC_FOG", "0.5"))
SHELL_FRAC_CLEAN = float(os.environ.get("HEALTH_FOG_SHELL_FRAC_CLEAN", "0.05"))
P50_CLEAN = float(os.environ.get("HEALTH_FOG_P50_CLEAN", "0.1"))       # median at real scene depth
ACC_MIN = float(os.environ.get("HEALTH_FOG_ACC_MIN", "0.98"))
CAM_FRAC = float(os.environ.get("HEALTH_FOG_CAM_FRAC", "0.66"))  # 2/3 majority (4/6 must pass)
MIN_VALID_PX = 500  # below this a camera sees almost nothing — don't let it vote


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path, help="nerfstudio checkpoint config.yml")
    ap.add_argument("out_dir", type=Path, help="artifact dir (e.g. <job_dir>/_health)")
    ap.add_argument("--cams", type=int, default=6)
    ap.add_argument("--max-width", type=int, default=640)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    m = pipeline.model.to(DEV)
    means, quats = m.means.detach(), m.quats.detach()
    scales = torch.exp(m.scales.detach())
    opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
    sh = torch.cat([m.features_dc.detach()[:, None, :], m.features_rest.detach()], dim=1)
    cams = pipeline.datamanager.train_dataset.cameras.to(DEV)
    n_cams = int(cams.camera_to_worlds.shape[0])
    k = max(1, min(args.cams, n_cams))
    cam_ids = sorted({round(i * (n_cams - 1) / max(1, k - 1)) for i in range(k)})
    print(f"[fog-gate] {means.shape[0]} gaussians, {n_cams} cameras, probing {cam_ids}", flush=True)

    zeros3 = means.new_zeros(means.shape[0], 3)
    white = torch.ones(3, device=DEV)
    turbo = cm.get_cmap("turbo")

    def viewmat_for(i):
        return get_viewmat(cams.camera_to_worlds[i:i + 1])

    def K_for(i, w, h):
        K = cams.get_intrinsics_matrices()[i:i + 1].clone().to(DEV)
        K[:, 0, :] *= (w / float(cams.width[i]))
        K[:, 1, :] *= (h / float(cams.height[i]))
        return K

    def render(colors, sh_degree, mode, i, w, h):
        out, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=viewmat_for(i), Ks=K_for(i, w, h), width=w, height=h, packed=False,
            near_plane=0.01, far_plane=1e10, render_mode=mode, sh_degree=sh_degree,
            rasterize_mode="antialiased")
        return out, alpha

    cam_rows, receipts = [], []
    for ci in cam_ids:
        full_w, full_h = int(cams.width[ci]), int(cams.height[ci])
        w = min(args.max_width, full_w)
        h = max(1, round(full_h * w / full_w))

        ed, alpha = render(zeros3, None, "ED", ci, w, h)
        depth = ed[0, ..., 0]
        a = alpha[0, ..., 0]
        valid = a > 0.5
        n_valid = int(valid.sum())
        acc_mean = float(a.mean())

        row = {"cam": int(ci), "valid_px": n_valid, "acc_mean": round(acc_mean, 4)}
        if n_valid < MIN_VALID_PX:
            row.update({"counted": False, "note": "too few opaque pixels to measure"})
            p5 = p50 = p95 = spread = shell_frac = None
        else:
            d = depth[valid]
            q = torch.quantile(d, torch.tensor([0.05, 0.5, 0.95], device=DEV))
            p5, p50, p95 = (float(v) for v in q)
            spread = p95 / (p5 + 1e-9)
            shell_frac = float((d <= SHELL_D).float().mean())
            row.update({
                "counted": True,
                "p5": round(p5, 5), "p50": round(p50, 5), "p95": round(p95, 5),
                "spread": round(spread, 3),
                "shell_frac": round(shell_frac, 4),
                "fog": bool(shell_frac >= SHELL_FRAC_FOG and acc_mean >= ACC_MIN),
                "healthy": bool(shell_frac <= SHELL_FRAC_CLEAN and p50 >= P50_CLEAN),
            })
        cam_rows.append(row)

        # Receipt: side-by-side [RGB render | turbo log-depth]. A fog scene reads
        # as a mushy RGB smear next to a flat monochrome depth blob — judgeable
        # at a glance, which is what the metric-trust doctrine demands.
        rgb_out, rgb_alpha = render(sh, 3, "RGB", ci, w, h)
        rgb = (rgb_out[0, ..., :3] + (1 - rgb_alpha[0]) * white).clamp(0, 1).cpu().numpy()
        dep = depth.clamp_min(1e-6).log().cpu().numpy()
        if p5 is not None and p95 is not None and p95 > p5:
            lo, hi = np.log(p5 + 1e-9), np.log(p95 * 1.2 + 1e-9)
        else:
            lo, hi = dep.min(), dep.max()
        dn = np.clip((dep - lo) / max(hi - lo, 1e-6), 0, 1)
        heat = turbo(dn)[..., :3]
        heat[~valid.cpu().numpy()] = 0.15
        strip = np.concatenate([rgb, heat], axis=1)
        img = Image.fromarray((strip * 255).astype(np.uint8))
        label = (f"cam {ci} | shell {shell_frac:.0%} | spread {spread:.2f} | p50 {p50:.4f} | acc {acc_mean:.3f}"
                 if spread is not None else f"cam {ci} | too few opaque pixels")
        ImageDraw.Draw(img).text((6, 4), label, fill=(255, 255, 0))
        spread_tag = f"{spread:.1f}" if spread is not None else "na"
        p50_tag = f"{p50:.3f}" if p50 is not None else "na"
        name = f"fog_cam{ci:03d}_spread{spread_tag}_p50-{p50_tag}.webp"
        img.save(args.out_dir / name)
        receipts.append(name)
        print(f"[fog-gate] {label}", flush=True)

    counted = [r for r in cam_rows if r.get("counted")]
    n_fog = sum(1 for r in counted if r["fog"])
    n_healthy = sum(1 for r in counted if r["healthy"])
    if not counted:
        verdict = "UNCERTAIN"
    elif n_fog / len(counted) >= CAM_FRAC:
        verdict = "FOG"
    elif n_healthy / len(counted) >= CAM_FRAC:
        verdict = "HEALTHY"
    else:
        verdict = "UNCERTAIN"

    med = lambda key: (round(float(np.median([r[key] for r in counted])), 5) if counted else None)  # noqa: E731
    result = {
        "v": 1,
        "verdict": verdict,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(time.time() - t0, 1),
        "config": str(args.config),
        "n_gaussians": int(means.shape[0]),
        "cameras": cam_rows,
        "summary": {
            "n_cams": len(cam_rows), "n_counted": len(counted),
            "n_fog": n_fog, "n_healthy": n_healthy,
            "median_shell_frac": med("shell_frac"),
            "median_spread": med("spread"), "median_p50": med("p50"),
            "median_acc": med("acc_mean"),
        },
        "thresholds": {
            "shell_d": SHELL_D, "shell_frac_fog": SHELL_FRAC_FOG,
            "shell_frac_clean": SHELL_FRAC_CLEAN, "p50_clean": P50_CLEAN,
            "acc_min": ACC_MIN, "cam_frac": CAM_FRAC,
        },
        "receipts": receipts,
    }
    (args.out_dir / "fog.json").write_text(json.dumps(result, indent=2))
    print(f"[fog-gate] VERDICT: {verdict} ({n_fog}/{len(counted)} cams fog, "
          f"median spread {result['summary']['median_spread']}) in {result['runtime_s']}s", flush=True)


if __name__ == "__main__":
    main()
