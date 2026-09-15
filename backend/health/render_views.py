#!/usr/bin/env python3
"""Render RGB views from a splatfacto checkpoint for the SCODA quality gate.

Mirrors fog_gate.py's loading + gsplat rasterization. Renders (a) the held-out
eval cameras — true novel views with a real photo to compare against — and (b)
perturbed copies of them (yaw ±deg, sideways shift) that stand in for walker
viewpoints no photo covers. Writes PNGs + cams.json into <out_dir>.
Run through backend/health/run_render_views.sh (compute-gated, langfield env).
"""
from __future__ import annotations

import argparse
import json
import math
import time
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
from PIL import Image                                         # noqa: E402

DEV = "cuda"


def _yaw(c2w: torch.Tensor, deg: float) -> torch.Tensor:
    """Rotate the camera about the world Z (up) axis around its own position."""
    a = math.radians(deg)
    rz = torch.tensor([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]],
                      dtype=c2w.dtype, device=c2w.device)
    out = c2w.clone()
    out[:3, :3] = rz @ c2w[:3, :3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cams", type=int, default=8, help="eval cameras to render")
    ap.add_argument("--max-width", type=int, default=640)
    ap.add_argument("--yaw-deg", type=float, default=12.0)
    ap.add_argument("--shift", type=float, default=0.15, help="sideways shift in scene units")
    ap.add_argument("--variants", choices=["all", "eval"], default="all", help="eval = held-out poses only, no perturbed copies")
    ap.add_argument("--rasterize-mode", choices=["auto", "classic", "antialiased"], default="auto",
                    help="auto = the mode the checkpoint was trained with (faithful render)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    m = pipeline.model.to(DEV)
    means, quats = m.means.detach(), m.quats.detach()
    scales = torch.exp(m.scales.detach())
    opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
    sh = torch.cat([m.features_dc.detach()[:, None, :], m.features_rest.detach()], dim=1)
    sh_degree = int(getattr(m.config, "sh_degree", 3))
    trained_mode = str(getattr(m.config, "rasterize_mode", "classic"))
    mode = trained_mode if args.rasterize_mode == "auto" else args.rasterize_mode
    dm = pipeline.datamanager
    dataset = getattr(dm, "eval_dataset", None) or dm.train_dataset
    split = "eval" if getattr(dm, "eval_dataset", None) is not None else "train"
    cams = dataset.cameras.to(DEV)
    n = int(cams.camera_to_worlds.shape[0])
    k = max(1, min(args.cams, n))
    ids = sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)})
    names = list(getattr(dataset._dataparser_outputs, "image_filenames", []))
    print(f"[render-views] {means.shape[0]} gaussians, {split} split {n} cams, rendering {ids}, rasterize_mode={mode} (trained {trained_mode})", flush=True)

    def K_for(i, w, h):
        K = cams.get_intrinsics_matrices()[i:i + 1].clone().to(DEV)
        K[:, 0, :] *= (w / float(cams.width[i]))
        K[:, 1, :] *= (h / float(cams.height[i]))
        return K

    def render_rgb(c2w, i, w, h):
        out, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=sh,
            viewmats=get_viewmat(c2w[None]), Ks=K_for(i, w, h), width=w, height=h, packed=False,
            near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=sh_degree,
            rasterize_mode=mode, backgrounds=torch.ones(1, 3, device=DEV))
        return out[0, ..., :3].clamp(0, 1)

    rows = []
    for ci in ids:
        full_w, full_h = int(cams.width[ci]), int(cams.height[ci])
        w = min(args.max_width, full_w)
        h = max(1, round(full_h * w / full_w))
        base = cams.camera_to_worlds[ci]
        variants = [("eval", base)]
        if args.variants == "all":
            for sgn in (+1, -1):
                c = _yaw(base, sgn * args.yaw_deg)
                c[:3, 3] = c[:3, 3] + sgn * args.shift * base[:3, 0]   # shift along camera right
                variants.append((f"yaw{sgn * args.yaw_deg:+.0f}", c))
        K = K_for(ci, w, h)[0].cpu().numpy()
        for tag, c2w in variants:
            img = (render_rgb(c2w, ci, w, h).cpu().numpy() * 255).astype(np.uint8)
            fn = f"cam{ci:04d}_{tag}.png"
            Image.fromarray(img).save(args.out_dir / fn)
            rows.append({"file": fn, "cam": ci, "variant": tag, "novel": tag != "eval",
                         "photo": str(names[ci]) if ci < len(names) else None, "w": w, "h": h,
                         "full_w": full_w, "full_h": full_h,
                         # camera-to-world (nerfstudio/OpenGL axes, normalised frame — same frame as the exported PLY)
                         "c2w": c2w[:3, :4].cpu().numpy().tolist(),
                         "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])})
    (args.out_dir / "cams.json").write_text(json.dumps({"split": split, "n_cams": n, "rendered": rows,
                                                        "rasterize_mode": mode, "trained_rasterize_mode": trained_mode,
                                                        "runtime_s": round(time.time() - t0, 1)}, indent=1))
    print(f"[render-views] wrote {len(rows)} PNGs to {args.out_dir} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
