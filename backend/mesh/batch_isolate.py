#!/usr/bin/env python3
"""P6c: materialize P6b's SAM3-determined instances into Blender-ready
per-instance splats + a background-remainder complement.

P6b already solved membership (SAM3 lift beats the old per-query DBSCAN --
proven on the garden vase, 2026-07-22: production langfield text search grabbed
a garden corner, SAM3 lift fixed it). This stage just claims indices with
first-claim-wins bookkeeping (largest instance first -- instances.json is
already size-sorted by P6b) and materializes raw/pre-activation PLYs, reusing
the write_splat_ply convention from object_isolate.py / the proven
langfield-isolate-probe/isolate_export.py (same "sanity_sum_ok" accounting:
claimed + background == total, exactly once each).

Runs in the langfield-spike env.

Usage: batch_isolate.py <config.yml> <scene_dir> <out_dir>
       [--min-members 200] [--views 2]
"""
import argparse
import json
import sys
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

DEV = "cuda"
PLY_FIELDS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
SH_C0 = 0.28209479177387814  # SH0 -> diffuse RGB, standard 3DGS constant


def write_splat_ply(path: Path, xyz, f_dc, opacity, scale, rot) -> None:
    """Raw/pre-activation fields -- the format the Blender importer round-trips
    (numerics mirrored from object_isolate.py / langfield-isolate-probe)."""
    m = xyz.shape[0]
    rows = np.concatenate([
        xyz.astype("<f4"), f_dc.astype("<f4"), opacity.astype("<f4")[:, None],
        scale.astype("<f4"), rot.astype("<f4"),
    ], axis=1)
    header = "ply\nformat binary_little_endian 1.0\n"
    header += "comment SplatLab batch isolate (P6c)\n"
    header += f"element vertex {m}\n"
    for f in PLY_FIELDS:
        header += f"property float {f}\n"
    header += "end_header\n"
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rows.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--min-members", type=int, default=200)
    ap.add_argument("--views", type=int, default=2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inventory = json.loads((args.scene_dir / "inventory.json").read_text())
    instances = inventory.get("instances", [])  # already size-sorted desc by P6b

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    m = pipeline.model.to(DEV)
    means = m.means.detach()
    N = means.shape[0]
    means_np = means.cpu().numpy()
    fdc_np = m.features_dc.detach().cpu().numpy()
    opac_np = m.opacities.detach().cpu().numpy().squeeze(-1)
    scale_np = m.scales.detach().cpu().numpy()
    rot_np = m.quats.detach().cpu().numpy()

    claimed = np.zeros(N, dtype=bool)
    results = []
    for inst in instances:
        slug = inst["slug"]
        npz_path = args.scene_dir / f"instance_{slug}.npz"
        if not npz_path.is_file():
            results.append({"slug": slug, "label": inst["label"],
                            "status": "SKIPPED:indices-file-missing"})
            continue
        idx = np.load(npz_path)["indices"]
        dedup = idx[~claimed[idx]]  # first-claim-wins: larger instances went first
        n_overlap_removed = int(idx.size - dedup.size)
        if dedup.size < args.min_members:
            results.append({
                "slug": slug, "label": inst["label"],
                "n_members_original": int(idx.size), "n_overlap_removed": n_overlap_removed,
                "status": "SKIPPED:too-few-members-after-dedup",
            })
            continue
        claimed[dedup] = True
        inst_dir = args.out_dir / slug
        inst_dir.mkdir(parents=True, exist_ok=True)
        write_splat_ply(inst_dir / "object.ply", means_np[dedup], fdc_np[dedup],
                        opac_np[dedup], scale_np[dedup], rot_np[dedup])
        np.savez_compressed(inst_dir / "object_indices.npz", indices=dedup)
        results.append({
            "slug": slug, "label": inst["label"],
            "n_members_original": int(idx.size), "n_overlap_removed": n_overlap_removed,
            "n_members_final": int(dedup.size), "status": "built",
        })

    background_mask = ~claimed
    n_claimed, n_background = int(claimed.sum()), int(background_mask.sum())
    write_splat_ply(args.out_dir / "background.ply", means_np[background_mask],
                    fdc_np[background_mask], opac_np[background_mask],
                    scale_np[background_mask], rot_np[background_mask])

    sanity = {"n_gaussians": N, "n_claimed": n_claimed, "n_background": n_background,
              "sanity_sum_ok": bool(n_claimed + n_background == N)}

    # ── receipt: a REAL gsplat RGB render with claimed gaussians' opacity
    # zeroed -- genuinely shows what the scene looks like with every built
    # instance removed, not just a photo with dots on it. Best-effort. ───────
    try:
        cams = pipeline.datamanager.train_dataset.cameras.to(DEV)
        fullW, fullH = int(cams.width[0]), int(cams.height[0])
        n_train = len(pipeline.datamanager.train_dataset)
        sample_views = sorted(set(
            np.linspace(0, n_train - 1, args.views).round().astype(int).tolist()))
        quats = m.quats.detach()
        scales = torch.exp(m.scales.detach())
        opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
        colors = (SH_C0 * m.features_dc.detach() + 0.5).clamp(0, 1)
        bg_opac = opac.clone()
        bg_opac[torch.from_numpy(claimed).to(DEV)] = 0.0

        def viewmat_for(i):
            return get_viewmat(cams.camera_to_worlds[i:i + 1])

        def K_for(i, w, h):
            K = cams.get_intrinsics_matrices()[i:i + 1].clone().to(DEV)
            K[:, 0, :] *= (w / fullW)
            K[:, 1, :] *= (h / fullH)
            return K

        from PIL import Image
        W, H = min(fullW, 960), min(fullH, 640)
        for v in sample_views:
            rgb, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=bg_opac, colors=colors,
                viewmats=viewmat_for(v), Ks=K_for(v, W, H), width=W, height=H,
                packed=False, near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=None,
                rasterize_mode="antialiased")
            img = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(args.out_dir / f"receipt_background_cam_{v:03d}.png")
    except Exception as e:                            # receipt is best-effort
        print(f"[batch-isolate] receipt render failed (non-fatal): {e}", flush=True)

    report = {"n_gaussians": N, "instances": results, "sanity": sanity}
    (args.out_dir / "batch_isolate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
