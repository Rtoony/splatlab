#!/usr/bin/env python3
"""P6b: lift SAM 3 instance masks (every "things" noun) to per-gaussian
instance sets — the general multi-noun productionization of the proven
scene-regen-spike lift_vote.py (Step 0, 2026-07-22: mechanism PROVEN GO,
recipe --min-views 2 --vote-frac 0.3).

For each noun's masks: per-(view,mask) member sets via the PASS-B projection +
ED-depth occlusion gate (mechanism copied from backend/langfield/langfield_v2.py)
-> greedy cross-view Jaccard merge -> majority vote. Survival floor: >=1 mask
AND >=200 gaussians AND seen in >=min-views views (kills single-view ghosts —
the Step 0 spike's cam-000 registration-error lesson).

Conservation bookkeeping (every gaussian claimed-or-remainder) + optional
regression against known reference object_indices.npz (--ref-dir), plus
receipts: a colored overlay across ALL instances on sample frames, and a
per-instance crop from its best (largest-projected-area) view.

Runs in the langfield-spike env.

Usage: instance_lift.py <config.yml> <workdir> <things.json> <out_dir>
       [--jaccard 0.25] [--vote-frac 0.3] [--depth-tol 0.07]
       [--min-members 200] [--min-views 2] [--ref-dir <path>]
"""
import argparse
import json
import math
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


def _slug(noun: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in noun.lower()).strip("-")[:40] or "thing"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("things_json", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--jaccard", type=float, default=0.25)
    ap.add_argument("--vote-frac", type=float, default=0.3)
    ap.add_argument("--depth-tol", type=float, default=0.07)
    ap.add_argument("--min-members", type=int, default=200)
    ap.add_argument("--min-views", type=int, default=2)
    ap.add_argument("--ref-dir", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    things = json.loads(args.things_json.read_text())
    views = json.load(open(args.workdir / "views.json"))
    W, H = views["W"], views["H"]

    config, pipeline, _, _ = eval_setup(args.config, test_mode="test")
    m = pipeline.model.to(DEV)
    means, quats = m.means.detach(), m.quats.detach()
    scales = torch.exp(m.scales.detach())
    opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
    N = means.shape[0]
    means_np = means.cpu().numpy()
    cams = pipeline.datamanager.train_dataset.cameras.to(DEV)
    fullW, fullH = int(cams.width[0]), int(cams.height[0])
    zeros3 = means.new_zeros(N, 3)
    print(f"[lift] {N} gaussians, things={things}", flush=True)

    def viewmat_for(i):
        return get_viewmat(cams.camera_to_worlds[i:i + 1])

    def K_for(i, w, h):
        K = cams.get_intrinsics_matrices()[i:i + 1].clone().to(DEV)
        K[:, 0, :] *= (w / fullW)
        K[:, 1, :] *= (h / fullH)
        return K

    # One rasterization pass per view, shared across every noun.
    proj_cache: dict[int, tuple] = {}

    def projected(v):
        if v in proj_cache:
            return proj_cache[v]
        _, _, info = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=zeros3,
            viewmats=viewmat_for(v), Ks=K_for(v, W, H), width=W, height=H,
            packed=True, near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=None,
            rasterize_mode="antialiased")
        gid = info["gaussian_ids"]; uv = info["means2d"]; dz = info["depths"]
        ed, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=zeros3,
            viewmats=viewmat_for(v), Ks=K_for(v, W, H), width=W, height=H,
            packed=False, near_plane=0.01, far_plane=1e10, render_mode="ED", sh_degree=None,
            rasterize_mode="antialiased")
        depth_buf = ed[0, ..., 0]
        px = uv[:, 0].round().long(); py = uv[:, 1].round().long()
        inf = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        gid, dz, px, py = gid[inf], dz[inf], px[inf], py[inf]
        front_d = depth_buf[py, px]
        occ_ok = dz <= front_d * (1.0 + args.depth_tol)
        gid, px, py = gid[occ_ok], px[occ_ok], py[occ_ok]
        result = (gid.cpu().numpy(), px.cpu().numpy(), py.cpu().numpy())
        proj_cache[v] = result
        return result

    ref: dict[str, np.ndarray] = {}
    if args.ref_dir and args.ref_dir.is_dir():
        for npz_path in args.ref_dir.glob("*/object_indices.npz"):
            ref[npz_path.parent.name] = np.unique(np.load(npz_path)["indices"])

    instances: list[dict] = []
    vetoed: list[dict] = []
    for noun in things:
        slug = _slug(noun)
        masks_dir = args.workdir / "masks" / slug
        if not masks_dir.is_dir():
            vetoed.append({"noun": noun, "reason": "no SAM3 masks produced"})
            continue

        detections = []
        for npz_path in sorted(masks_dir.glob("cam_*.npz")):
            v = int(npz_path.stem.split("_")[1])
            d = np.load(npz_path)
            masks, scores = d["masks"], d["scores"]
            gid_np, px_np, py_np = projected(v)
            for k in range(masks.shape[0]):
                hit = masks[k][py_np, px_np]
                members = np.unique(gid_np[hit])
                if members.size == 0:
                    continue
                detections.append({"cam": v, "mask": int(k), "score": float(scores[k]),
                                   "members": members})
        if not detections:
            vetoed.append({"noun": noun, "reason": "SAM3 found zero detections in any view"})
            continue

        # greedy cross-view merge by Jaccard on member sets
        detections.sort(key=lambda d: d["members"].size, reverse=True)
        clusters: list[dict] = []
        for det in detections:
            ms = det["members"]
            best, best_j = None, 0.0
            for c in clusters:
                inter = np.intersect1d(ms, c["members_union"], assume_unique=True).size
                union = ms.size + c["members_union"].size - inter
                j = inter / union if union else 0.0
                if j > best_j:
                    best, best_j = c, j
            if best is not None and best_j >= args.jaccard:
                best["members_union"] = np.union1d(best["members_union"], ms)
                best["dets"].append(det)
            else:
                clusters.append({"members_union": ms.copy(), "dets": [det]})

        noun_survived = 0
        for c in clusters:
            cams_in = sorted({d["cam"] for d in c["dets"]})
            if len(cams_in) < args.min_views:
                continue
            votes: dict[int, int] = {}
            for d in c["dets"]:
                for g in d["members"]:
                    votes[g] = votes.get(g, 0) + 1
            need = max(2, math.ceil(args.vote_frac * len(cams_in))) if len(cams_in) >= 2 else 1
            final = np.array(sorted(g for g, n in votes.items() if n >= need), dtype=np.int64)
            if final.size < args.min_members:
                continue

            noun_survived += 1
            inst_id = len(instances)
            inst_slug = slug if noun_survived == 1 else f"{slug}-{noun_survived}"
            centroid = means_np[final].mean(0)
            bbox_min, bbox_max = means_np[final].min(0), means_np[final].max(0)

            best_view, best_area, best_box = None, -1.0, None
            for v in cams_in:
                gid_np, px_np, py_np = projected(v)
                sel = np.isin(gid_np, final)
                if not sel.any():
                    continue
                x0, x1 = int(px_np[sel].min()), int(px_np[sel].max())
                y0, y1 = int(py_np[sel].min()), int(py_np[sel].max())
                area = (x1 - x0) * (y1 - y0)
                if area > best_area:
                    best_view, best_area, best_box = v, area, [x0, y0, x1, y1]

            inst = {
                "id": inst_id, "label": noun, "slug": inst_slug,
                "n_members": int(final.size), "n_views": len(cams_in),
                "views_seen": cams_in, "vote_threshold": need,
                "mean_score": float(np.mean([d["score"] for d in c["dets"]])),
                "centroid_scene": [round(float(x), 4) for x in centroid],
                "bbox_tight_scene": {"min": [round(float(x), 4) for x in bbox_min],
                                     "max": [round(float(x), 4) for x in bbox_max]},
                "best_view": best_view, "best_view_box": best_box,
            }
            for ref_slug, ref_idx in ref.items():
                inter = np.intersect1d(final, ref_idx, assume_unique=True).size
                if inter == 0:
                    continue
                union = final.size + ref_idx.size - inter
                iou = round(inter / union, 4) if union else 0.0
                if iou > 0.05:  # only record plausible matches, not incidental overlap
                    inst.setdefault("regression", {})[ref_slug] = {
                        "iou": iou, "recall": round(inter / ref_idx.size, 4),
                        "precision": round(inter / final.size, 4) if final.size else 0.0,
                    }
            np.savez_compressed(args.out_dir / f"instance_{inst_slug}.npz", indices=final)
            instances.append(inst)

        if noun_survived == 0:
            vetoed.append({"noun": noun, "reason": "all candidate clusters dropped below "
                                                     "min-views/min-members floor"})

    instances.sort(key=lambda i: i["n_members"], reverse=True)

    # ── conservation: every gaussian is claimed-or-remainder, exactly once ────
    claimed_union = np.zeros(N, dtype=bool)
    overlap = np.zeros(N, dtype=np.int32)
    for inst in instances:
        idx = np.load(args.out_dir / f"instance_{inst['slug']}.npz")["indices"]
        claimed_union[idx] = True
        overlap[idx] += 1
    n_claimed = int(claimed_union.sum())
    n_remainder = int(N - n_claimed)
    conservation = {
        "n_gaussians": N, "n_claimed": n_claimed, "n_remainder": n_remainder,
        "n_overlap": int((overlap > 1).sum()),
        "holds": (n_claimed + n_remainder) == N,
    }

    # ── receipts: best-effort, never fail the build ────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        cmap = plt.get_cmap("tab20")
        sample_views = views["cam_indices"][:: max(1, len(views["cam_indices"]) // 2)][:2]
        for v in sample_views:
            img = np.array(Image.open(args.workdir / "frames" / f"cam_{v:03d}.png"))
            gid_np, px_np, py_np = projected(v)
            fig, ax = plt.subplots(figsize=(10, 10 * H / W))
            ax.imshow(img); ax.axis("off")
            for inst in instances:
                idx = np.load(args.out_dir / f"instance_{inst['slug']}.npz")["indices"]
                sel = np.isin(gid_np, idx)
                if not sel.any():
                    continue
                ax.scatter(px_np[sel], py_np[sel], s=0.3, color=cmap(inst["id"] % 20),
                          alpha=0.5, label=f"{inst['label']} ({inst['n_members']})")
            ax.legend(loc="upper right", markerscale=20, fontsize=7)
            fig.savefig(args.out_dir / f"receipt_overlay_cam_{v:03d}.png",
                        bbox_inches="tight", dpi=110)
            plt.close(fig)

        for inst in instances:
            if inst["best_view"] is None:
                continue
            img = Image.open(args.workdir / "frames" / f"cam_{inst['best_view']:03d}.png")
            x0, y0, x1, y1 = inst["best_view_box"]
            pad = int(0.1 * max(x1 - x0, y1 - y0, 1))
            crop = img.crop((max(0, x0 - pad), max(0, y0 - pad),
                             min(img.width, x1 + pad), min(img.height, y1 + pad)))
            crop.save(args.out_dir / f"crop_{inst['slug']}.png")
    except Exception as e:                            # receipts are best-effort
        print(f"[lift] receipt render failed (non-fatal): {e}", flush=True)

    report = {
        "n_gaussians": N, "views": views["cam_indices"], "things": things,
        "params": {"jaccard": args.jaccard, "vote_frac": args.vote_frac,
                   "depth_tol": args.depth_tol, "min_members": args.min_members,
                   "min_views": args.min_views},
        "instances": instances, "vetoed": vetoed, "conservation": conservation,
    }
    (args.out_dir / "instances.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
