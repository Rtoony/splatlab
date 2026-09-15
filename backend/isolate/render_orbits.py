#!/usr/bin/env python3
"""Step 2 (langfield env): pick the best SAM3 grounding, lift the mask to a seed
gaussian set, synthesise a visibility-filtered virtual orbit around it and render
the frames the tracker will follow. Writes <work>/seed.npz, <work>/track/frames/<i>.jpg,
<work>/track/cams.json, <work>/track/seed_mask.png, <work>/orbit_receipt.json."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _ckpt import DEV, Splat, typical_spacing  # noqa: E402
from isolate import orbits as ob                # noqa: E402

MIN_MASK_FRAC = ob.MIN_MASK_FRAC


def pick_reference(ref_dir: Path, slug: str) -> dict:
    views = json.loads((ref_dir / "cams.json").read_text())["views"]
    def load(cam: int):
        npz = ref_dir / "masks" / slug / f"cam_{cam:03d}.npz"
        return np.load(npz) if npz.is_file() else None
    best = ob.pick_reference(views, load, slug)
    if best is None:
        raise SystemExit(f"no usable SAM3 detection for {slug!r} in {ref_dir / 'masks' / slug}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path); ap.add_argument("work", type=Path); ap.add_argument("--slug", required=True)
    ap.add_argument("--azimuths", default="-60,-45,-30,-15,15,30,45,60"); ap.add_argument("--elevations", default="-10,0,15,30")
    ap.add_argument("--distance-scales", default="1.0,0.7")
    ap.add_argument("--min-visible", type=float, default=0.3); ap.add_argument("--min-views", type=int, default=8)
    ap.add_argument("--max-views", type=int, default=16); ap.add_argument("--voxel-mult", type=float, default=2.0)
    ap.add_argument("--depth-tol-frac", type=float, default=0.03)
    a = ap.parse_args(); t0 = time.time()
    s = Splat(a.config)
    best = pick_reference(a.work / "ref", a.slug)
    row, mask = best["row"], best["mask"]
    w, h = int(row["w"]), int(row["h"])
    c2w = torch.tensor(row["c2w"], dtype=torch.float32, device=DEV)
    K = s.K_from(row["fx"], row["fy"], row["cx"], row["cy"])
    # depth at the reference camera -> lift the mask -> seed gaussians
    out, alpha, _ = s.render(c2w, K, w, h, mode="RGB+ED")
    depth = out[..., 3].cpu().numpy(); depth[alpha.cpu().numpy() < 0.5] = np.nan
    pts = ob.backproject(mask, depth, row["c2w"], row["fx"], row["fy"], row["cx"], row["cy"])
    spacing = typical_spacing(s.means)
    means_np = s.means.cpu().numpy()
    seed = ob.voxel_select(means_np, pts, voxel=a.voxel_mult * spacing, dilate=1)
    # keep only seed gaussians that project inside the (dilated) mask
    u, v, _ = ob.project(means_np[seed], row["c2w"], row["fx"], row["fy"], row["cx"], row["cy"], w, h)
    ok = ~np.isnan(u)
    from scipy.ndimage import binary_dilation
    dil = binary_dilation(mask, iterations=3)
    inside = np.zeros(len(seed), bool)
    inside[ok] = dil[v[ok].astype(int), u[ok].astype(int)]
    seed = seed[inside]
    stats = ob.seed_stats(means_np, seed)
    if stats["n"] < 50:
        raise SystemExit(f"seed too small ({stats['n']} gaussians) — grounding or depth lift failed")
    centroid = np.array(stats["centroid"])
    # virtual orbit, visibility-filtered
    az = [float(x) for x in a.azimuths.split(",")]; el = [float(x) for x in a.elevations.split(",")]
    ds = [float(x) for x in a.distance_scales.split(",")]
    cams = ob.orbit_cameras(np.asarray(row["c2w"]), centroid, az, el, distance_scales=ds)
    seed_opac = torch.zeros_like(s.opac); seed_opac[torch.as_tensor(seed, device=DEV)] = s.opac[torch.as_tensor(seed, device=DEV)]
    track = a.work / "track"; (track / "frames").mkdir(parents=True, exist_ok=True)
    for old in (track / "frames").glob("*.jpg"):
        old.unlink()
    scene_depth_scale = float(np.nanmedian(depth[mask])) if np.isfinite(depth[mask]).any() else 1.0
    for cam in cams:                      # score every candidate first (depth-only renders, cheap)
        c = torch.tensor(cam["c2w"], dtype=torch.float32, device=DEV)
        full, _, _ = s.render(c, K, w, h, mode="RGB+ED")
        seed_out, seed_alpha, _ = s.render(c, K, w, h, mode="RGB+ED", opac=seed_opac)
        vis = ob.visible_fraction(seed_alpha.cpu().numpy(), seed_out[..., 3].cpu().numpy(), full[..., 3].cpu().numpy(),
                                  tol=a.depth_tol_frac * scene_depth_scale)
        cover = float((seed_alpha > 0.5).float().mean())
        cam.update({"visible_fraction": round(vis, 4), "seed_cover": round(cover, 5)})
        if cover < MIN_MASK_FRAC:
            cam["visible_fraction"] = 0.0
    kept = ob.path_order(ob.select_views(cams, a.min_visible, a.min_views, a.max_views))   # tracker needs a smooth path
    for cam in cams:
        cam["kept"] = cam in kept
    (track / "seed").mkdir(exist_ok=True)
    for i, cam in enumerate(kept):        # render RGB + the seed's own silhouette for the chosen views
        cam["frame"] = i
        c = torch.tensor(cam["c2w"], dtype=torch.float32, device=DEV)
        full, _, _ = s.render(c, K, w, h, mode="RGB")
        Image.fromarray((full[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(track / "frames" / f"{i}.jpg", quality=95)
        _, seed_alpha, _ = s.render(c, K, w, h, mode="RGB", opac=seed_opac)
        sil = (seed_alpha > 0.5).cpu().numpy()
        Image.fromarray((sil * 255).astype(np.uint8)).save(track / "seed" / f"{i}.png")
        bb = ob.mask_bbox(sil)
        cam["seed_bbox"] = bb                                  # pixel [x0, y0, x1, y1] of the seed silhouette in this view
    Image.fromarray((mask * 255).astype(np.uint8)).save(track / "seed_mask.png")
    np.savez_compressed(a.work / "seed.npz", indices=seed.astype(np.int64), centroid=centroid, radius=stats["radius"],
                        voxel=a.voxel_mult * spacing, spacing=spacing)
    (track / "cams.json").write_text(json.dumps({"w": w, "h": h, "fx": row["fx"], "fy": row["fy"], "cx": row["cx"], "cy": row["cy"],
                                                 "frames": kept, "all": cams}, indent=1))
    receipt = {"schema": "dev.splatlab.isolate-by-reference.orbit/v1", "slug": a.slug, "reference": {"cam": row["cam"], "photo": row["photo"], "instance": best["instance"], "score": best["score"], "quality": best.get("quality"), "border_fraction": ob.border_fraction(mask), "mask_frac": best["frac"]},
               "seed": {**stats, "spacing": spacing, "voxel": a.voxel_mult * spacing, "lifted_points": int(len(pts))},
               "orbit": {"candidates": len(cams), "kept": len(kept), "min_visible": a.min_visible, "min_views": a.min_views,
                         "kept_views": [{"tag": c["tag"], "visible": c["visible_fraction"]} for c in kept],
                         "dropped": len(cams) - len(kept)}, "runtime_s": round(time.time() - t0, 1)}
    (a.work / "orbit_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"[orbits] ref cam {row['cam']} score {best['score']:.2f} mask {best['frac']:.3f} -> seed {stats['n']} gaussians; "
          f"orbit {len(kept)}/{len(cams)} views kept in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
