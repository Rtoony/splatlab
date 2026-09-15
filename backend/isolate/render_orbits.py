#!/usr/bin/env python3
"""Step 2 (langfield env): pick the best SAM3 grounding, lift the mask to a seed
gaussian set, synthesise a visibility-filtered virtual orbit around it and render
the frames the tracker will follow.

Stages (--stage):
  full   photo reference -> seed -> orbit in one pass (no re-grounding)
  seed   photo reference -> <work>/seed_initial.npz + pulled-back renders that frame
         the WHOLE seed under <work>/reground/ (frames/cam_%03d.png, seed/cam_%03d.png,
         views.json, cams.json) for a second SAM3 grounding — the photo may only
         show part of the object (a rear wheel), a pulled-back render shows all of it
  orbit  if <work>/reground/masks/<slug>/ exists: pick the instance that CONTAINS the
         seed silhouette, re-lift the seed from that render's depth and orbit from
         that camera; otherwise orbit from the initial (photo) seed
Writes <work>/seed.npz, <work>/track/{frames,seed}/, track/cams.json, track/seed_mask.png,
<work>/orbit_receipt.json."""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _ckpt import DEV, Splat, typical_spacing  # noqa: E402
from isolate import orbits as ob                # noqa: E402

MIN_MASK_FRAC = ob.MIN_MASK_FRAC
MIN_SEED = 50
MIN_REGROUND_OVERLAP = 0.4     # a re-grounded seed must keep this fraction of the initial seed (same object)
DEPTH_BAND = 0.3               # lift only mask pixels within +-30% of the mask's median depth


def pick_reference(ref_dir: Path, slug: str) -> dict:
    views = json.loads((ref_dir / "cams.json").read_text())["views"]
    def load(cam: int):
        npz = ref_dir / "masks" / slug / f"cam_{cam:03d}.npz"
        return np.load(npz) if npz.is_file() else None
    best = ob.pick_reference(views, load, slug)
    if best is None:
        raise SystemExit(f"no usable SAM3 detection for {slug!r} in {ref_dir / 'masks' / slug}")
    return best


def _c2w(row_or_list) -> torch.Tensor:
    return torch.tensor(row_or_list, dtype=torch.float32, device=DEV)


def depth_at(s: Splat, c2w, K, w: int, h: int) -> np.ndarray:
    out, alpha, _ = s.render(_c2w(c2w), K, w, h, mode="RGB+ED")
    depth = out[..., 3].cpu().numpy(); depth[alpha.cpu().numpy() < 0.5] = np.nan
    return depth


def lift_seed(means_np: np.ndarray, row: dict, mask: np.ndarray, depth: np.ndarray, voxel: float, w: int, h: int):
    """Mask + depth -> points -> voxel-selected gaussians, kept only where they project inside the dilated mask."""
    banded = ob.depth_band_mask(mask, depth, rel=DEPTH_BAND)      # see-through masks: keep the object's depth, not the wall behind
    pts = ob.backproject(banded, depth, row["c2w"], row["fx"], row["fy"], row["cx"], row["cy"])
    seed = ob.voxel_select(means_np, pts, voxel=voxel, dilate=1)
    u, v, _ = ob.project(means_np[seed], row["c2w"], row["fx"], row["fy"], row["cx"], row["cy"], w, h)
    ok = ~np.isnan(u); dil = ob.dilate(mask, 3)
    inside = np.zeros(len(seed), bool); inside[ok] = dil[v[ok].astype(int), u[ok].astype(int)]
    return seed[inside], pts


def seed_opacity(s: Splat, seed: np.ndarray) -> torch.Tensor:
    idx = torch.as_tensor(seed, device=DEV)
    opac = torch.zeros_like(s.opac); opac[idx] = s.opac[idx]
    return opac


def rgb_u8(s: Splat, c2w, K, w: int, h: int) -> np.ndarray:
    full, _, _ = s.render(_c2w(c2w), K, w, h, mode="RGB")
    return (full[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def silhouette(s: Splat, c2w, K, w: int, h: int, seed_opac: torch.Tensor) -> np.ndarray:
    _, alpha, _ = s.render(_c2w(c2w), K, w, h, mode="RGB", opac=seed_opac)
    return (alpha > 0.5).cpu().numpy()


def save_seed(path: Path, seed: np.ndarray, stats: dict, voxel: float, spacing: float, extra: dict | None = None) -> None:
    np.savez_compressed(path, indices=seed.astype(np.int64), centroid=np.array(stats["centroid"]), radius=stats["radius"],
                        voxel=voxel, spacing=spacing, **(extra or {}))


def stage_seed(s: Splat, a, row: dict, mask: np.ndarray, stats: dict, seed_opac: torch.Tensor, K, w: int, h: int) -> dict:
    """Pulled-back candidate renders around the seed for the second grounding. The
    framing radius comes from the photo mask's extent (what SAM3 saw) and the seed's
    MEDIAN spread, never its 95th percentile: before the depth band, wall gaussians
    lifted through the bonsai's leaves put p95 at 2.65 units for a 1.2-unit
    reference distance -> a 13-unit pull-back where SAM3 saw nothing."""
    fov_y = math.degrees(2 * math.atan(h / (2 * row["fy"])))
    az = [float(x) for x in a.pull_azimuths.split(",")]; el = [float(x) for x in a.pull_elevations.split(",")]
    extra = [float(x) for x in a.pull_scales.split(",") if float(x) != 1.0]
    d_ref = float(np.linalg.norm(np.asarray(row["c2w"])[:3, 3] - np.array(stats["centroid"])))
    half = ob.mask_half_extent(ob.mask_bbox(mask), d_ref, row["fx"], row["fy"])
    radius_pull = max(stats["radius_median"], half)
    cams = ob.pullback_cameras(np.asarray(row["c2w"]), np.array(stats["centroid"]), radius_pull, fov_y, w / h,
                               margin=a.pull_margin, azimuths_deg=az, elevations_deg=el, extra_scales=extra)
    rg = a.work / "reground"; (rg / "frames").mkdir(parents=True, exist_ok=True); (rg / "seed").mkdir(exist_ok=True)
    for old in list((rg / "frames").glob("*.png")) + list((rg / "seed").glob("*.png")):
        old.unlink()
    rows = []
    for i, cam in enumerate(cams):
        sil = silhouette(s, cam["c2w"], K, w, h, seed_opac)
        cover = float(sil.mean()); bb = ob.mask_bbox(sil)
        cam.update({"cam": i, "file": f"cam_{i:03d}.png", "w": w, "h": h, "fx": row["fx"], "fy": row["fy"], "cx": row["cx"], "cy": row["cy"],
                    "seed_cover": round(cover, 5), "seed_bbox": bb, "seed_fits": ob.fits_in_frame(bb, w, h), "seed_visible": cover >= MIN_MASK_FRAC})
        if not cam["seed_visible"]:
            continue                                   # seed not in view (inside a wall, behind geometry): no frame to ground on
        Image.fromarray(rgb_u8(s, cam["c2w"], K, w, h)).save(rg / "frames" / cam["file"])
        Image.fromarray((sil * 255).astype(np.uint8)).save(rg / "seed" / cam["file"])
        rows.append(cam)
    (rg / "views.json").write_text(json.dumps({"cam_indices": [r["cam"] for r in rows], "W": w, "H": h, "n_train": 0}))
    (rg / "cams.json").write_text(json.dumps({"views": rows, "all": cams, "source": "pullback-render"}, indent=1))
    return {"candidates": len(cams), "rendered": len(rows), "fov_y_deg": round(fov_y, 2), "distance_ref": round(d_ref, 4),
            "radius_pull": round(radius_pull, 4), "radius_mask": round(half, 4), "radius_median": round(stats["radius_median"], 4),
            "radius_p95": round(stats["radius"], 4), "distance_pull": round(cams[0]["distance"], 4) if cams else None}


def try_reground(s: Splat, a, means_np: np.ndarray, seed0: np.ndarray, voxel: float, K, w: int, h: int) -> tuple[dict | None, dict]:
    """Second grounding on the pulled-back renders. Returns (reference | None, receipt)."""
    rg = a.work / "reground"
    if not (rg / "cams.json").is_file() or not (rg / "masks" / a.slug).is_dir():
        return None, {"used": False, "reason": "no reground masks"}
    views = json.loads((rg / "cams.json").read_text())["views"]
    def load_npz(cam: int):
        p = rg / "masks" / a.slug / f"cam_{cam:03d}.npz"
        return np.load(p) if p.is_file() else None
    def load_seed(cam: int):
        p = rg / "seed" / f"cam_{cam:03d}.png"
        return (np.asarray(Image.open(p).convert("L")) > 127) if p.is_file() else None
    best = ob.pick_reground(views, load_npz, load_seed, min_containment=a.reground_containment)
    if best is None:
        return None, {"used": False, "reason": "no pulled-back instance contains the seed silhouette"}
    row, mask = best["row"], best["mask"]
    depth = depth_at(s, row["c2w"], K, w, h)
    seed, pts = lift_seed(means_np, row, mask, depth, voxel, w, h)
    overlap = float(np.isin(seed0, seed).mean()) if len(seed0) else 0.0
    info = {"view": row["tag"], "cam": row["cam"], "instance": best["instance"], "score": best["score"], "containment": best["containment"],
            "fits": best["fits"], "growth_px": best["growth"], "mask_frac": round(best["frac"], 4), "n_initial": int(len(seed0)),
            "n_regrounded": int(len(seed)), "initial_kept": round(overlap, 4), "lifted_points": int(len(pts))}
    if len(seed) < MIN_SEED or overlap < MIN_REGROUND_OVERLAP:
        return None, {"used": False, "reason": f"re-grounded seed rejected (n={len(seed)}, keeps {overlap:.2f} of the initial seed)", **info}
    return {"row": row, "mask": mask, "depth": depth, "seed": seed, "kind": "reground"}, {"used": True, **info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path); ap.add_argument("work", type=Path); ap.add_argument("--slug", required=True)
    ap.add_argument("--stage", choices=["full", "seed", "orbit"], default="full")
    ap.add_argument("--azimuths", default="-60,-45,-30,-15,15,30,45,60"); ap.add_argument("--elevations", default="-10,0,15,30")
    ap.add_argument("--distance-scales", default="1.0,0.7")
    ap.add_argument("--min-visible", type=float, default=0.3); ap.add_argument("--min-views", type=int, default=8)
    ap.add_argument("--max-views", type=int, default=16); ap.add_argument("--voxel-mult", type=float, default=2.0)
    ap.add_argument("--depth-tol-frac", type=float, default=0.03)
    ap.add_argument("--pull-azimuths", default="-40,0,40"); ap.add_argument("--pull-elevations", default="0,20")
    ap.add_argument("--pull-scales", default="1.0,1.5"); ap.add_argument("--pull-margin", type=float, default=1.6)
    ap.add_argument("--reground-containment", type=float, default=0.5)
    a = ap.parse_args(); t0 = time.time()
    s = Splat(a.config)
    best = pick_reference(a.work / "ref", a.slug)
    row, mask = best["row"], best["mask"]
    w, h = int(row["w"]), int(row["h"])
    K = s.K_from(row["fx"], row["fy"], row["cx"], row["cy"])
    spacing = typical_spacing(s.means); voxel = a.voxel_mult * spacing
    means_np = s.means.cpu().numpy()
    # depth at the photo reference -> lift the mask -> initial seed gaussians
    depth = depth_at(s, row["c2w"], K, w, h)
    seed0, pts0 = lift_seed(means_np, row, mask, depth, voxel, w, h)
    stats0 = ob.seed_stats(means_np, seed0)
    if stats0["n"] < MIN_SEED:
        raise SystemExit(f"seed too small ({stats0['n']} gaussians) — grounding or depth lift failed")
    save_seed(a.work / "seed_initial.npz", seed0, stats0, voxel, spacing, {"ref_cam": row["cam"]})
    ref = {"row": row, "mask": mask, "depth": depth, "seed": seed0, "kind": "photo"}
    reground = {"used": False, "reason": "not requested"}
    if a.stage == "seed":
        pull = stage_seed(s, a, row, mask, stats0, seed_opacity(s, seed0), K, w, h)
        receipt = {"schema": "dev.splatlab.isolate-by-reference.seed/v1", "slug": a.slug, "reference": {"cam": row["cam"], "photo": row["photo"], "instance": best["instance"], "score": best["score"]},
                   "seed": {**stats0, "spacing": spacing, "voxel": voxel, "lifted_points": int(len(pts0))}, "pullback": pull, "runtime_s": round(time.time() - t0, 1)}
        (a.work / "seed_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
        print(f"[seed] ref cam {row['cam']} -> seed {stats0['n']} gaussians; {pull['rendered']}/{pull['candidates']} pulled-back views rendered "
              f"(ref dist {pull['distance_ref']} -> {pull['distance_pull']}) in {time.time() - t0:.0f}s", flush=True)
        return 0
    if a.stage == "orbit":
        picked, reground = try_reground(s, a, means_np, seed0, voxel, K, w, h)
        if picked is not None:
            ref = picked
    row, mask, depth, seed = ref["row"], ref["mask"], ref["depth"], ref["seed"]
    stats = ob.seed_stats(means_np, seed); centroid = np.array(stats["centroid"])
    # virtual orbit around the (re-grounded) seed, visibility-filtered
    az = [float(x) for x in a.azimuths.split(",")]; el = [float(x) for x in a.elevations.split(",")]
    ds = [float(x) for x in a.distance_scales.split(",")]
    cams = ob.orbit_cameras(np.asarray(row["c2w"]), centroid, az, el, distance_scales=ds)
    seed_opac = seed_opacity(s, seed)
    track = a.work / "track"; (track / "frames").mkdir(parents=True, exist_ok=True); (track / "seed").mkdir(exist_ok=True)
    for old in list((track / "frames").glob("*.jpg")) + list((track / "seed").glob("*.png")):
        old.unlink()
    scene_depth_scale = float(np.nanmedian(depth[mask])) if np.isfinite(depth[mask]).any() else 1.0
    for cam in cams:                      # score every candidate first (depth-only renders, cheap)
        c = _c2w(cam["c2w"])
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
    for i, cam in enumerate(kept):        # render RGB + the seed's own silhouette for the chosen views
        cam["frame"] = i
        Image.fromarray(rgb_u8(s, cam["c2w"], K, w, h)).save(track / "frames" / f"{i}.jpg", quality=95)
        sil = silhouette(s, cam["c2w"], K, w, h, seed_opac)
        Image.fromarray((sil * 255).astype(np.uint8)).save(track / "seed" / f"{i}.png")
        cam["seed_bbox"] = ob.mask_bbox(sil)                   # pixel [x0, y0, x1, y1] of the seed silhouette in this view
    Image.fromarray((mask * 255).astype(np.uint8)).save(track / "seed_mask.png")
    save_seed(a.work / "seed.npz", seed, stats, voxel, spacing)
    (track / "cams.json").write_text(json.dumps({"w": w, "h": h, "fx": row["fx"], "fy": row["fy"], "cx": row["cx"], "cy": row["cy"],
                                                 "frames": kept, "all": cams, "reference_kind": ref["kind"]}, indent=1))
    reference = {"kind": ref["kind"], "cam": row["cam"], "photo": row.get("photo"), "instance": best["instance"] if ref["kind"] == "photo" else reground.get("instance"),
                 "score": best["score"] if ref["kind"] == "photo" else reground.get("score"), "quality": best.get("quality"),
                 "border_fraction": ob.border_fraction(mask), "mask_frac": float(mask.mean())}
    receipt = {"schema": "dev.splatlab.isolate-by-reference.orbit/v2", "slug": a.slug, "reference": reference,
               "photo_reference": {"cam": best["row"]["cam"], "photo": best["row"]["photo"], "instance": best["instance"], "score": best["score"], "mask_frac": best["frac"]},
               "reground": reground,
               "seed": {**stats, "spacing": spacing, "voxel": voxel, "n_initial": int(stats0["n"])},
               "orbit": {"candidates": len(cams), "kept": len(kept), "min_visible": a.min_visible, "min_views": a.min_views,
                         "kept_views": [{"tag": c["tag"], "visible": c["visible_fraction"]} for c in kept],
                         "dropped": len(cams) - len(kept)}, "runtime_s": round(time.time() - t0, 1)}
    (a.work / "orbit_receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    rg = f"re-grounded on {reground['view']} (seed {stats0['n']} -> {stats['n']})" if reground.get("used") else f"photo seed ({reground.get('reason')})"
    print(f"[orbits] ref cam {best['row']['cam']} score {best['score']:.2f} mask {best['frac']:.3f}; {rg}; "
          f"orbit {len(kept)}/{len(cams)} views kept in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
