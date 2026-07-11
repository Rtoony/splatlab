"""Compose per-arm masks under BOTH naming schemes.

Inputs:
  --crops    original colmap/images dir (crop-named frame_XXXXX_k.jpg) — read-only
  --seam     seam_mask_<k>.png dir (from make_seam_masks.py), optional
  --person   person masks dir keyed by SEQUENTIAL name (frame_NNNNN.jpg.png,
             from gen_person_masks.py run against processed/images), optional
Outputs (black=ignore, white=keep, 1-channel):
  <out>/colmap/frame_XXXXX_k.jpg.png   for --ImageReader.mask_path
  <out>/seq/frame_NNNNN.jpg.png        for nerfstudio mask_path injection

Mapping crop→sequential mirrors ns-process-data: sorted lexical order of the
crop names = frame_00001.jpg … frame_NNNNN.jpg.

  python3 compose_masks.py --crops DIR --out DIR [--seam DIR] [--person DIR]
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--crops", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seam", type=Path)
ap.add_argument("--person", type=Path)
args = ap.parse_args()
assert args.seam or args.person, "compose at least one mask source"

crop_names = sorted(p.name for p in args.crops.glob("*.jpg"))
assert crop_names, f"no crops in {args.crops}"
(args.out / "colmap").mkdir(parents=True, exist_ok=True)
(args.out / "seq").mkdir(parents=True, exist_ok=True)

seam_cache: dict[str, np.ndarray] = {}
if args.seam:
    for k in range(8):
        seam_cache[str(k)] = np.asarray(Image.open(args.seam / f"seam_mask_{k}.png").convert("L"))

stats = []
for i, crop_name in enumerate(crop_names, start=1):
    seq_name = f"frame_{i:05d}.jpg"
    mask = None
    if args.seam:
        k = crop_name.rsplit("_", 1)[1].split(".")[0]   # frame_00012_5.jpg -> "5"
        mask = seam_cache[k].copy()
    if args.person:
        pm_path = args.person / f"{seq_name}.png"
        pm = np.asarray(Image.open(pm_path).convert("L"))
        mask = pm.copy() if mask is None else np.minimum(mask, pm)   # black wins
    out = Image.fromarray(np.where(mask < 128, 0, 255).astype(np.uint8))
    out.save(args.out / "colmap" / f"{crop_name}.png")
    out.save(args.out / "seq" / f"{seq_name}.png")
    stats.append((np.asarray(out) == 0).mean())

print(f"[compose] {len(crop_names)} masks -> {args.out} "
      f"(mean masked {100 * float(np.mean(stats)):.1f}%, max {100 * float(np.max(stats)):.1f}%)", flush=True)
