"""Build per-crop-index SEAM-BAND masks for the dual-fisheye stitch seams.

The v360 dfisheye stitch puts the two lens boundaries at fixed yaws = equirect
x = 0.25W and 0.75W (visible parallax ghosting there in the pool capture).
Instead of doing the crop-projection trig by hand, feed a synthetic white
equirect with black seam bands through nerfstudio's OWN
generate_planar_projections_from_equirectangular with the exact same args the
pipeline uses — the projected crops ARE the per-crop-index masks, geometrically
exact by construction. Crop geometry is frame-independent, so one synthetic
frame yields the mask for every real frame with the same crop index.

Run in the splatops env:
  ~/miniconda3/envs/splatops/bin/python make_seam_masks.py <equirect_frames_dir> <out_dir> [band_deg]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from nerfstudio.process_data import equirect_utils

EQUIRECT_DIR = Path(sys.argv[1])          # original colmap/equirect_frames (read-only)
OUT = Path(sys.argv[2])
BAND_DEG = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
CROPS = 8
CROP_FACTOR = (0.0, 0.15, 0.0, 0.0)       # exact pipeline args (crop_bottom=0.15)

OUT.mkdir(parents=True, exist_ok=True)
sample = sorted(EQUIRECT_DIR.glob("*.jpg"))[0]
W, H = Image.open(sample).size
size = equirect_utils.compute_resolution_from_equirect(EQUIRECT_DIR, CROPS)
print(f"[seam] equirect {W}x{H}, crop size {size}, band ±{BAND_DEG}°", flush=True)

band_px = max(2, round(W * BAND_DEG / 360.0))
img = np.full((H, W, 3), 255, np.uint8)
for cx in (round(W * 0.25), round(W * 0.75)):
    img[:, max(0, cx - band_px):min(W, cx + band_px)] = 0

tmp = OUT / "_synth"
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir(parents=True)
Image.fromarray(img).save(tmp / "seam_00001.jpg", quality=98)
equirect_utils.generate_planar_projections_from_equirectangular(tmp, size, CROPS, crop_factor=CROP_FACTOR)
proj = sorted((tmp / "planar_projections").iterdir())
assert len(proj) == CROPS, f"expected {CROPS} projections, got {[p.name for p in proj]}"
for k, p in enumerate(proj):
    m = np.asarray(Image.open(p).convert("L"))
    binary = np.where(m < 128, 0, 255).astype(np.uint8)   # black = seam = ignore
    Image.fromarray(binary).save(OUT / f"seam_mask_{k}.png")
    print(f"[seam] crop {k}: masked {100 * (binary == 0).mean():.1f}% of pixels", flush=True)
shutil.rmtree(tmp)
print("[seam] DONE", flush=True)
