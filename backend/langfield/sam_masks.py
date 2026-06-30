"""PASS A (sam2 env, Py 3.10): SAM 2.1 automatic class-agnostic masks per frame.
Writes features/masks/view_{i:03d}.npz with label_map int16 [H,W] (-1 = no mask).
Large masks first so SMALL masks overwrite at overlaps -> finer object wins (the
granularity we want for region features). No SigLIP here (that runs in PASS B via
the langfield-spike env's open_clip/transformers).
"""
import sys, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

OUTD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/rtoony/tools/langfield-spike/features")
SAM2_CKPT = "/home/rtoony/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"   # Hydra config NAME, not a path

model = build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda", apply_postprocessing=False)
amg = SAM2AutomaticMaskGenerator(
    model=model,
    points_per_side=32,
    points_per_batch=64,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    crop_n_layers=0,
    box_nms_thresh=0.7,
    min_mask_region_area=200,
    output_mode="binary_mask",
)

meta = json.load(open(OUTD / "frames_meta.json"))
n = meta["n"]
masks_dir = OUTD / "masks"
masks_dir.mkdir(parents=True, exist_ok=True)

for i in range(n):
    img = np.array(Image.open(OUTD / "frames" / f"frame_{i:03d}.png").convert("RGB"))
    with torch.inference_mode():
        anns = amg.generate(img)
    anns.sort(key=lambda m: m["area"], reverse=True)  # large -> small (small overwrites)
    H, Wd = img.shape[:2]
    label = np.full((H, Wd), -1, dtype=np.int16)
    for k, m in enumerate(anns):
        label[m["segmentation"]] = k
    np.savez_compressed(masks_dir / f"view_{i:03d}.npz", label=label, n_masks=len(anns))
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  view {i}: {len(anns)} masks ({int((label >= 0).mean()*100)}% covered)", flush=True)

print(f"SAM_DONE {n} views -> {masks_dir}")
