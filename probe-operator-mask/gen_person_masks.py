"""Generate per-crop PERSON masks for the operator-masking A/B.

Mask R-CNN (COCO person class) over every crop image. Output per image:
<out_dir>/<image_name>.png — 8-bit, 255 = keep, 0 = masked (person).
That is COLMAP's --ImageReader.mask_path convention (zero = ignore for
features); nerfstudio mask semantics are checked separately before reuse.

Dilates person masks by ~2% of image width so feature points on the person's
silhouette edge (half on background) are also suppressed.

Run in the sam2 env:
  ~/miniconda3/envs/sam2/bin/python gen_person_masks.py <images_dir> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

IMAGES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda"
PERSON = 1
SCORE_MIN = 0.5
MASK_BIN = 0.5
BATCH = 8

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights).to(DEV).eval()
tf = weights.transforms()

def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    """Binary dilation via max-pool (GPU, no cv2 dependency)."""
    m = mask[None, None].float()
    m = torch.nn.functional.max_pool2d(m, kernel_size=2 * r + 1, stride=1, padding=r)
    return m[0, 0] > 0.5

paths = sorted(IMAGES.glob("*.jpg")) + sorted(IMAGES.glob("*.png"))
t0 = time.time()
n_person = 0
for s in range(0, len(paths), BATCH):
    batch_paths = paths[s:s + BATCH]
    imgs = [Image.open(p).convert("RGB") for p in batch_paths]
    xs = [tf(im).to(DEV) for im in imgs]
    with torch.no_grad():
        outs = model(xs)
    for p, im, out in zip(batch_paths, imgs, outs):
        keep = (out["labels"] == PERSON) & (out["scores"] >= SCORE_MIN)
        w, h = im.size
        if keep.any():
            union = (out["masks"][keep, 0] > MASK_BIN).any(0)
            union = dilate(union, max(2, int(0.02 * w)))
            n_person += 1
        else:
            union = torch.zeros((h, w), dtype=torch.bool, device=DEV)
        mask = np.where(union.cpu().numpy(), 0, 255).astype(np.uint8)
        Image.fromarray(mask).save(OUT / f"{p.name}.png")
    if (s // BATCH) % 10 == 0:
        print(f"[masks] {s + len(batch_paths)}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
print(f"[masks] DONE: {len(paths)} masks, {n_person} frames with person pixels, "
      f"{time.time() - t0:.0f}s", flush=True)
