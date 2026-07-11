"""Probe: can torchvision Mask R-CNN find the OPERATOR in equirect crop frames?

Runs person segmentation on a handful of frames from the pool test-flight job
and writes overlay visualizations + a coverage table. This decides mask-source
for the operator-masking spike (fallback: SAM2-auto + SigLIP person scoring).

Run in the sam2 env (torch/torchvision cu128):
  ~/miniconda3/envs/sam2/bin/python probe_person_masks.py <images_dir> <out_dir> [frame ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

IMAGES = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FRAMES = sys.argv[3:] or ["frame_00001.jpg", "frame_00090.jpg", "frame_00180.jpg",
                          "frame_00360.jpg", "frame_00540.jpg", "frame_00719.jpg"]
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PERSON = 1  # COCO class id
SCORE_MIN = 0.5
MASK_BIN = 0.5

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights).to(DEV).eval()
tf = weights.transforms()

print(f"[probe] torchvision {torchvision.__version__} on {DEV}", flush=True)
for name in FRAMES:
    path = IMAGES / name
    if not path.exists():
        print(f"[probe] {name}: MISSING", flush=True)
        continue
    img = Image.open(path).convert("RGB")
    x = tf(img).to(DEV)
    with torch.no_grad():
        out = model([x])[0]
    keep = (out["labels"] == PERSON) & (out["scores"] >= SCORE_MIN)
    masks = out["masks"][keep, 0] > MASK_BIN          # [K,H,W] bool
    scores = out["scores"][keep].tolist()
    union = masks.any(0).cpu().numpy() if masks.numel() else np.zeros((img.height, img.width), bool)
    cov = union.mean()

    arr = np.asarray(img).astype(np.float32)
    arr[union] = arr[union] * 0.35 + np.array([255, 40, 40], np.float32) * 0.65
    vis = Image.fromarray(arr.astype(np.uint8))
    vis.thumbnail((900, 900))
    vis.save(OUT / f"probe_{name.replace('.jpg', '')}_cov{cov:.2f}.webp")
    print(f"[probe] {name}: {int(keep.sum())} person det(s) "
          f"scores={[round(s, 2) for s in scores]} coverage={cov:.1%}", flush=True)
print("[probe] DONE", flush=True)
