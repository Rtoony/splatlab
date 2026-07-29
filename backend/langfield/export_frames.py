"""PASS 0 (langfield-spike env): dump the 116 train images at dataparser resolution
in nerfstudio camera order, so SAM (PASS A) and the lift (PASS B) index identical
images by identical position. Eliminates the frame-order / resolution ambiguity
(spec risks R-IMG, R-RES).
"""
import sys, json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402

CONFIG = Path(sys.argv[1])
OUTD = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/rtoony/tools/langfield-spike/features")
frames_dir = OUTD / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

config, pipeline, _, _ = eval_setup(CONFIG, test_mode="test")
ds = pipeline.datamanager.train_dataset
n = len(ds)
# ONE canonical resolution for the whole lift, taken from frame 0. Mixed
# per-frame dims are real (ETH3D Courtyard: 1551x1033 next to 1549x1032 —
# dataparser downscale rounding varies per source image), and the lift
# rasterizes, masks, and indexes every view at a single (W, H): the old code
# recorded only the LAST frame's dims, so the first mixed-res dataset died in
# PASS B's resolution assert. Sub-0.2% resamples are semantically nil for SAM
# masks and SigLIP features; the per-camera intrinsics scaling in
# langfield_v2.K_for absorbs the geometry side.
W = H = None
resized = 0
for i in range(n):
    img = (ds[i]["image"][..., :3].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # HWC RGB
    frame = Image.fromarray(img)
    if W is None:
        H, W = img.shape[:2]
    elif img.shape[:2] != (H, W):
        frame = frame.resize((W, H), Image.LANCZOS)
        resized += 1
    frame.save(frames_dir / f"frame_{i:03d}.png")
json.dump({"n": n, "W": int(W), "H": int(H), "resized_to_canonical": resized},
          open(OUTD / "frames_meta.json", "w"))
note = f" ({resized} frames resized to canonical)" if resized else ""
print(f"EXPORTED {n} frames @ {W}x{H}{note} -> {frames_dir}")
