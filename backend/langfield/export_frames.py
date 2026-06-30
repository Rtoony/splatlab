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
W = H = None
for i in range(n):
    img = (ds[i]["image"][..., :3].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # HWC RGB
    H, W = img.shape[:2]
    Image.fromarray(img).save(frames_dir / f"frame_{i:03d}.png")
json.dump({"n": n, "W": int(W), "H": int(H)}, open(OUTD / "frames_meta.json", "w"))
print(f"EXPORTED {n} frames @ {W}x{H} -> {frames_dir}")
