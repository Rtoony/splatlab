#!/usr/bin/env python3
"""P6b: export K evenly-spaced train views at dataparser resolution, named by
camera index, for the scene-inventory lane. Same access pattern as
export_frames.py / the proven scene-regen-spike select_views.py (frame-order/
res ambiguity is what let SAM3 and the lift agree on identical images).

Runs in the langfield-spike env.

Usage: scene_views.py <config.yml> <workdir> <K>
"""
import json
import sys
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


def main() -> int:
    config = Path(sys.argv[1])
    workdir = Path(sys.argv[2])
    k = int(sys.argv[3])
    frames_dir = workdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _, pipeline, _, _ = eval_setup(config, test_mode="test")
    ds = pipeline.datamanager.train_dataset
    n = len(ds)
    idxs = sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()))

    w = h = None
    for i in idxs:
        img = (ds[i]["image"][..., :3].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        h, w = img.shape[:2]
        Image.fromarray(img).save(frames_dir / f"cam_{i:03d}.png")

    (workdir / "views.json").write_text(json.dumps(
        {"cam_indices": idxs, "W": int(w), "H": int(h), "n_train": n}, indent=2))
    print(f"SCENE_VIEWS {len(idxs)}/{n} views @ {w}x{h} -> {frames_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
