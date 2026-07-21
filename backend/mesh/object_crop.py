#!/usr/bin/env python3
"""P5c: crop an isolated object out of the best source photo.

Projects the object's TIGHT bbox (cluster pool — the expanded bbox includes
debris and blows crops to full frame) through candidate dataparser train
cameras and picks the one with the largest fully-in-frame projection.
Runs in the dn-splatter-probe env, CPU-only.

Usage: object_crop.py <splatfacto_config.yml> <object.json> <out.png>
       [--candidates 12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

_orig_load = torch.load


def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _patched_load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("object_json")
    ap.add_argument("out_png")
    ap.add_argument("--candidates", type=int, default=12)
    args = ap.parse_args()

    obj = json.loads(Path(args.object_json).read_text())
    bb = obj.get("bbox_tight") or obj["bbox_scene"]
    lo, hi = np.array(bb["min"]), np.array(bb["max"])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    cfg_path = Path(args.config).resolve()
    # nerfstudio's own !!python/object config graph, self-produced on this box.
    config = yaml.load(cfg_path.read_text(), Loader=yaml.Loader)
    dp = config.pipeline.datamanager.dataparser
    dp.data = cfg_path.parents[2]
    outputs = dp.setup().get_dataparser_outputs(split="train")
    cams = outputs.cameras
    n = int(cams.camera_to_worlds.shape[0])
    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    best = None
    for ci in sorted({int(round(i)) for i in np.linspace(0, n - 1, args.candidates)}):
        c2w = np.eye(4)
        c2w[:3, :] = cams.camera_to_worlds[ci].numpy()
        w2c = np.linalg.inv(c2w @ flip)
        pc = (w2c[:3, :3] @ corners.T + w2c[:3, 3:4]).T
        if (pc[:, 2] <= 0.05).any():          # behind / grazing the camera
            continue
        uv = pc[:, :2] / pc[:, 2:3]
        fx, fy = float(cams.fx[ci]), float(cams.fy[ci])
        cx, cy = float(cams.cx[ci]), float(cams.cy[ci])
        px = np.stack([uv[:, 0] * fx + cx, uv[:, 1] * fy + cy], axis=1)
        W, H = int(cams.width[ci]), int(cams.height[ci])
        x0, y0 = px.min(0)
        x1, y1 = px.max(0)
        in_frame = x0 > -0.05 * W and y0 > -0.05 * H and x1 < 1.05 * W and y1 < 1.05 * H
        area = (min(x1, W) - max(x0, 0)) * (min(y1, H) - max(y0, 0)) / (W * H)
        score = area + (1.0 if in_frame else 0.0)   # prefer fully-framed views
        if 0 < area and (best is None or score > best["score"]):
            best = {"score": score, "cam": ci, "box": (x0, y0, x1, y1), "W": W, "H": H,
                    "area": area, "in_frame": in_frame}
    if best is None:
        print("FATAL: object projects into no candidate camera", file=sys.stderr)
        return 1

    ci = best["cam"]
    x0, y0, x1, y1 = best["box"]
    mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
    W, H = best["W"], best["H"]
    box = (max(0, int(x0 - mx)), max(0, int(y0 - my)),
           min(W, int(x1 + mx)), min(H, int(y1 + my)))
    img = Image.open(outputs.image_filenames[ci]).convert("RGB").resize((W, H))
    img.crop(box).save(args.out_png)
    print(json.dumps({"cam": int(ci), "box": [int(v) for v in box],
                      "area_frac": round(best["area"], 3),
                      "fully_in_frame": bool(best["in_frame"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
