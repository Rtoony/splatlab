#!/usr/bin/env python3
"""Step 1a (langfield env): stage K evenly spaced TRAINING photos (downscaled to the
render size) as SAM3 grounding frames, in the layout scene_sam3_masks.py expects:
<work>/ref/frames/cam_%03d.png + <work>/ref/views.json + <work>/ref/cams.json."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ckpt import Splat, render_size, scaled_intrinsics  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path); ap.add_argument("work", type=Path)
    ap.add_argument("--views", type=int, default=6); ap.add_argument("--max-width", type=int, default=960)
    a = ap.parse_args(); t0 = time.time()
    s = Splat(a.config); cams = s.train_cams(); photos = s.photo_paths()
    n = int(cams.camera_to_worlds.shape[0]); k = max(1, min(a.views, n))
    ids = sorted({round(i * (n - 1) / max(1, k - 1)) for i in range(k)})
    ref = a.work / "ref"; (ref / "frames").mkdir(parents=True, exist_ok=True)
    rows = []
    for ci in ids:
        w, h = render_size(cams, ci, a.max_width)
        Image.open(photos[ci]).convert("RGB").resize((w, h), Image.LANCZOS).save(ref / "frames" / f"cam_{ci:03d}.png")
        fx, fy, cx, cy = scaled_intrinsics(cams, ci, w, h)
        rows.append({"cam": ci, "file": f"cam_{ci:03d}.png", "photo": photos[ci], "w": w, "h": h,
                     "c2w": cams.camera_to_worlds[ci].cpu().numpy().tolist(), "fx": fx, "fy": fy, "cx": cx, "cy": cy})
    W, H = rows[0]["w"], rows[0]["h"]
    (ref / "views.json").write_text(json.dumps({"cam_indices": ids, "W": W, "H": H, "n_train": n}))
    (ref / "cams.json").write_text(json.dumps({"n_gaussians": s.n, "rasterize_mode": s.mode, "views": rows,
                                               "runtime_s": round(time.time() - t0, 1)}, indent=1))
    print(f"[prepare-ref] {len(ids)} photos staged at {W}x{H} from {n} training cams ({s.n} gaussians)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
