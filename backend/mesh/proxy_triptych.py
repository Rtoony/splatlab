#!/usr/bin/env python3
"""P6d: assemble the per-instance proxy receipt triptych (capture crop |
generated proxy preview | registered overlay).

The "registered overlay" panel is a top-down + front orthographic scatter of
the captured object.ply vs the ICP-registered proxy.ply (both already in the
same coordinate frame post-registration) -- deliberately simple/robust
matplotlib math instead of a full perspective render, to avoid a camera-
convention bug eating the time budget on a secondary receipt.

Runs in the dn-splatter-probe env.

Usage: proxy_triptych.py <object.ply> <proxy.ply> <crop.png> <proxy_preview.webp>
       <out.png>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plyfile import PlyData


def _xyz(path: Path) -> np.ndarray:
    v = PlyData.read(str(path))["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)


def main() -> int:
    object_ply, proxy_ply, crop_png, preview_webp, out_png = (Path(a) for a in sys.argv[1:6])

    obj_xyz = _xyz(object_ply)
    proxy_xyz = _xyz(proxy_ply)

    fig, (ax_top, ax_front) = plt.subplots(1, 2, figsize=(8, 4))
    for ax, (i, j), title in ((ax_top, (0, 1), "top (X-Y)"), (ax_front, (0, 2), "front (X-Z)")):
        ax.scatter(obj_xyz[:, i], obj_xyz[:, j], s=0.5, c="royalblue", alpha=0.4, label="captured")
        ax.scatter(proxy_xyz[:, i], proxy_xyz[:, j], s=0.5, c="darkorange", alpha=0.4, label="proxy")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    ax_top.legend(loc="upper right", markerscale=15, fontsize=7)
    overlay_path = out_png.with_name("_overlay_tmp.png")
    fig.savefig(overlay_path, bbox_inches="tight", dpi=110)
    plt.close(fig)

    # try/finally (review finding 2026-07-23): a bad crop/preview panel used
    # to raise before the unlink below ever ran, leaking _overlay_tmp.png.
    try:
        panels = []
        for p in (crop_png, preview_webp, overlay_path):
            if p.is_file():
                panels.append(Image.open(p).convert("RGB"))
        if not panels:
            print("FATAL: no panels available for triptych", file=sys.stderr)
            return 1

        h = 360
        resized = [im.resize((int(im.width * h / im.height), h)) for im in panels]
        total_w = sum(im.width for im in resized) + 10 * (len(resized) - 1)
        canvas = Image.new("RGB", (total_w, h), "white")
        x = 0
        for im in resized:
            canvas.paste(im, (x, 0))
            x += im.width + 10
        canvas.save(out_png)
    finally:
        overlay_path.unlink(missing_ok=True)
    print(f"TRIPTYCH_DONE -> {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
