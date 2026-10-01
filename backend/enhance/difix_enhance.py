"""Run in the isolated `difix` conda env: one viewer frame -> NVIDIA Difix -> an AI-enhanced still.

Difix ("remove degradation", single-step diffusion, nv-tlabs/Difix3D) repairs splat-render artifacts: blur, holes,
floater haze. With a reference photo (`nvidia/difix_ref`), the repair is guided by the nearest REAL capture frame,
which is what keeps it believable; without one it falls back to plain `nvidia/difix`.

Measured 2026-10-01 (realism lab): ~0.3 s for 512², ~0.7 s for 1024x576 with a reference; 8-12 GB peak VRAM.

  python difix_enhance.py <in.png> <ref.png|-> <out.png> [--long-side 1024]
Prints one JSON line: {"ok": true, "ms": ..., "model": ..., "size": [w, h]}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_CACHE", "/mnt/external-safe/models/model-caches/Hugging_Face_Hub")
DIFIX_SRC = os.environ.get("SPLAT_DIFIX_SRC", "/home/rtoony/tools/difix3d/src")
sys.path.insert(0, DIFIX_SRC)


def fit(size, long_side):
    """Largest size <= long_side on the long edge, both sides multiples of 8 (the UNet/VAE stride)."""
    w, h = size
    s = min(1.0, long_side / max(w, h))
    return max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)


def crop_to(im, size):
    from PIL import Image
    W, H = size
    w, h = im.size
    t = W / H
    if w / h > t:
        nw = int(h * t); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / t); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize(size, Image.BICUBIC)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--long-side", type=int, default=1024)
    a = ap.parse_args()
    import torch
    from PIL import Image
    from pipeline_difix import DifixPipeline

    src = Image.open(a.inp).convert("RGB")
    size = fit(src.size, a.long_side)
    use_ref = a.ref != "-" and os.path.isfile(a.ref)
    model = "nvidia/difix_ref" if use_ref else "nvidia/difix"
    pipe = DifixPipeline.from_pretrained(model, trust_remote_code=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    kw = dict(image=src.resize(size, Image.BICUBIC), num_inference_steps=1, timesteps=[199], guidance_scale=0.0,
              height=size[1], width=size[0])
    if use_ref:
        kw["ref_image"] = crop_to(Image.open(a.ref).convert("RGB"), size)
    torch.cuda.synchronize(); t0 = time.time()
    out = pipe("remove degradation", **kw).images[0]
    torch.cuda.synchronize(); ms = (time.time() - t0) * 1000
    out = out.resize(src.size, Image.BICUBIC)
    from PIL import PngImagePlugin
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "SplatLab Enhance (NVIDIA Difix)")
    info.add_text("Comment", "AI-enhanced still: generated detail, not a capture")
    out.save(a.out, pnginfo=info)
    print(json.dumps({"ok": True, "ms": round(ms), "model": model, "size": list(src.size),
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
