"""Deterministic procedural texture tiles per surface class.

v1 is procedural on purpose: proving the class-partitioned bake needs
distinguishable, plausible surfaces, not photoreal ones. When a real tileset
lands at backend/mesh/textures/<class_id>.png, `tile_for` prefers it — the
contract is already shaped for the upgrade. Everything is seeded from the
class id, so tiles are bit-identical across runs (receipt-friendly).

Numpy-only (runs in the probe env; also importable by the test suite).
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

TEXTURES_DIR = Path(__file__).with_name("textures")


def _value_noise(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """[size,size] float 0..1 — bilinear-upsampled random lattice (two octaves)."""
    out = np.zeros((size, size), np.float32)
    amp_total = 0.0
    for octave, amp in ((1, 1.0), (2, 0.5)):
        n = max(2, cells * octave)
        lattice = rng.random((n + 1, n + 1), dtype=np.float32)
        xs = np.linspace(0, n, size, endpoint=False)
        x0 = xs.astype(np.int64)
        fx = (xs - x0).astype(np.float32)
        a = lattice[x0][:, x0]
        b = lattice[x0 + 1][:, x0]
        c = lattice[x0][:, x0 + 1]
        d = lattice[x0 + 1][:, x0 + 1]
        fy = fx[None, :]
        fxv = fx[:, None]
        out += amp * ((a * (1 - fxv) + b * fxv) * (1 - fy)
                      + (c * (1 - fxv) + d * fxv) * fy)
        amp_total += amp
    return out / amp_total


def tile_for(class_def: dict, size: int = 512) -> np.ndarray:
    """[size,size,3] uint8 tile for one taxonomy class entry.

    Prefers a real texture at textures/<id>.png when present; else base_rgb
    modulated by seeded value noise (luma jitter) + per-class speckle."""
    cid = str(class_def.get("id", "unknown"))
    real = TEXTURES_DIR / f"{cid}.png"
    if real.is_file():
        from PIL import Image  # noqa: PLC0415 — only the real-tile path needs it
        img = Image.open(real).convert("RGB").resize((size, size))
        return np.asarray(img, dtype=np.uint8)

    gen = class_def.get("generative") or {}
    base = np.asarray(gen.get("base_rgb") or [0.5, 0.5, 0.5], np.float32)
    noise_cfg = gen.get("noise") or {}
    scale = int(noise_cfg.get("scale", 12))
    jitter = float(noise_cfg.get("luma_jitter", 0.1))
    speckle = float(noise_cfg.get("speckle", 0.3))

    seed = zlib.crc32(cid.encode())
    rng = np.random.default_rng(seed)
    luma = 1.0 + ((_value_noise(size, scale, rng) - 0.5) * 2.0) * jitter
    tile = base[None, None, :] * luma[..., None]
    if speckle > 0:
        spots = rng.random((size, size), dtype=np.float32)
        dark = (spots < speckle * 0.08).astype(np.float32) * 0.35
        light = (spots > 1.0 - speckle * 0.05).astype(np.float32) * 0.25
        tile *= (1.0 - dark[..., None])
        tile += light[..., None] * base[None, None, :]
    return (np.clip(tile, 0, 1) * 255).astype(np.uint8)


def sample_world(tile: np.ndarray, x: np.ndarray, y: np.ndarray,
                 texels_per_unit: float) -> np.ndarray:
    """Sample a tile in WORLD space (wrap-tiled) at xy coordinate arrays —
    the same physical texel density everywhere regardless of atlas layout."""
    size = tile.shape[0]
    u = np.mod((x * texels_per_unit).astype(np.int64), size)
    v = np.mod((y * texels_per_unit).astype(np.int64), size)
    return tile[v, u]
