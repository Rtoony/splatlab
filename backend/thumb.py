"""Lightweight scene thumbnails for the gallery.

Renders a small point-cloud projection of a splat's .ply (sampled by seeking, so
it's CPU-only and ~50ms even on a multi-million-point scene), colored by each
splat's base color (SH DC term). No GPU, no headless browser. Cached to
_preview/thumb.webp next to the scene.
"""
from __future__ import annotations

import math
import re
import struct
from pathlib import Path

from PIL import Image, ImageDraw

_FMT = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I", "short": "h", "ushort": "H"}
_C0 = 0.28209479177387814  # SH DC -> linear factor


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _render(ply_path: Path, out_path: Path, size: tuple[int, int] = (320, 180), n: int = 12000) -> bool:
    with ply_path.open("rb") as f:
        hdr = b""
        while b"end_header\n" not in hdr:
            chunk = f.read(1)
            if not chunk:
                return False
            hdr += chunk
        text = hdr.decode("latin1")
        base = len(hdr)
        m = re.search(r"element vertex (\d+)", text)
        if not m:
            return False
        vcount = int(m.group(1))
        props = re.findall(r"property (\w+) (\w+)", text)
        names = [p[1] for p in props]
        try:
            rowfmt = "<" + "".join(_FMT[a] for a, _ in props)
        except KeyError:
            return False
        rowsz = struct.calcsize(rowfmt)
        try:
            xi, zi, yi = names.index("x"), names.index("z"), names.index("y")
            d0, d1, d2 = names.index("f_dc_0"), names.index("f_dc_1"), names.index("f_dc_2")
        except ValueError:
            return False
        oi = names.index("opacity") if "opacity" in names else None

        step = max(1, vcount // n)
        pts: list[tuple[float, float, float, tuple[int, int, int]]] = []
        for i in range(0, vcount, step):
            f.seek(base + i * rowsz)
            raw = f.read(rowsz)
            if len(raw) < rowsz:
                break
            v = struct.unpack(rowfmt, raw)
            if oi is not None and 1 / (1 + math.exp(-max(-30.0, min(30.0, v[oi])))) < 0.3:
                continue

            def col(idx: int) -> int:
                return max(0, min(255, int((0.5 + _C0 * v[idx]) * 255)))

            pts.append((v[xi], v[zi], v[yi], (col(d0), col(d1), col(d2))))

    if len(pts) < 20:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = _percentile(xs, 0.03), _percentile(xs, 0.97)
    y0, y1 = _percentile(ys, 0.03), _percentile(ys, 0.97)
    if x1 <= x0 or y1 <= y0:
        return False

    W, H = size
    pad = 10
    scale = min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    img = Image.new("RGB", (W, H), (8, 10, 16))
    draw = ImageDraw.Draw(img)
    pts.sort(key=lambda p: -p[2])  # paint far points first
    for x, y, _depth, color in pts:
        px = int(W / 2 + (x - cx) * scale)
        py = int(H / 2 - (y - cy) * scale)
        if 0 <= px < W and 0 <= py < H:
            draw.point((px, py), fill=color)
            draw.point((px + 1, py), fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "WEBP", quality=80)
    return True


def get_or_make(preview_dir: Path) -> Path | None:
    """Return the cached thumbnail, generating it from web.ply/splat.ply if needed."""
    thumb = preview_dir / "thumb.webp"
    if thumb.is_file():
        return thumb
    src = None
    for candidate in ("web.ply", "splat.ply"):
        p = preview_dir / candidate
        if p.is_file():
            src = p
            break
    if src is None:
        return None
    try:
        if _render(src, thumb):
            return thumb
    except Exception:
        return None
    return None
