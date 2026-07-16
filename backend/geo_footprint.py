"""Top-down scene footprints for the Locate-in-the-world lane.

Renders a bird's-eye (plan view) point-cloud projection of a splat's .ply with a
TRANSPARENT background, plus the exact scene-unit ground bounds the image spans —
so the frontend can drape it over a satellite map and let the user drag/rotate/
scale it into place. Same CPU-only sampled-seek parsing as thumb.py (~50ms even
on millions of points); cached to _preview/footprint.webp + footprint.json.

Axis convention (nerfstudio exports are Z-up): ground plane = (x, y), height = z.
Image "up" (-py) is scene +Y; painting is sorted by ascending z so higher points
(roofs, canopy) overdraw lower ones, which is what a satellite photo shows.
"""
from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

_FMT = {"float": "f", "double": "d", "uchar": "B", "int": "i", "uint": "I", "short": "h", "ushort": "H"}
_C0 = 0.28209479177387814  # SH DC -> linear color factor

FOOTPRINT_IMAGE = "footprint.webp"
FOOTPRINT_META = "footprint.json"
_MAX_DIM = 768
_SAMPLES = 24000


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _read_points(ply_path: Path, n: int) -> list[tuple[float, float, float, tuple[int, int, int]]]:
    """Sampled (x, y, z, rgb) rows from a splat .ply — ground-plane order, z = height."""
    with ply_path.open("rb") as f:
        hdr = b""
        while b"end_header\n" not in hdr:
            chunk = f.read(1)
            if not chunk:
                return []
            hdr += chunk
        text = hdr.decode("latin1")
        base = len(hdr)
        m = re.search(r"element vertex (\d+)", text)
        if not m:
            return []
        vcount = int(m.group(1))
        props = re.findall(r"property (\w+) (\w+)", text)
        names = [p[1] for p in props]
        try:
            rowfmt = "<" + "".join(_FMT[a] for a, _ in props)
        except KeyError:
            return []
        rowsz = struct.calcsize(rowfmt)
        try:
            xi, yi, zi = names.index("x"), names.index("y"), names.index("z")
            d0, d1, d2 = names.index("f_dc_0"), names.index("f_dc_1"), names.index("f_dc_2")
        except ValueError:
            return []
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

            pts.append((v[xi], v[yi], v[zi], (col(d0), col(d1), col(d2))))
    return pts


def _render(ply_path: Path, out_path: Path, max_dim: int = _MAX_DIM, n: int = _SAMPLES) -> dict[str, Any] | None:
    pts = _read_points(ply_path, n)
    if len(pts) < 20:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Percentile-trim floaters so one stray gaussian can't blow up the extent.
    x0, x1 = _percentile(xs, 0.02), _percentile(xs, 0.98)
    y0, y1 = _percentile(ys, 0.02), _percentile(ys, 0.98)
    if x1 <= x0 or y1 <= y0:
        return None

    # No padding: bounds map exactly to the image edges so the frontend's
    # pixels<->scene-units math is a single ratio.
    xr, yr = x1 - x0, y1 - y0
    if xr >= yr:
        w, h = max_dim, max(2, round(max_dim * yr / xr))
    else:
        w, h = max(2, round(max_dim * xr / yr)), max_dim
    sx, sy = w / xr, h / yr

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts.sort(key=lambda p: p[2])  # low first; high points overdraw = plan view
    for x, y, _z, color in pts:
        px = int((x - x0) * sx)
        py = int(h - 1 - (y - y0) * sy)  # image up = scene +Y
        if 0 <= px < w and 0 <= py < h:
            fill = (*color, 255)
            draw.point((px, py), fill=fill)
            if px + 1 < w:
                draw.point((px + 1, py), fill=fill)
            if py + 1 < h:
                draw.point((px, py + 1), fill=fill)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "WEBP", quality=85)
    meta: dict[str, Any] = {
        "v": 1,
        "width": w,
        "height": h,
        # Scene-unit ground bounds spanned edge-to-edge by the image.
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "units_per_px": xr / w,
        "center": [(x0 + x1) / 2, (y0 + y1) / 2],
        "up_axis": "z",
    }
    return meta


def get_or_make(preview_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    """Return (image path, bounds meta), generating and caching both if needed."""
    image = preview_dir / FOOTPRINT_IMAGE
    meta_path = preview_dir / FOOTPRINT_META
    if image.is_file() and meta_path.is_file():
        try:
            return image, json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass  # stale/corrupt cache -> regenerate below
    src = None
    for candidate in ("web.ply", "splat.ply"):
        p = preview_dir / candidate
        if p.is_file():
            src = p
            break
    if src is None:
        return None
    try:
        meta = _render(src, image)
    except Exception:
        return None
    if meta is None:
        return None
    meta_path.write_text(json.dumps(meta))
    return image, meta
