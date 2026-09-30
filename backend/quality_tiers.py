"""Viewing-quality tiers for a splat (owner decision 2026-09-30).

"Full quality should only be shown when viewed locally on my Nexus PC; phones and browsers outside my network get
lower quality, with the higher-quality versions available to download."

  full  _preview/splat.ply            every splat, full colour detail (10M / 2.47 GB on an ultra scene)
  web   _preview/web.compressed.ply   3M (Raw 360) / 1.2M, SH0, PlayCanvas compressed PLY (49 MB vs 204 MB web.ply)
  lite  _preview/lite.compressed.ply  1M, SH0, compressed (16 MB) — phones

Measured on the storage room ULTRA (splat_ddb7ca9bf5): all three hold 60 fps in Spark on the 5090, and the
compressed web copy is visually identical to web.ply. Compressed PLY, not .spz: splat-transform writes SPZ v4
(bare "NGSP" header) and Spark 2.2 only reads the older gzip-wrapped SPZ ("Invalid gzip header"); the spz writer
also aborts at 10M splats with SH3.

Which tier a VIEWER may show is a presentation rule, not access control: downloads of every tier stay available
everywhere, and SuperSplat (the editor) still loads the full .ply.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

TIERS = ("full", "web", "lite")
FILES = {"full": "splat.ply", "web": "web.compressed.ply", "lite": "lite.compressed.ply"}
LEGACY_WEB = "web.ply"          # pre-tier jobs: the uncompressed web copy stands in for "web"
LITE_TARGET = 1_000_000
LABELS = {"full": "Full quality", "web": "Web", "lite": "Lite"}
# /preview/file?fmt= value per tier. "web" already means the uncompressed web.ply there (older links), so the
# compressed web tier gets its own name.
FMT = {"full": "full", "web": "webc", "lite": "lite"}
TIER_OF_FMT = {v: k for k, v in FMT.items()}
# Loopback only reaches :3416 from the PC itself; the Cloudflare tunnel (the only other way in) stamps every request
# with cf-* headers, and any proxy adds a forwarding header. Any of them means "not on the Nexus PC".
_PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "cf-ipcountry", "x-forwarded-for", "x-forwarded-host", "forwarded",
                  "x-real-ip")
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def is_local_request(client_host: str | None, headers: Any) -> bool:
    if (client_host or "") not in _LOOPBACK:
        return False
    return not any(h in headers for h in _PROXY_HEADERS)


def ply_vertex_count(path: Path) -> int | None:
    """`element vertex N` from a (binary or compressed) PLY header, without reading the body."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(65536)
    except OSError:
        return None
    end = head.find(b"end_header")
    for line in (head[:end] if end >= 0 else head).split(b"\n"):
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == b"element" and parts[1] == b"vertex":
            try:
                return int(parts[2])
            except ValueError:
                return None
    return None


def tier_path(preview_dir: Path, tier: str) -> Path | None:
    """The file behind a tier, or None when it does not exist (web falls back to the legacy web.ply)."""
    path = preview_dir / FILES[tier]
    if path.is_file():
        return path
    if tier == "web" and (preview_dir / LEGACY_WEB).is_file():
        return preview_dir / LEGACY_WEB
    return None


def tiers(preview_dir: Path, job_id: str) -> list[dict]:
    out = []
    for tier in TIERS:
        path = tier_path(preview_dir, tier)
        if path is None:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        out.append({"id": tier, "label": LABELS[tier], "splats": ply_vertex_count(path), "bytes": size,
                    "compressed": path.name.endswith(".compressed.ply"),
                    "url": f"/api/splat/jobs/{job_id}/preview/file?fmt={FMT[tier]}"})
    return out


def build_commands(transform: str, preview_dir: Path) -> list[tuple[str, list[str]]]:
    """splat-transform argv for the two compressed tiers, both from web.ply (already SH0 + decimated)."""
    web = preview_dir / LEGACY_WEB
    return [("webopt-webc", [transform, str(web), str(preview_dir / FILES["web"])]),
            ("webopt-lite", [transform, str(web), "--decimate", str(LITE_TARGET), str(preview_dir / FILES["lite"])])]
