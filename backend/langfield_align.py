"""Byte-exact row alignment: exported splat.ply row -> training-checkpoint row.

gauss_emb.npz (the lifted language field) is ordered by the CHECKPOINT's
gaussians (langfield_v2.py lifts onto m.means), but the client renders the
exported splat.ply / langweb.ply — and ns-export FILTERS gaussians on the way
out (Garden: 1,326,611 ckpt -> 1,321,833 ply, 4,778 dropped). Serving relevancy
in ckpt order therefore mistints every splat after the first dropped row.

The exporter copies positions bit-for-bit, so the ply row -> ckpt row map is
recoverable by exact float32-byte matching on xyz — no tolerance search, no
retraining, and it works retroactively for every legacy scene. The map is
cached to <lfdir>/ply_index_map.npy; a scene where matching isn't total (an
exporter that ever transforms positions) gets NO map and the caller serves
ckpt order — the client's row-count guard then fails loud rather than mistint.

Pure numpy + stdlib (imported lazily by the worker so the module never blocks
the no-heavy-deps test import path).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("langfield.align")

MAP_FILENAME = "ply_index_map.npy"


def read_ply_xyz(path: Path) -> np.ndarray:
    """(N, 3) float32 x/y/z from a binary_little_endian all-float PLY (the 3DGS
    layout ns-export writes). Memory-maps the body; never materializes full rows."""
    header = bytearray()
    with open(path, "rb") as fh:
        while not header.endswith(b"end_header\n"):
            chunk = fh.read(1)
            if not chunk:
                raise ValueError(f"{path}: no end_header")
            header += chunk
            if len(header) > 65536:
                raise ValueError(f"{path}: unreasonably large header")
        offset = fh.tell()
    lines = header.decode("ascii", "replace").splitlines()
    if len(lines) < 2 or "binary_little_endian" not in lines[1]:
        raise ValueError(f"{path}: not binary_little_endian")
    count: int | None = None
    props: list[str] = []
    in_vertex = False
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        if parts[:2] == ["element", "vertex"]:
            count = int(parts[2])
            in_vertex = True
        elif parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[0] == "property":
            if parts[1] != "float":
                raise ValueError(f"{path}: non-float vertex property {ln!r}")
            props.append(parts[2])
    if count is None or not props:
        raise ValueError(f"{path}: no vertex element")
    try:
        ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
    except ValueError as exc:
        raise ValueError(f"{path}: missing x/y/z properties") from exc
    data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(count, len(props)))
    return np.ascontiguousarray(data[:, [ix, iy, iz]])


def _void_keys(xyz: np.ndarray) -> np.ndarray:
    """View each float32 xyz row as one opaque 12-byte key (bit-exact compare)."""
    a = np.ascontiguousarray(xyz, dtype=np.float32)
    return a.view([("k", "V12")])["k"].reshape(-1)


def build_index_map(ply_xyz: np.ndarray, ckpt_xyz: np.ndarray) -> np.ndarray | None:
    """int64 [N_ply] with map[i] = ckpt row whose xyz bytes equal ply row i's,
    or None when any ply row has no byte-identical ckpt position (never a
    partial map — a wrong tint is worse than no tint)."""
    ply_k = _void_keys(ply_xyz)
    ck_k = _void_keys(ckpt_xyz)
    order = np.argsort(ck_k, kind="stable")
    ck_sorted = ck_k[order]
    pos = np.searchsorted(ck_sorted, ply_k)
    pos_c = np.minimum(pos, ck_sorted.size - 1)
    ok = (pos < ck_sorted.size) & (ck_sorted[pos_c] == ply_k)
    if not bool(ok.all()):
        log.warning(
            "ply->ckpt align failed: %d/%d ply rows have no byte-identical ckpt xyz",
            int((~ok).sum()), int(ply_k.size),
        )
        return None
    return order[pos_c].astype(np.int64)


def load_or_build_map(lfdir: Path, ckpt_xyz: np.ndarray) -> np.ndarray | None:
    """Cached ply->ckpt map for a scene. lfdir = <jobdir>/_langfield; the
    exported ply lives at <jobdir>/_preview/splat.ply. Returns None (caller
    serves ckpt order) whenever a trustworthy total map can't be produced."""
    lfdir = Path(lfdir)
    cache = lfdir / MAP_FILENAME
    n_ckpt = int(ckpt_xyz.shape[0])
    if cache.is_file():
        m = np.load(cache)
        if m.ndim == 1 and (m.size == 0 or (0 <= int(m.min()) and int(m.max()) < n_ckpt)):
            return m.astype(np.int64)
        log.warning("discarding invalid cached %s", cache)
    ply = lfdir.parent / "_preview" / "splat.ply"
    if not ply.is_file():
        log.warning("no exported ply at %s — relevancy stays in ckpt order", ply)
        return None
    try:
        ply_xyz = read_ply_xyz(ply)
    except (ValueError, OSError) as exc:
        log.warning("cannot parse %s (%s) — relevancy stays in ckpt order", ply, exc)
        return None
    if ply_xyz.shape[0] == n_ckpt and bool((_void_keys(ply_xyz) == _void_keys(ckpt_xyz)).all()):
        m = np.arange(n_ckpt, dtype=np.int64)  # exporter dropped nothing
    else:
        m = build_index_map(ply_xyz, ckpt_xyz)
        if m is None:
            return None
    np.save(cache, m)
    log.info(
        "ply->ckpt map ready: %d ply rows over %d ckpt rows -> %s",
        int(ply_xyz.shape[0]), n_ckpt, cache.name,
    )
    return m
