"""Row-subset writer for binary little-endian 3DGS PLYs (numpy only).

The exported splat.ply's rows are in checkpoint order (ns-export writes
model.means as-is), so a gaussian index set selects rows directly."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
          "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2",
          "short": "<i2", "int16": "<i2", "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}


def read_header(fh) -> tuple[list[str], np.dtype, int, str]:
    header: list[str] = []
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("PLY header never ended")
        header.append(line.decode("ascii", errors="replace").rstrip("\r\n"))
        if header[-1] == "end_header":
            break
    fmt = next((h.split()[1] for h in header if h.startswith("format ")), "")
    if fmt != "binary_little_endian":
        raise ValueError(f"unsupported PLY format {fmt!r} (need binary_little_endian)")
    count = next((int(h.split()[2]) for h in header if h.startswith("element vertex")), 0)
    props = [h.split() for h in header if h.startswith("property ") and len(h.split()) == 3]
    dtype = np.dtype([(p[2], _TYPES[p[1]]) for p in props])
    return header, dtype, count, fmt


def read_rows(path: Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open("rb") as fh:
        header, dtype, count, _ = read_header(fh)
        rows = np.frombuffer(fh.read(dtype.itemsize * count), dtype=dtype, count=count)
    return header, rows


def write_subset(src: Path, dst: Path, indices: np.ndarray) -> int:
    header, rows = read_rows(src)
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) and (idx.min() < 0 or idx.max() >= len(rows)):
        raise IndexError("index out of range for PLY rows")
    sub = rows[idx]
    out = [h if not h.startswith("element vertex") else f"element vertex {len(sub)}" for h in header]
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as fh:
        fh.write(("\n".join(out) + "\n").encode("ascii"))
        fh.write(sub.tobytes())
    return int(len(sub))
