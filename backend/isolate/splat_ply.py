"""Raw/pre-activation splat PLY writer (numpy only) — the same 14-field layout
batch_isolate.py / object_isolate.py write, so every downstream consumer
(solidify, Blender importer, pluck) reads isolate-by-reference output unchanged."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_FIELDS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_splat_ply(path: Path, xyz: np.ndarray, f_dc: np.ndarray, opacity: np.ndarray,
                    scale: np.ndarray, rot: np.ndarray, comment: str = "SplatLab isolate-by-reference") -> int:
    rows = np.concatenate([np.asarray(xyz, "<f4"), np.asarray(f_dc, "<f4"),
                           np.asarray(opacity, "<f4").reshape(-1, 1), np.asarray(scale, "<f4"),
                           np.asarray(rot, "<f4")], axis=1)
    if rows.shape[1] != len(PLY_FIELDS):
        raise ValueError(f"expected {len(PLY_FIELDS)} columns, got {rows.shape[1]}")
    header = "ply\nformat binary_little_endian 1.0\n" + f"comment {comment}\n" + f"element vertex {rows.shape[0]}\n"
    header += "".join(f"property float {f}\n" for f in PLY_FIELDS) + "end_header\n"
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(header.encode("ascii")); fh.write(rows.tobytes())
    return int(rows.shape[0])
