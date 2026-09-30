"""Geometry + cameras for a Raw 360 (Spirula) scene, shaped like what the checkpoint-based code reads.

The language-field lift (langfield_v2.py) and the query worker (langfield_worker.py) were written against a
nerfstudio pipeline: `m.means / quats / scales / opacities` and a `Cameras` object with `camera_to_worlds`,
`width`, `height` and `get_intrinsics_matrices()`. A Spirula scene has neither; it has the published splat PLY and
spirula_views.py's pinhole cuts (cameras.json, c2w in the nerfstudio/OpenGL convention). These two helpers give the
existing code the same shapes, so the lift and the query math run unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_splat(ply: Path, device: str, with_colors: bool = False):
    """means [N,3], quats [N,4] (wxyz), scales [N,3] (linear), opacities [N] (0-1) from a 3DGS PLY, in row order.
    With with_colors, also the DC colour as SH band 0 [N,1,3] for rendering."""
    from plyfile import PlyData

    el = PlyData.read(str(ply)).elements[0]

    def col(k):
        return torch.tensor(np.asarray(el[k], dtype=np.float32), device=device)

    means = torch.stack([col("x"), col("y"), col("z")], 1)
    quats = torch.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], 1)
    scales = torch.exp(torch.stack([col("scale_0"), col("scale_1"), col("scale_2")], 1))
    opac = torch.sigmoid(col("opacity"))
    if not with_colors:
        return means, quats, scales, opac
    sh = torch.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], 1)[:, None, :]
    return means, quats, scales, opac, sh


class PinholeCameras:
    """The slice of nerfstudio's Cameras the lift/worker use, backed by spirula_views.py's cameras.json."""

    def __init__(self, cameras_json: Path, device: str):
        views = json.loads(Path(cameras_json).read_text())["views"]
        self.camera_to_worlds = torch.tensor([v["c2w"] for v in views], dtype=torch.float32, device=device)
        self.width = torch.tensor([int(v["width"]) for v in views], device=device)
        self.height = torch.tensor([int(v["height"]) for v in views], device=device)
        K = torch.zeros(len(views), 3, 3, dtype=torch.float32, device=device)
        K[:, 0, 0] = torch.tensor([v["fx"] for v in views], device=device)
        K[:, 1, 1] = torch.tensor([v["fy"] for v in views], device=device)
        K[:, 0, 2] = torch.tensor([v["cx"] for v in views], device=device)
        K[:, 1, 2] = torch.tensor([v["cy"] for v in views], device=device)
        K[:, 2, 2] = 1.0
        self._K = K

    def get_intrinsics_matrices(self) -> torch.Tensor:
        return self._K

    def to(self, device):  # the worker calls .to(DEV) on cameras
        return self

    def __len__(self) -> int:
        return int(self.camera_to_worlds.shape[0])
