"""Shared checkpoint loader + gsplat render helpers for the langfield-env steps
(mirrors backend/health/render_views.py; torch.load patched for nerfstudio 1.1.5)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup            # noqa: E402
from nerfstudio.models.splatfacto import get_viewmat          # noqa: E402
from gsplat import rasterization                              # noqa: E402

DEV = "cuda"


class Splat:
    def __init__(self, config: Path):
        _, pipeline, _, _ = eval_setup(Path(config), test_mode="test")
        m = pipeline.model.to(DEV)
        self.means = m.means.detach(); self.quats = m.quats.detach()
        self.scales = torch.exp(m.scales.detach())
        self.opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
        self.sh = torch.cat([m.features_dc.detach()[:, None, :], m.features_rest.detach()], dim=1)
        self.sh_degree = int(getattr(m.config, "sh_degree", 3))
        self.mode = str(getattr(m.config, "rasterize_mode", "classic"))
        self.raw = {"f_dc": m.features_dc.detach(), "opacity": m.opacities.detach().squeeze(-1),
                    "scale": m.scales.detach(), "rot": m.quats.detach()}
        self.dm = pipeline.datamanager
        self.train = self.dm.train_dataset
        self.n = int(self.means.shape[0])

    def train_cams(self):
        return self.train.cameras.to(DEV)

    def photo_paths(self) -> list[str]:
        return [str(p) for p in getattr(self.train._dataparser_outputs, "image_filenames", [])]

    @staticmethod
    def K_from(fx, fy, cx, cy):
        return torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]], dtype=torch.float32, device=DEV)

    def render(self, c2w: torch.Tensor, K: torch.Tensor, w: int, h: int, *, mode: str = "RGB+ED",
               opac: torch.Tensor | None = None, colors: torch.Tensor | None = None, sh_degree=None,
               background: float = 1.0, packed: bool = False):
        cols = self.sh if colors is None else colors
        sh = self.sh_degree if colors is None else sh_degree
        out, alpha, info = rasterization(
            means=self.means, quats=self.quats, scales=self.scales, opacities=self.opac if opac is None else opac,
            colors=cols, viewmats=get_viewmat(c2w[None]), Ks=K, width=w, height=h, packed=packed,
            near_plane=0.01, far_plane=1e10, render_mode=mode, sh_degree=sh, rasterize_mode=self.mode,
            backgrounds=torch.full((1, 3), background, device=DEV) if (mode.startswith("RGB") and not packed) else None)
        return out[0], alpha[0, ..., 0], info


def scaled_intrinsics(cams, i: int, w: int, h: int) -> tuple[float, float, float, float]:
    K = cams.get_intrinsics_matrices()[i].clone()
    sx, sy = w / float(cams.width[i]), h / float(cams.height[i])
    return float(K[0, 0] * sx), float(K[1, 1] * sy), float(K[0, 2] * sx), float(K[1, 2] * sy)


def render_size(cams, i: int, max_width: int) -> tuple[int, int]:
    full_w, full_h = int(cams.width[i]), int(cams.height[i])
    w = min(max_width, full_w)
    return w, max(1, round(full_h * w / full_w))


def typical_spacing(means: torch.Tensor, sample: int = 4000) -> float:
    """Median nearest-neighbour distance over a random sample (one gaussian width)."""
    n = means.shape[0]
    idx = torch.randperm(n, device=means.device)[: min(sample, n)]
    pts = means[idx]
    d = torch.cdist(pts, means[torch.randperm(n, device=means.device)[: min(20000, n)]])
    d[d == 0] = float("inf")
    return float(d.min(dim=1).values.median())
