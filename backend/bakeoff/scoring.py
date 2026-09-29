"""One scorer for every arm of a trainer bake-off.

Each trainer writes its held-out renders in its own layout:

- nerfstudio `ns-eval --render-output-path`: `eval_img_NNNN.png`, the ground
  truth and the render side by side (GT left);
- Spirula `--save-eval-images 1`: `eval-gt-NNNNN.png` + `eval-render-NNNNN.png`.

Neither index is trusted to mean a particular photo. Every render is tied to
its photo by matching the trainer's own GT image against the photos on disk
(a coarse fingerprint picks the candidate, a full-resolution difference must
then confirm it), and the metrics are computed against the DISK photo, so both
arms are scored against identical pixels. A render whose GT matches no photo,
two renders claiming one photo, a photo nobody rendered, or a size mismatch
all raise: a partial score is not a score.

The metrics are nerfstudio's own objects (torchmetrics PSNR, pytorch_msssim
SSIM, torchmetrics LPIPS-alex with normalize=True, lib_bilagrid.color_correct),
imported lazily so the matching logic is testable without torch.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINGERPRINT = 24          # side of the coarse fingerprint image
MATCH_MEAN_ABS = 3.0 / 255  # full-res mean |diff| a GT must stay under to be "the same photo"
# (JPEG decoders differ by a level or two; a different photo is ~0.1+)


@dataclass
class Pair:
    key: str                 # the arm's own id for the view (file stem)
    gt: np.ndarray           # the arm's GT, HxWx3 float32 in [0,1]
    pred: np.ndarray         # the arm's render, HxWx3 float32 in [0,1]


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def pairs_nerfstudio(render_dir: Path) -> list[Pair]:
    out = []
    for p in sorted(Path(render_dir).glob("eval_img_*.png")):
        img = load_rgb(p)
        w = img.shape[1]
        if w % 2:
            raise ValueError(f"{p}: odd width {w}; not a GT|render concatenation")
        out.append(Pair(p.stem, img[:, : w // 2], img[:, w // 2:]))
    return out


def pairs_spirula(render_dir: Path) -> list[Pair]:
    out = []
    for gt_path in sorted(Path(render_dir).glob("eval-gt-*.png")):
        idx = re.fullmatch(r"eval-gt-(\d+)", gt_path.stem).group(1)
        pred_path = gt_path.with_name(f"eval-render-{idx}.png")
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        out.append(Pair(gt_path.stem, load_rgb(gt_path), load_rgb(pred_path)))
    return out


ARM_LAYOUTS = {"nerfstudio": pairs_nerfstudio, "spirula": pairs_spirula}


def fingerprint(img: np.ndarray, side: int = FINGERPRINT) -> np.ndarray:
    """Block-mean thumbnail (side x side x 3), resolution-independent."""
    h, w = img.shape[:2]
    ys = np.linspace(0, h, side + 1).astype(int)
    xs = np.linspace(0, w, side + 1).astype(int)
    return np.array([[img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean(axis=(0, 1))
                      for j in range(side)] for i in range(side)], dtype=np.float32)


def match_to_photos(pairs: list[Pair], photos: dict[str, np.ndarray]) -> dict[str, tuple[Pair, float]]:
    """name -> (pair, full-res mean |GT - photo|). Raises on any ambiguity."""
    names = list(photos)
    prints = np.stack([fingerprint(photos[n]) for n in names])
    matched: dict[str, tuple[Pair, float]] = {}
    shapes = {photos[n].shape for n in names}
    for pair in pairs:
        if pair.gt.shape not in shapes or pair.pred.shape != pair.gt.shape:
            raise ValueError(f"{pair.key}: size gt {pair.gt.shape} / render {pair.pred.shape} "
                             f"vs photo sizes {sorted(shapes)}")
        d = np.abs(prints - fingerprint(pair.gt)[None]).mean(axis=(1, 2, 3))
        d[[photos[n].shape != pair.gt.shape for n in names]] = np.inf
        name = names[int(np.argmin(d))]
        photo = photos[name]
        err = float(np.abs(photo - pair.gt).mean())
        if err > MATCH_MEAN_ABS:
            raise ValueError(f"{pair.key}: nearest photo {name} differs by {err * 255:.2f}/255 mean; "
                             "this render's GT is not one of the held-out photos")
        if name in matched:
            raise ValueError(f"{pair.key} and {matched[name][0].key} both match photo {name}")
        matched[name] = (pair, err)
    unrendered = sorted(set(names) - set(matched))
    if unrendered:
        raise ValueError(f"{len(unrendered)} held-out photos have no render, e.g. {unrendered[:3]}")
    return matched


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


class Metrics:
    """nerfstudio's splatfacto eval metrics, one instance per run."""

    def __init__(self, device: str = "cuda"):
        import torch
        from nerfstudio.model_components.lib_bilagrid import color_correct
        from pytorch_msssim import SSIM
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.torch, self.color_correct, self.device = torch, color_correct, device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    def __call__(self, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        t = self.torch
        with t.no_grad():
            g = t.from_numpy(gt).to(self.device)
            p = t.from_numpy(pred).to(self.device)
            cc = self.color_correct(p, g)
            out = {}
            for tag, img in (("", p), ("cc_", cc)):
                a = t.moveaxis(g, -1, 0)[None]
                b = t.moveaxis(img, -1, 0)[None].clamp(0, 1)
                out[tag + "psnr"] = float(self.psnr(a, b).item())
                out[tag + "ssim"] = float(self.ssim(a, b).item())
                out[tag + "lpips"] = float(self.lpips(a, b).item())
            return out


def score(pairs: list[Pair], photos: dict[str, np.ndarray], metrics) -> dict:
    matched = match_to_photos(pairs, photos)
    per_view = []
    for name in sorted(matched):
        pair, err = matched[name]
        row = {"photo": name, "arm_key": pair.key, "gt_match_mean_abs_255": round(err * 255, 3)}
        row.update(metrics(photos[name], pair.pred))
        per_view.append(row)
    keys = [k for k in per_view[0] if k not in ("photo", "arm_key", "gt_match_mean_abs_255")]
    mean = {k: float(np.mean([r[k] for r in per_view])) for k in keys}
    std = {k: float(np.std([r[k] for r in per_view])) for k in keys}
    return {"v": 1, "views": len(per_view), "mean": mean, "std": std,
            "max_gt_match_mean_abs_255": max(r["gt_match_mean_abs_255"] for r in per_view),
            "per_view": per_view}
