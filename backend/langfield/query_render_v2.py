"""PASS C (langfield-spike env): query the lifted SigLIP language field.
Loads outputs_v2/gauss_emb.npz, embeds the text query with SigLIP 2 (same joint
space), computes per-gaussian relevancy (LERF min-softmax vs negatives), and renders
the SAME composited-overlay visualization the spike used (so results are comparable).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup            # noqa: E402
from nerfstudio.models.splatfacto import get_viewmat          # noqa: E402
from gsplat import rasterization                              # noqa: E402
from transformers import AutoModel, AutoProcessor             # noqa: E402
import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
from matplotlib import cm                                     # noqa: E402
from PIL import Image                                         # noqa: E402

DEV = "cuda"
SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
CONFIG = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/rtoony/tools/langfield-spike/outputs_v2")
QUERIES = sys.argv[3].split("|") if len(sys.argv) > 3 else [
    "green bushes", "a window", "brick wall", "a lamp post",
]
NEGATIVES = ["object", "things", "stuff", "texture", "surface"]
QW, QH = 640, 480

# ── geometry (KEEP the spike load) ───────────────────────────────────────────────
config, pipeline, _, _ = eval_setup(CONFIG, test_mode="test")
m = pipeline.model.to(DEV)
means, quats = m.means.detach(), m.quats.detach()
scales = torch.exp(m.scales.detach())
opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
sh = torch.cat([m.features_dc.detach()[:, None, :], m.features_rest.detach()], dim=1)
cams = pipeline.datamanager.train_dataset.cameras.to(DEV)
fullW, fullH = int(cams.width[0]), int(cams.height[0])
# Adaptive representative views (don't hardcode indices — scenes vary in camera count).
n_cams = int(cams.camera_to_worlds.shape[0])
CAM_VIEWS = sorted({0, n_cams // 3, (2 * n_cams) // 3, n_cams - 1})[:3]

d = np.load(OUT / "gauss_emb.npz")
feat_n = torch.tensor(d["gauss_emb"], device=DEV).float()   # already L2-normed (zeros for unseen)

# ── SigLIP 2 text encoder (same joint space as the lift) ─────────────────────────
siglip = AutoModel.from_pretrained(SIGLIP_CKPT, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
sig_proc = AutoProcessor.from_pretrained(SIGLIP_CKPT)

@torch.no_grad()
def text_emb(prompts):
    inp = sig_proc(text=list(prompts), padding="max_length", max_length=64,
                   truncation=True, return_tensors="pt").to(DEV)
    feat = siglip.get_text_features(**inp).pooler_output     # [P,1152]
    return F.normalize(feat.float(), dim=-1)

neg_p = text_emb(NEGATIVES)

def relevancy_vec(query):
    q = text_emb([query])[0]
    sim_pos = feat_n @ q
    sim_neg = feat_n @ neg_p.T
    rel = torch.ones_like(sim_pos)
    for k in range(neg_p.shape[0]):
        pair = torch.stack([sim_pos, sim_neg[:, k]], dim=-1)
        rel = torch.minimum(rel, torch.softmax(pair * 10.0, dim=-1)[:, 0])
    return rel  # [N] in [0,1]

# ── render helpers (KEEP verbatim from the spike) ────────────────────────────────
def viewmat_for(i):
    return get_viewmat(cams.camera_to_worlds[i:i+1])

def K_for(i, w, h):
    K = cams.get_intrinsics_matrices()[i:i+1].clone().to(DEV)
    K[:, 0, :] *= (w / fullW); K[:, 1, :] *= (h / fullH)
    return K

def render(colors, sh_degree, i, w, h):
    out, alpha, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
        viewmats=viewmat_for(i), Ks=K_for(i, w, h), width=w, height=h, packed=False,
        near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=sh_degree,
        rasterize_mode="antialiased")
    return out, alpha

turbo = cm.get_cmap("turbo")
white = torch.ones(3, device=DEV)

rgb_cache = {}
rgb_tiles = []
for ci in CAM_VIEWS:
    out, alpha = render(sh, 3, ci, QW, QH)
    rgb = (out[0, ..., :3] + (1 - alpha[0]) * white).clamp(0, 1)
    rgb_cache[ci] = rgb
    rgb_tiles.append((rgb.cpu().numpy() * 255).astype(np.uint8))
Image.fromarray(np.concatenate(rgb_tiles, 1)).save(OUT / "_rgb_strip.png")

for q in QUERIES:
    rel = relevancy_vec(q)
    rel3 = rel[:, None].repeat(1, 3)
    tiles = []
    for ci in CAM_VIEWS:
        relmap, alpha = render(rel3, None, ci, QW, QH)
        a = alpha[0, ..., 0]
        R = relmap[0, ..., 0] / (a + 1e-6)
        valid = a > 0.5
        if valid.sum() > 100:
            lo = torch.quantile(R[valid], 0.85); hi = torch.quantile(R[valid], 0.99)
        else:
            lo, hi = R.min(), R.max()
        Rn = ((R - lo) / (hi - lo + 1e-6)).clamp(0, 1) * valid
        heat = torch.tensor(turbo(Rn.cpu().numpy())[..., :3], device=DEV, dtype=torch.float32)
        w = (0.75 * Rn)[..., None]
        overlay = (rgb_cache[ci] * (1 - w) + heat * w).clamp(0, 1)
        tiles.append((overlay.cpu().numpy() * 255).astype(np.uint8))
    safe = q.replace(" ", "_")
    Image.fromarray(np.concatenate(tiles, 1)).save(OUT / f"q_{safe}.png")
    print(f"saved q_{safe}.png  rel[max {rel.max():.2f} mean {rel.mean():.2f} p99 {torch.quantile(rel,0.99):.2f}]", flush=True)

print("QUERY_DONE")
