"""PASS B (langfield-spike env): training-free Occam's-LGS language-field lift.

For each of the 116 views: SAM 2.1 masks (PASS A) -> SigLIP 2 region embedding per
mask -> project the 545k frozen gaussians (gsplat packed meta) -> ED depth occlusion
gate -> assign each gaussian's center to the SAM mask it lands in -> accumulate the
opacity-weighted SigLIP feature. No optimizer, no loss: ~116x2 forward rasterizations.
Output: outputs_v2/gauss_emb.npz (per-gaussian 1152-D SigLIP feature, fp16).

Geometry + cameras + the get_viewmat convention are loaded from the SAME nerfstudio
pipeline (co-framed; pose-correct — proven in the spike).
"""
import os, sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from nerfstudio.utils.eval_utils import eval_setup            # noqa: E402
from nerfstudio.models.splatfacto import get_viewmat          # noqa: E402
from gsplat import rasterization                              # noqa: E402
from transformers import AutoModel, AutoProcessor             # noqa: E402

DEV = "cuda"
DEPTH_TOL = float(os.environ.get("LANGFIELD_DEPTH_TOL", "0.07"))  # front-surface occlusion tolerance; raise for dense scenes
SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
EMB_BATCH = 64

CONFIG = Path(sys.argv[1])
FEAT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/rtoony/tools/langfield-spike/features")
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/home/rtoony/tools/langfield-spike/outputs_v2")
OUT.mkdir(parents=True, exist_ok=True)

# ── geometry + cameras (KEEP the spike's pose-correct load) ──────────────────────
config, pipeline, _, _ = eval_setup(CONFIG, test_mode="test")
m = pipeline.model.to(DEV)
means, quats = m.means.detach(), m.quats.detach()
scales = torch.exp(m.scales.detach())
opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
N = means.shape[0]
cams = pipeline.datamanager.train_dataset.cameras.to(DEV)
fullW, fullH = int(cams.width[0]), int(cams.height[0])

meta = json.load(open(FEAT / "frames_meta.json"))
n_frames, LIFT_W, LIFT_H = meta["n"], meta["W"], meta["H"]
print(f"[lift] {N} gaussians, {n_frames} views @ {LIFT_W}x{LIFT_H}", flush=True)

def viewmat_for(i):
    return get_viewmat(cams.camera_to_worlds[i:i+1])

def K_for(i, w, h):
    K = cams.get_intrinsics_matrices()[i:i+1].clone().to(DEV)
    K[:, 0, :] *= (w / fullW); K[:, 1, :] *= (h / fullH)
    return K

# ── SigLIP 2 region encoder ──────────────────────────────────────────────────────
siglip = AutoModel.from_pretrained(SIGLIP_CKPT, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
sig_proc = AutoProcessor.from_pretrained(SIGLIP_CKPT)
D = siglip.config.text_config.projection_size   # 1152
assert D == 1152, f"unexpected SigLIP dim {D}"

@torch.no_grad()
def embed_crops(pil_crops):
    """list[PIL] -> L2-normed [M, 1152] in the joint text space."""
    outs = []
    for s in range(0, len(pil_crops), EMB_BATCH):
        batch = pil_crops[s:s+EMB_BATCH]
        inp = sig_proc(images=batch, return_tensors="pt").to(DEV)
        feat = siglip.get_image_features(**inp).pooler_output     # [b,1152]
        outs.append(F.normalize(feat.float(), dim=-1))
    return torch.cat(outs, 0) if outs else torch.zeros(0, D, device=DEV)

def view_region_embeds(view_i, label_np):
    """Build [maxlabel+1, 1152] region embeddings for the masks present in this view."""
    img = np.array(Image.open(FEAT / "frames" / f"frame_{view_i:03d}.png").convert("RGB"))
    present = np.unique(label_np)
    present = present[present >= 0]
    if present.size == 0:
        return torch.zeros(1, D, device=DEV)
    crops, ks = [], []
    for k in present.tolist():
        ys, xs = np.where(label_np == k)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = img[y0:y1, x0:x1].copy()
        cmask = (label_np[y0:y1, x0:x1] == k)
        crop[~cmask] = 0                                  # bg-zeroed region crop
        crops.append(Image.fromarray(crop)); ks.append(k)
    emb = embed_crops(crops)                              # [len(ks),1152]
    table = torch.zeros(int(label_np.max()) + 1, D, device=DEV)
    table[torch.tensor(ks, device=DEV)] = emb
    return table

# ── training-free lift ───────────────────────────────────────────────────────────
Facc = torch.zeros(N, D, device=DEV)
Wsum = torch.zeros(N, 1, device=DEV)
seen = torch.zeros(N, device=DEV, dtype=torch.bool)
zeros3 = means.new_zeros(N, 3)

for v in range(n_frames):
    label_np = np.load(FEAT / "masks" / f"view_{v:03d}.npz")["label"]
    assert label_np.shape == (LIFT_H, LIFT_W), f"mask res {label_np.shape} != {(LIFT_H, LIFT_W)}"
    label_map = torch.from_numpy(label_np.astype(np.int64)).to(DEV)
    embeds = view_region_embeds(v, label_np)             # [maxlabel+1, 1152]

    # (A) project: packed meta gives per-visible-gaussian center + depth + opacity
    _, _, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=zeros3,
        viewmats=viewmat_for(v), Ks=K_for(v, LIFT_W, LIFT_H), width=LIFT_W, height=LIFT_H,
        packed=True, near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=None,
        rasterize_mode="antialiased")
    gid = info["gaussian_ids"]; uv = info["means2d"]; dz = info["depths"]; op_g = info["opacities"]

    # (B) ED depth buffer = front-surface occlusion oracle
    ed, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opac, colors=zeros3,
        viewmats=viewmat_for(v), Ks=K_for(v, LIFT_W, LIFT_H), width=LIFT_W, height=LIFT_H,
        packed=False, near_plane=0.01, far_plane=1e10, render_mode="ED", sh_degree=None,
        rasterize_mode="antialiased")
    depth_buf = ed[0, ..., 0]                             # [H,W]

    # (C) in-frame gate (means2d can be <0 or >=W/H -> mask, don't clamp)
    px = uv[:, 0].round().long(); py = uv[:, 1].round().long()
    inf = (px >= 0) & (px < LIFT_W) & (py >= 0) & (py < LIFT_H)
    gid, dz, op_g, px, py = gid[inf], dz[inf], op_g[inf], px[inf], py[inf]

    # (D) occlusion gate + (E) mask assignment
    front_d = depth_buf[py, px]
    occ_ok = dz <= front_d * (1.0 + DEPTH_TOL)
    lab = label_map[py, px]
    valid = occ_ok & (lab >= 0)

    g = gid[valid]
    w = op_g[valid].unsqueeze(-1)
    emb = embeds[lab[valid]]
    # (F) opacity-weighted scatter-accumulate (index_add_ sums duplicate gaussians)
    Facc.index_add_(0, g, w * emb)
    Wsum.index_add_(0, g, w)
    seen[g] = True
    if (v + 1) % 20 == 0 or v == 0:
        print(f"  view {v}: {int(valid.sum())} gaussian-mask hits | seen so far {int(seen.sum())}", flush=True)

# ── finalize ─────────────────────────────────────────────────────────────────────
# Chunked, IN-PLACE normalize. 30k-iter scenes carry ~1.5-2M gaussians; the naive
# `F.normalize(Facc / Wsum)` allocates several full [N,1152] temporaries at once and
# OOMs a 32GB card. Divide in place, then L2-normalize row-chunks in place, so peak
# VRAM stays ~one Facc + one chunk instead of ~3x Facc.
Facc.div_(Wsum + 1e-15)
_NORM_CHUNK = 200_000
for _s in range(0, N, _NORM_CHUNK):
    _e = min(_s + _NORM_CHUNK, N)
    Facc[_s:_e] = F.normalize(Facc[_s:_e], dim=-1)
Facc[~seen] = 0.0
gauss_emb = Facc

frac_seen = seen.float().mean().item()
frac_w = (Wsum.squeeze(-1) > 0).float().mean().item()
print(f"[lift] seen={frac_seen:.1%}  got-feature={frac_w:.1%}", flush=True)
assert gid.max() < N, "gaussian_ids out of range — packed indexing broken"
_SEEN_MIN = float(os.environ.get("LANGFIELD_SEEN_MIN", "0.5"))
assert frac_seen > _SEEN_MIN, f"only {frac_seen:.1%} gaussians seen — occlusion gate too tight (raise DEPTH_TOL)"
assert frac_w > _SEEN_MIN, f"only {frac_w:.1%} gaussians got a mask — PASS A handoff / res mismatch"

np.savez_compressed(OUT / "gauss_emb.npz",
         gauss_emb=gauss_emb.half().cpu().numpy(),
         seen=seen.cpu().numpy(),
         model_id=SIGLIP_CKPT, lift_res=np.array([LIFT_W, LIFT_H]))
print(f"LIFT_DONE -> {OUT/'gauss_emb.npz'}", flush=True)
