"""WARM QUERY WORKER for Splat Lab's Language Field (P3).

Keeps the SigLIP 2 text encoder + an LRU of 1-2 lifted scenes resident so a
language query renders a relevancy heatmap-overlay strip in ~a second instead of
re-paying the cold eval_setup + model-load every time.

Relevancy + overlay math is COPIED VERBATIM from
``backend/langfield/query_render_v2.py`` (the cold renderer) so the warm path's
``q_<safe>.png`` is pixel-identical to the offline recipe. We copy rather than
import because query_render_v2 runs a full render at module import (it's a
script); copying the pure functions keeps the result identical with no side
effects.

GPU SAFETY: the SigLIP text-encode and the per-gaussian cosine relevancy run
LOCKLESS (cheap, ~no VRAM beyond the resident text encoder). The cross-process
``gpu_arbiter.HEAVY_GPU_LOCK`` is taken ONLY around the gsplat ``rasterization``
calls, with ``set_holder("langfield", <scene_id>)`` + ``acquire_gpu(4000)``, so a
query WAITS behind a training run or a TRELLIS job and never preempts/OOMs them.

Run in the ``langfield-spike`` conda env. Stateless across restarts: the scene
cache rebuilds from ``lfdir`` + ``config`` on demand, and idle-evicts after ~10
min so a long training run can reclaim the card.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

# gpu_arbiter is a sibling module (backend/gpu_arbiter.py). Import it the same way
# whether we're launched as ``backend.langfield_worker`` (uvicorn from repo root)
# or as a bare module — so the warm path shares the SAME Redis lock keys as the
# portal/splat lane.
try:  # package-relative first (uvicorn backend.langfield_worker:app)
    from . import gpu_arbiter  # type: ignore
except Exception:  # pragma: no cover - fallback for direct/script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gpu_arbiter  # type: ignore

log = logging.getLogger("langfield_worker")

# ── tunables ─────────────────────────────────────────────────────────────────────
DEV = "cuda"
SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
NEGATIVES = ["object", "things", "stuff", "texture", "surface"]

# Curated open-vocabulary noun list for the "what's in this scene" inventory. We score
# each word's per-gaussian relevancy against the scene, rank by presence, and keep the
# top-N. Bump INVENTORY_VERSION when this list or the scoring changes (invalidates cache).
INVENTORY_VERSION = 7
VOCAB = [
    # furniture / indoor
    "chair", "table", "desk", "sofa", "couch", "bed", "stool", "bench", "shelf",
    "bookshelf", "cabinet", "drawer", "counter", "countertop", "wardrobe", "dresser",
    # kitchen / dining
    "refrigerator", "oven", "stove", "microwave", "sink", "faucet", "toaster", "kettle",
    "pan", "pot", "bowl", "plate", "cup", "mug", "glass", "bottle", "jar", "can",
    "knife", "fork", "spoon", "cutting board", "dish", "napkin", "tray",
    # decor / small objects
    "lamp", "light", "candle", "vase", "painting", "picture frame", "mirror", "clock",
    "book", "magazine", "box", "basket", "bag", "pillow", "cushion", "blanket", "rug",
    "curtain", "toy", "doll", "lego", "ball", "camera", "phone", "remote", "speaker",
    # electronics
    "television", "monitor", "computer", "laptop", "keyboard", "mouse",
    # plants / nature
    "plant", "potted plant", "flower", "leaf", "tree", "bush",
    "grass", "moss", "branch", "trunk", "log", "stump", "vine", "fern",
    # food
    "fruit", "apple", "banana", "orange", "bread", "food", "vegetable",
    # outdoor / structural
    "building", "house", "roof", "wall", "brick wall", "window", "door", "gate",
    "fence", "stairs", "railing", "pillar", "column", "floor", "ceiling", "ground",
    "path", "sidewalk", "road", "gravel", "pavement",
    # outdoor objects
    "car", "bicycle", "wheel", "bench seat", "statue", "sculpture", "fountain",
    "pot", "planter", "umbrella", "sign", "pole", "lamp post", "trash can", "ladder",
    "bucket", "hose", "rope", "flag", "tent",
    # materials / surfaces
    "wood", "metal", "glass surface", "stone", "concrete", "fabric", "tile", "carpet",
    # scene elements
    "sky", "cloud", "water", "pond", "shadow", "reflection", "person",
    # musical / misc
    "piano", "guitar", "instrument",
    # small but distinctive handheld objects — surfaced by the smart ranking even at
    # low presence (a set of keys is tiny but salient).
    "keys", "keychain", "wallet", "watch", "ring", "coin", "pen", "pencil", "marker",
    "scissors", "cable", "charger", "headphones", "earbuds", "glasses", "sunglasses",
    "controller", "figurine", "trophy", "medal", "brush", "comb", "razor", "toothbrush",
    "spray bottle", "tube", "tin", "flashlight", "lighter", "matchbox", "usb drive",
    "battery", "lightbulb", "screwdriver", "hammer", "wrench", "pliers", "tape",
    "stapler", "calculator", "wallet", "purse", "hat", "cap", "shoe", "boot", "glove",
    "sock", "belt", "tie", "scarf", "mask", "helmet", "backpack", "notebook", "folder",
    "envelope", "card", "dice", "chess piece", "domino", "marble", "token", "whistle",
]
VOCAB = list(dict.fromkeys(VOCAB))   # de-dupe (a few words appear in >1 group)
QW, QH = 640, 480
THUMB = 224                    # per-match result thumbnail side (square, px)
QUERY_VRAM_MB = 4000            # headroom we ask acquire_gpu() to guarantee per render
SCENE_CACHE_MAX = int(os.environ.get("SPLAT_LANGFIELD_SCENE_CACHE", "2"))  # LRU size (1-2)
IDLE_EVICT_SEC = float(os.environ.get("SPLAT_LANGFIELD_IDLE_SEC", str(10 * 60)))  # ~10 min
IDLE_CHECK_SEC = 30.0
PORT = int(os.environ.get("SPLAT_LANGFIELD_WORKER_PORT", "3417"))
HOST = "127.0.0.1"             # LAN-only

# ── heavy deps are guarded so this file at least PARSES/imports for tooling that
#    can't load torch/gsplat. The real service (langfield-spike env) gets them. ──
try:
    import numpy as np
    import torch
    import torch.nn.functional as F

    _orig_load = torch.load

    def _patched_load(*a, **k):
        # SAME trust-load patch as query_render_v2.py — nerfstudio checkpoints
        # are full pickles, not weights_only.
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    torch.load = _patched_load

    from nerfstudio.utils.eval_utils import eval_setup
    from nerfstudio.models.splatfacto import get_viewmat
    from gsplat import rasterization
    from transformers import AutoModel, AutoProcessor
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    from PIL import Image

    _HEAVY_OK = True
    _HEAVY_ERR: Exception | None = None
except Exception as _e:  # pragma: no cover - lets py_compile / ast.parse pass anywhere
    _HEAVY_OK = False
    _HEAVY_ERR = _e
    np = torch = F = None  # type: ignore
    eval_setup = get_viewmat = rasterization = None  # type: ignore
    AutoModel = AutoProcessor = cm = Image = None  # type: ignore

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel


# ── relevancy + overlay math — COPIED VERBATIM from query_render_v2.py ────────────
# These are the only numerical primitives that decide what q_<safe>.png looks like.
# Keep byte-for-byte identical to the cold renderer.

def relevancy_vec(feat_n, q_emb, neg_p):
    """Per-gaussian LERF min-softmax relevancy in [0,1]. (query_render_v2.relevancy_vec)

    feat_n: [N,1152] L2-normed gaussian embeddings (zeros for unseen)
    q_emb:  [1152]   L2-normed query text embedding
    neg_p:  [K,1152] L2-normed negative-prompt embeddings
    """
    sim_pos = feat_n @ q_emb
    sim_neg = feat_n @ neg_p.T
    rel = torch.ones_like(sim_pos)
    for k in range(neg_p.shape[0]):
        pair = torch.stack([sim_pos, sim_neg[:, k]], dim=-1)
        rel = torch.minimum(rel, torch.softmax(pair * 10.0, dim=-1)[:, 0])
    return rel  # [N] in [0,1]


def _make_render(means, quats, scales, opac, cams, fullW, fullH):
    """Build the viewmat/K/render closures (verbatim from query_render_v2.py)."""

    def viewmat_for(i):
        return get_viewmat(cams.camera_to_worlds[i:i + 1])

    def K_for(i, w, h):
        K = cams.get_intrinsics_matrices()[i:i + 1].clone().to(DEV)
        K[:, 0, :] *= (w / fullW)
        K[:, 1, :] *= (h / fullH)
        return K

    def render(colors, sh_degree, i, w, h):
        out, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=viewmat_for(i), Ks=K_for(i, w, h), width=w, height=h, packed=False,
            near_plane=0.01, far_plane=1e10, render_mode="RGB", sh_degree=sh_degree,
            rasterize_mode="antialiased")
        return out, alpha

    return render


# ── scene cache ──────────────────────────────────────────────────────────────────
class Scene:
    """One lifted scene resident on the GPU, keyed by config_path."""

    def __init__(self, config_path: str, lfdir: str):
        self.config_path = config_path
        self.lfdir = lfdir
        self.last_used = time.monotonic()
        # refcount of in-flight queries holding this scene; the idle-evictor must
        # NOT free a scene that is mid-query (e.g. waiting on HEAVY_GPU_LOCK behind a
        # long training run, which can exceed IDLE_EVICT_SEC).
        self.in_use = 0

        config_p = Path(config_path)
        lf_p = Path(lfdir)

        # geometry (KEEP the spike load — same as query_render_v2.py)
        config, pipeline, _, _ = eval_setup(config_p, test_mode="test")
        m = pipeline.model.to(DEV)
        self.means = m.means.detach()
        self.quats = m.quats.detach()
        self.scales = torch.exp(m.scales.detach())
        self.opac = torch.sigmoid(m.opacities.detach()).squeeze(-1)
        self.sh = torch.cat(
            [m.features_dc.detach()[:, None, :], m.features_rest.detach()], dim=1)
        self.cams = pipeline.datamanager.train_dataset.cameras.to(DEV)
        self.fullW = int(self.cams.width[0])
        self.fullH = int(self.cams.height[0])
        self.n_cams = int(self.cams.camera_to_worlds.shape[0])

        # per-gaussian lifted SigLIP features (already L2-normed; zeros for unseen)
        d = np.load(lf_p / "gauss_emb.npz")
        self.feat_n = torch.tensor(d["gauss_emb"], device=DEV).float()

        self.render = _make_render(
            self.means, self.quats, self.scales, self.opac,
            self.cams, self.fullW, self.fullH)

        # keep the model only via tensors we pulled; drop pipeline ref
        del pipeline, m

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def free(self) -> None:
        """Release GPU tensors so a training run can reclaim the card."""
        for attr in ("means", "quats", "scales", "opac", "sh", "feat_n", "cams"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self.render = None
        if _HEAVY_OK:
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass


class WorkerState:
    def __init__(self) -> None:
        self.siglip = None
        self.sig_proc = None
        self.neg_p = None                      # cached negative-prompt embeddings
        self.scenes: "OrderedDict[str, Scene]" = OrderedDict()
        self.lock = asyncio.Lock()             # serialize scene build/evict (not render)
        self.ready = False

    # ── SigLIP (resident, loaded once) ──────────────────────────────────────────
    def load_siglip(self) -> None:
        self.siglip = AutoModel.from_pretrained(
            SIGLIP_CKPT, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
        self.sig_proc = AutoProcessor.from_pretrained(SIGLIP_CKPT)
        self.neg_p = self._text_emb(NEGATIVES)
        self.ready = True
        log.info("SigLIP %s resident; negatives cached", SIGLIP_CKPT)

    @torch.no_grad() if _HEAVY_OK else (lambda f: f)
    def _text_emb(self, prompts):
        """LOCKLESS text encode — verbatim from query_render_v2.text_emb."""
        inp = self.sig_proc(text=list(prompts), padding="max_length", max_length=64,
                            truncation=True, return_tensors="pt").to(DEV)
        feat = self.siglip.get_text_features(**inp).pooler_output     # [P,1152]
        return F.normalize(feat.float(), dim=-1)

    # ── scene LRU ────────────────────────────────────────────────────────────────
    def get_scene_cached(self, config_path: str) -> "Scene | None":
        sc = self.scenes.get(config_path)
        if sc is not None:
            sc.touch()
            self.scenes.move_to_end(config_path)
        return sc

    def insert_scene(self, sc: Scene) -> None:
        self.scenes[sc.config_path] = sc
        self.scenes.move_to_end(sc.config_path)
        while len(self.scenes) > SCENE_CACHE_MAX:
            _key, old = self.scenes.popitem(last=False)  # evict LRU
            log.info("evicting LRU scene %s", old.config_path)
            old.free()

    def evict_idle(self) -> list[str]:
        now = time.monotonic()
        dropped = []
        for key in list(self.scenes.keys()):
            sc = self.scenes[key]
            if sc.in_use <= 0 and now - sc.last_used >= IDLE_EVICT_SEC:
                self.scenes.pop(key)
                sc.free()
                dropped.append(key)
        return dropped


STATE = WorkerState()


# ── relevancy heatmap-overlay strip — overlay math VERBATIM from query_render_v2 ──
def _compute_relevancy(state: WorkerState, sc: Scene, text: str):
    """LOCKLESS: query text embed + per-gaussian LERF relevancy. Cheap, no gsplat,
    ~no extra VRAM beyond the resident text encoder. Returns rel3 = [N,3]."""
    q = state._text_emb([text])[0]
    rel = relevancy_vec(sc.feat_n, q, state.neg_p)
    return rel[:, None].repeat(1, 3)


def _cluster_extent(pts_np, c):
    import numpy as np
    return float(np.clip(np.quantile(np.linalg.norm(pts_np - c, axis=1), 0.75), 0.08, 1.5))


def _relevancy_focus(sc, rel3):
    """Cluster the top-relevancy gaussians into DISTINCT 3D instances (multiple
    references), in the .ply/viewer frame (sc.means == exported-.ply coords). LOCKLESS,
    cheap. Returns the primary focus (back-compat) + a `matches` list the viewer uses to
    show clickable results / highlight each instance. DBSCAN with a scale-adaptive eps."""
    import numpy as np
    from sklearn.cluster import DBSCAN

    rel = rel3[:, 0]
    n = rel.shape[0]
    # Wide-ish pool (lower threshold) so distinct instances are all represented, then
    # keep only clusters with a genuine strong CORE — that separates real repeats from
    # the diffuse tail.
    k = min(4000, n)
    topv, topi = torch.topk(rel, k)
    keep = topv > 0.42
    idx = topi[keep] if int(keep.sum()) >= 30 else topi[: min(80, n)]
    pts = sc.means[idx].cpu().numpy()
    wts = rel[idx].cpu().numpy()

    spread = float(np.linalg.norm(pts.std(0)) + 1e-6)
    eps = float(np.clip(spread * 0.15, 0.05, 0.8))
    labels = DBSCAN(eps=eps, min_samples=max(6, len(pts) // 60)).fit_predict(pts)

    matches = []
    for lab in set(labels.tolist()):
        if lab == -1:
            continue
        m = labels == lab
        if float(wts[m].max()) < 0.58:            # each instance needs a real strong core
            continue
        cp = pts[m]
        c = cp.mean(0)
        matches.append({
            "focus": [float(x) for x in c],
            "radius": _cluster_extent(cp, c),
            "score": float(wts[m].sum()),
            "count": int(m.sum()),
        })
    if not matches:                               # nothing distinct -> single centroid
        c = pts.mean(0)
        matches = [{"focus": [float(x) for x in c], "radius": _cluster_extent(pts, c),
                    "score": float(wts.sum()), "count": len(pts)}]
    matches.sort(key=lambda x: x["score"], reverse=True)
    matches = matches[:6]
    # pick the training camera that best FRAMES each instance, so its result
    # thumbnail actually looks at that object (not a fixed default view).
    for m in matches:
        m["view"] = _best_view_for(sc, m["focus"])
    top = matches[0]
    return {"focus": top["focus"], "radius": top["radius"], "matches": matches}


def _best_view_for(sc, focus) -> int:
    """Index of the training camera that best frames `focus` (viewer/.ply coords):
    the one whose forward axis points most directly at it (in front + centered),
    lightly preferring a closer camera on ties. LOCKLESS (reads the cams tensor)."""
    import numpy as np
    c2w = sc.cams.camera_to_worlds.detach().cpu().numpy()   # [N,3,4]
    pos = c2w[:, :3, 3]
    fwd = -c2w[:, :3, 2]                                     # nerfstudio: camera looks down -Z
    fwd = fwd / (np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9)
    d = np.asarray(focus, dtype=np.float64)[None, :] - pos
    dist = np.linalg.norm(d, axis=1) + 1e-9
    cos = (d / dist[:, None] * fwd).sum(1)                   # centered-ness; >0 = in front
    score = cos - 0.03 * (dist / (dist.max() + 1e-9))       # tie-break toward closer
    return int(np.argmax(score))


def _scene_inventory(state: "WorkerState", sc: Scene, topn: int = 20) -> list[dict]:
    """Open-vocabulary "what's in this scene" inventory. Scores every VOCAB word's
    per-gaussian relevancy against the scene, ranks by PRESENCE (share of gaussians the
    object occupies), and returns the top-N with presence + reliability (peak confidence)
    + clustered instances (for the toggle-to-highlight legend). LOCKLESS — cheap matmuls
    only, no gsplat render. sim_neg is word-independent so it's computed once."""
    import numpy as np
    feat_n = sc.feat_n                                  # [N,1152] L2-normed
    n = int(feat_n.shape[0])
    emb = state._text_emb(VOCAB)                        # [V,1152]
    neg = state.neg_p                                   # [K,1152]
    sim_neg = feat_n @ neg.T                            # [N,K] — same for every word

    # Scene solid-core center + radius (opacity-weighted). The sharp, well-reconstructed
    # SUBJECT (a toy on a table) sits here with opaque, tight gaussians; peripheral
    # floaters are low-opacity and far. Used to reward the clearly-rendered subject over
    # diffuse background surfaces in the ranking.
    op = sc.opac                                        # [N] in [0,1]
    wnorm = op / (op.sum() + 1e-9)
    core = (sc.means * wnorm[:, None]).sum(0)           # [3]
    core_r = float(torch.sqrt((((sc.means - core) ** 2).sum(1) * wnorm).sum()) + 1e-6)

    scored = []
    for i, word in enumerate(VOCAB):
        sim_pos = feat_n @ emb[i]                       # [N]
        rel = torch.ones_like(sim_pos)
        for k in range(neg.shape[0]):
            pair = torch.stack([sim_pos, sim_neg[:, k]], dim=-1)
            rel = torch.minimum(rel, torch.softmax(pair * 10.0, dim=-1)[:, 0])
        # unseen gaussians (zero feature) sit at exactly 0.5, so >0.55 counts a genuine match.
        strong = rel > 0.55
        cnt = int(strong.sum())
        if cnt < 12:
            continue
        reliab = float(torch.topk(rel, min(200, n)).values.mean())
        if reliab < 0.53:
            continue
        pres = cnt / max(n, 1)
        # RENDER-QUALITY / subject boost. A clearly-rendered object is SOLID (opaque),
        # CENTRAL (near the sharp core), and COMPACT (tight gaussians — a real localized
        # thing, not a diffuse match smeared across the scene). Each damps the diffuse
        # low-opacity background so the in-focus subject rises.
        om = sc.means[strong]
        solidity = float(op[strong].mean())
        centrality = float(torch.exp(-torch.linalg.norm(om.mean(0) - core) / core_r))
        spread = float(torch.linalg.norm(om.std(0)))
        compactness = float(np.exp(-spread / (core_r + 1e-6)))
        quality = (0.35 + 0.65 * solidity) * (0.45 + 0.75 * centrality) * (0.4 + 0.6 * compactness)
        # confidence-led, gently coverage-boosted, subject-weighted: a small sharp subject
        # outranks a big diffuse background surface.
        score = reliab * (0.55 + pres ** 0.5) * quality
        scored.append((word, pres, reliab, score, strong))
    scored.sort(key=lambda x: x[3], reverse=True)       # rank by smart score

    items = []
    for word, pres, reliab, score, strong in scored[:topn]:
        idx = torch.nonzero(strong, as_tuple=False).squeeze(-1)
        om = sc.means[idx]
        # robust center + extent: trim the scattered outlier tail so a spurious few
        # gaussians don't inflate the object or muddy the area wash / zoom framing.
        c0 = om.mean(0)
        d = torch.linalg.norm(om - c0, dim=1)
        keep = d <= torch.quantile(d, 0.85)
        if int(keep.sum()) >= 8:
            idx, om = idx[keep], om[keep]
        center = om.mean(0)
        extent = max(float(torch.quantile(torch.linalg.norm(om - center, dim=1), 0.9)), 0.08)
        # AREA highlight + zoom framing use the object's own gaussians (sampled).
        sel = idx if int(idx.numel()) <= 200 else idx[torch.randperm(int(idx.numel()), device=idx.device)[:200]]
        pts = sc.means[sel].detach().cpu().numpy()
        items.append({
            "label": word,
            "presence": round(pres, 4),
            "reliability": round(reliab, 3),
            "focus": [float(x) for x in center],     # whole-object center (zoom pivot)
            "radius": round(extent, 3),              # whole-object extent (zoom framing)
            "count": int(idx.numel()),
            "points": [[float(x) for x in p] for p in pts],
            "matches": [{"focus": [float(x) for x in center], "radius": round(extent, 3),
                         "score": float(score), "count": int(idx.numel())}],
        })
    return items


def _render_view_overlay(sc: Scene, rel3, ci: int):
    """LOCKED (caller holds HEAVY_GPU_LOCK): render one camera view as an RGB base with
    the relevancy heatmap composited over it. Overlay math VERBATIM from query_render_v2.
    Returns an HxWx3 uint8 array."""
    turbo = cm.get_cmap("turbo")
    white = torch.ones(3, device=DEV)
    out, alpha = sc.render(sc.sh, 3, ci, QW, QH)
    rgb = (out[0, ..., :3] + (1 - alpha[0]) * white).clamp(0, 1)
    relmap, alpha = sc.render(rel3, None, ci, QW, QH)
    a = alpha[0, ..., 0]
    R = relmap[0, ..., 0] / (a + 1e-6)
    valid = a > 0.5
    if valid.sum() > 100:
        lo = torch.quantile(R[valid], 0.85)
        hi = torch.quantile(R[valid], 0.99)
    else:
        lo, hi = R.min(), R.max()
    Rn = ((R - lo) / (hi - lo + 1e-6)).clamp(0, 1) * valid
    heat = torch.tensor(turbo(Rn.cpu().numpy())[..., :3], device=DEV, dtype=torch.float32)
    w = (0.75 * Rn)[..., None]
    overlay = (rgb * (1 - w) + heat * w).clamp(0, 1)
    return (overlay.cpu().numpy() * 255).astype(np.uint8)


def _render_hero_locked(sc: Scene, out_path: Path, long_side: int = 512) -> None:
    """LOCKED (caller holds HEAVY_GPU_LOCK): render ONE clean RGB hero view (a
    representative training camera) as a webp — a real splat image for the gallery
    thumbnail, framed by the scene's own aspect (no heatmap)."""
    ci = sc.n_cams // 2
    w_px = long_side
    h_px = max(1, int(round(long_side * sc.fullH / max(sc.fullW, 1))))
    white = torch.ones(3, device=DEV)
    out, alpha = sc.render(sc.sh, 3, ci, w_px, h_px)
    rgb = (out[0, ..., :3] + (1 - alpha[0]) * white).clamp(0, 1)
    Image.fromarray((rgb.cpu().numpy() * 255).astype(np.uint8)).save(out_path, "WEBP", quality=88)


def _project_point(sc: Scene, focus, ci: int):
    """Project a world/.ply point into the ci-th camera's QWxQH image. Returns
    (u, v, depth, fx) or None if the point is behind the camera."""
    viewmat = get_viewmat(sc.cams.camera_to_worlds[ci:ci + 1])[0]        # world->cam [4,4]
    K = sc.cams.get_intrinsics_matrices()[ci:ci + 1].clone().to(DEV)[0]  # [3,3]
    K[0, :] *= (QW / sc.fullW)
    K[1, :] *= (QH / sc.fullH)
    f = torch.tensor([float(focus[0]), float(focus[1]), float(focus[2]), 1.0],
                     device=DEV, dtype=viewmat.dtype)
    pc = viewmat @ f
    z = float(pc[2])
    if z <= 1e-3:
        return None
    u = float(K[0, 0] * pc[0] / pc[2] + K[0, 2])
    v = float(K[1, 1] * pc[1] / pc[2] + K[1, 2])
    return u, v, z, float(K[0, 0])


def _render_match_thumbs_locked(sc: Scene, rel3, matches: list[dict], text: str) -> list[str]:
    """LOCKED: one result thumbnail per match, each rendered from the camera that best
    frames it (match['view']) and cropped to the object. The primary match is saved as
    q_<safe>.png (so the app's heatmap_url + existence check resolve), the rest as
    q_<safe>_<i>.png. Returns the filenames aligned to `matches`."""
    safe = _safe_name(text)
    overlays: dict[int, Any] = {}     # cache per camera — matches often share a view
    names: list[str] = []
    for i, m in enumerate(matches):
        ci = int(m.get("view", 0))
        ci = ci if 0 <= ci < sc.n_cams else 0
        if ci not in overlays:
            overlays[ci] = _render_view_overlay(sc, rel3, ci)
        img = overlays[ci]
        H, W = img.shape[:2]
        crop = img
        proj = _project_point(sc, m["focus"], ci)
        if proj is not None:
            u, v, z, fx = proj
            s = float(np.clip(fx * 2.4 * float(m["radius"]) / z, 150.0, float(min(W, H))))
            x0 = int(np.clip(u - s / 2, 0, W - 1)); x1 = int(np.clip(u + s / 2, 1, W))
            y0 = int(np.clip(v - s / 2, 0, H - 1)); y1 = int(np.clip(v + s / 2, 1, H))
            if x1 - x0 >= 24 and y1 - y0 >= 24:
                crop = img[y0:y1, x0:x1]
        name = f"q_{safe}.png" if i == 0 else f"q_{safe}_{i}.png"
        Image.fromarray(crop).resize((THUMB, THUMB)).save(Path(sc.lfdir) / name)
        m["thumb"] = name
        names.append(name)
    return names


def quantize_relevancy(rel):
    """uint8-quantize a per-gaussian relevancy vector over [rel_min, rel_max].

    Pure numpy with a LOCAL import (the module-level ``np`` is None when the heavy
    deps are absent — same guard pattern as ``_cluster_extent``), so tests exercise
    the exact wire quantization without torch. Returns (payload, rel_min, rel_max);
    the client dequantizes with ``rel_min + q/255 * (rel_max - rel_min)``. A
    constant vector quantizes to all zeros with rel_min == rel_max, so the client
    dequantizes back to the constant; an empty vector yields (b"", 0.0, 0.0).
    """
    import numpy as np
    rel = np.asarray(rel, dtype=np.float32).reshape(-1)
    if rel.size == 0:
        return b"", 0.0, 0.0
    rmin = float(rel.min())
    rmax = float(rel.max())
    span = rmax - rmin
    if span <= 0.0:
        return bytes(rel.size), rmin, rmax
    q = np.round((rel - rmin) / span * 255.0).astype(np.uint8)
    return q.tobytes(), rmin, rmax


def _safe_name(text: str) -> str:
    """text.replace(' ','_') then sanitize to [\\w-] (drop everything else)."""
    s = text.replace(" ", "_")
    s = re.sub(r"[^\w-]", "", s)
    return s or "query"


def _default_views(n_cams: int) -> list[int]:
    """[0, n//3, 2n//3] (matches the cold renderer's CAM_VIEWS=[0,38,77] for n=116)."""
    n = max(n_cams, 1)
    return [0, n // 3, (2 * n) // 3]


# ── FastAPI app ──────────────────────────────────────────────────────────────────
class QueryReq(BaseModel):
    config: str
    lfdir: str
    text: str
    views: list[int] | None = None


class InventoryReq(BaseModel):
    config: str
    lfdir: str
    topn: int = 50


class RelevancyReq(BaseModel):
    config: str
    lfdir: str
    text: str


class HeroReq(BaseModel):
    config: str
    lfdir: str


app = FastAPI(title="Splat Lab Language Field — Warm Query Worker")


@app.on_event("startup")
async def _startup() -> None:
    logging.basicConfig(level=logging.INFO)
    if not _HEAVY_OK:
        # Don't crash the process on import-time dep gaps — /healthz will report
        # siglip=false so an operator sees exactly what's wrong.
        log.error("heavy deps unavailable, SigLIP NOT loaded: %s", _HEAVY_ERR)
        return
    # load SigLIP once, off the event loop (it touches the GPU but is small/lockless)
    await asyncio.to_thread(STATE.load_siglip)
    app.state._idle_task = asyncio.create_task(_idle_evictor())


@app.on_event("shutdown")
async def _shutdown() -> None:
    task = getattr(app.state, "_idle_task", None)
    if task is not None:
        task.cancel()


async def _idle_evictor() -> None:
    """Background task: drop scene tensors after ~10 min idle so a long training
    run can reclaim the card. SigLIP stays resident; only scenes are dropped."""
    try:
        while True:
            await asyncio.sleep(IDLE_CHECK_SEC)
            async with STATE.lock:
                dropped = STATE.evict_idle()
            if dropped:
                log.info("idle-evicted scenes: %s", dropped)
    except asyncio.CancelledError:
        pass


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {
        "ok": bool(_HEAVY_OK and STATE.ready),
        "siglip": bool(STATE.ready),
        "scenes": list(STATE.scenes.keys()),
    }


@app.post("/query")
async def query(req: QueryReq) -> dict[str, Any]:
    if not (_HEAVY_OK and STATE.ready):
        raise HTTPException(503, f"worker not ready (siglip loaded: {STATE.ready})")

    lf_p = Path(req.lfdir)
    if not (lf_p / "gauss_emb.npz").is_file():
        raise HTTPException(404, f"gauss_emb.npz not found under lfdir {req.lfdir}")
    if not Path(req.config).is_file():
        raise HTTPException(404, f"config not found: {req.config}")

    # ── build-or-reuse scene (serialized; scene build also touches the GPU, so it
    #    too goes behind the heavy lock to not race a training run) ──
    async with STATE.lock:
        sc = STATE.get_scene_cached(req.config)
        if sc is None:
            scene_id = lf_p.name or req.config
            sc = await _build_scene_locked(req.config, req.lfdir, scene_id)
            STATE.insert_scene(sc)
        else:
            sc.touch()
        sc.in_use += 1   # pin: the idle-evictor must not free us mid-query

    try:
        # ── render: heavy lock ONLY around the gsplat calls. Views are now chosen
        #    per-match (best camera for each instance), so req.views is unused. ──
        scene_id = Path(req.lfdir).name or req.config
        sc.touch()   # refresh right before the (possibly long) lock wait behind train
        heatmap_name, focus = await _render_locked(sc, req.text, scene_id)
    finally:
        async with STATE.lock:
            sc.in_use -= 1
            sc.touch()
    return {"heatmap_name": heatmap_name, "ready": True, **focus}


@app.post("/relevancy")
async def relevancy(req: RelevancyReq) -> Response:
    """Raw per-gaussian relevancy for the CLIENT-SIDE heatmap. ENTIRELY LOCKLESS:
    SigLIP text-encode + cosine relevancy + 3D match clustering only — no gsplat
    render, no PNG, so it never takes HEAVY_GPU_LOCK (a first-touch scene build
    still serializes via _build_scene_locked, same as every route). Body = the
    uint8-quantized [N] vector in gauss_emb.npz row order (== the exported
    splat.ply row order); X-Count/X-Min/X-Max carry the dequantization params
    (rel = min + q/255*(max-min)) and X-Matches the clustered instances JSON
    (same shape as /query's matches, minus the rendered thumbs)."""
    if not (_HEAVY_OK and STATE.ready):
        raise HTTPException(503, f"worker not ready (siglip loaded: {STATE.ready})")

    lf_p = Path(req.lfdir)
    if not (lf_p / "gauss_emb.npz").is_file():
        raise HTTPException(404, f"gauss_emb.npz not found under lfdir {req.lfdir}")
    if not Path(req.config).is_file():
        raise HTTPException(404, f"config not found: {req.config}")

    async with STATE.lock:
        sc = STATE.get_scene_cached(req.config)
        if sc is None:
            scene_id = lf_p.name or req.config
            sc = await _build_scene_locked(req.config, req.lfdir, scene_id)
            STATE.insert_scene(sc)
        else:
            sc.touch()
        sc.in_use += 1   # pin against idle-evict while computing

    try:
        def _compute():
            rel3 = _compute_relevancy(STATE, sc, req.text)
            focus = _relevancy_focus(sc, rel3)
            payload, rmin, rmax = quantize_relevancy(rel3[:, 0].detach().cpu().numpy())
            return payload, rmin, rmax, focus

        payload, rmin, rmax, focus = await asyncio.to_thread(_compute)
    finally:
        async with STATE.lock:
            sc.in_use -= 1
            sc.touch()

    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Count": str(len(payload)),
            "X-Min": repr(rmin),
            "X-Max": repr(rmax),
            "X-Matches": json.dumps(focus, separators=(",", ":")),
        },
    )


@app.post("/inventory")
async def inventory(req: InventoryReq) -> dict[str, Any]:
    """Top-N "what's in this scene" object inventory (open-vocab), cached per scene in
    <lfdir>/inventory.json. Cheap (no gsplat render), so it runs LOCKLESS."""
    if not (_HEAVY_OK and STATE.ready):
        raise HTTPException(503, f"worker not ready (siglip loaded: {STATE.ready})")

    lf_p = Path(req.lfdir)
    if not (lf_p / "gauss_emb.npz").is_file():
        raise HTTPException(404, f"gauss_emb.npz not found under lfdir {req.lfdir}")
    if not Path(req.config).is_file():
        raise HTTPException(404, f"config not found: {req.config}")

    cache = lf_p / "inventory.json"
    if cache.is_file():
        try:
            data = json.loads(cache.read_text())
            if data.get("version") == INVENTORY_VERSION and data.get("items"):
                return {"ready": True, "cached": True, "items": data["items"]}
        except Exception:  # pragma: no cover - stale/corrupt cache: recompute
            pass

    async with STATE.lock:
        sc = STATE.get_scene_cached(req.config)
        if sc is None:
            scene_id = lf_p.name or req.config
            sc = await _build_scene_locked(req.config, req.lfdir, scene_id)
            STATE.insert_scene(sc)
        else:
            sc.touch()
        sc.in_use += 1   # pin against idle-evict while scoring

    try:
        items = await asyncio.to_thread(_scene_inventory, STATE, sc, req.topn)
    finally:
        async with STATE.lock:
            sc.in_use -= 1
            sc.touch()

    try:
        cache.write_text(json.dumps({"version": INVENTORY_VERSION, "items": items}))
    except Exception:  # pragma: no cover - cache write is best-effort
        pass
    return {"ready": True, "cached": False, "items": items}


@app.post("/hero")
async def hero(req: HeroReq) -> dict[str, Any]:
    """Render a real RGB hero thumbnail for the scene, cached at <lfdir>/hero.webp.
    Takes the heavy GPU lock only around the single render."""
    if not (_HEAVY_OK and STATE.ready):
        raise HTTPException(503, f"worker not ready (siglip loaded: {STATE.ready})")
    lf_p = Path(req.lfdir)
    if not (lf_p / "gauss_emb.npz").is_file():
        raise HTTPException(404, f"gauss_emb.npz not found under lfdir {req.lfdir}")
    if not Path(req.config).is_file():
        raise HTTPException(404, f"config not found: {req.config}")

    hero_path = lf_p / "hero.webp"
    if hero_path.is_file():
        return {"ready": True, "cached": True, "name": "hero.webp"}

    async with STATE.lock:
        sc = STATE.get_scene_cached(req.config)
        if sc is None:
            scene_id = lf_p.name or req.config
            sc = await _build_scene_locked(req.config, req.lfdir, scene_id)
            STATE.insert_scene(sc)
        else:
            sc.touch()
        sc.in_use += 1

    try:
        scene_id = Path(req.lfdir).name or req.config
        async with gpu_arbiter.HEAVY_GPU_LOCK:
            gpu_arbiter.set_holder("langfield", scene_id)
            try:
                await gpu_arbiter.acquire_gpu(QUERY_VRAM_MB)
                await asyncio.to_thread(_render_hero_locked, sc, hero_path)
            finally:
                gpu_arbiter.clear_holder()
    finally:
        async with STATE.lock:
            sc.in_use -= 1
            sc.touch()
    return {"ready": True, "cached": False, "name": "hero.webp"}


async def _build_scene_locked(config_path: str, lfdir: str, scene_id: str) -> Scene:
    """Build a Scene with the heavy GPU lock held around the GPU-touching load."""
    async with gpu_arbiter.HEAVY_GPU_LOCK:
        gpu_arbiter.set_holder("langfield", scene_id)
        try:
            await gpu_arbiter.acquire_gpu(QUERY_VRAM_MB)
            sc = await asyncio.to_thread(Scene, config_path, lfdir)
        finally:
            gpu_arbiter.clear_holder()
    return sc


async def _render_locked(sc: Scene, text: str, scene_id: str) -> str:
    """The text-encode + per-gaussian cosine relevancy run LOCKLESS; the heavy GPU
    lock is taken ONLY around the gsplat rasterizations (the part that contends for
    VRAM with training/TRELLIS). A query thus WAITS behind a heavy lane and never
    preempts/OOMs it. Renders one result thumbnail per clustered match (each framed
    by its best camera); the primary is q_<safe>.png. Returns (primary_name, focus).
    """
    # LOCKLESS: SigLIP text-encode + per-gaussian relevancy + the 3D match centroids
    rel3 = await asyncio.to_thread(_compute_relevancy, STATE, sc, text)
    focus = await asyncio.to_thread(_relevancy_focus, sc, rel3)

    # LOCKED: gsplat render + overlay composite (per-match thumbnails)
    async with gpu_arbiter.HEAVY_GPU_LOCK:
        gpu_arbiter.set_holder("langfield", scene_id)
        try:
            await gpu_arbiter.acquire_gpu(QUERY_VRAM_MB)
            names = await asyncio.to_thread(
                _render_match_thumbs_locked, sc, rel3, focus["matches"], text)
            return names[0], focus   # names[0] == q_<safe>.png (primary, back-compat)
        finally:
            gpu_arbiter.clear_holder()


if __name__ == "__main__":  # pragma: no cover - convenience launcher
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
