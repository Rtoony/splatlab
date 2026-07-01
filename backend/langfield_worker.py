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
QW, QH = 640, 480
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

from fastapi import FastAPI, HTTPException
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


def _relevancy_focus(sc, rel3):
    """3D centroid + spread of the top-relevancy gaussians = the 'where' of the match,
    in the .ply/viewer frame (sc.means are the exact exported-.ply coords). LOCKLESS,
    cheap (tensor ops, no gsplat). Lets the viewer fly the camera to the found object."""
    rel = rel3[:, 0]
    n = rel.shape[0]
    k = min(max(48, n // 200), n)                 # top ~0.5% (the strongest core)
    topv, topi = torch.topk(rel, k)
    strong = topi[topv > 0.65]                    # only clearly-relevant gaussians
    idx = strong if strong.numel() >= 16 else topi[: min(16, n)]
    pts = sc.means[idx]
    c = pts.mean(0)
    # 70th-pct spread of the core, clamped so a diffuse match doesn't fling the
    # camera to the moon (viewer scenes are ~unit-scaled).
    r = float((pts - c).norm(dim=-1).quantile(0.70).clamp(0.08, 1.5))
    return {"focus": [float(x) for x in c.tolist()], "radius": r, "hits": int(idx.numel())}


def _render_strip_locked(sc: Scene, rel3, views: list[int], text: str) -> str:
    """LOCKED (caller holds HEAVY_GPU_LOCK): the gsplat rasterizations + overlay
    composite. Overlay math is VERBATIM from query_render_v2.py. Saves
    <lfdir>/q_<safe>.png and returns the filename."""
    turbo = cm.get_cmap("turbo")
    white = torch.ones(3, device=DEV)

    # RGB base tiles (cached per view within this call — verbatim from cold path)
    rgb_cache: dict[int, Any] = {}
    for ci in views:
        out, alpha = sc.render(sc.sh, 3, ci, QW, QH)
        rgb_cache[ci] = (out[0, ..., :3] + (1 - alpha[0]) * white).clamp(0, 1)

    tiles = []
    for ci in views:
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
        overlay = (rgb_cache[ci] * (1 - w) + heat * w).clamp(0, 1)
        tiles.append((overlay.cpu().numpy() * 255).astype(np.uint8))

    safe = _safe_name(text)
    out_path = Path(sc.lfdir) / f"q_{safe}.png"
    Image.fromarray(np.concatenate(tiles, 1)).save(out_path)
    return out_path.name


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
        views = req.views if req.views else _default_views(sc.n_cams)
        # clamp/validate view indices against the resident camera count
        views = [v for v in views if 0 <= int(v) < sc.n_cams] or _default_views(sc.n_cams)

        # ── render: heavy lock ONLY around the gsplat calls ──
        scene_id = Path(req.lfdir).name or req.config
        sc.touch()   # refresh right before the (possibly long) lock wait behind train
        heatmap_name, focus = await _render_locked(sc, req.text, views, scene_id)
    finally:
        async with STATE.lock:
            sc.in_use -= 1
            sc.touch()
    return {"heatmap_name": heatmap_name, "ready": True, **focus}


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


async def _render_locked(sc: Scene, text: str, views: list[int], scene_id: str) -> str:
    """The text-encode + per-gaussian cosine relevancy run LOCKLESS; the heavy GPU
    lock is taken ONLY around the gsplat rasterizations (the part that contends for
    VRAM with training/TRELLIS). A query thus WAITS behind a heavy lane and never
    preempts/OOMs it.
    """
    # LOCKLESS: SigLIP text-encode + per-gaussian relevancy + the 3D match centroid
    rel3 = await asyncio.to_thread(_compute_relevancy, STATE, sc, text)
    focus = await asyncio.to_thread(_relevancy_focus, sc, rel3)

    # LOCKED: gsplat render + overlay composite
    async with gpu_arbiter.HEAVY_GPU_LOCK:
        gpu_arbiter.set_holder("langfield", scene_id)
        try:
            await gpu_arbiter.acquire_gpu(QUERY_VRAM_MB)
            name = await asyncio.to_thread(_render_strip_locked, sc, rel3, views, text)
            return name, focus
        finally:
            gpu_arbiter.clear_holder()


if __name__ == "__main__":  # pragma: no cover - convenience launcher
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
