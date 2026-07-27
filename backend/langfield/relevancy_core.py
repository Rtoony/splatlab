"""The LERF relevancy math, in one place, with the unseen-gaussian fix.

Why this module exists
----------------------
Three copies of this arithmetic existed — `langfield_worker.relevancy_vec`,
`query_render_v2.relevancy_vec`, and an inline block in
`mesh/object_isolate.py` — kept in sync by a comment reading "COPIED VERBATIM
... keep byte-for-byte identical". They had already drifted, in the way that
matters most:

**`object_isolate.py` zeroed unseen gaussians. The other two did not.**

An unseen gaussian's lifted embedding is an all-zero row (`langfield_v2.py`
writes `Facc[~seen] = 0.0`). Feed a zero row through the min-softmax and every
term is `softmax([0, 0] * 10)[0]`, which is **exactly 0.5** — so every gaussian
the lift never observed scored 0.5 against every query anyone ever ran. Not a
low score: a mid-range one, immune to the query. Under the per-query min-max
normalisation the client applies, that pollutes:

  * heatmaps — unseen regions glow at mid-intensity;
  * top-k selection — for a query nothing matches well, a 0.5 floor of unseen
    rows can outrank the real, weakly-matching answer and *become* the match;
  * `_scene_inventory` scoring, which ranks nouns by relevancy statistics.

`seen` has been stored in `gauss_emb.npz` since the lift was written; only the
isolation lane ever read it. Zero is the honest score for "the lift never
observed this gaussian" — 0.5 asserts a half-match on no evidence.

Works on torch tensors (what all three consumers pass) and on numpy arrays (so
the arithmetic is testable in the app's test interpreter, which has no torch).
The torch branch performs exactly the operations the original copies did, in the
same order, so stored heatmaps stay bit-identical; an opt-in test runs both
branches in the langfield-spike env and asserts they agree.

Free of any app import so all three consumers — two of them in separate conda
environments — can share it.
"""

from __future__ import annotations

import numpy as np

try:                                        # the consumers all have torch
    import torch
except ImportError:                         # pragma: no cover - test interpreter
    torch = None

# The single definition of the SigLIP 2 checkpoint and the LERF negative
# prompts. Previously the checkpoint string was repeated in 6 files and the
# negatives in 4, where they could drift apart silently and change what every
# query means.
SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
NEGATIVES = ["object", "things", "stuff", "texture", "surface"]  # LERF canon

# LERF's softmax sharpness. Changing this changes what every stored heatmap and
# every relevancy threshold in the app means.
SOFTMAX_TEMP = 10.0

# The score an unseen gaussian would receive from the raw min-softmax: a zero
# embedding row makes both logits equal. Kept named so the tests can assert the
# bug is actually being prevented rather than accidentally absent.
UNSEEN_RAW_SCORE = 0.5


def relevancy_vec(feat_n, q_emb, neg_p, seen=None):
    """Per-gaussian LERF min-softmax relevancy in [0, 1].

    feat_n: [N, D] L2-normed gaussian embeddings (all-zero rows for unseen)
    q_emb:  [D]    L2-normed query text embedding
    neg_p:  [K, D] L2-normed negative-prompt embeddings
    seen:   [N] bool, optional. Rows that are False score 0.0.

    When `seen` is not supplied it is derived from the embeddings themselves —
    an all-zero row is exactly what the lift writes for an unobserved gaussian —
    so a caller that has not plumbed the mask through still gets correct scores.
    """
    if q_emb.ndim == 2:                     # tolerate a [1, D] query
        q_emb = q_emb.squeeze(0)

    if _is_torch(feat_n):
        # Exactly the operations the three copies performed, in the same order,
        # so existing heatmaps stay bit-identical.
        sim_pos = feat_n @ q_emb
        sim_neg = feat_n @ neg_p.T
        rel = torch.ones_like(sim_pos)
        for k in range(neg_p.shape[0]):
            pair = torch.stack([sim_pos, sim_neg[:, k]], dim=-1)
            rel = torch.minimum(rel, torch.softmax(pair * SOFTMAX_TEMP, dim=-1)[:, 0])
        mask = observed_mask(feat_n) if seen is None else _as_torch_bool(seen, rel)
        return torch.where(mask, rel, torch.zeros_like(rel))

    sim_pos = np.asarray(feat_n) @ np.asarray(q_emb)
    sim_neg = np.asarray(feat_n) @ np.asarray(neg_p).T
    rel = np.ones_like(sim_pos)
    for k in range(np.shape(neg_p)[0]):
        pair = np.stack([sim_pos, sim_neg[:, k]], axis=-1)
        rel = np.minimum(rel, _softmax_last_axis(pair * SOFTMAX_TEMP)[:, 0])
    mask = observed_mask(feat_n) if seen is None else np.asarray(seen, dtype=bool)
    return np.where(mask, rel, np.zeros_like(rel))


def observed_mask(feat_n):
    """True where the lift actually observed a gaussian.

    The lift L2-normalises every observed row, so an observed row has norm 1 and
    an unobserved one has norm 0; anything near zero is unobserved.
    """
    if _is_torch(feat_n):
        return feat_n.abs().sum(dim=-1) > 1e-6
    return np.abs(np.asarray(feat_n)).sum(axis=-1) > 1e-6


def _is_torch(value) -> bool:
    return torch is not None and torch.is_tensor(value)


def _as_torch_bool(seen, like):
    mask = seen if torch.is_tensor(seen) else torch.as_tensor(seen)
    return mask.to(device=like.device, dtype=torch.bool)


def _softmax_last_axis(logits):
    """Max-subtracted softmax, matching torch.softmax's stabilisation."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)
