"""Tests for the LERF relevancy math — previously three copies, zero tests.

`langfield_worker.relevancy_vec`, `query_render_v2.relevancy_vec` and an inline
block in `mesh/object_isolate.py` all implemented this, kept in sync by a
comment reading "COPIED VERBATIM ... keep byte-for-byte identical". They had
already drifted exactly where it mattered: object_isolate zeroed unseen
gaussians, the other two did not.

An unobserved gaussian's lifted embedding is an all-zero row, so both softmax
logits are equal and it scores **exactly 0.5** against every query ever run —
a mid-range score immune to the query, which polluted heatmaps, top-k selection
and inventory ranking. That number is asserted directly below, so the fix
cannot silently regress.

The app's test interpreter has no torch, so these drive relevancy_core's numpy
branch. The opt-in test at the bottom runs the real torch branch in the
langfield-spike env and asserts the two agree — the same shape as the other
heavy-env tests in this suite. Nothing here touches the GPU, which is reserved
for work launched through tools/splatlab-compute-gate.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "langfield"))

import relevancy_core as rc  # noqa: E402

DIM = 8
LANGFIELD_PYTHON = Path.home() / "miniconda3" / "envs" / "langfield-spike" / "bin" / "python"


def _unit(*components) -> np.ndarray:
    vector = np.array(components, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _basis(index: int, dim: int = DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


@pytest.fixture()
def negatives() -> np.ndarray:
    """Negative prompts pointing away from the query direction."""
    return np.stack([_basis(i) for i in range(2, 2 + len(rc.NEGATIVES))])


# ---------------------------------------------------------------------------
# The unseen-gaussian bug, pinned
# ---------------------------------------------------------------------------

def test_a_zero_embedding_would_score_exactly_one_half_without_the_fix(negatives):
    """The bug, reproduced against the raw math: equal logits -> 0.5."""
    zero_row = np.zeros((1, DIM), dtype=np.float32)
    query = _basis(0)

    sim_pos = zero_row @ query
    sim_neg = zero_row @ negatives.T
    raw = np.ones_like(sim_pos)
    for k in range(negatives.shape[0]):
        pair = np.stack([sim_pos, sim_neg[:, k]], axis=-1)
        raw = np.minimum(raw, rc._softmax_last_axis(pair * rc.SOFTMAX_TEMP)[:, 0])

    assert float(raw[0]) == pytest.approx(rc.UNSEEN_RAW_SCORE)
    assert float(raw[0]) == pytest.approx(0.5), "mid-range, and immune to the query"


def test_unseen_gaussians_score_zero(negatives):
    feat = np.stack([_basis(0), np.zeros(DIM, dtype=np.float32), _basis(0)])
    rel = rc.relevancy_vec(feat, _basis(0), negatives)

    assert float(rel[1]) == 0.0, "no evidence must not read as a half match"
    assert float(rel[0]) > 0.5 and float(rel[2]) > 0.5


def test_the_mask_is_derived_when_not_supplied(negatives):
    """A caller that has not plumbed `seen` through still gets correct scores."""
    feat = np.stack([_basis(0), np.zeros(DIM, dtype=np.float32)])
    assert float(rc.relevancy_vec(feat, _basis(0), negatives)[1]) == 0.0


def test_an_explicit_mask_overrides_a_nonzero_row(negatives):
    """Scenes whose lift recorded seenness separately are trusted over the
    zero-row heuristic."""
    feat = np.stack([_basis(0), _basis(0)])
    seen = np.array([True, False])

    rel = rc.relevancy_vec(feat, _basis(0), negatives, seen=seen)

    assert float(rel[0]) > 0.5
    assert float(rel[1]) == 0.0


def test_a_plain_list_mask_is_accepted(negatives):
    feat = np.stack([_basis(0), _basis(0)])
    rel = rc.relevancy_vec(feat, _basis(0), negatives, seen=[True, False])
    assert float(rel[1]) == 0.0


def test_unseen_rows_cannot_outrank_a_weak_real_match(negatives):
    """The failure that mattered: for a query nothing matches well, a floor of
    0.5 unseen rows became the top-k answer instead of the real geometry."""
    weak_match = _unit(1.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0)
    feat = np.stack([weak_match] + [np.zeros(DIM, dtype=np.float32)] * 20)

    rel = rc.relevancy_vec(feat, _basis(0), negatives)

    assert int(np.argmax(rel)) == 0
    assert float(rel[0]) > float(rel[1:].max())


def test_observed_mask_matches_the_lifts_zeroing_convention():
    feat = np.stack([_basis(0), np.zeros(DIM, dtype=np.float32),
                     _unit(1.0, 1.0, 0, 0, 0, 0, 0, 0)])
    assert rc.observed_mask(feat).tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# The relevancy contract itself
# ---------------------------------------------------------------------------

def test_relevancy_is_bounded_and_shaped(negatives):
    raw = np.random.default_rng(0).normal(size=(64, DIM)).astype(np.float32)
    feat = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    rel = rc.relevancy_vec(feat, _basis(0), negatives)

    assert rel.shape == (64,)
    assert float(rel.min()) >= 0.0 and float(rel.max()) <= 1.0


def test_an_exact_query_match_outscores_an_orthogonal_row(negatives):
    feat = np.stack([_basis(0), _basis(1)])
    rel = rc.relevancy_vec(feat, _basis(0), negatives)
    assert float(rel[0]) > float(rel[1])


def test_a_row_aligned_with_a_negative_scores_low(negatives):
    """The min over negatives is what suppresses generic "stuff"."""
    feat = np.stack([_basis(0), negatives[0]])
    rel = rc.relevancy_vec(feat, _basis(0), negatives)

    assert float(rel[1]) < 0.5
    assert float(rel[1]) < float(rel[0])


def test_a_two_dimensional_query_is_tolerated(negatives):
    """text_emb returns [1, D]; callers pass it both ways."""
    feat = np.stack([_basis(0), _basis(1)])
    flat = rc.relevancy_vec(feat, _basis(0), negatives)
    batched = rc.relevancy_vec(feat, _basis(0)[None, :], negatives)
    assert np.allclose(flat, batched)


def test_relevancy_is_deterministic(negatives):
    raw = np.random.default_rng(1).normal(size=(32, DIM)).astype(np.float32)
    feat = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
    query = _basis(0)
    assert np.array_equal(rc.relevancy_vec(feat, query, negatives),
                          rc.relevancy_vec(feat, query, negatives))


# ---------------------------------------------------------------------------
# Single-definition constants
# ---------------------------------------------------------------------------

def test_the_lerf_negatives_are_the_canonical_five():
    assert rc.NEGATIVES == ["object", "things", "stuff", "texture", "surface"]


def test_the_siglip_checkpoint_has_one_definition():
    assert rc.SIGLIP_CKPT == "google/siglip2-so400m-patch16-384"


def test_consumers_reexport_rather_than_redefine():
    """langfield_worker must not drift back to its own copy of the constants."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import langfield_worker  # noqa: E402

    assert langfield_worker.SIGLIP_CKPT is rc.SIGLIP_CKPT
    assert langfield_worker.NEGATIVES is rc.NEGATIVES


def test_softmax_temperature_is_pinned():
    """Changing this silently redefines every stored heatmap and threshold."""
    assert rc.SOFTMAX_TEMP == 10.0


# ---------------------------------------------------------------------------
# The torch branch, in the environment that actually runs it
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LANGFIELD_PYTHON.exists(),
                    reason="langfield-spike env not installed")
def test_torch_branch_agrees_with_the_numpy_branch(tmp_path):
    """Everything above tests the numpy branch, because the app's interpreter
    has no torch. Production takes the torch branch. This runs BOTH in the
    langfield-spike env on identical inputs and asserts they agree — otherwise
    the coverage above would be proving something about code nobody runs.

    CPU tensors only; no CUDA call is made.
    """
    import json
    import subprocess

    script = tmp_path / "compare.py"
    script.write_text(f'''
import sys, json
sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "langfield")!r})
import numpy as np, torch
import relevancy_core as rc

rng = np.random.default_rng(7)
raw = rng.normal(size=(256, 32)).astype(np.float32)
feat = raw / np.linalg.norm(raw, axis=-1, keepdims=True)
feat[::7] = 0.0                                   # unobserved gaussians
q = rng.normal(size=(32,)).astype(np.float32); q /= np.linalg.norm(q)
neg = rng.normal(size=(5, 32)).astype(np.float32)
neg /= np.linalg.norm(neg, axis=-1, keepdims=True)

np_rel = rc.relevancy_vec(feat, q, neg)
pt_rel = rc.relevancy_vec(torch.from_numpy(feat), torch.from_numpy(q),
                          torch.from_numpy(neg)).cpu().numpy()

print(json.dumps({{
    "max_abs_diff": float(np.abs(np_rel - pt_rel).max()),
    "unseen_np": float(np.abs(np_rel[::7]).max()),
    "unseen_pt": float(np.abs(pt_rel[::7]).max()),
    "used_cuda": torch.cuda.is_initialized(),
}}))
''')
    completed = subprocess.run([str(LANGFIELD_PYTHON), str(script)],
                              capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-3000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["max_abs_diff"] < 1e-6, result
    assert result["unseen_np"] == 0.0 and result["unseen_pt"] == 0.0
    assert result["used_cuda"] is False, "this comparison must stay on the CPU"
