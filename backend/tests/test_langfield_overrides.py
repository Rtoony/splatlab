"""Paint-override sidecar: guardrails, round-trip, revert (CPU-only, numpy).

The contract under test: a paint action must be (a) hard to commit by
accident — min count, max scene fraction, bounds — with human-readable
refusals, and (b) fully revertible — delete removes the record AND its
index file, and gauss_emb.npz is never touched by any of this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import langfield_overrides as lov  # noqa: E402

N = 10_000


def _idx(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).choice(N, size=n, replace=False)


def test_add_load_delete_roundtrip(tmp_path):
    rec = lov.add_override(
        tmp_path, label="trash can", aliases=["bin", "garbage"], op="assign",
        alpha=None, indices=_idx(500), n_ply=N,
    )
    assert rec["count"] == 500 and rec["op"] == "assign"
    assert rec["alpha"] == lov.DEFAULT_ALPHA["assign"]
    loaded = lov.load_overrides(tmp_path)
    assert len(loaded) == 1
    lrec, lidx = loaded[0]
    assert lrec["label"] == "trash can" and lidx.size == 500
    assert lov.delete_override(tmp_path, rec["id"]) is True
    assert lov.load_overrides(tmp_path) == []
    assert not (tmp_path / f"override_idx_{rec['id']}.npy").exists()


def test_tiny_selection_rejected(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        lov.add_override(tmp_path, label="oops", aliases=[], op="assign",
                         alpha=None, indices=_idx(3), n_ply=N)


def test_huge_selection_needs_force(tmp_path):
    big = _idx(int(N * 0.5))
    with pytest.raises(ValueError, match="force"):
        lov.add_override(tmp_path, label="half the scene", aliases=[], op="assign",
                         alpha=None, indices=big, n_ply=N)
    rec = lov.add_override(tmp_path, label="half the scene", aliases=[], op="assign",
                           alpha=None, indices=big, n_ply=N, force=True)
    assert rec["count"] == big.size


def test_out_of_range_rejected(tmp_path):
    bad = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, N + 5])
    with pytest.raises(ValueError, match="out-of-range"):
        lov.add_override(tmp_path, label="bad rows", aliases=[], op="assign",
                         alpha=None, indices=bad, n_ply=N)


def test_bad_op_and_label_rejected(tmp_path):
    with pytest.raises(ValueError, match="op"):
        lov.add_override(tmp_path, label="x", aliases=[], op="explode",
                         alpha=None, indices=_idx(50), n_ply=N)
    with pytest.raises(ValueError, match="label"):
        lov.add_override(tmp_path, label="  ", aliases=[], op="assign",
                         alpha=None, indices=_idx(50), n_ply=N)


def test_duplicate_indices_deduped_and_alpha_clamped(tmp_path):
    dup = np.concatenate([_idx(100, seed=2), _idx(100, seed=2)])
    rec = lov.add_override(tmp_path, label="dup", aliases=[], op="boost",
                           alpha=99.0, indices=dup, n_ply=N)
    assert rec["count"] == 100
    assert rec["alpha"] == 1.0  # clamped


def test_clean_label_recall_normalization():
    assert lov.clean_label("  The WEIRD-corner!! ") == "the weird corner"
    assert lov.clean_label("Trash   Can") == "trash can"  # whitespace collapsed
    assert lov.clean_label("dad's corner") == "dad s corner"


def test_missing_index_file_skipped(tmp_path):
    rec = lov.add_override(tmp_path, label="ghost", aliases=[], op="assign",
                           alpha=None, indices=_idx(50), n_ply=N)
    (tmp_path / f"override_idx_{rec['id']}.npy").unlink()
    assert lov.load_overrides(tmp_path) == []  # manifest advisory, indices truth
