"""Sidecar "paint" overrides for the language field.

Design: ~/reports/splatlab-embedding-paint-design-2026-07-04/DESIGN.md
RToony's feature: paint splat regions to CONFIRM or OVERRIDE the text-embedding
lift — including "liberal" labels for abstract things the lift never grounded.

Invariants (the guardrails live HERE, server-side, not in UI goodwill):
- gauss_emb.npz is NEVER mutated. Every paint action is a manifest record +
  an index file; deleting the record fully reverts it.
- Indices are EXPORTED-PLY order (the client/viewer space, post
  langfield_align). The worker maps to checkpoint rows at apply time.
- A selection must be big enough to be intentional (MIN_COUNT) and small
  enough to be sane (MAX_FRACTION of the scene) unless force=True.

Layout under <jobdir>/_langfield/:
  overrides.json            manifest: [{id, label, aliases, op, alpha, count,
                                        created_at, note}]
  override_idx_<id>.npy     uint32 ply-order rows for that record
"""

from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

import numpy as np

MANIFEST_NAME = "overrides.json"
MIN_COUNT = 10
MAX_FRACTION = 0.30
OPS = ("assign", "boost", "suppress")
# assign = trust the human fully; boost = nudge/confirm; suppress = "not this"
DEFAULT_ALPHA = {"assign": 0.95, "boost": 0.35, "suppress": 0.8}


def clean_label(text: str) -> str:
    """Normalization used for exact-label recall matching (query -> painted
    region). Mirrors the spirit of the app's _langfield_clean_text: lowercase,
    collapse non-alphanumerics."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _manifest_path(lfdir: Path) -> Path:
    return Path(lfdir) / MANIFEST_NAME


def _indices_path(lfdir: Path, oid: str) -> Path:
    return Path(lfdir) / f"override_idx_{oid}.npy"


def load_manifest(lfdir: Path) -> list[dict]:
    p = _manifest_path(lfdir)
    if not p.is_file():
        return []
    try:
        items = json.loads(p.read_text())
        return items if isinstance(items, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_manifest(lfdir: Path, items: list[dict]) -> None:
    _manifest_path(lfdir).write_text(json.dumps(items, indent=1))


def validate_selection(indices: np.ndarray, n_ply: int, force: bool) -> str | None:
    """Returns an error message, or None when the selection is acceptable."""
    if indices.size < MIN_COUNT:
        return f"selection too small ({indices.size} splats; need >= {MIN_COUNT}) — probably an accidental click"
    if int(indices.min(initial=0)) < 0 or int(indices.max(initial=0)) >= n_ply:
        return f"selection contains out-of-range rows (scene has {n_ply:,} splats)"
    frac = indices.size / max(n_ply, 1)
    if frac > MAX_FRACTION and not force:
        return (
            f"selection covers {frac:.0%} of the scene (> {MAX_FRACTION:.0%}) — "
            "that is usually a mistake; pass force=true to do it anyway"
        )
    return None


def add_override(
    lfdir: Path,
    *,
    label: str,
    aliases: list[str],
    op: str,
    alpha: float | None,
    indices: np.ndarray,
    n_ply: int,
    force: bool = False,
    note: str = "",
) -> dict:
    """Persist one paint action. Raises ValueError with a human message on any
    guardrail violation. Returns the manifest record."""
    label = label.strip()
    if not label or len(label) > 80:
        raise ValueError("label must be 1-80 characters")
    if op not in OPS:
        raise ValueError(f"op must be one of {OPS}")
    aliases = [a.strip() for a in aliases if a and a.strip()][:8]
    indices = np.unique(np.asarray(indices, dtype=np.int64))
    err = validate_selection(indices, n_ply, force)
    if err:
        raise ValueError(err)
    a = DEFAULT_ALPHA[op] if alpha is None else float(alpha)
    a = min(1.0, max(0.05, a))

    lfdir = Path(lfdir)
    oid = secrets.token_hex(4)
    np.save(_indices_path(lfdir, oid), indices.astype(np.uint32))
    record = {
        "id": oid,
        "label": label,
        "aliases": aliases,
        "op": op,
        "alpha": a,
        "count": int(indices.size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note[:200],
    }
    items = load_manifest(lfdir)
    items.append(record)
    _save_manifest(lfdir, items)
    return record


def delete_override(lfdir: Path, oid: str) -> bool:
    lfdir = Path(lfdir)
    items = load_manifest(lfdir)
    kept = [r for r in items if r.get("id") != oid]
    if len(kept) == len(items):
        return False
    _save_manifest(lfdir, kept)
    _indices_path(lfdir, oid).unlink(missing_ok=True)
    return True


def load_overrides(lfdir: Path) -> list[tuple[dict, np.ndarray]]:
    """Manifest records paired with their index arrays; silently skips records
    whose index file went missing (manifest is advisory, indices are truth)."""
    out: list[tuple[dict, np.ndarray]] = []
    for rec in load_manifest(lfdir):
        p = _indices_path(Path(lfdir), str(rec.get("id")))
        if not p.is_file():
            continue
        try:
            idx = np.load(p).astype(np.int64)
        except (OSError, ValueError):
            continue
        out.append((rec, idx))
    return out
