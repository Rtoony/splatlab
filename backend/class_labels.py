"""User-painted per-gaussian CLASS labels — the overrides store's sibling.

Where `langfield_overrides` biases search embeddings, this store records what
a region IS (grass/dirt/pavement/...), which the generative stages consume:
class-textured ground, class-partitioned shell bakes, per-class materials.

Invariants (same discipline as the overrides store):
- Nothing else is mutated. Every paint is a manifest record + an idx file +
  an xyz snapshot; deleting the record fully reverts it.
- Indices are EXPORTED-PLY rows (viewer space). The xyz snapshot — written on
  EVERY commit from day one — is what carries the paint across later crops
  (langfield_realign re-matches positions, never trusts dead row numbers).
- MIN_COUNT / MAX_FRACTION guardrails unless force=True.

Layout under <jobdir>/_langfield/ (deliberately: same row space, same stale
guard, same realign remap as the overrides):
  class_labels.json        [{id, class_id, count, created_at, note}]
  class_idx_<id>.npy       uint32 ply rows
  class_xyz_<id>.npy       float32 [n,3] positions at commit time

Worker-side numpy module — the app venv never imports it.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

import numpy as np

MANIFEST_NAME = "class_labels.json"
MIN_COUNT = 10
MAX_FRACTION = 0.30


def _manifest_path(lfdir: Path) -> Path:
    return Path(lfdir) / MANIFEST_NAME


def _indices_path(lfdir: Path, cid: str) -> Path:
    return Path(lfdir) / f"class_idx_{cid}.npy"


def _xyz_path(lfdir: Path, cid: str) -> Path:
    return Path(lfdir) / f"class_xyz_{cid}.npy"


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
    if indices.size < MIN_COUNT:
        return f"selection has {indices.size} gaussians — paint at least {MIN_COUNT}"
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_ply):
        return "selection indices out of range for this scene"
    if not force and indices.size > MAX_FRACTION * n_ply:
        return (
            f"selection covers {indices.size / max(n_ply, 1):.0%} of the scene "
            f"(max {MAX_FRACTION:.0%}) — pass force=true if this is intentional"
        )
    return None


def add_class_label(
    lfdir: Path,
    *,
    class_id: str,
    indices: np.ndarray,
    xyz: np.ndarray | None,
    n_ply: int,
    force: bool = False,
    note: str = "",
) -> dict:
    """Persist one class-paint action. The class_id is validated by the CALLER
    against the merged taxonomy (this module stays taxonomy-agnostic numpy).
    Raises ValueError on guardrail violation."""
    indices = np.unique(np.asarray(indices, dtype=np.int64))
    err = validate_selection(indices, n_ply, force)
    if err:
        raise ValueError(err)

    lfdir = Path(lfdir)
    rid = secrets.token_hex(4)
    np.save(_indices_path(lfdir, rid), indices.astype(np.uint32))
    if xyz is not None and len(xyz) == indices.size:
        np.save(_xyz_path(lfdir, rid), np.asarray(xyz, dtype=np.float32))
    record = {
        "id": rid,
        "class_id": class_id,
        "count": int(indices.size),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note[:200],
    }
    items = load_manifest(lfdir)
    items.append(record)
    _save_manifest(lfdir, items)
    return record


def delete_class_label(lfdir: Path, rid: str) -> bool:
    lfdir = Path(lfdir)
    items = load_manifest(lfdir)
    kept = [r for r in items if r.get("id") != rid]
    if len(kept) == len(items):
        return False
    _save_manifest(lfdir, kept)
    _indices_path(lfdir, rid).unlink(missing_ok=True)
    _xyz_path(lfdir, rid).unlink(missing_ok=True)
    return True


def resolve_class_map(lfdir: Path, n_ply: int, class_order: list[str]) -> np.ndarray:
    """int16 per-ply-row class index into `class_order`, -1 = unlabeled.
    Later records win — paint order is precedence, same as layers in any
    painting tool. Records for classes absent from `class_order` (a job
    extension that was later removed) are skipped, as are records whose idx
    file is missing or out of range (manifest is advisory, indices are truth)."""
    lfdir = Path(lfdir)
    class_pos = {cid: i for i, cid in enumerate(class_order)}
    out = np.full(n_ply, -1, dtype=np.int16)
    for record in load_manifest(lfdir):
        if record.get("invalid_reason"):
            continue
        pos = class_pos.get(record.get("class_id"))
        if pos is None:
            continue
        idx_path = _indices_path(lfdir, record.get("id", ""))
        if not idx_path.is_file():
            continue
        try:
            rows = np.load(idx_path).astype(np.int64)
        except (OSError, ValueError):
            continue
        rows = rows[(rows >= 0) & (rows < n_ply)]
        out[rows] = np.int16(pos)
    return out


def summary(lfdir: Path, n_ply: int, class_order: list[str]) -> dict:
    class_map = resolve_class_map(lfdir, n_ply, class_order)
    counts = {
        cid: int((class_map == i).sum()) for i, cid in enumerate(class_order)
    }
    labeled = int((class_map >= 0).sum())
    return {
        "n_ply": n_ply,
        "labeled": labeled,
        "unlabeled": n_ply - labeled,
        "by_class": {cid: c for cid, c in counts.items() if c},
        "records": load_manifest(lfdir),
    }
