"""Post-edit language-field rebuild = row REALIGNMENT, not a re-lift.

Edits (crop/floaters/semantic delete) never touch the training checkpoint, and
the lift's scratch is deleted after build — re-lifting would spend ~minutes to
reproduce the byte-identical checkpoint-order gauss_emb.npz. What an edit
actually breaks is the exported-ply -> checkpoint row map (rows were dropped),
so the rebuild is: re-derive `ply_index_map.npy` for the CURRENT splat.ply by
exact-xyz byte matching (the proven langfield_align technique; splat-transform
passthrough preserves xyz bytes — probe receipt in _regen_langweb's docstring),
carry every painted record across, and clear the STALE marker last.

Total-or-nothing: if any current ply row has no byte-identical checkpoint
position (geometry was transformed or externally rewritten), NOTHING is
written and the process exits 2 with a JSON error — a wrong tint is worse
than no tint, and the cure is a retrain with Language search on.

Painted-record remap, in preference order per record:
  1. xyz snapshot (`*_xyz_<id>.npy`, written on every commit since 2026-07-26
     and backfilled here): match snapshot positions against the new ply —
     PARTIAL survival allowed, kept/dropped counted honestly.
  2. legacy chain: old ply rows -> old map -> ckpt rows -> inverse(new map)
     -> new ply rows (valid because the stale guard blocks painting while
     stale, so record indices are always in the last-mapped row space).
  3. neither -> the record is marked invalid in its manifest, idx removed.

Runs in the langfield-spike conda env. First run on a scene loads the
checkpoint (GPU) to cache `_langfield/ckpt_xyz.npy`; every later realign is
pure numpy. Core functions are importable without torch for the test suite.

Usage: langfield_realign.py <job_dir> [--config <config.yml>]
Exit: 0 ok · 2 realign impossible · 3 precondition missing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from langfield_align import MAP_FILENAME, build_index_map, read_ply_xyz

RECEIPT_NAME = "realign.receipt.json"

# (manifest filename, idx prefix, xyz prefix) — overrides today, class labels
# from P3 on. Missing manifests are simply skipped.
RECORD_STORES = (
    ("overrides.json", "override_idx_", "override_xyz_"),
    ("class_labels.json", "class_idx_", "class_xyz_"),
)


def _void_keys(xyz: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(xyz, dtype=np.float32)
    return a.view([("k", "V12")])["k"].reshape(-1)


def match_xyz_rows(snapshot_xyz: np.ndarray, ply_xyz: np.ndarray) -> np.ndarray:
    """New-ply row for each snapshot position that still exists (byte-exact);
    unmatched snapshot rows are silently dropped (the edit removed them)."""
    ply_k = _void_keys(ply_xyz)
    snap_k = _void_keys(snapshot_xyz)
    order = np.argsort(ply_k, kind="stable")
    ply_sorted = ply_k[order]
    pos = np.searchsorted(ply_sorted, snap_k)
    pos_c = np.minimum(pos, ply_sorted.size - 1)
    ok = (pos < ply_sorted.size) & (ply_sorted[pos_c] == snap_k)
    return np.unique(order[pos_c[ok]].astype(np.int64))


def invert_map(new_map: np.ndarray, n_ckpt: int) -> np.ndarray:
    """ckpt row -> new ply row, -1 where the ckpt row is no longer exported."""
    inverse = np.full(n_ckpt, -1, dtype=np.int64)
    inverse[new_map] = np.arange(new_map.size, dtype=np.int64)
    return inverse


def remap_store(
    lfdir: Path,
    manifest_name: str,
    idx_prefix: str,
    xyz_prefix: str,
    ply_xyz: np.ndarray,
    old_map: np.ndarray | None,
    inverse_new: np.ndarray,
) -> list[dict]:
    """Carry one record store across the edit. Mutates idx/xyz files and the
    manifest ATOMICALLY (tmp+replace per file, manifest last). Returns the
    per-record receipt rows."""
    manifest_path = lfdir / manifest_name
    if not manifest_path.is_file():
        return []
    try:
        records = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return [{"store": manifest_name, "error": "unreadable manifest"}]
    if not isinstance(records, list):
        return []

    receipt: list[dict] = []
    for record in records:
        oid = record.get("id")
        if not isinstance(oid, str):
            continue
        idx_path = lfdir / f"{idx_prefix}{oid}.npy"
        xyz_path = lfdir / f"{xyz_prefix}{oid}.npy"
        before = int(record.get("count") or 0)
        new_rows: np.ndarray | None = None
        method = None

        if xyz_path.is_file():
            try:
                snap = np.load(xyz_path)
                new_rows = match_xyz_rows(np.asarray(snap, dtype=np.float32), ply_xyz)
                method = "xyz-snapshot"
            except (OSError, ValueError):
                new_rows = None
        if new_rows is None and old_map is not None and idx_path.is_file():
            try:
                old_rows = np.load(idx_path).astype(np.int64)
                valid = old_rows[(old_rows >= 0) & (old_rows < old_map.size)]
                ckpt_rows = old_map[valid]
                mapped = inverse_new[ckpt_rows]
                new_rows = np.unique(mapped[mapped >= 0])
                method = "old-map-chain"
            except (OSError, ValueError, IndexError):
                new_rows = None

        if new_rows is None or new_rows.size == 0:
            reason = (
                "all painted gaussians were removed by the edit"
                if new_rows is not None
                else "unmappable after edit (no xyz snapshot, no prior map)"
            )
            record["invalid_reason"] = reason
            record["count"] = 0
            idx_path.unlink(missing_ok=True)
            xyz_path.unlink(missing_ok=True)
            receipt.append({"store": manifest_name, "id": oid,
                            "label": record.get("label") or record.get("class_id"),
                            "kept": 0, "dropped": before, "method": method,
                            "invalid": reason})
            continue

        for path, payload in (
            (idx_path, new_rows.astype(np.uint32)),
            (xyz_path, ply_xyz[new_rows].astype(np.float32)),
        ):
            tmp = path.with_name(f".building-{path.name}")
            np.save(tmp, payload)
            # np.save appends .npy to names without it — tmp already ends .npy
            tmp.replace(path)
        record.pop("invalid_reason", None)
        record["count"] = int(new_rows.size)
        receipt.append({"store": manifest_name, "id": oid,
                        "label": record.get("label") or record.get("class_id"),
                        "kept": int(new_rows.size),
                        "dropped": max(0, before - int(new_rows.size)),
                        "method": method})

    tmp_manifest = manifest_path.with_name(f".building-{manifest_name}")
    tmp_manifest.write_text(json.dumps(records, indent=1))
    tmp_manifest.replace(manifest_path)
    return receipt


def load_ckpt_xyz(lfdir: Path, n_emb: int, config_path: Path | None) -> np.ndarray:
    """Checkpoint gaussian positions, cached beside the field. The first run
    needs the checkpoint (torch, GPU); every later realign is pure numpy."""
    cache = lfdir / "ckpt_xyz.npy"
    if cache.is_file():
        xyz = np.load(cache)
        if xyz.ndim == 2 and xyz.shape == (n_emb, 3):
            return np.asarray(xyz, dtype=np.float32)
    if config_path is None or not config_path.is_file():
        raise SystemExit(
            json.dumps({"error": "ckpt_xyz cache missing and no --config given"}))
    import torch  # noqa: PLC0415 — only the cold path needs the heavy env
    from nerfstudio.utils.eval_utils import eval_setup  # noqa: PLC0415

    _cfg, pipeline, _ckpt, _step = eval_setup(config_path, test_mode="inference")
    with torch.no_grad():
        xyz = pipeline.model.means.detach().cpu().numpy().astype(np.float32)
    if xyz.shape[0] != n_emb:
        raise SystemExit(json.dumps({
            "error": "checkpoint row count does not match gauss_emb",
            "ckpt_rows": int(xyz.shape[0]), "emb_rows": n_emb,
        }))
    tmp = cache.with_name(f".building-{cache.name}")
    np.save(tmp, xyz)
    tmp.replace(cache)
    return xyz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--config", default=None,
                    help="splatfacto config.yml (needed only when ckpt_xyz.npy "
                         "is not cached yet)")
    args = ap.parse_args()
    t0 = time.time()

    job_dir = Path(args.job_dir).expanduser().resolve()
    lfdir = job_dir / "_langfield"
    emb_path = lfdir / "gauss_emb.npz"
    ply_path = job_dir / "_preview" / "splat.ply"
    if not emb_path.is_file():
        print(json.dumps({"error": "no language field (gauss_emb.npz missing)"}))
        return 3
    if not ply_path.is_file():
        print(json.dumps({"error": "no exported splat.ply"}))
        return 3

    with np.load(emb_path) as npz:
        n_emb = int(npz["gauss_emb"].shape[0])

    ckpt_xyz = load_ckpt_xyz(
        lfdir, n_emb, Path(args.config) if args.config else None)
    ply_xyz = read_ply_xyz(ply_path)

    new_map = build_index_map(ply_xyz, ckpt_xyz)
    if new_map is None:
        unmatched = int(
            (~np.isin(_void_keys(ply_xyz), _void_keys(ckpt_xyz))).sum())
        print(json.dumps({
            "error": "realign_failed",
            "detail": "current geometry has positions with no byte-identical "
                      "checkpoint row — it was transformed or externally "
                      "rewritten; retrain with Language search to restore "
                      "the field",
            "unmatched": unmatched,
            "ply_rows": int(ply_xyz.shape[0]),
        }))
        return 2

    old_map_path = lfdir / MAP_FILENAME
    old_map = None
    if old_map_path.is_file():
        try:
            candidate = np.load(old_map_path).astype(np.int64)
            if candidate.ndim == 1 and (candidate.size == 0 or (
                    0 <= int(candidate.min()) and int(candidate.max()) < n_emb)):
                old_map = candidate
        except (OSError, ValueError):
            old_map = None

    inverse_new = invert_map(new_map, n_emb)
    record_receipts: list[dict] = []
    for manifest_name, idx_prefix, xyz_prefix in RECORD_STORES:
        record_receipts += remap_store(
            lfdir, manifest_name, idx_prefix, xyz_prefix,
            ply_xyz, old_map, inverse_new)

    tmp_map = old_map_path.with_name(f".building-{MAP_FILENAME}")
    np.save(tmp_map, new_map)
    tmp_map.replace(old_map_path)

    # STALE clears LAST — everything above must have landed for the field to
    # be honestly usable again.
    (lfdir / "STALE").unlink(missing_ok=True)

    receipt = {
        "ok": True,
        "ply_rows": int(ply_xyz.shape[0]),
        "ckpt_rows": n_emb,
        "dropped_ckpt_rows": n_emb - int(ply_xyz.shape[0]),
        "records": record_receipts,
        "seconds": round(time.time() - t0, 2),
    }
    (lfdir / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
