"""langfield_realign: exact-xyz post-edit realignment — the rebuild lane's
core. Pure-numpy paths only (the torch/eval_setup cold path is exercised live,
never here)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import langfield_realign as lr  # noqa: E402
from langfield_align import MAP_FILENAME, read_ply_xyz  # noqa: E402


def _write_splat_ply(path: Path, xyz: np.ndarray) -> None:
    props = ["x", "y", "z"]
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0",
         f"element vertex {len(xyz)}",
         *(f"property float {p}" for p in props),
         "end_header", ""]
    ).encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        for row in np.asarray(xyz, dtype="<f4"):
            fh.write(struct.pack("<3f", *row))


def _mk_scene(tmp_path: Path, n_ckpt: int = 100, drop: int = 10, seed: int = 3):
    """Synthetic job: ckpt xyz cached, edited ply = permuted subset, STALE set."""
    rng = np.random.default_rng(seed)
    ckpt = rng.random((n_ckpt, 3), dtype=np.float32)
    keep = rng.permutation(n_ckpt)[: n_ckpt - drop]
    ply = ckpt[keep]  # bit-identical positions, new order

    job = tmp_path / "splat_0e0001"
    lf = job / "_langfield"
    lf.mkdir(parents=True)
    (job / "_preview").mkdir()
    np.savez(lf / "gauss_emb.npz", gauss_emb=np.zeros((n_ckpt, 4), np.float16),
             seen=np.ones(n_ckpt, bool))
    np.save(lf / "ckpt_xyz.npy", ckpt)
    _write_splat_ply(job / "_preview" / "splat.ply", ply)
    (lf / "STALE").write_text("2026-07-26T00:00:00\n")
    return job, ckpt, ply, keep


def _run(job: Path) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(BACKEND / "langfield_realign.py"), str(job)],
        capture_output=True, text=True, cwd=str(BACKEND), timeout=120)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return proc.returncode, out


def test_realign_builds_total_map_and_clears_stale(tmp_path: Path) -> None:
    job, ckpt, ply, keep = _mk_scene(tmp_path)
    code, receipt = _run(job)
    assert code == 0, receipt
    assert receipt["ply_rows"] == len(ply)
    assert receipt["dropped_ckpt_rows"] == 10
    assert not (job / "_langfield" / "STALE").exists()

    new_map = np.load(job / "_langfield" / MAP_FILENAME)
    # every ply row maps to the ckpt row with byte-identical xyz
    assert np.array_equal(ckpt[new_map], ply)
    assert (job / "_langfield" / lr.RECEIPT_NAME).is_file()


def test_snapshot_records_survive_partially(tmp_path: Path) -> None:
    job, ckpt, ply, keep = _mk_scene(tmp_path)
    lf = job / "_langfield"
    # paint 20 ckpt positions; 5 of them were dropped by the edit
    dropped = np.setdiff1d(np.arange(len(ckpt)), keep)
    painted = np.concatenate([keep[:15], dropped[:5]])
    (lf / "overrides.json").write_text(json.dumps([
        {"id": "aa11", "label": "hedge", "op": "assign", "count": 20}
    ]))
    np.save(lf / "override_idx_aa11.npy", np.arange(20, dtype=np.uint32))  # stale rows
    np.save(lf / "override_xyz_aa11.npy", ckpt[painted])

    code, receipt = _run(job)
    assert code == 0, receipt
    rec = next(r for r in receipt["records"] if r["id"] == "aa11")
    assert rec == {"store": "overrides.json", "id": "aa11", "label": "hedge",
                   "kept": 15, "dropped": 5, "method": "xyz-snapshot"}

    new_rows = np.load(lf / "override_idx_aa11.npy")
    ply_xyz = read_ply_xyz(job / "_preview" / "splat.ply")
    # remapped rows point at the surviving painted positions
    surviving = {tuple(v) for v in np.round(ckpt[keep[:15]], 7).tolist()}
    got = {tuple(v) for v in np.round(ply_xyz[new_rows], 7).tolist()}
    assert got == surviving
    # snapshot refreshed to the surviving rows (self-healing)
    assert np.load(lf / "override_xyz_aa11.npy").shape == (15, 3)
    manifest = json.loads((lf / "overrides.json").read_text())
    assert manifest[0]["count"] == 15
    assert "invalid_reason" not in manifest[0]


def test_legacy_record_remaps_through_old_map_and_gains_snapshot(tmp_path: Path) -> None:
    job, ckpt, ply, keep = _mk_scene(tmp_path)
    lf = job / "_langfield"
    # pre-edit ply == ckpt (identity map); record painted old-ply rows 0..19,
    # no xyz snapshot (legacy)
    np.save(lf / MAP_FILENAME, np.arange(len(ckpt), dtype=np.int64))
    (lf / "overrides.json").write_text(json.dumps([
        {"id": "bb22", "label": "grass", "op": "boost", "count": 20}
    ]))
    np.save(lf / "override_idx_bb22.npy", np.arange(20, dtype=np.uint32))

    code, receipt = _run(job)
    assert code == 0, receipt
    rec = next(r for r in receipt["records"] if r["id"] == "bb22")
    assert rec["method"] == "old-map-chain"
    survivors = np.intersect1d(np.arange(20), keep)
    assert rec["kept"] == len(survivors)
    # snapshot backfilled for the next edit
    assert (lf / "override_xyz_bb22.npy").is_file()


def test_fully_deleted_record_is_marked_invalid(tmp_path: Path) -> None:
    job, ckpt, ply, keep = _mk_scene(tmp_path)
    lf = job / "_langfield"
    dropped = np.setdiff1d(np.arange(len(ckpt)), keep)
    (lf / "overrides.json").write_text(json.dumps([
        {"id": "cc33", "label": "gone", "op": "assign", "count": len(dropped)}
    ]))
    np.save(lf / "override_idx_cc33.npy", dropped.astype(np.uint32))
    np.save(lf / "override_xyz_cc33.npy", ckpt[dropped])

    code, receipt = _run(job)
    assert code == 0
    rec = next(r for r in receipt["records"] if r["id"] == "cc33")
    assert rec["kept"] == 0 and rec["invalid"]
    assert not (lf / "override_idx_cc33.npy").exists()
    manifest = json.loads((lf / "overrides.json").read_text())
    assert manifest[0]["invalid_reason"]


def test_transformed_geometry_fails_loud_and_changes_nothing(tmp_path: Path) -> None:
    job, ckpt, ply, keep = _mk_scene(tmp_path)
    # simulate a transform: every position shifted — no byte-identical rows
    _write_splat_ply(job / "_preview" / "splat.ply", ply + np.float32(0.001))
    code, out = _run(job)
    assert code == 2
    assert out["error"] == "realign_failed"
    assert out["unmatched"] == len(ply)
    # nothing written, STALE stays — the field remains honestly disabled
    assert (job / "_langfield" / "STALE").exists()
    assert not (job / "_langfield" / MAP_FILENAME).exists()


def test_realign_is_idempotent(tmp_path: Path) -> None:
    job, *_ = _mk_scene(tmp_path)
    assert _run(job)[0] == 0
    code, receipt = _run(job)  # crop-of-a-crop / repeat call
    assert code == 0
    assert receipt["ok"] is True


def test_unit_invert_map_roundtrip() -> None:
    new_map = np.array([5, 2, 9], dtype=np.int64)
    inv = lr.invert_map(new_map, 10)
    assert inv[5] == 0 and inv[2] == 1 and inv[9] == 2
    assert (inv[[0, 1, 3, 4, 6, 7, 8]] == -1).all()
