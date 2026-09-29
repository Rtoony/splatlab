"""Split mirror: a symlink copy of a job's `processed/` dataset whose image
names say which side of nerfstudio's held-out split they are on.

Spirula's `--eval-mode filename` holds out every image whose basename contains
`eval` and trains on every one containing `train` (and refuses any other name),
so renaming is how an external trainer is made to hold out EXACTLY the photos
`ns-eval` scores. The split itself is never re-derived here: the caller passes
the eval list nerfstudio's own dataparser produced (tools/trainer-bakeoff.py
`split`). Nothing under the source dataset is written; the mirror is symlinks
plus a rewritten transforms.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

TRAIN_PREFIX = "train_"
EVAL_PREFIX = "eval_"
PATH_KEYS = ("file_path", "mask_path", "depth_file_path")


def mirror_name(rel_path: str, is_eval: bool) -> str:
    """`images/sub/DSCF1.JPG` → `eval_sub__DSCF1.JPG` (flat, split-tagged)."""
    rel = Path(rel_path)
    parts = rel.parts[1:] if len(rel.parts) > 1 else rel.parts
    return (EVAL_PREFIX if is_eval else TRAIN_PREFIX) + "__".join(parts)


def _check_basename(rel_path: str) -> None:
    base = Path(rel_path).name.lower()
    if "train" in base or "eval" in base:
        raise ValueError(
            f"{rel_path}: an original name containing 'train'/'eval' would be misread by "
            "--eval-mode filename; refusing to build an ambiguous split")


def build_mirror(processed: Path, eval_names: list[str], out: Path) -> dict:
    """Write `out/transforms.json` + `out/images/*` symlinks. `eval_names` are
    frame `file_path`s exactly as they appear in the source transforms.json.
    Returns a receipt with the counts and the name map."""
    processed, out = Path(processed).resolve(), Path(out)
    meta = json.loads((processed / "transforms.json").read_text())
    frames = meta.get("frames") or []
    by_path = {f["file_path"]: f for f in frames}
    missing = sorted(set(eval_names) - set(by_path))
    if missing:
        raise ValueError(f"{len(missing)} eval names are not frames of transforms.json, e.g. {missing[:3]}")
    if len(set(eval_names)) != len(eval_names):
        raise ValueError("duplicate names in the eval list")
    eval_set = set(eval_names)
    (out / "images").mkdir(parents=True, exist_ok=True)
    name_map: dict[str, str] = {}
    new_frames = []
    for frame in frames:
        rel = frame["file_path"]
        _check_basename(rel)
        is_eval = rel in eval_set
        new = dict(frame)
        for key in PATH_KEYS:
            if key not in frame:
                continue
            src = processed / frame[key]
            if not src.is_file():
                raise FileNotFoundError(src)
            sub = "images" if key == "file_path" else key.split("_")[0] + "s"
            dst_rel = f"{sub}/{mirror_name(frame[key], is_eval)}"
            dst = out / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src, dst)
            new[key] = dst_rel
        name_map[rel] = new["file_path"]
        new_frames.append(new)
    ply = meta.get("ply_file_path")
    if ply:
        src = processed / ply
        dst = out / Path(ply).name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)
        meta["ply_file_path"] = dst.name
    meta["frames"] = new_frames
    (out / "transforms.json").write_text(json.dumps(meta, indent=2))
    receipt = {
        "v": 1,
        "source": str(processed),
        "frames": len(new_frames),
        "train": len(new_frames) - len(eval_set),
        "eval": len(eval_set),
        "eval_names": list(eval_names),
        "name_map": name_map,
    }
    (out / "split-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def rig_timestamp_eval_names(frame_paths: list[str], timestamps: int, every: int, offset: int = 0) -> list[str]:
    """Held-out names for a SplatLab rig job: `frame_NNNNN` is virtual view
    (N-1) // timestamps at timestamp (N-1) % timestamps, and every view of one
    timestamp shares one camera centre. Holding out single views would leave
    their siblings (same instant, overlapping content) in training, so whole
    timestamps are held out: every `every`-th one, all its views together."""
    import re
    out = []
    for rel in frame_paths:
        m = re.fullmatch(r"frame_(\d+)", Path(rel).stem)
        if not m:
            raise ValueError(f"{rel}: not a rig frame_NNNNN name")
        t = (int(m.group(1)) - 1) % timestamps
        if t % every == offset:
            out.append(rel)
    return out
