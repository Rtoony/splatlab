#!/usr/bin/env python3
"""P5b: isolate a named OBJECT from a scene's language field.

Runs in the langfield-spike env. Finds the object with the PRODUCTION focus
math (top-4k relevancy pool -> DBSCAN, eps=spread*0.15 clamp [0.05,0.8],
min_samples=max(6,len//60), peak>=0.58 — mirrored from langfield_worker /
isolate_export), then EXPANDS to full membership for meshing: every gaussian
within focus_radius*expand of the cluster AND above a relevancy floor.

Outputs (out_dir):
  object.ply          raw 3DGS splat of the object (Blender-importer fields)
  object_indices.npz  {indices int64, rel float32} checkpoint-order rows
  object.json         query, counts, focus, radius, bbox (scene units)

Usage: object_isolate.py <gauss_emb.npz> <splatfacto_config.yml> "<query>"
       <out_dir> [--cluster 0] [--expand 1.3] [--rel-floor 0.42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_orig_load = torch.load


def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _patched_load

from sklearn.cluster import DBSCAN  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

DEV = "cuda"
SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
NEGATIVES = ["object", "things", "stuff", "texture", "surface"]  # LERF canon

PLY_FIELDS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def write_splat_ply(path: Path, xyz, f_dc, opacity, scale, rot) -> None:
    """Raw/pre-activation fields — the format the Blender importer round-trips
    (numerics mirrored from the proven isolate_export.py writer)."""
    m = xyz.shape[0]
    rows = np.concatenate([
        xyz.astype("<f4"), f_dc.astype("<f4"), opacity.astype("<f4")[:, None],
        scale.astype("<f4"), rot.astype("<f4"),
    ], axis=1)
    header = "ply\nformat binary_little_endian 1.0\n"
    header += "comment SplatLab object isolate (P5b)\n"
    header += f"element vertex {m}\n"
    for f in PLY_FIELDS:
        header += f"property float {f}\n"
    header += "end_header\n"
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rows.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("emb")
    ap.add_argument("config")
    ap.add_argument("query")
    ap.add_argument("out_dir")
    ap.add_argument("--cluster", type=int, default=0)
    ap.add_argument("--expand", type=float, default=1.3)
    ap.add_argument("--rel-floor", type=float, default=0.42)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.emb)
    feat = torch.tensor(d["gauss_emb"]).float().to(DEV)
    seen = d["seen"].astype(bool)

    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(Path(args.config), test_mode="test")
    model = pipeline.model
    means = model.means.detach().cpu().numpy().astype(np.float32)
    if len(means) != feat.shape[0]:
        print(f"FATAL: row mismatch checkpoint {len(means)} vs emb {feat.shape[0]}", file=sys.stderr)
        return 1

    siglip = AutoModel.from_pretrained(
        SIGLIP_CKPT, dtype=torch.float16, attn_implementation="sdpa"
    ).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_CKPT)

    @torch.no_grad()
    def text_emb(prompts):
        inp = proc(text=list(prompts), padding="max_length", max_length=64,
                   truncation=True, return_tensors="pt").to(DEV)
        e = siglip.get_text_features(**inp).pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1)

    q = text_emb([args.query])
    n = text_emb(NEGATIVES)
    with torch.no_grad():
        sim_q = (feat @ q.T).squeeze(-1)
        sim_n = feat @ n.T
        pair = torch.stack([sim_q[:, None].expand(-1, sim_n.shape[1]), sim_n], dim=-1)
        rel = torch.softmax(pair * 10.0, dim=-1)[..., 0].min(dim=1).values
    rel[~torch.tensor(seen, device=DEV)] = 0.0

    # ── production focus math (top-4k pool -> DBSCAN) ────────────────────────
    k = min(4000, rel.shape[0])
    topv, topi = torch.topk(rel, k)
    keep = topv > 0.42
    idx = topi[keep] if int(keep.sum()) >= 30 else topi[: min(80, rel.shape[0])]
    idx_np = idx.cpu().numpy()
    pts = means[idx_np]
    wts = rel[idx].cpu().numpy()
    spread = float(np.linalg.norm(pts.std(0)) + 1e-6)
    eps = float(np.clip(spread * 0.15, 0.05, 0.8))
    labels = DBSCAN(eps=eps, min_samples=max(6, len(pts) // 60)).fit_predict(pts)

    matches = []
    for lab in set(labels.tolist()):
        if lab == -1:
            continue
        m = labels == lab
        if float(wts[m].max()) < 0.58:
            continue
        cp = pts[m]
        c = cp.mean(0)
        radius = float(np.linalg.norm(cp - c, axis=1).max())
        matches.append({"focus": c, "radius": radius,
                        "score": float(wts[m].sum()), "count": int(m.sum()),
                        "bbox_tight": (cp.min(0), cp.max(0))})
    if not matches:
        print(f"FATAL: no cluster with peak relevancy >= 0.58 for {args.query!r}", file=sys.stderr)
        return 1
    matches.sort(key=lambda x: x["score"], reverse=True)
    if args.cluster >= len(matches):
        print(f"FATAL: cluster {args.cluster} of {len(matches)} requested", file=sys.stderr)
        return 1
    chosen = matches[args.cluster]

    # ── expansion: full membership for meshing ───────────────────────────────
    rel_np = rel.cpu().numpy()
    dist = np.linalg.norm(means - chosen["focus"][None, :], axis=1)
    member = (dist <= chosen["radius"] * args.expand) & (rel_np >= args.rel_floor)
    indices = np.flatnonzero(member).astype(np.int64)
    if len(indices) < 200:
        print(f"FATAL: only {len(indices)} member gaussians after expansion", file=sys.stderr)
        return 1

    f_dc = model.features_dc.detach().cpu().numpy().astype(np.float32)
    opac = model.opacities.detach().cpu().numpy().astype(np.float32).squeeze(-1)
    scale = model.scales.detach().cpu().numpy().astype(np.float32)
    quat = model.quats.detach().cpu().numpy().astype(np.float32)
    write_splat_ply(out_dir / "object.ply", means[member], f_dc[member],
                    opac[member], scale[member], quat[member])
    np.savez_compressed(out_dir / "object_indices.npz",
                        indices=indices, rel=rel_np[indices].astype(np.float32))

    obj_xyz = means[member]
    report = {
        "query": args.query,
        "cluster": args.cluster,
        "clusters_found": len(matches),
        "pool_members": chosen["count"],
        "expanded_members": int(len(indices)),
        "expand": args.expand,
        "rel_floor": args.rel_floor,
        "focus_scene": [round(float(x), 4) for x in chosen["focus"]],
        "radius_scene": round(chosen["radius"], 4),
        "bbox_scene": {
            "min": [round(float(x), 4) for x in obj_xyz.min(0)],
            "max": [round(float(x), 4) for x in obj_xyz.max(0)],
        },
        # Cluster-pool bbox: TIGHT (no expansion debris) — use for photo crops.
        "bbox_tight": {
            "min": [round(float(x), 4) for x in chosen["bbox_tight"][0]],
            "max": [round(float(x), 4) for x in chosen["bbox_tight"][1]],
        },
        "artifacts": {"splat": "object.ply", "indices": "object_indices.npz"},
    }
    (out_dir / "object.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
