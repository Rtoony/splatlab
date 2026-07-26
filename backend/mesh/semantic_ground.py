#!/usr/bin/env python3
"""Per-gaussian GROUND relevancy from a scene's language field (P5a).

Runs in the langfield-spike env. Loads gauss_emb.npz + gaussian centers from
the CHECKPOINT (eval_setup — exported plys are NaN-culled a few thousand rows
short of the lift's row space and cannot be trusted for alignment), embeds
ground queries with SigLIP 2 (LERF min-softmax vs the canonical negatives),
and writes ground_gaussians.npz {xyz, rel, seen}.

Productionized 2026-07-21 from solidify-probe/probe_semantic_ground_a.py
(proven on garden: lawn confirmed, tabletop/hedge vetoed).

Class-aware since 2026-07-26: the per-query relevancy that was always computed
is now KEPT (class_rel [N,C] + class_ids in the npz) instead of collapsed away,
queries come from the class taxonomy's ground category when --taxonomy is
given, --live-map zeroes rows the current exported ply no longer contains
(crops used to resurrect deleted ground), and --class-labels gives USER paint
absolute precedence: painted ground classes become one-hot certainty, painted
non-ground classes veto ground regardless of what SigLIP thinks.

Usage: semantic_ground.py <gauss_emb.npz> <splatfacto_config.yml> <out.npz>
       [--taxonomy class_taxonomy.json] [--live-map ply_index_map.npy]
       [--class-labels <lfdir>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

_orig_load = torch.load


def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _patched_load

DEV = "cuda"
SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
GROUND_QUERIES = ["grass lawn", "dirt ground", "stone pavement", "concrete ground"]
# LERF canonical negatives + vegetation-mass negatives: ground must BEAT hedge,
# not just generic "stuff" — hedge-base gaussians were creeping into the ground
# set (section receipt, 2026-07-21) because nothing green competed for them.
NEGATIVES = ["object", "things", "stuff", "texture", "surface",
             "green bushes", "hedge", "leafy foliage", "tree canopy"]


def _load_ground_classes(taxonomy_path: str | None):
    """(class_ids, per-class queries) — taxonomy ground category when given,
    else the legacy hard-wired list mapped onto stable ids."""
    if taxonomy_path:
        doc = json.loads(Path(taxonomy_path).read_text())
        ground = [c for c in doc.get("classes", []) if c.get("category") == "ground"]
        if ground:
            return ([c["id"] for c in ground],
                    [c.get("queries") or [c["display"]] for c in ground])
    legacy_ids = ["grass", "dirt", "pavement", "concrete"]
    return legacy_ids, [[q] for q in GROUND_QUERIES]


def _apply_user_labels(class_rel: np.ndarray, class_ids: list[str],
                       lfdir: Path, n_rows: int, live_map: np.ndarray | None):
    """User paint = absolute truth. Painted ground classes become one-hot 1.0;
    painted NON-ground classes zero every ground channel for those rows.
    Class-label indices are EXPORTED-PLY rows; live_map converts to ckpt rows."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import class_labels as cl  # noqa: PLC0415 — worker-side numpy module

    pos = {cid: i for i, cid in enumerate(class_ids)}
    painted_rows = 0
    for record in cl.load_manifest(lfdir):
        if record.get("invalid_reason"):
            continue
        idx_path = lfdir / f"class_idx_{record.get('id', '')}.npy"
        if not idx_path.is_file():
            continue
        rows = np.load(idx_path).astype(np.int64)
        if live_map is not None:
            rows = rows[(rows >= 0) & (rows < live_map.size)]
            rows = live_map[rows]
        rows = rows[(rows >= 0) & (rows < n_rows)]
        if not rows.size:
            continue
        painted_rows += int(rows.size)
        channel = pos.get(record.get("class_id"))
        class_rel[rows, :] = 0.0
        if channel is not None:  # a ground class -> one-hot certainty
            class_rel[rows, channel] = 1.0
        # non-ground paint (vegetation/structure/...) leaves all channels 0 =
        # a hard ground veto for those gaussians.
    return painted_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_path")
    ap.add_argument("config_path")
    ap.add_argument("out_path")
    ap.add_argument("--taxonomy", default=None)
    ap.add_argument("--live-map", default=None,
                    help="ply_index_map.npy — zero rows absent from the "
                         "current exported ply (post-crop honesty)")
    ap.add_argument("--class-labels", default=None,
                    help="_langfield dir with user class paints (absolute "
                         "precedence over SigLIP)")
    args = ap.parse_args()
    emb_path, config_path, out_path = args.emb_path, args.config_path, args.out_path
    d = np.load(emb_path)
    feat = torch.tensor(d["gauss_emb"]).float().to(DEV)
    seen = d["seen"].astype(bool)

    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(Path(config_path), test_mode="test")
    xyz = pipeline.model.means.detach().cpu().numpy().astype(np.float32)
    if len(xyz) != feat.shape[0]:
        print(f"FATAL: row mismatch checkpoint {len(xyz)} vs emb {feat.shape[0]}", file=sys.stderr)
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

    class_ids, class_queries = _load_ground_classes(args.taxonomy)
    flat_queries = [q for group in class_queries for q in group]
    q = text_emb(flat_queries)
    n = text_emb(NEGATIVES)
    with torch.no_grad():
        sim_q = feat @ q.T
        sim_n = feat @ n.T
        rel_all = []
        for qi in range(sim_q.shape[1]):
            pair = torch.stack(
                [sim_q[:, qi : qi + 1].expand(-1, sim_n.shape[1]), sim_n], dim=-1
            )
            soft = torch.softmax(pair * 10.0, dim=-1)[..., 0]
            rel_all.append(soft.min(dim=1).values)
        per_query = torch.stack(rel_all, dim=1)  # [N, sum(queries)]
        # Collapse queries -> classes (max within a class's query group), and
        # KEEP the class axis this time — argmax over it is the free automatic
        # classification the paint tool's suggestion layer and the class-
        # textured bake consume.
        chunks = []
        start = 0
        for group in class_queries:
            chunks.append(per_query[:, start:start + len(group)].max(dim=1).values)
            start += len(group)
        class_rel_t = torch.stack(chunks, dim=1)  # [N, C]
        rel = class_rel_t.max(dim=1).values
    class_rel = class_rel_t.cpu().numpy().astype(np.float16)
    rel = rel.cpu().numpy().astype(np.float32)
    rel[~seen] = 0.0  # unseen = unknown, never "ground"
    class_rel[~seen, :] = 0.0

    # Post-crop honesty: rows the current exported ply no longer contains must
    # not resurrect as ground (this script reads CHECKPOINT rows).
    if args.live_map and Path(args.live_map).is_file():
        live = np.load(args.live_map).astype(np.int64)
        alive = np.zeros(len(xyz), dtype=bool)
        alive[live[(live >= 0) & (live < len(xyz))]] = True
        rel[~alive] = 0.0
        class_rel[~alive, :] = 0.0
        live_for_labels = live
    else:
        live_for_labels = None

    painted = 0
    if args.class_labels and Path(args.class_labels).is_dir():
        class_rel = class_rel.astype(np.float32)
        painted = _apply_user_labels(
            class_rel, class_ids, Path(args.class_labels), len(xyz), live_for_labels)
        rel = class_rel.max(axis=1).astype(np.float32)
        class_rel = class_rel.astype(np.float16)

    np.savez_compressed(out_path, xyz=xyz, rel=rel, seen=seen,
                        class_rel=class_rel,
                        class_ids=np.array(class_ids))
    print(json.dumps({
        "gaussians": int(len(xyz)),
        "seen": int(seen.sum()),
        "ground_at_0.5": int((rel > 0.5).sum()),
        "classes": class_ids,
        "painted_rows_applied": painted,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
