#!/usr/bin/env python3
"""Per-gaussian GROUND relevancy from a scene's language field (P5a).

Runs in the langfield-spike env. Loads gauss_emb.npz + gaussian centers from
the CHECKPOINT (eval_setup — exported plys are NaN-culled a few thousand rows
short of the lift's row space and cannot be trusted for alignment), embeds
ground queries with SigLIP 2 (LERF min-softmax vs the canonical negatives),
and writes ground_gaussians.npz {xyz, rel, seen}.

Productionized 2026-07-21 from solidify-probe/probe_semantic_ground_a.py
(proven on garden: lawn confirmed, tabletop/hedge vetoed).

Usage: semantic_ground.py <gauss_emb.npz> <splatfacto_config.yml> <out.npz>
"""
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
NEGATIVES = ["object", "things", "stuff", "texture", "surface"]  # LERF canon


def main() -> int:
    emb_path, config_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
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

    q = text_emb(GROUND_QUERIES)
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
        rel = torch.stack(rel_all, dim=1).max(dim=1).values
    rel = rel.cpu().numpy().astype(np.float32)
    rel[~seen] = 0.0  # unseen = unknown, never "ground"

    np.savez_compressed(out_path, xyz=xyz, rel=rel, seen=seen)
    print(json.dumps({
        "gaussians": int(len(xyz)),
        "seen": int(seen.sum()),
        "ground_at_0.5": int((rel > 0.5).sum()),
        "queries": GROUND_QUERIES,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
