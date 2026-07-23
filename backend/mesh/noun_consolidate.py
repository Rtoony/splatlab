#!/usr/bin/env python3
"""P6b: consolidate raw noun candidates (Qwen3-VL renders + langfield vocab)
into a deduped, stuff/things-split list.

The stuff/things split and the pure string classification are plain stdlib —
importable for unit tests with no heavy deps. SigLIP-cosine dedupe (the one
step that needs a model) is isolated inside siglip_dedupe() so importing this
module never drags in torch/transformers unless dedupe is actually called.

Runs in the langfield-spike env (CPU-only; the SigLIP text tower is small and
this only embeds a handful of short strings — not worth a GPU arbiter lease).

Usage: noun_consolidate.py <raw_nouns.json> <out.json> [--dedup-thresh 0.85]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slugify import slug  # noqa: E402

# Ground/vegetation/background classes: P5a's semantic_ground.py already owns
# these (GROUND_QUERIES there). A scene-inventory "thing" candidate that names
# one of these is stuff, not an object — never a proxy candidate. Curated list
# in the same spirit as GROUND_QUERIES/NEGATIVES elsewhere in this codebase.
STUFF_TERMS = (
    "ground", "grass", "lawn", "dirt", "soil", "gravel", "pavement", "path",
    "sidewalk", "driveway", "pavers", "concrete floor", "floor", "road",
    "hedge", "bush", "shrub", "foliage", "vegetation", "tree canopy",
    "tree", "leaves", "sky", "cloud", "wall", "fence", "background",
    "shadow", "mulch", "flower bed", "garden bed",
)


def classify_stuff(noun: str) -> bool:
    """True if `noun` names a ground/vegetation/background class (stuff, not
    a proxy-able thing). Substring match against STUFF_TERMS, lowercased."""
    n = noun.strip().lower()
    return any(term in n for term in STUFF_TERMS)


def clean_nouns(raw: list[str], max_nouns: int) -> list[str]:
    """Trim, lowercase-fold duplicates, drop empties, cap length."""
    seen: set[str] = set()
    out: list[str] = []
    for n in raw:
        n = str(n).strip()
        if not n or len(n) > 60:
            continue
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= max_nouns:
            break
    return out


def siglip_dedupe(nouns: list[str], thresh: float) -> tuple[list[str], dict[str, str]]:
    """Cosine-similarity dedupe over SigLIP2 text embeddings (CPU). Returns
    (representatives, {dropped_noun: kept_representative})."""
    if len(nouns) <= 1:
        return list(nouns), {}
    import torch
    from transformers import AutoModel, AutoProcessor

    SIGLIP_CKPT = "google/siglip2-so400m-patch16-384"
    siglip = AutoModel.from_pretrained(SIGLIP_CKPT, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_CKPT)

    with torch.no_grad():
        inp = proc(text=nouns, padding="max_length", max_length=64,
                   truncation=True, return_tensors="pt")
        emb = torch.nn.functional.normalize(
            siglip.get_text_features(**inp).pooler_output.float(), dim=-1)
    sim = (emb @ emb.T).numpy()

    kept: list[str] = []
    dropped: dict[str, str] = {}
    assigned = [False] * len(nouns)
    # Longer/more-specific-first: prefer keeping the more descriptive noun as
    # the cluster representative (e.g. "round wooden table" over "table").
    order = sorted(range(len(nouns)), key=lambda i: -len(nouns[i]))
    for i in order:
        if assigned[i]:
            continue
        kept.append(nouns[i])
        assigned[i] = True
        for j in order:
            if assigned[j] or j == i:
                continue
            if sim[i, j] >= thresh:
                assigned[j] = True
                dropped[nouns[j]] = nouns[i]
    return kept, dropped


def dedupe_slugs(things: list[str]) -> tuple[list[str], list[dict]]:
    """Downstream (scene_sam3_masks.py, instance_lift.py) key every noun's
    masks/instance files by slug(noun) — two things that reduce to the same
    slug (review finding 2026-07-23: e.g. "Fire Hydrant" and "Fire-Hydrant")
    would silently clobber each other's files. siglip_dedupe only catches
    near-duplicates PROBABILISTICALLY; this is the mechanical backstop.
    First-kept-wins, order-preserving."""
    seen: dict[str, str] = {}
    kept: list[str] = []
    collisions: list[dict] = []
    for n in things:
        s = slug(n)
        if s in seen:
            collisions.append({"noun": n, "slug": s, "collides_with": seen[s]})
            continue
        seen[s] = n
        kept.append(n)
    return kept, collisions


def consolidate(raw: list[str], max_nouns: int, dedup_thresh: float) -> dict:
    cleaned = clean_nouns(raw, max_nouns * 3)  # dedupe/split before the cap
    kept, dup_map = siglip_dedupe(cleaned, dedup_thresh) if cleaned else ([], {})
    kept = kept[:max_nouns]
    things, stuff = [], []
    for n in kept:
        (stuff if classify_stuff(n) else things).append(n)
    things, slug_collisions = dedupe_slugs(things)
    return {
        "raw_count": len(raw),
        "cleaned_count": len(cleaned),
        "dedup_map": dup_map,
        "things": things,
        "stuff": stuff,
        "slug_collisions": slug_collisions,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_nouns_json", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--max-nouns", type=int, default=12)
    ap.add_argument("--dedup-thresh", type=float, default=0.85)
    args = ap.parse_args()

    raw = json.loads(args.raw_nouns_json.read_text())
    report = consolidate(raw, args.max_nouns, args.dedup_thresh)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
