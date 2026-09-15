#!/usr/bin/env python3
"""Step 3 (sam3 env, PYTHONNOUSERSITE=1): track the reference object through the
rendered orbit with the SAM 3.1 multiplex video predictor (request API, the
only entry point that loads sam3.1_multiplex.pt). Frames: <work>/track/frames/<i>.jpg
(integer stems, frame 0 = reference camera). Prompting is PER FRAME: the seed
gaussians' silhouette in each view gives SAM3 a box prompt, and the returned mask
that best overlaps that silhouette is kept (>= 0.2 IoU) — no cross-frame tracking to lose.
Writes <work>/track/masks.npz {masks [F,H,W] bool, scores [F] float32, obj_id}."""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SAM3_ROOT = Path("/home/rtoony/projects/ml/sam3")
CKPT = SAM3_ROOT / "checkpoints" / "sam3.1_multiplex.pt"
BPE = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isolate.orbits import mask_bbox, mask_iou  # noqa: E402  (numpy-only)


def main() -> int:
    work = Path(sys.argv[1]); track = work / "track"; t0 = time.time()
    frames = sorted((track / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise SystemExit("no orbit frames")
    cams = json.loads((track / "cams.json").read_text())
    concept = json.loads((work / "concept.json").read_text()).get("concept") if (work / "concept.json").is_file() else None
    mask0 = np.asarray(Image.open(track / "seed_mask.png").convert("L")) > 127
    h, w = mask0.shape

    def rel_box(bb, margin=0.04):
        x0, y0, x1, y1 = bb; mx, my = margin * (x1 - x0), margin * (y1 - y0)
        return [max(0.0, (x0 - mx) / w), max(0.0, (y0 - my) / h), min(1.0, (x1 - x0 + 2 * mx) / w), min(1.0, (y1 - y0 + 2 * my) / h)]

    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=str(CKPT), bpe_path=str(BPE), use_fa3=False)
    sid = predictor.handle_request(request=dict(type="start_session", resource_path=str(track / "frames")))["session_id"]

    def obj_masks(out) -> dict[int, np.ndarray]:
        if not out: return {}
        ids = np.asarray(out["out_obj_ids"]).tolist(); res = {}
        for k, oid in enumerate(ids):
            m = np.asarray(out["out_binary_masks"][k]).astype(bool)
            if m.ndim == 3: m = m[0]
            if m.shape != (h, w):
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
            res[int(oid)] = (m, float(np.asarray(out.get("out_probs", [0] * len(ids)))[k]))
        return res

    masks, scores, valid, raw = [], [], [], {}
    try:
        # Per-frame prompting: the seed gaussians are visible in every kept view by construction, so their
        # silhouette gives SAM3 a box in each frame — no cross-frame tracking to lose.
        for i, cam in enumerate(cams["frames"]):
            sil = np.asarray(Image.open(track / "seed" / f"{i}.png").convert("L")) > 127 if (track / "seed" / f"{i}.png").is_file() else (mask0 if i == 0 else None)
            bb = cam.get("seed_bbox") or (mask_bbox(sil) if sil is not None else None)
            if bb is None:
                masks.append(np.zeros((h, w), bool)); scores.append(0.0); valid.append(False); raw[i] = "no seed silhouette"; continue
            # text names the whole concept, the box says which instance: SAM3 returns the full instance
            # overlapping the box rather than just the part the seed silhouette covers
            req = dict(type="add_prompt", session_id=sid, frame_index=i, bounding_boxes=[rel_box(bb)], bounding_box_labels=[1])
            if concept: req["text"] = concept
            resp = predictor.handle_request(request=req)
            cands = obj_masks(resp.get("outputs"))
            raw[i] = {oid: {"prob": round(p, 4), "cover": round(float(m.mean()), 5), "iou_seed": round(mask_iou(m, sil), 4) if sil is not None else None}
                      for oid, (m, p) in cands.items()}
            if not cands:
                masks.append(np.zeros((h, w), bool)); scores.append(0.0); valid.append(False); continue
            ref_sil = sil if sil is not None else mask0
            oid = max(cands, key=lambda o: mask_iou(cands[o][0], ref_sil))
            m, p = cands[oid]
            ok = bool(m.any()) and mask_iou(m, ref_sil) >= 0.2   # SAM3's pick must actually cover the seed
            masks.append(m if ok else np.zeros((h, w), bool)); scores.append(p if ok else 0.0); valid.append(ok)
    finally:
        try:
            predictor.handle_request(request=dict(type="close_session", session_id=sid))
        finally:
            if hasattr(predictor, "shutdown"):
                predictor.shutdown()   # the multiplex predictor keeps a worker process + the model on the GPU
    M = np.stack(masks); S = np.array(scores, dtype=np.float32); V = np.array(valid, dtype=bool)
    np.savez_compressed(track / "masks.npz", masks=M, scores=S, valid=V, obj_id=-1)
    (track / "raw_outputs.json").write_text(json.dumps(raw, indent=1, default=str))
    cover = M.reshape(len(frames), -1).mean(axis=1)
    iou0 = mask_iou(M[0], mask0)
    (track / "track_receipt.json").write_text(json.dumps({"frames": len(frames), "mode": "per-frame seed-box prompts" + (" + text" if concept else ""), "concept": concept,
                                                          "frame0_iou_vs_seed_mask": round(iou0, 4), "frames_with_object": int(V.sum()),
                                                          "coverage": [round(float(c), 5) for c in cover], "scores": [round(float(x), 4) for x in S],
                                                          "runtime_s": round(time.time() - t0, 1)}, indent=1))
    print(f"[sam3-track] {len(frames)} frames, object found in {int(V.sum())}; frame0 IoU vs seed mask {iou0:.3f}; coverage {cover.min():.3f}..{cover.max():.3f}; {time.time() - t0:.0f}s", flush=True)
    print("ISOLATE_SAM3_TRACK_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
