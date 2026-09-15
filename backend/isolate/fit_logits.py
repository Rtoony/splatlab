#!/usr/bin/env python3
"""Step 4 (langfield env): one temporary foreground logit per gaussian, fitted
by differentiable rendering against the tracked masks over the orbit; threshold
-> object gaussian set. Writes <work>/object_indices.npz, object.ply, receipt.json."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _ckpt import DEV, Splat  # noqa: E402
from isolate import orbits as ob          # noqa: E402
from isolate.splat_ply import write_splat_ply  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path); ap.add_argument("work", type=Path)
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--threshold", type=float, default=0.5); ap.add_argument("--seed-prior", type=float, default=0.1)
    ap.add_argument("--behind-tol", type=float, default=0.08, help="relative depth beyond the object's own front surface that counts as 'behind'")
    ap.add_argument("--behind-frac", type=float, default=0.8, help="drop a gaussian that is behind the front in this share of the views it appears in")
    ap.add_argument("--needle-mult", type=float, default=10.0, help="drop gaussians whose largest scale exceeds this many gaussian spacings")
    ap.add_argument("--vote-frac", type=float, default=0.6, help="drop gaussians whose projected pixel falls inside the tracked mask in fewer than this share of the views they appear in")
    ap.add_argument("--vote-min-views", type=int, default=3)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="existing object_indices.npz to compare against (e.g. _scene/isolated/<slug>/object_indices.npz)")
    a = ap.parse_args(); t0 = time.time()
    s = Splat(a.config)
    cams = json.loads((a.work / "track" / "cams.json").read_text())
    tracked = np.load(a.work / "track" / "masks.npz"); masks = tracked["masks"].astype(np.float32)
    valid = tracked["valid"].astype(bool) if "valid" in tracked.files else masks.reshape(len(masks), -1).any(axis=1)
    tscores = tracked["scores"].astype(np.float32) if "scores" in tracked.files else np.ones(len(masks), np.float32)
    seed = np.load(a.work / "seed.npz")["indices"]
    w, h = cams["w"], cams["h"]; K = s.K_from(cams["fx"], cams["fy"], cams["cx"], cams["cy"])
    frames = cams["frames"]
    assert masks.shape[0] == len(frames), f"{masks.shape[0]} masks for {len(frames)} frames"
    use = [i for i in range(len(frames)) if valid[i]]
    if not use:
        raise SystemExit("tracker found the object in no frame — nothing to fit")
    # frames where the tracker lost the object say nothing about background: skip them; low-confidence frames weigh less
    weights = {i: (1.0 if tscores[i] >= 0.5 else 0.5) for i in use}
    c2ws = {i: torch.tensor(frames[i]["c2w"], dtype=torch.float32, device=DEV) for i in use}
    targets = torch.from_numpy(masks).to(DEV)
    seed_t = torch.as_tensor(seed, device=DEV)
    logit = torch.full((s.n,), -1.0, device=DEV); logit[seed_t] = 1.0
    logit = logit.requires_grad_(True)
    opt = torch.optim.Adam([logit], lr=a.lr)
    history = []
    for it in range(a.iters):
        opt.zero_grad(set_to_none=True)
        prob = torch.sigmoid(logit)[:, None].expand(-1, 3)
        loss = 0.0
        for fi in use:
            out, _, _ = s.render(c2ws[fi], K, w, h, mode="RGB", colors=prob, sh_degree=None, background=0.0)
            pred = out[..., 0].clamp(1e-4, 1 - 1e-4)
            loss = loss + weights[fi] * torch.nn.functional.binary_cross_entropy(pred, targets[fi])
        loss = loss / sum(weights.values())
        # seed prior: the lifted gaussians were observed on the object; a weak pull keeps the fit from collapsing
        loss = loss + a.seed_prior * torch.nn.functional.binary_cross_entropy_with_logits(logit[seed_t], torch.ones_like(logit[seed_t]))
        loss = loss + 1e-3 * (logit ** 2).mean()
        loss.backward(); opt.step()
        history.append(float(loss))
    with torch.no_grad():
        prob = torch.sigmoid(logit)
        obj = torch.nonzero(prob > a.threshold).squeeze(-1)
        # self-consistency: IoU of the fitted foreground render vs the tracked masks
        ious = []
        for fi in use:
            out, _, _ = s.render(c2ws[fi], K, w, h, mode="RGB", colors=prob[:, None].expand(-1, 3), sh_degree=None, background=0.0)
            ious.append(ob.mask_iou(out[..., 0].cpu().numpy() > 0.5, masks[fi] > 0.5))
    # ── post-fit prune ──────────────────────────────────────────────────────────
    # Mask supervision from a cone of views cannot tell the object from what lies
    # BEHIND it inside the silhouette (wall, floor, floaters): those gaussians pass
    # every mask test. Two geometric rules catch them: (1) a gaussian that sits deeper
    # than the object's own front surface at its pixel in nearly every view it appears
    # in is behind the object, not part of it; (2) needle gaussians many spacings long
    # are reconstruction junk, never object detail.
    spacing = float(np.load(a.work / "seed.npz")["spacing"])
    with torch.no_grad():
        obj_mask = torch.zeros(s.n, dtype=torch.bool, device=DEV); obj_mask[obj] = True
        o_opac = torch.where(obj_mask, s.opac, torch.zeros_like(s.opac))
        seen = torch.zeros(s.n, device=DEV); behind = torch.zeros(s.n, device=DEV)
        appear = torch.zeros(s.n, device=DEV); inside = torch.zeros(s.n, device=DEV)
        zeros3 = s.means.new_zeros(s.n, 3)
        for fi in use:
            front, _, _ = s.render(c2ws[fi], K, w, h, mode="ED", colors=zeros3, sh_degree=None, opac=o_opac)
            front = front[..., 0]
            _, _, info = s.render(c2ws[fi], K, w, h, mode="RGB", colors=zeros3, sh_degree=None, packed=True)
            gid, uv, dz = info["gaussian_ids"], info["means2d"], info["depths"]
            px = uv[:, 0].round().long(); py = uv[:, 1].round().long()
            inf = (px >= 0) & (px < w) & (py >= 0) & (py < h) & obj_mask[gid]
            gid, px, py, dz = gid[inf], px[inf], py[inf], dz[inf]
            # silhouette vote (instance_lift's recipe): where does this gaussian's centre land in the tracked mask?
            appear.index_add_(0, gid, torch.ones_like(dz))
            hit = targets[fi][py, px] > 0.5
            inside.index_add_(0, gid[hit], torch.ones_like(dz[hit]))
            fd = front[py, px]
            valid_front = fd > 0
            seen.index_add_(0, gid[valid_front], torch.ones_like(dz[valid_front]))
            is_behind = valid_front & (dz > fd * (1.0 + a.behind_tol))
            behind.index_add_(0, gid[is_behind], torch.ones_like(dz[is_behind]))
        behind_frac = torch.where(seen > 0, behind / seen.clamp(min=1), torch.zeros_like(seen))
        drop_behind = obj_mask & (seen >= 2) & (behind_frac >= a.behind_frac)
        needle = s.scales.max(dim=1).values > a.needle_mult * spacing
        drop_needle = obj_mask & needle
        vote = torch.where(appear > 0, inside / appear.clamp(min=1), torch.zeros_like(appear))
        drop_vote = obj_mask & (appear >= a.vote_min_views) & (vote < a.vote_frac)
        keep = obj_mask & ~drop_behind & ~drop_needle & ~drop_vote
        obj = torch.nonzero(keep).squeeze(-1)
        prune = {"n_before": int(obj_mask.sum()), "dropped_behind": int(drop_behind.sum()), "dropped_needles": int(drop_needle.sum()),
                 "dropped_vote": int(drop_vote.sum()), "n_after": int(keep.sum()), "behind_tol": a.behind_tol, "behind_frac": a.behind_frac,
                 "needle_mult": a.needle_mult, "vote_frac": a.vote_frac, "vote_min_views": a.vote_min_views}
    idx = obj.cpu().numpy().astype(np.int64)
    # visual receipts: full render | object-only | baseline-only, for the reference and two orbit views
    base_idx = np.unique(np.load(a.baseline)["indices"]) if a.baseline and a.baseline.is_file() else None
    with torch.no_grad():
        picks = [use[0]] + use[len(use) // 2: len(use) // 2 + 1] + use[-1:]
        tiles = []
        for fi in dict.fromkeys(picks):
            full, _, _ = s.render(c2ws[fi], K, w, h, mode="RGB")
            o_opac = torch.zeros_like(s.opac); o_opac[obj] = s.opac[obj]
            only, _, _ = s.render(c2ws[fi], K, w, h, mode="RGB", opac=o_opac)
            row = [full, only]
            if base_idx is not None:
                b_opac = torch.zeros_like(s.opac); bt = torch.as_tensor(base_idx, device=DEV); b_opac[bt] = s.opac[bt]
                brow, _, _ = s.render(c2ws[fi], K, w, h, mode="RGB", opac=b_opac); row.append(brow)
            tiles.append(np.concatenate([(r[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for r in row], axis=1))
        Image.fromarray(np.concatenate(tiles, axis=0)).save(a.work / "receipt_object.png")
    np.savez_compressed(a.work / "object_indices.npz", indices=idx, logits=logit.detach().cpu().numpy().astype(np.float32))
    n_ply = write_splat_ply(a.work / "object.ply", s.means[obj].cpu().numpy(), s.raw["f_dc"][obj].cpu().numpy(),
                            s.raw["opacity"][obj].cpu().numpy(), s.raw["scale"][obj].cpu().numpy(), s.raw["rot"][obj].cpu().numpy())
    receipt = {"schema": "dev.splatlab.isolate-by-reference/v1", "n_gaussians": s.n, "n_seed": int(len(seed)), "n_object": int(len(idx)),
               "seed_retained": round(float(np.isin(seed, idx).mean()), 4), "iters": a.iters, "loss_first_last": [history[0], history[-1]],
               "mask_iou_mean": round(float(np.mean(ious)), 4), "mask_iou_min": round(float(np.min(ious)), 4), "views": len(frames),
               "frames_used": len(use), "seed_prior": a.seed_prior, "prune": prune,
               "threshold": a.threshold, "artifacts": {"indices": "object_indices.npz", "splat": "object.ply"}, "fit_runtime_s": round(time.time() - t0, 1)}
    if a.baseline and a.baseline.is_file():
        base = np.unique(np.load(a.baseline)["indices"]); inter = np.intersect1d(idx, base)
        receipt["baseline"] = {"path": str(a.baseline), "n": int(len(base)), "iou": round(len(inter) / max(1, len(np.union1d(idx, base))), 4),
                               "recall": round(len(inter) / max(1, len(base)), 4), "precision": round(len(inter) / max(1, len(idx)), 4)}
    (a.work / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(f"[fit] prune: {prune['n_before']} -> {prune['n_after']} (behind {prune['dropped_behind']}, needles {prune['dropped_needles']}, vote {prune['dropped_vote']})", flush=True)
    print(f"[fit] {len(idx)} object gaussians of {s.n} (seed {len(seed)}, retained {receipt['seed_retained']}); "
          f"mask IoU mean {receipt['mask_iou_mean']} min {receipt['mask_iou_min']}; loss {history[0]:.3f}->{history[-1]:.3f}; {time.time() - t0:.0f}s", flush=True)
    if "baseline" in receipt:
        b = receipt["baseline"]; print(f"[fit] vs baseline ({b['n']} gaussians): IoU {b['iou']} recall {b['recall']} precision {b['precision']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
