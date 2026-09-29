#!/usr/bin/env python3
"""External-trainer bake-off helpers (research only; see
~/reports/2026-09-29-spirula-bakeoff/). Runs in the `langfield-spike` env, the
one that has nerfstudio 1.1.5, so the held-out split and the metrics are
nerfstudio's own rather than re-implementations.

  split   JOB OUT             dump nerfstudio's eval split for JOB and build a
                              symlink mirror whose names carry the split
  score   ARM_DIR --mirror M  score one arm's held-out renders against the
                              photos on disk (PSNR/SSIM/LPIPS, raw + colour-corrected)
  to-ns-frame PLY --dataparser-transforms F  move a PLY trained on transforms.json into the
                              nerfstudio viewer frame, SH0 (Spark comparison)
  align / cross-render        score another SfM's model at SplatLab's held-out cameras
  xdataset                    their train images + SplatLab's held-out cameras, for their own renderer
  count   ARM_DIR             gaussian count of a trained arm
  summary SCENE_DIR           results table over every scored arm
  check-renders ARM_DIR       exit 0 iff every held-out photo has exactly one render
  sheet   OUT --mirror M --arms L=score.json..  photo | arm | arm contact sheet

CPU-only except `score` (LPIPS), which must go through
tools/splatlab-compute-gate.sh like every GPU command.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from bakeoff import cross  # noqa: E402
from bakeoff import ply_frame  # noqa: E402
from bakeoff import scoring  # noqa: E402
from bakeoff import split as bsplit  # noqa: E402

OUTPUTS = Path.home() / "projects" / "splatcli" / "outputs" / "3d"


def job_dir(job: str) -> Path:
    p = Path(job)
    return p if p.is_dir() else OUTPUTS / job


def latest_config(job: Path) -> Path:
    configs = sorted((job / "processed" / "splatfacto").glob("*/config.yml"))
    if not configs:
        raise SystemExit(f"{job}: no processed/splatfacto/*/config.yml")
    return configs[-1]


def nerfstudio_eval_names(job: Path) -> list[str]:
    """Frame file_paths of nerfstudio's `test` split, in eval-dataloader order,
    from the job's own dataparser config (so any non-default split is kept)."""
    import yaml
    processed = (job / "processed").resolve()
    # The job's own ns-train config (typed !!python/object tags); ns-eval loads it the same way.
    cfg = yaml.load(latest_config(job).read_text(), Loader=yaml.Loader)
    dp_cfg = cfg.pipeline.datamanager.dataparser
    dp_cfg.data = processed
    outs = dp_cfg.setup().get_dataparser_outputs(split="test")
    return [str(Path(p).resolve().relative_to(processed)) for p in outs.image_filenames]


def cmd_split(args) -> int:
    job = job_dir(args.job)
    if args.rig_timestamps:
        frames = json.loads((job / "processed" / "transforms.json").read_text())["frames"]
        names = bsplit.rig_timestamp_eval_names([f["file_path"] for f in frames], args.rig_timestamps,
                                                args.holdout_every)
    else:
        names = nerfstudio_eval_names(job)
    receipt = bsplit.build_mirror(job / "processed", names, Path(args.out))
    print(json.dumps({k: receipt[k] for k in ("source", "frames", "train", "eval")}, indent=2))
    print("first eval frames:", names[:3])
    return 0


def load_split(mirror: Path) -> dict:
    return json.loads((Path(mirror) / "split-receipt.json").read_text())


def cmd_score(args) -> int:
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    photos = {n: src / n for n in receipt["eval_names"]}   # paths: decoded one at a time
    pairs = scoring.ARM_LAYOUTS[args.layout](Path(args.renders))
    result = scoring.score(pairs, photos, scoring.Metrics(args.device), allow_missing=args.allow_missing)
    result.update({"layout": args.layout, "renders": str(Path(args.renders).resolve()),
                   "mirror": str(Path(args.mirror).resolve())})
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2))
    m = result["mean"]
    print(f"{out}: {result['views']} views  PSNR {m['psnr']:.3f}  SSIM {m['ssim']:.4f}  LPIPS {m['lpips']:.4f}"
          f"  | cc PSNR {m['cc_psnr']:.3f}  SSIM {m['cc_ssim']:.4f}  LPIPS {m['cc_lpips']:.4f}"
          f"  (max GT match {result['max_gt_match_mean_abs_255']:.2f}/255)")
    return 0


def cmd_check_renders(args) -> int:
    """Exit 0 iff the renders cover every held-out photo exactly once (CPU only).
    Spirula 2026.9.24's parallel eval writer sometimes writes one view twice and
    another never; the runner re-evaluates from the full checkpoint until this passes."""
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    photos = {n: src / n for n in receipt["eval_names"]}
    try:
        scoring.match_to_photos(scoring.ARM_LAYOUTS[args.layout](Path(args.renders)), photos)
    except ValueError as exc:
        print(f"INCOMPLETE: {exc}")
        return 4
    print(f"COMPLETE: {len(photos)} held-out photos, one render each")
    return 0


def cmd_to_ns_frame(args) -> int:
    from isolate.splat_ply import write_splat_ply
    cols = ply_frame.read_vertex_ply(Path(args.ply))
    if args.dataparser_transforms == "identity":  # already in the nerfstudio frame: SH0 strip only
        r, t, s = np.eye(3), np.zeros(3), 1.0
    else:
        r, t, s = ply_frame.load_dataparser_transform(Path(args.dataparser_transforms))
    # No applied_transform step: nerfstudio's saved dataparser transform already INCLUDES it (it maps the original
    # COLMAP frame to the viewer frame), and Spirula trained on transforms.json keeps its splats in that original
    # COLMAP frame — so the dataparser transform alone is exact (09-29: NN overlap with the arm's own ns-export
    # 0.03 % of the span; applying applied_transform on top = 2.7 %, FAIL).
    xyz, f_dc, opacity, scale, rot = ply_frame.to_ns_frame(cols, r, t, s)
    n = write_splat_ply(Path(args.out), xyz, f_dc, opacity, scale, rot,
                        comment="trainer-bakeoff to-ns-frame SH0")
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    report = {"gaussians": n, "centroid": xyz.mean(axis=0).round(4).tolist(),
              "p1": np.percentile(xyz, 1, axis=0).round(4).tolist(),
              "p99": np.percentile(xyz, 99, axis=0).round(4).tolist(),
              "bbox": [lo.round(4).tolist(), hi.round(4).tolist()], "scale": s}
    ok = True
    if args.reference:
        # Overlap, not a bounding box: a box test passed a model rotated 90° about x (09-29). Two splat models of
        # one scene in one frame put most gaussians within ~1% of the scene span of each other.
        from scipy.spatial import cKDTree
        ref = ply_frame.read_vertex_ply(Path(args.reference))
        rxyz = np.stack([ref["x"], ref["y"], ref["z"]], axis=1)
        rng = np.random.default_rng(0)
        a = xyz[rng.choice(len(xyz), min(50000, len(xyz)), replace=False)]
        b = rxyz[rng.choice(len(rxyz), min(50000, len(rxyz)), replace=False)]
        span = float(np.linalg.norm(np.percentile(rxyz, 99, 0) - np.percentile(rxyz, 1, 0)))
        d = cKDTree(b).query(a)[0] / span
        report["nn_to_reference_over_span"] = {"median": round(float(np.median(d)), 5),
                                               "p75": round(float(np.percentile(d, 75)), 5)}
        ok = report["nn_to_reference_over_span"]["median"] < args.max_nn
        report["frame_check"] = "pass" if ok else "FAIL"
    print(json.dumps(report, indent=2))
    if not ok:
        print(f"FRAME CHECK FAILED: median NN distance to the reference > {args.max_nn} of its span", file=sys.stderr)
        return 3
    return 0


def cmd_sheet(args) -> int:
    from PIL import Image, ImageDraw
    arms = []
    for spec in args.arms:
        label, score_path = spec.split("=", 1)
        arms.append((label, json.loads(Path(score_path).read_text())))
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    rows0 = {r["photo"]: r for r in arms[0][1]["per_view"]}
    rows1 = {r["photo"]: r for r in arms[-1][1]["per_view"]}
    by_gap = sorted(rows0, key=lambda n: rows1[n]["lpips"] - rows0[n]["lpips"])
    by_base = sorted(rows0, key=lambda n: rows0[n]["lpips"])
    picks = []
    for n in (by_gap[0], by_gap[-1], by_base[len(by_base) // 2], by_base[-1]):
        if n not in picks:
            picks.append(n)
    tile_w = args.tile_width
    tiles = []
    for name in picks:
        row = [("photo " + Path(name).name, scoring.load_rgb(src / name))]
        for label, res in arms:
            r = next(v for v in res["per_view"] if v["photo"] == name)
            pairs = scoring.ARM_LAYOUTS[res["layout"]](Path(res["renders"]))
            pair = next(p for p in pairs if p.key == r["arm_key"])
            row.append((f"{label}  {r['psnr']:.2f} dB  LPIPS {r['lpips']:.3f}", pair.load_pred()))
        tiles.append(row)
    h0, w0 = tiles[0][0][1].shape[:2]
    tile_h = int(round(tile_w * h0 / w0))
    sheet = Image.new("RGB", (tile_w * len(tiles[0]), (tile_h + 22) * len(tiles)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(tiles):
        for j, (label, img) in enumerate(row):
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((tile_w, tile_h), Image.LANCZOS)
            sheet.paste(im, (j * tile_w, i * (tile_h + 22) + 22))
            draw.text((j * tile_w + 6, i * (tile_h + 22) + 5), label, fill="black")
    sheet.save(args.out)
    print(f"{args.out}: {len(tiles)} views x {len(tiles[0])} columns: {picks}")
    return 0


def _instant_of_rig_frame(rel: str, timestamps: int) -> int:
    import re
    return (int(re.fullmatch(r"frame_(\d+)", Path(rel).stem).group(1)) - 1) % timestamps


def cmd_align(args) -> int:
    """Sim(3) SplatLab transforms.json frame → another SfM's COLMAP frame, from centres paired by instant."""
    from nerfstudio.data.utils import colmap_parsing_utils as cpu
    import re
    sparse = Path(args.colmap)
    images = (cpu.read_images_binary(sparse / "images.bin") if (sparse / "images.bin").is_file()
              else cpu.read_images_text(sparse / "images.txt"))
    theirs: dict[int, list] = {}
    for im in images.values():
        m = re.match(r"(\d+)", Path(im.name).stem)
        if m and int(m.group(1)) % args.stride == 0:
            theirs.setdefault(int(m.group(1)) // args.stride, []).append(cross.colmap_centre(im.qvec, im.tvec))
    frames = json.loads(Path(args.transforms).read_text())["frames"]
    ours: dict[int, np.ndarray] = {}
    for f in frames:
        ours.setdefault(_instant_of_rig_frame(f["file_path"], args.rig_timestamps), np.array(f["transform_matrix"])[:3, 3])
    common = sorted(set(theirs) & set(ours))
    src = np.stack([ours[i] for i in common])
    dst = np.stack([np.mean(theirs[i], axis=0) for i in common])
    s, r, t = cross.umeyama(src, dst)
    rep = cross.align_report(src, dst, s, r, t)
    rep.update({"from": "splatlab transforms.json frame", "to": str(sparse), "s": s, "R": r.tolist(), "t": t.tolist(),
                "instants_theirs": len(theirs), "instants_ours": len(ours),
                "lens_spread_median": float(np.median([np.linalg.norm(np.ptp(np.stack(v), 0)) for v in theirs.values() if len(v) > 1] or [0]))})
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("pairs", "scale", "rmse", "p95", "extent", "rmse_over_extent",
                                          "instants_theirs", "lens_spread_median")}, indent=1))
    if rep["rmse_over_extent"] > args.max_rel_rmse:
        print(f"ALIGN FAILED: rmse/extent {rep['rmse_over_extent']:.4f} > {args.max_rel_rmse}", file=sys.stderr)
        return 3
    return 0


def _scene_transform(path: str | None):
    if not path:
        return 1.0, np.eye(3), np.zeros(3)
    d = json.loads(Path(path).read_text())["train_from_world"]
    return float(d["scale"]), np.array(d["rotation"]["matrix_3x3"], float), np.array(d["translation"], float)


def cmd_cross_render(args) -> int:
    """Render a 3DGS PLY (its own frame) at the mirror's held-out cameras; write Spirula-layout
    eval-gt/eval-render PNG pairs so `score` / `check-renders` apply unchanged."""
    import torch
    from gsplat import rasterization
    from PIL import Image
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    meta = json.loads((src / "transforms.json").read_text())
    by_path = {f["file_path"]: f for f in meta["frames"]}
    if args.align:
        al = json.loads(Path(args.align).read_text())
        s, r, t = float(al["s"]), np.array(al["R"]), np.array(al["t"])
    else:
        s, r, t = 1.0, np.eye(3), np.zeros(3)
    if args.colmap_frame:   # Spirula trained on this transforms.json keeps its splats in the ORIGINAL COLMAP frame
        a = np.array(meta["applied_transform"], float)
        r = a[:3, :3].T @ r
        t = a[:3, :3].T @ (t - a[:3, 3])
    s2, r2, t2 = _scene_transform(args.scene_transform)          # their world → their training frame
    s, r, t = s2 * s, r2 @ r, s2 * r2 @ t + t2
    cols = ply_frame.read_vertex_ply(Path(args.ply))
    dev = "cuda"
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    means = f32(np.stack([cols["x"], cols["y"], cols["z"]], 1))
    quats = f32(np.stack([cols[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(f32(np.stack([cols[f"scale_{i}"] for i in range(3)], 1)))
    opac = torch.sigmoid(f32(cols["opacity"]))
    dc = np.stack([cols[f"f_dc_{i}"] for i in range(3)], 1)[:, None, :]
    n_rest = sum(1 for k in cols if k.startswith("f_rest_"))
    rest = np.stack([cols[f"f_rest_{i}"] for i in range(n_rest)], 1).reshape(len(dc), 3, n_rest // 3).transpose(0, 2, 1)
    sh = f32(np.concatenate([dc, rest], 1))
    deg = int(round(np.sqrt(sh.shape[1]))) - 1
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(receipt["eval_names"]):
        f = by_path[name]
        w, h = int(f.get("w", meta.get("w"))), int(f.get("h", meta.get("h")))
        k = np.array([[f.get("fl_x", meta.get("fl_x")), 0, f.get("cx", meta.get("cx"))],
                      [0, f.get("fl_y", meta.get("fl_y")), f.get("cy", meta.get("cy"))], [0, 0, 1]], float)
        view = cross.c2w_gl_to_viewmat(np.array(f["transform_matrix"]), s, r, t)
        with torch.no_grad():
            img, _, _ = rasterization(means, quats, scales, opac, sh, f32(view)[None], f32(k)[None], w, h,
                                      sh_degree=deg, rasterize_mode=args.mode)   # black background
        arr = (img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(out / f"eval-render-{i:05d}.png")
        with Image.open(src / name) as ph:
            ph.convert("RGB").save(out / f"eval-gt-{i:05d}.png")
    print(f"{out}: {len(receipt['eval_names'])} views rendered (sh_degree {deg}, {args.mode}, "
          f"{len(dc)} gaussians, camera map scale {s:.4f})")
    return 0


def cmd_xdataset(args) -> int:
    """COLMAP dataset = another SfM's TRAIN images (e.g. Spirula on the raw .insv) + SplatLab's held-out views
    placed in that SfM's frame (via `align`), named *_eval. Spirula then renders SplatLab's own held-out cameras
    with its OWN renderer, so the model can be scored against the exact photos splatfacto is scored on (a gsplat
    replica of Spirula's renderer measured 2-5 dB off on 09-29, so it is not used for scores)."""
    import re
    from nerfstudio.data.utils import colmap_parsing_utils as cpu
    sparse, img_root = Path(args.colmap), Path(args.images)
    mask_root = Path(args.masks) if args.masks else None
    out = Path(args.out)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    cams = cpu.read_cameras_binary(sparse / "cameras.bin")
    ims = cpu.read_images_binary(sparse / "images.bin")
    pts = cpu.read_points3D_binary(sparse / "points3D.bin")
    al = json.loads(Path(args.align).read_text())
    s_, r_, t_ = float(al["s"]), np.array(al["R"]), np.array(al["t"])
    receipt = load_split(args.mirror)
    src = Path(receipt["source"])
    meta = json.loads((src / "transforms.json").read_text())
    by_path = {f["file_path"]: f for f in meta["frames"]}

    def link(dst: Path, target: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(target, dst)

    cam_lines, img_lines = [], []
    k = args.camera_scale   # same poses, images k× the SfM's resolution: fx fy cx cy scale, fisheye k1..k4 do not
    for cid, c in sorted(cams.items()):
        if k != 1 and c.model not in ("OPENCV_FISHEYE", "OPENCV", "PINHOLE"):
            raise SystemExit(f"--camera-scale: camera model {c.model} not handled")
        params = [float(x) * (k if i < 4 else 1) for i, x in enumerate(c.params)]
        cam_lines.append(f"{cid} {c.model} {round(c.width * k)} {round(c.height * k)} " + " ".join(repr(x) for x in params))
    kept = 0
    for iid, im in sorted(ims.items()):
        if "_train" not in Path(im.name).stem:
            continue                                   # their held-out / guard frames never train
        link(out / "images" / im.name, img_root / im.name)
        if mask_root is not None:
            m = mask_root / Path(im.name).with_suffix(".png")
            if m.is_file():
                link(out / "masks" / Path(im.name).with_suffix(".png"), m)
        img_lines += [f"{iid} " + " ".join(repr(float(x)) for x in im.qvec) + " "
                      + " ".join(repr(float(x)) for x in im.tvec) + f" {im.camera_id} {im.name}", ""]
        kept += 1
    next_cam, next_img = max(cams) + 1, max(ims) + 1
    pin_cams: dict[tuple, int] = {}
    for name in receipt["eval_names"]:
        f = by_path[name]
        g = lambda k: f.get(k, meta.get(k))  # noqa: E731
        key = (int(g("w")), int(g("h")), float(g("fl_x")), float(g("fl_y")), float(g("cx")), float(g("cy")))
        if key not in pin_cams:
            pin_cams[key] = next_cam
            cam_lines.append(f"{next_cam} PINHOLE {key[0]} {key[1]} {key[2]!r} {key[3]!r} {key[4]!r} {key[5]!r}")
            next_cam += 1
        view = cross.c2w_gl_to_viewmat(np.array(f["transform_matrix"]), s_, r_, t_)   # OpenCV w2c = COLMAP
        q = ply_frame.quat_from_matrix(view[:3, :3])
        stem = re.sub(r"[^A-Za-z0-9_]", "_", Path(name).stem)
        new = f"splatlab/{stem}_eval{Path(name).suffix}"
        link(out / "images" / new, src / name)
        img_lines += [f"{next_img} " + " ".join(repr(float(x)) for x in q) + " "
                      + " ".join(repr(float(x)) for x in view[:3, 3]) + f" {pin_cams[key]} {new}", ""]
        next_img += 1
    (out / "sparse/0/cameras.txt").write_text("\n".join(cam_lines) + "\n")
    (out / "sparse/0/images.txt").write_text("\n".join(img_lines) + "\n")
    (out / "sparse/0/points3D.txt").write_text("".join(
        f"{pid} {p.xyz[0]!r} {p.xyz[1]!r} {p.xyz[2]!r} {int(p.rgb[0])} {int(p.rgb[1])} {int(p.rgb[2])} {float(p.error)!r}\n"
        for pid, p in pts.items()))
    print(json.dumps({"train_images": kept, "splatlab_eval_views": len(receipt["eval_names"]),
                      "pinhole_cameras": len(pin_cams), "points": len(pts), "out": str(out)}, indent=1))
    return 0


def gaussian_count(arm: Path) -> int | None:
    plys = sorted(arm.glob("step-*.ckpt/splat.ply"))
    if plys:
        head = plys[-1].read_bytes()[:4096].decode("ascii", "replace")
        for line in head.splitlines():
            if line.startswith("element vertex"):
                return int(line.split()[2])
    ckpts = sorted(arm.glob("*/splatfacto/*/nerfstudio_models/step-*.ckpt"))
    if ckpts:
        import torch
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)["pipeline"]
        return int(state["_model.gauss_params.means"].shape[0])
    return None


def cmd_summary(args) -> int:
    scene = Path(args.scene_dir)
    rows = []
    for arm in sorted(p for p in scene.iterdir() if (p / "score.json").is_file()):
        sc = json.loads((arm / "score.json").read_text())
        tm = json.loads((arm / "timing.json").read_text()) if (arm / "timing.json").is_file() else {}
        m = sc["mean"]
        rows.append({"arm": arm.name, "views": sc["views"], **{k: round(v, 4) for k, v in m.items()},
                     "train_s": tm.get("train_s"), "peak_vram_mb": tm.get("peak_vram_mb"),
                     "gaussians": gaussian_count(arm), "renders": sc["renders"]})
    (scene / "summary.json").write_text(json.dumps(rows, indent=2))
    per = {r["arm"]: {v["photo"]: v for v in json.loads((scene / r["arm"] / "score.json").read_text())["per_view"]}
           for r in rows}
    common = set.intersection(*(set(v) for v in per.values())) if per else set()
    if per and any(len(v) != len(common) for v in per.values()):
        print(f"\nOn the {len(common)} photos EVERY arm rendered (some arms have missing views):")
        print("| arm | psnr | ssim | lpips | cc_psnr | cc_lpips |\n|---|---|---|---|---|---|")
        for arm, v in per.items():
            m = {k: np.mean([v[p][k] for p in common]) for k in ("psnr", "ssim", "lpips", "cc_psnr", "cc_lpips")}
            print(f"| {arm} | {m['psnr']:.4f} | {m['ssim']:.4f} | {m['lpips']:.4f} | {m['cc_psnr']:.4f} | {m['cc_lpips']:.4f} |")
    cols = ["arm", "views", "psnr", "ssim", "lpips", "cc_psnr", "cc_ssim", "cc_lpips", "train_s", "peak_vram_mb",
            "gaussians"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("job")
    p.add_argument("out")
    p.add_argument("--rig-timestamps", type=int, default=0,
                   help="rig job: hold out whole timestamps instead of nerfstudio's split (value = timestamps per camera)")
    p.add_argument("--holdout-every", type=int, default=8)
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("score")
    p.add_argument("renders", help="directory holding the arm's held-out renders")
    p.add_argument("--layout", required=True, choices=sorted(scoring.ARM_LAYOUTS))
    p.add_argument("--mirror", required=True, help="split mirror (split-receipt.json names the photos)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-missing", action="store_true",
                   help="long runs only: drop exact duplicate renders and allow unrendered photos (disclosed in "
                        "the score; summary compares arms on the common photo set)")
    p.set_defaults(fn=cmd_score)
    p = sub.add_parser("check-renders")
    p.add_argument("renders")
    p.add_argument("--layout", required=True, choices=sorted(scoring.ARM_LAYOUTS))
    p.add_argument("--mirror", required=True)
    p.set_defaults(fn=cmd_check_renders)
    p = sub.add_parser("to-ns-frame")
    p.add_argument("ply")
    p.add_argument("--dataparser-transforms", required=True,
                   help="the job's dataparser_transforms.json, or `identity` to only strip to SH0")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", help="the job's own ns-export splat.ply, for the frame check")
    p.add_argument("--max-nn", type=float, default=0.01)
    p.set_defaults(fn=cmd_to_ns_frame)
    p = sub.add_parser("align")
    p.add_argument("--colmap", required=True, help="the other SfM's sparse model dir")
    p.add_argument("--transforms", required=True, help="SplatLab rig job transforms.json")
    p.add_argument("--rig-timestamps", type=int, default=410)
    p.add_argument("--stride", type=int, default=9, help="source frames per instant in the other model's names")
    p.add_argument("--max-rel-rmse", type=float, default=0.02)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_align)
    p = sub.add_parser("cross-render")
    p.add_argument("--ply", required=True)
    p.add_argument("--mirror", required=True)
    p.add_argument("--align", help="align.json (omit = same frame)")
    p.add_argument("--scene-transform", help="the trainer's scene_transform.json, if not identity")
    p.add_argument("--colmap-frame", action="store_true",
                   help="the PLY was trained by Spirula on this same transforms.json (undo applied_transform)")
    p.add_argument("--mode", default="classic", choices=["classic", "antialiased"])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_cross_render)
    p = sub.add_parser("xdataset")
    p.add_argument("--colmap", required=True, help="the other SfM's sparse model (binary)")
    p.add_argument("--images", required=True, help="image root its names are relative to")
    p.add_argument("--masks", help="mask root (<name>.png), optional")
    p.add_argument("--align", required=True, help="align.json: SplatLab transforms frame → that model's frame")
    p.add_argument("--mirror", required=True, help="SplatLab split mirror whose eval views to add")
    p.add_argument("--camera-scale", type=float, default=1.0,
                   help="images are this many times the SfM's resolution (same instants, same names)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_xdataset)
    p = sub.add_parser("count")
    p.add_argument("arm_dir")
    p.set_defaults(fn=lambda a: print(gaussian_count(Path(a.arm_dir))) or 0)
    p = sub.add_parser("summary")
    p.add_argument("scene_dir")
    p.set_defaults(fn=cmd_summary)
    p = sub.add_parser("sheet")
    p.add_argument("out")
    p.add_argument("--mirror", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="LABEL=score.json (first = baseline)")
    p.add_argument("--tile-width", type=int, default=560)
    p.set_defaults(fn=cmd_sheet)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
