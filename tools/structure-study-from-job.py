#!/usr/bin/env python3
"""Structure study from an ORDINARY SplatLab job (phone video, DSLR stills): the layout
capture_structure.prepare() writes for X5 fisheye packages, so `capture-structure.py masks`
and `analyze`, `moge-depth-views.py`, `architecture-scaffold.py` and the evidence exporter
run unchanged. CPU, langfield-spike or splatops env (OpenCV + nerfstudio's COLMAP readers).

  structure-study-from-job.py --job splat_xxx --output data/spatial/<name>-structure-<date>-01 [--views 32] [--max-width 1600] [--prompts a,b,c]

Reads <job>/processed/transforms.json (nerfstudio OpenGL poses, OPENCV intrinsics + k1 k2 p1 p2,
applied_transform) and <job>/processed/sparse/0/*.bin (COLMAP). Selects N registered frames evenly
along the sequence, undistorts + downsizes them to ONE pinhole intrinsics, brings the COLMAP points
into the pose frame (applied_transform), and records per view which points are observed in it —
error <= 2 px, track length >= 3, reprojection within 3 px of the COLMAP feature — exactly the
support rule of the fisheye study. Writes frames/, tracks.npz, views.json, things.json, receipt.json
(schema dev.splatlab.capture-structure/v1, status prepared-needs-masks, registration None).
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests                 # noqa: E402
import capture_structure as structure                 # noqa: E402
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary, read_images_binary, read_points3D_binary  # noqa: E402

OUTPUTS = Path.home() / "projects" / "splatcli" / "outputs" / "3d"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--views", type=int, default=32); ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--prompts", default=None); ap.add_argument("--min-points", type=int, default=40)
    a = ap.parse_args(); t0 = time.time()
    job = Path(a.job) if Path(a.job).is_dir() else OUTPUTS / a.job
    proc = job / "processed"; tr_path = proc / "transforms.json"; sparse = proc / "sparse" / "0"
    for p in (tr_path, sparse / "images.bin", sparse / "points3D.bin", sparse / "cameras.bin"):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    out = a.output.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    tr = json.loads(tr_path.read_text())
    if tr.get("camera_model") not in ("OPENCV", "PINHOLE", "SIMPLE_PINHOLE"):
        raise SystemExit(f"camera model {tr.get('camera_model')} is not a pinhole/OPENCV model; use the fisheye path")
    frames = sorted(tr["frames"], key=lambda f: f["file_path"])
    images = read_images_binary(sparse / "images.bin"); points3d = read_points3D_binary(sparse / "points3D.bin"); cams = read_cameras_binary(sparse / "cameras.bin")
    by_name = {im.name: im for im in images.values()}
    A = np.asarray(tr.get("applied_transform", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), dtype=np.float64).reshape(3, 4)
    # points into the pose frame; ids, colours, errors, track lengths, ambiguity (same image twice in a track)
    pids = np.array(sorted(points3d.keys()), dtype=np.int64); pid_index = {int(pid): i for i, pid in enumerate(pids)}
    xyz_colmap = np.stack([points3d[int(p)].xyz for p in pids]); points = xyz_colmap @ A[:3, :3].T + A[:3, 3]
    colors = np.stack([points3d[int(p)].rgb for p in pids]).astype(np.uint8); errors = np.array([points3d[int(p)].error for p in pids])
    track_len = np.array([len(points3d[int(p)].image_ids) for p in pids]); ambiguous = np.array([len(set(points3d[int(p)].image_ids.tolist())) != len(points3d[int(p)].image_ids) for p in pids])
    # one camera for the whole job (video / one body): intrinsics + distortion from transforms.json
    def intr(f):
        return {k: float(f.get(k, tr.get(k))) for k in ("fl_x", "fl_y", "cx", "cy")} | {k: int(f.get(k, tr.get(k))) for k in ("w", "h")} | {k: float(f.get(k, tr.get(k, 0.0)) or 0.0) for k in ("k1", "k2", "p1", "p2")}
    first = intr(frames[0])
    for f in frames:
        if intr(f) != first:
            raise SystemExit("per-frame intrinsics differ; this adapter assumes one camera")
    W, H = first["w"], first["h"]; K = np.array([[first["fl_x"], 0, first["cx"]], [0, first["fl_y"], first["cy"]], [0, 0, 1]]); dist = np.array([first["k1"], first["k2"], first["p1"], first["p2"]])
    undistort = bool(np.any(np.abs(dist) > 1e-9))
    if undistort:
        K2, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0, (W, H)); x0, y0, rw, rh = roi
    else:
        K2, (x0, y0, rw, rh) = K.copy(), (0, 0, W, H)
    scale = min(1.0, a.max_width / rw); w2, h2 = int(round(rw * scale)), int(round(rh * scale))
    K3 = K2.copy(); K3[0, 2] -= x0; K3[1, 2] -= y0; K3[:2] *= scale
    fx, fy, cx, cy = float(K3[0, 0]), float(K3[1, 1]), float(K3[0, 2]), float(K3[1, 2])
    # choose views evenly along the sequence among registered frames with enough support
    registered = [f for f in frames if Path(f["file_path"]).name in by_name and len([p for p in by_name[Path(f["file_path"]).name].point3D_ids if p >= 0]) >= a.min_points]
    if len(registered) < 4:
        raise SystemExit(f"only {len(registered)} registered frames with >= {a.min_points} points")
    n = min(a.views, len(registered)); picks = [registered[round(i * (len(registered) - 1) / max(1, n - 1))] for i in range(n)]
    out.mkdir(parents=True); (out / "frames").mkdir()
    prompts = tuple(p.strip() for p in a.prompts.split(",") if p.strip()) if a.prompts else structure.PROMPTS
    observed = np.zeros((n, len(pids)), dtype=bool); projected = np.zeros((n, len(pids), 2)); views = []
    for ordinal, f in enumerate(picks):
        name = Path(f["file_path"]).name; im = by_name[name]
        img = cv2.imread(str(proc / f["file_path"]))
        if img is None or img.shape[1] != W or img.shape[0] != H:
            raise SystemExit(f"{f['file_path']}: missing or not {W}x{H}")
        if undistort:
            img = cv2.undistort(img, K, dist, None, K2)
        img = img[y0:y0 + rh, x0:x0 + rw]
        if scale < 1.0:
            img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_AREA)
        photo = f"frames/cam_{ordinal:03d}.png"; cv2.imwrite(str(out / photo), img)
        c2w = np.asarray(f["transform_matrix"], dtype=np.float64)
        pix, inside = structure.project_points(points, {"transform_matrix": c2w.tolist()}, {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w2, "h": h2})
        projected[ordinal] = pix
        # COLMAP features of this image -> the transforms.json resolution (COLMAP may have run on full-size
        # images while the job's frames are downscaled: bicycle job 3118 px vs 1559 px), then undistorted, cropped, scaled
        cam = cams[im.camera_id]; feats = im.xys.astype(np.float64) * np.array([W / cam.width, H / cam.height]); ids = im.point3D_ids
        if undistort:
            feats = cv2.undistortPoints(feats.reshape(-1, 1, 2), K, dist, P=K2).reshape(-1, 2)
        feats = (feats - [x0, y0]) * scale
        residuals = []
        for feat, pid in zip(feats, ids):
            if pid < 0 or int(pid) not in pid_index:
                continue
            k = pid_index[int(pid)]
            if ambiguous[k] or errors[k] > 2 or track_len[k] < 3 or not inside[k]:
                continue
            r = float(np.linalg.norm(pix[k] - feat))
            if r <= 3:
                observed[ordinal, k] = True; residuals.append(r)
        views.append({"file_path": f["file_path"], "source_image": f["file_path"], "source_group": Path(f["file_path"]).stem, "split": "train",
                      "image_id": int(im.id), "colmap_im_id": f.get("colmap_im_id"), "ordinal": ordinal, "photo": photo, "transform_matrix": c2w.tolist(),
                      "tracked_supported_points": int(observed[ordinal].sum()), "maximum_raw_reprojection_px": max(residuals, default=None),
                      "pose_basis": "splatcli-job-transforms", "sha256": manifests.sha256_file(out / photo)})
        print(f"[study] cam {ordinal:03d} {name}: {int(observed[ordinal].sum())} supported points", flush=True)
    np.savez_compressed(out / "tracks.npz", points=points, colors=colors, point_ids=pids, observed=observed, pixels=projected)
    manifests.atomic_write_json(out / "views.json", {"cam_indices": list(range(n))})
    manifests.atomic_write_json(out / "things.json", list(prompts))
    source_hashes = {str(p): manifests.sha256_file(p) for p in (tr_path, sparse / "images.bin", sparse / "points3D.bin", sparse / "cameras.bin")}
    receipt = {"schema": structure.SCHEMA, "status": "prepared-needs-masks", "adapter": "structure-study-from-job/v1", "source_job": {"id": job.name, "dir": str(job)},
               "source_hashes": source_hashes, "started_at": manifests.utc_now(), "views": views, "prompts": list(prompts),
               "intrinsics": {"w": w2, "h": h2, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy},
               "camera": {"model": tr.get("camera_model"), "original": {"w": W, "h": H, **{k: first[k] for k in ("fl_x", "fl_y", "cx", "cy", "k1", "k2", "p1", "p2")}}, "undistorted": undistort, "roi": [int(x0), int(y0), int(rw), int(rh)], "scale": scale,
                          "colmap_feature_resolution": {int(k): [c.width, c.height] for k, c in cams.items()}},
               "applied_transform": A.tolist(), "sparse_points": int(len(pids)), "source_timestamps": n,
               "excluded_ambiguous_track_point_ids": [int(p) for p in pids[ambiguous]],
               "geometry_refined": False, "metric_scale": "unknown", "registration": None, "owner_accepted": False,
               "recipe": {"semantic_threshold": .5, "semantic_border_erosion_px": 3, "minimum_positive_timestamps": 2, "minimum_angle_degrees": 3,
                          "point_holdout_rule": "point ID modulo 5 equals zero reserved from plane fitting", "plane_tolerance_camera_span_fraction": .005},
               "scope": "Semantic predictions on undistorted training frames of an ordinary SplatLab job anchored to its COLMAP tracks; not semantic or structural ground truth",
               "elapsed_seconds": round(time.time() - t0, 1)}
    receipt["files"] = {str(p.relative_to(out)): manifests.sha256_file(p) for p in out.rglob("*") if p.is_file()}
    manifests.atomic_write_json(out / "receipt.json", receipt)
    print(f"[study] {n} views at {w2}x{h2} (undistorted={undistort}, scale {scale:.3f}), {len(pids)} points, "
          f"{int(observed.sum())} view-point supports, {int(ambiguous.sum())} ambiguous tracks excluded, in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
