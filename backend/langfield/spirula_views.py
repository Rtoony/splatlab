"""PASS 0 for a Raw 360 (Spirula) scene: cut PINHOLE views out of the dual-fisheye frames, so SAM (PASS A) and the
lift (PASS B) run unchanged on ordinary perspective images.

Why: the lift was written against nerfstudio pinhole cameras loaded from a training checkpoint; Spirula scenes have
neither a checkpoint nor pinhole images. Everything needed is on disk though:
  - poses       _spirula/data/sparse/0/images.bin   (COLMAP world->camera, OpenCV; the published splat's frame)
  - lenses      _spirula/data/sparse/0/cameras.bin  (OPENCV_FISHEYE fx fy cx cy k1..k4 per lens)
  - images      _spirula/data/images/cam{0,1}/*.jpg
  - keep-masks  _spirula/data/masks/cam{0,1}/*.png  (lens border AND the SAM 3 operator mask: white = keep)
Each chosen fisheye frame becomes VIEWS_PER_LENS virtual pinhole cameras (FOV_DEG wide) aimed along the lens axis
and around it. The operator/lens-border mask is carried into each view so masked pixels never feed a feature.

Writes <out>/frames/frame_NNN.png, <out>/valid/view_NNN.png (255 = usable pixel), <out>/frames_meta.json and
<out>/cameras.json: per view c2w (nerfstudio/OpenGL convention, 3x4 — what splatfacto.get_viewmat expects) + fx fy
cx cy + W H. Same frame as the published splat (_preview/web.ply).

  python spirula_views.py <job_dir> <out_dir> [--instants 36] [--size 640]      (langfield-spike env: cv2, numpy)
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spirula_lane  # noqa: E402  (stdlib-only: images.bin reader)

FOV_DEG = 90.0
# Upright virtual views in WORLD terms: 6 headings x 2 pitches (25 deg below / above the horizon), each rolled so
# world-up is image-up (SAM and SigLIP read upright images far better). A view is cut from a lens only if it points
# within MAX_OFF_AXIS_DEG of that lens's axis. Measured 09-30: the X5 was held with its lenses facing the FLOOR (cam0)
# and CEILING (cam1), so "around the lens axis" produced sideways views and ceilings; this works for any grip.
HEADINGS_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
PITCHES_DEG = (-25.0, 25.0)
MAX_OFF_AXIS_DEG = 75.0
WORLD_UP = np.array([0.0, 0.0, 1.0])   # the published splat is gravity-aligned Z-up (Spirula --telemetry)
CAMERA_MODELS = {5: ("OPENCV_FISHEYE", 8)}


def read_cameras_bin(path: Path) -> dict[int, dict]:
    out = {}
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            cid, model, w, h = struct.unpack("<iiQQ", fh.read(24))
            if model not in CAMERA_MODELS:
                raise ValueError(f"camera {cid}: unsupported COLMAP model id {model}")
            name, npar = CAMERA_MODELS[model]
            params = struct.unpack(f"<{npar}d", fh.read(8 * npar))
            out[cid] = {"model": name, "width": int(w), "height": int(h), "params": list(params)}
    return out


def levelled_c2w(heading_deg: float, pitch_deg: float) -> np.ndarray:
    """World rotation (columns = OpenCV right, down, forward) of an upright virtual camera."""
    h, p = math.radians(heading_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), math.sin(p)])
    right = np.cross(fwd, WORLD_UP); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--instants", type=int, default=30)
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args()
    job, out = Path(args.job_dir), Path(args.out_dir)
    data = job / "_spirula" / "data"
    cams = read_cameras_bin(data / "sparse" / "0" / "cameras.bin")
    images = sorted(spirula_lane.read_images_bin(data / "sparse" / "0" / "images.bin"), key=lambda im: im["name"])
    by_lens: dict[str, list[dict]] = {}
    for im in images:
        by_lens.setdefault(im["name"].split("/")[0], []).append(im)
    lenses = sorted(by_lens)
    n0 = len(by_lens[lenses[0]])
    picks = sorted({int(round(i)) for i in np.linspace(0, n0 - 1, min(args.instants, n0))})

    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "valid").mkdir(parents=True, exist_ok=True)
    S = args.size
    f = (S / 2) / math.tan(math.radians(FOV_DEG) / 2)
    Knew = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]])
    views, v = [], 0
    for lens in lenses:
        seq = by_lens[lens]
        for idx in picks:
            if idx >= len(seq):
                continue
            im = seq[idx]
            cam = cams[im["camera_id"]]
            fx, fy, cx, cy, k1, k2, k3, k4 = cam["params"]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            D = np.array([k1, k2, k3, k4])
            img = cv2.imread(str(data / "images" / im["name"]), cv2.IMREAD_COLOR)
            mpath = data / "masks" / (Path(im["name"]).with_suffix(".png"))
            keep = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) if mpath.is_file() else None
            if img is None:
                continue
            if img.shape[1] != cam["width"]:          # extraction scale < 1: intrinsics were solved at this size
                s = img.shape[1] / cam["width"]
                K = K * np.array([[s], [s], [1]])
            # original camera -> world (OpenCV) from COLMAP world->camera
            R_wc = np.array(spirula_lane._rotmat(im["qvec"]))
            t_wc = np.array(im["tvec"])
            R_cw, C = R_wc.T, -R_wc.T @ t_wc
            lens_fwd = R_cw @ np.array([0.0, 0.0, 1.0])
            for yaw, pitch in [(h, p) for p in PITCHES_DEG for h in HEADINGS_DEG]:
                Rvirt = levelled_c2w(yaw, pitch)
                if math.degrees(math.acos(float(np.clip(Rvirt[:, 2] @ lens_fwd, -1, 1)))) > MAX_OFF_AXIS_DEG:
                    continue                           # this direction belongs to the other lens
                Rv = R_wc @ Rvirt                      # virtual -> original lens coords
                # cv2's R maps ORIGINAL rays into the rectified (virtual) frame: the inverse of Rv
                m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, Rv.T, Knew, (S, S), cv2.CV_32FC1)
                view = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                valid = cv2.remap(keep if keep is not None else np.full(img.shape[:2], 255, np.uint8),
                                  m1, m2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                if (valid >= 128).mean() < 0.25:        # mostly outside the lens / the operator: skip
                    continue
                cv2.imwrite(str(out / "frames" / f"frame_{v:03d}.png"), view)
                cv2.imwrite(str(out / "valid" / f"view_{v:03d}.png"), np.where(valid >= 128, 255, 0).astype(np.uint8))
                c2w_cv = np.eye(4)
                c2w_cv[:3, :3] = R_cw @ Rv
                c2w_cv[:3, 3] = C
                c2w_gl = c2w_cv @ np.diag([1.0, -1.0, -1.0, 1.0])     # OpenCV -> OpenGL (nerfstudio) axes
                views.append({"c2w": c2w_gl[:3].tolist(), "fx": f, "fy": f, "cx": S / 2, "cy": S / 2,
                              "width": S, "height": S, "source": im["name"], "yaw": yaw, "pitch": pitch})
                v += 1
    (out / "cameras.json").write_text(json.dumps({"frame": "published splat (Z-up, metric)", "views": views}))
    json.dump({"n": v, "W": S, "H": S, "resized_to_canonical": 0, "source": "spirula fisheye -> pinhole",
               "instants": len(picks), "lenses": lenses}, open(out / "frames_meta.json", "w"))
    print(f"EXPORTED {v} upright pinhole views ({len(picks)} instants x {len(lenses)} lenses) @ {S}x{S}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
