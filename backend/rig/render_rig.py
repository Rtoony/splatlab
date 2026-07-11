"""Render equirect frames into a RIG of perspective virtual cameras + COLMAP
rig_config.json — the geometry fix for the 360 lane.

Root cause (probe 2026-07-11, probe-operator-mask/STATUS.md): the legacy 8-crop
fan-out solves each crop as a FREE camera; same-frame crop centers scatter a
median 5.1 units where truth is 0 (true step 0.13) → trajectory 12× the scene →
nerfstudio normalization collapses real geometry into the fog-cocoon fingerprint.
A rig config (zero translation, fixed relative rotations, rig-verified matching)
removes those degrees of freedom.

Vendored from colmap4-src python/examples/panorama_sfm.py (same rotation and
spherical-projection conventions), dependency-light: numpy/cv2/scipy/PIL only —
no pycolmap needed (rig application happens via the colmap binary's
rig_configurator reading the JSON this script writes).

Outputs into <out_dir>:
  images/pano_camera<k>/<frame>.jpg      rendered virtual views
  masks/pano_camera<k>/<frame>.jpg.png   per-pixel ownership masks (each pano
                                         pixel features in exactly ONE view)
  rig_config.json                        for `colmap rig_configurator`

Run in the colmap4 env:
  ~/miniconda3/envs/colmap4/bin/python render_rig.py <equirect_dir> <out_dir>
      [--yaw-steps 4] [--pitches=-35,0,35] [--hfov 90] [--vfov 90] [--workers N]
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def virtual_rotations(num_steps_yaw: int, pitches_deg: list[float]) -> list[np.ndarray]:
    """cam_from_pano rotations — same convention as panorama_sfm.get_virtual_rotations."""
    mats = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            mats.append(Rotation.from_euler("XY", [-pitch_deg, -yaw_deg], degrees=True).as_matrix())
    return mats


def spherical_img_from_cam(pano_w: int, pano_h: int, rays: np.ndarray) -> np.ndarray:
    """Project unit rays into equirect pixel coords (panorama_sfm convention)."""
    r = rays.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u * pano_w, v * pano_h], -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("equirect_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--yaw-steps", type=int, default=4)
    ap.add_argument("--pitches", default="-35,0,35")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--vfov", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=max(2, min(16, (os.cpu_count() or 4) // 2)))
    args = ap.parse_args()
    pitches = [float(p) for p in args.pitches.split(",")]

    panos = sorted(p for p in args.equirect_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert panos, f"no frames in {args.equirect_dir}"
    pano_w, pano_h = Image.open(panos[0]).size
    assert pano_w == pano_h * 2, f"not 2:1 equirect: {pano_w}x{pano_h}"

    # Virtual pinhole geometry (same formulas as panorama_sfm.create_virtual_camera).
    img_w = int(pano_w * args.hfov / 360)
    img_h = int(pano_h * args.vfov / 180)
    focal = img_w / (2 * np.tan(np.deg2rad(args.hfov) / 2))
    rotations = virtual_rotations(args.yaw_steps, pitches)
    print(f"[rig] {len(panos)} panos {pano_w}x{pano_h} -> {len(rotations)} views "
          f"{img_w}x{img_h} f={focal:.1f}", flush=True)

    # Precompute per-view remap grids + ownership masks (shared across panos).
    px, py = np.meshgrid(np.arange(img_w, dtype=np.float32) + 0.5,
                         np.arange(img_h, dtype=np.float32) + 0.5)
    rays_cam = np.stack([(px - img_w / 2) / focal, (py - img_h / 2) / focal,
                         np.ones_like(px)], -1).reshape(-1, 3)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    centers_in_pano = np.einsum("nij,i->nj", np.stack(rotations), [0.0, 0.0, 1.0])

    grids, masks = [], []
    for k, rot in enumerate(rotations):
        rays_pano = rays_cam @ rot
        xy = spherical_img_from_cam(pano_w, pano_h, rays_pano) - 0.5   # COLMAP->OpenCV origin
        grids.append(xy.reshape(img_h, img_w, 2).astype(np.float32))
        owner = np.argmax(rays_pano @ centers_in_pano.T, -1)
        masks.append(((owner == k) * 255).astype(np.uint8).reshape(img_h, img_w))

    for k in range(len(rotations)):
        (args.out_dir / "images" / f"pano_camera{k}").mkdir(parents=True, exist_ok=True)
        (args.out_dir / "masks" / f"pano_camera{k}").mkdir(parents=True, exist_ok=True)
        Image.fromarray(masks[k]).save(args.out_dir / "masks" / f"pano_camera{k}" / "_ownership.png")

    def render(pano_path: Path) -> None:
        pano = np.asarray(Image.open(pano_path).convert("RGB"))
        assert pano.shape[1] == pano_w and pano.shape[0] == pano_h, f"size mismatch {pano_path.name}"
        for k, grid in enumerate(grids):
            view = cv2.remap(pano, grid[..., 0], grid[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)
            Image.fromarray(view).save(args.out_dir / "images" / f"pano_camera{k}" / pano_path.name,
                                       quality=95)
            # COLMAP wants one mask per image: symlink to the shared ownership mask.
            mpath = args.out_dir / "masks" / f"pano_camera{k}" / f"{pano_path.name}.png"
            if not mpath.exists():
                mpath.symlink_to("_ownership.png")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render, p) for p in panos]
        for i, f in enumerate(as_completed(futures), 1):
            f.result()
            if i % 20 == 0 or i == len(panos):
                print(f"[rig] rendered {i}/{len(panos)}", flush=True)

    # rig_config.json for `colmap rig_configurator` (doc/rigs.rst schema).
    # cam_from_rig = cam_from_pano @ pano_from_ref (ref = view 0), zero translation.
    cameras = []
    # camera_params MUST be a JSON array of doubles — colmap's rig-config parser
    # iterates the node's children (rig.cc ReadRigConfig); a comma string parses
    # to an EMPTY params vector and poisons every camera row in the database
    # (matcher aborts in ReadCameraRow). Cost one live flight to learn.
    params = [float(focal), img_w / 2.0, img_h / 2.0]
    for k, rot in enumerate(rotations):
        cam = {"image_prefix": f"pano_camera{k}/",
               "camera_model_name": "SIMPLE_PINHOLE",
               "camera_params": params}
        if k == 0:
            cam["ref_sensor"] = True
        else:
            rel = rot @ rotations[0].T
            q = Rotation.from_matrix(rel).as_quat()   # scipy: [qx,qy,qz,qw]
            cam["cam_from_rig_rotation"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
            cam["cam_from_rig_translation"] = [0.0, 0.0, 0.0]
        cameras.append(cam)
    (args.out_dir / "rig_config.json").write_text(json.dumps([{"cameras": cameras}], indent=2))
    print(f"[rig] DONE -> {args.out_dir} (rig_config.json, {len(rotations)} views/pano)", flush=True)


if __name__ == "__main__":
    main()
