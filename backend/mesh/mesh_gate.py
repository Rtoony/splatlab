#!/usr/bin/env python3
"""WS1 mesh-fidelity gate (adopted from the Blender-lab report, 2026-07-21):
render the mesh flat through evenly-spaced TRAIN cameras and score it against
the real photos. The honest number for "does this mesh look like the site?".

Cameras come from the DATAPARSER (the lab's verified fact — hand-composing
transforms.json chains is subtly wrong). Rendering is Open3D offscreen
(defaultUnlit vertex colors; ONE renderer per process — two segfault).
Convention: nerfstudio c2w is OpenGL; OpenCV extrinsic = inv(c2w @ diag(1,-1,-1,1)).

Lab reference baseline (Blender path, garden o3dtsdf-vanilla): PSNR 17.64 dB /
SSIM 0.234 / coverage 74.6%. This gate records its OWN baseline per metric
convention — compare within-gate over time, not across implementations.

--transform registers a candidate that was authored in some OTHER frame (a Y-up
/ metres GLB) into the capture frame before rendering. It is applied to the
loaded Open3D mesh rather than to a re-exported copy on purpose: re-exporting a
UV-textured GLB to move it risks losing the very albedo the gate is scoring.
The bbox the renderer ACTUALLY saw is reported back (`bbox_render_frame`) so a
caller can verify its own registration against the geometry that was drawn
instead of against what it hoped was drawn.

Usage: mesh_gate.py <mesh.ply> <config.yml> <out_dir> [--cams 6] [--transform M]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_orig_load = torch.load


def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _patched_load


def _glb_has_image(path: Path) -> bool:
    """True only for a .glb whose JSON chunk declares at least one image.

    Guards the post-processing loader, which aborts the whole process on a GLB
    that has materials but no image. Anything that is not a readable binary
    glTF answers False, which is the safe (plain-load) path.
    """
    import struct

    if path.suffix.lower() != ".glb":
        return False
    try:
        with path.open("rb") as fh:
            magic, _ver, _len = struct.unpack("<III", fh.read(12))
            if magic != 0x46546C67:
                return False
            clen, _ctype = struct.unpack("<II", fh.read(8))
            doc = json.loads(fh.read(clen).decode("utf-8"))
        return bool(doc.get("images"))
    except Exception:  # noqa: BLE001 — unreadable means "do not risk it"
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--cams", type=int, default=6)
    ap.add_argument(
        "--transform", default=None,
        help="4x4 row-major matrix applied to the mesh before rendering, as a path to a "
             "JSON file or inline JSON (bare 4x4 list, or {\"matrix\": [[...]]}). Use it to "
             "register a candidate authored in another frame into the capture frame.",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # enable_post_processing loads UVs and embedded textures from .glb, which is
    # required to score a UV-textured candidate honestly — rendered without its
    # albedo the model loses exactly the colour detail the texture carries.
    #
    # But it is only safe to ask for when the file really has an image. On a GLB
    # carrying materials and NO image (our parametric builds, and twin.glb),
    # Open3D aborts the process outright: "Image has unrecognized
    # bytes_per_channel", a C++ terminate that cannot be caught from Python. So
    # the decision is made by reading the glTF JSON chunk first rather than by
    # trying and recovering.
    mesh = o3d.io.read_triangle_mesh(args.mesh, _glb_has_image(Path(args.mesh)))
    if len(mesh.triangles) == 0:
        print("FATAL: empty mesh", file=sys.stderr)
        return 1

    xform = None
    if args.transform:
        as_path = Path(args.transform)
        try:
            payload = json.loads(as_path.read_text() if as_path.is_file() else args.transform)
        except (OSError, ValueError) as exc:
            print(f"FATAL: --transform is neither a readable JSON file nor inline JSON: {exc}",
                  file=sys.stderr)
            return 1
        if isinstance(payload, dict):
            payload = payload.get("matrix")
        xform = np.asarray(payload, dtype=np.float64)
        if xform.shape != (4, 4) or not np.isfinite(xform).all():
            print(f"FATAL: --transform must be a finite 4x4 matrix, got shape {xform.shape}",
                  file=sys.stderr)
            return 1
        # Transform BEFORE normals: a non-uniform or mirrored linear part would
        # otherwise leave stale normals behind, and defaultUnlit hides that
        # until some later shader does not.
        mesh.transform(xform)
    mesh.compute_vertex_normals()

    albedo = None
    if mesh.textures:
        for tex in mesh.textures:
            # is_empty() FIRST and always. Open3D materialises an empty Image
            # placeholder for every material that has no image (twin.glb: 1,
            # a 3-material parametric build: 4), and np.asarray() on one of
            # those aborts the process with "Image has unrecognized
            # bytes_per_channel" — a C++ terminate, not a Python exception, so
            # there is nothing to catch. is_empty() is safe to call.
            if tex.is_empty():
                continue
            arr = np.asarray(tex)
            if arr.size and arr.ndim == 3:
                # .copy() is load-bearing: assigning a texture straight off a
                # loaded mesh raises "Unable to cast from non-held to held
                # instance" in this Open3D build.
                albedo = o3d.geometry.Image(arr.copy())
                break
    shading = ("texture" if albedo is not None
               else "vertex_color" if mesh.has_vertex_colors() else "untextured")

    cfg_path = Path(args.config).resolve()
    # nerfstudio's own !!python/object config graph, self-produced on this box.
    config = yaml.load(cfg_path.read_text(), Loader=yaml.Loader)
    dp_config = config.pipeline.datamanager.dataparser
    dp_config.data = cfg_path.parents[2]  # <job>/processed — configs carry stale paths
    outputs = dp_config.setup().get_dataparser_outputs(split="train")
    cams = outputs.cameras
    n = int(cams.camera_to_worlds.shape[0])
    sel = sorted({int(round(i)) for i in np.linspace(0, n - 1, args.cams)})

    # ONE renderer per process (two segfault), so mixed-size camera sets
    # (ETH3D DSLR: four calibrations) are normalized: every cam's intrinsics
    # scale to the first cam's frame; photos resize to match. Sub-0.1% aspect
    # distortion on ETH3D — fine for a fidelity gate.
    W, H = int(cams.width[sel[0]]), int(cams.height[sel[0]])
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    if albedo is not None:
        mat.albedo_img = albedo
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    per_cam = []
    for ci in sel:
        kx, ky = W / int(cams.width[ci]), H / int(cams.height[ci])
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(cams.fx[ci]) * kx, float(cams.fy[ci]) * ky,
            float(cams.cx[ci]) * kx, float(cams.cy[ci]) * ky,
        )
        c2w = np.eye(4)
        c2w[:3, :] = cams.camera_to_worlds[ci].numpy()
        extrinsic = np.linalg.inv(c2w @ flip)
        renderer.setup_camera(intr, extrinsic)
        render = np.asarray(renderer.render_to_image()).astype(np.float32) / 255.0
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        covered = np.isfinite(depth)

        photo = np.asarray(
            Image.open(outputs.image_filenames[ci]).convert("RGB").resize((W, H))
        ).astype(np.float32) / 255.0

        coverage = float(covered.mean())
        entry = {"cam": int(ci), "coverage": round(coverage, 3)}
        if coverage > 0.05:
            entry["psnr_covered"] = round(float(
                peak_signal_noise_ratio(photo[covered], render[covered], data_range=1.0)
            ), 2)
            entry["ssim_fullframe"] = round(float(
                structural_similarity(photo, render, channel_axis=2, data_range=1.0)
            ), 3)
        per_cam.append(entry)

        strip = np.concatenate([photo, render], axis=1)
        Image.fromarray((strip * 255).astype(np.uint8)).save(
            out_dir / f"gate_cam{ci:03d}.jpg", quality=88
        )

    scored = [e for e in per_cam if "psnr_covered" in e]
    # Measured on the geometry the renderer was handed, AFTER --transform. This
    # is the receipt that turns "I registered it" into "it rendered there".
    bb = mesh.get_axis_aligned_bounding_box()
    lo, hi = np.asarray(bb.min_bound), np.asarray(bb.max_bound)

    # Do the cameras sit INSIDE the thing being scored?
    #
    # A watertight shell built over an object-centric orbit capture encloses the
    # camera ring as well as the subject, because the cameras orbit outside the
    # subject and the solid must close around everything. Every render is then a
    # view of the mesh's own interior walls, which scores like noise and looks
    # like a jumble of slabs. Measured on splat_3aaf8067: 175/175 cameras inside,
    # PSNR 11.96 against a 17.64 reference, and no amount of texture resolution,
    # class radius, face budget or unobserved-face dropping moved it (0.23 dB
    # across 4x the geometry). It is not a tuning problem and it is invisible in
    # the scores alone, so state it.
    # lo/hi are measured AFTER --transform, i.e. in the frame the renderer saw,
    # which is the same capture frame the cameras live in.
    cam_pos = cams.camera_to_worlds[:, :3, 3].numpy()
    inside = np.all((cam_pos >= lo) & (cam_pos <= hi), axis=1)
    cameras_inside = float(inside.mean())

    report = {
        "v": 1,
        "cams": len(per_cam),
        "per_cam": per_cam,
        "median_coverage": round(float(np.median([e["coverage"] for e in per_cam])), 3),
        # >0 means some renders look at the mesh from the inside. Near 1.0 means
        # the mesh encloses the capture and cannot resemble it from any camera.
        "cameras_inside_mesh_bbox": round(cameras_inside, 3),
        "encloses_capture": bool(cameras_inside > 0.5),
        "median_psnr": round(float(np.median([e["psnr_covered"] for e in scored])), 2) if scored else None,
        "median_ssim": round(float(np.median([e["ssim_fullframe"] for e in scored])), 3) if scored else None,
        "convention": "o3d-unlit, psnr on covered px, ssim full-frame vs black-backed render",
        # Which colour source actually rendered. Comparing a "texture" score
        # against a "vertex_color" one is comparing different things, so the
        # bake-off records it per candidate rather than assuming.
        "shading": shading,
        "transform": xform.tolist() if xform is not None else None,
        "bbox_render_frame": {
            "centre": [round(float(x), 5) for x in (lo + hi) / 2.0],
            "extent": [round(float(x), 5) for x in hi - lo],
        },
        "lab_reference_blender": {"psnr": 17.64, "ssim": 0.234, "coverage": 0.746},
        "artifacts": [f"gate_cam{e['cam']:03d}.jpg" for e in per_cam],
    }
    (out_dir / "mesh_gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_cam", "transform")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
