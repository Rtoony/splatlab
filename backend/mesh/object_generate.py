#!/usr/bin/env python3
"""P7: Meta SAM 3D Objects as a GENERATIVE CANDIDATE SOURCE for one scene object.

WHAT THIS IS -- AND IS NOT
--------------------------
Every artifact this script writes is GENERATED, not captured. SAM 3D Objects
hallucinates the unseen 95% of an object from a single masked view; it is
plausible, not faithful. Output is render/VR lane ONLY and is tagged three
ways so the doctrine survives the files leaving the job tree:
  * path      -- default --out lives under <job>/_regen/... (provenance.path_is_generative)
  * PLY       -- GENERATIVE_TAG injected as a header `comment` (provenance.ply_is_generative)
  * glTF      -- GENERATIVE_TAG in asset.extras and EVERY node.extras under
                 provenance.GLTF_EXTRAS_KEY, which is exactly the key
                 blender_assemble.py's contamination gate reads back.
Both writes are read back off disk and asserted before this script claims
success -- a tag that isn't in the file is a tag that doesn't exist.

WHY THREE SUBPROCESSES
----------------------
Three incompatible conda envs, one per model, none of which can import the
others: sam3d-objects (torch 2.10 + the 13 GB SA-3DAO checkpoints), sam3
(SAM 3.1 text-prompted masks), dn-splatter-probe (nerfstudio dataparser =
the capture's cameras). Each worker is written to <out>/_work/ as a real,
re-runnable file rather than a temp -- the receipt IS the reproduction recipe.

THE MASK IS THE WHOLE BALLGAME
------------------------------
A misaligned mask produces a confident, wrong reconstruction, so the mask is
gated BEFORE the 13 GB model is loaded: the object's own captured gaussians
(inside bbox_tight_scene, solid only) are projected through the crop camera
that object_crop.py chose, rasterised with the same dilate/close/fill recipe,
and IoU'd against the SAM 3 mask. Measured 0.887 on splat_513e89171d's
hydrant. Below --min-mask-iou that object is REFUSED -- report + alignment.png
written, model never loaded for it, remaining objects continue (one bad object
must not cost the batch its 13 GB load), and the process exits non-zero.
--allow-unverified-mask overrides, and only with eyes on alignment.png: on
splat_aea04ab3's red-bicycle the gate scored 0.320 because SAM fills a spoked
wheel that the gaussians correctly leave hollow -- a real structural
disagreement, and exactly the input that would have yielded a solid-disc bike.

SCALE -- READ THE REPORT, NOT THIS DOCSTRING
--------------------------------------------
SAM 3D emits geometry in its own normalised frame (~unit cube at the origin)
with NO relation to capture units, plus a pose (rotation/translation/scale)
expressed in MoGe's scale-shift-invariant pointmap frame -- itself arbitrary.
So the capture-frame placement is not read out of the model; it is FITTED and
then SCORED:
  * uniform scale + translation come from the object's known bbox_tight_scene
    (an AABB fixes size and centre but CANNOT fix orientation);
  * orientation is searched (6 up-axis assignments x yaw) and each candidate
    is scored by silhouette IoU against the verified mask in the crop camera.
The report carries the winning IoU next to the IoU the CAPTURED object scores
through the identical rasteriser -- that ratio, not a bare number, is the
evidence. If the best candidate misses --min-placement-iou the report says
`placement_resolved: false` and the transform is withheld. A refusal is a
valid result; a guessed scale is not.

Usage:
  object_generate.py <job_dir> --object <slug>[,<slug>...] [--seed 42] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import (  # noqa: E402
    GENERATIVE_TAG, GLTF_EXTRAS_KEY, ply_header_comments)

# ---------------------------------------------------------------------------
# Fixed installation facts. All fail-loud in doctor(); none are guessed.
# ---------------------------------------------------------------------------
SAM3D_ROOT = Path("/home/rtoony/tools/sam-3d-objects")
SAM3D_PIPELINE_YAML = SAM3D_ROOT / "checkpoints" / "checkpoints" / "pipeline.yaml"
SAM3D_PYTHON = Path("/home/rtoony/miniconda3/envs/sam3d-objects/bin/python")
SAM3D_CKPTS = ("ss_generator.ckpt", "ss_encoder.ckpt", "ss_decoder.ckpt",
               "slat_generator.ckpt", "slat_decoder_gs.ckpt", "slat_decoder_mesh.ckpt")

SAM3_ROOT = Path("/home/rtoony/projects/ml/sam3")
SAM3_CKPT = SAM3_ROOT / "checkpoints" / "sam3.1_multiplex.pt"
SAM3_BPE = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
SAM3_PYTHON = Path("/home/rtoony/miniconda3/envs/sam3/bin/python")

PROBE_PYTHON = Path("/home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python")

MODEL_NAME = "meta/sam-3d-objects (SA-3DAO)"

# Silhouette rasterisation recipe. Calibrated on splat_513e89171d/fire-hydrant:
# captured-gaussian projection vs SAM 3 mask = IoU 0.887. Used IDENTICALLY for
# the captured reference and every generated candidate so the two are comparable.
RASTER_DILATE = 5
RASTER_CLOSE = 9
RASTER_MAX_POINTS = 120_000


class Fatal(RuntimeError):
    """Preflight or gate failure. Always actionable, never swallowed."""


def log(msg: str) -> None:
    print(f"[object-generate] {msg}", flush=True)


# ===========================================================================
# Embedded workers. Written to <out>/_work/ verbatim, then run in their own env.
# ===========================================================================

WORKER_SAM3D = r'''
"""SAM 3D Objects worker (env: sam3d-objects). Written by object_generate.py.
Usage: worker_sam3d.py <jobs.json>

Does NOT import notebook/inference.py: that module pulls seaborn + kaolin,
neither of which is in this env. It replicates the 5 lines of Inference.__init__
that matter and installs faithful, FAIL-LOUD shims for the three env holes:
  cv2            opencv 4.9 compiled against numpy 1.x, env has numpy 2.4.4
  kaolin         absent; only check_tensor() is used, by flexicubes asserts
  pkg_resources  setuptools >= 81 dropped it; lightning calls declare_namespace
Anything beyond the shimmed surface RAISES rather than silently degrading.
"""
import importlib.machinery
import json
import os
import sys
import time
import types

ROOT = os.environ["SAM3D_ROOT"]
os.environ.setdefault("CUDA_HOME", os.environ["CONDA_PREFIX"])
os.environ["LIDRA_SKIP_INIT"] = "true"
sys.path.insert(0, ROOT)

import numpy as np

SHIMS = {}


def _spec(mod, name):
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return mod


def install_cv2_shim():
    try:
        import cv2  # noqa: F401
        return "real"
    except Exception as exc:
        broken = repr(exc)
    from scipy import ndimage

    def _erode(src, kernel, **kw):
        return ndimage.grey_erosion(src, footprint=np.asarray(kernel) > 0)

    def _dilate(src, kernel, **kw):
        return ndimage.grey_dilation(src, footprint=np.asarray(kernel) > 0)

    def _dead(name):
        def f(*a, **k):
            raise RuntimeError(f"cv2.{name} is needed but opencv is ABI-broken: {broken}")
        return f

    m = types.ModuleType("cv2")
    m.erode, m.dilate = _erode, _dilate
    for n in ("inpaint", "putText", "getTextSize", "resize", "cvtColor"):
        setattr(m, n, _dead(n))
    for n, v in (("INPAINT_TELEA", 1), ("FONT_HERSHEY_SIMPLEX", 0), ("LINE_AA", 16)):
        setattr(m, n, v)
    sys.modules["cv2"] = _spec(m, "cv2")
    return f"shim: {broken}"


def install_kaolin_shim():
    try:
        import kaolin.utils.testing  # noqa: F401
        return "real"
    except Exception as exc:
        broken = repr(exc)
    import torch as _t

    def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
        def fail(msg):
            if throw:
                raise ValueError(msg)
            return False
        if not _t.is_tensor(tensor):
            return fail(f"not a tensor: {type(tensor)}")
        if shape is not None:
            if tensor.ndim != len(shape):
                return fail(f"ndim {tensor.ndim} != {len(shape)}")
            for i, s in enumerate(shape):
                if s is not None and tensor.shape[i] != s:
                    return fail(f"dim {i} = {tensor.shape[i]} != {s}")
        if dtype is not None and tensor.dtype != dtype:
            return fail(f"dtype {tensor.dtype} != {dtype}")
        if device is not None and _t.device(device).type != tensor.device.type:
            return fail(f"device {tensor.device} != {device}")
        return True

    kaolin = types.ModuleType("kaolin")
    utils = types.ModuleType("kaolin.utils")
    testing = types.ModuleType("kaolin.utils.testing")
    testing.check_tensor = check_tensor
    utils.testing, kaolin.utils = testing, utils
    for n, mod in (("kaolin", kaolin), ("kaolin.utils", utils),
                   ("kaolin.utils.testing", testing)):
        sys.modules[n] = _spec(mod, n)
    return f"shim: {broken}"


def install_pkg_resources_shim():
    try:
        import pkg_resources  # noqa: F401
        return "real"
    except Exception as exc:
        broken = repr(exc)
    m = types.ModuleType("pkg_resources")
    m.declare_namespace = lambda name: None

    def _missing(name):
        raise AttributeError(f"pkg_resources.{name} unavailable (setuptools>=81): {broken}")
    m.__getattr__ = _missing
    sys.modules["pkg_resources"] = _spec(m, "pkg_resources")
    return f"shim: {broken}"


SHIMS["cv2"] = install_cv2_shim()
SHIMS["kaolin"] = install_kaolin_shim()
SHIMS["pkg_resources"] = install_pkg_resources_shim()

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

import sam3d_objects  # noqa: E402,F401  (vendor: "do not remove this import")


def render_turntable(verts, colors, n_views=4, res=384, axis=1):
    """Dependency-free receipt render: orbit `axis`, painter's-algorithm point
    splat with real vertex colours. Proof the geometry looks like the object --
    NOT a quality-graded render."""
    v = np.asarray(verts, dtype=np.float64)
    c = np.asarray(colors, dtype=np.uint8)[:, :3]
    if len(v) > 200000:
        idx = np.random.default_rng(0).choice(len(v), 200000, replace=False)
        v, c = v[idx], c[idx]
    v = v - v.mean(0)
    span = float(np.abs(v).max()) * 2.2 + 1e-9
    other = [i for i in range(3) if i != axis]
    tiles = []
    for k in range(n_views):
        th = 2 * np.pi * k / n_views
        ct, st = np.cos(th), np.sin(th)
        a, b = v[:, other[0]], v[:, other[1]]
        x = ct * a + st * b
        depth = -st * a + ct * b
        y = -v[:, axis]
        px = ((x / span + 0.5) * (res - 1)).astype(np.int32)
        py = ((y / span + 0.5) * (res - 1)).astype(np.int32)
        ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
        px, py, dz, cc = px[ok], py[ok], depth[ok], c[ok]
        order = np.argsort(-dz)                  # far first, near overwrites
        img = np.full((res, res, 3), 24, np.uint8)
        img[py[order], px[order]] = cc[order]
        tiles.append(img)
    return np.concatenate(tiles, axis=1)


def main():
    jobs = json.loads(open(sys.argv[1]).read())
    cfg = OmegaConf.load(jobs["pipeline_yaml"])
    cfg.rendering_engine = "pytorch3d"          # vendor default: disables nvdiffrast
    cfg.compile_model = False
    cfg.workspace_dir = os.path.dirname(jobs["pipeline_yaml"])

    t0 = time.time()
    pipe = instantiate(cfg)
    load_s = time.time() - t0
    print(f"[worker] pipeline loaded in {load_s:.1f}s", flush=True)

    results = []
    for job in jobs["objects"]:
        rec = {"slug": job["slug"], "seed": job["seed"], "shims": SHIMS,
               "pipeline_load_seconds": round(load_s, 1)}
        try:
            rgba = np.load(job["rgba_npy"])
            assert rgba.ndim == 3 and rgba.shape[2] == 4 and rgba.dtype == np.uint8
            rec["input_rgba_shape"] = list(rgba.shape)
            rec["input_mask_coverage"] = round(float((rgba[..., 3] > 127).mean()), 5)

            t1 = time.time()
            torch.manual_seed(job["seed"])
            out = pipe.run(
                rgba, None, job["seed"],
                stage1_only=False,
                with_mesh_postprocess=False,     # needs cv2.inpaint / texture bake
                with_texture_baking=False,
                with_layout_postprocess=False,
                use_vertex_color=True,
                stage1_inference_steps=None,
                pointmap=None,                   # MoGe estimates it from the crop
            )
            rec["inference_seconds"] = round(time.time() - t1, 1)

            for key in ("rotation", "translation", "scale", "translation_scale"):
                if key in out and torch.is_tensor(out[key]):
                    rec[f"pose_{key}"] = [round(float(x), 8)
                                          for x in out[key].flatten().tolist()]
            if torch.is_tensor(out.get("coords")):
                rec["sparse_voxels"] = int(out["coords"].shape[0])

            glb = out.get("glb")
            if glb is not None:
                v = np.asarray(glb.vertices, dtype=np.float64)
                f = np.asarray(glb.faces)
                rec["mesh_vertices"], rec["mesh_faces"] = int(len(v)), int(len(f))
                rec["mesh_bounds_min"] = [round(float(x), 6) for x in v.min(0)]
                rec["mesh_bounds_max"] = [round(float(x), 6) for x in v.max(0)]
                colors = None
                try:
                    colors = np.asarray(glb.visual.vertex_colors)
                    rec["mesh_has_vertex_colors"] = bool(colors is not None and len(colors) == len(v))
                except Exception as exc:
                    rec["mesh_vertex_color_error"] = repr(exc)
                glb.export(job["mesh_out"])
                rec["mesh_out"] = job["mesh_out"]
                # Vertices + colours for the parent's placement search and gates.
                np.savez_compressed(
                    job["geom_npz"], vertices=v.astype(np.float32),
                    colors=(colors[:, :3].astype(np.uint8) if colors is not None
                            else np.full((len(v), 3), 200, np.uint8)))
                if colors is not None:
                    try:
                        Image.fromarray(render_turntable(v, colors)).save(job["preview_out"])
                        rec["preview_out"] = job["preview_out"]
                    except Exception as exc:
                        rec["preview_error"] = repr(exc)
            else:
                rec["mesh_error"] = "pipeline returned no glb (mesh decode unavailable)"

            gs = out.get("gs")
            if gs is not None:
                xyz = gs.get_xyz.detach().cpu().numpy()
                rec["splat_gaussians"] = int(len(xyz))
                rec["splat_bounds_min"] = [round(float(x), 6) for x in xyz.min(0)]
                rec["splat_bounds_max"] = [round(float(x), 6) for x in xyz.max(0)]
                gs.save_ply(job["splat_out"])
                rec["splat_out"] = job["splat_out"]
            else:
                rec["splat_error"] = "pipeline returned no gaussian"
            rec["ok"] = True
        except Exception as exc:
            import traceback
            rec["ok"] = False
            rec["error"] = repr(exc)
            rec["traceback"] = traceback.format_exc()
            print(rec["traceback"], flush=True)
        results.append(rec)
        with open(jobs["results_out"], "w") as fh:
            json.dump(results, fh, indent=2)

    print("WORKER_SAM3D_DONE", flush=True)


if __name__ == "__main__":
    main()
'''

WORKER_SAM3 = r'''
"""SAM 3.1 text-prompted crop masking worker (env: sam3). Written by
object_generate.py. Mirrors backend/mesh/scene_sam3_masks.py's proven API use;
one checkpoint load covers every requested crop.
Usage: worker_sam3.py <jobs.json>
"""
import json
import sys

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
from sam3.eval.postprocessors import PostProcessImage


def as_numpy(t):
    if torch.is_tensor(t):
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.detach().cpu().numpy()
    return np.asarray(t)


def main():
    jobs = json.loads(open(sys.argv[1]).read())
    model = build_sam3_image_model(checkpoint_path=jobs["checkpoint"], bpe_path=jobs["bpe"])
    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    post = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
        use_original_sizes_mask=True, convert_mask_to_rle=False,
        detection_threshold=jobs["threshold"], to_cpu=True)

    qid = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for job in jobs["crops"]:
            pil = Image.open(job["image"]).convert("RGB")
            w, h = pil.size
            qid += 1
            dp = Datapoint(find_queries=[], images=[])
            dp.images = [SAMImage(data=pil, objects=[], size=[h, w])]
            dp.find_queries.append(FindQueryLoaded(
                query_text=job["prompt"], image_id=0, object_ids_output=[],
                is_exhaustive=True, query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=qid, original_image_id=qid, original_category_id=1,
                    original_size=[h, w], object_id=qid, frame_index=0)))
            dp = transform(dp)
            batch = collate([dp], dict_key="d")["d"]
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            res = post.process_results(model(batch), batch.find_metadatas)
            (_, r), = res.items()
            m = as_numpy(r["masks"]) if "masks" in r else np.zeros((0, h, w), bool)
            if m.ndim == 4:
                m = m[:, 0]
            m = m.astype(bool)
            scores = (as_numpy(r["scores"]).astype(np.float32) if "scores" in r
                      else np.ones(m.shape[0], np.float32))
            assert m.shape[1:] == (h, w), f"mask res {m.shape[1:]} != image {(h, w)}"
            np.savez_compressed(job["out"], masks=m, scores=scores,
                                prompt=job["prompt"], size=np.array([h, w], np.int32))
            print(f"[worker] {job['prompt']!r}: {m.shape[0]} instance(s) "
                  f"scores={[round(float(s), 3) for s in scores]}", flush=True)
    print("WORKER_SAM3_DONE", flush=True)


if __name__ == "__main__":
    main()
'''

WORKER_CAM = r'''
"""Capture-camera resolver (env: dn-splatter-probe). Written by
object_generate.py. Loads the SAME nerfstudio dataparser split object_crop.py
used, so `cam` in crop.json indexes identically -- the mapping is reproduced,
never re-derived.
Usage: worker_cam.py <splatfacto_config.yml> <cam_index> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_orig_load = torch.load


def _patched_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_load(*a, **k)


torch.load = _patched_load

cfg_path = Path(sys.argv[1]).resolve()
ci = int(sys.argv[2])
out = Path(sys.argv[3])

# nerfstudio's own !!python/object config graph, self-produced on this box --
# identical to object_crop.py, which is the whole point (same env, same graph).
config = yaml.load(cfg_path.read_text(), Loader=yaml.Loader)
dp = config.pipeline.datamanager.dataparser
dp.data = cfg_path.parents[2]
outputs = dp.setup().get_dataparser_outputs(split="train")
cams = outputs.cameras
n = int(cams.camera_to_worlds.shape[0])
if not 0 <= ci < n:
    raise SystemExit(f"FATAL: camera index {ci} out of range (train split has {n})")

c2w = np.eye(4)
c2w[:3, :] = cams.camera_to_worlds[ci].numpy()
out.write_text(json.dumps({
    "n_train_cameras": n,
    "cam_index": ci,
    "image_filename": str(outputs.image_filenames[ci]),
    "camera_to_worlds_3x4_opengl": c2w[:3, :].tolist(),
    "fx": float(cams.fx[ci]), "fy": float(cams.fy[ci]),
    "cx": float(cams.cx[ci]), "cy": float(cams.cy[ci]),
    "width": int(cams.width[ci]), "height": int(cams.height[ci]),
    "dataparser_scale": float(getattr(outputs, "dataparser_scale", float("nan"))),
    "dataparser_transform": np.asarray(outputs.dataparser_transform).tolist(),
}, indent=2))
print("WORKER_CAM_DONE")
'''


# ===========================================================================
# Preflight
# ===========================================================================

def doctor() -> dict:
    """Fail-loud environment assertion. Cheap models -- and tired humans --
    cannot infer a broken install from a mid-run CUDA traceback."""
    checks = [
        ("sam3d-objects repo", SAM3D_ROOT.is_dir()),
        ("sam3d-objects pipeline.yaml", SAM3D_PIPELINE_YAML.is_file()),
        ("sam3d-objects env python", SAM3D_PYTHON.is_file()),
        ("sam3 repo", SAM3_ROOT.is_dir()),
        ("sam3.1 checkpoint (~3.3G)", SAM3_CKPT.is_file() and SAM3_CKPT.stat().st_size > 2e9),
        ("sam3 bpe asset", SAM3_BPE.is_file()),
        ("sam3 env python", SAM3_PYTHON.is_file()),
        ("dn-splatter-probe env python", PROBE_PYTHON.is_file()),
    ]
    ckpt_dir = SAM3D_PIPELINE_YAML.parent
    ckpt_bytes = 0
    for name in SAM3D_CKPTS:
        p = ckpt_dir / name
        present = p.is_file() and p.stat().st_size > 1_000_000
        if present:
            ckpt_bytes += p.stat().st_size
        checks.append((f"checkpoint {name}", present))

    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        log(f"{'OK  ' if ok else 'FAIL'} {n}")
    if bad:
        raise Fatal("preflight failed: " + "; ".join(bad))
    return {"checkpoint_dir": str(ckpt_dir),
            "checkpoint_bytes": ckpt_bytes,
            "checkpoints": list(SAM3D_CKPTS)}


# ===========================================================================
# Job-tree input resolution (two capture layouts exist in the wild)
# ===========================================================================

def _first_existing(*paths: Path):
    for p in paths:
        if p and p.is_file():
            return p
    return None


def resolve_inputs(job: Path, slug: str) -> dict:
    """Locate crop / crop metadata / captured object splat / inventory record.
    Handles both observed layouts: _objects/<slug>/... (P5c) and
    _scene/crop_<slug>.png + _scene/isolated/<slug>/ (batch_isolate)."""
    inv_path = job / "_scene" / "inventory.json"
    if not inv_path.is_file():
        raise Fatal(f"no {inv_path} -- run the scene enumeration lane first")
    inventory = json.loads(inv_path.read_text())
    inst = next((i for i in inventory.get("instances", []) if i.get("slug") == slug), None)
    if inst is None:
        have = [i.get("slug") for i in inventory.get("instances", [])]
        raise Fatal(f"object {slug!r} not in {inv_path} (have: {have})")

    crop = _first_existing(job / "_objects" / slug / "crop.png",
                           job / "_scene" / f"crop_{slug}.png")
    if crop is None:
        raise Fatal(f"no crop for {slug!r} "
                    f"(looked for _objects/{slug}/crop.png and _scene/crop_{slug}.png)")
    crop_json = crop.with_suffix(".json")
    if crop_json.is_file():                       # object_crop.py side-file (P5c)
        crop_meta = json.loads(crop_json.read_text())
        crop_meta.setdefault("source", "object_crop.py crop.json side-file")
    elif inst.get("best_view") is not None and inst.get("best_view_box"):
        # _scene/crop_<slug>.png (instance_lift.py) ships no side-file, but its
        # box is exactly reproducible: best_view_box padded by 10% of its long
        # edge and clamped to the frame (instance_lift.py, receipt block). `box`
        # stays None until the camera gives us the frame size to clamp against.
        crop_meta = {"cam": int(inst["best_view"]), "box": None,
                     "base_box": list(inst["best_view_box"]),
                     "source": "derived from inventory best_view/best_view_box "
                               "+ instance_lift.py's 10% pad"}
    else:
        crop_meta = None

    obj_ply = _first_existing(job / "_objects" / slug / "object.ply",
                              job / "_scene" / "isolated" / slug / "object.ply")
    configs = sorted((job / "processed").glob("splatfacto/*/config.yml"))

    return {
        "job_dir": str(job),
        "slug": slug,
        "label": inst.get("label"),
        "instance": inst,
        "inventory": str(inv_path),
        "crop_png": str(crop),
        "crop_json": str(crop_json) if crop_meta else None,
        "crop_meta": crop_meta,
        "captured_object_ply": str(obj_ply) if obj_ply else None,
        "splatfacto_config": str(configs[-1]) if configs else None,
        "bbox_tight_scene": inst.get("bbox_tight_scene"),
    }


# ===========================================================================
# Subprocess plumbing
# ===========================================================================

def run_worker(python: Path, script: Path, args: list[str], cwd: Path,
               log_path: Path, env_extra: dict | None = None,
               done_token: str = "", timeout: int = 5400) -> None:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    if env_extra:
        env.update(env_extra)
    log(f"run: {python.name} {script.name} {' '.join(args)}  (log: {log_path})")
    with open(log_path, "wb") as fh:
        proc = subprocess.run([str(python), str(script), *args], cwd=str(cwd),
                              env=env, stdout=fh, stderr=subprocess.STDOUT,
                              timeout=timeout)
    tail = log_path.read_text(errors="replace")[-4000:]
    if proc.returncode != 0:
        raise Fatal(f"{script.name} exited {proc.returncode}. Log tail:\n{tail}")
    if done_token and done_token not in tail:
        raise Fatal(f"{script.name} exited 0 but never printed {done_token} "
                    f"-- treat as failure. Log tail:\n{tail}")


# ===========================================================================
# Geometry helpers: capture camera projection + silhouette rasterisation
# ===========================================================================

# nerfstudio stores OpenGL-convention camera_to_worlds (+X right, +Y up, -Z fwd).
# This flip converts to the OpenCV convention the pinhole projection below
# assumes -- identical to object_crop.py, which produced the crop boxes.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def project_to_crop(points: np.ndarray, cam: dict, box) -> tuple[np.ndarray, np.ndarray]:
    """World/scene points -> pixel coords in the CROP's local frame."""
    c2w = np.eye(4)
    c2w[:3, :] = np.asarray(cam["camera_to_worlds_3x4_opengl"], dtype=np.float64)
    w2c = np.linalg.inv(c2w @ _GL_TO_CV)
    pc = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T
    front = pc[:, 2] > 1e-4
    uv = pc[front, :2] / pc[front, 2:3]
    px = np.stack([uv[:, 0] * cam["fx"] + cam["cx"],
                   uv[:, 1] * cam["fy"] + cam["cy"]], axis=1)
    px[:, 0] -= box[0]
    px[:, 1] -= box[1]
    return px, front


def rasterise_silhouette(px: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Point splat -> filled silhouette. Fixed recipe, applied identically to
    the captured reference and every generated candidate."""
    h, w = shape
    m = np.zeros((h, w), bool)
    if len(px):
        xi = np.round(px[:, 0]).astype(np.int64)
        yi = np.round(px[:, 1]).astype(np.int64)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        m[yi[ok], xi[ok]] = True
    if not m.any():
        return m
    m = ndimage.binary_dilation(m, np.ones((RASTER_DILATE, RASTER_DILATE), bool))
    m = ndimage.binary_closing(m, np.ones((RASTER_CLOSE, RASTER_CLOSE), bool))
    return ndimage.binary_fill_holes(m)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def load_splat_xyz(path: Path, bbox=None) -> np.ndarray:
    """xyz of SOLID gaussians (opacity > 0.5) from the raw 14-field 3DGS PLY
    this program writes everywhere; optionally clipped to a scene bbox."""
    with open(path, "rb") as f:
        header, n_props, names = b"", 0, []
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise Fatal(f"{path}: truncated PLY header")
            header += line
            if line.startswith(b"property"):
                n_props += 1
                names.append(line.split()[-1].decode())
        data = np.fromfile(f, dtype=np.float32)
    if n_props == 0 or data.size % n_props:
        raise Fatal(f"{path}: {data.size} floats not divisible by {n_props} properties")
    data = data.reshape(-1, n_props)
    xyz = data[:, :3].astype(np.float64)
    if "opacity" in names:
        opac = 1.0 / (1.0 + np.exp(-data[:, names.index("opacity")]))
        xyz = xyz[opac > 0.5]
    if bbox is not None:
        lo, hi = np.asarray(bbox["min"]), np.asarray(bbox["max"])
        xyz = xyz[((xyz >= lo) & (xyz <= hi)).all(axis=1)]
    return xyz


# ===========================================================================
# The mask gate -- runs BEFORE the 13 GB model load
# ===========================================================================

def pick_and_gate_mask(masks: np.ndarray, scores: np.ndarray,
                       captured_sil: np.ndarray | None, min_iou: float):
    """Choose the SAM 3 instance that agrees with the CAPTURED 3D object, then
    gate it. Selecting by agreement (not by detector confidence) is what stops a
    co-visible neighbour -- the bonsai sitting inside the bicycle crop -- from
    being reconstructed under the bicycle's name."""
    if masks is None or len(masks) == 0:
        raise Fatal("SAM 3 returned zero instances for this crop/prompt -- "
                    "no mask, no reconstruction (pass --mask to override)")
    rows = []
    for i, m in enumerate(masks):
        row = {"index": i, "score": round(float(scores[i]), 4),
               "coverage": round(float(m.mean()), 5)}
        if captured_sil is not None:
            row["iou_vs_captured"] = round(iou(m, captured_sil), 4)
        rows.append(row)
    key = "iou_vs_captured" if captured_sil is not None else "score"
    best = max(rows, key=lambda r: r[key])
    chosen = masks[best["index"]]

    verdict = {"candidates": rows, "chosen": best, "selected_by": key,
               "coverage": best["coverage"]}
    problems = []
    if best["coverage"] < 0.02:
        problems.append(f"mask covers only {best['coverage']:.3%} of the crop")
    if best["coverage"] > 0.98:
        problems.append(f"mask covers {best['coverage']:.3%} of the crop (no background left)")
    lab, n_cc = ndimage.label(chosen)
    if n_cc:
        sizes = ndimage.sum(chosen, lab, range(1, n_cc + 1))
        verdict["components"] = int(n_cc)
        verdict["largest_component_frac"] = round(float(sizes.max() / sizes.sum()), 4)
        if verdict["largest_component_frac"] < 0.5:
            problems.append(f"mask is fragmented: largest component is only "
                            f"{verdict['largest_component_frac']:.2f} of it")
    if captured_sil is not None:
        verdict["iou_vs_captured_object"] = best["iou_vs_captured"]
        if best["iou_vs_captured"] < min_iou:
            problems.append(f"mask/captured-object IoU {best['iou_vs_captured']:.3f} "
                            f"< --min-mask-iou {min_iou}")
        verdict["method"] = "captured-gaussian projection through the crop camera"
    else:
        verdict["method"] = "2d-sanity-only (no camera/captured splat available)"
    verdict["problems"] = problems
    verdict["passed"] = not problems
    return verdict, chosen


# ===========================================================================
# Provenance tagging + mandatory readback
# ===========================================================================

def tag_ply_generative(path: Path) -> None:
    """Insert `comment <GENERATIVE_TAG>` into the PLY header in place. The
    SAM 3D gaussian writer has no comment hook, and the tag must ride INSIDE
    the file so it survives the file leaving the job tree."""
    raw = path.read_bytes()
    if not raw.startswith(b"ply"):
        raise Fatal(f"{path}: not a PLY (cannot tag)")
    end = raw.find(b"end_header")
    if end == -1:
        raise Fatal(f"{path}: PLY header has no end_header (cannot tag)")
    if GENERATIVE_TAG.encode() in raw[:end]:
        return
    patched = raw[:end] + b"comment " + GENERATIVE_TAG.encode() + b"\n" + raw[end:]
    tmp = path.with_suffix(path.suffix + ".tagging")
    tmp.write_bytes(patched)
    tmp.replace(path)


def _glb_chunks(raw: bytes):
    magic, version, length = struct.unpack("<4sII", raw[:12])
    if magic != b"glTF":
        raise Fatal("not a GLB (bad magic)")
    off, chunks = 12, []
    while off + 8 <= len(raw):
        clen, ctype = struct.unpack("<II", raw[off:off + 8])
        chunks.append((ctype, raw[off + 8:off + 8 + clen]))
        off += 8 + clen
    return version, chunks


def tag_glb_generative(path: Path) -> None:
    """Write GENERATIVE_TAG into asset.extras and EVERY node.extras/mesh.extras
    under GLTF_EXTRAS_KEY -- the exact key blender_assemble.py's contamination
    gate reads back."""
    raw = path.read_bytes()
    version, chunks = _glb_chunks(raw)
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise Fatal(f"{path}: first GLB chunk is not JSON")
    doc = json.loads(chunks[0][1])
    for key in ("asset",):
        doc.setdefault(key, {}).setdefault("extras", {})[GLTF_EXTRAS_KEY] = GENERATIVE_TAG
    for coll in ("nodes", "meshes", "scenes"):
        for item in doc.get(coll, []):
            item.setdefault("extras", {})[GLTF_EXTRAS_KEY] = GENERATIVE_TAG

    new_json = json.dumps(doc, separators=(",", ":")).encode()
    new_json += b" " * ((4 - len(new_json) % 4) % 4)          # JSON pads with spaces
    body = struct.pack("<II", len(new_json), 0x4E4F534A) + new_json
    for ctype, data in chunks[1:]:
        pad = b"\x00" if ctype == 0x004E4942 else b" "
        data = data + pad * ((4 - len(data) % 4) % 4)
        body += struct.pack("<II", len(data), ctype) + data
    out = struct.pack("<4sII", b"glTF", version, 12 + len(body)) + body
    tmp = path.with_suffix(path.suffix + ".tagging")
    tmp.write_bytes(out)
    tmp.replace(path)


def verify_ply(path: Path) -> dict:
    """Open the WRITTEN file. Header counts + tag presence, from disk."""
    if not path.is_file():
        raise Fatal(f"{path}: expected PLY was not written")
    with open(path, "rb") as f:
        head = f.read(65536)
    if not head.startswith(b"ply"):
        raise Fatal(f"{path}: not a PLY")
    n_vertex, props = 0, 0
    for line in head[:head.find(b"end_header")].splitlines():
        if line.startswith(b"element vertex"):
            n_vertex = int(line.split()[-1])
        elif line.startswith(b"property"):
            props += 1
    tagged = any(GENERATIVE_TAG in c for c in ply_header_comments(path))
    if not tagged:
        raise Fatal(f"{path}: GENERATIVE_TAG missing from the written PLY header")
    if n_vertex <= 0:
        raise Fatal(f"{path}: PLY declares {n_vertex} vertices")
    return {"path": str(path), "bytes": path.stat().st_size,
            "gaussians": n_vertex, "properties": props, "generative_tag": True}


def verify_glb(path: Path) -> dict:
    """Open the WRITTEN file. Real accessor counts + tag on asset and every
    node, parsed straight out of the GLB's own JSON chunk."""
    if not path.is_file():
        raise Fatal(f"{path}: expected GLB was not written")
    _, chunks = _glb_chunks(path.read_bytes())
    if not chunks or chunks[0][0] != 0x4E4F534A:
        raise Fatal(f"{path}: first GLB chunk is not JSON")
    doc = json.loads(chunks[0][1])
    acc = doc.get("accessors", [])
    verts = tris = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None and pos < len(acc):
                verts += int(acc[pos].get("count", 0))
            idx = prim.get("indices")
            if idx is not None and idx < len(acc):
                tris += int(acc[idx].get("count", 0)) // 3
    if verts == 0 or tris == 0:
        raise Fatal(f"{path}: GLB has {verts} vertices / {tris} triangles -- empty mesh")
    nodes = doc.get("nodes", [])
    untagged = [n.get("name", f"#{i}") for i, n in enumerate(nodes)
                if (n.get("extras") or {}).get(GLTF_EXTRAS_KEY) != GENERATIVE_TAG]
    if untagged:
        raise Fatal(f"{path}: nodes missing the generative tag: {untagged}")
    if (doc.get("asset", {}).get("extras") or {}).get(GLTF_EXTRAS_KEY) != GENERATIVE_TAG:
        raise Fatal(f"{path}: asset.extras is missing the generative tag")
    has_colors = any("COLOR_0" in p.get("attributes", {})
                     for m in doc.get("meshes", []) for p in m.get("primitives", []))
    return {"path": str(path), "bytes": path.stat().st_size,
            "vertices": verts, "triangles": tris, "nodes": len(nodes),
            "vertex_colors": has_colors, "generative_tag": True}


# ===========================================================================
# Placement: fit the generated frame into the capture frame, then SCORE it
# ===========================================================================

def _yaw(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation of theta radians about `axis` (Rodrigues)."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def _axis_rotation(src_axis: np.ndarray, dst_axis: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit src_axis onto unit dst_axis (Rodrigues).
    The antiparallel case has no minimal axis, so pick any perpendicular one
    and rotate by pi -- a proper rotation, never a reflection."""
    a = src_axis / np.linalg.norm(src_axis)
    b = dst_axis / np.linalg.norm(dst_axis)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        return _yaw(perp / np.linalg.norm(perp), np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def fit_similarity(verts_rot: np.ndarray, bbox: dict) -> tuple[float, np.ndarray]:
    """Uniform scale + translation matching the ROTATED generated AABB to the
    object's known bbox_tight_scene. Robust percentiles, as in proxy_register.py.
    An AABB fixes size and centre only -- orientation must come from elsewhere."""
    lo = np.percentile(verts_rot, 1, axis=0)
    hi = np.percentile(verts_rot, 99, axis=0)
    tlo, thi = np.asarray(bbox["min"], float), np.asarray(bbox["max"], float)
    s = float(np.linalg.norm(thi - tlo) / max(np.linalg.norm(hi - lo), 1e-9))
    t = (tlo + thi) / 2.0 - s * (lo + hi) / 2.0
    return s, t


def derive_placement(verts: np.ndarray, bbox: dict, cam: dict | None, box,
                     mask: np.ndarray | None, captured_sil: np.ndarray | None,
                     min_iou: float, yaw_step_deg: float = 10.0,
                     applies_to: str = "") -> dict:
    """Fit generated -> capture frame and score every candidate against the
    verified mask. Returns a transform ONLY if a candidate clears min_iou.

    Run PER ARTIFACT: the mesh and gaussian decoders do not share an axis
    convention (measured on splat_513e89171d: mesh up = +Y, splat up = +Z --
    the vendor's own _fix_gaussian_alignment swaps them), so one artifact's
    transform is wrong for the other by a 90 degree rotation."""
    v = verts.astype(np.float64)
    if len(v) > RASTER_MAX_POINTS:
        v = v[np.random.default_rng(0).choice(len(v), RASTER_MAX_POINTS, replace=False)]
    v_c = v - (np.percentile(v, 1, axis=0) + np.percentile(v, 99, axis=0)) / 2.0

    ext = np.asarray(bbox["max"], float) - np.asarray(bbox["min"], float)
    gen_ext = np.percentile(v, 99, axis=0) - np.percentile(v, 1, axis=0)
    frame = {
        "applies_to": applies_to,
        "generated_frame": "SAM 3D normalised object frame (no relation to capture units)",
        "generated_extent_p1_p99": [round(float(x), 6) for x in gen_ext],
        "generated_diagonal": round(float(np.linalg.norm(gen_ext)), 6),
        "capture_bbox_tight_scene": bbox,
        "capture_extent": [round(float(x), 6) for x in ext],
        "capture_diagonal": round(float(np.linalg.norm(ext)), 6),
        "uniform_scale_from_bbox_diagonal": round(
            float(np.linalg.norm(ext) / max(np.linalg.norm(gen_ext), 1e-9)), 6),
        "note": ("Uniform scale is the bbox-diagonal ratio: the only "
                 "orientation-independent size statistic an AABB pair supports. "
                 "Per-axis ratios are NOT reported because they are meaningless "
                 "until orientation is fixed."),
    }
    if cam is None or mask is None or box is None:
        frame["placement_resolved"] = False
        frame["reason"] = ("no crop camera / verified mask -- orientation cannot be "
                           "scored, so no capture-frame transform is emitted. "
                           "Register with proxy_register.py (bbox init + scaled ICP "
                           "against the captured object.ply) instead.")
        return frame

    shape = (box[3] - box[1], box[2] - box[0])
    up_axes = [np.array(a, float) for a in
               ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]
    scene_up = np.array([0.0, 0.0, 1.0])   # nerfstudio auto-oriented world up

    def score(R: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        vr = v_c @ R.T
        s, t = fit_similarity(vr, bbox)
        px, _ = project_to_crop(vr * s + t, cam, box)
        sil = rasterise_silhouette(px, shape)
        return iou(sil, mask), s, t, sil

    t0 = time.time()
    trials = []
    coarse = np.arange(0.0, 360.0, yaw_step_deg)
    for ai, up in enumerate(up_axes):
        R_align = _axis_rotation(up, scene_up)
        for deg in coarse:
            R = _yaw(scene_up, np.radians(deg)) @ R_align
            sc, s, t, _ = score(R)
            trials.append({"up_axis": ai, "yaw_deg": float(deg), "iou": sc,
                           "scale": s, "t": t, "R": R})
    best = max(trials, key=lambda r: r["iou"])
    for deg in np.arange(best["yaw_deg"] - yaw_step_deg, best["yaw_deg"] + yaw_step_deg, 2.0):
        R = _yaw(scene_up, np.radians(deg)) @ _axis_rotation(up_axes[best["up_axis"]], scene_up)
        sc, s, t, _ = score(R)
        if sc > best["iou"]:
            best = {"up_axis": best["up_axis"], "yaw_deg": float(deg), "iou": sc,
                    "scale": s, "t": t, "R": R}
    search_s = time.time() - t0

    frame["orientation_search"] = {
        "method": ("6 up-axis assignments x yaw about the scene up axis, each scored "
                   "by silhouette IoU against the verified mask in the crop camera"),
        "scene_up_axis_assumed": scene_up.tolist(),
        "assumption": ("the object stands upright on the nerfstudio auto-oriented +Z. "
                       "A tilted object cannot be represented by this search and will "
                       "simply score badly -- which is reported, not hidden."),
        "scored_from_n_views": 1,
        "uniqueness_caveat": (
            "The score is a silhouette from the single crop camera, so it certifies "
            "CONSISTENCY WITH THAT VIEW, not a unique orientation. A symmetric object "
            "(a box, a bottle) has a whole equivalence class of yaws with the same "
            "silhouette and the search returns an arbitrary member of it. Read "
            "iou_spread: a narrow spread means the view barely discriminates."),
        "ceiling_note": (
            "fraction_of_captured_ceiling > 1 means the generated silhouette matches "
            "the mask BETTER than the captured gaussians do -- expected when the "
            "capture is sparse or holey, and a reason the generated candidate exists."),
        "candidates_scored": len(trials) + int(yaw_step_deg),
        "seconds": round(search_s, 1),
        "best_up_axis_index": int(best["up_axis"]),
        "best_yaw_deg": round(best["yaw_deg"], 1),
        "best_silhouette_iou": round(best["iou"], 4),
        "iou_spread": [round(float(min(r["iou"] for r in trials)), 4),
                       round(float(max(r["iou"] for r in trials)), 4)],
    }
    if captured_sil is not None:
        ref = iou(captured_sil, mask)
        frame["orientation_search"]["captured_object_silhouette_iou"] = round(ref, 4)
        frame["orientation_search"]["fraction_of_captured_ceiling"] = round(
            best["iou"] / ref, 4) if ref > 0 else None

    if best["iou"] < min_iou:
        frame["placement_resolved"] = False
        frame["reason"] = (f"best silhouette IoU {best['iou']:.3f} < --min-placement-iou "
                           f"{min_iou}: orientation is NOT resolved, so no transform is "
                           f"emitted. Scale alone "
                           f"({frame['uniform_scale_from_bbox_diagonal']}) is a bbox "
                           f"ratio and does not place the object.")
        return frame

    # x' = s*R*(x - centre) + t, folded into one 4x4 (proxy_register.py's form).
    centre = (np.percentile(v, 1, axis=0) + np.percentile(v, 99, axis=0)) / 2.0
    M = np.eye(4)
    M[:3, :3] = best["scale"] * best["R"]
    M[:3, 3] = best["t"] - best["scale"] * (best["R"] @ centre)
    frame["placement_resolved"] = True
    frame["transform_4x4_generated_to_capture"] = [
        [round(float(x), 8) for x in row] for row in M]
    frame["fitted_uniform_scale"] = round(float(best["scale"]), 6)
    frame["evidence"] = (f"silhouette IoU {best['iou']:.3f} against the SAM 3 mask "
                         f"through the crop camera -- fitted, not read out of the model")
    return frame


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("job_dir", help="capture job dir, e.g. .../outputs/3d/splat_xxxx")
    ap.add_argument("--object", required=True,
                    help="instance slug, or comma-separated slugs (one model load)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="default <job>/_regen/objects/<slug> (a provenance quarantine path)")
    ap.add_argument("--prompt", default=None, help="mask prompt; default = inventory label")
    ap.add_argument("--mask", default=None,
                    help="explicit mask (PNG/NPY, crop-sized); skips SAM 3")
    ap.add_argument("--mask-threshold", type=float, default=0.4)
    ap.add_argument("--min-mask-iou", type=float, default=0.35,
                    help="abort if mask vs captured-object projection falls below this")
    ap.add_argument("--min-placement-iou", type=float, default=0.50,
                    help="withhold the capture-frame transform below this silhouette IoU")
    ap.add_argument("--allow-unverified-mask", action="store_true",
                    help="proceed when the mask cannot be checked against the capture")
    ap.add_argument("--yaw-step", type=float, default=10.0)
    args = ap.parse_args()

    t_start = time.time()
    job = Path(args.job_dir).expanduser().resolve()
    if not job.is_dir():
        raise Fatal(f"{job} is not a directory")
    slugs = [s.strip() for s in args.object.split(",") if s.strip()]

    ckpt_info = doctor()

    # ---- resolve every input up front; nothing heavy has been loaded yet ----
    plans = []
    for slug in slugs:
        src = resolve_inputs(job, slug)
        out = Path(args.out).expanduser().resolve() if args.out else \
            job / "_regen" / "objects" / slug
        if args.out and len(slugs) > 1:
            out = out / slug
        work = out / "_work"
        work.mkdir(parents=True, exist_ok=True)
        src["out_dir"], src["work_dir"] = str(out), str(work)
        plans.append(src)
        log(f"{slug}: crop={src['crop_png']} label={src['label']!r} "
            f"cam={(src['crop_meta'] or {}).get('cam')}")

    work_root = Path(plans[0]["work_dir"])
    for name, body in (("worker_sam3.py", WORKER_SAM3),
                       ("worker_sam3d.py", WORKER_SAM3D),
                       ("worker_cam.py", WORKER_CAM)):
        (work_root / name).write_text(body)

    # ---- capture camera (optional but required for every real gate) --------
    for p in plans:
        p["camera"] = None
        cm, cfg = p["crop_meta"], p["splatfacto_config"]
        if not cm or cm.get("cam") is None or not cfg:
            log(f"{p['slug']}: no crop.json/config.yml -> no camera, gates degrade")
            continue
        cam_json = Path(p["work_dir"]) / "camera.json"
        try:
            run_worker(PROBE_PYTHON, work_root / "worker_cam.py",
                       [cfg, str(cm["cam"]), str(cam_json)], job,
                       Path(p["work_dir"]) / "worker_cam.log",
                       done_token="WORKER_CAM_DONE", timeout=900)
            p["camera"] = json.loads(cam_json.read_text())
            log(f"{p['slug']}: camera {cm['cam']} -> {Path(p['camera']['image_filename']).name}")
        except Exception as exc:
            log(f"{p['slug']}: camera probe FAILED ({exc}) -- gates degrade")
            continue
        if cm.get("box") is None and cm.get("base_box"):
            x0, y0, x1, y1 = cm["base_box"]
            pad = int(0.1 * max(x1 - x0, y1 - y0, 1))
            cm["box"] = [max(0, x0 - pad), max(0, y0 - pad),
                         min(p["camera"]["width"], x1 + pad),
                         min(p["camera"]["height"], y1 + pad)]
            log(f"{p['slug']}: reconstructed crop box {cm['box']} (pad {pad})")

    # ---- captured-object silhouette: the mask's independent yardstick ------
    for p in plans:
        p["captured_sil"] = None
        crop = Image.open(p["crop_png"]).convert("RGB")
        p["crop_size"] = crop.size                       # (w, h)
        box = (p["crop_meta"] or {}).get("box")
        p["box"] = box
        if not (p["camera"] and box and p["captured_object_ply"] and p["bbox_tight_scene"]):
            continue
        shape = (box[3] - box[1], box[2] - box[0])
        if shape != (crop.size[1], crop.size[0]):
            log(f"{p['slug']}: crop.json box {shape} != crop.png {crop.size[::-1]} "
                f"-- refusing to use a stale crop box")
            p["box"] = None
            continue
        xyz = load_splat_xyz(Path(p["captured_object_ply"]), p["bbox_tight_scene"])
        px, _ = project_to_crop(xyz, p["camera"], box)
        p["captured_sil"] = rasterise_silhouette(px, shape)
        log(f"{p['slug']}: captured silhouette from {len(xyz)} solid gaussians, "
            f"coverage {p['captured_sil'].mean():.3f}")

    # ---- masks (one SAM 3 load for all crops) ------------------------------
    need_sam3 = [p for p in plans if not args.mask]
    if need_sam3:
        jobs = {"checkpoint": str(SAM3_CKPT), "bpe": str(SAM3_BPE),
                "threshold": args.mask_threshold,
                "crops": [{"image": p["crop_png"],
                           "prompt": args.prompt or p["label"] or p["slug"].replace("-", " "),
                           "out": str(Path(p["work_dir"]) / "sam3_masks.npz")}
                          for p in need_sam3]}
        jf = work_root / "jobs_sam3.json"
        jf.write_text(json.dumps(jobs, indent=2))
        run_worker(SAM3_PYTHON, work_root / "worker_sam3.py", [str(jf)], SAM3_ROOT,
                   work_root / "worker_sam3.log", done_token="WORKER_SAM3_DONE")

    gate_failures = []
    for p in plans:
        if args.mask:
            mp = Path(args.mask)
            arr = np.load(mp) if mp.suffix == ".npy" else np.array(Image.open(mp))
            if arr.ndim == 3:
                arr = arr[..., -1]
            masks, scores = arr[None] > 0, np.array([1.0], np.float32)
            p["mask_source"] = f"explicit --mask {mp}"
        else:
            d = np.load(Path(p["work_dir"]) / "sam3_masks.npz")
            masks, scores = d["masks"], d["scores"]
            p["mask_source"] = (f"SAM 3.1 text-prompted on the crop, prompt="
                                f"{args.prompt or p['label']!r}, thr={args.mask_threshold}")
        if masks.shape[1:] != (p["crop_size"][1], p["crop_size"][0]):
            raise Fatal(f"{p['slug']}: mask {masks.shape[1:]} != crop "
                        f"{(p['crop_size'][1], p['crop_size'][0])}")
        verdict, chosen = pick_and_gate_mask(masks, scores,
                                             p["captured_sil"], args.min_mask_iou)
        p["mask_gate"], p["mask"] = verdict, chosen

        # Receipt FIRST: the operator has to be able to look at the thing the
        # refusal names, so it must exist before the refusal is emitted.
        rgb = np.array(Image.open(p["crop_png"]).convert("RGB"))
        panes = [rgb]
        ov = rgb.copy()
        ov[chosen] = (0.45 * ov[chosen] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
        panes.append(ov)
        if p["captured_sil"] is not None:
            ov2 = rgb.copy()
            sel = p["captured_sil"]
            ov2[sel] = (0.45 * ov2[sel] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
            panes.append(ov2)
        Image.fromarray(np.concatenate(panes, axis=1)).save(
            Path(p["out_dir"]) / "alignment.png")

        log(f"{p['slug']}: mask gate {'PASS' if verdict['passed'] else 'FAIL'} "
            f"({verdict['method']}) {verdict.get('iou_vs_captured_object', '-')}")
        reasons = list(verdict["problems"])
        if p["captured_sil"] is None:
            reasons.append("mask could not be checked against the capture "
                           "(no camera / captured splat / crop box)")
        if reasons and not args.allow_unverified_mask:
            # An open-topology object (a spoked wheel, a chair back) makes SAM
            # fill a region the captured gaussians legitimately leave hollow.
            # That is a real disagreement, not a bug -- name it so the operator
            # can judge from alignment.png instead of re-deriving it.
            if (p["captured_sil"] is not None
                    and verdict["coverage"] > 2.0 * float(p["captured_sil"].mean())):
                reasons.append(f"mask covers {verdict['coverage']:.3f} of the crop but the "
                               f"captured object only {p['captured_sil'].mean():.3f} -- "
                               f"likely an open/thin-topology object where SAM fills what "
                               f"the gaussians leave hollow, not a misalignment")
            gate_failures.append({"slug": p["slug"], "reasons": reasons,
                                  "alignment_png": str(Path(p["out_dir"]) / "alignment.png"),
                                  "gate": verdict})
            (Path(p["out_dir"]) / "generate_report.json").write_text(json.dumps(
                {"slug": p["slug"], "ok": False, "stage": "mask_alignment_gate",
                 "reasons": reasons, "mask_alignment_gate": verdict,
                 "alignment_png": str(Path(p["out_dir"]) / "alignment.png"),
                 "override": "--allow-unverified-mask (only with eyes on alignment.png)"},
                indent=2))
            log(f"{p['slug']}: SKIPPED -- {'; '.join(reasons)}")
            continue

        rgba = np.concatenate(
            [rgb, (chosen.astype(np.uint8) * 255)[..., None]], axis=-1).astype(np.uint8)
        np.save(Path(p["work_dir"]) / "input_rgba.npy", rgba)
        Image.fromarray(rgba).save(Path(p["out_dir"]) / "model_input_rgba.png")
        p["gated_in"] = True

    # One bad object must never cost the whole batch its model load -- same
    # doctrine as blender_assemble.py's per-element try/except.
    plans = [p for p in plans if p.get("gated_in")]
    if not plans:
        raise Fatal("every requested object failed the mask alignment gate: "
                    + json.dumps(gate_failures, indent=2))

    # ---- generate (one 13 GB model load for all objects) -------------------
    jobs = {"pipeline_yaml": str(SAM3D_PIPELINE_YAML),
            "results_out": str(work_root / "sam3d_results.json"),
            "objects": [{"slug": p["slug"], "seed": args.seed,
                         "rgba_npy": str(Path(p["work_dir"]) / "input_rgba.npy"),
                         "mesh_out": str(Path(p["out_dir"]) / "generated_mesh.glb"),
                         "splat_out": str(Path(p["out_dir"]) / "generated_splat.ply"),
                         "geom_npz": str(Path(p["work_dir"]) / "generated_geometry.npz"),
                         "preview_out": str(Path(p["out_dir"]) / "preview.png")}
                        for p in plans]}
    jf = work_root / "jobs_sam3d.json"
    jf.write_text(json.dumps(jobs, indent=2))
    run_worker(SAM3D_PYTHON, work_root / "worker_sam3d.py", [str(jf)], SAM3D_ROOT,
               work_root / "worker_sam3d.log",
               env_extra={"SAM3D_ROOT": str(SAM3D_ROOT),
                          "CONDA_PREFIX": str(SAM3D_PYTHON.parent.parent)},
               done_token="WORKER_SAM3D_DONE")
    results = {r["slug"]: r for r in json.loads((work_root / "sam3d_results.json").read_text())}

    # ---- verify what landed on disk, then place it -------------------------
    reports, failures = [], []
    for p in plans:
        slug, out = p["slug"], Path(p["out_dir"])
        r = results.get(slug)
        if r is None or not r.get("ok"):
            failures.append(slug)
            log(f"{slug}: GENERATION FAILED -- {(r or {}).get('error')}")
            (out / "generate_report.json").write_text(json.dumps(
                {"slug": slug, "ok": False, "worker": r}, indent=2))
            continue

        artifacts = {}
        if r.get("splat_out"):
            tag_ply_generative(Path(r["splat_out"]))
            artifacts["splat_ply"] = verify_ply(Path(r["splat_out"]))
        if r.get("mesh_out"):
            tag_glb_generative(Path(r["mesh_out"]))
            artifacts["mesh_glb"] = verify_glb(Path(r["mesh_out"]))
        if not artifacts:
            failures.append(slug)
            log(f"{slug}: worker reported ok but wrote no geometry")
            continue

        # Placement is fitted PER ARTIFACT -- see derive_placement's docstring
        # for why the mesh's transform is wrong for the splat.
        placement = {"warning": (
            "The mesh and gaussian decoders emit DIFFERENT axis conventions "
            "(measured, not assumed). NEVER apply one artifact's transform to "
            "the other -- each block below was fitted and scored on its own "
            "point set.")}
        geom = np.load(Path(p["work_dir"]) / "generated_geometry.npz")
        placement["mesh_glb"] = derive_placement(
            geom["vertices"], p["bbox_tight_scene"], p["camera"], p["box"],
            p.get("mask"), p.get("captured_sil"), args.min_placement_iou,
            args.yaw_step, applies_to="generated_mesh.glb vertices")
        if r.get("splat_out"):
            placement["splat_ply"] = derive_placement(
                load_splat_xyz(Path(r["splat_out"])), p["bbox_tight_scene"],
                p["camera"], p["box"], p.get("mask"), p.get("captured_sil"),
                args.min_placement_iou, args.yaw_step,
                applies_to="generated_splat.ply gaussian centres")

        report = {
            "schema": "splatlab.object_generate/1",
            "provenance": "generative",
            "provenance_tag": GENERATIVE_TAG,
            "provenance_note": ("GENERATED, not captured. Render/VR lane only. Never a "
                                "survey/geo artifact -- provenance.assert_not_generative() "
                                "refuses these files by path AND by content."),
            "model": {
                "name": MODEL_NAME,
                "repo": str(SAM3D_ROOT),
                "pipeline_yaml": str(SAM3D_PIPELINE_YAML),
                "checkpoint_dir": ckpt_info["checkpoint_dir"],
                "checkpoints": ckpt_info["checkpoints"],
                "checkpoint_bytes": ckpt_info["checkpoint_bytes"],
                "depth_prior": "MoGe (Ruicheng/moge-vitl) via the vendor pipeline",
                "decode_formats": ["gaussian", "mesh"],
                "options": {"with_mesh_postprocess": False, "with_texture_baking": False,
                            "with_layout_postprocess": False, "use_vertex_color": True},
                "env_shims": r.get("shims"),
            },
            "seed": args.seed,
            "determinism": (
                "The seed pins the diffusion sampling, but the run is NOT bit-"
                "reproducible: two identical-seed runs of splat_513e89171d/fire-hydrant "
                "produced 422052 vs 422020 mesh faces (~0.008%), from non-deterministic "
                "CUDA kernels in sparse-structure/flexicubes extraction. Expect "
                "equivalent, not identical, geometry on re-run."),
            "input_provenance": {
                "job_dir": str(job), "slug": slug, "label": p["label"],
                "crop_png": p["crop_png"], "crop_json": p["crop_json"],
                "crop_box_in_source_frame": p["box"],
                "source_camera_index": (p["crop_meta"] or {}).get("cam"),
                "source_image": (p["camera"] or {}).get("image_filename"),
                "camera_intrinsics": ({k: (p["camera"] or {}).get(k)
                                       for k in ("fx", "fy", "cx", "cy", "width", "height")}
                                      if p["camera"] else None),
                "dataparser_scale": (p["camera"] or {}).get("dataparser_scale"),
                "mask_source": p["mask_source"],
                "inventory": p["inventory"],
                "captured_object_ply": p["captured_object_ply"],
                "model_input_rgba": str(out / "model_input_rgba.png"),
                "input_rgba_shape": r.get("input_rgba_shape"),
                "input_mask_coverage": r.get("input_mask_coverage"),
            },
            "mask_alignment_gate": p["mask_gate"],
            "geometry": {
                "mesh_vertices": r.get("mesh_vertices"),
                "mesh_faces": r.get("mesh_faces"),
                "mesh_has_vertex_colors": r.get("mesh_has_vertex_colors"),
                "splat_gaussians": r.get("splat_gaussians"),
                "sparse_voxels": r.get("sparse_voxels"),
                "mesh_bounds_min": r.get("mesh_bounds_min"),
                "mesh_bounds_max": r.get("mesh_bounds_max"),
                "splat_bounds_min": r.get("splat_bounds_min"),
                "splat_bounds_max": r.get("splat_bounds_max"),
            },
            "model_frame_pose": {
                "note": ("SAM 3D's own layout head, expressed in MoGe's scale-shift-"
                         "invariant pointmap frame. Recorded verbatim; NOT a capture-"
                         "frame pose and not used to place the object."),
                "rotation_quat_local_to_camera_wxyz": r.get("pose_rotation"),
                "translation_local_to_camera": r.get("pose_translation"),
                "scale_local_to_camera": r.get("pose_scale"),
                "translation_scale": r.get("pose_translation_scale"),
            },
            "capture_frame_placement": placement,
            "artifacts": artifacts,
            "receipts": {
                "alignment_png": str(out / "alignment.png"),
                "preview_png": r.get("preview_out"),
                "worker_log": str(Path(p["work_dir"]) / "worker_sam3d.log"),
                "workers": [str(work_root / n) for n in
                            ("worker_sam3.py", "worker_sam3d.py", "worker_cam.py")],
            },
            "timings_seconds": {
                "pipeline_load": r.get("pipeline_load_seconds"),
                "inference": r.get("inference_seconds"),
                "total_wall": round(time.time() - t_start, 1),
            },
            "ok": True,
        }
        (out / "generate_report.json").write_text(json.dumps(report, indent=2))
        reports.append(report)
        log(f"{slug}: OK mesh={r.get('mesh_vertices')}v/{r.get('mesh_faces')}f "
            f"splat={r.get('splat_gaussians')}g placement_resolved="
            f"mesh:{placement['mesh_glb'].get('placement_resolved')}/"
            f"splat:{placement.get('splat_ply', {}).get('placement_resolved')} "
            f"-> {out}")

    print(json.dumps({
        "requested": slugs,
        "generated": [rp["input_provenance"]["slug"] for rp in reports],
        "refused_at_mask_gate": [g["slug"] for g in gate_failures],
        "generation_failed": failures,
        "gate_failures": gate_failures,
        "out_dirs": [p["out_dir"] for p in plans],
    }, indent=2, default=str))
    if failures or gate_failures:
        raise Fatal(f"incomplete: generation failed {failures}, "
                    f"mask gate refused {[g['slug'] for g in gate_failures]} "
                    f"(each has a generate_report.json and an alignment.png)")
    print("OBJECT_GENERATE_OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Fatal as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
