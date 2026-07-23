#!/usr/bin/env python3
"""P6b: SAM 3.1 text-prompted instance masks for EVERY noun in a scene's
"things" list, across the views scene_views.py exported. One model load
covers all nouns (loading the ~3.3GB checkpoint per-noun would be wasteful).

Adapted from the proven scene-regen-spike sam3_masks.py (Step 0 de-risk spike,
2026-07-22: mechanism PROVEN, recipe --min-views 2 --vote-frac 0.3 downstream
in instance_lift.py). Local checkpoint + bpe only, no downloads. Masks are
returned at original frame size.

Runs in the sam3 env (PYTHONNOUSERSITE=1).

Usage: scene_sam3_masks.py <workdir> <things.json> [threshold=0.5]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

WORK = Path(sys.argv[1])
THINGS = json.loads(Path(sys.argv[2]).read_text())
THRESH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

SAM3_ROOT = Path("/home/rtoony/projects/ml/sam3")
CKPT = SAM3_ROOT / "checkpoints" / "sam3.1_multiplex.pt"
BPE = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# NB: the upstream notebook does ctx().__enter__() on temporaries; in a plain
# script those get GC'd and the guards are released — use real with-blocks.

from sam3.model_builder import build_sam3_image_model                       # noqa: E402
from sam3.train.data.collator import collate_fn_api as collate              # noqa: E402
from sam3.model.utils.misc import copy_data_to_device                       # noqa: E402
from sam3.train.data.sam3_image_dataset import (                            # noqa: E402
    InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint)
from sam3.train.transforms.basic_for_api import (                           # noqa: E402
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
from sam3.eval.postprocessors import PostProcessImage                       # noqa: E402


def _slug(noun: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in noun.lower()).strip("-")[:40] or "thing"


def as_numpy(t):
    if torch.is_tensor(t):
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.detach().cpu().numpy()
    return np.asarray(t)


def main() -> int:
    views = json.load(open(WORK / "views.json"))
    model = build_sam3_image_model(checkpoint_path=str(CKPT), bpe_path=str(BPE))

    transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    postprocessor = PostProcessImage(
        max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
        use_original_sizes_mask=True, convert_mask_to_rle=False,
        detection_threshold=THRESH, to_cpu=True)

    manifest = {}
    qcounter = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for noun in THINGS:
            slug = _slug(noun)
            masks_dir = WORK / "masks" / slug
            masks_dir.mkdir(parents=True, exist_ok=True)
            manifest[noun] = {"slug": slug, "cams": []}
            for cam_i in views["cam_indices"]:
                pil = Image.open(WORK / "frames" / f"cam_{cam_i:03d}.png").convert("RGB")
                w, h = pil.size
                dp = Datapoint(find_queries=[], images=[])
                dp.images = [SAMImage(data=pil, objects=[], size=[h, w])]
                qcounter += 1
                dp.find_queries.append(FindQueryLoaded(
                    query_text=noun, image_id=0, object_ids_output=[], is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=qcounter, original_image_id=qcounter,
                        original_category_id=1, original_size=[h, w],
                        object_id=qcounter, frame_index=0)))
                dp = transform(dp)
                batch = collate([dp], dict_key="d")["d"]
                batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
                output = model(batch)
                res = postprocessor.process_results(output, batch.find_metadatas)
                (_, r), = res.items()
                m = as_numpy(r["masks"]) if "masks" in r else np.zeros((0, h, w), bool)
                if m.ndim == 4:
                    m = m[:, 0]
                m = m.astype(bool)
                scores = (as_numpy(r["scores"]).astype(np.float32)
                          if "scores" in r else np.ones(m.shape[0], np.float32))
                assert m.shape[1:] == (h, w), f"mask res {m.shape[1:]} != frame {(h, w)}"
                np.savez_compressed(masks_dir / f"cam_{cam_i:03d}.npz",
                                    masks=m, scores=scores, prompt=noun)
                manifest[noun]["cams"].append(cam_i)
                print(f"[scene-sam3] {noun!r} cam {cam_i:03d}: {m.shape[0]} instance(s)", flush=True)

    (WORK / "sam3_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SCENE_SAM3_MASKS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
