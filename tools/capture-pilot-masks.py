"""Source-bound raw-lens person/hand/sky exclusions for a bounded capture experiment."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests
from capture_structure import semantic_union

PROMPTS = ["person", "hand", "sky"]


def prepare(pilot, output):
    document = json.loads((pilot / "pilot.json").read_text())
    if document.get("status") != "decoded-needs-camera-and-mask-review" or len(document["views"]) > 192:
        raise ValueError("Use a completed bounded raw-lens pilot")
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    records = []
    for ordinal, view in enumerate(document["views"]):
        source = pilot / "images" / view["image"]
        if source.resolve() != source.absolute() or manifests.sha256_file(source) != view["sha256"]:
            raise ValueError("Pilot image changed or escaped its package")
        filename = f"frames/cam_{ordinal:03d}.png"
        with Image.open(source) as picture:
            picture.convert("RGB").save(output / filename)
        records.append({"ordinal": ordinal, "image": view["image"], "split": view["split"], "source_sha256": view["sha256"],
                        "file": filename, "sha256": manifests.sha256_file(output / filename)})
    manifests.atomic_write_json(output / "views.json", {"cam_indices": list(range(len(records)))})
    manifests.atomic_write_json(output / "things.json", PROMPTS)
    receipt = {"schema": "dev.splatlab.raw-lens-mask-study/v1", "status": "prepared-not-inferred", "pilot_sha256": manifests.sha256_file(pilot / "pilot.json"),
               "capture_id": document["capture_id"], "width": document["output_width"], "views": records, "prompts": PROMPTS,
               "files": {str(filename.relative_to(output)): manifests.sha256_file(filename) for filename in output.rglob("*") if filename.is_file()},
               "limitations": ["Local semantic predictions are not ground truth.", "Fixed prompts are applied independently to each split; held-out appearance is not scored or used to tune the model.",
                               "Geometric rim/nadir exclusion is not calibrated angular support.", "Foliage, traffic, reflections and shadows may remain."]}
    manifests.atomic_write_json(output / "prepared.json", receipt)
    return receipt


def verify(output):
    receipt = json.loads((output / "prepared.json").read_text())
    if receipt.get("schema") != "dev.splatlab.raw-lens-mask-study/v1" or receipt.get("status") != "prepared-not-inferred" or receipt["prompts"] != PROMPTS:
        raise ValueError("Expected the prepared fixed-prompt study")
    for filename, digest in receipt["files"].items():
        if manifests.sha256_file(output / filename) != digest:
            raise ValueError("Prepared source views changed")
    return receipt


def run(output, lease_path):
    if subprocess.run([str(ROOT / "tools/splatlab-compute-gate.sh"), "--is-contained"], capture_output=True).returncode:
        raise ValueError("Use the shared SplatLab compute gate")
    lease = json.loads(lease_path.read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(lease["no_new_starts_after"]):
        raise ValueError("The compute start window has closed")
    verify(output)
    if (output / "masks").exists() or (output / "inference.json").exists():
        raise ValueError("Preserve the prior inference attempt")
    script = ROOT / "backend/mesh/scene_sam3_masks.py"
    checkpoint = Path('/home/rtoony/projects/ml/sam3/checkpoints/sam3.1_multiplex.pt')
    snapshot = {str(filename): manifests.sha256_file(filename) for filename in [script, checkpoint, output / "prepared.json"]}
    receipt = {"status": "running", "started_at": manifests.utc_now(), "source_hashes": snapshot, "semantic_ground_truth": False}
    manifests.atomic_write_json(output / "inference.json", receipt)
    started = time.monotonic()
    try:
        environment = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONPATH='/home/rtoony/projects/ml/sam3')
        with (output / "sam3.log").open("x") as log:
            subprocess.run(['/home/rtoony/miniconda3/envs/sam3/bin/python', str(script), str(output), str(output / "things.json"), '.5'],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=480)
        verify(output)
        if snapshot != {filename: manifests.sha256_file(Path(filename)) for filename in snapshot}:
            raise ValueError("Model or prepared inputs changed")
        receipt.update(status="inferred-needs-review", files={str(filename.relative_to(output)): manifests.sha256_file(filename) for filename in (output / "masks").rglob("*.npz")})
    except BaseException as error:
        receipt.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        receipt.update(finished_at=manifests.utc_now(), seconds=time.monotonic() - started)
        manifests.atomic_write_json(output / "inference.json", receipt)
    return receipt


def combine_masks(excluded, width):
    if excluded.shape != (width, width) or excluded.dtype != bool or not 512 <= width <= 1920:
        raise ValueError("Need a matching boolean raw-lens exclusion mask")
    grown = np.asarray(Image.fromarray(excluded).filter(ImageFilter.MaxFilter(7)))
    vertical, horizontal = np.ogrid[:width, :width]
    return (np.hypot(horizontal + .5 - width / 2, vertical + .5 - width / 2) <= .43 * width) & (vertical + .5 < .76 * width) & ~grown


def review(output):
    prepared = verify(output)
    inferred = json.loads((output / "inference.json").read_text())
    if inferred["status"] != "inferred-needs-review" or (output / "mask-review.json").exists():
        raise ValueError("Use completed inference and preserve any prior review")
    for filename, digest in inferred["files"].items():
        if manifests.sha256_file(output / filename) != digest:
            raise ValueError("Inferred mask changed")
    width = prepared["width"]
    training = [view for view in prepared["views"] if view["split"] == "train"]
    sheet = Image.new("RGB", (4 * 256, math.ceil(len(training) / 4) * 280), '#20252d')
    drawing = ImageDraw.Draw(sheet)
    records = []
    contact = 0
    for view in prepared["views"]:
        excluded = np.zeros((width, width), dtype=bool)
        for prompt in PROMPTS:
            excluded |= semantic_union(output / f'masks/{prompt}/cam_{view["ordinal"]:03d}.npz', prompt, (width, width))
        keep = combine_masks(excluded, width)
        if keep.mean() < .03:
            raise ValueError("Less than 3% of a lens remains; inspect this clip rather than inventing support")
        filename = 'feature-masks/' + view["image"] + '.png'
        destination = output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(keep.astype(np.uint8) * 255).save(destination)
        records.append({**view, "mask": filename, "mask_sha256": manifests.sha256_file(destination), "retained_fraction": float(keep.mean())})
        if view["split"] == "train":
            with Image.open(output / view["file"]) as image:
                preview = Image.composite(image, Image.new('RGB', image.size, '#20252d'), Image.fromarray(keep.astype(np.uint8) * 255))
                preview.thumbnail((256, 256))
            left, top = (contact % 4) * 256, (contact // 4) * 280
            sheet.paste(preview, (left, top))
            drawing.text((left + 3, top + 259), view["image"], fill='white')
            contact += 1
    sheet.save(output / 'training-mask-review.png')
    result = {"schema": "dev.splatlab.raw-lens-exclusions/v1", "status": "prepared-needs-visual-review", "pilot_sha256": prepared["pilot_sha256"],
              "prepared_sha256": manifests.sha256_file(output / 'prepared.json'), "inference_sha256": manifests.sha256_file(output / 'inference.json'),
              "tool_sha256": manifests.sha256_file(Path(__file__)), "views": records, "prompts": PROMPTS, "width": width,
              "geometric_radius_fraction": .43, "nadir_exclusion_fraction": .76, "semantic_exclusion_dilation_pixels": 3,
              "semantic_ground_truth": False, "limitations": prepared["limitations"]}
    manifests.atomic_write_json(output / 'mask-review.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'review'])
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lease', type=Path)
    args = parser.parse_args()
    result = prepare(args.pilot, args.output) if args.action == 'prepare' else run(args.output, args.lease) if args.action == 'run' else review(args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ['files', 'views', 'source_hashes']}, indent=2))
