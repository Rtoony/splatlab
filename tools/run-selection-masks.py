#!/usr/bin/env python3
"""Run the installed SAM mask worker for a prepared study, inside the compute gate."""

import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import artifact_manifest as manifests
import capture_records
import selection_reviews as reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("review_id")
    args = parser.parse_args()
    with reviews.worker_slot(args.job.resolve(), args.review_id):
        run(args)


def run(args):
    job = args.job.resolve()
    receipt = reviews.read(job, args.review_id)
    reviews.verify(job, receipt)
    reviews.verify_prepared_files(job, receipt)
    output = reviews.directory(job, args.review_id)
    if (output / "masks").exists() or (output / "mask-run.json").exists():
        raise RuntimeError("Mask outputs already exist; prepare another review")
    checkpoint = Path("/home/rtoony/projects/ml/sam3/checkpoints/sam3.1_multiplex.pt")
    script = Path(__file__).resolve().parents[1] / "backend/mesh/scene_sam3_masks.py"
    identities = {"checkpoint": manifests.file_identity(checkpoint), "script": manifests.file_identity(script)}
    environment = capture_records.worker_env()
    environment.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    subprocess.run(["/home/rtoony/miniconda3/envs/sam3/bin/python", str(script), str(output), str(output / "things.json"), ".5"],
                   env=environment, check=True, timeout=210)
    reviews.verify(job, receipt)
    reviews.verify_prepared_files(job, receipt)
    if any(not manifests.same_file_identity(path, identities[key]) for key, path in (("checkpoint", checkpoint), ("script", script))):
        raise RuntimeError("SAM worker inputs changed during inference")
    reviews.record_run(job, receipt, "mask-run.json", {"inputs": identities,
        "seconds": time.monotonic() - started, "method": "installed SAM3.1 text-prompted masks on captured photos; no new training"},
        [output / "sam3_manifest.json", *sorted((output / "masks").glob("*/*.npz"))])


if __name__ == "__main__":
    main()
