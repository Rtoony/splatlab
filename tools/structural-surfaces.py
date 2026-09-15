#!/usr/bin/env python3
"""Prepare photo-semantic wall evidence, run installed SAM under the compute gate, or fit tracked surfaces."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import capture_records
import selection_reviews as selections
import structural_surfaces as surfaces


def masks(job, identifier):
    gate = ROOT / "tools/splatlab-compute-gate.sh"
    if subprocess.run([str(gate), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Structural mask inference must run through splatlab-compute-gate.sh --run")
    with surfaces.worker_slot(job, identifier):
        receipt = surfaces.read(job, identifier)
        surfaces.verify(job, receipt, prepared_files=True)
        output = surfaces.directory(job, identifier)
        if (output / "masks").exists() or (output / "masks.json").exists():
            raise RuntimeError("Structural masks already exist; prepare another study")
        checkpoint = Path("/home/rtoony/projects/ml/sam3/checkpoints/sam3.1_multiplex.pt")
        script = ROOT / "backend/mesh/scene_sam3_masks.py"
        identities = {"checkpoint": manifests.file_identity(checkpoint), "script": manifests.file_identity(script)}
        environment = capture_records.worker_env()
        environment.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        started = time.monotonic()
        subprocess.run(["/home/rtoony/miniconda3/envs/sam3/bin/python", str(script), str(output), str(output / "things.json"), ".5"],
                       env=environment, check=True, timeout=210)
        surfaces.verify(job, receipt, prepared_files=True)
        if any(not manifests.same_file_identity(path, identities[name]) for name, path in (("checkpoint", checkpoint), ("script", script))):
            raise RuntimeError("Structural mask model or worker changed during inference")
        return selections.seal(job, output, "masks.json", {"prepared_sha256": receipt["sha256"], "inputs": identities,
            "seconds": time.monotonic() - started, "method": "installed SAM3.1 wall masks on captured photos, not semantic ground truth"},
            [output / "sam3_manifest.json", *sorted((output / "masks/wall").glob("*.npz"))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "masks", "fit"])
    parser.add_argument("job", type=Path)
    parser.add_argument("--structure-id")
    parser.add_argument("--expected-generation", type=int)
    parser.add_argument("--image-id", type=int, action="append", default=[])
    args = parser.parse_args()
    if args.action == "prepare" and args.expected_generation is None or args.action != "prepare" and not args.structure_id:
        parser.error("Preparation needs a generation; masks/fit need a structural study ID")
    job = args.job.resolve()
    result = surfaces.prepare(job, args.expected_generation, args.image_id) if args.action == "prepare" else masks(job, args.structure_id) if args.action == "masks" else surfaces.fit(job, args.structure_id)
    print(json.dumps({key: result[key] for key in ("structure_id", "base", "sha256", "seconds", "status", "accepted_points", "planes", "views") if key in result}, indent=2))


if __name__ == "__main__":
    main()
