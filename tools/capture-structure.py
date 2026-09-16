#!/usr/bin/env python3
"""Prepare source-track structure evidence or run one bounded local SAM3 mask pass."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests
import capture_structure as structure
from mesh.slugify import slug


def require_gate(lease_path):
    gate = ROOT / "tools/splatlab-compute-gate.sh"
    if subprocess.run([str(gate), "--is-contained"], capture_output=True).returncode:
        raise ValueError("Use tools/splatlab-compute-gate.sh --run")
    lease = json.loads(lease_path.read_text())
    if datetime.now(timezone.utc) >= datetime.fromisoformat(lease["no_new_starts_after"]):
        raise ValueError("No new starts in the expiry buffer")


def masks(output, lease_path):
    require_gate(lease_path)
    output = output.resolve()
    receipt = structure.verify_study(output)
    if (output / "masks").exists() or (output / "masks.json").exists():
        raise ValueError("Mask attempt already exists; preserve it and prepare a new study")
    script = ROOT / "backend/mesh/scene_sam3_masks.py"
    checkpoint = Path("/home/rtoony/projects/ml/sam3/checkpoints/sam3.1_multiplex.pt")
    bpe = checkpoint.parents[1] / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    sources = {str(path): manifests.sha256_file(path) for path in (script, checkpoint, bpe, output / "receipt.json")}
    result = {"status": "running", "source_hashes": sources, "started_at": manifests.utc_now(), "model": "installed local SAM3.1",
              "semantic_ground_truth": False, "prompts": receipt["prompts"]}
    started = time.monotonic()
    manifests.atomic_write_json(output / "masks.json", result)
    try:
        environment = os.environ.copy()
        environment.update(PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="1", PYTHONPATH="/home/rtoony/projects/ml/sam3")
        with (output / "sam3.log").open("x") as log:
            subprocess.run(["/home/rtoony/miniconda3/envs/sam3/bin/python", str(script), str(output), str(output / "things.json"), ".5"],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=480, check=True)
        structure.verify_study(output)
        for path, digest in sources.items():
            if manifests.sha256_file(Path(path)) != digest:
                raise ValueError("SAM model, worker or structural receipt changed")
        expected = [output / "sam3_manifest.json"]
        for prompt in receipt["prompts"]:
            for view in receipt["views"]:
                path = output / "masks" / slug(prompt) / f"cam_{view['ordinal']:03d}.npz"
                structure.semantic_union(path, prompt, (receipt["intrinsics"]["h"], receipt["intrinsics"]["w"]))
                expected.append(path)
        result.update(status="inferred-needs-visual-review", files={str(path.relative_to(output)): manifests.sha256_file(path) for path in expected})
    except BaseException as error:
        result.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        result.update(finished_at=manifests.utc_now(), elapsed_seconds=time.monotonic() - started)
        manifests.atomic_write_json(output / "masks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "masks", "analyze"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--sfm", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--prompts", default=None, help="comma-separated SAM3 prompts for prepare (default: capture_structure.PROMPTS)")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.reference is None or args.sfm is None:
            parser.error("Preparation needs --reference and --sfm")
        if args.prompts:
            structure.PROMPTS = tuple(p.strip() for p in args.prompts.split(",") if p.strip())
        result = structure.prepare(args.reference, args.sfm, args.output)
    else:
        if args.lease is None:
            parser.error("Inference needs the current bounded --lease")
        if args.action == "masks":
            result = masks(args.output, args.lease)
        else:
            require_gate(args.lease)
            result = structure.analyze(args.output)
    print(json.dumps({key: result[key] for key in ("status", "sparse_points", "source_timestamps", "elapsed_seconds") if key in result}), flush=True)


if __name__ == "__main__":
    main()
