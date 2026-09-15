#!/usr/bin/env python3
"""Export all compared Gaussians without changing checkpoint row membership."""

import argparse
import importlib.util
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import reconstruction_comparison as comparison
import reconstruction_review as review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location("comparison_worker", ROOT / "tools/reconstruction-comparison.py")
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    worker.require_gate()
    worker.runtime_environment()
    study = args.comparison.resolve()
    result = comparison.summarize(study, args.runs)
    import numpy as np
    import torch

    torch.set_num_threads(4)
    destination = study / ("review-" + uuid.uuid4().hex[:24])
    destination.mkdir()
    runs = {name: comparison.read(study / name, "run.json") for name in args.runs}
    statistics = {}
    for name, run in runs.items():
        checkpoints = [relative for relative in run["artifacts"] if relative.endswith(".ckpt")]
        if len(checkpoints) != 1:
            raise ValueError("Review needs exactly one registered final checkpoint per arm")
        checkpoint = study / name / checkpoints[0]
        if checkpoint.stat().st_size > 2 * 1024 ** 3:
            raise ValueError("Checkpoint exceeds bounded review size")
        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["step"] + 1 != run["iterations"]:
            raise ValueError("Checkpoint does not contain the completed training step")
        pipeline = state["pipeline"]
        prefixes = [key.removesuffix("means") for key in pipeline if key.endswith("gauss_params.means")]
        if len(prefixes) != 1:
            raise ValueError("Checkpoint Gaussian parameter ownership is ambiguous")
        parameters = {key[len(prefixes[0]):]: value.detach().cpu().numpy() for key, value in pipeline.items() if key.startswith(prefixes[0])}
        if len(parameters["means"]) != run["gaussians"]:
            raise ValueError("Checkpoint row count differs from the retained evaluation")
        output = destination / name
        output.mkdir()
        geometry = review.export_splats(output / "model.ply", parameters)
        np.save(output / "checkpoint-rows.npy", np.arange(run["gaussians"], dtype=np.int64), allow_pickle=False)
        depth = {relative: review.depth_statistics(study / name / relative) for relative in run["artifacts"] if relative.endswith("-depth.npy")}
        if len(depth) != len(run["evaluation"]["views"]):
            raise ValueError("Retained depth does not cover every held-out view")
        statistics[name] = {"checkpoint": checkpoints[0], "checkpoint_identity": run["artifacts"][checkpoints[0]],
                            "geometry": geometry, "held_out_depth": depth}
        del parameters, pipeline, state
    (destination / "index.html").write_text(review.review_html(result, runs))
    if comparison.summarize(study, args.runs)["sha256"] != result["sha256"]:
        raise ValueError("Comparison changed during review export")
    artifacts = {str(path.relative_to(destination)): manifests.file_identity(path) for path in sorted(destination.rglob("*")) if path.is_file()}
    payload = comparison.seal(destination, "review.json", {"schema": "dev.splatlab.reconstruction-review/v1",
        "comparison_sha256": result["sha256"], "run_sha256": result["run_sha256"], "created_at": manifests.utc_now(),
        "status": "needs-review", "promoted": False, "methods": statistics, "artifacts": artifacts,
        "implementation_sources": {str(path): manifests.file_identity(path) for path in (Path(__file__), Path(review.__file__))},
        "torch_version": torch.__version__, "source_checkpoint_pickle_policy": "weights_only=True with explicit numpy scalar allowlist; no unrestricted fallback"})
    print(f"Review: {destination / 'index.html'}\nReceipt SHA-256: {payload['sha256']}", flush=True)


if __name__ == "__main__":
    main()
