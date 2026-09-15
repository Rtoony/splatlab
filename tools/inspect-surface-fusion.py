#!/usr/bin/env python3
"""Replay sealed training-depth samples to diagnose a failed mesh extraction."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import reconstruction_comparison as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-study", type=Path, required=True)
    parser.add_argument("--arm", choices=("splatfacto", "dn-splatter", "ags-mesh"), required=True)
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location("surface_worker", ROOT / "tools/reconstruction-surfaces.py")
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    gate = worker.worker_module()
    gate.require_gate()
    gate.runtime_environment()
    study = args.surface_study.resolve()
    receipt = worker.verify(study)
    source = study / args.arm
    run = comparison.read(source, "run.json")
    if run["surface_receipt_sha256"] != receipt["sha256"]:
        raise ValueError("Samples do not belong to this surface study")
    for relative, identity in run["artifacts"].items():
        filename = source / relative
        if filename.is_symlink() or not filename.resolve().is_relative_to(source) or manifests.sha256_file(filename) != identity["sha256"]:
            raise ValueError("Retained sample artifacts changed")
    records = json.loads((source / "training-cameras.json").read_text())
    if not isinstance(records, list) or len(records) != len(receipt["training_images"]) or {record["image"] for record in records} != set(receipt["training_images"]):
        raise ValueError("Replay requires every registered training camera and no held-out camera")
    import numpy as np
    import open3d as open3d
    from PIL import Image

    destination = source / "fusion-inspection"
    destination.mkdir()
    started = time.monotonic()
    parameters = receipt["parameters"]
    volume = open3d.pipelines.integration.ScalableTSDFVolume(voxel_length=parameters["voxel_length"], sdf_trunc=parameters["sdf_truncation"], color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    accepted = []
    for record in records:
        stem = Path(record["image"]).stem
        with np.load(source / "training-samples" / (stem + ".npz"), allow_pickle=False) as samples:
            depth = samples["depth"]
        rgb = np.asarray(Image.open(source / "training-samples" / (stem + ".png")))
        accepted.append(int((depth > 0).sum()))
        worker.integrate(volume, record, depth, rgb, open3d)
    mesh = volume.extract_triangle_mesh()
    vertices = np.asarray(mesh.vertices)
    payload = comparison.seal(destination, "result.json", {"source_run_sha256": run["sha256"], "surface_receipt_sha256": receipt["sha256"],
        "parameters": parameters, "training_views": len(records), "accepted_pixels_minimum": min(accepted), "accepted_pixels_maximum": max(accepted),
        "vertices": len(vertices), "triangles": len(mesh.triangles), "elapsed_seconds": time.monotonic() - started,
        "minimum": vertices.min(axis=0).tolist() if len(vertices) else None, "maximum": vertices.max(axis=0).tolist() if len(vertices) else None,
        "scope": "diagnostic replay only; not an accepted mesh or changed extraction recipe", "implementation": manifests.file_identity(Path(__file__))})
    print(payload, flush=True)


if __name__ == "__main__":
    main()
