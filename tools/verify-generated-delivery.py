#!/usr/bin/env python3
"""Independently reload generated GLB exports and compare retained fixed-camera renders."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import generated_objects as objects
import generated_color
import reconstruction_comparison as comparison
from mesh.provenance import GenerativeInputRefused, assert_not_generative, glb_is_generative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--generated-object-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location("generated_worker", ROOT / "tools/generated-object.py")
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    worker.require_gate()

    import numpy as np
    from PIL import Image
    import open3d
    import trimesh

    job, identifier = args.job.resolve(), args.generated_object_id
    receipt = objects.read(job, identifier)
    result = objects.read(job, identifier, True)
    for record, completed in ((receipt, False), (result, True)):
        for name in record["artifacts"]:
            objects.artifact(job, identifier, name, completed)
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "generated_object_id": identifier, "prepared_receipt_sha256": receipt["sha256"],
              "result_sha256": result["sha256"], "scope": "exported GLBs versus retained in-memory raycasts; not geometry truth, lighting, Gaussian placement or navigation acceptance",
              "tolerances": {"changed_hit_pixels": 0, "maximum_depth_error_m": 1e-5, "maximum_rgb_error_byte": 1}, "views": []}
    started = time.monotonic()
    try:
        meshes = []
        for name in ("placed-master.glb", "delivery.glb"):
            path = objects.artifact(job, identifier, name, True)
            if not glb_is_generative(path):
                raise ValueError("Export lost portable generative provenance")
            try:
                assert_not_generative(path, "generated mesh survey refusal check")
            except GenerativeInputRefused:
                pass
            else:
                raise ValueError("Survey lane accepted an inferred export")
            mesh = trimesh.load(path, file_type="glb", force="scene", process=False).to_geometry()
            color = generated_color.decoded_rgba(path)
            if color is not None:
                if len(color) != len(mesh.vertices):
                    raise ValueError("Generated color accessor lost mesh vertex membership")
                mesh.visual.vertex_colors = color
            if mesh.visual.kind != "vertex":
                raise ValueError("Export lost vertex colors")
            meshes.append(mesh)
        for camera in receipt["cameras"]:
            for kind in ("photo", "mask"):
                target = destination / camera[kind]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(objects.artifact(job, identifier, camera[kind]), target)
        worker.render_comparison(*meshes, receipt, destination)
        for camera in receipt["cameras"]:
            image_id = camera["image_id"]
            source_depth = objects.artifact(job, identifier, f"review/depth-{image_id}.npz", True)
            with np.load(source_depth, allow_pickle=False) as original, np.load(destination / f"review/depth-{image_id}.npz", allow_pickle=False) as exported:
                for label in ("master", "delivery"):
                    reference_depth, actual_depth = original[label], exported[label]
                    reference_hits, actual_hits = np.isfinite(reference_depth), np.isfinite(actual_depth)
                    overlap = reference_hits & actual_hits
                    if not overlap.any():
                        raise ValueError("Cannot verify an empty view")
                    with Image.open(objects.artifact(job, identifier, f"review/{label}-{image_id}.png", True)) as opened:
                        reference = np.asarray(opened).astype(float)
                    with Image.open(destination / f"review/{label}-{image_id}.png") as opened:
                        difference = np.abs(np.asarray(opened).astype(float) - reference)
                    reading = {"image_id": image_id, "split": camera["split"], "asset": label,
                               "changed_hit_pixels": int(np.count_nonzero(reference_hits ^ actual_hits)),
                               "maximum_depth_error_m": float(np.max(np.abs(reference_depth[overlap] - actual_depth[overlap]))),
                               "maximum_rgb_error_byte": float(difference.max()), "mean_rgb_error_byte": float(difference.mean())}
                    report["views"].append(reading)
                    if any(reading[key] > tolerance for key, tolerance in report["tolerances"].items()):
                        raise ValueError("Export differs beyond the declared fixed-camera tolerance")
        for completed, expected in ((False, receipt), (True, result)):
            if objects.read(job, identifier, completed)["sha256"] != expected["sha256"]:
                raise ValueError("Generated-object receipt changed during verification")
            for name in expected["artifacts"]:
                objects.artifact(job, identifier, name, completed)
        report.update(status="passed", master_triangles=len(meshes[0].faces), delivery_triangles=len(meshes[1].faces),
                      portable_provenance="present; both exports refused by survey lane")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.update(seconds=time.monotonic() - started, implementation=manifests.file_identity(Path(__file__)),
                      renderer_implementation=manifests.file_identity(ROOT / "tools/generated-object.py"),
                      open3d_version=open3d.__version__, trimesh_version=trimesh.__version__)
        comparison.seal(destination, "result.json", report)
    print({key: report[key] for key in ("status", "generated_object_id", "seconds")}, flush=True)


if __name__ == "__main__":
    main()
