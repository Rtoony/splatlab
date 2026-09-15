#!/usr/bin/env python3
"""Verify that exported inferred meshes preserve fixed-camera geometry and color."""

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
import reconstruction_surfaces as surfaces
from mesh.provenance import GenerativeInputRefused, assert_not_generative, ply_is_generative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-study", type=Path, required=True)
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location("surface_worker", ROOT / "tools/reconstruction-surfaces.py")
    worker = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(worker)
    gate = worker.worker_module()
    gate.require_gate()
    gate.runtime_environment()
    study = args.surface_study.resolve()
    worker.verify(study)
    summary, runs = surfaces.summarize(study)
    destination = study / "delivery-check"
    destination.mkdir()
    started = time.monotonic()
    result = {"status": "running", "comparison_sha256": summary["sha256"], "scope": "exported PLY versus retained in-memory mesh raycasts; not Spark splat-viewer parity, geometry truth or navigation acceptance", "methods": {}}
    import numpy as np
    import open3d as open3d
    from PIL import Image

    try:
        for arm, run in runs.items():
            source = study / arm
            filename = source / "surface.ply"
            if not ply_is_generative(filename):
                raise ValueError("Export lost its portable inferred-geometry tag")
            try:
                assert_not_generative(filename, "survey")
            except GenerativeInputRefused:
                pass
            else:
                raise ValueError("Survey guard accepted inferred surface geometry")
            mesh = open3d.io.read_triangle_mesh(str(filename), enable_post_processing=False)
            if len(mesh.triangles) != run["mesh"]["triangles"] or len(mesh.vertices) != run["mesh"]["vertices"] or not mesh.has_vertex_colors():
                raise ValueError("PLY reader changed mesh counts or lost its appearance")
            scene = open3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh))
            cameras = json.loads((source / "evaluation-cameras.json").read_text())
            if surfaces.camera_digest(cameras) != run["evaluation_camera_sha256"]:
                raise ValueError("Evaluation camera records changed")
            readings = []
            for camera in cameras:
                stem = Path(camera["image"]).stem
                depth, rgb, _normals, _rays = worker.cast_mesh(scene, mesh, camera, open3d)
                with np.load(source / "renders" / (stem + ".npz"), allow_pickle=False) as maps:
                    reference_depth = maps["mesh_depth"]
                hits, reference_hits = np.isfinite(depth), np.isfinite(reference_depth)
                if not np.array_equal(hits, reference_hits):
                    raise ValueError("Export changed fixed-camera surface-hit membership")
                depth_error = float(np.max(np.abs(depth[hits] - reference_depth[hits]))) if hits.any() else 0.
                reference_rgb = np.asarray(Image.open(source / "renders" / (stem + ".png")))
                current_rgb = np.rint(rgb.clip(0, 1) * 255).astype(np.uint8)
                difference = np.abs(current_rgb.astype(float) - reference_rgb)
                maximum_rgb_error = float(difference.max())
                if depth_error > 1e-5 or maximum_rgb_error > 1:
                    raise ValueError("PLY geometry/color differs beyond the declared double-position/RGB8 export tolerance")
                readings.append({"image": camera["image"], "maximum_depth_error_training_units": depth_error,
                                 "maximum_rgb_byte_difference": maximum_rgb_error, "mean_rgb_byte_difference": float(difference.mean())})
            result["methods"][arm] = {"vertices": len(mesh.vertices), "triangles": len(mesh.triangles), "views": readings,
                                     "portable_provenance": "present; survey lane refuses", "mesh_identity": manifests.file_identity(filename)}
        if surfaces.summarize(study)[0]["sha256"] != summary["sha256"]:
            raise ValueError("Surface comparison changed during delivery verification")
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(total_seconds=time.monotonic() - started, implementation=manifests.file_identity(Path(__file__)), open3d_version=open3d.__version__)
        comparison.seal(destination, "result.json", result)
    print({"status": result["status"], "methods": len(result["methods"]), "views_checked": sum(len(method["views"]) for method in result["methods"].values()), "total_seconds": result["total_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
