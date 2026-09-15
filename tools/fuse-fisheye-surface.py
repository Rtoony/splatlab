#!/usr/bin/env python3
"""Fuse only saved training-view depths into a separate, inferred Blender mesh candidate."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests
from reference_delivery import artifact_path, read_sparse_points
from reconstruction_surfaces import bounds_for_seeds, filtered_depth, rays, write_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=.1)
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--mask-mode", choices=("none", "static"), default="none")
    args = parser.parse_args()
    if subprocess.run([str(ROOT / "tools/splatlab-compute-gate.sh"), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Use the SplatLab compute gate")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(json.loads(args.lease.read_text())["no_new_starts_after"]):
        raise RuntimeError("No new starts in the expiry buffer")
    evaluation, dataset, output = args.evaluation.resolve(), args.dataset.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(evaluation) or output.is_relative_to(dataset):
        raise ValueError("Choose a new surface directory outside its sources")
    receipt_path = evaluation / "receipt.json"
    source = json.loads(receipt_path.read_text())
    if source["status"] != "evaluated-needs-review" or source["registration"] is not None:
        raise ValueError("Expected completed independent evaluation")
    transforms_path = dataset / "transforms.json"
    seed_path = dataset / "sparse.ply"
    snapshot = {receipt_path: manifests.sha256_file(receipt_path)}
    for path in (transforms_path, seed_path, dataset / "receipt.json"):
        snapshot[path] = source["source_hashes"][str(path)]
    for record in source["training_depth"]:
        path = artifact_path(evaluation, record["file"])
        snapshot[path] = source["files"][record["file"]]

    def verify():
        if any(manifests.sha256_file(path) != digest for path, digest in snapshot.items()):
            raise ValueError("Source-bound training depth or camera inputs changed")

    verify()
    transforms = json.loads(transforms_path.read_text())
    if sorted(record["image"] for record in source["training_depth"]) != sorted(transforms["train_filenames"]):
        raise ValueError("Fusion must use all and only training images, never held-out views")
    import numpy as np
    import open3d as open3d

    selected = source["training_depth"]
    semantic_paths = {}
    if args.structure:
        structure = args.structure.resolve()
        if output.is_relative_to(structure):
            raise ValueError("Surface output must be outside the structural evidence")
        study = json.loads((structure / "receipt.json").read_text())
        analysis = json.loads((structure / "analysis.json").read_text())
        dataset_receipt = json.loads((dataset / "receipt.json").read_text())
        reference_receipts = [digest for name, digest in study["source_hashes"].items() if Path(name).name == "receipt.json"]
        if (study.get("schema") != "dev.splatlab.capture-structure/v1" or analysis.get("status") != "analyzed-needs-visual-review"
                or reference_receipts != [dataset_receipt["source_reference_sha256"]]
                or analysis["source_receipt_sha256"] != manifests.sha256_file(structure / "receipt.json")):
            raise ValueError("Semantic study belongs to a different frozen capture")
        for filename in ("receipt.json", "analysis.json"):
            snapshot[structure / filename] = manifests.sha256_file(structure / filename)
        by_name = {record["image"]: record for record in selected}
        source_frames = {frame["file_path"]: frame for frame in transforms["frames"]}
        selected = []
        for view in study["views"]:
            if view["split"] != "train" or view["virtual_yaw_deg"] != 0 or view["file_path"] not in by_name:
                raise ValueError("Structural fusion selection is not a training center view")
            if not np.allclose(source_frames[view["file_path"]]["transform_matrix"], view["transform_matrix"], atol=1e-7, rtol=0):
                raise ValueError("Structural source camera changed")
            selected.append(by_name[view["file_path"]])
            name = f"classified-masks/cam_{view['ordinal']:03d}.npz"
            path = artifact_path(structure, name)
            snapshot[path] = analysis["files"][name]
            semantic_paths[view["file_path"]] = path
        if len({record["image"] for record in selected}) != len(selected) or not selected:
            raise ValueError("Structural selection is empty or duplicated")
        verify()
    elif args.mask_mode != "none":
        raise ValueError("Static masks require a source-bound structural study")

    seeds, _colors = read_sparse_points(seed_path)
    voxel = args.voxel
    if not np.isfinite(voxel) or not .05 <= voxel <= .5:
        raise ValueError("Choose a bounded voxel between .05 and .5 arbitrary units")
    bounds = bounds_for_seeds(seeds, voxel)
    parameters = {"voxel_length": voxel, "sdf_truncation": voxel * 4, "alpha_minimum": .7,
                  "depth_truncation": 120., "units": "arbitrary training units, not metres"}
    output.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"schema": "dev.splatlab.fisheye-surface/v1", "status": "running", "source_hashes": {str(path): digest for path, digest in snapshot.items()},
               "source": "TSDF of model-expected training depth; not measured surface truth", "parameters": parameters, "bounds": bounds,
               "registration": None, "metric_scale": "unknown", "promoted": False, "heldout_fused": False,
               "mask_mode": args.mask_mode, "selected_training_images": [record["image"] for record in selected],
               "selection_scope": "central training views selected by the semantic study" if args.structure else "all training views",
               "postprocessing": "none; no hole filling, smoothing, simplification or collider promotion"}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        volume = open3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel, sdf_trunc=voxel * 4,
                    color_type=open3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        records = []
        for index, record in enumerate(selected):
            with np.load(evaluation / record["file"], allow_pickle=False) as maps:
                selected_mask = maps["mask"].copy()
                if args.mask_mode == "static":
                    with np.load(semantic_paths[record["image"]], allow_pickle=False) as semantic:
                        mask = semantic["static_surface"]
                        if mask.dtype != bool or mask.shape != selected_mask.shape:
                            raise ValueError("Static semantic mask has different dimensions or type")
                        selected_mask &= mask
                accepted_depth, accepted = filtered_depth(maps["depth"], maps["alpha"], selected_mask, rays(record), bounds, parameters)
                rgb = np.ascontiguousarray(maps["rgb"], dtype=np.uint8)
            pose = np.eye(4)
            pose[:3] = record["camera_to_world_opengl"]
            pose = pose @ np.diag([1., -1., -1., 1.])
            intrinsic = open3d.camera.PinholeCameraIntrinsic(record["width"], record["height"], record["fx"], record["fy"], record["cx"] - .5, record["cy"] - .5)
            rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(open3d.geometry.Image(rgb), open3d.geometry.Image(accepted_depth),
                         depth_scale=1., depth_trunc=parameters["depth_truncation"], convert_rgb_to_intensity=False)
            if accepted.any():
                volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            records.append({"image": record["image"], "supported_pixels": int(accepted.sum()), "pixels": accepted.size})
            if index % 12 == 0 or index + 1 == len(selected):
                print(f"Fused {index + 1}/{len(selected)} training views", flush=True)
        mesh = volume.extract_triangle_mesh()
        vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        receipt.update(extracted_vertices=len(vertices), extracted_triangles=len(faces))
        print(f"Extracted {len(vertices)} vertices and {len(faces)} triangles", flush=True)
        if not 1 <= len(vertices) <= 2_000_000 or not 1 <= len(faces) <= 3_000_000:
            raise ValueError("Surface is empty or exceeds the bounded review size")
        write_mesh(output / "surface.ply", vertices, faces, colors)
        np.savez_compressed(output / "surface-arrays.npz", vertices=vertices, faces=faces, colors=colors)
        _labels, counts, _areas = mesh.cluster_connected_triangles()
        receipt.update(status="inferred-surface-needs-review", vertices=len(vertices), triangles=len(faces), components=len(counts),
                       largest_component_triangle_fraction=float(max(counts) / len(faces)), training_views=records,
                       boundary_or_nonmanifold_edges=len(mesh.get_non_manifold_edges(allow_boundary_edges=False)))
        verify()
        receipt["files"] = {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.iterdir()
                            if path.is_file() and path.name != "receipt.json"}
    except BaseException as error:
        receipt.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        receipt.update(finished_at=manifests.utc_now(), elapsed_seconds=time.monotonic() - started)
        manifests.atomic_write_json(output / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "vertices", "triangles", "components", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
