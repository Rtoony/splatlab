#!/usr/bin/env python3
"""Compare source-bound control and semantic-filtered meshes using identical cameras and masks."""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests
from reconstruction_surfaces import depth_agreement, filtered_depth, rays
from reference_delivery import artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("control", "filtered", "evaluation", "structure", "output", "lease"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--candidate-evaluation", type=Path)
    args = parser.parse_args()
    if subprocess.run([str(ROOT / "tools/splatlab-compute-gate.sh"), "--is-contained"], capture_output=True).returncode:
        raise ValueError("Use the shared SplatLab compute gate")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(json.loads(args.lease.read_text())["no_new_starts_after"]):
        raise ValueError("No new starts in the expiry buffer")
    output = args.output.resolve()
    directories = {"control": args.control.resolve(), "filtered": args.filtered.resolve()}
    if args.candidate_evaluation and output.is_relative_to(args.candidate_evaluation.resolve()):
        raise ValueError("Comparison must remain outside the candidate evaluation")
    if output.exists() or any(output.is_relative_to(path.resolve()) for path in (*directories.values(), args.evaluation, args.structure)):
        raise ValueError("Choose a new comparison directory outside its source inputs")
    receipts, snapshots = {}, {}
    for label, directory in directories.items():
        receipt_path = directory / "receipt.json"
        receipts[label] = json.loads(receipt_path.read_text())
        if receipts[label]["status"] != "inferred-surface-needs-review":
            raise ValueError("Require two completed inferred surfaces")
        snapshots[receipt_path] = manifests.sha256_file(receipt_path)
        for name, digest in receipts[label]["files"].items():
            snapshots[artifact_path(directory, name)] = digest
    for field in ("parameters", "bounds", "selected_training_images", "selection_scope"):
        if receipts["control"][field] != receipts["filtered"][field]:
            raise ValueError("Control and filtered mesh have different extraction settings or source cameras")
    shared_sources = receipts["control"]["source_hashes"]
    if not args.candidate_evaluation and receipts["filtered"]["source_hashes"] != shared_sources:
        raise ValueError("Control and filtered mesh have different frozen source evidence")
    expected_modes = ("none", "none") if args.candidate_evaluation else ("none", "static")
    if (receipts["control"]["mask_mode"], receipts["filtered"]["mask_mode"]) != expected_modes:
        raise ValueError("Require an unfiltered control and static-mask candidate")
    snapshots.update({Path(path): digest for path, digest in shared_sources.items()})
    if args.candidate_evaluation:
        for name, digest in receipts["filtered"]["source_hashes"].items():
            path = Path(name)
            if path in snapshots and snapshots[path] != digest:
                raise ValueError("A shared method-comparison input changed")
            snapshots[path] = digest

    def verify():
        if any(manifests.sha256_file(path) != digest for path, digest in snapshots.items()):
            raise ValueError("Source artifact changed during controlled comparison")

    verify()
    evaluation = json.loads((args.evaluation / "receipt.json").read_text())
    structure = json.loads((args.structure / "receipt.json").read_text())
    analysis = json.loads((args.structure / "analysis.json").read_text())
    if manifests.sha256_file(args.evaluation / "receipt.json") != shared_sources[str((args.evaluation / "receipt.json").resolve())]:
        raise ValueError("Evaluation differs from the extracted model")
    views = {record["image"]: record for record in evaluation["training_depth"]}
    semantic_views = {view["file_path"]: view for view in structure["views"]}
    import numpy as np
    import open3d as open3d
    from PIL import Image

    if args.candidate_evaluation:
        candidate_path = args.candidate_evaluation.resolve() / "receipt.json"
        candidate = json.loads(candidate_path.read_text())
        if (candidate.get("status") != "evaluated-needs-review"
                or manifests.sha256_file(candidate_path) != receipts["filtered"]["source_hashes"].get(str(candidate_path))):
            raise ValueError("Candidate mesh does not belong to this evaluated method")
        required = {name: digest for name, digest in evaluation["source_hashes"].items()
                    if Path(name).name in {"transforms.json", "sparse.ply"}}
        if len(required) != 2 or any(candidate["source_hashes"].get(name) != digest for name, digest in required.items()):
            raise ValueError("Methods do not share the frozen camera and sparse initialization inputs")
        candidate_views = {record["image"]: record for record in candidate["training_depth"]}
        if set(candidate_views) != set(views):
            raise ValueError("Methods do not use identical training photos")
        for name in views:
            for field in ("width", "height", "fx", "fy", "cx", "cy", "camera_to_world_opengl"):
                if not np.allclose(views[name][field], candidate_views[name][field], atol=1e-5, rtol=0):
                    raise ValueError("The candidate changed a comparison camera")

    def method_label(evaluated):
        if evaluated.get("method") != "gsplat-2dgs":
            name = "Splatfacto surface mesh"
        else:
            name = "Source-depth-anchored 2DGS mesh" if evaluated.get("source_depth_anchors") else "Unanchored 2DGS mesh"
        return name + f" ({evaluated['training_iterations']:,} steps)"

    with np.load(args.structure / "membership.npz", allow_pickle=False) as membership:
        points = membership["points"]
        observed = membership["observed"]
        supported_points = membership["facade_supported"] | membership["pavement_supported"]
    snapshots[args.structure / "membership.npz"] = analysis["files"]["membership.npz"]
    verify()
    output.mkdir(parents=True)
    (output / "images").mkdir()
    started = time.monotonic()
    result = {"schema": "dev.splatlab.capture-surface-comparison/v1", "status": "running", "methods": {},
              "source_hashes": {str(path): digest for path, digest in snapshots.items()},
              "scope": "Controlled semantic-filtering experiment; source-splat depth and sparse-track consistency, NOT independent geometric accuracy",
              "heldout_appearance_used": False, "registration": None, "owner_accepted": False,
              "semantic_review": "Codex inspected all source-mask contacts and corrected omitted garage doors; predictions are not ground truth"}
    if args.candidate_evaluation:
        result["scope"] = "Geometry-method pilot with shared inputs/extraction; NOT equal optimizer or training budget, and NOT independent geometric accuracy"
        result["depth_comparison_reference"] = "Fixed control model depth for both methods; diagnostic consistency only, not ground truth"
        result["method_labels"] = {"control": method_label(evaluation), "filtered": method_label(candidate)}
    manifests.atomic_write_json(output / "comparison.json", result)
    try:
        for label, directory in directories.items():
            mesh = open3d.io.read_triangle_mesh(str(directory / "surface.ply"), enable_post_processing=False)
            if len(mesh.triangles) != receipts[label]["triangles"] or len(mesh.vertices) != receipts[label]["vertices"]:
                raise ValueError("Independent mesh reader changed exported counts")
            scene = open3d.t.geometry.RaycastingScene(nthreads=4)
            scene.add_triangles(open3d.t.geometry.TriangleMesh.from_legacy(mesh))
            vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
            records = []
            for index, name in enumerate(receipts[label]["selected_training_images"]):
                camera = views[name]
                ordinal = semantic_views[name]["ordinal"]
                with np.load(args.evaluation / camera["file"], allow_pickle=False) as maps:
                    with np.load(args.structure / f"classified-masks/cam_{ordinal:03d}.npz", allow_pickle=False) as semantic:
                        selected_mask = maps["mask"] & semantic["static_surface"]
                    model_depth, supported = filtered_depth(maps["depth"], maps["alpha"], selected_mask,
                                                           rays(camera), receipts[label]["bounds"], receipts[label]["parameters"])
                    reference_rgb = maps["rgb"]
                hits = scene.cast_rays(open3d.core.Tensor(rays(camera)), nthreads=4)
                mesh_depth = hits["t_hit"].numpy()
                visible = np.isfinite(mesh_depth)
                rgb = np.full((*mesh_depth.shape, 3), .10, dtype=np.float32)
                uv = hits["primitive_uvs"].numpy()[visible]
                weights = np.column_stack((1 - uv.sum(axis=1), uv))
                face_rows = faces[hits["primitive_ids"].numpy()[visible]]
                rgb[visible] = (colors[face_rows] * weights[..., None]).sum(axis=1)
                Image.fromarray(np.rint(rgb * 255).clip(0, 255).astype(np.uint8)).save(output / "images" / f"{index:03d}-{label}.png")
                if label == "control":
                    Image.fromarray(reference_rgb).save(output / "images" / f"{index:03d}-reference.png")
                    Image.fromarray(selected_mask.astype(np.uint8) * 255).save(output / "images" / f"{index:03d}-mask.png")
                score = depth_agreement(mesh_depth, model_depth, supported, receipts[label]["parameters"]["voxel_length"])
                candidate_indices = np.flatnonzero(observed[ordinal] & supported_points)
                transform = np.asarray(camera["camera_to_world_opengl"])
                offsets = points[candidate_indices] - transform[:3, 3]
                expected_depth = -(offsets @ transform[:3, :3])[:, 2]
                directions = offsets / expected_depth[:, None]
                positions = np.column_stack(((offsets @ transform[:3, :3])[:, 0], -(offsets @ transform[:3, :3])[:, 1]))
                point_pixels = np.floor(positions / expected_depth[:, None] * [camera["fx"], camera["fy"]] + [camera["cx"], camera["cy"]]).astype(int)
                in_image = (point_pixels >= 0).all(axis=1) & (point_pixels < [camera["width"], camera["height"]]).all(axis=1) & (expected_depth > 0)
                select = np.flatnonzero(in_image)
                select = select[selected_mask[point_pixels[select, 1], point_pixels[select, 0]]]
                track_rays = np.column_stack((np.broadcast_to(transform[:3, 3], (len(select), 3)), directions[select])).astype(np.float32)
                track_hits = scene.cast_rays(open3d.core.Tensor(track_rays), nthreads=4)["t_hit"].numpy() if len(select) else np.empty(0)
                finite_tracks = np.isfinite(track_hits)
                track_errors = np.abs(track_hits[finite_tracks] - expected_depth[select][finite_tracks])
                with np.load(args.structure / "membership.npz", allow_pickle=False) as membership:
                    reserved = membership["plane_check_reserved"][candidate_indices[select]]
                reserved_hits = reserved & finite_tracks
                reserved_errors = np.abs(track_hits[reserved_hits] - expected_depth[select][reserved_hits])
                records.append({"image": name, **score, "sparse_track_samples": len(select), "sparse_track_hits": int(finite_tracks.sum()),
                                "reserved_track_samples": int(reserved.sum()), "reserved_track_hits": int(reserved_hits.sum()),
                                "reserved_track_within_two_voxels": int((reserved_errors <= 2 * receipts[label]["parameters"]["voxel_length"]).sum()),
                                "reserved_track_absolute_error_sum": float(reserved_errors.sum()),
                                "semantic_pixels": int(selected_mask.sum()), "semantic_mesh_hits": int((selected_mask & visible).sum()),
                                "sparse_track_within_two_voxels": int((track_errors <= 2 * receipts[label]["parameters"]["voxel_length"]).sum()),
                                "sparse_track_mean_error_arbitrary_units": float(track_errors.mean()) if len(track_errors) else None,
                                "sparse_track_median_error_arbitrary_units": float(np.median(track_errors)) if len(track_errors) else None})
                if index % 12 == 0 or index + 1 == len(receipts[label]["selected_training_images"]):
                    print(f"{label}: compared {index + 1}/{len(receipts[label]['selected_training_images'])} cameras", flush=True)
            support_count = sum(record["supported_model_pixels"] for record in records)
            result["methods"][label] = {"vertices": len(vertices), "triangles": len(faces), "components": receipts[label]["components"],
                "largest_component_triangle_fraction": receipts[label]["largest_component_triangle_fraction"],
                "supported_pixels": support_count,
                "mesh_hit_fraction_on_static_support": sum(record["mesh_hit_fraction_on_supported_model"] * record["supported_model_pixels"] for record in records) / support_count,
                "within_two_voxels_fraction_on_static_support": sum(record["within_two_voxels_fraction_of_supported_model"] * record["supported_model_pixels"] for record in records) / support_count,
                "sparse_track_samples": sum(record["sparse_track_samples"] for record in records),
                "reserved_track_samples": sum(record["reserved_track_samples"] for record in records),
                "reserved_track_hits": sum(record["reserved_track_hits"] for record in records),
                "reserved_track_within_two_voxels": sum(record["reserved_track_within_two_voxels"] for record in records),
                "reserved_track_mean_error_arbitrary_units": sum(record["reserved_track_absolute_error_sum"] for record in records) / max(1, sum(record["reserved_track_hits"] for record in records)),
                "semantic_pixels": sum(record["semantic_pixels"] for record in records),
                "mesh_hit_fraction_on_semantic_pixels": sum(record["semantic_mesh_hits"] for record in records) / sum(record["semantic_pixels"] for record in records),
                "sparse_track_within_two_voxels": sum(record["sparse_track_within_two_voxels"] for record in records),
                "sparse_track_mean_error_arbitrary_units": sum((record["sparse_track_mean_error_arbitrary_units"] or 0) * record["sparse_track_hits"] for record in records) / max(1, sum(record["sparse_track_hits"] for record in records)),
                "sparse_track_hits": sum(record["sparse_track_hits"] for record in records), "views": records}
            del scene, mesh, vertices, faces, colors
        if result["methods"]["control"]["supported_pixels"] != result["methods"]["filtered"]["supported_pixels"]:
            raise ValueError("Mesh comparison does not use exactly the same static support")
        verify()
        result["status"] = "compared-needs-review"
        sections = []
        for index, view in enumerate(result["methods"]["control"]["views"]):
            labels = (("reference", "Real source photo"), ("control", result["method_labels"]["control"] if args.candidate_evaluation else "Same-camera unfiltered mesh"),
                      ("filtered", result["method_labels"]["filtered"] if args.candidate_evaluation else "Semantic-filtered mesh"), ("mask", "Shared evaluated static region"))
            figures = "".join(f'<figure><figcaption>{title}</figcaption><img loading="lazy" src="images/{index:03d}-{kind}.png" alt="{title}"></figure>'
                              for kind, title in labels)
            sections.append(f'<section><h2>{escape(view["image"])}</h2><div class="views">{figures}</div></section>')
        (output / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Condo structural extraction comparison</title><style>body{font:16px system-ui;background:#101923;color:#e6eff8;margin:24px}h1{font-size:2rem}h2{font-size:1rem;overflow-wrap:anywhere}p{line-height:1.6;max-width:1100px}a{color:#8bd3f5}.views{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%}section{margin:28px 0}figcaption{padding:8px 0}@media(max-width:800px){.views{grid-template-columns:1fr 1fr}}</style>
<h1>Separating captured structure from sky and moving clutter</h1><p>Both meshes use the exact same 24 central training cameras, trained model, voxel and fusion settings. Only semantic filtering differs. Dark areas show absent geometry, not filled walls. Windows, sky, vegetation and vehicles are deliberately excluded. This remains an inferred, unregistered diagnostic—not survey truth or accepted architecture.</p><p><a href="comparison.json">Full metrics and provenance</a></p>''' + "".join(sections) + '</html>')
        if args.candidate_evaluation:
            page = (output / "index.html").read_text()
            beginning = page.index("<h1>")
            ending = page.index('<p><a href="comparison.json">', beginning)
            title = escape(result["method_labels"]["control"] + " versus " + result["method_labels"]["filtered"])
            page = page[:beginning] + f'<h1>{title}</h1><p>Actual source-aligned meshes. Identical frozen training photos, cameras, central-view selection and fusion settings; method settings and training budgets remain explicit in the source receipts. Both meshes are unfiltered. Static semantic regions and source tracks are shared evaluation evidence, not survey ground truth. Reserved track checks were not used by the source-depth loss, but were present in the original SfM initialization. Depth consistency uses the control model as a fixed diagnostic reference, not proof of correct geometry. No architectural acceptance or metric registration is claimed.</p>' + page[ending:]
            (output / "index.html").write_text(page)
        result["files"] = {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.rglob("*")
                            if path.is_file() and path.name != "comparison.json"}
    except BaseException as error:
        result.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        result.update(finished_at=manifests.utc_now(), elapsed_seconds=time.monotonic() - started)
        manifests.atomic_write_json(output / "comparison.json", result)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "methods": {label: {key: value for key, value in record.items() if key != "views"} for label, record in result["methods"].items()}}), flush=True)


if __name__ == "__main__":
    main()
