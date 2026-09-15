#!/usr/bin/env python3
"""Prepare local SAM-3D inputs or build an immutable generated-object candidate through the compute gate."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import artifact_manifest as manifests
import generated_objects as objects
import generated_color
import reconstruction_evidence as evidence
import scene_revisions as scenes
import selection_reviews as reviews


def current_code(receipt):
    for name, digest in receipt["implementation"].items():
        if manifests.sha256_file(ROOT / name) != digest:
            raise RuntimeError("Generated-object implementation changed; prepare a fresh candidate")


def require_gate():
    gate = str(ROOT / "tools/splatlab-compute-gate.sh")
    if subprocess.run([gate, "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Launch through tools/splatlab-compute-gate.sh --run")
    subprocess.run([gate, "--check"], check=True)


def fit_pose(vertices, target, receipt, output, model):
    source_center = (np.percentile(vertices, 1, axis=0) + np.percentile(vertices, 99, axis=0)) / 2
    sample = vertices[np.random.default_rng(0).choice(len(vertices), min(len(vertices), receipt["recipe"]["fit_samples"]), replace=False)] - source_center
    source_extent = np.percentile(vertices, 99, axis=0) - np.percentile(vertices, 1, axis=0)
    target_center = (target.min(0) + target.max(0)) / 2
    scale = np.linalg.norm(np.ptp(target, axis=0)) / np.linalg.norm(source_extent)
    if not np.isfinite(scale) or not 1e-5 < scale < 100:
        raise RuntimeError("Generated and captured extents cannot define a bounded uniform scale")
    metres = receipt["calibration"]["meters_per_unit"]
    support_center = np.asarray(receipt["support_plane"]["center"]) @ evidence.Y_UP / metres
    normal = np.asarray(receipt["support_plane"]["normal"]) @ evidence.Y_UP
    fit_views = []
    for camera in receipt["cameras"]:
        if camera["split"] != "fit":
            continue
        transform = np.linalg.inv(np.asarray(camera["raw_to_camera"])) @ evidence.GL_TO_CV
        intrinsics = dict(zip(("fx", "fy", "cx", "cy"), camera["parameters"]))
        intrinsics["camera_to_worlds_3x4_opengl"] = transform[:3].tolist()
        with Image.open(output / camera["mask"]) as opened:
            mask = np.asarray(opened) >= 128
        fit_views.append((camera, intrinsics, mask))
    trials = []
    for up_axis in np.concatenate([np.eye(3), -np.eye(3)]):
        align = model._axis_rotation(up_axis, normal)
        minimum = float(np.min((vertices - source_center) @ up_axis))
        correction = np.dot(target_center - support_center, normal) + scale * minimum
        center = target_center - normal * correction
        for yaw in range(0, 360, 10):
            rotation = model._yaw(normal, np.radians(yaw)) @ align
            placed = sample @ rotation.T * scale + center
            scores = []
            for camera, intrinsics, mask in fit_views:
                pixels, _front = model.project_to_crop(placed, intrinsics, [0, 0, camera["width"], camera["height"]])
                scores.append(model.iou(model.rasterise_silhouette(pixels, mask.shape), mask))
            trials.append({"mean_iou": float(np.mean(scores)), "view_ious": scores, "yaw": yaw, "up_axis": up_axis.tolist(),
                           "rotation": rotation, "center": center, "support_translation_m": float(correction * metres)})
    initial = max(trials, key=lambda trial: trial["mean_iou"])
    for scale_multiplier in np.linspace(.65, 1.15, 11):
        refined_scale = scale * scale_multiplier
        up_axis = np.asarray(initial["up_axis"])
        minimum = float(np.min((vertices - source_center) @ up_axis))
        correction = np.dot(target_center - support_center, normal) + refined_scale * minimum
        center = target_center - normal * correction
        for yaw in range(initial["yaw"] - 15, initial["yaw"] + 16, 5):
            rotation = model._yaw(normal, np.radians(yaw)) @ model._axis_rotation(up_axis, normal)
            placed = sample @ rotation.T * refined_scale + center
            scores = []
            for camera, intrinsics, mask in fit_views:
                pixels, _front = model.project_to_crop(placed, intrinsics, [0, 0, camera["width"], camera["height"]])
                scores.append(model.iou(model.rasterise_silhouette(pixels, mask.shape), mask))
            trials.append({"mean_iou": float(np.mean(scores)), "view_ious": scores, "yaw": yaw, "up_axis": up_axis.tolist(),
                           "rotation": rotation, "center": center, "support_translation_m": float(correction * metres), "scale": refined_scale})
    best = max(trials, key=lambda trial: trial["mean_iou"])
    scale = best.get("scale", scale)
    matrix = np.eye(4)
    matrix[:3, :3] = best["rotation"] * scale
    matrix[:3, 3] = best["center"] - matrix[:3, :3] @ source_center
    return matrix, {key: value for key, value in best.items() if key not in {"rotation", "center"}} | {
        "uniform_scale_raw": float(scale), "generated_to_raw": matrix.tolist(), "candidates": len(trials),
        "initial_bbox_scale_mean_iou": initial["mean_iou"], "scale_refinement_range": [.65, 1.15],
        "fit_image_ids": [camera["image_id"] for camera, _intrinsics, _mask in fit_views], "holdouts_used": False,
        "method": "sampled-vertex multi-view silhouette initialization with inferred-plane contact; not measured pose"}


def render_comparison(master, delivery, receipt, output):
    import open3d as geometry

    raycasters = []
    for mesh in (master, delivery):
        scene = geometry.t.geometry.RaycastingScene()
        scene.add_triangles(geometry.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                            geometry.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
        raycasters.append(scene)
    destination = output / "review"
    destination.mkdir()
    results = []
    for camera in receipt["cameras"]:
        size = (480, round(camera["height"] * 480 / camera["width"]))
        factor = np.array(size) / [camera["width"], camera["height"]]
        parameters = np.asarray(camera["parameters"]) * np.tile(factor, 2)
        intrinsic = np.array([[parameters[0], 0, parameters[2]], [0, parameters[1], parameters[3]], [0, 0, 1.]])
        rays = geometry.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic, objects.rigid_camera(camera), *size)
        with Image.open(output / camera["photo"]) as opened:
            reference = opened.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        reference.save(destination / f"reference-{camera['image_id']}.png")
        with Image.open(output / camera["mask"]) as opened:
            mask = np.asarray(opened.resize(size, Image.Resampling.NEAREST)) >= 128
        rendered = []
        for label, mesh, scene in zip(("master", "delivery"), (master, delivery), raycasters):
            hit = scene.cast_rays(rays)
            depth = hit["t_hit"].numpy()
            valid = np.isfinite(depth)
            color = np.zeros((*depth.shape, 3), dtype=np.uint8)
            face_ids = hit["primitive_ids"].numpy()[valid]
            barycentric = hit["primitive_uvs"].numpy()[valid]
            weights = np.column_stack([1 - barycentric.sum(1), barycentric])
            colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
            color[valid] = np.clip(np.sum(colors[mesh.faces[face_ids]] * weights[..., None], axis=1), 0, 255).round().astype(np.uint8)
            Image.fromarray(color).save(destination / f"{label}-{camera['image_id']}.png")
            rendered.append((valid, depth, color))
        full, small = rendered
        overlap = full[0] & small[0]
        record = {"image_id": camera["image_id"], "split": camera["split"], "master_mask_iou": float(np.count_nonzero(full[0] & mask) / max(1, np.count_nonzero(full[0] | mask))),
                  "delivery_mask_iou": float(np.count_nonzero(small[0] & mask) / max(1, np.count_nonzero(small[0] | mask))),
                  "master_hit_pixels": int(full[0].sum()), "delivery_hit_pixels": int(small[0].sum()),
                  "delivery_changed_hit_pixels": int(np.count_nonzero(full[0] ^ small[0])),
                  "delivery_depth_mae_m": float(np.mean(np.abs(full[1][overlap] - small[1][overlap]))) if overlap.any() else None,
                  "delivery_rgb_mae_byte": float(np.mean(np.abs(full[2][overlap].astype(float) - small[2][overlap]))) if overlap.any() else None}
        np.savez_compressed(destination / f"depth-{camera['image_id']}.npz", master=full[1], delivery=small[1], mask=mask)
        results.append(record)
    return results


def run_model(job, receipt, output, model):
    model.doctor()
    weights = {name: manifests.file_identity(model.SAM3D_PIPELINE_YAML.parent / name) for name in model.SAM3D_CKPTS}
    weights["pipeline.yaml"] = manifests.file_identity(model.SAM3D_PIPELINE_YAML)
    work = output / "model-run"
    work.mkdir()
    script = work / "worker.py"
    script.write_text(model.WORKER_SAM3D)
    with Image.open(objects.artifact(job, receipt["generated_object_id"], "input.png")) as opened:
        rgba = np.asarray(opened.convert("RGBA"))
    np.save(work / "input.npy", rgba)
    request = {"pipeline_yaml": str(model.SAM3D_PIPELINE_YAML), "results_out": str(work / "result.json"), "objects": [{
        "slug": receipt["selected_slug"], "seed": receipt["seed"], "rgba_npy": str(work / "input.npy"),
        "mesh_out": str(output / "master.glb"), "splat_out": str(output / "master-splat.ply"),
        "geom_npz": str(work / "geometry.npz"), "preview_out": str(output / "model-preview.png")}]}
    manifests.atomic_write_json(work / "request.json", request)
    environment = model.worker_env({"SAM3D_ROOT": str(model.SAM3D_ROOT), "CONDA_PREFIX": str(model.SAM3D_PYTHON.parent.parent),
                                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with (work / "worker.log").open("x") as log:
        process = subprocess.Popen([str(model.SAM3D_PYTHON), str(script), str(work / "request.json")],
                                   cwd=model.SAM3D_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=420)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    actual = json.loads((work / "result.json").read_text()) if (work / "result.json").is_file() else []
    if code or len(actual) != 1 or not actual[0].get("ok"):
        raise RuntimeError(f"Local SAM-3D worker failed; inspect {work / 'worker.log'}")
    for name, identity in weights.items():
        if not manifests.same_file_identity(model.SAM3D_PIPELINE_YAML.parent / name, identity):
            raise RuntimeError("SAM-3D checkpoint/config identity changed during inference")
    return {"name": "local Meta SAM-3D Objects", "worker": actual[0], "weights": weights,
            "worker_sha256": manifests.sha256_file(script), "offline": True,
            "native_splat_frame": "Gaussian decoder frame; mesh transform does not apply; splat placement remains unsolved"}


def build(job, identifier):
    require_gate()
    import open3d as geometry
    import trimesh

    sys.path.insert(0, str(ROOT / "backend/mesh"))
    import object_generate as model

    with objects.worker_slot(job, identifier):
        receipt = objects.read(job, identifier)
        objects.verify(job, receipt)
        current_code(receipt)
        output = objects.directory(job, identifier)
        if (output / "model-run").exists() or (output / "result.json").exists():
            raise RuntimeError("Generation was already attempted; retain its evidence and prepare a new candidate")
        for name in receipt["artifacts"]:
            if manifests.sha256_file(output / name) != manifests.sha256_file(objects.artifact(job, identifier, name)):
                raise RuntimeError("Prepared generated-object input changed")
        started = time.monotonic()
        result = {"generated_object_id": identifier, "prepared_receipt_sha256": receipt["sha256"], "status": "failed", "render_vr_only": True,
                  "placement_resolved": False, "navigation_accepted": False, "multi_view_appearance_accepted": False}

        def expire(_signum, _frame):
            raise TimeoutError("Generated-object build exceeded its 600-second budget")

        previous = signal.signal(signal.SIGALRM, expire)
        signal.alarm(receipt["recipe"]["seconds"])
        try:
            if receipt.get("reuse_generated_object_id"):
                parent_id = receipt["reuse_generated_object_id"]
                parent = objects.verify_reuse(job, receipt)
                for name in ("master.glb", "master-splat.ply"):
                    (output / name).write_bytes(objects.artifact(job, parent_id, name, True).read_bytes())
                (output / "model-run").mkdir()
                result["model"] = {**parent["model"], "reused_from": parent_id, "reused_result_sha256": parent["sha256"]}
                manifests.atomic_write_json(output / "model-run/reuse.json", result["model"])
            else:
                result["model"] = run_model(job, receipt, output, model)
                model.tag_glb_generative(output / "master.glb")
                model.tag_ply_generative(output / "master-splat.ply")
            model.verify_glb(output / "master.glb")
            model.verify_ply(output / "master-splat.ply")
            master = trimesh.load(output / "master.glb", force="scene", process=False).to_geometry()
            if master.visual.kind != "vertex" or len(master.faces) > 3_000_000 or len(master.vertices) > 2_000_000:
                raise RuntimeError("Generated master exceeds the mesh budget or lacks actual vertex colors")
            with np.load(objects.artifact(job, identifier, "target-points.npz"), allow_pickle=False) as target:
                matrix, placement = fit_pose(master.vertices, target["positions"], receipt, output, model)
            to_world = np.eye(4)
            to_world[:3, :3] = evidence.Y_UP * receipt["calibration"]["meters_per_unit"]
            master.apply_transform(to_world @ matrix)
            master.export(output / "placed-master-model-colors.glb")
            model.tag_glb_generative(output / "placed-master-model-colors.glb")
            generated_color.derive(output / "placed-master-model-colors.glb", output / "placed-master.glb")
            simplified = geometry.geometry.TriangleMesh(geometry.utility.Vector3dVector(master.vertices), geometry.utility.Vector3iVector(master.faces))
            simplified.vertex_colors = geometry.utility.Vector3dVector(np.asarray(master.visual.vertex_colors)[:, :3] / 255.)
            simplified = simplified.simplify_quadric_decimation(receipt["recipe"]["delivery_faces"])
            delivery = trimesh.Trimesh(vertices=np.asarray(simplified.vertices), faces=np.asarray(simplified.triangles),
                                      vertex_colors=np.round(np.clip(np.asarray(simplified.vertex_colors), 0, 1) * 255).astype(np.uint8), process=False)
            delivery.export(output / "delivery-model-colors.glb")
            model.tag_glb_generative(output / "delivery-model-colors.glb")
            result["mesh_color"] = generated_color.derive(output / "delivery-model-colors.glb", output / "delivery.glb")
            for name in ("placed-master.glb", "delivery.glb"):
                model.verify_glb(output / name)
            views = render_comparison(master, delivery, receipt, output)
            fit_iou = float(np.mean([view["master_mask_iou"] for view in views if view["split"] == "fit"]))
            placed = fit_iou >= receipt["recipe"]["minimum_fit_iou"]
            contact = (master.vertices - np.asarray(receipt["support_plane"]["center"])) @ np.asarray(receipt["support_plane"]["normal"])
            result.update(status="needs-review" if placed else "unplaced", placement_resolved=placed, placement=placement,
                          fit_mesh_mask_iou=fit_iou, views=views, master_triangles=len(master.faces), delivery_triangles=len(delivery.faces),
                          support_plane_contact_error_m=abs(float(contact.min())),
                          geometry_scope="all generated master components retained; QEM delivery is separate; no cleanup/refit, no measured surface claim")
            objects.verify(job, receipt)
            current_code(receipt)
        except Exception as exc:
            result.update(status="failed", placement_resolved=False, error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            result["seconds"] = time.monotonic() - started
            files = [path for path in output.rglob("*") if path.is_file() and str(path.relative_to(output)) not in receipt["artifacts"]
                     and str(path.relative_to(output)) not in {"receipt.json", "result.json", "worker.lock"}]
            if any(path.stat().st_size > evidence.MAX_FILE_BYTES for path in files):
                result.update(status="failed", placement_resolved=False, error="Generated file exceeds the 512 MiB artifact cap")
                files = [path for path in files if path.stat().st_size <= evidence.MAX_FILE_BYTES]
            result = reviews.seal(job, output, "result.json", result, files)
            print(json.dumps({key: result.get(key) for key in ("generated_object_id", "status", "seconds", "master_triangles", "delivery_triangles", "fit_mesh_mask_iou", "error", "sha256")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build", "rederive"])
    parser.add_argument("job", type=Path)
    parser.add_argument("--review-id")
    parser.add_argument("--recovery-id")
    parser.add_argument("--expected-generation", type=int)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--generated-object-id")
    parser.add_argument("--reuse-generated-object-id")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.review_id is None or args.recovery_id is None or args.expected_generation is None:
            parser.error("Preparation requires current selection, support and generation")
        receipt = objects.prepare(args.job.resolve(), args.review_id, args.recovery_id, args.expected_generation, args.image_id,
                                  reuse_generated_object_id=args.reuse_generated_object_id)
        print(json.dumps({key: receipt[key] for key in ("generated_object_id", "base", "input_image_id", "input_crop", "sha256")}))
    elif args.action == "rederive":
        if args.generated_object_id is None:
            parser.error("Rederivation requires --generated-object-id")
        receipt = objects.rederive(args.job.resolve(), args.generated_object_id)
        print(json.dumps({key: receipt[key] for key in ("generated_object_id", "base", "reuse_generated_object_id", "sha256")}))
    else:
        if args.generated_object_id is None:
            parser.error("Build requires --generated-object-id")
        build(args.job.resolve(), args.generated_object_id)


if __name__ == "__main__":
    main()
