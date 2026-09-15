"""Bounded, training-only surface studies with explicit inferred-geometry scope."""

import hashlib
from html import escape
from pathlib import Path
import uuid

import numpy as np

import artifact_manifest as manifests
import reconstruction_comparison as comparison
import scene_revisions as scenes
from mesh.provenance import GENERATIVE_TAG

MAXIMUM_TRIANGLES = 6_000_000
MAXIMUM_VERTICES = 4_000_000


def normalized_seeds(study, run):
    names = [name for name in run["artifacts"] if name.endswith("dataparser_transforms.json")]
    if len(names) != 1:
        raise ValueError("Surface extraction needs one registered dataparser normalization")
    normalization = manifests.read_json(study / run["arm"] / names[0])
    transforms = manifests.read_json(study / "dataset/transforms.json")
    raw = (study / "dataset/train-seeds.ply").read_bytes()
    header, body = raw.split(b"end_header\n", 1)
    expected = b"property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
    if not header.endswith(expected) or b"format binary_little_endian 1.0\n" not in header:
        raise ValueError("Unexpected registered seed PLY layout")
    vertices = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float64)
    applied = np.eye(4)
    stored_applied = np.asarray(transforms.get("applied_transform", np.eye(4)), dtype=np.float64)
    applied[:stored_applied.shape[0]] = stored_applied
    transform = np.eye(4)
    transform[:3] = np.asarray(normalization["transform"])
    scale = normalization["scale"]
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(transform).all() or not np.isfinite(applied).all():
        raise ValueError("Invalid training normalization")
    saved_to_normalized = transform @ np.linalg.inv(applied)
    points = (points @ saved_to_normalized[:3, :3].T + saved_to_normalized[:3, 3]) * scale
    return points, {"source_to_training": normalization, "saved_applied_transform": applied.tolist()}


def bounds_for_seeds(points, voxel):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError("Bounds need finite registered training seeds")
    lower, upper = np.quantile(points, [.005, .995], axis=0)
    padding = np.maximum((upper - lower) * .05, voxel * 3)
    lower, upper = lower - padding, upper + padding
    dimensions = np.ceil((upper - lower) / voxel)
    if np.any(dimensions > 1024) or np.prod(dimensions) > 512 ** 3:
        raise ValueError("Training-derived crop exceeds the bounded fusion grid; use a coarser shared voxel")
    return {"minimum": lower.tolist(), "maximum": upper.tolist(), "grid_dimensions": dimensions.astype(int).tolist(),
            "source": "0.5th/99.5th training-seed coordinate percentiles plus 5% span or three voxels; not full-scene coverage"}


def prepare(study, names, voxel=.01, seconds=600):
    if isinstance(voxel, bool) or not np.isfinite(voxel) or not .005 <= voxel <= .1 or type(seconds) is not int or not 60 <= seconds <= 1800:
        raise ValueError("Use a bounded shared voxel and wall-clock budget")
    result = comparison.summarize(study, names)
    runs = {name: comparison.read(study / name, "run.json") for name in names}
    if any(name != run["arm"] for name, run in runs.items()):
        raise ValueError("This surface protocol currently requires the original named arms")
    baseline = runs[result["baseline"]]
    points, normalization = normalized_seeds(study, baseline)
    for run in runs.values():
        other_points, other_normalization = normalized_seeds(study, run)
        if other_normalization != normalization or not np.array_equal(points, other_points):
            raise ValueError("Compared model coordinate normalizations differ")
    receipt = comparison.read(study)
    if len(points) != receipt["seed_points"]:
        raise ValueError("Registered seed count changed")
    output = study / ("surfaces-" + uuid.uuid4().hex[:24])
    output.mkdir()
    payload = comparison.seal(output, "receipt.json", {"schema": "dev.splatlab.surface-study/v1", "comparison_sha256": result["sha256"],
        "run_sha256": result["run_sha256"], "created_at": manifests.utc_now(), "arms": names,
        "parameters": {"voxel_length": float(voxel), "sdf_truncation": float(voxel * 4), "alpha_minimum": .5,
                       "depth_truncation": 4., "seconds_per_arm": seconds, "maximum_triangles": MAXIMUM_TRIANGLES, "maximum_vertices": MAXIMUM_VERTICES},
        "bounds": bounds_for_seeds(points, voxel), "normalization": normalization,
        "training_images": receipt["prepared_splits"]["train"], "evaluation_images": receipt["prepared_splits"][receipt["evaluation_split"]],
        "geometry_source": "TSDF of model-predicted expected depths from training views only",
        "color_source": "registered training photographs projected onto inferred depth; RGB8 TSDF vertex colors",
        "coordinate_scope": "normalized training coordinates; not established metres",
        "provenance": GENERATIVE_TAG, "promoted": False})
    return output, payload


def camera_record(camera_to_world, width, height, fx, fy, cx, cy):
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (3, 4) or not np.isfinite(pose).all() or not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("Invalid pinhole camera")
    if type(width) is not int or type(height) is not int or not 1 <= width * height <= 4_194_304 or min(width, height) < 1:
        raise ValueError("Camera exceeds bounded dimensions")
    rotation = pose[:, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
        raise ValueError("Camera pose must be rigid and right handed")
    return {"camera_to_world_opengl": pose.tolist(), "width": width, "height": height, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def rays(camera):
    rows, columns = np.indices((camera["height"], camera["width"]), dtype=np.float64)
    directions = np.stack([(columns + .5 - camera["cx"]) / camera["fx"], (rows + .5 - camera["cy"]) / camera["fy"], np.ones_like(rows)], axis=-1)
    pose = np.asarray(camera["camera_to_world_opengl"])
    rotation = pose[:, :3] @ np.diag([1, -1, -1])
    directions = directions @ rotation.T
    origins = np.broadcast_to(pose[:, 3], directions.shape)
    return np.concatenate([origins, directions], axis=-1).astype(np.float32)


def crop_mask(ray_values, bounds, depth=None):
    lower, upper = np.asarray(bounds["minimum"]), np.asarray(bounds["maximum"])
    origins, directions = ray_values[..., :3], ray_values[..., 3:]
    if depth is not None:
        points = origins + directions * np.asarray(depth)[..., None]
        return np.isfinite(points).all(axis=-1) & (points >= lower).all(axis=-1) & (points <= upper).all(axis=-1)
    parallel = np.abs(directions) < 1e-12
    safe = np.where(parallel, 1., directions)
    entering, leaving = (lower - origins) / safe, (upper - origins) / safe
    near, far = np.minimum(entering, leaving), np.maximum(entering, leaving)
    outside_parallel = parallel & ((origins < lower) | (origins > upper))
    near = np.where(parallel, -np.inf, near)
    far = np.where(parallel, np.inf, far)
    return ~outside_parallel.any(axis=-1) & (far.min(axis=-1) >= np.maximum(near.max(axis=-1), 0))


def filtered_depth(depth, opacity, valid, ray_values, bounds, parameters):
    depth, opacity, valid = np.asarray(depth), np.asarray(opacity), np.asarray(valid)
    if depth.shape != opacity.shape or depth.shape != valid.shape or ray_values.shape != (*depth.shape, 6):
        raise ValueError("Depth, opacity, masks and camera rays must align")
    if not np.isfinite(opacity).all() or np.any((opacity < 0) | (opacity > 1.00001)):
        raise ValueError("Invalid rendered opacity")
    accepted = np.isfinite(depth) & (depth > 0) & (depth < parameters["depth_truncation"]) & (opacity >= parameters["alpha_minimum"]) & valid.astype(bool)
    accepted &= crop_mask(ray_values, bounds, np.where(np.isfinite(depth), depth, 0))
    return np.where(accepted, depth, 0).astype(np.float32), accepted


def write_mesh(path, vertices, triangles, colors):
    vertices, triangles, colors = np.asarray(vertices), np.asarray(triangles), np.asarray(colors)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or len(vertices) > MAXIMUM_VERTICES or colors.shape != vertices.shape:
        raise ValueError("Invalid or oversized surface vertices/colors")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not 1 <= len(triangles) <= MAXIMUM_TRIANGLES or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("Invalid or oversized surface triangles")
    if not np.isfinite(vertices).all() or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1) or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("Nonfinite surface or out-of-range triangles/colors")
    packed_vertices = np.empty(len(vertices), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        packed_vertices[name] = vertices[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        packed_vertices[name] = np.rint(colors[:, index] * 255).astype(np.uint8)
    packed_faces = np.empty(len(triangles), dtype=[("count", "u1"), ("vertices", "<i4", (3,))])
    packed_faces["count"], packed_faces["vertices"] = 3, triangles
    header = f"ply\nformat binary_little_endian 1.0\ncomment {GENERATIVE_TAG}\ncomment RGB-derived expected-depth TSDF; normalized training coordinates; not independently measured\nelement vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(triangles)}\nproperty list uchar int vertex_indices\nend_header\n"
    with Path(path).open("xb") as stream:
        stream.write(header.encode())
        packed_vertices.tofile(stream)
        packed_faces.tofile(stream)


def depth_agreement(mesh_depth, model_depth, supported, voxel):
    mesh_depth, model_depth, supported = np.asarray(mesh_depth), np.asarray(model_depth), np.asarray(supported, dtype=bool)
    if mesh_depth.shape != model_depth.shape or supported.shape != mesh_depth.shape or not supported.any():
        raise ValueError("Depth comparison needs aligned nonempty support")
    if not np.isfinite(model_depth[supported]).all() or np.any(model_depth[supported] <= 0) or not np.isfinite(voxel) or voxel <= 0:
        raise ValueError("Model depth support and voxel must be finite and positive")
    hits = np.isfinite(mesh_depth) & (mesh_depth > 0) & supported
    differences = mesh_depth[hits] - model_depth[hits]
    return {"supported_model_pixels": int(supported.sum()), "mesh_hit_fraction_on_supported_model": float(hits.sum() / supported.sum()),
            "depth_mae_training_units": float(np.abs(differences).mean()) if differences.size else None,
            "depth_rmse_training_units": float(np.sqrt(np.mean(differences ** 2))) if differences.size else None,
            "within_two_voxels_fraction_of_supported_model": float((np.abs(differences) <= 2 * voxel).sum() / supported.sum()),
            "scope": "held-out self-consistency with model-predicted expected depth; NOT measured geometric accuracy"}


def camera_digest(records):
    return hashlib.sha256(scenes.canonical_bytes(records)).hexdigest()


def summarize(output):
    receipt = comparison.read(output)
    original = comparison.summarize(output.parent, receipt["arms"])
    if original["sha256"] != receipt["comparison_sha256"]:
        raise ValueError("Original comparison changed")
    runs = {name: comparison.read(output / name, "run.json") for name in receipt["arms"]}
    baseline = runs[original["baseline"]]
    expected = set(receipt["evaluation_images"])
    for name, run in runs.items():
        if run["status"] != "completed" or run["surface_receipt_sha256"] != receipt["sha256"] or run["trained_run_sha256"] != receipt["run_sha256"][name] or run["parameters"] != receipt["parameters"]:
            raise ValueError("Failed, stale or unequal-recipe surfaces cannot be compared")
        if any(run[key] != baseline[key] for key in ("training_camera_sha256", "evaluation_camera_sha256", "open3d_version", "torch_version")):
            raise ValueError("Surface cameras or runtimes differ between methods")
        if len(run["evaluation"]) != len(expected) or {view["image"] for view in run["evaluation"]} != expected:
            raise ValueError("Surface evaluation does not cover the exact held-out set")
        if len(run["training_views"]) != len(receipt["training_images"]) or {view["image"] for view in run["training_views"]} != set(receipt["training_images"]):
            raise ValueError("Surface fusion did not use the exact training split")
        for relative, identity in run["artifacts"].items():
            path = output / name / relative
            if path.is_symlink() or not path.resolve().is_relative_to((output / name).resolve()) or manifests.sha256_file(path) != identity["sha256"]:
                raise ValueError("Retained surface artifacts changed")
        for filename in expected:
            reference = "renders/" + Path(filename).stem + "-reference.png"
            if run["artifacts"][reference]["sha256"] != baseline["artifacts"][reference]["sha256"]:
                raise ValueError("Surface methods did not evaluate identical reference pixels")
            with np.load(output / name / "renders" / (Path(filename).stem + ".npz"), allow_pickle=False) as current:
                with np.load(output / original["baseline"] / "renders" / (Path(filename).stem + ".npz"), allow_pickle=False) as reference_maps:
                    if not np.array_equal(current["roi_rays"], reference_maps["roi_rays"]):
                        raise ValueError("Surface methods use different image-space crop masks")
    methods = {}
    for name, run in runs.items():
        views = run["evaluation"]
        errors = [view["depth_mae_training_units"] for view in views if view["depth_mae_training_units"] is not None]
        values = {"mean_source_photo_psnr_on_roi_rays": float(np.mean([view["source_photo_psnr_on_roi_rays"] for view in views])),
                  "mean_mesh_hit_fraction_on_roi_rays": float(np.mean([view["mesh_hit_fraction_on_roi_rays"] for view in views])),
                  "mean_mesh_hit_fraction_on_supported_model": float(np.mean([view["mesh_hit_fraction_on_supported_model"] for view in views])),
                  "mean_within_two_voxels_fraction_of_supported_model": float(np.mean([view["within_two_voxels_fraction_of_supported_model"] for view in views])),
                  "mean_depth_mae_training_units": float(np.mean(errors)) if errors else None,
                  "views_with_depth_hits": len(errors), **{key: run[key] for key in ("mesh", "fusion_seconds", "total_seconds", "max_rss_bytes", "max_cuda_allocated_bytes")},
                  "mesh_bytes": run["artifacts"]["surface.ply"]["bytes"]}
        if any(not np.isfinite(value) for value in values.values() if isinstance(value, (float, int))):
            raise ValueError("Surface summary contains nonfinite metrics")
        methods[name] = values
    payload = {"schema": "dev.splatlab.surface-comparison/v1", "surface_receipt_sha256": receipt["sha256"], "run_sha256": {name: run["sha256"] for name, run in runs.items()},
               "methods": methods, "status": "needs-review", "promoted": False, "parameters": receipt["parameters"], "bounds": receipt["bounds"],
               "geometric_reference_error": None, "scope": "cropped RGB-derived surfaces; photo PSNR includes holes within a shared ray-box mask; depth agreement uses each model's own prediction, not independent ground truth"}
    if (output / "comparison.json").exists():
        previous = comparison.read(output, "comparison.json")
        if previous["run_sha256"] != payload["run_sha256"]:
            raise ValueError("Do not overwrite a retained surface comparison")
        return previous, runs
    return comparison.seal(output, "comparison.json", payload), runs


def review_html(result, runs):
    names = list(result["methods"])
    rows = []
    for name, values in result["methods"].items():
        rows.append(f"<tr><th>{escape(name)}</th><td>{values['mesh']['triangles']:,}</td><td>{values['mean_mesh_hit_fraction_on_roi_rays']:.1%}</td><td>{values['mean_source_photo_psnr_on_roi_rays']:.2f}</td><td>{values['total_seconds']:.1f}</td><td><a href='{escape(name)}/surface.ply'>Inferred mesh</a></td></tr>")
    sections = []
    for view in runs[names[0]]["evaluation"]:
        stem = Path(view["image"]).stem
        figures = [("Reference", f"{names[0]}/renders/{stem}-reference.png")]
        figures += [(name, f"{name}/renders/{stem}.png") for name in names]
        figures += [(name + " geometry normals", f"{name}/renders/{stem}-normals.png") for name in names]
        content = "".join(f"<figure><figcaption>{escape(label)}</figcaption><a href='{escape(filename)}'><img loading='lazy' src='{escape(filename)}' alt='{escape(label)} at {escape(stem)}'></a></figure>" for label, filename in figures)
        sections.append(f"<section><h2>Held-out view {escape(stem)}</h2><div class='views'>{content}</div></section>")
    return """<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SplatLab inferred surface review</title><style>
body{font:16px system-ui;background:#10141b;color:#e8edf5;margin:24px}a{color:#8bc8ff}h1{font-size:1.7rem}
.notice{padding:16px;border:1px solid #ba924e;border-radius:8px;max-width:1050px}.table{overflow:auto}
table{border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #46515f}
.views{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}
figcaption{padding:8px}section{margin-top:32px}@media(max-width:650px){body{margin:12px}.views{grid-template-columns:1fr}}
</style><h1>Inferred surface review</h1>
<p class='notice'>Training-view depth fusion only. These are cropped, RGB-derived surfaces in normalized training coordinates,
not survey measurements or accepted colliders. Black pixels include holes and regions outside the crop. Colors come from
training photographs projected onto inferred depths. Mesh photo scores use the same projected crop mask, not the earlier full-image splat metric.
Depth agreement is self-consistency with each model, not independent geometric accuracy. No method is promoted.</p>
<p><a href='comparison.json'>Metrics and limitations</a> · <a href='receipt.json'>Frozen fusion recipe and crop</a></p>
<div class='table'><table><tr><th>Method</th><th>Triangles</th><th>Crop-ray coverage</th><th>Crop photo PSNR dB</th><th>Total seconds</th><th>Export</th></tr>
""" + "".join(rows) + "</table></div>" + "".join(sections) + "</html>"
