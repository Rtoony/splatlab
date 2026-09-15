"""CPU-only rectified reference bundles from a frozen, training-only fisheye model."""

from __future__ import annotations

import html
import json
import math
import re
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

import artifact_manifest as manifests
from rig_diagnostics import read_poses


SCHEMA = "dev.splatlab.fisheye-reference/v1"
VIEWS = (("left", -35.0), ("centre", 0.0), ("right", 35.0))
LIMITATIONS = [
    "Rectification uses estimated training intrinsics, not vendor calibration.",
    "Physical lens centers remain separate; virtual crops add no observations or parallax.",
    "Validation/test images have no estimated poses and never enter training transforms.",
    "Masks describe projection validity only, not reviewed sky/person/reflection exclusions.",
    "Sparse points are an existing reconstruction, not a new dense mesh or Gaussian splat.",
    "Scale and world up are arbitrary; no registration to the condo model is accepted.",
]


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 12 or fields[1] != "OPENCV_FISHEYE":
            raise ValueError("Expected COLMAP OPENCV_FISHEYE with eight parameters")
        identifier, width, height = int(fields[0]), int(fields[2]), int(fields[3])
        params = np.array([float(value) for value in fields[4:]])
        if (identifier in cameras or identifier < 1 or not 16 <= width <= 8192
                or not 16 <= height <= 8192 or not np.isfinite(params).all()
                or np.any(params[:2] <= 0) or not 0 <= params[2] <= width
                or not 0 <= params[3] <= height):
            raise ValueError("Invalid or duplicate fisheye camera")
        angles = np.linspace(0, math.radians(85), 1025)
        derivative = np.ones_like(angles)
        for index, coefficient in enumerate(params[4:]):
            derivative += (2 * index + 3) * coefficient * angles ** (2 * index + 2)
        if np.any(derivative <= 0):
            raise ValueError("Fisheye radial mapping folds within the supported cone")
        cameras[identifier] = {"width": width, "height": height, "params": params.tolist()}
    if len(cameras) != 2:
        raise ValueError("Expected two independently estimated physical-lens cameras")
    return cameras


def yaw_rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])


def project_fisheye(rays: np.ndarray, params: list[float]) -> tuple[np.ndarray, np.ndarray]:
    rays = np.asarray(rays, dtype=float)
    parameters = np.asarray(params, dtype=float)
    if (rays.shape[-1:] != (3,) or parameters.shape != (8,)
            or not np.isfinite(rays).all() or not np.isfinite(parameters).all()
            or np.any(parameters[:2] <= 0)):
        raise ValueError("Expected finite 3D rays and eight fisheye parameters")
    radius = np.linalg.norm(rays[..., :2], axis=-1)
    angle = np.arctan2(radius, rays[..., 2])
    radial = np.ones_like(angle)
    for index, coefficient in enumerate(parameters[4:]):
        radial += coefficient * angle ** (2 * index + 2)
    scale = np.divide(angle * radial, radius, out=np.zeros_like(radius), where=radius > 1e-12)
    pixels = rays[..., :2] * scale[..., None] * parameters[:2] + parameters[2:4]
    valid = (rays[..., 2] > 1e-12) & (angle <= math.radians(85))
    return pixels, valid


def remap_grid(camera: dict, width: int, fov_degrees: float, yaw_degrees: float) -> tuple:
    if (type(width) is not int or not 64 <= width <= 1024
            or not math.isfinite(fov_degrees) or not 60 <= fov_degrees <= 110
            or not math.isfinite(yaw_degrees) or abs(yaw_degrees) > 45):
        raise ValueError("Use a bounded rectilinear view: 64–1024 pixels, FOV 60–110, yaw ±45")
    focal = width / (2 * math.tan(math.radians(fov_degrees) / 2))
    rows, columns = np.indices((width, width), dtype=float)
    rays = np.stack(((columns + .5 - width / 2) / focal,
                     (rows + .5 - width / 2) / focal, np.ones_like(rows)), axis=-1)
    rotation = yaw_rotation(yaw_degrees)
    pixels, valid = project_fisheye(rays @ rotation.T, camera["params"])
    samples = pixels - .5
    valid &= ((samples[..., 0] >= 0) & (samples[..., 0] <= camera["width"] - 1)
              & (samples[..., 1] >= 0) & (samples[..., 1] <= camera["height"] - 1))
    return samples, valid, rotation, focal


def resample(image: Image.Image, samples: np.ndarray, valid: np.ndarray) -> Image.Image:
    values = np.asarray(image.convert("RGB"))
    horizontal = np.clip(samples[..., 0], 0, image.width - 1)
    vertical = np.clip(samples[..., 1], 0, image.height - 1)
    left, top = np.floor(horizontal).astype(int), np.floor(vertical).astype(int)
    right, bottom = np.minimum(left + 1, image.width - 1), np.minimum(top + 1, image.height - 1)
    fraction_x, fraction_y = (horizontal - left)[..., None], (vertical - top)[..., None]
    result = ((1 - fraction_y) * ((1 - fraction_x) * values[top, left] + fraction_x * values[top, right])
              + fraction_y * ((1 - fraction_x) * values[bottom, left] + fraction_x * values[bottom, right]))
    result[~valid] = 0
    return Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8))


def camera_to_world(pose: dict, virtual_to_lens: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = pose["rotation"].T @ virtual_to_lens @ np.diag([1, -1, -1])
    matrix[:3, 3] = pose["centre"]
    return matrix.tolist()


def verify_sources(pilot_root: Path, sfm_root: Path) -> tuple:
    pilot = json.loads((pilot_root / "pilot.json").read_text())
    receipt = json.loads((sfm_root / "receipt.json").read_text())
    if (pilot.get("schema") != "dev.splatlab.dual-fisheye-pilot/v1"
            or pilot.get("projection") != "raw-fisheye-resized-no-warp"
            or pilot.get("status") != "decoded-needs-camera-and-mask-review"
            or pilot.get("clock", {}).get("paired_container_pts_match") is not True
            or receipt.get("schema") != "dev.splatlab.dual-fisheye-sfm-probe/v1"
            or receipt.get("status") != "estimated-needs-rig-and-heldout-review"
            or receipt.get("heldout_used_in_mapping") is not False
            or receipt.get("pilot_sha256") != manifests.sha256_file(pilot_root / "pilot.json")):
        raise ValueError("Need an intact raw-lens pilot and its training-only SfM receipt")
    groups = {group["group_id"]: group for group in pilot["groups"]}
    if not 5 <= len(groups) <= 96 or len(groups) != len(pilot["groups"]):
        raise ValueError("Invalid or unbounded timestamp groups")
    views = pilot["views"]
    if len(views) != len(groups) * 2 or len({view["image"] for view in views}) != len(views):
        raise ValueError("Need exactly two unique views per timestamp group")
    sources = {"pilot": pilot_root / "pilot.json", "sfm-receipt": sfm_root / "receipt.json"}
    pairs = {}
    for view in views:
        name = view["image"]
        if not re.fullmatch(r"lens-[01]/frame-\d{6}\.jpg", name):
            raise ValueError("Invalid raw image path")
        group = groups.get(view["group_id"])
        if (not group or view["split"] != group["split"] or view["split"] not in {"train", "val", "test"}
                or view["lens_id"] != name.split("/")[0] or Path(name).stem != group["group_id"]
                or view["source_decoded_frame_index"] != group["source_decoded_frame_index"]):
            raise ValueError("Physical lens identity or timestamp split changed")
        path = pilot_root / "images" / name
        if (path.is_symlink() or not path.resolve().is_relative_to((pilot_root / "images").resolve())
                or manifests.sha256_file(path) != view["sha256"]):
            raise ValueError("Raw image changed or escapes input root")
        sources["image/" + name] = path
        pairs.setdefault(view["group_id"], []).append(view)
    for pair in pairs.values():
        if (len(pair) != 2 or {view["lens_id"] for view in pair} != {"lens-0", "lens-1"}
                or Fraction(pair[0]["pts"]) * Fraction(pair[0]["time_base"])
                != Fraction(pair[1]["pts"]) * Fraction(pair[1]["time_base"])):
            raise ValueError("Physical lens pair or exact presentation timestamps changed")
    if {view["split"] for view in views} != {"train", "val", "test"}:
        raise ValueError("Keep separate nonempty training, validation and test groups")
    model_root = sfm_root / "sparse/0"
    cameras = read_cameras(model_root / "cameras.txt")
    poses = read_poses(model_root / "images.txt")
    training_names = {view["image"] for view in views if view["split"] == "train"}
    if set(poses) != training_names or set(receipt["train_images"]) != training_names:
        raise ValueError("Training model names disagree or held-out poses leaked into training")
    lens_cameras = {}
    for lens in ("lens-0", "lens-1"):
        identifiers = {pose["camera_id"] for name, pose in poses.items() if name.startswith(lens + "/")}
        if len(identifiers) != 1 or not identifiers <= cameras.keys():
            raise ValueError("Intrinsics must stay shared per physical lens")
        lens_cameras[lens] = identifiers.pop()
    if len(set(lens_cameras.values())) != 2:
        raise ValueError("Physical lenses cannot be collapsed to one camera")
    for name in ("cameras.txt", "images.txt", "points3D.txt", "points.ply"):
        sources["model/" + name] = model_root / name
    snapshot = {name: manifests.sha256_file(path) for name, path in sources.items()}
    return pilot, cameras, poses, lens_cameras, sources, snapshot


def build_bundle(pilot_root: Path, sfm_root: Path, output: Path, width: int = 768, fov: float = 100) -> dict:
    pilot_root, sfm_root, output = pilot_root.resolve(), sfm_root.resolve(), output.resolve()
    if output.exists() or any(output.is_relative_to(root) for root in (pilot_root, sfm_root)):
        raise ValueError("Choose a new output directory outside frozen input roots")
    pilot, cameras, poses, lens_cameras, sources, snapshot = verify_sources(pilot_root, sfm_root)
    grids = {(identifier, label): remap_grid(camera, width, fov, yaw)
             for identifier, camera in cameras.items() for label, yaw in VIEWS}
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": SCHEMA, "status": "building-not-ready", "started_at": manifests.utc_now(),
               "source_hashes": snapshot, "tool_sha256": manifests.sha256_file(Path(__file__)),
               "metric_scale": "unknown", "registration": None, "training_ready": False,
               "heldout_evaluated": False, "limitations": LIMITATIONS}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    try:
        frames, training = [], []
        for view in pilot["views"]:
            camera_id = lens_cameras[view["lens_id"]]
            camera = cameras[camera_id]
            with Image.open(pilot_root / "images" / view["image"]) as original:
                if original.size != (camera["width"], camera["height"]):
                    raise ValueError("Raw image dimensions disagree with estimated camera")
                for label, yaw in VIEWS:
                    samples, valid, rotation, focal = grids[camera_id, label]
                    stem = f"{view['lens_id']}-{Path(view['image']).stem}-{label}"
                    image_path = Path(view["split"]) / (stem + ".png")
                    mask_path = Path("projection-masks") / (stem + ".png")
                    (output / image_path).parent.mkdir(exist_ok=True)
                    (output / mask_path).parent.mkdir(exist_ok=True)
                    resample(original, samples, valid).save(output / image_path)
                    Image.fromarray(valid.astype(np.uint8) * 255).save(output / mask_path)
                    pose = poses.get(view["image"])
                    frame = {"file_path": str(image_path), "mask_path": str(mask_path),
                             "sha256": manifests.sha256_file(output / image_path),
                             "mask_sha256": manifests.sha256_file(output / mask_path),
                             "source_image": view["image"], "source_sha256": view["sha256"],
                             "source_group": view["group_id"], "split": view["split"],
                             "physical_lens": view["lens_id"], "estimated_camera_id": camera_id,
                             "pts": view["pts"], "time_base": view["time_base"], "virtual_yaw_deg": yaw,
                             "valid_pixel_fraction": float(valid.mean()),
                             "transform_matrix": camera_to_world(pose, rotation) if pose else None,
                             "pose_basis": "frozen-training-sfm" if pose else "unlocalized-heldout"}
                    frames.append(frame)
                    if pose:
                        training.append({key: frame[key] for key in ("file_path", "mask_path", "transform_matrix")})
        intrinsics = {"camera_model": "OPENCV", "fl_x": focal, "fl_y": focal,
                      "cx": width / 2, "cy": width / 2, "w": width, "h": width,
                      "k1": 0, "k2": 0, "k3": 0, "k4": 0, "p1": 0, "p2": 0}
        document = {"schema": SCHEMA, **intrinsics, "frames": frames,
                    "coordinates": "Unchanged COLMAP world; camera-to-world OpenGL/Blender (+X right,+Y up,-Z forward)",
                    "pixel_coordinates": "Edge origin; pixel centers at (column+0.5,row+0.5)",
                    "units": "arbitrary", "registration": None, "limitations": LIMITATIONS}
        manifests.atomic_write_json(output / "camera-set.json", document)
        manifests.atomic_write_json(output / "training-transforms.json", {
            **intrinsics, "frames": training, "train_filenames": [frame["file_path"] for frame in training],
            "val_filenames": [], "test_filenames": [], "ply_file_path": "sparse.ply",
            "orientation_override": "none", "applied_transform": np.eye(4)[:3].tolist(),
            "splatlab_training_ready": False, "splatlab_limitations": LIMITATIONS})
        shutil.copyfile(sources["model/points.ply"], output / "sparse.ply")
        assets = []
        for kind, filename in (("camera-set", "camera-set.json"), ("point-cloud", "sparse.ply")):
            assets.append({"id": "raw-lens-" + kind, "kind": kind, "file": filename,
                           "sha256": manifests.sha256_file(output / filename),
                           "producer": {"tool": "SplatLab fisheye reference", "runId": output.name},
                           "sourceHashes": [pilot["source"]["sha256"], snapshot["pilot"], snapshot["sfm-receipt"]],
                           "units": "arbitrary", "coordinates": document["coordinates"],
                           "quality": {"heldoutEvaluated": False, "limitations": LIMITATIONS}, "registration": None})
        manifests.atomic_write_json(output / "external-references.json", {
            "schema": "chanate-external-references/v1", "assets": assets})
        cards = "".join(f'<figure><a href="{html.escape(frame["file_path"])}"><img loading="lazy" '
                        f'src="{html.escape(frame["file_path"])}" alt="Rectified {html.escape(frame["source_image"])}"></a>'
                        f'<figcaption>{html.escape(frame["source_image"])} · {frame["split"]} · '
                        f'{frame["virtual_yaw_deg"]:+g}° · {frame["pose_basis"]}</figcaption></figure>'
                        for frame in frames)
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Condo raw-video reconstruction inputs</title><style>'
            'body{background:#141c25;color:#edf1f7;font:16px system-ui;margin:24px}a{color:#8dd3ff}'
            '.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}'
            'figure{margin:0}img{width:100%;height:auto}figcaption{padding:8px;overflow-wrap:anywhere}</style>'
            '<h1>Real condo video → rectified reconstruction references</h1>'
            '<p>Actual raw-lens photos, not generated renders. Three virtual directions per lens retain the '
            'same physical optical center. This is input preparation, not a finished condo reconstruction.</p><ul>'
            + ''.join('<li>' + html.escape(note) + '</li>' for note in LIMITATIONS)
            + '</ul><p><a href="sparse.ply">Existing sparse 3D points</a> · '
            '<a href="camera-set.json">Cameras and image lineage</a> · '
            '<a href="external-references.json">Independent condo handoff</a></p><div class="views">' + cards + '</div>')
        if snapshot != {name: manifests.sha256_file(path) for name, path in sources.items()}:
            raise ValueError("Frozen source changed during rectification")
        receipt.update(status="prepared-needs-heldout-localization-and-mask-review", raw_images=len(pilot["views"]),
                       virtual_views=len(frames), splits=dict(Counter(frame["split"] for frame in frames)),
                       posed_views=len(training), min_valid_pixel_fraction=min(frame["valid_pixel_fraction"] for frame in frames),
                       artifacts={str(path.relative_to(output)): manifests.sha256_file(path)
                                  for path in sorted(output.rglob("*")) if path.is_file() and path.name != "receipt.json"})
    except Exception as error:
        receipt.update(status="failed-not-ready", error=str(error))
        raise
    finally:
        receipt["finished_at"] = manifests.utc_now()
        manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt
