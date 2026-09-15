"""Immutable generated-Gaussian frame conversion and registered review artifacts."""

import hashlib
from pathlib import Path
import re

import numpy as np

import artifact_manifest as manifests
import generated_color
import glb_transform
import generated_objects as objects
import reconstruction_evidence as evidence
import scene_revisions as scenes
from mesh.provenance import GENERATIVE_TAG
from mesh.provenance import glb_is_generative

STUDY_RE = re.compile(r"gaussians_[a-f0-9]{24}\Z")
FIELDS = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
GAUSSIAN_TO_MESH = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
SDK_FILES = (
    "sam3d_objects/pipeline/inference_pipeline.py",
    "sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py",
    "sam3d_objects/model/backbone/tdfy_dit/representations/gaussian/gaussian_model.py",
)
SDK_HASHES = (
    "e83e9560a727b06565134ea855503034d3510cbfbeeff2b6ca6dc13013536d2f",
    "7fdd03ebb0ac45ee41e0a35408d7e068b568793e677de4e485af0f9faf812b72",
    "522be3a65f8d316d2b22796c0a31cc87de064aa542ff4bf75b67660a6fc941f6",
)
FRAME_TOLERANCES = {"rgb_mae": .002, "rgb_p99": .01, "alpha_max": .05, "depth_mae_m": .0001}


def verify_sdk():
    root = Path("/home/rtoony/tools/sam-3d-objects")
    for name, digest in zip(SDK_FILES, SDK_HASHES):
        if manifests.sha256_file(root / name) != digest:
            raise evidence.EvidenceError("SAM-3D export implementation changed; revalidate the Gaussian-to-mesh frame")
    return dict(zip(SDK_FILES, SDK_HASHES))


def directory(job, object_id, identifier):
    if not STUDY_RE.fullmatch(identifier):
        raise evidence.EvidenceError("Invalid generated-Gaussian study identifier")
    path = objects.directory(job, object_id) / "gaussians" / identifier
    if path.resolve() != path.absolute():
        raise evidence.EvidenceError("Generated-Gaussian study paths cannot be symlinked")
    return path


def read(job, object_id, identifier):
    result = manifests.read_json(directory(job, object_id, identifier) / "result.json")
    if not result:
        raise evidence.EvidenceError("Generated-Gaussian review is missing")
    digest = hashlib.sha256(scenes.canonical_bytes({key: value for key, value in result.items() if key != "sha256"})).hexdigest()
    if result.get("sha256") != digest or result.get("generated_object_id") != object_id or result.get("gaussians_id") != identifier:
        raise evidence.EvidenceError("Generated-Gaussian review integrity failed")
    if result.get("generated_result_sha256") != objects.read(job, object_id, True)["sha256"]:
        raise evidence.EvidenceError("Generated-Gaussian review belongs to different model placement")
    return result


def artifact(job, object_id, identifier, name):
    record = read(job, object_id, identifier).get("artifacts", {}).get(name)
    if not record:
        raise evidence.EvidenceError("Artifact is not registered to this Gaussian review")
    path = scenes.blob_path(job, record["sha256"])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or manifests.sha256_file(path) != record["sha256"]:
        raise evidence.EvidenceError("Generated-Gaussian review artifact is missing or corrupt")
    return path


def read_native(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > evidence.MAX_FILE_BYTES:
        raise evidence.EvidenceError("Gaussian input is not a bounded regular file")
    with path.open("rb") as handle:
        lines = []
        while sum(map(len, lines)) < 65536:
            line = handle.readline(4096)
            if not line or len(line) >= 4096:
                raise evidence.EvidenceError("Invalid Gaussian PLY header")
            lines.append(line)
            if line == b"end_header\n":
                break
        else:
            raise evidence.EvidenceError("Gaussian PLY header exceeds limit")
        try:
            header = b"".join(lines).decode("ascii").splitlines()
        except UnicodeDecodeError as error:
            raise evidence.EvidenceError("Gaussian PLY header is not ASCII") from error
        significant = [line for line in header if not line.startswith("comment ")]
        if significant[:2] != ["ply", "format binary_little_endian 1.0"] or not significant[2].startswith("element vertex "):
            raise evidence.EvidenceError("Only binary little-endian Gaussian PLY is supported")
        try:
            count = int(significant[2].split()[-1])
        except ValueError as error:
            raise evidence.EvidenceError("Invalid Gaussian row count") from error
        if not 1 <= count <= 2_000_000 or significant[3:] != [f"property float {name}" for name in FIELDS] + ["end_header"]:
            raise evidence.EvidenceError("Expected bounded degree-zero SAM-3D Gaussian fields; refusing other schemas/SH degrees")
        if f"comment {GENERATIVE_TAG}" not in header:
            raise evidence.EvidenceError("Native Gaussian master lacks portable generative provenance")
        if path.stat().st_size - handle.tell() != count * len(FIELDS) * 4:
            raise evidence.EvidenceError("Gaussian row bytes do not match the declared shape")
        values = np.frombuffer(handle.read(), dtype="<f4").reshape(count, len(FIELDS)).copy()
    validate_rows(values)
    return values


def validate_rows(values):
    if values.ndim != 2 or values.shape[1] != len(FIELDS) or not 1 <= len(values) <= 2_000_000 or not np.isfinite(values).all():
        raise evidence.EvidenceError("Invalid Gaussian values")
    norms = np.linalg.norm(values[:, 13:17].astype(float), axis=1)
    if np.any(norms < 1e-8) or np.any(np.abs(values[:, 10:13]) > 80):
        raise evidence.EvidenceError("Gaussian covariance is degenerate or outside numeric range")
    if np.any(values[:, 3:6] != 0):
        raise evidence.EvidenceError("SAM-3D native normals must be zero placeholders, not measured normals")


def similarity(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise evidence.EvidenceError("Invalid generated-Gaussian placement")
    scale = float(np.linalg.norm(matrix[:3, 0]))
    if not 1e-6 <= scale <= 100:
        raise evidence.EvidenceError("Generated-Gaussian scale is outside bounds")
    rigid = matrix.copy()
    rigid[:3, :3] /= scale
    evidence.matrix4(rigid)
    return scale, rigid[:3, :3]


def rotation_quaternion(rotation):
    symmetric = np.array([
        [rotation[0, 0] - rotation[1, 1] - rotation[2, 2], rotation[0, 1] + rotation[1, 0], rotation[0, 2] + rotation[2, 0], rotation[2, 1] - rotation[1, 2]],
        [rotation[0, 1] + rotation[1, 0], rotation[1, 1] - rotation[0, 0] - rotation[2, 2], rotation[1, 2] + rotation[2, 1], rotation[0, 2] - rotation[2, 0]],
        [rotation[0, 2] + rotation[2, 0], rotation[1, 2] + rotation[2, 1], rotation[2, 2] - rotation[0, 0] - rotation[1, 1], rotation[1, 0] - rotation[0, 1]],
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1], np.trace(rotation)],
    ])
    _values, vectors = np.linalg.eigh(symmetric)
    quaternion = vectors[:, -1][[3, 0, 1, 2]]
    return quaternion if quaternion[0] >= 0 else -quaternion


def transform_rows(values, matrix):
    validate_rows(values)
    scale, rotation = similarity(matrix)
    result = values.copy()
    result[:, :3] = values[:, :3].astype(float) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
    result[:, 10:13] = values[:, 10:13].astype(float) + np.log(scale)
    source = values[:, 13:17].astype(float)
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    left = rotation_quaternion(rotation)
    transformed = np.empty_like(source)
    transformed[:, 0] = left[0] * source[:, 0] - source[:, 1:] @ left[1:]
    transformed[:, 1:] = left[0] * source[:, 1:] + source[:, :1] * left[1:] + np.cross(left[1:], source[:, 1:])
    result[:, 13:17] = transformed
    validate_rows(result)
    return result


def write_placed(path, values):
    validate_rows(values)
    header = ["ply", "format binary_little_endian 1.0", f"comment {GENERATIVE_TAG}",
              "comment SplatLab fitted generated object; Y-up metres; not measured geometry", f"element vertex {len(values)}"]
    header += [f"property float {name}" for name in FIELDS] + ["end_header", ""]
    with Path(path).open("xb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(values.astype("<f4").tobytes())


def world_matrix(receipt, result):
    if (receipt.get("recipe", {}).get("model") != "local SAM-3D Objects"
            or result.get("model", {}).get("name") != "local Meta SAM-3D Objects"
            or result.get("model", {}).get("worker", {}).get("ok") is not True):
        raise evidence.EvidenceError("This frame adapter requires a completed local SAM-3D model stage")
    if result.get("status") != "needs-review" or not result.get("placement_resolved"):
        raise evidence.EvidenceError("A fitted generated mesh is required before placing its native Gaussians")
    raw_to_world = np.eye(4)
    raw_to_world[:3, :3] = evidence.Y_UP * receipt["calibration"]["meters_per_unit"]
    matrix = raw_to_world @ np.asarray(result["placement"]["generated_to_raw"]) @ GAUSSIAN_TO_MESH
    similarity(matrix)
    return matrix


def proposal_asset(job, object_id, identifier, pointer, selected_slug=None):
    objects.proposal_asset(job, object_id, pointer, selected_slug)
    receipt = objects.read(job, object_id)
    parent = objects.read(job, object_id, True)
    result = read(job, object_id, identifier)
    if (result.get("status") != "needs-review" or result.get("base") != pointer
            or result.get("prepared_receipt_sha256") != receipt["sha256"]):
        raise evidence.EvidenceError("Native Gaussian placement requires current, completed parent evidence")
    if (result.get("sdk_export_sources") != dict(zip(SDK_FILES, SDK_HASHES))
            or result.get("native_identity") != parent["artifacts"].get("master-splat.ply")
            or result.get("frame_tolerances") != FRAME_TOLERANCES
            or not all(result.get(key) is True for key in ("all_rows_preserved", "appearance_parameters_byte_exact", "placed_export_byte_exact"))):
        raise evidence.EvidenceError("Native Gaussian lineage or full-row verification is incomplete")
    matrix = world_matrix(receipt, parent)
    reported_matrix = np.asarray(result.get("native_to_world"), dtype=float)
    if (not np.array_equal(result.get("gaussian_to_mesh"), GAUSSIAN_TO_MESH)
            or reported_matrix.shape != (4, 4) or not np.allclose(reported_matrix, matrix, rtol=0, atol=1e-12)):
        raise evidence.EvidenceError("Native Gaussian frame disagrees with the fitted mesh")
    expected_views = {(camera["image_id"], camera["split"]) for camera in receipt["cameras"]}
    views = result.get("views", [])
    if len(views) != len(expected_views) or {(view.get("image_id"), view.get("split")) for view in views} != expected_views:
        raise evidence.EvidenceError("Native Gaussian review is missing required camera evidence")
    for view in views:
        if view.get("opaque_pixels", 0) <= 0 or any(
            not 0 <= view.get("frame", {}).get(key, float("inf")) <= limit
            for key, limit in FRAME_TOLERANCES.items()
        ):
            raise evidence.EvidenceError("Native Gaussian frame verification did not pass")
    native = read_native(objects.artifact(job, object_id, "master-splat.ply", True))
    placed = read_native(artifact(job, object_id, identifier, "placed-splat.ply"))
    if len(native) != result.get("gaussians") or not np.array_equal(placed, transform_rows(native, matrix)):
        raise evidence.EvidenceError("Native Gaussian export does not preserve the full fitted master")
    mesh = artifact(job, object_id, identifier, "appearance-delivery.glb")
    if (result.get("mesh_color", {}).get("policy") != generated_color.COLOR_POLICY
            or not glb_is_generative(mesh) or generated_color.decoded_rgba(mesh) is None):
        raise evidence.EvidenceError("Native Gaussian mesh fallback requires its explicit generated color policy")
    source_mesh = objects.artifact(job, object_id, "delivery.glb", True)
    if source_mesh.read_bytes() != mesh.read_bytes():
        original, original_binary = glb_transform._chunks(source_mesh.read_bytes())
        derived, derived_binary = glb_transform._chunks(mesh.read_bytes())
        color_ids = {primitive["attributes"]["COLOR_0"] for part in original["meshes"] for primitive in part["primitives"]}
        if (not derived_binary.startswith(original_binary)
                or any(original.get(key) != derived.get(key) for key in ("meshes", "nodes", "scenes", "scene"))
                or len(original["accessors"]) != len(derived["accessors"])
                or any(accessor != derived["accessors"][index] for index, accessor in enumerate(original["accessors"]) if index not in color_ids)):
            raise evidence.EvidenceError("Native Gaussian mesh fallback must preserve the fitted delivery geometry")
    return mesh, result["artifacts"]["appearance-delivery.glb"]["sha256"]


def attach(job, revision, entry, slug, object_id, identifier):
    result = read(job, object_id, identifier)
    key = f"_world/elements/{slug}.ply"
    revision["artifacts"][key] = result["artifacts"]["placed-splat.ply"]
    entry["files"]["splat"] = "artifact:" + key
    entry["appearance_source"] = "native-generated-gaussians"
    entry["gaussian_appearance"] = {"gaussians_id": identifier, "generated_object_id": object_id,
        "result_sha256": result["sha256"], "rows": result["gaussians"], "frame": "world-y-up-metres",
        "collision_source": "generated-delivery-mesh", "render_vr_only": True}
    entry["generated"]["delivery"] = result["artifacts"]["appearance-delivery.glb"]
    entry["classification"].append("native Gaussian appearance; separate inferred mesh collision; not a measured surface")
    prefix = f"_studio/generated/{object_id}/gaussians/{identifier}"
    revision["artifacts"][prefix + "/result.json"] = scenes.store_json(job, result)
    for name, identity in result["artifacts"].items():
        artifact(job, object_id, identifier, name)
        revision["artifacts"][prefix + "/" + name] = identity
