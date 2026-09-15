"""Verify Blender mesh round trips and package unregistered condo references."""

import json
from pathlib import Path
import struct

import numpy as np

import artifact_manifest as manifests
from reference_delivery import artifact_path


def verify_surface_glb(data, vertices, faces):
    if len(data) < 28 or struct.unpack_from("<4sII", data) != (b"glTF", 2, len(data)):
        raise ValueError("Invalid GLB header")
    length, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON" or length % 4 or 28 + length > len(data):
        raise ValueError("Invalid GLB JSON chunk")
    document = json.loads(data[20:20 + length])
    size, kind = struct.unpack_from("<I4s", data, 20 + length)
    binary = data[28 + length:]
    if kind != b"BIN\0" or size != len(binary):
        raise ValueError("Require one embedded GLB buffer")
    nodes = document.get("nodes", [])
    if (len(nodes) != 1 or nodes[0].get("mesh") != 0
            or any(key in nodes[0] for key in ("matrix", "translation", "rotation", "scale", "children", "skin"))
            or len(document.get("meshes", [])) != 1 or document.get("extensionsRequired")
            or document.get("animations") or len(document.get("buffers", [])) != 1
            or "uri" in document["buffers"][0]
            or document.get("scenes") != [{"name": "Scene", "nodes": [0]}]):
        raise ValueError("Require the single identity-basis Blender surface export")
    primitives = document["meshes"][0]["primitives"]
    if len(primitives) != 1 or primitives[0].get("mode", 4) != 4 or primitives[0].get("targets"):
        raise ValueError("Require one uncompressed triangle primitive")

    def accessor(index, components, component_type):
        record = document["accessors"][index]
        view = document["bufferViews"][record["bufferView"]]
        if (record["componentType"] != component_type or record["type"] != ("VEC3" if components == 3 else "SCALAR")
                or record.get("sparse") or record.get("normalized") or view.get("buffer", 0) != 0
                or view.get("byteStride", components * 4) != components * 4):
            raise ValueError("Unsupported surface accessor")
        offset = view.get("byteOffset", 0) + record.get("byteOffset", 0)
        size = record["count"] * components * 4
        if (record["count"] <= 0 or offset < 0 or record.get("byteOffset", 0) < 0
                or record.get("byteOffset", 0) + size > view["byteLength"] or offset + size > len(binary)):
            raise ValueError("Surface accessor exceeds its buffer")
        return np.frombuffer(binary, dtype="<f4" if component_type == 5126 else "<u4",
                             offset=offset, count=record["count"] * components).reshape(-1, components)

    positions = accessor(primitives[0]["attributes"]["POSITION"], 3, 5126)
    indices = accessor(primitives[0]["indices"], 1, 5125).reshape(-1)
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices)
            or not np.isfinite(positions).all() or not np.isfinite(vertices).all()
            or indices.size != faces.size or indices.max() >= len(positions)):
        raise ValueError("Surface dimensions or coordinates differ")

    def coordinate_keys(values):
        values = np.array(values, dtype="<f4", order="C", copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Coordinates overflow float32")
        values[values == 0] = 0
        return values.view("V12").reshape(-1)

    source_keys, source_ids = np.unique(coordinate_keys(vertices), return_inverse=True)
    exported_keys = coordinate_keys(positions)
    exported_ids = np.searchsorted(source_keys, exported_keys)
    if np.any(exported_ids >= len(source_keys)) or not np.array_equal(source_keys[exported_ids], exported_keys):
        raise ValueError("GLB coordinates do not round-trip to the original SfM surface")

    def oriented_triangles(triangles):
        start = np.argmin(triangles, axis=1)
        rotated = np.take_along_axis(triangles, (start[:, None] + np.arange(3)) % 3, axis=1)
        return rotated[np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))]

    if not np.array_equal(oriented_triangles(source_ids[faces]), oriented_triangles(exported_ids[indices.reshape(-1, 3)])):
        raise ValueError("GLB triangle topology or winding differs from the original surface")
    return {"status": "passed", "source_vertices": len(vertices), "exported_vertices_with_normal_splits": len(positions),
            "triangles": len(faces), "coordinates_equal_after_float32_export": True, "oriented_triangles_equal": True,
            "registration": None, "scope": "Serialization round trip, not physical or architectural accuracy"}


def unlit_vertex_colors(data):
    length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + length])
    primitive = document["meshes"][0]["primitives"][0]
    if "COLOR_0" not in primitive["attributes"]:
        raise ValueError("Captured mesh has no vertex colors")
    primitive["material"] = 0
    document["materials"] = [{"name": "Captured vertex colors — unlit review", "doubleSided": True,
                              "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
                              "extensions": {"KHR_materials_unlit": {}}}]
    document["extensionsUsed"] = sorted(set(document.get("extensionsUsed", [])) | {"KHR_materials_unlit"})
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary_chunk = data[20 + length:]
    return (struct.pack("<4sII", b"glTF", 2, 20 + len(encoded) + len(binary_chunk))
            + struct.pack("<I4s", len(encoded), b"JSON") + encoded + binary_chunk)


def build_delivery(blender, surface, output):
    blender, surface, output = (Path(path).resolve() for path in (blender, surface, output))
    if output.exists() or any(output.is_relative_to(path) for path in (blender, surface)):
        raise ValueError("Choose a new delivery outside its sources")
    snapshot, receipts = {}, {}
    for directory in (blender, surface):
        receipt_path = artifact_path(directory, "receipt.json")
        receipts[directory] = json.loads(receipt_path.read_text())
        snapshot[receipt_path] = manifests.sha256_file(receipt_path)
        for name, digest in receipts[directory]["files"].items():
            snapshot[artifact_path(directory, name)] = digest
    if (receipts[blender].get("status") != "created-needs-owner-review"
            or receipts[surface].get("status") != "inferred-surface-needs-review"
            or not any(Path(name).resolve() == surface / "receipt.json" and digest == snapshot[surface / "receipt.json"]
                       for name, digest in receipts[blender]["source_hashes"].items())
            or any(path not in snapshot for path in (blender / "inferred-surface.glb", surface / "surface-arrays.npz"))):
        raise ValueError("Require a sealed Blender export of this completed surface")

    def verify():
        if any(manifests.sha256_file(path) != digest for path, digest in snapshot.items()):
            raise ValueError("Source changed during capture delivery")

    verify()
    with np.load(surface / "surface-arrays.npz", allow_pickle=False) as arrays:
        data = (blender / "inferred-surface.glb").read_bytes()
        verify_surface_glb(data, arrays["vertices"], arrays["faces"])
        delivered = unlit_vertex_colors(data)
        proof = verify_surface_glb(delivered, arrays["vertices"], arrays["faces"])
        proof["material_conversion"] = "KHR_materials_unlit with original vertex COLOR_0; original geometry/buffer unchanged, no baked lighting or invented texture"
    output.mkdir(parents=True)
    (output / "inferred-surface.glb").write_bytes(delivered)
    manifests.atomic_write_json(output / "coordinate-proof.json", proof)
    asset = {"id": output.name, "kind": "mesh", "file": "inferred-surface.glb",
             "sha256": manifests.sha256_file(output / "inferred-surface.glb"),
             "producer": {"tool": "SplatLab native surface delivery", "runId": surface.name},
             "sourceHashes": [snapshot[blender / "receipt.json"], snapshot[surface / "receipt.json"], snapshot[surface / "surface-arrays.npz"]],
             "units": "arbitrary", "coordinates": "Original COLMAP world numbers, right-handed. GLB triangle coordinates verified against source float32 surface, including winding. Metric scale and world up unverified.",
             "quality": {**proof, "components": receipts[surface]["components"], "ownerAccepted": False,
                         "limitations": ["Inferred training-depth TSDF; fragmented and not a building solid", "Not a survey, collider or accepted condo replacement", "No canonical condo alignment; registration requires reviewed controls"]},
             "registration": None}
    manifests.atomic_write_json(output / "external-references.json", {"schema": "chanate-external-references/v1", "assets": [asset]})
    verify()
    receipt = {"schema": "dev.splatlab.capture-mesh-delivery/v1", "status": "prepared-needs-registration-and-owner-review",
               "registration": None, "published": False, "source_hashes": {str(path): digest for path, digest in snapshot.items()},
               "implementation_sha256": manifests.sha256_file(Path(__file__)), "created_at": manifests.utc_now(),
               "files": {path.name: manifests.sha256_file(path) for path in output.iterdir()}}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    return receipt
