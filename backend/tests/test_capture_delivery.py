import json
import struct

import numpy as np
import pytest

import artifact_manifest as manifests
from capture_delivery import build_delivery, unlit_vertex_colors, verify_surface_glb


def glb(vertices, faces, change=None):
    positions = np.asarray(vertices, dtype="<f4").tobytes()
    indices = np.asarray(faces, dtype="<u4").tobytes()
    binary = positions + indices
    document = {"asset": {"version": "2.0"}, "nodes": [{"mesh": 0}], "scenes": [{"name": "Scene", "nodes": [0]}],
                "buffers": [{"byteLength": len(binary)}], "bufferViews": [
                    {"buffer": 0, "byteLength": len(positions)}, {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices)}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(vertices), "type": "VEC3"},
                              {"bufferView": 1, "componentType": 5125, "count": np.asarray(faces).size, "type": "SCALAR"}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 0}, "indices": 1}]}]}
    if change:
        change(document)
    metadata = json.dumps(document).encode()
    metadata += b" " * (-len(metadata) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(metadata) + len(binary))
            + struct.pack("<I4s", len(metadata), b"JSON") + metadata + struct.pack("<I4s", len(binary), b"BIN\0") + binary)


@pytest.fixture
def surface():
    return np.array([[0., 0., 0.], [1.25, 0., 0.], [0., 2.5, 0.], [0., 0., 3.75]]), np.array([[0, 1, 2], [0, 3, 1]])


def test_roundtrip_accepts_vertex_normal_splits_and_cyclic_triangle_order(surface):
    vertices, faces = surface
    split = vertices[faces[:, [1, 2, 0]].reshape(-1)]
    result = verify_surface_glb(glb(split, [[0, 1, 2], [3, 4, 5]]), vertices, faces)
    assert result["exported_vertices_with_normal_splits"] == 6
    assert result["source_vertices"] == 4
    assert result["oriented_triangles_equal"] is True
    assert result["registration"] is None


def test_unlit_material_fix_preserves_geometry_and_all_binary_vertex_colors(surface):
    vertices, faces = surface
    original = glb(vertices, faces)
    converted = unlit_vertex_colors(original)
    before_length = struct.unpack_from("<I", original, 12)[0]
    after_length = struct.unpack_from("<I", converted, 12)[0]
    assert original[20 + before_length:] == converted[20 + after_length:]
    material = json.loads(converted[20:20 + after_length])["materials"][0]
    assert material["extensions"] == {"KHR_materials_unlit": {}}
    assert material["pbrMetallicRoughness"]["baseColorFactor"] == [1, 1, 1, 1]
    assert "emissiveFactor" not in material
    assert verify_surface_glb(converted, vertices, faces)["oriented_triangles_equal"]
    with pytest.raises(ValueError, match="no vertex colors"):
        unlit_vertex_colors(glb(vertices, faces, lambda document: document["meshes"][0]["primitives"][0]["attributes"].pop("COLOR_0")))


@pytest.mark.parametrize("defect", ["coordinates", "winding", "indices", "truncated", "transform", "buffer", "count", "nonfinite"])
def test_roundtrip_rejects_changed_geometry_or_unsupported_storage(surface, defect):
    vertices, faces = surface
    exported, triangles = vertices.copy(), faces.copy()
    change = None
    if defect == "coordinates":
        exported[1, 0] += .01
    elif defect == "winding":
        triangles[0] = triangles[0, ::-1]
    elif defect == "indices":
        triangles[0, 0] = 100
    elif defect == "transform":
        change = lambda document: document["nodes"][0].update(translation=[1, 0, 0])
    elif defect == "buffer":
        change = lambda document: document["buffers"][0].update(uri="elsewhere.bin")
    elif defect == "count":
        change = lambda document: document["accessors"][0].update(count=1000000)
    elif defect == "nonfinite":
        exported[0, 0] = np.nan
    data = glb(exported, triangles, change)
    if defect == "truncated":
        data = data[:-1]
    with pytest.raises(ValueError):
        verify_surface_glb(data, vertices, faces)


@pytest.fixture
def package(tmp_path, surface):
    vertices, faces = surface
    extracted, blender = tmp_path / "surface", tmp_path / "blender"
    extracted.mkdir()
    blender.mkdir()
    np.savez(extracted / "surface-arrays.npz", vertices=vertices, faces=faces)
    manifests.atomic_write_json(extracted / "receipt.json", {"status": "inferred-surface-needs-review", "components": 1,
        "files": {"surface-arrays.npz": manifests.sha256_file(extracted / "surface-arrays.npz")}})
    (blender / "inferred-surface.glb").write_bytes(glb(vertices, faces))
    manifests.atomic_write_json(blender / "receipt.json", {"status": "created-needs-owner-review",
        "source_hashes": {str(extracted / "receipt.json"): manifests.sha256_file(extracted / "receipt.json")},
        "files": {"inferred-surface.glb": manifests.sha256_file(blender / "inferred-surface.glb")}})
    return blender, extracted


def test_delivery_seals_manifest_and_keeps_registration_unaccepted(package, tmp_path):
    blender, surface = package
    output = tmp_path / "delivery"
    result = build_delivery(blender, surface, output)
    asset = json.loads((output / "external-references.json").read_text())["assets"][0]
    assert asset["units"] == "arbitrary" and asset["kind"] == "mesh"
    assert asset["registration"] is None and asset["quality"]["ownerAccepted"] is False
    assert asset["sha256"] == manifests.sha256_file(output / asset["file"])
    assert result["published"] is False
    assert all(manifests.sha256_file(output / name) == digest for name, digest in result["files"].items())
    with pytest.raises(ValueError, match="new delivery"):
        build_delivery(blender, surface, output)


@pytest.mark.parametrize("defect", ["changed", "unsealed", "wrong-source"])
def test_delivery_refuses_unbound_or_tampered_export(package, tmp_path, defect):
    blender, surface = package
    receipt = json.loads((blender / "receipt.json").read_text())
    if defect == "changed":
        (blender / "inferred-surface.glb").write_bytes(b"changed")
    elif defect == "unsealed":
        receipt["files"] = {}
    else:
        receipt["source_hashes"] = {}
    manifests.atomic_write_json(blender / "receipt.json", receipt)
    with pytest.raises(ValueError):
        build_delivery(blender, surface, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
