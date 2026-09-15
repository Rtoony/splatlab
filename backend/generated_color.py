"""Derive linear-float glTF colors from explicitly interpreted SAM image-RGB bytes."""

from pathlib import Path

import numpy as np

import glb_check
import glb_transform
import reconstruction_evidence as evidence
from mesh.provenance import glb_is_generative

COLOR_POLICY = "sam-image-srgb-to-gltf-linear-float/v1"


def decoded_rgba(path):
    document, binary = glb_transform._chunks(Path(path).read_bytes())
    if document["asset"].get("extras", {}).get("splatlab_color_policy") != COLOR_POLICY:
        return None
    color_ids = {primitive["attributes"]["COLOR_0"] for mesh in document["meshes"] for primitive in mesh["primitives"]}
    if len(color_ids) != 1:
        raise evidence.EvidenceError("Fixed-camera color replay requires one shared generated color accessor")
    accessor = document["accessors"][color_ids.pop()]
    view = document["bufferViews"][accessor["bufferView"]]
    if accessor["componentType"] != 5126 or accessor["type"] != "VEC4" or accessor.get("normalized") or "sparse" in accessor or view.get("byteStride", 16) != 16:
        raise evidence.EvidenceError("Invalid linear-float generated color policy")
    linear = np.frombuffer(binary, dtype="<f4", count=accessor["count"] * 4,
                           offset=view.get("byteOffset", 0) + accessor.get("byteOffset", 0)).reshape(-1, 4)
    if not np.isfinite(linear).all() or np.any(linear < 0) or np.any(linear > 1):
        raise evidence.EvidenceError("Invalid generated linear colors")
    rgba = linear.astype(float).copy()
    rgba[:, :3] = linear_to_srgb(rgba[:, :3])
    return np.rint(rgba * 255).astype(np.uint8)


def srgb_to_linear(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise evidence.EvidenceError("Color channels must be finite and within [0,1]")
    return np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)


def linear_to_srgb(values):
    values = np.asarray(values, dtype=float)
    return np.where(values <= .0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - .055)


def derive(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > evidence.MAX_FILE_BYTES:
        raise evidence.EvidenceError("Color source must be a bounded regular GLB")
    glb_check.validate_glb(source)
    if not glb_is_generative(source):
        raise evidence.EvidenceError("Color interpretation only applies to tagged generated assets")
    document, binary = glb_transform._chunks(source.read_bytes())
    original_binary = bytes(binary)
    extras = document["asset"].setdefault("extras", {})
    if not isinstance(extras, dict) or "splatlab_color_policy" in extras:
        raise evidence.EvidenceError("Color policy is already applied or cannot be verified")
    if document.get("materials") or document.get("textures") or document.get("images") or len(document.get("buffers", [])) != 1:
        raise evidence.EvidenceError("This SAM color adapter requires untextured vertex-color-only geometry")
    color_ids = {primitive.get("attributes", {}).get("COLOR_0") for mesh in document["meshes"] for primitive in mesh["primitives"]}
    if not color_ids or None in color_ids:
        raise evidence.EvidenceError("Every generated primitive needs vertex colors")
    records = []
    for identifier in sorted(color_ids):
        accessor = document["accessors"][identifier]
        if accessor.get("componentType") != 5121 or accessor.get("normalized") is not True or accessor.get("type") != "VEC4" or "sparse" in accessor:
            raise evidence.EvidenceError("Expected original normalized RGBA8 SAM colors; refusing a different encoding")
        view = document["bufferViews"][accessor["bufferView"]]
        count = accessor["count"]
        offset, stride = accessor.get("byteOffset", 0), view.get("byteStride", 4)
        if (view.get("buffer", 0) != 0 or not 1 <= count <= 2_000_000 or stride < 4 or offset < 0
                or offset + (count - 1) * stride + 4 > view["byteLength"]
                or view.get("byteOffset", 0) + view["byteLength"] > len(original_binary)):
            raise evidence.EvidenceError("Generated color accessor is out of bounds")
        colors = np.ndarray((count, 4), dtype=np.uint8, buffer=original_binary, offset=view.get("byteOffset", 0) + offset, strides=(stride, 1))
        linear = colors.astype(float) / 255.
        linear[:, :3] = srgb_to_linear(linear[:, :3])
        linear = linear.astype("<f4")
        decoded = np.rint(linear_to_srgb(linear[:, :3]) * 255).astype(np.uint8)
        np.testing.assert_array_equal(decoded, colors[:, :3])
        np.testing.assert_array_equal(np.rint(linear[:, 3] * 255).astype(np.uint8), colors[:, 3])
        binary.extend(b"\x00" * (-len(binary) % 4))
        new_view = len(document["bufferViews"])
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": linear.nbytes})
        binary.extend(linear.tobytes())
        accessor.update(componentType=5126, normalized=False, byteOffset=0, bufferView=new_view,
                        min=linear.min(0).tolist(), max=linear.max(0).tolist())
        records.append({"accessor": identifier, "vertices": count, "rgb8_roundtrip_exact": True, "alpha_roundtrip_exact": True})
    assert bytes(binary[:len(original_binary)]) == original_binary
    document["buffers"][0]["byteLength"] = len(binary)
    extras["splatlab_color_policy"] = COLOR_POLICY
    if len(binary) > evidence.MAX_FILE_BYTES:
        raise evidence.EvidenceError("Derived color artifact exceeds its bound")
    with destination.open("xb") as handle:
        handle.write(glb_transform._assemble(document, binary))
    glb_check.validate_glb(destination)
    if glb_check.position_bounds(source) != glb_check.position_bounds(destination) or not glb_is_generative(destination):
        raise evidence.EvidenceError("Color derivation changed spatial bounds or provenance")
    return {"policy": COLOR_POLICY, "accessors": records, "original_bin_prefix_byte_exact": True,
            "scope": "explicit sRGB interpretation of generated image-like vertex colors; not measured albedo or relighting",
            "gltf_standard": "https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes"}
