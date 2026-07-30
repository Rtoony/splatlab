"""Uniform-scale + translate a self-contained GLB's geometry — stdlib only.

The Replace-with-asset flow needs a library asset (metre-scale, origin at
bottom-centre, identity transforms — the library contract) re-posed at a
captured element's spot WITHOUT a Blender round-trip: scale every POSITION
by one factor, then translate, and keep the accessor min/max honest. Normals
and tangents are direction data — untouched by uniform scale + translation.

Deliberately narrow: float32 VEC3 POSITION accessors in the embedded BIN
chunk, no sparse accessors, no external buffers. That is exactly what our
own normalized library exports contain; anything else is refused loudly
rather than silently mangled.
"""

from __future__ import annotations

import json
import struct

_MAGIC = b"glTF"
_JSON_TYPE = b"JSON"
_BIN_TYPE = b"BIN\x00"
_FLOAT32 = 5126


class GlbTransformError(ValueError):
    """The GLB cannot be safely transformed."""


def _chunks(data: bytes) -> tuple[dict, bytearray]:
    if len(data) < 20 or data[:4] != _MAGIC:
        raise GlbTransformError("not a GLB container")
    declared = struct.unpack_from("<I", data, 8)[0]
    if declared != len(data):
        raise GlbTransformError("container length mismatch")
    offset = 12
    doc: dict | None = None
    bin_chunk: bytearray | None = None
    while offset + 8 <= len(data):
        length, ctype = struct.unpack_from("<I4s", data, offset)
        payload = data[offset + 8:offset + 8 + length]
        if ctype == _JSON_TYPE and doc is None:
            doc = json.loads(payload)
        elif ctype == _BIN_TYPE and bin_chunk is None:
            bin_chunk = bytearray(payload)
        offset += 8 + length
    if doc is None or bin_chunk is None:
        raise GlbTransformError("missing JSON or BIN chunk")
    return doc, bin_chunk


def _assemble(doc: dict, bin_chunk: bytes) -> bytes:
    payload = json.dumps(doc, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    bin_padded = bytes(bin_chunk) + b"\x00" * (-len(bin_chunk) % 4)
    total = 12 + 8 + len(payload) + 8 + len(bin_padded)
    out = bytearray()
    out += _MAGIC + struct.pack("<II", 2, total)
    out += struct.pack("<I4s", len(payload), _JSON_TYPE) + payload
    out += struct.pack("<I4s", len(bin_padded), _BIN_TYPE) + bin_padded
    return bytes(out)


def translate_scale_glb(data: bytes, offset: tuple[float, float, float],
                        scale: float = 1.0) -> bytes:
    """POSITIONs -> position * scale + offset; accessor min/max updated.

    `scale` must be positive (a uniform positive scale keeps min/max mapping
    trivially and leaves normals/winding valid).
    """
    if not (scale > 0):
        raise GlbTransformError("scale must be > 0")
    doc, bin_chunk = _chunks(data)
    for buf in doc.get("buffers") or []:
        if "uri" in buf:
            raise GlbTransformError("external buffers are not supported")

    accessors = doc.get("accessors") or []
    views = doc.get("bufferViews") or []
    position_ids = sorted({
        prim["attributes"]["POSITION"]
        for mesh in doc.get("meshes") or []
        for prim in mesh.get("primitives") or []
        if "POSITION" in (prim.get("attributes") or {})})
    if not position_ids:
        raise GlbTransformError("no POSITION accessors")

    ox, oy, oz = (float(v) for v in offset)
    for aid in position_ids:
        acc = accessors[aid]
        if acc.get("componentType") != _FLOAT32 or acc.get("type") != "VEC3":
            raise GlbTransformError("POSITION must be float32 VEC3")
        if "sparse" in acc:
            raise GlbTransformError("sparse POSITION accessors are not supported")
        view = views[acc["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise GlbTransformError("POSITION outside the BIN chunk")
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or 12
        count = acc["count"]
        if base + (count - 1) * stride + 12 > len(bin_chunk):
            raise GlbTransformError("POSITION accessor overruns the BIN chunk")
        for i in range(count):
            at = base + i * stride
            x, y, z = struct.unpack_from("<fff", bin_chunk, at)
            struct.pack_into("<fff", bin_chunk, at,
                             x * scale + ox, y * scale + oy, z * scale + oz)
        for key, delta in (("min", (ox, oy, oz)), ("max", (ox, oy, oz))):
            old = acc.get(key)
            if isinstance(old, list) and len(old) == 3:
                acc[key] = [round(float(v) * scale + d, 6)
                            for v, d in zip(old, delta)]
    return _assemble(doc, bin_chunk)
