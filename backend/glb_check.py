"""Stdlib-only structural validation for GLB (binary glTF 2.0) uploads.

Why not a library: Open3D ABORTS the whole process (uncatchable C++ terminate)
on image-less GLBs and on empty texture placeholders — measured 2026-07-26 —
and trimesh is deliberately absent from this app's venv. An upload gate must be
able to reject garbage without ever being able to take the service down, which
is exactly what a bounds-checked header + JSON-chunk parse gives.

This validates STRUCTURE (magic, chunk layout, asset version, non-empty mesh
list, buffer/BIN consistency), not geometry. A structurally valid GLB with
nonsense triangles is the uploader's own polish decision; a truncated or
mislabeled file is not.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

GLB_MAGIC = b"glTF"
CHUNK_JSON = b"JSON"
CHUNK_BIN = b"BIN\x00"
_HEADER_LEN = 12
_CHUNK_HEADER_LEN = 8
# A polish upload has no business being a multi-GB monster; the route applies
# its own byte cap before this runs, so this is only a sanity backstop against
# absurd declared chunk lengths in a small file.
_MAX_JSON_CHUNK = 256 * 1024 * 1024


def validate_glb(path: Path) -> dict[str, Any]:
    """Raise ValueError with the precise defect, or return a summary dict."""
    size = path.stat().st_size
    if size < _HEADER_LEN + _CHUNK_HEADER_LEN:
        raise ValueError(f"file is {size} bytes — too small to be a GLB")

    with path.open("rb") as handle:
        magic, version, declared = struct.unpack("<4sII", handle.read(_HEADER_LEN))
        if magic != GLB_MAGIC:
            raise ValueError("not a GLB: missing glTF magic bytes")
        if version != 2:
            raise ValueError(f"unsupported GLB container version {version} (need 2)")
        if declared != size:
            raise ValueError(
                f"declared length {declared} does not match file size {size} "
                "— truncated or padded file"
            )

        chunks: list[tuple[bytes, int, int]] = []  # (type, offset, length)
        offset = _HEADER_LEN
        while offset + _CHUNK_HEADER_LEN <= size:
            handle.seek(offset)
            chunk_len, chunk_type = struct.unpack(
                "<I4s", handle.read(_CHUNK_HEADER_LEN)
            )
            data_start = offset + _CHUNK_HEADER_LEN
            if data_start + chunk_len > size:
                raise ValueError(
                    f"chunk {chunk_type!r} declares {chunk_len} bytes but the file "
                    "ends first — truncated"
                )
            chunks.append((chunk_type, data_start, chunk_len))
            offset = data_start + chunk_len

        if not chunks or chunks[0][0] != CHUNK_JSON:
            raise ValueError("first GLB chunk must be JSON")
        json_len = chunks[0][2]
        if json_len > _MAX_JSON_CHUNK:
            raise ValueError(f"JSON chunk of {json_len} bytes is not plausible")
        handle.seek(chunks[0][1])
        try:
            doc = json.loads(handle.read(json_len))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON chunk does not parse: {exc}") from exc

    asset = doc.get("asset")
    gltf_version = asset.get("version") if isinstance(asset, dict) else None
    if not isinstance(gltf_version, str) or not gltf_version.startswith("2."):
        raise ValueError(f"asset.version {gltf_version!r} is not glTF 2.x")
    meshes = doc.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise ValueError("glTF declares no meshes — nothing to render")

    # SELF-CONTAINMENT: a validated GLB is later handed to full glTF loaders
    # (Blender's importer via import_world_element, trimesh in the mesh env)
    # which happily dereference external buffer/image URIs — so an uploaded
    # file with `"uri": "/etc/..."` or an http reference would make a
    # server-side process read an arbitrary local or remote resource and
    # potentially embed it into a later export. Only data: URIs are inert.
    for section in ("buffers", "images"):
        entries = doc.get(section)
        if not isinstance(entries, list):
            continue
        for index, entry in enumerate(entries):
            uri = entry.get("uri") if isinstance(entry, dict) else None
            if uri is None:
                continue
            if not isinstance(uri, str) or not uri.startswith("data:"):
                raise ValueError(
                    f"{section}[{index}] references an external uri — a GLB "
                    "must be self-contained (BIN chunk or data: URIs only)"
                )

    bin_chunks = [c for c in chunks if c[0] == CHUNK_BIN]
    bin_len = bin_chunks[0][2] if bin_chunks else None
    buffers = doc.get("buffers")
    if isinstance(buffers, list) and buffers:
        first = buffers[0] if isinstance(buffers[0], dict) else {}
        if "uri" not in first:
            declared_bytes = first.get("byteLength")
            if not isinstance(declared_bytes, int) or declared_bytes < 0:
                raise ValueError("buffer 0 has no uri and no valid byteLength")
            if bin_len is None:
                raise ValueError(
                    "buffer 0 expects the binary chunk but the GLB has none"
                )
            # The BIN chunk may be padded up to 3 bytes past the buffer length.
            if not declared_bytes <= bin_len <= declared_bytes + 3:
                raise ValueError(
                    f"binary chunk is {bin_len} bytes but buffer 0 declares "
                    f"{declared_bytes} — mismatched or corrupt"
                )

    return {
        "gltf_version": gltf_version,
        "meshes": len(meshes),
        "nodes": len(doc.get("nodes") or []),
        "materials": len(doc.get("materials") or []),
        "images": len(doc.get("images") or []),
        "generator": (asset or {}).get("generator"),
        "json_bytes": json_len,
        "bin_bytes": bin_len,
    }


def _json_chunk(path: Path) -> dict[str, Any]:
    """The GLB's JSON document, minimally parsed. Callers run validate_glb
    first for the finely-worded structural errors; this only re-reads."""
    with path.open("rb") as handle:
        magic, _version, _declared = struct.unpack("<4sII", handle.read(_HEADER_LEN))
        if magic != GLB_MAGIC:
            raise ValueError("not a GLB: missing glTF magic bytes")
        chunk_len, chunk_type = struct.unpack(
            "<I4s", handle.read(_CHUNK_HEADER_LEN))
        if chunk_type != CHUNK_JSON:
            raise ValueError("first GLB chunk must be JSON")
        if chunk_len > _MAX_JSON_CHUNK:
            raise ValueError(f"JSON chunk of {chunk_len} bytes is not plausible")
        return json.loads(handle.read(chunk_len))


def _node_has_transform(node: dict[str, Any]) -> bool:
    matrix = node.get("matrix")
    if matrix is not None:
        identity = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
        return not (isinstance(matrix, list) and len(matrix) == 16
                    and all(abs(float(a) - b) <= 1e-9
                            for a, b in zip(matrix, identity)))
    for key, default in (("translation", (0.0, 0.0, 0.0)),
                         ("rotation", (0.0, 0.0, 0.0, 1.0)),
                         ("scale", (1.0, 1.0, 1.0))):
        value = node.get(key)
        if value is not None and any(abs(float(a) - b) > 1e-9
                                     for a, b in zip(value, default)):
            return True
    return False


def position_bounds(path: Path) -> dict[str, Any]:
    """POSITION-accessor bounds union + node-transform identity, from the
    JSON chunk alone — stdlib, no geometry decode.

    The shared-frame contract (world-walker.ts header) requires world-space-
    baked vertices under IDENTITY node transforms: reoriginObject and the
    manifest AABB path silently mis-place a node-TRS prop. glTF 2.0 requires
    min/max on POSITION accessors, so the bounds come free — and they equal
    world bounds exactly when the transforms are identity, which is what
    `identity_transforms` reports (a mesh under any non-identity node, own
    or inherited, lands in `transformed_nodes`).
    """
    doc = _json_chunk(path)
    accessors = doc.get("accessors") or []
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    found = False
    for mesh in doc.get("meshes") or []:
        for prim in (mesh or {}).get("primitives") or []:
            index = (prim.get("attributes") or {}).get("POSITION")
            if not isinstance(index, int) or not 0 <= index < len(accessors):
                continue
            accessor = accessors[index] if isinstance(accessors[index], dict) else {}
            amin, amax = accessor.get("min"), accessor.get("max")
            if (not isinstance(amin, list) or not isinstance(amax, list)
                    or len(amin) != 3 or len(amax) != 3):
                raise ValueError(
                    "POSITION accessor lacks 3-component min/max — the "
                    "exporter violated glTF 2.0")
            try:
                cur_min = [float(v) for v in amin]
                cur_max = [float(v) for v in amax]
            except (TypeError, ValueError) as exc:
                raise ValueError("POSITION min/max are not numbers") from exc
            if any(not math.isfinite(v) for v in cur_min + cur_max):
                raise ValueError("POSITION min/max are not finite")
            found = True
            lo = [min(a, b) for a, b in zip(lo, cur_min)]
            hi = [max(a, b) for a, b in zip(hi, cur_max)]
    if not found:
        raise ValueError("no POSITION data in any mesh — nothing to place")

    nodes = doc.get("nodes") or []
    child_of: set[int] = set()
    for node in nodes:
        for child in (node or {}).get("children") or []:
            if isinstance(child, int):
                child_of.add(child)
    transformed: list[int] = []
    visited: set[int] = set()
    stack = [(index, False) for index in range(len(nodes))
             if index not in child_of]
    while stack:
        index, inherited = stack.pop()
        if index in visited or not 0 <= index < len(nodes):
            continue  # visited-guard also bounds a malicious node cycle
        visited.add(index)
        node = nodes[index] if isinstance(nodes[index], dict) else {}
        tainted = inherited or _node_has_transform(node)
        if tainted and isinstance(node.get("mesh"), int):
            transformed.append(index)
        for child in node.get("children") or []:
            if isinstance(child, int):
                stack.append((child, tainted))

    return {
        "aabb": {"min": [round(v, 4) for v in lo],
                 "max": [round(v, 4) for v in hi]},
        "extent": [round(h - l, 4) for l, h in zip(lo, hi)],
        "identity_transforms": not transformed,
        "transformed_nodes": sorted(transformed),
    }
