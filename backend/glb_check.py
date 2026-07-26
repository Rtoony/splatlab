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
