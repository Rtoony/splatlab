"""Artifact inspection and manifest helpers for SplatLab interchange exports.

The canonical ``_preview/splat.ply`` stays the source of truth. Derived formats
record its file identity and checksum so callers can reject stale artifacts
after a crop, semantic edit, merge, or revert.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPORT_SCHEMA = "dev.splatlab.exports/v1"
UNREAL_SCHEMA = "dev.splatlab.unreal-bundle/v1"

_PLY_SCALARS: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


class ArtifactManifestError(ValueError):
    """An artifact or manifest cannot be inspected safely."""


@dataclass(frozen=True)
class PlyHeader:
    encoding: str
    version: str
    vertex_count: int
    vertex_properties: tuple[tuple[str, str], ...]
    header_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }
    if include_sha256:
        record["sha256"] = sha256_file(path)
    return record


def same_file_identity(path: Path, record: dict[str, Any] | None) -> bool:
    if not record or not path.is_file():
        return False
    stat = path.stat()
    return (
        record.get("bytes") == stat.st_size
        and record.get("mtime_ns") == stat.st_mtime_ns
        and record.get("inode") == stat.st_ino
    )


def read_ply_header(path: Path) -> PlyHeader:
    if not path.is_file():
        raise ArtifactManifestError(f"PLY not found: {path}")

    encoding = ""
    version = ""
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element = ""

    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != b"ply":
            raise ArtifactManifestError(f"Not a PLY file: {path}")
        while handle.tell() <= 1024 * 1024:
            raw = handle.readline()
            if not raw:
                raise ArtifactManifestError(f"PLY header has no end_header: {path}")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ArtifactManifestError(f"PLY header is not ASCII: {path}") from exc
            parts = line.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format" and len(parts) == 3:
                encoding, version = parts[1], parts[2]
            elif parts[0] == "element" and len(parts) == 3:
                current_element = parts[1]
                if current_element == "vertex":
                    try:
                        vertex_count = int(parts[2])
                    except ValueError as exc:
                        raise ArtifactManifestError("Invalid PLY vertex count") from exc
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) == 3:
                    scalar_type, name = parts[1], parts[2]
                    if scalar_type not in _PLY_SCALARS:
                        raise ArtifactManifestError(
                            f"Unsupported PLY vertex scalar type: {scalar_type}"
                        )
                    vertex_properties.append((name, scalar_type))
                elif len(parts) >= 2 and parts[1] == "list":
                    raise ArtifactManifestError(
                        "List-valued PLY vertex properties are unsupported"
                    )
            elif parts[0] == "end_header":
                if not encoding or vertex_count is None:
                    raise ArtifactManifestError(
                        "PLY header is missing format or vertex element"
                    )
                if encoding not in {
                    "ascii",
                    "binary_little_endian",
                    "binary_big_endian",
                }:
                    raise ArtifactManifestError(f"Unsupported PLY encoding: {encoding}")
                return PlyHeader(
                    encoding=encoding,
                    version=version,
                    vertex_count=vertex_count,
                    vertex_properties=tuple(vertex_properties),
                    header_bytes=handle.tell(),
                )

    raise ArtifactManifestError(f"PLY header exceeds 1 MiB: {path}")


def _sh_degree(properties: tuple[tuple[str, str], ...]) -> int | None:
    rest = sum(1 for name, _kind in properties if name.startswith("f_rest_"))
    if rest == 0:
        return (
            0 if any(name.startswith("f_dc_") for name, _kind in properties) else None
        )
    if rest % 3:
        return None
    coefficients_per_channel = rest // 3 + 1
    root = math.isqrt(coefficients_per_channel)
    return root - 1 if root * root == coefficients_per_channel else None


def _bounds_from_ascii(path: Path, header: PlyHeader) -> dict[str, list[float]] | None:
    names = [name for name, _kind in header.vertex_properties]
    try:
        xyz_indices = tuple(names.index(axis) for axis in ("x", "y", "z"))
    except ValueError:
        return None

    low = [math.inf, math.inf, math.inf]
    high = [-math.inf, -math.inf, -math.inf]
    valid = 0
    with path.open("rb") as handle:
        handle.seek(header.header_bytes)
        for _index in range(header.vertex_count):
            raw = handle.readline()
            if not raw:
                raise ArtifactManifestError("PLY ended before all vertices were read")
            values = raw.split()
            try:
                xyz = [float(values[i]) for i in xyz_indices]
            except (IndexError, ValueError) as exc:
                raise ArtifactManifestError("Invalid ASCII PLY vertex row") from exc
            if not all(math.isfinite(value) for value in xyz):
                continue
            valid += 1
            for axis, value in enumerate(xyz):
                low[axis] = min(low[axis], value)
                high[axis] = max(high[axis], value)
    if not valid:
        return None
    return {
        "min": low,
        "max": high,
        "extent": [high[i] - low[i] for i in range(3)],
    }


def _bounds_from_binary(path: Path, header: PlyHeader) -> dict[str, list[float]] | None:
    names = [name for name, _kind in header.vertex_properties]
    try:
        xyz_indices = tuple(names.index(axis) for axis in ("x", "y", "z"))
    except ValueError:
        return None

    byte_offsets: list[int] = []
    offset = 0
    for _name, scalar_type in header.vertex_properties:
        byte_offsets.append(offset)
        offset += _PLY_SCALARS[scalar_type][1]
    record_bytes = offset
    expected = header.header_bytes + header.vertex_count * record_bytes
    if path.stat().st_size < expected:
        raise ArtifactManifestError("Binary PLY ended before all vertices were read")

    endian = "<" if header.encoding == "binary_little_endian" else ">"
    xyz_unpackers = [
        struct.Struct(endian + _PLY_SCALARS[header.vertex_properties[i][1]][0])
        for i in xyz_indices
    ]
    xyz_offsets = [byte_offsets[i] for i in xyz_indices]
    low = [math.inf, math.inf, math.inf]
    high = [-math.inf, -math.inf, -math.inf]
    valid = 0

    with (
        path.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data,
    ):
        row = header.header_bytes
        for _index in range(header.vertex_count):
            xyz = [
                xyz_unpackers[axis].unpack_from(data, row + xyz_offsets[axis])[0]
                for axis in range(3)
            ]
            row += record_bytes
            if not all(math.isfinite(value) for value in xyz):
                continue
            valid += 1
            for axis, value in enumerate(xyz):
                low[axis] = min(low[axis], value)
                high[axis] = max(high[axis], value)
    if not valid:
        return None
    return {
        "min": low,
        "max": high,
        "extent": [high[i] - low[i] for i in range(3)],
    }


def inspect_ply(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    header = read_ply_header(path)
    bounds = (
        _bounds_from_ascii(path, header)
        if header.encoding == "ascii"
        else _bounds_from_binary(path, header)
    )
    return {
        "path": str(path),
        **file_identity(path, include_sha256=include_sha256),
        "ply": {
            "encoding": header.encoding,
            "version": header.version,
            "vertex_properties": [name for name, _kind in header.vertex_properties],
        },
        "gaussian_count": header.vertex_count,
        "sh_degree": _sh_degree(header.vertex_properties),
        "bounds_scene": bounds,
    }


def artifact_file_record(path: Path, root: Path, *, media_type: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactManifestError(f"Artifact escapes export root: {path}") from exc
    return {
        "path": relative.as_posix(),
        "media_type": media_type,
        **file_identity(path),
    }


def coordinate_record(meta: dict[str, Any]) -> dict[str, Any]:
    meters_per_unit = meta.get("meters_per_unit")
    calibrated = (
        not isinstance(meters_per_unit, bool)
        and isinstance(meters_per_unit, int | float)
        and math.isfinite(meters_per_unit)
        and meters_per_unit > 0
    )
    return {
        "source_frame": {
            "handedness": "right-handed",
            "right_axis": "+X",
            "forward_axis": "+Y",
            "up_axis": "+Z",
            "units": "scene-units",
            "meters_per_unit": float(meters_per_unit) if calibrated else None,
            "scale_calibrated": calibrated,
        },
        "georeference": meta.get("geo"),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def relative_job_path(path: Path, job_dir: Path) -> str:
    try:
        return path.resolve().relative_to(job_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ArtifactManifestError(f"Path escapes job directory: {path}") from exc
