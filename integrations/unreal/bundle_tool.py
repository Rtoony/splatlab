#!/usr/bin/env python3
"""Verify and stage SplatLab handoff bundles without importing Unreal APIs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse


BUNDLE_SCHEMA = "dev.splatlab.unreal-bundle/v1"
POLICY_SCHEMA = "dev.splatlab.unreal-mcp-policy/v1"
RENDERER_SCHEMA = "dev.splatlab.unreal-renderers/v1"
TARGET_VERSION = "5.6"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 20_000
MAX_ARCHIVE_BYTES = 256 * 1024**3
MAX_SINGLE_FILE_BYTES = 128 * 1024**3
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DANGEROUS_SUFFIXES = {
    ".bat",
    ".cmd",
    ".com",
    ".dll",
    ".exe",
    ".hta",
    ".lnk",
    ".msi",
    ".ps1",
    ".reg",
    ".scr",
    ".vbs",
}
WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
REQUIRED_DENIED_TOOLS = {"system_control", "manage_tools"}
REQUIRED_DENIED_ACTIONS = {
    "build",
    "delete",
    "execute_console_command",
    "execute_python",
    "package",
    "run_ubt",
}


class BundleToolError(ValueError):
    """A bundle, renderer descriptor, or policy violates the staging contract."""


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    gaussian_count: int | None
    total_bytes: int
    verified_files: int

    def summary(self) -> dict[str, Any]:
        coordinate = self.manifest["coordinate_system"]
        contract = self.manifest["import_contract"]
        return {
            "status": "verified",
            "schema": self.manifest["schema"],
            "job_id": self.manifest["job_id"],
            "manifest_sha256": self.manifest_sha256,
            "verified_files": self.verified_files,
            "total_bytes": self.total_bytes,
            "gaussian_count": self.gaussian_count,
            "coordinate_system": coordinate,
            "import_contract": contract,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, max_bytes: int = MAX_MANIFEST_BYTES) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BundleToolError(f"JSON file is missing or unsafe: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise BundleToolError(f"JSON file exceeds {max_bytes} bytes: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleToolError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise BundleToolError(f"Expected a JSON object: {path}")
    return value


def _safe_relative_path(raw: Any) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise BundleToolError("Manifest file path must be a non-empty string")
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not path.parts
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BundleToolError(f"Unsafe bundle path: {raw}")
    if len(normalized) > 240 or any(
        part.endswith((" ", "."))
        or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
        for part in path.parts
    ):
        raise BundleToolError(f"Bundle path is unsafe on Windows: {raw}")
    if path.suffix.lower() in DANGEROUS_SUFFIXES:
        raise BundleToolError(f"Executable content is not allowed in a bundle: {raw}")
    return path


def _contained_file(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise BundleToolError(f"Bundle path contains a symlink: {relative}")
    if not candidate.is_file():
        raise BundleToolError(f"Bundle file is missing or is a symlink: {relative}")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise BundleToolError(f"Bundle file escapes its root: {relative}") from exc
    return resolved


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise BundleToolError(f"{name} must be a positive number")
    return float(value)


def _validate_contract(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise BundleToolError(f"Unsupported bundle schema: {manifest.get('schema')!r}")
    job_id = manifest.get("job_id")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise BundleToolError("Bundle job_id is not safe for a staging directory")

    target = manifest.get("target")
    if not isinstance(target, dict):
        raise BundleToolError("Bundle target is missing")
    expected_target = {
        "engine": "Unreal Engine",
        "version": TARGET_VERSION,
        "platform": "Windows",
    }
    for name, expected in expected_target.items():
        if target.get(name) != expected:
            raise BundleToolError(
                f"Bundle target {name} must be {expected!r}, got {target.get(name)!r}"
            )

    coordinate = manifest.get("coordinate_system")
    if not isinstance(coordinate, dict):
        raise BundleToolError("Bundle coordinate_system is missing")
    source_frame = coordinate.get("source_frame")
    if not isinstance(source_frame, dict):
        raise BundleToolError("Bundle source_frame is missing")
    expected_frame = {
        "handedness": "right-handed",
        "right_axis": "+X",
        "forward_axis": "+Y",
        "up_axis": "+Z",
        "units": "scene-units",
    }
    if any(source_frame.get(name) != value for name, value in expected_frame.items()):
        raise BundleToolError("Bundle source frame must be right-handed +X/+Y/+Z")

    contract = manifest.get("import_contract")
    if not isinstance(contract, dict):
        raise BundleToolError("Bundle import_contract is missing")
    expected_contract = {
        "source_up_axis": "+Z",
        "source_forward_axis": "+Y",
        "target_up_axis": "+Z",
        "target_forward_axis": "+X",
        "target_units": "centimeters",
    }
    for name, expected in expected_contract.items():
        if contract.get(name) != expected:
            raise BundleToolError(f"Unexpected import contract value for {name}")

    scale_calibrated = source_frame.get("scale_calibrated")
    meters = source_frame.get("meters_per_unit")
    centimeters = contract.get("centimeters_per_source_unit")
    if scale_calibrated is True:
        meters = _positive_number(meters, "meters_per_unit")
        centimeters = _positive_number(centimeters, "centimeters_per_source_unit")
        if not math.isclose(centimeters, meters * 100.0, rel_tol=1e-9, abs_tol=1e-9):
            raise BundleToolError(
                "centimeters_per_source_unit does not match meters_per_unit"
            )
        if contract.get("requires_operator_alignment") is not False:
            raise BundleToolError("Calibrated bundles must not require scale alignment")
    elif scale_calibrated is False:
        if meters is not None or centimeters is not None:
            raise BundleToolError(
                "Uncalibrated bundles cannot assert a source-unit scale"
            )
        if contract.get("requires_operator_alignment") is not True:
            raise BundleToolError(
                "Uncalibrated bundles must require operator alignment"
            )
    else:
        raise BundleToolError("source_frame.scale_calibrated must be boolean")

    actor_structure = contract.get("actor_structure")
    if not isinstance(actor_structure, dict):
        raise BundleToolError("Bundle actor_structure is missing")
    if actor_structure.get("root") != "SplatLabSceneRoot":
        raise BundleToolError("Bundle root actor must be SplatLabSceneRoot")
    children = actor_structure.get("children")
    required_children = {"GaussianRender", "Collision", "ConventionalGeometry"}
    if (
        not isinstance(children, list)
        or not all(isinstance(child, str) for child in children)
        or not required_children.issubset(set(children))
    ):
        raise BundleToolError("Bundle actor structure omits required child actors")
    # WorldGeometry is required only when the bundle actually ships world-role
    # files: newer producers always emit it (subset check above keeps old tools
    # happy), while bundles from before the world lane stay valid here.
    files = manifest.get("files")
    ships_world = isinstance(files, list) and any(
        isinstance(item, dict)
        and isinstance(item.get("role"), str)
        and item["role"].startswith("world-")
        for item in files
    )
    if ships_world and "WorldGeometry" not in set(children):
        raise BundleToolError(
            "Bundle ships world geometry but actor structure omits WorldGeometry"
        )


def _ply_vertex_count(path: Path) -> int | None:
    with path.open("rb") as handle:
        if handle.readline().rstrip(b"\r\n") != b"ply":
            raise BundleToolError(f"Canonical Gaussian source is not a PLY: {path}")
        for _ in range(16_384):
            raw = handle.readline()
            if not raw:
                break
            if len(raw) > 1024 * 1024:
                raise BundleToolError("PLY header line exceeds 1 MiB")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise BundleToolError("PLY header is not ASCII") from exc
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                try:
                    count = int(parts[2])
                except ValueError as exc:
                    raise BundleToolError("PLY vertex count is invalid") from exc
                if count < 0:
                    raise BundleToolError("PLY vertex count cannot be negative")
                return count
            if line == "end_header":
                break
    return None


def verify_bundle(root: Path) -> VerifiedBundle:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path)
    _validate_contract(manifest)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise BundleToolError("Bundle manifest has no files")
    if len(files) > MAX_ARCHIVE_ENTRIES:
        raise BundleToolError("Bundle manifest has too many files")

    seen: set[str] = set()
    canonical_ply: Path | None = None
    total_bytes = 0
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise BundleToolError(f"Bundle file record {index} is not an object")
        relative = _safe_relative_path(item.get("path"))
        canonical_key = relative.as_posix().casefold()
        if canonical_key in seen:
            raise BundleToolError(f"Duplicate bundle path: {relative}")
        seen.add(canonical_key)

        expected_bytes = item.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or expected_bytes > MAX_SINGLE_FILE_BYTES
        ):
            raise BundleToolError(f"Invalid byte count for {relative}")
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or not SHA256_RE.fullmatch(expected_sha):
            raise BundleToolError(f"Invalid SHA-256 for {relative}")

        path = _contained_file(root, relative)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise BundleToolError(
                f"Size mismatch for {relative}: expected {expected_bytes}, got {actual_bytes}"
            )
        actual_sha = _sha256(path)
        if actual_sha.casefold() != expected_sha.casefold():
            raise BundleToolError(f"SHA-256 mismatch for {relative}")
        total_bytes += actual_bytes
        if total_bytes > MAX_ARCHIVE_BYTES:
            raise BundleToolError("Bundle exceeds the aggregate staging size limit")

        if item.get("role") == "canonical-gaussian-splat":
            if canonical_ply is not None:
                raise BundleToolError(
                    "Bundle declares multiple canonical Gaussian sources"
                )
            if path.suffix.lower() != ".ply":
                raise BundleToolError("Canonical Gaussian source must use PLY")
            canonical_ply = path

    if canonical_ply is None:
        raise BundleToolError("Bundle has no canonical Gaussian PLY")

    gaussian_count = _ply_vertex_count(canonical_ply)
    if gaussian_count is None:
        raise BundleToolError("Canonical Gaussian PLY has no vertex declaration")

    return VerifiedBundle(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_path),
        gaussian_count=gaussian_count,
        total_bytes=total_bytes,
        verified_files=len(files),
    )


def _safe_archive_name(raw: str) -> PurePosixPath:
    if "\x00" in raw:
        raise BundleToolError("Archive contains a NUL path")
    normalized = raw.replace("\\", "/").rstrip("/")
    if not normalized:
        return PurePosixPath(".")
    path = PurePosixPath(normalized)
    if (
        not path.parts
        or path.is_absolute()
        or ":" in path.parts[0]
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BundleToolError(f"Unsafe archive path: {raw}")
    if len(normalized) > 240 or any(
        part.endswith((" ", "."))
        or part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
        for part in path.parts
    ):
        raise BundleToolError(f"Archive path is unsafe on Windows: {raw}")
    return path


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BundleToolError(f"Invalid bundle zip: {archive_path}") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ARCHIVE_ENTRIES:
            raise BundleToolError("Bundle zip has too many entries")
        total = 0
        seen: set[str] = set()
        for entry in entries:
            relative = _safe_archive_name(entry.filename)
            if relative == PurePosixPath("."):
                continue
            key = relative.as_posix().casefold()
            if key in seen:
                raise BundleToolError(f"Bundle zip has a duplicate path: {relative}")
            seen.add(key)
            mode = entry.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise BundleToolError(f"Bundle zip contains a symlink: {relative}")
            if entry.file_size < 0 or entry.file_size > MAX_SINGLE_FILE_BYTES:
                raise BundleToolError(f"Bundle zip entry is too large: {relative}")
            total += entry.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise BundleToolError("Bundle zip exceeds the staging size limit")
            if (
                entry.compress_size > 0
                and entry.file_size > 1024 * 1024
                and entry.file_size / entry.compress_size > 1000
            ):
                raise BundleToolError(f"Suspicious compression ratio: {relative}")

            target = destination.joinpath(*relative.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)


def _find_bundle_root(root: Path) -> Path:
    direct = root / "manifest.json"
    if direct.is_file():
        return root
    manifests = [
        path
        for path in root.glob("*/manifest.json")
        if path.is_file() and not path.is_symlink()
    ]
    if len(manifests) != 1:
        raise BundleToolError(
            "Expected exactly one manifest.json at zip root or one level down"
        )
    return manifests[0].parent


@contextlib.contextmanager
def materialized_bundle(bundle_path: Path) -> Iterator[Path]:
    bundle_path = bundle_path.expanduser().resolve()
    if bundle_path.is_dir():
        yield _find_bundle_root(bundle_path)
        return
    if not bundle_path.is_file() or bundle_path.suffix.lower() != ".zip":
        raise BundleToolError("Bundle path must be a directory or .zip")
    with tempfile.TemporaryDirectory(prefix="splatlab-unreal-bundle-") as temp:
        root = Path(temp)
        _extract_zip_safely(bundle_path, root)
        yield _find_bundle_root(root)


def _load_renderer_config() -> dict[str, Any]:
    config = _load_json(Path(__file__).with_name("renderers.json"))
    if config.get("schema") != RENDERER_SCHEMA:
        raise BundleToolError("Unsupported renderer configuration schema")
    order = config.get("probe_order")
    renderers = config.get("renderers")
    if not isinstance(order, list) or not isinstance(renderers, list):
        raise BundleToolError("Renderer configuration is incomplete")
    ids = [item.get("id") for item in renderers if isinstance(item, dict)]
    if len(ids) != len(set(ids)) or not set(order).issubset(ids):
        raise BundleToolError("Renderer IDs or probe order are invalid")
    return config


def _descriptor_candidates(roots: Sequence[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for descriptor in root.rglob("*.uplugin"):
            resolved = descriptor.resolve()
            if (
                resolved not in seen
                and descriptor.is_file()
                and not descriptor.is_symlink()
            ):
                seen.add(resolved)
                yield resolved


def probe_renderers(project_root: Path, engine_roots: Sequence[Path]) -> dict[str, Any]:
    config = _load_renderer_config()
    project_root = project_root.expanduser().resolve()
    roots = [project_root / "Plugins"]
    for engine_root in engine_roots:
        roots.append(engine_root.expanduser().resolve() / "Engine" / "Plugins")

    by_name: dict[str, list[dict[str, Any]]] = {}
    for descriptor in _descriptor_candidates(roots):
        try:
            payload = _load_json(descriptor, max_bytes=2 * 1024 * 1024)
        except BundleToolError:
            continue
        record = {
            "path": str(descriptor),
            "sha256": _sha256(descriptor),
            "friendly_name": payload.get("FriendlyName"),
            "version": payload.get("Version"),
            "version_name": payload.get("VersionName"),
            "engine_version": payload.get("EngineVersion"),
            "enabled_by_default": payload.get("EnabledByDefault"),
        }
        by_name.setdefault(descriptor.name.casefold(), []).append(record)

    discovered = []
    found_ids: set[str] = set()
    for renderer in config["renderers"]:
        matches = []
        for name in renderer.get("descriptor_names", []):
            matches.extend(by_name.get(str(name).casefold(), []))
        status = "available" if matches else "not-installed"
        if matches:
            found_ids.add(renderer["id"])
        discovered.append(
            {
                "id": renderer["id"],
                "display_name": renderer["display_name"],
                "status": status,
                "recommendation": renderer["recommendation"],
                "descriptors": matches,
            }
        )
    selected = next(
        (
            renderer_id
            for renderer_id in config["probe_order"]
            if renderer_id in found_ids
        ),
        None,
    )
    return {
        "schema": RENDERER_SCHEMA,
        "target_version": TARGET_VERSION,
        "project_root": str(project_root),
        "search_roots": [str(path) for path in roots],
        "selected": selected,
        "renderers": discovered,
    }


def validate_mcp_policy(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    policy = _load_json(path)
    if policy.get("schema") != POLICY_SCHEMA:
        raise BundleToolError("Unsupported Unreal MCP policy schema")
    if policy.get("default") != "deny":
        raise BundleToolError("Unreal MCP policy must default to deny")

    upstream = policy.get("upstream")
    if not isinstance(upstream, dict):
        raise BundleToolError("Unreal MCP upstream policy is missing")
    parsed = urlparse(str(upstream.get("url", "")))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise BundleToolError("Unreal MCP must use a loopback HTTP endpoint")
    if upstream.get("allow_non_loopback") is not False:
        raise BundleToolError("Unreal MCP non-loopback access must be disabled")

    tools = policy.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise BundleToolError("Unreal MCP policy has no allowed tools")
    denied_tools = set(policy.get("denied_tools", []))
    denied_actions = set(policy.get("denied_actions", []))
    if not REQUIRED_DENIED_TOOLS.issubset(denied_tools):
        raise BundleToolError("Unreal MCP policy does not deny broad system tools")
    if not REQUIRED_DENIED_ACTIONS.issubset(denied_actions):
        raise BundleToolError("Unreal MCP policy does not deny dangerous actions")
    if set(tools).intersection(denied_tools):
        raise BundleToolError("Unreal MCP tool appears in both allow and deny lists")

    receipt = policy.get("receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("required_for_mutation") is not True
    ):
        raise BundleToolError("Unreal MCP mutation receipts must be mandatory")
    for name, tool in tools.items():
        if not isinstance(tool, dict):
            raise BundleToolError(f"Invalid Unreal MCP tool policy: {name}")
        actions = tool.get("allowed_actions")
        if (
            not isinstance(actions, list)
            or not actions
            or len(actions) != len(set(actions))
        ):
            raise BundleToolError(
                f"Invalid action allowlist for Unreal MCP tool: {name}"
            )
        overlap = set(actions).intersection(denied_actions)
        if overlap:
            raise BundleToolError(
                f"Denied actions allowed by {name}: {sorted(overlap)}"
            )

    return {
        "status": "valid",
        "schema": policy["schema"],
        "path": str(path),
        "sha256": _sha256(path),
        "allowed_tools": sorted(tools),
        "denied_tools": sorted(denied_tools),
        "enforcement": "contract-only",
    }


def _copy_verified_bundle(bundle: VerifiedBundle, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise BundleToolError(f"Staging directory is not empty: {destination}")
    for record in bundle.manifest["files"]:
        relative = _safe_relative_path(record["path"])
        source = _contained_file(bundle.root, relative)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(bundle.manifest_path, destination / "manifest.json")
    readme = bundle.root / "README.md"
    if (
        readme.is_file()
        and not readme.is_symlink()
        and readme.stat().st_size <= 1024 * 1024
    ):
        shutil.copy2(readme, destination / "README.md")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _project_owned_path(project_root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise BundleToolError("Staging path is outside the Unreal project") from exc
    cursor = project_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise BundleToolError(f"Staging path contains a symlink: {cursor}")
    try:
        path.resolve().relative_to(project_root)
    except ValueError as exc:
        raise BundleToolError("Staging path escapes the Unreal project") from exc
    return path


def stage_bundle(
    bundle: VerifiedBundle,
    project_root: Path,
    engine_roots: Sequence[Path],
    requested_renderer: str,
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    project_file = project_root / "SplatLabUE56.uproject"
    if not project_file.is_file():
        raise BundleToolError(f"Not a SplatLab UE 5.6 project: {project_root}")
    project = _load_json(project_file)
    if project.get("EngineAssociation") != TARGET_VERSION:
        raise BundleToolError("Staging project must target Unreal Engine 5.6")

    renderer_probe = probe_renderers(project_root, engine_roots)
    renderer_ids = {item["id"] for item in renderer_probe["renderers"]}
    available_ids = {
        item["id"]
        for item in renderer_probe["renderers"]
        if item["status"] == "available"
    }
    if requested_renderer != "auto" and requested_renderer not in renderer_ids:
        raise BundleToolError(f"Unknown renderer: {requested_renderer}")
    if requested_renderer != "auto" and requested_renderer not in available_ids:
        raise BundleToolError(
            f"Requested renderer is not installed: {requested_renderer}"
        )
    selected = (
        renderer_probe["selected"]
        if requested_renderer == "auto"
        else requested_renderer
    )

    job_id = bundle.manifest["job_id"]
    version = bundle.manifest_sha256[:16]
    job_root = _project_owned_path(
        project_root, project_root / "SplatLabImports" / job_id
    )
    destination = _project_owned_path(project_root, job_root / version)
    cached = destination.is_dir()
    if cached:
        existing = verify_bundle(destination)
        if existing.manifest_sha256 != bundle.manifest_sha256:
            raise BundleToolError("Existing immutable staging version does not match")
    else:
        job_root.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=job_root))
        try:
            _copy_verified_bundle(bundle, stage)
            verify_bundle(stage)
            os.replace(stage, destination)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    timestamp = _utc_now()
    receipt = {
        "schema": "dev.splatlab.unreal-stage-receipt/v1",
        "status": "cached" if cached else "staged",
        "timestamp": timestamp,
        "job_id": job_id,
        "bundle_schema": bundle.manifest["schema"],
        "manifest_sha256": bundle.manifest_sha256,
        "bundle_version": version,
        "staged_path": str(destination),
        "project": str(project_file),
        "content_destination": f"/Game/SplatLab/Jobs/{job_id}",
        "renderer": selected,
        "renderer_status": "available" if selected else "operator-selection-required",
        "renderer_probe": renderer_probe,
        "actor_structure": bundle.manifest["import_contract"]["actor_structure"],
        "coordinate_system": bundle.manifest["coordinate_system"],
        "import_contract": bundle.manifest["import_contract"],
        "verified_files": bundle.verified_files,
        "total_bytes": bundle.total_bytes,
        "gaussian_count": bundle.gaussian_count,
    }
    _atomic_json(
        job_root / "current.json",
        {
            "schema": "dev.splatlab.unreal-stage-pointer/v1",
            "job_id": job_id,
            "bundle_version": version,
            "manifest_sha256": bundle.manifest_sha256,
            "staged_path": str(destination),
            "updated_at": timestamp,
        },
    )
    safe_timestamp = timestamp.replace(":", "").replace("+", "_")
    _atomic_json(
        _project_owned_path(
            project_root,
            project_root
            / "Saved"
            / "SplatLab"
            / "Receipts"
            / f"{safe_timestamp}-stage-{job_id}.json",
        ),
        receipt,
    )
    return receipt


def _emit(value: dict[str, Any]) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and stage SplatLab Unreal 5.6 handoff bundles."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify-bundle")
    verify.add_argument("bundle", type=Path)

    stage = commands.add_parser("stage-bundle")
    stage.add_argument("bundle", type=Path)
    stage.add_argument("--project-root", type=Path, required=True)
    stage.add_argument("--engine-root", action="append", type=Path, default=[])
    stage.add_argument("--renderer", default="auto")

    probe = commands.add_parser("probe-renderers")
    probe.add_argument("--project-root", type=Path, required=True)
    probe.add_argument("--engine-root", action="append", type=Path, default=[])

    policy = commands.add_parser("validate-mcp-policy")
    policy.add_argument(
        "policy",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("mcp-policy.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify-bundle":
            with materialized_bundle(args.bundle) as root:
                _emit(verify_bundle(root).summary())
        elif args.command == "stage-bundle":
            with materialized_bundle(args.bundle) as root:
                verified = verify_bundle(root)
                _emit(
                    stage_bundle(
                        verified,
                        args.project_root,
                        args.engine_root,
                        args.renderer,
                    )
                )
        elif args.command == "probe-renderers":
            _emit(probe_renderers(args.project_root, args.engine_root))
        elif args.command == "validate-mcp-policy":
            _emit(validate_mcp_policy(args.policy))
        else:  # pragma: no cover - argparse enforces subcommands.
            raise BundleToolError(f"Unknown command: {args.command}")
    except BundleToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
