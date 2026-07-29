"""Portable Gaussian-splat, collision, and Unreal handoff exports.

All formats are derived from the canonical full-fidelity PLY. CPU conversion is
the default. GPU voxelization is explicit and passes through the same admission
and arbiter controls as SplatLab's existing edit operations.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

import artifact_manifest as manifests
import gpu_arbiter
import splat_route
from operator_audit import audit_operator_event


router = APIRouter()

EXPORT_DIRNAME = "_exports"
EXPORT_MANIFEST = "manifest.json"
COLLISION_VRAM_MB = 4_000
STREAMED_SOG_THRESHOLD = 1_000_000
CONVERSION_TIMEOUT_S = 60 * 60

ExportFormat = Literal["spz", "sog", "streamed-sog", "gltf"]
CollisionMode = Literal["interior", "exterior", "raw"]

_FORMAT_FILES: dict[str, tuple[str, str]] = {
    "spz": ("scene.spz", "application/octet-stream"),
    "sog": ("scene.sog", "application/octet-stream"),
    "gltf": ("scene.glb", "model/gltf-binary"),
    "streamed-sog": ("streamed-sog/lod-meta.json", "application/json"),
}

_export_locks: dict[str, asyncio.Lock] = {}
_export_locks_guard = asyncio.Lock()


class ExportRequest(BaseModel):
    formats: list[ExportFormat] = Field(
        default_factory=lambda: ["spz", "sog", "streamed-sog", "gltf"]
    )
    spz_version: Literal[3, 4] = 4
    sog_iterations: int = Field(default=10, ge=1, le=100)
    lod_chunk_count_k: int = Field(default=512, ge=32, le=4096)
    lod_chunk_extent: float = Field(default=16.0, gt=0, le=10_000)
    force_streamed_sog: bool = False
    overwrite: bool = False

    @field_validator("formats")
    @classmethod
    def unique_nonempty_formats(cls, values: list[ExportFormat]) -> list[ExportFormat]:
        if not values:
            raise ValueError("at least one export format is required")
        return list(dict.fromkeys(values))


class CollisionRequest(BaseModel):
    mode: CollisionMode = "exterior"
    seed_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    voxel_size: float = Field(default=0.05, gt=0, le=10)
    opacity_threshold: float = Field(default=0.1, ge=0, le=1)
    cluster_resolution: float = Field(default=1.0, gt=0, le=100)
    cluster_opacity: float = Field(default=0.999, ge=0, le=1)
    cluster_min_contribution: float = Field(default=0.1, ge=0, le=1)
    fill_size: float = Field(default=1.6, gt=0, le=100)
    carve: bool = False
    carve_height: float = Field(default=1.6, gt=0, le=20)
    carve_radius: float = Field(default=0.2, gt=0, le=10)
    mesh_style: Literal["smooth", "faces"] = "smooth"
    # Capture frame -> splat-transform's engine frame. SplatLab captures are
    # Z-up (bundle_tool.py declares source_up_axis "+Z"; mesh_report.py applies
    # rotation_matrix(-pi/2,[1,0,0]) for glTF), but splat-transform voxelizes
    # Y-up. --voxel-floor-fill "fills columns upward from bottom", so without
    # this rotation the floor was filled sideways along the wrong axis.
    # "-90,0,0" maps (x,y,z) -> (x,z,-y), the same conversion the rest of the
    # repo uses. Set to "" only if the input is already Y-up.
    rotate: str = "-90,0,0"
    # Spatial crop, in the CAPTURE frame, applied before the rotation.
    # A capture's gaussians extend far past the subject (sky, neighbouring
    # buildings, reconstruction floaters). Voxelizing all of it spans tens of
    # units of empty space instead of the scene, which inflates the collision
    # mesh by ~50x and buries the real geometry: unfiltered Bonsai voxelized to
    # 6.8M triangles over 64x53x70 units; cropped it is 137k over 8x10x4.
    # Explicit "x,y,z,X,Y,Z" wins; otherwise it is derived from the job's
    # reconstructed mesh. Set auto_filter_box=False for the old whole-cloud
    # behaviour.
    filter_box: str | None = None
    auto_filter_box: bool = True
    filter_box_percentile: float = Field(default=2.0, ge=0.0, lt=50.0)


class UnrealBundleRequest(BaseModel):
    include_zip: bool = False
    include_canonical_ply: bool = True
    include_survey: bool = True
    include_world: bool = True
    include_objects: bool = True
    require_current_exports: bool = True


def _job(job_id: str) -> tuple[dict[str, Any], Path, Path]:
    if not splat_route._safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = splat_route._read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    if meta.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Exports require a completed job. Current status: {meta.get('status')}",
        )
    job_dir = Path(meta.get("output_dir") or splat_route._job_dir(job_id)).resolve()
    preview = splat_route._preview_file_path(job_dir)
    if not preview.is_file():
        raise HTTPException(
            status_code=409,
            detail="Canonical preview PLY is missing. Generate the preview first.",
        )
    return meta, job_dir, preview


async def _export_lock(job_id: str) -> asyncio.Lock:
    async with _export_locks_guard:
        return _export_locks.setdefault(job_id, asyncio.Lock())


def _exports_dir(job_dir: Path) -> Path:
    return job_dir / EXPORT_DIRNAME


def _manifest_path(job_dir: Path) -> Path:
    return _exports_dir(job_dir) / EXPORT_MANIFEST


def _tool_info(transform: str) -> dict[str, Any]:
    resolved = Path(transform).resolve()
    package_json = next(
        (
            parent / "package.json"
            for parent in (resolved.parent, *resolved.parents)
            if (parent / "package.json").is_file()
        ),
        Path("/nonexistent/package.json"),
    )
    version = None
    license_name = None
    if package_json.is_file():
        try:
            package = json.loads(package_json.read_text())
            version = package.get("version")
            license_name = package.get("license")
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "name": "@playcanvas/splat-transform",
        "version": version,
        "license": license_name,
        "binary": str(Path(transform)),
    }


def _new_manifest(
    job_id: str,
    meta: dict[str, Any],
    job_dir: Path,
    source: dict[str, Any],
    transform: str,
) -> dict[str, Any]:
    source = {
        **source,
        "path": manifests.relative_job_path(Path(source["path"]), job_dir),
        "role": "canonical-full-fidelity-gaussian-splat",
    }
    return {
        "schema": manifests.EXPORT_SCHEMA,
        "job_id": job_id,
        "generation_id": uuid.uuid4().hex,
        "created_at": manifests.utc_now(),
        "updated_at": manifests.utc_now(),
        "source": source,
        "coordinate_system": manifests.coordinate_record(meta),
        "toolchain": {"converter": _tool_info(transform)},
        "artifacts": {},
        "warnings": [],
    }


def _source_is_current(source_path: Path, manifest: dict[str, Any] | None) -> bool:
    return bool(
        manifest
        and manifest.get("schema") == manifests.EXPORT_SCHEMA
        and manifests.same_file_identity(source_path, manifest.get("source"))
    )


def _mark_stale_artifacts(manifest: dict[str, Any]) -> None:
    for artifact in manifest.get("artifacts", {}).values():
        if isinstance(artifact, dict) and artifact.get("status") == "ready":
            artifact["status"] = "stale"
            artifact["stale_at"] = manifests.utc_now()


def _conversion_command(
    transform: str,
    source: Path,
    output: Path,
    fmt: ExportFormat,
    request: ExportRequest,
) -> list[str]:
    if fmt == "spz":
        return [
            transform,
            "--spz-version",
            str(request.spz_version),
            str(source),
            str(output),
        ]
    if fmt == "sog":
        return [
            transform,
            "-g",
            "cpu",
            "--iterations",
            str(request.sog_iterations),
            str(source),
            str(output),
        ]
    if fmt == "streamed-sog":
        # lod-meta.json output requires every gaussian to carry an LOD tag —
        # untagged input dies with "Missing lod assignment" (writeLod). `-l 0`
        # after the input tags the whole scene as level 0: spatial chunking
        # still happens (that's the streaming win); multi-level pyramids would
        # need multiple decimate+tag passes and are a future enhancement.
        return [
            transform,
            "-g",
            "cpu",
            "--iterations",
            str(request.sog_iterations),
            "--lod-chunk-count",
            str(request.lod_chunk_count_k),
            "--lod-chunk-extent",
            str(request.lod_chunk_extent),
            str(source),
            "-l",
            "0",
            str(output),
        ]
    return [transform, str(source), str(output)]


async def _execute_command(command: list[str]) -> tuple[int, str]:
    try:
        return_code, stdout, stderr = await asyncio.wait_for(
            splat_route._run_capture_subprocess(command),
            timeout=CONVERSION_TIMEOUT_S,
        )
    except TimeoutError:
        return -1, f"conversion exceeded {CONVERSION_TIMEOUT_S} seconds"
    log = (
        stdout.decode("utf-8", errors="replace")
        + "\n"
        + stderr.decode("utf-8", errors="replace")
    ).strip()
    return return_code, "\n".join(log.splitlines()[-20:])


async def _host_conversion(command: list[str]) -> tuple[int, str]:
    try:
        return await gpu_arbiter.run_host_operation(
            operation=lambda: _execute_command(command)
        )
    except gpu_arbiter.GPUArbiterUnavailable as exc:
        return -1, f"host admission failed: {exc}"


def _published_files(
    export_dir: Path, artifact_root: Path, fmt: ExportFormat
) -> list[dict[str, Any]]:
    if fmt == "streamed-sog":
        paths = sorted(path for path in artifact_root.rglob("*") if path.is_file())
    else:
        paths = [artifact_root]
    if not paths:
        raise manifests.ArtifactManifestError(f"{fmt} emitted no files")
    default_media = _FORMAT_FILES[fmt][1]
    records = []
    for path in paths:
        media = "application/json" if path.suffix == ".json" else default_media
        records.append(
            manifests.artifact_file_record(path, export_dir, media_type=media)
        )
    return records


def _artifact_parameters(fmt: ExportFormat, request: ExportRequest) -> dict[str, Any]:
    if fmt == "spz":
        return {"spz_version": request.spz_version}
    if fmt in {"sog", "streamed-sog"}:
        params: dict[str, Any] = {
            "device": "cpu",
            "iterations": request.sog_iterations,
        }
        if fmt == "streamed-sog":
            params.update(
                {
                    "lod_chunk_count_k": request.lod_chunk_count_k,
                    "lod_chunk_extent": request.lod_chunk_extent,
                }
            )
        return params
    return {"extension": "KHR_gaussian_splatting"}


def _cached_artifact(
    export_dir: Path,
    artifact: dict[str, Any] | None,
    parameters: dict[str, Any],
) -> bool:
    if (
        not artifact
        or artifact.get("status") != "ready"
        or artifact.get("parameters") != parameters
    ):
        return False
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and (export_dir / item["path"]).is_file()
        for item in files
    )


def _publish_path(staged: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if backup.exists():
            os.replace(backup, destination)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink(missing_ok=True)


async def _build_one_format(
    transform: str,
    source: Path,
    export_dir: Path,
    fmt: ExportFormat,
    request: ExportRequest,
) -> tuple[dict[str, Any] | None, str | None]:
    relative_output, _media = _FORMAT_FILES[fmt]
    destination = export_dir / relative_output
    stage_parent = Path(tempfile.mkdtemp(prefix=f".building-{fmt}-", dir=export_dir))
    if fmt == "streamed-sog":
        staged = stage_parent / "streamed-sog"
        staged.mkdir()
        command_output = staged / "lod-meta.json"
    else:
        staged = stage_parent / destination.name
        command_output = staged
    command = _conversion_command(transform, source, command_output, fmt, request)

    try:
        return_code, log = await _host_conversion(command)
        if return_code != 0 or not command_output.is_file():
            return None, f"{fmt} conversion failed (exit {return_code}): {log}"
        _publish_path(
            staged, destination if fmt != "streamed-sog" else destination.parent
        )
        files = _published_files(
            export_dir,
            destination if fmt != "streamed-sog" else destination.parent,
            fmt,
        )
        return {
            "status": "ready",
            "built_at": manifests.utc_now(),
            "parameters": _artifact_parameters(fmt, request),
            "primary_file": relative_output,
            "files": files,
            "command": command,
        }, None
    finally:
        shutil.rmtree(stage_parent, ignore_errors=True)


def _public_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    job_id = manifest["job_id"]
    payload = json.loads(json.dumps(manifest))
    payload["manifest_url"] = f"/api/splat/jobs/{job_id}/exports/manifest"
    for name, artifact in payload.get("artifacts", {}).items():
        if artifact.get("status") == "ready":
            artifact["url"] = (
                f"/api/splat/jobs/{job_id}/exports/streamed-sog/lod-meta.json"
                if name == "streamed-sog"
                else f"/api/splat/jobs/{job_id}/exports/file/{name}"
            )
    if (
        isinstance(payload.get("collision"), dict)
        and payload["collision"].get("status") == "ready"
    ):
        payload["collision"]["voxel_url"] = (
            f"/api/splat/jobs/{job_id}/exports/file/collision-voxel"
        )
        payload["collision"]["mesh_url"] = (
            f"/api/splat/jobs/{job_id}/exports/file/collision-mesh"
        )
    if isinstance(payload.get("unreal_bundle"), dict):
        payload["unreal_bundle"]["manifest_url"] = (
            f"/api/splat/jobs/{job_id}/unreal-bundle/manifest"
        )
        if payload["unreal_bundle"].get("zip"):
            payload["unreal_bundle"]["zip_url"] = (
                f"/api/splat/jobs/{job_id}/unreal-bundle/file"
            )
    return payload


def _annotate_world_current(payload: dict[str, Any], job_dir: Path) -> None:
    """Serve-time honesty flag: does the bundled world still match disk?

    world_current is True when the world_manifest.json recorded at bundle time
    is byte-identical now, False when it drifted or a world appeared/vanished
    after bundling, and None when neither the bundle nor the disk has a world
    (or the bundle predates world support entirely).
    """
    bundle = payload.get("unreal_bundle")
    if not isinstance(bundle, dict):
        return
    world = bundle.get("world")
    recorded = world.get("world_manifest_identity") if isinstance(world, dict) else None
    current = job_dir / "_world" / "world_manifest.json"
    if recorded is None:
        bundle["world_current"] = False if current.is_file() else None
    else:
        bundle["world_current"] = manifests.same_file_identity(current, recorded)


@router.get("/jobs/{job_id}/exports")
async def get_exports(job_id: str) -> dict[str, Any]:
    _meta, job_dir, source = _job(job_id)
    manifest = manifests.read_json(_manifest_path(job_dir))
    if not manifest:
        return {
            "schema": manifests.EXPORT_SCHEMA,
            "job_id": job_id,
            "status": "not-built",
            "artifacts": {},
        }
    if not _source_is_current(source, manifest):
        manifest = json.loads(json.dumps(manifest))
        _mark_stale_artifacts(manifest)
        manifest["status"] = "stale"
    else:
        manifest["status"] = "ready"
    payload = _public_manifest(manifest)
    _annotate_world_current(payload, job_dir)
    return payload


@router.get("/jobs/{job_id}/exports/manifest")
async def get_export_manifest(job_id: str):
    _meta, job_dir, source = _job(job_id)
    manifest = manifests.read_json(_manifest_path(job_dir))
    if not manifest:
        raise HTTPException(status_code=404, detail="Export manifest not built yet")
    if not _source_is_current(source, manifest):
        raise HTTPException(
            status_code=409,
            detail="Export manifest is stale because the canonical PLY changed",
        )
    return FileResponse(
        str(_manifest_path(job_dir)),
        media_type="application/json",
        filename=f"{job_id}-exports.json",
    )


@router.post("/jobs/{job_id}/exports")
async def build_exports(
    request: Request, job_id: str, body: ExportRequest
) -> dict[str, Any]:
    meta, job_dir, source_path = _job(job_id)
    transform = splat_route._splat_transform_path()
    if not transform:
        raise HTTPException(status_code=503, detail="splat-transform is unavailable")

    lock = await _export_lock(job_id)
    if lock.locked():
        raise HTTPException(
            status_code=409, detail=f"An export is already running for {job_id}"
        )

    async with lock:
        export_dir = _exports_dir(job_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        prior = manifests.read_json(_manifest_path(job_dir))
        source_current = _source_is_current(source_path, prior)
        source = await asyncio.to_thread(manifests.inspect_ply, source_path)
        if source_current and prior:
            manifest = prior
            manifest["updated_at"] = manifests.utc_now()
            manifest["warnings"] = []
        else:
            manifest = _new_manifest(job_id, meta, job_dir, source, transform)
            if prior:
                manifest["artifacts"] = prior.get("artifacts", {})
                _mark_stale_artifacts(manifest)

        failures: list[str] = []
        built: list[str] = []
        cached: list[str] = []
        skipped: list[str] = []
        for fmt in body.formats:
            parameters = _artifact_parameters(fmt, body)
            previous = manifest["artifacts"].get(fmt)
            if (
                fmt == "streamed-sog"
                and source["gaussian_count"] < STREAMED_SOG_THRESHOLD
                and not body.force_streamed_sog
            ):
                message = (
                    f"streamed-sog skipped below {STREAMED_SOG_THRESHOLD:,} Gaussians; "
                    "bundled SOG is more efficient for this scene"
                )
                manifest["artifacts"][fmt] = {
                    "status": "skipped",
                    "reason": message,
                    "parameters": parameters,
                }
                skipped.append(fmt)
                continue
            if (
                not body.overwrite
                and source_current
                and _cached_artifact(export_dir, previous, parameters)
            ):
                cached.append(fmt)
                continue
            artifact, error = await _build_one_format(
                transform, source_path, export_dir, fmt, body
            )
            if artifact:
                manifest["artifacts"][fmt] = artifact
                built.append(fmt)
            else:
                failures.append(error or f"{fmt} conversion failed")
                if previous and previous.get("status") == "ready" and source_current:
                    previous["last_error"] = failures[-1]
                else:
                    manifest["artifacts"][fmt] = {
                        "status": "failed",
                        "failed_at": manifests.utc_now(),
                        "parameters": parameters,
                        "error": failures[-1],
                    }

        manifest["updated_at"] = manifests.utc_now()
        manifest["warnings"] = failures
        manifests.atomic_write_json(_manifest_path(job_dir), manifest)
        splat_route._patch_meta(
            job_id,
            exports={
                "schema": manifests.EXPORT_SCHEMA,
                "updated_at": manifest["updated_at"],
                "source_sha256": manifest["source"]["sha256"],
                "ready": sorted(
                    name
                    for name, artifact in manifest["artifacts"].items()
                    if artifact.get("status") == "ready"
                ),
                "manifest_url": f"/api/splat/jobs/{job_id}/exports/manifest",
            },
        )

    if not built and not cached and failures:
        raise HTTPException(status_code=500, detail="; ".join(failures))
    await audit_operator_event(
        request=request,
        title="Built portable splat exports",
        description=f"{job_id}: {', '.join(built or cached)}",
        variant="success" if not failures else "warning",
        action="splat.exports",
        target=meta.get("mode", "3d"),
        metadata={
            "job_id": job_id,
            "built": built,
            "cached": cached,
            "skipped": skipped,
            "failures": failures,
        },
    )
    payload = _public_manifest(manifest)
    payload["result"] = {
        "built": built,
        "cached": cached,
        "skipped": skipped,
        "failures": failures,
    }
    return payload


def _derive_filter_box(job_dir: Path, source: Path, percentile: float) -> str | None:
    """A capture-frame crop around the actual scene, as "x,y,z,X,Y,Z".

    Preference order is deliberate: the reconstructed TSDF mesh first, because
    meshing already discarded most of the far-field that makes a raw gaussian
    cloud so much larger than its subject. The splat is the fallback so this
    still works before a mesh exists.

    Percentiles, not min/max — a handful of stray gaussians or one floater
    triangle would otherwise restore the very extent this is meant to remove.
    Blocking file I/O, so callers should run it off the event loop.
    """
    import numpy as np  # local: keep module import cost off the request path

    for candidate in (job_dir / splat_route.MESH_DIRNAME / "mesh.ply", source):
        if not candidate.is_file():
            continue
        points = _ply_xyz(candidate)
        if points is None or len(points) < 8:
            continue
        lo = np.percentile(points, percentile, axis=0)
        hi = np.percentile(points, 100.0 - percentile, axis=0)
        if np.all(hi > lo):
            return ",".join(f"{v:.6g}" for v in (*lo, *hi))
    return None


# PLY scalar type -> numpy dtype. Deliberately hand-rolled rather than using
# plyfile/trimesh: those live only in the mesh conda env, NOT in the backend
# venv, so importing them here silently returned None in production and the
# crop never happened.
_PLY_DTYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _ply_xyz(path: Path) -> Any:
    """x/y/z of a binary-little-endian PLY's vertex element, as an (N,3) array.

    Reads only the vertex block: the header gives the count and the property
    layout, which is enough to build a structured dtype and slice out the three
    columns without materialising the rest (a splat PLY carries ~60 properties
    per point, so this matters).
    """
    import numpy as np

    try:
        with path.open("rb") as handle:
            if handle.readline().strip() != b"ply":
                return None
            fmt = ""
            count = 0
            fields: list[tuple[str, str]] = []
            in_vertex = False
            while True:
                raw = handle.readline()
                if not raw:
                    return None
                line = raw.decode("ascii", "replace").strip()
                if line == "end_header":
                    break
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    in_vertex = parts[1] == "vertex"
                    if in_vertex:
                        count = int(parts[2])
                elif parts[0] == "property" and in_vertex:
                    if parts[1] == "list":  # not valid on vertex; bail out
                        return None
                    dtype = _PLY_DTYPES.get(parts[1])
                    if dtype is None:
                        return None
                    fields.append((parts[-1], dtype))
            names = [name for name, _ in fields]
            if fmt != "binary_little_endian" or count <= 0:
                return None
            if not {"x", "y", "z"} <= set(names):
                return None
            block = np.fromfile(handle, dtype=np.dtype(fields), count=count)
        if len(block) == 0:
            return None
        return np.stack(
            [block["x"], block["y"], block["z"]], axis=1
        ).astype(np.float64)
    except Exception:  # noqa: BLE001 — no crop is a valid outcome, never a 500
        return None


def _collision_command(
    transform: str,
    source: Path,
    output: Path,
    request: CollisionRequest,
    filter_box: str | None = None,
) -> list[str]:
    # splat-transform's contract is `input [ACTIONS] ... output`, and ACTIONS
    # execute IN ORDER. --rotate is a true action, so it only applies to the
    # working set if it comes after the input file — which is why the source
    # now leads instead of sitting second-to-last.
    seed_xyz = tuple(float(v) for v in request.seed_position)
    if request.rotate.strip() == "-90,0,0":
        # Rotate the seed with the cloud. Everything after the -r action is
        # evaluated in the rotated frame, so a caller-supplied seed given in
        # capture coords would otherwise point somewhere else entirely. The
        # default (0,0,0) is rotation-invariant, so this only bites non-default
        # seeds — silently, which is the worst kind.
        x, y, z = seed_xyz
        seed_xyz = (x, z, -y if y else 0.0)  # avoid "-0" in the argv
    seed = ",".join(f"{value:.9g}" for value in seed_xyz)

    command = [transform, str(source)]
    # Order is load-bearing: the box is expressed in the capture frame, so it
    # must be applied BEFORE the rotation into engine space. Everything after
    # -r (cluster, seed, voxel) is evaluated rotated.
    box = filter_box if filter_box is not None else request.filter_box
    if box:
        command.extend(["--filter-box", box.strip()])
    if request.rotate.strip():
        command.extend(["-r", request.rotate.strip()])
    command.extend([
        "--filter-cluster",
        (
            f"{request.cluster_resolution:.9g},"
            f"{request.cluster_opacity:.9g},"
            f"{request.cluster_min_contribution:.9g}"
        ),
        "--seed-pos",
        seed,
        "--voxel-params",
        f"{request.voxel_size:.9g},{request.opacity_threshold:.9g}",
    ])
    if request.mode == "interior":
        command.extend(["--voxel-external-fill", f"{request.fill_size:.9g}"])
    elif request.mode == "exterior":
        command.extend(["--voxel-floor-fill", f"{request.fill_size:.9g}"])
    if request.carve:
        command.extend(
            [
                "--voxel-carve",
                f"{request.carve_height:.9g},{request.carve_radius:.9g}",
            ]
        )
    command.extend(["--collision-mesh", request.mesh_style, str(output)])
    return command


@router.post("/jobs/{job_id}/collision")
async def build_collision(
    request: Request, job_id: str, body: CollisionRequest
) -> dict[str, Any]:
    splat_route.require_heavy_work_admitted()
    meta, job_dir, source = _job(job_id)
    transform = splat_route._splat_transform_path()
    if not transform:
        raise HTTPException(status_code=503, detail="splat-transform is unavailable")
    lock = await _export_lock(job_id)
    if lock.locked():
        raise HTTPException(
            status_code=409, detail=f"An export is already running for {job_id}"
        )

    async with lock:
        export_dir = _exports_dir(job_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        stage_dir: Path | None = Path(
            tempfile.mkdtemp(prefix=".building-collision-", dir=export_dir)
        )
        output = stage_dir / "scene.voxel.json"
        resolved_box = body.filter_box
        if not resolved_box and body.auto_filter_box:
            resolved_box = await asyncio.to_thread(
                _derive_filter_box, job_dir, source, body.filter_box_percentile
            )
        command = _collision_command(transform, source, output, body, resolved_box)
        try:
            try:
                return_code, log = await gpu_arbiter.run_gpu_operation(
                    lane="splat-collision",
                    operation_id=job_id,
                    vram_mb=COLLISION_VRAM_MB,
                    operation=lambda: _execute_command(command),
                )
            except gpu_arbiter.GPUArbiterUnavailable as exc:
                raise HTTPException(
                    status_code=503, detail=f"Collision build blocked: {exc}"
                ) from exc
            expected = [
                output,
                stage_dir / "scene.voxel.bin",
                stage_dir / "scene.collision.glb",
            ]
            if return_code != 0 or not all(path.is_file() for path in expected):
                raise HTTPException(
                    status_code=500,
                    detail=f"Collision build failed (exit {return_code}): {log}",
                )
            destination = export_dir / "collision"
            _publish_path(stage_dir, destination)
            stage_dir = None
        finally:
            if stage_dir is not None and stage_dir.is_dir():
                shutil.rmtree(stage_dir, ignore_errors=True)

        manifest = manifests.read_json(_manifest_path(job_dir))
        source_record = await asyncio.to_thread(manifests.inspect_ply, source)
        if not manifest or not _source_is_current(source, manifest):
            manifest = _new_manifest(job_id, meta, job_dir, source_record, transform)
        collision_dir = export_dir / "collision"
        collision_files = [
            manifests.artifact_file_record(
                collision_dir / "scene.voxel.json",
                export_dir,
                media_type="application/json",
            ),
            manifests.artifact_file_record(
                collision_dir / "scene.voxel.bin",
                export_dir,
                media_type="application/octet-stream",
            ),
            manifests.artifact_file_record(
                collision_dir / "scene.collision.glb",
                export_dir,
                media_type="model/gltf-binary",
            ),
        ]
        manifest["collision"] = {
            "status": "ready",
            "built_at": manifests.utc_now(),
            "mode": body.mode,
            "parameters": body.model_dump(),
            "files": collision_files,
            "command": command,
        }
        manifest["updated_at"] = manifests.utc_now()
        manifests.atomic_write_json(_manifest_path(job_dir), manifest)

    await audit_operator_event(
        request=request,
        title="Built splat collision artifacts",
        description=f"{job_id}: {body.mode}, {body.mesh_style}",
        variant="success",
        action="splat.collision",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "mode": body.mode},
    )
    return _public_manifest(manifest)["collision"]


def _resolve_manifest_file(
    export_dir: Path, manifest: dict[str, Any], artifact_name: str
) -> tuple[Path, str]:
    if artifact_name in manifest.get("artifacts", {}):
        artifact = manifest["artifacts"][artifact_name]
        if artifact.get("status") != "ready":
            raise HTTPException(
                status_code=409, detail=f"{artifact_name} is not current"
            )
        relative = artifact.get("primary_file")
        media = _FORMAT_FILES.get(artifact_name, ("", "application/octet-stream"))[1]
    elif artifact_name == "collision-voxel":
        relative, media = "collision/scene.voxel.json", "application/json"
    elif artifact_name == "collision-mesh":
        relative, media = "collision/scene.collision.glb", "model/gltf-binary"
    else:
        raise HTTPException(status_code=404, detail="Unknown export artifact")
    if not isinstance(relative, str):
        raise HTTPException(
            status_code=404, detail="Export artifact has no primary file"
        )
    path = (export_dir / relative).resolve()
    try:
        path.relative_to(export_dir.resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail="Invalid export artifact path"
        ) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Export artifact is missing")
    return path, media


@router.get("/jobs/{job_id}/exports/file/{artifact_name}")
async def get_export_file(job_id: str, artifact_name: str):
    _meta, job_dir, source = _job(job_id)
    manifest = manifests.read_json(_manifest_path(job_dir))
    if not manifest:
        raise HTTPException(status_code=404, detail="Exports not built yet")
    if not _source_is_current(source, manifest):
        raise HTTPException(
            status_code=409,
            detail="Exports are stale because the canonical PLY changed",
        )
    path, media = _resolve_manifest_file(_exports_dir(job_dir), manifest, artifact_name)
    return FileResponse(str(path), media_type=media, filename=path.name)


@router.get("/jobs/{job_id}/exports/streamed-sog/{relative_path:path}")
async def get_streamed_sog_file(job_id: str, relative_path: str):
    """Serve a streamed-SOG tree while preserving relative chunk URLs."""
    _meta, job_dir, source = _job(job_id)
    manifest = manifests.read_json(_manifest_path(job_dir))
    if not manifest or not _source_is_current(source, manifest):
        raise HTTPException(
            status_code=409, detail="Current streamed SOG not available"
        )
    artifact = manifest.get("artifacts", {}).get("streamed-sog")
    if not isinstance(artifact, dict) or artifact.get("status") != "ready":
        raise HTTPException(status_code=404, detail="Streamed SOG not built yet")
    if not relative_path or relative_path.startswith("."):
        raise HTTPException(status_code=404, detail="Invalid streamed SOG path")
    root = (_exports_dir(job_dir) / "streamed-sog").resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail="Invalid streamed SOG path"
        ) from exc
    declared = {
        Path(item["path"]).relative_to("streamed-sog").as_posix()
        for item in artifact.get("files", [])
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and item["path"].startswith("streamed-sog/")
    }
    if relative_path not in declared or not path.is_file():
        raise HTTPException(status_code=404, detail="Streamed SOG file is missing")
    media = "application/json" if path.suffix == ".json" else "application/octet-stream"
    return FileResponse(str(path), media_type=media, filename=path.name)


def _safe_export_relative(value: Any) -> PurePosixPath | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    raw_parts = value.split("/")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        return None
    return relative


# (source path, bundle-relative path, provenance, role, extra metadata)
BundleCandidate = tuple[Path, str, str, str, dict[str, Any]]


def _world_marker_provenance(world_dir: Path, polish_rel: str,
                             restyle_rel: str) -> tuple[str, bool]:
    """("polished" | "restyled" | "captured-derived", atlas_is_current).

    When both markers exist the newer one wins — a polish after a bake
    supersedes it and vice versa (ISO timestamps compare lexicographically).
    atlas_is_current is True only for a winning bake that refreshed the
    sidecar PNG; polish leaves the solidify-era PNG stale on purpose."""
    polish = manifests.read_json(world_dir / polish_rel)
    restyle = manifests.read_json(world_dir / restyle_rel)
    if not polish and not restyle:
        return "captured-derived", False
    newer_is_restyle = bool(restyle) and (
        not polish
        or str(restyle.get("baked_at") or "") >= str(polish.get("uploaded_at") or ""))
    if newer_is_restyle:
        return "restyled", bool(restyle.get("atlas_updated"))
    return "polished", False


def _world_bundle_candidates(job_dir: Path) -> list[BundleCandidate]:
    """Everything the navigable-world lane produced, for the World/ section.

    Reads world.json (scene_solidify receipt) and world_manifest.json
    (world_collision verdict) straight from disk — absent world = empty list,
    never an error, so pre-world jobs bundle exactly as before.
    """
    world_dir = job_dir / "_world"
    world = manifests.read_json(world_dir / "world.json") or {}
    verdict = manifests.read_json(world_dir / "world_manifest.json") or {}
    if not world and not verdict:
        return []
    out: list[BundleCandidate] = []

    shell = world.get("shell") if isinstance(world.get("shell"), dict) else {}
    if shell.get("built"):
        shell_prov, shell_atlas_current = _world_marker_provenance(
            world_dir, "shell.polish.json", "shell.restyle.json")
        out.append((world_dir / (shell.get("glb") or "shell.glb"),
                    "World/shell.glb", shell_prov, "world-visual-shell", {}))
        out.append((world_dir / "shell_atlas.png", "World/shell_atlas.png",
                    "restyled" if shell_atlas_current else "captured-derived",
                    "world-atlas", {}))
        out.append((world_dir / "shell.json", "World/shell.json",
                    "captured-derived", "world-receipt", {}))
    for name in ("collision_shell.glb", "collision_shell.json"):
        out.append((world_dir / name, f"World/{name}",
                    "captured-derived", "world-collision-shell", {}))
    for name in ("world.json", "world_manifest.json"):
        out.append((world_dir / name, f"World/{name}",
                    "captured-derived", "world-receipt", {}))

    elements = verdict.get("elements")
    if not isinstance(elements, list):
        elements = world.get("elements") if isinstance(world.get("elements"), list) else []
    for element in elements:
        if not isinstance(element, dict):
            continue
        slug = element.get("slug")
        glb = element.get("glb") or (f"{slug}.glb" if slug else None)
        if (not isinstance(slug, str) or not slug
                or not isinstance(glb, str) or "/" in glb or "\\" in glb):
            continue
        prov, atlas_current = _world_marker_provenance(
            world_dir, f"elements/{slug}.polish.json",
            f"elements/{slug}.restyle.json")
        out.append((world_dir / "elements" / glb, f"World/Elements/{glb}",
                    prov, "world-element",
                    {"element_role": element.get("role") or "prop"}))
        out.append((world_dir / "elements" / f"{slug}_atlas.png",
                    f"World/Elements/{slug}_atlas.png",
                    "restyled" if atlas_current else "captured-derived",
                    "world-atlas", {}))
        collision = element.get("collision")
        collision = collision if isinstance(collision, dict) else {}
        for hull in collision.get("files") or []:
            if not isinstance(hull, str) or "/" in hull or "\\" in hull:
                continue
            out.append((world_dir / "collision" / hull,
                        f"World/Collision/{hull}",
                        "captured-derived", "world-collision-hull",
                        {"collision_for": f"World/Elements/{glb}"}))
        ue_glb = collision.get("ue_glb")
        if isinstance(ue_glb, str) and "/" not in ue_glb and "\\" not in ue_glb:
            out.append((world_dir / "collision" / ue_glb,
                        f"World/Props/{slug}.glb",
                        "captured-derived", "world-prop-ue",
                        {"note": "single-file UE import; UCX_* nodes are the "
                                 "collision hulls for the sibling render node"}))
    return out


def _bakeoff_winner_candidate(
    job_dir: Path, slug: str, bakeoff: dict[str, Any]
) -> list[BundleCandidate]:
    verdict = bakeoff.get("verdict") if isinstance(bakeoff.get("verdict"), dict) else {}
    winner = verdict.get("winner")
    if not isinstance(winner, str) or not winner:
        return []
    candidate = next(
        (c for c in bakeoff.get("candidates", [])
         if isinstance(c, dict) and c.get("name") == winner),
        None,
    )
    source = candidate.get("source") if isinstance(candidate, dict) else None
    if not isinstance(source, str) or not source:
        return []
    source_path = Path(source)
    try:
        source_path.resolve().relative_to(job_dir.resolve())
    except (ValueError, OSError):
        # A winner supplied from outside the job tree (e.g. an operator's
        # --candidates path) is legitimate but not ours to ship; bakeoff.json
        # still records it, so the omission is visible rather than silent.
        return []
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    candidate_provenance = candidate.get("provenance")
    provenance = (
        "captured-derived"
        if candidate_provenance in ("capture-native", "ours")
        else "generated"
    )
    metadata = {
        "bakeoff": {
            "winner": winner,
            "candidate_provenance": candidate_provenance,
            "median_psnr_paired": scores.get("median_psnr_paired"),
        }
    }
    suffix = source_path.suffix or ".glb"
    return [(source_path, f"Objects/{slug}/winner-{winner}{suffix}",
             provenance, "object-bakeoff-winner", metadata)]


def _objects_bundle_candidates(job_dir: Path) -> list[BundleCandidate]:
    """Textured object assets + bake-off reports/winners for the Objects/ section."""
    objects_dir = job_dir / "_objects"
    if not objects_dir.is_dir():
        return []
    out: list[BundleCandidate] = []
    for obj_dir in sorted(objects_dir.iterdir()):
        slug = obj_dir.name
        if (not obj_dir.is_dir() or slug.startswith(".")
                or not (obj_dir / "object.json").is_file()):
            continue
        mesh_dir = obj_dir / "mesh"
        out.append((mesh_dir / "textured.glb", f"Objects/{slug}/textured.glb",
                    "captured-derived", "object-textured", {}))
        out.append((mesh_dir / "textured_atlas.png",
                    f"Objects/{slug}/textured_atlas.png",
                    "captured-derived", "object-atlas", {}))
        out.append((mesh_dir / "polished.glb", f"Objects/{slug}/polished.glb",
                    "polished", "object-polished", {}))
        out.append((mesh_dir / "polished.json", f"Objects/{slug}/polished.json",
                    "polished", "object-polish-receipt", {}))
        bakeoff_path = mesh_dir / "bakeoff" / "bakeoff.json"
        bakeoff = manifests.read_json(bakeoff_path)
        if bakeoff:
            out.append((bakeoff_path, f"Objects/{slug}/bakeoff.json",
                        "captured-derived", "bakeoff-report", {}))
            out.extend(_bakeoff_winner_candidate(job_dir, slug, bakeoff))
    return out


def _bundle_candidates(
    job_dir: Path,
    manifest: dict[str, Any],
    request: UnrealBundleRequest,
) -> list[BundleCandidate]:
    candidates: list[BundleCandidate] = []
    if request.include_canonical_ply:
        candidates.append(
            (
                splat_route._preview_file_path(job_dir),
                "Gaussian/scene.ply",
                "captured",
                "canonical-gaussian-splat",
                {},
            )
        )
    for name, artifact in manifest.get("artifacts", {}).items():
        if (
            name not in _FORMAT_FILES
            or not isinstance(artifact, dict)
            or artifact.get("status") != "ready"
        ):
            continue
        for file_record in artifact.get("files", []):
            relative = (
                file_record.get("path") if isinstance(file_record, dict) else None
            )
            source_relative = _safe_export_relative(relative)
            if source_relative is not None:
                artifact_relative = source_relative
                if source_relative.parts[0] == name:
                    artifact_relative = PurePosixPath(*source_relative.parts[1:])
                if not artifact_relative.parts:
                    continue
                candidates.append(
                    (
                        _exports_dir(job_dir).joinpath(*source_relative.parts),
                        (
                            PurePosixPath("Gaussian") / name / artifact_relative
                        ).as_posix(),
                        "captured-derived",
                        f"gaussian-{name}",
                        {},
                    )
                )
    collision = manifest.get("collision")
    if isinstance(collision, dict) and collision.get("status") == "ready":
        for file_record in collision.get("files", []):
            relative = (
                file_record.get("path") if isinstance(file_record, dict) else None
            )
            source_relative = _safe_export_relative(relative)
            if source_relative is not None:
                candidates.append(
                    (
                        _exports_dir(job_dir).joinpath(*source_relative.parts),
                        f"Collision/{source_relative.name}",
                        "captured-derived",
                        "collision",
                        {},
                    )
                )

    fixed_candidates: list[BundleCandidate] = [
        (
            job_dir / "_mesh" / "twin.glb",
            "Geometry/twin.glb",
            "captured-derived",
            "render-mesh",
            {},
        ),
        (
            job_dir / "_mesh" / "mesh.glb",
            "Geometry/mesh.glb",
            "captured-derived",
            "mesh",
            {},
        ),
        (
            job_dir / "_regen" / "scene.glb",
            "Scene/scene.glb",
            "composed",
            "assembled-scene",
            {},
        ),
        (
            job_dir / "_regen" / "scene.blend",
            "Scene/scene.blend",
            "composed",
            "blender-scene",
            {},
        ),
        (
            job_dir / "_regen" / "scene_manifest.json",
            "Scene/scene_manifest.json",
            "composed",
            "scene-manifest",
            {},
        ),
        (
            job_dir / "_preview" / "thumb.webp",
            "Receipts/thumb.webp",
            "captured-derived",
            "thumbnail",
            {},
        ),
        (
            job_dir / "_langfield" / "hero.webp",
            "Receipts/hero.webp",
            "captured-derived",
            "hero",
            {},
        ),
        (job_dir / "dimensions.json", "Survey/dimensions.json", "survey",
         "dimensions", {}),
    ]
    candidates.extend(item for item in fixed_candidates if item[0].is_file())
    if request.include_survey:
        survey_dir = job_dir / "_mesh" / "geo"
        for name in (
            "site.dxf",
            "surface.xml",
            "site.geojson",
            "contours.dxf",
            "ground_points.txt",
            "sections.png",
            "surface_iso.png",
        ):
            path = survey_dir / name
            if path.is_file():
                candidates.append(
                    (path, f"Survey/{name}", "survey", "survey-export", {})
                )
    if request.include_world:
        candidates.extend(_world_bundle_candidates(job_dir))
    if request.include_objects:
        candidates.extend(_objects_bundle_candidates(job_dir))

    unique: dict[str, BundleCandidate] = {}
    for item in candidates:
        if item[0].is_file():
            unique.setdefault(item[1], item)
    return list(unique.values())


def _hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_bundle_readme(path: Path, job_id: str, calibrated: bool) -> None:
    scale_note = (
        "The manifest includes a calibrated meters-per-unit value."
        if calibrated
        else "Scale is not calibrated. Align and scale the shared parent actor before use."
    )
    path.write_text(
        "\n".join(
            [
                "# SplatLab Unreal 5.6 Bundle",
                "",
                f"Job: `{job_id}`",
                "",
                "Verify `manifest.json` checksums before import.",
                "Import the Gaussian asset through the selected renderer plugin.",
                "Import collision and conventional meshes as separate actors, then parent them",
                "under one scene root so alignment changes remain reversible.",
                "",
                "`World/` holds the navigable-world artifacts (visual shell, walkable",
                "collision shell, elements, loose UCX hulls). `World/Props/<slug>.glb` is a",
                "single-file prop import whose `UCX_*` nodes are its collision hulls —",
                "verify the UCX binding on one prop before trusting it for glTF imports.",
                "`Objects/` holds textured object assets and bake-off winners; a winner with",
                "provenance `generated` is render-only and must never be treated as survey.",
                "",
                scale_note,
                "The source frame is right-handed, +Z up, with +Y forward.",
                "Do not treat render-health receipts as survey or geometric truth.",
                "",
            ]
        )
    )


def _build_unreal_bundle_sync(
    job_id: str,
    meta: dict[str, Any],
    job_dir: Path,
    export_manifest: dict[str, Any],
    request: UnrealBundleRequest,
) -> dict[str, Any]:
    export_dir = _exports_dir(job_dir)
    destination = export_dir / "unreal"
    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=".building-unreal-", dir=export_dir)
    )
    try:
        entries = []
        for source, relative, provenance, role, extra in _bundle_candidates(
            job_dir, export_manifest, request
        ):
            target = stage / relative
            materialization = _hardlink_or_copy(source, target)
            entries.append(
                {
                    "path": relative,
                    "source": manifests.relative_job_path(source, job_dir),
                    "provenance": provenance,
                    "role": role,
                    "materialization": materialization,
                    **extra,
                    **manifests.file_identity(target),
                }
            )

        coordinate = manifests.coordinate_record(meta)
        calibrated = coordinate["source_frame"]["scale_calibrated"]
        world_manifest_path = job_dir / "_world" / "world_manifest.json"
        world_block = {
            "present": any(
                isinstance(entry.get("role"), str)
                and entry["role"].startswith("world-")
                for entry in entries
            ),
            "collision_built": world_manifest_path.is_file(),
            "world_manifest_identity": (
                manifests.file_identity(world_manifest_path)
                if world_manifest_path.is_file()
                else None
            ),
        }
        bundle = {
            "schema": manifests.UNREAL_SCHEMA,
            "job_id": job_id,
            "created_at": manifests.utc_now(),
            "target": {
                "engine": "Unreal Engine",
                "version": "5.6",
                "platform": "Windows",
                "renderer_probe_order": [
                    "NanoGS",
                    "MLSLabsRenderer",
                    "UnrealSplat",
                ],
            },
            "coordinate_system": coordinate,
            "import_contract": {
                "source_up_axis": "+Z",
                "source_forward_axis": "+Y",
                "target_up_axis": "+Z",
                "target_forward_axis": "+X",
                "target_units": "centimeters",
                "centimeters_per_source_unit": (
                    float(meta["meters_per_unit"]) * 100.0 if calibrated else None
                ),
                "requires_operator_alignment": not calibrated,
                "actor_structure": {
                    "root": "SplatLabSceneRoot",
                    "children": [
                        "GaussianRender",
                        "Collision",
                        "ConventionalGeometry",
                        "WorldGeometry",
                    ],
                },
            },
            "world": world_block,
            "files": entries,
        }
        manifests.atomic_write_json(stage / "manifest.json", bundle)
        _write_bundle_readme(stage / "README.md", job_id, calibrated)
        _publish_path(stage, destination)
        stage = None

        zip_record = None
        if request.include_zip:
            zip_path = export_dir / f"{job_id}-unreal-5.6.zip"
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{job_id}-unreal-", suffix=".zip", dir=export_dir
            )
            os.close(descriptor)
            temp_zip = Path(temp_name)
            try:
                with zipfile.ZipFile(
                    temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
                ) as archive:
                    for file_path in sorted(destination.rglob("*")):
                        if file_path.is_file():
                            archive.write(
                                file_path,
                                file_path.relative_to(destination).as_posix(),
                            )
                os.replace(temp_zip, zip_path)
            finally:
                temp_zip.unlink(missing_ok=True)
            zip_record = manifests.artifact_file_record(
                zip_path, export_dir, media_type="application/zip"
            )
        return {
            "status": "ready",
            "built_at": manifests.utc_now(),
            "directory": "unreal",
            "manifest": "unreal/manifest.json",
            "files": len(entries),
            "world": world_block,
            "zip": zip_record,
        }
    finally:
        if stage is not None and stage.is_dir():
            shutil.rmtree(stage, ignore_errors=True)


@router.post("/jobs/{job_id}/unreal-bundle")
async def build_unreal_bundle(
    request: Request, job_id: str, body: UnrealBundleRequest
) -> dict[str, Any]:
    meta, job_dir, source = _job(job_id)
    lock = await _export_lock(job_id)
    if lock.locked():
        raise HTTPException(
            status_code=409, detail=f"An export is already running for {job_id}"
        )
    async with lock:
        export_manifest = manifests.read_json(_manifest_path(job_dir))
        if not export_manifest:
            raise HTTPException(
                status_code=409,
                detail="Build portable exports before creating an Unreal bundle",
            )
        if body.require_current_exports and not _source_is_current(
            source, export_manifest
        ):
            raise HTTPException(
                status_code=409,
                detail="Portable exports are stale; rebuild them before bundling",
            )
        try:
            bundle_record = await gpu_arbiter.run_host_operation(
                operation=lambda: asyncio.to_thread(
                    _build_unreal_bundle_sync,
                    job_id,
                    meta,
                    job_dir,
                    export_manifest,
                    body,
                )
            )
        except gpu_arbiter.GPUArbiterUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unreal bundle host admission failed: {exc}",
            ) from exc
        export_manifest["unreal_bundle"] = bundle_record
        export_manifest["updated_at"] = manifests.utc_now()
        manifests.atomic_write_json(_manifest_path(job_dir), export_manifest)

    await audit_operator_event(
        request=request,
        title="Built Unreal 5.6 handoff",
        description=f"{job_id}: {bundle_record['files']} files",
        variant="success",
        action="splat.unreal_bundle",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "include_zip": body.include_zip},
    )
    payload = _public_manifest(export_manifest)
    _annotate_world_current(payload, job_dir)
    return payload["unreal_bundle"]


@router.get("/jobs/{job_id}/unreal-bundle/manifest")
async def get_unreal_bundle_manifest(job_id: str):
    _meta, job_dir, source = _job(job_id)
    export_manifest = manifests.read_json(_manifest_path(job_dir))
    if not export_manifest or not _source_is_current(source, export_manifest):
        raise HTTPException(
            status_code=409, detail="Current Unreal bundle not available"
        )
    path = _exports_dir(job_dir) / "unreal" / "manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unreal bundle not built yet")
    return FileResponse(
        str(path), media_type="application/json", filename=f"{job_id}-unreal.json"
    )


@router.get("/jobs/{job_id}/unreal-bundle/file")
async def get_unreal_bundle_file(job_id: str):
    _meta, job_dir, source = _job(job_id)
    export_manifest = manifests.read_json(_manifest_path(job_dir))
    if not export_manifest or not _source_is_current(source, export_manifest):
        raise HTTPException(
            status_code=409, detail="Current Unreal bundle not available"
        )
    bundle = export_manifest.get("unreal_bundle")
    relative = bundle.get("zip", {}).get("path") if isinstance(bundle, dict) else None
    if not isinstance(relative, str):
        raise HTTPException(status_code=404, detail="Zipped Unreal bundle not built")
    path = (_exports_dir(job_dir) / relative).resolve()
    try:
        path.relative_to(_exports_dir(job_dir).resolve())
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail="Invalid Unreal bundle path"
        ) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Zipped Unreal bundle is missing")
    return FileResponse(
        str(path), media_type="application/zip", filename=f"{job_id}-unreal-5.6.zip"
    )
