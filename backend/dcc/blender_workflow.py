"""Restricted, versioned Blender operations for the SplatLab MCP surface."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import artifact_manifest as manifests
import glb_check


BLENDER_BIN = Path(
    os.environ.get(
        "SPLATLAB_BLENDER_BIN",
        str(Path.home() / "tools" / "blender-4.5.11-linux-x64" / "blender"),
    )
)
ACTION_SCRIPT = Path(__file__).with_name("blender_actions.py")
VERSION_RE = re.compile(r"^scene-v(?P<version>\d{4})\.blend$")
MUTATING_ACTIONS = {
    "snapshot",
    "toggle_collection",
    "transform_object",
    "import_world_element",
    "cleanup_mesh",
}
READ_ACTIONS = {"inspect"}
# World-element slugs come from splat_route._object_slug output: lowercase
# kebab, max 40 chars. Re-validated here so the dcc layer never trusts input.
ELEMENT_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
MAX_NAME_LENGTH = 256
BLENDER_TIMEOUT_S = 10 * 60
OUTPUT_ROOT = Path(
    os.environ.get("SPLATLAB_OUTPUT_ROOT", "/home/rtoony/projects/splatcli/outputs/3d")
).resolve()
JOB_ID_RE = re.compile(r"^splat_[A-Za-z0-9_-]{3,80}$")


class BlenderWorkflowError(ValueError):
    """A typed Blender operation is invalid or failed."""


def _job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.fullmatch(job_id):
        raise BlenderWorkflowError("invalid SplatLab job id")
    canonical_dir = (OUTPUT_ROOT / job_id).resolve()
    try:
        canonical_dir.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise BlenderWorkflowError(
            "job directory is outside the approved root"
        ) from exc
    meta = manifests.read_json(canonical_dir / "meta.json")
    if not meta:
        raise BlenderWorkflowError(f"SplatLab job not found: {job_id}")
    job_dir = Path(meta.get("output_dir") or canonical_dir).resolve()
    try:
        job_dir.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise BlenderWorkflowError(
            "job output directory is outside the approved root"
        ) from exc
    return job_dir


def _read_meta(job_id: str) -> dict[str, Any]:
    meta = manifests.read_json(OUTPUT_ROOT / job_id / "meta.json")
    if not meta:
        raise BlenderWorkflowError(f"SplatLab job not found: {job_id}")
    return meta


def _workflow_dir(job_dir: Path) -> Path:
    return job_dir / "_blender"


def _versions_dir(job_dir: Path) -> Path:
    return _workflow_dir(job_dir) / "versions"


def _receipts_dir(job_dir: Path) -> Path:
    return _workflow_dir(job_dir) / "receipts"


def _version_path(job_dir: Path, version: int) -> Path:
    if version < 1 or version > 9999:
        raise BlenderWorkflowError("version must be between 1 and 9999")
    return _versions_dir(job_dir) / f"scene-v{version:04d}.blend"


def _contained_scene_file(
    job_dir: Path,
    path: Path,
    *,
    missing: str = "assembled scene.blend is missing; build the scene assembly first",
) -> Path:
    try:
        relative = path.relative_to(job_dir)
    except ValueError as exc:
        raise BlenderWorkflowError("Blender scene is outside the approved job") from exc
    cursor = job_dir
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise BlenderWorkflowError("symlinked Blender scene inputs are not allowed")
    if not path.is_file():
        raise BlenderWorkflowError(missing)
    try:
        path.resolve(strict=True).relative_to(job_dir)
    except (OSError, ValueError) as exc:
        raise BlenderWorkflowError("Blender scene is outside the approved job") from exc
    return path


def list_versions(job_id: str) -> list[dict[str, Any]]:
    job_dir = _job_dir(job_id)
    versions = []
    for path in sorted(_versions_dir(job_dir).glob("scene-v*.blend")):
        match = VERSION_RE.match(path.name)
        if not match:
            continue
        _contained_scene_file(job_dir, path)
        version = int(match.group("version"))
        receipt = manifests.read_json(
            _receipts_dir(job_dir) / f"scene-v{version:04d}.json"
        )
        versions.append(
            {
                "version": version,
                "path": manifests.relative_job_path(path, job_dir),
                "bytes": path.stat().st_size,
                "receipt": receipt,
            }
        )
    return versions


def _latest_version(job_dir: Path) -> int | None:
    values = []
    for path in _versions_dir(job_dir).glob("scene-v*.blend"):
        match = VERSION_RE.match(path.name)
        if match:
            _contained_scene_file(job_dir, path)
            values.append(int(match.group("version")))
    return max(values) if values else None


def _source_blend(job_dir: Path, base_version: int | None) -> tuple[Path, int | None]:
    if base_version == 0:
        path = job_dir / "_regen" / "scene.blend"
        version = None
    elif base_version is not None:
        path = _version_path(job_dir, base_version)
        version = base_version
    else:
        version = _latest_version(job_dir)
        path = (
            _version_path(job_dir, version)
            if version is not None
            else job_dir / "_regen" / "scene.blend"
        )
    return _contained_scene_file(job_dir, path), version


def _safe_name(value: str, field: str) -> str:
    value = value.strip()
    if (
        not value
        or len(value) > MAX_NAME_LENGTH
        or any(ord(char) < 32 for char in value)
    ):
        raise BlenderWorkflowError(f"{field} must be a non-empty Blender name")
    return value


def _vec3(
    value: list[float] | tuple[float, float, float],
    field: str,
    *,
    positive: bool = False,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise BlenderWorkflowError(f"{field} must contain exactly three numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise BlenderWorkflowError(f"{field} must contain exactly three numbers")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise BlenderWorkflowError(f"{field} must contain finite numbers")
    if positive and not all(item > 0 for item in result):
        raise BlenderWorkflowError(f"{field} values must be positive")
    return result


def _bounded_float(
    value: Any,
    field: str,
    *,
    minimum: float,
    maximum: float,
    exclusive_min: bool = False,
    exclusive_max: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlenderWorkflowError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise BlenderWorkflowError(f"{field} must be a finite number")
    below = result <= minimum if exclusive_min else result < minimum
    above = result >= maximum if exclusive_max else result > maximum
    if below or above:
        lo = "(" if exclusive_min else "["
        hi = ")" if exclusive_max else "]"
        raise BlenderWorkflowError(
            f"{field} must be in {lo}{minimum}, {maximum}{hi}"
        )
    return result


def _element_slug(value: Any) -> str:
    slug = str(value or "").strip()
    if slug != "shell" and not ELEMENT_SLUG_RE.fullmatch(slug):
        raise BlenderWorkflowError(
            "slug must be a lowercase kebab-case world element name or 'shell'"
        )
    return slug


def _sanitize_params(action: str, params: dict[str, Any]) -> dict[str, Any]:
    if action == "inspect" or action == "snapshot":
        return {}
    if action == "import_world_element":
        return {"slug": _element_slug(params.get("slug"))}
    if action == "cleanup_mesh":
        result = {"object": _safe_name(str(params.get("object", "")), "object")}
        if params.get("merge_distance") is not None:
            result["merge_distance"] = _bounded_float(
                params["merge_distance"],
                "merge_distance",
                minimum=0.0,
                maximum=0.1,
                exclusive_min=True,
            )
        if params.get("decimate_ratio") is not None:
            result["decimate_ratio"] = _bounded_float(
                params["decimate_ratio"],
                "decimate_ratio",
                minimum=0.0,
                maximum=1.0,
                exclusive_min=True,
            )
        if params.get("min_component_frac") is not None:
            result["min_component_frac"] = _bounded_float(
                params["min_component_frac"],
                "min_component_frac",
                minimum=0.0,
                maximum=0.5,
                exclusive_max=True,
            )
        if params.get("shade_smooth") is not None:
            if not isinstance(params["shade_smooth"], bool):
                raise BlenderWorkflowError("shade_smooth must be boolean")
            result["shade_smooth"] = params["shade_smooth"]
        if len(result) == 1:
            raise BlenderWorkflowError(
                "cleanup_mesh requires at least one cleanup operation"
            )
        return result
    if action == "toggle_collection":
        if not isinstance(params.get("visible"), bool):
            raise BlenderWorkflowError("visible must be boolean")
        return {
            "collection": _safe_name(str(params.get("collection", "")), "collection"),
            "visible": params["visible"],
        }
    if action == "transform_object":
        result: dict[str, Any] = {
            "object": _safe_name(str(params.get("object", "")), "object")
        }
        for field in ("location", "rotation_degrees", "scale"):
            if params.get(field) is not None:
                result[field] = _vec3(params[field], field, positive=(field == "scale"))
        if len(result) == 1:
            raise BlenderWorkflowError(
                "transform_object requires at least one transform"
            )
        return result
    raise BlenderWorkflowError(f"unsupported Blender action: {action}")


def _sanitized_env() -> dict[str, str]:
    allowed = {
        "DISPLAY",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "TMPDIR",
        "USER",
        "WAYLAND_DISPLAY",
        "XDG_RUNTIME_DIR",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return env


def _run_process(
    command: list[str], timeout: int = BLENDER_TIMEOUT_S
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_sanitized_env(),
        start_new_session=True,
    )
    try:
        output, _unused = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _unused = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _unused = process.communicate()
        raise BlenderWorkflowError(
            f"Blender action exceeded {timeout} seconds"
        ) from exc
    return process.returncode, "\n".join(output.splitlines()[-30:])


@contextmanager
def _job_lock(job_dir: Path) -> Iterator[None]:
    workflow_dir = _workflow_dir(job_dir)
    workflow_dir.mkdir(parents=True, exist_ok=True)
    lock_path = workflow_dir / "workflow.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def inspect_job(job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    meta = _read_meta(job_id)
    export_manifest = manifests.read_json(job_dir / "_exports" / "manifest.json")
    scene_manifest = manifests.read_json(job_dir / "_regen" / "scene_manifest.json")
    dimensions_path = job_dir / "dimensions.json"
    dimensions = []
    if dimensions_path.is_file():
        try:
            value = json.loads(dimensions_path.read_text())
            dimensions = value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            pass
    source, base_version = _source_blend(job_dir, None)
    return {
        "job_id": job_id,
        "status": meta.get("status"),
        "blend": manifests.relative_job_path(source, job_dir),
        "base_version": base_version,
        "coordinate_system": manifests.coordinate_record(meta),
        "geo": meta.get("geo"),
        "dimensions": dimensions,
        "scene_manifest": scene_manifest,
        "exports": export_manifest,
        "versions": list_versions(job_id),
        "approved_root": str(OUTPUT_ROOT),
    }


def run_action(
    job_id: str,
    action: str,
    params: dict[str, Any] | None = None,
    *,
    base_version: int | None = None,
    note: str = "",
) -> dict[str, Any]:
    if action not in READ_ACTIONS | MUTATING_ACTIONS:
        raise BlenderWorkflowError(f"unsupported Blender action: {action}")
    if not BLENDER_BIN.is_file() or not ACTION_SCRIPT.is_file():
        raise BlenderWorkflowError("Blender 4.5 LTS workflow toolchain is unavailable")
    safe_params = _sanitize_params(action, params or {})
    job_dir = _job_dir(job_id)

    # Host-side path resolution: the Blender process never receives a
    # free-form path, only one resolved and containment-checked here.
    import_glb: Path | None = None
    if action == "import_world_element":
        slug = safe_params["slug"]
        relative = "shell.glb" if slug == "shell" else f"elements/{slug}.glb"
        import_glb = _contained_scene_file(
            job_dir,
            job_dir / "_world" / relative,
            missing=(
                f"no built world element '{slug}' — solidify the world first"
            ),
        )
        # Defense in depth: the polish route validates uploads, but this file
        # is about to be fed to Blender's full glTF importer — re-assert it is
        # a self-contained GLB (glb_check rejects external buffer/image URIs)
        # so a file that arrived by any other path gets the same gate.
        try:
            glb_check.validate_glb(import_glb)
        except ValueError as exc:
            raise BlenderWorkflowError(
                f"world element '{slug}' failed GLB validation: {exc}"
            ) from exc

    with _job_lock(job_dir):
        source, resolved_base = _source_blend(job_dir, base_version)
        version = None
        final_output = None
        staged_output = None
        if action in MUTATING_ACTIONS:
            version = (_latest_version(job_dir) or 0) + 1
            final_output = _version_path(job_dir, version)
            final_output.parent.mkdir(parents=True, exist_ok=True)
            staged_output = final_output.with_name(f".building-{final_output.name}")
            staged_output.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(prefix="splatlab-blender-") as temp_name:
            temp_dir = Path(temp_name)
            request_path = temp_dir / "request.json"
            response_path = temp_dir / "response.json"
            request_payload = {
                "schema": "dev.splatlab.blender-action/v1",
                "job_id": job_id,
                "action": action,
                "params": safe_params,
                "source_blend": str(source),
                "output_blend": str(staged_output) if staged_output else None,
            }
            if import_glb is not None:
                request_payload["import_glb"] = str(import_glb)
            request_path.write_text(json.dumps(request_payload))
            os.chmod(request_path, 0o600)
            command = [
                str(BLENDER_BIN),
                "--disable-autoexec",
                "--background",
                str(source),
                "--python",
                str(ACTION_SCRIPT),
                "--",
                str(request_path),
                str(response_path),
            ]
            return_code, log = _run_process(command)
            response = manifests.read_json(response_path)
            if return_code != 0 or not response or response.get("status") != "ok":
                if staged_output:
                    staged_output.unlink(missing_ok=True)
                detail = response.get("error") if response else log
                raise BlenderWorkflowError(
                    f"Blender {action} failed (exit {return_code}): {detail}"
                )

        if final_output and staged_output:
            if not staged_output.is_file():
                raise BlenderWorkflowError(
                    "Blender reported success without a version file"
                )
            os.replace(staged_output, final_output)
            receipt = {
                "schema": "dev.splatlab.blender-receipt/v1",
                "job_id": job_id,
                "version": version,
                "action": action,
                "params": safe_params,
                "note": note[:500],
                "base": {
                    "version": resolved_base,
                    "path": manifests.relative_job_path(source, job_dir),
                    **manifests.file_identity(source),
                },
                "output": {
                    "path": manifests.relative_job_path(final_output, job_dir),
                    **manifests.file_identity(final_output),
                },
                "created_at": manifests.utc_now(),
                "blender": response.get("blender"),
                "result": response.get("result"),
            }
            receipt_path = _receipts_dir(job_dir) / f"scene-v{version:04d}.json"
            manifests.atomic_write_json(receipt_path, receipt)
            return receipt

        return {
            "job_id": job_id,
            "action": action,
            "source": manifests.relative_job_path(source, job_dir),
            "base_version": resolved_base,
            "result": response.get("result"),
            "blender": response.get("blender"),
        }


def _exports_dir(job_dir: Path) -> Path:
    return _workflow_dir(job_dir) / "exports"


def export_glb(
    job_id: str,
    *,
    base_version: int | None = None,
    object_name: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Export a blend version (or the assembled scene) as a GLB artifact.

    Read-only with respect to the version history — no new .blend is created.
    Output lands at _blender/exports/scene-vNNNN.glb via a .building- stage,
    with a MANDATORY glb_check readback (the open3d lesson: never trust an
    exporter's exit code) and a receipt beside it. Re-exporting the same base
    version replaces the derived artifact and its receipt — versions are
    immutable, exports are derived. With object_name set, only that object is
    exported and the stem gains a slug suffix so the whole-scene export is
    never clobbered. The exported file feeds the polish-upload route;
    ingestion stays a separate, audited step.
    """
    if not BLENDER_BIN.is_file() or not ACTION_SCRIPT.is_file():
        raise BlenderWorkflowError("Blender 4.5 LTS workflow toolchain is unavailable")
    export_params: dict[str, Any] = {}
    stem_suffix = ""
    if object_name is not None:
        export_params["object"] = _safe_name(str(object_name), "object_name")
        object_slug = re.sub(
            r"[^a-z0-9]+", "-", export_params["object"].lower()
        ).strip("-")[:40]
        if not object_slug:
            raise BlenderWorkflowError(
                "object_name has no filename-safe characters"
            )
        stem_suffix = f"-{object_slug}"
    job_dir = _job_dir(job_id)
    with _job_lock(job_dir):
        source, resolved_base = _source_blend(job_dir, base_version)
        exports_dir = _exports_dir(job_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        stem = f"scene-v{(resolved_base or 0):04d}{stem_suffix}"
        final_output = exports_dir / f"{stem}.glb"
        # The stage name must still END in .glb — suffix-dispatching consumers
        # (the splat-transform lesson) and glb_check both key on it.
        staged_output = final_output.with_name(f".building-{final_output.name}")
        staged_output.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory(prefix="splatlab-blender-") as temp_name:
            temp_dir = Path(temp_name)
            request_path = temp_dir / "request.json"
            response_path = temp_dir / "response.json"
            request_path.write_text(
                json.dumps(
                    {
                        "schema": "dev.splatlab.blender-action/v1",
                        "job_id": job_id,
                        "action": "export_glb",
                        "params": export_params,
                        "source_blend": str(source),
                        "output_blend": None,
                        "output_glb": str(staged_output),
                    }
                )
            )
            os.chmod(request_path, 0o600)
            command = [
                str(BLENDER_BIN),
                "--disable-autoexec",
                "--background",
                str(source),
                "--python",
                str(ACTION_SCRIPT),
                "--",
                str(request_path),
                str(response_path),
            ]
            return_code, log = _run_process(command)
            response = manifests.read_json(response_path)
            if return_code != 0 or not response or response.get("status") != "ok":
                staged_output.unlink(missing_ok=True)
                detail = response.get("error") if response else log
                raise BlenderWorkflowError(
                    f"Blender export_glb failed (exit {return_code}): {detail}"
                )

        if not staged_output.is_file():
            raise BlenderWorkflowError(
                "Blender reported a successful export without a file"
            )
        try:
            gltf_summary = glb_check.validate_glb(staged_output)
        except ValueError as exc:
            staged_output.unlink(missing_ok=True)
            raise BlenderWorkflowError(
                f"exported GLB failed structural validation: {exc}"
            ) from exc
        os.replace(staged_output, final_output)
        receipt = {
            "schema": "dev.splatlab.blender-export-receipt/v1",
            "job_id": job_id,
            "action": "export_glb",
            "params": export_params,
            "note": note[:500],
            "base": {
                "version": resolved_base,
                "path": manifests.relative_job_path(source, job_dir),
                **manifests.file_identity(source),
            },
            "output": {
                "path": manifests.relative_job_path(final_output, job_dir),
                **manifests.file_identity(final_output),
            },
            "gltf": gltf_summary,
            "created_at": manifests.utc_now(),
            "blender": response.get("blender"),
            "result": response.get("result"),
        }
        manifests.atomic_write_json(exports_dir / f"{stem}.json", receipt)
        return receipt


def restore_version(job_id: str, version: int, note: str = "") -> dict[str, Any]:
    job_dir = _job_dir(job_id)
    with _job_lock(job_dir):
        source = _version_path(job_dir, version)
        if not source.exists() and not source.is_symlink():
            raise BlenderWorkflowError(f"Blender version {version} does not exist")
        _contained_scene_file(job_dir, source)
        new_version = (_latest_version(job_dir) or 0) + 1
        output = _version_path(job_dir, new_version)
        output.parent.mkdir(parents=True, exist_ok=True)
        staged_output = output.with_name(f".building-{output.name}")
        staged_output.unlink(missing_ok=True)
        try:
            shutil.copy2(source, staged_output)
            os.replace(staged_output, output)
        finally:
            staged_output.unlink(missing_ok=True)
        receipt = {
            "schema": "dev.splatlab.blender-receipt/v1",
            "job_id": job_id,
            "version": new_version,
            "action": "restore_version",
            "params": {"restored_version": version},
            "note": note[:500],
            "base": {
                "version": version,
                "path": manifests.relative_job_path(source, job_dir),
                **manifests.file_identity(source),
            },
            "output": {
                "path": manifests.relative_job_path(output, job_dir),
                **manifests.file_identity(output),
            },
            "created_at": manifests.utc_now(),
        }
        manifests.atomic_write_json(
            _receipts_dir(job_dir) / f"scene-v{new_version:04d}.json", receipt
        )
        return receipt


def open_blend(job_id: str, version: int | None = None) -> dict[str, Any]:
    if not BLENDER_BIN.is_file():
        raise BlenderWorkflowError("Blender 4.5 LTS is unavailable")
    job_dir = _job_dir(job_id)
    source, resolved_version = _source_blend(job_dir, version)
    process = subprocess.Popen(
        [str(BLENDER_BIN), "--disable-autoexec", str(source)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_sanitized_env(),
        start_new_session=True,
    )
    return {
        "job_id": job_id,
        "version": resolved_version,
        "blend": manifests.relative_job_path(source, job_dir),
        "pid": process.pid,
    }
