"""Dependency identities for geometry derived from a calibrated capture."""

from __future__ import annotations

from pathlib import Path
import math

import artifact_manifest as manifests


def collision_scale_to_world(world: dict, report: dict) -> float:
    frame = report.get("geometry_frame") or {}
    if frame.get("axis") != "y-up" or frame.get("units") != "scene-units" or world.get("units") != "meters":
        return 1.0
    scale = world.get("calibrated_from_meters_per_unit", world.get("meters_per_unit"))
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Metre world lacks a valid capture-to-world scale")
    return float(scale)


def scale_revision(job_dir: Path) -> dict:
    metadata = manifests.read_json(job_dir / "meta.json") or {}
    return {"scale_generation": int(metadata.get("scale_generation") or 0),
            "meters_per_unit": metadata.get("meters_per_unit")}


def shell_dependencies(job_dir: Path) -> dict:
    files = [job_dir / relative for relative in (
        "_scene/isolated/background.ply", "_scene/isolated/batch_isolate.json",
        "_mesh/mesh.ply", "_preview/splat.ply")]
    files.extend(sorted((job_dir / "_scene" / "surfaces").glob("patch_*.ply")))
    files.extend(sorted((job_dir / "_scene" / "ground").glob("*.npz")))
    return {"version": 1, "scale": scale_revision(job_dir),
            "files": {str(path.relative_to(job_dir)): manifests.file_identity(path, include_sha256=False)
                      for path in files if path.is_file()}}


def shell_is_current(job_dir: Path) -> bool:
    report = manifests.read_json(job_dir / "_world" / "collision_shell.json") or {}
    return (not report.get("source_override") and report.get("verdict") in {"PASS", "WALKABLE", "WALKABLE_NOT_WATERTIGHT"}
            and manifests.same_file_identity(job_dir / "_world" / "collision_shell.glb", report.get("output"))
            and report.get("dependencies") == shell_dependencies(job_dir))


def creative_dependencies(job_dir: Path) -> dict:
    paths = [job_dir / relative for relative in (
        "_preview/splat.ply", "_scene/inventory.json", "_scene/isolated/batch_isolate.json",
        "_langfield/class_labels.json", "_world/world.json", "_world/world_manifest.json")]
    paths.extend(sorted((job_dir / "_scene" / "surfaces").glob("patch_*.ply")))
    return {"scale": scale_revision(job_dir), "files": {
        str(path.relative_to(job_dir)): manifests.file_identity(path, include_sha256=False)
        for path in paths if path.is_file()}}
