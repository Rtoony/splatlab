"""scene_manifest.json — the single provenance-carrying record of a regenerated
scene (P6 lane). Written into <job>/_regen/, consumed by the Blender assembler
and the export endpoints; the geo/survey lane refuses anything it describes.

Pure stdlib (importable from the FastAPI venv, worker envs, and gates).
Validation is hand-rolled fail-loud checks, not jsonschema — no new deps.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from provenance import GENERATIVE_TAG, PROVENANCE_VALUES

MANIFEST_VERSION = 1
MANIFEST_NAME = "scene_manifest.json"

UNITS_MODES = ("meters", "scene-units")
STATES = ("building", "built", "approved")


class ManifestError(ValueError):
    """scene_manifest.json failed validation."""


def new_manifest(job_id: str, units_mode: str, meters_per_unit: float | None) -> dict:
    if units_mode not in UNITS_MODES:
        raise ManifestError(f"units mode {units_mode!r} not in {UNITS_MODES}")
    if units_mode == "meters" and not (meters_per_unit and meters_per_unit > 0):
        raise ManifestError("units mode 'meters' requires a positive meters_per_unit")
    return {
        "version": MANIFEST_VERSION,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "building",
        "doctrine": GENERATIVE_TAG,
        "units": {"mode": units_mode, "meters_per_unit": meters_per_unit},
        "elements": [],
    }


def add_element(manifest: dict, *, slug: str, provenance: str, files: dict,
                transform_4x4: list | None = None,
                registration: dict | None = None,
                skipped: str | None = None) -> dict:
    el = {"slug": slug, "provenance": provenance, "files": files,
          "transform_4x4": transform_4x4, "registration": registration,
          "skipped": skipped}
    manifest["elements"].append(el)
    return el


def validate_manifest(manifest: dict) -> None:
    """Raise ManifestError on the first structural problem (fail-loud)."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest is not an object")
    for key in ("version", "job_id", "created_at", "state", "doctrine", "units", "elements"):
        if key not in manifest:
            raise ManifestError(f"missing key: {key}")
    if manifest["version"] != MANIFEST_VERSION:
        raise ManifestError(f"version {manifest['version']} != {MANIFEST_VERSION}")
    if manifest["state"] not in STATES:
        raise ManifestError(f"state {manifest['state']!r} not in {STATES}")
    if manifest["doctrine"] != GENERATIVE_TAG:
        raise ManifestError("doctrine line missing/altered — provenance rails broken")
    units = manifest["units"]
    if units.get("mode") not in UNITS_MODES:
        raise ManifestError(f"units.mode {units.get('mode')!r} not in {UNITS_MODES}")
    if units["mode"] == "meters" and not (units.get("meters_per_unit") or 0) > 0:
        raise ManifestError("units.mode 'meters' without a positive meters_per_unit")
    slugs = set()
    for i, el in enumerate(manifest["elements"]):
        where = f"elements[{i}]"
        for key in ("slug", "provenance", "files"):
            if not el.get(key) and el.get(key) != {}:
                raise ManifestError(f"{where}: missing {key}")
        if el["slug"] in slugs:
            raise ManifestError(f"{where}: duplicate slug {el['slug']!r}")
        slugs.add(el["slug"])
        if el["provenance"] not in PROVENANCE_VALUES:
            raise ManifestError(
                f"{where}: provenance {el['provenance']!r} not in {PROVENANCE_VALUES}")
        if el["provenance"] == "proxy":
            if not el.get("skipped") and el.get("transform_4x4") is None:
                raise ManifestError(f"{where}: proxy element without transform_4x4")
        t = el.get("transform_4x4")
        if t is not None and (len(t) != 4 or any(len(row) != 4 for row in t)):
            raise ManifestError(f"{where}: transform_4x4 is not 4x4")


def write_manifest(dirpath: str | Path, manifest: dict) -> Path:
    """Validate then atomically write <dirpath>/scene_manifest.json."""
    validate_manifest(manifest)
    out = Path(dirpath) / MANIFEST_NAME
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(out)
    return out


def read_manifest(dirpath: str | Path) -> dict:
    manifest = json.loads((Path(dirpath) / MANIFEST_NAME).read_text())
    validate_manifest(manifest)
    return manifest
