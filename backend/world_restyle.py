"""Restyle a captured world: recolour, relight, re-surface.

The C2 starter for the decorating / dungeon-ifying destination. A capture is
a photograph of one moment under one light; restyling is how it becomes a
place you can dress. Three verbs, all NON-destructive — the document is a
diff the walker applies over the baked geometry, so the capture on disk is
never touched and any restyle can be dropped by deleting the file:

  recolour  per-element tint (multiplied into the baked albedo)
  resurface per-element material swap to a taxonomy CLASS, drawn from the
            same procedural tiles the ground/object bakes use, projected in
            world space (the baked UV charts are per-element atlases; a tile
            pasted through them would scramble)
  relight   one scene-wide lighting preset + intensity

Shaped exactly like world_interactions: a validated document with bounded
fields and a closed vocabulary, written through one gate. A restyle is
cosmetic, but it is authored content the walker trusts, so it is validated
with the same suspicion as an affordance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RESTYLE_FILENAME = "restyle.json"

# Closed vocabulary. Adding a preset is a deliberate edit here, not something
# a client can invent — the walker maps each name to a real light rig.
LIGHTING_PRESETS = ("as-captured", "noon", "golden-hour", "overcast",
                    "dusk", "night", "dungeon")
MAX_ELEMENTS = 256
MAX_SLUG = 64
MAX_CLASS = 48
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
INTENSITY_MIN, INTENSITY_MAX = 0.0, 3.0


class RestyleError(ValueError):
    """A restyle document that must not be written."""


def new_restyle(job_id: str) -> dict[str, Any]:
    return {"v": 1, "job_id": str(job_id)[:80], "elements": {},
            "lighting": {"preset": "as-captured", "intensity": 1.0}}


def _validate_element(slug: Any, raw: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(slug, str) or not slug.strip():
        raise RestyleError("every restyle key must be a non-empty element slug")
    slug = slug.strip()[:MAX_SLUG]
    if not isinstance(raw, dict):
        raise RestyleError(f"{slug}: a restyle entry must be an object")

    entry: dict[str, Any] = {}
    tint = raw.get("tint")
    if tint is not None:
        if not isinstance(tint, str) or not HEX_RE.match(tint):
            raise RestyleError(f"{slug}: tint must be #rrggbb, got {tint!r}")
        entry["tint"] = tint.lower()

    material = raw.get("material")
    if material is not None:
        if not isinstance(material, str) or not material.strip():
            raise RestyleError(f"{slug}: material must be a taxonomy class id")
        entry["material"] = material.strip()[:MAX_CLASS]

    scale = raw.get("material_scale")
    if scale is not None:
        if not isinstance(scale, (int, float)) or isinstance(scale, bool):
            raise RestyleError(f"{slug}: material_scale must be a number")
        if not 0.01 <= float(scale) <= 100.0:
            raise RestyleError(
                f"{slug}: material_scale {scale} outside 0.01..100 metres/tile")
        entry["material_scale"] = round(float(scale), 4)

    if not entry:
        raise RestyleError(
            f"{slug}: a restyle entry needs at least one of tint / material")
    return slug, entry


def _validate_lighting(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {"preset": "as-captured", "intensity": 1.0}
    if not isinstance(raw, dict):
        raise RestyleError("lighting must be an object")
    preset = raw.get("preset", "as-captured")
    if preset not in LIGHTING_PRESETS:
        raise RestyleError(
            f"unknown lighting preset {preset!r} — known: "
            f"{', '.join(LIGHTING_PRESETS)}")
    intensity = raw.get("intensity", 1.0)
    if not isinstance(intensity, (int, float)) or isinstance(intensity, bool):
        raise RestyleError("lighting intensity must be a number")
    if not INTENSITY_MIN <= float(intensity) <= INTENSITY_MAX:
        raise RestyleError(
            f"lighting intensity {intensity} outside "
            f"{INTENSITY_MIN}..{INTENSITY_MAX}")
    return {"preset": preset, "intensity": round(float(intensity), 3)}


def validate_restyle(document: Any,
                     known_slugs: set[str] | None = None,
                     known_classes: set[str] | None = None) -> dict[str, Any]:
    """The one gate. Returns the normalised document or raises RestyleError."""
    if not isinstance(document, dict):
        raise RestyleError("a restyle document must be an object")

    raw_elements = document.get("elements") or {}
    if not isinstance(raw_elements, dict):
        raise RestyleError("elements must be an object keyed by element slug")
    if len(raw_elements) > MAX_ELEMENTS:
        raise RestyleError(f"too many restyled elements (max {MAX_ELEMENTS})")

    elements: dict[str, Any] = {}
    for slug, raw in raw_elements.items():
        slug, entry = _validate_element(slug, raw)
        # An unknown slug or class would only ever surface as a mystery in the
        # walker — refuse it here, naming what does exist (interactions rule).
        if known_slugs is not None and slug not in known_slugs:
            raise RestyleError(
                f"{slug}: no such element in this world — known: "
                f"{', '.join(sorted(known_slugs)) or '(none)'}")
        if (known_classes is not None and "material" in entry
                and entry["material"] not in known_classes):
            raise RestyleError(
                f"{slug}: unknown material class {entry['material']!r} — known: "
                f"{', '.join(sorted(known_classes)) or '(none)'}")
        elements[slug] = entry

    return {
        "v": 1,
        "job_id": str(document.get("job_id", ""))[:80],
        "elements": elements,
        "lighting": _validate_lighting(document.get("lighting")),
    }


def read_restyle(world_dir: Path, job_id: str) -> dict[str, Any]:
    """The stored restyle, or a fresh empty one. Never raises on a bad file —
    a corrupt cosmetic document must not make a world unopenable."""
    path = Path(world_dir) / RESTYLE_FILENAME
    if not path.is_file():
        return new_restyle(job_id)
    try:
        return validate_restyle(json.loads(path.read_text()))
    except (OSError, ValueError):
        return new_restyle(job_id)


def write_restyle(world_dir: Path, document: dict[str, Any],
                  known_slugs: set[str] | None = None,
                  known_classes: set[str] | None = None) -> dict[str, Any]:
    validated = validate_restyle(document, known_slugs, known_classes)
    path = Path(world_dir) / RESTYLE_FILENAME
    path.write_text(json.dumps(validated, indent=2))
    return validated
