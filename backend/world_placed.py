"""The authored-asset registry: fantasy props PLACED into a captured world.

Captured elements are DERIVED — scene_solidify writes world.json from the
capture's instances, and world_collision derives world_manifest.json from
that — so anything not in the capture is erased by every rebuild. Operator
regrades solved this exact problem with a sticky sidecar (regrades.json)
that world_collision re-reads and re-applies each run; this module is the
same pattern for PLACED assets: `_world/placed.json` is written only by the
audited place/remove door, read by world_collision on every run, and
re-applied over the derived element list, so authored props survive
solidify/collision rebuilds.

Shaped like world_interactions/world_restyle: closed vocabulary, bounded
fields, fail-loud validator, one write gate. Pure stdlib (plus the stdlib-
only artifact_manifest) so the app venv AND the mesh env both import it.

Slug collisions: the door refuses any slug that exists at create time. If a
LATER re-inventory mints a captured instance with the same slug, the
captured element wins, the conflict is recorded loudly in the manifest's
`authored_conflicts`, and the placed GLB stays on disk for re-placing under
a new slug — silent shadowing in either direction is the failure mode this
contract exists to prevent.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import artifact_manifest as manifests

PLACED_SCHEMA = "dev.splatlab.world-placed/v1"
PLACED_FILENAME = "placed.json"
ROLES = ("static", "prop", "environment")
MAX_PLACED = 64
MAX_LABEL = 80
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")

_ENTRY_KEYS = {"slug", "label", "role", "provenance", "glb", "aabb", "extent",
               "gltf", "source_filename", "uploaded_bytes", "placed_at",
               "file", "warnings"}


class PlacedError(ValueError):
    """A placed-registry document that must not be written."""


def new_placed(job_id: str) -> dict[str, Any]:
    return {"schema": PLACED_SCHEMA, "v": 1, "job_id": str(job_id)[:80],
            "elements": []}


def _finite3(value: Any, name: str, slug: str) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                   or not math.isfinite(x) for x in value)):
        raise PlacedError(f"{slug}: {name} must be 3 finite numbers")
    return [round(float(x), 4) for x in value]


def _validate_entry(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PlacedError("a placed entry must be an object")
    slug = raw.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        raise PlacedError(
            f"bad placed slug {slug!r} — lowercase kebab, at most 40 chars")
    if slug == "shell":
        raise PlacedError('"shell" is the environment, not a placeable slug')
    unknown = set(raw) - _ENTRY_KEYS
    if unknown:
        raise PlacedError(f"{slug}: unknown entry keys {sorted(unknown)}")
    role = raw.get("role", "static")
    if role not in ROLES:
        raise PlacedError(f"{slug}: role must be one of {ROLES}")
    label = raw.get("label") or slug
    if not isinstance(label, str) or not label.strip():
        raise PlacedError(f"{slug}: label must be a non-empty string")

    entry: dict[str, Any] = {
        "slug": slug,
        "label": label.strip()[:MAX_LABEL],
        "role": role,
        "provenance": "authored",
        # Derived, never trusted from the document — one file per slug.
        "glb": f"{slug}.glb",
        "extent": _finite3(raw.get("extent"), "extent", slug),
        "aabb": {
            "min": _finite3((raw.get("aabb") or {}).get("min"), "aabb.min", slug),
            "max": _finite3((raw.get("aabb") or {}).get("max"), "aabb.max", slug),
        },
        "placed_at": str(raw.get("placed_at") or ""),
    }
    for lo, hi in zip(entry["aabb"]["min"], entry["aabb"]["max"]):
        if lo > hi:
            raise PlacedError(f"{slug}: aabb min exceeds max")
    for opt in ("gltf", "file"):
        if raw.get(opt) is not None:
            if not isinstance(raw[opt], dict):
                raise PlacedError(f"{slug}: {opt} must be an object")
            entry[opt] = raw[opt]
    if raw.get("source_filename") is not None:
        entry["source_filename"] = str(raw["source_filename"])[:120]
    if raw.get("uploaded_bytes") is not None:
        size = raw["uploaded_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise PlacedError(f"{slug}: uploaded_bytes must be a non-negative int")
        entry["uploaded_bytes"] = size
    if raw.get("warnings") is not None:
        warnings = raw["warnings"]
        if (not isinstance(warnings, list)
                or any(not isinstance(w, str) for w in warnings)):
            raise PlacedError(f"{slug}: warnings must be strings")
        entry["warnings"] = [w[:200] for w in warnings][:8]
    return entry


def validate_placed(document: Any) -> dict[str, Any]:
    """The one gate. Returns the normalised registry or raises PlacedError."""
    if not isinstance(document, dict):
        raise PlacedError("a placed registry must be an object")
    raw = document.get("elements") or []
    if not isinstance(raw, list):
        raise PlacedError("elements must be a list")
    if len(raw) > MAX_PLACED:
        raise PlacedError(f"too many placed assets (max {MAX_PLACED})")
    elements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        entry = _validate_entry(item)
        if entry["slug"] in seen:
            raise PlacedError(f"duplicate placed slug {entry['slug']!r}")
        seen.add(entry["slug"])
        elements.append(entry)
    return {"schema": PLACED_SCHEMA, "v": 1,
            "job_id": str(document.get("job_id", ""))[:80],
            "elements": elements}


def read_placed(world_dir: Path, job_id: str, *,
                strict: bool = True) -> dict[str, Any]:
    """The stored registry, or a fresh empty one when the file is absent.

    strict=True (the door): a corrupt registry RAISES — treating it as empty
    would let the next place silently orphan every prior entry's record.
    strict=False (rebuilds): degrade to empty; a collision run must not brick
    on registry damage — the caller logs it loudly instead.
    """
    path = Path(world_dir) / PLACED_FILENAME
    if not path.is_file():
        return new_placed(job_id)
    try:
        return validate_placed(json.loads(path.read_text()))
    except (OSError, ValueError) as exc:
        if strict:
            raise PlacedError(f"{PLACED_FILENAME} unreadable: {exc}") from exc
        return new_placed(job_id)


def write_placed(world_dir: Path, document: dict[str, Any]) -> dict[str, Any]:
    validated = validate_placed(document)
    manifests.atomic_write_json(Path(world_dir) / PLACED_FILENAME, validated)
    return validated


def manifest_element(entry: dict[str, Any]) -> dict[str, Any]:
    """A world_manifest.json elements[] record for one placed asset.

    The prop fallback block is HONEST, not aspirational: with no UCX files
    the walker builds a convex hull from the prop's own render mesh, so
    "render_mesh_fallback" describes what physics actually does. The next
    collision run upgrades it to real CoACD hulls.

    "environment" is authored walked-on architecture (terrain skirts, walls,
    platforms): the walker merges its render triangles straight into the
    player BVH, so complex_as_simple with zero hulls is the truthful record —
    no hull is ever built for it.
    """
    if entry["role"] == "prop":
        collision = {"ok": True, "strategy": "render_mesh_fallback",
                     "hulls": 0}
    else:  # static and environment both collide as their own render mesh
        collision = {"ok": True, "strategy": "complex_as_simple", "hulls": 0}
    classification = (["authored environment geometry (placed.json)"]
                      if entry["role"] == "environment"
                      else ["authored set-dressing (placed.json)"])
    return {"slug": entry["slug"], "label": entry["label"],
            "role": entry["role"], "provenance": "authored",
            "glb": entry["glb"], "extent": entry["extent"],
            "classification": classification,
            "collision": collision}


def apply_placed_to_manifest(manifest_doc: dict[str, Any],
                             placed_doc: dict[str, Any] | None,
                             ) -> tuple[dict[str, Any], list[str]]:
    """(merged manifest, conflicting slugs) — pure, never mutates its inputs.

    Strip-then-append makes it idempotent: existing authored entries are
    dropped and rebuilt from the registry, so re-applying can never
    duplicate. A captured slug always wins and is reported, never clobbered.
    """
    merged = json.loads(json.dumps(manifest_doc))
    kept = [e for e in merged.get("elements") or []
            if (e or {}).get("provenance") != "authored"]
    captured = {e.get("slug") for e in kept} | {"shell"}
    conflicts: list[str] = []
    appended: list[dict[str, Any]] = []
    for entry in (placed_doc or {}).get("elements") or []:
        if entry["slug"] in captured:
            conflicts.append(entry["slug"])
            continue
        appended.append(manifest_element(entry))
    merged["elements"] = kept + appended
    counts = merged.get("counts")
    if isinstance(counts, dict):
        counts["authored"] = len(appended)
        # The door is often the only writer between collision runs — an
        # environment piece must be countable the moment it lands, not only
        # after the next world_collision pass.
        counts["environment"] = sum(
            1 for e in merged["elements"] if e.get("role") == "environment")
    merged["authored_conflicts"] = conflicts
    return merged, conflicts
