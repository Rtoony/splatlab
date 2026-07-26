"""Fixed + extensible semantic-class taxonomy for the paint tool.

Pure stdlib on purpose: the app venv has no numpy, and routes need to
validate class ids without heavy imports. The checked-in
`class_taxonomy.json` is the fixed base; a job may EXTEND it (never
redefine built-ins) via `<jobdir>/_langfield/class_taxonomy_extra.json`
with the same `classes` list shape. Every entry carries its generative
meaning (base colour, PBR constants, noise params, capture-blend ratio),
so adding a class IS wiring what it means downstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TAXONOMY_PATH = Path(__file__).with_name("class_taxonomy.json")
EXTRA_NAME = "class_taxonomy_extra.json"
CATEGORIES = ("ground", "vegetation", "structure", "water", "object")

_ID_OK = set("abcdefghijklmnopqrstuvwxyz0123456789-_")


class TaxonomyError(ValueError):
    """The taxonomy (or a job extension) is malformed — fail loud."""


def _validate_class(entry: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TaxonomyError(f"{source}: class entry is not an object")
    cid = entry.get("id")
    if (not isinstance(cid, str) or not cid or len(cid) > 40
            or not set(cid) <= _ID_OK):
        raise TaxonomyError(f"{source}: bad class id {cid!r}")
    if not isinstance(entry.get("display"), str) or not entry["display"]:
        raise TaxonomyError(f"{source}: class {cid} has no display name")
    color = entry.get("color")
    if (not isinstance(color, str) or len(color) != 7 or color[0] != "#"
            or any(c not in "0123456789abcdefABCDEF" for c in color[1:])):
        raise TaxonomyError(f"{source}: class {cid} has bad color {color!r}")
    if entry.get("category") not in CATEGORIES:
        raise TaxonomyError(
            f"{source}: class {cid} category must be one of {CATEGORIES}")
    queries = entry.get("queries")
    if queries is not None and (
            not isinstance(queries, list)
            or any(not isinstance(q, str) or not q for q in queries)):
        raise TaxonomyError(f"{source}: class {cid} queries must be strings")
    gen = entry.get("generative")
    if gen is not None:
        if not isinstance(gen, dict):
            raise TaxonomyError(f"{source}: class {cid} generative is not an object")
        rgb = gen.get("base_rgb")
        if (not isinstance(rgb, list) or len(rgb) != 3
                or any(not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in rgb)):
            raise TaxonomyError(f"{source}: class {cid} base_rgb must be 3 floats in 0..1")
    return entry


def load_taxonomy(job_dir: Path | None = None) -> dict[str, Any]:
    """Built-ins merged with a job's extend-only extras. Raises TaxonomyError
    on any malformation (a broken taxonomy must never half-load)."""
    try:
        base = json.loads(TAXONOMY_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TaxonomyError(f"class_taxonomy.json unreadable: {exc}") from exc
    classes = [_validate_class(c, source="builtin") for c in base.get("classes", [])]
    ids = {c["id"] for c in classes}
    if len(ids) != len(classes):
        raise TaxonomyError("builtin taxonomy has duplicate ids")

    if job_dir is not None:
        extra_path = Path(job_dir) / "_langfield" / EXTRA_NAME
        if extra_path.is_file():
            try:
                extra = json.loads(extra_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise TaxonomyError(f"{EXTRA_NAME} unreadable: {exc}") from exc
            for entry in extra.get("classes", []):
                validated = _validate_class(entry, source=EXTRA_NAME)
                if validated["id"] in ids:
                    raise TaxonomyError(
                        f"{EXTRA_NAME}: cannot redefine built-in class "
                        f"{validated['id']!r} (extend-only)")
                classes.append(validated)
                ids.add(validated["id"])

    return {
        "schema": base.get("schema"),
        "version": base.get("version"),
        "classes": classes,
    }


def class_ids(job_dir: Path | None = None) -> set[str]:
    return {c["id"] for c in load_taxonomy(job_dir)["classes"]}


def class_by_id(taxonomy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in taxonomy["classes"]}
