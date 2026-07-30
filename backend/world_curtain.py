"""The curtain layer: where the photograph ends and the built world begins.

A sparse capture has a good core and a fringe of orbit floaters. The curtain
is a validated document beside interactions.json describing a boundary shape
(sphere or box, walker-frame scene units): the walker fades the gaussian
backdrop to nothing OUTSIDE it across `soft_edge` units, so authored
environment geometry takes over past the seam instead of competing with
noise. Same discipline as every world document — closed vocabulary, bounded
finite numbers, fail-loud validation — because a curtain is look-load-bearing:
a NaN centre or a zero-size shape is a black screen, not a tuning choice.

The document is data only; rendering lives in the walker (a Spark SplatEdit
SDF layer — non-destructive, nothing on disk is rewritten).
"""

from __future__ import annotations

import math
from typing import Any

CURTAIN_SCHEMA = "dev.splatlab.world-curtain/v1"
VERSION = 1

SHAPES = ("sphere", "box")
MAX_COORD = 1e5
MAX_RADIUS = 1e5
MAX_SOFT_EDGE = 1e4


class CurtainError(ValueError):
    """A curtain document violates its contract."""


def default_curtain(job_id: str,
                    world_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Disabled, centred, generously sized — a sane starting shape for the UI.

    Radius = 1.2x the shell half-diagonal when the manifest knows one, so
    enabling the default curtain changes nothing visible until it is pulled
    inward past the capture fringe.
    """
    radius = 10.0
    shell_extent = ((world_manifest or {}).get("shell") or {}).get("extent")
    if isinstance(shell_extent, list) and len(shell_extent) == 3:
        try:
            half_diag = sum((float(e) / 2) ** 2 for e in shell_extent) ** 0.5
            if math.isfinite(half_diag) and half_diag > 0:
                radius = round(half_diag * 1.2, 3)
        except (TypeError, ValueError):
            pass
    return {
        "schema": CURTAIN_SCHEMA,
        "version": VERSION,
        "job_id": job_id,
        "enabled": False,
        "shape": "sphere",
        "center": [0.0, 0.0, 0.0],
        "radius": radius,
        "half_extents": [radius, radius, radius],
        "soft_edge": round(max(radius * 0.15, 0.5), 3),
    }


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurtainError(f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise CurtainError(f"{name} must be within [{low}, {high}]")
    return v


def _vec3(value: Any, name: str, low: float, high: float) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise CurtainError(f"{name} must be a list of 3 numbers")
    return [_number(v, f"{name}[{i}]", low, high) for i, v in enumerate(value)]


def validate_curtain(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise CurtainError("curtain must be an object")
    if document.get("schema") != CURTAIN_SCHEMA:
        raise CurtainError(f"schema must be {CURTAIN_SCHEMA!r}")
    if document.get("version") != VERSION:
        raise CurtainError(f"version must be {VERSION}")
    job_id = document.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise CurtainError("job_id must be a non-empty string")
    if not isinstance(document.get("enabled"), bool):
        raise CurtainError("enabled must be a boolean")
    shape = document.get("shape")
    if shape not in SHAPES:
        raise CurtainError(f"shape must be one of {SHAPES}")

    known = {"schema", "version", "job_id", "enabled", "shape", "center",
             "radius", "half_extents", "soft_edge"}
    unknown = sorted(set(document) - known)
    if unknown:
        raise CurtainError(f"unknown keys: {unknown} — the vocabulary is "
                           "closed so a typo cannot silently do nothing")

    return {
        "schema": CURTAIN_SCHEMA,
        "version": VERSION,
        "job_id": job_id,
        "enabled": document["enabled"],
        "shape": shape,
        "center": _vec3(document.get("center"), "center",
                        -MAX_COORD, MAX_COORD),
        "radius": _number(document.get("radius"), "radius",
                          1e-3, MAX_RADIUS),
        "half_extents": _vec3(document.get("half_extents"), "half_extents",
                              1e-3, MAX_RADIUS),
        "soft_edge": _number(document.get("soft_edge"), "soft_edge",
                             0.0, MAX_SOFT_EDGE),
    }
