"""R3: propose an interaction for every named thing in the world.

Hand-authoring affordances is what makes game worlds expensive; a captured
world already KNOWS what its elements are (the inventory named them) and how
big they are (the manifest measured them). This module turns that knowledge
into proposed interaction records through deterministic, explainable rules —
no model call, every proposal carries its rationale — and the existing
authoring gate (world_interactions.validate_interactions + the PUT route)
remains the only door into the live document.

Rules, in order:
  1. static elements       -> inspect  (you cannot move the environment)
  2. toggleable-looking    -> toggle   (lamp/light/screen nouns; off/on + tint)
  3. carryable prop        -> pickup   (physics can lift it: every extent
                                        under CARRYABLE_MAX_M)
  4. everything else       -> inspect

Proposals never clobber authored records: slugs that already have one are
skipped and reported as such.
"""

from __future__ import annotations

import re
from typing import Any

CARRYABLE_MAX_M = 0.75
TOGGLE_NOUNS = re.compile(
    r"\b(lamp|light|lantern|bulb|screen|monitor|tv|television|display|"
    r"switch|candle|heater|fan|radio|speaker)\b", re.IGNORECASE)

PROMPT_MAX = 60


def _prompt_for(label: str, slug: str) -> str:
    text = (label or slug.replace("-", " ")).strip()
    return f"the {text}"[:PROMPT_MAX] if not text.lower().startswith("the ") \
        else text[:PROMPT_MAX]


def _extent_metres(element: dict[str, Any], mpu: float | None,
                   units: str | None) -> list[float] | None:
    extent = element.get("extent")
    if (not isinstance(extent, list) or len(extent) != 3
            or not all(isinstance(v, (int, float)) for v in extent)):
        return None
    # units == "meters" means the extents are ALREADY metres (old-format docs
    # recorded the capture factor as mpu beside metre-baked extents — applying
    # it would double the calibration; review finding). The factor applies
    # only to scene-unit documents that carry a restamped calibration.
    factor = float(mpu) if (mpu and units != "meters") else 1.0
    return [float(v) * factor for v in extent]


def propose_affordances(
    world_manifest: dict[str, Any],
    existing_slugs: set[str] | None = None,
) -> dict[str, Any]:
    """Proposed interaction records + per-element rationale.

    `world_manifest` is the collision-graded document (roles + extents +
    meters_per_unit); `existing_slugs` are elements that already have an
    authored record and must not be touched.
    """
    existing = existing_slugs or set()
    mpu = world_manifest.get("meters_per_unit")
    units = world_manifest.get("units")
    proposals: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for element in world_manifest.get("elements") or []:
        slug = element.get("slug")
        if not slug:
            continue
        if element.get("role") == "environment":
            skipped.append({"slug": slug, "reason":
                            "environment geometry is walked on, not "
                            "interacted with"})
            continue
        if element.get("role") not in ("prop", "static"):
            skipped.append({"slug": slug, "reason": "element is not built"})
            continue
        if slug in existing:
            skipped.append({"slug": slug,
                            "reason": "an authored record already exists"})
            continue

        label = str(element.get("label") or "")
        prompt = _prompt_for(label, slug)
        extent_m = _extent_metres(element, mpu, units)
        name = f"{label} {slug}"

        calibrated = bool(mpu) or units == "meters"
        if element.get("role") == "static":
            verb, states, initial, effects = "inspect", ["seen"], "seen", {}
            rationale = "static: the environment can be looked at, not moved"
        elif TOGGLE_NOUNS.search(name):
            verb, states, initial = "toggle", ["off", "on"], "off"
            effects = {"on": {"tint": "#ffd27f"}, "off": {"tint": None}}
            rationale = "its name suggests something that switches on and off"
        elif not calibrated:
            # Without real-world scale, "small enough to carry" is a guess in
            # arbitrary units — measured live: an UNCALIBRATED forest scene
            # proposed picking up a tree stump. No size claims, no pickups.
            verb, states, initial, effects = "inspect", ["seen"], "seen", {}
            rationale = ("uncalibrated capture: real size unknown, so no "
                         "carry proposal (calibrate, then re-propose)")
        elif extent_m is not None and max(extent_m) <= CARRYABLE_MAX_M:
            verb, states, initial = "pickup", ["placed", "held"], "placed"
            effects = {"held": {"tint": "#9be7ff"}, "placed": {"tint": None}}
            rationale = (f"a prop small enough to carry "
                         f"(max extent {max(extent_m):.2f} m <= {CARRYABLE_MAX_M} m)")
        else:
            verb, states, initial, effects = "inspect", ["seen"], "seen", {}
            size = f"{max(extent_m):.2f} m" if extent_m else "unknown size"
            rationale = f"a prop too large to carry ({size}); look, do not lift"

        proposals.append({
            "record": {
                "slug": slug, "verb": verb, "prompt": prompt,
                "states": states, "initial": initial, "effects": effects,
            },
            "rationale": rationale,
        })

    return {"proposals": proposals, "skipped": skipped}
