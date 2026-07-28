"""Authored affordances for a walkable world, and the player state over them.

Why a sidecar in the world lane, not a scene_manifest v2
--------------------------------------------------------
Two manifests exist and share nothing. `_regen/scene_manifest.json` is the
Blender assembly recipe; `_world/world_manifest.json` describes the thing you
actually walk. Affordances belong to the second.

The decisive reason is the HITL, not just which file the walker reads.
`scene_assemble_approve` (`splat_route.py:6045-6063`) re-writes the manifest with
`state: "approved"`, and that state means exactly one thing: a human graded this
assembly. Putting authored interactions in the same document forces a choice
between "approved" silently coming to mean "approved at some point, with unknown
edits since", or making someone re-approve geometry because they labelled a light
switch. Neither is acceptable, so interactions live beside the world instead.

Two supporting facts: the slug spaces differ (`_regen/` has `ground` and no
`shell`; `_world/` has `shell`, no `ground`, and drops elements that failed to
solidify), and `scene_manifest`'s `doctrine` field exists to carry the generative
quarantine — authored data is neither captured nor generated, and filing it under
that marker would dilute what the marker means.

Sidecar discipline also means authoring never requires rebuilding the world, and
rebuilding never destroys what was authored. `scene_solidify.py` and
`world_collision.py` keep writing exactly what they write today.

The vocabulary
--------------
All the verbs reduce to the same thing — a named finite state set over an
element, plus a prompt, plus a reach — which is how you can tell it is the
minimum rather than an abstraction reached for. `open` is deliberately NOT a
separate verb: it is `toggle` with the state names `closed`/`open`, and shipping
it separately would buy a synonym and a second code path.

Effect keys are a CLOSED set and an unknown one is rejected. This is a deliberate
deviation from `scene_manifest`'s permissiveness: an effect that silently does
nothing is exactly the failure this codebase's fail-loud discipline exists to
prevent.

Pure stdlib, no FastAPI import, so it is unit-testable standalone.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import artifact_manifest as manifests

INTERACTIONS_SCHEMA = "dev.splatlab.world-interactions/v1"
STATE_SCHEMA = "dev.splatlab.world-state/v1"
VERSION = 1
INTERACTIONS_NAME = "interactions.json"
STATE_NAME = "state.json"

# Closed vocabulary. `pickup` validates and persists like any other two-state
# flip; only its player-side container is deferred (see `player` in the state
# document, which is the named seam for it).
VERBS = ("inspect", "toggle", "pickup")

# Closed effect set. `visible` is accepted by the schema but the walker wires
# `tint` first: `object.visible` already has one owner (the HUD's per-element eye
# toggle), and a second writer would produce an object the panel calls hidden.
EFFECT_KEYS = ("visible", "tint")

MAX_PROMPT = 80
MAX_TEXT = 1000
MAX_STATES = 8
MAX_RANGE_M = 25.0
MAX_ELEMENTS = 256
# Physics-disturbed prop poses live under the reserved player seam; a scene
# has at most MAX_ELEMENTS props, so this bound is generous, not tight.
MAX_PLAYER_POSES = 256

_STATE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_TINT_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


class InteractionsError(ValueError):
    """Bad affordance or state document. Always actionable, never swallowed."""


# ---------------------------------------------------------------------------
# What the world actually contains
# ---------------------------------------------------------------------------

def world_slugs(world_manifest: dict[str, Any] | None) -> set[str]:
    """Every slug an affordance may legally reference.

    The shell is included: it is the thing you stand inside, and excluding it
    would be an arbitrary hole. Elements the solidify stage failed to build
    (`role: "unbuilt"`) are excluded — there is no geometry to aim at.
    """
    if not isinstance(world_manifest, dict):
        return set()
    slugs = {
        str(element.get("slug"))
        for element in (world_manifest.get("elements") or [])
        if element.get("slug") and element.get("role") != "unbuilt"
    }
    shell = world_manifest.get("shell") or {}
    if shell.get("slug"):
        slugs.add(str(shell["slug"]))
    return slugs


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

def _validate_effects(raw: Any, states: list[str], *, slug: str) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise InteractionsError(f"{slug}: effects must be an object keyed by state")

    effects: dict[str, dict[str, Any]] = {}
    for state, effect in raw.items():
        state = str(state)
        if state not in states:
            raise InteractionsError(
                f"{slug}: effects mention state {state!r}, which is not in {states}")
        if not isinstance(effect, dict):
            raise InteractionsError(f"{slug}: effect for {state!r} must be an object")

        checked: dict[str, Any] = {}
        for key, value in effect.items():
            if key not in EFFECT_KEYS:
                raise InteractionsError(
                    f"{slug}: unknown effect {key!r}; supported effects are "
                    f"{', '.join(EFFECT_KEYS)}")
            if key == "visible":
                if not isinstance(value, bool):
                    raise InteractionsError(f"{slug}: effect 'visible' must be true or false")
            elif key == "tint":
                if value is not None and not (
                        isinstance(value, str) and _TINT_RE.match(value)):
                    raise InteractionsError(
                        f"{slug}: effect 'tint' must be #rrggbb or null, got {value!r}")
            checked[key] = value
        effects[state] = checked
    return effects


def validate_record(raw: Any, *, index: int) -> dict[str, Any]:
    """One interaction record, normalised. Raises on anything unusable."""
    where = f"element[{index}]"
    if not isinstance(raw, dict):
        raise InteractionsError(f"{where}: each element must be an object")

    slug = str(raw.get("slug") or "").strip()
    if not slug:
        raise InteractionsError(f"{where}: slug is required")
    # Mirrors the containment rule the world file route applies to slugs.
    if "/" in slug or "\\" in slug or slug in (".", "..") or slug.startswith("."):
        raise InteractionsError(f"{where}: unsafe slug {slug!r}")

    verb = str(raw.get("verb") or "").strip()
    if verb not in VERBS:
        raise InteractionsError(
            f"{slug}: unknown verb {verb!r}; supported verbs are {', '.join(VERBS)}. "
            f"An 'open' door is a toggle with the states closed/open.")

    prompt = str(raw.get("prompt") or slug).strip()
    if len(prompt) > MAX_PROMPT:
        raise InteractionsError(f"{slug}: prompt longer than {MAX_PROMPT} characters")

    states = raw.get("states")
    if not isinstance(states, list) or not states:
        raise InteractionsError(f"{slug}: states must be a non-empty list")
    if not all(isinstance(s, str) for s in states):
        raise InteractionsError(f"{slug}: states must all be strings")
    states = [s.strip() for s in states]
    if len(states) > MAX_STATES:
        raise InteractionsError(f"{slug}: more than {MAX_STATES} states")
    if len(set(states)) != len(states):
        raise InteractionsError(f"{slug}: duplicate state names")
    for state in states:
        if not _STATE_RE.match(state):
            raise InteractionsError(
                f"{slug}: state {state!r} must be lowercase alphanumeric with - or _")

    initial = raw.get("initial")
    initial = states[0] if initial is None else str(initial).strip()
    if initial not in states:
        raise InteractionsError(f"{slug}: initial {initial!r} is not one of {states}")

    range_m = raw.get("range_m", 2.0)
    try:
        range_m = float(range_m)
    except (TypeError, ValueError):
        raise InteractionsError(f"{slug}: range_m must be a number") from None
    if not (math.isfinite(range_m) and 0.0 < range_m <= MAX_RANGE_M):
        raise InteractionsError(
            f"{slug}: range_m must be finite and within (0, {MAX_RANGE_M}]")

    record = {
        "slug": slug,
        "verb": verb,
        "prompt": prompt,
        "states": states,
        "initial": initial,
        "effects": _validate_effects(raw.get("effects"), states, slug=slug),
        "range_m": round(range_m, 4),
    }
    text = raw.get("text")
    if text is not None:
        text = str(text).strip()
        if len(text) > MAX_TEXT:
            raise InteractionsError(f"{slug}: text longer than {MAX_TEXT} characters")
        if text:
            record["text"] = text
    return record


def validate_interactions(document: Any,
                          known_slugs: set[str] | None = None) -> dict[str, Any]:
    """Validate and normalise a whole interactions document."""
    if not isinstance(document, dict):
        raise InteractionsError("interactions document must be an object")
    if document.get("schema") != INTERACTIONS_SCHEMA:
        raise InteractionsError(
            f"schema must be {INTERACTIONS_SCHEMA!r}, got {document.get('schema')!r}")
    if document.get("version") != VERSION:
        raise InteractionsError(
            f"version must be {VERSION}, got {document.get('version')!r}")

    raw_elements = document.get("elements")
    if not isinstance(raw_elements, list):
        raise InteractionsError("elements must be a list")
    if len(raw_elements) > MAX_ELEMENTS:
        raise InteractionsError(f"more than {MAX_ELEMENTS} elements")

    elements = [validate_record(raw, index=i) for i, raw in enumerate(raw_elements)]

    seen: set[str] = set()
    for record in elements:
        if record["slug"] in seen:
            raise InteractionsError(
                f"{record['slug']}: appears twice; one interaction per element in v1")
        seen.add(record["slug"])
        if known_slugs is not None and record["slug"] not in known_slugs:
            raise InteractionsError(
                f"no element {record['slug']!r} in this world; known slugs: "
                f"{', '.join(sorted(known_slugs)) or '(none)'}")

    return {
        "schema": INTERACTIONS_SCHEMA,
        "version": VERSION,
        "job_id": str(document.get("job_id") or ""),
        "authored_at": str(document.get("authored_at") or manifests.utc_now()),
        "elements": elements,
    }


def new_interactions(job_id: str) -> dict[str, Any]:
    return {
        "schema": INTERACTIONS_SCHEMA, "version": VERSION,
        "job_id": str(job_id), "authored_at": manifests.utc_now(), "elements": [],
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def new_state(job_id: str, world_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA, "version": VERSION,
        "job_id": str(job_id), "saved_at": manifests.utc_now(),
        "world": world_identity,
        "elements": {},
        # The named seam for `pickup`: a player-owned container is a second state
        # axis, and reserving the key now means adding the verb later is one
        # effect, not a schema change.
        "player": {},
    }


def validate_state(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise InteractionsError("state document must be an object")
    if document.get("schema") != STATE_SCHEMA:
        raise InteractionsError(
            f"schema must be {STATE_SCHEMA!r}, got {document.get('schema')!r}")
    if document.get("version") != VERSION:
        raise InteractionsError(
            f"version must be {VERSION}, got {document.get('version')!r}")

    elements = document.get("elements")
    if not isinstance(elements, dict):
        raise InteractionsError("state elements must be an object keyed by slug")
    normalised: dict[str, str] = {}
    for slug, value in elements.items():
        if not isinstance(value, str):
            raise InteractionsError(f"{slug}: saved state must be a string")
        normalised[str(slug)] = value

    player = document.get("player")
    if player is not None and not isinstance(player, dict):
        raise InteractionsError("player must be an object")

    return {
        "schema": STATE_SCHEMA, "version": VERSION,
        "job_id": str(document.get("job_id") or ""),
        "saved_at": str(document.get("saved_at") or manifests.utc_now()),
        "world": document.get("world"),
        "elements": normalised,
        "player": dict(player or {}),
    }


# ---------------------------------------------------------------------------
# Resolution — the heart of this module
# ---------------------------------------------------------------------------

def validate_player_poses(raw: Any) -> dict[str, Any]:
    """Validate the physics half of the player seam: prop poses.

    The seam itself is schema-free by design (`player: {}` — reserved
    2026-07-27), but the ROUTE that writes it must not be: an unbounded or
    non-finite pose store would feed NaNs straight into every future physics
    step and grow without limit. Position is three finite floats, quaternion
    four finite floats of ~unit length, slug count bounded."""
    if not isinstance(raw, dict):
        raise InteractionsError("poses must be an object keyed by slug")
    if len(raw) > MAX_PLAYER_POSES:
        raise InteractionsError(
            f"poses holds {len(raw)} entries; the cap is {MAX_PLAYER_POSES}")
    out: dict[str, Any] = {}
    for slug, pose in raw.items():
        name = str(slug)
        if not name or len(name) > 64 or any(ord(c) < 32 for c in name):
            raise InteractionsError(f"invalid pose slug {name!r}")
        if not isinstance(pose, dict):
            raise InteractionsError(f"{name}: pose must be an object")
        position = pose.get("position")
        quaternion = pose.get("quaternion")
        if (not isinstance(position, list) or len(position) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in position)):
            raise InteractionsError(f"{name}: position must be three finite numbers")
        if (not isinstance(quaternion, list) or len(quaternion) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in quaternion)):
            raise InteractionsError(f"{name}: quaternion must be four finite numbers")
        norm = sum(float(v) ** 2 for v in quaternion)
        if abs(norm - 1.0) > 0.05:
            raise InteractionsError(f"{name}: quaternion is not unit length")
        out[name] = {
            "position": [float(v) for v in position],
            "quaternion": [float(v) for v in quaternion],
        }
    return out


def resolve_state(interactions: dict[str, Any] | None,
                  state: dict[str, Any] | None,
                  world_manifest: dict[str, Any] | None = None,
                  world_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a saved state against what the world currently is.

    Per-element resolution is the real gate, not the world-level stamp. A
    rebuild rewrites `world_manifest.json` with a fresh `seconds` timestamp
    through a non-atomic `write_text`, so any file-identity comparison reports a
    change on EVERY rebuild whether or not anything a save depends on moved.
    Treating that as invalidation would mean you can never rebuild a world
    without losing your save — a hostile default for a sandbox where the worst
    stale entry is a door that is shut again.

    So the stamp is advisory (`world_rebuilt`) and each entry is resolved on its
    own merits, counted honestly, the same way `_world_collision` reports
    `missing_hulls` rather than serving a dead reference.

    Never writes. The saved document is left exactly as authored until the next
    explicit save, which keeps this a pure function.
    """
    records = {r["slug"]: r for r in ((interactions or {}).get("elements") or [])}
    saved = dict((state or {}).get("elements") or {})
    present = world_slugs(world_manifest) if world_manifest is not None else None

    applied: dict[str, str] = {r["slug"]: r["initial"] for r in records.values()}
    dropped: list[dict[str, str]] = []

    for slug, value in saved.items():
        record = records.get(slug)
        if record is None:
            dropped.append({"slug": slug, "saved": value,
                            "reason": "no interaction is authored for this element"})
            continue
        if present is not None and slug not in present:
            dropped.append({"slug": slug, "saved": value,
                            "reason": "element is not in the rebuilt world"})
            applied.pop(slug, None)
            continue
        if value not in record["states"]:
            dropped.append({"slug": slug, "saved": value,
                            "reason": f"state {value!r} is no longer authored"})
            continue
        applied[slug] = value

    # An authored element the world lost cannot be offered at all.
    if present is not None:
        for slug in [s for s in applied if s not in present]:
            applied.pop(slug, None)

    stamped = (state or {}).get("world")
    world_rebuilt = bool(
        stamped and world_identity
        and stamped.get("sha256") != world_identity.get("sha256"))

    return {
        "applied": applied,
        "dropped": dropped,
        "world_rebuilt": world_rebuilt,
        "player": dict((state or {}).get("player") or {}),
    }


def next_state(record: dict[str, Any], current: str | None) -> str:
    """The state one interaction advances to. Stable on a single-state record."""
    states = record["states"]
    if current not in states:
        return record["initial"]
    return states[(states.index(current) + 1) % len(states)]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def interactions_path(world_dir: Path) -> Path:
    return Path(world_dir) / INTERACTIONS_NAME


def state_path(world_dir: Path) -> Path:
    return Path(world_dir) / STATE_NAME


def read_interactions(world_dir: Path) -> dict[str, Any] | None:
    raw = manifests.read_json(interactions_path(world_dir))
    return validate_interactions(raw) if raw else None


def write_interactions(world_dir: Path, document: dict[str, Any],
                       known_slugs: set[str] | None = None) -> dict[str, Any]:
    normalised = validate_interactions(document, known_slugs)
    normalised["authored_at"] = manifests.utc_now()
    manifests.atomic_write_json(interactions_path(world_dir), normalised)
    return normalised


def read_state(world_dir: Path, job_id: str) -> dict[str, Any] | None:
    """Read the save. A save belonging to another job is refused, not applied.

    Job duplication copies `_world/` wholesale (`_DUP_SKIP_DIRS` is only
    `{"versions"}`), so without this check a duplicated scene would silently
    inherit its parent's save.
    """
    raw = manifests.read_json(state_path(world_dir))
    if not raw:
        return None
    document = validate_state(raw)
    if document["job_id"] and document["job_id"] != str(job_id):
        raise InteractionsError(
            f"this save belongs to job {document['job_id']!r}, not {job_id!r} — "
            f"it was most likely copied by a job duplication; delete it to start fresh")
    return document


def write_state(world_dir: Path, document: dict[str, Any]) -> dict[str, Any]:
    normalised = validate_state(document)
    normalised["saved_at"] = manifests.utc_now()
    manifests.atomic_write_json(state_path(world_dir), normalised)
    return normalised
