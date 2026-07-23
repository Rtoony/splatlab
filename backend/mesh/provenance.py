"""Provenance rails for the scene-regeneration (P6) lane.

Doctrine (approved 2026-07-22 plan): regenerated/generative content is
plausible-not-faithful — render/VR lane ONLY, never a survey/geo artifact.
Enforcement is mechanical, not narrative, and must survive files leaving the
job tree, so it is BOTH path-based (quarantine dirs / artifact names) and
in-file (PLY header comment, glTF extras).

Pure stdlib on purpose: this module is imported by the FastAPI app (splatlab
venv), the dn-splatter-probe worker scripts, and standalone gates — none of
which share third-party deps. The PLY header is parsed manually.
"""
from __future__ import annotations

from pathlib import Path

# In-file tag written into every generative PLY (header comment) and every
# generative glTF node/asset (extras[GLTF_EXTRAS_KEY]). Greppable, versionless.
GENERATIVE_TAG = "SplatLab-provenance: generative render-vr-only"

# Element provenance values allowed in scene_manifest.json.
PROVENANCE_VALUES = ("captured", "proxy", "ground-derived")

GLTF_EXTRAS_KEY = "splatlab_provenance"

# Read at most this much when scanning a PLY header (headers are tiny; a missing
# end_header on a huge binary file must not stall the guard).
_HEADER_CAP = 65536


class GenerativeInputRefused(RuntimeError):
    """A survey/geo lane was handed generative content."""


def ply_header_comments(path: str | Path) -> list[str]:
    """Header comments of a PLY file (binary or ascii). [] if not a PLY."""
    p = Path(path)
    try:
        with open(p, "rb") as fh:
            head = fh.read(_HEADER_CAP)
    except OSError:
        return []
    if not head.startswith(b"ply"):
        return []
    end = head.find(b"end_header")
    if end == -1:
        end = len(head)
    comments = []
    for line in head[:end].splitlines():
        if line.startswith(b"comment "):
            comments.append(line[len(b"comment "):].decode("utf-8", errors="replace").strip())
    return comments


def ply_is_generative(path: str | Path) -> bool:
    return any(GENERATIVE_TAG in c for c in ply_header_comments(path))


def path_is_generative(path: str | Path) -> bool:
    """Quarantine-path rule: anything under a _regen/ dir, or a proxy artifact
    inside an _objects/ tree, is generative regardless of file content."""
    p = Path(path)
    parts = p.parts
    if "_regen" in parts:
        return True
    if "_objects" in parts and p.name.startswith("proxy"):
        return True
    return False


def assert_not_generative(path: str | Path, lane: str) -> None:
    """Fail-loud guard for survey/geo lanes. Checks path AND file content, so
    the refusal survives copies out of the quarantine dir."""
    if path_is_generative(path):
        raise GenerativeInputRefused(
            f"REFUSED: {path} is generative content (quarantine path) — "
            f"the {lane} lane never consumes generative geometry")
    if ply_is_generative(path):
        raise GenerativeInputRefused(
            f"REFUSED: {path} carries the '{GENERATIVE_TAG}' tag — "
            f"the {lane} lane never consumes generative geometry")
