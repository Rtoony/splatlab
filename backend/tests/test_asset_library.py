"""The committed starter asset library keeps its contract forever: every GLB
validates, is ONE mesh under identity transforms, sits on the floor at its
origin, and stays small. The generator self-validates at build time; this
guards the committed artifacts against drift."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import glb_check  # noqa: E402
import world_placed as wp  # noqa: E402

LIBRARY = BACKEND.parent / "assets" / "library"


def test_library_exists_with_the_starters() -> None:
    names = sorted(p.name for p in LIBRARY.glob("*.glb"))
    assert {"torch-sconce.glb", "treasure-chest.glb",
            "wooden-signpost.glb"} <= set(names)


@pytest.mark.parametrize("path", sorted(LIBRARY.glob("*.glb")),
                         ids=lambda p: p.stem)
def test_every_library_glb_keeps_the_contract(path: Path) -> None:
    assert path.stat().st_size < 1024 * 1024, "library assets stay small"
    # Library names are placement slugs by construction.
    assert wp.SLUG_RE.match(path.stem), path.stem

    summary = glb_check.validate_glb(path)
    assert summary["meshes"] == 1, "ONE joined mesh per asset"

    bounds = glb_check.position_bounds(path)
    assert bounds["identity_transforms"] is True
    # Y is up in the exported GLB; origin is bottom-centre.
    assert abs(bounds["aabb"]["min"][1]) <= 0.005
    assert 0.1 <= bounds["extent"][1] <= 3.0, "human-scale metres"
