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
    # The library holds hand props AND environment architecture (walls,
    # crypts, stair runs) since the CC0 pack intake — the band is the intake
    # script's own sanity contract, not the old hand-prop 3 m ceiling.
    assert 0.05 <= max(bounds["extent"]) <= 30.0, "plausible metres"


def test_catalog_entries_reference_real_assets_with_licenses() -> None:
    """Every catalog entry points at a committed GLB and carries its license
    — the bookkeeping the flat directory itself cannot hold."""
    import json
    catalog_path = LIBRARY / "catalog.json"
    assert catalog_path.is_file(), "pack intake ships a catalog"
    catalog = json.loads(catalog_path.read_text())
    assert catalog["schema"] == "dev.splatlab.asset-catalog/v1"
    stems = {p.stem for p in LIBRARY.glob("*.glb")}
    for name, entry in catalog["assets"].items():
        assert name in stems, f"catalog names a missing asset: {name}"
        assert entry["license"], name
        assert entry["license_url"].startswith("http"), name
        assert entry["source_url"].startswith("http"), name
        assert entry["pack"], name


def test_list_asset_library_attaches_catalog_and_reports_corruption(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dcc import blender_workflow as bw
    lib = tmp_path / "library"
    lib.mkdir()
    # A real committed asset keeps the listing meaningful.
    (lib / "dungeon-wall.glb").write_bytes(
        (LIBRARY / "dungeon-wall.glb").read_bytes())
    monkeypatch.setattr(bw, "ASSET_LIBRARY_ROOT", lib)

    import json
    (lib / "catalog.json").write_text(json.dumps({
        "schema": "dev.splatlab.asset-catalog/v1",
        "assets": {"dungeon-wall": {"pack": "kaykit-dungeon-remastered",
                                    "license": "CC0-1.0",
                                    "license_url": "https://example",
                                    "source_url": "https://example",
                                    "imported_at": "2026-07-30"}}}))
    listing = bw.list_asset_library()
    entry = next(e for e in listing if e["name"] == "dungeon-wall")
    assert entry["catalog"]["license"] == "CC0-1.0"

    (lib / "catalog.json").write_text("{corrupt")
    listing = bw.list_asset_library()
    entry = next(e for e in listing if e["name"] == "dungeon-wall")
    assert "unreadable" in entry["catalog_error"]
