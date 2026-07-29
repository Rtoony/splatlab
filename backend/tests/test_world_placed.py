"""The authored-asset registry: validator (closed vocab, bounded), the pure
manifest merge, read strictness, and the world_collision preservation
contract — a rebuild re-applies placed entries instead of erasing them.

world_collision imports trimesh at module top but its static/authored paths
never touch it (and convex_hulls bails before trimesh without CoACD), so
the rebuild tests run in the default app suite with the stub, the same way
test_generated_texture.py does.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "mesh"))

for _name in ("trimesh",):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

import world_collision  # noqa: E402
import world_placed as wp  # noqa: E402


def _entry(slug: str = "torch-sconce", role: str = "static", **over) -> dict:
    base = {
        "slug": slug, "label": "torch sconce", "role": role,
        "extent": [0.2, 0.45, 0.2],
        "aabb": {"min": [1.0, 0.0, -2.0], "max": [1.2, 0.45, -1.8]},
        "placed_at": "2026-07-28T00:00:00Z",
    }
    base.update(over)
    return base


# ── validator ───────────────────────────────────────────────────────────────────


def test_validator_normalises_and_derives_glb() -> None:
    doc = wp.validate_placed({"job_id": "splat_ab12cd", "elements": [_entry()]})
    entry = doc["elements"][0]
    assert entry["glb"] == "torch-sconce.glb"  # derived, never trusted
    assert entry["provenance"] == "authored"
    assert doc["schema"] == wp.PLACED_SCHEMA


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e.update(slug="UPPER"), "bad placed slug"),
        (lambda e: e.update(slug="shell"), "not a placeable"),
        (lambda e: e.update(role="vehicle"), "role must be one of"),
        (lambda e: e.update(extent=[1, 2]), "3 finite numbers"),
        (lambda e: e.update(aabb={"min": [2, 2, 2], "max": [0, 0, 0]}),
         "min exceeds max"),
        (lambda e: e.update(surprise=True), "unknown entry keys"),
        (lambda e: e.update(label="   "), "label"),
        (lambda e: e.update(uploaded_bytes=-1), "non-negative"),
    ],
)
def test_validator_rejects_malformed_entries(mutate, message) -> None:
    entry = _entry()
    mutate(entry)
    with pytest.raises(wp.PlacedError, match=message):
        wp.validate_placed({"elements": [entry]})


def test_validator_rejects_duplicates_and_floods() -> None:
    with pytest.raises(wp.PlacedError, match="duplicate"):
        wp.validate_placed({"elements": [_entry(), _entry()]})
    flood = [_entry(slug=f"prop-{i:03d}") for i in range(wp.MAX_PLACED + 1)]
    with pytest.raises(wp.PlacedError, match="too many"):
        wp.validate_placed({"elements": flood})


def test_read_strict_raises_and_lax_degrades(tmp_path: Path) -> None:
    (tmp_path / wp.PLACED_FILENAME).write_text("{not json")
    with pytest.raises(wp.PlacedError, match="unreadable"):
        wp.read_placed(tmp_path, "splat_ab12cd")
    lax = wp.read_placed(tmp_path, "splat_ab12cd", strict=False)
    assert lax["elements"] == []
    # An absent file is a fresh registry either way.
    assert wp.read_placed(tmp_path / "nope", "splat_ab12cd")["elements"] == []


# ── the pure merge ──────────────────────────────────────────────────────────────


def _manifest() -> dict:
    return {
        "v": 1, "job_id": "splat_ab12cd",
        "elements": [{"slug": "crate", "role": "prop", "glb": "crate.glb"}],
        "shell": {"slug": "shell", "role": "static"},
        "counts": {"props": 1, "static": 0, "unbuilt": 0},
    }


def test_apply_placed_appends_authored_and_recounts() -> None:
    placed = wp.validate_placed({"elements": [_entry()]})
    merged, conflicts = wp.apply_placed_to_manifest(_manifest(), placed)
    assert conflicts == []
    authored = [e for e in merged["elements"] if e.get("provenance") == "authored"]
    assert [e["slug"] for e in authored] == ["torch-sconce"]
    assert authored[0]["collision"]["strategy"] == "complex_as_simple"
    assert merged["counts"]["authored"] == 1
    # Prop role gets the honest fallback block instead.
    prop = wp.manifest_element(
        wp.validate_placed({"elements": [_entry(role="prop")]})["elements"][0])
    assert prop["collision"]["strategy"] == "render_mesh_fallback"


def test_apply_placed_is_idempotent_and_pure() -> None:
    manifest = _manifest()
    snapshot = json.loads(json.dumps(manifest))
    placed = wp.validate_placed({"elements": [_entry()]})
    once, _ = wp.apply_placed_to_manifest(manifest, placed)
    twice, _ = wp.apply_placed_to_manifest(once, placed)
    assert twice["elements"] == once["elements"]  # strip-then-append
    assert manifest == snapshot  # inputs never mutated


def test_apply_placed_captured_slug_wins_and_reports() -> None:
    placed = wp.validate_placed({"elements": [_entry(slug="crate")]})
    merged, conflicts = wp.apply_placed_to_manifest(_manifest(), placed)
    assert conflicts == ["crate"]
    assert merged["authored_conflicts"] == ["crate"]
    crates = [e for e in merged["elements"] if e["slug"] == "crate"]
    assert len(crates) == 1 and crates[0].get("provenance") != "authored"
    assert merged["counts"]["authored"] == 0


# ── the preservation contract (world_collision re-run) ──────────────────────────


def _mk_world_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "splat_ab12cd"
    world = job_dir / "_world"
    (world / "elements").mkdir(parents=True)
    (world / "world.json").write_text(json.dumps({
        "v": 1, "job_id": "splat_ab12cd", "units": "meters",
        "meters_per_unit": 1.0,
        "shell": {"built": True, "glb": "shell.glb", "extent": [8, 3, 8],
                  "faces": 10},
        "elements": [{"slug": "crate", "built": True, "glb": "crate.glb",
                      "extent": [1.0, 1.0, 1.0], "faces": 10}],
    }))
    return job_dir


def _run_collision(monkeypatch: pytest.MonkeyPatch, job_dir: Path) -> dict:
    monkeypatch.setattr(sys, "argv", ["world_collision.py", str(job_dir)])
    assert world_collision.main() == 0
    manifest_path = job_dir / "_world" / "world_manifest.json"
    # Atomic write: no staged residue may survive a successful run.
    assert not list((job_dir / "_world").glob(".building-*"))
    return json.loads(manifest_path.read_text())


def test_collision_rerun_preserves_authored_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _mk_world_job(tmp_path)
    world = job_dir / "_world"
    wp.write_placed(world, {"job_id": "splat_ab12cd",
                            "elements": [_entry()]})
    (world / "elements" / "torch-sconce.glb").write_bytes(b"authored-glb")

    manifest = _run_collision(monkeypatch, job_dir)
    torch = next(e for e in manifest["elements"]
                 if e["slug"] == "torch-sconce")
    assert torch["provenance"] == "authored"
    assert torch["role"] == "static"
    assert torch["collision"]["strategy"] == "complex_as_simple"
    assert manifest["counts"]["authored"] == 1
    assert manifest["authored_conflicts"] == []

    # THE contract: a re-run derives from world.json again and the authored
    # entry is still there, exactly once.
    again = _run_collision(monkeypatch, job_dir)
    torches = [e for e in again["elements"] if e["slug"] == "torch-sconce"]
    assert len(torches) == 1
    assert again["counts"]["authored"] == 1


def test_collision_missing_placed_glb_marks_unbuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _mk_world_job(tmp_path)
    wp.write_placed(job_dir / "_world",
                    {"job_id": "splat_ab12cd", "elements": [_entry()]})
    manifest = _run_collision(monkeypatch, job_dir)
    torch = next(e for e in manifest["elements"]
                 if e["slug"] == "torch-sconce")
    assert torch["role"] == "unbuilt"
    assert torch["reason"] == "placed GLB missing"
    assert manifest["counts"]["unbuilt"] == 1


def test_collision_captured_slug_wins_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _mk_world_job(tmp_path)
    world = job_dir / "_world"
    wp.write_placed(world, {"job_id": "splat_ab12cd",
                            "elements": [_entry(slug="crate")]})
    (world / "elements" / "crate.glb").write_bytes(b"captured-glb")
    manifest = _run_collision(monkeypatch, job_dir)
    crates = [e for e in manifest["elements"] if e["slug"] == "crate"]
    assert len(crates) == 1
    assert crates[0].get("provenance") != "authored"
    assert manifest["authored_conflicts"] == ["crate"]
    assert manifest["counts"]["authored"] == 0


def test_collision_corrupt_registry_degrades_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    job_dir = _mk_world_job(tmp_path)
    (job_dir / "_world" / wp.PLACED_FILENAME).write_text("{broken")
    manifest = _run_collision(monkeypatch, job_dir)
    assert manifest["counts"]["authored"] == 0
    assert "WARNING" in capsys.readouterr().err
