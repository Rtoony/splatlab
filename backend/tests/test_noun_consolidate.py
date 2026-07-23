"""P6b noun consolidation: the stuff/things split and cleaning logic are pure
stdlib (no torch/transformers import at module load) so they're directly unit
testable; siglip_dedupe itself needs the langfield-spike env and is proven by
the Step 0 spike + a live P6b run, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))
import noun_consolidate as nc  # noqa: E402
import slugify  # noqa: E402


def test_module_has_no_module_level_torch_binding():
    # torch/transformers are imported INSIDE siglip_dedupe(), not at module
    # scope, so importing this module never drags in the heavy ML deps this
    # app's own venv intentionally lacks (doctrine: nothing added to
    # splatlab's venv; heavy deps live only in the langfield-spike conda env).
    assert "torch" not in vars(nc)
    assert "transformers" not in vars(nc)


def test_classify_stuff_matches_ground_and_vegetation():
    assert nc.classify_stuff("grass lawn")
    assert nc.classify_stuff("Green Hedge")
    assert nc.classify_stuff("gravel driveway")
    assert nc.classify_stuff("dirt path")


def test_classify_stuff_rejects_real_objects():
    assert not nc.classify_stuff("round wooden table")
    assert not nc.classify_stuff("fire hydrant")
    assert not nc.classify_stuff("flower vase")


def test_clean_nouns_dedupes_case_insensitive_and_trims():
    out = nc.clean_nouns(["Table", "table ", " TABLE", "Vase", ""], max_nouns=10)
    assert out == ["Table", "Vase"]


def test_clean_nouns_drops_overlong_and_caps():
    out = nc.clean_nouns(["a" * 61, "table", "vase", "chair"], max_nouns=2)
    assert out == ["table", "vase"]


def test_consolidate_single_noun_skips_dedupe_model(monkeypatch):
    # siglip_dedupe short-circuits for len(nouns)<=1 with no import — this is
    # the one consolidate() path this app's venv (no torch) can exercise
    # directly; siglip_dedupe's actual model path is proven by a live P6b run.
    report = nc.consolidate(["fire hydrant", "fire hydrant"], max_nouns=10, dedup_thresh=0.85)
    assert report["things"] == ["fire hydrant"]
    assert report["stuff"] == []
    assert report["cleaned_count"] == 1


def test_consolidate_splits_stuff_from_things(monkeypatch):
    monkeypatch.setattr(nc, "siglip_dedupe", lambda nouns, thresh: (nouns, {}))
    report = nc.consolidate(
        ["fire hydrant", "grass lawn", "fire hydrant"], max_nouns=10, dedup_thresh=0.85,
    )
    assert report["things"] == ["fire hydrant"]
    assert report["stuff"] == ["grass lawn"]
    assert report["cleaned_count"] == 2


def test_slugify_matches_the_two_implementations_it_replaced():
    # Review finding 2026-07-23: scene_sam3_masks.py and instance_lift.py
    # each had their own character-for-character-identical _slug() before
    # this shared module replaced both — pin the exact algorithm (per-char
    # substitution, NOT a collapsing regex) so on-disk slugs never shift.
    assert slugify.slug("fire hydrant") == "fire-hydrant"
    assert slugify.slug("Fire  Hydrant") == "fire--hydrant"  # two spaces -> two dashes
    assert slugify.slug("!!!") == "thing"
    assert slugify.slug("") == "thing"
    assert slugify.slug("a" * 50) == "a" * 40


def test_dedupe_slugs_drops_collisions_first_kept_wins():
    # The exact collision the review found: punctuation-only differences
    # reduce to the same slug ("fire-hydrant") and would otherwise let
    # scene_sam3_masks.py/instance_lift.py silently clobber each other's
    # per-instance files across a noun boundary.
    kept, collisions = nc.dedupe_slugs(["Fire Hydrant", "Fire-Hydrant", "Flower Vase"])
    assert kept == ["Fire Hydrant", "Flower Vase"]
    assert collisions == [{"noun": "Fire-Hydrant", "slug": "fire-hydrant", "collides_with": "Fire Hydrant"}]


def test_dedupe_slugs_no_collision_is_a_no_op():
    kept, collisions = nc.dedupe_slugs(["fire hydrant", "flower vase"])
    assert kept == ["fire hydrant", "flower vase"]
    assert collisions == []


def test_consolidate_wires_slug_dedupe_into_things(monkeypatch):
    monkeypatch.setattr(nc, "siglip_dedupe", lambda nouns, thresh: (nouns, {}))
    report = nc.consolidate(
        ["Fire Hydrant", "Fire-Hydrant"], max_nouns=10, dedup_thresh=0.85,
    )
    assert report["things"] == ["Fire Hydrant"]
    assert len(report["slug_collisions"]) == 1
