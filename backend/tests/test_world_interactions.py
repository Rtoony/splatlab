"""Authored affordances and player state for the walkable world.

The resolver is the whole design: it decides what a save still means after the
world it was saved against has been rebuilt. Most of this file is about that,
because the failure modes are silent ones — a save that resurrects an element
the world no longer has, or a save discarded wholesale because one prop moved.

Pure module, no FastAPI, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import world_interactions as wi  # noqa: E402


def _record(slug="lamp", verb="toggle", states=("off", "on"), **extra):
    record = {"slug": slug, "verb": verb, "states": list(states)}
    record.update(extra)
    return record


def _doc(*records, job_id="splat_abc"):
    return {"schema": wi.INTERACTIONS_SCHEMA, "version": wi.VERSION,
            "job_id": job_id, "elements": list(records)}


def _world(*slugs, shell=True, unbuilt=()):
    elements = [{"slug": s, "role": "prop"} for s in slugs]
    elements += [{"slug": s, "role": "unbuilt"} for s in unbuilt]
    manifest = {"v": 1, "elements": elements}
    if shell:
        manifest["shell"] = {"slug": "shell", "role": "static"}
    return manifest


# ---------------------------------------------------------------------------
# What the world contains
# ---------------------------------------------------------------------------

def test_world_slugs_includes_the_shell_you_stand_in():
    assert wi.world_slugs(_world("lamp", "chair")) == {"lamp", "chair", "shell"}


def test_world_slugs_excludes_elements_that_failed_to_build():
    """There is no geometry to aim at, so it cannot be an interactable."""
    assert wi.world_slugs(_world("lamp", unbuilt=("chair",))) == {"lamp", "shell"}


def test_world_slugs_of_nothing_is_empty():
    assert wi.world_slugs(None) == set()
    assert wi.world_slugs({}) == set()


# ---------------------------------------------------------------------------
# Interaction validation
# ---------------------------------------------------------------------------

def test_a_minimal_record_defaults_sensibly():
    doc = wi.validate_interactions(_doc(_record()))
    record = doc["elements"][0]

    assert record["verb"] == "toggle"
    assert record["states"] == ["off", "on"]
    assert record["initial"] == "off", "first state is the default initial"
    assert record["prompt"] == "lamp", "prompt falls back to the slug"
    assert record["range_m"] == 2.0
    assert record["effects"] == {}


def test_open_is_not_a_verb_it_is_a_toggle_with_labels():
    """Shipping `open` separately would buy a synonym and a second code path."""
    with pytest.raises(wi.InteractionsError, match="closed/open"):
        wi.validate_interactions(_doc(_record(verb="open", states=("closed", "open"))))

    doc = wi.validate_interactions(
        _doc(_record(slug="cupboard", verb="toggle", states=("closed", "open"))))
    assert doc["elements"][0]["states"] == ["closed", "open"]


@pytest.mark.parametrize("verb", ["inspect", "toggle", "pickup"])
def test_the_whole_vocabulary_validates(verb):
    doc = wi.validate_interactions(_doc(_record(verb=verb, states=("a", "b"))))
    assert doc["elements"][0]["verb"] == verb


def test_an_unknown_verb_fails_loud():
    with pytest.raises(wi.InteractionsError, match="unknown verb"):
        wi.validate_interactions(_doc(_record(verb="detonate")))


def test_initial_must_be_one_of_the_states():
    with pytest.raises(wi.InteractionsError, match="initial"):
        wi.validate_interactions(_doc(_record(initial="sideways")))


@pytest.mark.parametrize("states", [[], ["only"], ["a", "a"], ["Bad Caps"], ["x"] * 9])
def test_bad_state_lists_are_refused(states):
    if states == ["only"]:
        pytest.skip("a single state is legal; it just never advances")
    with pytest.raises(wi.InteractionsError):
        wi.validate_interactions(_doc(_record(states=states)))


def test_a_single_state_record_is_legal_and_never_advances():
    doc = wi.validate_interactions(_doc(_record(states=("seen",))))
    record = doc["elements"][0]
    assert wi.next_state(record, "seen") == "seen"


@pytest.mark.parametrize("bad", [0, -1, "far", float("nan"), float("inf"), 1e9])
def test_bad_range_is_refused(bad):
    with pytest.raises(wi.InteractionsError, match="range_m"):
        wi.validate_interactions(_doc(_record(range_m=bad)))


def test_an_unknown_effect_key_is_rejected():
    """An effect that silently does nothing is the exact failure this
    codebase's fail-loud discipline exists to prevent."""
    with pytest.raises(wi.InteractionsError, match="unknown effect"):
        wi.validate_interactions(_doc(_record(effects={"on": {"explode": True}})))


def test_an_effect_for_an_undeclared_state_is_rejected():
    with pytest.raises(wi.InteractionsError, match="not in"):
        wi.validate_interactions(_doc(_record(effects={"ajar": {"tint": None}})))


@pytest.mark.parametrize("tint", ["red", "#fff", "#gggggg", 123])
def test_a_bad_tint_is_rejected(tint):
    with pytest.raises(wi.InteractionsError, match="tint"):
        wi.validate_interactions(_doc(_record(effects={"on": {"tint": tint}})))


def test_a_good_tint_and_null_both_pass():
    doc = wi.validate_interactions(
        _doc(_record(effects={"on": {"tint": "#ffd27f"}, "off": {"tint": None}})))
    assert doc["elements"][0]["effects"]["on"]["tint"] == "#ffd27f"
    assert doc["elements"][0]["effects"]["off"]["tint"] is None


def test_visible_must_be_a_bool():
    with pytest.raises(wi.InteractionsError, match="visible"):
        wi.validate_interactions(_doc(_record(effects={"on": {"visible": "yes"}})))


def test_one_interaction_per_element_in_v1():
    with pytest.raises(wi.InteractionsError, match="appears twice"):
        wi.validate_interactions(_doc(_record(), _record()))


@pytest.mark.parametrize("slug", ["../etc", "a/b", ".", "..", ".hidden", ""])
def test_unsafe_slugs_are_refused(slug):
    with pytest.raises(wi.InteractionsError):
        wi.validate_interactions(_doc(_record(slug=slug)))


def test_an_affordance_for_an_element_not_in_the_world_is_refused():
    """Authoring an interaction for something absent would only ever surface as
    a mystery in the walker."""
    with pytest.raises(wi.InteractionsError, match="no element 'ghost'"):
        wi.validate_interactions(_doc(_record(slug="ghost")),
                                 known_slugs=wi.world_slugs(_world("lamp")))


def test_the_shell_can_carry_an_affordance():
    doc = wi.validate_interactions(_doc(_record(slug="shell", verb="inspect",
                                                states=("unseen", "seen"))),
                                   known_slugs=wi.world_slugs(_world("lamp")))
    assert doc["elements"][0]["slug"] == "shell"


@pytest.mark.parametrize("bad", [
    {"schema": "nope", "version": 1, "elements": []},
    {"schema": wi.INTERACTIONS_SCHEMA, "version": 99, "elements": []},
    {"schema": wi.INTERACTIONS_SCHEMA, "version": 1, "elements": "not-a-list"},
    "not a dict",
])
def test_malformed_documents_are_refused(bad):
    with pytest.raises(wi.InteractionsError):
        wi.validate_interactions(bad)


# ---------------------------------------------------------------------------
# next_state
# ---------------------------------------------------------------------------

def test_next_state_cycles():
    record = wi.validate_interactions(_doc(_record()))["elements"][0]
    assert wi.next_state(record, "off") == "on"
    assert wi.next_state(record, "on") == "off"


def test_next_state_recovers_from_an_unknown_current():
    record = wi.validate_interactions(_doc(_record()))["elements"][0]
    assert wi.next_state(record, "gibberish") == "off"
    assert wi.next_state(record, None) == "off"


def test_next_state_wraps_a_three_state_record():
    record = wi.validate_interactions(
        _doc(_record(states=("low", "mid", "high"))))["elements"][0]
    assert wi.next_state(record, "high") == "low"


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------

def _interactions(*records):
    return wi.validate_interactions(_doc(*records))


def test_an_empty_save_resolves_to_the_authored_initials():
    resolved = wi.resolve_state(_interactions(_record()), None, _world("lamp"))
    assert resolved["applied"] == {"lamp": "off"}
    assert resolved["dropped"] == []


def test_a_saved_state_overrides_the_initial():
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"lamp": "on"}}
    resolved = wi.resolve_state(_interactions(_record()), state, _world("lamp"))
    assert resolved["applied"] == {"lamp": "on"}


def test_an_element_the_rebuilt_world_lost_is_dropped_and_counted():
    state = {"schema": wi.STATE_SCHEMA, "version": 1,
             "elements": {"lamp": "on", "chair": "on"}}
    interactions = _interactions(_record(), _record(slug="chair"))

    resolved = wi.resolve_state(interactions, state, _world("lamp"))

    assert resolved["applied"] == {"lamp": "on"}, "the survivor still applies"
    assert [d["slug"] for d in resolved["dropped"]] == ["chair"]
    assert "not in the rebuilt world" in resolved["dropped"][0]["reason"]
    assert resolved["dropped"][0]["saved"] == "on"


def test_one_lost_element_does_not_discard_the_other_nineteen():
    slugs = [f"prop-{i}" for i in range(20)]
    interactions = _interactions(*[_record(slug=s) for s in slugs])
    state = {"schema": wi.STATE_SCHEMA, "version": 1,
             "elements": {s: "on" for s in slugs}}

    resolved = wi.resolve_state(interactions, state, _world(*slugs[:19]))

    assert len(resolved["applied"]) == 19
    assert len(resolved["dropped"]) == 1


def test_a_state_the_author_removed_is_dropped_and_counted():
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"lamp": "dimmed"}}
    resolved = wi.resolve_state(_interactions(_record()), state, _world("lamp"))

    assert resolved["applied"] == {"lamp": "off"}, "falls back to the initial"
    assert "no longer authored" in resolved["dropped"][0]["reason"]


def test_a_save_for_an_element_with_no_authored_interaction_is_dropped():
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"chair": "on"}}
    resolved = wi.resolve_state(_interactions(_record()), state, _world("lamp", "chair"))

    assert "chair" not in resolved["applied"]
    assert "no interaction is authored" in resolved["dropped"][0]["reason"]


def test_world_rebuilt_is_advisory_and_drops_nothing_by_itself():
    """A rebuild rewrites world_manifest.json with a fresh timestamp every time,
    so treating identity change as invalidation would mean never being able to
    rebuild without losing the save."""
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"lamp": "on"},
             "world": {"sha256": "old"}}

    resolved = wi.resolve_state(_interactions(_record()), state, _world("lamp"),
                                world_identity={"sha256": "new"})

    assert resolved["world_rebuilt"] is True
    assert resolved["applied"] == {"lamp": "on"}, "still applied"
    assert resolved["dropped"] == []


def test_an_unchanged_world_is_not_reported_as_rebuilt():
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {},
             "world": {"sha256": "same"}}
    resolved = wi.resolve_state(_interactions(_record()), state, _world("lamp"),
                                world_identity={"sha256": "same"})
    assert resolved["world_rebuilt"] is False


def test_resolution_without_a_world_manifest_skips_the_presence_check():
    """The local dev-static path has no manifest to check against."""
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"lamp": "on"}}
    resolved = wi.resolve_state(_interactions(_record()), state, None)
    assert resolved["applied"] == {"lamp": "on"}


def test_the_resolver_never_mutates_its_inputs():
    """It is a pure function; the saved document stays as authored until the
    next explicit write."""
    interactions = _interactions(_record())
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"lamp": "on"}}
    before = json.dumps(state, sort_keys=True), json.dumps(interactions, sort_keys=True)

    wi.resolve_state(interactions, state, _world("lamp"))

    assert (json.dumps(state, sort_keys=True),
            json.dumps(interactions, sort_keys=True)) == before


def test_player_state_survives_resolution():
    """The named seam for pickup — reserved from day one."""
    state = {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {},
             "player": {"carrying": ["red-bicycle"]}}
    resolved = wi.resolve_state(_interactions(_record()), state, _world("lamp"))
    assert resolved["player"] == {"carrying": ["red-bicycle"]}


# ---------------------------------------------------------------------------
# I/O and the duplication hazard
# ---------------------------------------------------------------------------

def test_interactions_round_trip(tmp_path):
    written = wi.write_interactions(tmp_path, _doc(_record()),
                                    known_slugs=wi.world_slugs(_world("lamp")))
    assert written["authored_at"]
    assert wi.read_interactions(tmp_path)["elements"][0]["slug"] == "lamp"


def test_an_invalid_document_never_reaches_disk(tmp_path):
    with pytest.raises(wi.InteractionsError):
        wi.write_interactions(tmp_path, _doc(_record(verb="detonate")))
    assert not wi.interactions_path(tmp_path).is_file()


def test_reading_an_absent_sidecar_is_none_not_an_error(tmp_path):
    assert wi.read_interactions(tmp_path) is None
    assert wi.read_state(tmp_path, "splat_abc") is None


def test_state_round_trip(tmp_path):
    document = wi.new_state("splat_abc", {"sha256": "abc"})
    document["elements"]["lamp"] = "on"
    wi.write_state(tmp_path, document)

    assert wi.read_state(tmp_path, "splat_abc")["elements"] == {"lamp": "on"}


def test_a_save_copied_by_job_duplication_is_refused_not_applied(tmp_path):
    """Duplicate copies _world/ wholesale — _DUP_SKIP_DIRS is only {"versions"}
    — so without this a duplicated scene inherits its parent's save."""
    wi.write_state(tmp_path, wi.new_state("splat_parent"))

    with pytest.raises(wi.InteractionsError, match="belongs to job"):
        wi.read_state(tmp_path, "splat_child")


def test_a_save_with_no_job_id_is_tolerated(tmp_path):
    document = wi.new_state("")
    document["elements"]["lamp"] = "on"
    wi.write_state(tmp_path, document)
    assert wi.read_state(tmp_path, "splat_abc")["elements"] == {"lamp": "on"}


@pytest.mark.parametrize("bad", [
    {"schema": "nope", "version": 1, "elements": {}},
    {"schema": wi.STATE_SCHEMA, "version": 9, "elements": {}},
    {"schema": wi.STATE_SCHEMA, "version": 1, "elements": []},
    {"schema": wi.STATE_SCHEMA, "version": 1, "elements": {"lamp": 3}},
])
def test_malformed_state_is_refused(bad):
    with pytest.raises(wi.InteractionsError):
        wi.validate_state(bad)
