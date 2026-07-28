"""The authoring + state routes for walkable-world interactions.

The case that matters most is the last one: save a state, rebuild the world with
one element gone, and confirm the survivor still applies while the loss is
reported rather than silently resurrected or silently wiped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_route  # noqa: E402
import world_interactions as wi  # noqa: E402
import world_interactions_route  # noqa: E402

JOB = "splat_1a2b3c"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(world_interactions_route.router, prefix="/api/splat")
    return TestClient(app), outputs


def _mk_world(outputs: Path, *slugs, graded=True) -> Path:
    world = outputs / JOB / "_world"
    world.mkdir(parents=True)
    (outputs / JOB / "meta.json").write_text(json.dumps({"job_id": JOB}))
    (world / "world.json").write_text(json.dumps({"v": 1, "job_id": JOB}))
    if graded:
        (world / "world_manifest.json").write_text(json.dumps({
            "v": 1, "job_id": JOB,
            "elements": [{"slug": s, "role": "prop"} for s in slugs],
            "shell": {"slug": "shell", "role": "static"},
        }))
    return world


def _lamp(slug="lamp", **extra):
    record = {"slug": slug, "verb": "toggle", "prompt": "the lamp",
              "states": ["off", "on"], "initial": "off",
              "effects": {"on": {"tint": "#ffd27f"}, "off": {"tint": None}}}
    record.update(extra)
    return record


def _author(http, *records):
    return http.put(f"/api/splat/jobs/{JOB}/world/interactions",
                    json={"elements": list(records)})


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def test_unknown_job_404s(client):
    http, _ = client
    assert http.get(f"/api/splat/jobs/{JOB}/world/interactions").status_code == 404


def test_a_scene_with_no_world_409s_with_the_remedy(client):
    http, outputs = client
    (outputs / JOB).mkdir(parents=True)
    (outputs / JOB / "meta.json").write_text("{}")

    r = http.get(f"/api/splat/jobs/{JOB}/world/interactions")

    assert r.status_code == 409
    assert "solidify" in r.json()["detail"]


def test_a_world_with_nothing_authored_returns_empty_not_an_error(client):
    http, outputs = client
    _mk_world(outputs, "lamp")

    body = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()

    assert body["interactions"] is None
    assert body["state"] is None
    assert body["resolved"]["applied"] == {}
    assert set(body["known_slugs"]) == {"lamp", "shell"}


# ---------------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------------

def test_authoring_round_trips(client):
    http, outputs = client
    _mk_world(outputs, "lamp")

    assert _author(http, _lamp()).status_code == 200
    body = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()

    assert body["interactions"]["elements"][0]["slug"] == "lamp"
    assert body["resolved"]["applied"] == {"lamp": "off"}, "starts at the initial"


def test_put_replaces_rather_than_merging(client):
    http, outputs = client
    _mk_world(outputs, "lamp", "chair")
    _author(http, _lamp(), _lamp(slug="chair"))

    _author(http, _lamp())

    body = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()
    assert [e["slug"] for e in body["interactions"]["elements"]] == ["lamp"]


def test_authoring_for_an_element_not_in_the_world_400s_with_the_known_slugs(client):
    http, outputs = client
    _mk_world(outputs, "lamp")

    r = _author(http, _lamp(slug="ghost"))

    assert r.status_code == 400
    assert "no element 'ghost'" in r.json()["detail"]
    assert "lamp" in r.json()["detail"], "tells you what you could have used"


def test_the_validator_message_reaches_the_client(client):
    http, outputs = client
    _mk_world(outputs, "lamp")

    r = _author(http, _lamp(effects={"on": {"explode": True}}))

    assert r.status_code == 400
    assert "unknown effect" in r.json()["detail"]


def test_an_unsafe_slug_is_refused(client):
    http, outputs = client
    _mk_world(outputs, "lamp")
    assert _author(http, _lamp(slug="../../etc/passwd")).status_code == 400


def test_an_ungraded_world_skips_the_presence_check(client):
    """A world solidified but never collision-graded has no world_manifest.json;
    authoring must still work rather than refusing everything."""
    http, outputs = client
    _mk_world(outputs, graded=False)

    assert _author(http, _lamp()).status_code == 200


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_setting_a_state_persists_across_a_fresh_read(client):
    """The end-to-end persistence proof."""
    http, outputs = client
    _mk_world(outputs, "lamp")
    _author(http, _lamp())

    r = http.post(f"/api/splat/jobs/{JOB}/world/state",
                  json={"slug": "lamp", "state": "on"})
    assert r.status_code == 200
    assert r.json()["resolved"]["applied"] == {"lamp": "on"}

    body = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()
    assert body["resolved"]["applied"] == {"lamp": "on"}
    assert body["state"]["elements"] == {"lamp": "on"}


def test_state_for_an_unauthored_element_404s(client):
    http, outputs = client
    _mk_world(outputs, "lamp", "chair")
    _author(http, _lamp())

    r = http.post(f"/api/splat/jobs/{JOB}/world/state",
                  json={"slug": "chair", "state": "on"})

    assert r.status_code == 404
    assert "no interaction is authored" in r.json()["detail"]


def test_an_undeclared_state_400s_and_lists_the_legal_ones(client):
    http, outputs = client
    _mk_world(outputs, "lamp")
    _author(http, _lamp())

    r = http.post(f"/api/splat/jobs/{JOB}/world/state",
                  json={"slug": "lamp", "state": "melted"})

    assert r.status_code == 400
    assert "['off', 'on']" in r.json()["detail"]


def test_delete_resets_to_the_authored_initials(client):
    http, outputs = client
    world = _mk_world(outputs, "lamp")
    _author(http, _lamp())
    http.post(f"/api/splat/jobs/{JOB}/world/state", json={"slug": "lamp", "state": "on"})

    r = http.delete(f"/api/splat/jobs/{JOB}/world/state")

    assert r.status_code == 200
    assert r.json()["resolved"]["applied"] == {"lamp": "off"}
    assert not wi.state_path(world).is_file()


def test_deleting_when_there_is_no_save_is_not_an_error(client):
    http, outputs = client
    _mk_world(outputs, "lamp")
    assert http.delete(f"/api/splat/jobs/{JOB}/world/state").status_code == 200


# ---------------------------------------------------------------------------
# The rebuild case
# ---------------------------------------------------------------------------

def test_a_rebuilt_world_keeps_the_survivor_and_reports_the_loss(client):
    http, outputs = client
    world = _mk_world(outputs, "lamp", "chair")
    _author(http, _lamp(), _lamp(slug="chair"))
    http.post(f"/api/splat/jobs/{JOB}/world/state", json={"slug": "lamp", "state": "on"})
    http.post(f"/api/splat/jobs/{JOB}/world/state", json={"slug": "chair", "state": "on"})

    # Rebuild: the chair did not survive this time.
    (world / "world_manifest.json").write_text(json.dumps({
        "v": 1, "job_id": JOB,
        "elements": [{"slug": "lamp", "role": "prop"}],
        "shell": {"slug": "shell", "role": "static"},
    }))

    resolved = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()["resolved"]

    assert resolved["applied"] == {"lamp": "on"}, "the survivor keeps its state"
    assert [d["slug"] for d in resolved["dropped"]] == ["chair"]
    assert "not in the rebuilt world" in resolved["dropped"][0]["reason"]


def test_a_rebuild_alone_does_not_invalidate_a_save(client):
    """world_manifest.json is rewritten with a fresh timestamp on every rebuild,
    so identity change must be advisory — otherwise rebuilding always costs the
    save."""
    http, outputs = client
    world = _mk_world(outputs, "lamp")
    _author(http, _lamp())
    http.post(f"/api/splat/jobs/{JOB}/world/state", json={"slug": "lamp", "state": "on"})

    manifest = json.loads((world / "world_manifest.json").read_text())
    manifest["seconds"] = 12.3            # what a real rebuild changes
    (world / "world_manifest.json").write_text(json.dumps(manifest))

    resolved = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()["resolved"]

    assert resolved["applied"] == {"lamp": "on"}
    assert resolved["dropped"] == []


def test_a_save_copied_in_by_job_duplication_is_reported_not_applied(client):
    """Duplicate copies _world/ wholesale, so this really happens."""
    http, outputs = client
    world = _mk_world(outputs, "lamp")
    _author(http, _lamp())
    foreign = wi.new_state("splat_999999")
    foreign["elements"]["lamp"] = "on"
    wi.write_state(world, foreign)

    body = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()

    assert body["state"] is None
    assert "belongs to job" in body["state_error"]
    assert body["resolved"]["applied"] == {"lamp": "off"}, "authored initial, not the copy"


# ---------------------------------------------------------------------------
# Player seam: physics poses (P1)
# ---------------------------------------------------------------------------

def _pose(x=1.0, y=2.0, z=3.0):
    return {"position": [x, y, z], "quaternion": [0.0, 0.0, 0.0, 1.0]}


def test_player_poses_round_trip_and_survive_element_state(client):
    http, outputs = client
    _mk_world(outputs, "lamp", "crate")
    _author(http, _lamp())

    saved = http.post(f"/api/splat/jobs/{JOB}/world/player",
                      json={"poses": {"crate": _pose(4.5, -1.25, 0.5)}})
    assert saved.status_code == 200, saved.text
    resolved = saved.json()["resolved"]
    assert resolved["player"]["physics"]["poses"]["crate"]["position"] == [4.5, -1.25, 0.5]

    # An element-state write must not clobber the player seam...
    assert http.post(f"/api/splat/jobs/{JOB}/world/state",
                     json={"slug": "lamp", "state": "on"}).status_code == 200
    payload = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()
    assert payload["resolved"]["player"]["physics"]["poses"]["crate"]["position"] == [4.5, -1.25, 0.5]
    assert payload["resolved"]["applied"]["lamp"] == "on"

    # ...and a new player write replaces only the physics half.
    assert http.post(f"/api/splat/jobs/{JOB}/world/player",
                     json={"poses": {}}).status_code == 200
    payload = http.get(f"/api/splat/jobs/{JOB}/world/interactions").json()
    assert payload["resolved"]["player"]["physics"]["poses"] == {}
    assert payload["resolved"]["applied"]["lamp"] == "on"


def test_player_poses_are_validated(client):
    http, outputs = client
    _mk_world(outputs, "crate")
    bad = [
        {"poses": {"crate": {"position": [1, 2], "quaternion": [0, 0, 0, 1]}}},
        {"poses": {"crate": {"position": [1, 2, float("nan")] if False else [1, 2, "x"],
                             "quaternion": [0, 0, 0, 1]}}},
        {"poses": {"crate": {"position": [1, 2, 3], "quaternion": [0, 0, 0, 2]}}},
        {"poses": {"crate": {"position": [1, 2, 3], "quaternion": [0, 0, 0, True]}}},
        {"poses": {"crate": "not-a-pose"}},
        {"poses": {"": _pose()}},
    ]
    for payload in bad:
        response = http.post(f"/api/splat/jobs/{JOB}/world/player", json=payload)
        assert response.status_code == 400, payload


def test_player_pose_count_is_bounded(client, monkeypatch):
    http, outputs = client
    _mk_world(outputs, "crate")
    monkeypatch.setattr(wi, "MAX_PLAYER_POSES", 2)
    over = {f"prop-{i}": _pose() for i in range(3)}
    response = http.post(f"/api/splat/jobs/{JOB}/world/player", json={"poses": over})
    assert response.status_code == 400
    assert "cap" in response.json()["detail"]
