"""Restyle: a cosmetic diff, validated as suspiciously as an affordance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import splat_route  # noqa: E402
import world_interactions_route  # noqa: E402
import world_restyle as wr  # noqa: E402

JOB = "splat_ab12cd34"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", outputs)
    app = FastAPI()
    app.include_router(world_interactions_route.router, prefix="/api/splat")
    world = outputs / JOB / "_world"
    world.mkdir(parents=True)
    (world / "world.json").write_text("{}")
    (world / "world_manifest.json").write_text(json.dumps({
        "v": 1, "job_id": JOB,
        "elements": [{"slug": "wall", "label": "wall", "role": "static"},
                     {"slug": "bonsai", "label": "bonsai", "role": "prop"}],
        "shell": {"slug": "shell", "role": "static"},
    }))
    return TestClient(app), world


def test_empty_world_restyles_to_as_captured(client):
    http, _world = client
    r = http.get(f"/api/splat/jobs/{JOB}/world/restyle")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["restyle"]["elements"] == {}
    assert body["restyle"]["lighting"]["preset"] == "as-captured"
    # The catalogue the author picks from comes from the SAME taxonomy the
    # ground/object bakes use, so a preview and a re-bake cannot disagree.
    # Fantasy classes (v2) ride the same file, so they need no extra wiring.
    assert {c["id"] for c in body["materials"]} >= {
        "grass", "concrete", "water", "cobblestone", "obsidian", "lava"}
    assert "dungeon" in body["presets"]


def test_put_stores_tints_materials_and_lighting(client):
    http, world = client
    r = http.put(f"/api/splat/jobs/{JOB}/world/restyle", json={
        "elements": {"wall": {"tint": "#8899FF"},
                     "bonsai": {"material": "vegetation", "material_scale": 0.5}},
        "lighting": {"preset": "dusk", "intensity": 0.8},
    })
    assert r.status_code == 200, r.text
    stored = json.loads((world / "restyle.json").read_text())
    assert stored["elements"]["wall"]["tint"] == "#8899ff"  # normalised
    assert stored["elements"]["bonsai"]["material"] == "vegetation"
    assert stored["lighting"] == {"preset": "dusk", "intensity": 0.8}
    # Nothing under _world/ was rewritten — the capture is untouched.
    assert json.loads((world / "world_manifest.json").read_text())["elements"]


def test_unknown_slug_class_preset_and_bad_tint_are_all_refused(client):
    http, _world = client
    for payload, expect in [
        ({"elements": {"ghost": {"tint": "#ffffff"}}}, "no such element"),
        ({"elements": {"wall": {"material": "vaporwave-chrome"}}},
         "unknown material class"),
        ({"elements": {"wall": {"tint": "blue"}}}, "#rrggbb"),
        ({"elements": {"wall": {}}}, "at least one of"),
        ({"lighting": {"preset": "vaporwave"}}, "unknown lighting preset"),
        ({"lighting": {"preset": "noon", "intensity": 99}}, "outside"),
        ({"elements": {"wall": {"material": "grass", "material_scale": 0}}}, "outside"),
    ]:
        r = http.put(f"/api/splat/jobs/{JOB}/world/restyle", json=payload)
        assert r.status_code == 400, (payload, r.text)
        assert expect in r.json()["detail"], (payload, r.json()["detail"])


def test_delete_restores_the_capture(client):
    http, world = client
    http.put(f"/api/splat/jobs/{JOB}/world/restyle",
             json={"elements": {"wall": {"tint": "#123456"}}})
    assert (world / "restyle.json").is_file()
    r = http.delete(f"/api/splat/jobs/{JOB}/world/restyle")
    assert r.status_code == 200
    assert not (world / "restyle.json").exists()
    assert r.json()["restyle"]["elements"] == {}


def test_a_corrupt_restyle_never_makes_a_world_unopenable(client):
    http, world = client
    (world / "restyle.json").write_text("{not json at all")
    r = http.get(f"/api/splat/jobs/{JOB}/world/restyle")
    assert r.status_code == 200
    assert r.json()["restyle"]["lighting"]["preset"] == "as-captured"


def test_restyle_requires_a_solidified_world(client, tmp_path):
    http, world = client
    (world / "world.json").unlink()
    r = http.get(f"/api/splat/jobs/{JOB}/world/restyle")
    assert r.status_code == 409
    assert "solidify" in r.json()["detail"]


def test_validator_rejects_a_flood_of_elements():
    doc = {"elements": {f"e{i}": {"tint": "#ffffff"}
                        for i in range(wr.MAX_ELEMENTS + 1)}}
    with pytest.raises(wr.RestyleError, match="too many"):
        wr.validate_restyle(doc)
