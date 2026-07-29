"""The scenario layer: a captured world becomes a playable one — as DATA.

The world stays the world (geometry, physics, interactions untouched); a
scenario is a validated document beside interactions.json describing what
PLAYS there: waves of actors, their stats, spawn policy, win/lose. The same
discipline as every other world document — closed vocabulary, bounded
numbers, fail-loud validation — because a scenario is gameplay-load-bearing:
a NaN zombie speed or a thousand-actor wave is a broken evening, not a
tuning choice.

v1 vocabulary is deliberately one verb deep: one actor type ("zombie"),
melee contact damage, hitscan-killable, waves that grow. The schema leaves
room (actors is a list; `type` is an enum) without pretending breadth we
have not built.
"""

from __future__ import annotations

import math
from typing import Any

SCENARIO_SCHEMA = "dev.splatlab.world-scenario/v1"
VERSION = 1

ACTOR_TYPES = ("zombie",)
MAX_WAVES = 20
MAX_ACTORS_PER_WAVE = 24
MAX_HEALTH = 1000.0
MAX_SPEED_MPS = 12.0
MAX_DAMAGE = 200.0
MAX_REACH_M = 3.0


class ScenarioError(ValueError):
    """A scenario document violates its contract."""


def default_scenario(job_id: str) -> dict[str, Any]:
    """Three growing zombie waves — the roadmap's thin-slice demo, as data."""
    return {
        "schema": SCENARIO_SCHEMA,
        "version": VERSION,
        "job_id": job_id,
        "name": "Zombie waves",
        "player": {"health": 100.0, "weapon": {
            "damage": 34.0, "range_m": 30.0, "cooldown_s": 0.35}},
        "waves": [
            {"actors": [{"type": "zombie", "count": 3, "health": 100.0,
                         "speed_mps": 1.2, "damage": 10.0, "reach_m": 0.9,
                         "attack_cooldown_s": 1.0}]},
            {"actors": [{"type": "zombie", "count": 5, "health": 120.0,
                         "speed_mps": 1.5, "damage": 12.0, "reach_m": 0.9,
                         "attack_cooldown_s": 0.9}]},
            {"actors": [{"type": "zombie", "count": 7, "health": 140.0,
                         "speed_mps": 1.8, "damage": 15.0, "reach_m": 0.9,
                         "attack_cooldown_s": 0.8}]},
        ],
        "spawn": {"policy": "far-walkable", "min_distance_m": 4.0},
        "rest_between_waves_s": 5.0,
    }


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioError(f"{name} must be a number")
    v = float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ScenarioError(f"{name} must be within [{low}, {high}]")
    return v


def validate_scenario(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ScenarioError("scenario must be an object")
    if document.get("schema") != SCENARIO_SCHEMA:
        raise ScenarioError(f"schema must be {SCENARIO_SCHEMA!r}")
    if document.get("version") != VERSION:
        raise ScenarioError(f"version must be {VERSION}")
    name = str(document.get("name") or "Scenario")[:60]

    player = document.get("player")
    if not isinstance(player, dict):
        raise ScenarioError("player must be an object")
    weapon = player.get("weapon")
    if not isinstance(weapon, dict):
        raise ScenarioError("player.weapon must be an object")
    out_player = {
        "health": _number(player.get("health"), "player.health", 1, MAX_HEALTH),
        "weapon": {
            "damage": _number(weapon.get("damage"), "weapon.damage", 1, MAX_DAMAGE),
            "range_m": _number(weapon.get("range_m"), "weapon.range_m", 1, 200),
            "cooldown_s": _number(weapon.get("cooldown_s"), "weapon.cooldown_s",
                                  0.05, 10),
        },
    }

    waves = document.get("waves")
    if not isinstance(waves, list) or not 1 <= len(waves) <= MAX_WAVES:
        raise ScenarioError(f"waves must be a list of 1..{MAX_WAVES}")
    out_waves = []
    for w_index, wave in enumerate(waves):
        if not isinstance(wave, dict) or not isinstance(wave.get("actors"), list):
            raise ScenarioError(f"waves[{w_index}] must have an actors list")
        out_actors = []
        total = 0
        for a_index, actor in enumerate(wave["actors"]):
            where = f"waves[{w_index}].actors[{a_index}]"
            if not isinstance(actor, dict):
                raise ScenarioError(f"{where} must be an object")
            if actor.get("type") not in ACTOR_TYPES:
                raise ScenarioError(
                    f"{where}.type must be one of {ACTOR_TYPES}")
            count = int(_number(actor.get("count"), f"{where}.count", 1,
                                MAX_ACTORS_PER_WAVE))
            total += count
            out_actors.append({
                "type": actor["type"],
                "count": count,
                "health": _number(actor.get("health"), f"{where}.health", 1, MAX_HEALTH),
                "speed_mps": _number(actor.get("speed_mps"), f"{where}.speed_mps",
                                     0.1, MAX_SPEED_MPS),
                "damage": _number(actor.get("damage"), f"{where}.damage", 0, MAX_DAMAGE),
                "reach_m": _number(actor.get("reach_m"), f"{where}.reach_m",
                                   0.1, MAX_REACH_M),
                "attack_cooldown_s": _number(actor.get("attack_cooldown_s"),
                                             f"{where}.attack_cooldown_s", 0.1, 30),
            })
        if total > MAX_ACTORS_PER_WAVE:
            raise ScenarioError(
                f"waves[{w_index}] spawns {total} actors; the cap is "
                f"{MAX_ACTORS_PER_WAVE}")
        out_waves.append({"actors": out_actors})

    spawn = document.get("spawn") or {}
    if not isinstance(spawn, dict) or spawn.get("policy") != "far-walkable":
        raise ScenarioError("spawn.policy must be 'far-walkable' (v1)")

    return {
        "schema": SCENARIO_SCHEMA,
        "version": VERSION,
        "job_id": str(document.get("job_id") or ""),
        "name": name,
        "player": out_player,
        "waves": out_waves,
        "spawn": {"policy": "far-walkable",
                  "min_distance_m": _number(spawn.get("min_distance_m", 4.0),
                                            "spawn.min_distance_m", 1, 100)},
        "rest_between_waves_s": _number(
            document.get("rest_between_waves_s", 5.0),
            "rest_between_waves_s", 0, 120),
    }
