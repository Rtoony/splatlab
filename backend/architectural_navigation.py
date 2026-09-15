"""Bounded, versioned local navigation probes for authored room connections."""

import math

import numpy as np

from reconstruction_evidence import EvidenceError

LEGACY_METHOD = "parallel-center-ray/v1"
ENTRY_METHOD = "shared-entry-footprint/v1"
LEGACY_RECIPE = {"floor_clearance_m": .015, "capsule_radius_m": .22, "capsule_height_m": 1.7, "floor_tolerance_m": .08}


def recipe():
    return {**LEGACY_RECIPE, "navigation_method": ENTRY_METHOD}


def method(value):
    if value == LEGACY_RECIPE:
        return LEGACY_METHOD
    if value == recipe():
        return ENTRY_METHOD
    raise EvidenceError("Architectural navigation recipe is unknown or changes its fixed body/clearance contract")


def local_routes(spec, value):
    selected = method(value)
    lateral = spec["opening_width"] / 2 - value["capsule_radius_m"] - .06
    spacing = .02 / math.hypot(1., lateral / .6) if selected == ENTRY_METHOD else .04
    count = math.ceil((spec["cut_depth"] + spec["room_depth"] + .1) / spacing) + 1
    near = -spec["cut_depth"] / 2
    along = np.linspace(near - .6, spec["cut_depth"] / 2 + spec["room_depth"] - .5, count)
    factor = np.clip((along - (near - .3)) / .6, 0., 1.) if selected == ENTRY_METHOD else np.ones(count)
    points = np.array([[offset * factor[index], 0., position] for offset in (-lateral, 0., lateral)
                       for index, position in enumerate(along)])
    return points, count


def floor_support(points, sample_floor, radius):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not 0 < len(points) <= 6000:
        raise EvidenceError("Footprint probes require a bounded finite world-point array")
    if type(radius) not in (int, float) or not np.isfinite(radius) or not .15 <= radius <= .5:
        raise EvidenceError("Footprint probes require an explicit finite body radius")
    extent = radius + .03
    offsets = [[0., 0., 0.]]
    for fraction in (.25, .5, .75, 1.):
        for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
            offsets.append([extent * fraction * np.cos(angle), 0., extent * fraction * np.sin(angle)])
    offsets = np.asarray(offsets)
    samples = np.asarray(sample_floor((points[:, None, :] + offsets).reshape(-1, 3)), dtype=float)
    if samples.shape != (len(points) * len(offsets),):
        raise EvidenceError("Floor sampler returned an invalid footprint array")
    samples = samples.reshape(len(points), len(offsets))
    lift = np.sqrt(np.maximum(0., extent ** 2 - np.sum(offsets[:, [0, 2]] ** 2, axis=1))) - extent
    supported = np.where(np.isfinite(samples), samples + lift, -np.inf).max(axis=1)
    center = samples[:, 0]
    usable = np.isfinite(center) & (supported - center <= .08)
    return center, np.where(usable, supported, np.nan)


def verify_document(spec, value, frame, document, identifier, gates):
    selected = method(value)
    expected_version = 2 if selected == ENTRY_METHOD else 1
    if (not isinstance(document, dict) or document.get("v") != expected_version
            or document.get("navigation_method", LEGACY_METHOD) != selected
            or document.get("architecture_id") != identifier or document.get("frame") != "world-y-up-metres"
            or document.get("gates") != gates or any(type(passed) is not bool for passed in document.get("gates", {}).values())
            or document.get("capsule_radius_m") != value["capsule_radius_m"]
            or document.get("capsule_height_m") != value["capsule_height_m"]):
        raise EvidenceError("Retained navigation does not match the prepared method, body, frame or gates")
    expected_local, count = local_routes(spec, value)
    try:
        points = np.asarray(document["route"])
        floor = np.asarray(document["floor_y"])
        center = np.asarray(document["center_floor_y"] if selected == ENTRY_METHOD else document["floor_y"])
    except (KeyError, ValueError, TypeError) as exc:
        raise EvidenceError("Retained navigation has malformed route/floor arrays") from exc
    if (any(array.dtype.kind not in "iuf" for array in (points, floor, center))
            or points.shape != expected_local.shape or floor.shape != (len(points),) or center.shape != floor.shape
            or not np.isfinite(points).all() or not np.isfinite(floor).all() or not np.isfinite(center).all()):
        raise EvidenceError("Retained navigation has missing or nonfinite route/floor samples")
    local = (points - frame["origin"]) @ frame["rotation"]
    if np.max(np.abs(local - expected_local)) > 1e-6:
        raise EvidenceError("Retained navigation route differs from its dimensioned approach and interior lanes")
    if (np.max(np.abs(floor - frame["origin"][1])) > .08 or np.max(np.abs(center - frame["origin"][1])) > .08
            or np.max(np.abs(np.diff(floor.reshape(3, count), axis=1))) > .04
            or np.max(np.abs(np.diff(center.reshape(3, count), axis=1))) > .04
            or np.any(floor < center - 1e-6) or np.any(floor - center > .08)):
        raise EvidenceError("Retained navigation floor or footprint adjustment exceeds the unchanged local tolerances")
