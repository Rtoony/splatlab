"""Versioned captured-appearance envelopes for paired authored architecture."""

import numpy as np

from reconstruction_evidence import EvidenceError

LEGACY_METHOD = "cut-only/v1"
ROOM_METHOD = "cut-and-authored-room/v1"


def default_method():
    return ROOM_METHOD


def method(clip):
    selected = clip.get("appearance_method", LEGACY_METHOD)
    if selected not in {LEGACY_METHOD, ROOM_METHOD}:
        raise EvidenceError("Unknown architectural appearance method")
    return selected


def room_envelope(spec):
    half_depth, thickness = spec["cut_depth"] / 2, spec["wall_thickness"]
    return (np.array([-spec["room_width"] / 2 - thickness, .005, half_depth - thickness / 2]),
            np.array([spec["room_width"] / 2 + thickness, spec["room_height"] + thickness,
                      half_depth + spec["room_depth"] + thickness / 2]))


def room_interior(spec):
    half_depth, thickness = spec["cut_depth"] / 2, spec["wall_thickness"]
    return (np.array([-spec["room_width"] / 2 + .0001, .0001, half_depth + thickness / 2 + .0001]),
            np.array([spec["room_width"] / 2 - .0001, spec["room_height"] - .0001,
                      half_depth + spec["room_depth"] - thickness / 2 - .0001]))


def contract(spec, coordinate, selected):
    clip = {"world_to_portal": coordinate["world_to_portal"].tolist(),
            "lower": coordinate["lower"].tolist(), "upper": coordinate["upper"].tolist()}
    if selected == ROOM_METHOD:
        lower, upper = room_envelope(spec)
        clip.update(appearance_method=ROOM_METHOD, room_lower=lower.tolist(), room_upper=upper.tolist())
    elif selected != LEGACY_METHOD:
        raise EvidenceError("Unknown architectural appearance method")
    return clip


def regions(clip):
    selected = method(clip)
    bounds = [(np.asarray(clip["lower"]), np.asarray(clip["upper"]))]
    if selected == ROOM_METHOD:
        bounds.append((np.asarray(clip["room_lower"]), np.asarray(clip["room_upper"])))
    return bounds


def _positions(local_positions, document):
    points = np.asarray(local_positions)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or document.get("n_rows") != len(points):
        raise EvidenceError("Protected appearance does not address the finite captured row space")
    return points


def _selection_rows(document, name, count):
    selection = document.get("elements", {}).get(name)
    if selection is None:
        return np.empty(0, dtype=np.int64)
    rows = np.asarray(selection.get("rows", []))
    if rows.ndim != 1 or (rows.size and rows.dtype.kind not in "iu") or np.any(rows < 0) or np.any(rows >= count):
        raise EvidenceError("Protected appearance contains invalid captured row indices")
    return np.unique(rows.astype(np.int64))


def protected_row_conflicts(local_positions, document, protected_names, clip):
    points = _positions(local_positions, document)
    inside = np.zeros(len(points), dtype=bool)
    for lower, upper in regions(clip):
        inside |= ((points >= lower) & (points <= upper)).all(axis=1)
    conflicts = {}
    for name in protected_names:
        rows = _selection_rows(document, name, len(points))
        count = int(inside[rows].sum())
        if count:
            conflicts[name] = count
    return conflicts


def capture_impact(local_positions, document, protected_names, replaced_names, clip):
    points = _positions(local_positions, document)
    masks = [((points >= lower) & (points <= upper)).all(axis=1) for lower, upper in regions(clip)]
    cut = masks[0]
    room = masks[1] if len(masks) > 1 else np.zeros(len(points), dtype=bool)
    union = cut | room
    protected, included = np.zeros(len(points), dtype=bool), np.zeros(len(points), dtype=bool)
    for name in protected_names:
        protected[_selection_rows(document, name, len(points))] = True
    for name in replaced_names:
        included[_selection_rows(document, name, len(points))] = True
    if (protected & included).any():
        raise EvidenceError("Captured impact requires disjoint protected and included rows")
    memberships = {"cut": cut, "room": room, "overlap": cut & room, "union": union,
                   "protected": union & protected, "included": union & included,
                   "unassigned": union & ~(protected | included)}
    rows = {name: np.flatnonzero(mask) for name, mask in memberships.items()}
    summary = {"basis": "captured-gaussian-centers/v1", "total_rows": len(points),
               **{name + "_rows": len(indices) for name, indices in rows.items()}}
    return summary, rows
