"""Building-agnostic matching of capture rectangles and planes to a building.json model:
openings by wall plane + IoU with ghost suppression, same-wall position shifts, wall-line
offsets, parallel-plane projections and the footprint test."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture import model_match as mm  # noqa: E402

BUILDING = {"levels": [
    {"id": "L1", "elevation": 0.0, "height": 3.0, "outline": [[0, 0], [6, 0], [6, 10], [0, 10]],
     "walls": [{"id": "street", "a": [0, 0], "b": [6, 0], "exterior": True, "openings": [{"id": "door", "kind": "garage", "offset": 0.5, "width": 4.0, "sill": 0.0, "height": 2.1},
                                                                                       {"id": "win", "kind": "window", "offset": 5.0, "width": 0.9, "sill": 1.0, "height": 1.5}]},
               {"id": "bay", "a": [2, -0.8], "b": [5, -0.8], "exterior": True, "openings": []},
               {"id": "inner", "a": [0, 3], "b": [6, 3], "exterior": False, "openings": [{"id": "closet", "kind": "door", "offset": 1, "width": 0.8, "sill": 0, "height": 2.0}]}]}]}


def rect(x0, x1, y, z0, z1):
    return np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], dtype=float)


def test_model_openings_and_exterior_filter():
    ops = mm.model_openings(BUILDING)
    assert [o["id"] for o in ops] == ["door", "win"] and ops[0]["head_z"] == 2.1 and ops[1]["sill_z"] == 1.0
    assert len(mm.model_openings(BUILDING, exterior_only=False)) == 3
    assert [w["id"] for w in mm.model_walls(BUILDING)] == ["street", "bay"]


def test_match_openings_with_ghost_suppression_and_shift():
    cap = [{"kind": "garage_door", "corners_m": rect(0.6, 4.4, 0.1, -0.1, 2.0)},          # the door, slightly off
           {"kind": "garage_door", "corners_m": rect(0.9, 4.7, 0.9, 0.0, 2.1)},           # ghost on a plane 0.9 m behind: lower IoU
           {"kind": "window", "corners_m": rect(1.0, 1.9, 0.0, 1.0, 2.5)},                # a window 4 m from the modelled one
           {"kind": "door", "corners_m": rect(1.0, 1.8, 3.05, 0.0, 2.0)}]                  # on the interior wall: exterior-only -> unmatched
    ops = mm.model_openings(BUILDING)
    pairs = mm.match_openings(cap, ops)
    assert pairs[0][1] == 0 and pairs[0][2] > 0.8 and pairs[1][1] == -1                    # best keeps the match, the ghost is dropped
    assert pairs[2][1] is None and pairs[3][1] is None
    m, du = mm.same_wall_shift(cap[2], ops)
    assert m["id"] == "win" and abs(du - (1.45 - 5.45)) < 1e-9                              # 4 m back along a->b


def test_wall_lines_parallel_offsets_and_footprint():
    walls = [{"name": "w0", "corners_m": rect(0, 6, 0.05, 0, 3), "normal_canonical": [0, -1, 0]},
             {"name": "w1", "corners_m": rect(2, 5, -0.9, 0, 3), "normal_canonical": [0, -1, 0]},
             {"name": "w2", "corners_m": rect(0, 6, 5.0, 0, 3), "normal_canonical": [0, -1, 0]}]        # far from every model line
    mw = mm.model_walls(BUILDING)
    lines = mm.wall_line_matches(walls, mw)
    assert {(x["patch"], x["wall"]) for x in lines} == {("w0", "street"), ("w1", "bay")}
    off = {x["patch"]: x["offset_m"] for x in lines}
    assert abs(off["w0"] - (-0.05)) < 1e-6 and abs(off["w1"] - 0.1) < 1e-6                 # + = outside the model line (toward the capture)
    proj = mm.parallel_offsets(walls, lines, mw)
    assert len(proj) == 1 and abs(proj[0]["capture_m"] - 0.95) < 1e-6 and abs(proj[0]["model_m"] - 0.8) < 1e-6 and abs(proj[0]["delta_m"] - 0.15) < 1e-6
    ghost = {"name": "w1b", "corners_m": rect(2, 5, -1.3, 0, 3), "normal_canonical": [0, -1, 0], "surfels": 10}   # a second, smaller patch on the bay line
    walls2 = [{**walls[0], "surfels": 100}, {**walls[1], "surfels": 50}, ghost]
    lines2 = mm.wall_line_matches(walls2, mw)
    assert {x["patch"] for x in lines2} == {"w0", "w1", "w1b"} and len(mm.parallel_offsets(walls2, lines2, mw)) == 1          # one patch per model wall
    dims = [{"id": "bay-projection", "expected": 0.8}]
    assert mm.parallel_offsets(walls2, lines2, mw, dimensions=dims)[0]["dimension_id"] == "bay-projection"
    assert mm.inside_footprint([3, -0.3], BUILDING) and not mm.inside_footprint([9, 3], BUILDING)
