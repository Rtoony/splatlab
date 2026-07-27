"""The world lane's numerical cores: static/prop classification, the walkable
shell's probe and ranking, and ground-cell binning.

These decide what the walkable world physically IS, and none of them had a test.
That mattered less while the world was CLI-only; now that POST /world/solidify
exists, a wrong call ships to a browser — and in a walkable world a wrong call is
immediately physical: you walk through the chair, or you cannot pick up the cup.

world_collision imports trimesh and ground_mesh_build imports open3d, neither of
which is in the app's test interpreter. `classify` is pure numpy, so trimesh is
stubbed at import the same way test_object_texture.py does it; the ground binning
was lifted into ground_binning.py, which has no heavy dependency at all.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

# world_collision imports trimesh at module level; `classify` never touches it.
for _name in ("trimesh", "coacd"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

import ground_binning as gb  # noqa: E402
import world_collision as wc  # noqa: E402
import world_shell as ws  # noqa: E402


# ===========================================================================
# world_collision.classify — the static/prop rule
# ===========================================================================

ROOM = (10.0, 8.0, 3.0)          # scene extent in scene units
FRAC = 0.16                       # prop if longest side <= 16% of the scene
METRES = 1.5                      # ...and, when calibrated, <= 1.5 m


def test_a_small_object_is_a_prop():
    kind, reasons = wc.classify((0.5, 0.4, 0.6), ROOM, None, FRAC, METRES)
    assert kind == "prop"
    assert any("prop threshold" in r for r in reasons)


def test_a_large_object_is_static():
    kind, _ = wc.classify((6.0, 0.4, 2.5), ROOM, None, FRAC, METRES)
    assert kind == "static"


def test_the_longest_axis_decides_not_the_volume():
    """A long thin rail is architecture even though it is barely any volume."""
    kind, _ = wc.classify((5.0, 0.05, 0.05), ROOM, None, FRAC, METRES)
    assert kind == "static"


def test_the_relative_rule_works_uncalibrated():
    """A capture with no meters_per_unit still has a meaningful 'how big is
    this compared to the room'."""
    kind, reasons = wc.classify((0.5, 0.5, 0.5), ROOM, None, FRAC, METRES)
    assert kind == "prop"
    assert not any(" m " in r for r in reasons), "no metre claim without a scale"


def test_the_metre_rule_only_demotes_never_promotes():
    """A 2 m object is furniture regardless of how large the room is."""
    huge_room = (100.0, 100.0, 10.0)
    # 3 units is only 3% of this room, so relative size says prop...
    assert wc.classify((3.0, 1.0, 1.0), huge_room, None, FRAC, METRES)[0] == "prop"
    # ...but at 1 m/unit it is 3 m long, which is not a prop.
    kind, reasons = wc.classify((3.0, 1.0, 1.0), huge_room, 1.0, FRAC, METRES)
    assert kind == "static"
    assert any("exceeds the 1.5 m prop limit" in r for r in reasons)


def test_a_scale_cannot_promote_a_large_object_to_a_prop():
    """The metre test is never reached for something already static, so a tiny
    mpu cannot turn a wall into a pickup."""
    kind, _ = wc.classify((6.0, 0.4, 2.5), ROOM, 0.001, FRAC, METRES)
    assert kind == "static"


def test_calibrated_and_within_both_limits_stays_a_prop():
    kind, reasons = wc.classify((0.5, 0.4, 0.6), ROOM, 1.0, FRAC, METRES)
    assert kind == "prop"
    assert any("within the 1.5 m prop limit" in r for r in reasons)


def test_the_boundary_is_inclusive():
    """Exactly at the fraction is still a prop; pinned so it cannot drift."""
    longest = ROOM[0] * FRAC                       # exactly 16%
    assert wc.classify((longest, 0.1, 0.1), ROOM, None, FRAC, METRES)[0] == "prop"
    assert wc.classify((longest * 1.001, 0.1, 0.1), ROOM, None, FRAC, METRES)[0] == "static"


def test_the_metre_boundary_is_inclusive():
    assert wc.classify((1.5, 0.1, 0.1), (100.0, 100.0, 10.0), 1.0, FRAC, METRES)[0] == "prop"
    assert wc.classify((1.51, 0.1, 0.1), (100.0, 100.0, 10.0), 1.0, FRAC, METRES)[0] == "static"


def test_a_degenerate_scene_extent_does_not_divide_by_zero():
    kind, _ = wc.classify((1.0, 1.0, 1.0), (0.0, 0.0, 0.0), None, FRAC, METRES)
    assert kind == "static", "an unknown scene must not silently yield props"


def test_every_decision_carries_its_numbers():
    """The reasons ARE the receipt — a regrade needs to know what decided."""
    _, reasons = wc.classify((3.0, 1.0, 1.0), (100.0, 100.0, 10.0), 1.0, FRAC, METRES)
    joined = " ".join(reasons)
    assert "3.00" in joined and "%" in joined and "m" in joined


# ===========================================================================
# world_shell — frame evidence, probe grid, candidate ranking
# ===========================================================================

def _room_cloud(n=40000, seed=0):
    """A room in the capture frame: dense flat floor at z=0 plus sparse walls.

    Dense on purpose — build_probe requires ~12 points per column at cell=0.1
    before it will call a cell occupied (see the area-scaled threshold test)."""
    rng = np.random.default_rng(seed)
    floor = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n),
                             rng.normal(0.0, 0.005, n)])
    walls = np.column_stack([rng.choice([-2, 2], n // 4), rng.uniform(-2, 2, n // 4),
                             rng.uniform(0, 2.5, n // 4)])
    return np.vstack([floor, walls])


def test_frame_check_detects_the_up_axis_from_the_cloud():
    """The floor is the axis whose histogram has one sharp peak."""
    result = ws.frame_check(_room_cloud())

    assert result["detected_up_axis"] == "z"
    assert result["agrees"] is True
    assert len(result["histogram_peak_fraction"]) == 3


def test_frame_check_reports_disagreement_rather_than_correcting_it():
    """Evidence recorded, not trusted: a Y-up cloud must say so, not be fixed."""
    cloud = _room_cloud()[:, [0, 2, 1]]              # swap so the floor is in y
    result = ws.frame_check(cloud)

    assert result["detected_up_axis"] == "y"
    assert result["agrees"] is False
    assert result["assumed_up_axis"].startswith("z")


def test_to_yup_is_a_rigid_frame_change():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    converted = ws.to_yup(points)

    assert converted.shape == points.shape
    assert np.allclose(np.linalg.norm(converted, axis=1), np.linalg.norm(points, axis=1))
    assert np.isclose(abs(np.linalg.det(ws.YUP_FROM_SCENE)), 1.0)


def test_room_box_uses_percentiles_so_one_flyaway_cannot_blow_up_the_grid(tmp_path):
    """The whole reason for percentile bounds: a single far-field gaussian would
    otherwise stretch the voxel grid across the scene."""
    cloud = _room_cloud()
    tight_lo, tight_hi, source = ws.room_box(tmp_path, cloud, margin=0.1)

    polluted = np.vstack([cloud, [[60.0, 60.0, 60.0]]])
    loose_lo, loose_hi, _ = ws.room_box(tmp_path, polluted, margin=0.1)

    assert "p2-p98" in source, "no mesh.ply present, so the cloud path is used"
    assert np.allclose(tight_hi, loose_hi, atol=0.05)
    assert float(loose_hi.max()) < 10.0


def test_room_box_margin_expands_the_box(tmp_path):
    cloud = _room_cloud()
    lo_a, hi_a, _ = ws.room_box(tmp_path, cloud, margin=0.0)
    lo_b, hi_b, _ = ws.room_box(tmp_path, cloud, margin=0.5)

    assert np.all(lo_b < lo_a) and np.all(hi_b > hi_a)


def test_build_probe_derives_a_footprint_interior_and_seed():
    probe = ws.build_probe(ws.to_yup(_room_cloud()), 0.1,
                           player_radius=0.25, player_height=1.7)

    assert probe.footprint.any(), "a room must have a footprint"
    assert probe.interior.sum() <= probe.footprint.sum(), "interior is eroded"
    assert probe.interior.shape == probe.footprint.shape == probe.shape
    assert np.isfinite(probe.floor_level) and np.isfinite(probe.top_level)
    assert probe.top_level > probe.floor_level
    assert probe.seed.shape == (3,) and np.all(np.isfinite(probe.seed))


def test_the_probe_interior_shrinks_as_the_player_gets_wider():
    yup = ws.to_yup(_room_cloud())

    narrow = ws.build_probe(yup, 0.1, player_radius=0.15, player_height=1.7)
    wide = ws.build_probe(yup, 0.1, player_radius=0.5, player_height=1.7)

    assert wide.interior.sum() < narrow.interior.sum()
    assert narrow.interior.sum() < narrow.footprint.sum()


def test_the_occupancy_threshold_scales_with_cell_area():
    """So the footprint — and every gate measured over it — means the same
    physical region at any --grid-res."""
    cloud = ws.to_yup(_room_cloud())

    coarse = ws.build_probe(cloud, 0.1, player_radius=0.15, player_height=1.7)
    fine = ws.build_probe(cloud, 0.05, player_radius=0.15, player_height=1.7)

    coarse_area = coarse.footprint.sum() * 0.1 ** 2
    fine_area = fine.footprint.sum() * 0.05 ** 2
    assert fine_area == pytest.approx(coarse_area, rel=0.25)


def test_a_cloud_too_sparse_to_be_a_room_yields_almost_no_footprint():
    """Honest by design: you cannot claim a room from a handful of points per
    column, and a thin footprint fails the gates rather than passing quietly."""
    sparse = ws.to_yup(_room_cloud(n=2000))
    probe = ws.build_probe(sparse, 0.1, player_radius=0.15, player_height=1.7)

    assert probe.footprint.sum() <= 4


def test_a_tiny_capture_falls_back_to_the_raw_footprint():
    """Erosion that would erase the room entirely returns the footprint, so the
    probe still has somewhere to measure rather than dividing by nothing."""
    probe = ws.build_probe(ws.to_yup(_room_cloud()), 0.1,
                           player_radius=50.0, player_height=1.7)

    assert probe.interior.any()
    assert probe.interior.sum() == probe.footprint.sum()


# ---------------------------------------------------------------------------
# rank_key — which candidate shell wins
# ---------------------------------------------------------------------------

def _cand(**gates):
    base = {"walkable": False, "gates_passed": 0, "walkable_frac": 0.0,
            "floor_continuity": 0.0, "max_hole_span": 1.0,
            "largest_component_frac": 0.0}
    base.update(gates)
    return {"gates": base}


def test_a_walkable_candidate_beats_a_prettier_unwalkable_one():
    """Walkability is the whole point; nothing else outranks it."""
    walkable = _cand(walkable=True, gates_passed=2, floor_continuity=0.96)
    pretty = _cand(walkable=False, gates_passed=3, floor_continuity=0.99,
                   largest_component_frac=1.0, walkable_frac=0.99)

    assert ws.rank_key(walkable) > ws.rank_key(pretty)


def test_more_gates_passed_wins_among_walkable_candidates():
    assert ws.rank_key(_cand(walkable=True, gates_passed=4)) > \
           ws.rank_key(_cand(walkable=True, gates_passed=3))


def test_a_smaller_hole_span_wins_when_gates_tie():
    """max_hole_span is negated in the key: smaller is better."""
    tight = _cand(walkable=True, gates_passed=4, walkable_frac=0.9,
                  floor_continuity=0.99, max_hole_span=0.1)
    holey = _cand(walkable=True, gates_passed=4, walkable_frac=0.9,
                  floor_continuity=0.99, max_hole_span=0.4)

    assert ws.rank_key(tight) > ws.rank_key(holey)


def test_ranking_survives_a_candidate_with_no_gates():
    """An errored candidate must sort last, not explode."""
    ordered = sorted([_cand(walkable=True, gates_passed=4), {}, _cand()],
                     key=ws.rank_key, reverse=True)
    assert ordered[0]["gates"]["walkable"] is True
    assert ordered[-1] in ({}, _cand())


# ===========================================================================
# ground_binning — the TIN's cells
# ===========================================================================

def _flat_ground(n_side=12, cell=0.1, per_cell=5, z=0.0, seed=0):
    """A grid of cells, each holding `per_cell` points near height z."""
    rng = np.random.default_rng(seed)
    points = []
    for i in range(n_side):
        for j in range(n_side):
            centre = ((i + 0.5) * cell, (j + 0.5) * cell)
            points.extend([[centre[0] + rng.normal(0, cell / 8),
                            centre[1] + rng.normal(0, cell / 8),
                            z + rng.normal(0, 0.001)] for _ in range(per_cell)])
    return np.array(points)


def test_binning_produces_one_cell_per_occupied_square():
    cells, _ = gb.bin_cells(_flat_ground(n_side=5), cell_units=0.1, min_pts_cell=3)
    assert len(cells) == 25


def test_a_sparse_cell_is_dropped():
    """Below min_pts_cell there is not enough evidence for a height."""
    cells, _ = gb.bin_cells(_flat_ground(n_side=4, per_cell=2),
                            cell_units=0.1, min_pts_cell=3)
    assert cells == {}


def test_cell_height_is_the_low_percentile_not_the_mean():
    """A cell holding foliage above the ground must report the GROUND."""
    ground = [[0.05, 0.05, 0.0] for _ in range(8)]
    foliage = [[0.05, 0.05, 2.0] for _ in range(8)]
    cells, _ = gb.bin_cells(np.array(ground + foliage), cell_units=0.1, min_pts_cell=3)

    height = cells[(0, 0)]
    assert height < 0.5, "the 15th percentile must sit in the ground cluster"
    assert height == pytest.approx(0.0, abs=0.01)


def test_class_votes_are_summed_not_argmaxed():
    """Every member gaussian's evidence counts — robust at class boundaries."""
    points = np.array([[0.05, 0.05, 0.0]] * 4)
    relevancy = np.array([[0.4, 0.3], [0.4, 0.3], [0.4, 0.3], [0.0, 0.9]])
    _, votes = gb.bin_cells(points, cell_units=0.1, min_pts_cell=3,
                            class_relevancy=relevancy)

    assert votes[(0, 0)] == pytest.approx([1.2, 1.8])
    assert int(np.argmax(votes[(0, 0)])) == 1, "the summed vote, not the plurality"


def test_a_spike_is_rejected():
    """THE fix: a stray high cell would make Delaunay interpolate a mountain."""
    cells = {(i, j): 0.0 for i in range(5) for j in range(5)}
    cells[(2, 2)] = 3.0

    kept, rejected = gb.reject_spikes(cells, spike_tol_units=0.15)

    assert rejected == 1
    assert (2, 2) not in kept
    assert len(kept) == 24


def test_real_terrain_slope_is_not_mistaken_for_a_spike():
    """A ramp must survive: every cell agrees with its neighbours' median."""
    cells = {(i, j): i * 0.02 for i in range(8) for j in range(8)}
    kept, rejected = gb.reject_spikes(cells, spike_tol_units=0.15)

    assert rejected == 0
    assert len(kept) == 64


def test_an_edge_cell_with_too_few_neighbours_is_kept():
    """Fewer than 3 neighbours is too little context to call something a spike;
    dropping it would erode the boundary of every ground surface."""
    cells = {(0, 0): 0.0, (0, 1): 0.0, (5, 5): 9.0}
    kept, rejected = gb.reject_spikes(cells, spike_tol_units=0.15)

    assert rejected == 0
    assert (5, 5) in kept, "isolated cells are dropped by connectivity, not here"


def test_a_detached_island_is_dropped_by_connectivity():
    main = {(i, j): 0.0 for i in range(6) for j in range(6)}
    island = {(20, 20): 0.0, (20, 21): 0.0, (21, 20): 0.0}

    kept, dropped = gb.largest_component({**main, **island})

    assert dropped == 3
    assert len(kept) == 36
    assert (20, 20) not in kept


def test_connectivity_is_eight_way():
    """Diagonally touching cells are one surface, not two."""
    cells = {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0}
    kept, dropped = gb.largest_component(cells)

    assert dropped == 0 and len(kept) == 3


def test_largest_component_of_nothing_is_nothing():
    assert gb.largest_component({}) == ({}, 0)


def test_cell_centres_land_in_the_middle_of_their_cell():
    centres = gb.cell_centres({(0, 0): 1.5, (3, 4): 2.5}, [(0, 0), (3, 4)], 0.1)

    assert centres[0] == pytest.approx([0.05, 0.05, 1.5])
    assert centres[1] == pytest.approx([0.35, 0.45, 2.5])


def test_a_high_cell_sharing_a_square_with_ground_is_absorbed_not_rejected():
    """The stages compose: the 15th percentile already ignores points above the
    ground in the SAME cell, so spike rejection never has to see them. Only a
    cell that is entirely high is a spike."""
    ground = np.array([[0.05, 0.05, 0.0]] * 5)
    foliage = np.array([[0.05, 0.05, 4.0]] * 20)

    cells, _ = gb.bin_cells(np.vstack([ground, foliage]), cell_units=0.1, min_pts_cell=3)

    assert cells[(0, 0)] == pytest.approx(0.0, abs=0.05)


def test_the_full_pipeline_reports_what_each_stage_dropped():
    ground = _flat_ground(n_side=6, cell=0.1, per_cell=5)
    # A cell of its own, touching the grid, holding ONLY high points.
    spike = np.array([[0.65, 0.35, 4.0]] * 20)
    island = np.array([[5.05, 5.05, 0.0]] * 5)          # far away

    result = gb.build_ground_cells(np.vstack([ground, spike, island]),
                                   cell_units=0.1, min_pts_cell=3)

    assert result["spikes_rejected"] >= 1
    assert result["disconnected_dropped"] >= 1
    assert result["binned"] > len(result["cells"])
    assert all(abs(z) < 0.1 for z in result["cells"].values())


def test_empty_input_is_handled():
    result = gb.build_ground_cells(np.zeros((0, 3)))
    assert result["cells"] == {} and result["binned"] == 0


# ---------------------------------------------------------------------------
# Refactor equivalence: the extracted module must match the code it replaced
# ---------------------------------------------------------------------------

def _reference_binning(pts, cell_units, min_pts_cell, spike_tol_units):
    """The algorithm exactly as it was inline in ground_mesh_build.main()
    before extraction. Kept here as the receipt that moving it changed nothing."""
    ij = np.floor(pts[:, :2] / cell_units).astype(np.int64)
    order = np.lexsort((ij[:, 1], ij[:, 0]))
    ij_sorted, z_sorted = ij[order], pts[order, 2]
    keys, starts = np.unique(ij_sorted, axis=0, return_index=True)
    cells = {}
    for k, (s, e) in enumerate(zip(starts, list(starts[1:]) + [len(z_sorted)])):
        if e - s >= min_pts_cell:
            cells[tuple(keys[k])] = float(np.percentile(z_sorted[s:e], 15))

    kept_cells, rejected = {}, 0
    for (i, j), z in cells.items():
        neigh = [cells[(i + di, j + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                 if (di or dj) and (i + di, j + dj) in cells]
        if len(neigh) >= 3 and abs(z - float(np.median(neigh))) > spike_tol_units:
            rejected += 1
            continue
        kept_cells[(i, j)] = z

    disconnected_dropped = 0
    if kept_cells:
        unvisited = set(kept_cells)
        best_comp = set()
        while unvisited:
            seed = unvisited.pop()
            comp, frontier = {seed}, [seed]
            while frontier:
                ci, cj = frontier.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        nb = (ci + di, cj + dj)
                        if nb in unvisited:
                            unvisited.remove(nb)
                            comp.add(nb)
                            frontier.append(nb)
            if len(comp) > len(best_comp):
                best_comp = comp
        disconnected_dropped = len(kept_cells) - len(best_comp)
        kept_cells = {k: kept_cells[k] for k in best_comp}
    return kept_cells, rejected, disconnected_dropped


@pytest.mark.parametrize("seed", range(6))
def test_extracted_binning_matches_the_original_inline_algorithm(seed):
    """Randomised clouds with spikes and islands, compared cell for cell."""
    rng = np.random.default_rng(seed)
    ground = _flat_ground(n_side=7, cell=0.1, per_cell=5, seed=seed)
    spikes = np.column_stack([rng.uniform(0, 0.7, 40), rng.uniform(0, 0.7, 40),
                              rng.uniform(1.0, 5.0, 40)])
    island = np.column_stack([rng.uniform(9, 9.3, 30), rng.uniform(9, 9.3, 30),
                              rng.normal(0, 0.01, 30)])
    cloud = np.vstack([ground, spikes, island])

    expected_cells, expected_rejected, expected_dropped = _reference_binning(
        cloud, 0.1, 3, 0.15)
    result = gb.build_ground_cells(cloud, cell_units=0.1, min_pts_cell=3,
                                   spike_tol_units=0.15)

    assert result["cells"].keys() == expected_cells.keys()
    for key, value in expected_cells.items():
        assert result["cells"][key] == pytest.approx(value)
    assert result["spikes_rejected"] == expected_rejected
    assert result["disconnected_dropped"] == expected_dropped


def test_the_cli_defaults_match_the_extracted_defaults():
    """The constants moved; the CLI must not have drifted from them."""
    source = (Path(__file__).resolve().parents[1] / "mesh" / "ground_mesh_build.py").read_text()
    assert f'default={gb.DEFAULT_CELL_UNITS}' in source
    assert f'default={gb.DEFAULT_MIN_PTS_CELL}' in source
    assert f'default={gb.DEFAULT_SPIKE_TOL_UNITS}' in source
