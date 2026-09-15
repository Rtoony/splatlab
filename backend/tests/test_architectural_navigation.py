import numpy as np
import pytest

import architectural_edits as architecture
import architectural_navigation as navigation
from reconstruction_evidence import EvidenceError
from test_architectural_edits import spec


def test_legacy_recipe_keeps_parallel_lanes_and_exact_sampling():
    points, count = navigation.local_routes(spec(), navigation.LEGACY_RECIPE)
    assert count == 94
    assert np.unique(points[:, 0]) == pytest.approx([-.27, 0, .27])
    assert points[:, 2].min() == pytest.approx(-.9)


def test_largest_authoring_spec_keeps_arc_spacing_and_footprint_budget_bounded():
    value = spec() | {"cut_depth": 6., "room_depth": 8., "opening_width": 3., "room_width": 4.}
    points, count = navigation.local_routes(value, navigation.recipe())
    assert len(points) <= 6000
    assert np.max(np.linalg.norm(np.diff(points.reshape(3, count, 3), axis=1), axis=2)) <= .020000001
    center, floor = navigation.floor_support(points, lambda query: np.zeros(len(query)), .22)
    assert center == pytest.approx(floor)


@pytest.mark.parametrize("depth", [.1, .6, 2., 4., 6.])
def test_entry_probe_fans_into_all_three_full_interior_lanes_without_shortening_the_approach(depth):
    value = spec() | {"cut_depth": depth}
    points, count = navigation.local_routes(value, navigation.recipe())
    lanes = points.reshape(3, count, 3)
    near = -depth / 2
    assert lanes[:, 0, 0] == pytest.approx([0, 0, 0])
    assert lanes[:, 0, 2] == pytest.approx([near - .6] * 3)
    assert np.max(np.diff(lanes[1, :, 2])) <= .020000001
    for index, lateral in enumerate([-.27, 0., .27]):
        assert lanes[index, lanes[index, :, 2] >= near + .3, 0] == pytest.approx(lateral)
    assert lanes[:, -1, 2] == pytest.approx([depth / 2 + value["room_depth"] - .5] * 3)
    assert navigation.recipe()["capsule_radius_m"] == .22
    assert navigation.recipe()["capsule_height_m"] == 1.7


@pytest.mark.parametrize("change", [{"capsule_radius_m": .2}, {"capsule_height_m": 1.6}, {"floor_tolerance_m": .1},
                                   {"navigation_method": "unknown"}, {"ignored": True}])
def test_unknown_or_weakened_recipe_refuses(change):
    with pytest.raises(EvidenceError, match="recipe"):
        navigation.method(navigation.recipe() | change)


def test_flat_floor_support_does_not_raise_the_body_or_change_the_plane():
    points = np.array([[0., -1., 0.], [2., -1., 3.]])
    seen = []
    def sample(query):
        seen.append(query.copy())
        return np.full(len(query), -1.)
    center, floor = navigation.floor_support(points, sample, .22)
    assert center == pytest.approx([-1., -1.])
    assert floor == pytest.approx(center)
    assert seen[0].shape == (258, 3)
    assert np.max(np.linalg.norm((seen[0][:129] - points[0])[:, [0, 2]], axis=1)) == pytest.approx(.25)


def test_neighboring_low_step_is_accounted_for_without_losing_the_center_ray():
    center, floor = navigation.floor_support(np.array([[0., 0., 0.]]), lambda query: np.where(query[:, 0] >= .05, .04, 0.), .22)
    assert center[0] == 0
    assert 0 < floor[0] < .04


@pytest.mark.parametrize("failure", ["missing-center", "raised-obstacle", "wrong-size"])
def test_footprint_cannot_bridge_missing_floor_or_lift_through_a_large_obstacle(failure):
    def sample(query):
        values = np.zeros(len(query))
        if failure == "missing-center":
            values[0] = np.nan
        elif failure == "raised-obstacle":
            values[query[:, 0] >= .05] = .2
        else:
            return values[:-1]
        return values
    if failure == "wrong-size":
        with pytest.raises(EvidenceError, match="sampler"):
            navigation.floor_support(np.array([[0., 0., 0.]]), sample, .22)
    else:
        _, floor = navigation.floor_support(np.array([[0., 0., 0.]]), sample, .22)
        assert np.isnan(floor[0])


def document_fixture():
    value = spec()
    frame = architecture.frame(value)
    points, _ = navigation.local_routes(value, navigation.recipe())
    document = {"v": 2, "navigation_method": navigation.ENTRY_METHOD, "architecture_id": "fixture",
        "frame": "world-y-up-metres", "gates": {"capsule_route_clear": True}, "capsule_radius_m": .22, "capsule_height_m": 1.7,
        "route": (points @ frame["rotation"].T + frame["origin"]).tolist(),
        "floor_y": [-1.] * len(points), "center_floor_y": [-1.] * len(points)}
    return value, frame, document


def test_navigation_document_binds_method_complete_routes_body_and_floor():
    value, frame, document = document_fixture()
    navigation.verify_document(value, navigation.recipe(), frame, document, "fixture", {"capsule_route_clear": True})


@pytest.mark.parametrize("defect", ["method", "version", "radius", "gate", "route", "floor", "center", "step", "rows"])
def test_inconsistent_navigation_document_refuses(defect):
    value, frame, document = document_fixture()
    if defect == "method": document["navigation_method"] = navigation.LEGACY_METHOD
    if defect == "version": document["v"] = 1
    if defect == "radius": document["capsule_radius_m"] = .2
    if defect == "gate": document["gates"] = {}
    if defect == "route": document["route"][0][0] += .001
    if defect == "floor": document["floor_y"][0] = None
    if defect == "center": document["center_floor_y"][0] = -1.09
    if defect == "step": document["floor_y"][0] = -.95
    if defect == "rows": document["route"].pop()
    with pytest.raises(EvidenceError):
        navigation.verify_document(value, navigation.recipe(), frame, document, "fixture", {"capsule_route_clear": True})
