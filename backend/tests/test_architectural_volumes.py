import numpy as np
import pytest

import architectural_edits as architecture
import architectural_volumes as volumes
from reconstruction_evidence import EvidenceError
from test_architectural_edits import spec


def test_legacy_contract_has_no_implicit_room_mask_or_new_gate():
    coordinate = architecture.frame(spec())
    clip = volumes.contract(spec(), coordinate, volumes.LEGACY_METHOD)
    assert set(clip) == {"world_to_portal", "lower", "upper"}
    assert volumes.method(clip) == volumes.LEGACY_METHOD
    assert len(volumes.regions(clip)) == 1
    assert architecture.required_gates({"clip": clip}) == architecture.GATES


@pytest.mark.parametrize("depth", [.1, .6, 4., 6.])
def test_room_envelope_is_dimensioned_separately_from_the_narrow_cut(depth):
    value = spec() | {"cut_depth": depth}
    clip = volumes.contract(value, architecture.frame(value), volumes.ROOM_METHOD)
    assert clip["room_lower"] == pytest.approx([-1.65, .005, depth / 2 - .075])
    assert clip["room_upper"] == pytest.approx([1.65, 2.85, depth / 2 + 3.075])
    assert len(volumes.regions(clip)) == 2
    assert architecture.required_gates({"clip": clip}) == architecture.GATES | {"authored_room_interior_clear"}
    lower, upper = volumes.room_interior(value)
    assert lower == pytest.approx([-1.4999, .0001, depth / 2 + .0751])
    assert upper == pytest.approx([1.4999, 2.6999, depth / 2 + 2.9249])
    assert (upper > lower).all()


@pytest.mark.parametrize("selected", [None, "unknown", "cut-only/v2"])
def test_unknown_appearance_method_refuses(selected):
    with pytest.raises(EvidenceError, match="appearance method"):
        volumes.method({"appearance_method": selected})
    with pytest.raises(EvidenceError, match="appearance method"):
        volumes.contract(spec(), architecture.frame(spec()), selected)


def test_named_gaussian_centers_are_protected_in_both_regions_even_when_mesh_bounds_miss_them():
    clip = volumes.contract(spec(), architecture.frame(spec()), volumes.ROOM_METHOD)
    positions = np.array([[0, 1, 0], [1, 1, 2], [4, 1, 2], [0, -.1, 0]])
    document = {"n_rows": 4, "elements": {"tripod": {"rows": [0]}, "box": {"rows": [1, 2]}, "floor": {"rows": [3]}}}
    assert volumes.protected_row_conflicts(positions, document, ["tripod", "box", "floor"], clip) == {"tripod": 1, "box": 1}
    legacy = volumes.contract(spec(), architecture.frame(spec()), volumes.LEGACY_METHOD)
    assert volumes.protected_row_conflicts(positions, document, ["tripod", "box"], legacy) == {"tripod": 1}
    assert volumes.protected_row_conflicts(positions, document, ["floor"], clip) == {}


@pytest.mark.parametrize("rows", [[-1], [2], [True], [.1], [[0]]])
def test_bad_named_membership_is_not_silently_ignored(rows):
    clip = volumes.contract(spec(), architecture.frame(spec()), volumes.ROOM_METHOD)
    with pytest.raises(EvidenceError, match="row indices"):
        volumes.protected_row_conflicts(np.array([[0., 1., 0.], [1., 1., 1.]]), {"n_rows": 2, "elements": {"box": {"rows": rows}}}, ["box"], clip)


def test_captured_impact_partitions_unique_rows_without_treating_unassigned_content_as_safe():
    clip = volumes.contract(spec(), architecture.frame(spec()), volumes.ROOM_METHOD)
    points = np.array([[0., 1., 0.], [0., 1., .25], [1., 1., 2.], [4., 1., 2.], [0., -.1, 2.]])
    document = {"n_rows": 5, "elements": {"keep": {"rows": [0, 0]}, "remove": {"rows": [2]}, "empty": {"rows": []}}}
    summary, rows = volumes.capture_impact(points, document, ["keep", "empty", "generated-without-captured-rows"], ["remove"], clip)
    assert summary == {"basis": "captured-gaussian-centers/v1", "total_rows": 5,
                       "cut_rows": 2, "room_rows": 2, "overlap_rows": 1, "union_rows": 3,
                       "protected_rows": 1, "included_rows": 1, "unassigned_rows": 1}
    assert rows["unassigned"].tolist() == [1]
    assert rows["union"].tolist() == [0, 1, 2]
    assert volumes.protected_row_conflicts(points, document, ["keep", "empty"], clip) == {"keep": 1}


def test_legacy_captured_impact_does_not_implicitly_count_the_authored_room():
    clip = volumes.contract(spec(), architecture.frame(spec()), volumes.LEGACY_METHOD)
    summary, rows = volumes.capture_impact(np.array([[0., 1., 0.], [1., 1., 2.]]), {"n_rows": 2, "elements": {}}, [], [], clip)
    assert summary["room_rows"] == summary["overlap_rows"] == 0
    assert summary["union_rows"] == summary["unassigned_rows"] == 1
    assert rows["union"].tolist() == [0]


@pytest.mark.parametrize("defect", ["overlap", "row-count", "nonfinite", "bad-included-rows"])
def test_captured_impact_refuses_inconsistent_membership(defect):
    points = np.array([[0., 1., 0.], [1., 1., 2.]])
    document = {"n_rows": 2, "elements": {"keep": {"rows": [0]}, "remove": {"rows": [1]}}}
    if defect == "overlap":
        document["elements"]["remove"]["rows"] = [0]
    elif defect == "row-count":
        document["n_rows"] = 3
    elif defect == "nonfinite":
        points[0, 0] = np.nan
    else:
        document["elements"]["remove"]["rows"] = [False]
    clip = volumes.contract(spec(), architecture.frame(spec()), volumes.ROOM_METHOD)
    with pytest.raises(EvidenceError):
        volumes.capture_impact(points, document, ["keep"], ["remove"], clip)
