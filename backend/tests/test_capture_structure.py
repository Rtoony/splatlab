from copy import deepcopy

import numpy as np
import pytest

import capture_structure as structure


def camera(group, center):
    matrix = np.eye(4)
    matrix[:3, 3] = center
    return {"source_group": group, "transform_matrix": matrix.tolist()}


def test_structural_votes_require_distinct_timestamps_and_parallax():
    points = np.array([[0., 0., 5.]])
    views = [camera("same", [-1, 0, 0]), camera("same", [1, 0, 0])]
    votes = np.ones((2, 1), dtype=bool)
    assert not structure.temporal_membership(points, views, votes, votes)[0].any()
    views[1]["source_group"] = "different"
    assert structure.temporal_membership(points, views, votes, votes)[0].all()
    views[1]["transform_matrix"] = deepcopy(views[0]["transform_matrix"])
    assert not structure.temporal_membership(points, views, votes, votes)[0].any()


def test_conflicting_lens_votes_do_not_inflate_positive_evidence():
    points = np.array([[0., 0., 5.]])
    views = [camera("one", [-1, 0, 0]), camera("one", [-.9, 0, 0]), camera("two", [1, 0, 0])]
    observed = np.ones((3, 1), dtype=bool)
    positive = np.array([[True], [False], [True]])
    supported, counts, negatives = structure.temporal_membership(points, views, observed, positive)
    assert not supported.any() and counts.tolist() == [1] and negatives.tolist() == [1]
    with pytest.raises(ValueError, match="tracked"):
        structure.temporal_membership(points, views, ~observed, positive)


def test_mask_erosion_excludes_boundary_and_holes():
    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False
    actual = structure.erode_mask(mask, 1)
    assert not actual[0].any() and not actual[:, -1].any()
    assert not actual[2:5, 2:5].any() and actual[1, 1]
    np.testing.assert_array_equal(structure.erode_mask(mask, 0), mask)


def test_semantic_union_requires_matching_prompt_dimensions_and_scores(tmp_path):
    path = tmp_path / "masks.npz"
    masks = np.ones((2, 5, 5), dtype=bool)
    masks[1] = False
    np.savez(path, masks=masks, scores=np.array([.4, .9]), prompt="wall")
    assert not structure.semantic_union(path, "wall", (5, 5)).any()
    with pytest.raises(ValueError):
        structure.semantic_union(path, "sky", (5, 5))
    np.savez(path, masks=masks, scores=np.array([np.nan, .9]), prompt="wall")
    with pytest.raises(ValueError):
        structure.semantic_union(path, "wall", (5, 5))


def test_ambiguous_sparse_tracks_remain_identified_not_silently_counted(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("1 0 0 5 20 30 40 .2 1 0 1 2 2 4\n2 0 1 5 20 30 40 .2 1 1 2 2 3 4\n")
    records = structure.read_points(path)
    assert records[0]["unambiguous_track"] is False and records[1]["unambiguous_track"] is True


def test_plane_fitting_is_orientation_and_scale_independent():
    random = np.random.default_rng(90)
    points = np.column_stack((random.uniform(-2, 2, 120), random.uniform(-1, 1, 120), random.normal(0, .005, 120)))
    fit = np.arange(96, dtype=int)
    check = np.arange(96, 120, dtype=int)
    planes = structure.fit_planes(points, fit, check, 4)
    assert len(planes) == 1 and planes[0]["status"] == "source-supported-plane-candidate"
    assert planes[0]["fit_points"] == 96 and planes[0]["check_points_near_plane"] >= 20
    assert set(planes[0]["point_indices"]).isdisjoint(check)
    rotated = np.column_stack((points[:, 2], points[:, 0], points[:, 1])) * 7
    other = structure.fit_planes(rotated, fit, check, 28)
    assert other[0]["point_indices"] == planes[0]["point_indices"]
    assert other[0]["fit_rms_arbitrary_units"] == pytest.approx(planes[0]["fit_rms_arbitrary_units"] * 7)


def test_plane_checks_do_not_fit_or_hide_disagreeing_evidence():
    random = np.random.default_rng(22)
    points = np.column_stack((random.uniform(-2, 2, 120), random.uniform(-1, 1, 120), np.zeros(120)))
    fit, check = np.arange(96, dtype=int), np.arange(96, 120, dtype=int)
    points[check, 2] = 1
    plane = structure.fit_planes(points, fit, check, 4)[0]
    assert plane["status"] == "insufficient-independent-point-support"
    assert plane["check_points_near_plane"] == 0 and plane["check_median_distance_arbitrary_units"] == 1
    with pytest.raises(ValueError, match="overlap"):
        structure.fit_planes(points, fit, fit[:4], 4)
