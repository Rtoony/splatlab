import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import architectural_edits as architecture
import architectural_volumes as volumes
from test_architectural_edits import spec

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools/review-architectural-candidates.py"
MODULE_SPEC = importlib.util.spec_from_file_location("candidate_review", MODULE_PATH)
reviewer = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(reviewer)


def recipe():
    value = spec()
    return {"spec": value, "clip": volumes.contract(value, architecture.frame(value), volumes.ROOM_METHOD),
            "geometry_method": architecture.JOINED_GEOMETRY,
            "protected_elements": [{"slug": "keep"}], "replaced_elements": []}


def world(local, value):
    frame = architecture.frame(value)
    return np.asarray(local) @ frame["rotation"].T + frame["origin"]


def test_named_gaussian_in_room_is_detected_even_without_a_mesh_conflict():
    receipt = recipe()
    points = world([[1, 1, 2], [5, 1, 2]], receipt["spec"])
    document = {"n_rows": 2, "elements": {"keep": {"rows": [0]}}}
    result = reviewer.review(receipt["spec"], receipt, points, document, np.empty((0, 3)))
    assert result["named_clipped_center_conflicts"] == {"keep": 1}
    assert result["passes_center_and_source_point_screen"] is False


def test_source_guard_reserves_room_interior_and_floor_slab():
    receipt = recipe()
    points = world([[5, 1, 2]], receipt["spec"])
    guards = world([[1, 1, 2], [0, -.04, 0]], receipt["spec"])
    result = reviewer.review(receipt["spec"], receipt, points, {"n_rows": 1, "elements": {}}, guards)
    assert result["source_guard_conflicts"] == 2
    assert result["passes_center_and_source_point_screen"] is False


def test_unassigned_impact_is_reported_not_mistaken_for_preservation():
    receipt = recipe()
    points = world([[1, 1, 2], [5, 1, 2]], receipt["spec"])
    result = reviewer.review(receipt["spec"], receipt, points, {"n_rows": 2, "elements": {}}, np.empty((0, 3)))
    assert result["capture_impact"]["unassigned_rows"] == 1
    assert result["passes_center_and_source_point_screen"]
    assert result["full_preservation_proven"] is False


def test_source_view_selection_refuses_projection_behind_camera():
    camera = {"width": 600, "height": 400, "world_to_camera": np.eye(4),
              "model": "PINHOLE", "parameters": [300, 300, 300, 200], "center": np.zeros(3), "image_key": "photo.jpg"}
    source = SimpleNamespace(cameras={1: camera})
    assert reviewer.source_views(source, [0, 0, -2]) == []


def test_nearby_observations_are_not_implicitly_promoted_to_floor():
    source = SimpleNamespace(points=np.array([[0, .01, 0], [.1, .2, .1], [5, 0, 0], [0, 0, 0]]),
                             records=[{"id": index, "error": 1 if index != 3 else 4, "track": [1, 2, 3]}
                                      for index in range(4)])
    result = reviewer.nearby_source_samples(source, [0, 0, 0])
    assert result["quality_points"] == 2
    assert result["points_within_8cm_height"] == 1
    assert result["physical_floor_proven"] is False
