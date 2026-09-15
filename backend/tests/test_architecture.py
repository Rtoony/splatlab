import pytest

from dcc import architecture, blender_workflow


def test_opening_keeps_jambs_and_header_without_door_collision():
    boxes = architecture.wall_boxes(4, 3, 0.2, {"width": 1, "height": 2, "offset": 0})
    assert len(boxes) == 3
    for center, size in boxes:
        assert not all(abs(point - origin) < extent / 2 for point, origin, extent in zip((0, 0, 1), center, size))
    vertices, faces = architecture.box_mesh(boxes)
    assert len(vertices) == 24 and len(faces) == 18


@pytest.mark.parametrize("opening", [{"width": 4, "height": 2}, {"width": 1, "height": 3}, {"width": 1, "height": 2, "sill": -1}, {"width": 1, "height": 2, "offset": 2}, {"width": float("nan"), "height": 2}])
def test_outside_openings_are_rejected(opening):
    with pytest.raises(ValueError):
        architecture.wall_boxes(4, 3, 0.2, opening)


def test_room_floor_and_doorway_contract():
    boxes = architecture.room_boxes(4, 5, 3, 0.2, 1, 2)
    assert len(boxes) == 7
    assert boxes[-1] == ((0, 2.5, -0.1), (4.4, 5.2, 0.2))
    assert len(architecture.room_boxes(4, 5, 3, 0.2, 1, 2, True)) == 8


def test_host_sanitizes_dimensions_material_and_paths():
    result = blender_workflow._sanitize_params("create_room", {"name": "extension", "origin": [1, 2, 3]})
    assert result["width"] == 4
    for action, params in (("create_wall", {"name": "../escape"}), ("create_room", {"name": "room", "door_height": 99}), ("assign_material", {"object": "Wall", "color": [0, -1, 0]})):
        with pytest.raises(blender_workflow.BlenderWorkflowError):
            blender_workflow._sanitize_params(action, params)
