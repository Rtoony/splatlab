from collections import Counter

import numpy as np
import pytest

import architectural_edits as architecture
from architectural_solids import joined_box_mesh
from test_architectural_edits import spec


def triangles_from_boxes(boxes):
    vertices, quads = joined_box_mesh(boxes)
    triangles = [[face[0], face[1], face[2]] for face in quads] + [[face[0], face[2], face[3]] for face in quads]
    return np.asarray(vertices), np.asarray(triangles)


def volume(vertices, faces):
    triangles = vertices[faces]
    return np.sum(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))) / 6


def assert_oriented_closed(vertices, faces):
    directed = Counter((int(face[index]), int(face[(index + 1) % 3])) for face in faces for index in range(3))
    undirected = Counter(tuple(sorted(edge)) for face in faces
                         for edge in zip(face, np.roll(face, -1)))
    assert set(undirected.values()) == {2}, [(vertices[list(edge)].tolist(), count)
                                            for edge, count in undirected.items() if count != 2]
    assert all(count == 1 and directed[(second, first)] == 1 for (first, second), count in directed.items())
    assert len({tuple(sorted(face)) for face in faces}) == len(faces)
    assert volume(vertices, faces) > 0


def contains(points, boxes):
    result = np.zeros(len(points), dtype=bool)
    for center, size in boxes:
        lower, upper = np.asarray(center) - np.asarray(size) / 2, np.asarray(center) + np.asarray(size) / 2
        result |= ((points > lower) & (points < upper)).all(axis=1)
    return result


@pytest.mark.parametrize("offset,expected_volume,expected_area", [(0, 8, 24), (1, 12, 32), (2, 16, 40)])
def test_exposed_union_drops_duplicate_internal_and_partially_overlapping_faces(offset, expected_volume, expected_area):
    boxes = [((0, 0, 0), (2, 2, 2)), ((offset, 0, 0), (2, 2, 2))]
    vertices, faces = triangles_from_boxes(boxes)
    assert_oriented_closed(vertices, faces)
    assert volume(vertices, faces) == pytest.approx(expected_volume)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.linalg.norm(normals, axis=1).sum() / 2 == pytest.approx(expected_area)


def test_boundary_is_deterministic_under_box_order_and_roundoff_at_shared_faces():
    boxes = [((0, 0, 0), (2, 2, 2)), ((2 + 1e-15, 0, 0), (2, 2, 2))]
    assert joined_box_mesh(boxes) == joined_box_mesh(list(reversed(boxes)))
    assert joined_box_mesh(boxes) == joined_box_mesh([((0, 0, 0), (2, 2, 2)), ((2, 0, 0), (2, 2, 2))])


@pytest.mark.parametrize("boxes", [[], [((0, 0, 0), (0, 1, 1))], [((0, 0, 0), (1e-12, 1, 1))],
                                    [((float("nan"), 0, 0), (1, 1, 1))], [((1001, 0, 0), (1, 1, 1))],
                                    [((0, 0, 0), (1, 1, 1))] * 33])
def test_invalid_or_unbounded_box_sets_refuse(boxes):
    with pytest.raises(ValueError):
        joined_box_mesh(boxes)


@pytest.mark.parametrize("depth", [.1, .6, 4., 6.])
@pytest.mark.parametrize("yaw", [0., 69.43, -132.17])
def test_joined_room_preserves_intended_solid_floor_and_outward_boundary(depth, yaw):
    value = spec() | {"cut_depth": depth, "yaw_degrees": yaw}
    finished_boxes = architecture.authored_boxes(value, architecture.FINISHED_GEOMETRY)
    joined_boxes = architecture.authored_boxes(value, architecture.JOINED_GEOMETRY)
    assert joined_boxes[:-1] == finished_boxes[:-1]
    assert joined_boxes[-1][0] == finished_boxes[-1][0]
    assert joined_boxes[-1][1] == (value["opening_width"] + 2 * value["wall_thickness"],
                                   value["cut_depth"] + value["wall_thickness"], value["wall_thickness"])
    vertices, faces = architecture.room_geometry(value, architecture.JOINED_GEOMETRY)
    assert_oriented_closed(vertices, faces)
    local_world = architecture.local_coordinates(vertices, value)
    local = local_world[:, [0, 2, 1]]
    triangles = local[faces]
    centers = triangles.mean(axis=1)
    inward = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    assert contains(centers + inward * 1e-7, joined_boxes).all()
    assert not contains(centers - inward * 1e-7, joined_boxes).any()
    assert local[:, 2].min() == pytest.approx(.016 - value["wall_thickness"])
    for joined, finished in zip(architecture.protected_volume_bounds(value, architecture.JOINED_GEOMETRY),
                                architecture.protected_volume_bounds(value, architecture.FINISHED_GEOMETRY)):
        assert np.array_equal(joined[0], finished[0]) and np.array_equal(joined[1], finished[1])
    assert architecture.floor_finish_m(value, architecture.JOINED_GEOMETRY) == .016


def test_current_narrow_four_metre_entry_has_no_unexposed_jamb_or_lintel_surface():
    value = spec() | {"cut_depth": 4., "opening_width": .8, "opening_height": 2.25}
    boxes = architecture.authored_boxes(value, architecture.JOINED_GEOMETRY)
    vertices, faces = triangles_from_boxes(boxes)
    assert_oriented_closed(vertices, faces)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    centers = triangles.mean(axis=1)
    assert contains(centers - normals * 1e-7, boxes).all()
    assert not contains(centers + normals * 1e-7, boxes).any()


@pytest.mark.parametrize("thickness", [.08, .4])
@pytest.mark.parametrize("depth", [.1, 6.])
def test_joined_roof_and_boundary_remain_closed_at_small_room_limits(thickness, depth):
    value = spec() | {"wall_thickness": thickness, "cut_depth": depth, "room_width": 1.5,
                      "opening_width": .8, "opening_height": 1.9, "room_height": 2.1}
    vertices, faces = architecture.room_geometry(value, architecture.JOINED_GEOMETRY)
    assert_oriented_closed(vertices, faces)
