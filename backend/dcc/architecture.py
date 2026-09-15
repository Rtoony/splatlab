"""Dimensioned Z-up architectural primitives, independent of Blender."""

from __future__ import annotations

import math


def wall_boxes(width: float, height: float, thickness: float,
               opening: dict | None = None) -> list[tuple]:
    if not all(math.isfinite(value) and value > 0 for value in (width, height, thickness)):
        raise ValueError("Wall dimensions must be positive and finite")
    if not opening:
        return [((0, 0, height / 2), (width, thickness, height))]
    opening_width, opening_height = opening["width"], opening["height"]
    offset, sill = opening.get("offset", 0), opening.get("sill", 0)
    if not all(math.isfinite(value) for value in (opening_width, opening_height, offset, sill)):
        raise ValueError("Opening dimensions must be finite")
    left, right = offset - opening_width / 2, offset + opening_width / 2
    top = sill + opening_height
    if opening_width <= 0 or opening_height <= 0 or sill < 0 or left <= -width / 2 or right >= width / 2 or top >= height:
        raise ValueError("Opening must fit inside the wall, retaining both jambs and a lintel")
    boxes = [
        (((-width / 2 + left) / 2, 0, height / 2), (left + width / 2, thickness, height)),
        (((right + width / 2) / 2, 0, height / 2), (width / 2 - right, thickness, height)),
        ((offset, 0, (top + height) / 2), (opening_width, thickness, height - top)),
    ]
    if sill > 0:
        boxes.append(((offset, 0, sill / 2), (opening_width, thickness, sill)))
    return boxes


def box_mesh(boxes: list[tuple]) -> tuple[list, list]:
    vertices, faces = [], []
    corners = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
               (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
    indices = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
               (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    for center, size in boxes:
        base = len(vertices)
        vertices.extend(tuple(center[axis] + corner[axis] * size[axis] / 2 for axis in range(3)) for corner in corners)
        faces.extend(tuple(base + index for index in face) for face in indices)
    return vertices, faces


def room_boxes(width: float, depth: float, height: float, thickness: float,
               door_width: float, door_height: float, ceiling: bool = False) -> list[tuple]:
    if not math.isfinite(depth) or depth <= thickness:
        raise ValueError("Room depth must exceed wall thickness")
    boxes = wall_boxes(width, height, thickness, {"width": door_width, "height": door_height})
    boxes.extend([
        ((0, depth, height / 2), (width, thickness, height)),
        ((-width / 2 - thickness / 2, depth / 2, height / 2), (thickness, depth + thickness, height)),
        ((width / 2 + thickness / 2, depth / 2, height / 2), (thickness, depth + thickness, height)),
        ((0, depth / 2, -thickness / 2), (width + 2 * thickness, depth + thickness, thickness)),
    ])
    if ceiling:
        boxes.append(((0, depth / 2, height + thickness / 2), (width + 2 * thickness, depth + thickness, thickness)))
    return boxes
