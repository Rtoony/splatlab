"""Deterministic exposed boundaries for bounded unions of orthogonal authored boxes."""

import numpy as np


GRID_UNITS_PER_METRE = 1_000_000_000


def joined_box_mesh(boxes):
    values = np.asarray(boxes, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != (2, 3) or not 1 <= len(values) <= 32
            or not np.isfinite(values).all() or np.any(values[:, 1] <= 0)):
        raise ValueError("Provide one to 32 finite positive orthogonal boxes")
    lower = values[:, 0] - values[:, 1] / 2
    upper = values[:, 0] + values[:, 1] / 2
    if np.any(np.abs(np.concatenate([lower, upper])) > 1000):
        raise ValueError("Authored box coordinates exceed the bounded local frame")
    lower = np.rint(lower * GRID_UNITS_PER_METRE).astype(np.int64)
    upper = np.rint(upper * GRID_UNITS_PER_METRE).astype(np.int64)
    if np.any(upper <= lower):
        raise ValueError("An authored box collapses at nanometre coordinate precision")
    axes = [np.unique(np.concatenate([lower[:, axis], upper[:, axis]])) for axis in range(3)]
    shape = tuple(len(coordinates) - 1 for coordinates in axes)
    if np.prod(shape) > 262144:
        raise ValueError("Authored boundary grid exceeds the bounded cell budget")
    occupied = np.zeros(shape, dtype=bool)
    for minimum, maximum in zip(lower, upper):
        spans = tuple(slice(int(np.searchsorted(axes[axis], minimum[axis])),
                            int(np.searchsorted(axes[axis], maximum[axis]))) for axis in range(3))
        occupied[spans] = True
    vertices, faces, lookup = [], [], {}
    for cell in np.ndindex(shape):
        if not occupied[cell]:
            continue
        for axis in range(3):
            first_axis, second_axis = (axis + 1) % 3, (axis + 2) % 3
            for direction in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += direction
                if 0 <= neighbor[axis] < shape[axis] and occupied[tuple(neighbor)]:
                    continue
                face = []
                for first_offset, second_offset in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    corner = list(cell)
                    corner[axis] += int(direction > 0)
                    corner[first_axis] += first_offset
                    corner[second_axis] += second_offset
                    key = tuple(int(axes[index][corner[index]]) for index in range(3))
                    if key not in lookup:
                        lookup[key] = len(vertices)
                        vertices.append(tuple(value / GRID_UNITS_PER_METRE for value in key))
                    face.append(lookup[key])
                faces.append(tuple(face if direction > 0 else reversed(face)))
    return vertices, faces
