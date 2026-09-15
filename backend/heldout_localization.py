"""Construct held-out 2D/3D correspondences without changing training geometry."""

from collections import Counter, defaultdict
from pathlib import Path
import sqlite3

import numpy as np


PAIR_BASE = 2147483647


def read_tracks(path: Path) -> dict[str, dict]:
    tracks = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Malformed training image record")
            name = fields[9].strip()
            observations_line = next(handle, None)
            if observations_line is None or name in tracks:
                raise ValueError("Missing observations or duplicate training image")
            fields = observations_line.split()
            if len(fields) % 3:
                raise ValueError("Malformed training observations")
            points = np.array([int(fields[index]) for index in range(2, len(fields), 3)], dtype=np.int64)
            pixels = np.array([[float(fields[index]), float(fields[index + 1])]
                               for index in range(0, len(fields), 3)]).reshape(-1, 2)
            if not np.isfinite(pixels).all():
                raise ValueError("Non-finite training observations")
            tracks[name] = {"point_ids": points, "pixels": pixels}
    return tracks


def read_keypoints(database: sqlite3.Connection, image_id: int) -> np.ndarray:
    row = database.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    if row is None or row[1] not in (2, 4, 6) or not 0 <= row[0] <= 100000:
        raise ValueError("Missing or unsupported image keypoints")
    values = np.frombuffer(row[2], dtype=np.float32).reshape(row[0], row[1])[:, :2].astype(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite keypoints")
    return values


def validate_training_database(database: sqlite3.Connection, tracks: dict[str, dict]) -> dict[int, dict]:
    images = list(database.execute("SELECT image_id,name FROM images"))
    if {name for _, name in images} != set(tracks):
        raise ValueError("Database must contain exactly the frozen training images")
    mapped = {}
    for identifier, name in images:
        keypoints = read_keypoints(database, identifier)
        expected = tracks[name]["pixels"]
        if keypoints.shape != expected.shape or not np.allclose(keypoints, expected, atol=1e-4, rtol=0):
            raise ValueError("Training feature indices no longer match reconstructed observations")
        mapped[identifier] = tracks[name]
    return mapped


def select_correspondences(query_id: int, query_count: int, pairs: list[tuple], training: dict[int, dict]) -> list[tuple[int, int, int]]:
    votes = defaultdict(Counter)
    for pair_id, rows, columns, blob in pairs:
        first, second = divmod(pair_id, PAIR_BASE)
        if query_id not in (first, second):
            continue
        other = second if query_id == first else first
        if other not in training:
            continue
        if columns != 2 or not 0 <= rows <= 100000:
            raise ValueError("Unsupported pair correspondence dimensions")
        matches = np.frombuffer(blob, dtype=np.uint32).reshape(rows, columns)
        query_column = 0 if query_id == first else 1
        seen = set()
        for match in matches:
            query_index, train_index = int(match[query_column]), int(match[1 - query_column])
            if not 0 <= query_index < query_count or train_index >= len(training[other]["point_ids"]):
                raise ValueError("Pair feature index is outside the frozen image")
            point_id = int(training[other]["point_ids"][train_index])
            if point_id >= 0 and (query_index, point_id) not in seen:
                votes[query_index][point_id] += 1
                seen.add((query_index, point_id))
    candidates = []
    for query_index, counts in votes.items():
        ranked = counts.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            candidates.append((query_index, ranked[0][0], ranked[0][1]))
    used_points, selected = set(), []
    for candidate in sorted(candidates, key=lambda value: (-value[2], value[0], value[1])):
        if candidate[1] not in used_points:
            used_points.add(candidate[1])
            selected.append(candidate)
    return sorted(selected)
