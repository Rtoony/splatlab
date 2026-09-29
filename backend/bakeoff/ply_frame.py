"""Move a splat PLY trained on a job's transforms.json into the nerfstudio
viewer frame, so SplatLab's viewer tools (Spark agreement, camera markers)
read it like the job's own export.

A trainer fed the mirror of `processed/transforms.json` keeps its splats in
that file's frame (Spirula: `scene_transform.json` = identity). ns-export's
splat.ply is in nerfstudio's dataparser frame instead:
`p_ns = scale * (R @ p + t)` with `[R | t]` and `scale` read from the run's
`dataparser_transforms.json`. Both factors are KNOWN, so nothing is fitted:
means are mapped exactly, log-scales gain `log(scale)`, rotations are
left-multiplied by R. Only SH0 is written (the web.ply layout Spark is fed);
rotating the higher bands is not needed for that comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
              "uchar": "u1", "uint8": "u1", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"}


def read_vertex_ply(path: Path) -> dict[str, np.ndarray]:
    """Binary-little-endian, vertex-only PLY → {property: column}."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: only binary_little_endian PLY is supported")
    count, props, in_vertex = None, [], False
    for line in header:
        parts = line.split()
        if parts[:1] == ["element"]:
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
            elif int(parts[2]):
                raise ValueError(f"{path}: non-empty element {parts[1]} after vertex is not supported")
        elif parts[:1] == ["property"] and in_vertex:
            if parts[1] == "list":
                raise ValueError(f"{path}: list property in vertex element")
            props.append((parts[2], _PLY_TYPES[parts[1]]))
    data = np.frombuffer(raw, dtype=np.dtype(props), count=count, offset=end)
    return {name: np.asarray(data[name]) for name, _ in props}


def quat_from_matrix(r: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion (w, x, y, z)."""
    r = np.asarray(r, dtype=np.float64)
    tr = np.trace(r)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b, (w, x, y, z); `a` is (4,), `b` is (N, 4)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def load_dataparser_transform(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    d = json.loads(Path(path).read_text())
    m = np.asarray(d["transform"], dtype=np.float64)
    r, t, s = m[:3, :3], m[:3, 3], float(d["scale"])
    if not (np.allclose(r @ r.T, np.eye(3), atol=1e-5) and abs(np.linalg.det(r) - 1) < 1e-5):
        raise ValueError(f"{path}: transform is not a proper rotation")
    return r, t, s


def to_ns_frame(cols: dict[str, np.ndarray], r: np.ndarray, t: np.ndarray, s: float):
    """Columns of a 3DGS PLY in the transforms.json frame → (xyz, f_dc, opacity,
    log-scale, quat wxyz) in the nerfstudio frame."""
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], axis=1).astype(np.float64)
    xyz_ns = s * (xyz @ r.T + t)
    f_dc = np.stack([cols[f"f_dc_{i}"] for i in range(3)], axis=1)
    scale = np.stack([cols[f"scale_{i}"] for i in range(3)], axis=1) + np.log(s)
    q = np.stack([cols[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q_ns = quat_mul(quat_from_matrix(r), q)
    return xyz_ns, f_dc, cols["opacity"], scale, q_ns
