"""R3.2 plane scaffold core: a synthetic façade with a garage-door hole, two
windows and a sloping pavement in front, seen by four cameras with SAM3-style
masks — planes, up vector, Manhattan frame, openings, gaps, the reserved-track
check and metre conversion must all come back right from noisy surfels."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture import scaffold_core as sc  # noqa: E402

RNG = np.random.default_rng(7)
CLASSES = sc.FACADE_CLASSES + sc.NUISANCE_CLASSES + (sc.PAVEMENT_CLASS,)


def _look_at(pos, target, up=(0, 0, 1.0)):
    pos, target, up = (np.asarray(a, dtype=float) for a in (pos, target, up))
    fwd = target - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right); true_up = np.cross(right, fwd)
    m = np.eye(4); m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, true_up, -fwd, pos
    return m


def _scene():
    """Façade: plane y=0, x in [0, 6.3], z in [0, 2.8]; garage hole x 0.66..5.54, z 0..2.13; windows above.
    Pavement: z = -0.02 * y for y in [0, 8]. Surfels normal +y (towards the street) / +z."""
    xs, zs = np.meshgrid(np.arange(0, 6.3, 0.05), np.arange(0, 2.8, 0.05)); xs, zs = xs.ravel(), zs.ravel()
    door = (xs > 0.66) & (xs < 5.54) & (zs < 2.13)
    win_a = (xs > 0.9) & (xs < 1.8) & (zs > 2.25) & (zs < 2.7)
    win_b = (xs > 4.5) & (xs < 5.4) & (zs > 2.25) & (zs < 2.7)
    keep = ~door
    fx, fz = xs[keep], zs[keep]
    facade = np.stack([fx, RNG.normal(0, 0.01, len(fx)), fz], axis=1)
    f_lab = np.where(win_a[keep] | win_b[keep], 2, 0)                   # 0 wall, 2 window
    px, py = np.meshgrid(np.arange(-1, 7.3, 0.08), np.arange(0.2, 8, 0.08)); px, py = px.ravel(), py.ravel()
    pavement = np.stack([px, py, -0.02 * py + RNG.normal(0, 0.01, len(px))], axis=1)
    pts = np.concatenate([facade, pavement])
    normals = np.concatenate([np.tile([0, 1.0, 0], (len(facade), 1)), np.tile([0, 0.02, 1.0], (len(pavement), 1))])
    normals += RNG.normal(0, 0.03, normals.shape); normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals[: len(facade) // 2] *= -1                                    # half the surfel normals point the wrong way
    labels = np.concatenate([f_lab, np.full(len(pavement), 6)])          # 6 = pavement
    return pts, normals, labels, len(facade)


def _views(pts, labels):
    views = []
    for x, y in ((1.0, 7.0), (3.0, 7.5), (5.0, 7.0), (3.0, 5.0)):
        c2w = _look_at((x, y, 1.6), (3.15, 0.0, 1.3))
        masks = {name: np.zeros((256, 256), bool) for name in CLASSES}
        u, v, d, ok = sc.project(pts, c2w, 160.0, 160.0, 128.0, 128.0, 256, 256)
        for k, name in enumerate(CLASSES):
            sel = ok & (labels == k)
            masks[name][v[sel].astype(int), u[sel].astype(int)] = True
            masks[name] = np.asarray(masks[name])
        # the garage door: paint the hole region "garage door" in the masks (SAM3 sees the door leaf)
        gx, gz = np.meshgrid(np.linspace(0.7, 5.5, 60), np.linspace(0.02, 2.1, 30))
        door_pts = np.stack([gx.ravel(), np.full(gx.size, 0.02), gz.ravel()], axis=1)
        du, dv, _, dok = sc.project(door_pts, c2w, 160.0, 160.0, 128.0, 128.0, 256, 256)
        masks["garage door"][dv[dok].astype(int), du[dok].astype(int)] = True
        views.append({"c2w": c2w, "fx": 160.0, "fy": 160.0, "cx": 128.0, "cy": 128.0, "w": 256, "h": 256, "depth": None, "masks": masks})
    return views


def test_quat_to_normal_identity_and_rotation():
    assert np.allclose(sc.quat_to_normal(np.array([[1, 0, 0, 0]])), [[0, 0, 1]])
    s = math.sqrt(0.5)
    assert np.allclose(sc.quat_to_normal(np.array([[s, s, 0, 0]])), [[0, -1, 0]], atol=1e-9)   # 90° about x: z -> -y
    assert np.allclose(sc.to_original_frame(np.array([[1.0, 2, 3]]), np.array([10, 0, 0]), 2.0), [[12, 4, 6]])


def test_project_matches_capture_structure_convention():
    c2w = _look_at((0, 5, 0), (0, 0, 0)); u, v, d, ok = sc.project(np.array([[0, 0, 0.0], [0, 6, 0.0]]), c2w, 100, 100, 50, 50, 100, 100)
    assert ok.tolist() == [True, False] and abs(u[0] - 50) < 1e-9 and abs(v[0] - 50) < 1e-9 and abs(d[0] - 5) < 1e-9


def test_scaffold_recovers_facade_pavement_up_frame_and_openings():
    pts, normals, labels, n_facade = _scene()
    views = _views(pts, labels)
    votes = sc.label_votes(pts, views, CLASSES)
    cls = sc.classify(votes, CLASSES, min_votes=2)
    facade_ids = np.flatnonzero(np.isin(cls, [0, 1, 2])); pave_ids = np.flatnonzero(cls == 6)
    assert (facade_ids < n_facade).mean() > 0.97 and (pave_ids >= n_facade).mean() > 0.97   # labels land on the right surfels
    cams = np.array([v["c2w"][:3, 3] for v in views])
    n_or = sc.orient_towards(normals, pts, cams.mean(axis=0))
    w = np.ones(len(pts)); tol = 0.05
    fac = sc.extract_planes(pts[facade_ids], n_or[facade_ids], w[facade_ids], tol=tol, min_inliers=200, min_cells=20)
    pav = sc.extract_planes(pts[pave_ids], n_or[pave_ids], w[pave_ids], tol=tol, min_inliers=200, min_cells=20)
    assert fac and pav
    f, p = fac[0], pav[0]
    assert abs(abs(f["normal"][1]) - 1) < 0.02 and f["rms"] < 0.02 and len(f["ids"]) > 0.9 * len(facade_ids)
    up = sc.up_vector({"normal": p["normal"], "centre": p["centre"]}, cams)
    assert up[2] > 0.99                                                   # cameras above the pavement -> +z
    frame = sc.manhattan_frame(up, f["normal"])
    assert abs(abs(frame[0][0]) - 1) < 0.03 and frame[1][1] > 0.99 and frame[2][2] > 0.99   # right = x, out = +y (street), up = z
    assert sc.tilt_deg(f["normal"], up) < 2.0
    # openings named by the masks: windows; the door hole is a gap the masks call "garage door" only on the leaf
    f["ids"] = facade_ids[f["ids"]]                                       # back to global surfel ids for label lookup
    wins = sc.opening_rectangles(f, pts, cls, class_index=2, cell=2 * tol, min_cells=4)
    assert len(wins) == 2 and all(0.7 < r["width"] < 1.1 and 0.3 < r["height"] < 0.6 for r in wins)
    assert not sc.gap_rectangles(f, pts, cell=2 * tol, min_cells=12)              # the door hole is open to the ground line
    gaps = sc.gap_rectangles(f, pts, cell=2 * tol, min_cells=12, closed_edges=(sc.ground_edge(f, up),))
    assert gaps and abs(gaps[0]["width"] - 4.88) < 0.25 and abs(gaps[0]["height"] - 2.13) < 0.25   # the garage door hole
    corners = sc.rect_corners_3d(f, gaps[0]["lower_uv"], gaps[0]["upper_uv"])
    assert corners.shape == (4, 3) and np.all(np.abs(corners[:, 1]) < 0.05)
    # reserved-track check: points on the plane inside the footprint are "near"
    chk = sc.reserved_track_check(f, np.array([[3.0, 0.0, 2.5], [0.3, 0.0, 1.0], [3.0, 0.4, 1.0], [30.0, 0.0, 1.0]]), tol)
    assert chk["in_footprint"] == 3 and chk["near"] == 2


def test_similarity_parts_and_apply():
    s = 1.5; R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]); t = np.array([1, 2, 3.0])
    m = np.eye(4); m[:3, :3] = s * R; m[:3, 3] = t
    r, scale, tr = sc.similarity_parts(m)
    assert abs(scale - 1.5) < 1e-9 and np.allclose(r, R) and np.allclose(tr, t)
    assert np.allclose(sc.apply_similarity(np.array([[1, 0, 0.0]]), m), [[1, 3.5, 3]])
    assert np.allclose(sc.offset_peaks(np.array([0, 0.01, 0.02, 1.0, 1.01, 1.02]), np.ones(6), 0.05, 1.0), [0.025, 1.025], atol=0.03)


def test_backproject_depth_recovers_plane_points_and_normals():
    c2w = _look_at((0, 5, 0), (0, 0, 0)); fx = fy = 100.0; cx = cy = 50.0
    # a wall at y = 0 facing the camera: depth = 5 everywhere except a step edge in the right third
    depth = np.full((100, 100), 5.0); depth[:, 70:] = 6.0
    pts, n, rows, cols = sc.backproject_depth(depth, c2w, fx, fy, cx, cy, step=5)
    assert len(pts) > 200 and np.all(np.abs(pts[cols < 70, 1]) < 1e-9) and np.all(np.abs(pts[cols >= 70, 1] + 1) < 1e-9)
    assert np.allclose(np.abs(n[:, 1]), 1, atol=1e-6) and np.all(n[:, 1] > 0)      # normals face the camera (+y)
    assert not np.any((cols >= 66) & (cols <= 72))                                  # the depth edge is dropped
    # centre pixel maps to the optical axis
    u = np.flatnonzero((rows == 50) & (cols == 50)); assert len(u) == 1 and np.allclose(pts[u[0]], [0, 0, 0], atol=1e-9)


def test_gravity_basis_rebase_and_track_anchoring():
    up = np.array([0, 0, 1.0]); n = np.array([0, 1.0, 0])
    b = sc.gravity_basis(n, up)
    assert np.allclose(b[1], up) and np.allclose(np.abs(b[0]), [1, 0, 0]) and abs(b[0] @ n) < 1e-12
    pts = np.array([[x, 0.0, z] for x in np.linspace(0, 4, 21) for z in np.linspace(0, 2, 11)])
    patch = {"centre": pts.mean(axis=0), "normal": n, "basis": np.array([[0.6, 0, 0.8], [-0.8, 0, 0.6]]), "ids": np.arange(len(pts)), "rms": 0.0, "extent": [0, 0]}
    rb = sc.rebase_patch(patch, pts, b)
    assert np.allclose(rb["extent"], [4, 2], atol=1e-9)                                   # plumb bounds, not the rotated bbox
    tracks = np.array([[x, 0.3, 1.0] for x in np.linspace(0.2, 3.8, 10)] + [[2.0, 2.5, 1.0]])   # wall really sits 0.3 further; one outlier
    anchored, info = sc.anchor_to_tracks(rb, tracks, min_tracks=8)
    assert info["applied"] and info["tracks_used"] == 10 and abs(info["shift_units"] - 0.3) < 1e-9 and abs(anchored["centre"][1] - 0.3) < 1e-9
    _, info2 = sc.anchor_to_tracks(rb, tracks[:5], min_tracks=8)
    assert not info2["applied"] and info2["tracks_in_footprint"] == 5
    outside = np.array([[x, 0.3, 2.3] for x in np.linspace(0.2, 3.8, 10)])                 # just above the patch (z 0..2)
    assert sc.anchor_to_tracks(rb, outside, min_tracks=8)[1]["tracks_in_footprint"] == 0
    assert sc.anchor_to_tracks(rb, outside, min_tracks=8, dilate=0.2)[1]["applied"]          # dilated footprint catches them
    assert sc.anchor_to_tracks(rb, tracks, min_tracks=8, band=0.1)[1]["tracks_in_footprint"] == 0   # band excludes the 0.3-offset tracks
    chk = sc.reserved_track_check(rb, np.array([[1.0, 0.02, 1.0], [2.0, 0.9, 1.0]]), tol=0.05, band=0.5)
    assert chk["in_footprint"] == 2 and chk["far"] == 1 and chk["near"] == 1


def test_ray_plane_openings_and_coplanar_merge():
    up = np.array([0, 0, 1.0]); n = np.array([0, 1.0, 0])
    patch = {"centre": np.array([3.0, 0.0, 1.4]), "normal": n, "basis": sc.gravity_basis(n, up), "lower_uv": np.array([-3.0, -1.4]), "upper_uv": np.array([3.0, 1.4])}
    c2w = _look_at((3.0, 8.0, 1.6), (3.0, 0.0, 1.4)); fx = fy = 200.0; cx = cy = 128.0
    # a door 0.66..5.54 x 0..2.13 on the wall, painted as a mask by projecting its interior
    gx, gz = np.meshgrid(np.linspace(0.67, 5.53, 80), np.linspace(0.01, 2.12, 40))
    u, v, _, ok = sc.project(np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=1), c2w, fx, fy, cx, cy, 256, 256)
    mask = np.zeros((256, 256), bool); mask[v[ok].astype(int), u[ok].astype(int)] = True
    uv, hit = sc.ray_plane_uv(np.nonzero(mask), c2w, fx, fy, cx, cy, patch)
    assert hit.all() and abs((uv[:, 0].max() - uv[:, 0].min()) - 4.86) < 0.1 and abs((uv[:, 1].max() - uv[:, 1].min()) - 2.11) < 0.1
    assert not sc.touches_image_border(mask) and sc.touches_image_border(np.ones((4, 4), bool))
    rects = [{"lower_uv": [0.6, -1.4], "upper_uv": [5.5, 0.7], "view": 0}, {"lower_uv": [0.7, -1.4], "upper_uv": [5.6, 0.75], "view": 1},
             {"lower_uv": [-2.9, 0.2], "upper_uv": [-1.9, 1.0], "view": 0}]
    groups = sc.cluster_rectangles(rects, join_dist=0.5)
    assert len(groups) == 2 and groups[0]["n_views"] == 2 and abs(groups[0]["width"] - 4.9) < 1e-9 and groups[1]["n_views"] == 1
    pts = np.array([[x, 0.0, z] for x in np.linspace(0, 4, 21) for z in np.linspace(0, 2, 11)])
    a = {"centre": pts[:100].mean(axis=0), "normal": n, "basis": sc.gravity_basis(n, up), "ids": np.arange(100), "rms": 0.0, "extent": [0, 0]}
    b = {"centre": pts[100:].mean(axis=0) + [0, 0.05, 0], "normal": n, "basis": sc.gravity_basis(n, up), "ids": np.arange(100, len(pts)), "rms": 0.0, "extent": [0, 0]}
    far = {**b, "centre": b["centre"] + [0, 1.0, 0], "ids": np.arange(100, 120)}
    merged = sc.merge_coplanar([a, b, far], pts, angle_deg=6, offset_tol=0.1)
    assert len(merged) == 2 and merged[0].get("merged") == 2 and len(merged[0]["ids"]) == len(pts)
