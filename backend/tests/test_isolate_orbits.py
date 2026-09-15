import numpy as np
import pytest

from isolate import orbits as ob
from isolate import ply_subset as ps


def test_backproject_and_project_roundtrip():
    c2w = ob.look_at([0.0, -3.0, 1.0], [0.0, 0.0, 1.0])       # looking along +y from y=-3
    fx = fy = 100.0; cx, cy = 32.0, 24.0; w, h = 64, 48
    depth = np.full((h, w), 2.0); mask = np.zeros((h, w), bool); mask[20:28, 28:36] = True
    pts = ob.backproject(mask, depth, c2w, fx, fy, cx, cy)
    assert pts.shape == (64, 3)
    assert np.allclose(np.linalg.norm(pts - c2w[:, 3], axis=1) >= 2.0, True)
    u, v, d = ob.project(pts, c2w, fx, fy, cx, cy, w, h)
    assert np.allclose(d, 2.0) and np.nanmin(u) >= 28 and np.nanmax(u) < 36 and np.nanmin(v) >= 20 and np.nanmax(v) < 28
    behind = ob.project(np.array([c2w[:, 3] - 5 * (-c2w[:, 2])]), c2w, fx, fy, cx, cy, w, h)   # 5 units behind the camera
    assert np.isnan(behind[0]).all()


def test_voxel_select_picks_neighbours_only():
    means = np.array([[0, 0, 0], [0.05, 0, 0], [1.0, 0, 0], [5, 5, 5.0]])
    picked = ob.voxel_select(means, np.array([[0.02, 0.0, 0.0]]), voxel=0.1, dilate=1)
    assert set(picked.tolist()) == {0, 1}
    assert ob.voxel_select(means, np.zeros((0, 3)), 0.1).size == 0
    stats = ob.seed_stats(means, picked)
    assert stats["n"] == 2 and abs(stats["centroid"][0] - 0.025) < 1e-9


def test_orbit_cameras_keep_distance_and_look_at_centroid():
    ref = ob.look_at([2.0, 0.0, 1.5], [0.0, 0.0, 1.0])
    cams = ob.orbit_cameras(ref, [0.0, 0.0, 1.0], azimuths_deg=[-60, 0, 60], elevations_deg=[0, 20])
    assert cams[0]["tag"] == "ref" and len(cams) == 1 + 5
    d0 = np.linalg.norm(ref[:, 3] - [0, 0, 1])
    for c in cams:
        m = np.asarray(c["c2w"]); pos = m[:, 3]; fwd = -m[:, 2]
        assert abs(np.linalg.norm(pos - [0, 0, 1]) - d0 * c["distance_scale"]) < 1e-9
        to_c = np.array([0, 0, 1.0]) - pos; to_c /= np.linalg.norm(to_c)
        assert np.dot(fwd, to_c) > 0.999
        assert abs(np.linalg.norm(m[:, 0]) - 1) < 1e-9 and abs(np.dot(m[:, 0], m[:, 1])) < 1e-9
    az60 = next(c for c in cams if c["tag"] == "az+60_el+0_d1.00"); az_60 = next(c for c in cams if c["tag"] == "az-60_el+0_d1.00")
    assert not np.allclose(az60["c2w"], az_60["c2w"])
    dolly = ob.orbit_cameras(ref, [0.0, 0.0, 1.0], [0.0], [0.0], distance_scales=[1.0, 0.5])
    assert len(dolly) == 2 and abs(np.linalg.norm(np.asarray(dolly[1]["c2w"])[:, 3] - [0, 0, 1]) - d0 * 0.5) < 1e-9


def test_select_views_keeps_reference_and_tops_up():
    cams = [{"tag": "ref", "visible_fraction": 0.9}] + [{"tag": f"c{i}", "visible_fraction": v} for i, v in enumerate([0.05, 0.6, 0.2, 0.4, 0.15, 0.0])]
    chosen = ob.select_views(cams, min_visible=0.3, min_views=5, max_views=16)
    assert [c["tag"] for c in chosen] == ["ref", "c1", "c3", "c2", "c4"]       # 2 good + top-up from >=0.1, never the 0.05/0.0 ones
    assert [c["tag"] for c in ob.select_views(cams, 0.3, 2, 2)] == ["ref", "c1"]  # max_views cap


def test_visible_fraction_and_masks():
    seed_alpha = np.array([[1.0, 1.0], [0.0, 1.0]]); seed_depth = np.array([[2.0, 2.0], [9.0, 2.0]])
    full = np.array([[2.0, 1.0], [9.0, 2.05]])
    assert abs(ob.visible_fraction(seed_alpha, seed_depth, full, tol=0.1) - 2 / 3) < 1e-9
    assert ob.visible_fraction(np.zeros((2, 2)), seed_depth, full, 0.1) == 0.0
    m = np.zeros((5, 5), bool); m[1:3, 2:4] = True
    assert ob.mask_bbox(m) == [2, 1, 4, 3] and ob.mask_bbox(np.zeros((2, 2), bool)) is None
    assert ob.mask_iou(m, m) == 1.0 and ob.mask_iou(m, ~m) == 0.0


def test_ply_subset_roundtrip(tmp_path):
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
           "property float z\nproperty float opacity\nproperty uchar flag\nend_header\n").encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4"), ("flag", "u1")])
    rows = np.array([(0, 0, 0, 0.1, 1), (1, 1, 1, 0.2, 2), (2, 2, 2, 0.3, 3)], dtype=dt)
    src = tmp_path / "a.ply"; src.write_bytes(hdr + rows.tobytes())
    n = ps.write_subset(src, tmp_path / "sub.ply", np.array([2, 0]))
    header, sub = ps.read_rows(tmp_path / "sub.ply")
    assert n == 2 and "element vertex 2" in header and sub["z"].tolist() == [2.0, 0.0] and sub["flag"].tolist() == [3, 1]
    with pytest.raises(IndexError):
        ps.write_subset(src, tmp_path / "bad.ply", np.array([5]))


def test_pick_reference_prefers_score_within_plausible_mask_fraction():
    big = np.ones((10, 10), bool); small = np.zeros((10, 10), bool); small[2:5, 2:5] = True; tiny = np.zeros((10, 10), bool)
    data = {0: {"masks": np.stack([big, small]), "scores": np.array([0.99, 0.6])},
            1: {"masks": np.stack([small]), "scores": np.array([0.8])}, 2: None}
    views = [{"cam": 0}, {"cam": 1}, {"cam": 2}]
    best = ob.pick_reference(views, lambda c: data.get(c), "x")
    assert best["row"]["cam"] == 1 and best["score"] == 0.8 and best["instance"] == 0   # the 100 % mask is rejected
    edge = np.zeros((10, 10), bool); edge[0:4, 0:4] = True                                # touches the border -> discounted
    data[1] = {"masks": np.stack([edge]), "scores": np.array([0.95])}
    assert ob.pick_reference(views, lambda c: data.get(c), "x")["row"]["cam"] == 0        # the interior 0.6 wins over an edge 0.95
    assert ob.pick_reference(views, lambda c: {"masks": np.stack([tiny]), "scores": np.array([1.0])}, "x") is None


def test_prompt_points_inside_and_outside_mask():
    m = np.zeros((60, 80), bool); m[15:45, 20:60] = True
    pts, labels = ob.prompt_points(m, n_pos=4, n_neg=3)
    assert labels.tolist() == [1, 1, 1, 1, 0, 0, 0] and pts.shape == (7, 2)
    for (x, y), lab in zip(pts.astype(int), labels):
        assert bool(m[y, x]) == bool(lab)
    x0, y0 = pts[0].astype(int)
    assert 25 <= x0 <= 55 and 20 <= y0 <= 40                   # first positive is deep inside, not on the edge
    assert ob.erode(m, 5).sum() < m.sum() < ob.dilate(m, 5).sum()


def test_path_order_walks_neighbours_from_the_reference():
    def cam(tag, x): return {"tag": tag, "c2w": [[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0]]}
    cams = [cam("far", 5.0), cam("ref", 0.0), cam("near", 1.0), cam("mid", 2.5), cam("left", -1.0)]
    assert [c["tag"] for c in ob.path_order(cams)] == ["ref", "near", "mid", "far", "left"]
    assert ob.path_order([]) == []


def test_border_fraction_and_reference_quality():
    m = np.zeros((20, 20), bool); m[5:15, 5:15] = True
    assert ob.border_fraction(m) == 0.0 and ob.reference_quality(0.8, m) == 0.8
    m[0:2, 5:15] = True                     # 20 of 120 pixels on the border -> cut-off object
    assert abs(ob.border_fraction(m) - 20 / 120) < 1e-9 and ob.touches_border(m) and ob.reference_quality(0.8, m) < 0.8 * 0.2 * 0.3
    cut = np.zeros((20, 20), bool); cut[5:15, 0:10] = True   # box reaches the left edge with few edge pixels
    assert ob.touches_border(cut) and ob.reference_quality(1.0, cut) < 0.3


def test_pullback_cameras_frame_the_seed_and_fits_in_frame():
    ref = ob.look_at([0.0, -1.0, 1.0], [0.0, 0.0, 1.0])            # 1 unit away, looking along +y
    cams = ob.pullback_cameras(ref, [0.0, 0.0, 1.0], radius=1.0, fov_y_deg=60.0, aspect=1.5, margin=1.6, extra_scales=[1.3])
    assert len(cams) == 2 and cams[0]["distance"] > 1.0                # pulled back beyond the reference distance
    d0 = cams[0]["distance"]; import math as _m
    assert abs(d0 - 1.6 / _m.tan(_m.radians(30))) < 1e-9              # sphere fills 1/1.6 of the vertical FOV
    for c in cams:
        m = np.asarray(c["c2w"]); fwd = -m[:, 2]; to_c = np.array([0, 0, 1.0]) - m[:, 3]
        assert np.dot(fwd, to_c / np.linalg.norm(to_c)) > 0.999 and abs(np.linalg.norm(to_c) - c["distance"]) < 1e-9
    assert ob.fits_in_frame([10, 10, 90, 60], 100, 100) and not ob.fits_in_frame([0, 10, 90, 60], 100, 100)
    assert not ob.fits_in_frame(None, 100, 100)


def test_pick_reground_prefers_the_instance_that_contains_and_grows_the_seed():
    h = w = 100
    sil = np.zeros((h, w), bool); sil[40:60, 40:60] = True                   # partial seed silhouette (400 px)
    whole = np.zeros((h, w), bool); whole[30:70, 20:80] = True               # full object, contains the seed, fits
    other = np.zeros((h, w), bool); other[5:25, 5:25] = True                 # unrelated instance, higher score
    edge = np.zeros((h, w), bool); edge[30:100, 20:80] = True                # contains the seed but touches the border
    small = sil.copy()                                                       # identical to the seed: adds nothing, top score
    blob = np.zeros((h, w), bool); blob[10:90, 10:90] = True                 # huge low-confidence blob (fog) — score floor drops it
    views = [{"cam": 0}, {"cam": 1}, {"cam": 2}]
    npz = {0: {"masks": np.stack([other, whole]), "scores": np.array([0.95, 0.7])},
           1: {"masks": np.stack([edge]), "scores": np.array([0.9])},
           2: {"masks": np.stack([small, blob]), "scores": np.array([0.99, 0.35])}}
    best = ob.pick_reground(views, lambda c: npz[c], lambda c: sil)
    assert best["row"]["cam"] == 0 and best["instance"] == 1 and best["fits"] and best["containment"] == 1.0
    assert best["growth"] == 6.0                                             # 2400 / 400
    # growth outranks a border touch: with the fitting candidate gone, the edge mask beats the no-growth high-score one
    npz[0] = {"masks": np.stack([other]), "scores": np.array([0.95])}
    best = ob.pick_reground(views, lambda c: npz[c], lambda c: sil)
    assert best["row"]["cam"] == 1 and not best["fits"]
    assert ob.pick_reground(views, lambda c: None, lambda c: sil) is None    # no masks at all
    assert ob.pick_reground(views, lambda c: npz[c], lambda c: np.zeros((h, w), bool)) is None   # seed not visible


def test_seed_stats_median_radius_and_mask_half_extent_ignore_depth_outliers():
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.normal(0, 0.1, (950, 3)), rng.normal(0, 3.0, (50, 3))])   # tight object + 5% wall outliers
    st = ob.seed_stats(pts, np.arange(len(pts)))
    assert st["radius_median"] < 0.3 < st["radius"]                                     # p95 is blown up, median is not
    assert abs(ob.mask_half_extent([100, 100, 500, 300], 2.0, 1000.0, 1000.0) - 0.4) < 1e-9   # 400 px wide at f=1000, d=2
    assert ob.mask_half_extent(None, 2.0, 1000.0, 1000.0) == 0.0


def test_depth_band_mask_drops_wall_pixels_seen_through_the_object():
    mask = np.zeros((10, 10), bool); mask[2:8, 2:8] = True
    depth = np.full((10, 10), 1.0); depth[4:6, 4:6] = 2.2            # gaps between leaves: the wall, 2.2x further
    depth[2, 2] = np.nan                                              # no alpha there
    kept = ob.depth_band_mask(mask, depth, rel=0.3)
    assert kept.sum() == 36 - 4 - 1 and not kept[4:6, 4:6].any() and not kept[2, 2]
    assert not ob.depth_band_mask(np.zeros((3, 3), bool), depth[:3, :3]).any()
