"""Tests for the SAM 3D generative gate in mesh/object_generate.py.

Why this file exists: `object_generate.py` decides, before a 13 GB model load,
whether a SAM 3 mask may stand in for a captured object, and afterwards whether
a fitted orientation is trustworthy enough to emit a capture-frame transform.
Both decisions were previously exercised only by real GPU runs, so a regression
in either would have surfaced as a confidently-wrong reconstruction rather than
a failure. These are CPU-only tests of that decision logic.

The module imports cleanly under the test interpreter (numpy + scipy + PIL are
all present); no stubbing is needed and nothing here loads a model or touches
the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mesh"))

import object_generate as og  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic camera + object helpers
#
# object_generate consumes nerfstudio's OpenGL camera_to_worlds (+X right,
# +Y up, -Z forward) and flips it to OpenCV internally. These helpers build a
# camera the same way round so the tests exercise the real projection path.
# ---------------------------------------------------------------------------

IMG = 128
BOX = (0, 0, IMG, IMG)
SHAPE = (BOX[3] - BOX[1], BOX[2] - BOX[0])


def _camera(position=(0.0, -3.0, 0.0), fx=260.0):
    """Camera at `position` looking at the origin, world up = +Z."""
    forward = np.array([0.0, 0.0, 0.0]) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    c2w_cv = np.eye(4)
    c2w_cv[:3, 0] = right
    c2w_cv[:3, 1] = down
    c2w_cv[:3, 2] = forward
    c2w_cv[:3, 3] = np.asarray(position, float)

    # The module computes w2c = inv(c2w_gl @ _GL_TO_CV); _GL_TO_CV is its own
    # inverse, so the OpenGL matrix to hand back is c2w_cv @ _GL_TO_CV.
    c2w_gl = c2w_cv @ og._GL_TO_CV
    return {
        "camera_to_worlds_3x4_opengl": c2w_gl[:3, :].tolist(),
        "fx": fx, "fy": fx, "cx": IMG / 2.0, "cy": IMG / 2.0,
    }


def _asymmetric_object(rng=None, n=3000):
    """An L-shaped point cloud: asymmetric, so a wrong yaw scores differently."""
    rng = rng or np.random.default_rng(0)
    tall = rng.uniform([-0.2, -0.2, -0.5], [0.2, 0.2, 0.5], size=(n // 2, 3))
    arm = rng.uniform([0.2, -0.2, 0.1], [0.8, 0.2, 0.5], size=(n // 2, 3))
    return np.vstack([tall, arm])


def _silhouette(points, cam, box=BOX):
    px, _ = og.project_to_crop(points, cam, box)
    return og.rasterise_silhouette(px, SHAPE)


def _bbox(points):
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------

def test_iou_identical_is_one():
    a = np.zeros((8, 8), bool)
    a[2:6, 2:6] = True
    assert og.iou(a, a) == pytest.approx(1.0)


def test_iou_disjoint_is_zero():
    a = np.zeros((8, 8), bool)
    b = np.zeros((8, 8), bool)
    a[0:3, 0:3] = True
    b[5:8, 5:8] = True
    assert og.iou(a, b) == 0.0


def test_iou_empty_union_is_zero_not_nan():
    """Both masks empty must not divide by zero."""
    empty = np.zeros((8, 8), bool)
    assert og.iou(empty, empty) == 0.0


def test_iou_half_overlap():
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:, 0:4] = True          # 40 px
    b[:, 2:6] = True          # 40 px, 20 shared
    assert og.iou(a, b) == pytest.approx(20 / 60)


# ---------------------------------------------------------------------------
# rasterise_silhouette
# ---------------------------------------------------------------------------

def test_rasterise_empty_input_returns_empty_mask():
    m = og.rasterise_silhouette(np.zeros((0, 2)), SHAPE)
    assert m.shape == SHAPE and not m.any()


def test_rasterise_all_out_of_bounds_returns_empty_mask():
    px = np.array([[-50.0, -50.0], [500.0, 500.0]])
    assert not og.rasterise_silhouette(px, SHAPE).any()


def test_rasterise_fills_interior_holes():
    """A ring of points must come back as a filled disc: the dilate/close/fill
    recipe is what makes a sparse gaussian projection comparable to a dense
    SAM mask."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    px = np.stack([64 + 30 * np.cos(t), 64 + 30 * np.sin(t)], axis=1)
    m = og.rasterise_silhouette(px, SHAPE)
    assert m[64, 64], "centre of the ring should be filled"
    assert not m[5, 5], "far outside the ring should stay empty"


def test_rasterise_is_deterministic():
    px = np.random.default_rng(1).uniform(20, 100, size=(500, 2))
    assert np.array_equal(
        og.rasterise_silhouette(px, SHAPE), og.rasterise_silhouette(px, SHAPE))


# ---------------------------------------------------------------------------
# pick_and_gate_mask — selection contract
# ---------------------------------------------------------------------------

def _mask(rows, cols, shape=(64, 64)):
    m = np.zeros(shape, bool)
    m[rows, cols] = True
    return m


def test_gate_raises_when_sam_returned_nothing():
    with pytest.raises(og.Fatal, match="zero instances"):
        og.pick_and_gate_mask(np.zeros((0, 64, 64), bool), np.array([]), None, 0.35)


def test_gate_raises_on_none_masks():
    with pytest.raises(og.Fatal, match="zero instances"):
        og.pick_and_gate_mask(None, np.array([]), None, 0.35)


def test_gate_selects_by_agreement_not_detector_confidence():
    """THE contract: a co-visible neighbour with a higher SAM score must lose to
    the instance that actually agrees with the captured 3D object. This is what
    stops the bonsai inside the bicycle crop being rebuilt as the bicycle."""
    captured = _mask(slice(10, 40), slice(10, 40))     # the real object
    neighbour = _mask(slice(45, 60), slice(45, 60))    # confident, but elsewhere
    overlapping = _mask(slice(12, 42), slice(12, 42))  # agrees with captured

    masks = np.stack([neighbour, overlapping])
    scores = np.array([0.99, 0.42])                    # neighbour "wins" on score

    verdict, chosen = og.pick_and_gate_mask(masks, scores, captured, 0.35)

    assert verdict["selected_by"] == "iou_vs_captured"
    assert verdict["chosen"]["index"] == 1
    assert np.array_equal(chosen, overlapping)
    assert verdict["passed"] is True


def test_gate_falls_back_to_score_without_captured_silhouette():
    a = _mask(slice(10, 40), slice(10, 40))
    b = _mask(slice(12, 42), slice(12, 42))
    verdict, chosen = og.pick_and_gate_mask(np.stack([a, b]), np.array([0.2, 0.8]), None, 0.35)

    assert verdict["selected_by"] == "score"
    assert verdict["chosen"]["index"] == 1
    assert np.array_equal(chosen, b)
    assert verdict["method"].startswith("2d-sanity-only")
    assert "iou_vs_captured_object" not in verdict


def test_gate_records_every_candidate_for_the_receipt():
    a = _mask(slice(10, 40), slice(10, 40))
    b = _mask(slice(45, 60), slice(45, 60))
    verdict, _ = og.pick_and_gate_mask(np.stack([a, b]), np.array([0.9, 0.1]), a, 0.35)

    assert len(verdict["candidates"]) == 2
    assert all("iou_vs_captured" in c for c in verdict["candidates"])
    assert all("score" in c and "coverage" in c for c in verdict["candidates"])


# ---------------------------------------------------------------------------
# pick_and_gate_mask — refusal contract ("a refusal is a valid result")
# ---------------------------------------------------------------------------

def test_gate_refuses_below_min_iou():
    captured = _mask(slice(10, 40), slice(10, 40))
    poor = _mask(slice(38, 60), slice(38, 60))         # barely touches
    verdict, _ = og.pick_and_gate_mask(np.stack([poor]), np.array([0.9]), captured, 0.35)

    assert verdict["passed"] is False
    assert any("min-mask-iou" in p for p in verdict["problems"])


def test_gate_refuses_a_speck():
    tiny = _mask(slice(0, 4), slice(0, 4))             # 16 / 4096 = 0.4 %
    verdict, _ = og.pick_and_gate_mask(np.stack([tiny]), np.array([0.9]), None, 0.35)

    assert verdict["passed"] is False
    assert any("covers only" in p for p in verdict["problems"])


def test_gate_refuses_a_mask_with_no_background_left():
    everything = np.ones((64, 64), bool)
    verdict, _ = og.pick_and_gate_mask(
        np.stack([everything]), np.array([0.9]), None, 0.35)

    assert verdict["passed"] is False
    assert any("no background left" in p for p in verdict["problems"])


def test_gate_refuses_a_fragmented_mask():
    """Three comparable blobs: no single component is half the mask."""
    frag = np.zeros((64, 64), bool)
    frag[5:20, 5:20] = True
    frag[25:40, 25:40] = True
    frag[45:60, 45:60] = True
    verdict, _ = og.pick_and_gate_mask(np.stack([frag]), np.array([0.9]), None, 0.35)

    assert verdict["components"] == 3
    assert verdict["largest_component_frac"] == pytest.approx(1 / 3, abs=0.01)
    assert verdict["passed"] is False
    assert any("fragmented" in p for p in verdict["problems"])


def test_gate_fragmentation_threshold_is_strictly_below_half():
    """Boundary: two equal components sit at exactly 0.50 and are ACCEPTED —
    the check is `< 0.5`. Pinned so the threshold cannot drift unnoticed."""
    frag = np.zeros((64, 64), bool)
    frag[5:20, 5:20] = True
    frag[40:55, 40:55] = True
    verdict, _ = og.pick_and_gate_mask(np.stack([frag]), np.array([0.9]), None, 0.35)

    assert verdict["components"] == 2
    assert verdict["largest_component_frac"] == pytest.approx(0.5)
    assert not any("fragmented" in p for p in verdict["problems"])


def test_gate_accepts_a_single_solid_blob():
    solid = _mask(slice(16, 48), slice(16, 48))        # 25 % coverage, 1 component
    verdict, _ = og.pick_and_gate_mask(np.stack([solid]), np.array([0.9]), solid, 0.35)

    assert verdict["components"] == 1
    assert verdict["largest_component_frac"] == pytest.approx(1.0)
    assert verdict["problems"] == []
    assert verdict["passed"] is True
    assert verdict["iou_vs_captured_object"] == pytest.approx(1.0)


def test_gate_passed_is_exactly_no_problems():
    """`passed` must never diverge from `problems` — the report is the receipt."""
    frag = np.zeros((64, 64), bool)
    frag[5:20, 5:20] = True
    frag[40:55, 40:55] = True
    for masks, cap in ((np.stack([frag]), None),
                       (np.stack([_mask(slice(16, 48), slice(16, 48))]), None)):
        verdict, _ = og.pick_and_gate_mask(masks, np.array([0.9]), cap, 0.35)
        assert verdict["passed"] == (not verdict["problems"])


# ---------------------------------------------------------------------------
# Rotation helpers — must stay proper rotations, never reflections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0.0, 37.0, 90.0, 180.0, 275.0])
def test_yaw_is_a_proper_rotation(deg):
    R = og._yaw(np.array([0.0, 0.0, 1.0]), np.radians(deg))
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert R @ R.T == pytest.approx(np.eye(3))


def test_yaw_full_turn_is_identity():
    R = og._yaw(np.array([0.0, 0.0, 1.0]), 2 * np.pi)
    assert R == pytest.approx(np.eye(3), abs=1e-12)


def test_yaw_rotates_about_the_given_axis():
    axis = np.array([0.0, 0.0, 1.0])
    R = og._yaw(axis, np.radians(90.0))
    assert R @ axis == pytest.approx(axis)
    assert R @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("src", [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 1], [0.3, -0.7, 0.2],
])
def test_axis_rotation_maps_src_onto_dst_and_stays_proper(src):
    a = np.asarray(src, float)
    dst = np.array([0.0, 0.0, 1.0])
    R = og._axis_rotation(a, dst)

    assert np.linalg.det(R) == pytest.approx(1.0), "a reflection would mirror the object"
    assert R @ (a / np.linalg.norm(a)) == pytest.approx(dst, abs=1e-12)


def test_axis_rotation_antiparallel_case_is_a_rotation_not_a_reflection():
    """The degenerate case the docstring calls out: no minimal axis exists."""
    R = og._axis_rotation(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]))
    assert np.linalg.det(R) == pytest.approx(1.0)
    assert R @ np.array([0.0, 0.0, -1.0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-12)


# ---------------------------------------------------------------------------
# fit_similarity
# ---------------------------------------------------------------------------

def test_fit_similarity_recovers_a_known_scale_and_translation():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1, 1, size=(4000, 3))
    scale, offset = 4.0, np.array([10.0, -2.0, 0.5])
    target = pts * scale + offset

    s, t = og.fit_similarity(pts, _bbox(target))

    assert s == pytest.approx(scale, rel=0.05)
    assert (pts * s + t).mean(axis=0) == pytest.approx(target.mean(axis=0), abs=0.1)


def test_fit_similarity_is_robust_to_a_far_outlier():
    """Percentile-based fitting is the point: one stray gaussian must not
    collapse the scale."""
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1, 1, size=(4000, 3))
    target = pts * 4.0
    clean_s, _ = og.fit_similarity(pts, _bbox(target))

    dirty = np.vstack([pts, [[500.0, 500.0, 500.0]]])
    dirty_s, _ = og.fit_similarity(dirty, _bbox(target))

    assert dirty_s == pytest.approx(clean_s, rel=0.05)


# ---------------------------------------------------------------------------
# derive_placement
# ---------------------------------------------------------------------------

def test_placement_degrades_honestly_without_a_camera():
    verts = _asymmetric_object()
    frame = og.derive_placement(verts, _bbox(verts), None, None, None, None, 0.5)

    assert frame["placement_resolved"] is False
    assert "no crop camera" in frame["reason"]
    assert "transform_4x4_generated_to_capture" not in frame
    # Scale statistics are still reported — they just do not place the object.
    assert frame["uniform_scale_from_bbox_diagonal"] > 0


def test_placement_reports_orientation_independent_scale_only():
    verts = _asymmetric_object()
    frame = og.derive_placement(verts, _bbox(verts * 3.0), None, None, None, None, 0.5)

    assert frame["uniform_scale_from_bbox_diagonal"] == pytest.approx(3.0, rel=0.05)
    assert "per_axis" not in frame, "per-axis ratios are meaningless before orientation"


def test_placement_withholds_the_transform_when_the_silhouette_disagrees():
    """A mask that has nothing to do with the object must not yield a transform."""
    cam = _camera()
    scene = _asymmetric_object()
    wrong_mask = np.zeros(SHAPE, bool)
    wrong_mask[0:12, 0:12] = True

    frame = og.derive_placement(
        scene, _bbox(scene), cam, BOX, wrong_mask, None, min_iou=0.5, yaw_step_deg=30.0)

    assert frame["placement_resolved"] is False
    assert "min-placement-iou" in frame["reason"]
    assert "transform_4x4_generated_to_capture" not in frame
    assert frame["orientation_search"]["best_silhouette_iou"] < 0.5


def test_placement_resolves_a_recoverable_orientation_and_emits_a_transform():
    """End-to-end on synthetic data: rotate an object out of the capture frame by
    a rotation inside the search family, then check the search puts it back and
    the emitted 4x4 actually maps generated -> capture."""
    cam = _camera()
    scene = _asymmetric_object()
    bbox = _bbox(scene)
    mask = _silhouette(scene, cam)

    scene_up = np.array([0.0, 0.0, 1.0])
    r_true = og._yaw(scene_up, np.radians(40.0)) @ og._axis_rotation(
        np.array([0.0, 1.0, 0.0]), scene_up)
    centre = scene.mean(axis=0)
    generated = (scene - centre) @ r_true * 0.3      # SAM3D-like normalised frame

    frame = og.derive_placement(
        generated, bbox, cam, BOX, mask, mask, min_iou=0.5, yaw_step_deg=10.0)

    assert frame["placement_resolved"] is True, frame.get("reason")
    assert frame["orientation_search"]["best_silhouette_iou"] >= 0.5
    assert frame["fitted_uniform_scale"] == pytest.approx(1 / 0.3, rel=0.1)

    m = np.asarray(frame["transform_4x4_generated_to_capture"], float)
    assert m.shape == (4, 4)
    assert m[3].tolist() == [0.0, 0.0, 0.0, 1.0]

    placed = (m[:3, :3] @ generated.T).T + m[:3, 3]
    assert placed.mean(axis=0) == pytest.approx(scene.mean(axis=0), abs=0.15)
    assert og.iou(_silhouette(placed, cam), mask) >= 0.5


def test_placement_reports_the_captured_ceiling_for_comparison():
    """The evidence is the ratio to what the captured object itself scores, not
    a bare IoU number."""
    cam = _camera()
    scene = _asymmetric_object()
    mask = _silhouette(scene, cam)

    frame = og.derive_placement(
        scene, _bbox(scene), cam, BOX, mask, mask, min_iou=0.5, yaw_step_deg=30.0)

    search = frame["orientation_search"]
    assert search["captured_object_silhouette_iou"] == pytest.approx(1.0)
    assert search["fraction_of_captured_ceiling"] is not None
    assert search["scored_from_n_views"] == 1
    assert len(search["iou_spread"]) == 2
    assert search["iou_spread"][0] <= search["iou_spread"][1]


def test_placement_search_covers_every_up_axis_and_yaw():
    cam = _camera()
    scene = _asymmetric_object(n=600)
    mask = _silhouette(scene, cam)

    frame = og.derive_placement(
        scene, _bbox(scene), cam, BOX, mask, mask, min_iou=0.99, yaw_step_deg=90.0)

    # 6 up-axis assignments x 4 coarse yaws, plus the refinement pass.
    search = frame["orientation_search"]
    assert search["candidates_scored"] >= 24
    assert 0 <= search["best_up_axis_index"] <= 5
    # The refinement pass sweeps best +/- yaw_step, so a coarse winner at 0 deg
    # can report a small negative yaw. That is a display artifact of an angle,
    # not an invalid rotation: the emitted transform carries the matrix itself.
    assert -360.0 < search["best_yaw_deg"] < 720.0
