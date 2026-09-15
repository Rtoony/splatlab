import numpy as np
import pytest

import rig_diagnostics as rig


def fixed_rig():
    poses = {}
    turn = rig.rotation([0, 0, 1, 0])
    for index in range(4):
        centre = np.array([index, 0, 0], dtype=float)
        for lens, matrix in [(0, np.eye(3)), (1, turn)]:
            sensor_centre = centre + np.array([.03 * lens, 0, 0])
            poses[f"lens-{lens}/frame-{index:06d}.jpg"] = {"rotation": matrix, "translation": -matrix @ sensor_centre,
                                                        "centre": sensor_centre}
    return poses


def test_fixed_rig_reports_constant_rotation_baseline_and_unknown_metric_scale():
    poses = fixed_rig()
    report = rig.summarize(poses, list(poses))
    assert report["paired_timestamps"] == 4
    assert report["rotation_error_deg"]["max"] == 0
    assert report["baseline_arbitrary_units"]["median"] == pytest.approx(.03)
    assert report["median_baseline_over_trajectory"] == pytest.approx(.01)
    assert report["calibration_accepted"] is False
    assert report["metric_scale"] == "unknown"


def test_drifting_sensor_and_leaked_holdout_are_detected():
    poses = fixed_rig()
    poses["lens-1/frame-000003.jpg"]["translation"] += np.array([1, 0, 0])
    assert rig.summarize(poses, list(poses))["max_translation_deviation_arbitrary_units"] > .9
    with pytest.raises(ValueError, match="frozen training"):
        rig.summarize(poses, list(poses)[:-1])
    del poses["lens-1/frame-000003.jpg"]
    with pytest.raises(ValueError, match="complete"):
        rig.summarize(poses, list(poses))


def test_pose_parser_preserves_world_to_camera_convention(tmp_path):
    source = tmp_path / "images.txt"
    source.write_text("# header\n1 1 0 0 0 -1 -2 -3 1 lens-0/frame-000001.jpg\n\n")
    pose = rig.read_poses(source)["lens-0/frame-000001.jpg"]
    np.testing.assert_array_equal(pose["centre"], [1, 2, 3])
    source.write_text("1 0 0 0 0 -1 -2 -3 1 name.jpg\n\n")
    with pytest.raises(ValueError, match="unit quaternion"):
        rig.read_poses(source)


def test_rotation_sign_and_quarter_turn():
    np.testing.assert_allclose(rig.rotation([1, 0, 0, 0]), rig.rotation([-1, 0, 0, 0]))
    angle = np.sqrt(.5)
    assert rig.angular_distance(np.eye(3), rig.rotation([angle, 0, angle, 0])) == pytest.approx(90)
