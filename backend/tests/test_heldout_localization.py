import importlib.util
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

import heldout_localization as localization


def pair(first, second, matches):
    values = np.array(matches, dtype=np.uint32).reshape(-1, 2)
    return (first * localization.PAIR_BASE + second, len(values), 2, values.tobytes())


def test_pair_orientation_and_heldout_only_pairs_are_handled():
    training = {1: {"point_ids": np.array([70, 71])}, 8: {"point_ids": np.array([70, 72])}}
    pairs = [pair(1, 5, [[0, 1], [1, 0]]), pair(5, 8, [[1, 0], [2, 1]]), pair(5, 9, [[3, 100]])]
    assert localization.select_correspondences(5, 4, pairs, training) == [(0, 71, 1), (1, 70, 2), (2, 72, 1)]


def test_ambiguous_matches_and_duplicate_3d_points_are_not_extra_evidence():
    training = {1: {"point_ids": np.array([70, 71, -1])}, 2: {"point_ids": np.array([70, 71])}}
    pairs = [pair(1, 5, [[0, 0], [0, 0], [0, 1], [0, 2], [2, 3]]), pair(2, 5, [[1, 0], [0, 2]])]
    assert localization.select_correspondences(5, 4, pairs, training) == [(2, 70, 2)]


@pytest.mark.parametrize("matches", [[[2, 0]], [[0, 2]]])
def test_out_of_bounds_correspondences_fail(matches):
    with pytest.raises(ValueError, match="outside"):
        localization.select_correspondences(5, 2, [pair(1, 5, matches)], {1: {"point_ids": np.array([70])}})


def test_tracks_preserve_untriangulated_feature_indices(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# heading\n1 1 0 0 0 0 0 0 1 lens-0/frame-000000.jpg\n0.5 0.5 -1 5.5 7.5 70\n")
    tracks = localization.read_tracks(path)
    np.testing.assert_array_equal(tracks["lens-0/frame-000000.jpg"]["point_ids"], [-1, 70])
    np.testing.assert_array_equal(tracks["lens-0/frame-000000.jpg"]["pixels"], [[.5, .5], [5.5, 7.5]])
    path.write_text("1 1 0 0 0 0 0 0 1 lens-0/frame-000000.jpg\n0.5 0.5\n")
    with pytest.raises(ValueError, match="Malformed"):
        localization.read_tracks(path)


def test_training_database_requires_exact_feature_identity():
    database = sqlite3.connect(":memory:")
    database.execute("CREATE TABLE images(image_id INTEGER,name TEXT)")
    database.execute("CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB)")
    pixels = np.array([[.5, .5], [5.5, 7.5]], dtype=np.float32)
    database.execute("INSERT INTO images VALUES(1,'training.jpg')")
    database.execute("INSERT INTO keypoints VALUES(1,2,2,?)", (pixels.tobytes(),))
    tracks = {"training.jpg": {"pixels": pixels, "point_ids": np.array([-1, 70])}}
    mapped = localization.validate_training_database(database, tracks)
    assert list(mapped) == [1]
    tracks["training.jpg"]["pixels"] = pixels + .5
    with pytest.raises(ValueError, match="indices"):
        localization.validate_training_database(database, tracks)
    database.execute("INSERT INTO images VALUES(2,'heldout.jpg')")
    with pytest.raises(ValueError, match="exactly"):
        localization.validate_training_database(database, tracks)
    database.close()


def test_localization_runner_refuses_direct_launch_before_loading_colmap(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/localize-dual-fisheye-holdout.py"
    spec = importlib.util.spec_from_file_location("localize_fisheye_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = []

    def denied(command, **kwargs):
        observed.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", denied)
    with pytest.raises(ValueError, match="compute-gate"):
        module.run(tmp_path / "pilot", tmp_path / "sfm", tmp_path / "output")
    assert observed == [[str(module.GATE), "--is-contained"]]
    assert not (tmp_path / "output").exists()
