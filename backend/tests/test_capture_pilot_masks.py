import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fisheye_training import sample_raw_mask

spec = importlib.util.spec_from_file_location("capture_pilot_masks", Path(__file__).resolve().parents[2] / "tools/capture-pilot-masks.py")
masks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(masks)


def test_geometric_support_and_semantic_dilation_exclude_rim_nadir_and_person():
    excluded = np.zeros((512, 512), dtype=bool)
    excluded[256, 256] = True
    keep = masks.combine_masks(excluded, 512)
    assert not keep[0, 0] and not keep[450, 256]
    assert not keep[253:260, 253:260].any()
    assert keep[200, 200]


def test_raw_mask_rectification_uses_source_pixel_centers_and_never_enables_invalid_rays():
    source = np.array([[True, False], [False, True]])
    grid = np.array([[[0., 0.], [1., 0.]], [[0., 1.], [1., 1.]]])
    np.testing.assert_array_equal(sample_raw_mask(source, grid, np.ones((2, 2), dtype=bool)), source)
    assert not sample_raw_mask(source, grid, np.zeros((2, 2), dtype=bool)).any()
    with pytest.raises(ValueError):
        sample_raw_mask(source.astype(np.uint8), grid, np.ones((2, 2), dtype=bool))
