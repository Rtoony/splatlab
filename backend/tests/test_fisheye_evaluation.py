import pytest

from fisheye_evaluation import validation_scope


@pytest.mark.parametrize("timestamps", [4, 12])
def test_capture_scope_counts_paired_timestamps_not_virtual_crops(timestamps):
    records = [{"image": f"val/lens-{lens}-frame-{index * 30:06d}-{direction}.png"}
               for index in range(timestamps) for lens in (0, 1) for direction in ("left", "centre", "right")]
    assert validation_scope(records) == f"{timestamps * 6} validation crops share {timestamps} held-out timestamps and overlapping directions."


def test_unknown_image_identity_does_not_invent_independence():
    assert "grouping is not available" in validation_scope([{"image": "photo.jpg"}])
