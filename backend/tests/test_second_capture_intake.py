import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest

import artifact_manifest as manifests

spec = importlib.util.spec_from_file_location("second_capture_intake", Path(__file__).resolve().parents[2] / "tools/verify-dual-fisheye-pilot.py")
intake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intake)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "second.insv"
    source.write_bytes(b"synthetic fixture media, not reconstruction evidence")
    root, previous = tmp_path / "pilot", tmp_path / "previous"
    root.mkdir()
    previous.mkdir()
    groups, views = [], []
    for ordinal, split in enumerate(["train", "train", "test", "val", "train"]):
        group = {"group_id": f"frame-{ordinal * 60:06d}", "source_decoded_frame_index": ordinal * 60, "split": split}
        groups.append(group)
        for lens in range(2):
            name = f"lens-{lens}/{group['group_id']}.jpg"
            image = root / "images" / name
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (512, 512), "white").save(image)
            views.append({**group, "image": name, "lens_id": f"lens-{lens}", "sha256": manifests.sha256_file(image),
                          "pts": ordinal * 60060, "time_base": "1/60000", "time_s": ordinal * 1.001})
    document = {"schema": "dev.splatlab.dual-fisheye-pilot/v1", "status": "decoded-needs-camera-and-mask-review",
                "clock": {"paired_container_pts_match": True}, "capture_id": "second", "source": {"path": str(source), **manifests.file_identity(source)},
                "groups": groups, "views": views, "output_width": 512, "streams": []}
    (root / "pilot.json").write_text(json.dumps(document))
    (previous / "pilot.json").write_text(json.dumps({"capture_id": "first", "source": {"sha256": "a" * 64}}))
    for split in ["train", "val", "test"]:
        (root / f"{split}-images.txt").write_text("".join(view["image"] + "\n" for view in views if view["split"] == split))
    return root, previous, document


def test_second_clip_intake_does_not_claim_reconstruction(prepared):
    root, previous, _ = prepared
    result = intake.inspect(root, previous)
    assert result["splits"] == {"train": 6, "val": 2, "test": 2}
    assert result["paired_container_pts_match"] is True
    assert result["reconstruction_validated"] is False
    assert result["registration"] is None
    with pytest.raises(ValueError, match="different video"):
        intake.inspect(root, root)


@pytest.mark.parametrize("fault", ["split", "timestamp", "pixels", "split-file", "source"])
def test_second_clip_refuses_changed_lineage_and_bytes(prepared, fault):
    root, previous, document = prepared
    if fault == "split":
        document["views"][1]["split"] = "test"
    elif fault == "timestamp":
        document["views"][1]["pts"] += 1
    elif fault == "pixels":
        (root / "images" / document["views"][0]["image"]).write_bytes(b"changed")
    elif fault == "split-file":
        (root / "train-images.txt").write_text("changed\n")
    else:
        Path(document["source"]["path"]).write_bytes(b"changed source")
    (root / "pilot.json").write_text(json.dumps(document))
    with pytest.raises(ValueError):
        intake.inspect(root, previous)
