import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

import artifact_manifest as manifests
from test_dual_fisheye_pilot import capture, fake_decode, pilot


TOOL = Path(__file__).resolve().parents[2] / "tools/probe-dual-fisheye-sfm.py"
MODULE_SPEC = importlib.util.spec_from_file_location("dual_fisheye_sfm", TOOL)
sfm = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(sfm)


@pytest.fixture
def prepared(capture, tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.subprocess, "run", fake_decode)
    root = tmp_path / "pilot"
    document = pilot.extract(pilot.make_plan(capture, 0, 30, 5, 512), root, 30)
    masks = tmp_path / "masks"
    for view in document["views"]:
        if view["split"] == "train":
            path = masks / (view["image"] + ".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (512, 512), 255).save(path)
    return root, masks, document


def test_mapping_reads_only_training_names_but_verifies_all_lineage(prepared):
    root, masks, document = prepared
    result = sfm.verified_inputs(root, masks)
    assert len(result["train_names"]) == len(result["masks"]) == 6
    assert {name for name in result["train_names"]} == {
        view["image"] for view in document["views"] if view["split"] == "train"}


@pytest.mark.parametrize("change", ["failed", "unsynced", "split", "path", "duplicate", "pts", "missing_lens", "frame_index"])
def test_input_tampering_refuses_mapping(prepared, change):
    root, masks, document = prepared
    if change == "failed":
        document["status"] = "failed-not-for-reconstruction"
    elif change == "unsynced":
        document["clock"]["paired_container_pts_match"] = False
    elif change == "split":
        document["views"][0]["split"] = "test"
    elif change == "path":
        document["views"][0]["image"] = "../outside.jpg"
    elif change == "duplicate":
        document["views"][1]["image"] = document["views"][0]["image"]
    elif change == "pts":
        document["views"][0]["pts"] += 1
    elif change == "missing_lens":
        document["views"].pop()
    else:
        document["views"][0]["source_decoded_frame_index"] += 1
    manifests.atomic_write_json(root / "pilot.json", document)
    with pytest.raises(ValueError):
        sfm.verified_inputs(root, masks)


@pytest.mark.parametrize("kind", ["missing", "black", "wrong_size", "color"])
def test_training_mask_is_required_and_validated(prepared, kind):
    root, masks, document = prepared
    path = masks / (document["views"][0]["image"] + ".png")
    if kind == "missing":
        path.unlink()
    else:
        Image.new("RGB" if kind == "color" else "L", (20, 20) if kind == "wrong_size" else (512, 512),
                  0 if kind == "black" else 255).save(path)
    with pytest.raises((ValueError, FileNotFoundError)):
        sfm.verified_inputs(root, masks)


def test_modified_heldout_image_is_not_silently_accepted(prepared):
    root, masks, document = prepared
    view = next(view for view in document["views"] if view["split"] == "test")
    (root / "images" / view["image"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        sfm.verified_inputs(root, masks)


def fake_model(names):
    return SimpleNamespace(images={index: SimpleNamespace(name=name) for index, name in enumerate(names)},
        reg_image_ids=lambda: list(range(len(names))), num_points3D=lambda: 10,
        compute_mean_track_length=lambda: 3.0, compute_mean_reprojection_error=lambda: 0.8, cameras={})


def test_model_summary_distinguishes_components_from_shared_physical_lens_pairs(prepared):
    _, _, document = prepared
    training = [view for view in document["views"] if view["split"] == "train"]
    first_lens = [view["image"] for view in training if view["lens_id"] == "lens-0"]
    separate = sfm.model_summary(fake_model(first_lens), document["views"])
    assert separate["registered_by_lens"] == {"lens-0": 3}
    assert separate["same_timestamp_both_lenses_registered"] == 0
    combined = sfm.model_summary(fake_model([view["image"] for view in training]), document["views"])
    assert combined["same_timestamp_both_lenses_registered"] == 3
    assert combined["registered_images"] == 6


def test_model_summary_refuses_heldout_leakage(prepared):
    _, _, document = prepared
    heldout = next(view["image"] for view in document["views"] if view["split"] == "test")
    with pytest.raises(ValueError, match="Held-out"):
        sfm.model_summary(fake_model([heldout]), document["views"])
