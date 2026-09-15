from copy import deepcopy
import os
import struct
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

import artifact_manifest as manifests
import background_recovery as recovery
import reconstruction_comparison as comparison
import reconstruction_evidence as evidence
from test_background_recovery import observed_scene


def frames(count=24):
    return [{"file_path": f"images/photo-{index:03d}.png"} for index in range(count)]


def test_deterministic_group_holdout_covers_every_image_without_overlap():
    images = frames()
    split = comparison.partition(images)
    assert split == comparison.partition(list(reversed(images)))
    assert {name: len(members) for name, members in split.items()} == {"train": 20, "val": 2, "test": 2}
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])


def test_virtual_views_of_one_timestamp_stay_together():
    images = frames(48)
    for index, frame in enumerate(images):
        frame["timestamp_group"] = f"group-{index // 2:03d}"
    split = comparison.partition(images)
    ownership = {name: split for split, members in split.items() for name in members}
    for index in range(0, 48, 2):
        assert ownership[images[index]["file_path"]] == ownership[images[index + 1]["file_path"]]


@pytest.mark.parametrize("defect", ["overlap", "missing", "unknown", "empty", "timestamp"])
def test_invalid_supplied_split_refuses(defect):
    images = frames()
    split = comparison.partition(images)
    if defect == "overlap":
        split["train"].append(split["test"][0])
    elif defect == "missing":
        split["train"].pop()
    elif defect == "unknown":
        split["train"][0] = "outside.png"
    elif defect == "empty":
        split["train"] += split["test"]
        split["test"] = []
    else:
        for frame in images[:2]:
            frame["timestamp_group"] = "shared"
    with pytest.raises(evidence.EvidenceError):
        comparison.partition(images, {name + "_filenames": members for name, members in split.items()})


@pytest.mark.parametrize("iterations,seconds,downscale", [(0, 1200, 2), (30001, 1200, 2), (3000, 0, 2), (3000, 7201, 2), (3000, 1200, 8), (3000, 1200, True)])
def test_unbounded_training_budget_refuses(tmp_path, iterations, seconds, downscale):
    with pytest.raises(evidence.EvidenceError):
        comparison.prepare(tmp_path, iterations, seconds, downscale)


def test_frozen_receipt_detects_artifact_tamper_even_with_restored_timestamps(observed_scene, tmp_path):
    source = evidence.load(observed_scene)
    output = tmp_path / "comparison"
    output.mkdir()
    image = output / "input.png"
    image.write_bytes(b"original")
    identity = manifests.file_identity(image)
    comparison.seal(output, "receipt.json", {"source_job": str(observed_scene), "sources": source.sources,
                    "calibration": source.calibration, "artifacts": {"input.png": identity}})
    assert comparison.verify(output)["artifacts"]["input.png"] == identity
    image.write_bytes(b"modified")
    os.utime(image, ns=(identity["mtime_ns"], identity["mtime_ns"]))
    with pytest.raises(evidence.EvidenceError, match="Prepared comparison"):
        comparison.verify(output)


def test_receipt_digest_prevents_budget_or_split_edits(tmp_path):
    comparison.seal(tmp_path, "receipt.json", {"budget": {"iterations": 32}})
    payload = deepcopy(comparison.read(tmp_path))
    payload["budget"]["iterations"] = 30000
    manifests.atomic_write_json(tmp_path / "receipt.json", payload)
    with pytest.raises(evidence.EvidenceError, match="corrupt"):
        comparison.read(tmp_path)


def test_held_out_cameras_are_selected_after_lazy_undistortion(tmp_path):
    class Manager:
        def __init__(self):
            self.eval_dataset = SimpleNamespace(image_filenames=[tmp_path / "first.png", tmp_path / "second.png"], cameras=["distorted-first", "distorted-second"])

        @property
        def cached_eval(self):
            self.eval_dataset.cameras[:] = ["undistorted-first", "undistorted-second"]
            return ["first-batch", "second-batch"]

    result = list(comparison.held_out_frames(Manager(), tmp_path, ["first.png", "second.png"]))
    assert result == [("first.png", ["undistorted-first"], "first-batch"), ("second.png", ["undistorted-second"], "second-batch")]


@pytest.mark.parametrize("defect", ["membership", "empty", "cache"])
def test_invalid_held_out_batches_refuse(tmp_path, defect):
    manager = SimpleNamespace(eval_dataset=SimpleNamespace(image_filenames=[tmp_path / "first.png"], cameras=["first-camera"]), cached_eval=["first-batch"])
    if defect == "membership":
        manager.eval_dataset.image_filenames = [tmp_path / "wrong.png"]
    elif defect == "empty":
        manager.eval_dataset.image_filenames = []
    else:
        manager.cached_eval = []
    with pytest.raises(evidence.EvidenceError):
        list(comparison.held_out_frames(manager, tmp_path, ["first.png"]))


@pytest.fixture
def comparison_source(observed_scene, tmp_path, monkeypatch):
    job = observed_scene
    monkeypatch.setattr(comparison, "ROOT", tmp_path / "studies")
    monkeypatch.setattr(recovery, "WORKER_LOCK", tmp_path / "comparison.lock")
    sparse = job / "processed/sparse/0"
    document = manifests.read_json(job / "processed/transforms.json")
    original_frames = deepcopy(document["frames"])
    original_images = evidence.read_images(sparse / "images.bin")
    for frame in original_frames:
        changed = deepcopy(frame)
        changed["colmap_im_id"] += 6
        changed["file_path"] = "images/copy-" + frame["file_path"].split("/")[-1]
        (job / "processed" / changed["file_path"]).write_bytes((job / "processed" / frame["file_path"]).read_bytes())
        document["frames"].append(changed)
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 12))
        for frame in document["frames"]:
            image_id = frame["colmap_im_id"]
            original = original_images[(image_id - 1) % 6 + 1]
            handle.write(struct.pack("<idddddddi", image_id, 0., 1., 0., 0., *original["raw_world_to_camera"][:3, 3], 1))
            handle.write(frame["file_path"].split("/")[-1].encode() + b"\0")
            handle.write(struct.pack("<Q", len(original["observations"])))
            handle.write(original["observations"].tobytes())
    records = evidence.read_points(sparse / "points3D.bin")
    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(records)))
        for record in records:
            copied = record["track"].copy()
            copied[:, 0] += 6
            tracks = np.concatenate([record["track"], copied])
            handle.write(struct.pack("<QdddBBBdQ", record["id"], *record["xyz"], 255, 0, 255, record["error"], len(tracks)))
            handle.write(tracks.astype("<i4").tobytes())
    manifests.atomic_write_json(job / "processed/transforms.json", document)
    return job


def test_preparation_uses_train_only_colors_keeps_masks_and_never_warm_starts(comparison_source):
    job = comparison_source
    document = manifests.read_json(job / "processed/transforms.json")
    Image.new("L", (64, 64), 255).save(job / "processed/mask.png")
    for frame in document["frames"]:
        frame["mask_path"] = "mask.png"
    manifests.atomic_write_json(job / "processed/transforms.json", document)
    before, receipt = comparison.prepare(job, 32, 60, 2)
    comparison.verify(before)
    assert receipt["warm_start"] is False
    assert receipt["evaluation_split"] == "val"
    assert receipt["masking"] == "registered masks retained"
    assert receipt["seed_points"] >= 100
    prepared = manifests.read_json(before / "dataset/transforms.json")
    assert prepared["frames"][0]["w"] == prepared["frames"][0]["h"] == 32
    assert prepared["frames"][0]["fl_x"] == 30
    assert (before / "dataset" / prepared["frames"][0]["mask_path"]).is_file()
    with np.load(before / "dataset/seed-lineage.npz", allow_pickle=False) as lineage:
        assert np.all(lineage["training_color_observations"] >= 3)
    Image.new("RGB", (64, 64), (255, 0, 0)).save(job / "processed" / receipt["splits"]["test"][0])
    after, _ = comparison.prepare(job, 32, 60, 2)
    assert (before / "dataset/train-seeds.ply").read_bytes() == (after / "dataset/train-seeds.ply").read_bytes()
    with pytest.raises(evidence.EvidenceError, match="changed"):
        comparison.verify(before)


@pytest.fixture
def compared_runs(tmp_path, monkeypatch):
    receipt = {"sha256": "a" * 64, "budget": {"iterations_per_arm": 1000}, "prepared_splits": {"test": ["images/000001.png"]}, "evaluation_split": "test"}
    monkeypatch.setattr(comparison, "verify", lambda _output: receipt)
    for name, score in (("splatfacto", 20), ("dn-splatter", 21)):
        output = tmp_path / name
        (output / "renders").mkdir(parents=True)
        reference = output / "renders/000001-reference.png"
        reference.write_bytes(b"identical synthetic reference fixture")
        comparison.seal(output, "run.json", {"arm": name, "status": "completed", "budget": receipt["budget"], "prepared_receipt_sha256": receipt["sha256"],
            "iterations": 1000, "initial_positions_sha256": "b" * 64, "initial_gaussians": 100,
            "implementation_sha256": "c" * 64, "packages": {"fixture": "1"}, "implementation_sources": {},
            "evaluation": {"split": "test", "views": [{"image": "images/000001.png", "psnr_db": score, "ssim": .8, "covered_fraction_alpha_0_5": .9}]},
            "gaussians": 200, "training_seconds": 10, "total_seconds": 12, "max_cuda_allocated_bytes": 1000, "max_rss_bytes": 2000,
            "artifacts": {"renders/000001-reference.png": manifests.file_identity(reference)}})
    return tmp_path, receipt


def test_comparison_reports_deltas_without_promoting_a_method(compared_runs):
    output, _ = compared_runs
    result = comparison.summarize(output, ["splatfacto", "dn-splatter"])
    assert result["candidate_minus_baseline"]["dn-splatter"]["psnr_db"] == 1
    assert result["status"] == "needs-review" and result["promoted"] is False
    assert result["geometry_error"] is None
    assert comparison.summarize(output, ["splatfacto", "dn-splatter"]) == result


@pytest.mark.parametrize("defect", ["budget", "initialization", "harness", "failed", "empty-eval", "nonfinite", "artifact"])
def test_invalid_pair_cannot_be_reported_as_a_success(compared_runs, defect):
    output, _ = compared_runs
    record = comparison.read(output / "dn-splatter", "run.json")
    if defect == "budget":
        record["iterations"] = 999
    elif defect == "initialization":
        record["initial_positions_sha256"] = "0" * 64
    elif defect == "harness":
        record["implementation_sha256"] = "0" * 64
    elif defect == "failed":
        record["status"] = "failed"
    elif defect == "empty-eval":
        record["evaluation"]["views"] = []
    elif defect == "nonfinite":
        record["evaluation"]["views"][0]["psnr_db"] = float("nan")
    else:
        (output / "dn-splatter/renders/000001-reference.png").write_bytes(b"different reference")
    if defect == "nonfinite":
        with pytest.raises(ValueError):
            comparison.seal(output / "dn-splatter", "run.json", record)
        return
    comparison.seal(output / "dn-splatter", "run.json", record)
    with pytest.raises(evidence.EvidenceError):
        comparison.summarize(output, ["splatfacto", "dn-splatter"])
