import json

import pytest
from PIL import Image

import artifact_manifest as manifests
import capture_records as captures
import route_builder as routes
import route_reconstruction as reconstruction


@pytest.fixture
def ready_route(tmp_path, monkeypatch):
    monkeypatch.setattr(captures, "DATA_ROOT", tmp_path)
    route_id = "route_" + "a" * 16
    directory = routes.route_dir(route_id)
    directory.mkdir(parents=True)
    points = []
    for index in range(20):
        path = directory / f"pano-{index:05d}.jpg"
        Image.new("RGB", (100, 50), (100, 150, 200)).save(path)
        points.append({"index": index, "time_s": index * 5, "status": "ready", "sha256": manifests.sha256_file(path), "quality": {"sharpness": 50 + index}})
    routes.save_route({"schema": routes.ROUTE_SCHEMA, "route_id": route_id, "source_sha256": "source", "clock": {}, "status": "completed", "spec": {"start_s": 0, "interval_s": 5}, "points": points})
    return route_id


def test_sections_overlap_without_splitting_timestamp_groups(ready_route):
    plan = reconstruction.prepare(ready_route, reconstruction.ReconstructionSpec())
    assert len(plan["sections"]) == 3
    assert set(plan["sections"][0]["point_indices"]) & set(plan["sections"][1]["point_indices"])
    transforms = {"frames": [{"file_path": f"images/pano_camera{camera}/{group['timestamp_group']}.jpg"} for group in plan["groups"] for camera in range(4)]}
    split = reconstruction.apply_group_split(transforms, plan)
    filenames = [set(split[f"{name}_filenames"]) for name in ("train", "val", "test")]
    assert sum(map(len, filenames)) == 80
    assert not (filenames[0] & filenames[1] or filenames[0] & filenames[2] or filenames[1] & filenames[2])
    for group in plan["groups"]:
        assert sum(group["timestamp_group"] in filename for filename in split[f"{group['split']}_filenames"]) == 4
    assert len(plan["semantic_point_indices"]) < len(plan["groups"])


def test_mask_excludes_nadir_without_modifying_observation(ready_route):
    directory = routes.route_dir(ready_route)
    before = (directory / "pano-00000.jpg").read_bytes()
    plan = reconstruction.prepare(ready_route, reconstruction.ReconstructionSpec(nadir_exclusion_deg=60))
    mask = Image.open(directory / "reconstruction" / plan["fingerprint"][:16] / plan["groups"][0]["mask"])
    assert mask.getpixel((50, 25)) == 255
    assert mask.getpixel((50, 49)) == 0
    assert (directory / "pano-00000.jpg").read_bytes() == before


def test_unknown_crop_lineage_refuses_eval_leakage(ready_route):
    plan = reconstruction.prepare(ready_route, reconstruction.ReconstructionSpec())
    with pytest.raises(ValueError, match="timestamp lineage"):
        reconstruction.apply_group_split({"frames": [{"file_path": "images/unidentified.jpg"}]}, plan)


def test_modified_source_and_unusable_selection_refuse(ready_route):
    with pytest.raises(ValueError, match="three usable"):
        reconstruction.prepare(ready_route, reconstruction.ReconstructionSpec(min_sharpness=1000))
    (routes.route_dir(ready_route) / "pano-00000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        reconstruction.prepare(ready_route, reconstruction.ReconstructionSpec())
