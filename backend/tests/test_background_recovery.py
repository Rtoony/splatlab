from io import BytesIO
import json
import struct

import numpy as np
from PIL import Image
import pytest

import artifact_manifest as manifests
import background_recovery as recovery
import glb_check
import reconstruction_evidence as evidence
import scene_revisions as scenes
from mesh import provenance
from test_scene_revisions import scene, placement


@pytest.fixture
def observed_scene(tmp_path):
    job = tmp_path / "splat_f00123"
    sparse = job / "processed/sparse/0"
    sparse.mkdir(parents=True)
    images = job / "processed/images"
    images.mkdir()
    run = job / "processed/splatfacto/run"
    run.mkdir(parents=True)
    manifests.atomic_write_json(job / "meta.json", {"meters_per_unit": 1, "scale_generation": 1})
    manifests.atomic_write_json(run / "dataparser_transforms.json", {"transform": np.eye(4)[:3].tolist(), "scale": 1})
    points = np.array([[axis_x, axis_y, 0.] for axis_x in np.linspace(-1., 1., 35) for axis_y in np.linspace(-1., 1., 35)])
    positions = [-.6, -.3, 0., .3, .6, .8]
    frames = []
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<QiiQQdddd", 1, 1, 1, 64, 64, 60., 60., 32., 32.))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(positions)))
        for image_id, axis_x in enumerate(positions, 1):
            name = f"photo-{image_id}.png"
            Image.new("RGB", (64, 64), (70, 120, 160)).save(images / name)
            world_to_camera = np.diag([1., -1., -1., 1.])
            world_to_camera[:3, 3] = [-axis_x, 0, 3]
            pose = np.linalg.inv(world_to_camera) @ evidence.GL_TO_CV
            frames.append({"colmap_im_id": image_id, "file_path": "images/" + name, "transform_matrix": pose.tolist()})
            handle.write(struct.pack("<idddddddi", image_id, 0., 1., 0., 0., -axis_x, 0., 3., 1))
            handle.write(name.encode() + b"\0")
            handle.write(struct.pack("<Q", len(points)))
            for point_id, point in enumerate(points, 1):
                horizontal = (point[0] - axis_x) / 3 * 60 + 32
                vertical = -point[1] / 3 * 60 + 32
                handle.write(struct.pack("<ddq", horizontal, vertical, point_id))
    with (sparse / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(points)))
        for point_id, point in enumerate(points, 1):
            handle.write(struct.pack("<QdddBBBdQ", point_id, *point, 70, 120, 160, .1, len(positions)))
            for image_id in range(1, len(positions) + 1):
                handle.write(struct.pack("<ii", image_id, point_id - 1))
    manifests.atomic_write_json(job / "processed/transforms.json", {"w": 64, "h": 64, "fl_x": 60., "fl_y": 60., "cx": 32., "cy": 32., "applied_transform": np.eye(4)[:3].tolist(), "frames": frames})
    return job


def test_real_binary_observation_chain_and_reprojection(observed_scene):
    source = evidence.load(observed_scene)
    assert source.report["median_reprojection_px"] < 1e-10
    assert source.report["cameras"] == 6
    assert np.max(np.abs(source.points[:, 1])) == 0
    assert source.cameras[1]["center"][1] == 3


def test_recorded_resize_is_verified_not_guessed(observed_scene):
    path = observed_scene / "processed/transforms.json"
    document = manifests.read_json(path)
    for key in ("w", "h", "fl_x", "fl_y", "cx", "cy"):
        document[key] /= 2
    document["w"], document["h"] = int(document["w"]), int(document["h"])
    manifests.atomic_write_json(path, document)
    source = evidence.load(observed_scene)
    assert source.report["median_reprojection_px"] < 1e-10
    assert source.cameras[1]["width"] == 32
    document["fl_x"] += 1
    manifests.atomic_write_json(path, document)
    with pytest.raises(evidence.EvidenceError, match="verified resize"):
        evidence.load(observed_scene)


def test_mismatched_saved_pose_is_refused(observed_scene):
    path = observed_scene / "processed/transforms.json"
    document = manifests.read_json(path)
    document["frames"][0]["transform_matrix"][0][3] += .1
    manifests.atomic_write_json(path, document)
    with pytest.raises(evidence.EvidenceError, match="disagrees with COLMAP"):
        evidence.load(observed_scene)


def test_recovery_omits_unobserved_cells_and_keeps_per_pixel_lineage(observed_scene, tmp_path):
    source = evidence.load(observed_scene)
    bounds = {"min": [-.2, 0., -.2], "max": [.2, .5, .2]}
    mesh, texture, diagnostic, report, pixels = recovery.recover(source, [bounds], size=64)
    assert 0 < report["supported_fraction"] < 1
    assert report["plane"]["rms_m"] < 1e-10
    assert 0 < report["texture_support_points"] < report["support_points"]
    assert report["recipe"]["version"] == 2
    assert set(report["source_image_ids"]).isdisjoint(report["appearance_check_image_ids"])
    assert all(check["mean_rgb_absolute_error"] < 1e-6 for check in report["appearance_checks"])
    atlas = np.asarray(Image.open(BytesIO(texture)))
    assert atlas[32, 32, 3] == 0
    assert (atlas[~pixels["accepted"], :3] == 0).all()
    assert (pixels["source_image_ids"][:, ~pixels["accepted"].reshape(-1)] == -1).all()
    accepted_sources = (pixels["source_image_ids"] >= 0).sum(axis=0)
    assert (accepted_sources[pixels["accepted"].reshape(-1)] >= 2).all()
    assert len(pixels["support_point_ids"]) == report["support_points"]
    assert pixels["support_points_world"].shape == (report["support_points"], 3)
    path = tmp_path / "supported.glb"
    path.write_bytes(mesh)
    glb_check.validate_glb(path)
    assert glb_check.position_bounds(path)["identity_transforms"]
    assert report["supported_fraction"] == pixels["accepted"].mean()
    assert (diagnostic[~pixels["accepted"]] == [210, 40, 160]).all()
    json_size = struct.unpack_from("<I", mesh, 12)[0]
    document = json.loads(mesh[20:20 + json_size])
    assert document["asset"]["extras"][recovery.GLTF_EXTRAS_KEY] == recovery.GENERATIVE_TAG
    assert all(item["extras"][recovery.GLTF_EXTRAS_KEY] == recovery.GENERATIVE_TAG for item in document["nodes"] + document["meshes"])
    with pytest.raises(provenance.GenerativeInputRefused, match="GLB tag"):
        provenance.assert_not_generative(path, "survey")


def test_duplicate_viewpoints_do_not_establish_multiview_support():
    points = np.array([[0., 0., 0.]])
    colors = np.ones((3, 1, 3)) * .4
    valid = np.ones((3, 1), dtype=bool)
    _, accepted, _, _ = recovery.consensus(colors, valid, np.array([[0., 3., 0.]] * 3), points)
    assert not accepted.any()
    _, accepted, _, _ = recovery.consensus(colors, valid, np.array([[-1., 3., 0.], [0., 3., 0.], [1., 3., 0.]]), points)
    assert accepted.all()
    _, accepted, _, _ = recovery.consensus(colors, valid, np.array([[-1., 3., 0.], [0., 3., 0.], [1., 3., 0.]]), points, groups=["same-capture"] * 3)
    assert not accepted.any()


def test_object_bounds_occlude_pixels_but_not_unrelated_rays():
    box = {"min": [-.2, 0., -.2], "max": [.2, 1., .2]}
    result = recovery.ray_box_occluded(np.array([0., 3., 0.]), np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]]), box)
    assert result.tolist() == [True, False, False]


def test_observation_mask_is_respected(observed_scene):
    path = observed_scene / "processed/transforms.json"
    document = manifests.read_json(path)
    for frame in document["frames"]:
        frame["mask_path"] = "mask.png"
    manifests.atomic_write_json(path, document)
    Image.new("L", (64, 64), 0).save(observed_scene / "processed/mask.png")
    with pytest.raises(evidence.EvidenceError, match="No multi-view-supported cells"):
        recovery.recover(evidence.load(observed_scene), [{"min": [-.2, 0., -.2], "max": [.2, .5, .2]}], size=64)


def test_truncated_binary_is_refused(observed_scene):
    path = observed_scene / "processed/sparse/0/cameras.bin"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(evidence.EvidenceError, match="Truncated"):
        evidence.load(observed_scene)


def test_evidence_allows_contained_mask_links_but_not_external_files(observed_scene, tmp_path):
    link = observed_scene / "processed/mask-alias.png"
    link.symlink_to("images/photo-1.png")
    sources = {}
    assert evidence.input_path(observed_scene, "processed/mask-alias.png", sources).name == "photo-1.png"
    evidence.verify_sources(observed_scene, sources, evidence.scale_revision(observed_scene))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside the registered job")
    escaping = observed_scene / "processed/escaping.png"
    escaping.symlink_to(outside)
    with pytest.raises(evidence.EvidenceError, match="escaping"):
        evidence.input_path(observed_scene, "processed/escaping.png", {})


def test_cpu_recovery_admission_has_only_one_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "WORKER_LOCK", tmp_path / "worker.lock")
    with recovery.worker_slot():
        with pytest.raises(evidence.EvidenceError, match="already running"):
            with recovery.worker_slot():
                pytest.fail("Concurrent recovery must not be admitted")


def test_unverifiable_glb_metadata_is_not_accepted_as_survey_input(tmp_path):
    path = tmp_path / "unverifiable.glb"
    path.write_bytes(struct.pack("<4sIII4s", b"glTF", 2, 20, 64 * 1024 ** 2, b"JSON"))
    with pytest.raises(provenance.GenerativeInputRefused, match="cannot verify GLB provenance"):
        provenance.assert_not_generative(path, "survey")


def test_recovery_proposal_pins_evidence_and_refuses_changed_source(scene):
    job, pointer = scene
    candidate = job / "_blender/exports" / placement()["blender_export"]
    provenance.assert_not_generative(candidate, "survey")
    photo = job / "processed/images/photo.png"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"captured photo")
    identifier = "recovery_" + "a" * 24
    receipt = {"recovery_id": identifier, "base": pointer, "render_vr_only": True, "selected_slug": "chair", "sources": {"processed/images/photo.png": manifests.file_identity(photo)},
               "calibration": evidence.scale_revision(job), "report": {"supported_fraction": .5}, "artifacts": {"candidate.glb": scenes.store_file(job, candidate)}}
    receipt["sha256"] = recovery.hashlib.sha256(scenes.canonical_bytes(receipt)).hexdigest()
    manifests.atomic_write_json(scenes.root(job) / "recoveries" / identifier / "receipt.json", receipt)
    proposal = scenes.propose(job, {"kind": "place", "slug": "background", "recovery_id": identifier}, "Inspect real-photo cells", 0)
    revision = scenes.read_revision(job, proposal["preview_revision"])
    assert revision["state"]["viewer"]["elements"][0]["geometry_source"] == "sfm-plane-fit"
    assert f"_studio/{identifier}.json" in revision["artifacts"]
    photo.write_bytes(b"changed photo")
    with pytest.raises(scenes.SceneRevisionError, match="evidence changed"):
        scenes.activate(job, proposal["proposal_id"], 0)
    assert scenes.active(job) == pointer
