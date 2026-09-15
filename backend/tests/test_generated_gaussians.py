from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import generated_gaussians as gaussians
import generated_objects as objects
import reconstruction_evidence as evidence
import scene_revisions as scenes
import scene_studio_route
import selection_reviews as reviews
from test_generated_objects import generated_scene, support_scene, observed_scene, selected_scene, seal_generated


def sample_rows():
    rows = np.zeros((3, 17), dtype=np.float32)
    rows[:, :3] = [[1, 2, 3], [-3, 2, 1], [2, -1, 4]]
    rows[:, 6:10] = [[.2, -.4, .3, 1], [.3, -.2, .4, -3], [2, 0, 1, 4]]
    rows[:, 10:13] = np.log([[.02, .1, .3], [.2, .4, .3], [.1, .4, .2]])
    rows[:, 13:17] = [[1, 0, 0, 0], [.5, .5, .5, .5], [1, 2, 3, 4]]
    return rows


def test_sdk_row_vector_conversion_is_correctly_transposed():
    row_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_array_equal(sample_rows()[:, :3] @ row_matrix, sample_rows()[:, :3] @ gaussians.GAUSSIAN_TO_MESH[:3, :3].T)
    assert np.linalg.det(gaussians.GAUSSIAN_TO_MESH[:3, :3]) == 1


@pytest.mark.parametrize("rotation", [np.eye(3), gaussians.GAUSSIAN_TO_MESH[:3, :3], np.diag([-1, -1, 1])])
def test_similarity_preserves_all_rows_appearance_and_rotates_full_covariance(rotation):
    rows = sample_rows()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * 2.3
    matrix[:3, 3] = [4, 6, -2]
    placed = gaussians.transform_rows(rows, matrix)
    np.testing.assert_allclose(placed[:, :3], rows[:, :3] @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-6)
    np.testing.assert_array_equal(placed[:, 3:10], rows[:, 3:10])
    for original, transformed in zip(rows, placed):
        original_rotation = evidence.rotation(original[13:17] / np.linalg.norm(original[13:17]))
        transformed_rotation = evidence.rotation(transformed[13:17] / np.linalg.norm(transformed[13:17]))
        original_covariance = original_rotation @ np.diag(np.exp(2 * original[10:13])) @ original_rotation.T
        actual_covariance = transformed_rotation @ np.diag(np.exp(2 * transformed[10:13])) @ transformed_rotation.T
        np.testing.assert_allclose(actual_covariance, matrix[:3, :3] @ original_covariance @ matrix[:3, :3].T, atol=5e-7)
    restored = gaussians.transform_rows(placed, np.linalg.inv(matrix))
    np.testing.assert_allclose(restored[:, :13], rows[:, :13], atol=1e-6)


@pytest.mark.parametrize("defect", ["nan", "zero-quaternion", "normals", "scale"])
def test_invalid_native_parameters_refuse(defect):
    rows = sample_rows()
    if defect == "nan":
        rows[0, 6] = np.nan
    elif defect == "zero-quaternion":
        rows[0, 13:17] = 0
    elif defect == "normals":
        rows[0, 3] = 1
    else:
        rows[0, 10] = 100
    with pytest.raises(ValueError):
        gaussians.transform_rows(rows, np.eye(4))


@pytest.mark.parametrize("defect", ["reflection", "anisotropic", "shear", "zero", "nan"])
def test_unsupported_frames_refuse(defect):
    matrix = np.eye(4)
    if defect == "reflection":
        matrix[0, 0] = -1
    elif defect == "anisotropic":
        matrix[0, 0] = 2
    elif defect == "shear":
        matrix[0, 1] = .1
    elif defect == "zero":
        matrix[:3, :3] = 0
    else:
        matrix[0, 3] = np.nan
    with pytest.raises(ValueError):
        gaussians.transform_rows(sample_rows(), matrix)


def test_export_retains_native_schema_bytes_and_refuses_overwrite(tmp_path):
    path = tmp_path / "generated.ply"
    rows = sample_rows()
    gaussians.write_placed(path, rows)
    np.testing.assert_array_equal(gaussians.read_native(path), rows)
    with pytest.raises(FileExistsError):
        gaussians.write_placed(path, rows)


@pytest.mark.parametrize("defect", ["missing-tag", "extra-sh", "extra-body", "truncated", "ascii", "symlink"])
def test_unverified_or_incompatible_ply_refuses(tmp_path, defect):
    path = tmp_path / "candidate.ply"
    gaussians.write_placed(path, sample_rows())
    payload = path.read_bytes()
    if defect == "missing-tag":
        payload = payload.replace(f"comment {gaussians.GENERATIVE_TAG}\n".encode(), b"")
    elif defect == "extra-sh":
        payload = payload.replace(b"end_header\n", b"property float f_rest_0\nend_header\n")
    elif defect == "extra-body":
        payload += b"extraneous"
    elif defect == "truncated":
        payload = payload[:-4]
    elif defect == "ascii":
        payload = payload.replace(b"binary_little_endian", b"ascii")
    elif defect == "symlink":
        linked = tmp_path / "linked.ply"
        linked.symlink_to(path)
        path = linked
    if defect != "symlink":
        path.write_bytes(payload)
    with pytest.raises(ValueError):
        gaussians.read_native(path)


def test_world_placement_composes_native_mesh_raw_and_calibrated_frames():
    receipt = {"calibration": {"meters_per_unit": .3}, "recipe": {"model": "local SAM-3D Objects"}}
    result = {"status": "needs-review", "placement_resolved": True, "placement": {"generated_to_raw": np.eye(4).tolist()},
              "model": {"name": "local Meta SAM-3D Objects", "worker": {"ok": True}}}
    matrix = gaussians.world_matrix(receipt, result)
    np.testing.assert_allclose(matrix[:3, :3], .3 * evidence.Y_UP @ gaussians.GAUSSIAN_TO_MESH[:3, :3])
    unplaced = deepcopy(result)
    unplaced["placement_resolved"] = False
    with pytest.raises(ValueError):
        gaussians.world_matrix(receipt, unplaced)
    different_model = deepcopy(receipt)
    different_model["recipe"]["model"] = "unverified adapter"
    with pytest.raises(ValueError, match="SAM-3D model stage"):
        gaussians.world_matrix(different_model, result)


@pytest.mark.parametrize("identifier", ["../../etc", "gaussians_" + "0" * 23, "gaussians_" + "a" * 24 + "/x"])
def test_unregistered_study_paths_refuse(identifier):
    with pytest.raises(ValueError):
        gaussians.directory(Path("/tmp/job"), "generated_" + "a" * 24, identifier)


def test_registered_gaussian_reviews_are_read_only_and_bound_to_parent(generated_scene):
    job, pointer, receipt = generated_scene
    parent = seal_generated(job, receipt)
    identifier = "gaussians_" + "a" * 24
    object_id = receipt["generated_object_id"]
    output = gaussians.directory(job, object_id, identifier)
    output.mkdir(parents=True)
    gaussians.write_placed(output / "placed-splat.ply", sample_rows())
    result = reviews.seal(job, output, "result.json", {"gaussians_id": identifier, "generated_object_id": object_id,
                          "generated_result_sha256": parent["sha256"], "created_at": "2026-09-06", "status": "needs-review",
                          "scope": "synthetic unit fixture; no model inference"}, [output / "placed-splat.ply"])
    app = FastAPI()
    app.include_router(scene_studio_route.router)
    endpoint = f"/jobs/{job.name}/studio/generated-objects/{object_id}/gaussians"
    with TestClient(app) as client:
        assert client.get(endpoint).json()["reviews"][0]["sha256"] == result["sha256"]
        assert client.get(endpoint + "/" + identifier).json() == result
        response = client.get(endpoint + f"/{identifier}/artifact", params={"name": "placed-splat.ply"})
        assert response.status_code == 200 and "immutable" in response.headers["cache-control"]
        assert response.content == (output / "placed-splat.ply").read_bytes()
        assert client.get(endpoint + f"/{identifier}/artifact", params={"name": "../../meta.json"}).status_code == 409
        assert client.post(endpoint, json={"run": True}).status_code == 405
        corrupt = objects.artifact(job, object_id, "master.glb", True)
        assert corrupt.is_file()
        parent_changed = seal_generated(job, receipt, status="unplaced")
        assert parent_changed["sha256"] != parent["sha256"]
        assert client.get(endpoint + "/" + identifier).status_code == 409
    assert scenes.active(job) == pointer


def test_gaussian_artifact_corruption_never_serves(generated_scene):
    job, _, receipt = generated_scene
    parent = seal_generated(job, receipt)
    identifier = "gaussians_" + "b" * 24
    object_id = receipt["generated_object_id"]
    output = gaussians.directory(job, object_id, identifier)
    output.mkdir(parents=True)
    gaussians.write_placed(output / "placed-splat.ply", sample_rows())
    reviews.seal(job, output, "result.json", {"gaussians_id": identifier, "generated_object_id": object_id,
                 "generated_result_sha256": parent["sha256"]}, [output / "placed-splat.ply"])
    artifact = gaussians.artifact(job, object_id, identifier, "placed-splat.ply")
    artifact.chmod(0o644)
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        gaussians.artifact(job, object_id, identifier, "placed-splat.ply")
