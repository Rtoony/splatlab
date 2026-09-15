from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

import artifact_manifest as manifests
import reconstruction_comparison as comparison
import reconstruction_surfaces as surfaces
from mesh.provenance import GenerativeInputRefused, assert_not_generative, ply_is_generative


def camera():
    return surfaces.camera_record(np.column_stack([np.eye(3), [0., 0., 1.]]), 3, 3, 2., 2., 1.5, 1.5)


def test_opengl_camera_rays_preserve_axial_depth_not_euclidean_distance():
    values = surfaces.rays(camera())
    np.testing.assert_allclose(values[1, 1], [0, 0, 1, 0, 0, -1])
    np.testing.assert_allclose(values[0, 0, 3:], [-.5, .5, -1])
    assert np.linalg.norm(values[0, 0, 3:]) > 1
    points = values[..., :3] + values[..., 3:]
    np.testing.assert_allclose(points[..., 2], 0)


def test_rays_rotate_with_camera_without_changing_depth_scale():
    pose = np.column_stack([np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]]), [0., 0., 1.]])
    rotated = surfaces.camera_record(pose, 3, 3, 2., 2., 1.5, 1.5)
    np.testing.assert_allclose(surfaces.rays(rotated)[0, 0, 3:], [-.5, -.5, -1])


@pytest.mark.parametrize("defect", ["reflection", "nonrigid", "intrinsic", "dimensions", "nan"])
def test_invalid_camera_refuses(defect):
    pose = np.column_stack([np.eye(3), [0., 0., 1.]])
    width, focal = 3, 2.
    if defect == "reflection":
        pose[0, 0] = -1
    elif defect == "nonrigid":
        pose[0, 0] = 2
    elif defect == "intrinsic":
        focal = 0
    elif defect == "dimensions":
        width = 10_000_000
    else:
        pose[0, 0] = np.nan
    with pytest.raises(ValueError):
        surfaces.camera_record(pose, width, 3, focal, 2., 1.5, 1.5)


def test_crop_ray_intersection_handles_parallel_outside_and_behind():
    bounds = {"minimum": [-.1, -.1, -.1], "maximum": [.1, .1, .1]}
    rays = np.array([[0, 0, 1, 0, 0, -1], [1, 0, 1, 0, 0, -1], [0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(surfaces.crop_mask(rays, bounds), [True, False, False, True])
    np.testing.assert_array_equal(surfaces.crop_mask(rays, bounds, np.array([1, 1, 1, .5])), [True, False, False, False])


def test_filtered_depth_excludes_masks_opacity_invalid_and_outside_crop():
    values = surfaces.rays(camera())
    depth = np.ones((3, 3), dtype=np.float32)
    opacity = np.ones_like(depth)
    valid = np.ones_like(depth, dtype=bool)
    depth[0, 0], depth[0, 1], depth[0, 2] = np.nan, 0, 5
    depth[1, 0] = 3
    opacity[1, 1] = .4
    valid[1, 2] = False
    bounds = {"minimum": [-1, -1, -.5], "maximum": [1, 1, .5]}
    result, mask = surfaces.filtered_depth(depth, opacity, valid, values, bounds, {"depth_truncation": 4, "alpha_minimum": .5})
    assert not mask[:2].any()
    assert mask[2].all()
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:2], 0)


def test_depth_alignment_and_invalid_opacity_refuse():
    with pytest.raises(ValueError, match="align"):
        surfaces.filtered_depth(np.ones((3, 3)), np.ones((2, 2)), np.ones((3, 3)), surfaces.rays(camera()), {}, {})
    with pytest.raises(ValueError, match="opacity"):
        surfaces.filtered_depth(np.ones((3, 3)), np.full((3, 3), np.nan), np.ones((3, 3)), surfaces.rays(camera()), {}, {})


def test_crop_bounds_exclude_extreme_seed_outliers_and_bound_grid():
    points = np.tile(np.linspace(-1, 1, 1000)[:, None], (1, 3))
    points[-1] = 1000
    bounds = surfaces.bounds_for_seeds(points, .01)
    assert max(bounds["maximum"]) < 2
    assert min(bounds["minimum"]) > -2
    with pytest.raises(ValueError, match="grid"):
        surfaces.bounds_for_seeds(points * 1000, .01)


@pytest.mark.parametrize("voxel,seconds", [(0, 600), (.001, 600), (.2, 600), (True, 600), (float("nan"), 600), (.01, 0), (.01, 1801), (.01, True)])
def test_unbounded_recipe_refuses_before_reading_data(tmp_path, voxel, seconds):
    with pytest.raises(ValueError):
        surfaces.prepare(tmp_path, ["splatfacto", "ags-mesh"], voxel, seconds)


def test_seed_normalization_removes_saved_applied_transform(tmp_path):
    raw = np.array([[0., 1., 2.], [3., 4., 5.]])
    applied = np.array([[0., 1, 0, .25], [1, 0, 0, .5], [0, 0, -1, .75]])
    saved = raw @ applied[:, :3].T + applied[:, 3]
    packed = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for index, name in enumerate(("x", "y", "z")):
        packed[name] = saved[:, index]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    (dataset / "train-seeds.ply").write_bytes(header + packed.tobytes())
    manifests.atomic_write_json(dataset / "transforms.json", {"applied_transform": applied.tolist()})
    normalization = {"transform": [[0., -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3]], "scale": 2.}
    destination = tmp_path / "splatfacto"
    destination.mkdir()
    manifests.atomic_write_json(destination / "dataparser_transforms.json", normalization)
    points, lineage = surfaces.normalized_seeds(tmp_path, {"arm": "splatfacto", "artifacts": {"dataparser_transforms.json": {}}})
    transform = np.asarray(normalization["transform"])
    np.testing.assert_allclose(points, (raw @ transform[:, :3].T + transform[:, 3]) * 2)
    assert lineage["source_to_training"] == normalization


def test_surface_export_preserves_double_positions_and_portable_non_survey_tag(tmp_path):
    vertices = np.array([[0.123456789012345, 0, 0], [1, 0, 0], [0, 1, 0]])
    triangles = np.array([[0, 1, 2]])
    colors = np.tile([1., .5, 0], (3, 1))
    output = tmp_path / "surface.ply"
    surfaces.write_mesh(output, vertices, triangles, colors)
    assert ply_is_generative(output)
    copied = tmp_path / "unrelated-name.ply"
    copied.write_bytes(output.read_bytes())
    with pytest.raises(GenerativeInputRefused):
        assert_not_generative(copied, "survey")
    _header, body = output.read_bytes().split(b"end_header\n", 1)
    packed = np.frombuffer(body, dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1")], count=3)
    np.testing.assert_array_equal(packed["x"], vertices[:, 0])
    np.testing.assert_array_equal(packed["green"], 128)
    with pytest.raises(FileExistsError):
        surfaces.write_mesh(output, vertices, triangles, colors)


@pytest.mark.parametrize("defect", ["nan", "index", "float-index", "color"])
def test_invalid_surface_refuses_without_partial_output(tmp_path, defect):
    vertices = np.eye(3)
    triangles = np.array([[0, 1, 2]])
    colors = np.ones((3, 3))
    if defect == "nan":
        vertices[0, 0] = np.nan
    elif defect == "index":
        triangles[0, 0] = 5
    elif defect == "float-index":
        triangles = triangles.astype(float)
    else:
        colors[0, 0] = 1.1
    output = tmp_path / "mesh.ply"
    with pytest.raises(ValueError):
        surfaces.write_mesh(output, vertices, triangles, colors)
    assert not output.exists()


def test_missing_mesh_hits_are_not_removed_from_coverage_denominator():
    result = surfaces.depth_agreement(np.array([1., np.inf]), np.array([1., 1.]), np.array([True, True]), .01)
    assert result["mesh_hit_fraction_on_supported_model"] == .5
    assert result["within_two_voxels_fraction_of_supported_model"] == .5
    assert result["depth_mae_training_units"] == 0
    assert "NOT measured" in result["scope"]
    missing = surfaces.depth_agreement(np.array([np.inf]), np.array([1.]), np.array([True]), .01)
    assert missing["depth_mae_training_units"] is None
    assert missing["mesh_hit_fraction_on_supported_model"] == 0
    with pytest.raises(ValueError):
        surfaces.depth_agreement(np.array([1.]), np.array([np.nan]), np.array([True]), .01)


@pytest.fixture
def surface_study(tmp_path, monkeypatch):
    output = tmp_path / "surfaces-fixture"
    output.mkdir()
    names = ["splatfacto", "ags-mesh"]
    receipt = comparison.seal(output, "receipt.json", {"arms": names, "comparison_sha256": "original", "run_sha256": {name: "trained-" + name for name in names},
        "parameters": {"voxel_length": .01}, "bounds": {}, "training_images": ["images/train.png"], "evaluation_images": ["images/test.png"]})
    monkeypatch.setattr(comparison, "summarize", lambda *args: {"sha256": "original", "baseline": "splatfacto"})
    for name in names:
        directory = output / name
        (directory / "renders").mkdir(parents=True)
        (directory / "surface.ply").write_bytes(b"controlled-mesh-fixture")
        Image.new("RGB", (2, 2), "white").save(directory / "renders/test-reference.png")
        np.savez(directory / "renders/test.npz", roi_rays=np.ones((2, 2), dtype=bool))
        evaluation = [{"image": "images/test.png", "depth_mae_training_units": .01, "source_photo_psnr_on_roi_rays": 20.,
                       "mesh_hit_fraction_on_roi_rays": .8, "mesh_hit_fraction_on_supported_model": .9, "within_two_voxels_fraction_of_supported_model": .7}]
        comparison.seal(directory, "run.json", {"status": "completed", "surface_receipt_sha256": receipt["sha256"], "trained_run_sha256": "trained-" + name,
            "parameters": receipt["parameters"], "training_camera_sha256": "same", "evaluation_camera_sha256": "same", "open3d_version": "same", "torch_version": "same",
            "evaluation": evaluation, "training_views": [{"image": "images/train.png"}], "mesh": {"triangles": 1}, "fusion_seconds": 1., "total_seconds": 2.,
            "max_rss_bytes": 100, "max_cuda_allocated_bytes": 100, "artifacts": {str(path.relative_to(directory)): manifests.file_identity(path) for path in directory.rglob("*") if path.is_file()}})
    return output


def test_surface_comparison_is_explicitly_unpromoted_and_repeatable(surface_study):
    result, runs = surfaces.summarize(surface_study)
    assert result["status"] == "needs-review"
    assert result["promoted"] is False
    assert result["geometric_reference_error"] is None
    assert surfaces.summarize(surface_study)[0] == result
    html = surfaces.review_html(result, runs)
    assert "not survey measurements" in html
    assert "ags-mesh/renders/test-normals.png" in html
    assert "<script" not in html


@pytest.mark.parametrize("defect", ["failed", "recipe", "camera", "missing-eval", "heldout-fusion", "artifact", "crop", "nonfinite"])
def test_surface_comparison_rejects_invalid_evidence(surface_study, defect):
    directory = surface_study / "ags-mesh"
    run = deepcopy(comparison.read(directory, "run.json"))
    if defect == "failed":
        run["status"] = "failed"
    elif defect == "recipe":
        run["parameters"]["voxel_length"] = .02
    elif defect == "camera":
        run["evaluation_camera_sha256"] = "different"
    elif defect == "missing-eval":
        run["evaluation"] = []
    elif defect == "heldout-fusion":
        run["training_views"] = [{"image": "images/test.png"}]
    elif defect == "artifact":
        (directory / "surface.ply").write_bytes(b"changed")
    elif defect == "nonfinite":
        run["evaluation"][0]["depth_mae_training_units"] = float("nan")
    else:
        np.savez(directory / "renders/test.npz", roi_rays=np.zeros((2, 2), dtype=bool))
        run["artifacts"]["renders/test.npz"] = manifests.file_identity(directory / "renders/test.npz")
    if defect == "nonfinite":
        with pytest.raises(ValueError):
            comparison.seal(directory, "run.json", run)
        return
    comparison.seal(directory, "run.json", run)
    with pytest.raises(ValueError):
        surfaces.summarize(surface_study)
