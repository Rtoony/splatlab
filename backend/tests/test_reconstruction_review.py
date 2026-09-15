import numpy as np
import pytest

import reconstruction_review as review


def parameters(count=3):
    return {"means": np.arange(count * 3).reshape(count, 3).astype(np.float32),
            "scales": np.full((count, 3), -2, dtype=np.float32),
            "quats": np.tile([2, 0, 0, 0], (count, 1)).astype(np.float32),
            "features_dc": np.arange(count * 3).reshape(count, 3).astype(np.float32),
            "features_rest": np.arange(count * 15 * 3).reshape(count, 15, 3).astype(np.float32),
            "opacities": np.array([-10, 0, 10], dtype=np.float32).reshape(count, 1)}


def test_lossless_export_retains_low_opacity_rows_and_all_sh_coefficients(tmp_path):
    values = parameters()
    output = tmp_path / "model.ply"
    result = review.export_splats(output, values)
    header, body = output.read_bytes().split(b"end_header\n", 1)
    names = [line.split()[-1].decode() for line in header.splitlines() if line.startswith(b"property")]
    vertices = np.frombuffer(body, dtype=[(name, "<f4") for name in names])
    assert len(vertices) == result["rows"] == 3
    assert result["sh_degree"] == 3
    assert result["opacity_below_1_over_255_rows"] == 1
    assert result["geometry_error"] is None
    for index, name in enumerate(("x", "y", "z")):
        np.testing.assert_array_equal(vertices[name], values["means"][:, index])
    for index in range(45):
        np.testing.assert_array_equal(vertices[f"f_rest_{index}"], values["features_rest"].transpose(0, 2, 1).reshape(3, -1)[:, index])
    np.testing.assert_array_equal(vertices["rot_0"], values["quats"][:, 0])
    np.testing.assert_array_equal(vertices["opacity"], values["opacities"][:, 0])
    np.testing.assert_array_equal(vertices["scale_2"], values["scales"][:, 2])
    with pytest.raises(FileExistsError):
        review.export_splats(output, values)


@pytest.mark.parametrize("defect", ["nonfinite", "dimension", "rotation", "scale", "sh", "dtype"])
def test_invalid_parameters_refuse_without_partial_export(tmp_path, defect):
    values = parameters()
    if defect == "nonfinite":
        values["means"][0, 0] = np.nan
    elif defect == "dimension":
        values["opacities"] = values["opacities"].ravel()
    elif defect == "rotation":
        values["quats"][0] = 0
    elif defect == "scale":
        values["scales"][0, 0] = -1000
    elif defect == "sh":
        values["features_rest"] = values["features_rest"][:, :2]
    else:
        values["means"] = values["means"].astype(np.float64)
    output = tmp_path / "model.ply"
    with pytest.raises(ValueError):
        review.export_splats(output, values)
    assert not output.exists()


def test_depth_statistics_do_not_invent_geometric_truth(tmp_path):
    output = tmp_path / "depth.npy"
    np.save(output, np.array([[1, np.nan], [-1, 3]], dtype=np.float32), allow_pickle=False)
    result = review.depth_statistics(output)
    assert result["positive_fraction"] == .5
    assert result["finite_fraction"] == .75
    assert result["positive_depth_training_units"]["median"] == 2
    assert "not an independent" in result["scope"]


@pytest.mark.parametrize("depth", [np.zeros((2, 2, 3)), np.array(1.), np.zeros((2, 2), dtype=np.int32)])
def test_malformed_depth_refuses(tmp_path, depth):
    output = tmp_path / "depth.npy"
    np.save(output, depth, allow_pickle=False)
    with pytest.raises(ValueError):
        review.depth_statistics(output)


def test_html_review_is_local_read_only_and_labels_limits():
    metrics = {"psnr_db": 20., "ssim": .8, "gaussians": 3, "training_seconds": 1.}
    result = {"metrics": {"splatfacto": metrics, "dn-splatter": metrics}, "held_out_views": 1, "budget": {"iterations_per_arm": 10000}}
    runs = {"splatfacto": {"evaluation": {"views": [{"image": "images/000001.png"}]}}}
    html = review.review_html(result, runs)
    assert "../dn-splatter/renders/000001.png" in html
    assert "../splatfacto/renders/000001-reference.png" in html
    assert "not survey metres" in html
    assert "no method promoted" in html
    assert "<script" not in html
    assert "https://" not in html
