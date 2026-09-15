import copy
import json

from PIL import Image
import pytest

import artifact_manifest as manifests
from capture_appearance import build_comparison
from fisheye_evaluation import image_scores
import numpy as np


@pytest.fixture
def evaluations(tmp_path):
    roots = [tmp_path / "before", tmp_path / "after"]
    for ordinal, root in enumerate(roots):
        (root / "renders").mkdir(parents=True)
        reference = np.full((8, 8, 3), 120, dtype=np.uint8)
        rendered = np.full((8, 8, 3), 100 + ordinal * 10, dtype=np.uint8)
        for kind, pixels in (("reference", reference), ("render", rendered)):
            Image.fromarray(pixels).save(root / f"renders/000-{kind}.png")
        record = {"image": "val/lens-0-frame-000030-centre.png", "width": 8, "height": 8,
                  "fx": 5, "fy": 5, "cx": 4, "cy": 4,
                  "camera_to_world_opengl": np.eye(4)[:3].tolist(),
                  **image_scores(reference / 255, rendered / 255, np.ones((8, 8), dtype=bool))}
        document = {"schema": "dev.splatlab.fisheye-baseline-evaluation/v1", "status": "evaluated-needs-review",
                    "method": "gsplat-2dgs", "test_split_evaluated": False, "training_iterations": 3000 * (ordinal + 1),
                    "source_hashes": {"/frozen/dataset/transforms.json": "a" * 64,
                                      "/frozen/dataset/photo.png": "b" * 64},
                    "validation": [record], "gaussians": {"rows": 20},
                    "files": {path.relative_to(root).as_posix(): manifests.sha256_file(path)
                              for path in (root / "renders").iterdir()}}
        manifests.atomic_write_json(root / "receipt.json", document)
    return *roots, tmp_path / "comparison"


def change_receipt(root, change):
    path = root / "receipt.json"
    document = json.loads(path.read_text())
    change(document)
    manifests.atomic_write_json(path, document)


def test_comparison_copies_all_real_views_and_keeps_inputs(evaluations):
    before, after, output = evaluations
    old_receipts = [(root / "receipt.json").read_bytes() for root in (before, after)]
    result = build_comparison(before, after, output)
    assert result["all_validation_views_included"] and not result["generated_completion"]
    assert result["aggregate_appearance_improved"]
    assert result["after"]["summary"]["masked_psnr_db"] > result["before"]["summary"]["masked_psnr_db"]
    assert len(list((output / "images").iterdir())) == 3
    assert "not geometric accuracy" in (output / "index.html").read_text()
    assert old_receipts == [(root / "receipt.json").read_bytes() for root in (before, after)]


def test_no_gain_is_not_promoted_as_a_sharper_result(evaluations):
    before, after, output = evaluations
    result = build_comparison(after, before, output)
    assert not result["aggregate_appearance_improved"]
    assert "No overall appearance gain" in (output / "index.html").read_text()


@pytest.mark.parametrize("change", [
    lambda receipt: receipt.update(test_split_evaluated=True),
    lambda receipt: receipt.update(status="running"),
    lambda receipt: receipt["source_hashes"].update({"/frozen/dataset/transforms.json": "c" * 64}),
    lambda receipt: receipt["source_hashes"].update({"/frozen/dataset/photo.png": "c" * 64}),
    lambda receipt: receipt["validation"][0].update(fx=6),
    lambda receipt: receipt["validation"][0].update(image="different-view.png"),
    lambda receipt: receipt["validation"].append(copy.deepcopy(receipt["validation"][0])),
    lambda receipt: receipt["files"].pop("renders/000-render.png"),
])
def test_mismatched_or_incomplete_comparison_refuses(evaluations, change):
    before, after, output = evaluations
    change_receipt(after, change)
    with pytest.raises(ValueError):
        build_comparison(before, after, output)
    assert not output.exists()


def test_changed_image_and_resealed_reference_refuse(evaluations):
    before, after, output = evaluations
    path = after / "renders/000-reference.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    with pytest.raises(ValueError):
        build_comparison(before, after, output)
    change_receipt(after, lambda receipt: receipt["files"].update({"renders/000-reference.png": manifests.sha256_file(path)}))
    with pytest.raises(ValueError, match="byte-identical"):
        build_comparison(before, after, output)


def test_existing_or_source_nested_output_refuses(evaluations):
    before, after, output = evaluations
    for destination in (before / "nested", after):
        with pytest.raises(ValueError, match="new comparison"):
            build_comparison(before, after, destination)
