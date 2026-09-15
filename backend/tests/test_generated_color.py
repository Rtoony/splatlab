import numpy as np
import pytest

import generated_color as colors
import glb_transform
from mesh.provenance import GENERATIVE_TAG, GLTF_EXTRAS_KEY, glb_is_generative


def fixture_glb():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4").tobytes()
    rgba = np.array([[35, 24, 17, 255], [155, 129, 94, 255], [128, 128, 128, 128]], dtype=np.uint8).tobytes()
    document = {"asset": {"version": "2.0", "extras": {GLTF_EXTRAS_KEY: GENERATIVE_TAG}},
                "buffers": [{"byteLength": len(positions + rgba)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}, {"buffer": 0, "byteOffset": len(positions), "byteLength": len(rgba)}],
                "accessors": [{"componentType": 5126, "type": "VEC3", "count": 3, "bufferView": 0, "min": [0, 0, 0], "max": [1, 1, 0]},
                              {"componentType": 5121, "normalized": True, "type": "VEC4", "count": 3, "bufferView": 1}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}}]}], "nodes": [{"mesh": 0}]}
    return document, positions + rgba


def test_srgb_curve_uses_standard_toe_and_preserves_all_rgb8_values():
    values = np.arange(256) / 255.
    np.testing.assert_array_equal(np.rint(colors.linear_to_srgb(colors.srgb_to_linear(values)) * 255), np.arange(256))
    assert colors.srgb_to_linear([.5])[0] == pytest.approx(.21404114)
    assert colors.srgb_to_linear([.01])[0] == pytest.approx(.01 / 12.92)


def test_export_keeps_original_geometry_bytes_and_roundtrips_colors(tmp_path):
    original, derived = tmp_path / "original.glb", tmp_path / "derived.glb"
    document, binary = fixture_glb()
    original.write_bytes(glb_transform._assemble(document, binary))
    original_bytes = original.read_bytes()
    report = colors.derive(original, derived)
    result, output = glb_transform._chunks(derived.read_bytes())
    assert original.read_bytes() == original_bytes
    assert output[:len(binary)] == binary
    assert result["accessors"][0] == document["accessors"][0]
    assert result["meshes"] == document["meshes"] and result["nodes"] == document["nodes"]
    assert result["accessors"][1]["componentType"] == 5126
    assert glb_is_generative(derived)
    assert report["accessors"][0]["rgb8_roundtrip_exact"]
    expected = np.array([[35, 24, 17, 255], [155, 129, 94, 255], [128, 128, 128, 128]], dtype=np.uint8)
    np.testing.assert_array_equal(colors.decoded_rgba(derived), expected)
    assert colors.decoded_rgba(original) is None
    with pytest.raises(ValueError, match="already applied"):
        colors.derive(derived, tmp_path / "twice.glb")


@pytest.mark.parametrize("defect", ["untagged", "texture", "float", "missing-color", "sparse", "overrun"])
def test_ambiguous_color_interpretation_refuses(tmp_path, defect):
    document, binary = fixture_glb()
    if defect == "untagged":
        document["asset"]["extras"] = {}
    elif defect == "texture":
        document["materials"] = [{"pbrMetallicRoughness": {}}]
    elif defect == "float":
        document["accessors"][1]["componentType"] = 5126
    elif defect == "missing-color":
        del document["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]
    elif defect == "sparse":
        document["accessors"][1]["sparse"] = {}
    else:
        document["accessors"][1]["count"] = 4
    original = tmp_path / "original.glb"
    original.write_bytes(glb_transform._assemble(document, binary))
    with pytest.raises(ValueError):
        colors.derive(original, tmp_path / "derived.glb")
    assert not (tmp_path / "derived.glb").exists()
