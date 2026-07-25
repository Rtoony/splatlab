"""Pure artifact-manifest inspection tests."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifact_manifest  # noqa: E402


def _write_binary_ply(path: Path) -> None:
    properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [
        f"f_rest_{index}" for index in range(9)
    ]
    header = ["ply", "format binary_little_endian 1.0", "element vertex 2"]
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")
    rows = [
        [-2.0, 3.0, 1.0] + [0.0] * 12,
        [4.0, -1.0, 8.0] + [0.0] * 12,
    ]
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        for row in rows:
            handle.write(struct.pack("<" + "f" * len(row), *row))


def test_binary_ply_inspection_reports_count_sh_and_bounds(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    _write_binary_ply(path)

    result = artifact_manifest.inspect_ply(path)

    assert result["gaussian_count"] == 2
    assert result["sh_degree"] == 1
    assert result["bounds_scene"] == {
        "min": [-2.0, -1.0, 1.0],
        "max": [4.0, 3.0, 8.0],
        "extent": [6.0, 4.0, 7.0],
    }
    assert len(result["sha256"]) == 64


def test_ascii_ply_inspection(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "0 1 2",
                "-1 4 3",
                "",
            ]
        )
    )
    result = artifact_manifest.inspect_ply(path)
    assert result["gaussian_count"] == 2
    assert result["sh_degree"] is None
    assert result["bounds_scene"]["extent"] == [1.0, 3.0, 1.0]


def test_truncated_binary_ply_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "scene.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with pytest.raises(artifact_manifest.ArtifactManifestError, match="ended"):
        artifact_manifest.inspect_ply(path)


def test_atomic_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    artifact_manifest.atomic_write_json(path, {"generation": 1})
    artifact_manifest.atomic_write_json(path, {"generation": 2, "ok": True})
    assert json.loads(path.read_text()) == {"generation": 2, "ok": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_coordinate_record_keeps_scale_and_geo_separate() -> None:
    geo = {"lat": 38.0, "lon": -122.0, "heading_deg": 12.0}
    result = artifact_manifest.coordinate_record({"meters_per_unit": 2.5, "geo": geo})
    assert result["source_frame"]["up_axis"] == "+Z"
    assert result["source_frame"]["meters_per_unit"] == 2.5
    assert result["source_frame"]["scale_calibrated"] is True
    assert result["georeference"] == geo


@pytest.mark.parametrize("value", [True, False, 0, -1, math.inf, math.nan, "1.0"])
def test_coordinate_record_rejects_invalid_calibration(value) -> None:
    result = artifact_manifest.coordinate_record({"meters_per_unit": value})

    assert result["source_frame"]["scale_calibrated"] is False
    assert result["source_frame"]["meters_per_unit"] is None
