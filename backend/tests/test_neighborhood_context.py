import io
import json
import struct

import numpy as np
from PIL import Image
import pytest

from neighborhood_context import DownloadBudget, SCHEMA, US_FOOT_METERS, capture_center, check_surfaces, elevation_grid, glb_bytes, projected_bounds, public_url, read_elevation, refresh_preview


class Response(io.BytesIO):
    def __init__(self, data, declared=None):
        super().__init__(data)
        self.headers = {} if declared is None else {"Content-Length": str(declared)}


def simple_grid(values=None):
    values = np.full((4, 4), 50.0) if values is None else values
    return elevation_grid(values, [100, 200, 140, 240], [120, 220], 45)


def test_us_survey_foot_xy_and_meter_z_are_not_confused():
    mesh = simple_grid()
    assert np.ptp(mesh["positions"][:, 0]) == pytest.approx(30 * US_FOOT_METERS)
    assert np.allclose(mesh["positions"][:, 1], 5)
    assert np.ptp(mesh["positions"][:, 2]) == pytest.approx(30 * US_FOOT_METERS)
    assert mesh["positions"][0, 2] < 0
    assert mesh["stats"]["source_pixel_size_m"] == pytest.approx([10 * US_FOOT_METERS] * 2)


def test_half_mile_bbox_is_metric_and_coarse():
    bounds = projected_bounds(10000, 20000, 804.672)
    assert (bounds[2] - bounds[0]) * US_FOOT_METERS == pytest.approx(1609.344)
    assert (bounds[3] + bounds[1]) / 2 == pytest.approx(20000)


@pytest.mark.parametrize("half_width", [0, 9, 805, float("nan"), float("inf")])
def test_oversized_invalid_bounds_refuse(half_width):
    with pytest.raises(ValueError):
        projected_bounds(100, 200, half_width)


def test_upward_winding_normals_and_top_origin_texture_coordinates():
    mesh = simple_grid()
    triangles = mesh["positions"][mesh["indices"]]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(cross[:, 1] > 0)
    assert np.allclose(mesh["normals"], [0, 1, 0])
    assert mesh["uv"][0].tolist() == pytest.approx([0.125, 0.125])
    assert mesh["uv"][-1].tolist() == pytest.approx([0.875, 0.875])


def test_nodata_is_removed_not_filled_or_joined_by_triangles():
    values = np.full((4, 4), 50.0)
    values[1, 1] = np.nan
    mesh = simple_grid(values)
    assert mesh["stats"]["vertices"] == 15
    assert mesh["stats"]["triangles"] < 18
    assert np.isfinite(mesh["positions"]).all()
    assert np.allclose(mesh["positions"][:, 1], 5)
    assert mesh["stats"]["valid_fraction"] == 15 / 16


def test_numeric_nodata_and_extreme_values_do_not_create_spikes():
    values = np.full((4, 4), 50.0)
    values[1, 1] = -9999
    values[2, 2] = 3.4e38
    mesh = elevation_grid(values, [100, 200, 140, 240], [120, 220], 45, nodata=-9999)
    assert np.allclose(mesh["positions"][:, 1], 5)
    assert mesh["stats"]["valid_fraction"] == 14 / 16


def test_large_local_elevation_jump_discards_incident_faces():
    values = np.full((4, 4), 50.0)
    values[1, 1] = 500
    mesh = simple_grid(values)
    assert mesh["stats"]["discarded_spike_faces"] > 0
    assert np.allclose(mesh["positions"][:, 1], 5)


@pytest.mark.parametrize("values", [np.zeros((1, 3)), np.zeros((257, 257)), np.full((4, 4), np.nan), np.array([1, 2])])
def test_invalid_grid_refuses(values):
    with pytest.raises(ValueError):
        simple_grid(values)


def test_glb_has_finite_metric_geometry_embedded_texture_and_aligned_buffers():
    texture = io.BytesIO()
    Image.new("RGB", (2, 2), (120, 150, 100)).save(texture, format="PNG")
    mesh = simple_grid()
    content = glb_bytes(mesh, texture.getvalue(), "test")
    magic, version, length = struct.unpack_from("<4sII", content)
    assert (magic, version, length) == (b"glTF", 2, len(content))
    json_length, chunk_type = struct.unpack_from("<I4s", content, 12)
    assert chunk_type == b"JSON"
    document = json.loads(content[20:20 + json_length])
    assert document["accessors"][0]["count"] == 16
    assert document["accessors"][3]["count"] == 54
    assert document["extras"]["generative_completion"] is False
    assert all(view["byteOffset"] % 4 == 0 for view in document["bufferViews"])
    binary_start = 20 + json_length + 8
    image_view = document["bufferViews"][document["images"][0]["bufferView"]]
    assert content[binary_start + image_view["byteOffset"]:][:8] == b"\x89PNG\r\n\x1a\n"


def test_capture_origin_is_rounded_unknown_accuracy_and_ignores_gps_altitude(tmp_path):
    source = tmp_path / "capture.json"
    source.write_text(json.dumps({"capture_id": "capture-test", "gps": {"samples": [{"latitude": 38.465236666, "longitude": -122.717916666, "altitude_m": 999}]}}))
    center = capture_center(source)
    assert center["latitude"] == 38.4652 and center["longitude"] == -122.7179
    assert center["horizontal_accuracy_m"] is None
    assert not center["gps_altitude_used"] and not center["survey_verified"]
    assert len(center["source_manifest_sha256"]) == 64


@pytest.mark.parametrize("url", ["http://imagery.nationalmap.gov/data", "https://evil.example/data", "https://socogis.sonomacounty.ca.gov.evil.example/data", "https://user:password@imagery.nationalmap.gov/data", "https://imagery.nationalmap.gov:8443/data"])
def test_private_or_unapproved_download_urls_refuse(url):
    with pytest.raises(ValueError):
        public_url(url)


def test_budget_counts_each_download_and_refuses_oversized_declared_length(tmp_path):
    budget = DownloadBudget(limit=20, used=0)
    assert budget.receive(Response(b"abc"), tmp_path / "first", 10) == 3
    assert budget.used == 3
    with pytest.raises(ValueError, match="Declared"):
        budget.receive(Response(b"more", 30), tmp_path / "second", 10)
    assert budget.used == 3 and not (tmp_path / "second").exists()


def test_unknown_length_stream_cannot_read_past_aggregate_ceiling(tmp_path):
    budget = DownloadBudget(limit=10, used=2)
    response = Response(b"0123456789abcdef")
    with pytest.raises(ValueError, match="ceiling"):
        budget.receive(response, tmp_path / "partial", 100)
    assert budget.used == 10 and response.tell() == 8
    assert (tmp_path / "partial").stat().st_size == 8


def test_download_does_not_overwrite_existing_sources(tmp_path):
    existing = tmp_path / "source"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        DownloadBudget(limit=20, used=0).receive(Response(b"new"), existing, 10)
    assert existing.read_bytes() == b"original"


def test_dsm_below_dem_catches_real_source_unit_mismatch():
    terrain_feet = np.full((4, 4), 164.0)
    surface_meters = np.full((4, 4), 58.0)
    with pytest.raises(ValueError, match="check source units"):
        check_surfaces(terrain_feet, surface_meters)
    assert check_surfaces(terrain_feet * 0.3048, surface_meters)["below_terrain_by_more_than_2m_fraction"] == 0


@pytest.fixture
def elevation_tiff(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("pyproj")
    from rasterio.transform import from_bounds

    path = tmp_path / "elevation.tif"
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32", crs="EPSG:6418", transform=from_bounds(100, 200, 140, 240, 4, 4), nodata=-9999) as dataset:
        values = np.full((4, 4), 100, dtype=np.float32)
        values[0, 0] = -9999
        dataset.write(values, 1)
    return path


def test_explicit_feet_dem_normalizes_z_but_does_not_scale_xy(elevation_tiff):
    values, receipt = read_elevation(elevation_tiff, [100, 200, 140, 240], 4, "feet")
    assert np.isnan(values[0, 0])
    assert values[1, 1] == pytest.approx(30.48)
    assert receipt["pixel_size_us_feet"] == [10, 10]
    assert receipt["vertical_scale_to_meters"] == 0.3048


def test_meter_surface_keeps_z_scale(elevation_tiff):
    values, receipt = read_elevation(elevation_tiff, [100, 200, 140, 240], 4, "meters")
    assert values[1, 1] == 100 and receipt["vertical_scale_to_meters"] == 1


def test_raster_changed_bounds_or_units_refuse(elevation_tiff):
    with pytest.raises(ValueError, match="bounds"):
        read_elevation(elevation_tiff, [101, 200, 140, 240], 4)
    with pytest.raises(ValueError, match="vertical units"):
        read_elevation(elevation_tiff, [100, 200, 140, 240], 4, "guess")


def test_wrong_projected_raster_crs_is_rejected(elevation_tiff):
    import rasterio

    with rasterio.open(elevation_tiff, "r+") as dataset:
        dataset.crs = "EPSG:3857"
    with pytest.raises(ValueError, match="EPSG:6418"):
        read_elevation(elevation_tiff, [100, 200, 140, 240], 4)


def test_preview_refresh_cannot_reseal_changed_sources(tmp_path):
    (tmp_path / "source").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "status": "prepared-needs-review", "files": {"source": {"sha256": "a" * 64}}}))
    with pytest.raises(ValueError, match="hash changed"):
        refresh_preview(tmp_path, tmp_path / "three")


def test_preview_refresh_refuses_rejected_candidate(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "status": "rejected-source-vertical-unit-mismatch", "files": {"source": {"sha256": "a" * 64}}}))
    with pytest.raises(ValueError, match="complete prepared"):
        refresh_preview(tmp_path, tmp_path / "three")
