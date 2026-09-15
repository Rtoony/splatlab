import json
import struct

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from neighborhood_buildings import ContextGrid, colored_glb, elevation_source_units, esri_polygon, extrude_building, normalize_footprints


def test_esri_holes_and_multiple_shells_survive_orientation():
    outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]
    second = [[20, 0], [25, 0], [25, 5], [20, 5], [20, 0]]
    geometry = esri_polygon([hole, second, outer])
    assert geometry.geom_type == "MultiPolygon"
    assert geometry.area == 89
    assert sum(len(part.interiors) for part in geometry.geoms) == 1


def test_invalid_source_ring_refuses():
    with pytest.raises(ValueError):
        esri_polygon([[[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]])


def test_extrusion_keeps_hole_open_and_outward_wall_normals():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], holes=[[(3, 3), (3, 7), (7, 7), (7, 3)]])
    mesh = extrude_building(polygon, 0, 5, [0, 0, 0])
    positions = mesh["positions"].reshape(-1, 3, 3)
    normals = mesh["normals"].reshape(-1, 3, 3)[:, 0]
    roof = positions[np.all(positions[:, :, 1] == 5, axis=1)]
    roof_area = np.linalg.norm(np.cross(roof[:, 1] - roof[:, 0], roof[:, 2] - roof[:, 0]), axis=1).sum() / 2
    assert roof_area == pytest.approx(84)
    assert np.all(normals[np.all(positions[:, :, 1] == 5, axis=1), 1] > 0)
    south_wall = np.all(positions[:, :, 2] == 0, axis=1) & np.any(positions[:, :, 1] != positions[:, :1, 1], axis=1)
    assert np.all(normals[south_wall, 2] > 0)


def test_metric_local_frame_preserves_ground_origin():
    mesh = extrude_building(box(1937300, 588800, 1937310, 588810), 50, 56, [1937300, 588800, 50])
    assert mesh["positions"].min(axis=0).tolist() == [0, 0, -10]
    assert mesh["positions"].max(axis=0).tolist() == [10, 6, 0]


def test_per_feature_glb_nodes_preserve_pick_identity():
    mesh = extrude_building(box(0, 0, 5, 5), 0, 5, [0, 0, 0])
    payload = colored_glb([{"id": "county-buildings-7", "mesh": mesh, "extras": {"featureId": "county-buildings-7"}}])
    magic, version, length = struct.unpack_from("<4sII", payload)
    assert (magic, version, length) == (b"glTF", 2, len(payload))
    json_length = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20:20 + json_length])
    assert document["nodes"][0]["extras"]["featureId"] == "county-buildings-7"
    assert "COLOR_0" in document["meshes"][0]["primitives"][0]["attributes"]


def test_source_crs_is_transformed_and_duplicate_geometry_removed():
    from pyproj import Transformer

    source = [[6356110, 1932040], [6356140, 1932040], [6356140, 1932070], [6356110, 1932070], [6356110, 1932040]]
    center = Transformer.from_crs(2226, 6417, always_xy=True).transform(6356125, 1932055)
    features = [{"attributes": {"OBJECTID": identifier}, "geometry": {"rings": [source]}} for identifier in (1, 2)]
    result, receipt = normalize_footprints(features, center)
    assert len(result) == 1 and receipt["skipped"]["duplicate_geometry"] == 1
    assert result[0]["geometry"].area == pytest.approx((30 * 1200 / 3937) ** 2, rel=1e-4)


def synthetic_grid(green=False, height=8):
    from rasterio.transform import from_bounds

    grid = object.__new__(ContextGrid)
    grid.patch_id = "test"
    grid.terrain_basis = "2022-public-DEM"
    grid.color_basis = "synthetic test color reference"
    grid.bounds = [0, 0, 16, 16]
    grid.rows = grid.columns = 16
    grid.pixel_size = 1
    grid.transform = from_bounds(0, 0, 16, 16, 16, 16)
    grid.terrain = np.full((16, 16), 50.0)
    grid.surface = grid.terrain + height
    grid.green = np.full((16, 16), green, dtype=bool)
    return grid


def test_robust_height_uses_actual_samples_not_a_silent_fallback():
    result = synthetic_grid().height(box(2, 2, 14, 14))
    assert result["approximateHeightMeters"] == 8
    assert result["heightBasis"] == "2022-dsm-minus-dem-interior-35th-percentile"
    assert result["confidence"] == "medium"


def test_tree_contamination_and_sparse_samples_have_explicit_fallbacks():
    result = synthetic_grid(green=True).height(box(2, 2, 14, 14))
    assert result["heightBasis"] == "explicit-6m-approximate-fallback"
    assert result["confidence"] == "low" and "tree_contamination_possible" in result["flags"]
    sparse = synthetic_grid().height(box(2, 2, 2.2, 2.2))
    assert sparse["heightBasis"] == "explicit-6m-approximate-fallback"


def test_outside_circle_is_not_mislabeled_as_small_building():
    feature = {"attributes": {"OBJECTID": 1}, "geometry": {"rings": [[[6356100, 1932000], [6356200, 1932000], [6356200, 1932100], [6356100, 1932100], [6356100, 1932000]]]}}
    normalized, receipt = normalize_footprints([feature], [0, 0])
    assert normalized == []
    assert receipt["skipped"] == {"outside_circle": 1}


def test_public_elevation_provenance_does_not_infer_survey_foot_from_horizontal_crs():
    manifest = {"coordinate_system": {"source_horizontal_units": "US survey feet", "vertical_datum": "NAVD88", "geoid": "GEOID18"}}
    provenance = elevation_source_units(manifest)
    assert provenance["dem"]["original_owner_source"] is None
    assert provenance["dem"]["context_raster_encoding"]["vertical_unit"] == "feet; source subtype unspecified"
    assert provenance["dem"]["public_fallback"]["vertical_scale_to_meters"] == 0.3048
    assert provenance["dem"]["public_fallback"]["scope"] == "entire context"
    assert provenance["dsm"]["vertical_scale_to_meters"] == 1.0
    assert provenance["dsm"]["vertical_datum"] == "NAVD88"


@pytest.mark.parametrize("source_key,kind,unit_key,unit,scale", [
    ("elevation_geotiff", "geotiff_elevation", "vertical_unit", "USSurveyFoot", 1200 / 3937),
    ("landxml", "landxml_tin", "linear_unit", "USSurveyFoot", 1200 / 3937),
    ("landxml", "landxml_tin", "linear_unit", "meter", 1.0),
])
def test_owner_elevation_provenance_separates_original_units_and_intermediate_encoding(source_key, kind, unit_key, unit, scale):
    owner = {"kind": kind, unit_key: unit, "vertical_scale_to_meters": scale, "vertical_datum": "NAVD88",
             "source": {"name": "terrain-source", "sha256": "a" * 64, "path": "/private/original"}}
    manifest = {"coordinate_system": {"vertical_datum": "NAVD88", "geoid": "owner realization unverified"},
                "owner_terrain_composite": {source_key: owner, "public_geoid_declaration": "GEOID18", "owner_geoid_declaration": "GEOID18",
                                            "vertical_datum_conversion": False, "geoid_compatibility_verified": False}}
    before = json.dumps(manifest, sort_keys=True)
    provenance = elevation_source_units(manifest)
    original = provenance["dem"]["original_owner_source"]
    assert original["kind"] == kind
    assert original["vertical_unit"] == unit
    assert original["vertical_scale_to_meters"] == scale
    assert original["vertical_datum"] == "NAVD88" and original["geoid"] == "GEOID18"
    assert "path" not in original and "/private/original" not in json.dumps(provenance)
    intermediate = provenance["dem"]["context_raster_encoding"]
    assert intermediate["vertical_unit"] == "international foot"
    assert intermediate["vertical_scale_to_meters"] == 0.3048
    assert provenance["dem"]["original_owner_scale_reapplied_by_reader"] is False
    assert provenance["dem"]["geoid_compatibility_verified"] is False
    assert provenance["dsm"]["vertical_unit"] == "meters"
    assert provenance["dsm"]["geoid"] == "GEOID18"
    assert provenance["dsm"]["owner_dem_unit_conversion_applied"] is False
    assert json.dumps(manifest, sort_keys=True) == before


def test_geotiff_provenance_retains_band_scale_offset_and_unknown_vertical_metadata():
    manifest = {"coordinate_system": {"source_horizontal_units": "US survey feet"},
                "owner_terrain_composite": {"elevation_geotiff": {"kind": "geotiff_elevation", "scales": [2.0], "offsets": [3.0]}}}
    original = elevation_source_units(manifest)["dem"]["original_owner_source"]
    assert original["band_scales"] == [2.0] and original["band_offsets"] == [3.0]
    assert original["vertical_unit"] is None and original["vertical_scale_to_meters"] is None
    assert original["vertical_datum"] is None and original["geoid"] is None
