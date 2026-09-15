import json

import fiona
import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point, Polygon, box, mapping

import neighborhood_layers
from neighborhood_buildings import clip_owner_footprints, owner_coverage
from neighborhood_layers import build_collection, circular_textured_mesh, gps_records, sample_terrain, sealed_path, shape_records, shape_sources


FRAME = {"crs": "EPSG:6417", "origin": [1937347.0, 588889.0, 50.0], "units": "metres", "axes": "east-up-south"}
CENTER = FRAME["origin"][:2]


def test_lines_share_metric_frame_and_do_not_publish_private_properties():
    east, north = CENTER
    line = LineString([(east, north), (east + 20, north + 10)])
    collection, receipt = build_collection([{"id": "1", "geometry": line, "OWNER": "PRIVATE", "SITE_ADDR": "PRIVATE"}], FRAME,
                                          Point(CENTER).buffer(100), lambda east, north: 53, "roads")
    feature = collection["features"][0]
    assert feature["properties"]["localCoordinates"][0] == [0, 3, 0]
    assert feature["properties"]["localCoordinates"][-1] == [20, 3, -10]
    assert collection["displayCoordinateSystem"] == FRAME
    assert "PRIVATE" not in json.dumps(collection)
    assert receipt["display_features"] == 1
    json.dumps(receipt)
    expected = Transformer.from_crs(6417, 4326, always_xy=True).transform(east, north)
    assert feature["geometry"]["coordinates"][0] == pytest.approx(expected)


def test_polygon_holes_and_point_site_survive_display_conversion():
    east, north = CENTER
    polygon = Polygon(box(east, north, east + 40, north + 40).exterior.coords,
                      holes=[box(east + 10, north + 10, east + 20, north + 20).exterior.coords])
    result, _ = build_collection([{"id": "1", "geometry": polygon}], FRAME, Point(CENTER).buffer(100), lambda east, north: 50, "parcels")
    assert len(result["features"][0]["properties"]["localCoordinates"]) == 2
    site, _ = build_collection([{"id": "0", "geometry": Point(CENTER)}], FRAME, Point(CENTER).buffer(100), lambda east, north: 55, "site")
    assert site["features"][0]["properties"]["localCoordinates"] == [0, 5, 0]


def test_outside_geometry_clips_without_recentering():
    east, north = CENTER
    records = [{"id": "1", "geometry": LineString([(east - 200, north), (east + 200, north)])},
               {"id": "2", "geometry": Point(east + 200, north)}]
    result, receipt = build_collection(records, FRAME, Point(CENTER).buffer(100), lambda east, north: 50, "roads")
    assert receipt["outside_context"] == 1
    local = result["features"][0]["properties"]["localCoordinates"]
    assert local[0][0] == -100 and local[-1][0] == 100


def test_contours_use_explicit_source_elevation_not_terrain_or_sentinel():
    east, north = CENTER
    records = [{"id": "1", "geometry": LineString([(east, north), (east + 10, north)]), "elevation": 200}]
    result, _ = build_collection(records, FRAME, Point(CENTER).buffer(100), lambda east, north: 999, "contours", "USSurveyFoot")
    feature = result["features"][0]
    assert feature["properties"]["elevationMeters"] == pytest.approx(200 * 1200 / 3937)
    assert feature["properties"]["localCoordinates"][0][1] == pytest.approx(200 * 1200 / 3937 - 50, abs=0.0001)
    with pytest.raises(ValueError, match="explicit vertical"):
        build_collection(records, FRAME, Point(CENTER).buffer(100), lambda east, north: 50, "contours")
    records[0]["elevation"] = -9999
    with pytest.raises(ValueError, match="sentinel"):
        build_collection(records, FRAME, Point(CENTER).buffer(100), lambda east, north: 50, "contours", "foot")


def test_frame_and_vertex_limits_refuse(monkeypatch):
    with pytest.raises(ValueError, match="metric"):
        build_collection([], {**FRAME, "units": "feet"}, Point(CENTER).buffer(100), lambda east, north: 50, "roads")
    monkeypatch.setattr(neighborhood_layers, "MAX_VERTICES", 2)
    east, north = CENTER
    with pytest.raises(ValueError, match="bounded display"):
        build_collection([{"id": "1", "geometry": LineString([(east, north), (east + 50, north)])}], FRAME,
                         Point(CENTER).buffer(100), lambda east, north: 50, "roads")


def test_bilinear_terrain_preserves_nodata_and_refuses_outside():
    class Grid:
        bounds = [0, 0, 2, 2]
        rows = columns = 2
        transform = from_bounds(0, 0, 2, 2, 2, 2)
        terrain = np.array([[50., 60.], [70., 80.]])

    grid = Grid()
    assert sample_terrain([grid], 1, 1) == 65
    with pytest.raises(ValueError, match="coverage"):
        sample_terrain([grid], 3, 1)
    grid.terrain[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sample_terrain([grid], 1, 1)


def test_shapefile_source_requires_sidecars_and_strips_owner_fields(tmp_path):
    filename = tmp_path / "parcels.shp"
    with fiona.open(filename, "w", driver="ESRI Shapefile", crs="EPSG:6417",
                    schema={"geometry": "Polygon", "properties": {"OWNER": "str", "SITE_ADDR": "str"}}) as dataset:
        dataset.write({"geometry": mapping(box(CENTER[0], CENTER[1], CENTER[0] + 20, CENTER[1] + 20)),
                       "properties": {"OWNER": "Private owner", "SITE_ADDR": "Private address"}})
    records, receipt = shape_records(filename, "parcels")
    collection, _ = build_collection(records, FRAME, Point(CENTER).buffer(100), lambda east, north: 50, "parcels")
    assert "Private" not in json.dumps(collection)
    assert len(receipt["files"]) >= 4
    json.dumps(receipt)
    filename.with_suffix(".prj").unlink()
    with pytest.raises(ValueError, match="sidecars"):
        shape_sources(filename)


def test_gps_routes_stay_unregistered_without_camera_heights(tmp_path):
    filename = tmp_path / "gps.geojson"
    filename.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {
        "type": "LineString", "coordinates": [[-122.7177, 38.4651, 12345], [-122.7178, 38.4652, 12346]]}, "properties": {"source": "private"}}]}))
    records, receipt = gps_records(filename)
    assert not records[0]["geometry"].has_z
    assert receipt["camera_registration"] == "unregistered"
    assert "not measured camera altitude" in receipt["height_basis"]
    assert "time" not in records[0]


def test_owner_coverage_accepts_aoi_and_site_but_never_uses_site_as_area(tmp_path):
    path = tmp_path / "aoi.geojson"
    polygon = mapping(box(-122.73, 38.45, -122.70, 38.48))
    point = mapping(Point(-122.7177, 38.4651))
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": value} for value in [polygon, point]]}))
    assert owner_coverage(path).covers(Point(CENTER))
    path.write_text(json.dumps({"type": "Feature", "geometry": point}))
    with pytest.raises(ValueError, match="polygon"):
        owner_coverage(path)


def test_owner_whole_intersecting_polygons_get_clipped_derivatives_not_raw_mutation():
    geometry = box(-20, -20, 20, 20)
    layer = {"receipt": {"feature_type": "building_footprints"}, "features": [
        {"id": "owner-1", "geometry": geometry, "properties": {"NAME": "Building", "OWNER": "Private"}}]}
    clipped, skipped = clip_owner_footprints(layer, box(-10, -10, 10, 10), Point(0, 0).buffer(100))
    assert clipped["features"][0]["geometry"].area == 400
    assert layer["features"][0]["geometry"].area == 1600
    assert clipped["features"][0]["properties"]["clippedAtContextBoundary"]
    assert "OWNER" not in clipped["features"][0]["properties"]
    with pytest.raises(ValueError, match="building footprints"):
        clip_owner_footprints({**layer, "receipt": {"feature_type": "parcels"}}, geometry, geometry)


def test_sealed_asset_key_refuses_absolute_traversal_symlinks_and_changed_bytes(tmp_path):
    import hashlib

    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    expected = {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    assert sealed_path(tmp_path, "input.json", expected) == path
    for relative in (str(path), "../" + tmp_path.name + "/input.json"):
        with pytest.raises(ValueError, match="relative"):
            sealed_path(tmp_path, relative, expected)
    (tmp_path / "link.json").symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        sealed_path(tmp_path, "link.json", expected)
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="changed"):
        sealed_path(tmp_path, "input.json", expected)


def test_textured_circle_removes_unused_corners_without_changing_coordinates():
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [10, 0, 10]], dtype="float32")
    mesh = {"positions": positions, "normals": np.tile([0, 1, 0], (4, 1)), "uv": np.zeros((4, 2)),
            "indices": np.array([[0, 2, 1], [1, 2, 3]]), "stats": {}}
    clipped = circular_textured_mesh(mesh, 2)
    np.testing.assert_array_equal(clipped["positions"], positions[:3])
    assert clipped["indices"].tolist() == [[0, 2, 1]]
    assert clipped["stats"] == {"vertices": 3, "triangles": 1}
