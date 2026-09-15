import json
import struct

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds

from neighborhood_context import sha256, read_elevation, elevation_grid, glb_bytes
from neighborhood_inputs import inspect_geotiff, load_landxml, load_shapefile, merge_owner_footprints, parcel_boundaries_geojson, sample_tin_grid, prepare_owner_terrain, composite_owner_imagery, inspect_elevation_geotiff, sample_elevation_geotiff


def tin(tmp_path, units="meter", code="6417", points=None, faces="<F>10 20 30</F>"):
    path = tmp_path / "terrain.xml"
    points = points or '<P id="10">588800 1937300 50</P><P id="20">588800 1937310 51</P><P id="30">588810 1937300 52</P>'
    coordinate = f'<CoordinateSystem epsgCode="{code}"/>' if code else ""
    path.write_text(f'<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2"><Units><Metric linearUnit="{units}"/></Units>{coordinate}<Surfaces><Surface name="Ground"><Definition surfType="TIN"><Pnts>{points}</Pnts><Faces>{faces}</Faces></Definition></Surface></Surfaces></LandXML>')
    return path


def test_tin_preserves_north_east_order_and_faces(tmp_path):
    path = tin(tmp_path)
    original = path.read_bytes()
    result = load_landxml(path, vertical_datum="NAVD88")
    assert result["vertices"][0].tolist() == pytest.approx([1937300, 588800, 50])
    assert result["triangles"].tolist() == [[0, 1, 2]]
    assert result["receipt"]["triangulation_preserved"]
    assert path.read_bytes() == original


def test_tin_without_crs_does_not_claim_registration(tmp_path):
    result = load_landxml(tin(tmp_path, code=""))
    assert result["receipt"]["status"] == "needs-coordinate-or-vertical-datum-metadata"
    assert result["receipt"]["output_xy_crs"] is None


def test_international_versus_survey_foot_conflict_refuses(tmp_path):
    with pytest.raises(ValueError, match="foot subtype"):
        load_landxml(tin(tmp_path, units="foot", code="2226"))


def test_survey_foot_vertical_conversion_is_separate_and_exact(tmp_path):
    result = load_landxml(tin(tmp_path, units="USSurveyFoot", code="2226"))
    assert result["vertices"][0, 2] == pytest.approx(50 * 1200 / 3937)


@pytest.mark.parametrize("faces", ['<F>10 20 999</F>', '<F>10 10 20</F>', '<F>10 20 30 40</F>'])
def test_bad_tin_references_refuse(tmp_path, faces):
    with pytest.raises(ValueError):
        load_landxml(tin(tmp_path, faces=faces))


def test_xml_entities_are_forbidden(tmp_path):
    path = tmp_path / "bad.xml"
    path.write_text('<!DOCTYPE LandXML [<!ENTITY bad SYSTEM "file:///etc/passwd">]><LandXML>&bad;</LandXML>')
    with pytest.raises(Exception, match="Forbidden"):
        load_landxml(path)


def test_raster_metadata_and_bounded_png_preview(tmp_path):
    path = tmp_path / "aerial.tif"
    with rasterio.open(path, "w", driver="GTiff", width=20, height=10, count=3, dtype="uint8", crs="EPSG:6417", transform=from_origin(1937300, 588800, 1, 1)) as output:
        output.write(np.full((3, 10, 20), 100, dtype=np.uint8))
    receipt = inspect_geotiff(path, "imagery", tmp_path / "preview.png")
    assert receipt["shape"] == [10, 20]
    assert receipt["source_crs"]["authority"] == ("EPSG", "6417")
    assert receipt["preview_valid_fraction"] == 1


def write_shapefile(tmp_path, geometry):
    import fiona

    path = tmp_path / "owner.shp"
    with fiona.open(path, "w", driver="ESRI Shapefile", schema={"geometry": "Polygon", "properties": {"name": "str"}}, crs="EPSG:6417") as output:
        output.write({"geometry": mapping(geometry), "properties": {"name": "Owner geometry"}})
    return path


def test_shapefile_requires_projection_sidecar(tmp_path):
    path = write_shapefile(tmp_path, box(1937300, 588800, 1937310, 588810))
    path.with_suffix(".prj").unlink()
    with pytest.raises(ValueError, match=".prj"):
        load_shapefile(path, "building_footprints")


def test_shapefile_holes_and_multipolygons_are_retained(tmp_path):
    holed = Polygon([(1937300, 588800), (1937320, 588800), (1937320, 588820), (1937300, 588820)], holes=[[(1937305, 588805), (1937305, 588810), (1937310, 588810), (1937310, 588805)]])
    path = write_shapefile(tmp_path, MultiPolygon([holed, box(1937340, 588800, 1937350, 588810)]))
    result = load_shapefile(path, "building_footprints")
    geometry = result["features"][0]["geometry"]
    assert geometry.geom_type == "MultiPolygon"
    assert sum(len(part.interiors) for part in geometry.geoms) == 1


def test_parcels_can_never_replace_building_footprints(tmp_path):
    parcel = load_shapefile(write_shapefile(tmp_path, box(1937300, 588800, 1937310, 588810)), "parcels")
    with pytest.raises(ValueError, match="Parcel"):
        merge_owner_footprints([], parcel, box(1937200, 588700, 1937400, 589000), True)


def test_owner_footprints_require_validation_and_coverage(tmp_path):
    owner = load_shapefile(write_shapefile(tmp_path, box(1937300, 588800, 1937310, 588810)), "building_footprints")
    coverage = box(1937290, 588790, 1937320, 588820)
    county = [{"id": "overridden", "geometry": box(1937300, 588800, 1937310, 588810)}, {"id": "retained", "geometry": box(1937400, 588800, 1937410, 588810)}]
    with pytest.raises(ValueError, match="Owner validation"):
        merge_owner_footprints(county, owner, coverage)
    merged = merge_owner_footprints(county, owner, coverage, True)
    assert [item["id"] for item in merged] == ["retained", owner["features"][0]["id"]]


def test_sidecar_symlink_is_refused(tmp_path):
    path = write_shapefile(tmp_path, box(1937300, 588800, 1937310, 588810))
    path.with_suffix(".qix").symlink_to(path.with_suffix(".dbf"))
    with pytest.raises(ValueError, match="sidecars"):
        load_shapefile(path, "building_footprints")


def test_parcel_boundary_export_is_grounded_but_never_extruded(tmp_path):
    layer = load_shapefile(write_shapefile(tmp_path, box(1937300, 588800, 1937310, 588810)), "parcels")
    result = parcel_boundaries_geojson(layer, [1937300, 588800, 50], lambda east, north: 52)
    properties = result["features"][0]["properties"]
    assert not properties["buildingExtrusion"]
    assert all(point[1] == 2 for point in properties["localRings"][0])
    assert result["features"][0]["geometry"]["coordinates"][0][0][0] < 0


def test_tin_sampling_uses_only_supplied_faces_and_keeps_hole():
    vertices = np.array([[0, 0, 10], [4, 0, 14], [0, 4, 18], [4, 4, 22]], dtype=float)
    result, receipt = sample_tin_grid(vertices, np.array([[0, 1, 2]]), [0, 0, 4, 4], 4)
    assert result[3, 0] == pytest.approx(11.5)
    assert np.isnan(result[0, 3])
    assert receipt["owner_cells"] == 10


def test_tin_conflicting_overlap_and_excessive_work_refuse():
    vertices = np.array([[0, 0, 10], [4, 0, 10], [0, 4, 10], [0, 0, 20], [4, 0, 20], [0, 4, 20]], dtype=float)
    with pytest.raises(ValueError, match="disagree"):
        sample_tin_grid(vertices, np.array([[0, 1, 2], [3, 4, 5]]), [0, 0, 4, 4], 4)
    with pytest.raises(ValueError, match="bounded"):
        sample_tin_grid(vertices, np.array([[0, 1, 2]]), [0, 0, 4, 4], 4, maximum_checks=4)


def public_context(tmp_path):
    root = tmp_path / "public-context"
    (root / "sources").mkdir(parents=True)
    origin = [1937300, 588800, 50]
    projected = Transformer.from_crs(6417, 6418, always_xy=True)
    patches = []
    for name, half_width in (("neighborhood", 32), ("village", 16)):
        bounds = projected.transform_bounds(origin[0] - half_width, origin[1] - half_width, origin[0] + half_width, origin[1] + half_width)
        area = root / name
        area.mkdir()
        Image.new("RGB", (64, 64), (80, 100, 120)).save(area / "aerial.png")
        mesh = elevation_grid(np.full((16, 16), 64.0), bounds, projected.transform(*origin[:2]), origin[2])
        (area / "surface.glb").write_bytes(glb_bytes(mesh, (area / "aerial.png").read_bytes(), "synthetic-public-surface"))
        for kind, value in (("terrain", 50 / 0.3048), ("surface", 64)):
            with rasterio.open(root / "sources" / f"{name}-{kind}.tif", "w", driver="GTiff", width=16, height=16, count=1, dtype="float32", crs="EPSG:6418", transform=from_bounds(*bounds, 16, 16), nodata=-9999) as output:
                output.write(np.full((16, 16), value, dtype=np.float32), 1)
        patches.append({"id": name, "grid": 16, "bounds_epsg6418": bounds, "aerial_path": f"{name}/aerial.png", "meshes": {"surface": {"path": f"{name}/surface.glb"}}})
    manifest = {"schema": "dev.splatlab.neighborhood-context/v1", "status": "prepared-needs-review", "coordinate_system": {"source_vertical_units": {"terrain": "feet", "surface": "meters"}, "vertical_datum": "NAVD88"},
                "projected_origin_xy_us_feet": projected.transform(*origin[:2]), "origin_height_m_navd88": origin[2], "center": {"latitude": 38.46, "longitude": -122.71, "method": "synthetic test origin"}, "patches": patches,
                "files": {path.relative_to(root).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in root.rglob("*") if path.is_file()}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, manifest


def owner_terrain_packet(tmp_path):
    points = '<P id="10">588784 1937284 60</P><P id="20">588784 1937300 60</P><P id="30">588816 1937284 60</P><P id="40">588816 1937300 60</P>'
    terrain = tin(tmp_path, points=points, faces="<F>10 20 30</F><F>20 40 30</F>")
    imagery = tmp_path / "owner-aerial.tif"
    with rasterio.open(imagery, "w", driver="GTiff", width=16, height=32, count=3, dtype="uint8", crs="EPSG:6417", transform=from_origin(1937284, 588816, 1, 1)) as output:
        pixels = np.zeros((3, 32, 16), dtype=np.uint8)
        pixels[0] = 220
        output.write(pixels)
    return terrain, imagery


def test_owner_bridge_renders_composite_and_rebases_buildings(tmp_path, monkeypatch):
    import neighborhood_buildings
    from neighborhood_buildings import ContextGrid, prepare_lowpoly

    context, manifest = public_context(tmp_path)
    terrain, imagery = owner_terrain_packet(tmp_path)
    original_hashes = [sha256(terrain), sha256(imagery), sha256(context / "manifest.json")]
    output = tmp_path / "composite"
    receipt = prepare_owner_terrain(context, terrain, imagery, output, vertical_datum="NAVD88")
    assert receipt["status"] == "prepared-needs-owner-review"
    composite_manifest = json.loads((output / "manifest.json").read_text())
    patch = composite_manifest["patches"][1]
    values, _ = read_elevation(output / "sources/village-terrain.tif", patch["bounds_epsg6418"], 16, "feet")
    assert np.allclose(values[:, :8], 60, atol=0.0001)
    assert np.allclose(values[:, 8:], 50, atol=0.0001)
    pixels = np.asarray(Image.open(output / "village/aerial.png"))
    assert pixels[32, 8].tolist() == [220, 0, 0]
    assert pixels[32, 56].tolist() == [80, 100, 120]
    contract = json.loads((output / "terrain-import.json").read_text())
    assert contract["coordinateSystem"]["origin"] == pytest.approx([1937300, 588800, 50])
    assert contract["regenerateBuildingsRequired"]
    for relative, record in composite_manifest["files"].items():
        assert sha256(output / relative) == record["sha256"]
    payload = (output / "village/terrain.glb").read_bytes()
    assert payload[:4] == b"glTF"
    document = json.loads(payload[20:20 + struct.unpack_from("<I", payload, 12)[0]])
    assert "Owner" in document["materials"][0]["name"]
    assert document["accessors"][0]["min"][1] == pytest.approx(0, abs=0.0001)
    assert document["accessors"][0]["max"][1] == pytest.approx(10, abs=0.0001)
    grid = ContextGrid(output, patch, contract["coordinateSystem"]["origin"])
    height = grid.height(box(1937286, 588786, 1937296, 588796))
    assert height["baseElevationMeters"] == pytest.approx(59.75, abs=0.001)
    assert height["roofElevationMeters"] == pytest.approx(64, abs=0.001)
    assert height["heightBasis"] == "historic-dsm-minus-owner-public-terrain-interior-35th-percentile"
    assert "mixed_date_owner_terrain_and_historic_surface" in height["flags"]
    assert original_hashes == [sha256(terrain), sha256(imagery), sha256(context / "manifest.json")]
    source_projection = Transformer.from_crs(6417, 2226, always_xy=True)
    ring = [source_projection.transform(*point) for point in box(1937286, 588786, 1937296, 588796).exterior.coords]
    source_feature = {"attributes": {"OBJECTID": 1}, "geometry": {"rings": [ring]}}
    monkeypatch.setattr(neighborhood_buildings, "snapshot_footprints", lambda downloader, bounds: ({"copyrightText": "Synthetic county fixture; no external calls"}, [source_feature]))
    lowpoly = tmp_path / "regenerated-lowpoly"
    regenerated = prepare_lowpoly(output, lowpoly)
    assert regenerated["counts"]["neighborhood"]["buildings"] == 1
    assert regenerated["counts"]["neighborhood"]["canopy_candidates"] == 0
    assert regenerated["owner_terrain_composite_used"]
    features = json.loads((lowpoly / "building-features.geojson").read_text())["features"]
    assert features[0]["properties"]["baseElevationMeters"] == pytest.approx(59.75, abs=0.001)
    assert "mixed dates" in features[0]["properties"]["source"]
    lowpoly_contract = json.loads((lowpoly / "condo-import.json").read_text())
    assert {layer["role"] for layer in lowpoly_contract["neighborhood"]["layers"]} == {"terrain", "buildings", "surface"}
    for relative, record in lowpoly_contract["artifacts"].items():
        assert sha256(lowpoly / relative) == record["sha256"]


@pytest.mark.parametrize("datum", [None, "ellipsoidal", "NGVD29"])
def test_owner_bridge_unknown_or_incompatible_vertical_datum_refuses(tmp_path, datum):
    context, _ = public_context(tmp_path)
    terrain, imagery = owner_terrain_packet(tmp_path)
    with pytest.raises(ValueError, match="NAVD88"):
        prepare_owner_terrain(context, terrain, imagery, tmp_path / "composite", vertical_datum=datum)
    assert not (tmp_path / "composite").exists()


def test_owner_bridge_outside_tin_does_not_claim_import(tmp_path):
    context, _ = public_context(tmp_path)
    terrain, imagery = owner_terrain_packet(tmp_path)
    terrain.write_text(terrain.read_text().replace("588784", "580784").replace("588816", "580816"))
    with pytest.raises(ValueError, match="overlap"):
        prepare_owner_terrain(context, terrain, imagery, tmp_path / "composite", vertical_datum="NAVD88")


def test_imagery_reprojection_uses_own_crs_and_alpha(tmp_path):
    public = tmp_path / "fallback.png"
    Image.new("RGB", (32, 32), (80, 100, 120)).save(public)
    bounds = [1937284, 588784, 1937316, 588816]
    projected = Transformer.from_crs(6417, 3857, always_xy=True)
    source_bounds = projected.transform_bounds(1937284, 588784, 1937300, 588816)
    imagery = tmp_path / "mercator.tif"
    with rasterio.open(imagery, "w", driver="GTiff", width=16, height=32, count=4, dtype="uint8", crs="EPSG:3857", transform=from_bounds(*source_bounds, 16, 32)) as output:
        pixels = np.zeros((4, 32, 16), dtype=np.uint8)
        pixels[0] = 220
        pixels[3, 16:] = 255
        output.write(pixels)
    image, receipt = composite_owner_imagery(imagery, public, bounds)
    pixels = np.asarray(image)
    assert pixels[24, 6].tolist() == [220, 0, 0]
    assert pixels[6, 6].tolist() == [80, 100, 120]
    assert pixels[24, 28].tolist() == [80, 100, 120]
    assert 0.2 < receipt["owner_valid_fraction"] < 0.3


def test_owner_bridge_requires_sealed_fallback_assets(tmp_path):
    context, manifest = public_context(tmp_path)
    terrain, imagery = owner_terrain_packet(tmp_path)
    manifest["files"].pop("village/aerial.png")
    (context / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="sealed source hash"):
        prepare_owner_terrain(context, terrain, imagery, tmp_path / "composite", vertical_datum="NAVD88")


def elevation_tif(tmp_path, unit="US survey foot", scale=1200 / 3937, crs=6418, nodata_hole=True, band_scale=1, band_offset=0):
    path = tmp_path / "owner-dem.tif"
    bounds = Transformer.from_crs(6417, crs, always_xy=True).transform_bounds(1937284, 588784, 1937300, 588816)
    values = np.full((32, 16), (60 / scale - band_offset) / band_scale, dtype=np.float32)
    if nodata_hole:
        values[12:20, 4:12] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=16, height=32, count=1, dtype="float32", crs=f"EPSG:{crs}", transform=from_bounds(*bounds, 16, 32), nodata=-9999) as output:
        output.write(values, 1)
        if unit:
            output.set_band_unit(1, unit)
        output.scales = (band_scale,)
        output.offsets = (band_offset,)
    return path


@pytest.mark.parametrize("units,band_units,factor", [("meter", "metre", 1), ("foot", "foot", 0.3048), ("USSurveyFoot", "US survey foot", 1200 / 3937)])
def test_elevation_bridge_units_nodata_fallback_and_geometry(tmp_path, units, band_units, factor):
    from neighborhood_buildings import ContextGrid

    context, manifest = public_context(tmp_path)
    terrain = elevation_tif(tmp_path, band_units, factor)
    original_hash = sha256(terrain)
    output = tmp_path / "elevation-context"
    receipt = prepare_owner_terrain(context, terrain, None, output, vertical_datum="NAVD88", terrain_kind="elevation", vertical_units=units, geoid="GEOID18", texture_size=128)
    assert receipt["landxml"] is None
    assert receipt["elevation_geotiff"]["vertical_scale_to_meters"] == factor
    composite = json.loads((output / "manifest.json").read_text())
    assert composite["projected_origin_xy_us_feet"] == list(manifest["projected_origin_xy_us_feet"])
    assert composite["origin_height_m_navd88"] == manifest["origin_height_m_navd88"]
    values, _ = read_elevation(output / "sources/village-terrain.tif", composite["patches"][1]["bounds_epsg6418"], 16, "feet")
    assert values[13, 2] == pytest.approx(60, abs=0.001)
    assert values[8, 4] == pytest.approx(50, abs=0.001)
    assert values[8, 12] == pytest.approx(50, abs=0.001)
    assert receipt["patches"][1]["terrain"]["owner_fraction"] < 0.5
    assert receipt["patches"][1]["imagery"]["public_fallback_fraction"] == 1
    assert Image.open(output / "village/aerial.png").size == (128, 128)
    grid = ContextGrid(output, composite["patches"][1], receipt["coordinateSystem"]["origin"])
    assert grid.sample(1937287, 588787) == pytest.approx(60, abs=0.001)
    payload = (output / "village/terrain.glb").read_bytes()
    document = json.loads(payload[20:20 + struct.unpack_from("<I", payload, 12)[0]])
    assert document["accessors"][0]["max"][1] == pytest.approx(10, abs=0.001)
    assert "GeoTIFF" in document["extras"]["observed_source"]
    assert document["materials"][0]["name"] == "Historic public aerial fallback"
    assert sha256(terrain) == original_hash


def test_elevation_band_scale_offset_and_independent_crs(tmp_path):
    path = elevation_tif(tmp_path, "metre", 1, crs=3857, band_scale=0.5, band_offset=10)
    receipt = inspect_elevation_geotiff(path, "meter", "NAVD88")
    values, proof = sample_elevation_geotiff(receipt, [1937284, 588784, 1937316, 588816], 16)
    assert values[13, 2] == pytest.approx(60, abs=0.001)
    assert np.isnan(values[8, 4])
    assert np.isnan(values[8, 12])
    assert proof["band_scale"] == 0.5 and proof["band_offset"] == 10


@pytest.mark.parametrize("units", [None, "foot", "meter"])
def test_elevation_requires_explicit_matching_vertical_units(tmp_path, units):
    terrain = elevation_tif(tmp_path)
    with pytest.raises(ValueError, match="units"):
        inspect_elevation_geotiff(terrain, units, "NAVD88")


def test_elevation_crs_override_mismatch_is_refused(tmp_path):
    terrain = elevation_tif(tmp_path)
    with pytest.raises(ValueError, match="CRS conflicts"):
        inspect_elevation_geotiff(terrain, "USSurveyFoot", "NAVD88", source_crs="EPSG:3857")


def test_elevation_bridge_outside_coverage_and_missing_datum_refuse(tmp_path):
    context, _ = public_context(tmp_path)
    terrain = elevation_tif(tmp_path)
    with pytest.raises(ValueError, match="NAVD88"):
        prepare_owner_terrain(context, terrain, None, tmp_path / "unknown", terrain_kind="elevation", vertical_units="USSurveyFoot")
    with rasterio.open(terrain, "r+") as output:
        output.transform = from_origin(6400000, 1999999, 3, 3)
    with pytest.raises(ValueError, match="overlap"):
        prepare_owner_terrain(context, terrain, None, tmp_path / "outside", terrain_kind="elevation", vertical_units="USSurveyFoot", vertical_datum="NAVD88")


def test_elevation_with_owner_rgb_uses_same_writer(tmp_path):
    context, _ = public_context(tmp_path)
    terrain = elevation_tif(tmp_path)
    _, imagery = owner_terrain_packet(tmp_path)
    output = tmp_path / "elevation-and-rgb"
    receipt = prepare_owner_terrain(context, terrain, imagery, output, terrain_kind="elevation", vertical_units="USSurveyFoot", vertical_datum="NAVD88", texture_size=128)
    assert receipt["patches"][1]["imagery"]["owner_valid_fraction"] > 0
    assert np.asarray(Image.open(output / "village/aerial.png"))[64, 8].tolist() == [220, 0, 0]


def test_owner_display_rgb_retains_separate_sealed_analysis_aerial(tmp_path):
    context, manifest = public_context(tmp_path)
    terrain = elevation_tif(tmp_path)
    _, imagery = owner_terrain_packet(tmp_path)
    output = tmp_path / "separate-analysis"
    receipt = prepare_owner_terrain(context, terrain, imagery, output, terrain_kind="elevation", vertical_units="USSurveyFoot", vertical_datum="NAVD88", texture_size=128)
    composite = json.loads((output / "manifest.json").read_text())
    for patch in composite["patches"]:
        source_path = context / patch["id"] / "aerial.png"
        analysis_path = output / patch["analysis_aerial_path"]
        assert analysis_path.read_bytes() == source_path.read_bytes()
        assert sha256(analysis_path) == manifest["files"][f"{patch['id']}/aerial.png"]["sha256"]
        assert composite["files"][patch["analysis_aerial_path"]]["sha256"] == sha256(analysis_path)
        assert sha256(output / patch["aerial_path"]) != sha256(analysis_path)
        assert patch["analysis_aerial_provenance"]["source_acquisition_not_inferred_from_display_imagery"]
    assert len(receipt["analysis_imagery"]) == 2


def test_existing_analysis_aerial_survives_chained_owner_context(tmp_path):
    context, _ = public_context(tmp_path)
    terrain = elevation_tif(tmp_path)
    _, imagery = owner_terrain_packet(tmp_path)
    first = tmp_path / "first-context"
    prepare_owner_terrain(context, terrain, imagery, first, terrain_kind="elevation", vertical_units="USSurveyFoot", vertical_datum="NAVD88")
    second = tmp_path / "second-context"
    prepare_owner_terrain(first, terrain, None, second, terrain_kind="elevation", vertical_units="USSurveyFoot", vertical_datum="NAVD88")
    assert sha256(second / "village/analysis-aerial.png") == sha256(context / "village/aerial.png")
    assert sha256(second / "village/analysis-aerial.png") != sha256(first / "village/aerial.png")


@pytest.mark.parametrize("xmin,xmax", [(-5, 5), (-4, 6)])
def test_partial_owner_coverage_preserves_county_exterior_without_overlap(xmin, xmax):
    coverage = box(-10, -10, 0, 10)
    footprint = box(xmin, -5, xmax, 5)
    county = {"id": "county-1", "geometry": footprint, "properties": {"name": "Existing outline"}}
    owner = {"receipt": {"feature_type": "building_footprints"}, "features": [{"id": "owner-1", "geometry": footprint.intersection(coverage)}]}
    merged = merge_owner_footprints([county], owner, coverage, True)
    public, supplied = merged
    assert public["geometry"].union(supplied["geometry"]).equals(footprint)
    assert public["geometry"].intersection(supplied["geometry"]).area == 0
    assert public["properties"]["clippedAtOwnerCoverageBoundary"]
    assert county["geometry"].equals(footprint)
    assert "clippedAtOwnerCoverageBoundary" not in county["properties"]


def test_owner_coverage_hole_retains_public_building_geometry():
    footprint = box(0, 0, 20, 20)
    hole = box(5, 5, 15, 15)
    coverage = Polygon(footprint.exterior.coords, holes=[hole.exterior.coords])
    owner = {"receipt": {"feature_type": "building_footprints"}, "features": [{"id": "owner", "geometry": coverage}]}
    merged = merge_owner_footprints([{"id": "county", "geometry": footprint}], owner, coverage, True)
    assert merged[0]["geometry"].equals(hole)
    assert sum(feature["geometry"].area for feature in merged) == footprint.area


def test_partial_coverage_can_leave_multiple_public_parts():
    footprint = box(-10, -10, 10, 10)
    coverage = box(-2, -20, 2, 20)
    owner = {"receipt": {"feature_type": "building_footprints"}, "features": [{"id": "owner", "geometry": footprint.intersection(coverage)}]}
    merged = merge_owner_footprints([{"id": "county", "geometry": footprint}], owner, coverage, True)
    assert merged[0]["geometry"].geom_type == "MultiPolygon"
    assert len(merged[0]["geometry"].geoms) == 2
    assert sum(feature["geometry"].area for feature in merged) == footprint.area
