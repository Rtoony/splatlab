"""Private GIS context layers in the neighborhood's existing metric frame."""

from collections import Counter
import json
import math
from pathlib import Path
import shutil

import fiona
import numpy as np
from pyproj import CRS, Transformer
import shapely
from shapely.geometry import Point, mapping, shape
from shapely.ops import transform as transform_geometry

from neighborhood_buildings import ContextGrid
from neighborhood_context import elevation_grid, sha256, utc_now, write_json
from neighborhood_inputs import LINEAR_UNITS, METRIC_CRS, local_file, owner_terrain_glb, source_record


SCHEMA = "dev.splatlab.neighborhood-context-layers/v1"
MAX_VERTICES = 150000
LAYER_SPECS = {
    "roads": ("roads", "Roads and shared approaches", "City of Santa Rosa", "Source road centerlines; not a navigability or access guarantee", "ROAD_NAME"),
    "parcels": ("parcels", "Parcel reference boundaries", "County of Sonoma", "GIS reference boundaries, not surveyed property lines", None),
    "parks": ("parks", "Parks and open-space outlines", "County of Sonoma", "Source park outlines; ground-draped display", "NAME"),
    "streams": ("streams", "Streams and channels", "Sonoma Water", "Source hydrography; ground-draped display, not current water level", "NAME"),
    "drainage": ("drainage", "Drainage context", "County of Sonoma LiDAR derivatives", "Supplementary drainage lines, not buried pipes or verified flow", None),
    "contours": ("contours5", "Five-foot terrain contours", "Owner-supplied county 2022 DEM derivative", "Elevation-labelled contours; interval is not an accuracy claim", None),
    "site": ("site", "2286 county address point", "County of Sonoma address-point record", "County address record, not a surveyed entrance or interior registration", None),
    "floodzones": ("floodzones", "Flood-map reference", "FEMA via supplied county GIS export", "Source map outlines, not a property-specific flood-risk determination", None),
    "addresses": ("addresses", "Address-location reference points", "County of Sonoma", "Location points only; no address or ownership attributes published", None),
}


def shape_sources(path):
    path = local_file(path, {".shp"}, 128 * 1024**2)
    siblings = {}
    for sibling in path.parent.iterdir():
        if sibling.stem == path.stem:
            suffix = sibling.suffix.lower()
            if sibling.is_symlink() or suffix in siblings:
                raise ValueError("Ambiguous or symlink shapefile sidecars")
            siblings[suffix] = sibling
    if any(suffix not in siblings for suffix in (".shp", ".shx", ".dbf", ".prj")):
        raise ValueError("Missing required shapefile sidecars")
    paths = [local_file(siblings[suffix], {suffix}, 128 * 1024**2) for suffix in (".shp", ".shx", ".dbf", ".prj")]
    if ".cpg" in siblings:
        paths.append(local_file(siblings[".cpg"], {".cpg"}, 1024))
    if sum(item.stat().st_size for item in paths) > 256 * 1024**2:
        raise ValueError("Shapefile packet exceeds context intake limit")
    return path, CRS.from_wkt(siblings[".prj"].read_text()), [source_record(item) for item in paths]


def sealed_path(root, relative, expected):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Sealed asset keys must be safe relative paths")
    root = Path(root).resolve()
    path = root / relative
    if path.resolve() != path or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("Sealed assets must be regular files without symlink components")
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError("Sealed asset bytes changed")
    return path


def circular_textured_mesh(mesh, radius):
    inside = np.linalg.norm(mesh["positions"][:, [0, 2]], axis=1) <= radius
    faces = mesh["indices"][np.all(inside[mesh["indices"]], axis=1)]
    if not len(faces):
        raise ValueError("Textured terrain has no faces inside context")
    used, reverse = np.unique(faces, return_inverse=True)
    return {**mesh, **{key: mesh[key][used] for key in ("positions", "normals", "uv")},
            "indices": reverse.reshape(-1, 3).astype("<u4"),
            "stats": {**mesh["stats"], "vertices": len(used), "triangles": len(faces)}}


def sample_terrain(grids, east, north):
    grid = next((item for item in reversed(grids) if item.bounds[0] <= east <= item.bounds[2] and item.bounds[1] <= north <= item.bounds[3]), None)
    if grid is None:
        raise ValueError("Context geometry falls outside terrain coverage")
    column, row = ~grid.transform * (east, north)
    column = min(max(column - 0.5, 0), grid.columns - 1)
    row = min(max(row - 0.5, 0), grid.rows - 1)
    first_row, first_column = math.floor(row), math.floor(column)
    second_row, second_column = min(first_row + 1, grid.rows - 1), min(first_column + 1, grid.columns - 1)
    fraction_row, fraction_column = row - first_row, column - first_column
    values = [grid.terrain[first_row, first_column], grid.terrain[first_row, second_column],
              grid.terrain[second_row, first_column], grid.terrain[second_row, second_column]]
    weights = [(1 - fraction_row) * (1 - fraction_column), (1 - fraction_row) * fraction_column,
               fraction_row * (1 - fraction_column), fraction_row * fraction_column]
    if any(not math.isfinite(value) for value, weight in zip(values, weights) if weight > 0):
        raise ValueError("Context geometry has no finite terrain height")
    return float(sum(value * weight for value, weight in zip(values, weights) if weight > 0))


def geometry_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "GeometryCollection":
        return [part for child in geometry.geoms for part in geometry_parts(child)]
    if geometry.geom_type not in ("Point", "LineString", "MultiLineString", "Polygon", "MultiPolygon"):
        raise ValueError("Unsupported context geometry type")
    return [geometry]


def display_feature(geometry, identifier, frame, sample_height, label=None, fixed_height=None):
    inverse = Transformer.from_crs(METRIC_CRS, 4326, always_xy=True)
    coordinates = mapping(geometry)["coordinates"]
    origin = frame["origin"]

    def local(values):
        if isinstance(values[0], (float, int)):
            east, north = values[:2]
            height = sample_height(east, north) if fixed_height is None else fixed_height
            if not all(math.isfinite(value) for value in (east, north, height)):
                raise ValueError("Display coordinates must be finite")
            return [round(east - origin[0], 4), round(height - origin[2], 4), round(origin[1] - north, 4)]
        return [local(child) for child in values]

    properties = {"localCoordinates": local(coordinates)}
    if label:
        properties["label"] = str(label)[:200]
    if fixed_height is not None:
        properties["elevationMeters"] = fixed_height
    return {"type": "Feature", "id": identifier, "geometry": mapping(transform_geometry(inverse.transform, geometry)), "properties": properties}


def build_collection(records, frame, circle, sample_height, role, contour_units=None):
    if frame.get("crs") != "EPSG:6417" or frame.get("units") != "metres" or frame.get("axes") != "east-up-south":
        raise ValueError("Context layers require the existing metric neighborhood frame")
    if len(frame.get("origin", [])) != 3 or not np.isfinite(frame["origin"]).all():
        raise ValueError("Context origin must contain three finite coordinates")
    features, counts = [], Counter()
    vertices = 0
    for record in records:
        counts["source_features"] += 1
        geometry = record["geometry"]
        if not geometry.is_valid or geometry.is_empty or not np.isfinite(geometry.bounds).all():
            counts["invalid_geometry"] += 1
            continue
        clipped = shapely.force_2d(geometry).intersection(circle)
        if clipped.is_empty:
            counts["outside_context"] += 1
            continue
        clipped = clipped.simplify(0.5, preserve_topology=True).segmentize(10)
        fixed_height = None
        if role == "contours":
            if contour_units not in LINEAR_UNITS:
                raise ValueError("Contour elevation needs explicit vertical units")
            fixed_height = float(record["elevation"]) * LINEAR_UNITS[contour_units]
            if not -100 <= fixed_height <= 1500:
                raise ValueError("Contour elevation is invalid or a missing-value sentinel")
        for index, part in enumerate(geometry_parts(clipped)):
            vertices += int(shapely.get_num_coordinates(part))
            if vertices > MAX_VERTICES or len(features) >= 10000:
                raise ValueError("Context layer exceeds bounded display features/vertices")
            features.append(display_feature(part, f"{role}-{record['id']}-{index}", frame, sample_height, record.get("label"), fixed_height))
    counts.update(display_features=len(features), display_vertices=vertices)
    return {"type": "FeatureCollection", "displayCoordinateSystem": frame, "features": features}, dict(counts)


def shape_records(path, role):
    path, declared, sources = shape_sources(path)
    records = []
    raw_vertices = 0
    label_field = LAYER_SPECS[role][4]
    with fiona.open(path, enabled_drivers=["ESRI Shapefile"]) as dataset:
        actual = CRS.from_user_input(dataset.crs_wkt or dataset.crs)
        if not declared.equals(actual, ignore_axis_order=True):
            raise ValueError("Context shapefile CRS disagrees with .prj")
        if len(dataset) > 100000:
            raise ValueError("Source context layer exceeds feature limit")
        transformer = Transformer.from_crs(actual, METRIC_CRS, always_xy=True)
        for index, feature in enumerate(dataset):
            geometry = shape(feature["geometry"])
            raw_vertices += int(shapely.get_num_coordinates(geometry))
            if raw_vertices > 2000000:
                raise ValueError("Source context layer exceeds vertex limit")
            record = {"id": str(index), "geometry": transform_geometry(transformer.transform, geometry)}
            if label_field:
                record["label"] = feature["properties"].get(label_field)
            if role == "site":
                record["label"] = "2286 county address point (not surveyed entrance)"
            if role == "contours":
                record["elevation"] = feature["properties"].get("ELEV_FT")
            records.append(record)
    return records, {"files": sources, "source_crs": declared.to_string(), "source_vertices": raw_vertices}


def gps_records(path):
    path = local_file(path, {".json", ".geojson"}, 16 * 1024**2)
    collection = json.loads(path.read_text())
    if collection.get("type") != "FeatureCollection" or "crs" in collection or not 1 <= len(collection.get("features", [])) <= 100:
        raise ValueError("GPS routes need bounded WGS84 GeoJSON")
    transformer = Transformer.from_crs(4326, METRIC_CRS, always_xy=True)
    records = []
    for index, feature in enumerate(collection["features"]):
        geometry = shapely.force_2d(shape(feature["geometry"]))
        if geometry.geom_type not in ("LineString", "MultiLineString") or shapely.get_num_coordinates(geometry) > 50000:
            raise ValueError("GPS routes must be bounded line geometry")
        coordinates = shapely.get_coordinates(geometry)
        if not np.isfinite(coordinates).all() or np.any(np.abs(coordinates[:, 0]) > 180) or np.any(np.abs(coordinates[:, 1]) > 85):
            raise ValueError("GPS longitude/latitude coordinates are invalid")
        records.append({"id": str(index), "geometry": transform_geometry(transformer.transform, geometry), "label": f"Capture GPS route {index + 1}; timing and camera registration unverified"})
    return records, {"files": [source_record(path)], "source_crs": "EPSG:4326", "camera_registration": "unregistered", "height_basis": "terrain-draped display, not measured camera altitude", "time_synchronization": "unverified; no frame cursor attached"}


def prepare_layers(context, lowpoly, shapefiles, output, gps=None, contour_units=None, resources=None):
    context, lowpoly, output = Path(context).resolve(), Path(lowpoly).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Context-layer output must be a new private directory")
    contract = json.loads((lowpoly / "condo-import.json").read_text())
    source = json.loads((context / "manifest.json").read_text())
    proof = json.loads((lowpoly / "source-receipt.json").read_text())
    if proof["source_context_manifest_sha256"] != sha256(context / "manifest.json"):
        raise ValueError("Buildings and context terrain do not have the same source")
    frame = contract["neighborhood"]["coordinateSystem"]
    projected = Transformer.from_crs(6418, METRIC_CRS, always_xy=True).transform(*source["projected_origin_xy_us_feet"])
    if frame["origin"] != [*projected, source["origin_height_m_navd88"]]:
        raise ValueError("Layer origin differs from the terrain/building frame")
    for relative, expected in source["files"].items():
        sealed_path(context, relative, expected)
    grids = [ContextGrid(context, patch, frame["origin"]) for patch in source["patches"]]
    circle = Point(frame["origin"][:2]).buffer(contract["neighborhood"]["radiusMeters"], quad_segs=128)
    sample = lambda east, north: sample_terrain(grids, east, north)
    output.mkdir(parents=True)
    for relative, expected in contract["artifacts"].items():
        path = sealed_path(lowpoly, relative, expected)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sealed_path(output, relative, expected)
    for layer in contract["neighborhood"]["layers"]:
        detail = "village/" + Path(layer["meshFile"]).name
        if detail in contract["artifacts"]:
            layer["detailMeshFile"] = detail
    if source.get("owner_terrain_composite"):
        for patch, grid in zip(source["patches"], grids):
            mesh = elevation_grid(grid.terrain, patch["bounds_epsg6418"], source["projected_origin_xy_us_feet"], frame["origin"][2])
            if patch["id"] == "neighborhood":
                mesh = circular_textured_mesh(mesh, contract["neighborhood"]["radiusMeters"])
            relative = patch["id"] + "/owner-aerial-terrain.glb"
            (output / relative).write_bytes(owner_terrain_glb(mesh, (context / patch["aerial_path"]).read_bytes(),
                                                            patch["id"] + "-owner-aerial-terrain", grid.terrain_basis))
            contract["artifacts"][relative] = {"sha256": sha256(output / relative), "bytes": (output / relative).stat().st_size}
        contract["neighborhood"]["layers"].append({"id": "owner-aerial", "title": "Owner aerial on ground terrain", "role": "aerial",
            "meshFile": "neighborhood/owner-aerial-terrain.glb", "detailMeshFile": "village/owner-aerial-terrain.glb", "defaultVisible": False})
    descriptors, receipts = [], {}
    (output / "context").mkdir()
    for role, (stem, title, attribution, basis, label_field) in LAYER_SPECS.items():
        path = Path(shapefiles) / f"{stem}.shp"
        if not path.exists():
            receipts[role] = {"status": "not-supplied"}
            continue
        records, receipt = shape_records(path, role)
        collection, counts = build_collection(records, frame, circle, sample, role, contour_units)
        relative = f"context/{role}.geojson"
        (output / relative).write_text(json.dumps(collection, separators=(",", ":"), allow_nan=False))
        if (output / relative).stat().st_size > 16 * 1024**2:
            raise ValueError("Context GeoJSON exceeds browser byte cap")
        descriptors.append({"id": role, "label": title, "role": role, "file": relative, "source": attribution, "basis": basis,
                            "defaultVisible": role in ("site", "roads")})
        receipts[role] = {**receipt, **counts, "status": "prepared-needs-review", "display_simplification_meters": 0.5,
                          "maximum_segment_meters": 10, "source_attributes_published": ["label"] if label_field else [],
                          "contour_vertical_units": contour_units if role == "contours" else None}
    if gps:
        records, receipt = gps_records(gps)
        collection, counts = build_collection(records, frame, circle, sample, "capture-route")
        relative = "context/capture-route.geojson"
        (output / relative).write_text(json.dumps(collection, separators=(",", ":"), allow_nan=False))
        descriptors.append({"id": "capture-route", "label": "360 capture GPS paths", "role": "capture-route", "file": relative,
                            "source": "Owner X5 embedded GPS", "basis": "Approximate GPS; terrain-draped display, not camera altitude. Timing and native reconstruction registration remain unverified.", "defaultVisible": True})
        receipts["capture-route"] = {**receipt, **counts, "status": "prepared-needs-review"}
    (output / "sources").mkdir(exist_ok=True)
    write_json(output / "sources/context-layer-receipt.json", {"schema": SCHEMA, "created_at": utc_now(), "coordinateSystem": frame,
               "source_context_sha256": sha256(context / "manifest.json"), "layers": receipts, "owner_attributes_published": False, "gpu_work": False})
    contract["neighborhood"]["contextLayers"] = descriptors
    if resources:
        resource_path = local_file(resources, {".json"}, 2 * 1024**2)
        resource_bytes = resource_path.read_bytes()
        catalog = json.loads(resource_bytes)
        if catalog.get("schema") != "chanate-resource-catalog/v1" or not isinstance(catalog.get("resources"), list) or len(catalog["resources"]) > 200:
            raise ValueError("Unsupported or oversized modeling resource catalog")
        relative = "context/modeling-resources.json"
        (output / relative).write_bytes(resource_bytes)
        contract["neighborhood"]["resourcesFile"] = relative
        contract["artifacts"][relative] = {"sha256": sha256(output / relative), "bytes": len(resource_bytes)}
    for descriptor in descriptors:
        path = output / descriptor["file"]
        contract["artifacts"][descriptor["file"]] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(output / "condo-import.json", contract)
    result = {"schema": SCHEMA, "status": "prepared-needs-review", "created_at": utc_now(), "coordinateSystem": frame,
              "layers": {role: {key: value for key, value in receipt.items() if key in ("status", "source_features", "display_features", "display_vertices")} for role, receipt in receipts.items()},
              "source_lowpoly_contract_sha256": sha256(lowpoly / "condo-import.json"), "contract_sha256": sha256(output / "condo-import.json"),
              "resource_catalog_included": resources is not None}
    write_json(output / "context-layer-summary.json", result)
    return result
