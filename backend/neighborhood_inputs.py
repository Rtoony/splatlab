"""Bounded local GIS intake and review-only neighborhood terrain preparation."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import struct

from defusedxml import ElementTree
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from shapely import force_2d
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

from neighborhood_context import elevation_grid, glb_bytes, read_elevation, sha256, utc_now, write_json


INPUT_SCHEMA = "dev.splatlab.neighborhood-input/v1"
METRIC_CRS = 6417
LINEAR_UNITS = {"meter": 1.0, "metre": 1.0, "foot": 0.3048, "USSurveyFoot": 1200 / 3937}
RASTER_UNIT_NAMES = {"meter": "meter", "meters": "meter", "metre": "meter", "metres": "meter", "m": "meter",
                     "foot": "foot", "feet": "foot", "internationalfoot": "foot", "ft": "foot",
                     "ussurveyfoot": "USSurveyFoot", "ussurveyfeet": "USSurveyFoot", "ftus": "USSurveyFoot"}


def local_file(path, suffixes, maximum):
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError("Expected a regular local file with the requested format, not a symlink or URL")
    if not 0 < path.stat().st_size <= maximum:
        raise ValueError("Input file is empty or exceeds its bounded intake size")
    return path.resolve()


def source_record(path):
    path = Path(path)
    return {"name": path.name, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def crs_record(crs):
    crs = CRS.from_user_input(crs)
    return {"authority": crs.to_authority(), "name": crs.name, "wkt": crs.to_wkt(), "projected": crs.is_projected,
            "axes": [{"name": axis.name, "direction": axis.direction, "unit": axis.unit_name, "to_si": axis.unit_conversion_factor} for axis in crs.axis_info]}


def load_landxml(path, surface_name=None, source_crs=None, vertical_datum=None):
    path = local_file(path, {".xml", ".landxml"}, 64 * 1024 * 1024)
    content = path.read_bytes()
    root = ElementTree.fromstring(content, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    if root.tag.rsplit("}", 1)[-1] != "LandXML":
        raise ValueError("Expected LandXML root")
    units = root.find("{*}Units")
    declarations = [] if units is None else list(units)
    if len(declarations) != 1 or declarations[0].get("linearUnit") not in LINEAR_UNITS:
        raise ValueError("Explicit supported LandXML linearUnit is required")
    linear_unit = declarations[0].get("linearUnit")
    scale = LINEAR_UNITS[linear_unit]
    coordinate = root.find("{*}CoordinateSystem")
    declared_code = None if coordinate is None else coordinate.get("epsgCode")
    if source_crs is not None and declared_code is not None and CRS.from_user_input(source_crs) != CRS.from_epsg(int(declared_code)):
        raise ValueError("Supplied CRS conflicts with the LandXML EPSG declaration")
    resolved_crs = CRS.from_user_input(source_crs or (f"EPSG:{declared_code}" if declared_code else "")) if source_crs or declared_code else None
    if resolved_crs is not None:
        if not resolved_crs.is_projected or not math.isclose(resolved_crs.axis_info[0].unit_conversion_factor, scale, rel_tol=1e-10):
            raise ValueError("LandXML linearUnit conflicts with its projected CRS; do not guess foot subtype")
    surfaces = root.findall("{*}Surfaces/{*}Surface")
    matches = [surface for surface in surfaces if surface_name is None or surface.get("name") == surface_name]
    if len(matches) != 1:
        raise ValueError("Select exactly one named LandXML surface")
    surface = matches[0]
    definition = surface.find("{*}Definition")
    if definition is None or definition.get("surfType") != "TIN":
        raise ValueError("Only explicit TIN surfaces are supported; no guessed retriangulation")
    points = definition.findall("{*}Pnts/{*}P")
    faces = definition.findall("{*}Faces/{*}F")
    if not 3 <= len(points) <= 250000 or not 1 <= len(faces) <= 500000:
        raise ValueError("TIN point/face count is absent or exceeds bounded intake")
    identifiers = {}
    coordinates = []
    for point in points:
        identifier = point.get("id")
        if not identifier or identifier in identifiers:
            raise ValueError("TIN point IDs must be present and unique")
        values = [float(value) for value in (point.text or "").split()]
        if len(values) != 3 or not all(math.isfinite(value) and abs(value) < 1e9 for value in values):
            raise ValueError("TIN points must have finite northing, easting, elevation")
        identifiers[identifier] = len(coordinates)
        coordinates.append([values[1], values[0], values[2]])
    vertices = np.asarray(coordinates, dtype=np.float64)
    if resolved_crs is not None:
        transformer = Transformer.from_crs(resolved_crs, METRIC_CRS, always_xy=True)
        vertices[:, 0], vertices[:, 1] = transformer.transform(vertices[:, 0], vertices[:, 1])
    else:
        vertices[:, :2] *= scale
    vertices[:, 2] *= scale
    triangles = []
    ignored = 0
    for face in faces:
        if face.get("i") in ("1", "true"):
            ignored += 1
            continue
        references = (face.text or "").split()
        if len(references) != 3 or len(set(references)) != 3 or any(reference not in identifiers for reference in references):
            raise ValueError("TIN faces must reference three distinct existing points")
        triangle = [identifiers[reference] for reference in references]
        positions = vertices[triangle]
        normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
        if abs(normal[2]) < 1e-9:
            raise ValueError("Degenerate or vertical TIN faces are not usable terrain")
        if normal[2] < 0:
            triangle.reverse()
        triangles.append(triangle)
    if not triangles or not np.isfinite(vertices).all():
        raise ValueError("TIN has no usable triangles or its projection failed")
    receipt = {"schema": INPUT_SCHEMA, "kind": "landxml_tin", "inspected_at": utc_now(), "source": source_record(path),
               "surface_name": surface.get("name"), "available_surfaces": [item.get("name") for item in surfaces],
               "source_coordinate_order": "northing,easting,elevation", "linear_unit": linear_unit, "vertical_scale_to_meters": scale,
               "source_crs": None if resolved_crs is None else crs_record(resolved_crs),
               "output_xy_crs": "EPSG:6417" if resolved_crs else None, "vertical_datum": vertical_datum,
               "status": "inspectable-needs-owner-review" if resolved_crs and vertical_datum else "needs-coordinate-or-vertical-datum-metadata",
               "vertices": len(vertices), "triangles": len(triangles), "ignored_face_flags": ignored,
               "bounds_xyz_meters": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
               "triangulation_preserved": True, "original_changed": False,
               "warnings": ["No vertical datum conversion is performed.", "Supplied TIN is not automatically registered or accepted."]}
    return {"receipt": receipt, "vertices": vertices, "triangles": np.asarray(triangles, dtype=np.uint32)}


def inspect_geotiff(path, role, preview_path=None):
    path = local_file(path, {".tif", ".tiff"}, 8 * 1024 * 1024 * 1024)
    if role not in ("imagery", "elevation"):
        raise ValueError("GeoTIFF role must explicitly be imagery or elevation")
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=32 * 1024 * 1024):
        with rasterio.open(path) as dataset:
            if dataset.driver != "GTiff" or not dataset.crs or not 1 <= dataset.count <= 8:
                raise ValueError("A real GeoTIFF with its own CRS and bounded band count is required")
            if max(dataset.shape) > 200000:
                raise ValueError("Raster dimensions exceed bounded intake")
            crs = CRS.from_user_input(dataset.crs)
            transformer = Transformer.from_crs(crs, METRIC_CRS, always_xy=True)
            bounds = transformer.transform_bounds(*dataset.bounds, densify_pts=21)
            if not all(math.isfinite(value) for value in bounds):
                raise ValueError("GeoTIFF bounds cannot be projected into the neighborhood frame")
            receipt = {"schema": INPUT_SCHEMA, "kind": "geotiff", "role": role, "inspected_at": utc_now(), "source": source_record(path),
                       "source_crs": crs_record(crs), "source_bounds": list(dataset.bounds), "bounds_epsg6417_meters": list(bounds),
                       "shape": list(dataset.shape), "bands": dataset.count, "dtypes": list(dataset.dtypes),
                       "color_interpretations": [value.name for value in dataset.colorinterp], "overviews": [dataset.overviews(index) for index in dataset.indexes],
                       "band_units": list(dataset.units), "scales": list(dataset.scales), "offsets": list(dataset.offsets),
                       "nodata": dataset.nodata if dataset.nodata is None or math.isfinite(dataset.nodata) else str(dataset.nodata),
                       "affine": list(dataset.transform)[:6], "status": "inspectable-needs-owner-review", "original_changed": False,
                       "warnings": ["Each input retains its own CRS; no same-CRS assumption with LandXML or footprints.", "Elevation units/datum are not inferred from horizontal units."]}
            if preview_path is not None:
                preview_path = Path(preview_path)
                if preview_path.exists() or preview_path.is_symlink() or role != "imagery":
                    raise ValueError("Preview requires imagery role and a new destination")
                if dataset.count < 3 or any(value != "uint8" for value in dataset.dtypes[:3]):
                    raise ValueError("Automatic preview requires three uint8 color bands; no guessed radiometric stretch")
                factor = min(1.0, 2048 / max(dataset.shape))
                rows, columns = max(1, round(dataset.height * factor)), max(1, round(dataset.width * factor))
                pixels = dataset.read([1, 2, 3], out_shape=(3, rows, columns), resampling=Resampling.bilinear)
                mask = dataset.dataset_mask(out_shape=(rows, columns), resampling=Resampling.nearest)
                image = Image.fromarray(np.moveaxis(pixels, 0, -1)).convert("RGBA")
                image.putalpha(Image.fromarray(mask))
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(preview_path, format="PNG")
                receipt["preview"] = source_record(preview_path)
                receipt["preview_valid_fraction"] = float(np.mean(mask > 0))
    return receipt


def load_shapefile(path, feature_type):
    import fiona

    if feature_type not in ("building_footprints", "parcels"):
        raise ValueError("Explicit feature_type must be building_footprints or parcels")
    path = local_file(path, {".shp"}, 128 * 1024 * 1024)
    candidates = [sibling for sibling in path.parent.iterdir() if sibling.name.startswith(path.stem + ".")]
    if any(sibling.is_symlink() for sibling in candidates):
        raise ValueError("Shapefile sidecars must not be symlinks")
    siblings = {}
    for sibling in candidates:
        if sibling.stem == path.stem:
            if sibling.suffix.lower() in siblings:
                raise ValueError("Ambiguous duplicate shapefile sidecar extensions")
            siblings[sibling.suffix.lower()] = sibling
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        if suffix not in siblings:
            raise ValueError(f"Shapefile packet is missing required {suffix} sidecar")
    sidecars = [local_file(siblings[suffix], {suffix}, 128 * 1024 * 1024) for suffix in (".shp", ".shx", ".dbf", ".prj")]
    if ".cpg" in siblings:
        sidecars.append(local_file(siblings[".cpg"], {".cpg"}, 1024))
    if sum(item.stat().st_size for item in sidecars) > 256 * 1024 * 1024:
        raise ValueError("Shapefile packet exceeds bounded intake")
    declared = CRS.from_wkt(siblings[".prj"].read_text())
    features = []
    with fiona.open(path, enabled_drivers=["ESRI Shapefile"]) as dataset:
        if len(dataset) > 100000 or dataset.schema["geometry"] not in ("Polygon", "MultiPolygon", "3D Polygon"):
            raise ValueError("Only bounded polygon shapefile layers are supported")
        actual = CRS.from_user_input(dataset.crs_wkt or dataset.crs)
        if not actual.equals(declared, ignore_axis_order=True):
            raise ValueError("Shapefile CRS disagrees with its .prj sidecar")
        transformer = Transformer.from_crs(actual, METRIC_CRS, always_xy=True)
        for feature in dataset:
            geometry = force_2d(shape(feature["geometry"]))
            if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty or not geometry.is_valid:
                raise ValueError("Shapefile contains invalid or non-polygon geometry")
            geometry = transform_geometry(transformer.transform, geometry)
            if not all(math.isfinite(value) for value in geometry.bounds):
                raise ValueError("Shapefile projection produced invalid coordinates")
            identifier = f"owner-{feature_type}-{feature['id']}"
            features.append({"id": identifier, "geometry": geometry, "properties": dict(feature["properties"]), "feature_type": feature_type})
    receipt = {"schema": INPUT_SCHEMA, "kind": "shapefile", "feature_type": feature_type, "inspected_at": utc_now(),
               "sources": [source_record(item) for item in sidecars], "source_crs": crs_record(declared), "output_xy_crs": "EPSG:6417",
               "features": len(features), "status": "inspectable-needs-owner-validation", "owner_validated": False,
               "building_extrusion_permitted": feature_type == "building_footprints", "original_changed": False, "source_z_used_as_height": False,
               "warnings": ["Parcel polygons are boundaries, never building footprints.", "Owner-provided files do not automatically replace county geometry or establish an accepted coordinate alignment."]}
    return {"receipt": receipt, "features": features}


def merge_owner_footprints(county_features, owner_layer, coverage, owner_validated=False):
    if owner_layer["receipt"].get("feature_type") != "building_footprints":
        raise ValueError("Parcel boundaries must never be extruded as buildings")
    if owner_validated is not True or coverage is None or not coverage.is_valid or coverage.is_empty:
        raise ValueError("Owner validation and an explicit valid coverage polygon are required for replacement")
    owners = owner_layer["features"]
    if any(not coverage.covers(item["geometry"]) for item in owners):
        raise ValueError("Owner footprint extends outside declared replacement coverage")
    retained = []
    for item in county_features:
        original = item["geometry"]
        if original.geom_type not in ("Polygon", "MultiPolygon") or not original.is_valid:
            raise ValueError("County building footprint must be valid polygon geometry")
        remainder = original.difference(coverage)
        if remainder.is_empty:
            continue
        if remainder.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("Footprint coverage subtraction produced unexpected non-polygon geometry")
        retained.append({**item, "geometry": remainder, "properties": {**item.get("properties", {}),
                         "clippedAtOwnerCoverageBoundary": not remainder.equals(original)}})
    return retained + owners


def parcel_boundaries_geojson(layer, origin, sample_height):
    if layer["receipt"].get("feature_type") != "parcels":
        raise ValueError("Boundary export requires an explicitly typed parcel layer")
    if len(origin) != 3 or not all(math.isfinite(value) for value in origin):
        raise ValueError("Boundary origin must contain three finite metric coordinates")
    inverse = Transformer.from_crs(METRIC_CRS, 4326, always_xy=True)
    features = []
    for feature in layer["features"]:
        geometry = feature["geometry"]
        polygons = [geometry] if geometry.geom_type == "Polygon" else list(geometry.geoms)
        local_rings = []
        for polygon in polygons:
            for ring in [polygon.exterior, *polygon.interiors]:
                points = []
                for east, north in ring.coords:
                    height = float(sample_height(east, north))
                    if not math.isfinite(height):
                        raise ValueError("Parcel boundary lacks finite terrain elevation")
                    points.append([east - origin[0], height - origin[2], origin[1] - north])
                local_rings.append(points)
        features.append({"type": "Feature", "id": feature["id"], "geometry": mapping(transform_geometry(inverse.transform, geometry)),
                         "properties": {"featureId": feature["id"], "featureType": "parcel_boundary", "source": "Owner-provided parcel polygon; display boundary only",
                                        "localRings": local_rings, "localAxes": "east-up-south", "localUnits": "metres", "buildingExtrusion": False}})
    return {"type": "FeatureCollection", "features": features}


def sample_tin_grid(vertices, triangles, bounds, grid, maximum_checks=5000000):
    if not 2 <= grid <= 256 or len(bounds) != 4 or not np.isfinite(bounds).all():
        raise ValueError("TIN sampling needs a bounded finite target grid")
    west, south, east, north = bounds
    if east <= west or north <= south:
        raise ValueError("TIN sampling bounds must have positive area")
    result = np.full((grid, grid), np.nan, dtype=np.float64)
    step_east, step_north = (east - west) / grid, (north - south) / grid
    points = vertices[triangles]
    minima, maxima = points[:, :, :2].min(axis=1), points[:, :, :2].max(axis=1)
    candidates = np.where((maxima[:, 0] >= west) & (minima[:, 0] <= east) & (maxima[:, 1] >= south) & (minima[:, 1] <= north))[0]
    checks = 0
    for index in candidates:
        first_column = max(0, math.ceil((minima[index, 0] - west) / step_east - 0.50000001))
        last_column = min(grid - 1, math.floor((maxima[index, 0] - west) / step_east - 0.49999999))
        first_row = max(0, math.ceil((north - maxima[index, 1]) / step_north - 0.50000001))
        last_row = min(grid - 1, math.floor((north - minima[index, 1]) / step_north - 0.49999999))
        if first_column > last_column or first_row > last_row:
            continue
        checks += (last_column - first_column + 1) * (last_row - first_row + 1)
        if checks > maximum_checks:
            raise ValueError("TIN interpolation exceeds bounded candidate checks; simplify a separate source export")
        columns, rows = np.meshgrid(np.arange(first_column, last_column + 1), np.arange(first_row, last_row + 1))
        query_east = west + (columns + 0.5) * step_east
        query_north = north - (rows + 0.5) * step_north
        anchor, second, third = points[index]
        edge_second, edge_third = second - anchor, third - anchor
        denominator = edge_second[0] * edge_third[1] - edge_third[0] * edge_second[1]
        if abs(denominator) < 1e-9:
            raise ValueError("TIN has degenerate projected triangles")
        offset_east, offset_north = query_east - anchor[0], query_north - anchor[1]
        weight_second = (offset_east * edge_third[1] - offset_north * edge_third[0]) / denominator
        weight_third = (edge_second[0] * offset_north - edge_second[1] * offset_east) / denominator
        inside = (weight_second >= -1e-8) & (weight_third >= -1e-8) & (weight_second + weight_third <= 1 + 1e-8)
        interpolated = anchor[2] + weight_second * edge_second[2] + weight_third * edge_third[2]
        existing = result[rows[inside], columns[inside]]
        if np.any(np.isfinite(existing) & (np.abs(existing - interpolated[inside]) > 0.05)):
            raise ValueError("Overlapping TIN faces disagree in elevation")
        result[rows[inside], columns[inside]] = interpolated[inside]
    return result, {"candidate_checks": checks, "owner_cells": int(np.isfinite(result).sum()),
                    "owner_fraction": float(np.isfinite(result).mean()), "source_faces_used_without_retriangulation": True,
                    "output_is_resampled_grid_not_original_tin": True}


def composite_owner_imagery(path, public_image, bounds, texture_size=None):
    with Image.open(public_image) as original:
        if max(original.size) > 2048:
            raise ValueError("Public fallback texture exceeds bounded dimensions")
        fallback = original.convert("RGB")
    if texture_size is not None:
        if not isinstance(texture_size, int) or not 16 <= texture_size <= 2048:
            raise ValueError("Requested texture size must be 16..2048 pixels")
        fallback = fallback.resize((texture_size, texture_size), Image.Resampling.BILINEAR)
    width, height = fallback.size
    if path is None:
        return fallback, {"owner_valid_fraction": 0.0, "public_fallback_fraction": 1.0, "shape": [height, width], "crs": "EPSG:6417",
                          "resampling": "public fallback only", "owner_imagery_used": False}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=32 * 1024 * 1024):
        with rasterio.open(path) as source:
            if source.count < 3 or any(dtype != "uint8" for dtype in source.dtypes[:3]):
                raise ValueError("Owner imagery requires uint8 RGB; no guessed radiometric stretch")
            if [interpretation.name for interpretation in source.colorinterp[:3]] != ["red", "green", "blue"]:
                raise ValueError("Owner imagery must explicitly identify RGB color bands")
            has_alpha = any(interpretation.name == "alpha" for interpretation in source.colorinterp)
            with WarpedVRT(source, crs="EPSG:6417", transform=from_bounds(*bounds, width, height), width=width, height=height,
                           resampling=Resampling.bilinear, add_alpha=not has_alpha, warp_mem_limit=32) as warped:
                pixels = warped.read([1, 2, 3])
                mask = warped.dataset_mask()
    replacement = Image.fromarray(np.moveaxis(pixels, 0, -1)).convert("RGBA")
    replacement.putalpha(Image.fromarray(mask))
    image = Image.alpha_composite(fallback.convert("RGBA"), replacement).convert("RGB")
    return image, {"owner_valid_fraction": float(np.mean(mask > 0)), "public_fallback_fraction": float(np.mean(mask == 0)),
                   "shape": [height, width], "crs": "EPSG:6417", "resampling": "bilinear RGB, source validity/alpha retained"}


def raster_vertical_scale(unit):
    normalized = "" if not isinstance(unit, str) else "".join(character for character in unit.lower() if character.isalnum())
    canonical = RASTER_UNIT_NAMES.get(normalized)
    if canonical is None:
        raise ValueError("Explicit supported elevation units are required; do not infer height units from horizontal CRS")
    return canonical, LINEAR_UNITS[canonical]


def inspect_elevation_geotiff(path, vertical_units, vertical_datum, geoid=None, source_crs=None):
    receipt = inspect_geotiff(path, "elevation")
    canonical, vertical_scale = raster_vertical_scale(vertical_units)
    if receipt["bands"] != 1 or np.dtype(receipt["dtypes"][0]).kind not in "fiu":
        raise ValueError("Terrain preparation requires one real numeric elevation band")
    declared = receipt["band_units"][0]
    if declared:
        _, declared_scale = raster_vertical_scale(declared)
        if not math.isclose(vertical_scale, declared_scale, rel_tol=1e-12):
            raise ValueError("Supplied vertical units conflict with the raster band; foot subtypes are distinct")
    if source_crs is not None and not CRS.from_user_input(source_crs).equals(CRS.from_wkt(receipt["source_crs"]["wkt"]), ignore_axis_order=True):
        raise ValueError("Supplied CRS conflicts with GeoTIFF metadata; no silent override")
    if any(not math.isfinite(value) for value in [*receipt["scales"], *receipt["offsets"]]) or receipt["scales"][0] <= 0:
        raise ValueError("Elevation band scale/offset must be finite with positive scale")
    receipt.update(kind="geotiff_elevation", vertical_unit=canonical, vertical_scale_to_meters=vertical_scale,
                   vertical_units_source="explicit argument matched band metadata" if declared else "explicit argument; band has no unit declaration",
                   vertical_datum=vertical_datum, geoid=geoid, output_xy_crs="EPSG:6417", vertical_datum_conversion=False)
    return receipt


def sample_elevation_geotiff(receipt, bounds, grid):
    if not isinstance(grid, int) or not 2 <= grid <= 256 or len(bounds) != 4 or not np.isfinite(bounds).all() or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Elevation warp requires bounded positive target dimensions")
    options = {"crs": "EPSG:6417", "transform": from_bounds(*bounds, grid, grid), "width": grid, "height": grid,
               "dtype": "float64", "nodata": float("nan"), "warp_mem_limit": 32}
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=32 * 1024 * 1024):
        with rasterio.open(receipt["source"]["path"]) as source:
            with WarpedVRT(source, resampling=Resampling.bilinear, **options) as warped:
                samples = warped.read(1, masked=True).filled(np.nan)
            with WarpedVRT(source, resampling=Resampling.nearest, **options) as validity:
                nearest = validity.read(1, masked=True).filled(np.nan)
    samples[~np.isfinite(nearest)] = np.nan
    values = (samples * receipt["scales"][0] + receipt["offsets"][0]) * receipt["vertical_scale_to_meters"]
    return values, {"owner_cells": int(np.isfinite(values).sum()), "owner_fraction": float(np.isfinite(values).mean()),
                    "source_kind": "geotiff_elevation", "source_vertical_units": receipt["vertical_unit"], "vertical_scale_to_meters": receipt["vertical_scale_to_meters"],
                    "band_scale": receipt["scales"][0], "band_offset": receipt["offsets"][0],
                    "resampling": "bilinear elevation with nearest source-validity guard; source nodata and outside coverage retain public fallback",
                    "output_is_resampled_grid_not_original_raster": True}


def owner_terrain_glb(mesh, texture, name, terrain_basis="owner-TIN/public-DEM-composite", owner_imagery=True):
    payload = glb_bytes(mesh, texture, name)
    json_size = struct.unpack_from("<I", payload, 12)[0]
    document = json.loads(payload[20:20 + json_size])
    document["asset"]["generator"] = "SplatLab bounded owner/public terrain composite"
    document["materials"][0]["name"] = "Owner aerial with historic public fallback" if owner_imagery else "Historic public aerial fallback"
    document["extras"]["observed_source"] = terrain_basis + "; resampled, public DEM outside valid owner coverage"
    document["extras"]["owner_accepted"] = False
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    remainder = payload[20 + json_size:]
    return struct.pack("<4sII", b"glTF", 2, 20 + len(encoded) + len(remainder)) + struct.pack("<I4s", len(encoded), b"JSON") + encoded + remainder


def prepare_owner_terrain(context, landxml_path, imagery_path, output, surface_name=None, source_crs=None, vertical_datum=None,
                          terrain_kind="landxml", vertical_units=None, geoid=None, texture_size=None):
    context, output = Path(context).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Owner terrain requires a new output directory; originals and prior packages stay intact")
    manifest = json.loads((context / "manifest.json").read_text())
    coordinates = manifest.get("coordinate_system", {})
    if manifest.get("schema") != "dev.splatlab.neighborhood-context/v1" or manifest.get("status") != "prepared-needs-review" or coordinates.get("source_vertical_units") != {"terrain": "feet", "surface": "meters"}:
        raise ValueError("Owner bridge requires the corrected verified public context")
    if coordinates.get("vertical_datum") != "NAVD88" or vertical_datum != "NAVD88":
        raise ValueError("Explicit NAVD88 vertical datum is required for compositing; unknown datums remain separate inspection inputs")
    if [patch.get("id") for patch in manifest.get("patches", [])] != ["neighborhood", "village"]:
        raise ValueError("Expected bounded neighborhood and village patches")
    required_files = set()
    for patch in manifest["patches"]:
        required_files.update([f"sources/{patch['id']}-terrain.tif", f"sources/{patch['id']}-surface.tif", patch["aerial_path"], patch["meshes"]["surface"]["path"]])
        if patch.get("analysis_aerial_path"):
            required_files.add(patch["analysis_aerial_path"])
    if not required_files.issubset(manifest.get("files", {})):
        raise ValueError("Every required public fallback file must have a sealed source hash")
    for relative, record in manifest.get("files", {}).items():
        target = (context / relative).resolve()
        if not target.is_relative_to(context) or target.stat().st_size != record["bytes"] or sha256(target) != record["sha256"]:
            raise ValueError("Verified public fallback artifact changed")
    if terrain_kind == "landxml":
        terrain = load_landxml(landxml_path, surface_name, source_crs, vertical_datum)
        if terrain["receipt"]["output_xy_crs"] != "EPSG:6417":
            raise ValueError("TIN must have an explicit projected CRS before scene compositing")
        terrain_basis = "owner-TIN/public-DEM-composite"
    elif terrain_kind == "elevation":
        if surface_name is not None:
            raise ValueError("Named TIN surface does not apply to elevation GeoTIFF")
        terrain = {"receipt": inspect_elevation_geotiff(landxml_path, vertical_units, vertical_datum, geoid, source_crs)}
        terrain_basis = "owner-elevation-GeoTIFF/public-DEM-composite"
    else:
        raise ValueError("Terrain kind must be landxml or elevation")
    imagery = inspect_geotiff(imagery_path, "imagery") if imagery_path is not None else None
    projected = Transformer.from_crs(6418, METRIC_CRS, always_xy=True)
    origin_xy = projected.transform(*manifest["projected_origin_xy_us_feet"])
    origin = [*origin_xy, manifest["origin_height_m_navd88"]]
    frame = {"crs": "EPSG:6417", "origin": origin, "units": "metres", "axes": "east-up-south"}
    prepared = []
    for patch in manifest["patches"]:
        bounds, grid = patch["bounds_epsg6418"], patch["grid"]
        if not isinstance(grid, int) or not 2 <= grid <= 256:
            raise ValueError("Public fallback grid exceeds the bounded bridge")
        metric_bounds = projected.transform_bounds(*bounds)
        if terrain_kind == "landxml":
            owner_values, tin_receipt = sample_tin_grid(terrain["vertices"], terrain["triangles"], metric_bounds, grid)
        else:
            owner_values, tin_receipt = sample_elevation_geotiff(terrain["receipt"], metric_bounds, grid)
        public_values, _ = read_elevation(context / "sources" / f"{patch['id']}-terrain.tif", bounds, grid, "feet")
        owner_mask = np.isfinite(owner_values)
        if np.any(owner_mask & ((owner_values < -100) | (owner_values > 1500))):
            raise ValueError("Owner elevations fall outside the bounded neighborhood domain")
        combined = np.where(owner_mask, owner_values, public_values)
        differences = (owner_values - public_values)[owner_mask & np.isfinite(public_values)]
        tin_receipt["owner_minus_public_meters"] = None if not len(differences) else {"min": float(differences.min()), "median": float(np.median(differences)), "max": float(differences.max())}
        aerial_path = (context / patch["aerial_path"]).resolve()
        if not aerial_path.is_relative_to(context):
            raise ValueError("Public imagery path escapes its package")
        image, imagery_receipt = composite_owner_imagery(None if imagery is None else imagery["source"]["path"], aerial_path, metric_bounds, texture_size)
        prepared.append({"patch": patch, "terrain": combined, "image": image, "tin": tin_receipt, "imagery": imagery_receipt})
    if not any(item["tin"]["owner_cells"] for item in prepared) or (imagery is not None and not any(item["imagery"]["owner_valid_fraction"] > 0 for item in prepared)):
        raise ValueError("Owner terrain and any supplied imagery must overlap the neighborhood; no all-public result claimed as an import")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    receipt = {"schema": "dev.splatlab.owner-terrain-composite/v1", "status": "preparing", "created_at": utc_now(),
               "coordinateSystem": frame, "source_context_manifest_sha256": sha256(context / "manifest.json"),
               "landxml": terrain["receipt"] if terrain_kind == "landxml" else None,
               "elevation_geotiff": terrain["receipt"] if terrain_kind == "elevation" else None,
               "terrain_basis": terrain_basis, "imagery": imagery, "originals_changed": False, "owner_accepted": False,
               "vertical_datum": "NAVD88", "vertical_datum_conversion": False, "geoid_compatibility_verified": False,
               "owner_geoid_declaration": geoid, "public_geoid_declaration": coordinates.get("geoid"),
               "geoid_declarations_match": geoid is not None and geoid == coordinates.get("geoid"),
               "warnings": ["Owner terrain replaces public terrain only at valid supplied coverage; holes retain public fallback.",
                            "Output is a bounded resampled terrain grid, not the original detailed TIN or full-resolution source raster.",
                            "Historic DSM is retained; regenerate building bases/heights and canopy candidates using this context before publication.",
                            "Mixed capture dates and unknown geoid realization require owner review; no survey registration is claimed."]}
    try:
        patches = []
        for item in prepared:
            patch, name = item["patch"], item["patch"]["id"]
            area = output / name
            area.mkdir()
            item["image"].save(area / "aerial.png")
            analysis_relative = patch.get("analysis_aerial_path", patch["aerial_path"])
            analysis_source = (context / analysis_relative).resolve()
            if not analysis_source.is_relative_to(context):
                raise ValueError("Analysis imagery path escapes the sealed fallback package")
            shutil.copyfile(analysis_source, area / "analysis-aerial.png")
            analysis_basis = patch.get("analysis_aerial_basis", "retained source-context aerial for historic DSM analysis; not the new display orthophoto")
            analysis_receipt = {"source_context_manifest_sha256": sha256(context / "manifest.json"), "source_path": analysis_relative,
                                "sha256": sha256(analysis_source), "bytes": analysis_source.stat().st_size,
                                "basis": analysis_basis, "source_acquisition_not_inferred_from_display_imagery": True,
                                "purpose": "historic DSM color/vegetation screening only; not measured current tree identities"}
            values = np.where(np.isfinite(item["terrain"]), item["terrain"] / 0.3048, -9999).astype("float32")
            with rasterio.open(output / "sources" / f"{name}-terrain.tif", "w", driver="GTiff", width=patch["grid"], height=patch["grid"], count=1,
                               dtype="float32", crs="EPSG:6418", transform=from_bounds(*patch["bounds_epsg6418"], patch["grid"], patch["grid"]), nodata=-9999, compress="deflate") as raster:
                raster.write(values, 1)
                raster.update_tags(vertical_units="feet", vertical_scale_to_meters="0.3048", vertical_datum="NAVD88", basis=terrain_basis,
                                   source_vertical_units=terrain["receipt"].get("vertical_unit", terrain["receipt"].get("linear_unit")))
            shutil.copyfile(context / "sources" / f"{name}-surface.tif", output / "sources" / f"{name}-surface.tif")
            surface = (context / patch["meshes"]["surface"]["path"]).resolve()
            if not surface.is_relative_to(context):
                raise ValueError("Public surface path escapes its package")
            shutil.copyfile(surface, area / "surface.glb")
            mesh = elevation_grid(item["terrain"], patch["bounds_epsg6418"], manifest["projected_origin_xy_us_feet"], origin[2])
            (area / "terrain.glb").write_bytes(owner_terrain_glb(mesh, (area / "aerial.png").read_bytes(), name + "-owner-terrain", terrain_basis, imagery is not None))
            patches.append({**patch, "aerial_path": f"{name}/aerial.png", "terrain_basis": terrain_basis,
                            "analysis_aerial_path": f"{name}/analysis-aerial.png", "analysis_aerial_basis": analysis_basis,
                            "analysis_aerial_provenance": analysis_receipt,
                            "meshes": {"terrain": {**mesh["stats"], "path": f"{name}/terrain.glb"}, "surface": {**patch["meshes"]["surface"], "path": f"{name}/surface.glb"}},
                            "owner_terrain": item["tin"], "owner_imagery": item["imagery"]})
        receipt.update(status="prepared-needs-owner-review", patches=[{"id": item["patch"]["id"], "terrain": item["tin"], "imagery": item["imagery"]} for item in prepared], finished_at=utc_now())
        receipt["analysis_imagery"] = [{"id": patch["id"], "path": patch["analysis_aerial_path"], **patch["analysis_aerial_provenance"]} for patch in patches]
        receipt["warnings"].append("Historic source-context analysis aerial is retained separately from owner display imagery; color thresholds do not establish current vegetation identity.")
        write_json(output / "source-receipt.json", receipt)
        files = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*")) if path.is_file()}
        composite_manifest = {**manifest, "created_at": utc_now(), "patches": patches, "files": files, "owner_terrain_composite": receipt,
                              "coordinate_system": {**coordinates, "geoid": "owner realization not verified; public fallback GEOID18"}}
        write_json(output / "manifest.json", composite_manifest)
        artifacts = {relative: record for relative, record in files.items() if Path(relative).suffix in (".glb", ".png", ".json") and not relative.startswith("sources/")}
        contract = {"schema": "dev.splatlab.owner-terrain-import/v1", "status": receipt["status"], "coordinateSystem": frame,
                    "layers": [{"id": patch["id"] + "-terrain", "meshFile": patch["meshes"]["terrain"]["path"], "aerialFile": patch["aerial_path"]} for patch in patches],
                    "artifacts": artifacts, "regenerateBuildingsRequired": True,
                    "nextStep": {"tool": "tools/prepare-neighborhood-buildings.py", "context": str(output.resolve()), "outputMustBeNew": True},
                    "context_manifest_sha256": sha256(output / "manifest.json")}
        write_json(output / "terrain-import.json", contract)
        return receipt
    except Exception as error:
        receipt.update(status="incomplete-not-passed", error=str(error))
        write_json(output / "incomplete-receipt.json", receipt)
        raise


def inspect_input(path, kind, output, surface_name=None, source_crs=None, vertical_datum=None, preview=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Inspection receipt must use a new path")
    if kind == "landxml":
        receipt = load_landxml(path, surface_name, source_crs, vertical_datum)["receipt"]
    elif kind in ("imagery", "elevation"):
        receipt = inspect_geotiff(path, kind, preview)
    elif kind in ("building_footprints", "parcels"):
        receipt = load_shapefile(path, kind)["receipt"]
    else:
        raise ValueError("Unsupported explicit input kind")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    return receipt
