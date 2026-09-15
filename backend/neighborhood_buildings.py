"""Clean, provenance-labeled neighborhood massing from bounded public footprints."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from urllib.parse import urlencode

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from scipy.ndimage import maximum_filter
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping, shape
from shapely.ops import transform as transform_geometry, unary_union

from neighborhood_context import DownloadBudget, PublicDownloader, read_elevation, sha256, utc_now, write_json
from neighborhood_inputs import METRIC_CRS, load_shapefile, local_file, merge_owner_footprints


SCHEMA = "dev.splatlab.neighborhood-lowpoly/v1"
FOOTPRINT_SERVICE = "https://socogis.sonomacounty.ca.gov/map/rest/services/BASEPublic/Buildings/FeatureServer/0"
RADIUS_METERS = 804.672
COLORS = {"terrain": [0.48, 0.54, 0.41], "wall": [0.64, 0.60, 0.52], "roof": [0.29, 0.34, 0.35],
          "canopy": [0.23, 0.37, 0.25], "trunk": [0.29, 0.22, 0.16]}


def polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [part for child in geometry.geoms for part in polygon_parts(child)]
    return []


def esri_polygon(rings):
    if not rings or sum(len(ring) for ring in rings) > 20000:
        raise ValueError("Footprint rings absent or too large")
    polygons = []
    for ring in rings:
        coordinates = np.asarray(ring, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 4 or not np.isfinite(coordinates).all():
            raise ValueError("Footprint rings require finite 2D coordinates")
        polygon = Polygon(coordinates)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("Invalid source footprint ring")
        polygons.append(polygon)
    polygons.sort(key=lambda value: value.area, reverse=True)
    parents = []
    depths = []
    for index, polygon in enumerate(polygons):
        candidates = [ordinal for ordinal in range(index) if polygons[ordinal].contains(polygon.representative_point())]
        parent = min(candidates, key=lambda ordinal: polygons[ordinal].area) if candidates else None
        parents.append(parent)
        depths.append(0 if parent is None else depths[parent] + 1)
    results = [Polygon(polygon.exterior.coords, [polygons[child].exterior.coords for child, parent in enumerate(parents) if parent == index])
               for index, polygon in enumerate(polygons) if depths[index] % 2 == 0]
    result = MultiPolygon(results) if len(results) > 1 else results[0]
    if not result.is_valid:
        raise ValueError("Source footprint shell/hole topology is invalid")
    return result


def snapshot_footprints(downloader, bounds):
    metadata = downloader.json(FOOTPRINT_SERVICE + "?f=pjson", "county-buildings-service.json")
    reference = metadata.get("extent", {}).get("spatialReference", {})
    if reference.get("latestWkid", reference.get("wkid")) != 2226:
        raise ValueError("County footprint source CRS changed; expected EPSG:2226")
    envelope = dict(zip(("xmin", "ymin", "xmax", "ymax"), bounds))
    envelope["spatialReference"] = {"wkid": METRIC_CRS}
    common = {"f": "json", "where": "1=1", "geometry": json.dumps(envelope), "geometryType": "esriGeometryEnvelope", "spatialRel": "esriSpatialRelIntersects"}
    count = downloader.json(FOOTPRINT_SERVICE + "/query?" + urlencode({**common, "returnCountOnly": "true"}), "county-count-before.json")["count"]
    if not 1 <= count <= 5000:
        raise ValueError("County footprint count is empty or exceeds bounded pilot")
    features = []
    seen = set()
    for offset in range(0, count, 500):
        query = {**common, "outFields": "OBJECTID,Name,PropType,GovtType,last_edited_date", "outSR": 2226,
                 "returnGeometry": "true", "returnTrueCurves": "false", "orderByFields": "OBJECTID ASC",
                 "resultOffset": offset, "resultRecordCount": 500}
        path = downloader.fetch(FOOTPRINT_SERVICE + "/query?" + urlencode(query), f"county-page-{offset:04}.json", maximum=8 * 1024 * 1024)
        response = json.loads(path.read_text())
        if "error" in response or not response.get("features"):
            raise ValueError("County footprint page is missing or refused")
        for feature in response["features"]:
            identifier = feature["attributes"]["OBJECTID"]
            if identifier in seen:
                raise ValueError("County pagination repeated an object ID; snapshot is inconsistent")
            seen.add(identifier)
            features.append(feature)
    final_count = downloader.json(FOOTPRINT_SERVICE + "/query?" + urlencode({**common, "returnCountOnly": "true"}), "county-count-after.json")["count"]
    if len(features) != count or final_count != count:
        raise ValueError("County feature count changed during snapshot")
    return metadata, features


def normalize_footprints(features, center, radius=RADIUS_METERS):
    transformer = Transformer.from_crs(2226, METRIC_CRS, always_xy=True)
    clip = Point(center).buffer(radius, quad_segs=128)
    normalized = []
    fingerprints = set()
    skipped = Counter()
    for feature in features:
        attributes = feature["attributes"]
        try:
            native = esri_polygon(feature.get("geometry", {}).get("rings"))
            projected = transform_geometry(transformer.transform, native)
        except (ValueError, TypeError):
            skipped["invalid_source_geometry"] += 1
            continue
        clipped = projected.intersection(clip)
        parts = polygon_parts(clipped)
        if not parts:
            skipped["outside_circle"] += 1
            continue
        clipped = MultiPolygon(parts) if len(parts) > 1 else parts[0]
        if clipped.area < 5:
            skipped["tiny_or_point_proxy"] += 1
            continue
        fingerprint = hashlib.sha256(shapely.set_precision(clipped, 0.02).normalize().wkb).hexdigest()
        if fingerprint in fingerprints:
            skipped["duplicate_geometry"] += 1
            continue
        fingerprints.add(fingerprint)
        normalized.append({"id": f"county-buildings-{attributes['OBJECTID']}", "geometry": clipped, "feature_type": "building_footprints",
                           "properties": {"sourceObjectId": attributes["OBJECTID"], "name": attributes.get("Name") or f"County building {attributes['OBJECTID']}",
                                          "countyPropertyType": attributes.get("PropType"), "lastEditedUtcMs": attributes.get("last_edited_date"),
                                          "clippedAtContextBoundary": not projected.equals(clipped)}})
    return normalized, {"input_features": len(features), "output_features": len(normalized), "skipped": dict(skipped),
                        "source_crs": "EPSG:2226", "target_crs": "EPSG:6417", "transform_description": transformer.description,
                        "transform_accuracy_meters": transformer.accuracy, "circle_segments": 512}


class ContextGrid:
    def __init__(self, root, patch, origin):
        self.patch_id = patch["id"]
        self.terrain_basis = patch.get("terrain_basis", "2022-public-DEM")
        self.origin = origin
        grid = patch["grid"]
        bounds = patch["bounds_epsg6418"]
        projected = Transformer.from_crs(6418, METRIC_CRS, always_xy=True)
        self.bounds = projected.transform_bounds(*bounds)
        self.transform = from_bounds(*self.bounds, grid, grid)
        self.terrain, _ = read_elevation(root / "sources" / f"{patch['id']}-terrain.tif", bounds, grid, "feet")
        self.surface, _ = read_elevation(root / "sources" / f"{patch['id']}-surface.tif", bounds, grid, "meters")
        self.rows, self.columns = self.terrain.shape
        self.color_basis = patch.get("analysis_aerial_basis", "context aerial color heuristic; not vegetation classification")
        image = Image.open(root / patch.get("analysis_aerial_path", patch["aerial_path"])).convert("RGB").resize((grid, grid), Image.Resampling.BILINEAR)
        color = np.asarray(image).astype(float)
        self.green = (color[:, :, 1] > color[:, :, 0] * 1.07) & (color[:, :, 1] > color[:, :, 2] * 1.04)
        self.pixel_size = (self.bounds[2] - self.bounds[0]) / self.columns

    def mask(self, geometry):
        return geometry_mask([mapping(geometry)], out_shape=self.terrain.shape, transform=self.transform, invert=True)

    def sample(self, east, north):
        column, row = ~self.transform * (east, north)
        row = int(np.clip(math.floor(row), 0, self.rows - 1))
        column = int(np.clip(math.floor(column), 0, self.columns - 1))
        return float(self.terrain[row, column])

    def height(self, polygon):
        interior = polygon.buffer(-min(1.0, self.pixel_size * 0.2))
        mask = self.mask(interior if not interior.is_empty else polygon)
        valid = mask & np.isfinite(self.terrain) & np.isfinite(self.surface)
        terrain = self.terrain[valid]
        heights = (self.surface - self.terrain)[valid]
        flags = ["mixed_date_owner_terrain_and_historic_surface"] if self.terrain_basis != "2022-public-DEM" else []
        if terrain.size:
            base = float(np.percentile(terrain, 10)) - 0.25
            reference = float(np.median(terrain))
            ground_max = float(terrain.max())
        else:
            centroid = polygon.representative_point()
            reference = self.sample(centroid.x, centroid.y)
            base, ground_max = reference - 0.25, reference
        green_fraction = float(np.mean(self.green[valid])) if terrain.size else 0
        candidates = heights[(heights >= 2) & (heights <= 18) & ~self.green[valid]]
        spread = float(np.percentile(candidates, 90) - np.percentile(candidates, 10)) if len(candidates) >= 4 else None
        if green_fraction > 0.2 or (spread is not None and spread > 5):
            flags.append("tree_contamination_possible")
        if len(candidates) < 4 or green_fraction > 0.55 or (spread is not None and spread > 10):
            approximate_height = 6.0
            basis = "explicit-6m-approximate-fallback"
            confidence = "low"
            flags.append("insufficient_uncontaminated_height_samples")
        else:
            approximate_height = float(np.clip(np.percentile(candidates, 35), 2.5, 18))
            basis = "2022-dsm-minus-dem-interior-35th-percentile" if self.terrain_basis == "2022-public-DEM" else "historic-dsm-minus-owner-public-terrain-interior-35th-percentile"
            confidence = "low" if flags else "medium"
        roof = max(reference + approximate_height, ground_max + 2)
        if roof > reference + approximate_height + 0.01:
            flags.append("roof_raised_for_sloping_ground_clearance")
            confidence = "low"
        return {"approximateHeightMeters": round(roof - reference, 3), "heightBasis": basis, "confidence": confidence,
                "flags": flags + ["flat_roof_is_simplified_not_measured"], "baseElevationMeters": round(base, 3),
                "roofElevationMeters": round(roof, 3), "heightSampleCount": len(candidates),
                "greenContaminationFraction": round(green_fraction, 4), "heightSampleSpreadMeters": spread,
                "heightGrid": self.patch_id, "colorAnalysisBasis": self.color_basis}


def choose_grid(grids, geometry):
    return next((grid for grid in reversed(grids) if box(*grid.bounds).covers(geometry)), grids[0])


def triangle_mesh(triangles, colors):
    positions = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
    cross = np.cross(positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    if np.any(lengths < 1e-10) or not np.isfinite(positions).all():
        raise ValueError("Mesh has invalid or degenerate triangles")
    normals = np.repeat((cross / lengths[:, None])[:, None, :], 3, axis=1)
    color_values = np.repeat(np.asarray(colors, dtype=float)[:, None, :], 3, axis=1)
    return {"positions": positions.reshape(-1, 3).astype("<f4"), "normals": normals.reshape(-1, 3).astype("<f4"),
            "colors": color_values.reshape(-1, 3).astype("<f4")}


def extrude_building(geometry, base, roof, origin):
    if roof <= base:
        raise ValueError("Building roof must be above its base")
    triangles, colors = [], []

    def local(point, height):
        return [point[0] - origin[0], height - origin[2], origin[1] - point[1]]

    for polygon in polygon_parts(geometry):
        polygon = shapely.geometry.polygon.orient(polygon, sign=1)
        triangulation = shapely.constrained_delaunay_triangles(polygon)
        for triangle in triangulation.geoms:
            coordinates = list(triangle.exterior.coords)[:3]
            points = [local(point, roof) for point in coordinates]
            if np.cross(np.subtract(points[1], points[0]), np.subtract(points[2], points[0]))[1] < 0:
                points.reverse()
            triangles.append(points)
            colors.append(COLORS["roof"])
            bottom = [local(point, base) for point in coordinates]
            if np.cross(np.subtract(bottom[1], bottom[0]), np.subtract(bottom[2], bottom[0]))[1] > 0:
                bottom.reverse()
            triangles.append(bottom)
            colors.append(COLORS["wall"])
        for ring in [polygon.exterior, *polygon.interiors]:
            coordinates = list(ring.coords)
            for first, second in zip(coordinates[:-1], coordinates[1:]):
                lower_first, lower_second = local(first, base), local(second, base)
                upper_first, upper_second = local(first, roof), local(second, roof)
                triangles.extend([[lower_first, lower_second, upper_first], [lower_second, upper_second, upper_first]])
                colors.extend([COLORS["wall"], COLORS["wall"]])
    return triangle_mesh(triangles, colors)


def colored_glb(objects):
    binary, views, accessors, nodes, meshes = bytearray(), [], [], [], []
    for item in objects:
        attributes = {}
        for key, attribute in (("positions", "POSITION"), ("normals", "NORMAL"), ("colors", "COLOR_0")):
            values = np.asarray(item["mesh"][key], dtype="<f4")
            binary.extend(b"\0" * (-len(binary) % 4))
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": values.nbytes, "target": 34962})
            binary.extend(values.tobytes())
            accessor = {"bufferView": len(views) - 1, "componentType": 5126, "count": len(values), "type": "VEC3"}
            if key == "positions":
                accessor.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
            accessors.append(accessor)
            attributes[attribute] = len(accessors) - 1
        meshes.append({"name": item["id"], "primitives": [{"attributes": attributes, "material": 0}]})
        nodes.append({"name": item["id"], "mesh": len(meshes) - 1, "extras": item.get("extras", {})})
    if not meshes:
        raise ValueError("Refusing an empty GLB layer")
    document = {"asset": {"version": "2.0", "generator": "SplatLab low-poly neighborhood"}, "scene": 0,
                "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
                "materials": [{"name": "Muted neighborhood colors", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1}, "doubleSided": False}],
                "bufferViews": views, "accessors": accessors, "buffers": [{"byteLength": len(binary)}],
                "extras": {"units": "metres", "axes": "east-up-south", "height_exaggeration": 1}}
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    return struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary)) + struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\0") + binary


def terrain_object(grid, origin, circular=True):
    west, south, east, north = grid.bounds
    rows, columns = grid.terrain.shape
    horizontal, depth = np.meshgrid(west + (np.arange(columns) + 0.5) * (east - west) / columns - origin[0],
                                   origin[1] - north + (np.arange(rows) + 0.5) * (north - south) / rows)
    points = np.stack((horizontal, grid.terrain - origin[2], depth), axis=-1).reshape(-1, 3)
    indices = np.arange(rows * columns).reshape(rows, columns)
    faces = np.concatenate((np.stack((indices[:-1, :-1].ravel(), indices[1:, :-1].ravel(), indices[:-1, 1:].ravel()), axis=1),
                            np.stack((indices[:-1, 1:].ravel(), indices[1:, :-1].ravel(), indices[1:, 1:].ravel()), axis=1)))
    valid = np.isfinite(points).all(axis=1)
    if circular:
        valid &= np.linalg.norm(points[:, [0, 2]], axis=1) <= RADIUS_METERS
    faces = faces[np.all(valid[faces], axis=1)]
    return {"id": grid.patch_id + "-terrain", "mesh": triangle_mesh(points[faces], [COLORS["terrain"]] * len(faces)),
            "extras": {"role": "terrain", "basis": grid.terrain_basis + ", no height exaggeration"}}


def canopy_candidates(grid, buildings, origin, circular=True):
    building_mask = grid.mask(shapely.union_all([feature["geometry"].buffer(3) for feature in buildings]))
    heights = grid.surface - grid.terrain
    size = max(3, round(12 / grid.pixel_size) | 1)
    peaks = maximum_filter(np.where(np.isfinite(heights), heights, -1000), size=size, mode="nearest")
    eligible = grid.green & ~building_mask & np.isfinite(heights) & (heights >= 3) & (heights <= 42) & (heights == peaks)
    rows, columns = np.where(eligible)
    candidates, occupied = [], set()
    for row, column in zip(rows, columns):
        east, north = grid.transform * (float(column) + 0.5, float(row) + 0.5)
        if circular and math.hypot(east - origin[0], north - origin[1]) > RADIUS_METERS - 8:
            continue
        cell = (round((east - origin[0]) / 8), round((north - origin[1]) / 8))
        if cell in occupied:
            continue
        occupied.add(cell)
        candidates.append({"id": f"canopy-candidate-{grid.patch_id}-{row}-{column}", "east": east, "north": north,
                           "ground": float(grid.terrain[row, column]), "height": float(heights[row, column]),
                           "radius": float(np.clip(heights[row, column] * 0.25, 2.5, 7)), "basis": "unclassified-historic-DSM-canopy-peak-candidate", "terrainBasis": grid.terrain_basis})
    return candidates[:1500]


def canopy_object(candidates, origin, name):
    triangles, colors = [], []
    for candidate in candidates:
        center = np.array([candidate["east"] - origin[0], candidate["ground"] - origin[2], origin[1] - candidate["north"]])
        height, radius = candidate["height"], candidate["radius"]
        ring = [center + [radius * math.cos(angle), height * 0.55, radius * math.sin(angle)] for angle in np.linspace(0, math.tau, 9)[:-1]]
        top, bottom = center + [0, height, 0], center + [0, height * 0.2, 0]
        for index in range(8):
            first, second = ring[index], ring[(index + 1) % 8]
            triangles.extend([[first, top, second], [first, second, bottom]])
            colors.extend([COLORS["canopy"], COLORS["canopy"]])
    return {"id": name, "mesh": triangle_mesh(triangles, colors), "extras": {"role": "vegetation", "basis": "simplified-canopy-candidates-not-measured-tree-identities"}}


def owner_coverage(path):
    record = json.loads(local_file(path, {".json", ".geojson"}, 8 * 1024**2).read_text())
    if "crs" in record:
        raise ValueError("Replacement coverage must use standard WGS84 GeoJSON")
    records = record.get("features", []) if record.get("type") == "FeatureCollection" else [record]
    if not 1 <= len(records) <= 100:
        raise ValueError("Replacement coverage has no bounded polygon records")
    polygons = []
    for feature in records:
        if "crs" in feature:
            raise ValueError("Replacement coverage must use standard WGS84 GeoJSON")
        geometry = shape(feature["geometry"] if feature.get("type") == "Feature" else feature)
        if not geometry.is_valid or geometry.is_empty or not box(-180, -90, 180, 90).covers(geometry):
            raise ValueError("Replacement coverage must have valid WGS84 coordinates")
        if geometry.geom_type in ("Polygon", "MultiPolygon"):
            polygons.append(geometry)
        elif geometry.geom_type != "Point":
            raise ValueError("Coverage accepts polygons with optional site points only")
    if not polygons:
        raise ValueError("Replacement coverage needs a polygon, not just a site point")
    return transform_geometry(Transformer.from_crs(4326, METRIC_CRS, always_xy=True).transform, unary_union(polygons))


def clip_owner_footprints(layer, coverage, circle):
    if layer["receipt"].get("feature_type") != "building_footprints":
        raise ValueError("Only explicit building footprints may become building massing")
    features, skipped = [], Counter()
    for feature in layer["features"]:
        geometry = feature["geometry"].intersection(coverage).intersection(circle)
        parts = polygon_parts(geometry)
        if not parts or geometry.area < 5:
            skipped["outside_context_or_tiny"] += 1
            continue
        geometry = MultiPolygon(parts) if len(parts) > 1 else parts[0]
        properties = feature["properties"]
        features.append({**feature, "geometry": geometry, "properties": {
            "name": str(properties.get("NAME") or properties.get("name") or feature["id"])[:200],
            "clippedAtContextBoundary": not geometry.equals(feature["geometry"]),
            "sourceFeatureId": feature["id"],
        }})
    return {**layer, "features": features}, dict(skipped)


def elevation_source_units(source_manifest):
    coordinates = source_manifest["coordinate_system"]
    composite = source_manifest.get("owner_terrain_composite") or {}
    owner = composite.get("elevation_geotiff") or composite.get("landxml")
    public_geoid = composite.get("public_geoid_declaration", coordinates.get("geoid"))
    original_owner = None
    if owner:
        source = owner.get("source", {})
        original_owner = {
            "kind": owner.get("kind"),
            "name": source.get("name"),
            "sha256": source.get("sha256"),
            "vertical_unit": owner.get("vertical_unit", owner.get("linear_unit")),
            "vertical_scale_to_meters": owner.get("vertical_scale_to_meters"),
            "vertical_datum": owner.get("vertical_datum"),
            "geoid": owner.get("geoid", composite.get("owner_geoid_declaration")),
            "band_scales": owner.get("scales"),
            "band_offsets": owner.get("offsets"),
        }
    return {
        "dem": {
            "original_owner_source": original_owner,
            "context_raster_encoding": {
                "vertical_unit": "international foot" if composite else "feet; source subtype unspecified",
                "vertical_scale_to_meters": 0.3048,
                "basis": "metre-valued owner/public composite divided by 0.3048 for compatibility TIFF storage" if composite else "public DEM service feet-valued pixels; standard 0.3048 conversion assumption",
            },
            "public_fallback": {
                "vertical_unit": "feet; source subtype unspecified",
                "vertical_scale_to_meters": 0.3048,
                "vertical_datum": coordinates.get("vertical_datum"),
                "geoid": public_geoid,
                "scope": "outside valid owner coverage" if composite else "entire context",
            },
            "read_elevation_output_unit": "meters",
            "original_owner_scale_reapplied_by_reader": False,
            "vertical_datum_conversion": composite.get("vertical_datum_conversion", False),
            "geoid_compatibility_verified": composite.get("geoid_compatibility_verified", False),
        },
        "dsm": {
            "basis": "historic public DSM retained from source context",
            "vertical_unit": "meters",
            "vertical_scale_to_meters": 1.0,
            "vertical_datum": coordinates.get("vertical_datum"),
            "geoid": public_geoid,
            "owner_dem_unit_conversion_applied": False,
        },
    }


def prepare_lowpoly(context, output, owner_shapefile=None, coverage_path=None, owner_validated=False):
    context, output = Path(context).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Low-poly output must be a new directory")
    source_manifest = json.loads((context / "manifest.json").read_text())
    if source_manifest.get("status") != "prepared-needs-review" or source_manifest["coordinate_system"].get("source_vertical_units") != {"terrain": "feet", "surface": "meters"}:
        raise ValueError("Only the corrected verified context can be used")
    for relative, record in source_manifest["files"].items():
        target = (context / relative).resolve()
        if not target.is_relative_to(context) or sha256(target) != record["sha256"]:
            raise ValueError("Verified source context hash changed")
    origin_xy = Transformer.from_crs(6418, METRIC_CRS, always_xy=True).transform(*source_manifest["projected_origin_xy_us_feet"])
    origin = [*origin_xy, source_manifest["origin_height_m_navd88"]]
    grids = [ContextGrid(context, patch, origin) for patch in source_manifest["patches"]]
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    downloader = PublicDownloader(output / "sources", DownloadBudget(limit=20 * 1024 * 1024, used=0))
    receipt = {"schema": SCHEMA, "created_at": utc_now(), "status": "preparing", "source_context_manifest_sha256": sha256(context / "manifest.json")}
    try:
        owner, coverage = None, None
        circle = Point(origin[:2]).buffer(RADIUS_METERS, quad_segs=128)
        if owner_shapefile is not None:
            if coverage_path is None or owner_validated is not True:
                raise ValueError("Owner footprint replacement needs validation and explicit coverage GeoJSON")
            coverage = owner_coverage(coverage_path)
            owner, owner_skipped = clip_owner_footprints(load_shapefile(owner_shapefile, "building_footprints"), coverage, circle)
        if owner is not None and coverage.covers(circle):
            metadata, footprints = {"copyrightText": "Owner-provided building outlines; retain original source attribution"}, owner["features"]
            normalization = {"input_features": owner["receipt"]["features"], "output_features": len(footprints), "skipped": owner_skipped,
                             "source_crs": owner["receipt"]["source_crs"], "target_crs": "EPSG:6417",
                             "basis": "validated-owner-coverage-contains-entire-context; no redundant public download"}
        else:
            metadata, raw = snapshot_footprints(downloader, grids[0].bounds)
            footprints, normalization = normalize_footprints(raw, origin[:2])
        if owner is not None and not coverage.covers(circle):
            footprints = merge_owner_footprints(footprints, owner, coverage, owner_validated)
            footprints = [{**feature, "geometry": feature["geometry"].intersection(circle)} for feature in footprints if feature["geometry"].intersects(circle)]
            footprints = [feature for feature in footprints if feature["geometry"].area >= 5]
        if owner is not None:
            write_json(output / "sources" / "owner-footprint-receipt.json", owner["receipt"])
            write_json(output / "sources" / "owner-coverage-receipt.json", {"sha256": sha256(coverage_path), "clippedDerivative": True,
                       "sourceOriginalChanged": False, "coversEntireContext": coverage.covers(circle), "retainedFeatures": len(owner["features"]), "skipped": owner_skipped})
        inverse = Transformer.from_crs(METRIC_CRS, 4326, always_xy=True)
        objects, features = [], []
        elevation_basis = "2022 DEM/DSM" if not source_manifest.get("owner_terrain_composite") else "owner/public terrain composite and historic 2022 DSM; mixed dates"
        for footprint in footprints:
            grid = choose_grid(grids, footprint["geometry"])
            height = grid.height(footprint["geometry"])
            center = footprint["geometry"].centroid
            longitude, latitude = inverse.transform(center.x, center.y)
            source = ("County of Sonoma building outlines" if footprint["id"].startswith("county-") else "Owner-validated footprint") + "; heights estimated from " + elevation_basis
            properties = {**footprint["properties"], **height, "featureId": footprint["id"], "buildingId": footprint["id"], "source": source,
                          "centroid": [longitude, latitude], "footprintAreaSquareMeters": round(footprint["geometry"].area, 3),
                          "geometryBasis": "source-footprint-clipped-and-extruded", "roofType": "flat-simplified"}
            features.append({"type": "Feature", "id": footprint["id"], "geometry": mapping(transform_geometry(inverse.transform, footprint["geometry"])), "properties": properties})
            mesh = extrude_building(footprint["geometry"], height["baseElevationMeters"], height["roofElevationMeters"], origin)
            objects.append({"id": footprint["id"], "mesh": mesh, "extras": {"featureId": footprint["id"], "role": "building", "confidence": height["confidence"]}})
        overview_canopies = canopy_candidates(grids[0], footprints, origin)
        village_canopies = canopy_candidates(grids[1], footprints, origin, circular=False)
        village_bounds = box(*grids[1].bounds)
        overview_canopies = [candidate for candidate in overview_canopies if not village_bounds.covers(Point(candidate["east"], candidate["north"]))] + village_canopies
        counts = {}
        for grid in grids:
            area = output / grid.patch_id
            area.mkdir()
            if grid.patch_id == "neighborhood":
                selected_objects, trees = objects, overview_canopies
            else:
                indices = [index for index, footprint in enumerate(footprints) if village_bounds.covers(footprint["geometry"].representative_point())]
                selected_objects, trees = [objects[index] for index in indices], village_canopies
            terrain = terrain_object(grid, origin, circular=grid.patch_id == "neighborhood")
            layers = {"terrain": [terrain]}
            if selected_objects:
                layers["buildings"] = selected_objects
            if trees:
                layers["vegetation"] = [canopy_object(trees, origin, grid.patch_id + "-vegetation")]
            for name, layer in layers.items():
                (area / f"{name}.glb").write_bytes(colored_glb(layer))
            source_patch = next(patch for patch in source_manifest["patches"] if patch["id"] == grid.patch_id)
            shutil.copyfile(context / source_patch["aerial_path"], area / "aerial.png")
            shutil.copyfile(context / source_patch["meshes"]["surface"]["path"], area / "surface.glb")
            counts[grid.patch_id] = {"buildings": len(selected_objects), "canopy_candidates": len(trees), "terrain_triangles": len(terrain["mesh"]["positions"]) // 3,
                                    "building_triangles": sum(len(item["mesh"]["positions"]) // 3 for item in selected_objects)}
        feature_collection = {"type": "FeatureCollection", "features": features}
        write_json(output / "building-features.geojson", feature_collection)
        write_json(output / "canopy-candidates.json", {"basis": "simplified canopy peaks, not measured tree identities", "candidates": overview_canopies})
        source_summary = "County of Sonoma building outlines (mixed acquisition ages; not complete). Flat roofs and heights are simplified/estimated from 2022 DEM/DSM, with explicit 6m fallbacks and tree-contamination flags. Muted low-poly scene; trees are canopy candidates, not measured identities. USDA aerial comparison acquired 2022-05-24. Coarse GPS center; no accepted condo registration."
        if source_manifest.get("owner_terrain_composite"):
            source_summary = "County/owner building outlines with freshly recomputed bases and approximate heights from owner/public terrain plus historic 2022 DSM. Mixed dates, simplified flat roofs, explicit 6m height fallbacks and tree-contamination flags require review. Owner aerial has historic USDA fallback outside coverage; surface comparison remains historic. No accepted condo or survey registration."
        neighborhood = {"center": {"longitude": source_manifest["center"]["longitude"], "latitude": source_manifest["center"]["latitude"], "basis": source_manifest["center"]["method"]},
                        "radiusMeters": RADIUS_METERS, "coordinateSystem": {"crs": "EPSG:6417", "origin": origin, "units": "metres", "axes": "east-up-south"},
                        "layers": [{"id": name, "title": title, "role": name, "meshFile": f"neighborhood/{name}.glb", "defaultVisible": name != "surface"}
                                   for name, title in (("terrain", "Muted terrain"), ("buildings", "Clean building massing"), ("vegetation", "Simplified canopy"), ("surface", "Historic textured surface comparison")) if (output / "neighborhood" / f"{name}.glb").exists()],
                        "featuresFile": "building-features.geojson", "aerialFile": "neighborhood/aerial.png", "sourceSummary": source_summary}
        height_counts = Counter(feature["properties"]["heightBasis"] for feature in features)
        receipt.update(status="prepared-needs-owner-review", finished_at=utc_now(), coordinateSystem=neighborhood["coordinateSystem"], normalization=normalization,
                       counts=counts, height_basis_counts=dict(height_counts), source_summary=source_summary, attribution=metadata.get("copyrightText"),
                       source_receipts=downloader.records, downloaded_bytes=downloader.budget.used,
                       native_source_crs=owner["receipt"]["source_crs"] if owner is not None and coverage.covers(circle) else "EPSG:2226",
                       elevation_source_units=elevation_source_units(source_manifest),
                       owner_input_used=owner_shapefile is not None or bool(source_manifest.get("owner_terrain_composite")),
                       owner_terrain_composite_used=bool(source_manifest.get("owner_terrain_composite")), accepted_condo_changed=False, generative_ai_used=False)
        write_json(output / "source-receipt.json", receipt)
        artifacts = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                     for path in sorted(output.rglob("*")) if path.is_file() and "sources" not in path.relative_to(output).parts}
        write_json(output / "condo-import.json", {"neighborhood": neighborhood, "artifacts": artifacts})
        write_json(output / "manifest.json", receipt)
        return receipt
    except Exception as error:
        receipt.update(status="incomplete-not-passed", error=str(error), source_receipts=downloader.records, downloaded_bytes=downloader.budget.used)
        write_json(output / "manifest.json", receipt)
        raise
