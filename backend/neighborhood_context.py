"""Bounded public elevation and aerial context, independent of captured splats."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
from PIL import Image


SCHEMA = "dev.splatlab.neighborhood-context/v1"
US_FOOT_METERS = 1200 / 3937
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
DISCOVERY_RESERVE_BYTES = 4 * 1024 * 1024
COUNTY_BASE = "https://socogis.sonomacounty.ca.gov/image2/rest/services/Rasters/"
ELEVATION_SERVICES = {
    "terrain": COUNTY_BASE + "Lidar_HydroFlat_BareEarth_DEM_2022_WM/ImageServer",
    "surface": COUNTY_BASE + "Lidar_Highest_Hit_DSM_2022/ImageServer",
}
NAIP_SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
ALLOWED_HOSTS = {"socogis.sonomacounty.ca.gov", "imagery.nationalmap.gov", "www.arcgis.com"}
ITEM_IDS = {"terrain": "8d47c0e1709c4acc9a34bc788b20252e", "surface": "e5111ae4fd584ddfade4d4cf53294cb2"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def public_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("Only allowlisted primary public data HTTPS hosts are allowed")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("Credentials and nonstandard ports are forbidden")
    return url


class PublicRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        public_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


@dataclass
class DownloadBudget:
    limit: int = MAX_DOWNLOAD_BYTES
    used: int = DISCOVERY_RESERVE_BYTES

    def __post_init__(self):
        if not 0 <= self.used < self.limit <= MAX_DOWNLOAD_BYTES:
            raise ValueError("Download budget must be positive and at most 100 MiB")

    def receive(self, response, destination, maximum):
        allowance = min(maximum, self.limit - self.used)
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > allowance:
            raise ValueError("Declared response exceeds the download budget")
        total = 0
        with Path(destination).open("xb") as target:
            while total < allowance:
                chunk = response.read(min(65536, allowance - total))
                if not chunk:
                    return total
                self.used += len(chunk)
                total += len(chunk)
                target.write(chunk)
        raise ValueError("Response reached its byte ceiling; no oversized retry is allowed")


class PublicDownloader:
    def __init__(self, root, budget=None):
        self.root = Path(root)
        self.budget = budget or DownloadBudget()
        self.records = []
        self.opener = build_opener(PublicRedirect())

    def fetch(self, url, name, maximum=16 * 1024 * 1024):
        public_url(url)
        if Path(name).name != name:
            raise ValueError("Download names must be simple filenames")
        destination = self.root / name
        record = {"url": url, "path": name, "requested_at": utc_now(), "status": "requested"}
        self.records.append(record)
        try:
            request = Request(url, headers={"User-Agent": "SplatLab-neighborhood-context/1", "Accept-Encoding": "identity"})
            with self.opener.open(request, timeout=30) as response:
                public_url(response.geturl())
                record["bytes"] = self.budget.receive(response, destination, maximum)
                record["content_type"] = response.headers.get("Content-Type")
                record["last_modified"] = response.headers.get("Last-Modified")
                record["resolved_url"] = response.geturl()
            record.update(status="downloaded", sha256=sha256(destination), finished_at=utc_now())
            return destination
        except Exception as error:
            record.update(status="failed", error=str(error), finished_at=utc_now())
            raise

    def json(self, url, name):
        path = self.fetch(url, name, maximum=1024 * 1024)
        result = json.loads(path.read_text())
        if "error" in result:
            raise ValueError(f"Public image service refused request: {result['error']}")
        return result


def capture_center(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Capture manifest is too large for this bounded context")
    capture = json.loads(path.read_text())
    samples = capture.get("gps", {}).get("samples", [])
    if not samples:
        raise ValueError("A source capture with GPS samples is required")
    sample = samples[0]
    latitude = float(sample["latitude"])
    longitude = float(sample["longitude"])
    if not math.isfinite(latitude) or not math.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("Invalid source GPS coordinate")
    return {
        "latitude": round(latitude, 4), "longitude": round(longitude, 4),
        "method": "First capture GPS sample rounded to four decimal degrees; coarse area selection only",
        "horizontal_accuracy_m": None, "survey_verified": False,
        "source_capture_id": capture.get("capture_id"), "source_manifest": str(path.resolve()),
        "source_manifest_sha256": sha256(path), "gps_altitude_used": False,
    }


def projected_bounds(east, north, half_width_m):
    values = (east, north, half_width_m)
    if not all(math.isfinite(value) for value in values) or not 10 <= half_width_m <= 804.672:
        raise ValueError("Context half-width must be 10–804.672 meters")
    half_width_feet = half_width_m / US_FOOT_METERS
    return [east - half_width_feet, north - half_width_feet, east + half_width_feet, north + half_width_feet]


def make_plan(capture_path, grid=256, texture_size=2048):
    from pyproj import Transformer

    if type(grid) is not int or not 16 <= grid <= 256:
        raise ValueError("Grid size must be an integer between 16 and 256")
    if type(texture_size) is not int or not 256 <= texture_size <= 2048:
        raise ValueError("Texture size must be an integer between 256 and 2048")
    center = capture_center(capture_path)
    transform = Transformer.from_crs(4326, 6418, always_xy=True)
    east, north = transform.transform(center["longitude"], center["latitude"])
    return {
        "schema": SCHEMA, "status": "planned", "center": center,
        "projected_origin_xy_us_feet": [east, north],
        "coordinate_system": {"source_horizontal_epsg": 6418, "source_horizontal_units": "US survey feet",
                              "source_vertical_units": {"terrain": "feet", "surface": "meters"},
                              "normalized_vertical_units": "meters", "vertical_datum": "NAVD88", "geoid": "GEOID18",
                              "scene_units": "meters", "scene_axes": {"x": "grid east", "y": "up", "z": "grid south"},
                              "grid_north_not_true_north": True, "source_to_scene_alignment_verified": False},
        "patches": [
            {"id": "neighborhood", "half_width_m": 804.672, "bounds_epsg6418": projected_bounds(east, north, 804.672), "grid": grid, "texture_size": texture_size},
            {"id": "village", "half_width_m": 192.0, "bounds_epsg6418": projected_bounds(east, north, 192.0), "grid": grid, "texture_size": min(texture_size, 1024)},
        ],
        "maximum_download_bytes": MAX_DOWNLOAD_BYTES, "preflight_discovery_reserve_bytes": DISCOVERY_RESERVE_BYTES,
        "elevation_acquisition_period": "2022-09 through 2022-11",
        "generative_completion": False, "accepted_condo_geometry_changed": False,
        "warnings": ["Historic elevation and imagery are context, not current survey evidence.",
                     "DSM contains trees and roofs; it is not classified building geometry.",
                     "Coarse GPS alignment has unknown error; numeric transform digits do not imply accuracy.",
                     "DEM WM service explicitly documents feet-valued pixels despite copied item prose saying meters; normalize only DEM Z.",
                     "DEM says feet without specifying survey/international subtype; standard 0.3048m/ft is recorded, not a survey-grade datum conversion.",
                     "Commercial 2025 EagleView imagery excluded because its metadata contains conflicting reuse terms."],
    }


def elevation_grid(values, bounds, origin, origin_height_m, nodata=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or max(values.shape) > 256:
        raise ValueError("A 2D elevation grid of 2–256 samples per side is required")
    bounds = np.asarray(bounds, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if bounds.shape != (4,) or origin.shape != (2,) or not np.isfinite(bounds).all() or not np.isfinite(origin).all() or not math.isfinite(origin_height_m):
        raise ValueError("Invalid projected bounds or origin")
    west, south, east, north = bounds
    if east <= west or north <= south:
        raise ValueError("Raster bounds must have positive width and height")
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= values != nodata
    valid &= (values >= -100) & (values <= 1500)
    if np.count_nonzero(valid) < values.size * 0.8:
        raise ValueError("Insufficient valid elevation coverage")
    rows, columns = values.shape
    eastings = west + (np.arange(columns) + 0.5) * (east - west) / columns
    northings = north - (np.arange(rows) + 0.5) * (north - south) / rows
    horizontal, depth = np.meshgrid((eastings - origin[0]) * US_FOOT_METERS, (origin[1] - northings) * US_FOOT_METERS)
    positions = np.stack((horizontal, np.where(valid, values - origin_height_m, 0), depth), axis=-1).reshape(-1, 3)
    texture_horizontal, texture_vertical = np.meshgrid((np.arange(columns) + 0.5) / columns, (np.arange(rows) + 0.5) / rows)
    texture_coordinates = np.stack((texture_horizontal, texture_vertical), axis=-1).reshape(-1, 2)
    indices = np.arange(rows * columns).reshape(rows, columns)
    northwest, northeast = indices[:-1, :-1].ravel(), indices[:-1, 1:].ravel()
    southwest, southeast = indices[1:, :-1].ravel(), indices[1:, 1:].ravel()
    faces = np.concatenate((np.stack((northwest, southwest, northeast), axis=1), np.stack((northeast, southwest, southeast), axis=1)))
    faces = faces[np.all(valid.ravel()[faces], axis=1)]
    height_ranges = np.ptp(positions[faces, 1], axis=1)
    discarded_spike_faces = int(np.count_nonzero(height_ranges > 100))
    faces = faces[height_ranges <= 100]
    if not len(faces):
        raise ValueError("Elevation grid has no usable triangles")
    used, reverse = np.unique(faces, return_inverse=True)
    positions = positions[used]
    texture_coordinates = texture_coordinates[used]
    faces = reverse.reshape(-1, 3)
    face_normals = np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]])
    if np.any(face_normals[:, 1] <= 0):
        raise ValueError("Surface triangle winding must face upward")
    normals = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return {
        "positions": positions.astype("<f4"), "normals": normals.astype("<f4"),
        "uv": texture_coordinates.astype("<f4"), "indices": faces.astype("<u4"),
        "stats": {"vertices": len(positions), "triangles": len(faces), "valid_fraction": float(valid.mean()),
                  "elevation_min_m_navd88": float(values[valid].min()), "elevation_max_m_navd88": float(values[valid].max()),
                  "discarded_spike_faces": discarded_spike_faces, "winding": "counterclockwise_up",
                  "source_pixel_size_m": [(east - west) / columns * US_FOOT_METERS, (north - south) / rows * US_FOOT_METERS]},
    }


def glb_bytes(mesh, texture_png, name):
    binary = bytearray()
    views = []
    accessors = []

    def append_view(payload, target=None):
        binary.extend(b"\0" * (-len(binary) % 4))
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        views.append(view)
        binary.extend(payload)
        return len(views) - 1

    for key, kind, component, target in (("positions", "VEC3", 5126, 34962), ("normals", "VEC3", 5126, 34962), ("uv", "VEC2", 5126, 34962), ("indices", "SCALAR", 5125, 34963)):
        values = mesh[key].reshape(-1) if key == "indices" else mesh[key]
        accessor = {"bufferView": append_view(values.tobytes(), target), "componentType": component, "count": len(values), "type": kind}
        if key == "positions":
            accessor.update(min=values.min(axis=0).tolist(), max=values.max(axis=0).tolist())
        accessors.append(accessor)
    image_view = append_view(texture_png)
    document = {
        "asset": {"version": "2.0", "generator": "SplatLab bounded public neighborhood context"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
        "materials": [{"name": "Historic USDA aerial texture", "doubleSided": False, "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1}}],
        "textures": [{"source": 0, "sampler": 0}], "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "images": [{"bufferView": image_view, "mimeType": "image/png"}], "accessors": accessors,
        "bufferViews": views, "buffers": [{"byteLength": len(binary)}],
        "extras": {"units": "meters", "axes": "+X grid east, +Y up, +Z grid south", "observed_source": "Historic LiDAR-derived surface; resampled", "generative_completion": False},
    }
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    return struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary)) + struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\0") + binary


def read_elevation(path, expected_bounds, grid, source_vertical_units="meters"):
    import rasterio
    from pyproj import CRS

    if source_vertical_units not in ("meters", "feet"):
        raise ValueError("Unsupported explicit source vertical units")
    scale = 0.3048 if source_vertical_units == "feet" else 1.0
    with rasterio.open(path) as dataset:
        if dataset.count != 1 or dataset.shape != (grid, grid) or dataset.dtypes[0] != "float32":
            raise ValueError("Expected one float32 elevation band at the requested grid size")
        if not dataset.crs or not CRS(dataset.crs).equals(CRS.from_epsg(6418), ignore_axis_order=True):
            raise ValueError("Elevation raster must be in EPSG:6418 US survey feet")
        transform = dataset.transform
        if transform.a <= 0 or transform.e >= 0 or transform.b != 0 or transform.d != 0:
            raise ValueError("Only north-up rasters with increasing eastings are supported")
        if not np.allclose(tuple(dataset.bounds), expected_bounds, rtol=0, atol=0.05):
            raise ValueError("Returned raster bounds differ from the requested area")
        values = dataset.read(1, masked=True).astype(float).filled(np.nan) * scale
        return values, {"bounds_epsg6418": list(dataset.bounds), "crs_wkt": dataset.crs.to_wkt(), "nodata": dataset.nodata,
                        "pixel_size_us_feet": [transform.a, -transform.e], "source_vertical_units": source_vertical_units,
                        "vertical_scale_to_meters": scale, "elevation_units": "meters", "north_up": True}


def check_surfaces(terrain, surface):
    if terrain.shape != surface.shape:
        raise ValueError("DEM and DSM grids must have identical shapes")
    difference = surface - terrain
    finite = difference[np.isfinite(difference)]
    if not finite.size:
        raise ValueError("DEM/DSM comparison has no common coverage")
    below_fraction = float(np.mean(finite < -2))
    result = {"min": float(finite.min()), "max": float(finite.max()), "below_terrain_by_more_than_2m_fraction": below_fraction}
    if below_fraction > 0.05:
        raise ValueError("DSM falls below DEM over more than 5% of common coverage; check source units and alignment")
    return result


def export_raster(downloader, service, stem, bounds, size, image=False, mosaic=None):
    parameters = {"f": "json", "bbox": ",".join(map(str, bounds)), "bboxSR": 6418, "imageSR": 6418,
                  "size": f"{size},{size}", "adjustAspectRatio": "false", "interpolation": "RSP_BilinearInterpolation"}
    if image:
        parameters.update(format="png32", bandIds="0,1,2")
        if mosaic:
            parameters["mosaicRule"] = json.dumps(mosaic, separators=(",", ":"))
    else:
        parameters.update(format="tiff", pixelType="F32", noData=-9999, renderingRule='{"rasterFunction":"None"}')
    metadata = downloader.json(service + "/exportImage?" + urlencode(parameters), stem + "-export.json")
    if metadata.get("width") != size or metadata.get("height") != size:
        raise ValueError("Image service did not return the requested pixel dimensions")
    extent = metadata.get("extent", {})
    returned = [extent.get(key, math.nan) for key in ("xmin", "ymin", "xmax", "ymax")]
    if not np.allclose(returned, bounds, rtol=0, atol=0.05):
        raise ValueError("Image service unexpectedly changed the export bounds")
    filename = stem + (".png" if image else ".tif")
    return downloader.fetch(metadata["href"], filename)


def prepare_context(capture_path, output, three_root=None, grid=256, texture_size=2048):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Context output must be a new directory")
    plan = make_plan(capture_path, grid, texture_size)
    prior_download_bytes = 0
    prior_manifest = output.parent / "manifest.json"
    if prior_manifest.is_file():
        prior = json.loads(prior_manifest.read_text())
        if prior.get("schema") == SCHEMA:
            prior_download_bytes = max(0, int(prior.get("accounted_bytes_including_discovery_reserve", 0)) - DISCOVERY_RESERVE_BYTES)
    output.mkdir(parents=True)
    sources = output / "sources"
    sources.mkdir()
    downloader = PublicDownloader(sources, DownloadBudget(used=DISCOVERY_RESERVE_BYTES + prior_download_bytes))
    manifest = {**plan, "created_at": utc_now(), "status": "preparing", "source_receipts": downloader.records,
                "prior_attempt_download_bytes": prior_download_bytes}
    write_json(output / "plan.json", plan)
    try:
        for kind, service in ELEVATION_SERVICES.items():
            service_info = downloader.json(service + "?f=pjson", kind + "-service.json")
            if kind == "terrain" and "pixel values (ground elevations) in feet" not in service_info.get("description", ""):
                raise ValueError("DEM source no longer explicitly documents feet-valued elevations")
            downloader.json("https://www.arcgis.com/sharing/rest/content/items/" + ITEM_IDS[kind] + "?f=pjson", kind + "-item.json")
        downloader.json(NAIP_SERVICE + "?f=pjson", "naip-service.json")
        bounds = plan["patches"][0]["bounds_epsg6418"]
        envelope = dict(zip(("xmin", "ymin", "xmax", "ymax"), bounds))
        envelope["spatialReference"] = {"wkid": 6418}
        query = {"f": "json", "geometry": json.dumps(envelope), "geometryType": "esriGeometryEnvelope", "spatialRel": "esriSpatialRelIntersects",
                 "where": "Category=1 AND agency='USDA'", "outFields": "*", "returnGeometry": "false", "resultRecordCount": 100}
        catalog = downloader.json(NAIP_SERVICE + "/query?" + urlencode(query), "naip-local-catalog.json")
        if catalog.get("exceededTransferLimit") or not catalog.get("features"):
            raise ValueError("A complete bounded USDA imagery catalog is required")
        attributes = [feature["attributes"] for feature in catalog["features"]]
        latest_year = max(item["Year"] for item in attributes)
        selected = [item for item in attributes if item["Year"] == latest_year]
        if any(not item.get("acquisition_date") or item.get("agency") != "USDA" for item in selected):
            raise ValueError("Selected imagery must have USDA provenance and acquisition dates")
        mosaic = {"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [item["OBJECTID"] for item in selected], "mosaicOperation": "MT_FIRST"}
        manifest["imagery"] = {"provider": "USGS / USDA NAIP", "selected_tiles": selected, "mosaic_rule": mosaic,
                               "source_service_crs": "EPSG:3857", "export_crs": "EPSG:6418",
                               "attribution": "USGS, USDA, The National Map. Historic aerial imagery; resampled and draped by SplatLab.",
                               "usage_basis": "USGS service describes public-domain orthoimagery; selected USDA tile records retained."}
        prepared = []
        origin_height = None
        for patch in plan["patches"]:
            patch_root = output / patch["id"]
            patch_root.mkdir()
            values = {}
            raster_receipts = {}
            for kind, service in ELEVATION_SERVICES.items():
                path = export_raster(downloader, service, patch["id"] + "-" + kind, patch["bounds_epsg6418"], grid)
                values[kind], raster_receipts[kind] = read_elevation(path, patch["bounds_epsg6418"], grid, "feet" if kind == "terrain" else "meters")
            surface_comparison = check_surfaces(values["terrain"], values["surface"])
            if origin_height is None:
                center_values = values["terrain"][grid // 2 - 1:grid // 2 + 1, grid // 2 - 1:grid // 2 + 1]
                if not np.isfinite(center_values).all():
                    raise ValueError("The coarse area center needs valid terrain elevation")
                origin_height = float(np.median(center_values))
            image_path = export_raster(downloader, NAIP_SERVICE, patch["id"] + "-aerial", patch["bounds_epsg6418"], patch["texture_size"], image=True, mosaic=mosaic)
            with Image.open(image_path) as image:
                if image.size != (patch["texture_size"], patch["texture_size"]):
                    raise ValueError("Aerial texture dimensions mismatch")
                pixels = np.asarray(image.convert("RGBA"))
                coverage = float(np.mean(pixels[:, :, 3] > 0))
                if coverage < 0.99 or float(np.std(pixels[:, :, :3])) < 2:
                    raise ValueError("Aerial texture has inadequate coverage or is blank")
                image.convert("RGB").save(patch_root / "aerial.png")
            texture_bytes = (patch_root / "aerial.png").read_bytes()
            meshes = {}
            for kind in ELEVATION_SERVICES:
                mesh = elevation_grid(values[kind], patch["bounds_epsg6418"], plan["projected_origin_xy_us_feet"], origin_height)
                asset = patch_root / (kind + ".glb")
                asset.write_bytes(glb_bytes(mesh, texture_bytes, patch["id"] + "-" + kind))
                meshes[kind] = {**mesh["stats"], "path": asset.relative_to(output).as_posix(), "sha256": sha256(asset), "bytes": asset.stat().st_size,
                                "raster": raster_receipts[kind]}
            prepared.append({**patch, "meshes": meshes, "aerial_path": (patch_root / "aerial.png").relative_to(output).as_posix(),
                             "texture_valid_fraction": coverage, "surface_minus_terrain_m": surface_comparison})
        manifest.update(status="prepared-needs-review", patches=prepared, origin_height_m_navd88=origin_height,
                        source_receipts=downloader.records, downloaded_bytes=downloader.budget.used - DISCOVERY_RESERVE_BYTES - prior_download_bytes,
                        accounted_bytes_including_discovery_reserve=downloader.budget.used, finished_at=utc_now())
        if three_root is not None:
            write_preview(output, Path(three_root))
            manifest["preview"] = "index.html"
        manifest["files"] = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                             for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(status="incomplete-not-passed", error=str(error), source_receipts=downloader.records,
                        accounted_bytes_including_discovery_reserve=downloader.budget.used, finished_at=utc_now())
        write_json(output / "manifest.json", manifest)
        raise


def write_preview(output, three_root):
    copies = {"build/three.module.js": "modules/three.module.js", "build/three.core.js": "modules/three.core.js",
              "examples/jsm/controls/OrbitControls.js": "modules/controls/OrbitControls.js",
              "examples/jsm/loaders/GLTFLoader.js": "modules/loaders/GLTFLoader.js",
              "examples/jsm/utils/BufferGeometryUtils.js": "modules/utils/BufferGeometryUtils.js",
              "examples/jsm/utils/SkeletonUtils.js": "modules/utils/SkeletonUtils.js", "LICENSE": "modules/THREE-LICENSE.txt"}
    for source, destination in copies.items():
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(three_root / source, target)
    (output / "index.html").write_text('''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chanate Village — real neighborhood context</title>
<style>body{margin:0;background:#e6e8e7;color:#192327;font:16px system-ui}header{padding:18px 24px;background:white}h1{font-size:24px;margin:0 0 6px}p{margin:6px 0;font-size:14px}button,a{padding:8px 12px;margin:4px;border:1px solid #718582;background:white;color:#173d37;cursor:pointer}#view{height:72vh}#status{font-weight:600}footer{padding:16px 24px}canvas{display:block}#map{max-width:440px;width:100%;height:auto}details{margin-top:14px}</style>
<header><h1>Chanate Village: neighborhood and village context</h1><p>Actual 2022 LiDAR-derived elevations + dated USDA aerial imagery. Not a generated concept image.</p>
<nav><button data-area="neighborhood">Neighborhood · 1.6 km</button><button data-area="village">Village · 384 m</button><button data-kind="surface">Roof / tree surface</button><button data-kind="terrain">Bare-earth terrain</button><button id="top">Top-down</button><button id="orbit">Oblique 3D</button></nav><p id="status">Loading manifest…</p></header>
<main id="view"></main><footer><p>Drag to orbit, scroll to zoom, right-drag to pan. +X grid east, +Y up, +Z grid south. Heights are not exaggerated.</p><p>Coarse capture GPS center; unknown alignment error. Roofs and trees are not separately classified. Accepted condo geometry is unchanged.</p><p id="attribution"></p><a href="manifest.json">Source hashes, bounds and dates</a><details><summary>Inspect the aerial texture and source comparison</summary><img id="map" alt="Dated USDA aerial image of the selected area"><p>The mesh uses measured historic elevations, resampled from DEM/DSM. This image is not a new capture and is not a substitute for 3D.</p></details></footer>
<script type="importmap">{"imports":{"three":"./modules/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from './modules/controls/OrbitControls.js';
import {GLTFLoader} from './modules/loaders/GLTFLoader.js';
const host=document.getElementById('view'), status=document.getElementById('status');
const scene=new THREE.Scene(); scene.background=new THREE.Color('#cdd9dc');
const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); host.appendChild(renderer.domElement);
const camera=new THREE.PerspectiveCamera(48,1,0.5,12000); const controls=new OrbitControls(camera,renderer.domElement); controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x555948,2.4)); const sun=new THREE.DirectionalLight(0xffffff,1.2); sun.position.set(-500,1200,600); scene.add(sun);
const loader=new GLTFLoader(); let selected='neighborhood',kind='surface',model=null,request=0;
const manifest=await fetch('./manifest.json').then(response=>{if(!response.ok)throw Error('Manifest unavailable');return response.json()});
if(manifest.status!=='prepared-needs-review')throw Error('This candidate is not cleared for preview: '+manifest.status);
const dates=manifest.imagery.selected_tiles.map(tile=>new Date(tile.acquisition_date).toISOString().slice(0,10));
document.getElementById('attribution').textContent=manifest.imagery.attribution+' Imagery acquired '+[...new Set(dates)].join(', ')+'. Elevation: Sonoma County / NV5 / Tukman Geospatial, Sep–Nov 2022; full credits in source metadata.';
function pose(top=false){const patch=manifest.patches.find(value=>value.id===selected),span=patch.half_width_m; camera.position.set(top?0:span*0.9,span*(top?2.65:1.15),top?0.001:span*1.35);camera.up.set(0,top?0:1,top?-1:0);controls.target.set(0,0,0);controls.update()}
async function show(){const token=++request;status.textContent='Loading real '+selected+' '+kind+' mesh…';const patch=manifest.patches.find(value=>value.id===selected);const loaded=await loader.loadAsync('./'+patch.meshes[kind].path);if(token!==request){dispose(loaded.scene);return}if(model){scene.remove(model);dispose(model)}model=loaded.scene;scene.add(model);pose();document.getElementById('map').src=patch.aerial_path;status.textContent=selected+' · '+kind+' · '+patch.meshes[kind].triangles.toLocaleString()+' triangles · historic context, needs review';window.neighborhoodPreview={area:selected,kind,triangles:patch.meshes[kind].triangles,loaded:true}}
function dispose(object){object.traverse(child=>{child.geometry?.dispose();if(child.material){for(const material of Array.isArray(child.material)?child.material:[child.material]){material.map?.dispose();material.dispose()}}})}
document.querySelectorAll('[data-area]').forEach(button=>button.onclick=()=>{selected=button.dataset.area;show().catch(fail)});document.querySelectorAll('[data-kind]').forEach(button=>button.onclick=()=>{kind=button.dataset.kind;show().catch(fail)});
document.getElementById('top').onclick=()=>pose(true);document.getElementById('orbit').onclick=()=>pose(false);
function fail(error){status.textContent='Preview error: '+error.message;window.neighborhoodPreview={loaded:false,error:error.message}}
function resize(){renderer.setSize(host.clientWidth,host.clientHeight);camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix()}window.addEventListener('resize',resize);resize();renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});show().catch(fail);
</script></html>''')


def refresh_preview(output, three_root):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "prepared-needs-review" or not manifest.get("files"):
        raise ValueError("Only a complete prepared candidate can refresh its preview")
    for relative, record in manifest["files"].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output) or not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError("Candidate file hash changed before preview refresh")
    write_preview(output, Path(three_root))
    manifest["files"] = {path.relative_to(output).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
                         for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    manifest["preview_refreshed_at"] = utc_now()
    write_json(output / "manifest.json", manifest)
    return {"status": "preview-refreshed", "source_downloads": 0, "geometry_changed": False}
