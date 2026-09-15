"""Horizontal similarity alignment from map or measured control pairs."""

from __future__ import annotations

import math


def ecef(latitude: float, longitude: float) -> tuple[float, float, float]:
    latitude, longitude = math.radians(latitude), math.radians(longitude)
    eccentricity = 0.0066943799901413165
    radius = 6378137 / math.sqrt(1 - eccentricity * math.sin(latitude) ** 2)
    return (radius * math.cos(latitude) * math.cos(longitude),
            radius * math.cos(latitude) * math.sin(longitude),
            radius * (1 - eccentricity) * math.sin(latitude))


def fit_controls(controls: list[dict]) -> dict:
    if not 2 <= len(controls) <= 100:
        raise ValueError("Provide 2–100 control pairs; use at least 3 for residual checks")
    origin = controls[0]
    latitude, longitude = math.radians(origin["lat"]), math.radians(origin["lon"])
    base = ecef(origin["lat"], origin["lon"])
    target = []
    for control in controls:
        if not (-90 <= control["lat"] <= 90 and -180 <= control["lon"] <= 180):
            raise ValueError("Control coordinates are outside WGS84 bounds")
        numbers = [control["lat"], control["lon"], *control["scene"]]
        if len(control["scene"]) != 2 or not all(math.isfinite(value) for value in numbers):
            raise ValueError("Control coordinates must be finite XY and latitude/longitude")
        offset = [value - anchor for value, anchor in zip(ecef(control["lat"], control["lon"]), base)]
        east = -math.sin(longitude) * offset[0] + math.cos(longitude) * offset[1]
        north = (-math.sin(latitude) * math.cos(longitude) * offset[0]
                 - math.sin(latitude) * math.sin(longitude) * offset[1] + math.cos(latitude) * offset[2])
        if math.hypot(east, north) > 10000:
            raise ValueError("Control footprint exceeds the local 10 km alignment limit")
        target.append((east, north))
    count = len(controls)
    mean_scene = [sum(control["scene"][axis] for control in controls) / count for axis in (0, 1)]
    mean_target = [sum(point[axis] for point in target) / count for axis in (0, 1)]
    denominator = cosine_sum = sine_sum = 0.0
    for control, point in zip(controls, target):
        scene_x, scene_y = [control["scene"][axis] - mean_scene[axis] for axis in (0, 1)]
        east, north = [point[axis] - mean_target[axis] for axis in (0, 1)]
        denominator += scene_x * scene_x + scene_y * scene_y
        cosine_sum += scene_x * east + scene_y * north
        sine_sum += scene_x * north - scene_y * east
    if denominator <= 1e-12 or math.hypot(cosine_sum, sine_sum) <= 1e-9:
        raise ValueError("Control points do not constrain a nonzero scale and heading")
    cosine_scale, sine_scale = cosine_sum / denominator, sine_sum / denominator
    scale = math.hypot(cosine_scale, sine_scale)
    translation = [mean_target[0] - cosine_scale * mean_scene[0] + sine_scale * mean_scene[1],
                   mean_target[1] - sine_scale * mean_scene[0] - cosine_scale * mean_scene[1]]
    anchor_scene = [-(cosine_scale * translation[0] + sine_scale * translation[1]) / scale ** 2,
                    -(-sine_scale * translation[0] + cosine_scale * translation[1]) / scale ** 2]
    residuals = []
    for control, point in zip(controls, target):
        scene_x, scene_y = control["scene"]
        residual = math.hypot(cosine_scale * scene_x - sine_scale * scene_y + translation[0] - point[0],
                              sine_scale * scene_x + cosine_scale * scene_y + translation[1] - point[1])
        residuals.append(residual)
    return {"meters_per_unit": scale, "geo": {"lat": origin["lat"], "lon": origin["lon"],
             "heading_deg": (-math.degrees(math.atan2(sine_scale, cosine_scale))) % 360,
             "anchor_scene": anchor_scene}, "residuals_m": residuals,
            "rms_m": math.sqrt(sum(value * value for value in residuals) / count),
            "max_residual_m": max(residuals), "control_count": count,
            "redundant_control_check": count >= 3, "independent_check_points": 0,
            "dimensions": "horizontal-only", "horizontal_crs": "EPSG:4326",
            "fit_frame": "local-ENU", "distance_units": "metres",
            "vertical_datum": "unchanged", "controls": controls}
