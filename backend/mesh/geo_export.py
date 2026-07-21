#!/usr/bin/env python3
"""Survey export: place a job's triangle mesh in a projected CRS and write CAD
deliverables. Runs in the dn-splatter-probe env (open3d + pyproj + ezdxf).

Usage: geo_export.py <mesh.ply> <out_dir> --params-json '<json>'

params: {"job_id", "meters_per_unit", "epsg", "max_faces",
         "geo": {"lat", "lon", "alt_m", "heading_deg", "anchor_scene"}}

Products (out_dir): site.dxf (3DFACE TIN, grid coordinates), surface.xml
(LandXML TIN surface — Civil 3D imports this as a real surface object),
site.geojson (anchor + footprint hull in WGS84, for a QGIS drop-in check),
geo_export.json (measured report incl. self-audits).

Grid calibration is PROBE-DERIVED, never convention-derived: the anchor plus
geodesic points 100 m true-north/true-east are projected through pyproj and the
grid rotation + point scale are measured from where they land, then a
known-distance self-audit must pass or the export FAILS (exit 1). Elevations
are unit-converted only (no grid scale) and carry whatever vertical datum the
anchor's alt_m was given in — EXIF altitudes are approximate; scene-relative
when alt_m is absent (flagged in the report).
"""
import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo_transform import enu_to_grid, scene_to_enu, xy_convex_hull  # noqa: E402

PROBE_DIST_M = 100.0


def _fail(msg: str) -> int:
    print(f"FATAL: {msg}", file=sys.stderr)
    return 1


def _crs_calibration(epsg: int, lat: float, lon: float) -> dict:
    """Project the anchor + two geodesic probes; measure the grid frame."""
    from pyproj import CRS, Geod, Transformer

    crs = CRS.from_epsg(epsg)
    if crs.is_geographic:
        raise ValueError(f"EPSG:{epsg} is geographic — survey export needs a projected CRS")
    unit_factor = float(crs.axis_info[0].unit_conversion_factor)  # meters per CRS unit
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    geod = Geod(ellps="WGS84")

    e0, n0 = tf.transform(lon, lat)
    lon_n, lat_n, _ = geod.fwd(lon, lat, 0.0, PROBE_DIST_M)
    lon_e, lat_e, _ = geod.fwd(lon, lat, 90.0, PROBE_DIST_M)
    en, nn = tf.transform(lon_n, lat_n)
    ee, ne = tf.transform(lon_e, lat_e)

    dn = np.array([en - e0, nn - n0]) * unit_factor  # north probe vector, meters
    de = np.array([ee - e0, ne - n0]) * unit_factor
    grid_rot_deg = math.degrees(math.atan2(dn[0], dn[1]))  # grid azimuth of true north
    k_north = float(np.hypot(*dn)) / PROBE_DIST_M
    k_east = float(np.hypot(*de)) / PROBE_DIST_M
    # East probe must land 90 deg CW of the north probe in the grid frame too.
    east_az = math.degrees(math.atan2(de[0], de[1]))
    ortho_residual_deg = abs(((east_az - grid_rot_deg - 90.0) + 180.0) % 360.0 - 180.0)
    return {
        "epsg": epsg,
        "crs_name": crs.name,
        "unit_factor_m": unit_factor,
        "e0": float(e0),
        "n0": float(n0),
        "grid_rot_deg": round(grid_rot_deg, 6),
        "scale_factor": round((k_north + k_east) / 2.0, 9),
        "probe": {
            "k_north": round(k_north, 9),
            "k_east": round(k_east, 9),
            "ortho_residual_deg": round(ortho_residual_deg, 6),
        },
        "transformer": tf,
    }


def _unit_name(unit_factor: float) -> tuple[str, str, int]:
    """-> (human unit, LandXML linearUnit, DXF $INSUNITS)."""
    if abs(unit_factor - 1200.0 / 3937.0) < 1e-9:
        return "US survey foot", "USSurveyFoot", 2
    if abs(unit_factor - 0.3048) < 1e-9:
        return "foot", "foot", 2
    if abs(unit_factor - 1.0) < 1e-12:
        return "meter", "meter", 6
    return f"{unit_factor} m/unit", "meter", 0


def _write_dxf(path: Path, tris_xyz: np.ndarray, faces: np.ndarray, anchor_en: tuple,
               job_id: str, epsg: int, insunits: int) -> dict:
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = insunits
    msp = doc.modelspace()
    doc.layers.add("SPLAT-TIN", color=8)
    doc.layers.add("SPLAT-ANCHOR", color=1)
    for f in faces:
        a, b, c = (tuple(tris_xyz[i]) for i in f)
        msp.add_3dface([a, b, c, c], dxfattribs={"layer": "SPLAT-TIN"})
    msp.add_point((*anchor_en, 0.0), dxfattribs={"layer": "SPLAT-ANCHOR"})
    msp.add_text(
        f"SplatLab anchor {job_id} EPSG:{epsg}",
        height=5.0,
        dxfattribs={"layer": "SPLAT-ANCHOR", "insert": (anchor_en[0] + 5.0, anchor_en[1] + 5.0)},
    )
    doc.saveas(path)
    audit = ezdxf.readfile(path).audit()
    return {
        "entities": len(ezdxf.readfile(path).modelspace()),
        "audit_errors": len(audit.errors),
    }


def _write_landxml(path: Path, pts: np.ndarray, faces: np.ndarray, linear_unit: str,
                   job_id: str, epsg: int, crs_name: str) -> None:
    """LandXML 1.2 TIN surface. Point records are NORTHING EASTING ELEV order."""
    now = datetime.now(timezone.utc)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2"',
        f'  date="{now.date().isoformat()}" time="{now.strftime("%H:%M:%S")}">',
        f'  <Units><Imperial linearUnit="{linear_unit}" areaUnit="squareFoot"'
        if "oot" in linear_unit
        else f'  <Units><Metric linearUnit="{linear_unit}" areaUnit="squareMeter"',
        '    temperatureUnit="fahrenheit" pressureUnit="inHG"/></Units>'
        if "oot" in linear_unit
        else '    temperatureUnit="celsius" pressureUnit="milliBars"/></Units>',
        f'  <Application name="SplatLab" desc="splat-derived TIN {job_id} ({crs_name} EPSG:{epsg})"/>',
        "  <Surfaces>",
        f'    <Surface name="SPLAT_{job_id}">',
        "      <Definition surfType=\"TIN\">",
        "        <Pnts>",
    ]
    for i, p in enumerate(pts, start=1):
        lines.append(f'          <P id="{i}">{p[1]:.4f} {p[0]:.4f} {p[2]:.4f}</P>')
    lines.append("        </Pnts>")
    lines.append("        <Faces>")
    for f in faces:
        lines.append(f"          <F>{f[0] + 1} {f[1] + 1} {f[2] + 1}</F>")
    lines.append("        </Faces>")
    lines += ["      </Definition>", "    </Surface>", "  </Surfaces>", "</LandXML>"]
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--params-json", required=True)
    args = ap.parse_args()

    p = json.loads(args.params_json)
    job_id = p.get("job_id", "unknown")
    mpu = float(p["meters_per_unit"])
    geo = p["geo"]
    epsg = int(p.get("epsg", 2226))
    max_faces = int(p.get("max_faces", 50_000))
    anchor_scene = geo.get("anchor_scene") or [0.0, 0.0]

    mesh_path, out_dir = Path(args.mesh), Path(args.out_dir)
    if not mesh_path.is_file():
        return _fail(f"mesh not found: {mesh_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    n_tris_full = len(mesh.triangles)
    if n_tris_full == 0:
        return _fail("mesh is empty")
    if n_tris_full > max_faces:
        mesh = mesh.simplify_quadric_decimation(max_faces)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        return _fail("decimation produced an empty mesh")

    cal = _crs_calibration(epsg, geo["lat"], geo["lon"])
    tf = cal.pop("transformer")
    unit_name, landxml_unit, insunits = _unit_name(cal["unit_factor_m"])
    alt_m = geo.get("alt_m")
    elev0_units = (alt_m or 0.0) / cal["unit_factor_m"]

    enu = scene_to_enu(verts, mpu, geo.get("heading_deg", 0.0), anchor_scene)
    grid = enu_to_grid(
        enu, cal["e0"], cal["n0"], cal["unit_factor_m"],
        cal["grid_rot_deg"], cal["scale_factor"], elev0_units,
    )

    # ── self-audits: fail loud, never ship a silently-misplaced surface ──────
    audits: dict = {"probe": cal["probe"]}
    if cal["probe"]["ortho_residual_deg"] > 0.1:
        return _fail(f"probe orthogonality residual {cal['probe']['ortho_residual_deg']} deg")
    i, j = 0, int(np.argmax(np.linalg.norm(verts - verts[0], axis=1)))
    ground_m = float(np.linalg.norm(verts[j, :2] - verts[i, :2])) * mpu
    grid_m = float(np.linalg.norm(grid[j, :2] - grid[i, :2])) * cal["unit_factor_m"]
    dist_residual = abs(grid_m / cal["scale_factor"] - ground_m)
    audits["known_distance"] = {
        "ground_m": round(ground_m, 6),
        "grid_m": round(grid_m, 6),
        "residual_m": round(dist_residual, 9),
    }
    if ground_m > 0 and dist_residual > max(1e-6 * ground_m, 1e-6):
        return _fail(f"known-distance audit failed: residual {dist_residual} m over {ground_m} m")
    back_lon, back_lat = tf.transform(cal["e0"], cal["n0"], direction="INVERSE")
    audits["anchor_roundtrip_deg"] = round(
        float(np.hypot(back_lat - geo["lat"], back_lon - geo["lon"])), 12
    )

    dxf_info = _write_dxf(
        out_dir / "site.dxf", grid, faces, (cal["e0"], cal["n0"]), job_id, epsg, insunits
    )
    if dxf_info["audit_errors"]:
        return _fail(f"DXF audit reported {dxf_info['audit_errors']} errors")
    _write_landxml(
        out_dir / "surface.xml", grid, faces, landxml_unit, job_id, epsg, cal["crs_name"]
    )

    hull = xy_convex_hull(grid[:, :2])
    hull_ll = [list(tf.transform(e, n, direction="INVERSE")) for e, n in hull]
    site_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [geo["lon"], geo["lat"]]},
             "properties": {"name": f"SplatLab anchor {job_id}"}},
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [hull_ll + hull_ll[:1]]},
             "properties": {"name": f"SplatLab mesh footprint {job_id}", "epsg": epsg}},
        ],
    }
    (out_dir / "site.geojson").write_text(json.dumps(site_geojson, indent=2))

    report = {
        "v": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epsg": epsg,
        "crs_name": cal["crs_name"],
        "unit": unit_name,
        "grid_rot_deg": cal["grid_rot_deg"],
        "scale_factor": cal["scale_factor"],
        "anchor": {"lat": geo["lat"], "lon": geo["lon"], "e": round(cal["e0"], 4),
                   "n": round(cal["n0"], 4), "scene": anchor_scene,
                   "scene_defaulted": geo.get("anchor_scene") is None},
        "meters_per_unit": mpu,
        "heading_deg": geo.get("heading_deg", 0.0),
        "vertical": (
            f"anchor alt {alt_m} m (datum as given; EXIF altitudes are approximate)"
            if alt_m is not None else "scene-relative (no anchor altitude set)"
        ),
        "tris_source": n_tris_full,
        "tris_exported": int(len(faces)),
        "verts_exported": int(len(verts)),
        "extent_units": [round(float(v), 3) for v in (grid.max(axis=0) - grid.min(axis=0))],
        "audits": audits,
        "dxf_entities": dxf_info["entities"],
        "artifacts": {"dxf": "site.dxf", "landxml": "surface.xml", "geojson": "site.geojson"},
    }
    (out_dir / "geo_export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
