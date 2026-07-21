#!/usr/bin/env python3
"""PNEZD ground points -> cdt survey_to_surface -> contour DXF + result JSON.

Runs in the CDT venv (~/projects/civil-design-tools/.venv) — this IS the
production civil recipe (TIN + contours on standards-resolved office layers,
provenance sidecar, fail-loud on zero points/contours). Zero geometry code here.

Productionized 2026-07-21 from ~/tools/solidify-probe/cdt_contours.py.

Usage: contours_build.py <pnezd.txt> <out.dxf> [--epsg 2226] [--minor 0.5]
       [--major 2.5] [--tin-faces]
Writes <out.dxf> plus contours_result.json alongside it.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("dxf_out")
    ap.add_argument("--epsg", type=int, default=2226)
    ap.add_argument("--minor", type=float, default=0.5)
    ap.add_argument("--major", type=float, default=2.5)
    ap.add_argument("--tin-faces", action="store_true")
    args = ap.parse_args()

    from workflows.recipes import survey_to_surface

    result = survey_to_surface(
        args.points,
        args.dxf_out,
        epsg=args.epsg,
        points_format="pnezd",
        minor_interval_ft=args.minor,
        major_interval_ft=args.major,
        draw_tin_faces=args.tin_faces,
        project_ref="splatlab-contours",
    )
    out = {
        "points_imported": result.point_count,
        "tin_triangles": result.triangle_count,
        "contours_drawn": result.contour_emissions,
        "minor_interval_ft": args.minor,
        "major_interval_ft": args.major,
        "watermarked": result.watermarked,
        "warnings": [str(w) for w in result.warnings],
    }
    (Path(args.dxf_out).parent / "contours_result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
