#!/usr/bin/env python3
"""Inspect local terrain/imagery/vector packets without changing originals or publishing."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from neighborhood_inputs import inspect_input, prepare_owner_terrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--kind", choices=["landxml", "imagery", "elevation", "building_footprints", "parcels"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface")
    parser.add_argument("--source-crs")
    parser.add_argument("--vertical-datum")
    parser.add_argument("--vertical-units", choices=["meter", "metre", "foot", "USSurveyFoot"], help="Required explicit elevation-GeoTIFF vertical units")
    parser.add_argument("--geoid", help="Source-declared geoid realization; no geoid conversion is performed")
    parser.add_argument("--texture-size", type=int, help="Bounded square output texture size,16..2048")
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--prepare-terrain", action="store_true", help="Create a review-only composite context, not just an inspection receipt")
    parser.add_argument("--context", type=Path, help="Corrected verified public neighborhood context")
    parser.add_argument("--imagery-geotiff", type=Path, help="Owner RGB GeoTIFF, using its own explicit CRS")
    args = parser.parse_args()
    if args.prepare_terrain:
        if args.kind not in ("landxml", "elevation") or not args.context or args.preview:
            parser.error("--prepare-terrain requires --kind landxml|elevation and --context; --output is a new directory")
        if args.kind == "elevation" and not args.vertical_units:
            parser.error("Elevation preparation requires --vertical-units matching the source band metadata")
        result = prepare_owner_terrain(args.context, args.input, args.imagery_geotiff, args.output, args.surface, args.source_crs, args.vertical_datum,
                                       terrain_kind=args.kind, vertical_units=args.vertical_units, geoid=args.geoid, texture_size=args.texture_size)
    else:
        if args.context or args.imagery_geotiff or args.vertical_units or args.geoid or args.texture_size:
            parser.error("Composite context, units/geoid and texture options require --prepare-terrain")
        result = inspect_input(args.input, args.kind, args.output, args.surface, args.source_crs, args.vertical_datum, args.preview)
    print(json.dumps({"status": result["status"], "kind": result.get("kind", "owner-terrain-composite"), "receipt": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
