#!/usr/bin/env python3
"""Compose private, ground-referenced GIS overlays with a sealed low-poly package."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from neighborhood_layers import prepare_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--lowpoly", type=Path, required=True)
    parser.add_argument("--shapefiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gps-geojson", type=Path)
    parser.add_argument("--contour-vertical-units", choices=["meter", "metre", "foot", "USSurveyFoot"])
    parser.add_argument("--resource-catalog", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_layers(args.context, args.lowpoly, args.shapefiles, args.output, args.gps_geojson, args.contour_vertical_units, args.resource_catalog), indent=2))


if __name__ == "__main__":
    main()
