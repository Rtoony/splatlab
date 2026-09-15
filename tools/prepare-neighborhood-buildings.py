#!/usr/bin/env python3
"""Build bounded clean neighborhood layers without changing the source context."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from neighborhood_buildings import prepare_lowpoly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner-footprints-shp", type=Path)
    parser.add_argument("--owner-coverage-geojson", type=Path)
    parser.add_argument("--owner-validated", action="store_true")
    args = parser.parse_args()
    result = prepare_lowpoly(args.context, args.output, args.owner_footprints_shp, args.owner_coverage_geojson, args.owner_validated)
    print(json.dumps({"status": result["status"], "counts": result["counts"], "downloaded_bytes": result["downloaded_bytes"], "height_basis_counts": result["height_basis_counts"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
