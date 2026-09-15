#!/usr/bin/env python3
"""Prepare a CPU-only, bounded public-data context; /usr/bin/python3 has GIS dependencies."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from neighborhood_context import make_plan, prepare_context, refresh_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--three-root", type=Path)
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--refresh-preview", action="store_true")
    args = parser.parse_args()
    if args.refresh_preview:
        if args.output is None or args.three_root is None:
            parser.error("--output and --three-root are required for --refresh-preview")
        result = refresh_preview(args.output, args.three_root)
    elif args.capture_manifest is None:
        parser.error("--capture-manifest is required to plan or prepare a context")
    elif args.plan_only:
        result = make_plan(args.capture_manifest, args.grid, args.texture_size)
    else:
        if args.output is None:
            parser.error("--output is required unless --plan-only is selected")
        result = prepare_context(args.capture_manifest, args.output, args.three_root, args.grid, args.texture_size)
        result = {"status": result["status"], "output": str(args.output), "downloaded_bytes": result["downloaded_bytes"],
                  "patches": [{"id": patch["id"], "meshes": {name: mesh["triangles"] for name, mesh in patch["meshes"].items()}} for patch in result["patches"]]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
