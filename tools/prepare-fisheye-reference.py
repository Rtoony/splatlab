#!/usr/bin/env python3
"""Prepare CPU-only rectilinear references without reconstruction, training or model mutation."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fisheye_reference import build_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--sfm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--fov", type=float, default=100)
    args = parser.parse_args()
    receipt = build_bundle(args.pilot, args.sfm, args.output, args.width, args.fov)
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in {"artifacts", "source_hashes"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
