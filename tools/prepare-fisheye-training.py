#!/usr/bin/env python3
"""Prepare split-preserving fixed-camera training inputs; never launches training."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from fisheye_training import prepare


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--localization", required=True, type=Path)
    parser.add_argument("--sfm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-mask-review", type=Path)
    args = parser.parse_args()
    result = prepare(args.reference, args.localization, args.sfm, args.output, args.raw_mask_review)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
