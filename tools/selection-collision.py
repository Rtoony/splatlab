#!/usr/bin/env python3
"""Prepare a collision partition without launching the bounded mesh worker."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import selection_collision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("review_id")
    parser.add_argument("--expected-generation", type=int, required=True)
    parser.add_argument("--voxel-m", type=float, default=.05)
    args = parser.parse_args()
    receipt = selection_collision.prepare(args.job.resolve(), args.review_id, args.expected_generation, args.voxel_m)
    print(json.dumps({key: receipt[key] for key in ("collision_id", "counts", "base", "recipe")}, indent=2))


if __name__ == "__main__":
    main()
