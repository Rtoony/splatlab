#!/usr/bin/env python3
"""Prepare an inferred-visibility study or derive a separately reviewable recovery; no GPU launch."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import background_visibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "derive"])
    parser.add_argument("job", type=Path)
    parser.add_argument("identifier", help="Recovery ID for preparation, visibility ID for derivation")
    parser.add_argument("--expected-generation", type=int, required=True)
    args = parser.parse_args()
    result = getattr(background_visibility, args.action)(args.job.resolve(), args.identifier, args.expected_generation)
    print(json.dumps({key: result[key] for key in ("visibility_id", "recovery_id", "base", "sha256", "visibility_review", "report") if key in result}, indent=2))


if __name__ == "__main__":
    main()
