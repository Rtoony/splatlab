#!/usr/bin/env python3
"""Build an all-view before/source/after review without launching GPU work."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from capture_appearance import build_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = build_comparison(args.before, args.after, args.output)
    print(json.dumps({"status": result["status"], "views": len(result["views"]),
                      "before": result["before"], "after": result["after"]}, indent=2))


if __name__ == "__main__":
    main()
