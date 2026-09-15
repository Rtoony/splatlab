#!/usr/bin/env python3
"""Create a new, unregistered mesh handoff without changing the condo model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from capture_delivery import build_delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("blender", "surface", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    receipt = build_delivery(args.blender, args.surface, args.output)
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
