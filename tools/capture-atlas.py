#!/usr/bin/env python3
"""Build a private GPS/360 review atlas from an explicitly selected local handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import capture_atlas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--title", default="Exterior capture atlas")
    args = parser.parse_args()
    documents = capture_atlas.inspect_sources(args.handoff.resolve(), args.raw_root.resolve())
    document = capture_atlas.write_atlas(args.handoff.resolve(), documents, args.title)
    print(json.dumps({"atlas_id": document["atlas_id"], "summary": document["summary"],
                      "directory": str(capture_atlas.atlas_dir(document["atlas_id"]))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
