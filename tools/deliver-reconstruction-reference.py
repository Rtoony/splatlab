#!/usr/bin/env python3
"""Package existing reconstructed points/cameras; no new geometry estimation or GPU execution."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from reference_delivery import build_delivery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_delivery(args.reference, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in {"files", "source_hashes"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
