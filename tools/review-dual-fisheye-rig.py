#!/usr/bin/env python3
"""Inspect a frozen training-only COLMAP text model without launching reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests
import rig_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--training-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new diagnostic output; existing evidence is preserved")
    before = {str(path): manifests.sha256_file(path) for path in (args.images, args.training_list)}
    names = [name.strip() for name in args.training_list.read_text().splitlines() if name.strip()]
    result = rig_diagnostics.summarize(rig_diagnostics.read_poses(args.images), names)
    if before != {str(path): manifests.sha256_file(path) for path in (args.images, args.training_list)}:
        raise ValueError("Frozen model inputs changed during review")
    result["source_hashes"] = before
    manifests.atomic_write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"pairs", "limitations", "source_hashes"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
