#!/usr/bin/env python3
"""Prepare dimensioned paired portal/room edits; execute the contained builder separately."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import architectural_edits as architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "inspect"])
    parser.add_argument("job", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--expected-generation", type=int)
    parser.add_argument("--architecture-id")
    parser.add_argument("--replace-slug", action="append", default=[])
    parser.add_argument("--geometry-method", choices=[architecture.LEGACY_GEOMETRY, architecture.FINISHED_GEOMETRY, architecture.JOINED_GEOMETRY],
                        default=architecture.LEGACY_GEOMETRY, help="Opt-in experimental floor finish; existing nominal floor remains the default")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.spec or not args.instruction or args.expected_generation is None:
            parser.error("Preparation requires the dimensioned spec, instruction and active generation")
        value = architecture.prepare(args.job.resolve(), json.loads(args.spec.read_text()), args.expected_generation, args.instruction,
                                     args.replace_slug, selected_geometry=args.geometry_method)
    else:
        if not args.architecture_id:
            parser.error("Inspect requires an architectural edit identifier")
        value = architecture.read(args.job.resolve(), args.architecture_id)
    print(json.dumps({key: value[key] for key in ("architecture_id", "base", "spec", "clip", "sha256", "scope")}, indent=2))


if __name__ == "__main__":
    main()
