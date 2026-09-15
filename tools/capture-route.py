#!/usr/bin/env python3
"""Register an explicitly selected local video or build a checkpointed panorama route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import capture_records
import route_builder
import route_reconstruction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("source", type=Path)
    build = commands.add_parser("build")
    build.add_argument("capture_id")
    build.add_argument("--start", type=float, default=0)
    build.add_argument("--end", type=float)
    build.add_argument("--interval", type=float, default=5)
    build.add_argument("--width", type=int, default=2048)
    build.add_argument("--budget", type=int, default=3600)
    resume = commands.add_parser("resume")
    resume.add_argument("route_id")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("route_id")
    prepare.add_argument("--section", type=float, default=45)
    prepare.add_argument("--overlap", type=float, default=10)
    prepare.add_argument("--nadir-exclusion", type=float, default=50)
    split = commands.add_parser("split-transforms")
    split.add_argument("plan", type=Path)
    split.add_argument("transforms", type=Path)
    split.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "split-transforms":
        if args.output.exists():
            parser.error("Output already exists; choose a new filename to preserve the original")
        result = route_reconstruction.apply_group_split(json.loads(args.transforms.read_text()), json.loads(args.plan.read_text()))
        capture_records.manifests.atomic_write_json(args.output, result)
        print(json.dumps({"output": str(args.output), "frames": len(result["frames"])}))
        return 0
    if args.command == "prepare":
        result = route_reconstruction.prepare(args.route_id, route_reconstruction.ReconstructionSpec(
            section_s=args.section, overlap_s=args.overlap, nadir_exclusion_deg=args.nadir_exclusion))
        print(json.dumps({"fingerprint": result["fingerprint"], "groups": len(result["groups"]),
                          "sections": len(result["sections"]), "status": result["status"]}, indent=2))
        return 0
    if args.command == "inspect":
        document = capture_records.inspect_capture(args.source)
        print(json.dumps({"capture_id": document["capture_id"], "duration_s": document["duration_s"],
                          "gps_count": document["gps"]["count"], "clock": document["clock"]}, indent=2))
        return 0
    if args.command == "build":
        spec = route_builder.RouteSpec(capture_id=args.capture_id, start_s=args.start, end_s=args.end,
                                       interval_s=args.interval, width=args.width, budget_s=args.budget)
        route_id = route_builder.create_route(spec)["route_id"]
    else:
        route_id = args.route_id
        (route_builder.route_dir(route_id) / "stop-requested").unlink(missing_ok=True)
    print(json.dumps({"route_id": route_id}), flush=True)
    result = route_builder.build_route(route_id)
    print(json.dumps({"route_id": route_id, "status": result["status"], "seconds": result["seconds"],
                      "ready": sum(point["status"] == "ready" for point in result["points"]),
                      "error": result.get("error")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
