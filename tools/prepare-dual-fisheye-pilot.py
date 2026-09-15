#!/usr/bin/env python3
"""Prepare a bounded, unstitched two-lens pilot without estimating camera poses."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests
import capture_records as captures


SCHEMA = "dev.splatlab.dual-fisheye-pilot/v1"


def make_plan(capture: dict, start_frame: int, stride: int, groups: int, width: int) -> dict:
    bounds = ((start_frame, 0, 3600), (stride, 1, 300), (groups, 5, 96), (width, 512, 1920))
    if any(type(value) is not int or not lower <= value <= upper for value, lower, upper in bounds):
        raise ValueError("Invalid bounded pilot selection")
    indices = [start_frame + index * stride for index in range(groups)]
    if indices[-1] > 3600 or width % 2:
        raise ValueError("Select at most 3600 decoded frames and an even output width")
    streams = capture["streams"]
    if (len(streams) != 2 or any(stream.get("width", 0) != stream.get("height")
                               or stream.get("width", 0) < width for stream in streams)
            or streams[0]["index"] == streams[1]["index"]):
        raise ValueError("Select two distinct square video streams without upscaling")
    if any(type(stream["index"]) is not int or stream["index"] < 0 for stream in streams):
        raise ValueError("Invalid video stream indices")
    source = capture["source"]
    if not re.fullmatch(r"[a-f0-9]{64}", source.get("sha256", "")):
        raise ValueError("Inspect and hash the source capture first")
    timestamp_groups = []
    for ordinal, frame_index in enumerate(indices):
        split = "test" if ordinal % 5 == 2 else "val" if ordinal % 5 == 3 else "train"
        timestamp_groups.append({"group_id": f"frame-{frame_index:06d}",
                                 "source_decoded_frame_index": frame_index, "split": split})
    return {"schema": SCHEMA, "status": "planned-not-decoded", "capture_id": capture["capture_id"],
            "source": source, "streams": streams, "output_width": width, "groups": timestamp_groups,
            "projection": "raw-fisheye-resized-no-warp", "autorotate": False,
            "calibration": {"intrinsics": "unknown", "rig_extrinsics": "unknown",
                            "physical_optical_centers_assumed_equal": False},
            "clock": {"basis": "decoded-container-pts", "exposure_sync_verified": False,
                      "utc_alignment_verified": False},
            "mask_status": "not-prepared-requires-visual-review",
            "evaluation": {"split_unit": "paired-source-frame-index",
                           "heldout_pose_policy": "localize-against-frozen-training-model",
                           "heldout_results": "not-evaluated"},
            "warnings": ["Square streams alone do not prove fisheye calibration; explicitly select known raw lenses.",
                         "Matching container PTS does not establish hardware exposure synchronization.",
                         "No stitching, stabilization, pose estimation, metric scale or training is performed.",
                         "Physical lenses are not co-centered virtual crops from a stitched panorama.",
                         "Inspect operator, sky, reflections and lens-edge masks before reconstruction."]}


def decode_command(source: Path, stream_index: int, indices: list[int], width: int, output: Path) -> list[str]:
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    filters = f"select={selection},showinfo=checksum=0,scale={width}:{width}:flags=lanczos,setsar=1"
    return [shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-n",
            "-loglevel", "info", "-copyts", "-hwaccel", "none", "-threads", "2",
            "-noautorotate", "-i", str(source), "-map", f"0:{stream_index}", "-an", "-sn", "-dn",
            "-filter_threads", "2", "-vf", filters, "-frames:v", str(len(indices)),
            "-fps_mode", "passthrough", "-q:v", "2", "-threads", "2", "-start_number", "0",
            str(output / "decoded-%06d.jpg")]


def parse_timestamps(log: str, count: int) -> list[dict]:
    bases = re.findall(r"config in time_base:\s*(\d+/\d+)", log)
    if len(bases) != 1:
        raise ValueError("Missing or reinitialized decoded timestamp time base")
    try:
        time_base = Fraction(bases[0])
    except ZeroDivisionError as error:
        raise ValueError("Invalid decoded timestamp time base") from error
    if time_base <= 0:
        raise ValueError("Invalid decoded timestamp time base")
    values = re.findall(r"\[Parsed_showinfo_[^\]]+\]\s+n:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", log)
    if len(values) != count or [int(ordinal) for ordinal, _ in values] != list(range(count)):
        raise ValueError("Decoded timestamp count/order does not match requested frames")
    times = [int(pts) * time_base for _, pts in values]
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("Decoded presentation timestamps are not strictly increasing")
    return [{"pts": int(pts), "time_base": str(time_base), "time_s": float(timestamp)}
            for (_, pts), timestamp in zip(values, times)]


def paired_timestamps(first: list[dict], second: list[dict]) -> bool:
    return bool(first) and len(first) == len(second) and all(
        left["pts"] * Fraction(left["time_base"]) == right["pts"] * Fraction(right["time_base"])
        for left, right in zip(first, second))


def extract(plan: dict, output: Path, budget_s: int) -> dict:
    if type(budget_s) is not int or not 30 <= budget_s <= 1200:
        raise ValueError("Decode budget must be 30–1200 seconds")
    source = Path(plan["source"]["path"]).resolve(strict=True)
    output = output.resolve()
    if output.exists() or output == source.parent or source.parent in output.parents:
        raise ValueError("Choose a new output directory outside the source-media directory")
    if not manifests.same_file_identity(source, plan["source"]):
        raise ValueError("Source identity changed; inspect it again")
    started = time.monotonic()
    deadline = started + budget_s
    if manifests.sha256_file(source) != plan["source"]["sha256"]:
        raise ValueError("Source content hash changed; inspect it again")
    if not manifests.same_file_identity(source, plan["source"]):
        raise ValueError("Source changed during content verification")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {**plan, "status": "decoding", "started_at": manifests.utc_now(), "views": [],
               "tool_sha256": manifests.sha256_file(Path(__file__)), "python_version": sys.version}
    manifests.atomic_write_json(output / "pilot.json", receipt)
    try:
        all_timestamps = []
        indices = [group["source_decoded_frame_index"] for group in plan["groups"]]
        for lens_index, stream in enumerate(plan["streams"]):
            directory = output / "images" / f"lens-{lens_index}"
            directory.mkdir(parents=True)
            command = decode_command(source, stream["index"], indices, plan["output_width"], directory)
            log_path = output / f"decode-lens-{lens_index}.log"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pilot decode budget exhausted")
            with log_path.open("w") as log_handle:
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log_handle,
                                        timeout=remaining, env=captures.worker_env())
            if result.returncode:
                raise ValueError(f"Lens {lens_index} decode failed; inspect its retained log")
            timestamps = parse_timestamps(log_path.read_text(), len(indices))
            if len(list(directory.glob("decoded-*.jpg"))) != len(indices):
                raise ValueError("Decoded image count does not match requested frames")
            all_timestamps.append(timestamps)
            for ordinal, (group, timestamp) in enumerate(zip(plan["groups"], timestamps)):
                path = directory / f"decoded-{ordinal:06d}.jpg"
                with Image.open(path) as picture:
                    if picture.size != (plan["output_width"], plan["output_width"]):
                        raise ValueError("Unexpected decoded image dimensions")
                    picture.verify()
                destination = directory / f"{group['group_id']}.jpg"
                path.rename(destination)
                receipt["views"].append({**group, **timestamp, "lens_id": f"lens-{lens_index}",
                                         "stream_index": stream["index"],
                                         "image": destination.relative_to(output / "images").as_posix(),
                                         **manifests.file_identity(destination)})
        if not paired_timestamps(*all_timestamps):
            raise ValueError("Lens frame indices do not have matching container PTS; do not treat them as synchronized")
        if not manifests.same_file_identity(source, plan["source"]):
            raise ValueError("Source changed during decode")
        for split in ("train", "val", "test"):
            names = [view["image"] for view in receipt["views"] if view["split"] == split]
            (output / f"{split}-images.txt").write_text("".join(f"{name}\n" for name in names))
        receipt["clock"] = {**receipt["clock"], "paired_container_pts_match": True}
        receipt["status"] = "decoded-needs-camera-and-mask-review"
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        receipt["status"] = "failed-not-for-reconstruction"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["seconds"] = round(time.monotonic() - started, 3)
        receipt["finished_at"] = manifests.utc_now()
        manifests.atomic_write_json(output / "pilot.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--budget", type=int, default=600)
    args = parser.parse_args()
    plan = make_plan(captures.load_capture(args.capture_id), args.start_frame, args.stride, args.groups, args.width)
    if args.output is None:
        print(json.dumps(plan, indent=2))
        return 0
    result = extract(plan, args.output, args.budget)
    print(json.dumps({"status": result["status"], "capture_id": result["capture_id"],
                      "groups": len(result["groups"]), "views": len(result["views"]),
                      "seconds": result["seconds"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
