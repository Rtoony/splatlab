"""Verify a second raw-lens intake and prepare a local review gallery, without reconstruction."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import html
import json
import math
from pathlib import Path
import re
import shutil
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import artifact_manifest as manifests


def inspect(pilot_root: Path, previous_root: Path) -> dict:
    document = json.loads((pilot_root / "pilot.json").read_text())
    previous = json.loads((previous_root / "pilot.json").read_text())
    if (document.get("schema") != "dev.splatlab.dual-fisheye-pilot/v1"
            or document.get("status") != "decoded-needs-camera-and-mask-review"
            or document.get("clock", {}).get("paired_container_pts_match") is not True):
        raise ValueError("A completed paired raw-lens pilot is required")
    if (document["capture_id"] == previous["capture_id"]
            or document["source"]["sha256"] == previous["source"]["sha256"]):
        raise ValueError("The second pilot must come from a genuinely different video")
    source = Path(document["source"]["path"])
    if (not manifests.same_file_identity(source, document["source"])
            or manifests.sha256_file(source) != document["source"]["sha256"]):
        raise ValueError("Raw source changed since intake")
    groups, views = document["groups"], document["views"]
    if (not 5 <= len(groups) <= 96 or len(views) != len(groups) * 2
            or len({group["group_id"] for group in groups}) != len(groups)
            or len({view["image"] for view in views}) != len(views)):
        raise ValueError("Invalid bounded paired frame inventory")
    if {group["split"] for group in groups} != {"train", "val", "test"}:
        raise ValueError("Preserve all three nonempty timestamp-grouped splits")
    previous_time = None
    for group in groups:
        pair = [view for view in views if view["group_id"] == group["group_id"]]
        if len(pair) != 2 or {view["lens_id"] for view in pair} != {"lens-0", "lens-1"}:
            raise ValueError("Each group requires two physical lenses")
        timestamps = []
        for view in pair:
            if (view["split"] != group["split"]
                    or view["source_decoded_frame_index"] != group["source_decoded_frame_index"]
                    or not re.fullmatch(r"lens-[01]/frame-\d{6}\.jpg", view["image"])
                    or view["image"] != view["lens_id"] + "/" + group["group_id"] + ".jpg"):
                raise ValueError("Image lineage or paired split changed")
            image_path = pilot_root / "images" / view["image"]
            if image_path.resolve() != image_path.absolute() or manifests.sha256_file(image_path) != view["sha256"]:
                raise ValueError("Pilot image changed or escaped its package")
            with Image.open(image_path) as picture:
                if picture.size != (document["output_width"], document["output_width"]):
                    raise ValueError("Image dimensions changed")
                picture.verify()
            time_base = Fraction(view["time_base"])
            if time_base <= 0 or not math.isfinite(view["time_s"]) or abs(float(view["pts"] * time_base) - view["time_s"]) > 1e-6:
                raise ValueError("Invalid container timestamp basis")
            timestamps.append(view["pts"] * time_base)
        if timestamps[0] != timestamps[1]:
            raise ValueError("Paired container timestamps disagree")
        if previous_time is not None and timestamps[0] <= previous_time:
            raise ValueError("Timestamp groups must increase")
        previous_time = timestamps[0]
    for split in ("train", "val", "test"):
        actual = (pilot_root / f"{split}-images.txt").read_text().splitlines()
        expected = [view["image"] for view in views if view["split"] == split]
        if actual != expected:
            raise ValueError("Split list differs from the source lineage")
    if not manifests.same_file_identity(source, document["source"]):
        raise ValueError("Raw source changed during verification")
    return {"schema": "dev.splatlab.second-capture-intake-review/v1", "status": "intake-verified-reconstruction-not-run",
            "created_at": manifests.utc_now(), "capture_id": document["capture_id"], "source_name": source.name,
            "source_sha256": document["source"]["sha256"], "independent_from_capture": previous["capture_id"],
            "pilot_sha256": manifests.sha256_file(pilot_root / "pilot.json"), "groups": len(groups), "views": len(views),
            "splits": dict(Counter(view["split"] for view in views)), "native_streams": document["streams"],
            "paired_container_pts_match": True, "reconstruction_validated": False, "registration": None,
            "limitations": ["This validates reusable intake, not Gaussian or geometry quality on the second clip.",
                            "Container timestamp agreement does not verify physical exposure synchronization.",
                            "Lens intrinsics, rig extrinsics, metric scale and GPS-to-frame alignment remain unverified.",
                            "Operator, hand, foliage, flare and reflections need clip-specific masking and review."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior evidence; use a new review directory")
    receipt = inspect(args.pilot, args.previous)
    document = json.loads((args.pilot / "pilot.json").read_text())
    args.output.mkdir(parents=True)
    cards = []
    for view in sorted(document["views"], key=lambda item: (item["source_decoded_frame_index"], item["lens_id"])):
        filename = "images/" + view["image"]
        destination = args.output / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.pilot / filename, destination)
        if manifests.sha256_file(destination) != view["sha256"]:
            raise ValueError("Review image copy changed")
        cards.append(f'<article><b>{html.escape(view["lens_id"])} · {view["time_s"]:.3f}s · {html.escape(view["split"])}</b>'
                     f'<a href="{filename}"><img loading="lazy" src="{filename}" alt="Unstitched source lens"></a></article>')
    body = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Second condo capture — raw-lens intake</title><style>body{font:16px system-ui;margin:3vw;background:#152128;color:#edf3eb}'
            'section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}img{width:100%}article{padding:12px;background:#26343c}'
            'a{color:#acded0}aside{padding:20px;background:#614b22}@media(max-width:600px){section{grid-template-columns:1fr}}</style>'
            f'<h1>A genuinely different video: {html.escape(receipt["source_name"])}</h1><aside>{receipt["views"]} raw lens images, {receipt["groups"]} paired timestamps. '
            f'{receipt["splits"]["train"]} train / {receipt["splits"]["val"]} validation / {receipt["splits"]["test"]} test. Intake verified; reconstruction NOT run. Original lens views are not stitched panoramas.</aside>'
            '<p>Inspect both lenses before reconstruction. Operator, hand, foliage, flare and reflections may require clip-specific exclusion. '
            'Do not reuse the frontage masks blindly. No metric scale, GPS placement or condo alignment is claimed.</p>'
            '<p><a href="receipt.json">Verification receipt</a></p><section>' + ''.join(cards) + '</section>')
    (args.output / "index.html").write_text(body)
    receipt["tool_sha256"] = manifests.sha256_file(Path(__file__))
    receipt["files"] = {str(filename.relative_to(args.output)): manifests.sha256_file(filename)
                        for filename in args.output.rglob("*") if filename.is_file()}
    manifests.atomic_write_json(args.output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
