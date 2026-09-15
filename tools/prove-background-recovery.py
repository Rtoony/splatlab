#!/usr/bin/env python3
"""Build a CPU background-recovery candidate in the private scene-proof clone."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import artifact_manifest as manifests
import background_recovery
import scene_revisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-generation", type=int, required=True)
    parser.add_argument("--selected-slug", default="cardboard-box-2")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    source = Path("/home/rtoony/projects/splatcli/outputs/3d/splat_aea04ab3")
    clone = repository / "data/spatial/scene-proof/outputs/splat_c0ffee"
    pointer = scene_revisions.active(clone)
    if not pointer or pointer["generation"] != args.expected_generation:
        raise SystemExit("Inspect the existing private fixture generation before proceeding")
    files = [source / "processed/transforms.json"]
    files += list((source / "processed/sparse/0").glob("*.bin"))
    files += list((source / "processed").glob("splatfacto*/**/dataparser_transforms.json"))
    files += list((source / "processed/images").iterdir())
    identities = {str(path.relative_to(source)): manifests.file_identity(path) for path in files}
    for relative, identity in identities.items():
        destination = clone / relative
        if destination.exists():
            if manifests.sha256_file(destination) != identity["sha256"]:
                raise SystemExit("Private evidence copy differs; refusing to overwrite it")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    started = time.monotonic()
    receipt = background_recovery.build(clone, args.selected_slug, args.expected_generation)
    if any(not manifests.same_file_identity(source / relative, identity) for relative, identity in identities.items()):
        raise SystemExit("Original evidence changed during the proof; inspect the result")
    report = {"source_unchanged": True, "active_unchanged": scene_revisions.active(clone) == pointer,
              "recovery_id": receipt["recovery_id"], "selected_slug": args.selected_slug,
              "seconds": time.monotonic() - started, "evidence": receipt["evidence"], "recovery": receipt["report"]}
    manifests.atomic_write_json(clone.parent.parent / "background-recovery-proof.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
