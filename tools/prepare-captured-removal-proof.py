#!/usr/bin/env python3
"""Copy retained training/instance evidence into the private scene proof fixture."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import artifact_manifest as manifests
import scene_revisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-generation", type=int, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    source = Path("/home/rtoony/projects/splatcli/outputs/3d/splat_aea04ab3")
    clone = repository / "data/spatial/scene-proof/outputs/splat_c0ffee"
    pointer = scene_revisions.active(clone)
    if not pointer or pointer["generation"] != args.expected_generation:
        raise SystemExit("Inspect the private fixture generation before proceeding")
    runs = list((source / "processed").glob("splatfacto/*/config.yml"))
    if len(runs) != 1:
        raise SystemExit("Expected one retained training run")
    original_config = runs[0]
    files = [original_config, source / "_scene/inventory.json", source / "_mesh/mesh.ply"]
    files += list(original_config.parent.glob("nerfstudio_models/*.ckpt"))
    files += list((source / "_scene").glob("instance_*.npz"))
    files += list((source / "processed").glob("*.ply"))
    identities = {str(path.relative_to(source)): manifests.file_identity(path) for path in files}
    original_prefix = "\n".join("- " + part for part in source.parts)
    private_prefix = "\n".join("- " + part for part in clone.parts)
    for relative, identity in identities.items():
        destination = clone / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source / relative == original_config:
            config = original_config.read_text()
            if config.count(original_prefix) != 2:
                raise SystemExit("Unexpected training YAML paths; refusing an unsafe rewrite")
            rewritten = config.replace(original_prefix, private_prefix).replace("cache_images: gpu", "cache_images: cpu")
            if destination.exists() and destination.read_text() != rewritten:
                raise SystemExit("Private training config differs; refusing to overwrite")
            destination.write_text(rewritten)
        elif destination.exists():
            if manifests.sha256_file(destination) != identity["sha256"]:
                raise SystemExit("Private input differs; refusing to overwrite")
        else:
            shutil.copy2(source / relative, destination)
    if any(not manifests.same_file_identity(source / relative, identity) for relative, identity in identities.items()):
        raise SystemExit("Original evidence changed during the copy")
    report = {"original_inputs": identities, "source_unchanged": True,
              "active_unchanged": scene_revisions.active(clone) == pointer,
              "private_config": str(clone / original_config.relative_to(source)),
              "config_changes": ["private capture/output paths", "CPU image cache"],
              "scope": "private evidence preparation only; no training or activation"}
    manifests.atomic_write_json(clone.parent.parent / "captured-removal-inputs.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "original_inputs"}, indent=2))


if __name__ == "__main__":
    main()
