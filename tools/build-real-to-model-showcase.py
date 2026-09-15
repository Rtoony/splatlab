#!/usr/bin/env python3
"""Package actual reconstructed assets and local renderer code without publishing anything."""

import argparse
from html import escape
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import artifact_manifest as manifests
from reference_delivery import artifact_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--blender-alternative", type=Path)
    parser.add_argument("--tour", type=Path)
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    parser.add_argument("--mesh-delivery", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Your condo, beyond the photograph.")
    args = parser.parse_args()
    if not 1 <= len(args.title.strip()) <= 160 or any(ord(character) < 32 for character in args.title):
        parser.error("Use a short, nonempty capture title without control characters")
    output = args.output.resolve()
    sources = [args.evaluation.resolve(), args.reference.resolve()]
    if args.blender:
        sources.append(args.blender.resolve())
    if args.blender_alternative:
        sources.append(args.blender_alternative.resolve())
    if args.tour:
        sources.append(args.tour.resolve())
    sources.extend(path.resolve() for path in args.mesh_delivery)
    comparison_roots = [path.resolve() for path in args.comparison]
    if output.exists() or any(output.is_relative_to(path) for path in sources + comparison_roots):
        raise ValueError("Choose an unused showcase directory outside its sources")
    snapshot = {}
    for root in sources:
        receipt = json.loads((root / "receipt.json").read_text())
        snapshot[root / "receipt.json"] = manifests.sha256_file(root / "receipt.json")
        for name, digest in receipt["files"].items():
            path = artifact_path(root, name)
            if manifests.sha256_file(path) != digest:
                raise ValueError("Source artifact changed")
            snapshot[path] = digest
    comparisons = []
    mesh_deliveries = []
    for ordinal, directory in enumerate(args.mesh_delivery):
        delivery = json.loads((directory / "receipt.json").read_text())
        allowed_blenders = [path.resolve() for path in (args.blender, args.blender_alternative) if path]
        if (delivery.get("status") != "prepared-needs-registration-and-owner-review"
                or delivery.get("registration") is not None
                or not any(delivery["source_hashes"].get(str(root / "receipt.json")) == snapshot[root / "receipt.json"] for root in allowed_blenders)):
            raise ValueError("Mesh handoff must belong to a displayed, unregistered Blender candidate")
        role = "mesh" if args.blender and delivery["source_hashes"].get(str(args.blender.resolve() / "receipt.json")) == snapshot[args.blender.resolve() / "receipt.json"] else "mesh-alternative"
        mesh_deliveries.append({"href": f"mesh-deliveries/{ordinal}/external-references.json", "label": directory.name,
                                "mesh_href": f"mesh-deliveries/{ordinal}/inferred-surface.glb", "representation": role})
    if ((args.evaluation / "surfels.npz").exists()
            and args.evaluation.resolve() / "surfels.npz" not in snapshot):
        raise ValueError("Native surface parameters must be sealed by the evaluation receipt")
    for ordinal, directory in enumerate(comparison_roots):
        source_receipt = directory / "comparison.json"
        receipt = json.loads(source_receipt.read_text())
        if receipt.get("status") != "compared-needs-review":
            raise ValueError("Require completed geometry comparisons")
        snapshot[source_receipt] = manifests.sha256_file(source_receipt)
        for name, digest in receipt["files"].items():
            path = artifact_path(directory, name)
            if manifests.sha256_file(path) != digest:
                raise ValueError("Geometry comparison artifact changed")
            snapshot[path] = digest
        labels = receipt.get("method_labels", {"control": "Control", "filtered": "Filtered"})
        comparisons.append({"href": f"geometry/{ordinal}/index.html", "label": labels["control"] + " vs " + labels["filtered"]})
    if args.tour:
        tour = json.loads((args.tour / "receipt.json").read_text())
        if (tour.get("status") != "reopened-and-rendered-needs-review"
                or tour["source_hashes"].get(str(args.evaluation.resolve() / "receipt.json")) != snapshot[args.evaluation.resolve() / "receipt.json"]):
            raise ValueError("Native tour must belong to the displayed evaluated model")
    output.mkdir(parents=True)
    for path in (ROOT / "tools/real-to-model-viewer").iterdir():
        shutil.copyfile(path, output / path.name)
    page = output / "index.html"
    page.write_text(page.read_text().replace("Your condo, beyond the photograph.", escape(args.title)))
    shutil.copyfile(args.evaluation / "splat.ply", output / "splat.ply")
    shutil.copytree(args.reference, output / "reference")
    (output / "evaluation").mkdir()
    shutil.copytree(args.evaluation / "renders", output / "evaluation/renders")
    for name in ("index.html", "receipt.json"):
        shutil.copyfile(args.evaluation / name, output / "evaluation" / name)
    shutil.copyfile(args.evaluation / "splat.ply", output / "evaluation/splat.ply")
    native_parameters = (args.evaluation / "surfels.npz").is_file()
    if native_parameters:
        shutil.copyfile(args.evaluation / "surfels.npz", output / "evaluation/surfels.npz")
    if args.blender:
        shutil.copytree(args.blender, output / "blender")
    if args.blender_alternative:
        shutil.copytree(args.blender_alternative, output / "blender-alternative")
    if args.tour:
        shutil.copytree(args.tour, output / "tour")
    for ordinal, directory in enumerate(comparison_roots):
        shutil.copytree(directory, output / "geometry" / str(ordinal))
    for ordinal, directory in enumerate(args.mesh_delivery):
        shutil.copytree(directory, output / "mesh-deliveries" / str(ordinal))
    vendor = output / "vendor"
    vendor.mkdir()
    dependencies = ROOT / "frontend/node_modules"
    for name in ("three.module.js", "three.core.js"):
        shutil.copyfile(dependencies / "three/build" / name, vendor / name)
    shutil.copyfile(dependencies / "three/examples/jsm/controls/OrbitControls.js", vendor / "OrbitControls.js")
    shutil.copyfile(dependencies / "three/examples/jsm/postprocessing/Pass.js", vendor / "Pass.js")
    (vendor / "loaders").mkdir()
    (vendor / "utils").mkdir()
    shutil.copyfile(dependencies / "three/examples/jsm/loaders/GLTFLoader.js", vendor / "loaders/GLTFLoader.js")
    shutil.copyfile(dependencies / "three/examples/jsm/utils/BufferGeometryUtils.js", vendor / "utils/BufferGeometryUtils.js")
    shutil.copyfile(dependencies / "three/examples/jsm/utils/SkeletonUtils.js", vendor / "utils/SkeletonUtils.js")
    shutil.copyfile(dependencies / "three/LICENSE", vendor / "three-LICENSE.txt")
    shutil.copyfile(dependencies / "@sparkjsdev/spark/dist/spark.module.js", vendor / "spark.module.js")
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if (dependencies / "@sparkjsdev/spark" / name).exists():
            shutil.copyfile(dependencies / "@sparkjsdev/spark" / name, vendor / "spark-LICENSE.txt")
            break
    for path, digest in snapshot.items():
        if manifests.sha256_file(path) != digest:
            raise ValueError("Source changed during packaging")
    receipt = {"schema": "dev.splatlab.real-to-model-showcase/v1", "status": "packaged-needs-browser-review",
               "title": args.title,
               "created_at": manifests.utc_now(), "blender_scene": bool(args.blender), "registration": None,
               "native_tour": bool(args.tour), "native_parameters": native_parameters, "geometry_comparisons": comparisons,
               "blender_alternative": bool(args.blender_alternative),
               "mesh_deliveries": mesh_deliveries,
               "published": False, "source_hashes": {str(path): digest for path, digest in snapshot.items()},
               "evaluation_copy_scope": "review images, original receipt and Gaussian export; training depths stay in the original local evaluation directory",
               "renderer": {"three": "0.183.2", "spark": "2.1.0", "delivery": "copied installed local dependencies; no CDN"},
               "files": {str(path.relative_to(output)): manifests.sha256_file(path) for path in output.rglob("*") if path.is_file()}}
    manifests.atomic_write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output), "files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
