"""Read back the saved independent Blender scene without modifying it."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    gate = Path(__file__).resolve().parents[1] / "tools/splatlab-compute-gate.sh"
    if subprocess.run([str(gate), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Use the SplatLab compute gate")
    receipt = json.loads((args.directory / "receipt.json").read_text())
    if args.proof.exists():
        raise ValueError("Choose a new proof file")
    for name, digest in receipt["files"].items():
        if hashlib.sha256((args.directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Saved Blender delivery changed")
    scene = bpy.context.scene
    if Path(bpy.data.filepath).resolve() != (args.directory / "condo-capture-review.blend").resolve():
        raise ValueError("The expected independent scene was not reopened")
    surfaces = [obj for obj in scene.objects if obj.type == "MESH" and len(obj.data.polygons)]
    points = [obj for obj in scene.objects if obj.type == "MESH" and not len(obj.data.polygons)]
    cameras = [obj for obj in scene.objects if obj.type == "CAMERA"]
    if len(surfaces) != 1 or len(points) != 1 or len(cameras) != receipt["source_cameras"] + 1:
        raise ValueError("Saved objects did not survive reopening")
    surface = surfaces[0]
    if (len(surface.data.vertices) != receipt["mesh_vertices"] or len(surface.data.polygons) != receipt["mesh_triangles"]
            or len(points[0].data.vertices) != receipt["sparse_points"] or surface.get("owner_accepted") is not False):
        raise ValueError("Geometry counts or acceptance metadata changed")
    packed = [image for image in bpy.data.images if image.packed_file]
    if len(packed) != receipt["source_cameras"]:
        raise ValueError("Source photographs are not fully packed")
    positions = np.empty(len(surface.data.vertices) * 3, dtype=np.float32)
    surface.data.vertices.foreach_get("co", positions)
    if not np.isfinite(positions).all() or scene["registration"] != "unregistered":
        raise ValueError("Invalid geometry or invented registration")
    result = {"status": "passed", "blender_version": bpy.app.version_string, "source_cameras": receipt["source_cameras"],
              "packed_photographs": len(packed), "mesh_vertices": receipt["mesh_vertices"], "mesh_triangles": receipt["mesh_triangles"],
              "source_scene_sha256": receipt["files"]["condo-capture-review.blend"], "reopened_without_changes": True,
              "architectural_accuracy_or_acceptance": False}
    args.proof.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
