"""Build a new Blender scene from inferred geometry and independently check GLB cameras."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if subprocess.run([str(ROOT / "tools/splatlab-compute-gate.sh"), "--is-contained"], capture_output=True).returncode:
        raise RuntimeError("Use the shared SplatLab compute gate")
    output = args.output.resolve()
    if output.exists() or any(output.is_relative_to(path.resolve()) for path in (args.reference, args.surface, args.evaluation)):
        raise ValueError("Choose a new independent Blender output")
    sources = {}
    for directory in (args.reference, args.surface, args.evaluation):
        receipt = json.loads((directory / "receipt.json").read_text())
        sources[directory / "receipt.json"] = checksum(directory / "receipt.json")
        for name, digest in receipt["files"].items():
            if checksum(directory / name) != digest:
                raise ValueError("Source artifact changed")
            sources[directory / name] = digest
    output.mkdir(parents=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = "NONE"
    scene.render.resolution_x = scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene["scope"] = "Independent photo-derived review; NOT the accepted condo model, metric survey or collision mesh"
    scene["registration"] = "unregistered"
    scene["world_up"] = "unverified; fixed glTF-to-Blender basis conversion only"
    scene["source_to_blender"] = "x,y,z -> x,-z,y; no inferred metric scaling or architectural alignment"
    basis = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    bpy.ops.import_scene.gltf(filepath=str(args.reference / "reference.glb"))
    source_cameras = json.loads((args.reference / "camera-set.json").read_text())
    frames = source_cameras["frames"]
    point_count = json.loads((args.reference / "receipt.json").read_text())["source_points"]
    imported = [obj for obj in scene.objects if obj.type == "CAMERA"]
    if len(imported) != len(frames):
        raise ValueError("Blender did not import all source cameras")
    by_identity = {(frame["source_image"], frame["virtual_yaw_deg"]): frame for frame in frames}
    point_objects = [obj for obj in scene.objects if obj.type == "MESH"]
    if len(point_objects) != 1 or len(point_objects[0].data.vertices) != point_count or len(point_objects[0].data.polygons) != 0:
        raise ValueError("Unexpected sparse point import; do not confuse points with faces")
    projection_checks = 0
    for camera in imported:
        frame = by_identity[(camera["source_image"], camera["virtual_yaw_deg"])]
        expected = basis @ Matrix(frame["transform_matrix"])
        if not np.allclose(np.array(camera.matrix_world), np.array(expected), atol=1e-5):
            raise ValueError("Blender camera basis differs from the frozen source")
        scene.camera = camera
        for vertex in list(point_objects[0].data.vertices)[::113]:
            world = point_objects[0].matrix_world @ vertex.co
            local = camera.matrix_world.inverted() @ world
            if local.z >= 0:
                continue
            projected = world_to_camera_view(scene, camera, world)
            expected_pixel = (source_cameras["fl_x"] * local.x / -local.z + source_cameras["cx"],
                              -source_cameras["fl_y"] * local.y / -local.z + source_cameras["cy"])
            actual_pixel = (projected.x * 768, (1 - projected.y) * 768)
            if not (0 <= expected_pixel[0] < 768 and 0 <= expected_pixel[1] < 768):
                continue
            if not np.allclose(expected_pixel, actual_pixel, atol=.01):
                raise ValueError(f"Blender projection differs: expected {expected_pixel}, got {actual_pixel}; {frame['file_path']}")
            projection_checks += 1
        background = camera.data.background_images.new()
        background.image = bpy.data.images.load(str(args.reference / frame["file_path"]), check_existing=True)
        camera.data.show_background_images = True
    if projection_checks < 100:
        raise ValueError("Insufficient on-image Blender projection checks")
    with np.load(args.surface / "surface-arrays.npz", allow_pickle=False) as arrays:
        positions, faces, colors = arrays["vertices"], arrays["faces"], arrays["colors"]
    positions = positions @ np.array(basis)[:3, :3].T
    mesh = bpy.data.meshes.new("Inferred training-depth surface — not survey")
    mesh.from_pydata(positions.tolist(), [], faces.tolist())
    mesh.update()
    attribute = mesh.color_attributes.new(name="CapturedColor", type="FLOAT_COLOR", domain="POINT")
    linear = np.where(colors <= .04045, colors / 12.92, ((colors + .055) / 1.055) ** 2.4)
    attribute.data.foreach_set("color", np.column_stack((linear, np.ones(len(linear)))).ravel())
    surface = bpy.data.objects.new("INFERRED surface — review only", mesh)
    scene.collection.objects.link(surface)
    surface_receipt = json.loads((args.surface / "receipt.json").read_text())
    surface["provenance"] = f"AI-inferred expected-depth TSDF from {len(surface_receipt['training_views'])} training cameras; no held-out fusion"
    surface["units"] = "arbitrary"
    surface["owner_accepted"] = False
    material = bpy.data.materials.new("Captured photo vertex color")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    color_node = nodes.new("ShaderNodeVertexColor")
    color_node.layer_name = attribute.name
    emission = nodes.new("ShaderNodeEmission")
    destination = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(color_node.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs[0], destination.inputs["Surface"])
    mesh.materials.append(material)
    evaluation = json.loads((args.evaluation / "receipt.json").read_text())
    view = evaluation["validation"][1]
    camera_data = bpy.data.cameras.new("Held-out review camera")
    camera_data.type = "PERSP"
    camera_data.lens_unit = "FOV"
    camera_data.angle = float(2 * np.arctan(view["width"] / (2 * view["fx"])))
    camera_data.clip_start, camera_data.clip_end = .005, 1000
    camera = bpy.data.objects.new("Held-out review camera", camera_data)
    scene.collection.objects.link(camera)
    pose = Matrix([*view["camera_to_world_opengl"], [0, 0, 0, 1]])
    camera.matrix_world = basis @ pose
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    for layer in scene.view_layers:
        layer.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.world.color = (.02, .02, .02)
    scene.render.filepath = str(output / "mesh-preview.png")
    bpy.ops.render.render(write_still=True)
    for obj in scene.objects:
        obj.select_set(obj == surface)
    bpy.context.view_layer.objects.active = surface
    bpy.ops.export_scene.gltf(filepath=str(output / "inferred-surface.glb"), export_format="GLB", use_selection=True)
    for image in bpy.data.images:
        if image.source == "FILE" and image.filepath:
            image.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "condo-capture-review.blend"))
    if any(checksum(path) != digest for path, digest in sources.items()):
        raise ValueError("Source changed during Blender review")
    receipt = {"schema": "dev.splatlab.blender-capture-review/v1", "status": "created-needs-owner-review",
               "blender_version": bpy.app.version_string, "source_cameras": len(imported), "sparse_points": point_count,
               "projection_checks": projection_checks, "mesh_vertices": len(positions), "mesh_triangles": len(faces),
               "registration": None, "units": "arbitrary", "accepted_condo_model_opened": False,
               "source_to_blender": [list(row) for row in basis], "source_hashes": {str(path): digest for path, digest in sources.items()},
               "files": {path.name: checksum(path) for path in output.iterdir() if path.is_file()}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "source_cameras", "sparse_points", "projection_checks", "mesh_vertices", "mesh_triangles")}), flush=True)


if __name__ == "__main__":
    main()
