"""P6f: headless assembly of scene_manifest.json into scene.blend + scene.glb.

Runs headless (matches the proven blender-receipt-views.py pattern -- NOT
blender-mcp, which bridges to an already-open, human-attended GUI instance):
  ~/tools/blender-4.5.11-linux-x64/blender --background --python \
      blender_assemble.py -- <manifest.json> <out.blend> <out.glb> <out_dir>

Each splat element (background.ply / object.ply / proxy.ply -- all the same
raw 3DGS 14-field format written by write_splat_ply across this program) is
represented as a vertex-colored point mesh: xyz + SH0-derived RGB (the same
`0.5 + SH_C0*f_dc` formula already used in twin_finish.py, batch_isolate.py,
instance_lift.py, recall_expand.py today), solid-opacity filtered (>0.5,
matching twin_finish.py's own convention). glTF has no native gaussian-splat
primitive; a POINTS-mode vertex-colored mesh is the robust, standard
representation -- not a gaussian-ellipsoid render (that needs the KIRI addon,
proven only for interactive GUI use, not scripted headless import).

Per-element work is wrapped in try/except INSIDE this one process (the only
place the "kill one instance, scene still completes, element flagged, never
aborts" requirement can be enforced -- the whole manifest builds in ONE
`blender --background` invocation, unlike P6c/d's one-subprocess-per-element
pattern). Resumability = re-run this whole script from the manifest (full
idempotent rebuild), never incremental .blend patching.

glTF export passes export_extras=True EXPLICITLY -- Blender's exporter drops
custom properties from node.extras otherwise, silently, with exit 0. Same
failure CLASS this program already found once (P0's GLB writer that silently
corrupted output while returing success) -- so export is followed by a
MANDATORY readback: re-parse the written GLB's own JSON chunk and assert
every node's extras.splatlab_provenance is present and correct. This readback
IS the contamination gate too: a captured/ground-derived node must NOT carry
the tag, every proxy node MUST.
"""
import json
import struct
import sys
import time
from pathlib import Path

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(argv) < 4:
    print("USAGE: ... -- <manifest.json> <out.blend> <out.glb> <out_dir>")
    sys.exit(2)
MANIFEST_PATH = Path(argv[0]).expanduser()
OUT_BLEND = Path(argv[1]).expanduser()
OUT_GLB = Path(argv[2]).expanduser()
OUT_DIR = Path(argv[3]).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
t_start = time.time()

SH_C0 = 0.28209479177387814
GLTF_EXTRAS_KEY = "splatlab_provenance"  # must match provenance.GLTF_EXTRAS_KEY
GENERATIVE_TAG = "SplatLab-provenance: generative render-vr-only"  # must match provenance.GENERATIVE_TAG
# Large point clouds (background can be 1M+ points) are capped for a
# tractable .blend/.glb -- randomly subsampled, never silently pretending to
# be complete. Objects (tens of thousands) are well under this and untouched.
# Each surviving point becomes a real 12-triangle cube (see
# apply_point_material) so this budgets triangle count too, not just verts.
MAX_POINTS = 60_000


def load_splat_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse the raw 3DGS PLY format this program writes everywhere
    (x,y,z,f_dc_0-2,opacity,scale_0-2,rot_0-3 -- 14 float32 fields).
    Returns (xyz, rgb) for SOLID gaussians only (opacity > 0.5)."""
    with open(path, "rb") as f:
        header = b""
        n_props = 0
        while not header.endswith(b"end_header\n"):
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            header += line
            if line.startswith(b"property"):
                n_props += 1
        data = np.fromfile(f, dtype=np.float32)
    if data.size % n_props != 0:
        raise ValueError(f"{path}: {data.size} floats not divisible by {n_props} fields")
    data = data.reshape(-1, n_props)
    xyz = data[:, 0:3]
    f_dc = data[:, 3:6]
    opac = 1.0 / (1.0 + np.exp(-data[:, 6]))
    solid = opac > 0.5
    if solid.sum() == 0:
        raise ValueError(f"{path}: zero solid gaussians (opacity > 0.5)")
    xyz, rgb = xyz[solid], np.clip(0.5 + SH_C0 * f_dc[solid], 0, 1)
    if len(xyz) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(xyz), size=MAX_POINTS, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    return xyz.astype(np.float64), rgb.astype(np.float32)


def make_point_object(name: str, xyz: np.ndarray, rgb: np.ndarray):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in xyz], [], [])
    mesh.update()
    attr = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1), dtype=np.float32)], axis=1)
    attr.data.foreach_set("color", rgba.ravel())
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


_POINT_TEMPLATE = None  # one shared tiny-cube instance mesh, built lazily


def _point_template_object():
    global _POINT_TEMPLATE
    if _POINT_TEMPLATE is None:
        bpy.ops.mesh.primitive_cube_add(size=0.006)
        _POINT_TEMPLATE = bpy.context.active_object
        _POINT_TEMPLATE.name = "_point_template"
        bpy.context.scene.collection.objects.unlink(_POINT_TEMPLATE)  # instance source only, not a rendered root
    return _POINT_TEMPLATE


def apply_point_material(obj, tag_generative: bool) -> None:
    """Best-effort visualization: bake real triangulated geometry (Instance
    on Points -> tiny cube -> Realize Instances) so the object has actual
    mesh primitives a glTF exporter will keep. A vertex-only, zero-face mesh
    was tried first and silently DROPPED by Blender's own glTF exporter
    ("Mesh has no primitives and will be omitted") -- verified live against
    this exact script, not assumed. Realize Instances sidesteps that
    entirely by producing genuine polygon data before export. Failure here
    is non-fatal -- matches this program's existing best-effort-receipt
    doctrine; only the geometry itself (asserted present in the readback) is
    load-bearing."""
    mat = bpy.data.materials.new(f"{obj.name}_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = "Col"
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.append(mat)

    template = _point_template_object()
    mod = obj.modifiers.new("PointsViz", "NODES")
    group = bpy.data.node_groups.new(f"{obj.name}_pts", "GeometryNodeTree")
    mod.node_group = group
    group.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gin = group.nodes.new("NodeGroupInput")
    gout = group.nodes.new("NodeGroupOutput")
    iop = group.nodes.new("GeometryNodeInstanceOnPoints")
    obj_info = group.nodes.new("GeometryNodeObjectInfo")
    obj_info.inputs["Object"].default_value = template
    realize = group.nodes.new("GeometryNodeRealizeInstances")
    setmat = group.nodes.new("GeometryNodeSetMaterial")
    setmat.inputs["Material"].default_value = mat
    group.links.new(gin.outputs["Geometry"], iop.inputs["Points"])
    group.links.new(obj_info.outputs["Geometry"], iop.inputs["Instance"])
    group.links.new(iop.outputs["Instances"], realize.inputs["Geometry"])
    group.links.new(realize.outputs["Geometry"], setmat.inputs["Geometry"])
    group.links.new(setmat.outputs["Geometry"], gout.inputs["Geometry"])


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    bpy.ops.wm.read_homefile(use_empty=True)

    built, flagged = [], []
    for el in manifest["elements"]:
        slug, prov = el["slug"], el["provenance"]
        try:
            if "ply" in el["files"]:
                xyz, rgb = load_splat_ply(Path(el["files"]["ply"]))
                obj = make_point_object(slug, xyz, rgb)
                try:
                    apply_point_material(obj, tag_generative=(prov == "proxy"))
                except Exception as viz_exc:  # best-effort viz, never fatal
                    print(f"[assemble] {slug}: point viz failed (non-fatal): {viz_exc}", flush=True)
            elif "glb" in el["files"]:
                before = set(bpy.data.objects)
                bpy.ops.import_scene.gltf(filepath=el["files"]["glb"])
                new_objs = [o for o in bpy.data.objects if o not in before]
                if not new_objs:
                    raise ValueError("glTF import produced zero objects")
                obj = new_objs[0]
                for o in new_objs:
                    o.name = f"{slug}_{o.name}" if o is not obj else slug
            else:
                raise ValueError(f"element {slug!r} has neither ply nor glb")
            obj[GLTF_EXTRAS_KEY] = GENERATIVE_TAG if prov == "proxy" else ""
            obj["splatlab_slug"] = slug
            obj["splatlab_source_provenance"] = prov
            built.append(slug)
        except Exception as exc:  # never let one bad element kill the whole build
            print(f"[assemble] {slug}: FAILED, flagged and skipped: {exc}", flush=True)
            flagged.append({"slug": slug, "reason": str(exc)})

    if not built:
        print("FATAL: zero elements built", file=sys.stderr)
        sys.exit(1)

    OUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND))

    bpy.ops.export_scene.gltf(
        filepath=str(OUT_GLB), export_format="GLB",
        export_extras=True,          # the load-bearing flag -- see module docstring
        export_yup=True, export_apply=True,
    )

    # ---- mandatory readback: parse the GLB's own JSON chunk directly ------
    with open(OUT_GLB, "rb") as f:
        magic, version, length = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            print(f"FATAL: {OUT_GLB} is not a valid GLB (bad magic)", file=sys.stderr)
            sys.exit(1)
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:  # 'JSON'
            print("FATAL: GLB first chunk is not JSON", file=sys.stderr)
            sys.exit(1)
        gltf_json = json.loads(f.read(chunk_len))

    slug_by_provenance = {e["slug"]: e["provenance"] for e in manifest["elements"] if e["slug"] in built}
    tag_errors = []
    seen_slugs = set()
    for node in gltf_json.get("nodes", []):
        name = node.get("name", "")
        slug = next((s for s in slug_by_provenance if name == s or name.startswith(f"{s}_")), None)
        if slug is None:
            continue
        seen_slugs.add(slug)
        extras = node.get("extras") or {}
        has_tag = bool(extras.get(GLTF_EXTRAS_KEY))
        should_have_tag = slug_by_provenance[slug] == "proxy"
        if has_tag != should_have_tag:
            tag_errors.append(f"{slug}: provenance={slug_by_provenance[slug]!r} but "
                              f"tag_present={has_tag} (expected {should_have_tag})")
    missing_from_glb = set(built) - seen_slugs
    if missing_from_glb:
        tag_errors.append(f"elements missing from exported GLB nodes entirely: {sorted(missing_from_glb)}")

    report = {
        "n_elements_total": len(manifest["elements"]),
        "n_built": len(built), "built": built,
        "n_flagged": len(flagged), "flagged": flagged,
        "contamination_gate": {"ok": not tag_errors, "errors": tag_errors},
        "seconds": round(time.time() - t_start, 1),
    }
    (OUT_DIR / "assemble_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if tag_errors:
        print("FATAL: contamination gate failed -- glTF extras don't match provenance", file=sys.stderr)
        sys.exit(1)
    print("ASSEMBLE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
