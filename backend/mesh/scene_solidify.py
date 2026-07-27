#!/usr/bin/env python3
"""Whole-capture solidify: every scene element becomes a textured, simplified mesh.

object_texture.py turns ONE isolated object into a simplified UV-textured GLB.
This drives it across an entire capture, from _scene/inventory.json, and adds the
piece an object lane cannot produce on its own: the SHELL — the walls, floor and
fixed surroundings with every named instance cut out of it. Shell + elements is
what a navigable world is made of.

Two deliberately different recipes, because the two halves have opposite needs:

  PROPS  (named instances)  Poisson refit ON, low face budget.
         They get picked up, rotated and collided against, so they must be
         closed solids and cheap.

  SHELL  (everything else)  Poisson refit OFF, high face budget.
         Screened Poisson always closes a surface. On a room that means it
         invents a lid over the open side the cameras never crossed and welds
         the walls into a blob. The raw TSDF surface is already the right
         topology for an environment, so the shell is cleaned and decimated but
         never refitted. The high budget is deliberate too: Unreal renders
         static geometry through Nanite, which does its own LOD — spending the
         simplification effort on props instead of walls is the correct
         inversion for a UE target.

Per element, object_texture.py runs as its own SUBPROCESS. One malformed
instance then fails alone with an honest exit code instead of aborting the
capture, and every element keeps its own report on disk.

Usage: scene_solidify.py <job_dir>
       [--meters-per-unit MPU] [--prop-faces 8000] [--shell-faces 60000]
       [--texture-size 1024] [--only slug,slug] [--skip-shell]
       [--python PATH] [--object-texture PATH]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

DEFAULT_PY = Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_inventory(job_dir: Path) -> list[dict]:
    """Instances from the scene lane, newest source first.

    _scene/inventory.json is the consolidated P6b product (SAM3 masks -> lift ->
    majority vote -> noun consolidation). _scene/instances.json is the
    pre-consolidation form and is accepted as a fallback so this also runs on
    captures built before consolidation existed.
    """
    for name in ("inventory.json", "instances.json"):
        p = job_dir / "_scene" / name
        if p.is_file():
            try:
                doc = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            inst = doc.get("instances") or []
            if inst:
                return inst
    return []


def instance_boxes(instances: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tight semantic boxes, in scene units — what to cut OUT of the shell."""
    boxes = []
    for inst in instances:
        b = inst.get("bbox_tight_scene") or inst.get("bbox_tight")
        if not b or "min" not in b or "max" not in b:
            continue
        boxes.append((np.asarray(b["min"], dtype=np.float64),
                      np.asarray(b["max"], dtype=np.float64)))
    return boxes


def build_shell_mesh(job_dir: Path, boxes, out_ply: Path, margin: float = 0.05) -> dict:
    """The scene mesh with every named instance carved out, plus the ground lane.

    Cutting by the instances' own semantic boxes is what leaves walls/floor/fixed
    surroundings behind. The ground mesh is unioned in because the TSDF floor is
    routinely the worst-reconstructed surface in a capture (it is seen at
    grazing angles) while the ground lane rebuilds it properly from the
    gaussians that were classified as ground.
    """
    scene_ply = job_dir / "_mesh" / "mesh.ply"
    if not scene_ply.is_file():
        raise FileNotFoundError(f"no scene mesh at {scene_ply}")
    mesh = trimesh.load(str(scene_ply), force="mesh", process=False)
    faces_in = int(len(mesh.faces))

    v = np.asarray(mesh.vertices)
    drop = np.zeros(len(mesh.faces), dtype=bool)
    for lo, hi in boxes:
        pad = (hi - lo) * margin
        inside = np.all((v >= lo - pad) & (v <= hi + pad), axis=1)
        # a face belongs to an instance if ANY corner is inside its box —
        # keeping partial faces would leave shards of the prop welded to the wall
        drop |= inside[mesh.faces].any(axis=1)
    keep = ~drop
    if keep.sum() < 4:
        raise ValueError(f"instance boxes consumed the whole scene ({keep.sum()} faces left)")
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()

    stats = {"scene_faces_in": faces_in, "faces_after_instance_cut": int(len(mesh.faces)),
             "instances_cut": len(boxes), "ground_merged": False}

    ground = job_dir / "_scene" / "ground" / "ground_mesh_raw.ply"
    if ground.is_file():
        try:
            g = trimesh.load(str(ground), force="mesh", process=False)
            if len(g.faces):
                mesh = trimesh.util.concatenate([mesh, g])
                stats["ground_merged"] = True
                stats["ground_faces"] = int(len(g.faces))
        except Exception as exc:  # noqa: BLE001 — ground is an enhancement, not a gate
            stats["ground_error"] = f"{type(exc).__name__}: {exc}"[:160]

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_ply))
    stats["faces_out"] = int(len(mesh.faces))
    return stats


def _generated_candidate(job_dir: Path, slug: str) -> dict | None:
    """A generated asset for this element, only if it knows where it belongs.

    object_generate.py withholds `transform_4x4_generated_to_capture` whenever
    its silhouette-IoU orientation search falls below threshold, so a missing
    transform means "we could not place this", not "we forgot". Refused objects
    (the mask gate) have no mesh at all and land here as None.
    """
    base = job_dir / "_regen" / "objects" / slug
    mesh, report = base / "generated_mesh.glb", base / "generate_report.json"
    if not (mesh.is_file() and report.is_file()):
        return None
    try:
        doc = json.loads(report.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    placement = (doc.get("capture_frame_placement") or {}).get("mesh_glb") or {}
    matrix = placement.get("transform_4x4_generated_to_capture")
    if not placement.get("placement_resolved") or not matrix:
        return None
    return {
        "mesh": mesh,
        "matrix": matrix,
        "iou": float((placement.get("orientation_search") or {}).get("best_silhouette_iou") or 0.0),
        "report": {
            "source": str(mesh),
            "model": doc.get("model") or "sam-3d-objects",
            "seed": doc.get("seed"),
            "mask_gate_iou": (doc.get("mask_alignment_gate") or {}).get("iou_vs_captured_object"),
            "placement_silhouette_iou": (placement.get("orientation_search") or {}).get("best_silhouette_iou"),
            "fitted_uniform_scale": placement.get("fitted_uniform_scale"),
            "provenance_tag": doc.get("provenance_tag"),
            # The placement is FITTED, not recovered from a known convention.
            # Recorded so a consumer never mistakes it for a measured pose.
            "placement_is_fitted": True,
        },
    }


def place_generated(gen: dict, out_glb: Path, *, mpu, faces: int, tex: int) -> dict:
    """Transform a generated mesh into the capture frame and write it as an element.

    Deliberately does NOT re-run object_texture: the generated asset is already a
    clean closed surface with its own vertex colours, and a second Poisson refit
    would only degrade the thing that made it worth choosing.

    It DOES decimate. SAM 3D emits 0.5-1.2M triangles per object — a cardboard
    box arrived at 1,186,232 — which is unusable in a browser walking a room.
    Quadric collapse with vertex colours carried through keeps the appearance
    while meeting the same face budget as every other prop; that is a different
    operation from refitting the surface.
    """
    import numpy as np
    import open3d as o3d
    import trimesh

    t0 = time.time()
    try:
        scene = trimesh.load(str(gen["mesh"]))
        mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
        mesh.apply_transform(np.asarray(gen["matrix"], dtype=np.float64))

        faces_in = int(len(mesh.faces))
        if faces and faces_in > faces:
            om = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
                o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))
            vc = getattr(mesh.visual, "vertex_colors", None)
            if vc is not None and len(vc) == len(mesh.vertices):
                om.vertex_colors = o3d.utility.Vector3dVector(
                    np.asarray(vc, dtype=np.float64)[:, :3] / 255.0)
            om = om.simplify_quadric_decimation(target_number_of_triangles=int(faces))
            om.remove_unreferenced_vertices()
            mesh = trimesh.Trimesh(
                vertices=np.asarray(om.vertices), faces=np.asarray(om.triangles),
                vertex_colors=(np.clip(np.asarray(om.vertex_colors), 0, 1) * 255).astype(np.uint8)
                if om.has_vertex_colors() else None,
                process=False)
        if mpu:
            # Elements are exported in metres, Y-up — the same convention
            # object_texture.py uses — so the capture-frame result is scaled and
            # rotated to match its siblings.
            v = np.asarray(mesh.vertices) * float(mpu)
            mesh.vertices = np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_glb))
        back = trimesh.load(str(out_glb), force="mesh")
        if len(back.faces) != len(mesh.faces):
            out_glb.unlink(missing_ok=True)
            return {"ok": False, "error": f"readback mismatch {len(back.faces)} != {len(mesh.faces)}"}
        return {"ok": True, "seconds": round(time.time() - t0, 1),
                "faces": int(len(back.faces)),
                "extent": [round(float(x), 3) for x in back.extents]}
    except Exception as exc:  # noqa: BLE001 — fall back to captured geometry
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200],
                "seconds": round(time.time() - t0, 1)}


def run_object_texture(py: Path, script: Path, mesh: Path, splat: Path, out_glb: Path,
                       *, mpu, faces: int, tex: int, reconstruct: bool,
                       crop: bool, smooth: bool = True, timeout: int = 1800,
                       min_gaussians: int | None = None,
                       unwrap_chunks: int = 1,
                       class_map: Path | None = None,
                       class_radius: float | None = None) -> dict:
    cmd = [str(py), str(script), str(mesh), str(splat), str(out_glb),
           "--target-faces", str(faces), "--texture-size", str(tex)]
    if unwrap_chunks > 1:
        cmd += ["--unwrap-chunks", str(unwrap_chunks)]
    if class_map is not None:
        taxonomy = Path(__file__).resolve().parents[1] / "class_taxonomy.json"
        cmd += ["--class-map", str(class_map), "--class-taxonomy", str(taxonomy)]
        if class_radius:
            cmd += ["--class-radius", str(round(float(class_radius), 4))]
    if mpu:
        cmd += ["--meters-per-unit", str(mpu)]
    if smooth:
        cmd += ["--smooth"]
    if not reconstruct:
        cmd += ["--no-reconstruct"]
    if not crop:
        cmd += ["--no-crop"]
    if min_gaussians is not None:
        cmd += ["--min-gaussians", str(min_gaussians)]
    t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out_glb.is_file():
        tail = "\n".join((proc.stderr or "").splitlines()[-4:])
        return {"ok": False, "exit": proc.returncode, "error": tail[:400],
                "seconds": round(time.time() - t, 1)}
    report_path = out_glb.parent / "object_texture.json"
    report = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
            report_path.replace(out_glb.with_suffix(".json"))
        except (OSError, json.JSONDecodeError):
            pass
    return {"ok": True, "seconds": round(time.time() - t, 1), "report": report}


def _shell_colour_source(job_dir: Path) -> Path:
    """background.ply is the exact complement of every claimed instance — a
    cleaner colour source for the shell than the full splat, which still
    contains the props cut out of the geometry."""
    splat = job_dir / "_scene" / "isolated" / "background.ply"
    if not splat.is_file():
        splat = job_dir / "_preview" / "splat.ply"
    if not splat.is_file():
        raise FileNotFoundError("no shell colour source (background.ply or splat.ply)")
    return splat


def _materialize_class_map(job_dir: Path, conf_floor: float = 0.6) -> Path | None:
    """xyz-keyed class map for the shell bake, from semantic_ground's
    class-aware npz (user paints are one-hot 1.0 there, so paint precedence
    rides along for free). None when the ground lane hasn't produced classes."""
    gg = job_dir / "_scene" / "ground" / "ground_gaussians.npz"
    if not gg.is_file():
        return None
    d = np.load(gg)
    if "class_rel" not in d.files:
        return None
    class_rel = d["class_rel"].astype(np.float32)
    conf = class_rel.max(axis=1)
    keep = conf >= conf_floor
    if not bool(keep.any()):
        return None
    out = job_dir / "_world" / "_work" / "class_map.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        xyz=d["xyz"][keep].astype(np.float32),
        class_idx=class_rel[keep].argmax(axis=1).astype(np.int16),
        class_ids=d["class_ids"],
    )
    _log(f"  class map: {int(keep.sum())} classed gaussians -> {out.name}")
    return out


def voxel_to_capture_frame(src: Path, dst: Path) -> Path:
    """Undo world_shell.py's Y-up conversion so object_texture.py sees what it expects.

    world_shell.py writes collision_shell.glb in the Y-up three.js frame
    (documented at its module head). object_texture.py assumes CAPTURE-frame
    input: it samples colour from the capture-frame splat, and applies the
    capture->Y-up rotation itself on export (object_texture.py:863-866, applied
    unconditionally — only the SCALE depends on meters_per_unit).

    Handing it a Y-up mesh therefore did two wrong things at once: rotated the
    shell a second time, so the visual shell sat 90 degrees off the props and
    the collision solid it is supposed to coincide with; and sampled every
    texel's colour from the wrong neighbourhood, which is why voxel shells came
    out blotchy. object_texture.py:460 warns about exactly this hazard.

    The TSDF path was always correct — it feeds _work/shell.ply, which is
    capture-frame — so this only ever affected --shell-source voxel.

    Forward is (x, y, z) -> (x, z, -y); the inverse is (X, Y, Z) -> (X, -Z, Y).
    """
    mesh = trimesh.load(str(src), force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertices = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(dst))
    return dst


def build_shell(job_dir: Path, args, instances, py: Path, script: Path, mpu) -> dict:
    """The visual shell, from one of two geometry sources.

    tsdf  — the original path: TSDF surface with instance boxes cut out.
            Accurate to the capture but lacy (bonsai: 2,088 components at any
            budget) — fails its own connectivity/floor gates.
    voxel — bake colour onto world_shell.py's watertight voxel solid instead.
            Blockier detail, but ONE component with floor continuity 1.0, so
            the shell passes its gates, survives Blender editing, and the
            visual surface coincides exactly with what the walker collides
            against (no fall-throughs where the eye sees ground).
    """
    world = job_dir / "_world"
    _log(f"  shell ({args.shell_source}) ...")
    shell: dict = {"role": "shell", "source": args.shell_source}
    try:
        if args.shell_source == "voxel":
            geom = world / "collision_shell.glb"
            if not geom.is_file():
                _log("  no collision_shell.glb — running world_shell.py first")
                proc = subprocess.run(
                    [str(py), str(Path(__file__).with_name("world_shell.py")),
                     str(job_dir)],
                    capture_output=True, text=True, timeout=1800)
                if proc.returncode != 0 or not geom.is_file():
                    tail = "\n".join((proc.stderr or "").splitlines()[-4:])
                    raise RuntimeError(f"world_shell.py failed: {tail[:300]}")
            # collision_shell.glb is Y-up; object_texture.py expects the capture
            # frame both for colour sampling and for its own Y-up export.
            geom = voxel_to_capture_frame(geom, world / "_work" / "shell_voxel_capture.ply")
            cut = None
        else:
            geom = world / "_work" / "shell.ply"
            cut = build_shell_mesh(job_dir, instance_boxes(instances), geom)
        splat = _shell_colour_source(job_dir)
        out = world / "shell.glb"
        class_map = _materialize_class_map(job_dir)
        if class_map is not None:
            shell["classed"] = True
        # Chunked unwrap is what makes dense-shell budgets tractable
        # (13x measured at 400k faces); the layout change is confined to
        # the shell, whose consumers treat GLB+atlas as a regenerated pair.
        # --class-radius was unreachable from here, so the shell always used
        # object_texture's 0.15 default. It is exposed now, but note the
        # measurement that came with it: on splat_3aaf8067 the median distance
        # from shell surface to the nearest classed gaussian is 3.44 scene
        # units, so NO radius rescues class coverage. See the STATUS entry:
        # the shell encloses ~3x the volume of the ground. Knob, not the fix.
        res = run_object_texture(py, script, geom, splat, out, mpu=mpu,
                                 faces=args.shell_faces, tex=args.shell_texture_size,
                                 reconstruct=False, crop=False, timeout=3600,
                                 unwrap_chunks=4, class_map=class_map,
                                 class_radius=getattr(args, "class_radius", None))
        shell.update(cut=cut, built=res["ok"], seconds=res["seconds"],
                     glb=out.name if res["ok"] else None)
        if not res["ok"]:
            shell["reason"] = res.get("error")
            _log(f"  FAIL shell: {str(res.get('error'))[:200]}")
        else:
            r = res.get("report") or {}
            shell["faces"] = r.get("faces")
            shell["extent"] = r.get("extent")
            shell["texture"] = (r.get("texture") or {}).get("baked")
    except Exception as exc:  # noqa: BLE001 — a failed shell must not lose the props
        shell.update(built=False, reason=f"{type(exc).__name__}: {exc}"[:300])
        _log(f"  FAIL shell: {shell['reason']}")
    return shell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    ap.add_argument("--prop-faces", type=int, default=8_000)
    ap.add_argument("--shell-faces", type=int, default=60_000)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--shell-texture-size", type=int, default=2048)
    ap.add_argument("--only", default=None, help="comma-separated slugs")
    ap.add_argument("--skip-shell", action="store_true")
    ap.add_argument("--class-radius", type=float, default=None,
                    help="scene units a shell texel may reach for a class; "
                         "default scales with the shell's voxel size")
    ap.add_argument("--shell-source", choices=["tsdf", "voxel"], default="tsdf",
                    help="voxel = bake colour onto world_shell.py's watertight "
                         "solid (gate-passing); tsdf = the original cut surface")
    ap.add_argument("--shell-only", action="store_true",
                    help="rebuild ONLY the shell and patch it into the existing "
                         "world.json — elements are untouched")
    ap.add_argument("--prefer-generated", action="store_true",
                    help="use a placed generative reconstruction in place of the captured "
                         "geometry when one exists (render lane only; survey still refuses it)")
    ap.add_argument("--min-gaussians", type=int, default=200,
                    help="floor for reconstructing a prop from its gaussians")
    ap.add_argument("--python", default=str(DEFAULT_PY))
    ap.add_argument("--object-texture", default=str(Path(__file__).with_name("object_texture.py")))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    t0 = time.time()
    job_dir = Path(args.job_dir).expanduser()
    if not job_dir.is_dir():
        print(f"FATAL: no such job dir {job_dir}", file=sys.stderr)
        return 1
    py, script = Path(args.python), Path(args.object_texture)
    if not py.is_file() or not script.is_file():
        print(f"FATAL: toolchain missing ({py}, {script})", file=sys.stderr)
        return 1

    mpu = args.meters_per_unit
    if mpu is None:
        try:
            mpu = json.loads((job_dir / "meta.json").read_text()).get("meters_per_unit")
        except (OSError, json.JSONDecodeError):
            mpu = None

    world = job_dir / "_world"
    (world / "elements").mkdir(parents=True, exist_ok=True)

    if args.shell_only:
        # Patch the shell into the EXISTING world.json — never clobber the
        # element records a full solidify wrote.
        world_json = world / "world.json"
        try:
            doc = json.loads(world_json.read_text())
        except (OSError, json.JSONDecodeError):
            print("FATAL: --shell-only needs an existing _world/world.json — "
                  "run the full solidify first", file=sys.stderr)
            return 1
        instances = load_inventory(job_dir) if args.shell_source == "tsdf" else []
        if args.shell_source == "tsdf" and not instances:
            print("FATAL: tsdf shell needs _scene/inventory.json for the "
                  "instance cut boxes", file=sys.stderr)
            return 1
        shell = build_shell(job_dir, args, instances, py, script, mpu)
        doc["shell"] = shell
        doc.setdefault("counts", {})["shell_built"] = bool(shell.get("built"))
        world_json.write_text(json.dumps(doc, indent=2))
        print(json.dumps({"shell": shell}, indent=2))
        return 0 if shell.get("built") else 1

    instances = load_inventory(job_dir)
    if not instances:
        print("FATAL: no _scene/inventory.json (or instances.json) — run the scene "
              "inventory first", file=sys.stderr)
        return 1
    only = {s.strip() for s in args.only.split(",")} if args.only else None

    elements = []
    for inst in instances:
        slug = inst.get("slug") or inst.get("label", "object").replace(" ", "-")
        if only and slug not in only:
            continue
        # Geometry source, best first:
        #   1. _objects/<slug>/mesh/mesh.ply  — a real per-object TSDF, if the
        #      single-object lane happened to build one.
        #   2. _scene/isolated/<slug>/object.ply — the scene lane's claimed
        #      gaussians. batch_isolate.py never meshes, so this is the normal
        #      case; object_texture.py Poisson-reconstructs it directly.
        splat_ply = None
        for cand in (job_dir / "_scene" / "isolated" / slug / "object.ply",
                     job_dir / "_objects" / slug / "object.ply"):
            if cand.is_file():
                splat_ply = cand
                break
        tsdf = job_dir / "_objects" / slug / "mesh" / "mesh.ply"
        geom = tsdf if tsdf.is_file() else splat_ply
        entry = {"slug": slug, "label": inst.get("label"), "role": "prop",
                 "n_members": inst.get("n_members"),
                 "geometry_source": "tsdf" if tsdf.is_file() else "gaussians"}
        if splat_ply is None:
            entry.update(built=False, reason="not isolated yet (no object.ply)")
            elements.append(entry)
            _log(f"  SKIP {slug}: {entry['reason']}")
            continue
        out = world / "elements" / f"{slug}.glb"

        # GENERATED geometry, when it exists and knows where it goes.
        # For a sparse capture a generative reconstruction can be plainly better
        # than anything the scan supports — operator-graded on the fire hydrant,
        # in form AND colour, against a 17.96% LCC source. It is only usable when
        # object_generate.py resolved a capture-frame placement; an unplaced
        # asset is a nice model floating in the wrong spot.
        #
        # This is a RENDER decision and stays one. The generative quarantine is
        # enforced for lane="survey" (geo_export.py, ground_extract.py) and is
        # untouched — measurement must never consume invented geometry. The
        # world manifest records the substitution so a viewer can label it.
        gen = _generated_candidate(job_dir, slug) if args.prefer_generated else None
        if gen:
            entry.update(geometry_source="generated", generated=gen["report"])
            res = place_generated(gen, out, mpu=mpu, faces=args.prop_faces,
                                  tex=args.texture_size)
            entry.update(built=res["ok"], seconds=res.get("seconds"),
                         glb=out.name if res["ok"] else None)
            if res["ok"]:
                entry["faces"] = res.get("faces")
                entry["extent"] = res.get("extent")
                entry["provenance"] = "generative render-only"
                _log(f"  prop {slug}: GENERATED (placement iou {gen['iou']:.3f})")
                elements.append(entry)
                continue
            # Fall through to the captured path rather than losing the element.
            _log(f"  prop {slug}: generated placement failed ({res.get('error')}); "
                 f"falling back to captured geometry")
            entry.pop("provenance", None)
            entry["geometry_source"] = "tsdf" if tsdf.is_file() else "gaussians"

        _log(f"  prop {slug} ...")
        res = run_object_texture(py, script, geom, splat_ply, out, mpu=mpu,
                                 faces=args.prop_faces, tex=args.texture_size,
                                 reconstruct=True, crop=True,
                                 min_gaussians=args.min_gaussians)
        entry.update(built=res["ok"], seconds=res["seconds"], glb=out.name if res["ok"] else None)
        if not res["ok"]:
            entry["reason"] = res.get("error")
            _log(f"  FAIL {slug}: {str(res.get('error'))[:160]}")
        else:
            r = res.get("report") or {}
            entry["faces"] = r.get("faces")
            entry["extent"] = r.get("extent")
            entry["texture"] = (r.get("texture") or {}).get("baked")
        elements.append(entry)

    shell = None
    if not args.skip_shell:
        shell = build_shell(job_dir, args, instances, py, script, mpu)

    built = [e for e in elements if e.get("built")]
    doc = {
        "v": 1,
        "job_id": job_dir.name,
        "units": "meters" if mpu else "scene-units (uncalibrated)",
        "meters_per_unit": mpu,
        "up_axis": "Y",
        "counts": {"instances": len(instances), "props_built": len(built),
                   "props_failed": len(elements) - len(built),
                   "shell_built": bool(shell and shell.get("built"))},
        "shell": shell,
        "elements": elements,
        "seconds": round(time.time() - t0, 1),
    }
    out_report = Path(args.report) if args.report else world / "world.json"
    out_report.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    # Props failing individually is survivable and recorded; nothing built at all is not.
    return 0 if (built or (shell and shell.get("built"))) else 1


if __name__ == "__main__":
    sys.exit(main())
