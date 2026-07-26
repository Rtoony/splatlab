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


def run_object_texture(py: Path, script: Path, mesh: Path, splat: Path, out_glb: Path,
                       *, mpu, faces: int, tex: int, reconstruct: bool,
                       crop: bool, smooth: bool = True, timeout: int = 1800,
                       min_gaussians: int | None = None) -> dict:
    cmd = [str(py), str(script), str(mesh), str(splat), str(out_glb),
           "--target-faces", str(faces), "--texture-size", str(tex)]
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
        _log("  shell ...")
        try:
            shell_ply = world / "_work" / "shell.ply"
            cut = build_shell_mesh(job_dir, instance_boxes(instances), shell_ply)
            # background.ply is the exact complement of every claimed instance —
            # a cleaner colour source for the shell than the full splat, which
            # still contains the props we just cut out of the geometry.
            splat = job_dir / "_scene" / "isolated" / "background.ply"
            if not splat.is_file():
                splat = job_dir / "_preview" / "splat.ply"
            if not splat.is_file():
                raise FileNotFoundError("no shell colour source (background.ply or splat.ply)")
            out = world / "shell.glb"
            res = run_object_texture(py, script, shell_ply, splat, out, mpu=mpu,
                                     faces=args.shell_faces, tex=args.shell_texture_size,
                                     reconstruct=False, crop=False, timeout=3600)
            shell = {"role": "shell", "cut": cut, "built": res["ok"],
                     "seconds": res["seconds"], "glb": out.name if res["ok"] else None}
            if not res["ok"]:
                shell["reason"] = res.get("error")
                _log(f"  FAIL shell: {str(res.get('error'))[:200]}")
            else:
                r = res.get("report") or {}
                shell["faces"] = r.get("faces")
                shell["extent"] = r.get("extent")
                shell["texture"] = (r.get("texture") or {}).get("baked")
        except Exception as exc:  # noqa: BLE001 — a failed shell must not lose the props
            shell = {"role": "shell", "built": False,
                     "reason": f"{type(exc).__name__}: {exc}"[:300]}
            _log(f"  FAIL shell: {shell['reason']}")

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
