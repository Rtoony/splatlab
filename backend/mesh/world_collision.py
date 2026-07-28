#!/usr/bin/env python3
"""Classify solidified scene elements into static vs prop, and give props collision.

scene_solidify.py produces textured meshes. This decides what each one IS in a
navigable world, and emits the collision geometry that makes it solid.

Two collision strategies, because static environments and movable props want
opposite things:

  STATIC (shell, large fixed furniture)
      No hulls generated — the render triangles ARE the collision. In three.js
      that is a three-mesh-bvh capsule sweep against the mesh; in Unreal it is
      "Use Complex Collision As Simple". Either way it is exact, free, and
      immune to the convex-hull artifacts that would round off a doorway.
      CAVEAT that matters: this only holds for a SOLID surface. A lacy or
      fragmented shell lets a capsule fall straight through, which is why the
      walkable shell is voxel-solidified separately (see world_shell.py) rather
      than reusing the visual mesh.

  PROPS (small movable objects)
      Convex decomposition via CoACD. Physics needs CONVEX geometry — a concave
      mesh used as simple collision behaves as its own convex hull, so a
      bicycle would collide as a solid slab. Emitted as `UCX_<mesh>_NN`, which
      Unreal auto-imports as collision and which is a harmless, self-describing
      convention for a three.js/rapier consumer to glob.

Runtime target is three.js as of 2026-07-26; the naming above stays UE-friendly
because it costs nothing and keeps the Unreal bundle lane usable.

Classification is deliberately explainable rather than clever: an element is a
prop when it is small relative to the scene AND small in absolute terms (when
the capture is calibrated). Every decision is recorded with the numbers that
produced it, so a wrong call can be seen and overridden rather than guessed at.

Usage: world_collision.py <job_dir> [--prop-max-frac 0.16] [--prop-max-metres 1.5]
       [--max-hulls 24] [--force-prop slug,...] [--force-static slug,...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

# CoACD is chatty on stdout; keep the stage's own reporting readable.
try:
    import coacd
    coacd.set_log_level("error")
    HAVE_COACD = True
except Exception:  # noqa: BLE001
    HAVE_COACD = False


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def classify(extent, scene_extent, mpu, prop_max_frac: float, prop_max_metres: float):
    """Prop or static, with the numbers that decided it.

    Relative size is the primary signal and works uncalibrated — a capture with
    no meters_per_unit still has a meaningful "how big is this compared to the
    room". The absolute metre test is applied only when a scale exists, and can
    only ever DEMOTE something to static: a 2 m object is furniture regardless
    of how large the room is.
    """
    e = np.asarray(extent, dtype=float)
    longest = float(e.max())
    scene_longest = float(np.asarray(scene_extent, dtype=float).max())
    frac = longest / scene_longest if scene_longest > 0 else 1.0

    reasons = []
    is_prop = frac <= prop_max_frac
    reasons.append(f"longest {longest:.2f} = {frac:.1%} of scene ({scene_longest:.2f}); "
                   f"prop threshold {prop_max_frac:.0%}")
    if is_prop and mpu:
        metres = longest * mpu
        if metres > prop_max_metres:
            is_prop = False
            reasons.append(f"but {metres:.2f} m exceeds the {prop_max_metres} m prop limit")
        else:
            reasons.append(f"{metres:.2f} m within the {prop_max_metres} m prop limit")
    return ("prop" if is_prop else "static"), reasons


def convex_hulls(mesh_path: Path, out_dir: Path, base: str, max_hulls: int,
                 threshold: float = 0.05) -> dict:
    """CoACD decomposition -> UCX_<base>_NN.glb, plus the numbers to judge it."""
    if not HAVE_COACD:
        return {"ok": False, "reason": "coacd not installed in this environment"}
    m = trimesh.load(str(mesh_path), force="mesh", process=False)
    if len(m.faces) == 0:
        return {"ok": False, "reason": "empty mesh"}
    t = time.time()
    try:
        parts = coacd.run_coacd(
            coacd.Mesh(np.asarray(m.vertices, dtype=np.float64),
                       np.asarray(m.faces, dtype=np.int32)),
            threshold=threshold, max_convex_hull=max_hulls, merge=True)
    except Exception as exc:  # noqa: BLE001 — a prop without hulls is still a prop
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear THIS prop's previous hulls first. Hull count varies run to run (a
    # threshold change took the Bonsai props from 24 hulls to 8), and writing
    # over the survivors leaves the old high-numbered files orphaned — 96 files
    # on disk against 32 in the manifest, which is dead geometry a consumer can
    # still fetch. Scoped to UCX_<base>_NN.glb so a sibling prop's hulls are
    # never touched.
    stale = sorted(out_dir.glob(f"UCX_{base}_*.glb"))
    for old in stale:
        old.unlink()
    # The combined UE file is derived from the hulls; a stale one from a prior
    # run must not outlive a failed decomposition below.
    (out_dir / f"UE_{base}.glb").unlink(missing_ok=True)
    written, total_faces = [], 0
    for i, (vs, fs) in enumerate(parts):
        hull = trimesh.Trimesh(vertices=np.asarray(vs), faces=np.asarray(fs), process=False)
        if len(hull.faces) == 0:
            continue
        name = f"UCX_{base}_{i:02d}.glb"
        hull.export(str(out_dir / name))
        written.append(name)
        total_faces += int(len(hull.faces))
    if not written:
        return {"ok": False, "reason": "decomposition produced no usable hulls"}
    # COVERAGE, measured by sampling — not by comparing volumes. Volume was the
    # obvious check and it is wrong here: density-trimmed Poisson meshes are not
    # watertight, and trimesh.volume on an open mesh is meaningless, so the
    # comparison reported failures that were artifacts of the metric. Sampling
    # points on the render surface and asking whether each falls inside some
    # hull needs no closure assumption and answers the question that actually
    # matters — will the prop fall through its own collision.
    coverage = None
    try:
        pts, _ = trimesh.sample.sample_surface(m, 4000)
        inside = np.zeros(len(pts), dtype=bool)
        for n in written:
            h = trimesh.load(str(out_dir / n), force="mesh", process=False)
            if len(h.faces) and h.is_watertight:
                inside |= h.contains(pts)
        coverage = round(float(inside.mean()), 4)
    except Exception:  # noqa: BLE001 — the metric failing must not fail the prop
        coverage = None
    # Combined single-file UE import: render mesh + hulls in ONE glb, hull
    # nodes named UCX_<base>_NN. Unreal's UCX convention binds collision to the
    # render mesh only within a single imported file — the loose hull files can
    # never bind on their own, so this is what a UE operator imports, while the
    # loose hulls stay for three.js/rapier consumers that glob them.
    ue_name = f"UE_{base}.glb"
    try:
        combined = trimesh.Scene()
        combined.add_geometry(m, node_name=base, geom_name=base)
        for n in written:
            h = trimesh.load(str(out_dir / n), force="mesh", process=False)
            node = n[:-len(".glb")]
            combined.add_geometry(h, node_name=node, geom_name=node)
        combined.export(str(out_dir / ue_name))
        back = trimesh.load(str(out_dir / ue_name))
        got = len(getattr(back, "geometry", {}) or {})
        if got != len(written) + 1:
            raise ValueError(f"readback held {got} geometries, "
                             f"expected {len(written) + 1}")
        ue_glb = ue_name
    except Exception as exc:  # noqa: BLE001 — hulls alone are still a valid prop
        (out_dir / ue_name).unlink(missing_ok=True)
        ue_glb = None
        _log(f"  {base}: combined UE glb failed "
             f"({type(exc).__name__}: {exc})")
    return {"ok": True, "hulls": len(written), "files": written,
            "hull_faces_total": total_faces,
            "stale_removed": len(stale),
            "capped": len(written) >= max_hulls,
            "surface_coverage": coverage,
            "coverage_ok": (coverage is None) or coverage >= 0.9,
            "ue_glb": ue_glb,
            "seconds": round(time.time() - t, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--prop-max-frac", type=float, default=0.16,
                    help="prop if its longest side is at most this fraction of the scene's")
    ap.add_argument("--prop-max-metres", type=float, default=1.5,
                    help="absolute ceiling for a prop, applied only when calibrated")
    # 24 hulls on a cardboard box is not a decomposition, it is the cap being hit:
    # every Bonsai prop maxed out at threshold 0.05, because that concavity target
    # is tighter than the noise floor of a reconstructed surface, so CoACD keeps
    # splitting to chase detail that is not real. Game props are conventionally
    # 4-8 hulls; a looser threshold gets there and runs far faster.
    ap.add_argument("--max-hulls", type=int, default=8)
    ap.add_argument("--coacd-threshold", type=float, default=0.12)
    ap.add_argument("--force-prop", default="")
    ap.add_argument("--force-static", default="")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    t0 = time.time()
    job_dir = Path(args.job_dir).expanduser()
    world_json = job_dir / "_world" / "world.json"
    if not world_json.is_file():
        print(f"FATAL: no {world_json} — run scene_solidify.py first", file=sys.stderr)
        return 1
    world = json.loads(world_json.read_text())
    mpu = world.get("meters_per_unit")

    # Scene scale reference: the shell if we have it, else the union of elements.
    shell = world.get("shell") or {}
    scene_extent = shell.get("extent")
    if not scene_extent:
        exts = [e["extent"] for e in world.get("elements", []) if e.get("extent")]
        scene_extent = list(np.max(np.asarray(exts), axis=0)) if exts else [1.0, 1.0, 1.0]

    force_prop = {s.strip() for s in args.force_prop.split(",") if s.strip()}
    force_static = {s.strip() for s in args.force_static.split(",") if s.strip()}

    # Operator regrades are STICKY: the audited route records them in
    # _world/regrades.json precisely so a later rebuild does not silently
    # revert a human's static<->prop call back to the heuristic's.
    regrades: dict = {}
    regrades_path = job_dir / "_world" / "regrades.json"
    if regrades_path.is_file():
        try:
            raw = json.loads(regrades_path.read_text()).get("regrades") or {}
            regrades = {k: v for k, v in raw.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            _log("  WARNING: regrades.json unreadable — ignoring it")
    for slug, r in regrades.items():
        if r.get("role") == "prop":
            force_prop.add(slug)
        elif r.get("role") == "static":
            force_static.add(slug)

    hull_dir = job_dir / "_world" / "collision"
    out = []
    for el in world.get("elements", []):
        slug = el["slug"]
        if not el.get("built") or not el.get("extent"):
            out.append({**el, "role": "unbuilt", "reason": el.get("reason")})
            continue
        role, reasons = classify(el["extent"], scene_extent, mpu,
                                 args.prop_max_frac, args.prop_max_metres)
        if slug in force_prop:
            why = (regrades.get(slug) or {}).get("why")
            role = "prop"
            reasons = reasons + [f"forced prop by operator{': ' + why if why else ''}"]
        elif slug in force_static:
            why = (regrades.get(slug) or {}).get("why")
            role = "static"
            reasons = reasons + [f"forced static by operator{': ' + why if why else ''}"]

        rec = {"slug": slug, "label": el.get("label"), "role": role,
               "extent": el["extent"], "faces": el.get("faces"),
               "classification": reasons, "glb": el.get("glb")}
        if role == "prop":
            glb = job_dir / "_world" / "elements" / (el.get("glb") or "")
            if glb.is_file():
                rec["collision"] = convex_hulls(glb, hull_dir, slug, args.max_hulls,
                                                args.coacd_threshold)
                _log(f"  {slug}: prop -> {rec['collision'].get('hulls', 0)} hulls")
            else:
                rec["collision"] = {"ok": False, "reason": f"missing {glb}"}
        else:
            # Complex-as-simple: the render mesh IS the collision. Nothing to build.
            rec["collision"] = {"ok": True, "strategy": "complex_as_simple", "hulls": 0}
            _log(f"  {slug}: static -> complex-as-simple")
        out.append(rec)

    shell_rec = None
    if shell.get("built"):
        shell_rec = {"slug": "shell", "role": "static", "glb": shell.get("glb"),
                     "faces": shell.get("faces"), "extent": shell.get("extent"),
                     "collision": {"ok": True, "strategy": "complex_as_simple", "hulls": 0},
                     "classification": ["the shell is the environment; always static"]}
        _log("  shell: static -> complex-as-simple")

    doc = {
        "v": 1,
        "job_id": world.get("job_id"),
        "units": world.get("units"),
        "meters_per_unit": mpu,
        "scene_extent": [round(float(x), 3) for x in scene_extent],
        "thresholds": {"prop_max_frac": args.prop_max_frac,
                       "prop_max_metres": args.prop_max_metres if mpu else None,
                       "max_hulls": args.max_hulls},
        "coacd_available": HAVE_COACD,
        "shell": shell_rec,
        "elements": out,
        "counts": {
            "props": sum(1 for e in out if e.get("role") == "prop"),
            "static": sum(1 for e in out if e.get("role") == "static"),
            "unbuilt": sum(1 for e in out if e.get("role") == "unbuilt"),
            "hulls_total": sum((e.get("collision") or {}).get("hulls", 0) or 0 for e in out),
        },
        "seconds": round(time.time() - t0, 1),
    }
    dest = Path(args.report) if args.report else job_dir / "_world" / "world_manifest.json"
    dest.write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
