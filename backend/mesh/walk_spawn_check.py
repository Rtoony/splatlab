#!/usr/bin/env python3
"""Is there ground under the seed — or is the seed buried inside the collision solid?

world_shell.py's gates cast rays DOWN FROM ABOVE the scene and grade the first surface they hit. On outdoor Raw 360
scenes the splat-transform exterior skin wraps the sky floaters too, so that first surface is a lid at ~9 m, the
floor gate passes, and the walker spawns INSIDE the solid (street walk, splat_c75feef03f: under the seed the only
surfaces were an up-facing lid at y=9.0 and a down-facing bottom at y=-2.6). The seed is a real capture-camera
position, i.e. certainly free space, so ask the question from there:

  - down: the first surface hit must face UP (ground), within --max-drop metres;
  - up:   the first surface hit (if any) must face DOWN (a ceiling) — an up-facing one means we are inside solid.

  walk_spawn_check.py <job_dir>      (dn-splatter-probe env: trimesh)   -> one JSON line; exit 0 always
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--max-drop", type=float, default=4.0)
    args = ap.parse_args()
    world = Path(args.job_dir) / "_world"
    report = json.loads((world / "collision_shell.json").read_text())
    seed = (report.get("params") or {}).get("seed_yup")
    if not (isinstance(seed, list) and len(seed) == 3):
        print(json.dumps({"ok": False, "reason": "no seed in collision_shell.json"}))
        return 0
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(world / "collision_shell.glb"), force="mesh", process=False)

    def first_hit(direction: float):
        locs, _, tri = mesh.ray.intersects_location([seed], [[0.0, direction, 0.0]], multiple_hits=True)
        if not len(locs):
            return None
        i = int(np.argmin(np.abs(locs[:, 1] - seed[1])))
        return float(locs[i, 1]), float(mesh.face_normals[tri[i]][1])

    down, up = first_hit(-1.0), first_hit(1.0)
    out = {"seed_yup": seed, "down": down, "up": up}
    if down is None:
        out.update(ok=False, reason="nothing under the capture path")
    elif down[1] <= 0.0:
        out.update(ok=False, reason="the capture path is inside the collision solid (first surface below faces down)")
    elif seed[1] - down[0] > args.max_drop:
        out.update(ok=False, reason=f"ground is {seed[1] - down[0]:.1f} m under the capture path")
    elif up is not None and up[1] > 0.0:
        out.update(ok=False, reason="the capture path is inside the collision solid (first surface above faces up)")
    else:
        out.update(ok=True, ground_below_m=round(seed[1] - down[0], 3))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
