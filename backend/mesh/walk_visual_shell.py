#!/usr/bin/env python3
"""A light visual stand-in for a Raw 360 walk: _world/collision_shell.glb decimated to <= --max-faces as shell.glb.

The walker downloads the visual shell AND the collision solid, and hides the shell whenever the splat is showing.
A plain copy doubled the download; on the 434 s pool clip that was 2 x 298 MB (16.6M triangles) and the walker
never finished loading. Collision keeps the full-resolution solid; only the stand-in is decimated.

  walk_visual_shell.py <job_dir> [--max-faces 200000]      (dn-splatter-probe env: open3d + trimesh)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir")
    ap.add_argument("--max-faces", type=int, default=200_000)
    args = ap.parse_args()
    world = Path(args.job_dir) / "_world"
    src, dst = world / "collision_shell.glb", world / "shell.glb"
    import trimesh

    mesh = trimesh.load(str(src), force="mesh", process=False)
    faces_in = int(len(mesh.faces))
    if faces_in <= args.max_faces:
        shutil.copyfile(src, dst)
        faces_out = faces_in
    else:
        import numpy as np
        import open3d as o3d

        o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
                                       o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)))
        o3 = o3.simplify_quadric_decimation(target_number_of_triangles=args.max_faces)
        out = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles), process=False)
        tmp = dst.with_suffix(".tmp.glb")
        out.export(str(tmp))
        tmp.replace(dst)
        faces_out = int(len(out.faces))
    print(json.dumps({"faces_in": faces_in, "faces_out": faces_out, "bytes": dst.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
