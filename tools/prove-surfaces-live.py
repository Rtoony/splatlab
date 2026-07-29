#!/usr/bin/env python3
"""PW-A9 live proof #2: painted structure becomes a wall you cannot walk
through — measured by ray coverage, before and after.

  1. paint a sparse wall region as `structure` (the UI's own endpoints)
  2. POST /scene/surfaces -> planar patches (assert rms, provenance tags)
  3. PRE-measure: ray grid through the patch region against the CURRENT
     shell.glb + collision_shell.glb (mesh-env trimesh, on-box)
  4. POST /world/prepare {"force": ["solidify"]} -> patches join the shell
     and the voxel collision solid
  5. POST-measure the same grid; assert coverage improved by the floor
     amounts and print both numbers regardless

The paint record and patches are LEFT IN PLACE — they are the improvement.

Usage: prove-surfaces-live.py JOB CX,CY,CZ RADIUS
         [--min-hit-frac 0.95] [--min-delta 0.3] [--min-cluster-points 150]
"""
import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")
ROOT = Path.home() / "projects/splatcli/outputs/3d"
MESH_ENV_PYTHON = Path.home() / "miniconda3/envs/dn-splatter-probe/bin/python"


def _load_token() -> str:
    """RAM secret store first, env fallback. The store can be unreadable from
    a MAC-confined shell (EACCES despite owner match) — don't crash on the
    read, fall through and say what to do."""
    try:
        for l in Path("/dev/shm/nexus-env-splatlab").read_text().splitlines():
            if l.startswith("PORTAL_TOKEN="):
                return l.split("=", 1)[1].strip()
    except OSError:
        pass
    tok = os.environ.get("PORTAL_TOKEN", "")
    if not tok:
        sys.exit("PORTAL_TOKEN unavailable — /dev/shm/nexus-env-splatlab "
                 "unreadable and env unset; run from a normal shell or "
                 "export PORTAL_TOKEN")
    return tok


TOKEN = _load_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def ray_coverage(job_dir: Path, plane: dict, grid: int = 20) -> dict:
    """hits/rays through the patch plane region, per shell artifact. Runs in
    the mesh env (trimesh); rays go along -normal from +0.75 units out."""
    script = f"""
import json
import numpy as np
import trimesh

plane = {json.dumps(plane)}
n = np.asarray(plane["normal"], dtype=np.float64)
d = float(plane["d"])
centre = -d * n + np.asarray(plane["centre_hint"])  # projected paint centre
extent = float(plane["extent"])
axis = np.zeros(3); axis[int(np.argmin(np.abs(n)))] = 1.0
u = np.cross(n, axis); u /= np.linalg.norm(u)
v = np.cross(n, u)
g = {grid}
lin = np.linspace(-extent, extent, g)
origins = np.array([centre + u * a + v * b + n * 0.75
                    for a in lin for b in lin])
dirs = np.tile(-n, (len(origins), 1))
out = {{}}
for name in ("shell.glb", "collision_shell.glb"):
    path = r"{job_dir}/_world/" + name
    try:
        m = trimesh.load(path, force="mesh", process=False)
        hits = m.ray.intersects_any(ray_origins=origins, ray_directions=dirs)
        out[name] = round(float(hits.mean()), 4)
    except Exception as exc:
        out[name] = f"error: {{type(exc).__name__}}"
print("RAYS " + json.dumps(out))
"""
    proc = subprocess.run([str(MESH_ENV_PYTHON), "-c", script],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-1500:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RAYS "))
    return json.loads(line[5:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("center", help="x,y,z of the wall paint sphere")
    ap.add_argument("radius", type=float)
    ap.add_argument("--min-hit-frac", type=float, default=0.95)
    ap.add_argument("--min-delta", type=float, default=0.3)
    ap.add_argument("--min-cluster-points", type=int, default=150)
    args = ap.parse_args()
    api = f"{BASE}/api/splat/jobs/{args.job}"
    job_dir = ROOT / args.job
    center = [float(x) for x in args.center.split(",")]

    # 1. Paint the wall (the UI's own endpoints).
    r = requests.post(f"{api}/langfield/select/sphere",
                      json={"center": center, "radius": args.radius},
                      headers=AUTH, timeout=300)
    if r.status_code == 503:
        sys.exit("FATAL: langfield worker is down — "
                 "systemctl --user start splatlab-langfield")
    assert r.status_code == 200, r.text
    count = int(r.headers.get("X-Count", "0"))
    payload = {"class_id": "structure",
               "indices_b64": base64.b64encode(r.content).decode(),
               "note": "prove-surfaces-live A9/B2"}
    r = requests.post(f"{api}/class-labels", json=payload, headers=AUTH,
                      timeout=300)
    if r.status_code == 400 and "force" in r.text.lower():
        r = requests.post(f"{api}/class-labels", json={**payload, "force": True},
                          headers=AUTH, timeout=300)
    assert r.status_code == 200, r.text
    print(f"1  PAINT: {count} gaussians committed as structure", flush=True)

    # 2. Build the patches.
    r = requests.post(f"{api}/scene/surfaces",
                      json={"min_cluster_points": args.min_cluster_points},
                      headers=AUTH, timeout=3600)
    assert r.status_code == 200, r.text
    report = r.json()
    assert report["provenance"] == "paint-synthesized"
    assert len(report["patches"]) >= 1, report
    patch = report["patches"][0]
    print(f"2  SURFACES: {len(report['patches'])} patch(es); patch_00 "
          f"rms={patch['plane']['rms_units']} inliers={patch['plane']['inliers']} "
          f"area={patch['area_units2']}u²", flush=True)
    ply_head = (job_dir / "_scene" / "surfaces" / patch["ply"]).read_bytes()[:2048]
    assert b"generative" in ply_head.lower(), \
        "patch ply is missing its provenance tag"
    print("2b PROVENANCE: GENERATIVE tag verified inside the patch ply header",
          flush=True)

    plane = {"normal": patch["plane"]["normal"], "d": patch["plane"]["d"],
             "centre_hint": center, "extent": max(0.3, args.radius)}

    # 3. PRE coverage on the current shells.
    pre = ray_coverage(job_dir, plane)
    print(f"3  PRE-COVERAGE: {pre}", flush=True)

    # 4. Rebuild the world with the patches in it.
    r = requests.post(f"{api}/world/prepare", json={"force": ["solidify"]},
                      headers=AUTH, timeout=7200)
    assert r.status_code == 200, r.text
    print(f"4  PREPARE: {r.json().get('stages')}", flush=True)

    # 4b. The union receipts from the build artifacts themselves.
    world = json.loads((job_dir / "_world" / "world.json").read_text())
    merged = ((world.get("shell") or {}).get("cut") or {}).get("surfaces_merged")
    collision_rec = {}
    cs_json = job_dir / "_world" / "collision_shell.json"
    if cs_json.is_file():
        collision_rec = json.loads(cs_json.read_text())
    sampled = None
    for key in ("chosen", "winner"):
        rec = collision_rec.get(key)
        if isinstance(rec, dict) and "surface_patches_sampled" in rec:
            sampled = rec["surface_patches_sampled"]
    if sampled is None:
        sampled = collision_rec.get("surface_patches_sampled")
    print(f"4c RECEIPTS: shell surfaces_merged={merged} "
          f"collision surface_patches_sampled={sampled}", flush=True)
    assert (merged or 0) >= 1 or (sampled or 0) >= 1, \
        "no artifact records the patch union — the wall never reached a shell"

    # 5. POST coverage + the verdict.
    post = ray_coverage(job_dir, plane)
    print(f"5  POST-COVERAGE: {post}", flush=True)
    for name in ("shell.glb", "collision_shell.glb"):
        pre_v, post_v = pre.get(name), post.get(name)
        if not isinstance(pre_v, float) or not isinstance(post_v, float):
            continue
        delta = post_v - pre_v
        print(f"   {name}: {pre_v:.3f} -> {post_v:.3f} (Δ {delta:+.3f})",
              flush=True)
    cs_pre, cs_post = pre.get("collision_shell.glb"), post.get("collision_shell.glb")
    assert isinstance(cs_post, float) and cs_post >= args.min_hit_frac, \
        f"collision coverage {cs_post} < floor {args.min_hit_frac}"
    if isinstance(cs_pre, float):
        assert cs_post - cs_pre >= args.min_delta or cs_pre >= args.min_hit_frac, \
            f"coverage barely moved ({cs_pre} -> {cs_post}) and was not already solid"
    print("ALL RECEIPTS LANDED — the painted wall is real geometry now",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
