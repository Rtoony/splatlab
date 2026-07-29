#!/usr/bin/env python3
"""Measure which ground-probe rule is right for a given world.

    ~/miniconda3/envs/langfield-spike/bin/python tools/measure-ground-probe.py <job_id> [samples] [units_per_metre]

The walker has to answer "what height is the ground at (x, z)?" for every actor,
and on a real solidified world that is NOT obvious: the collider is a voxel
terrain SLAB, often under canopy or ceiling geometry, so a column typically
returns 2-4 faces. This samples walkable columns and scores the candidate rules
against the navmesh's graded floor, which is the only independent statement of
where the ground is.

W2 shipped `nearest-within-band` on the strength of this measurement over 600
Stump columns at the app's own 1.9184 u/m: |error| p50 0.15 / p95 0.76 / max
0.76, with no canopy tail at all — versus max 1.88 for highest-below-cap and
p95 2.87 / max 3.72 for naive-highest. Re-run it on any NEW world — especially
an indoor walk-through capture, whose ceilings sit far closer to the floor than
a forest canopy does — before assuming the band still fits. Measure, then decide;
do not carry the constant over on faith.

Read-only: opens the world's collision shell and navmesh, writes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

JOBS = Path.home() / "projects/splatcli/outputs/3d"
BAND_M = 1.6      # what world-game.ts ships (GROUND_BAND_M)
CAP_M = 1.0       # the rejected highest-below-cap rule, for comparison


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    job = sys.argv[1]
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    # Bands are expressed in METRES, so the walker's scale dial decides how wide
    # they really are. On an UNCALIBRATED world that dial is a guess the operator
    # can move, and it changes the answer — pass the value the HUD is showing.
    upm_override = float(sys.argv[3]) if len(sys.argv) > 3 else None
    world = JOBS / job / "_world"
    shell = world / "collision_shell.glb"
    if not shell.is_file():
        shell = world / "shell.glb"
    if not shell.is_file() or not (world / "navmesh.json").is_file():
        print(f"no collider + navmesh under {world}")
        return 1

    mesh = trimesh.load(shell, force="mesh")
    nav = json.loads((world / "navmesh.json").read_text())
    floor_y, cell = nav["floor_y"], nav["cell"]
    ox, oz = nav["origin"]
    walkable = [(i, j) for j, row in enumerate(nav["rows"])
                for i, c in enumerate(row) if c == "1"]
    if not walkable:
        print("navmesh has no walkable cells")
        return 1

    # Scale: metre-baked worlds declare 1.0; uncalibrated ones have no metres,
    # so the bands below are in scene units and read as "about a body height".
    manifest = json.loads((world / "world_manifest.json").read_text())
    mpu = manifest.get("meters_per_unit")
    upm = upm_override if upm_override else (1.0 / mpu if mpu else 1.0)
    origin = ("manifest" if (mpu and not upm_override)
              else "given" if upm_override else "ASSUMED — pass the HUD's u/m")
    print(f"{job}: {len(walkable)} walkable cells, floor_y={floor_y:.3f}, "
          f"collider y {mesh.bounds[0][1]:.2f}..{mesh.bounds[1][1]:.2f}, "
          f"{'calibrated' if mpu else 'UNCALIBRATED'} | {upm:.4f} u/m ({origin})")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(walkable), size=min(samples, len(walkable)), replace=False)
    origins, top = [], mesh.bounds[1][1] + 5
    for k in pick:
        i, j = walkable[k]
        origins.append([ox + (i + 0.5) * cell, top, oz + (j + 0.5) * cell])
    origins = np.array(origins)
    locs, idx, _ = mesh.ray.intersects_location(
        origins, np.tile([0, -1, 0], (len(origins), 1)), multiple_hits=True)

    columns: dict[int, list[float]] = {}
    for point, ray in zip(locs, idx):
        columns.setdefault(int(ray), []).append(float(point[1]))
    if not columns:
        print("no column hit the collider at all — check the frames match")
        return 1

    counts = np.array([len(v) for v in columns.values()])
    print(f"faces per column: mean {counts.mean():.2f}, max {counts.max()} "
          f"({len(columns)} columns hit)")

    def score(name: str, choose) -> None:
        picked, missed = [], 0
        for hits in columns.values():
            y = choose(hits)
            if y is None:
                missed += 1
            else:
                picked.append(y)
        if not picked:
            print(f"  {name:26s} no pick in any column")
            return
        err = np.abs(np.array(picked) - floor_y)
        print(f"  {name:26s} |err| p50 {np.percentile(err, 50):5.2f} "
              f"p95 {np.percentile(err, 95):5.2f} max {err.max():5.2f} "
              f"| fell back {missed:3d}/{len(columns)}")

    band, cap = BAND_M * upm, CAP_M * upm
    print(f"error vs the graded floor (band {BAND_M} m = {band:.2f} u, "
          f"cap {CAP_M} m = {cap:.2f} u):")
    score("highest (naive)", max)
    score("lowest (naive)", min)
    score("highest-below-cap", lambda h: max((y for y in h if y <= floor_y + cap),
                                             default=None))
    score("nearest-within-band  <-", lambda h: min(
        (y for y in h if abs(y - floor_y) <= band), key=lambda y: abs(y - floor_y),
        default=None))
    print("The shipped rule is walker.surfaceNear() = nearest-within-band. If a "
          "different rule wins here by a clear margin, that is a finding — say so "
          "with these numbers rather than tuning the constant quietly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
