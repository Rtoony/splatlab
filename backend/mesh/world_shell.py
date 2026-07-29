#!/usr/bin/env python3
"""world_shell.py — the WALKABLE COLLISION SOLID for a navigable scene shell.

The render shell and the collision shell cannot be the same mesh, and this is
measured, not assumed. The TSDF surface of an indoor capture (_mesh/mesh.ply)
is a thin LACY WEB: it looks like a room, but its largest connected component
is only ~60% of it and every decimation budget shatters it further (400k tris
-> 2088 components, 60k tris -> 819). A physics engine walking a player over
that lace drops them through the floor.

So: keep the lace for rendering, and build a SEPARATE solid for collision.
This module builds that solid and, more importantly, PROVES it walkable with
four executed gates written next to the artifact:

  1. components          connected components; one should hold >90% of tris
  2. watertight          trimesh .is_watertight on the exported solid
  3. floor_continuity    grid of downward rays over the interior footprint;
                         fraction that hit geometry. A lacy floor fails this.
  4. max_hole_span       largest inscribed gap in that ray grid, scene units.
                         A capsule falls through anything wider than 2*radius.

Two independent routes are built and SCORED against each other; the winner is
exported. Nothing is claimed that a gate did not measure.

  ROUTE splat-transform  the existing collision lane (backend/export_route.py
        POST /jobs/{id}/collision). Voxelizes the gaussians on the GPU and
        emits a .collision.glb. --voxel-floor-fill walks every column upward
        from the grid floor until it hits solid, which is exactly a walkable
        ground slab; empty columns become wall pillars.
  ROUTE voxel            self-contained fallback: rasterize the TSDF surface
        into an occupancy volume, morphologically close it so the lace welds
        into thick walls, fill every interior column down to the floor plane,
        then marching-cubes a closed surface out of it.

FRAME. The capture frame is Z-up (repo canon: mesh_report.py exports GLB with
rotation_matrix(-pi/2, [1,0,0])). splat-transform voxelizes in engine space,
which is Y-up, so the input is rotated -90 deg about X on the way in -- the
same rotation the rest of the repo already applies for glTF. The collision GLB
is therefore Y-up and drops straight into three.js next to _world/shell.glb.

UNITS. The test capture is UNCALIBRATED (meta.json meters_per_unit: null).
Every length here -- voxel size, hole span, player radius -- is in SCENE UNITS.

Usage: world_shell.py <job_dir>
       [--route auto|splat-transform|voxel] [--voxel-size N] [--report PATH]
       [--out-glb PATH] [--st-mode exterior|interior|both] [--seed x,y,z]
       [--player-radius N] [--player-height N] [--grid-res N] [--source PLY]
       [--keep-largest] [--json]

Requires the geometry env: ~/miniconda3/envs/dn-splatter-probe/bin/python
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_PY = Path.home() / "miniconda3" / "envs" / "dn-splatter-probe" / "bin" / "python"
SPLAT_TRANSFORM = (
    Path.home() / "projects" / "supersplat" / "node_modules" / ".bin" / "splat-transform"
)

# Capture frame is Z-up; glTF is Y-up. (x, y, z) -> (x, z, -y).
SCENE_UP_AXIS = 2
YUP_FROM_SCENE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)
# The Euler triple that makes splat-transform apply the same rotation. Verified
# empirically: -r -90,0,0 maps input (x,y,z) -> output (x, z, -y).
ST_ROTATE = "-90,0,0"

SOLID_OPACITY = 0.5  # sigmoid(opacity) > this == "solid", repo convention


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight() -> dict:
    """Fail loud on a wrong interpreter instead of dying halfway through a build."""
    missing = []
    versions = {}
    for name in ("trimesh", "open3d", "scipy", "skimage", "plyfile"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001 - report every miss, not the first
            missing.append(name)
    if missing:
        raise SystemExit(
            f"world_shell: missing {', '.join(missing)} under {sys.executable}\n"
            f"  run it with: {DEFAULT_PY} {Path(__file__).name} <job_dir>"
        )
    if not SPLAT_TRANSFORM.is_file():
        versions["splat_transform"] = None
    else:
        versions["splat_transform"] = str(SPLAT_TRANSFORM)
    return versions


# ---------------------------------------------------------------- inputs


def load_gaussians(ply: Path, solid_only: bool = True) -> np.ndarray:
    """Gaussian centres from a 3DGS .ply, in the capture (Z-up) frame."""
    from plyfile import PlyData

    el = PlyData.read(str(ply)).elements[0]
    pts = np.stack(
        [np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1
    ).astype(np.float64)
    names = {p.name for p in el.properties}
    if solid_only and "opacity" in names:
        alpha = 1.0 / (1.0 + np.exp(-np.asarray(el["opacity"]).astype(np.float64)))
        pts = pts[alpha > SOLID_OPACITY]
    return pts


def to_yup(pts: np.ndarray) -> np.ndarray:
    return pts @ YUP_FROM_SCENE.T


def room_box(job_dir: Path, pts_scene: np.ndarray, margin: float) -> tuple:
    """Crop box in the capture frame.

    The TSDF mesh is the tightest honest statement of where the room is: it was
    carved from the depth renders, so it excludes the far-field floaters that
    would otherwise blow the voxel grid up to 60 units across. Percentile bounds
    (not min/max) because the mesh itself carries a few stragglers.
    """
    mesh_ply = job_dir / "_mesh" / "mesh.ply"
    if mesh_ply.is_file():
        import trimesh

        verts = np.asarray(
            trimesh.load(str(mesh_ply), process=False).vertices, dtype=np.float64
        )
        lo = np.percentile(verts, 0.1, axis=0) - margin
        hi = np.percentile(verts, 99.9, axis=0) + margin
        return lo, hi, "mesh.ply p0.1-p99.9"
    lo = np.percentile(pts_scene, 2.0, axis=0) - margin
    hi = np.percentile(pts_scene, 98.0, axis=0) + margin
    return lo, hi, "source cloud p2-p98"


def frame_check(pts_scene: np.ndarray) -> dict:
    """Evidence for the Z-up assumption, recorded rather than trusted.

    A room's up axis is the one whose coordinate histogram has the sharpest
    single peak: every floor point shares one height, while x/y spread evenly.
    """
    peaks = []
    for axis in range(3):
        col = pts_scene[:, axis]
        lo, hi = np.percentile(col, [1, 99])
        hist, _ = np.histogram(col, bins=120, range=(lo, hi))
        peaks.append(float(hist.max() / max(1, hist.sum())))
    detected = int(np.argmax(peaks))
    return {
        "assumed_up_axis": "z (capture frame)",
        "histogram_peak_fraction": [round(p, 4) for p in peaks],
        "detected_up_axis": "xyz"[detected],
        "agrees": detected == SCENE_UP_AXIS,
    }


# ---------------------------------------------------------------- the probe grid


@dataclass
class Probe:
    """Everything the gates need that is derived from the INPUT, not the result.

    Deriving the interior mask and the floor height from the source cloud keeps
    the gates honest: a candidate cannot pass by having no floor to test.
    """

    cell: float
    origin: np.ndarray  # (x0, z0) lower corner of the grid, Y-up frame
    shape: tuple
    footprint: np.ndarray  # bool[nx, nz] room outline, holes filled
    interior: np.ndarray  # footprint eroded by the player radius
    column_top: np.ndarray  # float[nx, nz] highest source point per column
    capsule_free: np.ndarray  # bool[nx, nz] no source point in the capsule band
    seed: np.ndarray  # a known-clear point, Y-up frame
    floor_level: float
    top_level: float

    def centres(self) -> tuple:
        nx, nz = self.shape
        cx = self.origin[0] + (np.arange(nx) + 0.5) * self.cell
        cz = self.origin[1] + (np.arange(nz) + 0.5) * self.cell
        return np.meshgrid(cx, cz, indexing="ij")


def build_probe(
    pts_yup: np.ndarray, cell: float, player_radius: float, player_height: float
) -> Probe:
    from scipy import ndimage

    lo = pts_yup.min(axis=0)
    hi = pts_yup.max(axis=0)
    nx = max(2, int(np.ceil((hi[0] - lo[0]) / cell)))
    nz = max(2, int(np.ceil((hi[2] - lo[2]) / cell)))
    ix = np.clip(((pts_yup[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iz = np.clip(((pts_yup[:, 2] - lo[2]) / cell).astype(int), 0, nz - 1)
    flat = ix * nz + iz

    count = np.zeros(nx * nz, dtype=np.int64)
    np.add.at(count, flat, 1)
    col_min = np.full(nx * nz, np.inf)
    col_max = np.full(nx * nz, -np.inf)
    np.minimum.at(col_min, flat, pts_yup[:, 1])
    np.maximum.at(col_max, flat, pts_yup[:, 1])

    # Scale the occupancy threshold with cell AREA so the footprint (and every
    # gate measured over it) means the same region at any --grid-res.
    min_pts = max(1, int(round(3 * (cell / 0.05) ** 2)))
    occupied = (count >= min_pts).reshape(nx, nz)
    closed = ndimage.binary_closing(occupied, np.ones((3, 3), bool))
    footprint = ndimage.binary_fill_holes(closed)
    labels, n = ndimage.label(footprint)
    if n > 1:  # keep the room, drop detached satellites
        sizes = ndimage.sum(footprint, labels, range(1, n + 1))
        footprint = labels == (int(np.argmax(sizes)) + 1)

    erode = max(1, int(np.ceil(player_radius / cell)))
    struct = np.ones((2 * erode + 1, 2 * erode + 1), bool)
    interior = ndimage.binary_erosion(footprint, struct)
    if not interior.any():  # tiny capture: fall back to the raw footprint
        interior = footprint

    # Floor = the median of the per-column minima inside the room. Robust to
    # both the far-field tail below the floor and to furniture above it. The
    # naive alternative (mode of the height histogram) picks whichever flat
    # surface happens to be densest, which on this capture is not the floor.
    grid_min = col_min.reshape(nx, nz)
    grid_max = col_max.reshape(nx, nz)
    inside = interior & np.isfinite(grid_min)
    floor_level = float(np.median(grid_min[inside])) if inside.any() else float(lo[1])
    top_level = float(np.percentile(pts_yup[:, 1], 99.5))

    # Which columns the CAPTURE itself leaves clear for a standing capsule.
    # This is the ceiling on any candidate's walkable area -- a solid cannot
    # open up space the room does not have -- so it is what walkable_frac gets
    # compared against instead of an invented 100%.
    band = (pts_yup[:, 1] >= floor_level + 0.15) & (
        pts_yup[:, 1] <= floor_level + player_height
    )
    blocked = np.zeros((nx, nz), bool)
    blocked[ix[band], iz[band]] = True
    capsule_free = interior & ~blocked

    if capsule_free.any():  # seed the cluster/carve passes from real free space
        fx, fz = np.nonzero(capsule_free)
        cx, cz = np.mean(np.nonzero(interior), axis=1)
        pick = int(np.argmin((fx - cx) ** 2 + (fz - cz) ** 2))
        seed = np.array(
            [
                lo[0] + (fx[pick] + 0.5) * cell,
                floor_level + 0.3,
                lo[2] + (fz[pick] + 0.5) * cell,
            ]
        )
    else:
        seed = np.array([float(np.median(pts_yup[:, 0])), floor_level + 0.3,
                         float(np.median(pts_yup[:, 2]))])

    return Probe(
        cell=cell,
        origin=np.array([lo[0], lo[2]]),
        shape=(nx, nz),
        footprint=footprint,
        interior=interior,
        column_top=grid_max,
        capsule_free=capsule_free,
        seed=seed,
        floor_level=floor_level,
        top_level=top_level,
    )


# ---------------------------------------------------------------- the gates


def evaluate(mesh, probe: Probe, player_radius: float, player_height: float) -> dict:
    """Run all four acceptance gates plus the diagnostics that keep them honest."""
    import open3d as o3d
    import trimesh.graph as tg
    from scipy import ndimage

    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    if len(faces) == 0:
        return {"error": "empty mesh", "components": 0, "watertight": False,
                "floor_continuity": 0.0, "max_hole_span": float("inf")}

    # gate 1 + 2
    comps = tg.connected_components(mesh.face_adjacency, nodes=np.arange(len(faces)))
    sizes = sorted((len(c) for c in comps), reverse=True)
    gates = {
        "triangles": int(len(faces)),
        "vertices": int(len(verts)),
        "components": int(len(comps)),
        "largest_component_frac": round(float(sizes[0] / len(faces)), 4),
        "watertight": bool(mesh.is_watertight),
    }

    # gates 3 + 4: downward rays over the interior footprint.
    #
    # WHERE the rays start decides what is measured, and the obvious choice is
    # wrong. Starting above the whole scene measures the TOPMOST surface, which
    # on this capture is a canopy of oversized far-field gaussians ~0.5 above
    # the room -- that scores a perfect 1.0 while saying nothing about the
    # floor. Rays therefore start at the player's HEAD height above the floor
    # plane: a hit then proves both that ground exists below the player and
    # (because the ray travelled the whole way) that the capsule column above
    # that ground is clear. The literal from-above cast is kept as `canopy_*`
    # diagnostics.
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts.astype(np.float64)),
                o3d.utility.Vector3iVector(faces.astype(np.int32)),
            )
        )
    )
    gx, gz = probe.centres()
    sel = probe.interior
    n_rays = int(sel.sum())
    gates["rays"] = n_rays

    def cast_down(start_y: float) -> np.ndarray:
        origins = np.stack(
            [gx[sel], np.full(n_rays, start_y, np.float32), gz[sel]], axis=1
        ).astype(np.float32)
        down = np.tile(np.array([0, -1, 0], np.float32), (n_rays, 1))
        return scene.cast_rays(o3d.core.Tensor(np.hstack([origins, down])))[
            "t_hit"
        ].numpy()

    head_y = float(probe.floor_level + player_height)
    t_head = cast_down(head_y)
    hit = np.isfinite(t_head)
    hit_y = head_y - t_head
    gates["ray_start_y"] = round(head_y, 4)
    # The ACCEPTED floor number is the shared measurement world_gate also
    # consumes (floor_support.measure_floor_support), so the two graders can
    # never disagree on the same mesh again (reconciliation, 2026-07-28).
    # The probe-grid head-cast stays as a diagnostic and keeps feeding the
    # capsule/walkability machinery below, which is tied to the source-cloud
    # probe by design.
    gates["floor_continuity_probe"] = (
        round(float(hit.mean()), 4) if n_rays else 0.0
    )
    try:
        import floor_support
        fs = floor_support.measure_floor_support(verts, faces, "Y")
        gates["floor_continuity"] = fs["coverage"]
        gates["floor_support"] = {
            key: fs[key]
            for key in ("largest_gap_frac", "largest_gap_span", "ground_level",
                        "probe_level", "ground_band_coverage")
        }
    except ValueError as exc:
        gates["floor_continuity"] = 0.0
        gates["floor_support"] = {"error": str(exc)[:200]}
    if hit.any():
        gates["hit_height_pct"] = [
            round(float(v), 3) for v in np.percentile(hit_y[hit], [5, 50, 95])
        ]
        gates["ground_level_frac"] = round(
            float(
                (
                    (hit_y[hit] >= probe.floor_level - 0.25)
                    & (hit_y[hit] <= probe.floor_level + 0.5)
                ).mean()
            ),
            4,
        )

    # Is the capsule column actually empty, or did the ray start buried inside
    # a solid fill? Occupancy is intersection parity, so it is exact on the
    # watertight candidates and indicative on the rest -- reported either way.
    #
    # The sample heights are measured from the GROUND THIS COLUMN ACTUALLY HAS,
    # not from the floor plane: a fill slab and its marching-cubes skin raise
    # the standing surface a few voxels, and probing from the plane instead
    # samples inside the slab and reports the whole room blocked.
    n_s = 5
    base = np.where(hit, hit_y, probe.floor_level)
    offs = np.linspace(0.1, player_height, n_s)
    probes = np.stack(
        [
            np.repeat(gx[sel], n_s),
            (base[:, None] + offs[None, :]).reshape(-1),
            np.repeat(gz[sel], n_s),
        ],
        axis=1,
    ).astype(np.float32)
    occ = scene.compute_occupancy(o3d.core.Tensor(probes)).numpy().reshape(n_rays, -1)
    clear = ~occ.astype(bool).any(axis=1)
    gates["capsule_blocked_by_height"] = [
        round(float(v), 4) for v in occ.astype(bool).mean(axis=0)
    ]
    gates["capsule_sample_offsets"] = [round(float(v), 3) for v in offs]
    gates["capsule_clear_frac"] = round(float(clear.mean()), 4)
    # ground to stand on AND room to stand in
    gates["walkable_frac"] = round(float((hit & clear).mean()), 4)
    # The per-cell mask, kept for the navmesh export (P3): actors path over
    # exactly the cells the acceptance gates graded as standable. Underscore
    # prefix = side-channel, stripped before any JSON serialization.
    walk_grid = np.zeros(probe.shape, bool)
    walk_grid[sel] = hit & clear
    gates["_walkable"] = {
        "cell": float(probe.cell),
        "origin": [float(probe.origin[0]), float(probe.origin[1])],
        "shape": [int(probe.shape[0]), int(probe.shape[1])],
        "floor_y": float(probe.floor_level),
        "rows": ["".join("1" if v else "0" for v in row) for row in walk_grid],
    }
    src_free = float(probe.capsule_free[sel].mean())
    gates["source_capsule_free_frac"] = round(src_free, 4)
    gates["walkable_vs_source"] = (
        round(float(gates["walkable_frac"] / src_free), 4) if src_free > 0 else None
    )

    # diagnostics: the literal from-above cast, and whether that surface sits
    # where the capture says the scene actually is
    canopy_y = float(max(verts[:, 1].max(), probe.top_level) + 1.0)
    t_can = cast_down(canopy_y)
    can_hit = np.isfinite(t_can)
    gates["canopy_continuity"] = round(float(can_hit.mean()), 4) if n_rays else 0.0
    src_top = probe.column_top[sel]
    both = can_hit & np.isfinite(src_top)
    if both.any():
        delta = (canopy_y - t_can)[both] - src_top[both]
        gates["canopy_delta_pct"] = [
            round(float(v), 3) for v in np.percentile(delta, [5, 50, 95])
        ]
        gates["phantom_roof_frac"] = round(
            float((delta > max(0.25, 4 * probe.cell)).mean()), 4
        )

    # gate 4: biggest circle that fits in a hole == what a capsule falls through
    miss = np.zeros(probe.shape, bool)
    miss[sel] = ~hit
    if miss.any():
        dt = ndimage.distance_transform_edt(miss)
        gates["max_hole_span"] = round(float(2.0 * dt.max() * probe.cell), 4)
        _, n_holes = ndimage.label(miss)
        gates["hole_regions"] = int(n_holes)
        gates["hole_area_frac"] = round(float(miss.sum() / max(1, n_rays)), 4)
    else:
        gates["max_hole_span"] = 0.0
        gates["hole_regions"] = 0
        gates["hole_area_frac"] = 0.0

    gates["targets"] = {
        "largest_component_frac>=0.90": gates["largest_component_frac"] >= 0.90,
        "watertight": gates["watertight"],
        "floor_continuity>0.95": gates["floor_continuity"] > 0.95,
        f"max_hole_span<={player_radius}": gates["max_hole_span"] <= player_radius,
    }
    gates["gates_passed"] = int(sum(gates["targets"].values()))
    # "walkable" is the required-gate question: can a player stand anywhere in
    # the room without falling through? Room clutter is NOT a failure, so the
    # standing-room numbers are reported and ranked on, never gated on.
    gates["walkable"] = bool(
        gates["floor_continuity"] > 0.95 and gates["max_hole_span"] <= player_radius
    )
    return gates


def rank_key(cand: dict) -> tuple:
    g = cand.get("gates") or {}
    return (
        1 if g.get("walkable") else 0,
        g.get("gates_passed", 0),
        g.get("walkable_frac", 0.0),
        g.get("floor_continuity", 0.0),
        -g.get("max_hole_span", 1e9),
        g.get("largest_component_frac", 0.0),
    )


# ---------------------------------------------------------------- route 1


def route_splat_transform(
    source: Path,
    work: Path,
    box: tuple,
    seed_yup: np.ndarray,
    voxel: float,
    mode: str,
    style: str,
    fill: float,
    carve: tuple | None,
    cluster: bool,
    timeout: int,
) -> dict:
    """Drive the existing collision lane (same flags as export_route.py).

    Order matters: --filter-box is expressed in the capture frame, so it runs
    BEFORE the rotation into engine space; everything after it (cluster, voxel,
    seed) is Y-up.
    """
    lo, hi = box
    tag = f"{mode}_{style}{'_carve' if carve else ''}"
    out = work / f"st_{tag}.voxel.json"
    cmd = [str(SPLAT_TRANSFORM), str(source)]
    cmd += ["--filter-box", ",".join(f"{v:.6g}" for v in (*lo, *hi))]
    cmd += ["-r", ST_ROTATE]
    if cluster:
        cmd += ["--filter-cluster", "1,0.999,0.1"]
    cmd += ["--seed-pos", ",".join(f"{v:.6g}" for v in seed_yup)]
    cmd += ["--voxel-params", f"{voxel:.6g},0.1"]
    if mode == "exterior":
        cmd += ["--voxel-floor-fill", f"{fill:.6g}"]
    elif mode == "interior":
        cmd += ["--voxel-external-fill", f"{fill:.6g}"]
    if carve:
        cmd += ["--voxel-carve", f"{carve[0]:.6g},{carve[1]:.6g}"]
    cmd += ["--collision-mesh", style, "-w", "--no-tty", "-q", str(out)]

    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    glb = out.with_name(out.name.replace(".voxel.json", ".collision.glb"))
    rec = {
        "route": f"splat-transform/{mode}/{style}" + ("+carve" if carve else ""),
        "command": cmd,
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "glb": str(glb),
    }
    if proc.returncode != 0 or not glb.is_file():
        rec["error"] = (proc.stderr or proc.stdout or "no output")[-800:]
        return rec
    rec["glb_bytes"] = glb.stat().st_size
    if out.is_file():
        try:
            rec["voxel_json"] = json.loads(out.read_text())
        except Exception:  # noqa: BLE001 - metadata only
            pass
    return rec


# ---------------------------------------------------------------- route 2


def route_voxel(
    job_dir: Path,
    pts_yup: np.ndarray,
    probe: Probe,
    box_yup: tuple,
    voxel: float,
    close_radius: int,
    work: Path,
    floor_mode: str = "plane",
) -> dict:
    """Voxel-solidify the TSDF surface ourselves.

    Three moves, in this order, each fixing what the previous one leaves:
      rasterize  surface samples (not just vertices) so no thin triangle is
                 missed between two voxel centres;
      close      dilate then partially erode, welding the lace into walls that
                 are thicker than they are holey;
      column-fill make the ground continuous under the whole room outline.

    floor_mode decides the last move, and it is the difference between a solid
    that is merely un-fall-through-able and one that is walkable:
      plane        fill up to the LOCAL GROUND -- the lowest solid voxel within
                   +-0.4 of the floor plane, or the plane itself where the
                   capture never saw the floor. Space under tables stays open.
      first-solid  fill up to the lowest solid voxel whatever its height, which
                   is splat-transform's --voxel-floor-fill semantic. Guarantees
                   ground, but buries every under-table volume in the room.
    marching_cubes on a zero-padded volume closes the surface by construction.
    """
    import trimesh
    from scipy import ndimage
    from skimage import measure

    started = time.time()
    rec = {"route": "voxel", "voxel_size": voxel, "close_radius": close_radius}
    mesh_ply = job_dir / "_mesh" / "mesh.ply"
    lo, hi = box_yup

    samples = []
    if mesh_ply.is_file():
        tm = trimesh.load(str(mesh_ply), process=False)
        tm.vertices = to_yup(np.asarray(tm.vertices, dtype=np.float64))
        area = float(tm.area)
        n = int(min(12_000_000, max(2_000_000, area / (voxel * 0.4) ** 2)))
        samples.append(np.asarray(tm.sample(n)))
        samples.append(np.asarray(tm.vertices))
        rec["source"] = f"{mesh_ply.name} ({len(tm.faces)} tris, {n} surface samples)"
    # Painted surface patches (PW-A8): higher-authority gap-closers — a wall
    # the user painted must reach the COLLISION solid, or it stays
    # walk-through-able on the mainline (voxel) route.
    patch_faces = 0
    for patch in sorted((job_dir / "_scene" / "surfaces").glob("patch_*.ply")):
        try:
            pm = trimesh.load(str(patch), process=False)
            if len(pm.faces):
                pm.vertices = to_yup(np.asarray(pm.vertices, dtype=np.float64))
                n_patch = int(min(500_000,
                                  max(20_000, float(pm.area) / (voxel * 0.4) ** 2)))
                samples.append(np.asarray(pm.sample(n_patch)))
                samples.append(np.asarray(pm.vertices))
                patch_faces += int(len(pm.faces))
        except Exception as exc:  # noqa: BLE001 — recorded, never fatal
            rec.setdefault("surface_patch_errors", []).append(
                f"{patch.name}: {type(exc).__name__}: {exc}"[:160])
    rec["surface_patches_sampled"] = patch_faces
    samples.append(pts_yup)  # gaussians close the gaps the TSDF never carved
    pts = np.concatenate(samples, axis=0)
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]

    dims = np.maximum(2, np.ceil((hi - lo) / voxel).astype(int))
    idx = np.clip(((pts - lo) / voxel).astype(int), 0, dims - 1)
    vol = np.zeros(tuple(dims), bool)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    rec["occupied_voxels_raw"] = int(vol.sum())

    if close_radius > 0:
        ball = np.ones((2 * close_radius + 1,) * 3, bool)
        vol = ndimage.binary_dilation(vol, ball)
        if close_radius > 1:  # net thickening of one voxel per side
            small = np.ones((2 * close_radius - 1,) * 3, bool)
            vol = ndimage.binary_erosion(vol, small)
    rec["occupied_voxels_closed"] = int(vol.sum())

    # column fill, restricted to the room outline so the solid stays room-shaped
    foot = _resample_mask(probe.footprint, probe.origin, probe.cell, lo, voxel, dims)
    floor_idx = int(np.clip((probe.floor_level - lo[1]) / voxel, 0, dims[1] - 1))
    ramp = np.arange(dims[1])[None, :, None]
    if floor_mode == "first-solid":
        slack = max(1, int(round(0.15 / voxel)))
        first = np.where(vol.any(axis=1), vol.argmax(axis=1), floor_idx)
        fill_to = np.clip(np.maximum(first, floor_idx - slack), 0, dims[1] - 1)
    else:
        # Only ever fill UP to local ground, never down to it. Splat noise sits
        # below the floor plane as often as above it, and following it downward
        # turns a flat floor into a 0.8-unit-deep bumpy trench.
        hi_i = min(dims[1], floor_idx + int(round(0.4 / voxel)) + 1)
        near = vol[:, floor_idx:hi_i, :]
        has = near.any(axis=1)
        fill_to = np.where(has, floor_idx + near.argmax(axis=1), floor_idx)
    fill_to = np.where(foot, fill_to, -1)
    vol |= ramp <= fill_to[:, None, :]
    rec["floor_mode"] = floor_mode
    rec["occupied_voxels_filled"] = int(vol.sum())

    padded = np.pad(vol.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1.0) * voxel + lo
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    glb = work / "voxel_solid.glb"
    mesh.export(str(glb))
    rec.update(
        {
            "glb": str(glb),
            "glb_bytes": glb.stat().st_size,
            "grid_dims": [int(v) for v in dims],
            "seconds": round(time.time() - started, 2),
            "returncode": 0,
        }
    )
    return rec


def _resample_mask(mask, origin, cell, lo, voxel, dims):
    """Footprint mask (probe grid) -> the voxel route's XZ grid."""
    nx, nz = dims[0], dims[2]
    xs = lo[0] + (np.arange(nx) + 0.5) * voxel
    zs = lo[2] + (np.arange(nz) + 0.5) * voxel
    ix = np.clip(((xs - origin[0]) / cell).astype(int), 0, mask.shape[0] - 1)
    iz = np.clip(((zs - origin[1]) / cell).astype(int), 0, mask.shape[1] - 1)
    return mask[np.ix_(ix, iz)]


# ---------------------------------------------------------------- driver


def load_candidate(rec: dict, keep_largest: bool):
    import trimesh

    mesh = trimesh.load(rec["glb"], force="mesh", process=False)
    if keep_largest and len(mesh.faces):
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            mesh = max(parts, key=lambda m: len(m.faces))
            rec["kept_largest_component"] = True
    return mesh


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + prove a walkable collision shell")
    ap.add_argument("job_dir")
    ap.add_argument("--route", choices=["auto", "splat-transform", "voxel"], default="auto")
    ap.add_argument("--voxel-size", type=float, default=0.07, help="scene units")
    ap.add_argument("--report", default=None)
    ap.add_argument("--out-glb", default=None)
    ap.add_argument("--source", default=None, help="input splat .ply")
    ap.add_argument("--st-mode", choices=["exterior", "interior", "both"], default="exterior")
    ap.add_argument("--st-style", default="smooth,faces", help="collision-mesh styles to try")
    ap.add_argument("--fill-size", type=float, default=1.6)
    ap.add_argument("--carve", default=None, metavar="H,R")
    ap.add_argument("--no-carve-variant", action="store_true",
                    help="auto: skip the extra carved splat-transform candidate")
    ap.add_argument("--floor-mode", choices=["plane", "first-solid"], default="plane")
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--seed", default=None, metavar="X,Y,Z", help="Y-up frame")
    ap.add_argument("--player-radius", type=float, default=0.25)
    ap.add_argument("--player-height", type=float, default=1.2)
    ap.add_argument("--grid-res", type=float, default=0.05, help="ray grid cell")
    ap.add_argument("--box-margin", type=float, default=0.25)
    ap.add_argument("--close-radius", type=int, default=2, help="voxel route")
    ap.add_argument("--keep-largest", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json", action="store_true", help="report to stdout too")
    args = ap.parse_args()

    versions = preflight()
    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.is_dir():
        raise SystemExit(f"world_shell: no such job dir {job_dir}")
    source = Path(args.source) if args.source else job_dir / "_scene" / "isolated" / "background.ply"
    if not source.is_file():
        raise SystemExit(f"world_shell: source splat missing: {source}")

    world = job_dir / "_world"
    world.mkdir(parents=True, exist_ok=True)
    out_glb = Path(args.out_glb) if args.out_glb else world / "collision_shell.glb"
    report_path = Path(args.report) if args.report else world / "collision_shell.json"

    started = time.time()
    _log(f"[1/5] source {source.name}")
    pts_scene = load_gaussians(source)
    lo_s, hi_s, box_src = room_box(job_dir, pts_scene, args.box_margin)
    inside = np.all((pts_scene >= lo_s) & (pts_scene <= hi_s), axis=1)
    pts_yup = to_yup(pts_scene[inside])
    _log(f"      {len(pts_scene)} solid gaussians, {int(inside.sum())} inside room box ({box_src})")

    probe = build_probe(pts_yup, args.grid_res, args.player_radius, args.player_height)
    _log(
        f"[2/5] probe {probe.shape[0]}x{probe.shape[1]} cells @ {args.grid_res} "
        f"| footprint {int(probe.footprint.sum())} interior {int(probe.interior.sum())} "
        f"| floor y={probe.floor_level:.3f} top y={probe.top_level:.3f} "
        f"| source capsule-free {probe.capsule_free[probe.interior].mean():.3f}"
    )
    seed = (
        np.array([float(v) for v in args.seed.split(",")]) if args.seed else probe.seed
    )
    corners_yup = to_yup(np.array([lo_s, hi_s, [lo_s[0], hi_s[1], lo_s[2]],
                                   [hi_s[0], lo_s[1], hi_s[2]]]))
    box_yup = (corners_yup.min(axis=0), corners_yup.max(axis=0))

    work = Path(tempfile.mkdtemp(prefix=".world-shell-", dir=str(world)))
    candidates: list[dict] = []
    carve = tuple(float(v) for v in args.carve.split(",")) if args.carve else None
    try:
        _log("[3/5] building candidates")
        if args.route in ("auto", "splat-transform"):
            if versions.get("splat_transform"):
                modes = ["exterior", "interior"] if args.st_mode == "both" else [args.st_mode]
                plans = [
                    (mode, style, carve)
                    for mode in modes
                    for style in [s for s in args.st_style.split(",") if s]
                ]
                if args.route == "auto" and not args.no_carve_variant and not carve:
                    # --voxel-carve flood-fills a capsule out of the filled grid
                    # from the seed: the only lever splat-transform has for
                    # giving back the under-furniture volume floor-fill buries.
                    plans.append(
                        (modes[0], "smooth", (args.player_height, args.player_radius))
                    )
                for mode, style, cv in plans:
                    rec = route_splat_transform(
                        source, work, (lo_s, hi_s), seed, args.voxel_size, mode,
                        style, args.fill_size, cv, not args.no_cluster, args.timeout,
                    )
                    _log(f"      {rec['route']}: rc={rec['returncode']} {rec.get('seconds')}s"
                         f" {rec.get('error','')[:120]}")
                    candidates.append(rec)
            else:
                candidates.append({"route": "splat-transform", "returncode": 127,
                                   "error": f"binary not found at {SPLAT_TRANSFORM}"})
        if args.route in ("auto", "voxel"):
            try:
                rec = route_voxel(job_dir, pts_yup, probe, box_yup, args.voxel_size,
                                  args.close_radius, work, args.floor_mode)
            except Exception as exc:  # noqa: BLE001 - a dead route must not kill the run
                rec = {"route": "voxel", "returncode": 1, "error": repr(exc)}
            _log(f"      voxel: rc={rec['returncode']} {rec.get('seconds')}s {rec.get('error','')[:200]}")
            candidates.append(rec)

        _log("[4/5] gates")
        for rec in candidates:
            if rec.get("returncode") != 0 or not rec.get("glb"):
                continue
            try:
                mesh = load_candidate(rec, args.keep_largest)
                rec["gates"] = evaluate(mesh, probe, args.player_radius, args.player_height)
                rec["_mesh"] = mesh
            except Exception as exc:  # noqa: BLE001
                rec["gate_error"] = repr(exc)
            g = rec.get("gates", {})
            _log(
                f"      {rec['route']}: tris={g.get('triangles')} comps={g.get('components')}"
                f" lcc={g.get('largest_component_frac')} watertight={g.get('watertight')}"
                f" floor={g.get('floor_continuity')} hole={g.get('max_hole_span')}"
                f" walkable={g.get('walkable_frac')} passed={g.get('gates_passed')}/4"
            )

        scored = [c for c in candidates if c.get("gates")]
        winner = max(scored, key=rank_key) if scored else None

        report = {
            "v": 1,
            "generated_at": _now(),
            "job_id": job_dir.name,
            "job_dir": str(job_dir),
            "source_splat": str(source),
            "units": "scene units (UNCALIBRATED - meta.json meters_per_unit is null)",
            "frame": {
                "capture": "Z-up",
                "output_glb": "Y-up (rotation_matrix(-pi/2,[1,0,0]), repo glTF convention)",
                "splat_transform_rotate": ST_ROTATE,
                "check": frame_check(pts_scene),
            },
            "room_box_scene": {"min": [round(float(v), 4) for v in lo_s],
                               "max": [round(float(v), 4) for v in hi_s],
                               "derived_from": box_src},
            "probe": {
                "grid_res": args.grid_res,
                "grid": list(probe.shape),
                "footprint_cells": int(probe.footprint.sum()),
                "interior_cells": int(probe.interior.sum()),
                "floor_level_y": round(probe.floor_level, 4),
                "top_level_y": round(probe.top_level, 4),
                "source_capsule_free_frac": round(
                    float(probe.capsule_free[probe.interior].mean()), 4
                ),
                "interior_defn": "source-cloud footprint, hole-filled, largest "
                                 "component, eroded by the player radius",
                "note": "source_capsule_free_frac is a reference ceiling for "
                        "walkable_frac, not a gate; it counts a column blocked "
                        "if ANY gaussian centre sits in the capsule band, so it "
                        "moves with --grid-res. The four gates do not.",
            },
            "params": {
                "route": args.route, "voxel_size": args.voxel_size,
                "st_mode": args.st_mode, "fill_size": args.fill_size,
                "carve": carve, "cluster": not args.no_cluster,
                "seed_yup": [round(float(v), 4) for v in seed],
                "player_radius": args.player_radius,
                "player_height": args.player_height,
                "close_radius": args.close_radius,
            },
            "environment": versions,
            "candidates": [
                {k: v for k, v in c.items() if not k.startswith("_")} for c in candidates
            ],
        }

        if winner is None:
            report["verdict"] = "FAILED"
            report["reason"] = "no candidate produced a mesh the gates could read"
            report_path.write_text(json.dumps(report, indent=2))
            _log(f"[5/5] FAILED - no usable candidate. report: {report_path}")
            return 2

        _log(f"[5/5] winner {winner['route']}")
        mesh = winner.pop("_mesh")
        # Floating collision crumbs (voxelization islands under 1% of faces)
        # snag capsules and fail shell_connectivity on every rebuild — drop
        # them here, durably, instead of as a one-off data fix (2026-07-28:
        # 26 crumbs, 5,464 tris on the bonsai). The largest island is always
        # kept, so a degenerate threshold can never empty the solid.
        import trimesh.graph as _tg

        comps = _tg.connected_components(mesh.face_adjacency,
                                         nodes=np.arange(len(mesh.faces)))
        if len(comps) > 1:
            sizes = [(len(c), i) for i, c in enumerate(comps)]
            largest = max(sizes)[1]
            keep = np.zeros(len(mesh.faces), bool)
            dropped = 0
            for size, index in sizes:
                if index == largest or size >= 0.01 * len(mesh.faces):
                    keep[comps[index]] = True
                else:
                    dropped += 1
            if dropped:
                mesh.update_faces(keep)
                mesh.remove_unreferenced_vertices()
                _log(f"      dropped {dropped} collision crumb component(s)")
                # The gates and the navmesh were measured BEFORE the drop —
                # exporting a different mesh than the report describes would
                # make every number a lie (review finding). Re-grade the
                # dropped mesh and let THOSE numbers ship.
                winner["gates"] = evaluate(mesh, probe, args.player_radius,
                                           args.player_height)
        mesh.export(str(out_glb))
        # MANDATORY readback: the export is only real if it reloads identically.
        import trimesh

        back = trimesh.load(str(out_glb), force="mesh", process=False)
        readback = {
            "path": str(out_glb),
            "bytes": out_glb.stat().st_size,
            "triangles_written": int(len(mesh.faces)),
            "triangles_read_back": int(len(back.faces)),
            "vertices_read_back": int(len(back.vertices)),
            "bounds": [[round(float(v), 4) for v in b] for b in back.bounds],
        }
        readback["match"] = readback["triangles_written"] == readback["triangles_read_back"]
        # The navmesh: the winner's own walkable cells, written beside the
        # collision solid. Actors will path over exactly what the gates
        # graded as standable — no second opinion about where feet may go.
        nav = None
        for cand in scored:
            popped = (cand.get("gates") or {}).pop("_walkable", None)
            if cand is winner:
                nav = popped
        if nav:
            nav_path = out_glb.parent / "navmesh.json"
            nav_path.write_text(json.dumps({"v": 1, "frame": "y-up", **nav}))
            report["navmesh"] = {"path": nav_path.name, "shape": nav["shape"],
                                 "cell": nav["cell"],
                                 "walkable_cells": sum(r.count("1") for r in nav["rows"])}

        g = winner["gates"]
        report.update(
            {
                "route_used": winner["route"],
                "gates": {
                    "components": g["components"],
                    "largest_component_frac": g["largest_component_frac"],
                    "watertight": g["watertight"],
                    "floor_continuity": g["floor_continuity"],
                    "max_hole_span": g["max_hole_span"],
                },
                "gates_detail": g,
                "artifact": readback,
                "seconds": round(time.time() - started, 2),
            }
        )
        if not readback["match"]:
            report["verdict"] = "FAILED"
            report["reason"] = "GLB readback triangle-count mismatch"
        elif g["gates_passed"] == 4:
            report["verdict"] = "PASS"
        elif g["walkable"]:
            report["verdict"] = "WALKABLE_NOT_WATERTIGHT" if not g["watertight"] else "WALKABLE"
        else:
            report["verdict"] = "NOT_WALKABLE"
        report_path.write_text(json.dumps(report, indent=2))

        _log(
            f"      {report['verdict']} | components={g['components']}"
            f" (largest {g['largest_component_frac']}) watertight={g['watertight']}"
            f" floor_continuity={g['floor_continuity']} max_hole_span={g['max_hole_span']}"
        )
        _log(f"      glb {out_glb} {readback['bytes']}B "
             f"{readback['triangles_read_back']} tris (readback match={readback['match']})")
        _log(f"      report {report_path}")
        if args.json:
            print(json.dumps(report, indent=2))
        return 0 if readback["match"] else 3
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
