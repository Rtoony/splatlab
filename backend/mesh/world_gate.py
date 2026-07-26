#!/usr/bin/env python3
"""Acceptance gates for a solidified world: is this capture actually walkable?

The pipeline can and does emit confident-looking garbage. The Bonsai shell that
motivated this module reports `built: true`, 59,999 faces and a baked texture in
its own build report -- and is in fact a colander of hundreds of disconnected
islands with a third of its floor missing. A build report is a record of what ran,
not evidence that the result is usable. This module produces the evidence.

Every number here is computed from the exported geometry on disk (the GLBs an
engine would actually load) and the atlas PNGs beside them. Nothing is read from
shell.json / <slug>.json except metadata that is not a quality claim (units,
meters_per_unit, up_axis, which elements were supposed to exist). Where a build
report does state a quality number, the gate recomputes it independently and
records both.

Gates
  1. shell_connectivity  - a walkable environment is one dominant surface.
  2. floor_continuity    - downward raycast grid: do you fall through the floor?
  3. prop_integrity      - per prop: components, degenerate faces, UVs, texture.
  4. texture_coverage    - fraction of the atlas that actually got painted.
  5. scale_sanity        - only when calibrated; SKIPPED loudly when not.

Exit status is 0 only when every gate that ran passed. --strict additionally
fails on any gate that could not be measured.

Usage: world_gate.py <job_dir> [--report PATH] [--strict]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy import ndimage

# --------------------------------------------------------------------------
# Thresholds. Each one is a stated engineering position, not a tuned constant.
# --------------------------------------------------------------------------

# A solidified environment shell is one surface. Decimation slivers and a
# genuinely detached fixture or two are tolerable; hundreds of islands mean the
# solidifier never closed the surface and the "room" is a cloud of shards.
SHELL_MAX_COMPONENTS = 20
# ...and the islands must be trivia. 95% of triangles in one piece means at most
# a twentieth of the surface area is floating debris.
SHELL_MIN_LARGEST_COMPONENT_FRAC = 0.95

# Floor: every interior column must have *something* under it, or the player
# falls to infinity. 10% slack absorbs furniture shadows and capture edges.
FLOOR_MIN_COVERAGE = 0.90
# A void is only a bug if it is contiguous and big enough to fall into. 3% of the
# floor plan is larger than any plausible single unobserved furniture shadow.
FLOOR_MAX_GAP_FRAC = 0.03
# Grid is sized so the longer horizontal axis gets this many cells. Recorded in
# the report along with the resulting cell size so the number is reproducible.
FLOOR_GRID_CELLS = 128
# Interior footprint is eroded by this many cells to keep the wall ring itself
# out of the floor test.
FLOOR_INTERIOR_ERODE = 2
# Diagnostic only (not gated): a column counts as "ground supported" when its
# lowest surface sits within this fraction of scene height of the median ground.
FLOOR_GROUND_BAND_FRAC = 0.15

# A prop may legitimately reconstruct in a few pieces (a bicycle's wheels need
# not weld to its frame), but it must be recognisably one object.
PROP_MIN_LARGEST_COMPONENT_FRAC = 0.80
# Zero-area triangles are engine poison (NaN normals, degenerate collision).
PROP_MAX_DEGENERATE_FRAC = 0.02
# Relative linear tolerance for calling a triangle collapsed, scaled by the
# model diagonal so it means the same thing on a bottle and on a room.
DEGENERATE_REL_EPS = 1e-6

# The atlas is dilated by the baker before it is written, so a healthy atlas is
# almost entirely non-black. Below half means over half the atlas was touched by
# neither rasterisation nor dilation: the bake failed.
TEXTURE_MIN_COVERAGE = 0.50

# Physically absurd sizes. This gate catches a broken scale factor (the 40 m
# chair) or a segmentation that swallowed the scene, not furniture taste. An
# element is a labelled noun - a vase, a bicycle, a sofa, at the very top end a
# car at ~5 m. Past that it is a chunk of scene wearing an object's name. Below
# a centimetre it is a collapsed element, not a small one.
SCALE_PROP_MIN_M = 0.01
SCALE_PROP_MAX_M = 5.0
SCALE_SHELL_MIN_M = 0.5
SCALE_SHELL_MAX_M = 500.0


class GateError(RuntimeError):
    """Something the gate needs is absent or malformed. Never swallowed."""


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _scene_to_mesh(path: Path) -> tuple[trimesh.Trimesh, trimesh.Scene]:
    """Load a GLB and flatten it WITH node transforms applied.

    trimesh.util.concatenate(scene.geometry.values()) silently drops node
    transforms; a prop placed by a node matrix would then be measured at the
    origin at the wrong scale. Use the scene's own flattening.
    """
    scene = trimesh.load(path, process=False, force="scene")
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise GateError(f"{path.name}: no geometry in file")
    if hasattr(scene, "to_geometry"):
        mesh = scene.to_geometry()
    else:  # trimesh < 4.6
        mesh = scene.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise GateError(f"{path.name}: flattened to {type(mesh).__name__}, not a mesh")
    if len(mesh.faces) == 0:
        raise GateError(f"{path.name}: zero faces")
    return mesh, scene


def _weld(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge vertices by POSITION only.

    Load-bearing. A GLB stores a separate vertex per UV chart corner, so an
    as-exported mesh looks shattered no matter how sound it is: the Bonsai shell
    counts 11,026 components unwelded and 645 welded. Counting components on the
    raw index buffer measures the unwrapper, not the geometry.
    """
    out = mesh.copy()
    out.merge_vertices(merge_tex=True, merge_norm=True)
    return out


def _components(mesh: trimesh.Trimesh) -> tuple[int, float, list[int]]:
    """Edge-connected face components of an already-welded mesh."""
    cc = trimesh.graph.connected_components(
        mesh.face_adjacency, min_len=1, nodes=np.arange(len(mesh.faces))
    )
    sizes = sorted((len(c) for c in cc), reverse=True)
    total = len(mesh.faces)
    return len(sizes), (sizes[0] / total if sizes else 0.0), sizes[:5]


def _components_open3d(mesh: trimesh.Trimesh) -> tuple[int, float]:
    """Independent second opinion from a different library.

    Two implementations agreeing on a component count is the difference between
    a measurement and an assertion. Recorded as a cross-check in the report.
    """
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    om.remove_duplicated_vertices()
    om.remove_degenerate_triangles()
    _, counts, _ = om.cluster_connected_triangles()
    counts = np.asarray(counts)
    if counts.size == 0:
        return 0, 0.0
    return int(counts.size), float(counts.max() / counts.sum())


def _degenerate_faces(mesh: trimesh.Trimesh) -> tuple[int, float]:
    """Count collapsed triangles by area, on the mesh as exported.

    Computed directly rather than via a library default so the tolerance is
    explicit and scale-relative: a triangle is collapsed when its area is below
    that of an equilateral triangle whose side is 1e-6 of the model diagonal.
    """
    tri = mesh.triangles
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    diag = float(np.linalg.norm(mesh.extents))
    eps_len = DEGENERATE_REL_EPS * max(diag, 1e-12)
    n = int((area <= 0.5 * eps_len * eps_len).sum())
    return n, (n / len(mesh.faces) if len(mesh.faces) else 0.0)


def _uv_and_texture(scene: trimesh.Scene) -> tuple[bool, bool, list]:
    """Per-geometry UV + baseColorTexture presence, read back off the GLB."""
    have_uv, have_tex, sizes = [], [], []
    for geom in scene.geometry.values():
        vis = getattr(geom, "visual", None)
        uv = getattr(vis, "uv", None)
        have_uv.append(uv is not None and len(uv) == len(geom.vertices) and len(uv) > 0)
        mat = getattr(vis, "material", None)
        img = None
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None)
            if img is None:
                img = getattr(mat, "image", None)  # SimpleMaterial
        have_tex.append(img is not None)
        if img is not None:
            sizes.append(list(getattr(img, "size", ()) or ()))
    return (bool(have_uv) and all(have_uv)), (bool(have_tex) and all(have_tex)), sizes


def _axis_indices(up_axis: str) -> tuple[int, tuple[int, int]]:
    idx = {"X": 0, "Y": 1, "Z": 2}.get(str(up_axis).upper())
    if idx is None:
        raise GateError(f"unsupported up_axis {up_axis!r} (expected X, Y or Z)")
    return idx, tuple(a for a in (0, 1, 2) if a != idx)  # type: ignore[return-value]


def _check(value, threshold, ok: bool, comparison: str, note: str | None = None) -> dict:
    entry = {"value": value, "threshold": threshold, "comparison": comparison,
             "pass": bool(ok)}
    if note:
        entry["note"] = note
    return entry


def _gate(checks: dict, **extra) -> dict:
    hard = [c for c in checks.values() if c.get("threshold") is not None]
    return {"pass": all(c["pass"] for c in hard) if hard else None,
            "skipped": False, "checks": checks, **extra}


def _skip(reason: str, **extra) -> dict:
    return {"pass": None, "skipped": True, "reason": reason, "checks": {}, **extra}


# --------------------------------------------------------------------------
# Gate 1 - shell connectivity
# --------------------------------------------------------------------------

def gate_shell_connectivity(shell: trimesh.Trimesh) -> dict:
    raw_n, raw_frac, _ = _components(shell)
    welded = _weld(shell)
    n, frac, top = _components(welded)
    o3d_n, o3d_frac = _components_open3d(welded)

    cross = {"open3d_components": o3d_n,
             "open3d_largest_component_frac": round(o3d_frac, 4)}
    if o3d_n != n or abs(o3d_frac - frac) > 0.01:
        cross["disagreement"] = (
            "trimesh (edge adjacency) and open3d (cluster_connected_triangles) "
            "disagree; treat both numbers as suspect until reconciled")

    return _gate(
        {
            "components": _check(
                n, SHELL_MAX_COMPONENTS, n <= SHELL_MAX_COMPONENTS, "<=",
                "edge-connected face components after welding vertices by position"),
            "largest_component_frac": _check(
                round(frac, 4), SHELL_MIN_LARGEST_COMPONENT_FRAC,
                frac >= SHELL_MIN_LARGEST_COMPONENT_FRAC, ">=",
                "fraction of triangles in the single biggest connected piece"),
        },
        faces=int(len(shell.faces)),
        vertices_as_exported=int(len(shell.vertices)),
        vertices_welded=int(len(welded.vertices)),
        components_unwelded=raw_n,
        largest_component_frac_unwelded=round(raw_frac, 4),
        top_component_faces=top,
        cross_check=cross,
        method=("vertices welded by exact position before counting; the GLB "
                "index buffer splits vertices at UV seams, so an unwelded count "
                "measures the unwrapper rather than the geometry"),
    )


# --------------------------------------------------------------------------
# Gate 2 - floor continuity
# --------------------------------------------------------------------------

def gate_floor_continuity(shell: trimesh.Trimesh, up_axis: str, units: str) -> dict:
    up, hax = _axis_indices(up_axis)
    welded = _weld(shell)
    lo, hi = welded.bounds
    span = hi - lo
    if span[hax[0]] <= 0 or span[hax[1]] <= 0 or span[up] <= 0:
        raise GateError("shell has zero extent on an axis; cannot lay out a floor grid")

    cell = max(span[hax[0]], span[hax[1]]) / FLOOR_GRID_CELLS
    na = max(int(np.ceil(span[hax[0]] / cell)), 2)
    nb = max(int(np.ceil(span[hax[1]] / cell)), 2)
    ca = lo[hax[0]] + (np.arange(na) + 0.5) * cell
    cb = lo[hax[1]] + (np.arange(nb) + 0.5) * cell

    A, B = np.meshgrid(ca, cb, indexing="ij")
    pa, pb = A.ravel(), B.ravel()

    rs = o3d.t.geometry.RaycastingScene()
    rs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(welded.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(welded.faces, dtype=np.int32)))))

    pad = 0.01 * span[up] + 1e-3

    def _cast(origin_level: float, direction: float) -> np.ndarray:
        rays = np.zeros((len(pa), 6), dtype=np.float32)
        rays[:, hax[0]] = pa
        rays[:, hax[1]] = pb
        rays[:, up] = origin_level
        rays[:, 3 + up] = direction
        return rs.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()

    # Up from beneath the model: the first hit is the LOWEST surface in that
    # column -- the floor, if there is one -- in a single cast. The downward cast
    # is the independent cross-check: the two must agree on which columns are
    # empty, since an unsigned raycast cannot care which side it approaches from.
    t_down = _cast(float(hi[up] + pad), -1.0)
    t_up = _cast(float(lo[up] - pad), +1.0)
    hit_down = np.isfinite(t_down)
    hit_up = np.isfinite(t_up)
    disagree = int((hit_down != hit_up).sum())
    hit_grid = (hit_down | hit_up).reshape(na, nb)

    # Interior footprint from the columns that actually contain surface, NOT from
    # vertex occupancy: a mesh whose vertex spacing is coarser than a grid cell
    # marks a checkerboard lattice, which erodes to nothing and would abort the
    # gate on a perfectly sound but coarse shell. Ray coverage is independent of
    # tessellation density. Holes are then filled back in on purpose -- a missing
    # patch of floor is an enclosed void, and letting it shrink the footprint
    # would excuse the very defect this gate exists to find. Erosion drops the
    # boundary/wall ring. A void that breaches the outline is not enclosed and so
    # falls outside the footprint: this gate under-reports rather than over.
    footprint = ndimage.binary_fill_holes(hit_grid)
    interior = (ndimage.binary_erosion(footprint, iterations=FLOOR_INTERIOR_ERODE)
                if FLOOR_INTERIOR_ERODE else footprint)
    n_interior = int(interior.sum())
    if n_interior < 16:
        raise GateError(
            f"interior footprint collapsed to {n_interior} cells; shell is too "
            "sparse or too thin to test for a floor")

    any_geometry = float(hit_grid[interior].mean())

    # "Is there anything in this column" is NOT the floor test. In a room with a
    # ceiling, a ray fired down through a hole in the floor still hits the
    # ceiling and reports a hit -- a false pass on precisely the defect this gate
    # exists to catch (demonstrated on a synthetic ceilinged room with a disc of
    # floor removed). What matters is the LOWEST surface in the column: fire up
    # from beneath the model and the first thing hit is the floor, if there is
    # one. Over a hole, the first hit is the ceiling, far above the ground plane.
    lowest_all = np.where(hit_up, (lo[up] - pad) + t_up, np.nan).reshape(na, nb)
    lowest = lowest_all[interior]
    ground_level = float(np.nanmedian(lowest)) if np.isfinite(lowest).any() else None
    if ground_level is None:
        raise GateError("no interior column contains any surface; shell is empty")

    band = FLOOR_GROUND_BAND_FRAC * float(span[up])
    supported_grid = (np.nan_to_num(lowest_all, nan=np.inf) <= ground_level + band)
    supported_grid &= hit_up.reshape(na, nb)
    coverage = float(supported_grid[interior].mean())

    # Largest contiguous void, 4-connected (diagonal touching does not merge
    # regions - the conservative choice, it under-reports rather than over).
    holes = interior & ~supported_grid
    labels, n_regions = ndimage.label(holes)
    if n_regions:
        sizes = np.bincount(labels.ravel())[1:]
        gap_cells = int(sizes.max())
    else:
        gap_cells = 0
    gap_frac = gap_cells / n_interior

    return _gate(
        {
            "coverage": _check(
                round(coverage, 4), FLOOR_MIN_COVERAGE, coverage >= FLOOR_MIN_COVERAGE,
                ">=", "fraction of interior columns whose lowest surface sits within "
                      "the ground band, i.e. that have a floor under them"),
            "largest_gap_frac": _check(
                round(gap_frac, 4), FLOOR_MAX_GAP_FRAC, gap_frac <= FLOOR_MAX_GAP_FRAC,
                "<=", "largest 4-connected void as a fraction of interior footprint"),
        },
        grid={"cells": [na, nb], "cell_size": round(float(cell), 5),
              "units": units, "footprint_cells": int(footprint.sum()),
              "interior_cells": n_interior,
              "interior_frac_of_bbox": round(n_interior / (na * nb), 4)},
        rays_cast=int(2 * len(pa)),
        up_axis=up_axis,
        unsupported_cells=int(n_interior - int(supported_grid[interior].sum())),
        void_regions=int(n_regions),
        largest_gap_cells=gap_cells,
        largest_gap_span=round(float(np.sqrt(gap_cells) * cell), 4),
        ground_level=round(ground_level, 4),
        ground_band=round(float(band), 4),
        any_geometry_frac=round(any_geometry, 4),
        any_geometry_note=("diagnostic only: interior columns containing any surface "
                           "at all. Always >= coverage, and blind to a floor hole "
                           "under a ceiling, which is why it is not the gated value."),
        raycast_consistency={"down_up_disagreements": disagree,
                             "note": "up- and down-cast must agree on which columns "
                                     "are empty; a nonzero count means the raycast "
                                     "itself is unreliable"},
        method=("shell only - props are not allowed to patch a hole in the floor"),
    )


# --------------------------------------------------------------------------
# Gate 3 - prop integrity
# --------------------------------------------------------------------------

def gate_prop_integrity(elements: list[dict], elements_dir: Path) -> dict:
    per_prop, failures = [], []
    for el in elements:
        slug = el.get("slug", "?")
        if not el.get("built"):
            failures.append(slug)
            per_prop.append({"slug": slug, "pass": False,
                             "reason": "element was never built (build reported failure)"})
            continue
        glb = elements_dir / (el.get("glb") or f"{slug}.glb")
        if not glb.is_file():
            failures.append(slug)
            per_prop.append({"slug": slug, "pass": False,
                             "reason": f"build reported built=true but {glb.name} is missing"})
            continue

        mesh, scene = _scene_to_mesh(glb)
        welded = _weld(mesh)
        n, frac, top = _components(welded)
        deg_n, deg_frac = _degenerate_faces(mesh)
        has_uv, has_tex, tex_sizes = _uv_and_texture(scene)

        checks = {
            "largest_component_frac": _check(
                round(frac, 4), PROP_MIN_LARGEST_COMPONENT_FRAC,
                frac >= PROP_MIN_LARGEST_COMPONENT_FRAC, ">="),
            "degenerate_face_frac": _check(
                round(deg_frac, 5), PROP_MAX_DEGENERATE_FRAC,
                deg_frac <= PROP_MAX_DEGENERATE_FRAC, "<="),
            "has_uv": _check(has_uv, True, has_uv, "=="),
            "has_base_color_texture": _check(has_tex, True, has_tex, "=="),
            # Reported because it was asked for and because it is real
            # information, but NOT gated: every prop here is an open shell, which
            # is what photogrammetry of a scene you cannot walk around produces.
            # Gating on it would fail every good prop the pipeline can make.
            "watertight": _check(
                bool(welded.is_watertight), None, bool(welded.is_watertight), "advisory",
                "measured, not gated - reconstructed props are open shells by "
                "construction, so gating on this would fail every good prop"),
        }
        ok = all(c["pass"] for c in checks.values() if c["threshold"] is not None)
        if not ok:
            failures.append(slug)
        per_prop.append({
            "slug": slug, "glb": glb.name, "pass": ok, "checks": checks,
            "faces": int(len(mesh.faces)),
            "vertices_as_exported": int(len(mesh.vertices)),
            "vertices_welded": int(len(welded.vertices)),
            "components": n, "top_component_faces": top,
            "degenerate_faces": deg_n,
            "winding_consistent": bool(welded.is_winding_consistent),
            "texture_sizes": tex_sizes,
            "extent": [round(float(x), 4) for x in mesh.extents],
        })

    if not per_prop:
        return _skip("world declares no elements")

    n_pass = sum(1 for p in per_prop if p["pass"])
    return _gate(
        {"props_passing": _check(n_pass, len(per_prop), n_pass == len(per_prop), "==",
                                 "every declared element must pass its own checks")},
        props=len(per_prop), failing=failures, per_prop=per_prop,
        method="each GLB re-opened from disk; build reports are not consulted",
    )


# --------------------------------------------------------------------------
# Gate 4 - texture coverage
# --------------------------------------------------------------------------

def _atlas_coverage(png: Path) -> tuple[float, list[int]]:
    """Fraction of texels the baker actually painted.

    Background is exactly black (or alpha 0): object_texture.bake_texture starts
    from np.zeros and dilate_atlas only ever spreads baked colour, so an
    all-zero texel was touched by neither. This measures the atlas AS SHIPPED,
    i.e. after dilation. The side-by-side build-report value is the pre-dilation
    rasterised fraction (`texture.coverage_rasterized`; in reports written
    before 2026-07-26 the same quantity was misfiled under `texture.coverage`).
    Caveat: a genuinely pure-black painted region reads as background.
    """
    img = Image.open(png)
    arr = np.asarray(img)
    if arr.ndim == 2:
        painted = arr > 0
    elif arr.shape[2] == 4:
        painted = (arr[:, :, 3] > 0) & (arr[:, :, :3].max(axis=2) > 0)
    else:
        painted = arr[:, :, :3].max(axis=2) > 0
    return float(painted.mean()), [int(arr.shape[1]), int(arr.shape[0])]


def gate_texture_coverage(entries: list[tuple[str, Path, float | None]]) -> dict:
    per_atlas, failures = [], []
    for name, png, reported in entries:
        if not png.is_file():
            failures.append(name)
            per_atlas.append({"element": name, "pass": False,
                              "reason": f"atlas {png.name} missing"})
            continue
        cov, size = _atlas_coverage(png)
        ok = cov >= TEXTURE_MIN_COVERAGE
        if not ok:
            failures.append(name)
        per_atlas.append({
            "element": name, "atlas": png.name, "pass": ok,
            "check": _check(round(cov, 4), TEXTURE_MIN_COVERAGE, ok, ">=",
                            "non-background texels in the shipped (dilated) atlas"),
            "atlas_size": size,
            "build_report_coverage": reported,
            "build_report_note": ("pre-dilation rasterised mask; a different and "
                                  "always smaller quantity than the measured value"),
        })

    if not per_atlas:
        return _skip("no atlases declared")

    n_pass = sum(1 for a in per_atlas if a["pass"])
    return _gate(
        {"atlases_passing": _check(n_pass, len(per_atlas), n_pass == len(per_atlas),
                                   "==", "every atlas must clear the coverage floor")},
        atlases=len(per_atlas), failing=failures, per_atlas=per_atlas,
    )


# --------------------------------------------------------------------------
# Gate 5 - scale sanity
# --------------------------------------------------------------------------

def gate_scale_sanity(mpu, measured: list[tuple[str, str, list[float]]]) -> dict:
    if mpu is None:
        return _skip(
            "capture is UNCALIBRATED (meters_per_unit is null): geometry is in "
            "arbitrary scene units, so no real-world size can be checked. This "
            "gate invents nothing and reports no size verdict.",
            meters_per_unit=None)
    try:
        mpu = float(mpu)
    except (TypeError, ValueError):
        raise GateError(f"meters_per_unit is present but not a number: {mpu!r}")
    if not np.isfinite(mpu) or mpu <= 0:
        raise GateError(f"meters_per_unit is non-physical: {mpu}")

    per_item, failures = [], []
    for name, role, extent in measured:
        longest = float(max(extent))
        lo_m, hi_m = ((SCALE_SHELL_MIN_M, SCALE_SHELL_MAX_M) if role == "shell"
                      else (SCALE_PROP_MIN_M, SCALE_PROP_MAX_M))
        ok = lo_m <= longest <= hi_m
        if not ok:
            failures.append(name)
        per_item.append({
            "element": name, "role": role, "pass": ok,
            "longest_dimension_m": round(longest, 4),
            "extent_m": [round(float(x), 4) for x in extent],
            "plausible_range_m": [lo_m, hi_m],
        })

    n_pass = sum(1 for i in per_item if i["pass"])
    return _gate(
        {"elements_plausible": _check(n_pass, len(per_item), n_pass == len(per_item),
                                      "==", "no element of physically absurd size")},
        meters_per_unit=mpu, elements=len(per_item), failing=failures,
        per_element=per_item,
        method=("GLB vertices are exported already multiplied by meters_per_unit "
                "(object_texture.py), so measured extents are metres"),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def resolve_world_dir(job_dir: Path) -> Path:
    if (job_dir / "world.json").is_file():
        return job_dir
    if (job_dir / "_world" / "world.json").is_file():
        return job_dir / "_world"
    raise GateError(f"no world.json under {job_dir} or {job_dir}/_world")


def run(world: Path, strict: bool) -> dict:
    world_json = world / "world.json"
    try:
        wj = json.loads(world_json.read_text())
    except Exception as exc:
        raise GateError(f"unreadable {world_json}: {exc}") from exc

    up_axis = wj.get("up_axis", "Y")
    units = wj.get("units", "unknown")
    mpu = wj.get("meters_per_unit")
    elements = wj.get("elements") or []
    elements_dir = world / "elements"

    gates: dict[str, dict] = {}
    measured_sizes: list[tuple[str, str, list[float]]] = []
    atlases: list[tuple[str, Path, float | None]] = []

    # ---- shell ----
    shell_meta = wj.get("shell")
    shell_mesh = None
    if not shell_meta or not shell_meta.get("built"):
        reason = ("world declares no shell (shell_built is false): there is no "
                  "environment surface to walk on or fall through")
        gates["shell_connectivity"] = _skip(reason)
        gates["floor_continuity"] = _skip(reason)
    else:
        shell_glb = world / (shell_meta.get("glb") or "shell.glb")
        if not shell_glb.is_file():
            raise GateError(
                f"world.json reports shell built=true but {shell_glb} does not exist")
        shell_mesh, _shell_scene = _scene_to_mesh(shell_glb)
        gates["shell_connectivity"] = gate_shell_connectivity(shell_mesh)
        gates["floor_continuity"] = gate_floor_continuity(shell_mesh, up_axis, units)
        measured_sizes.append(("shell", "shell", list(shell_mesh.extents)))
        shell_tex = shell_meta.get("texture")
        shell_tex = shell_tex if isinstance(shell_tex, dict) else {}
        atlases.append(("shell", world / "shell_atlas.png",
                        shell_tex.get("coverage_rasterized",
                                      shell_tex.get("coverage"))))

    # ---- props ----
    gates["prop_integrity"] = gate_prop_integrity(elements, elements_dir)
    for el in elements:
        slug = el.get("slug", "?")
        glb = elements_dir / (el.get("glb") or f"{slug}.glb")
        if glb.is_file():
            mesh, _ = _scene_to_mesh(glb)
            measured_sizes.append((slug, el.get("role", "prop"), list(mesh.extents)))
            atlases.append((slug, elements_dir / f"{slug}_atlas.png", None))

    # Sidecar element JSONs carry the build-time coverage; pulled only for the
    # side-by-side comparison, never as the gated value.
    fixed: list[tuple[str, Path, float | None]] = []
    for name, png, reported in atlases:
        if reported is None:
            sj = (world / "shell.json") if name == "shell" else (elements_dir / f"{name}.json")
            if sj.is_file():
                try:
                    tex = json.loads(sj.read_text()).get("texture")
                    if isinstance(tex, dict):
                        reported = tex.get("coverage_rasterized",
                                           tex.get("coverage"))
                except Exception:  # noqa: BLE001 - comparison value only
                    reported = None
        fixed.append((name, png, reported))

    gates["texture_coverage"] = gate_texture_coverage(fixed)
    gates["scale_sanity"] = gate_scale_sanity(mpu, measured_sizes)

    failed = [k for k, g in gates.items() if g["pass"] is False]
    skipped = [k for k, g in gates.items() if g.get("skipped")]
    ok = not failed and (not skipped if strict else True)

    return {
        "v": 1,
        "job_id": wj.get("job_id"),
        "world_dir": str(world),
        "generated": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "units": units,
        "meters_per_unit": mpu,
        "up_axis": up_axis,
        "strict": strict,
        "thresholds": {
            "shell_max_components": SHELL_MAX_COMPONENTS,
            "shell_min_largest_component_frac": SHELL_MIN_LARGEST_COMPONENT_FRAC,
            "floor_min_coverage": FLOOR_MIN_COVERAGE,
            "floor_max_gap_frac": FLOOR_MAX_GAP_FRAC,
            "floor_grid_cells": FLOOR_GRID_CELLS,
            "prop_min_largest_component_frac": PROP_MIN_LARGEST_COMPONENT_FRAC,
            "prop_max_degenerate_frac": PROP_MAX_DEGENERATE_FRAC,
            "texture_min_coverage": TEXTURE_MIN_COVERAGE,
            "scale_prop_range_m": [SCALE_PROP_MIN_M, SCALE_PROP_MAX_M],
            "scale_shell_range_m": [SCALE_SHELL_MIN_M, SCALE_SHELL_MAX_M],
        },
        "gates": gates,
        "verdict": {
            "pass": ok,
            "failed_gates": failed,
            "skipped_gates": skipped,
            "summary": ("all gates passed" if ok and not skipped else
                        "all measurable gates passed" if ok else
                        f"FAILED: {', '.join(failed)}" if failed else
                        f"strict mode: unmeasurable gates {', '.join(skipped)}"),
        },
    }


def _print_summary(rep: dict) -> None:
    w = print
    w("=" * 72)
    w(f"WORLD GATE  job={rep['job_id']}  units={rep['units']}"
      f"  mpu={rep['meters_per_unit']}")
    w(f"  {rep['world_dir']}")
    w("=" * 72)

    for name, g in rep["gates"].items():
        if g.get("skipped"):
            w(f"\n[SKIP] {name}\n       {g['reason']}")
            continue
        w(f"\n[{'PASS' if g['pass'] else 'FAIL'}] {name}")
        for cname, c in g["checks"].items():
            if c["threshold"] is None:
                w(f"       ---- {cname:<28} = {c['value']}  (advisory)")
                continue
            w(f"       {'ok  ' if c['pass'] else 'FAIL'} {cname:<28} "
              f"= {c['value']}   (need {c['comparison']} {c['threshold']})")

        if name == "shell_connectivity":
            w(f"       . {g['faces']} faces, {g['vertices_as_exported']} verts as "
              f"exported -> {g['vertices_welded']} welded")
            w(f"       . unwelded component count would be {g['components_unwelded']}"
              f" (meaningless: UV seams split vertices)")
            w(f"       . biggest pieces: {g['top_component_faces']}")
            w(f"       . open3d cross-check: {g['cross_check']['open3d_components']}"
              f" components / {g['cross_check']['open3d_largest_component_frac']} largest")
            if "disagreement" in g["cross_check"]:
                w(f"       ! {g['cross_check']['disagreement']}")
        elif name == "floor_continuity":
            gr = g["grid"]
            w(f"       . grid {gr['cells'][0]}x{gr['cells'][1]} @ {gr['cell_size']} "
              f"{gr['units']}/cell, {gr['interior_cells']} interior cells, "
              f"{g['rays_cast']} rays")
            w(f"       . {g['unsupported_cells']} columns with no floor under them, "
              f"in {g['void_regions']} voids")
            w(f"       . largest void {g['largest_gap_cells']} cells "
              f"(~{g['largest_gap_span']} units across)")
            w(f"       . ground level {g['ground_level']} +/- {g['ground_band']} band; "
              f"any-geometry {g['any_geometry_frac']} (diagnostic)")
        elif name == "prop_integrity":
            for p in g["per_prop"]:
                if "checks" not in p:
                    w(f"       {'ok  ' if p['pass'] else 'FAIL'} {p['slug']:<28} "
                      f"{p['reason']}")
                    continue
                c = p["checks"]
                w(f"       {'ok  ' if p['pass'] else 'FAIL'} {p['slug']:<28} "
                  f"largest={c['largest_component_frac']['value']} "
                  f"comps={p['components']} "
                  f"degen={p['degenerate_faces']} "
                  f"uv={c['has_uv']['value']} tex={c['has_base_color_texture']['value']} "
                  f"watertight={c['watertight']['value']}")
        elif name == "texture_coverage":
            for a in g["per_atlas"]:
                if "check" not in a:
                    w(f"       FAIL {a['element']:<28} {a['reason']}")
                    continue
                w(f"       {'ok  ' if a['pass'] else 'FAIL'} {a['element']:<28} "
                  f"measured={a['check']['value']} "
                  f"(build report said {a['build_report_coverage']}, pre-dilation)")
        elif name == "scale_sanity":
            for i in g["per_element"]:
                w(f"       {'ok  ' if i['pass'] else 'FAIL'} {i['element']:<28} "
                  f"longest={i['longest_dimension_m']} m  "
                  f"plausible {i['plausible_range_m']}")

    v = rep["verdict"]
    w("\n" + "=" * 72)
    w(f"VERDICT: {'PASS' if v['pass'] else 'FAIL'} - {v['summary']}")
    if v["skipped_gates"]:
        w(f"  not measurable: {', '.join(v['skipped_gates'])}"
          f"{' (strict: counted as failure)' if rep['strict'] else ''}")
    w("=" * 72)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Acceptance gates for a solidified world capture.")
    ap.add_argument("job_dir", help="job directory, or the _world directory itself")
    ap.add_argument("--report", help="where to write world_gate.json "
                                     "(default: <world>/world_gate.json)")
    ap.add_argument("--strict", action="store_true",
                    help="also fail when a gate cannot be measured")
    args = ap.parse_args(argv)

    try:
        world = resolve_world_dir(Path(args.job_dir).expanduser().resolve())
        report = run(world, args.strict)
    except GateError as exc:
        print(f"WORLD GATE ABORTED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - a gate that cannot measure must be loud
        print(f"WORLD GATE ABORTED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    out = Path(args.report).expanduser() if args.report else world / "world_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    _print_summary(report)
    print(f"\nreport: {out}")
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
