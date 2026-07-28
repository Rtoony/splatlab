#!/usr/bin/env python3
"""Texture leg for GENERATED (SAM-3D) elements — UVs + a baked atlas.

place_generated() ships a placed, decimated mesh with vertex colours and
nothing else: no UVs, no texture. world_gate's prop_integrity requires both
per prop (`has_uv`, `has_base_color_texture`) — so every --prefer-generated
world failed the gate BY CONSTRUCTION. This module completes the artifact
instead of touching the gate:

  drop interior debris islands (measured on the bonsai cardboard-box: every
  secondary component sits INSIDE the main component's bbox — near-coincident
  generation shells, not real detached parts) -> xatlas unwrap -> k-NN bake
  of the mesh's own vertex colours into an atlas -> dilate -> textured GLB.

The unwrap/bake/dilate machinery is object_texture.py's, reused verbatim —
its k-NN bake never asks whether the colour points are gaussians or vertices.
Runs in the dn-splatter-probe env (trimesh/xatlas), like every mesh script.

Island policy: components are dropped smallest-first, only while the gate's
own criterion (largest/total >= target_frac) is unmet, and never any single
component above max_single_drop_frac of the mesh — the largest component is
always kept. The report records exactly what was dropped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def island_drop_plan(
    component_sizes: list[int],
    *,
    target_frac: float = 0.8,
    max_single_drop_frac: float = 0.2,
) -> dict:
    """Which components to drop, decided purely from their face counts.

    Pure function (no trimesh) so the policy is unit-testable in the app env.
    Returns indices into component_sizes, largest-component index, and the
    achieved fraction after the plan is applied.
    """
    if not component_sizes:
        return {"drop": [], "largest": None, "achieved_frac": 0.0}
    total_original = sum(component_sizes)
    order = sorted(range(len(component_sizes)), key=lambda i: component_sizes[i])
    largest = order[-1]
    remaining = total_original
    drop: list[int] = []
    for index in order[:-1]:
        if component_sizes[largest] / remaining >= target_frac:
            break
        if component_sizes[index] / total_original > max_single_drop_frac:
            break  # refuse to silently delete substantial geometry
        drop.append(index)
        remaining -= component_sizes[index]
    return {
        "drop": drop,
        "largest": largest,
        "achieved_frac": round(component_sizes[largest] / remaining, 4),
    }


def drop_debris_islands(
    mesh,
    *,
    target_frac: float = 0.8,
    max_single_drop_frac: float = 0.2,
):
    """Apply island_drop_plan to a trimesh mesh. Returns (mesh, report).

    Connectivity is measured on a position-welded copy: the raw index buffer
    splits vertices at generation-artifact seams, so unwelded adjacency
    overcounts components (the same lesson world_gate.py records).
    """
    import trimesh

    welded = mesh.copy()
    welded.merge_vertices()
    components = trimesh.graph.connected_components(
        welded.face_adjacency, nodes=np.arange(len(welded.faces))
    )
    sizes = [int(len(c)) for c in components]
    plan = island_drop_plan(
        sizes,
        target_frac=target_frac,
        max_single_drop_frac=max_single_drop_frac,
    )
    report = {
        "components": sorted(sizes, reverse=True),
        "dropped_components": len(plan["drop"]),
        "dropped_faces": int(sum(sizes[i] for i in plan["drop"])),
        "achieved_frac": plan["achieved_frac"],
    }
    if not plan["drop"]:
        return mesh, report
    keep = np.ones(len(mesh.faces), dtype=bool)
    for index in plan["drop"]:
        keep[components[index]] = False
    # merge_vertices preserves face order, so the welded component face
    # indices address the original mesh's faces directly.
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    return mesh, report


def texture_generated_mesh(
    mesh,
    out_glb: Path,
    *,
    mpu: float | None,
    tex_size: int = 1024,
    target_frac: float = 0.8,
    k: int = 24,
) -> dict:
    """Unwrap + bake a capture-frame vertex-coloured mesh into a textured GLB.

    Input mesh is CAPTURE-frame (pre scale/Y-up), exactly what place_generated
    holds after decimation; the export applies the same metres/Y-up convention
    as object_texture.py and place_generated, so siblings keep matching.
    Raises on any failure — the caller decides whether to fall back to the
    untextured export.
    """
    import trimesh
    from PIL import Image

    import object_texture as ot

    t0 = time.time()
    mesh, islands = drop_debris_islands(mesh, target_frac=target_frac)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(verts):
        raise ValueError("generated mesh has no per-vertex colours to bake from")
    gx = verts.copy()
    gc = np.asarray(colors, dtype=np.float32)[:, :3] / 255.0

    u_verts, u_faces, uvs = ot._parametrize_chunked(
        np.asarray(verts, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int64),
        chunks=1,
    )
    tex, mask = ot.bake_texture(u_verts, u_faces, uvs, gx, gc, tex_size, k=k)
    if tex is None:
        raise ValueError("atlas rasterization covered no texels")
    coverage_rasterized = float(mask.mean())
    tex, shipped_mask = ot.dilate_atlas(tex, mask, passes=max(16, tex_size // 48))
    image = Image.fromarray(tex, mode="RGB")

    scale = float(mpu) if mpu else 1.0
    scaled = u_verts * scale
    yup = np.stack(
        [scaled[:, 0], scaled[:, 2], -scaled[:, 1]], axis=1
    ).astype(np.float32)
    out = trimesh.Trimesh(vertices=yup, faces=u_faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        image=image,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=image, metallicFactor=0.0, roughnessFactor=0.75
        ),
    )
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out.export(str(out_glb))
    atlas_path = out_glb.with_name(out_glb.stem + "_atlas.png")
    image.save(atlas_path)

    back = trimesh.load(str(out_glb), force="mesh")
    if len(back.faces) != len(u_faces):
        out_glb.unlink(missing_ok=True)
        raise ValueError(
            f"GLB readback mismatch ({len(back.faces)} != {len(u_faces)})"
        )
    has_uv = getattr(getattr(back.visual, "uv", None), "shape", None) is not None
    texture_back = getattr(
        getattr(back.visual, "material", None), "baseColorTexture", None
    )
    if not has_uv or texture_back is None:
        out_glb.unlink(missing_ok=True)
        raise ValueError("GLB readback lost UVs or texture")

    return {
        "faces": int(len(back.faces)),
        "verts": int(len(back.vertices)),
        "extent": [round(float(x), 3) for x in back.extents],
        "seconds": round(time.time() - t0, 1),
        "islands": islands,
        "texture": {
            "baked": True,
            "size": tex_size,
            "coverage": round(float(shipped_mask.mean()), 4),
            "coverage_rasterized": round(coverage_rasterized, 4),
            "atlas_png": atlas_path.name,
            "atlas_bytes": atlas_path.stat().st_size,
            "readback_texture_size": list(texture_back.size),
        },
    }
