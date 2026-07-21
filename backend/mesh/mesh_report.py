#!/usr/bin/env python3
"""Post-export report for the splat→mesh stage -> stable artifacts + mesh.json.

Usage: mesh_report.py <exported_mesh.ply> <out_dir> [--recipe JSON]

Normalizes the gs-mesh output to <out_dir>/mesh.ply, writes a best-effort
mesh.glb (Blender/web import), renders receipt views (top + 3 exterior orbits),
and writes mesh.json. Every metric is measured from the mesh itself (diagnose.py
lineage) — no narrative success anywhere.

Exit 0 = report written. Exit nonzero = mesh missing/empty/unreadable — the
stage runner treats that as a failed export.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open3d as o3d


def _render_receipts(mesh, out_dir: Path) -> tuple[list[str], str | None]:
    """Top + 3 exterior orbit views (orbit captures of objects/sites read better
    from outside — unlike the room-tuned interior views in mesh-trial)."""
    v = np.asarray(mesh.vertices)
    lo, hi = np.percentile(v, 1, axis=0), np.percentile(v, 99, axis=0)
    center = ((lo + hi) / 2).astype(np.float32)
    ext = hi - lo
    diag = float(np.linalg.norm(ext))

    renderer = o3d.visualization.rendering.OffscreenRenderer(1024, 768)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)
    renderer.scene.set_background([0.18, 0.18, 0.20, 1.0])
    renderer.scene.scene.set_sun_light([-0.3, -0.3, -0.9], [1, 1, 1], 90000)
    renderer.scene.scene.enable_sun_light(True)

    views = {
        "view_top": (
            center + np.array([0.0, 0.001, diag * 1.1], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
        )
    }
    for i, az in enumerate([0.0, 2.094, 4.189]):
        eye = center + np.array(
            [np.cos(az) * diag * 0.7, np.sin(az) * diag * 0.7, diag * 0.35],
            dtype=np.float32,
        )
        views[f"view_ext{i}"] = (eye, np.array([0, 0, 1], dtype=np.float32))

    written = []
    for name, (eye, up) in views.items():
        renderer.setup_camera(70.0, center, eye, up)
        o3d.io.write_image(str(out_dir / f"{name}.png"), renderer.render_to_image())
        written.append(f"{name}.png")
    return written, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--recipe", default=None, help="JSON provenance blob from the runner")
    args = ap.parse_args()

    mesh_path = Path(args.mesh)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not mesh_path.is_file():
        print(f"FATAL: mesh not found: {mesh_path}", file=sys.stderr)
        return 2

    canonical = out_dir / "mesh.ply"
    if mesh_path.resolve() != canonical.resolve():
        os.replace(mesh_path, canonical)

    mesh = o3d.io.read_triangle_mesh(str(canonical))
    if len(mesh.triangles) == 0:
        print(f"FATAL: exported mesh is empty: {canonical}", file=sys.stderr)
        return 1
    mesh.compute_vertex_normals()

    _, cluster_tris, _ = mesh.cluster_connected_triangles()
    cluster_tris = np.asarray(cluster_tris)
    ext = mesh.get_axis_aligned_bounding_box().get_extent()
    verts = np.asarray(mesh.vertices)
    # Robust extent (1-99th pct): raw bbox is gameable by floater specks.
    lo, hi = np.percentile(verts, 1, axis=0), np.percentile(verts, 99, axis=0)

    report: dict = {
        "v": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verts": len(mesh.vertices),
        "tris": len(mesh.triangles),
        "watertight": bool(mesh.is_watertight()),
        "clusters": int(len(cluster_tris)),
        "lcc_pct": round(float(100 * cluster_tris.max() / len(mesh.triangles)), 2),
        # RAW SCENE UNITS, not meters — the report has no meters_per_unit.
        # (Mislabeled "_m" until 2026-07-21; caught by the Blender-lab session.)
        "bbox_extent_units": [round(float(x), 3) for x in ext],
        "bbox_extent_robust_units": [round(float(x), 3) for x in (hi - lo)],
        "artifacts": {"ply": "mesh.ply"},
    }
    if args.recipe:
        try:
            report["recipe"] = json.loads(args.recipe)
        except json.JSONDecodeError:
            report["recipe_error"] = "unparseable recipe JSON from runner"

    # GLB is a convenience copy (Blender drag-in); PLY stays the artifact of
    # record. Written via trimesh — open3d 0.19's GLB writer emits corrupt
    # buffer views (found 2026-07-21) — and VERIFIED by readback: a GLB that
    # doesn't load with the same triangle count is deleted, never shipped.
    try:
        import trimesh

        glb = out_dir / "mesh.glb"
        tm = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_colors=(
                (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
                if mesh.has_vertex_colors()
                else None
            ),
            process=False,
        )
        # glTF is a Y-up format; our scene frame is Z-up. Export rotated
        # (Z-up -> Y-up) so importers' Y-up->native conversion restores the
        # true orientation — without this, Blender imports the site on its side.
        tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        tm.export(str(glb))
        back = trimesh.load(str(glb), force="mesh")
        if len(back.faces) == len(tm.faces):
            report["artifacts"]["glb"] = "mesh.glb"
        else:
            glb.unlink(missing_ok=True)
            report["glb_error"] = (
                f"readback mismatch ({len(back.faces)} != {len(tm.faces)} faces)"
            )
    except Exception as exc:  # noqa: BLE001 — convenience artifact, never fatal
        report["glb_error"] = str(exc)

    try:
        receipts, _ = _render_receipts(mesh, out_dir)
        report["artifacts"]["receipts"] = receipts
    except Exception as exc:  # noqa: BLE001 — receipts are receipts, not gates
        report["receipts_error"] = str(exc)

    (out_dir / "mesh.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
