#!/usr/bin/env python3
"""P6e: top-down + oblique receipt renders of the finished, colored
ground_mesh.glb. Same Open3D EGL OffscreenRenderer pattern as
mesh_report.py's _render_receipts (proven headless in this environment),
adapted for vertex-colored geometry (defaultUnlit shader reads the GLB's
baked vertex colors directly, no scene lighting needed).

Runs in the dn-splatter-probe env.

Usage: ground_mesh_receipt.py <ground_mesh.glb> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> int:
    glb_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    mesh = o3d.io.read_triangle_mesh(str(glb_path))
    if len(mesh.triangles) == 0:
        print("FATAL: empty ground mesh", file=sys.stderr)
        return 1

    v = np.asarray(mesh.vertices)
    lo, hi = np.percentile(v, 1, axis=0), np.percentile(v, 99, axis=0)
    center = ((lo + hi) / 2).astype(np.float32)
    diag = float(np.linalg.norm(hi - lo))

    renderer = o3d.visualization.rendering.OffscreenRenderer(1024, 768)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"  # read the GLB's baked vertex colors as-is
    renderer.scene.add_geometry("ground", mesh, mat)
    renderer.scene.set_background([0.18, 0.18, 0.20, 1.0])

    # GLB is Y-up (twin_finish.py convention); top-down looks down -Y, oblique
    # offsets in X/Z at a 3/4 elevation.
    views = {
        "top": (center + np.array([0.0, diag * 1.1, 0.001], dtype=np.float32),
                np.array([0, 0, -1], dtype=np.float32)),
        "oblique": (center + np.array([diag * 0.7, diag * 0.5, diag * 0.7], dtype=np.float32),
                    np.array([0, 1, 0], dtype=np.float32)),
    }
    written = []
    for name, (eye, up) in views.items():
        renderer.setup_camera(60.0, center, eye, up)
        path = out_dir / f"receipt_{name}.png"
        o3d.io.write_image(str(path), renderer.render_to_image())
        written.append(path.name)
    print(f"GROUND_MESH_RECEIPT_DONE {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
