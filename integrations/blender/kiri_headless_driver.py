"""Headless driver for the KIRI 3DGS Render add-on (research lane).

Proven 2026-07-28 on the bonsai capture (spike receipt in STATUS.md):
605,391 splats in -> exact edit -> 605,391 splats out with the full
canonical 3DGS layout (SH3, 45 f_rest coefficients) preserved.

RESEARCH LANE — this drives Blender 5.1.2 + the Apache-2.0 KIRI extension
(`dgs_render_by_kiri_engine`, user_default repo), NOT the 4.5.11-pinned
studio-loop toolchain. Nothing here touches blender_workflow's action
surface; graduating splat-editing into the audited workflow is a separate,
deliberate step.

Usage (Blender 5.1.2 only; the add-on's 5.2 patch is untested upstream):

  ~/Apps/blender-5.1.2/blender --background --factory-startup \
      --python integrations/blender/kiri_headless_driver.py -- \
      --in-ply  <capture>/_preview/splat.ply \
      --out-dir /tmp/exports/ \
      --scale 2.0

Two headless gotchas this file exists to encode:

1. The add-on is Serpens-generated: every operator lives under `bpy.ops.sna`
   with a hash suffix (e.g. sna.dgs_render_import_ply_e0a3a). There are no
   kiri/splat-named categories to discover.
2. Export DEFERS its real work through bpy.app.timers, and --background
   quits before any timer ticks — bpy.ops returns FINISHED with no file
   written (a silent failure). The fix: monkeypatch timers.register to
   capture callbacks and drain the chain synchronously (drain_deferred).

Known open validation item: the exported `scale_*` attributes shift by more
than ln(s) under a uniform scale s (measured 1.621 vs ln(2)=0.693 on the
bonsai), so KIRI may reparametrize gaussian scale on export. Render-compare
an edited export in the Spark viewer before trusting edited splats anywhere
user-facing.
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy

EXTENSION_MODULE = "bl_ext.user_default.dgs_render_by_kiri_engine"
OP_IMPORT = "dgs_render_import_ply_e0a3a"
OP_APPLY_TRANSFORMS = "dgs_render_apply_3dgs_tranforms_5b665"
OP_EXPORT = "dgs_render_export_mesh_as_3dgs4dgs_ce2f7"


def drain_deferred(pending: list, max_ticks: int = 2000) -> int:
    """Run captured timer callbacks synchronously until the chain is dry."""
    ticks = 0
    while pending and ticks < max_ticks:
        fn = pending.pop(0)
        ret = fn()
        ticks += 1
        if isinstance(ret, (int, float)):
            pending.append(fn)
    return ticks


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-ply", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scale", type=float, default=None,
                    help="uniform scale to apply as the edit")
    ap.add_argument("--suffix", default="_edited")
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    bpy.ops.preferences.addon_enable(module=EXTENSION_MODULE)

    result = getattr(bpy.ops.sna, OP_IMPORT)(filepath=args.in_ply)
    if result != {"FINISHED"}:
        print(f"FATAL: import returned {result}", file=sys.stderr)
        return 1
    splat = next(
        (obj for obj in bpy.data.objects
         if obj.type == "MESH" and "KIRI_3DGS_Render_GN" in obj.modifiers),
        None,
    )
    if splat is None:
        print("FATAL: no KIRI splat object after import", file=sys.stderr)
        return 1
    count_in = len(splat.data.vertices)
    print(f"imported {count_in} splats -> {splat.name}")

    bpy.context.view_layer.objects.active = splat
    splat.select_set(True)
    if args.scale is not None:
        splat.scale = (args.scale,) * 3
        getattr(bpy.ops.sna, OP_APPLY_TRANSFORMS)(sna_apply_location=True)

    sp = bpy.context.scene.sna_dgs_scene_properties
    sp.export_output_path = args.out_dir if args.out_dir.endswith("/") else args.out_dir + "/"
    sp.export_suffix = args.suffix

    pending: list = []
    bpy.app.timers.register = lambda fn, **kw: pending.append(fn)
    result = getattr(bpy.ops.sna, OP_EXPORT)(sna_send_to_world_centre=False)
    ticks = drain_deferred(pending)
    print(f"export op {result}, drained {ticks} deferred ticks")

    exported = [f for f in sorted(os.listdir(args.out_dir)) if f.endswith(".ply")]
    if not exported:
        print("FATAL: export produced no file (deferred chain incomplete?)",
              file=sys.stderr)
        return 1
    for name in exported:
        path = os.path.join(args.out_dir, name)
        print(f"EXPORTED {name} ({os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
