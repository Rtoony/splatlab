#!/usr/bin/env python3
"""P6f: resolve the fidelity dial (mode + overrides) against P6b-e's outputs
into one concrete scene_manifest.json.

`provenance` (existing, per-file) stays a FACT about what a script produced.
Fidelity mode is a SEPARATE, new concept: a selection POLICY this script
applies once, at assembly time, choosing which available representation of
each object goes into THIS build:
  - faithful: every instance uses its captured (P6c) file. Proxies excluded
    even when built. A pure reality-capture twin.
  - styled (default): uses each instance's proxy (P6d) when one was built,
    falls back to captured only where P6d was skipped.
  - overrides: an optional sparse {slug: "captured"|"proxy"} map layered on
    top of either base mode -- "styled scene but keep the hydrant real" is a
    real use case neither base mode alone can express. An override naming an
    unavailable representation, or an unknown slug, is a hard fail (this is
    a deliberate creative choice, not a best-effort batch op).

Every element records WHY it ended up where it did (the `selection` field) --
without it, "faithful mode deliberately excluded a working proxy" and "the
proxy build actually failed" are indistinguishable months later.

No GPU, no Blender -- pure stdlib JSON in, validated manifest out. Runs in
any env (matches noun_consolidate.py's zero-heavy-deps precedent).

Usage: scene_assemble.py <scene_dir> <job_id> <out_manifest.json>
       [--mode faithful|styled] [--overrides '{"slug": "captured"}']
       [--units-mode meters|scene-units] [--meters-per-unit 2.35]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene_manifest as sm  # noqa: E402


class AssembleError(ValueError):
    """A fidelity choice could not be resolved -- fail loud, never guess."""


def resolve_elements(scene_dir: Path, mode: str, overrides: dict) -> list[dict]:
    isolated_dir = scene_dir / "isolated"
    proxied_dir = scene_dir / "proxied"
    isolate_report = json.loads((isolated_dir / "batch_isolate.json").read_text())
    proxy_report_path = proxied_dir / "batch_proxy.json"
    proxy_by_slug = {}
    if proxy_report_path.is_file():
        for r in json.loads(proxy_report_path.read_text())["instances"]:
            proxy_by_slug[r["slug"]] = r

    built_instances = [r for r in isolate_report["instances"] if r["status"] == "built"]
    known_slugs = {r["slug"] for r in built_instances}
    unknown_overrides = set(overrides) - known_slugs
    if unknown_overrides:
        raise AssembleError(f"overrides name unknown slug(s): {sorted(unknown_overrides)}")

    elements = []

    # background: always captured, no override possible (it's the complement, not a named object)
    bg_ply = isolated_dir / "background.ply"
    if not bg_ply.is_file():
        raise AssembleError(f"missing {bg_ply} -- run /scene/isolate first")
    elements.append(dict(
        slug="background", provenance="captured",
        files={"ply": str(bg_ply)}, transform_4x4=None, registration=None,
        selection={"mode": mode, "chosen": "captured", "available": ["captured"],
                   "reason": "background-always-captured"},
    ))

    for r in built_instances:
        slug, label = r["slug"], r["label"]
        proxy = proxy_by_slug.get(slug)
        proxy_built = bool(proxy and proxy["status"] == "built")
        available = ["captured"] + (["proxy"] if proxy_built else [])

        if slug in overrides:
            chosen = overrides[slug]
            if chosen not in available:
                raise AssembleError(
                    f"override requests {chosen!r} for slug {slug!r} but only "
                    f"{available} is available")
            reason = f"override:{chosen}"
        elif mode == "faithful":
            chosen = "captured"
            reason = "faithful-excludes-proxy" if proxy_built else "faithful-mode"
        elif mode == "styled":
            if proxy_built:
                chosen, reason = "proxy", "styled-mode"
            else:
                upstream = proxy["status"] if proxy else "not-attempted"
                chosen, reason = "captured", f"upstream-skip:{upstream}"
        else:
            raise AssembleError(f"unknown mode {mode!r}")

        if chosen == "proxy":
            files = {"ply": str(proxied_dir / slug / "proxy.ply")}
            transform_4x4 = proxy.get("transform_4x4")
            registration = {k: proxy.get(k) for k in ("icp_fitness", "icp_rmse", "total_scale")}
        else:
            files = {"ply": str(isolated_dir / slug / "object.ply")}
            transform_4x4 = None
            registration = None

        elements.append(dict(
            slug=slug, provenance=chosen, files=files,
            transform_4x4=transform_4x4, registration=registration,
            selection={"mode": mode, "chosen": chosen, "available": available,
                      "reason": reason, "label": label},
        ))

    ground_glb = scene_dir / "ground" / "ground_mesh.glb"
    if ground_glb.is_file():
        elements.append(dict(
            slug="ground", provenance="ground-derived",
            files={"glb": str(ground_glb)}, transform_4x4=None, registration=None,
            selection={"mode": mode, "chosen": "ground-derived",
                      "available": ["ground-derived"], "reason": "ground-always-derived"},
        ))

    return elements


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("job_id")
    ap.add_argument("out_manifest", type=Path)
    ap.add_argument("--mode", choices=("faithful", "styled"), default="styled")
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--units-mode", choices=("meters", "scene-units"), default="scene-units")
    ap.add_argument("--meters-per-unit", type=float, default=None)
    args = ap.parse_args()

    try:
        overrides = json.loads(args.overrides)
        elements = resolve_elements(args.scene_dir, args.mode, overrides)
        manifest = sm.new_manifest(args.job_id, args.units_mode, args.meters_per_unit)
        for el in elements:
            sm.add_element(manifest, **el)
        sm.write_manifest(args.out_manifest.parent, manifest)
    except (AssembleError, sm.ManifestError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    summary = {
        "n_elements": len(elements),
        "by_provenance": {
            p: sum(1 for e in elements if e["provenance"] == p)
            for p in ("captured", "proxy", "ground-derived")
        },
        "mode": args.mode, "overrides": overrides,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
