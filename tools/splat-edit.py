#!/usr/bin/env python3
"""Drive the audited splat-edit lane from the terminal.

Edits are always non-destructive (each lands as _splat/versions/splat-vNNNN
with a receipt; the live asset is untouched). `promote` is the one mutating
step and is guarded server-side: index-keyed consumers (language field,
isolated claims) refuse promotion without --force, and the pristine original
is preserved to _splat/original.ply on first promote.

  tools/splat-edit.py <job> versions
  tools/splat-edit.py <job> clean --min-opacity 0.05 --max-scale 0.5
  tools/splat-edit.py <job> crop --min -2,-2,-2 --max 2,2,2
  tools/splat-edit.py <job> transform --scale 2.0 --translate 0,0,1
  tools/splat-edit.py <job> promote --version 2 [--force]

Requires PORTAL_TOKEN in the environment (vault-injected). Deliberately not
an MCP tool: mutations of live-adjacent assets go through the audited HTTP
door only, the same doctrine as polish uploads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

API = "http://127.0.0.1:3416/api/splat"


def _vec(text: str) -> list[float]:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise SystemExit("FATAL: vectors are three comma-separated numbers")
    return parts


def _call(method: str, path: str, token: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=660) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        raise SystemExit(f"FATAL: HTTP {exc.code}: {detail}") from exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_id")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("versions")
    clean = sub.add_parser("clean")
    clean.add_argument("--min-opacity", type=float, default=None)
    clean.add_argument("--max-scale", type=float, default=None)
    clean.add_argument("--max-dist", type=float, default=None)
    crop = sub.add_parser("crop")
    crop.add_argument("--min", required=True)
    crop.add_argument("--max", required=True)
    transform = sub.add_parser("transform")
    transform.add_argument("--scale", type=float, default=None)
    transform.add_argument("--translate", default=None)
    promote = sub.add_parser("promote")
    promote.add_argument("--version", type=int, required=True)
    promote.add_argument("--force", action="store_true")
    for parser in (clean, crop, transform, promote):
        parser.add_argument("--note", default="")
    args = ap.parse_args()

    token = os.environ.get("PORTAL_TOKEN", "")
    if not token:
        print("FATAL: PORTAL_TOKEN missing (vault-injected; run under "
              "nexus-inject)", file=sys.stderr)
        return 1

    if args.command == "versions":
        print(json.dumps(_call("GET", f"/jobs/{args.job_id}/splat/versions",
                               token), indent=2))
        return 0
    if args.command == "promote":
        result = _call("POST", f"/jobs/{args.job_id}/splat/promote", token,
                       {"version": args.version, "force": args.force,
                        "note": args.note})
        print(json.dumps(result["receipt"], indent=2))
        return 0

    if args.command == "clean":
        params = {k: v for k, v in (("min_opacity", args.min_opacity),
                                    ("max_scale", args.max_scale),
                                    ("max_dist", args.max_dist)) if v is not None}
        op = "clean"
    elif args.command == "crop":
        params = {"min": _vec(args.min), "max": _vec(args.max)}
        op = "crop_box"
    else:
        params = {}
        if args.scale is not None:
            params["scale"] = args.scale
        if args.translate is not None:
            params["translate"] = _vec(args.translate)
        op = "transform"
    result = _call("POST", f"/jobs/{args.job_id}/splat/edit", token,
                   {"op": op, "params": params, "note": args.note})
    receipt = result["receipt"]
    engine = receipt["engine"]
    print(f"v{result['version']:04d}: {op} kept {engine['out_count']} of "
          f"{engine['in_count']} gaussians (removed {engine['removed']}) "
          f"-> {receipt['output']['path']}")
    print(f"sha {receipt['output']['sha256'][:12]}; promote with:\n"
          f"  tools/splat-edit.py {args.job_id} promote --version {result['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
