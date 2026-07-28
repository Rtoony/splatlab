#!/usr/bin/env python3
"""Polish one world element end to end: Blender legs + audited upload.

Dry-run by default: runs the recipe (snapshot -> import -> cleanup ->
validated export — real versions and receipts are created) and PRINTS the
upload command without firing it. --upload performs the POST through the
same audited route the UI uses and verifies the served artifact's identity
against the export receipt. Requires PORTAL_TOKEN in the environment for
--upload (vault-injected; never written to disk).

  tools/polish-element.py splat_aea04ab3 red-bicycle
  tools/polish-element.py splat_aea04ab3 red-bicycle --upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from dcc import blender_workflow, polish_recipe  # noqa: E402

API = "http://127.0.0.1:3416/api/splat"


def _post_glb(url: str, token: str, glb: Path, filename: str) -> dict:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: model/gltf-binary\r\n\r\n"
    ).encode() + glb.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("job_id")
    ap.add_argument("slug")
    ap.add_argument("--merge-distance", type=float, default=1e-4)
    ap.add_argument("--decimate-ratio", type=float, default=None)
    ap.add_argument("--min-component-frac", type=float, default=0.02)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--upload", action="store_true",
                    help="actually POST the export through the polish route "
                         "(default: dry-run prints the command)")
    args = ap.parse_args()

    try:
        receipt = polish_recipe.run_recipe(
            args.job_id,
            args.slug,
            merge_distance=args.merge_distance,
            decimate_ratio=args.decimate_ratio,
            min_component_frac=args.min_component_frac,
            shade_smooth=not args.no_smooth,
            note=args.note,
        )
    except blender_workflow.BlenderWorkflowError as exc:
        done = [s["step"] for s in getattr(exc, "partial_receipts", [])]
        print(f"FAILED after {done or 'no steps'}: {exc}", file=sys.stderr)
        return 1

    job_dir = blender_workflow._job_dir(args.job_id)
    glb = job_dir / receipt["export"]["path"]
    print(json.dumps({k: receipt[k] for k in
                      ("job_id", "slug", "object", "cleanup", "export")},
                     indent=2))

    upload_url = f"{API}/jobs/{args.job_id}/world/elements/{args.slug}/polish"
    if not args.upload:
        print(f"\nDRY RUN — export is staged, nothing uploaded. To ingest:\n"
              f"  {sys.argv[0]} {args.job_id} {args.slug} --upload\n"
              f"  (or POST {glb} to {upload_url})")
        return 0

    token = os.environ.get("PORTAL_TOKEN", "")
    if not token:
        print("FATAL: --upload needs PORTAL_TOKEN in the environment "
              "(vault-injected; run under nexus-inject)", file=sys.stderr)
        return 1
    response = _post_glb(
        upload_url, token, glb, f"{args.slug}-polish-{receipt['export']['sha256'][:8]}.glb"
    )
    served_sha = response.get("receipt", {}).get("sha256")
    if not response.get("ok") or served_sha != receipt["export"]["sha256"]:
        print(f"FATAL: upload landed wrong ({json.dumps(response)[:400]})",
              file=sys.stderr)
        return 1
    # Independent verification: hash the file the route now serves.
    disk_sha = hashlib.sha256(
        (job_dir / "_world" /
         ("shell.glb" if args.slug == "shell" else f"elements/{args.slug}.glb")
         ).read_bytes()
    ).hexdigest()
    if disk_sha != receipt["export"]["sha256"]:
        print("FATAL: landed file hash mismatch", file=sys.stderr)
        return 1
    print(f"\nUPLOADED + VERIFIED: {args.slug} -> {served_sha[:12]} "
          f"(supersedes {response['receipt'].get('supersedes', {}).get('sha256', '')[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
