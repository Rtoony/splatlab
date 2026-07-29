#!/usr/bin/env python3
"""PW-A9 live proof #1: the paint-rescue of a ground-refused capture.

The honest scriptable brush — every step is the UI's own endpoint, nothing
synthetic touches disk:

  1. select gaussians in a sphere  (POST /langfield/select/sphere)
  2. commit them as ground paint   (POST /class-labels)
  3. capture the BEFORE receipt    (default-knob /scene/ground refusal text)
  4. rebuild ground, tuned         (POST /scene/ground + bounded floors)
     -> assert the paint-authority ledger (painted_cells > 0, coherent sums)
  5. finish the world              (POST /world/prepare, resumes past ground)
  6. prove it plays                (tools/prove-game-live.py)

The paint record is deliberately LEFT IN PLACE — it is the fix, not
scaffolding. Requires the splatlab-langfield worker (fails loud with the
service name if it is down).

Usage: prove-paint-ground-live.py JOB CX,CY,CZ RADIUS [CLASS_ID]
         [--min-ground-points 20] [--min-pts-cell 1] [--cell-units 0.03]
"""
import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")


def _load_token() -> str:
    """RAM secret store first, env fallback. The store can be unreadable from
    a MAC-confined shell (EACCES despite owner match) — don't crash on the
    read, fall through and say what to do."""
    try:
        for l in Path("/dev/shm/nexus-env-splatlab").read_text().splitlines():
            if l.startswith("PORTAL_TOKEN="):
                return l.split("=", 1)[1].strip()
    except OSError:
        pass
    tok = os.environ.get("PORTAL_TOKEN", "")
    if not tok:
        sys.exit("PORTAL_TOKEN unavailable — /dev/shm/nexus-env-splatlab "
                 "unreadable and env unset; run from a normal shell or "
                 "export PORTAL_TOKEN")
    return tok


TOKEN = _load_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("center", help="x,y,z of the paint sphere (scene units)")
    ap.add_argument("radius", type=float)
    ap.add_argument("class_id", nargs="?", default="pavement")
    ap.add_argument("--min-ground-points", type=int, default=20)
    ap.add_argument("--min-pts-cell", type=int, default=1)
    ap.add_argument("--cell-units", type=float, default=0.03)
    args = ap.parse_args()
    api = f"{BASE}/api/splat/jobs/{args.job}"
    center = [float(x) for x in args.center.split(",")]

    # 1. The UI's own brush selection.
    r = requests.post(f"{api}/langfield/select/sphere",
                      json={"center": center, "radius": args.radius},
                      headers=AUTH, timeout=300)
    if r.status_code == 503:
        sys.exit("FATAL: langfield worker is down — "
                 "systemctl --user start splatlab-langfield")
    assert r.status_code == 200, r.text
    count = int(r.headers.get("X-Count", "0"))
    print(f"1  SELECT: {count} gaussians in r={args.radius} @ {center}", flush=True)
    assert count >= 10, "selection too small for a paint record (MIN_COUNT=10)"

    # 2. Commit as ground-class paint (retry with force if the fraction
    #    guardrail asks for it — that is the UI's own Force button).
    payload = {"class_id": args.class_id,
               "indices_b64": base64.b64encode(r.content).decode(),
               "note": "prove-paint-ground-live A9/B1"}
    r = requests.post(f"{api}/class-labels", json=payload, headers=AUTH,
                      timeout=300)
    if r.status_code == 400 and "force" in r.text.lower():
        print("2a PAINT: fraction guardrail tripped — retrying with force "
              "(the UI's own Force button)", flush=True)
        r = requests.post(f"{api}/class-labels", json={**payload, "force": True},
                          headers=AUTH, timeout=300)
    assert r.status_code == 200, r.text
    record_id = (r.json().get("record") or {}).get("id")
    print(f"2  PAINT: committed {count} gaussians as '{args.class_id}' "
          f"(record {record_id})", flush=True)

    # 3. The BEFORE receipt: default knobs must still refuse (that is why
    #    this scene needed rescuing). If it passes, say so honestly.
    r = requests.post(f"{api}/scene/ground", json={}, headers=AUTH, timeout=3600)
    if r.status_code == 500:
        print(f"3  BEFORE: default knobs still refuse — {r.json()['detail']}",
              flush=True)
    else:
        print(f"3  BEFORE: default knobs now pass (status {r.status_code}) — "
              "the paint alone rescued it; tuned run still follows for the "
              "ledger", flush=True)

    # 4. The tuned rebuild + the authority ledger.
    r = requests.post(f"{api}/scene/ground", json={
        "min_ground_points": args.min_ground_points,
        "min_pts_cell": args.min_pts_cell,
        "cell_units": args.cell_units,
    }, headers=AUTH, timeout=3600)
    assert r.status_code == 200, r.text
    report = r.json()
    ledger = (report["cells_with_data"] - report["cells_spike_rejected"]
              - report["cells_disconnected_dropped"])
    assert ledger == report["ground_points"], report
    assert report.get("painted_cells", 0) > 0, \
        "paint never reached the binning — no painted cells in the ledger"
    print(f"4  GROUND: {report['ground_points']} cells "
          f"(painted {report['painted_cells']}, rescued sparse "
          f"{report.get('painted_rescued_sparse', 0)} / component "
          f"{report.get('painted_rescued_component', 0)})", flush=True)

    # 5. Finish the world — prepare resumes past the built ground stage.
    r = requests.post(f"{api}/world/prepare", json={}, headers=AUTH,
                      timeout=7200)
    assert r.status_code == 200, r.text
    stages = r.json().get("stages") or {}
    print(f"5  PREPARE: {stages}", flush=True)
    assert stages.get("ground") == "skipped", \
        "prepare re-ran ground — the tuned build should have satisfied it"

    # 6. The playable receipt.
    print("6  GAME PROOF:", flush=True)
    rc = subprocess.call([sys.executable,
                          str(Path(__file__).with_name("prove-game-live.py")),
                          args.job, f"/tmp/{args.job}-paint-rescue-proof"])
    if rc != 0:
        return rc
    print("ALL RECEIPTS LANDED — paint record kept (it IS the fix)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
