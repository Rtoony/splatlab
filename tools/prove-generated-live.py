#!/usr/bin/env python3
"""Triage treatment 3 live proof: Propose-generated — candidate, HITL
promote, exact revert. Runs on the fire-hydrant job (the operator's own
reference case), INCLUDING a real SAM-3D propose under the GPU arbiter.

  1. PROPOSE   POST .../generate/propose — full GPU run (~2-4 min); the
               candidate lands with the gates' verdict, nothing applied
  2. REVIEW    GET candidate: placed, mask IoU, served preview (no-cache)
  3. PROMOTE   the captured element GLB is versioned and replaced by the
               placed candidate (mesh-env transform+decimate+bake);
               world.json reads geometry_source=generated
  4. REVERT    the exact versioned pair returns — byte-identity receipt
  5. as-found  candidate remains on disk (regenerable review material)

Usage: prove-generated-live.py [JOB] [SLUG]  (default: splat_513e89171d fire-hydrant)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_513e89171d"
SLUG = sys.argv[2] if len(sys.argv) > 2 else "fire-hydrant"
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")

drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = next((l.split("=", 1)[1].strip() for l in drop.read_text().splitlines()
              if l.startswith("PORTAL_TOKEN=")), os.environ.get("PORTAL_TOKEN", ""))
if not TOKEN:
    sys.exit("PORTAL_TOKEN unavailable")
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = f"{BASE}/api/splat/jobs/{JOB}"
ELEMENT = Path.home() / f"projects/splatcli/outputs/3d/{JOB}/_world/elements/{SLUG}.glb"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# 0. As-found state; revert any stale promotion from an aborted run.
requests.post(f"{API}/objects/{SLUG}/generate/revert", headers=AUTH, timeout=120)
assert ELEMENT.is_file(), f"no captured element at {ELEMENT}"
before = sha(ELEMENT)
print(f"0  BASELINE: captured {SLUG}.glb sha {before}", flush=True)

# 1. A real propose under the arbiter (the lane solidify never gave it).
r = requests.post(f"{API}/objects/{SLUG}/generate/propose",
                  headers=AUTH, timeout=2400)
assert r.status_code == 200, f"propose: {r.status_code} {r.text[:300]}"
prop = r.json()
assert prop["placed"], f"gates refused this run: {prop['report'].get('mask')}"
iou = (prop["report"].get("mask_alignment_gate") or {}).get("iou_vs_captured_object")
print(f"1  PROPOSE: fresh SAM-3D candidate, PLACED, mask IoU {iou}", flush=True)

# 2. Review payload + served preview.
r = requests.get(f"{API}/objects/{SLUG}/generate/candidate",
                 headers=AUTH, timeout=60)
assert r.status_code == 200, r.text
cand = r.json()
assert cand["placed"] and cand["marker"] is None
pv = requests.get(f"{BASE}{cand['files']['preview']}", headers=AUTH, timeout=60)
assert pv.status_code == 200 and pv.headers["cache-control"] == "no-cache"
print(f"2  REVIEW: candidate served (preview {len(pv.content)} bytes, "
      "no-cache)", flush=True)

# 3. Promote.
r = requests.post(f"{API}/objects/{SLUG}/generate/promote",
                  headers=AUTH, timeout=1200)
assert r.status_code == 200, f"promote: {r.status_code} {r.text[:400]}"
marker = r.json()["marker"]
promoted = sha(ELEMENT)
assert promoted != before, "element GLB did not change"
world = json.loads((ELEMENT.parents[1] / "world.json").read_text())
entry = next(e for e in world["elements"] if e.get("slug") == SLUG)
assert entry.get("geometry_source") == "generated", entry
versions = ELEMENT.parents[1] / "versions" / marker["prior_glb_version"]
assert versions.is_file(), marker
print(f"3  PROMOTE: element now GENERATED ({marker['faces']} faces, "
      f"sha {promoted}); prior versioned as {marker['prior_glb_version']}",
      flush=True)

# 4. Exact revert.
r = requests.post(f"{API}/objects/{SLUG}/generate/revert",
                  headers=AUTH, timeout=120)
assert r.status_code == 200, r.text
restored = sha(ELEMENT)
assert restored == before, f"revert not exact: {restored} != {before}"
world = json.loads((ELEMENT.parents[1] / "world.json").read_text())
entry = next(e for e in world["elements"] if e.get("slug") == SLUG)
assert entry.get("geometry_source") == "gaussians", entry
assert not (ELEMENT.parent / f"{SLUG}.generated.json").is_file()
print(f"4  REVERT: byte-exact restore (sha {restored}); marker gone; "
      "world.json honest again", flush=True)
print("ALL RECEIPTS LANDED — propose, review, promote, revert: the human "
      "decides at every step.")
