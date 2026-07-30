#!/usr/bin/env python3
"""Lane 6 live proof: authored environment geometry is REAL floor.

  1. build a world-frame-baked platform GLB (trimesh, identity transforms)
  2. place it through the door with role=environment -> manifest receipts
     (role, provenance, counts.environment)
  3. open the walker: the element loads with collides=true, the collider
     source reads collision_shell+authored, and the BVH grew
  4. the WALK-ON receipt: drop the player above the platform, gravity runs,
     and they stand ON it (grounded, eye height above the platform top —
     measurably higher than standing on the captured floor below)
  5. DELETE the element, reload: collider back to the bare collision shell

Usage: prove-environment-live.py [JOB] [OUT_DIR]   (default: splat_716a9122)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_716a9122"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/environment-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")
MESH_ENV_PYTHON = Path.home() / "miniconda3/envs/dn-splatter-probe/bin/python"
SLUG = "env-proof-platform"

# Platform sized to the Truck's proven interior: seed (0.42, -0.02, -0.46),
# floor -0.32, probe ceiling 1.95 — top at floor+0.6 keeps the standing
# capsule inside the measured headroom.
PLATFORM_CENTRE = (1.9, 0.13, -0.46)
PLATFORM_TOP = 0.28


def _load_token() -> str:
    try:
        for l in Path("/dev/shm/nexus-env-splatlab").read_text().splitlines():
            if l.startswith("PORTAL_TOKEN="):
                return l.split("=", 1)[1].strip()
    except OSError:
        pass
    tok = os.environ.get("PORTAL_TOKEN", "")
    if not tok:
        sys.exit("PORTAL_TOKEN unavailable — run from a normal shell or export it")
    return tok


TOKEN = _load_token()
AUTH = {"Authorization": f"Bearer {TOKEN}"}
OUT.mkdir(parents=True, exist_ok=True)
API = f"{BASE}/api/splat/jobs/{JOB}"

# 0. Clean slate: remove a leftover platform from an aborted run.
requests.delete(f"{API}/world/elements/{SLUG}", headers=AUTH, timeout=60)

# 1. The platform GLB — world-frame-baked vertices, identity transforms.
glb_path = OUT / "platform.glb"
proc = subprocess.run([str(MESH_ENV_PYTHON), "-c", f"""
import trimesh
box = trimesh.creation.box(extents=[2.0, 0.3, 2.0])
box.apply_translation([{PLATFORM_CENTRE[0]}, {PLATFORM_CENTRE[1]}, {PLATFORM_CENTRE[2]}])
box.export(r"{glb_path}")
print("verts", len(box.vertices), "faces", len(box.faces))
"""], capture_output=True, text=True, timeout=120)
assert proc.returncode == 0, proc.stderr[-500:]
print(f"1  GLB: baked platform ({proc.stdout.strip()})", flush=True)

# 2. Through the door as environment.
with glb_path.open("rb") as fh:
    r = requests.post(f"{API}/world/elements",
                      params={"slug": SLUG, "role": "environment",
                              "label": "proof platform"},
                      files={"file": (glb_path.name, fh, "model/gltf-binary")},
                      headers=AUTH, timeout=120)
assert r.status_code == 200, f"door refused: {r.status_code} {r.text[:300]}"
r = requests.get(f"{API}/world/file", params={"name": "world_manifest.json"},
                 headers=AUTH, timeout=60)
manifest = r.json()
rec = next(e for e in manifest["elements"] if e["slug"] == SLUG)
assert rec["role"] == "environment" and rec["provenance"] == "authored", rec
print(f"2  DOOR: placed as environment; counts={manifest.get('counts')}", flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    ctx = browser.new_context(viewport={"width": 1600, "height": 900},
                              extra_http_headers=AUTH)
    page = ctx.new_page()
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    deadline = time.time() + 120
    state = {}
    while time.time() < deadline:
        state = page.evaluate("""() => {
          const w = window.__worldWalker;
          if (!w || !w.elements || !w.elements.length) return null;
          const el = w.elements.find((e) => e.slug === %r);
          return el ? { collides: el.collides, role: el.role,
                        provenance: el.provenance,
                        colliderTris: w.colliderTris,
                        colliderSource: w.colliderSource } : null;
        }""" % SLUG)
        if state:
            break
        time.sleep(1.0)
    assert state, "platform element never loaded in the walker"
    assert state["collides"] and state["role"] == "environment", state
    assert state["colliderSource"] == "collision_shell+authored", state
    print(f"3  WALKER: collides=true, colliderSource={state['colliderSource']}, "
          f"tris={state['colliderTris']}", flush=True)

    # 4. The walk-on receipt. Standing eye on the captured floor is ~1.26
    #    scene units (floor -0.32 + clamped capsule 1.575); on the platform
    #    (top 0.28) it is ~1.85 — cleanly separable.
    page.evaluate("""() => {
      const w = window.__worldWalker;
      w.flying = false;
      w.camera.position.set(%f, 2.6, %f);
      w.velocity.set(0, 0, 0);
    }""" % (PLATFORM_CENTRE[0], PLATFORM_CENTRE[2]))
    # Wait for a SETTLED stand, not the first grounded flag — the flag is
    # stale from the pre-teleport stance for a frame or two, and reading it
    # at the teleport height was a measured false pass (y=2.60, the exact
    # drop height, "grounded"). Settled = grounded AND y unchanged 0.5s apart.
    deadline = time.time() + 15
    landed, prev_y = {}, None
    while time.time() < deadline:
        landed = page.evaluate("""() => {
          const w = window.__worldWalker;
          return { y: w.camera.position.y, x: w.camera.position.x,
                   z: w.camera.position.z, grounded: w.grounded };
        }""")
        if landed["grounded"] and prev_y is not None \
                and abs(landed["y"] - prev_y) < 0.005:
            break
        prev_y = landed["y"]
        time.sleep(0.5)
    assert landed.get("grounded"), f"never grounded: {landed}"
    # Standing eye on the platform ≈ top 0.28 + clamped capsule ≈ 1.85;
    # on the captured floor it is ≈ 1.26; at the drop height it was 2.60.
    assert PLATFORM_TOP + 1.2 < landed["y"] < 2.3, \
        f"not standing ON the platform (floor ≈1.26, drop 2.60): {landed}"
    assert abs(landed["x"] - PLATFORM_CENTRE[0]) < 1.2 \
        and abs(landed["z"] - PLATFORM_CENTRE[2]) < 1.2, \
        f"slid off the platform: {landed}"
    print(f"4  WALK-ON: standing on the authored platform at eye y="
          f"{landed['y']:.2f} (captured floor would be ~1.26)", flush=True)
    page.screenshot(path=str(OUT / "standing-on-platform.png"))
    ctx.close()

    # 5. Cleanup + the collider returns to baseline.
    r = requests.delete(f"{API}/world/elements/{SLUG}", headers=AUTH, timeout=60)
    assert r.status_code == 200, f"delete: {r.status_code} {r.text[:200]}"
    ctx2 = browser.new_context(viewport={"width": 1600, "height": 900},
                               extra_http_headers=AUTH)
    page2 = ctx2.new_page()
    page2.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page2.wait_for_selector("canvas", timeout=120_000)
    deadline = time.time() + 120
    after = {}
    while time.time() < deadline:
        after = page2.evaluate("""() => {
          const w = window.__worldWalker;
          return (w && w.colliderTris)
            ? { tris: w.colliderTris, source: w.colliderSource } : null;
        }""") or {}
        if after:
            break
        time.sleep(1.0)
    assert after.get("source") == "collision_shell", \
        f"collider did not return to baseline: {after}"
    assert after["tris"] == state["colliderTris"] - 12, \
        f"tri count did not shrink by the platform's 12: {after} vs {state}"
    print(f"5  CLEANUP: element deleted; collider back to bare "
          f"collision_shell ({after['tris']} tris)", flush=True)
    browser.close()

print(f"ALL RECEIPTS LANDED — authored geometry is real floor. screenshots: {OUT}")
