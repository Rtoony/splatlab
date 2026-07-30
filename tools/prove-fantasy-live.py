#!/usr/bin/env python3
"""Lane 6, layer 1 — THE integration receipt: a fantasy scene wrapped around
a splat foundation, walked, curtained, and still playable.

  1. AUTHOR+LAND   three CC0 library pieces through the Blender loop into the
                   Truck as role=environment (a raised dungeon platform, a
                   doorway wall, a graveyard fence) — the fire-hydrant
                   pattern, formalized
  2. WALK-ON       the player drops onto the RAISED dungeon platform and
                   stands there — authored fantasy floor above the captured
                   ground, cleanly separable heights
  3. CURTAIN       sphere at the capture core: the splat fringe fades while
                   the authored geometry past the seam stays — screenshot is
                   the money shot (photograph core, fantasy beyond)
  4. PLAYABLE      the zombie scenario still runs on the captured ground
                   (prove-game-live re-run, unchanged)
  5. CLEANUP       elements tombstoned, curtain disabled, state reset — the
                   world as found; the audit trail remains by design

Usage: prove-fantasy-live.py [JOB] [OUT_DIR]   (default: splat_716a9122)
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
import glb_check  # noqa: E402
from dcc import blender_workflow as wf  # noqa: E402

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_716a9122"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/fantasy-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")

drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = next((l.split("=", 1)[1].strip() for l in drop.read_text().splitlines()
              if l.startswith("PORTAL_TOKEN=")), os.environ.get("PORTAL_TOKEN", ""))
if not TOKEN:
    sys.exit("PORTAL_TOKEN unavailable")
OUT.mkdir(parents=True, exist_ok=True)
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = f"{BASE}/api/splat/jobs/{JOB}"

FLOOR = -0.32  # the Truck's proven floor (collision_shell.json probe)
# (slug, library asset, world x, world y(floor), world z) — Blender authoring
# frame is Z-up: location = [x, -z, y].
PIECES = [
    ("fantasy-platform", "dungeon-floor-tile-small", 3.2, FLOOR + 0.8, -0.4),
    ("fantasy-wall", "dungeon-wall-doorway", 5.2, FLOOR, -0.4),
    ("fantasy-fence", "graveyard-fence", 3.2, FLOOR, 1.6),
]
PLATFORM_TOP = FLOOR + 0.8 + 0.15  # tile is 0.15 m thick


def put_curtain(enabled: bool) -> None:
    doc = {"schema": "dev.splatlab.world-curtain/v1", "version": 1,
           "job_id": JOB, "enabled": enabled, "shape": "sphere",
           "center": [0.42, 0.5, -0.46], "radius": 2.5,
           "half_extents": [2.5, 2.5, 2.5], "soft_edge": 0.6}
    r = requests.put(f"{API}/world/curtain", json={"curtain": doc},
                     headers=AUTH, timeout=60)
    assert r.status_code == 200, f"PUT curtain: {r.status_code} {r.text[:200]}"


def cleanup() -> None:
    for slug, _a, _x, _y, _z in PIECES:
        requests.delete(f"{API}/world/elements/{slug}", headers=AUTH, timeout=60)
    put_curtain(False)
    requests.delete(f"{API}/world/state", headers=AUTH, timeout=60)


cleanup()  # a previous aborted run must not pollute this one
job_dir = wf._job_dir(JOB)

# ── 1. AUTHOR + LAND ────────────────────────────────────────────────────────────
for slug, asset, wx, wy, wz in PIECES:
    wf.run_action(JOB, "import_asset", {"name": asset, "slug": slug},
                  note="lane-6 fantasy proof")
    wf.run_action(JOB, "transform_object",
                  {"object": f"asset_{slug}", "location": [wx, -wz, wy]},
                  note="place at the fantasy seam")
    export = wf.export_glb(JOB, object_name=f"asset_{slug}",
                           bake_world_transform=True, note="lane-6 fantasy proof")
    export_path = job_dir / export["output"]["path"]
    bounds = glb_check.position_bounds(export_path)
    assert bounds["identity_transforms"], bounds
    with export_path.open("rb") as fh:
        landed = requests.post(
            f"{API}/world/elements",
            params={"slug": slug, "role": "environment", "label": slug},
            files={"file": (export_path.name, fh, "model/gltf-binary")},
            headers=AUTH, timeout=120)
    assert landed.status_code == 200, f"{slug}: {landed.status_code} {landed.text[:300]}"
    print(f"1  LAND: {asset} -> {slug} (environment) at "
          f"({wx}, {wy}, {wz})", flush=True)

manifest = requests.get(f"{API}/world/file",
                        params={"name": "world_manifest.json"},
                        headers=AUTH, timeout=60).json()
assert manifest["counts"]["environment"] == 3, manifest["counts"]
print(f"1b DOOR: counts.environment=3 in the manifest", flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    ctx = browser.new_context(viewport={"width": 1600, "height": 900},
                              extra_http_headers=AUTH)
    page = ctx.new_page()
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    deadline = time.time() + 180
    state = None
    while time.time() < deadline:
        state = page.evaluate("""() => {
          const w = window.__worldWalker;
          if (!w || !w.elements || !w.elements.length) return null;
          const envs = w.elements.filter((e) => e.role === "environment");
          const ready = !!(w.backdrop && w.backdrop.packedSplats
                           && w.backdrop.packedSplats.numSplats > 0);
          return (envs.length === 3 && ready)
            ? { collides: envs.every((e) => e.collides),
                source: w.colliderSource } : null;
        }""")
        if state:
            break
        time.sleep(1.0)
    assert state and state["collides"], f"environment elements not solid: {state}"
    assert state["source"] == "collision_shell+authored", state
    print(f"2a WALKER: 3 environment pieces in the BVH "
          f"({state['source']})", flush=True)

    # 2. Walk-on: the RAISED platform. Standing eye ~= top + clamped capsule
    #    (1.575) = 2.53; the captured floor would stand at 1.26.
    page.mouse.click(800, 450)
    page.evaluate("""() => {
      const w = window.__worldWalker;
      w.flying = false;
      w.camera.position.set(3.2, 3.4, -0.4);
      w.velocity.set(0, 0, 0);
    }""")
    deadline = time.time() + 15
    landed, prev_y = {}, None
    while time.time() < deadline:
        landed = page.evaluate("""() => {
          const w = window.__worldWalker;
          return { y: w.camera.position.y, grounded: w.grounded };
        }""")
        if landed["grounded"] and prev_y is not None \
                and abs(landed["y"] - prev_y) < 0.005:
            break
        prev_y = landed["y"]
        time.sleep(0.5)
    assert landed.get("grounded"), f"never grounded: {landed}"
    assert landed["y"] > PLATFORM_TOP + 1.2, \
        f"fell past the platform (floor eye ~1.26): {landed}"
    print(f"2b WALK-ON: standing on the dungeon platform at eye "
          f"y={landed['y']:.2f} (captured floor would be ~1.26)", flush=True)

    # 3. The curtain: photograph core stays, fringe fades, fantasy remains.
    page.evaluate("""() => {
      const w = window.__worldWalker;
      w.setCurtain({ enabled: true, shape: "sphere",
                     center: [0.42, 0.5, -0.46], radius: 2.5,
                     halfExtents: [2.5, 2.5, 2.5], softEdge: 0.6 });
      // Look from the platform toward the doorway wall past the seam.
      w.camera.lookAt(5.2, 1.2, -0.4);
    }""")
    time.sleep(2.0)
    page.screenshot(path=str(OUT / "01-fantasy-past-the-seam.png"))
    cs = page.evaluate("() => window.__worldWalker.curtainState()")
    assert cs and cs["enabled"], cs
    print(f"3  CURTAIN: enabled r={cs['radius']} soft={cs['softEdge']} — "
          f"screenshot {OUT / '01-fantasy-past-the-seam.png'}", flush=True)
    ctx.close()
    browser.close()

# 4. Still playable: the unchanged game receipt on the captured ground.
game = subprocess.run([sys.executable, str(REPO / "tools" / "prove-game-live.py"),
                       JOB, str(OUT / "game")],
                      capture_output=True, text=True, timeout=600)
assert game.returncode == 0, f"game proof failed:\n{game.stdout[-800:]}\n{game.stderr[-400:]}"
tail = [l for l in game.stdout.splitlines() if l.strip()][-2:]
print(f"4  PLAYABLE: {' | '.join(tail)}", flush=True)

# 5. Leave the world as found.
cleanup()
manifest = requests.get(f"{API}/world/file",
                        params={"name": "world_manifest.json"},
                        headers=AUTH, timeout=60).json()
assert manifest["counts"]["environment"] == 0, manifest["counts"]
print("5  CLEANUP: elements tombstoned, curtain disabled, counts.environment=0",
      flush=True)
print(f"ALL RECEIPTS LANDED — a fantasy scene on a splat foundation. {OUT}")
