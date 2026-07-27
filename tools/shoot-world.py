#!/usr/bin/env python3
"""Screenshot the walkable world on the real GPU, for review.

Why this exists: every appearance argument in this project has been settled by
looking at a picture, not by reading a number — the mesh_gate side-by-side is
what finally explained a 11.96 dB score. But mesh_gate renders the MESH, and
after the hybrid split the mesh is deliberately no longer the whole picture. The
only honest view of what a person actually sees is the walker itself.

Renders through --use-angle=gl so it binds the RTX 5090. SwiftShader was tried
for the editor gate and is unusably slow on a scene this size (see
verify-editor-live.py), and it would not exercise the same path anyway.

The token is read from the RAM-only vault drop and never printed. Nothing is
written except the screenshots.

Usage:
  tools/shoot-world.py <job_id> [out_dir]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_3aaf8067"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/world-shots")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")

drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = ""
if drop.is_file():
    for line in drop.read_text().splitlines():
        if line.startswith("PORTAL_TOKEN="):
            TOKEN = line.split("=", 1)[1].strip()
            break
TOKEN = TOKEN or os.environ.get("PORTAL_TOKEN", "")
if not TOKEN:
    sys.exit("PORTAL_TOKEN unavailable (vault sealed? run nexus-unlock)")

OUT.mkdir(parents=True, exist_ok=True)
errors: list[str] = []

with sync_playwright() as p:
    browser = p.chromium.launch(args=[
        "--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist",
    ])
    ctx = browser.new_context(
        viewport={"width": 1600, "height": 900},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"},
    )
    page = ctx.new_page()
    page.set_default_timeout(30_000)
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))

    print(f"--- {JOB} ---", flush=True)
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    print("canvas present; waiting for the splat to stream in...", flush=True)
    # web.ply is tens of MB and Spark streams it; the first frames show an empty
    # or half-loaded scene, which would make the screenshot a lie.
    time.sleep(45)

    # Panels cover a third of the frame and are not what is being reviewed.
    for label in ("Hide panels",):
        try:
            page.get_by_role("button", name=label).click(timeout=5_000)
            time.sleep(1.5)
        except Exception:
            print(f"  ({label} not clickable — leaving panels up)", flush=True)

    shots = []
    page.screenshot(path=str(OUT / "world-01-spawn.png"))
    shots.append("world-01-spawn.png")

    # Look around from the spawn point. The walker only accepts input under
    # pointer lock, so drive the camera by clicking into the canvas first.
    # The "Click to walk" overlay is a real button covering the canvas; clicking
    # the canvas hits the overlay instead. Click the overlay itself to lock.
    try:
        page.get_by_role("button", name="Click to walk").click(timeout=8_000)
    except Exception:
        page.locator("canvas").first.click(timeout=8_000, force=True)
    try:
        time.sleep(1.5)
        # Look DOWN first: the spawn faces the horizon and an outdoor capture
        # is mostly sky from eye height.
        for i, (dx, dy) in enumerate([(0, 220), (700, 0), (700, 0)], start=2):
            page.mouse.move(800, 450)
            page.mouse.move(800 + dx, 450 + dy, steps=12)
            time.sleep(2.0)
            name = f"world-{i:02d}-look.png"
            page.screenshot(path=str(OUT / name))
            shots.append(name)
    except Exception as exc:
        print(f"  (look-around skipped: {exc})", flush=True)

    stats = page.evaluate("""() => {
        const t = document.body.innerText;
        const m = t.match(/tris[\\s\\S]{0,40}/);
        return m ? m[0].replace(/\\s+/g, ' ').trim() : '';
    }""")
    print(f"  HUD: {stats}", flush=True)
    browser.close()

print(f"\nwrote {len(shots)} screenshots to {OUT}")
for s in shots:
    print(f"  {OUT / s}")
if errors:
    print(f"\n{len(errors)} console error(s):")
    for e in errors[:6]:
        print(f"  {e[:160]}")
