#!/usr/bin/env python3
"""P1 live proof: shove a prop in the real walker, verify it PERSISTED.

Drives the world page on the real GPU (same --use-angle=gl posture as
shoot-world.py), teleports to a physics prop, walks into it, and then asks
the SERVER whether the prop's pose landed in the player seam — the receipt
is the persisted pose delta, not a pixel judgment. Screenshots are written
for the human eyeball on top.

Usage: tools/prove-physics-live.py [job_id] [slug] [out_dir]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_aea04ab3"
SLUG = sys.argv[2] if len(sys.argv) > 2 else "cardboard-box-2"
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "/tmp/physics-proof")
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


def get_poses() -> dict:
    req = urllib.request.Request(
        f"{BASE}/api/splat/jobs/{JOB}/world/interactions",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    return (((payload.get("resolved") or {}).get("player") or {})
            .get("physics") or {}).get("poses") or {}


OUT.mkdir(parents=True, exist_ok=True)
before = get_poses()
print(f"poses BEFORE: {json.dumps(before) or '{}'}")

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

    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    print("world loading (splat streams; physics engages during load)...", flush=True)
    time.sleep(30)

    body = page.evaluate("() => document.body.innerText")
    if "Prop physics unavailable" in body:
        sys.exit("FAIL: the page reports physics degraded to display-only")

    # Teleport next to the prop via its row's Go button (faces the element).
    row = page.locator(f"text={SLUG}").first
    go = row.locator("xpath=ancestor::*[.//button[normalize-space()='Go']][1]") \
            .get_by_role("button", name="Go").first
    go.click(timeout=10_000)
    time.sleep(2)
    page.screenshot(path=str(OUT / "01-before-shove.png"))

    # Lock the pointer and walk INTO the box for a second and a half.
    try:
        page.get_by_role("button", name="Click to walk").click(timeout=8_000)
    except Exception:
        page.locator("canvas").first.click(timeout=8_000, force=True)
    time.sleep(1.0)
    page.keyboard.down("w")
    time.sleep(1.6)
    page.keyboard.up("w")
    time.sleep(1.0)
    page.screenshot(path=str(OUT / "02-after-shove.png"))
    page.keyboard.press("Escape")

    # The page saves disturbed poses every 5 s; give it one full cycle.
    time.sleep(7)
    browser.close()

after = get_poses()
print(f"poses AFTER:  {json.dumps(after) or '{}'}")

moved = []
for slug, pose in after.items():
    prior = before.get(slug)
    if prior is None:
        moved.append((slug, None, pose["position"]))
    else:
        delta = math.dist(prior["position"], pose["position"])
        if delta > 1e-4:
            moved.append((slug, round(delta, 4), pose["position"]))

print(f"screenshots: {OUT}/01-before-shove.png  {OUT}/02-after-shove.png")
if errors:
    print(f"{len(errors)} console error(s):")
    for e in errors[:5]:
        print(f"  {e[:160]}")
if not moved:
    sys.exit("FAIL: no prop pose was disturbed/persisted — the shove did not land")
for slug, delta, pos in moved:
    tag = "newly disturbed" if delta is None else f"moved {delta}u"
    print(f"PROOF: {slug} {tag}, persisted at {[round(v, 3) for v in pos]}")
