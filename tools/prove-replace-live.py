#!/usr/bin/env python3
"""Triage slice-2 live proof: Replace-with-asset, the fire-hydrant pattern.

  1. POST /world/elements/umbrella/replace?asset=dungeon-torch — the torch is
     scaled to the umbrella's height and lands at its floor point, carrying
     replaces=umbrella
  2. the walker: the authored torch is VISIBLE at the umbrella's spot; the
     captured umbrella mesh is hidden; its photograph ghost is auto-plucked
     (the walker acts on the replaces link, no hand-holding)
  3. DELETE the replacement, reload: the umbrella is fully back (no pluck,
     photograph intact) — replace is perfectly reversible

Usage: prove-replace-live.py [JOB]   (default: splat_716a9122)
"""
import os
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_716a9122"
OUT = Path("/tmp/replace-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")
CAPTURED = "umbrella"
ASSET = "dungeon-torch"

drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = next((l.split("=", 1)[1].strip() for l in drop.read_text().splitlines()
              if l.startswith("PORTAL_TOKEN=")), os.environ.get("PORTAL_TOKEN", ""))
if not TOKEN:
    sys.exit("PORTAL_TOKEN unavailable")
OUT.mkdir(parents=True, exist_ok=True)
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = f"{BASE}/api/splat/jobs/{JOB}"

requests.delete(f"{API}/world/elements/re-{CAPTURED}", headers=AUTH, timeout=60)
requests.delete(f"{API}/world/state", headers=AUTH, timeout=60)

r = requests.post(f"{API}/world/elements/{CAPTURED}/replace",
                  params={"asset": ASSET}, headers=AUTH, timeout=120)
assert r.status_code == 200, f"replace: {r.status_code} {r.text[:300]}"
body = r.json()
pose = body["receipt"]["pose"]
assert body["replaces"] == CAPTURED and body["slug"] == f"re-{CAPTURED}"
print(f"1  REPLACE: {ASSET} -> re-{CAPTURED} at {pose['floor_point']} "
      f"scale {pose['scale']}", flush=True)


def open_and_read(page):
    deadline = time.time() + 240
    while time.time() < deadline:
        state = page.evaluate("""([cap, rep]) => {
          const w = window.__worldWalker;
          if (!w || !w.elements || !w.elements.length) return null;
          if (!(w.backdrop && w.backdrop.packedSplats
                && w.backdrop.packedSplats.numSplats > 0)) return null;
          const c = w.elements.find((e) => e.slug === cap);
          const a = w.elements.find((e) => e.slug === rep);
          const ps = w.pluckState();
          return {
            captured: c ? { visible: c.visible } : null,
            authored: a ? { visible: a.visible,
                            center: a.center, provenance: a.provenance } : null,
            plucked: ps[cap] ? ps[cap].plucked : null,
          };
        }""", [CAPTURED, f"re-{CAPTURED}"])
        if state and state["captured"] is not None:
            return state
        time.sleep(1.0)
    raise AssertionError("world never became readable")


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    ctx = browser.new_context(viewport={"width": 1600, "height": 900},
                              extra_http_headers=AUTH)
    page = ctx.new_page()
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    # Give the 4 Hz pluck tick a moment after readiness.
    state = open_and_read(page)
    deadline = time.time() + 10
    while time.time() < deadline and not state["plucked"]:
        time.sleep(0.5)
        state = open_and_read(page)
    assert state["authored"] and state["authored"]["visible"], state
    assert state["authored"]["provenance"] == "authored", state
    assert state["captured"]["visible"] is False, state
    assert state["plucked"] is True, f"ghost not plucked: {state}"
    fp = pose["floor_point"]
    ac = state["authored"]["center"]
    assert abs(ac[0] - fp[0]) < 0.3 and abs(ac[2] - fp[2]) < 0.3, (ac, fp)
    print(f"2  WALKER: torch visible at the umbrella's spot "
          f"(centre {[round(v, 2) for v in ac]}), umbrella mesh hidden, "
          f"photograph ghost plucked", flush=True)
    page.evaluate("""(slug) => {
      window.__worldWalker.teleportTo(slug);
    }""", f"re-{CAPTURED}")
    time.sleep(2.0)
    page.screenshot(path=str(OUT / "01-replaced.png"))
    ctx.close()

    # 3. Reversibility.
    r = requests.delete(f"{API}/world/elements/re-{CAPTURED}",
                        headers=AUTH, timeout=60)
    assert r.status_code == 200, r.text
    ctx = browser.new_context(viewport={"width": 1600, "height": 900},
                              extra_http_headers=AUTH)
    page = ctx.new_page()
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    state = open_and_read(page)
    assert state["authored"] is None, state
    assert state["plucked"] is False, f"ghost still plucked after delete: {state}"
    print("3  REVERSIBLE: replacement deleted; the umbrella's photograph "
          "is back untouched", flush=True)
    ctx.close()
    browser.close()

print(f"ALL RECEIPTS LANDED — replace is a two-click fire-hydrant. {OUT}")
