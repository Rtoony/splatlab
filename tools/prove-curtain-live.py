#!/usr/bin/env python3
"""Lane 6 live proof: the curtain fades the photograph SPATIALLY.

  1. PUT the curtain disabled (clean slate), open the walker, dead-still
     fly-cam at the spawn, settle, baseline screenshot
  2. enable a tight sphere around the player via the walker API: the OUTSIDE
     band of the frame fades, the INSIDE band stays — the two-sided verdict
     that proves a spatial boundary, not global dimming
  3. persistence: PUT the same curtain enabled, reload the page — the walker
     reinstalls it from curtain.json (curtainState receipt + pixels again)
  4. PUT disabled (cleanup)

Usage: prove-curtain-live.py [JOB] [OUT_DIR]   (default: splat_716a9122)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_716a9122"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/curtain-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")

# Truck spawn seed (0.42, -0.02, -0.46); a 2.5-unit sphere around it keeps
# nearby ground INSIDE and the tree line OUTSIDE.
CENTER = (0.42, 0.5, -0.46)
RADIUS = 2.5
SOFT = 0.5

INSIDE_BAND = (620, 600, 980, 830)   # lower-centre: ground within the sphere
OUTSIDE_BAND = (40, 80, 360, 460)    # upper-left: capture fringe beyond it


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


def band_delta(a: Path, b: Path, band) -> float:
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    l, t, r, btm = band
    return float(np.abs(ia[t:btm, l:r] - ib[t:btm, l:r]).mean())


def put_curtain(enabled: bool) -> dict:
    doc = {"schema": "dev.splatlab.world-curtain/v1", "version": 1,
           "job_id": JOB, "enabled": enabled, "shape": "sphere",
           "center": list(CENTER), "radius": RADIUS,
           "half_extents": [RADIUS] * 3, "soft_edge": SOFT}
    r = requests.put(f"{API}/world/curtain", json={"curtain": doc},
                     headers=AUTH, timeout=60)
    assert r.status_code == 200, f"PUT curtain: {r.status_code} {r.text[:200]}"
    return r.json()["curtain"]


put_curtain(False)
print("0  RESET: curtain stored disabled", flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])

    def open_world():
        ctx = browser.new_context(viewport={"width": 1600, "height": 900},
                                  extra_http_headers=AUTH)
        page = ctx.new_page()
        page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded",
                  timeout=90_000)
        page.wait_for_selector("canvas", timeout=120_000)
        deadline = time.time() + 300
        while time.time() < deadline:
            ready = page.evaluate("""() => {
              const w = window.__worldWalker;
              return !!(w && w.backdrop && w.backdrop.packedSplats
                        && w.backdrop.packedSplats.numSplats > 0);
            }""")
            if ready:
                break
            time.sleep(1.0)
        assert ready, "backdrop never loaded"
        page.mouse.click(800, 450)  # pointer lock kills the click-to-walk card
        page.evaluate("() => { window.__worldWalker.flying = true; }")
        return ctx, page

    def settle(page, tag: str, budget_s: float = 25.0) -> float:
        deadline = time.time() + budget_s
        prev = OUT / f"settle-{tag}-a.png"
        page.screenshot(path=str(prev))
        drift = float("inf")
        while time.time() < deadline:
            time.sleep(1.2)
            cur = OUT / f"settle-{tag}-b.png"
            page.screenshot(path=str(cur))
            drift = max(band_delta(prev, cur, INSIDE_BAND),
                        band_delta(prev, cur, OUTSIDE_BAND))
            if drift < 0.5:
                break
            prev.write_bytes(cur.read_bytes())
        return drift

    # 1. Baseline.
    ctx, page = open_world()
    drift = settle(page, "pre")
    page.screenshot(path=str(OUT / "01-no-curtain.png"))
    print(f"1  BASELINE: still fly-cam, drift {drift:.2f} mean-RGB", flush=True)

    # 2. Enable in-page: the two-sided pixel verdict.
    page.evaluate("""([cx, cy, cz, r, soft]) => {
      window.__worldWalker.setCurtain({
        enabled: true, shape: "sphere", center: [cx, cy, cz],
        radius: r, halfExtents: [r, r, r], softEdge: soft });
    }""", [*CENTER, RADIUS, SOFT])
    time.sleep(2.0)
    page.screenshot(path=str(OUT / "02-curtained.png"))
    outside = band_delta(OUT / "01-no-curtain.png", OUT / "02-curtained.png",
                         OUTSIDE_BAND)
    inside = band_delta(OUT / "01-no-curtain.png", OUT / "02-curtained.png",
                        INSIDE_BAND)
    print(f"2  PIXELS: outside Δ{outside:.2f} | inside Δ{inside:.2f} "
          f"| drift {drift:.2f}", flush=True)
    assert outside > max(3 * drift, 0.5), \
        f"the fringe never faded (outside Δ{outside:.2f} vs drift {drift:.2f})"
    assert inside < max(2 * drift, outside * 0.25), \
        f"the core faded too — that is global dimming, not a curtain " \
        f"(inside Δ{inside:.2f} vs outside Δ{outside:.2f})"
    ctx.close()

    # 3. Persistence: stored enabled, fresh page load reinstalls it.
    put_curtain(True)
    ctx, page = open_world()
    state = page.evaluate("() => window.__worldWalker.curtainState()")
    assert state and state["enabled"] and abs(state["radius"] - RADIUS) < 1e-6, \
        f"curtain did not come back from curtain.json: {state}"
    drift2 = settle(page, "post")
    page.screenshot(path=str(OUT / "03-reloaded.png"))
    outside2 = band_delta(OUT / "01-no-curtain.png", OUT / "03-reloaded.png",
                          OUTSIDE_BAND)
    print(f"3  PERSISTENCE: curtainState round-tripped; outside Δ{outside2:.2f} "
          f"vs uncurtained baseline (drift {drift2:.2f})", flush=True)
    assert outside2 > max(3 * drift2, 0.5), \
        "reloaded page shows no curtain effect"
    ctx.close()
    browser.close()

put_curtain(False)
print("4  CLEANUP: curtain stored disabled", flush=True)
print(f"ALL RECEIPTS LANDED — the photograph ends where told. screenshots: {OUT}")
