#!/usr/bin/env python3
"""PW live proof: a disturbed prop takes its photograph ghost with it.

  1. GET /world/pluck            -> doc is fresh, maps >= 1 prop
  2. open /world/{job}           -> backdrop upgrades to fmt=langweb (row-
                                    identical), walker installs the doc
  3. teleport to the prop, lock the pointer (the click-to-walk card covers
     frame centre), let the 1.46M-splat sorter SETTLE, and measure ambient
     drift — every pixel verdict is relative to it
  4. disturb through physics     -> applyPoses (the saved-game path: body
                                    pose + disturbed=true); the frame loop
                                    must auto-pluck WITHOUT being asked.
                                    At rest the mesh renders exactly over its
                                    ghost, so only a MOVED prop exposes it —
                                    which is the whole point of the feature.
  5. unpluck / re-pluck at the old spot -> the ghost visibly RETURNS (pixel
     band delta vs drift) and visibly leaves again; the sampled live opacity
     goes 0 -> 1 -> 0, byte-exact.

Usage: prove-pluck-live.py [JOB] [OUT_DIR]   (default: splat_5c1db781)
"""
import os, sys, time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from playwright.sync_api import sync_playwright

# Measurement band (1600x900 viewport): frame centre, where teleportTo puts
# the prop — clear of the side panels and the top-right HUD. The click-to-walk
# card also lives here, so the proof enters pointer lock before measuring.
BAND = (500, 250, 1100, 700)  # left, top, right, bottom


def band_delta(a: Path, b: Path) -> float:
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    l, t, r, btm = BAND
    return float(np.abs(ia[t:btm, l:r] - ib[t:btm, l:r]).mean())

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_5c1db781"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/pluck-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")


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
OUT.mkdir(parents=True, exist_ok=True)

# 0. Pristine state: the walker persists disturbed poses every 5 s, so a
#    previous run's shoved props reload mid-air and tumble — ambient motion
#    that drowns the pixel verdicts (measured: drift 11+ vs a 16 ghost) and
#    pre-plucks the target at load. Each run starts from the photograph.
r = requests.delete(f"{BASE}/api/splat/jobs/{JOB}/world/state",
                    headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60)
assert r.status_code in (200, 404), f"state reset: HTTP {r.status_code} {r.text[:200]}"
print(f"0  STATE: saved poses cleared (HTTP {r.status_code})", flush=True)

# 1. The doc itself, with the server's staleness verdict.
r = requests.get(f"{BASE}/api/splat/jobs/{JOB}/world/pluck",
                 headers={"Authorization": f"Bearer {TOKEN}"}, timeout=60)
assert r.status_code == 200, f"pluck doc: HTTP {r.status_code} {r.text[:200]}"
doc = r.json()
assert doc["ok"] and not doc["stale"], f"pluck doc stale/broken: {doc.get('reasons')}"
slugs = list((doc["pluck"] or {}).get("elements", {}))
assert slugs, "pluck doc maps no props"
print(f"1  DOC: fresh, {len(slugs)} prop(s) mapped: {slugs}", flush=True)

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    page = browser.new_context(viewport={"width": 1600, "height": 900},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"}).new_page()
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)

    # 2. Wait for the langweb backdrop (can be the RAW ply — hundreds of MB
    #    locally) and the installed doc: sampleOpacity goes non-null only once
    #    packed data exists AND the row-count guard passed.
    state = {}
    deadline = time.time() + 300
    while time.time() < deadline:
        state = page.evaluate(
            "() => window.__worldWalker ? window.__worldWalker.pluckState() : {}")
        if state and all(v["sampleOpacity"] is not None for v in state.values()):
            break
        time.sleep(1.0)
    assert state, "walker never installed the pluck doc"
    assert all(v["sampleOpacity"] is not None for v in state.values()), \
        f"backdrop rows never became addressable (fmt=web still?): {state}"

    # The MOST-splats prop: a 528-row bicycle in a 1.46M-splat photograph is
    # pixel-invisible (measured Δ0.00); the 4k-row table is not.
    unplucked = [s for s in slugs if not state[s]["plucked"]]
    assert unplucked, f"every mapped prop is already plucked at load: {state}"
    target = max(unplucked, key=lambda s: state[s]["rows"])
    before = state[target]
    assert before["sampleOpacity"] > 0, f"row already transparent: {before}"
    print(f"2  BEFORE: {target} rows={before['rows']} plucked=False "
          f"sampleOpacity={before['sampleOpacity']:.3f}", flush=True)

    def settle(tag: str, budget_s: float = 30.0) -> float:
        """Wait until two 1.2s-apart band screenshots agree; return the drift."""
        deadline = time.time() + budget_s
        prev = OUT / f"settle-{tag}-a.png"
        page.screenshot(path=str(prev))
        drift = float("inf")
        while time.time() < deadline:
            time.sleep(1.2)
            cur = OUT / f"settle-{tag}-b.png"
            page.screenshot(path=str(cur))
            drift = band_delta(prev, cur)
            if drift < 0.5:
                break
            prev.write_bytes(cur.read_bytes())
        return drift

    # 3. Frame the ghost with a DEAD-STILL camera. teleportTo leaves the
    #    player under physics — the camera kept settling (HUD pos drifted
    #    every frame) and that motion swamped every pixel verdict. Fly mode
    #    with no keys held moves nothing, so: lock the pointer (kills the
    #    click-to-walk card), switch to fly, aim at the prop's PHOTOGRAPHED
    #    location, and HIDE its mesh — at rest the mesh renders exactly over
    #    its ghost, so the eye-toggle leaves the ghost alone in frame.
    page.mouse.click(800, 450)
    aimed = page.evaluate("""(slug) => {
      const w = window.__worldWalker;
      const el = w.elements.find((e) => e.slug === slug);
      if (!el) return "no element";
      w.flying = true;  // no gravity, no keys: the camera holds still
      const c = el.center, s = el.size, box = w.sceneBox;
      const wc = [(box.min.x + box.max.x) / 2, (box.min.z + box.max.z) / 2];
      let dx = wc[0] - c[0], dz = wc[1] - c[2];
      const len = Math.hypot(dx, dz) || 1; dx /= len; dz /= len;
      const back = Math.max(s[0], s[2]) * 1.2 + 2.0;
      w.camera.position.set(c[0] + dx * back, c[1] + 1.2, c[2] + dz * back);
      w.camera.lookAt(c[0], c[1], c[2]);
      w.setElementVisible(slug, false);
      return "ok";
    }""", target)
    assert aimed == "ok", f"could not frame {target}: {aimed}"
    time.sleep(0.5)
    drift = settle("pre")
    print(f"3  SETTLE: fly-cam framed on the ghost, mesh hidden; ambient band "
          f"drift {drift:.2f} mean-RGB", flush=True)
    page.screenshot(path=str(OUT / "01-ghost-alone.png"))

    # 4. The real trigger: disturb it through the physics saved-game path
    #    (sets the body pose AND disturbed=true) — the frame loop must pluck
    #    it WITHOUT being asked, and the only visible change in the band is
    #    the ghost itself vanishing (the mesh is hidden).
    moved = page.evaluate("""(slug) => {
      const w = window.__worldWalker;
      const el = w.elements.find((e) => e.slug === slug);
      if (!el) return "no element";
      const p = el.object.position;
      const poses = {}; poses[slug] = {
        position: [p.x + 1.5, p.y + 0.3, p.z],
        quaternion: [0, 0, 0, 1] };
      w.applyPhysicsPoses(poses);
      return "ok";
    }""", target)
    assert moved == "ok", f"could not disturb {target}: {moved}"
    deadline = time.time() + 15
    after = {}
    while time.time() < deadline:
        after = page.evaluate("(s) => window.__worldWalker.pluckState()[s]", target)
        if after["plucked"]:
            break
        time.sleep(0.5)
    assert after.get("plucked"), f"disturbed prop never got plucked: {after}"
    assert after["sampleOpacity"] == 0, f"plucked but rows still visible: {after}"
    print(f"4  TRIGGER: physics-disturbed {target} auto-plucked "
          f"(sampleOpacity=0 in the live packed buffer)", flush=True)
    time.sleep(1.2)
    page.screenshot(path=str(OUT / "02-ghost-gone.png"))
    ghost = band_delta(OUT / "01-ghost-alone.png", OUT / "02-ghost-gone.png")
    print(f"   PIXELS: ghost Δ{ghost:.2f} vs drift {drift:.2f} "
          f"(mean-RGB over the band)", flush=True)
    assert ghost > max(3 * drift, 0.5), \
        f"the ghost never visibly vanished (Δ{ghost:.2f} vs drift {drift:.2f})"

    # 5. unpluck restores the bytes (receipt: sampled opacity returns) — and
    #    the frame loop RE-plucks a still-disturbed prop within its 0.25s
    #    tick: the ghost of a moved prop must never come back on its own.
    state = page.evaluate("""(slug) => {
      const w = window.__worldWalker;
      w.unpluckElement(slug);
      return w.pluckState()[slug];
    }""", target)
    assert not state["plucked"] and state["sampleOpacity"] > 0.5, \
        f"byte restore failed: {state}"
    deadline = time.time() + 5
    reasserted = {}
    while time.time() < deadline:
        reasserted = page.evaluate("(s) => window.__worldWalker.pluckState()[s]", target)
        if reasserted["plucked"]:
            break
        time.sleep(0.25)
    assert reasserted.get("plucked") and reasserted["sampleOpacity"] == 0, \
        f"one-way invariant broke — unplucked disturbed prop stayed: {reasserted}"
    print(f"5  INVARIANT: unpluck restored the bytes "
          f"(opacity {state['sampleOpacity']:.3f}), and the frame loop "
          f"re-plucked the still-disturbed prop on its own", flush=True)
    page.evaluate("(s) => window.__worldWalker.setElementVisible(s, true)", target)
    browser.close()

print(f"ALL RECEIPTS LANDED — the ghost leaves with the prop. screenshots: {OUT}")
