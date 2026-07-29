#!/usr/bin/env python3
"""W2-C2 live proof: restyle a captured world, then prove it is reversible.

Three receipts, all pixel-measured on the 5090 (ANGLE/GL, never SwiftShader):
  1. the restyle VISIBLY changes the world (before != after, by mean channel)
  2. the class material lands on the element that asked for it
  3. DELETE restores the capture — after-reset pixels match before, closely

Reversibility is the claim that matters: a restyle is a diff over a capture,
so if reset does not return the original frame the feature is destructive.
"""
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_aea04ab3"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/restyle-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")

drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = next((l.split("=", 1)[1].strip() for l in drop.read_text().splitlines()
              if l.startswith("PORTAL_TOKEN=")), os.environ.get("PORTAL_TOKEN", ""))
if not TOKEN:
    sys.exit("PORTAL_TOKEN unavailable")
OUT.mkdir(parents=True, exist_ok=True)


def mean_rgb(png: Path) -> tuple[float, float, float]:
    """Mean RGB of the canvas region (panels excluded from the sample)."""
    from PIL import Image
    with Image.open(png) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        crop = rgb.crop((int(w * 0.35), int(h * 0.1), int(w * 0.75), int(h * 0.9)))
        px = list(crop.getdata())
    n = len(px)
    return (sum(p[0] for p in px) / n,
            sum(p[1] for p in px) / n,
            sum(p[2] for p in px) / n)


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    page = browser.new_context(
        viewport={"width": 1600, "height": 900},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"}).new_page()

    def api(method: str, body=None):
        return page.evaluate(
            """async ([m, url, b, tok]) => {
                 const r = await fetch(url, {method: m,
                   headers: {'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + tok},
                   body: b ? JSON.stringify(b) : undefined});
                 return {status: r.status, body: await r.json()};
               }""",
            [method, f"/api/splat/jobs/{JOB}/world/restyle", body, TOKEN])

    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    time.sleep(22)

    # Park the camera somewhere with wall and props in frame, and hold it.
    pose = """() => {
      const w = window.__worldWalker;
      w.flying = true;
      w.camera.position.set(1.2, 1.5, 2.2);
      w.camera.lookAt(0, 0.6, -1.5);
      w.camera.updateMatrixWorld(true);
    }"""
    page.evaluate(pose); time.sleep(1.2); page.evaluate(pose); time.sleep(0.6)
    page.screenshot(path=str(OUT / "01-before.png"))

    r = api("PUT", {
        "elements": {
            "shell": {"material": "concrete", "material_scale": 1.5},
            "red-bicycle": {"tint": "#ff3355"},
        },
        "lighting": {"preset": "dungeon", "intensity": 1.2},
    })
    assert r["status"] == 200, r
    print("restyle stored:", r["body"]["restyle"]["elements"], flush=True)

    # Reload: proves the look PERSISTS and is applied on open, not just live.
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("canvas", timeout=120_000)
    time.sleep(20)
    page.evaluate(pose); time.sleep(1.0); page.evaluate(pose); time.sleep(0.6)
    page.screenshot(path=str(OUT / "02-restyled.png"))

    # The panel is the product surface: it must show what is stored.
    panel = page.evaluate("""() => {
      const text = document.body.innerText;
      const sel = [...document.querySelectorAll('select')]
        .map((s) => s.value);
      return {hasPanel: text.includes('RESTYLE') || text.includes('Restyle'),
              selects: sel};
    }""")
    assert panel["hasPanel"], "no Restyle panel in the UI"
    assert "dungeon" in panel["selects"], panel["selects"]
    print("panel shows:", panel["selects"], flush=True)

    # The class material must be on the element that asked for it.
    applied = page.evaluate("""() => {
      const w = window.__worldWalker;
      const el = w.elements.find((e) => e.slug === 'shell');
      let restyled = 0, total = 0;
      el.object.traverse((o) => {
        if (!o.isMesh) return;
        total++;
        if (o.userData.restyleMaterial) restyled++;
      });
      const bike = w.elements.find((e) => e.slug === 'red-bicycle');
      let tinted = null;
      bike.object.traverse((o) => {
        if (o.isMesh && tinted === null) tinted = o.material.color.getHexString();
      });
      return {restyled, total, tinted};
    }""")
    print("shell meshes restyled:", applied["restyled"], "/", applied["total"],
          "| bike colour:", applied["tinted"], flush=True)
    assert applied["restyled"] == applied["total"] and applied["total"] > 0
    assert applied["tinted"] == "ff3355", applied["tinted"]

    r = api("DELETE")
    assert r["status"] == 200, r
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("canvas", timeout=120_000)
    time.sleep(20)
    page.evaluate(pose); time.sleep(1.0); page.evaluate(pose); time.sleep(0.6)
    page.screenshot(path=str(OUT / "03-reset.png"))

    before = mean_rgb(OUT / "01-before.png")
    after = mean_rgb(OUT / "02-restyled.png")
    reset = mean_rgb(OUT / "03-reset.png")
    changed = max(abs(a - b) for a, b in zip(before, after))
    restored = max(abs(a - b) for a, b in zip(before, reset))
    print(f"mean rgb  before={tuple(round(v,1) for v in before)}  "
          f"restyled={tuple(round(v,1) for v in after)}  "
          f"reset={tuple(round(v,1) for v in reset)}", flush=True)
    print(f"changed by {changed:.1f} | restored to within {restored:.1f}", flush=True)
    assert changed > 8, "the restyle did not visibly change the world"
    assert restored < 3, "reset did NOT return the capture — restyle is destructive"
    print("screenshots:", OUT, flush=True)
    browser.close()
