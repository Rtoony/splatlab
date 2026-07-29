#!/usr/bin/env python3
"""W3 Lane-A live proof: bake a restyle permanently, prove preview==bake in
pixels, and prove revert restores the capture.

Five receipts, all measured on the 5090 (ANGLE/GL, never SwiftShader):
  1. the PREVIEW visibly changes the world        dist(preview, base) > 8
  2. the BAKE survives with the overlay cleared   dist(baked, base) > 8
     (without the walker's bakedLook rule this frame EQUALS base — the splat
      backdrop photograph would hide the baked shell)
  3. preview and bake are the same look           dist(baked, preview) < 12
     (the "same numbers by construction" promise, finally in pixels)
  4. the look lives in the ATLAS now              zero restyleMaterial meshes
  5. revert returns the capture                   dist(reverted, base) < 3

Every frame is captured with unlit:false — the preview's class material is a
LIT MeshStandardMaterial, so any comparison under as-captured (no-lights)
rendering would be a comparison against black.

Uses a FANTASY class (cobblestone) on the shell so the taxonomy lane is
proven end-to-end, plus a tint on the red bicycle. Leaves the job exactly as
found: bake -> revert -> restore the original restyle document.
"""
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_aea04ab3"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/bake-proof")
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


def dist(a: tuple, b: tuple) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    page = browser.new_context(
        viewport={"width": 1600, "height": 900},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"}).new_page()

    def api(method: str, path: str = "", body=None):
        return page.evaluate(
            """async ([m, url, b, tok]) => {
                 const r = await fetch(url, {method: m,
                   headers: {'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + tok},
                   body: b ? JSON.stringify(b) : undefined});
                 return {status: r.status, body: await r.json()};
               }""",
            [method, f"/api/splat/jobs/{JOB}/world/restyle{path}", body, TOKEN])

    def settle_and_shoot(name: str) -> Path:
        page.wait_for_selector("canvas", timeout=120_000)
        time.sleep(20)
        # Lit mode for every frame, camera parked and held (applied twice so
        # the walker's own update cannot drift it).
        pose = """() => {
          const w = window.__worldWalker;
          w.setParams({unlit: false});
          w.flying = true;
          w.camera.position.set(1.2, 1.5, 2.2);
          w.camera.lookAt(0, 0.6, -1.5);
          w.camera.updateMatrixWorld(true);
        }"""
        page.evaluate(pose); time.sleep(1.2); page.evaluate(pose); time.sleep(0.6)
        shot = OUT / name
        page.screenshot(path=str(shot))
        return shot

    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)

    # As-found state, to restore at the end. Refuse a world that already
    # carries a bake — this proof would stack on top of it.
    found = api("GET")
    assert found["status"] == 200, found
    original = found["body"]["restyle"]
    if found["body"].get("baked_elements"):
        sys.exit(f"{JOB} already carries a baked restyle "
                 f"({found['body']['baked_elements']}) — revert it first")

    r = api("DELETE")
    assert r["status"] == 200, r
    page.reload(wait_until="domcontentloaded")
    f_base = settle_and_shoot("01-base.png")

    r = api("PUT", body={
        "elements": {
            "shell": {"material": "cobblestone", "material_scale": 1.5},
            "red-bicycle": {"tint": "#ff3355"},
        },
        "lighting": {"preset": "as-captured", "intensity": 1.0},
    })
    assert r["status"] == 200, r
    print("restyle stored:", r["body"]["restyle"]["elements"], flush=True)
    page.reload(wait_until="domcontentloaded")
    f_preview = settle_and_shoot("02-preview.png")

    r = api("POST", "/bake", {"units_per_metre": 1.0})
    assert r["status"] == 200, r
    baked_payload = r["body"]["restyle"]
    assert baked_payload["restyle"]["elements"] == {}, baked_payload
    assert set(baked_payload.get("baked_elements") or []) == {"shell", "red-bicycle"}, \
        baked_payload.get("baked_elements")
    print("baked:", [e["slug"] for e in r["body"]["elements"]],
          "| versions:", [e["versioned_as"] for e in r["body"]["elements"]], flush=True)

    page.reload(wait_until="domcontentloaded")
    f_baked = settle_and_shoot("03-baked.png")

    # Object truth: the look is in the atlas now — no restyle shader anywhere,
    # and the bike's material colour is back to plain white (its tint is baked
    # into the texture, not multiplied at render time).
    state = page.evaluate("""() => {
      const w = window.__worldWalker;
      let restyled = 0, meshes = 0;
      for (const el of w.elements) el.object.traverse((o) => {
        if (!o.isMesh) return;
        meshes++;
        if (o.userData.restyleMaterial) restyled++;
      });
      const bike = w.elements.find((e) => e.slug === 'red-bicycle');
      let tinted = null;
      bike.object.traverse((o) => {
        if (o.isMesh && tinted === null) tinted = o.material.color.getHexString();
      });
      return {restyled, meshes, tinted};
    }""")
    print("post-bake:", state, flush=True)
    assert state["meshes"] > 0 and state["restyled"] == 0, state
    assert state["tinted"] == "ffffff", state

    r = api("POST", "/bake/revert")
    assert r["status"] == 200, r
    # Revert restored the pre-bake OVERLAY; drop it so the reverted frame is
    # comparable to the bare base frame.
    r = api("DELETE")
    assert r["status"] == 200, r
    page.reload(wait_until="domcontentloaded")
    f_reverted = settle_and_shoot("04-reverted.png")

    # Leave the job as found.
    if original.get("elements") or (original.get("lighting") or {}).get("preset", "as-captured") != "as-captured":
        r = api("PUT", body={"elements": original["elements"],
                             "lighting": original["lighting"]})
        assert r["status"] == 200, r
        print("original restyle document restored", flush=True)

    base = mean_rgb(f_base)
    preview = mean_rgb(f_preview)
    baked = mean_rgb(f_baked)
    reverted = mean_rgb(f_reverted)
    d_preview = dist(preview, base)
    d_baked = dist(baked, base)
    d_agree = dist(baked, preview)
    d_revert = dist(reverted, base)
    print(f"mean rgb  base={tuple(round(v, 1) for v in base)}  "
          f"preview={tuple(round(v, 1) for v in preview)}  "
          f"baked={tuple(round(v, 1) for v in baked)}  "
          f"reverted={tuple(round(v, 1) for v in reverted)}", flush=True)
    print(f"preview moved {d_preview:.1f} | bake holds {d_baked:.1f} | "
          f"preview<->bake agree within {d_agree:.1f} | "
          f"revert restored to within {d_revert:.1f}", flush=True)
    assert d_preview > 8, "the preview restyle did not visibly change the world"
    assert d_baked > 8, ("the bake is INVISIBLE — the backdrop rule "
                         "(setBakedLook) is not holding")
    assert d_agree < 12, "preview and bake disagree — the promise is broken"
    assert d_revert < 3, "revert did NOT return the capture"
    print("screenshots:", OUT, flush=True)
    browser.close()
