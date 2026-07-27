"""Live acceptance gate for the Spark editor's interaction layer.

Nothing else in this repo can test the viewer: its entire input layer lives
inside one useEffect keyed on [url], there is no DOM test runner (vitest is
pure-function only), and tsc+eslint cannot see a mode that stays armed or a
panel with no background. This drives the REAL app with real keystrokes and
real clicks and asserts against the real DOM.

Usage (from the repo root, with splatlab running on :3416):

    python3 tools/verify-editor-live.py <job_id> [--paint]

--paint adds the langfield-only checks (brush, class picking, Esc-keeps-
selection); it needs a scene with a LIVE language field. Known-good pair:
splat_7f98469203 (14k gaussians, fast, no langfield) and splat_6b2e82e5
(487k, langfield). Env: VW/VH viewport, SHOTS_DIR for screenshots.

Auth reads PORTAL_TOKEN from the service's RAM-only env file - zero-disk
policy, nothing is written and the token is never printed.

Renders on the real GPU (--use-angle=gl -> RTX 5090). SwiftShader was tried
first and is NOT viable here: a 487k-gaussian scene pushed CDP round trips to
~15s and a 900k one wedged the page outright. The responsiveness probe below
stays as a guard so any future regression fails fast instead of hanging.
"""
import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:3416"
JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_7f98469203"
DO_PAINT = "--paint" in sys.argv
SHOTS = Path(os.environ.get("SHOTS_DIR", "/tmp/splatlab-editor-shots"))
SHOTS.mkdir(exist_ok=True)

TOKEN = [
    l.split("=", 1)[1].strip().strip('"')
    for l in open("/dev/shm/nexus-env-splatlab")
    if l.startswith("PORTAL_TOKEN=")
][0]

results = []
console_errors = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)


def shot(page, name):
    try:
        page.screenshot(path=str(SHOTS / name), timeout=15_000, animations="disabled")
    except PWError:
        print(f"      (screenshot {name} skipped - page never idles)", flush=True)


def panel_text(page):
    try:
        el = page.query_selector(".absolute.left-3.top-3")
        return el.inner_text() if el else ""
    except PWError:
        return ""


def hud_text(page):
    try:
        return page.evaluate("""() => {
          for (const el of document.querySelectorAll('div')) {
            const c = el.className || '';
            if (typeof c === 'string' && c.includes('left-1/2')
                && c.includes('bg-[#0a0f1a]')) return el.innerText;
          }
          return '';
        }""")
    except PWError:
        return ""


def ensure_paint_armed(page):
    """`b` toggles, so pressing it blind can DISARM. Assert the end state."""
    if "Painting" not in panel_text(page):
        page.keyboard.press("b")
        page.wait_for_timeout(1500)
    return "Painting" in panel_text(page)


def size_input(page, label_word):
    for el in page.query_selector_all("input[aria-label]"):
        if label_word in (el.get_attribute("aria-label") or ""):
            return el
    return None


with sync_playwright() as p:
    # Real GPU, not SwiftShader. Measured on this box: software raster of a
    # 487k-gaussian scene takes ~15s per CDP round trip (unusable and flaky);
    # --use-angle=gl binds the RTX 5090 and the same page stays interactive.
    browser = p.chromium.launch(args=[
        "--use-gl=angle", "--use-angle=gl", "--enable-gpu", "--ignore-gpu-blocklist",
    ])
    VW, VH = int(os.environ.get("VW", 1280)), int(os.environ.get("VH", 800))
    ctx = browser.new_context(
        viewport={"width": VW, "height": VH},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"},
    )
    page = ctx.new_page()
    page.set_default_timeout(20_000)
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: console_errors.append(f"PAGEERROR {e}"))

    print(f"--- {JOB} (paint checks: {DO_PAINT}) ---", flush=True)
    print(f"viewport {VW}x{VH}", flush=True)
    page.goto(f"{BASE}/view/{JOB}", wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector("canvas", timeout=90_000)
    print("canvas present, settling...", flush=True)
    time.sleep(6)

    # Responsiveness probe — if the render loop has eaten the main thread,
    # every later check would hang instead of failing.
    t0 = time.time()
    try:
        page.evaluate("1+1")
        alive = time.time() - t0
    except PWError:
        alive = None
    check("page stays responsive under the render loop", alive is not None and alive < 10,
          f"{alive:.2f}s round trip" if alive else "WEDGED")
    if alive is None:
        browser.close()
        sys.exit(2)

    shot(page, f"{JOB}-01-loaded.png")
    check("scene loads with a canvas", page.query_selector("canvas") is not None)

    page.mouse.click(VW // 2, VH // 2 + 20)  # focus the body, not an input
    time.sleep(0.4)

    # --- C from the View tab: switches tab AND arms -------------------------
    page.keyboard.press("c")
    time.sleep(1.5)
    check("C from View jumps to the Edit tab", "tab=edit" in page.url, page.url.split("?")[-1])
    check("...and arms crop", "Cropping" in panel_text(page))
    check("HUD names the armed tool", "Crop sphere" in hud_text(page),
          hud_text(page)[:60].replace("\n", " / "))
    shot(page, f"{JOB}-02-crop-armed.png")

    # --- [ ] resize, clamped ------------------------------------------------
    el = size_input(page, "radius")
    check("crop radius has a typed numeric box", el is not None)
    if el:
        r0 = float(el.input_value())
        page.keyboard.press("BracketRight")
        time.sleep(0.5)
        r1 = float(size_input(page, "radius").input_value())
        check("] grows the radius", r1 > r0, f"{r0} -> {r1}")
        for _ in range(40):
            page.keyboard.press("BracketRight")
        time.sleep(0.6)
        r_max = float(size_input(page, "radius").input_value())
        for _ in range(90):
            page.keyboard.press("BracketLeft")
        time.sleep(0.6)
        r_min = float(size_input(page, "radius").input_value())
        check("[ ] clamp to the slider bounds instead of running away",
              0 < r_min < r_max < 1e6, f"min {r_min} / max {r_max}")
        check("small radii still print a real number (not 0.00)", r_min > 0, f"{r_min}")

    # --- Esc ladder: placement first, then disarm ---------------------------
    page.mouse.click(VW // 2, VH // 2 + 20)  # place a crop centre
    time.sleep(1.2)
    pt = panel_text(page)
    placed = "Cropping" in pt and ("placed" in hud_text(page) or "would be removed" in pt or "Counting" in pt)
    if placed:
        page.keyboard.press("Escape")
        time.sleep(1.0)
        check("Esc #1 clears the placement but STAYS in the tool", "Cropping" in panel_text(page),
              hud_text(page)[:60].replace("\n", " / "))
        page.keyboard.press("Escape")
        time.sleep(1.0)
        check("Esc #2 disarms the tool", "Cropping" not in panel_text(page))
    else:
        page.keyboard.press("Escape")
        time.sleep(1.0)
        check("Esc disarms the tool", "Cropping" not in panel_text(page),
              "no placement registered (raycast miss) - ladder step 1 untested here")

    # --- shortcut card ------------------------------------------------------
    page.keyboard.press("?")
    time.sleep(1.0)
    body = page.inner_text("body")
    check("? opens the shortcuts card", "shortcuts" in body.lower())
    for row in ["back out one level", "pick class", "crop to box", "redo stroke", "reset view"]:
        check(f"  card lists '{row}'", row in body)
    shot(page, f"{JOB}-03-shortcuts.png")
    page.keyboard.press("?")
    time.sleep(0.6)

    # --- H hides chrome -----------------------------------------------------
    page.keyboard.press("h")
    time.sleep(1.0)
    check("H hides the tool panel", panel_text(page) == "")
    shot(page, f"{JOB}-04-chrome-hidden.png")
    page.keyboard.press("h")
    time.sleep(0.8)
    check("H restores it", panel_text(page) != "")

    # --- modified keys are left to the browser ------------------------------
    prevented = page.evaluate("""() => {
      let seen = null;
      const h = (e) => { seen = e.defaultPrevented; };
      window.addEventListener('keydown', h, false);
      document.body.dispatchEvent(new KeyboardEvent('keydown',
        {key:'s', code:'KeyS', ctrlKey:true, bubbles:true, cancelable:true}));
      window.removeEventListener('keydown', h, false);
      return seen;
    }""")
    check("Ctrl+S is not swallowed by the viewer", prevented is False, f"defaultPrevented={prevented}")

    # --- typing in a field must not drive the camera or fire tool keys ------
    page.keyboard.press("Escape")
    time.sleep(0.4)
    page.keyboard.press("c")
    time.sleep(1.2)
    el = size_input(page, "radius")
    if el:
        el.click()
        el.fill("")
        page.keyboard.type("0.42")
        time.sleep(0.3)
        check("typing '0.42' in the box does not fire tool keys", "Cropping" in panel_text(page),
              "otherwise 4/2 would pick classes and 0 would reset the view")
        page.keyboard.press("Enter")
        time.sleep(0.8)
        got = size_input(page, "radius").input_value()
        check("typed radius commits on Enter", abs(float(got) - 0.42) < 0.02, got)
    page.keyboard.press("Escape")
    time.sleep(0.5)

    # --- paint-only checks --------------------------------------------------
    if DO_PAINT:
        page.keyboard.press("b")
        time.sleep(2.0)
        check("B jumps to Measure and arms paint",
              "tab=measure" in page.url and "Painting" in panel_text(page), page.url.split("?")[-1])
        shot(page, f"{JOB}-05-paint-armed.png")

        for dx, dy in [(0, 0), (30, 18), (-30, -18), (18, -28), (-22, 26)]:
            page.mouse.click(VW // 2 + dx, VH // 2 + 20 + dy)
            time.sleep(0.8)
        time.sleep(2.5)
        hud = hud_text(page)
        painted = "selected" in hud.lower()
        check("painting selects splats (HUD shows the count)", painted, hud[:80].replace("\n", " / "))
        shot(page, f"{JOB}-06-painted.png")

        if painted:
            def sel_count():
                for tok in hud_text(page).replace("\n", " ").split():
                    t = tok.replace(",", "")
                    if t.isdigit():
                        return int(t)
                return None

            before = sel_count()
            page.keyboard.press("Escape")
            time.sleep(1.2)
            after = hud_text(page)
            check("Esc disarms paint", "Painting" not in panel_text(page))
            low = after.lower()
            check("...and the selection SURVIVES Esc (RToony's rule)",
                  "selected" in low and "not committed" in low, after[:80].replace("\n", " / "))
            shot(page, f"{JOB}-07-esc-keeps-selection.png")

            ensure_paint_armed(page)
            page.keyboard.press("Control+z")
            time.sleep(1.2)
            undone = sel_count()
            check("Ctrl+Z undoes a stroke",
                  undone is not None and before is not None and undone < before, f"{before} -> {undone}")
            page.keyboard.press("Control+Shift+z")
            time.sleep(1.2)
            redone = sel_count()
            check("Ctrl+Shift+Z redoes it", redone == before, f"{undone} -> {redone} (was {before})")

            page.keyboard.press("Delete")
            time.sleep(1.2)
            check("Del discards the selection", "selected" not in hud_text(page).lower())

            # Orbit WHILE the brush is armed (RToony, 07-26): right-drag must
            # move the camera and left-drag must still paint, so you can frame
            # the next stroke without disarming.
            check("brush re-arms for the orbit checks", ensure_paint_armed(page))
            shot_a = page.screenshot(timeout=15_000)
            page.mouse.move(VW // 2 + 60, VH // 2)
            page.mouse.down(button="right")
            for i in range(12):
                page.mouse.move(VW // 2 + 60 - 14 * i, VH // 2, steps=1)
            page.mouse.up(button="right")
            time.sleep(1.5)
            check("right-drag orbits while the brush is armed",
                  page.screenshot(timeout=15_000) != shot_a)
            n_before = sel_count() or 0
            page.mouse.move(VW // 2, VH // 2 + 10)
            page.mouse.down()
            for i in range(10):
                page.mouse.move(VW // 2 + 9 * i, VH // 2 + 10 + 5 * i, steps=1)
            page.mouse.up()
            time.sleep(2.5)
            check("...and left-drag still paints rather than orbiting",
                  (sel_count() or 0) > n_before, f"{n_before} -> {sel_count()}")

        ensure_paint_armed(page)
        cls = page.query_selector("button:text-is('Class')")
        if cls:
            cls.click()
            time.sleep(1.5)
            page.mouse.click(VW // 2, VH // 2 + 20)
            time.sleep(0.6)
            page.keyboard.press("4")
            time.sleep(1.0)
            t4 = panel_text(page)
            page.keyboard.press("1")
            time.sleep(1.0)
            t1 = panel_text(page)
            check("1-9 change the selected class", bool(t4) and bool(t1) and t4 != t1,
                  "panel differs between class 4 and class 1")
            check("selection stays cyan, not the class colour",
                  "shown" in panel_text(page).lower() and "cyan" in panel_text(page).lower(),
                  "class-coloured selection is invisible on same-coloured scene content")
            shot(page, f"{JOB}-08-class-brush.png")
        else:
            check("1-9 change the selected class", False, "Class toggle not found")

    real_errors = [e for e in console_errors if "favicon" not in e.lower()]
    check("no console/page errors", not real_errors, "; ".join(real_errors[:2])[:140])
    browser.close()

failed = [n for n, ok, _ in results if not ok]
print("\n" + "=" * 62)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:\n  - " + "\n  - ".join(failed))
(SHOTS / f"results-{JOB}.json").write_text(json.dumps(results, indent=2))
sys.exit(1 if failed else 0)
