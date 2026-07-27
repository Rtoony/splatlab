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

try:  # optional: the paint-visibility check needs it, nothing else does
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None
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
    # NOTE: innerText returns CSS-UPPERCASED text for headings and labels
    # (tracking-[0.28em] uppercase). Compare case-insensitively — this has
    # produced three separate false failures in this file already.
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


def lit_pixels(page, vw, vh):
    """Count selection-cyan pixels in the CANVAS area.

    The gate used to assert only that the HUD count went up. This catches the
    stronger failure: painting registers splats and draws NOTHING.

    SCOPE, honestly: this does NOT catch depth-burying (a sweep whose plane
    drifts off the surface so the paint lands behind it). Measured on
    splat_6b2e82e5 the buggy and fixed builds BOTH clear this threshold,
    because that scene has little depth variation. Catching that needs a scene
    with strongly receding ground (splat_3aaf8067) and a check that the lit
    pixels track the swept screen path — not yet written.

    Read from Playwright's screenshot, never the canvas: a WebGL canvas drawn
    into a 2D context reads back BLANK without preserveDrawingBuffer, which
    produced a confident false alarm here.
    """
    if Image is None:
        return None
    import io

    im = Image.open(io.BytesIO(page.screenshot(timeout=20_000))).convert("RGB")
    im = im.crop((int(vw * 0.27), 110, im.width, im.height - 60))  # skip the tool panel
    n = 0
    for r, g, b in im.getdata():
        if g > 90 and b > 110 and g > r + 35 and b > r + 45:
            n += 1
    return n


def click_when_enabled(page, selector, tries=6):
    """Click a control once it stops being disabled, or give up and say so.
    The lane toggles are deliberately disabled while paint is pending, so a
    plain .click() here hangs for the full timeout instead of failing."""
    for _ in range(tries):
        el = page.query_selector(selector)
        if el and not el.is_disabled():
            el.click()
            page.wait_for_timeout(900)
            return True
        discard_selection(page)
        page.wait_for_timeout(400)
    return False


def _sel_from_hud(page):
    for tok in hud_text(page).replace(chr(10), ' ').split():
        t = tok.replace(',', '')
        if t.isdigit():
            return int(t)
    return None


def discard_selection(page):
    """Clear any pending paint. The lane toggles are deliberately DISABLED
    while a selection is live (switching would carry it into the wrong class),
    so the gate has to finish a selection before it can change lanes."""
    # Del is the discard shortcut and is separately asserted above, so it is
    # the reliable path — the HUD button lives inside a pointer-events-none
    # wrapper and is fussier to hit.
    for _ in range(3):
        if not _sel_from_hud(page):
            return
        # Move focus OFF any text field first: the viewer deliberately ignores
        # shortcuts while you're typing, so a Delete sent from the label input
        # is silently swallowed (which is correct, and cost a stalled run).
        page.evaluate("() => (document.activeElement instanceof HTMLElement) && document.activeElement.blur()")
        page.wait_for_timeout(150)
        page.keyboard.press("Delete")
        page.wait_for_timeout(900)


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

            # ...and it must be VISIBLE, not just counted.
            ensure_paint_armed(page)
            before_px = lit_pixels(page, VW, VH)
            page.mouse.move(VW // 2 - 90, VH // 2 + 60)
            page.mouse.down()
            for i in range(26):
                page.mouse.move(VW // 2 - 90 + 11 * i, VH // 2 + 60 + 4 * i, steps=1)
                page.wait_for_timeout(45)
            page.mouse.up()
            time.sleep(2.5)
            after_px = lit_pixels(page, VW, VH)
            if before_px is None:
                print("SKIP  swept paint is visible on screen   [Pillow not installed]", flush=True)
            else:
                check("swept paint is VISIBLE on screen, not buried behind the surface",
                      after_px - before_px > 800, f"cyan px {before_px} -> {after_px}")

        ensure_paint_armed(page)
        # Switching class with paint still selected must be INTERCEPTED - that
        # is how a grass->asphalt->vegetation pass became one 293,004-splat
        # vegetation commit that clobbered the earlier asphalt by precedence.
        ensure_paint_armed(page)
        if sel_count():
            active_label = "paint · label" in panel_text(page).lower()
            other = "Class" if active_label else "Label"
            lane = page.query_selector(f"button:text-is('{other}')")
            check("the OTHER lane toggle is blocked while paint is pending",
                  lane is not None and lane.is_disabled(), f"other lane = {other}")

        # --- commit guards (added after a paint pass pinned 186,710 splats of
        # flat ground to a mistyped label without the operator noticing) ------
        ensure_paint_armed(page)
        pt = panel_text(page)
        check("panel heading names the active paint target", "paint · " in pt.lower(),
              [l for l in pt.split("\n") if "PAINT" in l][:1])
        check("the active lane says what it actually does",
              "teaches" in pt.lower() or "generative" in pt.lower())
        if sel_count():
            check("commit block describes WHAT is selected", "About to label" in pt,
                  [l[:80] for l in pt.split("\n") if "About to label" in l][:1])
        lab = page.query_selector("input[placeholder*='Label']")
        if lab and not lab.is_disabled():
            lab.click()
            lab.fill("bikr")
            time.sleep(1.0)
            pt = panel_text(page)
            check("an unknown label is flagged before commit", "new to this scene" in pt.lower())
            btn = page.query_selector("button:has-text('Pin ')")
            check("commit button names the act, not just a count", btn is not None,
                  btn.inner_text().replace("\n", " ") if btn else "no Pin button")
            opts = page.eval_on_selector_all(
                "#splatlab-known-labels option", "els => els.map(e => e.value)")
            check("label field autocompletes from the scene vocabulary", len(opts) > 0,
                  f"{len(opts)} known labels")
            lab.fill("")
        else:
            check("commit guards present", False, "label input not found")

        # The Class toggle is deliberately disabled while paint is pending, so
        # this waits for it rather than hanging on a plain .click().
        if click_when_enabled(page, "button:text-is('Class')"):
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
            check("1-9 change the selected class", False,
                  "Class lane never became clickable (selection stuck?)")

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
