#!/usr/bin/env python3
"""P3 live proof: start the zombie scenario in the real walker, verify the
loop fires (spawn, chase, damage or kills), screenshot the fight."""
import os, re, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_aea04ab3"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/game-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")


def _load_token() -> str:
    """RAM secret store first, env fallback. The store can be unreadable from
    a MAC-confined shell (EACCES despite owner match) — don't crash on the
    read, fall through and say what to do."""
    try:
        for l in Path("/dev/shm/nexus-env-splatlab").read_text().splitlines():
            if l.startswith("PORTAL_TOKEN="):
                return l.split("=", 1)[1].strip()
    except OSError:
        pass
    tok = os.environ.get("PORTAL_TOKEN", "")
    if not tok:
        sys.exit("PORTAL_TOKEN unavailable — /dev/shm/nexus-env-splatlab "
                 "unreadable and env unset; run from a normal shell or "
                 "export PORTAL_TOKEN")
    return tok


TOKEN = _load_token()
OUT.mkdir(parents=True, exist_ok=True)
errors = []


def wait_for_text(page, needles, timeout_s):
    """Poll the page text until any needle appears (replaces the fixed sleeps
    — a freshly rebuilt world can need longer than 4s to spawn wave 1). On
    timeout, screenshot + raise with the HUD tail: a failure must leave
    evidence. (The 2026-07-29 Truck run asserted before any screenshot and
    left nothing to diagnose.)"""
    if isinstance(needles, str):
        needles = (needles,)
    deadline = time.time() + timeout_s
    text = ""
    while time.time() < deadline:
        text = page.evaluate("() => document.body.innerText")
        if any(n in text for n in needles):
            return text
        time.sleep(0.5)
    tag = "-".join(needles).replace(" ", "_").replace("/", "-")[:40]
    page.screenshot(path=str(OUT / f"FAIL-{tag}.png"))
    raise AssertionError(
        f"{needles} never appeared in {timeout_s}s — HUD tail: {text[-300:]}")


with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle","--use-angle=gl","--enable-gpu","--ignore-gpu-blocklist"])
    page = browser.new_context(viewport={"width":1600,"height":900},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"}).new_page()
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)

    wait_for_text(page, "Zombie waves", 90)  # world + HUD up (was: fixed 25s)
    page.get_by_role("button", name="Zombie waves").click(timeout=10_000)
    print("scenario started", flush=True)

    # Rest ends, wave 1 spawns — poll, don't sleep (was: fixed 4s, the line
    # that timed out on the rebuilt Truck world with zero evidence left).
    # The HUD already reads "wave 1/3" during the initial 1.5s rest (the rest
    # phase offsets the wave label) and the spawn count only renders once the
    # phase flips to "wave" — so the spawn receipt is "undead" itself.
    wait_for_text(page, "wave 1/3", 30)
    hud = wait_for_text(page, "undead", 30)
    print("HUD:", [l for l in hud.splitlines() if "wave" in l or "undead" in l or "HP" in l][:3], flush=True)
    page.screenshot(path=str(OUT/"01-wave1.png"))

    # Fight badly for a while: spin and shoot, sampling the HUD after EVERY
    # shot. The receipts (kills, HP, OVERRUN) live on a HUD that disappears
    # the moment the scenario ends, so evidence is accumulated during the
    # fight, never inferred from the final frame — the first cut of this tool
    # read "no 'kills 0' on screen" as a kill after the HUD had already gone.
    # And the loop stops the moment an endgame dialog appears: one more blind
    # click would dismiss it and erase the outcome (observed on the Truck —
    # OVERRUN at ~7s, dialog gone by ~8s).
    min_hp, max_kills, outcome = 100, 0, None
    for i in range(14):
        page.mouse.move(800, 450)
        page.mouse.move(800 + (i % 4 - 2) * 250, 470, steps=6)
        page.mouse.down(); page.mouse.up()
        text = page.evaluate("() => document.body.innerText")
        m = re.search(r"HP\s*\n?\s*(\d+)", text)
        if m: min_hp = min(min_hp, int(m.group(1)))
        m = re.search(r"kills (\d+)", text)
        if m: max_kills = max(max_kills, int(m.group(1)))
        m = re.search(r"(\d+) zombies down", text)
        if m: max_kills = max(max_kills, int(m.group(1)))
        if "OVERRUN" in text or "YOU SURVIVED" in text:
            outcome = "OVERRUN" if "OVERRUN" in text else "SURVIVED"
            break
        time.sleep(0.7)
    page.screenshot(path=str(OUT/"02-fight.png"))
    fps = page.evaluate("() => { const m = document.body.innerText.match(/fps\\s*\\n?\\s*(\\d+)/); return m ? m[1] : '?'; }")
    print(f"fps={fps} min_hp={min_hp} kills={max_kills} outcome={outcome}", flush=True)
    assert min_hp < 100 or max_kills > 0 or outcome, "no combat evidence — loop did not engage"

    # The endgame dialog must be CLICKABLE (review finding: it soft-locked
    # behind the pointer lock + the click-to-walk overlay).
    if outcome:
        page.get_by_role("button", name="Play again").click(timeout=10_000)
        wait_for_text(page, ("wave 1/3", "next wave"), 15)
        print("endgame dialog clickable; scenario restarted", flush=True)
    browser.close()
print(f"screenshots: {OUT}")
if errors:
    print(f"{len(errors)} console errors:"); [print(" ", e[:140]) for e in errors[:5]]
