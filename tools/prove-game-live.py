#!/usr/bin/env python3
"""P3 live proof: start the zombie scenario in the real walker, verify the
loop fires (spawn, chase, damage or kills), screenshot the fight."""
import os, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_aea04ab3"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/game-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")
drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = next((l.split("=",1)[1].strip() for l in drop.read_text().splitlines()
              if l.startswith("PORTAL_TOKEN=")), os.environ.get("PORTAL_TOKEN",""))
if not TOKEN: sys.exit("PORTAL_TOKEN unavailable")
OUT.mkdir(parents=True, exist_ok=True)
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle","--use-angle=gl","--enable-gpu","--ignore-gpu-blocklist"])
    page = browser.new_context(viewport={"width":1600,"height":900},
        extra_http_headers={"Authorization": f"Bearer {TOKEN}"}).new_page()
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded", timeout=90_000)
    page.wait_for_selector("canvas", timeout=120_000)
    time.sleep(25)

    body = page.evaluate("() => document.body.innerText")
    assert "Zombie waves" in body, "start button missing"
    page.get_by_role("button", name="Zombie waves").click(timeout=10_000)
    print("scenario started", flush=True)
    time.sleep(4)  # rest ends, wave 1 spawns

    hud = page.evaluate("() => document.body.innerText")
    assert "wave 1/3" in hud, f"wave HUD missing: {hud[-300:]}"
    assert "undead" in hud, "spawn count missing"
    print("HUD:", [l for l in hud.splitlines() if "wave" in l or "undead" in l or "HP" in l][:3], flush=True)
    page.screenshot(path=str(OUT/"01-wave1.png"))

    # Fight badly for a while: spin and shoot. The deterministic loop test
    # proves skill-play; here we prove the real page wires it all.
    for i in range(14):
        page.mouse.move(800, 450)
        page.mouse.move(800 + (i % 4 - 2) * 250, 470, steps=6)
        page.mouse.down(); page.mouse.up()
        time.sleep(0.7)
    page.screenshot(path=str(OUT/"02-fight.png"))
    final = page.evaluate("() => document.body.innerText")
    took_damage = ("HP" in final) and ("100" not in final.split("HP")[1][:24])
    killed = "kills 0" not in final
    print("after fight:", [l for l in final.splitlines() if "wave" in l or "kills" in l or "OVERRUN" in l or "SURVIVED" in l][:3], flush=True)
    fps = page.evaluate("() => { const m = document.body.innerText.match(/fps\\s*\\n?\\s*(\\d+)/); return m ? m[1] : '?'; }")
    print(f"fps={fps} took_damage={took_damage} killed_something={killed}", flush=True)
    assert took_damage or killed or "OVERRUN" in final, "no combat evidence — loop did not engage"
    browser.close()
print(f"screenshots: {OUT}")
if errors:
    print(f"{len(errors)} console errors:"); [print(" ", e[:140]) for e in errors[:5]]
