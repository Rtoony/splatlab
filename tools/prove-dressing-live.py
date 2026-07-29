#!/usr/bin/env python3
"""W3 Lane-B live proof: author a fantasy asset in Blender, land it through
the audited door, and prove the full set-dressing contract on the 5090.

Receipts, in order (each printed as it lands):
  1. AUTHOR    import_asset -> transform_object -> export with
               bake_world_transform; the exported GLB's POSITION bounds sit at
               the commanded floor point under identity transforms
  2. LAND      POST /world/elements -> 200 authored element; the SAME slug
               re-POSTs 409 (placement invents, never replaces) — live
  3. VISIBLE   locked-pose pixel delta before/after placement; the slug shows
               in the PlacedPanel/ElementsPanel DOM
  4. INTERACT  an authored toggle lights the torch (tint effect) — object
               truth on the walker's material
  5. PERSIST   the "on" state survives a full reload (resolve_state, live)
  6. REBUILD   a real world_collision re-run preserves the authored entry
               (the sticky-registry contract, live)
  7. ROUNDTRIP import_world_element("<placed slug>") pulls it back into
               Blender — a placed asset is a first-class element to the loop
  8. REMOVE    DELETE tombstones it; the frame returns to baseline

Leaves the job as found: the asset is removed, the original interactions
document is restored, and only version history (Blender versions, the
tombstone) remains — which is the audit trail working as designed.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
import glb_check  # noqa: E402
from dcc import blender_workflow as wf  # noqa: E402

JOB = sys.argv[1] if len(sys.argv) > 1 else "splat_aea04ab3"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/dressing-proof")
BASE = os.environ.get("SPLATLAB_BASE", "http://127.0.0.1:3416")
SLUG = "proof-torch-sconce"
MESH_ENV_PYTHON = Path.home() / "miniconda3/envs/dn-splatter-probe/bin/python"

drop = Path("/dev/shm/nexus-env-splatlab")
TOKEN = next((l.split("=", 1)[1].strip() for l in drop.read_text().splitlines()
              if l.startswith("PORTAL_TOKEN=")), os.environ.get("PORTAL_TOKEN", ""))
if not TOKEN:
    sys.exit("PORTAL_TOKEN unavailable")
OUT.mkdir(parents=True, exist_ok=True)
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = f"{BASE}/api/splat/jobs/{JOB}"


def mean_rgb(png: Path, box=(0.35, 0.1, 0.75, 0.9)) -> tuple[float, float, float]:
    from PIL import Image
    with Image.open(png) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        crop = rgb.crop((int(w * box[0]), int(h * box[1]),
                         int(w * box[2]), int(h * box[3])))
        px = list(crop.getdata())
    n = len(px)
    return (sum(p[0] for p in px) / n, sum(p[1] for p in px) / n,
            sum(p[2] for p in px) / n)


# The pose below parks the torch at ~25% screen width; this band sits between
# the left panels (end ~19%) and the centered "Click to walk" card (starts
# ~36%), so the delta measures WORLD pixels, not UI chrome.
TORCH_BOX = (0.20, 0.25, 0.35, 0.85)


def dist(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


# ── as-found state + the placement point ────────────────────────────────────────

manifest = requests.get(f"{API}/world/file", params={"name": "world_manifest.json"},
                        headers=AUTH, timeout=30).json()
if any(e.get("slug") == SLUG for e in manifest.get("elements", [])):
    sys.exit(f"{JOB} already carries {SLUG} — remove it first")
original_interactions = requests.get(f"{API}/world/interactions",
                                     headers=AUTH, timeout=30).json()

job_dir = wf._job_dir(JOB)
probe = json.loads((job_dir / "_world" / "collision_shell.json").read_text())
floor_y = float((probe.get("probe") or {}).get("floor_level_y") or 0.0)
# World-frame (glTF Y-up) target, in front of the proof camera. Blender is
# Z-up: glTF (x, y, z) == Blender (x, -z, y).
gx, gz = 0.3, -1.2
blender_location = [gx, -gz, floor_y]

# ── 1. AUTHOR in Blender ────────────────────────────────────────────────────────

imported = wf.run_action(JOB, "import_asset",
                         {"name": "torch-sconce", "slug": SLUG},
                         note="W3 dressing proof")
dims = imported["result"]["dimensions"]
print(f"1a AUTHOR: imported torch-sconce as asset_{SLUG} "
      f"({dims[0]}x{dims[1]}x{dims[2]} m)", flush=True)
wf.run_action(JOB, "transform_object",
              {"object": f"asset_{SLUG}", "location": blender_location},
              note="place at the proof floor point")
export = wf.export_glb(JOB, object_name=f"asset_{SLUG}",
                       bake_world_transform=True, note="W3 dressing proof")
export_path = job_dir / export["output"]["path"]
bounds = glb_check.position_bounds(export_path)
assert bounds["identity_transforms"], bounds
aabb = bounds["aabb"]
centre = [(lo + hi) / 2 for lo, hi in zip(aabb["min"], aabb["max"])]
assert abs(aabb["min"][1] - floor_y) < 0.02, (aabb, floor_y)
assert abs(centre[0] - gx) < 0.02 and abs(centre[2] - gz) < 0.02, (centre, gx, gz)
print(f"1b AUTHOR: baked export sits at x={centre[0]:.2f} z={centre[2]:.2f} "
      f"floor_y={aabb['min'][1]:.3f} under identity transforms", flush=True)

# ── 2. LAND through the audited door ────────────────────────────────────────────

with export_path.open("rb") as handle:
    files = {"file": (export_path.name, handle, "model/gltf-binary")}
    landed = requests.post(f"{API}/world/elements",
                           params={"slug": SLUG, "role": "static",
                                   "label": "proof torch"},
                           files=files, headers=AUTH, timeout=120)
assert landed.status_code == 200, landed.text
body = landed.json()
assert body["element"]["provenance"] == "authored", body["element"]
print(f"2a LAND: 200, provenance=authored, warnings={body['warnings']}", flush=True)
with export_path.open("rb") as handle:
    dup = requests.post(f"{API}/world/elements",
                        params={"slug": SLUG, "role": "static"},
                        files={"file": (export_path.name, handle,
                                        "model/gltf-binary")},
                        headers=AUTH, timeout=120)
assert dup.status_code == 409, (dup.status_code, dup.text)
print("2b LAND: same slug re-POST -> 409 (placement invents, never replaces)",
      flush=True)

# ── 3-5, 8: pixels + interaction + persistence in the real walker ───────────────

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=gl",
                                      "--enable-gpu", "--ignore-gpu-blocklist"])
    page = browser.new_context(
        viewport={"width": 1600, "height": 900},
        extra_http_headers=AUTH).new_page()

    def settle_and_shoot(name: str) -> Path:
        page.wait_for_selector("canvas", timeout=120_000)
        time.sleep(20)
        # Pixel receipts run in MESH view: the splat backdrop is the
        # photograph, and a 0.44 m torch blended against/behind gaussians is
        # a needle in it — against the reconstructed shell it is unmissable.
        pose = f"""() => {{
          const w = window.__worldWalker;
          w.clearBackdrop();
          w.flying = true;
          w.camera.position.set({gx + 0.6}, {floor_y + 0.7}, {gz + 0.8});
          w.camera.lookAt({gx + 0.45}, {floor_y + 0.22}, {gz - 0.1});
          w.camera.updateMatrixWorld(true);
        }}"""
        page.evaluate(pose); time.sleep(1.2); page.evaluate(pose); time.sleep(0.6)
        shot = OUT / name
        page.screenshot(path=str(shot))
        return shot

    # The "before" frame is a lie-detector: the asset landed on disk already,
    # so before/after must be captured around a RELOAD of the walker.
    # First: a frame with the walker still holding the PRE-placement world is
    # impossible now — instead prove visibility by remove/compare at the end,
    # and by DOM + pixel presence here.
    page.goto(f"{BASE}/world/{JOB}", wait_until="domcontentloaded",
              timeout=90_000)
    f_placed = settle_and_shoot("01-placed.png")
    dom = page.evaluate("() => document.body.innerText")
    assert "proof torch" in dom, "PlacedPanel/ElementsPanel do not show the asset"
    walker_truth = page.evaluate(f"""() => {{
      const el = window.__worldWalker.elements.find((e) => e.slug === {SLUG!r});
      if (!el) return null;
      let meshes = 0;
      el.object.traverse((o) => {{ if (o.isMesh) meshes++; }});
      return {{meshes, visible: el.object.visible}};
    }}""")
    assert walker_truth and walker_truth["meshes"] > 0, walker_truth
    assert walker_truth["visible"] is True, walker_truth
    print(f"3  VISIBLE: walker loaded {SLUG} ({walker_truth['meshes']} mesh"
          f" primitives); DOM shows it", flush=True)

    # 4. INTERACT: author a toggle, flip it on, prove the tint on the object.
    records = list((original_interactions.get("interactions") or {})
                   .get("elements") or [])
    records.append({"slug": SLUG, "verb": "toggle", "prompt": "Light the torch",
                    "states": ["off", "on"], "initial": "off",
                    "effects": {"on": {"tint": "#ffd27f"},
                                "off": {"tint": None}}})
    r = requests.put(f"{API}/world/interactions", json={"elements": records},
                     headers=AUTH, timeout=30)
    assert r.status_code == 200, r.text
    r = requests.post(f"{API}/world/state", json={"slug": SLUG, "state": "on"},
                      headers=AUTH, timeout=30)
    assert r.status_code == 200, r.text

    page.reload(wait_until="domcontentloaded")
    settle_and_shoot("02-lit.png")
    tinted = page.evaluate(f"""() => {{
      const el = window.__worldWalker.elements.find((e) => e.slug === {SLUG!r});
      let hex = null;
      el.object.traverse((o) => {{
        if (o.isMesh && hex === null) hex = o.material.color.getHexString();
      }});
      return hex;
    }}""")
    assert tinted == "ffd27f", f"tint effect did not land: {tinted}"
    print("4  INTERACT: authored toggle 'on' tints the torch #ffd27f in the "
          "walker", flush=True)

    # 5. PERSIST: the state survived that reload (resolve_state, live).
    payload = requests.get(f"{API}/world/interactions", headers=AUTH,
                           timeout=30).json()
    resolved = (payload.get("resolved") or {}).get("applied") or {}
    assert resolved.get(SLUG) == "on", resolved
    print("5  PERSIST: resolved state still 'on' after reload", flush=True)

    # 6. REBUILD: a real collision re-run must preserve the authored entry.
    rebuild = subprocess.run(
        [str(MESH_ENV_PYTHON), str(REPO / "backend/mesh/world_collision.py"),
         str(job_dir)],
        capture_output=True, text=True, timeout=600)
    assert rebuild.returncode == 0, rebuild.stderr[-2000:]
    manifest = requests.get(f"{API}/world/file",
                            params={"name": "world_manifest.json"},
                            headers=AUTH, timeout=30).json()
    torch = next(e for e in manifest["elements"] if e["slug"] == SLUG)
    assert torch["provenance"] == "authored", torch
    assert manifest["counts"]["authored"] >= 1
    print("6  REBUILD: world_collision re-run preserved the authored entry "
          f"(counts.authored={manifest['counts']['authored']})", flush=True)

    # 7. ROUNDTRIP: the placed slug is a first-class element to the MCP loop.
    back = wf.run_action(JOB, "import_world_element", {"slug": SLUG},
                         note="W3 dressing proof round-trip")
    assert back["result"]["faces"] > 0, back["result"]
    print(f"7  ROUNDTRIP: import_world_element({SLUG!r}) -> "
          f"{back['result']['faces']} faces back in Blender", flush=True)

    # 8. REMOVE: tombstone, manifest clean, frame returns to baseline.
    removed = requests.delete(f"{API}/world/elements/{SLUG}", headers=AUTH,
                              timeout=60)
    assert removed.status_code == 200, removed.text
    tombstone = removed.json()["tombstone"]
    assert tombstone and (job_dir / "_world" / tombstone).is_file()
    # Restore the world's own interaction document (leave the job as found).
    r = requests.put(
        f"{API}/world/interactions",
        json={"elements": list((original_interactions.get("interactions") or {})
                               .get("elements") or [])},
        headers=AUTH, timeout=30)
    assert r.status_code == 200, r.text

    page.reload(wait_until="domcontentloaded")
    f_removed = settle_and_shoot("03-removed.png")
    dom = page.evaluate("() => document.body.innerText")
    assert "proof torch" not in dom
    delta = dist(mean_rgb(f_placed, TORCH_BOX), mean_rgb(f_removed, TORCH_BOX))
    assert delta > 2.0, (
        f"placement/removal never changed the frame (delta {delta:.2f}) — "
        "the asset was invisible at the proof pose")
    print(f"8  REMOVE: tombstoned to {tombstone}; frame delta placed->removed "
          f"{delta:.1f} (the torch was really there)", flush=True)
    browser.close()

print("screenshots:", OUT, flush=True)
print("ALL RECEIPTS LANDED", flush=True)
