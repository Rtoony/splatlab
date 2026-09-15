import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveFixedWalking(page, request, base, jobId, output, errors, expectedGeneration) {
  assert.equal(jobId, "splat_c0ffee");
  assert.equal(process.env.SPATIAL_PROOF_GPU, "1");
  const endpoint = new URL(base);
  assert.ok(endpoint.protocol === "http:" && endpoint.hostname === "127.0.0.1" && endpoint.port);
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => { assert.ok(response.ok(), `Read-only proof request failed: ${response.status()}`); return response.json(); };
  const save = (name, value) => writeFile(path.join(output, name), JSON.stringify(value, null, 2));
  const initial = await read(await request.get(api));
  assert.equal(initial.active.generation, expectedGeneration);
  assert.equal(initial.legacy_sources_changed, false);
  assert.equal(initial.state.viewer.units, "meters");
  await page.route("**/api/**", route => route.request().method() === "GET" ? route.continue() : route.abort());
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.id = "fixed-walking-proof-size";
    style.textContent = "canvas {width:960px!important;height:640px!important}";
    document.head.append(style);
  }));
  page.setDefaultTimeout(60000);
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__sceneStudioWalker?.bvh && window.__sceneStudioWalker.params.bodySizing === "fixed-metric");
  const canvas = page.locator("canvas").first();
  await canvas.evaluate(node => { node.parentElement.style.width = "960px"; node.parentElement.style.height = "640px"; });
  await page.evaluate(() => window.__sceneStudioWalker.resize());
  const initialPose = await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    walker.stop();
    return { position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray() };
  });
  const frames = [];
  async function capture(label) {
    const value = await page.evaluate(async () => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const spark = walker.spark;
      if (spark) spark.autoUpdate = false;
      const started = performance.now();
      const settle = async () => {
        while (spark && (spark.sorting || spark.sortDirty || spark.sortTimeoutId !== -1 || spark.updateTimeoutId !== -1)) {
          if (performance.now() - started > 30000) throw new Error("Fixed-walking frame did not settle");
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      };
      await settle();
      for (let pass = 0; pass < 2; pass++) {
        if (spark) await spark.update({ scene: walker.scene, camera: walker.camera });
        await settle();
        walker.renderer.render(walker.scene, walker.camera);
      }
      const graphics = walker.renderer.getContext();
      if (graphics.isContextLost() || graphics.getError() !== graphics.NO_ERROR) throw new Error("Walking frame has a WebGL error");
      return { png: walker.renderer.domElement.toDataURL("image/png"), width: graphics.drawingBufferWidth,
        height: graphics.drawingBufferHeight, position: walker.camera.position.toArray(), body: walker.walkingBody, flying: walker.isFlying };
    });
    const { png, ...record } = value;
    assert.match(png, /^data:image\/png;base64,/);
    assert.equal(record.width, 960);
    assert.equal(record.height, 640);
    const bytes = Buffer.from(png.split(",")[1], "base64");
    await writeFile(path.join(output, label + ".png"), bytes);
    const frame = { label, ...record, sha256: createHash("sha256").update(bytes).digest("hex") };
    frames.push(frame);
    return frame;
  }
  const baseline = await capture("walking-baseline");
  const studies = await read(await request.get(api + "/architectural-edits"));
  const failed = studies.edits.find(item => item.result?.metrics?.protected_conflicts?.includes("red-bicycle"));
  assert.ok(failed, "The retained real conflict result is missing");
  await page.getByLabel("Retained architectural edit").selectOption(failed.architecture_id);
  assert.equal(await page.getByRole("button", { name: "Preview paired doorway and room", exact: true }).isEnabled(), false);
  const guidance = page.getByLabel("Architectural adjustments needed");
  assert.match(await guidance.textContent(), /red-bicycle/);
  assert.match(await guidance.textContent(), /plastic-storage-container/);
  await guidance.screenshot({ path: path.join(output, "walking-architectural-refusal.png") });
  await page.getByLabel("Total player height (m)", { exact: true }).fill("1.7");
  await page.getByLabel("Player radius (m)", { exact: true }).fill("0.22");
  await page.waitForFunction(() => {
    const body = window.__sceneStudioWalker.walkingBody;
    return Math.abs(body.totalHeightM - 1.7) < 1e-9 && body.radiusM === .22;
  });
  const scan = await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    const outcomes = [];
    for (let axisX = -1.3; axisX <= .81; axisX += .1) {
      for (let axisZ = -1.3; axisZ <= 1.81; axisZ += .1) {
        const result = walker.assessWalkingStart(walker.camera.position.clone().set(axisX, -.1, axisZ));
        outcomes.push({ x: axisX, z: axisZ, ok: result.ok, message: result.message });
        if (!result.ok) continue;
        for (const direction of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const route = [];
          for (let step = 0; step <= 6; step++) {
            const sample = walker.assessWalkingStart(walker.camera.position.clone().set(axisX + direction[0] * .05 * step, -.1, axisZ + direction[1] * .05 * step));
            if (!sample.ok || Math.abs(sample.floorY - result.floorY) > .03) break;
            route.push(sample);
          }
          if (route.length === 7) return { outcomes, route, direction };
        }
      }
    }
    return { outcomes, route: null };
  });
  await save("walking-admission-scan.json", scan);
  assert.ok(scan.route, "No short full-size supported route found in the retained baseline; do not shrink or bypass admission");
  await page.evaluate(({ route, direction }) => {
    const walker = window.__sceneStudioWalker;
    walker.camera.position.fromArray(route[0].position);
    walker.camera.lookAt(route[0].position[0] + direction[0], route[0].position[1], route[0].position[2] + direction[1]);
  }, scan);
  await page.getByLabel("Walking / collision", { exact: true }).check();
  const admitted = await capture("walking-admitted");
  assert.equal(admitted.flying, false);
  assert.equal(admitted.body.radiusM, .22);
  assert.ok(Math.abs(admitted.body.totalHeightM - 1.7) < 1e-9);
  await page.bringToFront();
  const pointerInput = await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    return { connected: walker.controls.domElement.isConnected, same_canvas: walker.controls.domElement === walker.renderer.domElement,
      same_document: walker.controls.domElement.ownerDocument === document, root_is_document: walker.controls.domElement.getRootNode() === document,
      focused: document.hasFocus(), visibility: document.visibilityState, active_element: document.activeElement?.tagName };
  });
  await save("walking-pointer-input.json", pointerInput);
  assert.ok(pointerInput.connected && pointerInput.same_canvas && pointerInput.same_document && pointerInput.root_is_document);
  await page.getByRole("button", { name: "Explore with WASD · Esc to release", exact: true }).click();
  await page.waitForFunction(() => window.__sceneStudioWalker.controls.isLocked, null, { timeout: 10000 });
  await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    walker.setParams({ walkSpeedMps: .5 });
    const original = walker.renderer.render.bind(walker.renderer);
    const samples = [];
    window.__fixedWalkingRenderProbe = { samples, original };
    walker.renderer.render = (...args) => {
      original(...args);
      if (args[0] === walker.scene && samples.length < 500) samples.push({ time: performance.now(),
        position: walker.camera.position.toArray(), radius: walker.capsuleRadius, height: walker.capsuleHeight + walker.capsuleRadius,
        flying: walker.isFlying, pointer_locked: walker.controls.isLocked, grounded: walker.grounded });
    };
    if (walker.spark) walker.spark.autoUpdate = true;
    walker.start();
  });
  try {
    await page.keyboard.down("KeyW");
    await page.waitForTimeout(400);
  } finally {
    await page.keyboard.up("KeyW");
  }
  const samples = await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    walker.stop();
    const probe = window.__fixedWalkingRenderProbe;
    walker.renderer.render = probe.original;
    delete window.__fixedWalkingRenderProbe;
    return probe.samples;
  });
  await save("walking-rendered-trace.json", samples);
  assert.ok(samples.length >= 5, "Too few actual rendered movement samples");
  assert.ok(samples.every(sample => !sample.flying && sample.pointer_locked && Math.abs(sample.radius - .22) < 1e-9 && Math.abs(sample.height - 1.7) < 1e-9));
  const travel = Math.hypot(samples.at(-1).position[0] - admitted.position[0], samples.at(-1).position[2] - admitted.position[2]);
  assert.ok(travel > .05 && travel < .4, `Unexpected real short-walk travel: ${travel}`);
  await capture("walking-after-short-movement");
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => !document.pointerLockElement && !window.__sceneStudioWalker.controls.isLocked, null, { timeout: 10000 });
  await page.getByLabel("Walking / collision", { exact: true }).uncheck();
  await page.evaluate(floor => {
    const walker = window.__sceneStudioWalker;
    walker.camera.position.y = floor + .05;
  }, scan.route[0].floorY);
  await page.getByLabel("Walking / collision", { exact: true }).click();
  assert.equal(await page.getByLabel("Walking / collision", { exact: true }).isChecked(), false);
  assert.equal(await page.evaluate(() => window.__sceneStudioWalker.isFlying), true);
  assert.match(await page.getByLabel("Walking admission").textContent(), /raised surface|No upward-facing floor/);
  await page.getByLabel("Walking admission").screenshot({ path: path.join(output, "walking-unsafe-start-refusal.png") });
  await page.getByLabel("Player radius (m)", { exact: true }).fill("0");
  assert.equal(await page.getByLabel("Walking / collision", { exact: true }).isEnabled(), false);
  await page.getByLabel("Player radius (m)", { exact: true }).fill("0.32");
  await page.getByLabel("Total player height (m)", { exact: true }).fill("2.02");
  await page.waitForFunction(() => {
    const body = window.__sceneStudioWalker.walkingBody;
    return body.radiusM === .32 && Math.abs(body.totalHeightM - 2.02) < 1e-9;
  });
  await page.evaluate(pose => {
    const walker = window.__sceneStudioWalker;
    walker.camera.position.fromArray(pose.position);
    walker.camera.quaternion.fromArray(pose.quaternion);
  }, initialPose);
  assert.equal((await capture("walking-restored-view")).sha256, baseline.sha256);
  const final = await read(await request.get(api));
  assert.deepEqual(final.active, initial.active);
  assert.deepEqual(final.state, initial.state);
  assert.deepEqual(final.proposals.map(item => item.proposal_id), initial.proposals.map(item => item.proposal_id));
  await page.setViewportSize({ width: 390, height: 844 });
  await canvas.evaluate(node => { node.style.removeProperty("width"); node.parentElement.style.removeProperty("width"); });
  await page.evaluate(() => { document.getElementById("fixed-walking-proof-size")?.remove(); window.__sceneStudioWalker.resize(); });
  const dimensions = page.getByRole("group", { name: "Explicit player dimensions" });
  await dimensions.screenshot({ path: path.join(output, "walking-mobile-controls.png") });
  const layout = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
  assert.ok(layout.scroll <= layout.width + 1);
  assert.deepEqual(errors, []);
  await save("fixed-walking-browser-proof.json", { status: "passed", base: initial.active, final: final.active,
    body: admitted.body, admission_candidates: scan.outcomes.length, rendered_samples: samples.length, horizontal_travel_m: travel,
    trace_sha256: createHash("sha256").update(JSON.stringify(samples, null, 2)).digest("hex"),
    frames, exact_pixel_restore: true, exact_state_unchanged: true, page_errors: errors, mobile: layout,
    scope: "Real fixed-body admission/refusal and short RAF-rendered baseline movement. Not doorway traversal, repaired-floor acceptance or a completed connected-room demo." });
}
