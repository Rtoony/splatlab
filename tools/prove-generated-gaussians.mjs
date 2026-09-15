import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveGeneratedGaussians(page, request, base, jobId, output, errors, expectedGeneration) {
  const objectId = process.env.SPATIAL_GAUSSIAN_OBJECT;
  if (jobId !== "splat_c0ffee" || !/^generated_[a-f0-9]{24}$/.test(objectId || "")) throw new Error("Use the retained private generated-object review");
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async url => {
    const response = await request.get(url);
    if (!response.ok()) throw new Error(`Gaussian review API failed: ${response.status()}`);
    return response.json();
  };
  const initial = await read(api);
  if (initial.active.generation !== expectedGeneration) throw new Error("Private baseline changed");
  const parent = await read(`${api}/generated-objects/${objectId}`);
  const originalResult = await read(`${api}/generated-objects/${objectId}?result=true`);
  const studies = await read(`${api}/generated-objects/${objectId}/gaussians`);
  const study = studies.reviews.filter(item => item.status === "needs-review").at(-1);
  if (!study) throw new Error("No completed native-Gaussian frame review");
  await page.route("**/api/**", route => route.request().url().startsWith(base + "/api/") && ["GET", "HEAD"].includes(route.request().method()) ? route.continue() : route.abort());
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(480px, 100%) !important; height: 320px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  page.setDefaultTimeout(90000);
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await page.getByLabel("Retained recovery", { exact: true }).selectOption(parent.recovery_id);
  await page.getByLabel("Retained generated object", { exact: true }).selectOption(objectId);
  const panel = page.getByRole("region", { name: "Native Gaussian frame review", exact: true });
  await panel.getByLabel("Retained Gaussian frame review").selectOption(study.gaussians_id);
  await panel.locator("details").evaluateAll(items => items.forEach(item => { item.open = true; }));
  await panel.locator("img").evaluateAll(items => items.forEach(item => { item.loading = "eager"; }));
  await page.waitForFunction(() => [...document.querySelectorAll('section[aria-label="Native Gaussian frame review"] img')].every(image => image.complete && image.naturalWidth > 0));
  const decoded = await panel.locator("img").evaluateAll(async items => { for (const image of items) await image.decode(); return items.length; });
  if (decoded !== study.views.length * 5) throw new Error("Not all matched Gaussian review images decoded");
  await panel.screenshot({ path: path.join(output, "gaussian-review-panel.png") });
  await panel.locator("details").evaluateAll(items => items.forEach(item => { item.open = false; }));
  await page.waitForFunction(() => window.__sceneStudioWalker && window.__sceneStudioWalker.running === false);
  await panel.getByRole("button", { name: "Open 3D Gaussian comparison", exact: true }).click();
  await page.waitForFunction(count => {
    const canvas = document.querySelector('canvas[aria-label="Generated mesh and splat 3D comparison"]');
    return Number(canvas?.parentElement.dataset.splatCount) === count && Number(canvas.parentElement.dataset.renderCount) > 0;
  }, study.gaussians);
  const canvas = page.getByLabel("Generated mesh and splat 3D comparison", { exact: true });
  const captures = [];
  const capture = async label => {
    await canvas.scrollIntoViewIfNeeded();
    const state = await canvas.evaluate(element => {
      const context = element.getContext("webgl2");
      if (!context || context.isContextLost()) throw new Error("Generated inspection lost its renderer");
      const pixels = new Uint8Array(element.width * element.height * 4);
      context.readPixels(0, 0, element.width, element.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
      const colors = new Set();
      for (let offset = 0; offset < pixels.length; offset += 4) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
      if (colors.size < 16) throw new Error("Generated inspection is blank");
      return { ...element.parentElement.dataset, distinct_colors: colors.size, width: element.width, height: element.height };
    });
    captures.push({ label, ...state });
    await canvas.screenshot({ path: path.join(output, label + ".png") });
  };
  const setMode = async mode => {
    const before = await canvas.evaluate(element => Number(element.parentElement.dataset.renderCount));
    await panel.getByLabel("Generated 3D appearance").selectOption(mode);
    await page.waitForFunction(({ mode, before }) => {
      const host = document.querySelector('canvas[aria-label="Generated mesh and splat 3D comparison"]').parentElement;
      return host.dataset.mode === mode && Number(host.dataset.renderCount) > before;
    }, { mode, before });
  };
  await capture("native-3d-0");
  await setMode("mesh");
  await capture("mesh-3d-0");
  const bounds = await canvas.boundingBox();
  const rendered = await canvas.evaluate(element => Number(element.parentElement.dataset.renderCount));
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2 + 60, bounds.y + bounds.height / 2 + 25, { steps: 4 });
  await page.mouse.up();
  await page.waitForFunction(before => Number(document.querySelector('canvas[aria-label="Generated mesh and splat 3D comparison"]').parentElement.dataset.renderCount) > before, rendered);
  await capture("mesh-3d-1");
  await setMode("splat");
  await capture("native-3d-1");
  await panel.getByRole("button", { name: "Close 3D Gaussian comparison", exact: true }).click();
  if (await canvas.count()) throw new Error("Closing the inspection retained its renderer canvas");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const mobile = await page.evaluate(() => ({ viewport: innerWidth, document_width: document.documentElement.scrollWidth }));
  if (mobile.document_width > mobile.viewport + 1) throw new Error("Gaussian review overflows on mobile");
  await panel.screenshot({ path: path.join(output, "gaussian-review-mobile.png") });
  await page.getByLabel("Revision-pinned 3D scene", { exact: true }).scrollIntoViewIfNeeded();
  await page.waitForFunction(() => window.__sceneStudioWalker?.running === true);
  const final = await read(api);
  const finalResult = await read(`${api}/generated-objects/${objectId}?result=true`);
  if (JSON.stringify(final.active) !== JSON.stringify(initial.active) || finalResult.sha256 !== originalResult.sha256 || errors.length) throw new Error("Read-only Gaussian review changed state or reported errors");
  const report = { status: "passed", gaussians_id: study.gaussians_id, review_sha256: study.sha256, generated_object_id: objectId,
    decoded, captures, mobile, active: final.active, active_unchanged: true, parent_result_unchanged: true, offscreen_room_paused: true, onscreen_room_resumed: true, page_errors: errors,
    scope: "actual Spark all-row import and two-angle mesh/splat inspection; software rendering, not Spark-vs-gsplat pixel parity or room/collision acceptance" };
  await writeFile(path.join(output, "gaussian-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
}
