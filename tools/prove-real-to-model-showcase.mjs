import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {spawnSync} from "node:child_process";

const [directoryArgument, proofDirectory, playwrightModule] = process.argv.slice(2);
assert.ok(directoryArgument && proofDirectory && playwrightModule, "Supply showcase, new proof directory and installed Playwright module");
assert.equal(spawnSync(path.resolve("tools/splatlab-compute-gate.sh"), ["--is-contained"]).status, 0, "Use the shared compute gate for WebGL proof");
const directory = path.resolve(directoryArgument);
const readJson = filename => JSON.parse(fs.readFileSync(filename, "utf8"));
const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const receipt = readJson(path.join(directory, "receipt.json"));
const evaluation = readJson(path.join(directory, "evaluation/receipt.json"));
const validationCount = evaluation.validation.length;
assert(validationCount > 0);
const verify = () => {
  for (const [name, checksum] of Object.entries(receipt.files)) assert.equal(digest(fs.readFileSync(path.join(directory, name))), checksum, name);
};
verify();
fs.mkdirSync(proofDirectory);
const mime = {".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4", ".webm": "video/webm"};
const server = http.createServer((request, response) => {
  let name;
  try { name = decodeURIComponent(new URL(request.url, "http://localhost").pathname).slice(1) || "index.html"; }
  catch { response.writeHead(400).end(); return; }
  if (name === "favicon.ico") { response.writeHead(204).end(); return; }
  if ((request.method !== "GET" && request.method !== "HEAD") || (name !== "receipt.json" && !Object.hasOwn(receipt.files, name))) {
    response.writeHead(404).end(); return;
  }
  response.writeHead(200, {"Content-Type": mime[path.extname(name)] || "application/octet-stream", "Cache-Control": "no-store"});
  if (request.method === "HEAD") response.end();
  else fs.createReadStream(path.join(directory, name)).pipe(response);
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;
const {chromium} = await import(pathToFileURL(playwrightModule).href);
let browser;
const result = {status: "running", errors: [], external_requests: [], checked_views: [], browser_console: []};
try {
  browser = await chromium.launch({headless: true, args: ["--use-gl=angle", "--use-angle=gl", "--enable-webgl", "--ignore-gpu-blocklist", "--disable-dev-shm-usage", "--disable-accelerated-video-decode"]});
  const page = await browser.newPage({viewport: {width: 1480, height: 1150}, deviceScaleFactor: 1});
  page.on("pageerror", error => result.errors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error" || message.type() === "warning") result.browser_console.push(message.text());
  });
  await page.route("**/*", async route => {
    const url = route.request().url();
    if (url.startsWith(base + "/") || /^(blob|data):/.test(url)) await route.continue();
    else { result.external_requests.push(url); await route.abort(); }
  });
  await page.goto(base, {waitUntil: "load"});
  if (receipt.title) assert.equal(await page.locator("h1").textContent(), receipt.title);
  await page.waitForFunction(() => document.querySelector("#viewport").dataset.ready === "true" || document.querySelector("#viewport").dataset.error, null, {timeout: 60000});
  assert.equal(await page.locator("#viewport").getAttribute("data-error"), null, await page.locator("#status").textContent());
  assert.equal(Number(await page.locator("#viewport").getAttribute("data-splats")), evaluation.gaussians.rows);
  const hasMeshes = Boolean(receipt.blender_scene || receipt.blender_alternative || receipt.mesh_deliveries?.length);
  await page.waitForFunction(expected => !document.querySelector("#representation-bar").hidden === expected, hasMeshes);
  assert.equal(await page.locator("#representation-bar").isVisible(), hasMeshes);
  result.before_video = await page.locator("canvas").evaluate(canvas => ({lost: canvas.getContext("webgl2").isContextLost(), width: canvas.width, height: canvas.height}));
  if (evaluation.method === "gsplat-2dgs") {
    assert.match(await page.locator("#representation-notice").textContent(), /thin-3D approximation/);
    assert.match(await page.locator("#splat-download").textContent(), /preview/);
    result.representation_disclosure = true;
  }
  if (receipt.native_tour) {
    const video = page.locator("#tour-video");
    await video.evaluate(element => element.play());
    await page.waitForFunction(() => document.querySelector("#tour-video").currentTime > .3);
    result.native_tour = await video.evaluate(element => ({width: element.videoWidth, height: element.videoHeight, duration: element.duration, currentTime: element.currentTime}));
    assert.equal(result.native_tour.width, evaluation.validation[0].width);
    assert.equal(result.native_tour.height, evaluation.validation[0].height);
    assert(Math.abs(result.native_tour.duration - 8) < .1);
    await video.evaluate(element => element.pause());
  }
  await page.waitForTimeout(2500);
  result.webgl = await page.locator("canvas").evaluate(canvas => {
    const context = canvas.getContext("webgl2");
    const extension = context.getExtension("WEBGL_debug_renderer_info");
    const pixels = new Uint8Array(canvas.width * canvas.height * 4);
    context.readPixels(0, 0, canvas.width, canvas.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    const colors = new Set();
    for (let offset = 0; offset < pixels.length; offset += 124) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
    return {renderer: extension ? context.getParameter(extension.UNMASKED_RENDERER_WEBGL) : context.getParameter(context.RENDERER), sampled_colors: colors.size, lost: context.isContextLost(), width: canvas.width, height: canvas.height};
  });
  assert.ok(result.webgl.sampled_colors > 100, "Actual splat rendering must not be an empty canvas");
  await page.screenshot({path: path.join(proofDirectory, "live-overview.png"), fullPage: true});
  const beforeCamera = await page.locator("#viewport").getAttribute("data-camera");
  const beforePixels = await page.locator("canvas").evaluate(canvas => canvas.toDataURL());
  await page.locator("#viewport").focus();
  await page.keyboard.press("ArrowRight");
  await page.waitForTimeout(1200);
  assert.notEqual(await page.locator("#viewport").getAttribute("data-camera"), beforeCamera);
  assert.notEqual(await page.locator("canvas").evaluate(canvas => canvas.toDataURL()), beforePixels, "Camera movement must change actual 3D pixels");
  await page.screenshot({path: path.join(proofDirectory, "shifted-3d.png"), fullPage: true});
  await page.locator("#reset").click();
  assert.equal(await page.locator("#viewport").getAttribute("data-camera"), beforeCamera);
  result.live_meshes = [];
  if (hasMeshes) {
    for (const selected of ["mesh-alternative", "mesh"]) {
      if (!await page.locator(`#representation option[value="${selected}"]`).count()) continue;
      await page.locator("#representation").selectOption(selected);
      await page.waitForFunction(value => document.querySelector("#viewport").dataset.representation === value, selected);
      await page.waitForTimeout(1200);
      assert.equal(await page.locator("#viewport").getAttribute("data-camera"), beforeCamera);
      const actual = await page.locator("canvas").evaluate(canvas => {
        const context = canvas.getContext("webgl2");
        const pixels = new Uint8Array(canvas.width * canvas.height * 4);
        context.readPixels(0, 0, canvas.width, canvas.height, context.RGBA, context.UNSIGNED_BYTE, pixels);
        const colors = new Set();
        for (let offset = 0; offset < pixels.length; offset += 124) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
        return {lost: context.isContextLost(), colors: colors.size, pixels: canvas.toDataURL()};
      });
      assert.equal(actual.lost, false);
      assert(actual.colors > 100);
      assert.notEqual(actual.pixels, beforePixels);
      await page.screenshot({path: path.join(proofDirectory, selected + ".png"), fullPage: true});
      result.live_meshes.push({representation: selected, sampled_colors: actual.colors, source_camera_unchanged: true});
    }
    await page.locator("#representation").selectOption("splat");
    await page.waitForFunction(() => document.querySelector("#viewport").dataset.representation === "splat");
  }
  const checkedIndices = [...new Set([0, .25, .5, .75, 1].map(fraction => Math.round(fraction * (validationCount - 1))))];
  for (const index of checkedIndices) {
    await page.locator("#view").selectOption(String(index));
    await page.waitForFunction(width => document.querySelector("#reference").complete && document.querySelector("#reference").naturalWidth === width, evaluation.validation[index].width);
    await page.waitForTimeout(1200);
    assert.equal(await page.locator("#viewport").getAttribute("data-selected-view"), evaluation.validation[index].image);
    await page.screenshot({path: path.join(proofDirectory, `view-${index}.png`), fullPage: true});
    result.checked_views.push(evaluation.validation[index].image);
  }
  await page.waitForTimeout(1500);
  const idleBefore = await page.locator("#viewport").getAttribute("data-render-count");
  await page.waitForTimeout(1500);
  assert.equal(await page.locator("#viewport").getAttribute("data-render-count"), idleBefore, "Idle viewer must release GPU rendering work");
  result.idle_rendering_pauses = true;
  await page.setViewportSize({width: 390, height: 844});
  await page.waitForTimeout(1200);
  await page.screenshot({path: path.join(proofDirectory, "phone.png"), fullPage: true});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  const comparisonPage = await browser.newPage();
  await comparisonPage.goto(base + "/evaluation/index.html");
  assert.equal(await comparisonPage.locator("section").count(), validationCount);
  assert.equal(await page.locator("#comparison-link").textContent(), `All ${validationCount} source/render/error comparisons`);
  for (const [index, section] of (await comparisonPage.locator("section").all()).entries()) {
    assert.equal(await section.locator("img").count(), 3);
    for (const image of await section.locator("img").all()) {
      await image.scrollIntoViewIfNeeded();
      await image.evaluate(image => image.decode());
      assert.equal(await image.evaluate(image => image.naturalWidth), evaluation.validation[index].width);
      assert.equal(await image.evaluate(image => image.naturalHeight), evaluation.validation[index].height);
    }
  }
  if (await page.locator("#reload-live").count()) {
    await page.bringToFront();
    await page.locator("canvas").evaluate(canvas => canvas.getContext("webgl2").getExtension("WEBGL_lose_context").loseContext());
    await page.waitForFunction(() => document.querySelector("#viewport").dataset.error === "WebGL context lost");
    assert.equal(await page.locator("#reload-live").isVisible(), true);
    assert.match(await page.locator("#status").textContent(), /saved camera tour, comparisons and downloads remain available/);
    if (receipt.native_tour) {
      await page.locator("#tour-video").evaluate(async element => { element.currentTime = 1; await element.play(); });
      await page.waitForFunction(() => document.querySelector("#tour-video").currentTime > 1.3);
      await page.locator("#tour-video").evaluate(element => element.pause());
    }
    result.context_loss_fallback = "Simulated WebGL loss exposes reload and keeps native video playable";
    await page.screenshot({path: path.join(proofDirectory, "gpu-unavailable.png"), fullPage: true});
  }
  assert.deepEqual(result.errors, []);
  assert.deepEqual(result.external_requests, []);
  result.mesh_handoffs = [];
  for (const handoff of receipt.mesh_deliveries || []) {
    const response = await page.request.get(base + "/" + handoff.href);
    assert.equal(response.status(), 200);
    const manifest = await response.json();
    assert.equal(manifest.schema, "chanate-external-references/v1");
    for (const asset of manifest.assets) {
      assert.equal(asset.registration, null);
      assert.equal(asset.quality.oriented_triangles_equal, true);
      const meshResponse = await page.request.get(new URL(asset.file, base + "/" + handoff.href).href);
      assert.equal(meshResponse.status(), 200);
      assert.equal(digest(await meshResponse.body()), asset.sha256);
    }
    result.mesh_handoffs.push(handoff.label);
  }
  verify();
  result.status = "passed";
  result.gaussians = evaluation.gaussians.rows;
  result.comparison_images = validationCount * 3;
  result.source_hashes_unchanged = true;
} catch (error) {
  result.status = "failed";
  result.failure = error.stack;
  throw error;
} finally {
  fs.writeFileSync(path.join(proofDirectory, "proof.json"), JSON.stringify(result, null, 2));
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
console.log(JSON.stringify(result));
