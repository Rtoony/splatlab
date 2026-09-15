import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveSelection(page, request, base, jobId, output, errors, expectedGeneration) {
  const width = process.env.SPATIAL_PROOF_GPU === "1" ? 960 : 480;
  const height = width * 2 / 3;
  await page.addInitScript(({ width, height }) => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = `div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: ${width}px !important; height: ${height}px !important; min-height: ${height}px !important; }`;
    document.head.append(style);
  }), { width, height });
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Selection proof API failed: ${response.status()} ${await response.text()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration) throw new Error("Inspect the active generation before this read-only proof");
  if (initial.state.hidden_capture_slugs.length) throw new Error("Residual comparison requires an initially unremoved captured baseline");
  const studies = await read(await request.get(api + "/selection-reviews"));
  const study = studies.reviews.filter(item => item.result && !item.stale).at(-1);
  if (!study) throw new Error("No sealed current selection study exists");
  const cameras = [study.cameras.find(camera => camera.split === "check"), study.cameras[0]];
  page.setDefaultTimeout(90000);
  const waitViewer = async (count, remaining = false) => {
    await page.waitForFunction(({ count, remaining }) => {
      const walker = window.__sceneStudioWalker;
      if (!walker?.backdrop || !walker.spark) return false;
      const mask = walker.captureInspectionMask;
      return count === null ? !mask : mask?.reduce((total, value) => total + Number(value === (remaining ? 255 : 0)), 0) === count;
    }, { count, remaining }, { timeout: 90000 });
    return page.evaluate(() => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      return { collision_triangles: walker.colliderTris, pluck_state: walker.pluckState() };
    });
  };
  const capture = async (label, sourceCamera) => {
    const result = await page.evaluate(async ({ sourceCamera, width, height }) => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      if (sourceCamera) {
      const transform = new walker.camera.matrix.constructor();
      transform.set(...sourceCamera.world_to_camera.flat()).invert();
      const values = transform.elements;
      walker.camera.position.setFromMatrixPosition(transform);
      walker.camera.up.set(-values[4], -values[5], -values[6]).normalize();
      walker.camera.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
      walker.camera.fov = 2 * Math.atan(sourceCamera.height / (2 * sourceCamera.parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      }
      walker.spark.autoUpdate = false;
      const started = performance.now();
      const settle = async () => {
        while (walker.spark.sorting || walker.spark.sortDirty || walker.spark.sortTimeoutId !== -1 || walker.spark.updateTimeoutId !== -1) {
          if (performance.now() - started > 30000) throw new Error("Selection renderer did not settle");
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      };
      await settle();
      for (let warmup = 0; warmup < 2; warmup++) {
        await walker.spark.update({ scene: walker.scene, camera: walker.camera });
        await settle();
        walker.renderer.render(walker.scene, walker.camera);
      }
      if (walker.spark.current.mappingVersion !== walker.spark.display.mappingVersion) throw new Error("Selection mapping is not displayed");
      const context = walker.renderer.getContext();
      if (context.drawingBufferWidth !== width || context.drawingBufferHeight !== height) throw new Error("Selection canvas dimensions changed");
      const samples = new Uint8Array(width * height * 4);
      context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, samples);
      const colors = new Set();
      for (let offset = 0; offset < samples.length; offset += 4) colors.add(`${samples[offset]},${samples[offset + 1]},${samples[offset + 2]}`);
      if (context.isContextLost() || colors.size < 16) throw new Error("Selection evidence rendered blank");
      return { image: sourceCamera?.image_key || "framed-selection", distinct_colors: colors.size,
        png: walker.renderer.domElement.toDataURL("image/png") };
    }, { sourceCamera, width, height });
    await writeFile(path.join(output, label + ".png"), Buffer.from(result.png.split(",")[1], "base64"));
    const { png, ...metadata } = result;
    return metadata;
  };
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  const baseline = await waitViewer(null);
  await page.getByLabel("Retained selection review").selectOption(study.review_id);
  await capture("context-before", cameras[0]);
  const views = [];
  for (const [candidate, count] of [[false, study.core_count], [true, study.result.candidate_count]]) {
    const name = candidate ? "refined" : "current";
    await page.getByRole("button", { name: `Inspect ${name} selection in 3D`, exact: true }).click();
    const state = await waitViewer(count);
    if (JSON.stringify(state) !== JSON.stringify(baseline)) throw new Error("Inspection changed removal state or collision");
    if (!await page.getByRole("checkbox", { name: "Walking / collision", exact: true }).isDisabled()) throw new Error("Walking must be disabled in selection isolation");
    if (!await page.getByRole("checkbox", { name: "Captured appearance", exact: true }).isDisabled()) throw new Error("Selection isolation must require captured appearance");
    for (const [index, camera] of cameras.entries()) views.push({ mode: name, rows: count, ...await capture(`${name}-${index}`, camera) });
    await page.getByRole("button", { name: "Frame selected splats", exact: true }).click();
    views.push({ mode: name, rows: count, ...await capture(`${name}-framed`, null) });
    await page.getByRole("checkbox", { name: "Show remaining captured appearance", exact: true }).check();
    const residualState = await waitViewer(count, true);
    if (JSON.stringify(residualState) !== JSON.stringify(baseline)) throw new Error("Remaining-capture inspection changed removal state or collision");
    if (!await page.getByRole("checkbox", { name: "Walking / collision", exact: true }).isDisabled()) throw new Error("Residual diagnostics must disable walking");
    const removalButton = page.getByRole("button", { name: "Propose refined removal", exact: true });
    if (await removalButton.count() && !await removalButton.isDisabled()) throw new Error("Remaining-capture inspection must not authorize a removal proposal");
    for (const [index, camera] of cameras.entries()) views.push({ mode: `${name}-capture-only-residual`, rows: count,
      ...await capture(`${name}-residual-${index}`, camera) });
    await page.getByRole("checkbox", { name: "Show remaining captured appearance", exact: true }).uncheck();
    await waitViewer(count);
  }
  await page.getByRole("button", { name: "Exit selection inspection", exact: true }).click();
  const restored = await waitViewer(null);
  if (JSON.stringify(restored) !== JSON.stringify(baseline)) throw new Error("Exiting inspection did not restore normal state");
  await capture("context-restored", cameras[0]);
  if (Buffer.compare(await readFile(path.join(output, "context-before.png")), await readFile(path.join(output, "context-restored.png")))) throw new Error("Selection inspection did not restore a pixel-identical settled frame");
  if (study.result.recipe?.visibility_method === "alpha-compositing-color-gradient/v1") {
    if (await page.getByText(/Experimental footprint refinement:/).count() !== 1) throw new Error("Missing footprint recipe explanation");
    const associated = study.result.views.filter(view => view.selected_mask !== null).length;
    if (await page.getByText(/Captured alpha mass inside mask retained by selection:/).count() !== associated) throw new Error("Missing per-view footprint diagnostics");
  }
  await page.getByLabel("Selection evidence layer").selectOption("candidate");
  const photographs = page.locator('img[alt^="candidate selection evidence"]');
  if (await photographs.count() !== study.cameras.length) throw new Error("Missing photographic evidence views");
  for (const photograph of await photographs.all()) {
    await photograph.scrollIntoViewIfNeeded();
    await photograph.evaluate(image => image.decode());
  }
  await page.getByRole("heading", { name: "Inspect and refine captured selection" }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, "selection-evidence-panel.png"), fullPage: true });
  const final = await read(await request.get(api));
  if (JSON.stringify(final.active) !== JSON.stringify(initial.active) || JSON.stringify(final.state) !== JSON.stringify(initial.state)) throw new Error("Read-only selection proof changed the scene");
  const report = { review_id: study.review_id, active: final.active, core_rows: study.core_count,
    candidate_rows: study.result.candidate_count, photos_loaded: await photographs.count(), views,
    active_and_collision_unchanged: true, inspection_guards_verified: true, pixel_identical_restoration: true,
    canvas: { width, height }, page_errors: errors,
    scope: "Read-only private-fixture selection inspection and capture-only residual diagnostics with walking disabled; no background fill, active removal, physical-object recall, collision rebuild or performance acceptance" };
  await writeFile(path.join(output, "selection-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length) throw new Error("Selection proof reported browser errors");
}
