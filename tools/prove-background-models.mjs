import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveBackgroundModels(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Background model comparison only targets the private fixture");
  const comparison = JSON.parse(await readFile(process.env.SPATIAL_COMPLETION_COMPARISON, "utf8"));
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Background comparison API failed: ${response.status()} ${await response.text()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed
    || comparison.job_id !== jobId || comparison.status !== "needs-review"
    || JSON.stringify(comparison.base) !== JSON.stringify(initial.active) || comparison.candidates.length !== 2) throw new Error("Inspect a fresh private two-candidate comparison first");
  const studies = await read(await request.get(api + "/selection-reviews"));
  const study = studies.reviews.find(item => item.review_id === comparison.selection_review_id);
  const collisions = await read(await request.get(api + "/selection-collisions"));
  const collision = collisions.collisions.find(item => item.collision_id === comparison.collision_id);
  if (!study || study.stale || !collision || collision.stale || collision.result?.verdict !== "PASS_LOCAL_EDIT") throw new Error("Fresh selection and matching local clearance are required");
  const completions = await read(await request.get(api + "/completions"));
  for (const candidate of comparison.candidates) {
    const item = completions.completions.find(item => item.completion_id === candidate.completion_id);
    if (!item || item.stale || item.result?.sha256 !== candidate.result_sha256) throw new Error("A comparison candidate differs from its retained import");
  }
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    Object.defineProperty(window, "__sceneStudioWalker", { configurable: true, set(walker) {
      walker.stop();
      Object.getPrototypeOf(walker).start = function () { this.stop(); };
      Object.defineProperty(window, "__sceneStudioWalker", { value: walker, writable: true, configurable: true });
    } });
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(800px, 100%) !important; height: 533px !important; min-height: 533px !important; }';
    document.head.append(style);
  }));
  page.setDefaultTimeout(90000);
  const selected = study.selected_slug;
  const cameras = [study.cameras.find(camera => camera.split === "check"), study.cameras.find(camera => camera.image_id === 58)];
  const waitState = async slug => {
    await page.waitForFunction(({ selected, slug, core, refined }) => {
      const walker = window.__sceneStudioWalker;
      const state = walker?.pluckState()[selected];
      return walker?.colliderTris > 0 && state?.rows === (slug ? refined : core) && state.plucked === Boolean(slug)
        && state.sampleOpacity === (slug ? 0 : 1) && (!slug || walker.elements.some(item => item.slug === slug && item.object.visible));
    }, { selected, slug, core: study.core_count, refined: study.result.candidate_count });
    return page.evaluate(({ selected, slug }) => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      return { triangles: walker.colliderTris, selected_triangles: walker.elements.find(item => item.slug === selected)?.tris || 0,
        patch_triangles: walker.elements.find(item => item.slug === slug)?.tris || 0, rows: walker.pluckState(),
        hidden_rows: walker.pluckMask?.reduce((total, value) => total + Number(value === 255), 0) || 0 };
    }, { selected, slug });
  };
  const capture = async (label, camera, next = null, blend = 0, isolated = null) => {
    console.log(JSON.stringify({ stage: "capture", label }));
    await page.getByLabel("Revision-pinned 3D scene").scrollIntoViewIfNeeded();
    const view = await page.evaluate(async ({ camera, next, blend, isolated }) => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const transform = new walker.camera.matrix.constructor().set(...camera.world_to_camera.flat()).invert();
      const values = transform.elements;
      walker.camera.position.setFromMatrixPosition(transform);
      walker.camera.up.set(-values[4], -values[5], -values[6]).normalize();
      walker.camera.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
      if (next) {
        transform.set(...next.world_to_camera.flat()).invert();
        const target = walker.camera.clone();
        target.position.setFromMatrixPosition(transform);
        target.up.set(-values[4], -values[5], -values[6]).normalize();
        target.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
        walker.camera.position.lerp(target.position, blend);
        walker.camera.quaternion.slerp(target.quaternion, blend);
      }
      if (isolated) {
        const element = walker.elements.find(item => item.slug === isolated.slug);
        const center = element.box.getCenter(walker.camera.position.clone());
        walker.camera.position.copy(center).add(new walker.camera.position.constructor(Math.cos(isolated.angle), 1.7, Math.sin(isolated.angle)));
        walker.camera.up.set(0, 1, 0);
        walker.camera.lookAt(center);
      }
      walker.camera.fov = 2 * Math.atan(camera.height / (2 * camera.parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      const spark = walker.spark;
      if (!spark && !isolated) throw new Error("Full-context comparison lost captured appearance");
      if (spark) spark.autoUpdate = false;
      const started = performance.now();
      const settle = async () => {
        while (spark && (spark.sorting || spark.sortDirty || spark.sortTimeoutId !== -1 || spark.updateTimeoutId !== -1)) {
          if (performance.now() - started > 30000) throw new Error("Background frame did not settle");
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      };
      await settle();
      for (const frame of [0, 1]) {
        if (spark) await spark.update({ scene: walker.scene, camera: walker.camera });
        await settle();
        walker.renderer.render(walker.scene, walker.camera);
        if (frame === 1 && spark && spark.current.mappingVersion !== spark.display.mappingVersion) throw new Error("Background frame mapping is not displayed");
      }
      const context = walker.renderer.getContext();
      const pixels = new Uint8Array(context.drawingBufferWidth * context.drawingBufferHeight * 4);
      context.readPixels(0, 0, context.drawingBufferWidth, context.drawingBufferHeight, context.RGBA, context.UNSIGNED_BYTE, pixels);
      const colors = new Set();
      for (let offset = 0; offset < pixels.length; offset += 4) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
      if (context.isContextLost() || colors.size < 16) throw new Error("Background comparison frame is blank");
      return { canvas_png: walker.renderer.domElement.toDataURL("image/png"), source_camera: camera.image_id, interpolation: next ? blend : null, isolated,
        position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray(), projection: walker.camera.projectionMatrix.toArray(),
        width: context.drawingBufferWidth, height: context.drawingBufferHeight, distinct_colors: colors.size,
        wireframe: walker.colliderWire.visible, update_render_readback_ms: performance.now() - started };
    }, { camera, next, blend, isolated });
    const { canvas_png: canvasPng, ...metrics } = view;
    if (!canvasPng.startsWith("data:image/png;base64,")) throw new Error("Canvas readback did not encode PNG");
    await writeFile(path.join(output, label + ".json"), JSON.stringify(metrics, null, 2));
    await writeFile(path.join(output, label + ".png"), Buffer.from(canvasPng.split(",")[1], "base64"));
    return metrics;
  };
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  const baseline = await waitState(null);
  await capture("baseline", cameras[0]);
  const previews = [];
  for (const [index, candidate] of comparison.candidates.entries()) {
    await page.getByLabel("Retained recovery", { exact: true }).selectOption(comparison.recovery_id);
    const panel = page.getByRole("region", { name: "Unknown-region completion", exact: true });
    await panel.getByLabel("Retained completion", { exact: true }).selectOption(candidate.completion_id);
    await panel.locator("img").evaluateAll(async images => { for (const image of images) await image.decode(); });
    await panel.locator("details").evaluateAll(items => items.forEach(item => { item.open = true; }));
    await panel.screenshot({ path: path.join(output, `model-${index}-panel.png`) });
    const proposed = page.waitForResponse(response => response.url().endsWith("/studio/proposals") && response.request().method() === "POST");
    await panel.getByRole("button", { name: "Propose removal + completion", exact: true }).click();
    const proposal = await read(await proposed);
    const slug = `completion-${candidate.completion_id.slice(-8)}`;
    const state = await waitState(slug);
    const expected = baseline.triangles - baseline.selected_triangles - collision.result.current.triangles + collision.result.candidate.triangles + 32768;
    if (state.triangles !== expected || state.patch_triangles !== 32768 || state.selected_triangles !== 0 || state.hidden_rows !== study.result.candidate_count) throw new Error("Completion did not change refined visibility, local clearance and support collision together");
    for (const [name, value] of Object.entries(baseline.rows)) {
      if (name !== selected && JSON.stringify(state.rows[name]) !== JSON.stringify(value)) throw new Error("Another captured instance changed");
    }
    const fixed = [];
    for (const [view, camera] of cameras.entries()) fixed.push(await capture(`model-${index}-view-${view}`, camera));
    const moving = [];
    for (const fraction of [.2, .4, .6, .8]) moving.push(await capture(`model-${index}-moving-${fraction}`, cameras[0], cameras[1], fraction));
    await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).check();
    await page.waitForFunction(slug => {
      const visible = window.__sceneStudioWalker?.elements.filter(item => item.visible);
      return visible?.length === 1 && visible[0].slug === slug;
    }, slug);
    for (const angle of [0, 1.2]) await capture(`model-${index}-isolated-${angle}`, cameras[0], null, 0, { slug, angle });
    if (await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).isEnabled()) throw new Error("Isolated view granted full-context review");
    await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).uncheck();
    await waitState(slug);
    await page.getByRole("checkbox", { name: "Collision wireframe", exact: true }).check();
    await capture(`model-${index}-collision`, cameras[0]);
    await page.getByRole("checkbox", { name: "Collision wireframe", exact: true }).uncheck();
    if (JSON.stringify((await read(await request.get(api))).active) !== JSON.stringify(initial.active)) throw new Error("Comparison preview changed the active scene");
    previews.push({ request_id: candidate.request_id, completion_id: candidate.completion_id, proposal, state, fixed, moving });
    if (index === 0) {
      await page.getByRole("button", { name: "Keep unapplied", exact: true }).click();
      await waitState(null);
    }
  }
  const tested = previews.at(-1);
  await writeFile(path.join(output, "comparison-previews.json"), JSON.stringify({ baseline, previews, initial: initial.active }, null, 2));
  await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify({ base: initial.active, preview_revision: tested.proposal.preview_revision }, null, 2));
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  const slug = `completion-${tested.completion_id.slice(-8)}`;
  const reloaded = await waitState(slug);
  if (JSON.stringify(reloaded) !== JSON.stringify(tested.state)) throw new Error("Removal and material completion did not survive reload");
  await capture("applied-reloaded", cameras[0]);
  if (Buffer.compare(await readFile(path.join(output, "model-1-view-0.png")), await readFile(path.join(output, "applied-reloaded.png")))) throw new Error("Applied model changed appearance on reload");
  const applied = await read(await request.get(api));
  if (applied.state.semantics[selected].active || applied.state.capture_partition.collision_id !== comparison.collision_id) throw new Error("Edited selection semantics did not persist");
  const history = initial.history.find(item => item.revision_id === initial.active.revision_id);
  await page.getByRole("button", { name: `${history.operation.kind} · ${history.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
  const restored = await read(await request.get(api));
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state) || JSON.stringify(await waitState(null)) !== JSON.stringify(baseline)) throw new Error("Completion undo changed the baseline state");
  await capture("restored", cameras[0]);
  if (Buffer.compare(await readFile(path.join(output, "baseline.png")), await readFile(path.join(output, "restored.png")))) throw new Error("Completion undo is not pixel-identical");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const layout = await page.evaluate(() => ({ viewport: innerWidth, document_width: document.documentElement.scrollWidth }));
  await page.screenshot({ path: path.join(output, "mobile.png"), fullPage: true });
  const report = { status: "PASS_PRIVATE_TRANSACTION", comparison_sha256: comparison.sha256, baseline, previews,
    tested_request_id: tested.request_id, exact_reload_pixels: true, exact_undo_pixels: true, exact_state_undo: true,
    artifact_map_verification: "Requires separate local audit; revision API does not expose artifact maps",
    restored: restored.active, mobile: layout, page_errors: errors, automatic_quality_winner: null,
    scope: "Two actual cloud-generated materials in the private captured scene; one reversible test transaction, not operator approval, hidden-surface truth, ghost-free appearance or navigation acceptance" };
  await writeFile(path.join(output, "background-model-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length || layout.document_width > layout.viewport + 1) throw new Error("Background model comparison has page or layout errors");
}
