import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveBackgroundVisibility(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Visibility comparison only targets the private fixture");
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Visibility proof API failed: ${response.status()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  const listed = await read(await request.get(api + "/recoveries"));
  const qualified = listed.recoveries.find(item => item.recovery_id === process.env.SPATIAL_STUDIO_VISIBILITY);
  const parent = listed.recoveries.find(item => item.recovery_id === qualified?.report.visibility_qualification?.parent_recovery_id);
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed || !qualified || qualified.stale || !parent || parent.stale
    || !qualified.report.visibility_comparison) throw new Error("Inspect a fresh visibility-qualified recovery and its parent");
  const selections = await read(await request.get(api + "/selection-reviews"));
  const selection = selections.reviews.filter(item => !item.stale && item.result).at(-1);
  if (!selection) throw new Error("Retained camera evidence is required");
  const camera = selection.cameras.find(item => item.split === "check");
  const width = 960, height = 640;
  await page.addInitScript(({ width, height }) => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = `div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(${width}px, 100%) !important; height: ${height}px !important; min-height: ${height}px !important; }`;
    document.head.append(style);
  }), { width, height });
  page.setDefaultTimeout(90000);
  const loaded = async (slug, isolated = false) => {
    await page.waitForFunction(({ slug, isolated }) => {
      const walker = window.__sceneStudioWalker;
      return walker?.renderer && (isolated || walker.spark && walker.backdrop) && walker.colliderTris > 0
        && (!slug || walker.elements.some(element => element.slug === slug));
    }, { slug, isolated });
    await page.evaluate(() => window.__sceneStudioWalker.stop());
  };
  const capture = async (label, slug = null, angle = 0) => {
    await page.getByLabel("Revision-pinned 3D scene").scrollIntoViewIfNeeded();
    const view = await page.evaluate(async ({ camera, slug, angle, width, height, planeCenter }) => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const transform = new walker.camera.matrix.constructor().set(...camera.world_to_camera.flat()).invert();
      const values = transform.elements;
      walker.camera.position.setFromMatrixPosition(transform);
      walker.camera.up.set(-values[4], -values[5], -values[6]).normalize();
      walker.camera.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
      if (slug) {
        const center = walker.camera.position.clone().fromArray(planeCenter);
        walker.camera.position.copy(center).add(new walker.camera.position.constructor(Math.cos(angle) * 1.5, 1.4, Math.sin(angle) * 1.5));
        walker.camera.up.set(0, 1, 0);
        walker.camera.lookAt(center);
      }
      walker.camera.fov = 2 * Math.atan(camera.height / (2 * camera.parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      const spark = walker.spark;
      if (spark) spark.autoUpdate = false;
      const started = performance.now();
      const settle = async () => {
        while (spark && (spark.sorting || spark.sortDirty || spark.sortTimeoutId !== -1 || spark.updateTimeoutId !== -1)) {
          if (performance.now() - started > 30000) throw new Error("Visibility renderer did not settle");
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      };
      await settle();
      for (let warmup = 0; warmup < 2; warmup++) {
        if (spark) await spark.update({ scene: walker.scene, camera: walker.camera });
        await settle();
        walker.renderer.render(walker.scene, walker.camera);
      }
      const graphics = walker.renderer.getContext();
      if (graphics.isContextLost() || graphics.drawingBufferWidth !== width || graphics.drawingBufferHeight !== height
        || spark && spark.current.mappingVersion !== spark.display.mappingVersion) throw new Error("Visibility frame is not fully displayed");
      const pixels = new Uint8Array(width * height * 4);
      graphics.readPixels(0, 0, width, height, graphics.RGBA, graphics.UNSIGNED_BYTE, pixels);
      const colors = new Set();
      for (let offset = 0; offset < pixels.length; offset += 4) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
      if (colors.size < 16) throw new Error("Visibility preview rendered blank");
      return { png: walker.renderer.domElement.toDataURL("image/png"), distinct_colors: colors.size,
        collider_triangles: walker.colliderTris, patch_triangles: slug ? walker.elements.find(element => element.slug === slug).tris : null,
        camera_position: walker.camera.position.toArray(), camera_quaternion: walker.camera.quaternion.toArray() };
    }, { camera, slug, angle, width, height, planeCenter: parent.report.plane.center });
    await writeFile(path.join(output, label + ".png"), Buffer.from(view.png.split(",")[1], "base64"));
    const { png, ...metadata } = view;
    return { label, ...metadata };
  };
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await loaded(null);
  const baseline = await capture("baseline");
  const views = [], decoded = [];
  for (const [label, recovery] of [["parent", parent], ["qualified", qualified]]) {
    await page.getByLabel("Retained recovery").selectOption(recovery.recovery_id);
    const panel = page.locator("section").filter({ has: page.getByRole("heading", { name: "Recover observed background", exact: true }) });
    await page.waitForFunction(identifier => [...document.images].filter(image => image.src.includes(`/recoveries/${identifier}/artifacts/`)).length === 2
      && [...document.images].filter(image => image.src.includes(`/recoveries/${identifier}/artifacts/`)).every(image => image.complete && image.naturalWidth === 128), recovery.recovery_id);
    decoded.push(recovery.recovery_id);
    if (label === "qualified") {
      await panel.getByRole("heading", { name: "Inferred-depth qualification", exact: true }).waitFor();
      await panel.getByText("Same-pixel appearance checks", { exact: true }).click();
      for (const check of recovery.report.visibility_comparison.paired_appearance_checks) {
        await panel.getByText(`View ${check.image_id} · ${check.shared_pixels.toLocaleString()} shared pixels`, { exact: false }).waitFor();
      }
    }
    await panel.screenshot({ path: path.join(output, `${label}-panel.png`) });
    await page.getByRole("button", { name: "Preview recovered cells", exact: true }).click();
    const slug = `background-${recovery.recovery_id.slice(-8)}`;
    await loaded(slug);
    views.push(await capture(`${label}-context`));
    await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).check();
    await loaded(slug, true);
    await page.waitForFunction(() => window.__sceneStudioWalker?.elements.filter(element => element.object.visible).length === 1);
    if (!await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications.", exact: true }).isDisabled()) throw new Error("Isolated visibility review cannot approve a proposal");
    for (const angle of [0, 1.2]) {
      const view = await capture(`${label}-isolated-${angle}`, slug, angle);
      if (view.patch_triangles !== Math.round(recovery.report.supported_fraction * 128 * 128) * 2
        || view.collider_triangles <= baseline.collider_triangles) throw new Error("Delivered patch geometry/collision disagrees with observed cells");
      views.push(view);
    }
    await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).uncheck();
    await loaded(slug);
    await page.getByRole("button", { name: "Compare original", exact: true }).click();
    await loaded(null);
    await page.waitForFunction(slug => window.__sceneStudioWalker?.elements.length && !window.__sceneStudioWalker.elements.some(element => element.slug === slug), slug);
  }
  await page.reload({ waitUntil: "domcontentloaded" });
  await loaded(null);
  const restored = await capture("restored");
  if (Buffer.compare(await readFile(path.join(output, "baseline.png")), await readFile(path.join(output, "restored.png")))
    || restored.collider_triangles !== baseline.collider_triangles) throw new Error("Visibility preview did not restore the exact baseline");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel("Retained recovery").selectOption(qualified.recovery_id);
  await page.getByRole("heading", { name: "Inferred-depth qualification", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(output, "visibility-mobile.png"), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  const final = await read(await request.get(api));
  for (const angle of [0, 1.2]) {
    const original = views.find(view => view.label === `parent-isolated-${angle}`);
    const filtered = views.find(view => view.label === `qualified-isolated-${angle}`);
    if (JSON.stringify(original.camera_position) !== JSON.stringify(filtered.camera_position)
      || JSON.stringify(original.camera_quaternion) !== JSON.stringify(filtered.camera_quaternion)) throw new Error("A/B isolated camera poses differ");
  }
  if (JSON.stringify(final.active) !== JSON.stringify(initial.active) || JSON.stringify(final.state) !== JSON.stringify(initial.state)) throw new Error("Visibility preview mutated active state");
  const report = { qualified_recovery: qualified.recovery_id, parent_recovery: parent.recovery_id, active: final.active,
    views, decoded_recoveries: decoded, decoded_images: decoded.length * 2, mobile_horizontal_overflow: overflow,
    baseline_pixel_identical: true, matched_isolated_cameras: true, active_unchanged: true, page_errors: errors,
    scope: "Private preview only: two inferred photo-textured planes in captured context and two isolated angles each; no removal, application, generation, measured geometry or whole-scene acceptance" };
  await writeFile(path.join(output, "visibility-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length || overflow) throw new Error("Visibility browser review failed");
}
