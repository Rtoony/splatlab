import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveRemoval(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("This mutation proof only targets the private fixture");
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: 480px !important; height: 320px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const selected = "cardboard-box-2";
  const read = async response => {
    if (!response.ok()) throw new Error(`Proof API failed: ${response.status()} ${await response.text()}`);
    return response.json();
  };
  let initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration) throw new Error("Inspect the private fixture generation before running this proof");
  page.setDefaultTimeout(90000);
  const review = page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." });
  await page.goto(`${base}/studio/${jobId}?appearance=mesh`, { waitUntil: "domcontentloaded" });
  if (initial.legacy_sources_changed) {
    await page.getByRole("button", { name: "Preview fresh capture baseline", exact: true }).click();
    await review.check();
    if ((await read(await request.get(api))).active.generation !== expectedGeneration) throw new Error("Baseline preview changed active state");
    await page.getByRole("button", { name: "Apply together", exact: true }).click();
    await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
    initial = await read(await request.get(api));
  }
  if (!initial.state.captured_collision_current || !initial.state.selections.elements[selected].coordinate_verified) throw new Error("Fresh removal evidence is missing");
  const recovery = await read(await request.post(api + "/recoveries", { data: { expected_generation: initial.active.generation, selected_slug: selected, texture_size: 128 } }));
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  const waitCaptured = async removed => {
    await page.waitForFunction(({ selected, removed }) => {
      const walker = window.__sceneStudioWalker;
      const state = walker?.pluckState()[selected];
      return state?.sampleOpacity === (removed ? 0 : 1) && state.plucked === removed;
    }, { selected, removed }, { timeout: 90000 });
    await page.evaluate(() => window.__sceneStudioWalker.stop());
  };
  await waitCaptured(false);
  const before = await page.evaluate(selected => {
    const walker = window.__sceneStudioWalker;
    const element = walker.elements.find(element => element.slug === selected);
    return { collider_triangles: walker.colliderTris, selected_triangles: element.tris,
      center: [(element.box.min.x + element.box.max.x) / 2, (element.box.min.y + element.box.max.y) / 2, (element.box.min.z + element.box.max.z) / 2],
      backdrop_scale: walker.backdrop.scale.x, collision_bounds: walker.collisionShellGeom.boundingBox,
      rows: walker.pluckState()[selected] };
  }, selected);
  const capture = async (label, sourceCamera) => {
    const camera = await page.evaluate(async sourceCamera => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const container = walker.renderer.domElement.parentElement;
      container.style.width = "480px";
      container.style.height = "320px";
      walker.resize();
      const transform = new walker.camera.matrix.constructor();
      transform.set(...sourceCamera.world_to_camera.flat()).invert();
      const values = transform.elements;
      walker.camera.position.setFromMatrixPosition(transform);
      walker.camera.up.set(-values[4], -values[5], -values[6]).normalize();
      walker.camera.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
      walker.camera.fov = 2 * Math.atan(sourceCamera.height / (2 * sourceCamera.parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      walker.spark.autoUpdate = false;
      await walker.spark.update({ scene: walker.scene, camera: walker.camera });
      walker.renderer.render(walker.scene, walker.camera);
      const context = walker.renderer.getContext();
      const samples = new Uint8Array(64 * 64 * 4);
      context.readPixels(208, 128, 64, 64, context.RGBA, context.UNSIGNED_BYTE, samples);
      const colors = new Set();
      for (let offset = 0; offset < samples.length; offset += 4) colors.add(`${samples[offset]},${samples[offset + 1]},${samples[offset + 2]}`);
      if (context.isContextLost() || colors.size < 16) throw new Error("Captured evidence view is blank or its graphics context failed");
      return { image: sourceCamera.image_key, position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray(), pixel_colors: colors.size };
    }, sourceCamera);
    await page.evaluate(() => {
      const canvas = document.querySelector('canvas[aria-label="Revision-pinned 3D scene"]');
      canvas.style.scrollMarginTop = "70px";
      canvas.scrollIntoView({ behavior: "instant", block: "start" });
    });
    const region = await page.getByLabel("Revision-pinned 3D scene").boundingBox();
    await page.screenshot({ path: path.join(output, label + ".png"), clip: region });
    return camera;
  };
  const available = Object.values(recovery.cameras);
  const fixedViews = [available[0], available[6], available[12]];
  const cameras = [];
  for (const [index, camera] of fixedViews.entries()) cameras.push(await capture(`before-${index}`, camera));
  await page.getByLabel("Retained recovery").selectOption(recovery.recovery_id);
  await page.getByRole("button", { name: "Propose removal + recovery", exact: true }).click();
  await waitCaptured(true);
  const preview = await page.evaluate(selected => {
    const walker = window.__sceneStudioWalker;
    const patch = walker.elements.find(element => element.slug.startsWith("background-"));
    return { collider_triangles: walker.colliderTris, patch_triangles: patch.tris,
      selected_absent: !walker.elements.some(element => element.slug === selected),
      rows: walker.pluckState(), hidden_mask_rows: walker.pluckMask.reduce((count, value) => count + Number(value === 255), 0) };
  }, selected);
  if (!preview.selected_absent || preview.hidden_mask_rows !== before.rows.rows) throw new Error("Selected mesh/rows did not disappear together");
  if (preview.collider_triangles !== before.collider_triangles - before.selected_triangles + preview.patch_triangles) throw new Error("Collision did not remove the prop and add only the supported patch");
  for (const [index, camera] of fixedViews.entries()) await capture(`after-${index}`, camera);
  for (const [index, camera] of available.slice(0, 6).entries()) await capture(`moving-${index}`, camera);
  if ((await read(await request.get(api))).active.generation !== initial.active.generation) throw new Error("Removal preview changed active state");
  const pending = await read(await request.get(api));
  const proposal = pending.proposals.find(item => item.operation.recovery_id === recovery.recovery_id && item.operation.kind === "replace");
  if (!proposal) throw new Error("Cannot retain the private rollback checkpoint");
  await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify({ base: initial.active, preview_revision: proposal.preview_revision }, null, 2));
  await review.check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${initial.active.generation + 1}`, { exact: true }).waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  await waitCaptured(true);
  const applied = await read(await request.get(api));
  if (applied.state.semantics[selected].active || !applied.state.hidden_capture_slugs.includes(selected)) throw new Error("Removal did not survive reload");
  const baseline = initial.history.find(item => item.revision_id === initial.active.revision_id);
  await page.getByRole("button", { name: `${baseline.operation.kind} · ${baseline.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${initial.active.generation + 2}`, { exact: true }).waitFor();
  await waitCaptured(false);
  const restored = await read(await request.get(api));
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state)) throw new Error("Undo did not restore the complete baseline state");
  await capture("restored", fixedViews[0]);
  const report = { recovery_id: recovery.recovery_id, selected_slug: selected, before, preview, cameras,
    baseline_generation: initial.active.generation, final_generation: restored.active.generation,
    preview_kept_active_unchanged: true, reload_preserved_removal: true, exact_state_undo: true,
    page_errors: errors, render_resolution: [480, 320], scope: "Private fixture only; software-rendered removal of existing instance rows and partial recovered cells from retained camera poses. No full-object recall, full background completion, navigation or real-time performance acceptance." };
  await writeFile(path.join(output, "captured-removal-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length) throw new Error("Removal proof reported browser errors");
}
