import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveNativeRoomResume(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Native transaction resumption only targets the private fixture");
  const previous = process.env.SPATIAL_NATIVE_RESUME;
  const checkpointBefore = JSON.parse(await readFile(path.join(previous, "private-rollback-checkpoint.json"), "utf8"));
  const cleanup = JSON.parse(await readFile(path.join(previous, "failed-proof-restore.json"), "utf8"));
  const candidate = JSON.parse(await readFile(process.env.SPATIAL_COMPOUND_RECEIPT, "utf8"));
  const read = async response => {
    if (!response.ok()) throw new Error(`Native resume API ${response.status()}: ${await response.text()}`);
    return response.json();
  };
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const initial = await read(await request.get(api));
  const original = await read(await request.get(api + "/revisions/" + checkpointBefore.base.revision_id));
  const paired = await read(await request.get(api + "/revisions/" + cleanup.applied.revision_id));
  const candidateBase = await read(await request.get(api + "/revisions/" + candidate.base.revision_id));
  const ownedRestore = checkpointBefore.rollback_from.some(item => item.revision_id === cleanup.applied.revision_id && item.generation === cleanup.applied.generation)
    && paired.parent === checkpointBefore.base.revision_id
    && initial.history.find(item => item.revision_id === paired.revision_id)?.operation.revision_id === checkpointBefore.preview_revision;
  let ownedReinstatement = false;
  const cycle = checkpointBefore.rollback_from;
  if (cycle.length === 3 && cycle.every((pointer, index) => pointer.generation === checkpointBefore.base.generation + index + 1)
    && cycle[2].revision_id === cleanup.applied.revision_id && cycle[2].generation === cleanup.applied.generation
    && cycle.every(pointer => initial.history.some(item => item.revision_id === pointer.revision_id))) {
    const first = await read(await request.get(api + "/revisions/" + cycle[0].revision_id));
    const removed = await read(await request.get(api + "/revisions/" + cycle[1].revision_id));
    const operations = cycle.map(pointer => initial.history.find(item => item.revision_id === pointer.revision_id).operation);
    ownedReinstatement = first.parent === checkpointBefore.base.revision_id
      && operations[0].kind === "restore" && operations[0].revision_id === checkpointBefore.preview_revision
      && removed.parent === first.revision_id && operations[1].kind === "remove" && operations[1].selected_slug === candidate.object_slug
      && paired.parent === removed.revision_id && operations[2].kind === "restore" && operations[2].revision_id === first.revision_id
      && JSON.stringify(paired.state) === JSON.stringify(first.state);
  }
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed
    || JSON.stringify(initial.active) !== JSON.stringify(cleanup.restored)
    || JSON.stringify(initial.state) !== JSON.stringify(original.state)
    || (cleanup.applied.revision_id !== checkpointBefore.preview_revision && !ownedRestore && !ownedReinstatement)
    || cleanup.applied.generation !== checkpointBefore.base.generation + (ownedReinstatement ? 3 : 1)
    || !initial.history.some(item => item.revision_id === cleanup.applied.revision_id)
    || JSON.stringify(candidateBase.state) !== JSON.stringify(original.state)) throw new Error("Resume requires exactly the owned, activated and rolled-back private transaction");
  const entry = paired.viewer.elements.find(item => item.slug === candidate.object_slug);
  if (entry?.gaussian_appearance?.gaussians_id !== candidate.gaussians_id) throw new Error("Historical native pairing differs from the candidate");
  const studies = await read(await request.get(api + "/selection-reviews"));
  const study = studies.reviews.find(item => item.review_id === candidate.selection_review_id);
  const cameras = [study.cameras.find(camera => camera.split === "check"), study.cameras[0]];
  const checkpoint = { base: initial.active, preview_revision: paired.revision_id, rollback_from: [], restore_targets: [] };
  const save = () => writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify(checkpoint, null, 2));
  const waitState = async (nativePresent, capturedRemoved) => {
    console.log(JSON.stringify({ stage: "wait-state", nativePresent, capturedRemoved }));
    await page.waitForFunction(({ slug, selected, nativePresent, capturedRemoved, nativeRows, capturedRows }) => {
      const walker = window.__sceneStudioWalker;
      const element = walker?.elements.find(item => item.slug === slug);
      const captured = walker?.pluckState()[selected];
      return walker && captured?.plucked === capturedRemoved && captured.rows === capturedRows
        && (nativePresent ? element?.gaussianAppearance?.numSplats === nativeRows && element.gaussianAppearance.visible && !element.object.visible : !element);
    }, { slug: candidate.object_slug, selected: study.selected_slug, nativePresent, capturedRemoved,
      nativeRows: candidate.gaussians, capturedRows: capturedRemoved ? study.result.candidate_count : study.core_count }, { timeout: 120000, polling: 250 });
    return page.evaluate(slug => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      return { triangles: walker.colliderTris, object_triangles: walker.elements.find(item => item.slug === slug)?.tris || 0,
        native_layers: walker.scene.children.filter(item => item.name.endsWith(":generated-appearance")).length,
        hidden_rows: walker.pluckMask?.reduce((total, value) => total + Number(value === 255), 0) || 0 };
    }, candidate.object_slug);
  };
  const capture = async (label, camera) => {
    console.log(JSON.stringify({ stage: "capture", label }));
    await page.evaluate(() => window.__sceneStudioWalker.stop());
    await page.getByLabel("Revision-pinned 3D scene").scrollIntoViewIfNeeded();
    const metrics = await page.evaluate(async camera => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const transform = new walker.camera.matrix.constructor().set(...camera.world_to_camera.flat()).invert();
      const values = transform.elements;
      walker.camera.position.setFromMatrixPosition(transform);
      walker.camera.up.set(-values[4], -values[5], -values[6]).normalize();
      walker.camera.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
      walker.camera.fov = 2 * Math.atan(camera.height / (2 * camera.parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      walker.spark.autoUpdate = false;
      const started = performance.now();
      const pending = () => walker.spark.sorting || walker.spark.sortDirty || walker.spark.sortTimeoutId !== -1 || walker.spark.updateTimeoutId !== -1;
      const settle = async () => {
        while (pending()) {
          if (performance.now() - started > 30000) throw new Error("Native frame sort did not settle within its budget");
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      };
      const sortPendingBefore = pending();
      await settle();
      await walker.spark.update({ scene: walker.scene, camera: walker.camera });
      const sortPendingAfterUpdate = pending();
      await settle();
      if (walker.spark.current.mappingVersion !== walker.spark.display.mappingVersion) throw new Error("Native frame mapping is not displayed yet");
      walker.renderer.render(walker.scene, walker.camera);
      await walker.spark.update({ scene: walker.scene, camera: walker.camera });
      await settle();
      walker.renderer.render(walker.scene, walker.camera);
      const context = walker.renderer.getContext();
      const extension = context.getExtension("WEBGL_debug_renderer_info");
      const renderer = context.getParameter(extension ? extension.UNMASKED_RENDERER_WEBGL : context.RENDERER);
      const pixels = new Uint8Array(480 * 320 * 4);
      context.readPixels(0, 0, 480, 320, context.RGBA, context.UNSIGNED_BYTE, pixels);
      const colors = new Set();
      for (let offset = 0; offset < pixels.length; offset += 4) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
      if (colors.size < 16 || context.isContextLost()) throw new Error("Native room frame is blank");
      return { camera: camera.image_id, distinct_colors: colors.size, renderer, update_render_readback_ms: performance.now() - started,
        sort_pending_before: sortPendingBefore, sort_pending_after_update: sortPendingAfterUpdate, sort_settled_before_readback: !pending(),
        camera_state: { position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray(), up: walker.camera.up.toArray(),
          projection: walker.camera.projectionMatrix.toArray(), world: walker.camera.matrixWorld.toArray(), near: walker.camera.near, far: walker.camera.far },
        spark_state: { active: walker.spark.activeSplats, current: walker.spark.current.numSplats, display: walker.spark.display.numSplats,
          render_size: walker.spark.renderSize.toArray(), sorted_center: walker.spark.sortedCenter.toArray(), sorted_direction: walker.spark.sortedDir.toArray(),
          sort_radial: walker.spark.sortRadial, cov_splats: walker.spark.covSplats, accum_ext: walker.spark.accumExtSplats,
          blur: walker.spark.blurAmount, pre_blur: walker.spark.preBlurAmount, min_alpha: walker.spark.minAlpha,
          linear: walker.spark.encodeLinear, lod: walker.spark.enableLod },
        splat_transforms: walker.scene.children.filter(child => child.numSplats).map(child => ({ name: child.name, rows: child.numSplats,
          visible: child.visible, world: child.matrixWorld.toArray() })) };
    }, camera);
    await writeFile(path.join(output, label + ".json"), JSON.stringify(metrics, null, 2));
    await page.getByLabel("Revision-pinned 3D scene").screenshot({ path: path.join(output, label + ".png") });
    return metrics;
  };
  const restore = async (revisionId, generation) => {
    console.log(JSON.stringify({ stage: "restore", revisionId, generation }));
    const before = await read(await request.get(api));
    checkpoint.restore_targets.push({ target: revisionId, parent: before.active.revision_id, generation });
    await save();
    const history = before.history.find(item => item.revision_id === revisionId);
    await page.getByRole("button", { name: `${history.operation.kind} · ${revisionId.slice(-8)}`, exact: true }).click();
    await page.getByText(`Active generation ${generation}`, { exact: true }).waitFor();
    const restored = await read(await request.get(api));
    checkpoint.rollback_from.push({ revision_id: restored.active.revision_id, generation });
    await save();
    return restored;
  };
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    Object.defineProperty(window, "__sceneStudioWalker", { configurable: true, set(walker) {
      walker.stop();
      Object.getPrototypeOf(walker).start = function () { this.stop(); };
      Object.defineProperty(window, "__sceneStudioWalker", { value: walker, writable: true, configurable: true });
    } });
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(480px, 100%) !important; height: 320px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  page.setDefaultTimeout(120000);
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  const baseline = await waitState(false, false);
  await capture("before-0", cameras[0]);
  const restoredPair = await restore(paired.revision_id, expectedGeneration + 1);
  if (JSON.stringify(restoredPair.state) !== JSON.stringify(paired.state)) throw new Error("Historical restore did not reproduce the activated pairing");
  const preview = await waitState(true, true);
  const fixed = [];
  for (const [index, camera] of cameras.entries()) fixed.push(await capture(`after-${index}`, camera));
  await page.reload({ waitUntil: "domcontentloaded" });
  if (JSON.stringify(await waitState(true, true)) !== JSON.stringify(preview)) throw new Error("Native pairing did not survive reload");
  await page.getByLabel("Operation", { exact: true }).selectOption("remove");
  await page.getByLabel("Selected element", { exact: true }).selectOption(candidate.object_slug);
  await page.getByLabel("Intent / review note", { exact: true }).fill("Private current-revision removal of paired native appearance and inferred mesh collision; retain captured removal and observed background.");
  await page.getByRole("button", { name: "Build review proposal", exact: true }).click();
  const removed = await waitState(false, true);
  if (removed.triangles !== preview.triangles - preview.object_triangles || removed.native_layers !== 0 || removed.hidden_rows !== preview.hidden_rows) throw new Error("Paired removal changed the wrong layers or collider");
  await capture("native-removed", cameras[0]);
  const pending = await read(await request.get(api));
  if (pending.active.revision_id !== restoredPair.active.revision_id) throw new Error("Removal preview changed the active scene");
  const removal = pending.proposals.findLast(item => item.base.generation === expectedGeneration + 1 && item.operation.kind === "remove" && item.operation.selected_slug === candidate.object_slug);
  checkpoint.rollback_from.push({ revision_id: removal.preview_revision, generation: expectedGeneration + 2 });
  await save();
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  const removedReloaded = await waitState(false, true);
  if (JSON.stringify(removedReloaded) !== JSON.stringify(removed)) throw new Error("Paired removal changed after reload");
  const reinstatedState = await restore(restoredPair.active.revision_id, expectedGeneration + 3);
  const reinstated = await waitState(true, true);
  if (JSON.stringify(reinstated) !== JSON.stringify(preview) || JSON.stringify(reinstatedState.state) !== JSON.stringify(restoredPair.state)) throw new Error("Paired undo did not restore exact state");
  await capture("native-reinstated", cameras[0]);
  if (Buffer.compare(await readFile(path.join(output, "after-0.png")), await readFile(path.join(output, "native-reinstated.png")))) throw new Error("Paired undo changed matched-camera pixels");
  const final = await restore(initial.active.revision_id, expectedGeneration + 4);
  const restored = await waitState(false, false);
  if (JSON.stringify(final.state) !== JSON.stringify(initial.state) || JSON.stringify(restored) !== JSON.stringify(baseline)) throw new Error("Final undo differs from the untouched baseline");
  await capture("restored", cameras[0]);
  if (Buffer.compare(await readFile(path.join(output, "before-0.png")), await readFile(path.join(output, "restored.png")))) throw new Error("Baseline undo changed matched-camera pixels");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(output, "native-room-mobile.png"), fullPage: true });
  const mobile = await page.evaluate(() => ({ viewport: innerWidth, document_width: document.documentElement.scrollWidth }));
  if (mobile.document_width > mobile.viewport + 1 || errors.length) throw new Error("Native room UI overflowed or reported errors");
  const report = { status: "passed", previous_attempt: previous, resumed_from: initial.active, resumed_activation: cleanup.applied,
    baseline_generation: expectedGeneration, final_generation: final.active.generation, generated_object_id: candidate.generated_object_id,
    gaussians_id: candidate.gaussians_id, previous_review_views_retained: true, no_stale_proposal_created: true, manual_frames_only: true,
    baseline, preview, fixed, mobile, native_removal: { removed, removedReloaded, reinstated, exact_state_undo: true, exact_pixel_undo: true },
    exact_state_undo: true, exact_pixel_undo: true, page_errors: errors,
    scope: "Resumes only a previously activated private revision via ordinary history restore; current paired removal and exact undo; not smooth software interaction, hardware FPS or navigation acceptance" };
  await writeFile(path.join(output, "refined-removal-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
}
