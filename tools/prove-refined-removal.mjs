import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveRefinedRemoval(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Refined-removal mutation proof only targets the private fixture");
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(480px, 100%) !important; height: 320px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Refined-removal API failed: ${response.status()} ${await response.text()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed) throw new Error("Inspect the private baseline before this proof");
  const compound = process.env.SPATIAL_COMPOUND_RECEIPT ? JSON.parse(await readFile(process.env.SPATIAL_COMPOUND_RECEIPT, "utf8")) : null;
  let generatedImages = 0;
  if (compound && (compound.status !== "needs-review" || compound.job_id !== jobId || JSON.stringify(compound.base) !== JSON.stringify(initial.active))) throw new Error("Compound preparation is stale or incomplete");
  const collisions = await read(await request.get(api + "/selection-collisions"));
  const collision = collisions.collisions.filter(item => !item.stale && item.result?.verdict === "PASS_LOCAL_EDIT" && (!compound || item.collision_id === compound.collision_id)).at(-1);
  if (!collision) throw new Error("No current passing local collision edit exists");
  const studies = await read(await request.get(api + "/selection-reviews"));
  const study = studies.reviews.find(item => item.review_id === collision.selection_review_id);
  if (!study || study.stale) throw new Error("Collision selection evidence is stale");
  const selected = collision.selected_slug;
  const waitCaptured = async (count, removed) => {
    await page.waitForFunction(({ count, removed, selected }) => {
      const walker = window.__sceneStudioWalker;
      const state = walker?.pluckState()[selected];
      return state?.rows === count && state.plucked === removed && state.sampleOpacity === (removed ? 0 : 1);
    }, { count, removed, selected }, { timeout: 90000 });
    return page.evaluate(selected => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      return { triangles: walker.colliderTris, selected_triangles: walker.elements.find(item => item.slug === selected)?.tris || 0,
        rows: walker.pluckState(), hidden_rows: walker.pluckMask?.reduce((count, value) => count + Number(value === 255), 0) || 0 };
    }, selected);
  };
  const capture = async (label, sourceCamera, nextCamera = null, blend = 0) => {
    const view = await page.evaluate(async ({ sourceCamera, nextCamera, blend }) => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const transform = new walker.camera.matrix.constructor();
      transform.set(...sourceCamera.world_to_camera.flat()).invert();
      const values = transform.elements;
      walker.camera.position.setFromMatrixPosition(transform);
      walker.camera.up.set(-values[4], -values[5], -values[6]).normalize();
      walker.camera.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
      if (nextCamera) {
        const nextTransform = transform.clone().set(...nextCamera.world_to_camera.flat()).invert();
        const nextValues = nextTransform.elements;
        const target = walker.camera.clone();
        target.position.setFromMatrixPosition(nextTransform);
        target.up.set(-nextValues[4], -nextValues[5], -nextValues[6]).normalize();
        target.lookAt(nextValues[12] + nextValues[8], nextValues[13] + nextValues[9], nextValues[14] + nextValues[10]);
        walker.camera.position.lerp(target.position, blend);
        walker.camera.quaternion.slerp(target.quaternion, blend);
      }
      walker.camera.fov = 2 * Math.atan(sourceCamera.height / (2 * sourceCamera.parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      walker.spark.autoUpdate = false;
      await walker.spark.update({ scene: walker.scene, camera: walker.camera });
      walker.renderer.render(walker.scene, walker.camera);
      const context = walker.renderer.getContext();
      const samples = new Uint8Array(480 * 320 * 4);
      context.readPixels(0, 0, 480, 320, context.RGBA, context.UNSIGNED_BYTE, samples);
      const colors = new Set();
      for (let offset = 0; offset < samples.length; offset += 4) colors.add(`${samples[offset]},${samples[offset + 1]},${samples[offset + 2]}`);
      if (context.isContextLost() || colors.size < 16) throw new Error("Refined removal view is blank");
      return { image: sourceCamera.image_key, distinct_colors: colors.size, wireframe: walker.colliderWire.visible,
        position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray(), interpolation: nextCamera ? blend : null };
    }, { sourceCamera, nextCamera, blend });
    await page.getByLabel("Revision-pinned 3D scene").scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(output, label + ".png"), clip: await page.getByLabel("Revision-pinned 3D scene").boundingBox() });
    return view;
  };
  page.setDefaultTimeout(90000);
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  const before = await waitCaptured(study.core_count, false);
  const cameras = [study.cameras.find(camera => camera.split === "check"), study.cameras[0]];
  for (const [index, camera] of cameras.entries()) await capture(`before-${index}`, camera);
  if (compound?.generated_object_id) {
    await page.getByLabel("Retained recovery", { exact: true }).selectOption(compound.recovery_id);
    const panel = page.getByRole("region", { name: "Local generated object", exact: true });
    await panel.getByLabel("Retained generated object", { exact: true }).selectOption(compound.generated_object_id);
    await panel.locator("details").evaluateAll(items => items.filter(item => !item.closest('section[aria-label="Native Gaussian frame review"]')).forEach(item => { item.open = true; }));
    await panel.locator("img").evaluateAll(items => items.filter(item => !item.closest('section[aria-label="Native Gaussian frame review"]')).forEach(item => { item.loading = "eager"; }));
    await page.waitForFunction(() => [...document.querySelectorAll('section[aria-label="Local generated object"] img')].filter(image => !image.closest('section[aria-label="Native Gaussian frame review"]')).every(image => image.complete && image.naturalWidth > 0));
    generatedImages = await panel.locator("img").evaluateAll(async items => { const original = items.filter(item => !item.closest('section[aria-label="Native Gaussian frame review"]')); for (const image of original) await image.decode(); return original.length; });
    if (generatedImages !== 19) throw new Error("Expected the input and all six reference/master/delivery comparisons");
    await panel.screenshot({ path: path.join(output, "generated-object-panel.png") });
    await panel.locator("details").evaluateAll(items => items.forEach(item => { item.open = false; }));
    if (compound.gaussians_id) {
      await panel.getByLabel("Retained Gaussian frame review", { exact: true }).selectOption(compound.gaussians_id);
      await panel.getByRole("button", { name: "Propose native splat replacement + background", exact: true }).click();
    } else {
      await panel.getByRole("button", { name: "Propose generated replacement + background", exact: true }).click();
    }
  } else if (compound) {
    await page.getByLabel("Operation", { exact: true }).selectOption("replace");
    await page.getByLabel("Selected element", { exact: true }).selectOption(selected);
    await page.getByLabel("Versioned Blender export", { exact: true }).selectOption(compound.export);
    await page.getByLabel("New element name", { exact: true }).fill(compound.object_slug);
    await page.getByLabel("Companion background", { exact: true }).selectOption(compound.recovery_id);
    await page.getByLabel("Intent / review note", { exact: true }).fill("Private creative review: substitute an authored chest and preserve observed background in one revision. Unknown cells remain absent; not model inference or navigation acceptance.");
    await page.getByRole("button", { name: "Build review proposal", exact: true }).click();
  } else {
    await page.getByLabel("Retained selection review").selectOption(study.review_id);
    await page.getByLabel("Retained selection collision").selectOption(collision.collision_id);
    await page.getByRole("button", { name: "Propose refined removal", exact: true }).click();
  }
  const preview = await waitCaptured(study.result.candidate_count, true);
  const additions = compound ? await page.evaluate(slugs => {
    const walker = window.__sceneStudioWalker;
    return slugs.map(slug => { const element = walker.elements.find(item => item.slug === slug); if (!element) throw new Error("Missing compound asset"); return { slug, triangles: element.tris }; });
  }, [compound.object_slug, `background-${compound.recovery_id.slice(-8)}`]) : [];
  const expectedTriangles = before.triangles - before.selected_triangles - collision.result.current.triangles + collision.result.candidate.triangles + additions.reduce((total, item) => total + item.triangles, 0);
  await writeFile(path.join(output, "preview-collision-check.json"), JSON.stringify({ before, preview, additions, expectedTriangles }, null, 2));
  if (preview.triangles !== expectedTriangles || preview.selected_triangles !== 0 || preview.hidden_rows !== study.result.candidate_count) throw new Error("Rows and collision did not change together");
  if (compound?.generated_object_id) {
    const generated = await page.evaluate(slug => {
      const element = window.__sceneStudioWalker.elements.find(item => item.slug === slug);
      let meshes = 0;
      let colored = 0;
      element.object.traverse(mesh => { if (mesh.isMesh) { meshes += 1; if (mesh.material.vertexColors && mesh.geometry.getAttribute("color")) colored += 1; } });
      const native = element.gaussianAppearance;
      return { visible: element.visible && (element.object.visible || Boolean(native?.visible)), meshes, colored, collides: element.collides,
        native_rows: native?.numSplats || 0, native_visible: Boolean(native?.visible), mesh_visible: element.object.visible,
        native_matrix: native?.matrix.toArray() || null };
    }, compound.object_slug);
    if (!generated.visible || !generated.collides || !generated.meshes || generated.colored !== generated.meshes) throw new Error("Generated delivery lost visibility, vertex color or collision");
    if (compound.gaussians_id && (generated.native_rows !== compound.gaussians || !generated.native_visible || generated.mesh_visible
      || JSON.stringify(generated.native_matrix) !== JSON.stringify([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))) throw new Error("Native appearance lost its full row count, world frame or exclusive visibility");
    await writeFile(path.join(output, "generated-delivery-check.json"), JSON.stringify(generated, null, 2));
  }
  for (const [slug, state] of Object.entries(before.rows)) {
    if (slug !== selected && JSON.stringify(preview.rows[slug]) !== JSON.stringify(state)) throw new Error("Another instance's row visibility changed");
  }
  const fixed = [];
  for (const [index, camera] of cameras.entries()) fixed.push(await capture(`after-${index}`, camera));
  const nativeComparison = [];
  if (compound?.gaussians_id) {
    await page.getByRole("checkbox", { name: "Native generated appearance (mesh collision unchanged)", exact: true }).uncheck();
    await page.waitForFunction(slug => {
      const element = window.__sceneStudioWalker?.elements.find(item => item.slug === slug);
      return element?.object.visible && !element.gaussianAppearance.visible;
    }, compound.object_slug);
    for (const [index, camera] of cameras.entries()) nativeComparison.push(await capture(`mesh-fallback-${index}`, camera));
    const fallback = await waitCaptured(study.result.candidate_count, true);
    if (JSON.stringify(fallback) !== JSON.stringify(preview)) throw new Error("Appearance mode changed collision or captured membership");
    await page.getByRole("checkbox", { name: "Native generated appearance (mesh collision unchanged)", exact: true }).check();
    await page.waitForFunction(slug => {
      const element = window.__sceneStudioWalker?.elements.find(item => item.slug === slug);
      return element?.gaussianAppearance.visible && !element.object.visible;
    }, compound.object_slug);
  }
  const sequence = [];
  for (const [index, camera] of study.cameras.slice(0, 6).entries()) sequence.push(await capture(`sequence-${index}`, camera));
  const moving = [];
  if (compound) {
    for (const fraction of [.2, .4, .6, .8]) moving.push(await capture(`moving-${fraction}`, cameras[0], cameras[1], fraction));
    await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).check();
    await page.waitForFunction(slugs => {
      const visible = window.__sceneStudioWalker?.elements.filter(item => item.visible).map(item => item.slug).sort();
      return JSON.stringify(visible) === JSON.stringify(slugs.sort());
    }, additions.map(item => item.slug));
    await page.evaluate(async slug => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const element = walker.elements.find(item => item.slug === slug);
      const target = element.box.getCenter(walker.camera.position.clone());
      walker.camera.position.copy(target).add(new walker.camera.position.constructor(1, .8, 1));
      walker.camera.up.set(0, 1, 0);
      walker.camera.lookAt(target);
      if (walker.spark) await walker.spark.update({ scene: walker.scene, camera: walker.camera });
      walker.renderer.render(walker.scene, walker.camera);
    }, compound.object_slug);
    await page.getByLabel("Revision-pinned 3D scene").screenshot({ path: path.join(output, "compound-isolated.png") });
    if (await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).isEnabled()) throw new Error("Isolated compound view granted approval");
    await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).uncheck();
    await waitCaptured(study.result.candidate_count, true);
  }
  await page.getByRole("checkbox", { name: "Collision wireframe", exact: true }).check();
  const collisionView = await capture("collision-wireframe", cameras[0]);
  if (!collisionView.wireframe) throw new Error("Collision overlay did not enable");
  await page.getByRole("checkbox", { name: "Collision wireframe", exact: true }).uncheck();
  const pending = await read(await request.get(api));
  if (JSON.stringify(pending.active) !== JSON.stringify(initial.active)) throw new Error("Preview changed the active scene");
  const proposal = pending.proposals.filter(item => item.operation.selection_collision_id === collision.collision_id && item.base.generation === expectedGeneration && (!compound || item.operation.slug === compound.object_slug)).at(-1);
  if (!proposal) throw new Error("Missing private rollback checkpoint identity");
  const checkpoint = { base: initial.active, preview_revision: proposal.preview_revision,
    rollback_from: [{ revision_id: proposal.preview_revision, generation: expectedGeneration + 1 }] };
  await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify(checkpoint, null, 2));
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  const reloaded = await waitCaptured(study.result.candidate_count, true);
  if (JSON.stringify(reloaded) !== JSON.stringify(preview)) throw new Error("Refined removal did not survive reload");
  const applied = await read(await request.get(api));
  if (applied.state.semantics[selected].active || applied.state.capture_partition.collision_id !== collision.collision_id) throw new Error("Selection/collision semantics were not persisted");
  let removalProof = null;
  if (compound?.gaussians_id) {
    await page.waitForFunction(({ slug, rows }) => {
      const element = window.__sceneStudioWalker?.elements.find(item => item.slug === slug);
      return element?.gaussianAppearance?.visible && element.gaussianAppearance.numSplats === rows;
    }, { slug: compound.object_slug, rows: compound.gaussians });
    await page.getByLabel("Operation", { exact: true }).selectOption("remove");
    await page.getByLabel("Selected element", { exact: true }).selectOption(compound.object_slug);
    await page.getByLabel("Intent / review note", { exact: true }).fill("Private paired-layer test: remove native appearance and inferred collision together, then undo without resurrecting captured rows.");
    await page.getByRole("button", { name: "Build review proposal", exact: true }).click();
    await page.waitForFunction(slug => !window.__sceneStudioWalker?.elements.some(item => item.slug === slug), compound.object_slug);
    const removed = await waitCaptured(study.result.candidate_count, true);
    const objectTriangles = additions.find(item => item.slug === compound.object_slug).triangles;
    if (removed.triangles !== preview.triangles - objectTriangles) throw new Error("Removing native appearance did not remove exactly its mesh collider");
    await capture("native-removed", cameras[0]);
    const proposedRemoval = await read(await request.get(api));
    const removal = proposedRemoval.proposals.findLast(item => item.base.generation === expectedGeneration + 1 && item.operation.kind === "remove" && item.operation.selected_slug === compound.object_slug);
    if (!removal) throw new Error("Missing paired-layer removal proposal");
    checkpoint.rollback_from.push({ revision_id: removal.preview_revision, generation: expectedGeneration + 2 });
    await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify(checkpoint, null, 2));
    await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
    await page.getByRole("button", { name: "Apply together", exact: true }).click();
    await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
    await page.reload({ waitUntil: "domcontentloaded" });
    const removedReloaded = await waitCaptured(study.result.candidate_count, true);
    if (JSON.stringify(removedReloaded) !== JSON.stringify(removed)) throw new Error("Native removal changed after reload");
    const nativeLayers = await page.evaluate(() => window.__sceneStudioWalker.scene.children.filter(item => item.name.endsWith(":generated-appearance")).length);
    if (nativeLayers !== 0) throw new Error("Removed native appearance remains in the renderer");
    await page.getByRole("button", { name: `replace · ${proposal.preview_revision.slice(-8)}`, exact: true }).click();
    await page.getByText(`Active generation ${expectedGeneration + 3}`, { exact: true }).waitFor();
    const reinstated = await waitCaptured(study.result.candidate_count, true);
    const restoredNative = await read(await request.get(api));
    checkpoint.rollback_from.push({ revision_id: restoredNative.active.revision_id, generation: restoredNative.active.generation });
    await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify(checkpoint, null, 2));
    if (JSON.stringify(reinstated) !== JSON.stringify(preview) || JSON.stringify(restoredNative.state) !== JSON.stringify(applied.state)) throw new Error("Undo did not restore both native and mesh layers exactly");
    await capture("native-reinstated", cameras[0]);
    if (Buffer.compare(await readFile(path.join(output, "after-0.png")), await readFile(path.join(output, "native-reinstated.png")))) throw new Error("Native removal undo is not pixel-identical");
    removalProof = { removed, removedReloaded, reinstated, removed_native_layers: nativeLayers, exact_state_undo: true, exact_pixel_undo: true };
  }
  const baseline = initial.history.find(item => item.revision_id === initial.active.revision_id);
  await page.getByRole("button", { name: `${baseline.operation.kind} · ${baseline.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + (compound?.gaussians_id ? 4 : 2)}`, { exact: true }).waitFor();
  await waitCaptured(study.core_count, false);
  const restored = await read(await request.get(api));
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state)) throw new Error("Undo did not restore complete original state");
  await capture("restored", cameras[0]);
  if (Buffer.compare(await readFile(path.join(output, "before-0.png")), await readFile(path.join(output, "restored.png")))) throw new Error("Matched-camera undo is not pixel-identical");
  let mobile = null;
  if (compound) {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    mobile = await page.evaluate(() => ({ viewport: innerWidth, document_width: document.documentElement.scrollWidth }));
    await page.screenshot({ path: path.join(output, "compound-mobile.png"), fullPage: true });
    if (mobile.document_width > mobile.viewport + 1) throw new Error("Compound UI overflows on mobile");
  }
  const report = { collision_id: collision.collision_id, selection_review_id: study.review_id, before, preview,
    baseline_generation: expectedGeneration, final_generation: restored.active.generation, fixed, sequence,
    preview_unchanged_active: true, reload_preserved_removal: true, exact_state_undo: true,
    additions, moving, mobile, compound_receipt_sha256: compound?.sha256, generated_images_decoded: generatedImages,
    gaussians_id: compound?.gaussians_id, native_comparison: nativeComparison, native_removal: removalProof,
    generated_object_id: compound?.generated_object_id, generated_result_sha256: compound?.generated_result_sha256,
    page_errors: errors, scope: "Private reviewed transaction, 480x320 software views; authored or generated replacement when specified, incomplete observed background, fitted local clearance, no complete-object, model-quality or navigation acceptance" };
  await writeFile(path.join(output, "refined-removal-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length) throw new Error("Refined removal reported browser errors");
}
