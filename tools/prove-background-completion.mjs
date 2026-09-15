import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import path from "node:path";

async function inspectLayout(page, output) {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const layout = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document_width: document.documentElement.scrollWidth,
    overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    offenders: Array.from(document.querySelectorAll("main *")).flatMap(element => {
      const rectangle = element.getBoundingClientRect();
      return rectangle.right > window.innerWidth + 1 && rectangle.width > 0 ? [{ tag: element.tagName, label: element.getAttribute("aria-label"),
        class_name: typeof element.className === "string" ? element.className : "", left: rectangle.left, right: rectangle.right, width: rectangle.width }] : [];
    }).slice(0, 30),
  }));
  await writeFile(path.join(output, "completion-layout.json"), JSON.stringify(layout, null, 2));
  await page.screenshot({ path: path.join(output, "completion-mobile.png"), fullPage: true });
  return layout;
}

export async function proveCompletion(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Completion mutation proof only targets the private fixture");
  page.setDefaultTimeout(90000);
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(480px, 100%) !important; height: 320px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Completion API failed: ${response.status()} ${await response.text()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed) throw new Error("Inspect the private baseline before this proof");
  if (process.env.SPATIAL_COMPLETION_LAYOUT_ONLY) {
    await page.goto(`${base}/studio/${jobId}?appearance=mesh`, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.__sceneStudioWalker?.colliderTris > 0);
    await page.getByRole("region", { name: "Unknown-region completion", exact: true }).waitFor();
    await page.evaluate(() => window.__sceneStudioWalker.stop());
    const layout = await inspectLayout(page, output);
    if (JSON.stringify((await read(await request.get(api))).active) !== JSON.stringify(initial.active)) throw new Error("Read-only layout inspection changed the scene");
    console.log(JSON.stringify({ ...layout, active_unchanged: true, page_errors: errors, scope: "read-only mobile completion layout" }));
    if (layout.overflow || errors.length) throw new Error("Read-only layout check failed");
    return;
  }
  const recoveries = await read(await request.get(`${api}/recoveries`));
  const recovery = recoveries.recoveries.filter(item => !item.stale && item.selected_slug === "cardboard-box-2" && item.report.plane.selection?.method === "photo-linked-sfm-anchors").at(-1);
  if (!recovery) throw new Error("A fresh anchored recovery is required; do not silently rebuild or rebase it");
  await page.goto(`${base}/studio/${jobId}?appearance=mesh`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__sceneStudioWalker?.colliderTris > 0);
  const before = await page.evaluate(() => { const walker = window.__sceneStudioWalker; walker.stop(); return walker.colliderTris; });
  await page.getByLabel("Retained recovery", { exact: true }).selectOption(recovery.recovery_id);
  const panel = page.getByRole("region", { name: "Unknown-region completion", exact: true });
  await panel.getByLabel("Intended unknown-region material", { exact: true }).fill("Synthetic striped contract fixture only. This tests disjoint material ownership and review/undo, not a generated floor or model quality.");
  const preparedResponse = page.waitForResponse(response => response.url().endsWith("/studio/completions") && response.request().method() === "POST");
  await panel.getByRole("button", { name: "Prepare creative completion", exact: true }).click();
  const receipt = await read(await preparedResponse);
  await panel.getByLabel("Retained completion", { exact: true }).selectOption(receipt.completion_id);
  const material = Buffer.from(await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 128;
    const context = canvas.getContext("2d");
    for (let row = 0; row < 8; row++) {
      context.fillStyle = row % 2 ? "#f0a030" : "#6240a0";
      context.fillRect(0, row * 16, 128, 16);
    }
    return canvas.toDataURL("image/png").split(",")[1];
  }), "base64");
  await writeFile(path.join(output, "synthetic-material-fixture.png"), material);
  await panel.getByLabel("Material provider", { exact: true }).fill("Local software test fixture");
  await panel.getByLabel("Model or source label", { exact: true }).fill("Canvas stripes; no model inference");
  await panel.getByLabel("UV-aligned completion image", { exact: true }).setInputFiles({ name: "synthetic-fixture.png", mimeType: "image/png", buffer: material });
  const importedResponse = page.waitForResponse(response => response.url().endsWith(`/completions/${receipt.completion_id}/image`) && response.request().method() === "POST");
  await panel.getByRole("button", { name: "Import completion material", exact: true }).click();
  const result = await read(await importedResponse);
  const meshResponse = await request.get(`${api}/completions/${receipt.completion_id}/artifact?name=candidate.glb&result=true`);
  if (!meshResponse.ok()) throw new Error("Completion GLB failed to load");
  const mesh = await meshResponse.body();
  const jsonLength = mesh.readUInt32LE(12);
  const document = JSON.parse(mesh.subarray(20, 20 + jsonLength).toString());
  const binary = mesh.subarray(28 + jsonLength);
  const embedded = document.images.map(image => {
    const view = document.bufferViews[image.bufferView];
    return createHash("sha256").update(binary.subarray(view.byteOffset, view.byteOffset + view.byteLength)).digest("hex");
  });
  if (embedded[0] !== receipt.artifacts["atlas.png"].sha256 || embedded[1] !== createHash("sha256").update(material).digest("hex")) throw new Error("Source or imported texture bytes were altered");
  const layers = document.meshes[0].primitives.map(primitive => ({ ...primitive.extras, triangles: document.accessors[primitive.indices].count / 3 }));
  if (layers[0].triangles !== receipt.observed_cells * 2 || layers[1].triangles !== receipt.generated_cells * 2) throw new Error("GLB material ownership counts disagree");
  if (await panel.getByRole("button", { name: "Propose removal + completion", exact: true }).isEnabled()) throw new Error("Stale collision evidence authorized removal");
  await panel.screenshot({ path: path.join(output, "completion-import-panel.png") });
  const proposedResponse = page.waitForResponse(response => response.url().endsWith("/studio/proposals") && response.request().method() === "POST");
  await panel.getByRole("button", { name: "Preview completed support", exact: true }).click();
  const proposal = await read(await proposedResponse);
  const slug = `completion-${receipt.completion_id.slice(-8)}`;
  await page.waitForFunction(slug => window.__sceneStudioWalker?.colliderTris > 0 && window.__sceneStudioWalker.elements.some(element => element.slug === slug), slug);
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).check();
  await page.waitForFunction(slug => {
    const walker = window.__sceneStudioWalker;
    return walker?.colliderTris > 0 && walker.elements.filter(element => element.object.visible).length === 1 && walker.elements.some(element => element.slug === slug && element.object.visible);
  }, slug);
  const views = [];
  for (const angle of [0, 1.2]) {
    views.push(await page.evaluate(({ slug, angle }) => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const element = walker.elements.find(element => element.slug === slug);
      const center = element.box.getCenter(walker.camera.position.clone());
      walker.camera.position.set(center.x + Math.cos(angle) * 1.2, center.y + 1.2, center.z + Math.sin(angle) * 1.2);
      walker.camera.lookAt(center);
      walker.renderer.render(walker.scene, walker.camera);
      return { camera: walker.camera.position.toArray(), patch_triangles: element.tris, collider_triangles: walker.colliderTris };
    }, { slug, angle }));
    await page.getByLabel("Revision-pinned 3D scene").screenshot({ path: path.join(output, `completion-angle-${views.length}.png`) });
  }
  if (views.some(view => view.patch_triangles !== result.triangles || view.collider_triangles !== before + result.triangles)) throw new Error("Completed plane did not join visual geometry and collision together");
  if (await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).isEnabled()) throw new Error("Isolation granted full-context approval");
  if (JSON.stringify((await read(await request.get(api))).active) !== JSON.stringify(initial.active)) throw new Error("Completion preview changed the active scene");
  await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).uncheck();
  await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify({ base: initial.active, preview_revision: proposal.preview_revision }, null, 2));
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(slug => window.__sceneStudioWalker?.colliderTris > 0 && window.__sceneStudioWalker.elements.some(element => element.slug === slug), slug);
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  const baseline = initial.history.find(item => item.revision_id === initial.active.revision_id);
  await page.getByRole("button", { name: `${baseline.operation.kind} · ${baseline.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
  const restored = await read(await request.get(api));
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state)) throw new Error("Completion undo did not restore baseline state");
  await page.waitForFunction(() => window.__sceneStudioWalker?.colliderTris > 0);
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  const layout = await inspectLayout(page, output);
  const overflow = layout.overflow;
  const report = { completion_id: receipt.completion_id, recovery_id: recovery.recovery_id, receipt_sha256: receipt.sha256,
    result_sha256: result.sha256, source_atlas_bytes_preserved: true, imported_bytes_preserved: true, layers, views,
    baseline_generation: expectedGeneration, restored_generation: restored.active.generation, exact_state_undo: true,
    mobile_horizontal_overflow: overflow, page_errors: errors,
    scope: "private browser software-fixture import/ownership/apply/reload/undo proof; no model inference, full captured removal, photorealism, 30 FPS or navigation acceptance" };
  await writeFile(path.join(output, "completion-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length || overflow) throw new Error("Completion browser proof failed; inspect the report");
}
