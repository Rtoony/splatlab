import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveSupport(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Support mutation proof only targets the private fixture");
  page.setDefaultTimeout(90000);
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: 480px !important; height: 320px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Support API failed: ${response.status()} ${await response.text()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed) throw new Error("Inspect the private baseline before this proof");
  const observations = {};
  for (const imageId of [47, 280]) {
    observations[imageId] = await read(await request.get(`${api}/recovery-support?selected_slug=cardboard-box-2&expected_generation=${expectedGeneration}&image_id=${imageId}`));
  }
  await page.goto(`${base}/studio/${jobId}?appearance=mesh`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__sceneStudioWalker?.colliderTris > 0);
  const before = await page.evaluate(() => { const walker = window.__sceneStudioWalker; walker.stop(); return walker.colliderTris; });
  await page.getByLabel("Captured element for background recovery").selectOption("cardboard-box-2");
  await page.getByRole("checkbox", { name: "Choose support plane from tracked photo points", exact: true }).check();
  await page.getByLabel("Support reference photograph", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Expand photo picker", exact: true }).click();
  const selectedPoints = [[47, "9279"], [280, "121451"], [280, "121358"], [280, "9542"]];
  const photos = [];
  for (const imageId of [47, 280]) {
    const view = observations[imageId];
    await page.getByLabel("Support reference photograph", { exact: true }).selectOption(String(imageId));
    await page.waitForFunction(identifier => document.querySelector('svg[aria-label="Tracked support observations"] image')?.getAttribute("href")?.includes(`/recovery-support/${identifier}/photo?`), imageId);
    const photo = await page.evaluate(async () => {
      const url = document.querySelector('svg[aria-label="Tracked support observations"] image').getAttribute("href");
      const image = new Image();
      image.src = url;
      await image.decode();
      return { width: image.naturalWidth, height: image.naturalHeight };
    });
    if (photo.width !== view.width || photo.height !== view.height) throw new Error("Support photo dimensions changed");
    photos.push({ image_id: imageId, ...photo });
    for (const [identifier, pointId] of selectedPoints.filter(([identifier]) => identifier === imageId)) {
      const feature = view.features.find(feature => feature.point_id === pointId);
      if (!feature) throw new Error(`Reviewed floor point ${pointId} is not eligible in the current capture`);
      if (pointId === "9279") {
        const picture = page.getByLabel("Tracked support observations", { exact: true });
        await picture.scrollIntoViewIfNeeded();
        const rectangle = await picture.boundingBox();
        await picture.click({ position: { x: feature.pixel[0] / view.width * rectangle.width, y: feature.pixel[1] / view.height * rectangle.height } });
        if (!(await page.getByLabel(`Remove support anchor ${pointId}`, { exact: true }).count())) {
          const duplicate = await page.locator('button[aria-label^="Remove support anchor "]').all();
          if (duplicate.length !== 1) throw new Error("Photo click did not select one tracked point");
          await duplicate[0].click();
          await page.getByLabel("Tracked support point ID", { exact: true }).fill(pointId);
          await page.getByRole("button", { name: "Add tracked point", exact: true }).click();
        }
      } else {
        await page.getByLabel("Tracked support point ID", { exact: true }).fill(pointId);
        await page.getByRole("button", { name: "Add tracked point", exact: true }).click();
      }
      await page.getByLabel(`Remove support anchor ${pointId}`, { exact: true }).waitFor();
      if (identifier !== imageId) throw new Error("Unexpected photo selection");
    }
    await page.getByLabel("Tracked support observations", { exact: true }).screenshot({ path: path.join(output, `support-photo-${imageId}.png`) });
  }
  await page.getByRole("dialog", { name: "Choose support surface", exact: true }).getByRole("button", { name: "Close", exact: true }).click();
  const builtResponse = page.waitForResponse(response => response.url().endsWith("/studio/recoveries") && response.request().method() === "POST");
  await page.getByRole("button", { name: "Recover surface observations", exact: true }).click();
  const built = await read(await builtResponse);
  const receipt = built;
  if (receipt.base.generation !== expectedGeneration || receipt.report.plane.selection.anchors.length !== 4) throw new Error("Recovery did not retain all four chosen anchors");
  await page.getByLabel("Retained recovery", { exact: true }).selectOption(receipt.recovery_id);
  await page.screenshot({ path: path.join(output, "support-recovery-panel.png"), fullPage: true });
  const proposedResponse = page.waitForResponse(response => response.url().endsWith("/studio/proposals") && response.request().method() === "POST");
  await page.getByRole("button", { name: "Preview recovered cells", exact: true }).click();
  const proposal = await read(await proposedResponse);
  const slug = `background-${receipt.recovery_id.slice(-8)}`;
  await page.waitForFunction(slug => window.__sceneStudioWalker?.elements.some(element => element.slug === slug), slug);
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).check();
  await page.waitForFunction(slug => {
    const walker = window.__sceneStudioWalker;
    return walker?.colliderTris > 0 && walker.elements.filter(element => element.object.visible).length === 1
      && walker.elements.some(element => element.slug === slug && element.object.visible);
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
    await page.getByLabel("Revision-pinned 3D scene").screenshot({ path: path.join(output, `support-angle-${views.length}.png`) });
  }
  if (views.some(view => view.collider_triangles !== before + view.patch_triangles || view.patch_triangles <= 0)) throw new Error("Anchored cells did not enter geometry and collision together");
  if (await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).isEnabled()) throw new Error("Isolated support inspection granted full-context approval");
  if (JSON.stringify((await read(await request.get(api))).active) !== JSON.stringify(initial.active)) throw new Error("Support preview changed the active scene");
  await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).uncheck();
  await writeFile(path.join(output, "private-rollback-checkpoint.json"), JSON.stringify({ base: initial.active, preview_revision: proposal.preview_revision }, null, 2));
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(slug => window.__sceneStudioWalker?.elements.some(element => element.slug === slug), slug);
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  const baseline = initial.history.find(item => item.revision_id === initial.active.revision_id);
  await page.getByRole("button", { name: `${baseline.operation.kind} · ${baseline.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
  const restored = await read(await request.get(api));
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state)) throw new Error("Support undo did not restore baseline state");
  const report = { recovery_id: receipt.recovery_id, anchors: receipt.report.plane.selection, plane: receipt.report.plane,
    supported_fraction: receipt.report.supported_fraction, unknown_fraction: receipt.report.unknown_fraction, photos, views,
    baseline_generation: expectedGeneration, restored_generation: restored.active.generation, exact_state_undo: true,
    page_errors: errors, scope: "private source-photo anchor UI plus mesh-only support preview/apply/reload/undo; not full removal, generated completion, semantic ground truth or navigation acceptance" };
  await writeFile(path.join(output, "support-surface-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length) throw new Error("Support browser proof reported page errors");
}
