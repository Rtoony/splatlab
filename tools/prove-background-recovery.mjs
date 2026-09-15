import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveBackground(page, request, base, jobId, output, errors, expectedGeneration) {
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const initial = await (await request.get(api)).json();
  if (initial.active.generation !== expectedGeneration) throw new Error("Inspect the private fixture generation before this proof");
  const listed = await (await request.get(api + "/recoveries")).json();
  const recovery = listed.recoveries.filter(item => !item.stale).at(-1);
  if (!recovery) throw new Error("No current recovery is available on the private fixture");
  await page.goto(`${base}/studio/${jobId}?appearance=mesh`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.__sceneStudioWalker?.elements.length), undefined, { timeout: 90000 });
  await page.screenshot({ path: path.join(output, "background-panel.png"), fullPage: true });
  await page.getByLabel("Retained recovery").selectOption(recovery.recovery_id);
  const atlas = page.getByAltText("Recovered photo samples; transparent areas are unknown");
  await atlas.waitFor();
  await page.waitForFunction(() => [...document.images].some(image => image.alt.startsWith("Recovered photo") && image.complete && image.naturalWidth > 0));
  const before = await page.evaluate(() => window.__sceneStudioWalker.colliderTris);
  await page.getByRole("button", { name: "Preview recovered cells", exact: true }).click();
  const slug = `background-${recovery.recovery_id.slice(-8)}`;
  await page.waitForFunction(selected => window.__sceneStudioWalker?.elements.some(element => element.slug === selected), slug, { timeout: 90000 });
  await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).check();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.filter(element => element.object.visible).length === 1);
  const views = [];
  for (const angle of [0, 1.2]) {
    views.push(await page.evaluate(({ slug, angle }) => {
      const walker = window.__sceneStudioWalker;
      const element = walker.elements.find(element => element.slug === slug);
      const bounds = element.box;
      const center = { x: (bounds.min.x + bounds.max.x) / 2, y: (bounds.min.y + bounds.max.y) / 2, z: (bounds.min.z + bounds.max.z) / 2 };
      walker.camera.position.set(center.x + Math.cos(angle) * 1.5, center.y + 1.4, center.z + Math.sin(angle) * 1.5);
      walker.camera.lookAt(center.x, center.y, center.z);
      return { camera: walker.camera.position.toArray(), collider_triangles: walker.colliderTris, patch_triangles: element.tris };
    }, { slug, angle }));
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.screenshot({ path: path.join(output, `background-angle-${views.length}.png`), fullPage: true });
  }
  if (views.some(view => view.collider_triangles <= before)) throw new Error("Recovered observed cells did not enter the collider");
  if ((await (await request.get(api)).json()).active.generation !== expectedGeneration) throw new Error("Background preview changed active state");
  if (await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).isEnabled()) throw new Error("Isolated inspection must not stand in for full-context review");
  await page.getByRole("checkbox", { name: "Candidate only (inspection)", exact: true }).uncheck();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.filter(element => element.object.visible).length > 1);
  await page.getByRole("button", { name: "Compare original", exact: true }).click();
  await page.waitForFunction(selected => window.__sceneStudioWalker?.elements.length && !window.__sceneStudioWalker.elements.some(element => element.slug === selected), slug);
  await page.screenshot({ path: path.join(output, "background-original.png"), fullPage: true });
  await page.getByRole("button", { name: "Show proposal", exact: true }).click();
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
  const baseline = initial.history.find(item => item.revision_id === initial.active.revision_id);
  await page.getByRole("button", { name: `${baseline.operation.kind} · ${baseline.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
  const restored = await (await request.get(api)).json();
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state)) throw new Error("Recovery undo changed the baseline state");
  const report = { recovery_id: recovery.recovery_id, supported_fraction: recovery.report.supported_fraction,
    unknown_fraction: recovery.report.unknown_fraction, before_collider_triangles: before, views,
    preview_kept_active_unchanged: true, restored_generation: restored.active.generation, exact_state_restored: true,
    page_errors: errors, scope: "private fixture, two software-rendered camera views, observed cells only; no captured removal or complete-fill acceptance" };
  await writeFile(path.join(output, "background-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length) throw new Error("Background browser proof reported page errors");
}
