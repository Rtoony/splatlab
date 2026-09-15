import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveStudio(page, request, base, jobId, output, errors, expectedGeneration = 0) {
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  let initial = await (await request.get(api)).json();
  if (initial.active.generation !== expectedGeneration) throw new Error("Fixture generation differs from the explicitly selected generation; inspect before rerunning");
  if (initial.state.semantics["review-room"]) throw new Error("Prior review-room state is still present; inspect before starting a new proof");
  await page.goto(`${base}/studio/${jobId}?appearance=mesh`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.__sceneStudioWalker?.elements.length), undefined, { timeout: 90000 });
  const refreshed = initial.legacy_sources_changed;
  if (refreshed) {
    await page.getByRole("button", { name: "Preview fresh capture baseline", exact: true }).click();
    const review = page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." });
    await review.check();
    if ((await (await request.get(api)).json()).active.generation !== expectedGeneration) throw new Error("Refresh preview changed active state");
    await page.getByRole("button", { name: "Apply together", exact: true }).click();
    await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
    initial = await (await request.get(api)).json();
    if (initial.legacy_sources_changed) throw new Error("Applied refresh still reports stale capture inputs");
  }
  await page.waitForFunction(() => Boolean(window.__sceneStudioWalker?.elements.length), undefined, { timeout: 90000 });
  const before = await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    const bounds = walker.sceneBox;
    const span = Math.max(bounds.max.x - bounds.min.x, bounds.max.z - bounds.min.z);
    walker.camera.position.set(bounds.max.x + span, bounds.max.y + span * 0.75, bounds.min.z - span);
    walker.camera.lookAt((bounds.min.x + bounds.max.x) / 2, bounds.min.y + 1, (bounds.min.z + bounds.max.z) / 2);
    return { elements: walker.elements.map(element => element.slug), collider_triangles: walker.colliderTris };
  });
  await page.getByLabel("Versioned Blender export").selectOption(initial.blender_exports[0].filename);
  await page.getByLabel("New element name").fill("review-room");
  await page.getByLabel("Intent / review note").fill("Review adjacent authored room; exercise atomic apply and full undo on the private fixture.");
  await page.getByRole("button", { name: "Build review proposal", exact: true }).click();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.some(element => element.slug === "review-room"), undefined, { timeout: 90000 });
  const preview = await page.evaluate(() => ({ elements: window.__sceneStudioWalker.elements.map(element => element.slug), collider_triangles: window.__sceneStudioWalker.colliderTris }));
  const during = await (await request.get(api)).json();
  if (during.active.generation !== initial.active.generation) throw new Error("Preview mutated the active scene");
  if (preview.collider_triangles <= before.collider_triangles) throw new Error("Authored room did not enter the collider");
  await page.screenshot({ path: path.join(output, "scene-proposal.png"), fullPage: true });
  await page.getByRole("button", { name: "Compare original", exact: true }).click();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.length && !window.__sceneStudioWalker.elements.some(element => element.slug === "review-room"));
  await page.screenshot({ path: path.join(output, "scene-base.png"), fullPage: true });
  await page.getByRole("button", { name: "Show proposal", exact: true }).click();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.some(element => element.slug === "review-room"));
  await page.getByRole("button", { name: "Keep unapplied", exact: true }).click();
  await page.reload({ waitUntil: "domcontentloaded" });
  const saved = during.proposals.find(proposal => proposal.operation.slug === "review-room" && !proposal.stale);
  if (!saved) throw new Error("Review proposal was not durably listed");
  await page.getByRole("button", { name: `place · ${saved.proposal_id.slice(-8)} · unapplied`, exact: true }).click();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.some(element => element.slug === "review-room"));
  await page.getByRole("checkbox", { name: "I reviewed the geometry, appearance and collision implications." }).check();
  await page.getByRole("button", { name: "Apply together", exact: true }).click();
  await page.getByText(`Active generation ${initial.active.generation + 1}`, { exact: true }).waitFor();
  const applied = await (await request.get(api)).json();
  if (!applied.state.semantics["review-room"].active) throw new Error("Applied semantics did not advance");
  const baselineKind = initial.history.find(revision => revision.revision_id === initial.active.revision_id).operation.kind;
  await page.getByRole("button", { name: `${baselineKind} · ${initial.active.revision_id.slice(-8)}`, exact: true }).click();
  await page.getByText(`Active generation ${initial.active.generation + 2}`, { exact: true }).waitFor();
  await page.waitForFunction(() => window.__sceneStudioWalker?.elements.length && !window.__sceneStudioWalker.elements.some(element => element.slug === "review-room"));
  const restored = await (await request.get(api)).json();
  if (JSON.stringify(restored.state) !== JSON.stringify(initial.state)) throw new Error("Undo did not restore complete scene state");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(output, "scene-mobile.png"), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  const report = { job_id: jobId, before, preview, initial_generation: initial.active.generation, applied_generation: applied.active.generation,
    restored_generation: restored.active.generation, baseline_refreshed: refreshed,
    baseline_operation: baselineKind, saved_proposal_reopened: true,
    mobile_horizontal_overflow: overflow, exact_state_restored: true, page_errors: errors,
    scope: "private clone of real capture; source job unchanged; software-rendered mesh/collision proof, not photoreal reconstruction acceptance" };
  await writeFile(path.join(output, "studio-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length || overflow) throw new Error("Studio browser proof reported page errors or overflow");
}
