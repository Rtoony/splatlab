import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function proveStructuralSurfaces(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee") throw new Error("Structural review proof only targets the private fixture");
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Structural proof API failed: ${response.status()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  const listed = await read(await request.get(api + "/structural-surfaces"));
  const study = listed.studies.find(item => item.structure_id === process.env.SPATIAL_STUDIO_STRUCTURE);
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed || !study || study.stale || !study.result)
    throw new Error("Structural review requires fresh retained evidence");
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__sceneStudioWalker?.backdrop && window.__sceneStudioWalker?.colliderTris > 0);
  await page.evaluate(() => window.__sceneStudioWalker.stop());
  await page.getByLabel("Retained wall study").selectOption(study.structure_id);
  const panel = page.locator("section").filter({ has: page.getByRole("heading", { name: "Captured wall evidence", exact: true }) });
  const decoded = [];
  for (const camera of study.cameras) {
    await page.getByLabel("Wall evidence photograph").selectOption(String(camera.image_id));
    await page.waitForFunction(({ identifier, imageId }) => {
      const images = [...document.images].filter(image => image.src.includes(`/structural-surfaces/${identifier}/artifact`));
      return images.length === 2 && images.every(image => image.complete && image.naturalWidth === 960)
        && images.some(image => decodeURIComponent(image.src).includes(`wall-tracks-${imageId}.png`));
    }, { identifier: study.structure_id, imageId: camera.image_id });
    const view = study.result.views.find(item => item.image_id === camera.image_id);
    await panel.getByText(`View ${camera.image_id}: ${view.positive_points.toLocaleString()} wall-labelled tracks out of`, { exact: false }).waitFor();
    await panel.screenshot({ path: path.join(output, `wall-evidence-${camera.image_id}.png`) });
    decoded.push({ image_id: camera.image_id, split: camera.split, positive_tracks: view.positive_points });
  }
  if (study.result.status === "no-supported-wall")
    await panel.getByText("No wall has enough tracked support for a fitted surface.", { exact: false }).waitFor();
  const download = await request.get(new URL(await panel.getByRole("link", { name: "Download retained point membership" }).getAttribute("href"), base).href);
  if (!download.ok()) throw new Error("Retained point-membership download failed");
  const payload = await download.body();
  const membershipSha = createHash("sha256").update(payload).digest("hex");
  await page.setViewportSize({ width: 390, height: 844 });
  await panel.scrollIntoViewIfNeeded();
  await panel.screenshot({ path: path.join(output, "wall-evidence-mobile.png") });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.getByLabel("Retained wall study").selectOption(study.structure_id);
  await panel.getByText(`${study.result.accepted_points.toLocaleString()} multi-view-supported wall points`, { exact: false }).waitFor();
  const final = await read(await request.get(api));
  if (JSON.stringify(final.active) !== JSON.stringify(initial.active) || JSON.stringify(final.state) !== JSON.stringify(initial.state))
    throw new Error("Read-only structural review changed active scene state");
  const report = { structure_id: study.structure_id, result_sha256: study.result.sha256, active: final.active,
    decoded, decoded_images: decoded.length * 2, downloaded_membership_bytes: payload.length, downloaded_membership_sha256: membershipSha,
    fitted_planes: study.result.planes.length, accepted_points: study.result.accepted_points, mobile_horizontal_overflow: overflow,
    active_unchanged: true, reload_retains_evidence: true, page_errors: errors,
    scope: "Private photo/mask/track evidence UI, not structural geometry acceptance, a wall cut or navigable extension" };
  await writeFile(path.join(output, "structural-browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (overflow || errors.length) throw new Error("Structural review proof failed");
}
