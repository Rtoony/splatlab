import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { architecturalWorld, validateArchitecturalInputs, validateArchitecturalRoomTriangles, validateCapsuleTrace, validateRenderedCapsuleTrace, LEGACY_GEOMETRY, JOINED_GEOMETRY } from "./architectural-proof-contract.mjs";
import { traceArchitecturalLane, beginRenderedArchitecturalLane, finishRenderedArchitecturalLane } from "./architectural-navigation-proof.mjs";

export async function proveArchitecturalEdit(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee" || process.env.SPATIAL_PROOF_GPU !== "1")
    throw new Error("Architectural acceptance requires the private NVIDIA fixture");
  const endpoint = new URL(base);
  assert.ok(endpoint.protocol === "http:" && endpoint.hostname === "127.0.0.1" && endpoint.port,
    "Architectural proof must target the ephemeral loopback server");
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Architectural proof API failed: ${response.status()}`);
    return response.json();
  };
  const save = (name, value) => writeFile(path.join(output, name), JSON.stringify(value, null, 2));
  const initial = await read(await request.get(api));
  const listing = await read(await request.get(api + "/architectural-edits"));
  const study = listing.edits.find(item => item.architecture_id === process.env.SPATIAL_STUDIO_ARCHITECTURE);
  if (!study) throw new Error("The specified architectural study is missing");
  const navigation = await read(await request.get(api + `/architectural-edits/${study.architecture_id}/artifact?result=true&name=navmesh.json`));
  const contract = validateArchitecturalInputs(initial, study, navigation, expectedGeneration);
  const slug = "extension-" + study.architecture_id.slice(-8);
  const checkpoint = { base: initial.active, preview_revision: null, rollback_from: [], restore_targets: [] };
  const consoleErrors = [];
  const recordConsole = message => {
    if (message.type() === "error" && /WebGLProgram|shader.*error|VALIDATE_STATUS|GL_INVALID/i.test(message.text()))
      consoleErrors.push(message.text());
  };
  page.on("console", recordConsole);
  page.setDefaultTimeout(60000);
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { width: min(960px, 100%) !important; height: 640px !important; min-height: 320px !important; }';
    document.head.append(style);
  }));
  const waitScene = async (paired, captured = true) => {
    await page.waitForFunction(({ identifier, slug, paired, captured }) => {
      const walker = window.__sceneStudioWalker;
      return walker?.colliderTris > 0 && Boolean(walker.backdrop) === captured
        && walker.architecturalPortalState().some(item => item.architecture_id === identifier) === paired
        && walker.elements.some(item => item.slug === slug) === paired;
    }, { identifier: study.architecture_id, slug, paired, captured });
    return page.evaluate(() => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      return { collider_triangles: walker.colliderTris, portals: walker.architecturalPortalState(),
        elements: walker.elements.map(item => ({ slug: item.slug, triangles: item.tris, provenance: item.provenance })),
        hidden_rows: walker.pluckMask?.reduce((total, value) => total + Number(value === 255), 0) || 0 };
    });
  };
  const poses = [
    { position: architecturalWorld(contract.spec, [0, 1.45, -contract.spec.cut_depth / 2 - 1.8]),
      target: architecturalWorld(contract.spec, [0, 1.1, contract.spec.cut_depth / 2 + 1]) },
    { position: architecturalWorld(contract.spec, [-0.5, 1.45, -contract.spec.cut_depth / 2 - 1.5]),
      target: architecturalWorld(contract.spec, [0, 1.1, contract.spec.cut_depth / 2 + 1]) },
  ];
  const frames = [];
  const capture = async (label, pose = poses[0]) => {
    await page.getByLabel("Revision-pinned 3D scene").scrollIntoViewIfNeeded();
    const value = await page.evaluate(async pose => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      walker.camera.position.fromArray(pose.position);
      walker.camera.up.set(0, 1, 0);
      walker.camera.lookAt(...pose.target);
      walker.camera.fov = 65;
      walker.camera.updateProjectionMatrix();
      const spark = walker.spark;
      if (spark) spark.autoUpdate = false;
      const started = performance.now();
      const pending = () => spark && (spark.sorting || spark.sortDirty || spark.sortTimeoutId !== -1 || spark.updateTimeoutId !== -1);
      const settle = async () => {
        while (pending()) {
          if (performance.now() - started > 30000) throw new Error("Architectural frame did not settle");
          await new Promise(resolve => setTimeout(resolve, 10));
        }
      };
      await settle();
      for (let pass = 0; pass < 2; pass++) {
        if (spark) await spark.update({ scene: walker.scene, camera: walker.camera });
        await settle();
        walker.renderer.render(walker.scene, walker.camera);
      }
      if (spark && spark.current.mappingVersion !== spark.display.mappingVersion) throw new Error("The rendered splat mapping is stale");
      const graphics = walker.renderer.getContext();
      const pixels = new Uint8Array(graphics.drawingBufferWidth * graphics.drawingBufferHeight * 4);
      graphics.readPixels(0, 0, graphics.drawingBufferWidth, graphics.drawingBufferHeight, graphics.RGBA, graphics.UNSIGNED_BYTE, pixels);
      const colors = new Set();
      for (let offset = 0; offset < pixels.length; offset += 4) colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
      if (graphics.isContextLost() || graphics.getError() !== graphics.NO_ERROR || colors.size < 16)
        throw new Error("Architectural frame is blank or has a WebGL error");
      return { png: walker.renderer.domElement.toDataURL("image/png"), distinct_colors: colors.size,
        width: graphics.drawingBufferWidth, height: graphics.drawingBufferHeight,
        depth_bits: graphics.getParameter(graphics.DEPTH_BITS),
        logarithmic_depth_renderer: walker.renderer.capabilities.logarithmicDepthBuffer,
        position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray(),
        portals: walker.architecturalPortalState(), guide_present: Boolean(walker.scene.getObjectByName("architectural-draft-guide")) };
    }, pose);
    const { png, ...metrics } = value;
    assert.match(png, /^data:image\/png;base64,/);
    const bytes = Buffer.from(png.split(",")[1], "base64");
    await writeFile(path.join(output, label + ".png"), bytes);
    const record = { label, ...metrics, png_sha256: createHash("sha256").update(bytes).digest("hex") };
    await save(label + ".json", record);
    frames.push(record);
    return record;
  };
  try {
    await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
    const baseline = await waitScene(false);
    const before = await capture("architecture-baseline");
    await page.getByLabel("Retained architectural edit").selectOption(study.architecture_id);
    const geometryMethod = study.geometry_method ?? LEGACY_GEOMETRY;
    const floorDescription = await page.getByLabel("Architectural floor recipe", { exact: true }).innerText();
    if (geometryMethod !== LEGACY_GEOMETRY) assert.match(floorDescription, /16 mm/);
    if (geometryMethod === JOINED_GEOMETRY) assert.match(floorDescription, /overlapping authored faces/);
    await page.getByRole("button", { name: "Adjust as a new candidate", exact: true }).click();
    assert.equal(await page.getByLabel("Authored floor recipe", { exact: true }).inputValue(), geometryMethod,
      "Adjusting a retained candidate must preserve its geometry recipe");
    const floorRecipe = { geometry_method: geometryMethod, floor_finish_m: study.result.metrics?.floor_finish_m ?? 0,
      description: floorDescription, adjusted_candidate_preserves_recipe: true };
    await page.waitForFunction(() => Boolean(window.__sceneStudioWalker?.scene.getObjectByName("architectural-draft-guide")));
    const guide = await capture("architecture-draft");
    assert.equal(guide.guide_present, true);
    assert.notEqual(guide.png_sha256, before.png_sha256, "The positioning guide did not render");
    assert.equal(await page.getByLabel("Walking / collision", { exact: true }).isDisabled(), true);
    await page.getByRole("button", { name: "Hide positioning guide", exact: true }).click();
    await page.waitForFunction(() => !window.__sceneStudioWalker.scene.getObjectByName("architectural-draft-guide"));
    const hidden = await capture("architecture-guide-hidden");
    assert.equal(hidden.png_sha256, before.png_sha256, "Guide cleanup changed the captured view");
    const unchanged = await read(await request.get(api));
    assert.deepEqual(unchanged.active, initial.active);
    assert.deepEqual(unchanged.state, initial.state);
    await page.getByRole("button", { name: "Preview paired doorway and room", exact: true }).click();
    const preview = await waitScene(true);
    validateArchitecturalRoomTriangles(study, preview.elements.find(item => item.slug === slug)?.triangles);
    for (const key of Object.keys(study.clip))
      assert.deepEqual(preview.portals.find(item => item.architecture_id === study.architecture_id)[key], study.clip[key]);
    const listed = await read(await request.get(api));
    const proposal = listed.proposals.findLast(item => item.operation.architecture_id === study.architecture_id && !item.stale);
    assert.ok(proposal, "No current paired proposal was created");
    checkpoint.preview_revision = proposal.preview_revision;
    await save("private-rollback-checkpoint.json", checkpoint);
    const previewState = await read(await request.get(api + "/revisions/" + proposal.preview_revision));
    const paired = await capture("architecture-preview");
    assert.notEqual(paired.png_sha256, before.png_sha256);
    await capture("architecture-preview-oblique", poses[1]);
    const retainedPortals = await page.evaluate(() => window.__sceneStudioWalker.architecturalPortalState());
    await page.evaluate(() => window.__sceneStudioWalker.setArchitecturalPortals([]));
    const unclipped = await capture("architecture-splat-unclipped-diagnostic");
    assert.notEqual(unclipped.png_sha256, paired.png_sha256, "The actual Spark cut has no visible effect");
    await page.evaluate(portals => window.__sceneStudioWalker.setArchitecturalPortals(portals), retainedPortals);
    assert.equal((await capture("architecture-splat-reclipped")).png_sha256, paired.png_sha256);
    await page.getByLabel("Captured appearance", { exact: true }).uncheck();
    await waitScene(true, false);
    const mesh = await capture("architecture-mesh-cut");
    await page.evaluate(() => window.__sceneStudioWalker.setArchitecturalPortals([]));
    const uncutMesh = await capture("architecture-mesh-uncut-diagnostic");
    assert.notEqual(mesh.png_sha256, uncutMesh.png_sha256, "The actual captured mesh cut has no visible effect");
    await page.evaluate(portals => window.__sceneStudioWalker.setArchitecturalPortals(portals), retainedPortals);
    assert.equal((await capture("architecture-mesh-reclipped")).png_sha256, mesh.png_sha256);
    await page.getByLabel("Captured appearance", { exact: true }).check();
    await waitScene(true);
    const traces = [], renderedTraces = [];
    for (const profile of [{ name: "retained-worker", radius: contract.radius, height: contract.height },
      { name: "viewer-default-dimensions", radius: 0.32, height: 2.02 }]) {
      for (const reverse of [false, true]) {
        await page.getByLabel("Total player height (m)", { exact: true }).fill(String(profile.height));
        await page.getByLabel("Player radius (m)", { exact: true }).fill(String(profile.radius));
        await page.waitForFunction(profile => {
          const walker = window.__sceneStudioWalker;
          return walker.params.bodySizing === "fixed-metric" && Math.abs(walker.walkingBody.radiusM - profile.radius) < 1e-9
            && Math.abs(walker.walkingBody.totalHeightM - profile.height) < 1e-9;
        }, profile);
        await page.evaluate(({ lane, floor, profile, reverse }) => {
          const index = reverse ? lane.length - 1 : 0;
          const walker = window.__sceneStudioWalker;
          walker.stop();
          walker.camera.position.set(lane[index][0], floor[index] + profile.height - profile.radius + .01, lane[index][2]);
        }, { lane: contract.lane, floor: contract.floor, profile, reverse });
        await page.getByLabel("Walking / collision", { exact: true }).check();
        await page.bringToFront();
        await page.getByRole("button", { name: "Explore with WASD · Esc to release", exact: true }).click();
        await page.waitForFunction(() => window.__sceneStudioWalker.controls.isLocked);
        const trace = await page.evaluate(traceArchitecturalLane, { ...contract, ...profile, reverse });
        await page.keyboard.press("Escape");
        await page.waitForFunction(() => !document.pointerLockElement && !window.__sceneStudioWalker.controls.isLocked);
        await page.getByLabel("Walking / collision", { exact: true }).uncheck();
        const filename = `architecture-trace-${profile.name}-${reverse ? "return" : "outward"}.json`;
        await save(filename, trace);
        traces.push({ profile: profile.name, ...validateCapsuleTrace(trace, { ...contract, ...profile }, reverse),
          filename, sha256: createHash("sha256").update(await readFile(path.join(output, filename))).digest("hex") });
        const entryLabel = reverse ? "Start in extension" : "Start at captured side";
        await page.getByRole("group", { name: `Walking entry ${study.architecture_id}`, exact: true })
          .getByRole("button", { name: entryLabel, exact: true }).click();
        await page.waitForFunction(() => !window.__sceneStudioWalker.isFlying);
        assert.equal(await page.getByLabel("Walking / collision", { exact: true }).isChecked(), true);
        const entryPose = await page.evaluate(() => ({ position: window.__sceneStudioWalker.camera.position.toArray(),
          body: window.__sceneStudioWalker.walkingBody, flying: window.__sceneStudioWalker.isFlying }));
        const entryIndex = reverse ? contract.lane.length - 1 : 0;
        assert.ok(Math.hypot(entryPose.position[0] - contract.lane[entryIndex][0],
          entryPose.position[2] - contract.lane[entryIndex][2]) < 1e-6, "UI entry did not reach the retained endpoint");
        assert.equal(entryPose.body.radiusM, profile.radius);
        assert.ok(Math.abs(entryPose.body.totalHeightM - profile.height) < 1e-9);
        await page.bringToFront();
        await page.getByRole("button", { name: "Explore with WASD · Esc to release", exact: true }).click();
        await page.waitForFunction(() => window.__sceneStudioWalker.controls.isLocked);
        await page.evaluate(beginRenderedArchitecturalLane, { ...contract, ...profile, reverse });
        let movementError = null;
        try {
          await page.keyboard.down("KeyW");
          await page.waitForFunction(() => window.__architecturalRenderProbe?.done, null, { timeout: 65000 });
        } catch (error) {
          movementError = error;
        } finally {
          await page.keyboard.up("KeyW");
        }
        const rendered = await page.evaluate(finishRenderedArchitecturalLane);
        rendered.ui_entry = { method: "pinned-architectural-entry/v1", button: entryLabel,
          architecture_id: study.architecture_id, ...entryPose };
        const renderedName = `architecture-rendered-${profile.name}-${reverse ? "return" : "outward"}.json`;
        for (const [index, snapshot] of rendered.snapshots.entries()) {
          for (const [suffix, frame] of [["", snapshot], ["-scene-hidden", snapshot.render_control.hidden],
            ["-scene-restored", snapshot.render_control.restored]]) {
            assert.match(frame.png, /^data:image\/png;base64,/);
            const bytes = Buffer.from(frame.png.split(",")[1], "base64");
            frame.filename = renderedName.replace(".json", `-${index}${suffix}.png`);
            frame.sha256 = createHash("sha256").update(bytes).digest("hex");
            await writeFile(path.join(output, frame.filename), bytes);
            delete frame.png;
          }
        }
        await save(renderedName, rendered);
        await page.keyboard.press("Escape");
        await page.waitForFunction(() => !document.pointerLockElement && !window.__sceneStudioWalker.controls.isLocked);
        await page.getByLabel("Walking / collision", { exact: true }).uncheck();
        if (movementError) throw movementError;
        assert.equal(rendered.error, null, rendered.error ?? "Rendered traversal failed");
        assert.equal(rendered.snapshots.length, 3, "Missing rendered start, middle or end walking image");
        renderedTraces.push({ profile: profile.name, ...validateRenderedCapsuleTrace(rendered, { ...contract, ...profile }, reverse),
          filename: renderedName, sha256: createHash("sha256").update(await readFile(path.join(output, renderedName))).digest("hex") });
      }
    }
    assert.equal((await capture("architecture-after-traversal")).png_sha256, paired.png_sha256,
      "Capsule traversal changed the paired appearance");
    if (process.env.SPATIAL_ARCHITECTURE_PREVIEW_ONLY === "1") {
      await page.reload({ waitUntil: "domcontentloaded" });
      await waitScene(false);
      assert.equal((await capture("architecture-preview-dismissed")).png_sha256, before.png_sha256);
      const final = await read(await request.get(api));
      assert.deepEqual(final.active, initial.active);
      assert.deepEqual(final.state, initial.state);
      assert.deepEqual(errors, []);
      assert.deepEqual(consoleErrors, []);
      await save("architectural-preview-proof.json", { status: "passed_preview_only", architecture_id: study.architecture_id,
        floor_recipe: floorRecipe,
        result_sha256: study.result.sha256, base: initial.active, final: final.active, baseline, preview, traces,
        rendered_traces: renderedTraces, frames, preview_dismissed: true, page_errors: errors, shader_errors: consoleErrors,
        scope: "Private shader and rendered-traversal preview only. No activation, reload-of-applied-edit or undo acceptance; photographed unnamed-object protection remains unresolved." });
      return;
    }
    checkpoint.rollback_from.push({ revision_id: proposal.preview_revision, generation: expectedGeneration + 1 });
    await save("private-rollback-checkpoint.json", checkpoint);
    await page.getByLabel("I reviewed the geometry, appearance and collision implications.", { exact: true }).check();
    await page.getByRole("button", { name: "Apply together", exact: true }).click();
    await page.getByText(`Active generation ${expectedGeneration + 1}`, { exact: true }).waitFor();
    await page.reload({ waitUntil: "domcontentloaded" });
    await waitScene(true);
    const applied = await read(await request.get(api));
    assert.deepEqual(applied.state, previewState.state);
    assert.equal(applied.active.revision_id, proposal.preview_revision);
    assert.equal((await capture("architecture-applied-reloaded")).png_sha256, paired.png_sha256);
    const history = applied.history.find(item => item.revision_id === initial.active.revision_id);
    assert.ok(history, "The original revision is absent from undo history");
    await page.getByRole("button", { name: `${history.operation.kind} · ${initial.active.revision_id.slice(-8)}`, exact: true }).click();
    await page.getByText(`Active generation ${expectedGeneration + 2}`, { exact: true }).waitFor();
    const restored = await waitScene(false);
    assert.deepEqual(restored, baseline);
    assert.equal((await capture("architecture-restored")).png_sha256, before.png_sha256);
    await page.reload({ waitUntil: "domcontentloaded" });
    await waitScene(false);
    const final = await read(await request.get(api));
    assert.deepEqual(final.state, initial.state);
    assert.equal(final.active.generation, expectedGeneration + 2);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(output, "architecture-mobile.png"), fullPage: true });
    const mobile = await page.evaluate(() => ({ viewport: innerWidth, document_width: document.documentElement.scrollWidth }));
    assert.ok(mobile.document_width <= mobile.viewport + 1);
    assert.deepEqual(errors, []);
    assert.deepEqual(consoleErrors, []);
    await save("architectural-browser-proof.json", { status: "passed", architecture_id: study.architecture_id,
      floor_recipe: floorRecipe,
      result_sha256: study.result.sha256, base: initial.active, final: final.active, baseline, preview, traces, rendered_traces: renderedTraces, frames, mobile,
      exact_state_undo: true, exact_pixel_undo: true, page_errors: errors, shader_errors: consoleErrors,
      scope: "Private real shader A/B, CPU capsule integration and keyboard-driven rendered traversal at both explicit body profiles; not measured geometry, photo-real quality, touch acceptance or a performance target" });
    assert.ok((await readFile(path.join(output, "architecture-restored.png"))).length > 0);
  } finally {
    page.off("console", recordConsole);
    await page.evaluate(() => { const walker = window.__sceneStudioWalker; walker?.stop(); walker?.controls.unlock(); }).catch(() => {});
  }
}
