import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { architecturalProofReportName, raisedBridgeFloor, architecturalAppearanceModes, legacyArchitecturalProjection } from "./architectural-diagnostic-contract.mjs";

export async function inspectCaptured(page, request, base, jobId, output, errors, expectedGeneration) {
  if (process.env.SPATIAL_INSPECT_ARCHITECTURE_PROOF)
    return inspectArchitecturalAppearance(page, request, base, jobId, output, errors, expectedGeneration);
  const initial = await (await request.get(`${base}/api/splat/jobs/${jobId}/studio`)).json();
  if (initial.active.generation !== expectedGeneration) throw new Error("Inspect the fixture generation before running");
  const messages = [];
  page.on("console", message => {
    if (["warning", "error"].includes(message.type())) messages.push(message.text().slice(0, 2000));
  });
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.__sceneStudioWalker), undefined, { timeout: 90000 });
  const readings = [];
  for (let frame = 0; frame < 4; frame++) {
    const reading = await page.evaluate(() => {
      const walker = window.__sceneStudioWalker;
      walker.stop();
      const element = walker.elements.find(element => element.slug === "cardboard-box-2");
      const center = [(element.box.min.x + element.box.max.x) / 2, (element.box.min.y + element.box.max.y) / 2, (element.box.min.z + element.box.max.z) / 2];
      walker.camera.position.set(center[0] + 1.4, center[1] + 1, center[2]);
      walker.camera.lookAt(...center);
      walker.renderer.render(walker.scene, walker.camera);
      const context = walker.renderer.getContext();
      const samples = new Uint8Array(64 * 64 * 4);
      context.readPixels(Math.floor(context.drawingBufferWidth / 2) - 32, Math.floor(context.drawingBufferHeight / 2) - 32, 64, 64, context.RGBA, context.UNSIGNED_BYTE, samples);
      const colors = new Set();
      for (let offset = 0; offset < samples.length; offset += 4) colors.add(`${samples[offset]},${samples[offset + 1]},${samples[offset + 2]}`);
      return { render: walker.renderer.info.render, context_lost: context.isContextLost(), error: context.getError(),
        buffer: [context.drawingBufferWidth, context.drawingBufferHeight], canvas: walker.renderer.domElement.getBoundingClientRect().toJSON(),
        camera: walker.camera.position.toArray(), target: center, backdrop_scale: walker.backdrop.scale.x,
        splats: walker.backdrop.numSplats, pixel_colors: colors.size, first_pixel: [...samples.slice(0, 4)] };
    });
    readings.push(reading);
    await writeFile(path.join(output, "captured-inspection.json"), JSON.stringify({ readings, messages, errors }, null, 2));
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  await page.screenshot({ path: path.join(output, "captured-inspection.png") });
  const final = await (await request.get(`${base}/api/splat/jobs/${jobId}/studio`)).json();
  if (JSON.stringify(final.active) !== JSON.stringify(initial.active)) throw new Error("Read-only inspection changed the active scene");
  console.log(JSON.stringify({ readings, messages, errors }));
}

async function inspectArchitecturalAppearance(page, request, base, jobId, output, errors, expectedGeneration) {
  assert.equal(jobId, "splat_c0ffee");
  assert.equal(process.env.SPATIAL_PROOF_GPU, "1");
  assert.equal(new URL(base).hostname, "127.0.0.1");
  const projectionComparison = process.env.SPATIAL_ARCHITECTURE_PROJECTION_COMPARISON === "1";
  const shaderErrors = [];
  const recordConsole = message => {
    if (message.type() === "error" && /WebGLProgram|shader.*error|VALIDATE_STATUS|GL_INVALID/i.test(message.text()))
      shaderErrors.push(message.text());
  };
  page.on("console", recordConsole);
  const proof = path.resolve(process.env.SPATIAL_INSPECT_ARCHITECTURE_PROOF);
  assert.ok(proof.startsWith("/home/rtoony/reports/2026-09-06-splatlab-next-phases/"));
  const read = async response => {
    assert.equal(response.ok(), true);
    return response.json();
  };
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const initial = await read(await request.get(api));
  assert.equal(initial.active.generation, expectedGeneration);
  const checkpoint = JSON.parse(await readFile(path.join(proof, "private-rollback-checkpoint.json"), "utf8"));
  const reportName = architecturalProofReportName(await readdir(proof));
  const browser = JSON.parse(await readFile(path.join(proof, reportName), "utf8"));
  assert.equal(browser.status, reportName === "architectural-preview-proof.json" ? "passed_preview_only" : "passed");
  assert.deepEqual(initial.active, browser.final);
  const revisionId = checkpoint.preview_revision;
  assert.match(revisionId, /^scene_[a-f0-9]{24}$/);
  const revision = await read(await request.get(api + "/revisions/" + revisionId));
  const study = (await read(await request.get(api + "/architectural-edits"))).edits.find(item => item.architecture_id === browser.architecture_id);
  assert.ok(study && study.result.sha256 === browser.result_sha256);
  const traceRecord = browser.rendered_traces.find(item => item.profile === "viewer-default-dimensions" && item.reverse === false);
  const traceBytes = await readFile(path.join(proof, traceRecord.filename));
  assert.equal(createHash("sha256").update(traceBytes).digest("hex"), traceRecord.sha256);
  const trace = JSON.parse(traceBytes.toString());
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__sceneStudioWalker?.bvh && window.__sceneStudioWalker.backdrop);
  await page.evaluate(async ({ revision, revisionId, jobId }) => {
    const walker = window.__sceneStudioWalker;
    walker.stop();
    const canvas = walker.renderer.domElement;
    canvas.parentElement.style.width = "960px";
    canvas.parentElement.style.height = "640px";
    walker.resize();
    walker.setParams({ physicsProps: false });
    const source = { kind: "local", describe: revisionId,
      fileUrl: name => `/api/splat/jobs/${jobId}/studio/revisions/${revisionId}/artifact?key=${encodeURIComponent("_world/" + name)}` };
    await walker.loadWorld(source, revision.viewer);
    walker.setPluckDoc(revision.state.selections);
    await walker.setBackdrop(revision.backdrop_url, revision.backdrop_meters_per_unit);
    await walker.applyCapturedVisibility(revision.state.hidden_capture_slugs);
    walker.setArchitecturalPortals(revision.state.architectural_portals || []);
  }, { revision, revisionId, jobId });
  const modes = architecturalAppearanceModes(study.result.appearance_method, study.result.geometry_method, projectionComparison);
  const legacyProjection = projectionComparison
    ? legacyArchitecturalProjection(await page.evaluate(() => window.__sceneStudioWalker.spark.material.fragmentShader)) : null;
  const floorVariants = modes.includes("raised-bridge-diagnostic");
  const bridge = floorVariants ? await page.evaluate(slug => {
    const room = window.__sceneStudioWalker.elements.find(element => element.slug === slug);
    if (!room) throw new Error("Missing authored room for floor diagnostic");
    const geometries = [];
    room.object.updateWorldMatrix(true, true);
    room.object.traverse(object => {
      if (!object.isMesh) return;
      if (object.matrixWorld.elements.some((value, index) => value !== (index % 5 === 0 ? 1 : 0)))
        throw new Error("Floor diagnostic requires the retained baked world-metre frame");
      const positions = object.geometry.getAttribute("position");
      geometries.push({ uuid: object.geometry.uuid, positions: Array.from(positions.array) });
    });
    if (geometries.length !== 1) throw new Error("Floor diagnostic requires one retained room mesh");
    return geometries[0];
  }, revision.state.architectural_portals[0].slug) : null;
  const raised = bridge ? raisedBridgeFloor(bridge.positions, study.spec, study.clip.lower[1] + .001) : null;
  const frames = [];
  const poses = trace.snapshots.map(item => ({ position: item.position,
    target: trace.samples.at(-1).position.map((value, axis) => axis === 1 ? item.position[1] : value) }));
  const roomEntry = trace.samples[Math.floor(trace.samples.length * .67)].position;
  poses[2].target = [roomEntry[0], study.spec.origin[1] + .9, roomEntry[2]];
  for (const mode of modes) {
    for (const [index, pose] of poses.entries()) {
      const value = await page.evaluate(async ({ mode, pose, portals, spec, bridge, raised, legacyProjection }) => {
        const walker = window.__sceneStudioWalker;
        walker.stop();
        walker.setParams({ unlit: !mode.startsWith("lit") });
        walker.spark.visible = mode !== "no-capture";
        const clips = structuredClone(portals);
        if (bridge) {
          const positions = mode.startsWith("raised-bridge") ? raised.positions : bridge.positions;
          const room = walker.elements.find(element => element.slug === portals[0].slug);
          room.object.traverse(object => {
            if (!object.isMesh || object.geometry.uuid !== bridge.uuid) return;
            const attribute = object.geometry.getAttribute("position");
            attribute.array.set(positions);
            attribute.needsUpdate = true;
            object.geometry.computeBoundingBox();
            object.geometry.computeBoundingSphere();
          });
        }
        if (mode.includes("threshold-mask")) {
          clips[0].lower[1] = -.001;
          clips[0].room_lower[1] = -.001;
        }
        if (mode.endsWith("room-mask")) clips.push({ ...structuredClone(portals[0]),
          lower: [-spec.room_width / 2 - spec.wall_thickness, -.01, spec.cut_depth / 2],
          upper: [spec.room_width / 2 + spec.wall_thickness, spec.room_height + spec.wall_thickness,
            spec.cut_depth / 2 + spec.room_depth + spec.wall_thickness] });
        walker.__architecturalDepthRestore?.();
        walker.setArchitecturalPortals(clips);
        if (mode.startsWith("legacy-render")) {
          if (!legacyProjection) throw new Error("Missing explicit legacy projection control");
          walker.spark.material.fragmentShader = legacyProjection;
          walker.spark.material.needsUpdate = true;
        }
        const linearDepth = mode.startsWith("legacy-render") || mode === "projection-only-diagnostic";
        if (legacyProjection && walker.renderer.capabilities.logarithmicDepthBuffer !== true)
          throw new Error("Precision comparison requires the product logarithmic depth renderer");
        if (linearDepth) {
          const materials = new Set();
          walker.scene.traverse(object => {
            if (object.material) for (const material of Array.isArray(object.material) ? object.material : [object.material])
              materials.add(material);
          });
          const restores = [];
          for (const material of materials) {
            const compile = material.onBeforeCompile, key = material.customProgramCacheKey;
            material.onBeforeCompile = function(shader, renderer) {
              compile.call(this, shader, renderer);
              shader.fragmentShader = shader.fragmentShader.replace("#include <logdepthbuf_fragment>", "");
            };
            material.customProgramCacheKey = function() { return key.call(this) + "|architectural-linear-depth-control/v1"; };
            material.needsUpdate = true;
            restores.push(() => {
              material.onBeforeCompile = compile;
              material.customProgramCacheKey = key;
              material.needsUpdate = true;
            });
          }
          walker.__architecturalDepthRestore = () => {
            for (const restore of restores) restore();
            delete walker.__architecturalDepthRestore;
          };
        }
        walker.camera.position.fromArray(pose.position);
        walker.camera.up.set(0, 1, 0);
        walker.camera.lookAt(...pose.target);
        walker.camera.fov = 65;
        walker.camera.updateProjectionMatrix();
        const spark = walker.spark;
        spark.autoUpdate = false;
        const started = performance.now();
        const settle = async () => {
          while (spark.sorting || spark.sortDirty || spark.sortTimeoutId !== -1 || spark.updateTimeoutId !== -1) {
            if (performance.now() - started > 30000) throw new Error("Appearance diagnostic did not settle");
            await new Promise(resolve => setTimeout(resolve, 10));
          }
        };
        await settle();
        for (let pass = 0; pass < 2; pass++) {
          await spark.update({ scene: walker.scene, camera: walker.camera });
          await settle();
          walker.renderer.render(walker.scene, walker.camera);
        }
        const graphics = walker.renderer.getContext();
        if (graphics.isContextLost() || graphics.getError() !== graphics.NO_ERROR) throw new Error("Appearance diagnostic WebGL error");
        return { png: walker.renderer.domElement.toDataURL("image/png"), width: graphics.drawingBufferWidth,
          height: graphics.drawingBufferHeight, position: walker.camera.position.toArray(), quaternion: walker.camera.quaternion.toArray(),
          splat_depth_test: spark.material.depthTest, splat_depth_write: spark.material.depthWrite,
          camera_near: walker.camera.near, camera_far: walker.camera.far,
          depth_bits: graphics.getParameter(graphics.DEPTH_BITS),
          logarithmic_depth_renderer: walker.renderer.capabilities.logarithmicDepthBuffer,
          legacy_linear_depth_control: linearDepth,
          projection_matrix: walker.camera.projectionMatrix.toArray(),
          inverse_projection_matrix: walker.camera.projectionMatrixInverse.toArray(),
          fragment_projection: spark.material.fragmentShader.includes("gl_FragCoord.w") ? "homogeneous-w/v1" : "legacy-ndc-depth/v1",
          bridge_visual_lift_m: mode.startsWith("raised-bridge") ? raised.lift_m : 0,
          bridge_changed_vertices: mode.startsWith("raised-bridge") ? raised.indices.length : 0,
          clipping_volumes: clips, collider_triangles: walker.colliderTris };
      }, { mode, pose, portals: revision.state.architectural_portals, spec: study.spec, bridge, raised, legacyProjection });
      const { png, ...metadata } = value;
      const bytes = Buffer.from(png.split(",")[1], "base64");
      const filename = `appearance-${mode}-${index}.png`;
      await writeFile(path.join(output, filename), bytes);
      frames.push({ mode, index, filename, sha256: createHash("sha256").update(bytes).digest("hex"), ...metadata });
    }
  }
  const final = await read(await request.get(api));
  assert.deepEqual(final.active, initial.active);
  assert.deepEqual(final.state, initial.state);
  assert.deepEqual(errors, []);
  assert.deepEqual(shaderErrors, []);
  if (projectionComparison) {
    for (const index of [0, 1, 2]) {
      const selected = mode => frames.find(frame => frame.mode === mode && frame.index === index);
      for (const [before, after] of [["original", "original-restored"], ["legacy-render-diagnostic", "legacy-render-restored"]])
        assert.equal(selected(before).sha256, selected(after).sha256, "Projection control did not restore exact pixels");
      for (const frame of frames.filter(frame => frame.index === index))
        for (const key of ["position", "quaternion", "camera_near", "camera_far", "projection_matrix", "clipping_volumes", "collider_triangles"])
          assert.deepEqual(frame[key], selected("original")[key], "Projection comparison changed its scene or camera");
    }
  }
  await writeFile(path.join(output, "architectural-appearance-diagnostic.json"), JSON.stringify({ base: initial.active,
    inspected_revision: revisionId, architecture_id: study.architecture_id, result_sha256: study.result.sha256, frames,
    active_scene_unchanged: true, page_errors: errors, shader_errors: shaderErrors, projection_comparison: projectionComparison,
    retained_report: reportName,
    scope: "Read-only retained preview rendering. Lighting, appearance masks and bridge height changes are in-memory diagnostics only. The diagnostic does not rebuild collision or prove walking, accept a recipe, create a proposal or activate an edit." }, null, 2));
  await page.evaluate(() => { window.__sceneStudioWalker.stop(); });
  page.off("console", recordConsole);
}
