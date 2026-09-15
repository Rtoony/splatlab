import { writeFile } from "node:fs/promises";
import path from "node:path";

export async function provePerformance(page, request, base, jobId, output, errors, expectedGeneration) {
  if (jobId !== "splat_c0ffee" || process.env.SPATIAL_PROOF_GPU !== "1") throw new Error("Performance measurement requires the private NVIDIA fixture");
  const api = `${base}/api/splat/jobs/${jobId}/studio`;
  const read = async response => {
    if (!response.ok()) throw new Error(`Performance input API failed: ${response.status()}`);
    return response.json();
  };
  const initial = await read(await request.get(api));
  if (initial.active.generation !== expectedGeneration || initial.legacy_sources_changed) throw new Error("Performance fixture changed; inspect its baseline first");
  const studies = await read(await request.get(api + "/selection-reviews"));
  const study = studies.reviews.findLast(item => item.cameras.some(camera => camera.image_id === 58) && item.cameras.some(camera => camera.image_id === 47));
  if (!study) throw new Error("Recorded camera path is unavailable");
  const sourceRevision = await read(await request.get(api + "/revisions/" + study.base.revision_id));
  if (JSON.stringify(sourceRevision.state.calibration) !== JSON.stringify(initial.state.calibration)) throw new Error("Recorded camera calibration does not match the baseline");
  const cameras = [47, 58].map(imageId => study.cameras.find(camera => camera.image_id === imageId));
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.addInitScript(() => document.addEventListener("DOMContentLoaded", () => {
    Object.defineProperty(window, "__sceneStudioWalker", { configurable: true, set(walker) {
      walker.stop();
      Object.getPrototypeOf(walker).start = function () { this.stop(); };
      Object.defineProperty(window, "__sceneStudioWalker", { value: walker, writable: true, configurable: true });
    } });
    const style = document.createElement("style");
    style.textContent = 'div:has(> canvas[aria-label="Revision-pinned 3D scene"]) { position: fixed !important; inset: 0 !important; width: 1920px !important; height: 1080px !important; min-height: 1080px !important; z-index: 100 !important; border-radius: 0 !important; }';
    document.head.append(style);
  }));
  await page.goto(`${base}/studio/${jobId}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__sceneStudioWalker?.backdrop?.numSplats > 0 && window.__sceneStudioWalker.colliderTris > 0, null, { timeout: 90000 });
  const identity = await page.evaluate(() => {
    const walker = window.__sceneStudioWalker;
    walker.stop();
    walker.setFlying(true);
    walker.keys.clear();
    walker.renderer.setPixelRatio(1);
    walker.renderer.setSize(1920, 1080, false);
    walker.camera.aspect = 1920 / 1080;
    walker.camera.updateProjectionMatrix();
    walker.spark.autoUpdate = true;
    const context = walker.renderer.getContext();
    const extension = context.getExtension("WEBGL_debug_renderer_info");
    return { width: context.drawingBufferWidth, height: context.drawingBufferHeight, pixel_ratio: walker.renderer.getPixelRatio(),
      renderer: context.getParameter(extension ? extension.UNMASKED_RENDERER_WEBGL : context.RENDERER),
      captured_rows: walker.backdrop.numSplats, collision_triangles: walker.colliderTris,
      spark_lod_enabled: walker.spark.enableLod, spark_auto_update: walker.spark.autoUpdate,
      worker_loop: "actual WorldWalker.frame, manually scheduled once per requestAnimationFrame", walking_collision_tested: false };
  });
  if (identity.width !== 1920 || identity.height !== 1080 || identity.pixel_ratio !== 1 || !/NVIDIA/.test(identity.renderer)) throw new Error("Measured viewport is not actual 1080p NVIDIA rendering");
  await writeFile(path.join(output, "performance-identity.json"), JSON.stringify(identity, null, 2));
  const measurements = [];
  for (const mode of ["stationary", "moving"]) {
    console.log(JSON.stringify({ stage: "1080p-performance", mode, warmup_frames: 60, measured_frames: 360 }));
    const result = await page.evaluate(async ({ cameras, mode }) => {
      const walker = window.__sceneStudioWalker;
      const source = cameras.map(camera => {
        const transform = new walker.camera.matrix.constructor().set(...camera.world_to_camera.flat()).invert();
        const values = transform.elements;
        const target = walker.camera.clone();
        target.position.setFromMatrixPosition(transform);
        target.up.set(-values[4], -values[5], -values[6]).normalize();
        target.lookAt(values[12] + values[8], values[13] + values[9], values[14] + values[10]);
        return target;
      });
      walker.camera.fov = 2 * Math.atan(cameras[0].height / (2 * cameras[0].parameters[1])) * 180 / Math.PI;
      walker.camera.updateProjectionMatrix();
      const started = performance.now();
      const intervals = [], cpu = [], pending = [];
      const pathSamples = [], displayedCounts = [], gpuQueries = [];
      const context = walker.renderer.getContext();
      const timer = context.getExtension("EXT_disjoint_timer_query_webgl2");
      const renderFrameBefore = walker.renderer.info.render.frame;
      let previous = null;
      for (let frame = 0; frame < 420; frame += 1) {
        const timestamp = await new Promise(resolve => requestAnimationFrame(resolve));
        if (performance.now() - started > 90000) throw new Error("1080p sample exceeded its 90-second scenario budget");
        const fraction = mode === "moving" ? (1 - Math.cos(Math.max(0, frame - 60) / 359 * Math.PI)) / 2 : 0;
        walker.camera.position.copy(source[0].position).lerp(source[1].position, fraction);
        walker.camera.quaternion.copy(source[0].quaternion).slerp(source[1].quaternion, fraction);
        const submitted = performance.now();
        const query = frame >= 60 && timer ? context.createQuery() : null;
        if (query) context.beginQuery(timer.TIME_ELAPSED_EXT, query);
        walker.frame();
        if (query) {
          context.endQuery(timer.TIME_ELAPSED_EXT);
          gpuQueries.push(query);
        }
        if (frame >= 60) {
          intervals.push(timestamp - previous);
          cpu.push(performance.now() - submitted);
          pending.push(Boolean(walker.spark.sorting || walker.spark.sortDirty));
          displayedCounts.push(walker.spark.display.numSplats);
          if ([60, 240, 419].includes(frame)) pathSamples.push({ frame, fraction, position: walker.camera.position.toArray(),
            quaternion: walker.camera.quaternion.toArray(), displayed_splats: walker.spark.display.numSplats, active_splats: walker.spark.activeSplats });
        }
        previous = timestamp;
      }
      const summarize = values => {
        const sorted = [...values].sort((first, second) => first - second);
        return { mean_ms: values.reduce((total, value) => total + value, 0) / values.length,
          p50_ms: sorted[Math.ceil(sorted.length * .5) - 1], p95_ms: sorted[Math.ceil(sorted.length * .95) - 1],
          p99_ms: sorted[Math.ceil(sorted.length * .99) - 1], max_ms: sorted.at(-1) };
      };
      const timing = summarize(intervals);
      const renderFrames = walker.renderer.info.render.frame - renderFrameBefore;
      const gpuTimes = [];
      const gpuDeadline = performance.now() + 10000;
      let gpuValid = Boolean(timer) && gpuQueries.length === 360;
      try {
        for (const query of gpuQueries) {
          while (!context.getQueryParameter(query, context.QUERY_RESULT_AVAILABLE) && performance.now() < gpuDeadline) await new Promise(resolve => setTimeout(resolve, 10));
          if (context.getParameter(timer.GPU_DISJOINT_EXT) || !context.getQueryParameter(query, context.QUERY_RESULT_AVAILABLE)) {
            gpuValid = false;
            break;
          }
          gpuTimes.push(context.getQueryParameter(query, context.QUERY_RESULT) / 1e6);
        }
      } finally {
        for (const query of gpuQueries) context.deleteQuery(query);
      }
      walker.renderer.render(walker.scene, walker.camera);
      const png = walker.renderer.domElement.toDataURL("image/png");
      if (context.isContextLost()) throw new Error("Measured rendering lost its WebGL context");
      return { mode, frames: intervals.length, warmup_frames: 60, elapsed_seconds: (performance.now() - started) / 1000,
        frame_intervals: timing, mean_fps: 1000 / timing.mean_ms, javascript_frame: summarize(cpu),
        frames_over_33_34_ms: intervals.filter(value => value > 33.34).length,
        frames_with_pending_sort: pending.filter(Boolean).length, raw_interval_ms: intervals, raw_javascript_ms: cpu,
        render_frame_count: renderFrames, path_samples: pathSamples, displayed_splats_min: Math.min(...displayedCounts), displayed_splats_max: Math.max(...displayedCounts),
        gpu_timer: { extension_available: Boolean(timer), valid: gpuValid, samples: gpuTimes.length, timings: gpuValid ? summarize(gpuTimes) : null },
        passes_scoped_30fps_target: timing.mean_ms <= 1000 / 30 && timing.p95_ms <= 33.34, canvas_png: png };
    }, { cameras, mode });
    const { canvas_png: canvasPng, ...measurement } = result;
    await writeFile(path.join(output, `performance-${mode}.png`), Buffer.from(canvasPng.split(",")[1], "base64"));
    await writeFile(path.join(output, `performance-${mode}.json`), JSON.stringify(measurement, null, 2));
    measurements.push(measurement);
  }
  const final = await read(await request.get(api));
  if (JSON.stringify(final.active) !== JSON.stringify(initial.active) || JSON.stringify(final.state) !== JSON.stringify(initial.state)) throw new Error("Read-only performance measurement changed the active scene");
  const report = { status: "MEASURED", active: initial.active, identity, measurements, page_errors: errors, active_scene_unchanged: true,
    scope: "Private headless NVIDIA browser, two recorded-camera scenarios, actual application frame path at 1920x1080 DPR1. RAF intervals and optional disjoint-checked GPU timer samples are separate metrics; neither certifies end-user display latency, walking acceptance or a universal frame-rate guarantee." };
  await writeFile(path.join(output, "spatial-performance.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ ...report, measurements: measurements.map(({ raw_interval_ms, raw_javascript_ms, ...summary }) => summary) }));
  if (errors.length) throw new Error("Performance measurement produced page errors");
}
