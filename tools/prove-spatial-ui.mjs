import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { proveStudio } from "./prove-scene-studio.mjs";
import { proveFixedWalking } from "./prove-fixed-walking.mjs";
import { proveBackground } from "./prove-background-recovery.mjs";
import { proveRemoval } from "./prove-captured-removal.mjs";
import { inspectCaptured } from "./inspect-captured-view.mjs";
import { proveSelection } from "./prove-selection-review.mjs";
import { proveRefinedRemoval } from "./prove-refined-removal.mjs";
import { proveSupport } from "./prove-support-surface.mjs";
import { proveCompletion } from "./prove-background-completion.mjs";
import { proveGeneratedGaussians } from "./prove-generated-gaussians.mjs";
import { proveNativeRoomResume } from "./prove-native-room-resume.mjs";
import { proveBackgroundModels } from "./prove-background-models.mjs";
import { provePerformance } from "./prove-spatial-performance.mjs";
import { proveBackgroundVisibility } from "./prove-background-visibility.mjs";
import { proveStructuralSurfaces } from "./prove-structural-surfaces.mjs";
import { proveArchitecturalEdit } from "./prove-architectural-edit.mjs";

const require = createRequire(import.meta.url);
const { chromium, request } = require(process.env.SPATIAL_PROOF_PLAYWRIGHT);
const base = process.env.SPATIAL_PROOF_BASE;
const output = process.env.SPATIAL_PROOF_OUTPUT;
await mkdir(output, { recursive: true });
const unauthenticated = await request.newContext();
const refused = await unauthenticated.get(`${base}/api/splat/captures`);
if (refused.status() !== 401) throw new Error("Capture API must require authentication");
if (process.env.SPATIAL_STUDIO_JOB) {
  const studioRefused = await unauthenticated.post(`${base}/api/splat/jobs/${process.env.SPATIAL_STUDIO_JOB}/studio/initialize`);
  if (studioRefused.status() !== 401) throw new Error("Studio mutation API must require authentication");
}
await unauthenticated.dispose();
const gpuBrowser = process.env.SPATIAL_PROOF_GPU === "1";
const browser = await chromium.launch({
  executablePath: path.join(process.env.HOME, gpuBrowser ? ".cache/ms-playwright/chromium-1223/chrome-linux64/chrome" : ".cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell"),
  args: gpuBrowser ? ["--enable-gpu", "--use-angle=vulkan", "--enable-features=Vulkan", "--disable-vulkan-surface"] : ["--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
try {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    extraHTTPHeaders: { Authorization: `Bearer ${process.env.SPATIAL_PROOF_TOKEN}` },
  });
  await context.route("**/*", route => {
    const url = route.request().url();
    return url.startsWith(base) || url.startsWith("data:") || url.startsWith("blob:") ? route.continue() : route.abort();
  });
  const page = await context.newPage();
  const rendererEvidence = await page.evaluate(() => {
    const graphics = document.createElement("canvas").getContext("webgl2");
    if (!graphics) throw new Error("Proof browser has no WebGL2");
    const extension = graphics.getExtension("WEBGL_debug_renderer_info");
    const renderer = graphics.getParameter(extension ? extension.UNMASKED_RENDERER_WEBGL : graphics.RENDERER);
    graphics.getExtension("WEBGL_lose_context")?.loseContext();
    return { renderer };
  });
  await writeFile(path.join(output, "browser-renderer.json"), JSON.stringify({ ...rendererEvidence, requested_gpu: gpuBrowser, browser: browser.version(), scope: "Renderer identity, not application performance acceptance" }, null, 2));
  if (gpuBrowser && (!/NVIDIA/i.test(rendererEvidence.renderer) || /SwiftShader|llvmpipe/i.test(rendererEvidence.renderer))) throw new Error("GPU proof requires actual NVIDIA-backed WebGL");
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  if (process.env.SPATIAL_STUDIO_JOB) {
    const prove = process.env.SPATIAL_STUDIO_COMPLETION ? proveCompletion : process.env.SPATIAL_STUDIO_SUPPORT ? proveSupport : process.env.SPATIAL_STUDIO_REFINED ? proveRefinedRemoval : process.env.SPATIAL_STUDIO_SELECTION ? proveSelection : process.env.SPATIAL_STUDIO_INSPECT ? inspectCaptured : process.env.SPATIAL_STUDIO_REMOVAL ? proveRemoval : process.env.SPATIAL_STUDIO_RECOVERY ? proveBackground : proveStudio;
    try {
      await (process.env.SPATIAL_STUDIO_WALKING ? proveFixedWalking : process.env.SPATIAL_STUDIO_ARCHITECTURE ? proveArchitecturalEdit : process.env.SPATIAL_STUDIO_STRUCTURE ? proveStructuralSurfaces : process.env.SPATIAL_STUDIO_VISIBILITY ? proveBackgroundVisibility : process.env.SPATIAL_STUDIO_PERFORMANCE ? provePerformance : process.env.SPATIAL_COMPLETION_COMPARISON ? proveBackgroundModels : process.env.SPATIAL_NATIVE_RESUME ? proveNativeRoomResume : process.env.SPATIAL_GAUSSIAN_OBJECT ? proveGeneratedGaussians : prove)(page, context.request, base, process.env.SPATIAL_STUDIO_JOB, output, errors, Number(process.env.SPATIAL_STUDIO_GENERATION || 0));
    } catch (error) {
      const failure = { error: error.message, page_errors: errors };
      await writeFile(path.join(output, "failed-proof.json"), JSON.stringify(failure, null, 2));
      try {
        await page.screenshot({ path: path.join(output, "failed-proof.png"), fullPage: true, timeout: 10000 });
      } catch (screenshotError) {
        failure.screenshot_error = screenshotError.message;
        await writeFile(path.join(output, "failed-proof.json"), JSON.stringify(failure, null, 2));
      }
      throw error;
    }
  } else {
  const loaded = page.waitForResponse(response => response.url().endsWith("/panoramas/0") && response.status() === 200);
  await page.goto(`${base}/routes/${process.env.SPATIAL_PROOF_ROUTE}`, { waitUntil: "domcontentloaded" });
  await loaded;
  await page.locator("canvas").waitFor({ state: "visible" });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await page.screenshot({ path: path.join(output, "route-pilot.png"), fullPage: true });
  const nextLoaded = page.waitForResponse(response => response.url().endsWith("/panoramas/1") && response.status() === 200);
  await page.getByRole("button", { name: "Next viewpoint", exact: true }).click();
  await nextLoaded;
  const timeLabel = await page.getByText("35.0s", { exact: true }).textContent();
  await page.getByRole("button", { name: "All routes", exact: true }).click();
  await page.getByRole("heading", { name: "Build a route", exact: true }).waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(output, "route-mobile.png"), fullPage: true });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
  const report = { route: process.env.SPATIAL_PROOF_ROUTE, unauthorized_status: refused.status(), next_viewpoint: timeLabel, mobile_horizontal_overflow: overflow, page_errors: errors, rendering: "software WebGL; not a hardware performance benchmark", publication: "staged loopback server only; no production restart" };
  await writeFile(path.join(output, "browser-proof.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (errors.length || overflow) throw new Error("Browser proof failed; inspect the report");
  }
} finally {
  await browser.close();
}
