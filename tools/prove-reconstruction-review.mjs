import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.SPATIAL_PROOF_PLAYWRIGHT);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../data/spatial/reconstruction-comparisons");
const filename = await realpath(process.argv[2]);
if (!filename.startsWith(`${root}${path.sep}`) || path.basename(filename) !== "index.html") throw new Error("Use a retained private comparison review");
const directory = path.dirname(filename);
const isSurface = path.basename(directory).startsWith("surfaces-");
const receiptPath = path.join(directory, isSurface ? "comparison.json" : "review.json");
const before = await readFile(receiptPath, "utf8");
const receipt = JSON.parse(before);
if (receipt.schema !== (isSurface ? "dev.splatlab.surface-comparison/v1" : "dev.splatlab.reconstruction-review/v1")) throw new Error("Unrecognized review receipt");
const methodCount = Object.keys(receipt.methods).length;
const viewCount = isSurface
  ? JSON.parse(await readFile(path.join(directory, Object.keys(receipt.methods)[0], "run.json"), "utf8")).evaluation.length
  : Object.keys(Object.values(receipt.methods)[0].held_out_depth).length;
const expectedImages = viewCount * (isSurface ? methodCount * 2 + 1 : methodCount + 1);
const attempt = process.argv[3] || "browser-proof";
if (!/^browser-proof(?:-[0-9]{2})?$/.test(attempt)) throw new Error("Use a bounded browser-proof attempt name");
const output = path.join(directory, attempt);
await mkdir(output);
const errors = [];
const checks = [];
const browser = await chromium.launch({
  executablePath: path.join(process.env.HOME, ".cache/ms-playwright/chromium_headless_shell-1223/chrome-headless-shell-linux64/chrome-headless-shell"),
  args: ["--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
let status = "failed";
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", error => errors.push(error.message));
  page.on("requestfailed", request => {
    if (request.resourceType() === "image") errors.push(`${request.url()}: ${request.failure()?.errorText}`);
  });
  await page.route("**/*", route => route.request().url().startsWith(pathToFileURL(path.dirname(directory)).href + "/") ? route.continue() : route.abort());
  await page.goto(pathToFileURL(filename).href, { waitUntil: "load" });
  await page.locator("img").evaluateAll(elements => {
    for (const element of elements) element.loading = "eager";
  });
  await page.waitForFunction(() => [...document.images].every(element => element.complete), undefined, { timeout: 30000 });
  const images = await page.locator("img").evaluateAll(async elements => {
    for (const element of elements) {
      if (!element.naturalWidth) throw new Error(`Image did not load: ${element.src}`);
      await element.decode().catch(error => { throw new Error(`${element.src}: ${error.message}`); });
    }
    return elements.map(element => ({ width: element.naturalWidth, height: element.naturalHeight }));
  });
  if (images.length !== expectedImages || images.some(image => image.width < 1)) throw new Error("Not every matched-camera image decoded");
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    const layout = await page.evaluate(() => ({ width: innerWidth, content: document.documentElement.scrollWidth }));
    checks.push(layout);
    if (layout.content > width) throw new Error("Review has horizontal overflow");
    await page.screenshot({ path: path.join(output, `review-${width}.png`) });
  }
  if (errors.length || before !== await readFile(receiptPath, "utf8")) throw new Error("Read-only review changed state or raised errors");
  status = "passed";
  checks.push({ images_decoded: images.length });
} catch (error) {
  errors.push(error.message);
  throw error;
} finally {
  await writeFile(path.join(output, "result.json"), JSON.stringify({ status, errors, checks, review_sha256: receipt.sha256 }, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ status, checks, errors }));
