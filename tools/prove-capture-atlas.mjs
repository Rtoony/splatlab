import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";

const [directory, proofDirectory, playwrightPath] = process.argv.slice(2);
assert.ok(directory && proofDirectory && playwrightPath, "Supply atlas directory, new proof directory and installed Playwright module path");
const atlas = JSON.parse(fs.readFileSync(path.join(directory, "atlas.json"), "utf8"));
const receipt = JSON.parse(fs.readFileSync(path.join(directory, "receipt.json"), "utf8"));
const digest = value => createHash("sha256").update(value).digest("hex");
for (const [filename, hash] of Object.entries(receipt.files)) assert.equal(digest(fs.readFileSync(path.join(directory, filename))), hash);
fs.mkdirSync(proofDirectory, {recursive:false});
const {chromium} = await import(pathToFileURL(playwrightPath).href);
const browser = await chromium.launch({headless:true, args:["--disable-gpu", "--disable-software-rasterizer", "--disable-dev-shm-usage"]});
const errors = [], external = [], seen = [];
try {
  const page = await browser.newPage({viewport:{width:1440,height:1000},deviceScaleFactor:1});
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {if (!request.url().startsWith("file:") && !request.url().startsWith("data:")) external.push(request.url());});
  await page.goto(pathToFileURL(path.resolve(directory, "index.html")).href, {waitUntil:"load"});
  const pixelHash = async () => digest(await page.locator("#panorama").evaluate(canvas => canvas.toDataURL()));
  const ready = async () => {
    await page.waitForFunction(() => {
      const canvas = document.getElementById("panorama");
      const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
      return pixels.some((value, index) => index % 4 === 3 && value === 255);
    });
  };
  await ready();
  await page.screenshot({path:path.join(proofDirectory,"overview.png"),fullPage:true});
  for (const [index, view] of atlas.views.entries()) {
    await page.getByRole("button", {name:`Select viewpoint ${index+1}`,exact:true}).click();
    await ready();
    assert.match(await page.locator("#view-title").innerText(), new RegExp(`^View ${index+1} ·`));
    const before = await pixelHash();
    const beforePixels = await page.locator("#panorama").evaluate(canvas => canvas.toDataURL());
    await page.locator("#panorama").focus();
    await page.keyboard.press("ArrowRight");
    await page.waitForFunction(previous => document.getElementById("panorama").toDataURL() !== previous,
      beforePixels);
    const turned = await pixelHash();
    assert.notEqual(turned, before, "Look-around changes the actual rendered viewpoint");
    await page.locator("#reset").click();
    await page.waitForFunction(previous => document.getElementById("panorama").toDataURL() === previous, beforePixels);
    assert.equal(await pixelHash(), before, "Reset returns exact original viewing direction");
    seen.push({view_id:view.view_id,source_sha256:view.sha256,render_sha256:before,turned_sha256:turned,location_status:view.location?.status || "unplaced"});
    if ([9,12,14,16,18].includes(index)) await page.screenshot({path:path.join(proofDirectory,view.view_id+".png"),fullPage:true});
  }
  assert.equal(new Set(seen.map(view=>view.render_sha256)).size, atlas.views.length, "Distinct actual source views render distinctly");
  const emptyClip = atlas.clips.find(clip => !clip.view_ids.length);
  if (emptyClip) {
    await page.locator("#clip").selectOption(emptyClip.capture_id);
    assert.equal(await page.locator("#empty").isVisible(), true);
    assert.equal(await page.locator("#next").isDisabled(), true);
  }
  const located = atlas.views.findIndex(view => view.location);
  await page.getByRole("button", {name:`Viewpoint ${located+1}`,exact:true}).focus();
  await page.keyboard.press("Enter");
  await ready();
  assert.match(await page.locator("#view-title").innerText(), new RegExp(`^View ${located+1} ·`));
  await page.locator("#flip").click();
  await page.waitForTimeout(180);
  await page.screenshot({path:path.join(proofDirectory,"rotation-control.png"),fullPage:true});
  await page.locator("#reset").click();
  await page.waitForTimeout(180);
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(proofDirectory,"phone.png"),fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth), true, "No mobile horizontal overflow");
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  for (const [filename, hash] of Object.entries(receipt.files)) assert.equal(digest(fs.readFileSync(path.join(directory, filename))), hash);
  const result = {status:"passed",atlas_id:atlas.atlas_id,views:seen,errors,external,source_files_unchanged:true,
    browser_mode:"CPU canvas; Chromium GPU and software 3D rasterizer disabled",mobile_width:390};
  fs.writeFileSync(path.join(proofDirectory,"proof.json"), JSON.stringify(result,null,2));
  console.log(JSON.stringify({status:result.status,atlas_id:atlas.atlas_id,views:seen.length,proofDirectory}));
} finally {
  await browser.close();
}
