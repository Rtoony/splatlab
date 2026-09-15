import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";
import {GLTFLoader} from "../frontend/node_modules/three/examples/jsm/loaders/GLTFLoader.js";
import {Vector3} from "../frontend/node_modules/three/build/three.module.js";

const [directory, proofDirectory, playwrightPath, originalReference] = process.argv.slice(2);
assert.ok(directory && proofDirectory && playwrightPath && originalReference, "Supply delivery, new proof directory, installed Playwright module, original reference directory");
const readJson = filename => JSON.parse(fs.readFileSync(filename, "utf8"));
const digest = value => createHash("sha256").update(value).digest("hex");
const receipt = readJson(path.join(directory, "receipt.json"));
const data = readJson(path.join(directory, "viewer-data.json"));
const original = readJson(path.join(originalReference, "camera-set.json"));
const verifyHashes = () => {
  for (const [filename, checksum] of Object.entries(receipt.files)) assert.equal(digest(fs.readFileSync(path.join(directory, filename))), checksum);
  for (const [filename, checksum] of Object.entries(receipt.source_hashes)) assert.equal(digest(fs.readFileSync(path.join(originalReference, filename))), checksum);
};
verifyHashes();
fs.mkdirSync(proofDirectory, {recursive: false});
const bytes = fs.readFileSync(path.join(directory, "reference.glb"));
const loaded = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), "");
loaded.scene.updateMatrixWorld(true);
const points = [], cameras = [];
loaded.scene.traverse(object => {if (object.isPoints) points.push(object); if (object.isPerspectiveCamera) cameras.push(object);});
assert.equal(points.length, 1, "Independent loader recognizes a point primitive, not invented triangles");
assert.equal(points[0].geometry.getAttribute("position").count, data.points.length);
const decodedPositions = points[0].geometry.getAttribute("position");
for (const [index, expected] of data.points.entries()) {
  assert.deepEqual([decodedPositions.getX(index), decodedPositions.getY(index), decodedPositions.getZ(index)], expected);
}
assert.equal(cameras.length, data.cameras.frames.length);
const frameKey = frame => `${frame.source_image}:${frame.virtual_yaw_deg}`;
const originals = new Map(original.frames.filter(frame => frame.split === "train").map(frame => [frameKey(frame), frame]));
let checkedProjections = 0;
for (const camera of cameras) {
  const frame = originals.get(frameKey(camera.userData));
  assert.ok(frame, "Every GLB camera maps to a frozen original training frame");
  for (let row = 0; row < 4; row += 1) for (let column = 0; column < 4; column += 1) {
    assert.ok(Math.abs(camera.matrixWorld.elements[column * 4 + row] - frame.transform_matrix[row][column]) < 1e-8);
  }
  camera.updateProjectionMatrix();
  for (const position of data.points.filter((_, index) => index % 113 === 0)) {
    const world = new Vector3(...position);
    const local = world.clone().applyMatrix4(camera.matrixWorldInverse);
    if (local.z >= 0) continue;
    const expected = [original.fl_x * local.x / -local.z + original.cx, -original.fl_y * local.y / -local.z + original.cy];
    const normalized = world.clone().project(camera);
    const actual = [(normalized.x + 1) * original.w / 2, (1 - normalized.y) * original.h / 2];
    assert.ok(Math.abs(actual[0] - expected[0]) < 1e-6 && Math.abs(actual[1] - expected[1]) < 1e-6);
    checkedProjections += 1;
  }
}
assert.ok(checkedProjections > 0);
assert.equal(loaded.parser.json.asset.extras.registration, null);
assert.equal(loaded.parser.json.asset.extras.units, "arbitrary");
const {chromium} = await import(pathToFileURL(playwrightPath).href);
const browser = await chromium.launch({headless: true, args: ["--disable-gpu", "--disable-software-rasterizer", "--disable-dev-shm-usage"]});
const errors = [], external = [], reviewed = [];
try {
  const page = await browser.newPage({viewport: {width: 1500, height: 1050}, deviceScaleFactor: 1});
  page.on("pageerror", error => errors.push(error.message));
  page.on("request", request => {if (!/^(file|data):/.test(request.url())) external.push(request.url());});
  await page.goto(pathToFileURL(path.resolve(directory, "index.html")).href, {waitUntil: "load"});
  const photoReady = async () => page.waitForFunction(() => {
    const photo = document.getElementById("photo");
    return photo.complete && photo.naturalWidth > 0;
  });
  await photoReady();
  assert.equal(Number(await page.locator("#orbit").getAttribute("data-drawn-cameras")), cameras.length);
  assert.ok(Number(await page.locator("#orbit").getAttribute("data-drawn-points")) > 0);
  await page.screenshot({path: path.join(proofDirectory, "overview.png"), fullPage: true});
  const initialPixels = await page.locator("#orbit").evaluate(canvas => canvas.toDataURL());
  await page.locator("#orbit").focus();
  await page.keyboard.press("ArrowRight");
  assert.notEqual(await page.locator("#orbit").evaluate(canvas => canvas.toDataURL()), initialPixels);
  await page.locator("#reset").click();
  assert.equal(await page.locator("#orbit").evaluate(canvas => canvas.toDataURL()), initialPixels);
  await page.locator("#show-cameras").uncheck();
  assert.equal(await page.locator("#orbit").getAttribute("data-drawn-cameras"), "0");
  await page.screenshot({path: path.join(proofDirectory, "sparse-points-only.png"), fullPage: true});
  await page.locator("#show-cameras").check();
  for (const [index, frame] of data.cameras.frames.entries()) {
    await page.locator("#source").selectOption(String(index));
    await photoReady();
    assert.equal(await page.locator("#photo").getAttribute("src"), frame.file_path);
    assert.equal(await page.locator("#orbit").getAttribute("data-selected-frame"), frame.file_path);
    assert.equal(await page.locator("#photo").evaluate(image => image.naturalWidth), data.cameras.w);
    const count = Number(await page.locator("#projected").getAttribute("data-point-count"));
    reviewed.push({file: frame.file_path, sha256: frame.sha256, projected_points: count});
    if ([17, 35, 53, 71].includes(index)) await page.screenshot({path: path.join(proofDirectory, `source-${index}.png`), fullPage: true});
  }
  assert.ok(reviewed.some(frame => frame.projected_points > 20));
  await page.locator("#overlay").uncheck();
  assert.equal(await page.locator("#projected").evaluate(canvas => {
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    return pixels.some((value, index) => index % 4 === 3 && value > 0);
  }), false);
  await page.locator("#overlay").check();
  const downloadHref = await page.getByRole("link", {name: "Download points + cameras GLB"}).getAttribute("href");
  assert.equal(downloadHref, "reference.glb");
  assert.ok(fs.statSync(path.join(directory, downloadHref)).size > 0);
  await page.setViewportSize({width: 390, height: 844});
  await page.screenshot({path: path.join(proofDirectory, "phone.png"), fullPage: true});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  verifyHashes();
  const proof = {status: "passed", source_points: data.points.length, gltf_cameras: cameras.length,
    checked_gltf_projections: checkedProjections, source_views: reviewed, errors, external_requests: external,
    browser_mode: "On-demand CPU Canvas2D; GPU and software 3D rasterizer disabled",
    source_hashes_unchanged: true, independent_gltf_loader: "installed Three.js GLTFLoader",
    blender_import_executed: false, new_reconstruction: false};
  fs.writeFileSync(path.join(proofDirectory, "proof.json"), JSON.stringify(proof, null, 2));
  console.log(JSON.stringify({status: proof.status, points: proof.source_points, cameras: cameras.length, reviewed_views: reviewed.length, checked_gltf_projections: checkedProjections}));
} finally {
  await browser.close();
}
