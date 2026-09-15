import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {test} from "node:test";

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(new URL("reference-viewer/math.js", import.meta.url), "utf8"), context);
const math = context.ReferenceMath;
const plain = value => JSON.parse(JSON.stringify(value));
const camera = {fl_x: 50, fl_y: 50, cx: 32, cy: 32, w: 64, h: 64};
const matrix = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]];

test("source-camera projection uses inverse rigid transform and negative Z forward", () => {
  assert.deepEqual(plain(math.sourceProjection([2, 3, 0], matrix, camera)), [32, 32, 4]);
  assert.deepEqual(plain(math.sourceProjection([3, 4, 0], matrix, camera)), [44.5, 19.5, 4]);
  assert.equal(math.sourceProjection([2, 3, 5], matrix, camera), null);
  assert.equal(math.sourceProjection([200, 3, 0], matrix, camera), null);
});

test("camera rotation, translation and half-pixel rays round-trip", () => {
  const rotated = [[0, 0, 1, 2], [0, 1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]];
  for (const [horizontal, vertical] of [[.5, .5], [32.5, 32.5], [63.5, 63.5]]) {
    const depth = 4;
    const local = [(horizontal - camera.cx) * depth / camera.fl_x, -(vertical - camera.cy) * depth / camera.fl_y, -depth];
    const world = math.transform(rotated, local);
    const projected = math.sourceProjection(world, rotated, camera);
    assert.ok(Math.abs(projected[0] - horizontal) < 1e-10);
    assert.ok(Math.abs(projected[1] - vertical) < 1e-10);
    assert.equal(projected[2], depth);
  }
});

test("orbit camera is presentation-only and clips points behind the observer", () => {
  assert.deepEqual(plain(math.orbitProjection([0, 0, 0], [0, 0, 10], [0, 0, 0], [0, 1, 0], 800, 600)), [400, 300, 10]);
  assert.equal(math.orbitProjection([0, 0, 11], [0, 0, 10], [0, 0, 0], [0, 1, 0], 800, 600), null);
});
