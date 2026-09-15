import assert from "node:assert/strict";
import { test } from "node:test";
import "./capture-atlas-viewer/math.js";

const { projectPixel, mapFrame } = globalThis.CaptureAtlasMath;

test("centre ray, cardinal turns and pitch sample the expected panorama", () => {
  const centre = projectPixel(0, 0, 1, 1, 0, 0, 65);
  assert.equal(centre.u, .5);
  assert.equal(centre.v, .5);
  assert.equal(projectPixel(0, 0, 1, 1, Math.PI/2, 0, 65).u, .75);
  assert.equal(projectPixel(0, 0, 1, 1, Math.PI, 0, 65).u, 0);
  assert.ok(projectPixel(0, 0, 1, 1, 0, Math.PI/4, 65).v < .5);
});

test("panorama samples stay bounded through seam and pole views", () => {
  for (const yaw of [-4, -Math.PI, 0, Math.PI, 7]) {
    for (const pitch of [-Math.PI/2, 0, Math.PI/2]) {
      for (const horizontal of [0, 319, 639]) {
        const sample = projectPixel(horizontal, 0, 640, 360, yaw, pitch, 100);
        assert.ok(Number.isFinite(sample.u) && sample.u >= 0 && sample.u < 1);
        assert.ok(Number.isFinite(sample.v) && sample.v >= 0 && sample.v <= 1);
      }
    }
  }
});

test("roll corrects inspection orientation without mutating source geometry", () => {
  const normal = projectPixel(0, 0, 3, 3, 0, 0, 65);
  const flipped = projectPixel(0, 0, 3, 3, 0, 0, 65, Math.PI);
  assert.ok(normal.u < .5 && normal.v < .5);
  assert.ok(flipped.u > .5 && flipped.v > .5);
});

test("map uses equal metre scale and north up, including single fixes", () => {
  const frame = mapFrame([{east_m:0,north_m:0},{east_m:20,north_m:20}]);
  const first = frame.project({east_m:0,north_m:0});
  const second = frame.project({east_m:20,north_m:20});
  assert.equal(second.horizontal-first.horizontal, first.vertical-second.vertical);
  assert.ok(second.vertical < first.vertical);
  assert.equal(mapFrame([]), null);
  assert.deepEqual(mapFrame([{east_m:10,north_m:20}]).project({east_m:10,north_m:20}), {horizontal:300,vertical:230});
});
