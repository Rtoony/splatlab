import assert from "node:assert/strict";
import test from "node:test";
import {captureSummary} from "./real-to-model-viewer/capture-summary.js";

for (const count of [4, 12]) {
  test(`Counts ${count} timestamps rather than six overlapping crops each`, () => {
    const validation = Array.from({length: count}, (_, index) => [0, 1].flatMap(lens =>
      ["left", "centre", "right"].map(direction => ({image: `val/lens-${lens}-frame-${String(index * 30).padStart(6, "0")}-${direction}.png`})))).flat();
    const result = captureSummary({validation});
    assert.equal(result.viewCount, count * 6);
    assert.equal(result.timestampCount, count);
    assert.match(result.label, /crops overlap/);
  });
}

test("Unknown image names do not invent timestamp independence", () => {
  assert.equal(captureSummary({validation: [{image: "photo.jpg"}]}).timestampCount, null);
});

test("Empty and duplicated view identities refuse", () => {
  assert.throws(() => captureSummary({validation: []}), /nonempty/);
  assert.throws(() => captureSummary({validation: [{image: "photo.jpg"}, {image: "photo.jpg"}]}), /unique/);
});
