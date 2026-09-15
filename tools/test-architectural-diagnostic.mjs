import assert from "node:assert/strict";
import test from "node:test";
import { architecturalProofReportName, raisedBridgeFloor, architecturalAppearanceModes, legacyArchitecturalProjection } from "./architectural-diagnostic-contract.mjs";

test("projection comparison stays explicit and paired with its original restoration", () => {
  assert.deepEqual(architecturalAppearanceModes("cut-and-authored-room/v1", "joined-floor-envelope/v1", true),
    ["legacy-render-diagnostic", "projection-only-diagnostic", "original", "no-capture", "legacy-render-restored", "original-restored"]);
  assert.throws(() => architecturalAppearanceModes("cut-and-authored-room/v1", "raised-floor-finish/v1", true));
  const source = "before\n    vec2 architecturalNdc = ndc;\n    vec3 position = architecturalRay / (-architecturalRay.z * gl_FragCoord.w);\n    for (int portalIndex = 0; portalIndex < 4; portalIndex++) { unchanged; }";
  const legacy = legacyArchitecturalProjection(source);
  assert.ok(legacy.includes("architecturalView.xyz / architecturalView.w"));
  assert.ok(legacy.endsWith("for (int portalIndex = 0; portalIndex < 4; portalIndex++) { unchanged; }"));
  assert.throws(() => legacyArchitecturalProjection(legacy));
  assert.throws(() => legacyArchitecturalProjection(source + source));
});

test("appearance comparison never raises an already finished floor again", () => {
  const modes = architecturalAppearanceModes("cut-and-authored-room/v1", "raised-floor-finish/v1");
  assert.deepEqual(modes, ["original", "lit", "no-capture", "diagnostic-room-mask", "lit-room-mask"]);
  assert.equal(architecturalAppearanceModes("cut-and-authored-room/v1").length, 8);
  assert.equal(architecturalAppearanceModes().length, 5);
  assert.equal(architecturalAppearanceModes("cut-only/v1", "raised-floor-finish/v1").length, 5);
  assert.deepEqual(architecturalAppearanceModes("cut-and-authored-room/v1", "joined-floor-envelope/v1"), modes);
});

for (const method of [null, false, "future/v2"]) test(`unknown diagnostic method ${method} refuses`, () => {
  assert.throws(() => architecturalAppearanceModes(method), /appearance method/);
  assert.throws(() => architecturalAppearanceModes("cut-and-authored-room/v1", method), /geometry method/);
});

const spec = { origin: [-3.34, -1.62, 1.73], yaw_degrees: -80, opening_width: .8, cut_depth: 4, wall_thickness: .15 };

function bridge() {
  const vertices = [];
  const cosine = Math.cos(spec.yaw_degrees * Math.PI / 180), sine = Math.sin(spec.yaw_degrees * Math.PI / 180);
  for (const localX of [-.45, .45]) for (const localY of [-.15, 0]) for (const localZ of [-2.3, 2])
    vertices.push(cosine * localX + sine * localZ + spec.origin[0], localY + spec.origin[1],
      -sine * localX + cosine * localZ + spec.origin[2]);
  return new Float32Array(vertices);
}

test("bridge diagnostic raises only known corners and preserves the original input", () => {
  const vertices = [...bridge(), 20, 30, 40];
  const original = [...vertices];
  const changed = raisedBridgeFloor(vertices, spec, .016);
  assert.deepEqual(vertices, original);
  assert.equal(changed.indices.length, 8);
  assert.deepEqual(changed.positions.slice(-3), [20, 30, 40]);
  for (const index of changed.indices) {
    assert.equal(changed.positions[index * 3], original[index * 3]);
    assert.equal(changed.positions[index * 3 + 1], original[index * 3 + 1] + .016);
    assert.equal(changed.positions[index * 3 + 2], original[index * 3 + 2]);
  }
});

test("duplicated GLB corners remain supported without inventing more unique corners", () => {
  const changed = raisedBridgeFloor([...bridge(), ...bridge()], spec, .016);
  assert.equal(changed.indices.length, 16);
  assert.equal(changed.unique_corners, 8);
});

test("zero lift restores the exact retained positions", () => {
  assert.deepEqual(raisedBridgeFloor(bridge(), spec, 0).positions, [...bridge()]);
});

for (const lift of [-.001, .031, NaN, Infinity]) test(`invalid diagnostic lift ${lift} refuses`, () => {
  assert.throws(() => raisedBridgeFloor(bridge(), spec, lift), /bridge lift/);
});

test("missing or unrelated floor corners are not silently guessed", () => {
  assert.throws(() => raisedBridgeFloor(bridge().slice(3), spec, .016), /all eight/);
  assert.throws(() => raisedBridgeFloor(bridge(), { ...spec, opening_width: 1.1 }, .016), /all eight/);
});

test("preview-only evidence stays explicitly distinct from full transaction evidence", () => {
  assert.equal(architecturalProofReportName(["architectural-preview-proof.json", "image.png"]), "architectural-preview-proof.json");
  assert.equal(architecturalProofReportName(["architectural-browser-proof.json"]), "architectural-browser-proof.json");
  assert.throws(() => architecturalProofReportName([]), /exactly one/);
  assert.throws(() => architecturalProofReportName(["architectural-browser-proof.json", "architectural-preview-proof.json"]), /exactly one/);
});
