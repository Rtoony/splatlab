import assert from "node:assert/strict";
import { test } from "node:test";
import { ARCHITECTURAL_GATES, ENTRY_NAVIGATION, LEGACY_NAVIGATION, ROOM_APPEARANCE, architecturalWorld, architecturalLocal, validateArchitecturalInputs,
  validateCapsuleTrace, validateRenderedCapsuleTrace, validateArchitecturalRoomTriangles, FINISHED_GEOMETRY, JOINED_GEOMETRY } from "./architectural-proof-contract.mjs";

test("rendered triangle count tracks sealed legacy and joined masters, not a constant", () => {
  for (const triangles of [144, 656]) {
    const study = { result: { metrics: { room_triangles: triangles } } };
    validateArchitecturalRoomTriangles(study, triangles);
    assert.throws(() => validateArchitecturalRoomTriangles(study, triangles + 2));
    assert.throws(() => validateArchitecturalRoomTriangles(study, undefined));
  }
  for (const triangles of [undefined, null, false, 0, -1, 1.5])
    assert.throws(() => validateArchitecturalRoomTriangles({ result: { metrics: { room_triangles: triangles } } }, triangles));
});

test("joined geometry proof requires the added solid-boundary gate", () => {
  const { initial, study, navigation } = fixture();
  study.geometry_method = JOINED_GEOMETRY;
  study.result.geometry_method = JOINED_GEOMETRY;
  study.result.metrics = { floor_finish_m: .016 };
  assert.throws(() => validateArchitecturalInputs(initial, study, navigation, 4));
  study.result.gates.authored_room_watertight = true;
  navigation.gates.authored_room_watertight = true;
  validateArchitecturalInputs(initial, study, navigation, 4);
});

test("finished floor proof binds preparation, result and explicit datum", () => {
  const { initial, study, navigation } = fixture();
  study.geometry_method = FINISHED_GEOMETRY;
  study.result.geometry_method = FINISHED_GEOMETRY;
  study.result.metrics = { floor_finish_m: .016 };
  validateArchitecturalInputs(initial, study, navigation, 4);
});

for (const defect of ["missing-result", "unknown", "null", "wrong-finish", "missing-finish", "legacy-finish"])
  test(`finished floor proof rejects ${defect}`, () => {
    const { initial, study, navigation } = fixture();
    study.geometry_method = FINISHED_GEOMETRY;
    study.result.geometry_method = FINISHED_GEOMETRY;
    study.result.metrics = { floor_finish_m: .016 };
    if (defect === "missing-result") delete study.result.geometry_method;
    if (defect === "unknown") study.geometry_method = "raised-floor-finish/v2";
    if (defect === "null") study.geometry_method = null;
    if (defect === "wrong-finish") study.result.metrics.floor_finish_m = .03;
    if (defect === "missing-finish") delete study.result.metrics.floor_finish_m;
    if (defect === "legacy-finish") { delete study.geometry_method; delete study.result.geometry_method; }
    assert.throws(() => validateArchitecturalInputs(initial, study, navigation, 4));
  });

function fixture(depth = 0.6, method = LEGACY_NAVIGATION) {
  const spec = { origin: [2, -1, 3], yaw_degrees: 35, opening_width: 1.1, opening_height: 2.1,
    cut_depth: depth, room_width: 3, room_depth: 3, room_height: 2.7, wall_thickness: 0.15 };
  const initial = { active: { revision_id: "scene", generation: 4, proposal_id: null }, legacy_sources_changed: false,
    state: { viewer: { units: "meters" } } };
  const identifier = "architecture_" + "a".repeat(24);
  const gates = Object.fromEntries(ARCHITECTURAL_GATES.map(name => [name, true]));
  const study = { architecture_id: identifier, base: { ...initial.active }, stale: false, spec,
    result: { method: "authored-portal-and-connected-room/v1", verdict: "PASS_ARCHITECTURAL_EDIT", architecture_id: identifier,
      sha256: "b".repeat(64), gates, navigation_method: method } };
  const lateral = spec.opening_width / 2 - 0.22 - 0.06;
  const spacing = method === ENTRY_NAVIGATION ? 0.02 / Math.hypot(1, lateral / 0.6) : 0.04;
  const count = Math.ceil((spec.cut_depth + spec.room_depth + 0.1) / spacing) + 1;
  const start = -spec.cut_depth / 2 - 0.6, end = spec.cut_depth / 2 + spec.room_depth - 0.5;
  const route = [-lateral, 0, lateral].flatMap(offset => Array.from({ length: count }, (_, index) => {
    const along = start + (end - start) * index / (count - 1);
    const factor = method === ENTRY_NAVIGATION ? Math.max(0, Math.min(1, (along + spec.cut_depth / 2 + 0.3) / 0.6)) : 1;
    return architecturalWorld(spec, [offset * factor, 0, along]);
  }));
  const navigation = { v: method === ENTRY_NAVIGATION ? 2 : 1, navigation_method: method,
    architecture_id: identifier, frame: "world-y-up-metres", gates: { ...gates }, route,
    floor_y: route.map(() => -1), center_floor_y: route.map(() => -1), capsule_radius_m: 0.22, capsule_height_m: 1.7 };
  return { initial, study, navigation };
}

function syntheticTrace(contract, reverse = false) {
  const first = architecturalLocal(contract.spec, reverse ? contract.lane.at(-1) : contract.lane[0]);
  const last = architecturalLocal(contract.spec, reverse ? contract.lane[0] : contract.lane.at(-1));
  const count = Math.ceil(Math.abs(last[2] - first[2]) / (0.5 / 120));
  const samples = Array.from({ length: count + 1 }, (_, index) => ({ grounded: true, capsule_height_m: 1.48,
    position: architecturalWorld(contract.spec, [0, 1.48, first[2] + (last[2] - first[2]) * index / count]) }));
  return { flying: false, pointer_locked: true, physics_method: "WorldWalker.stepPlayer", units_per_metre: 1,
    radius_m: 0.22, capsule_height_m: 1.48, step_seconds: 1 / 120, walk_speed_mps: 0.5, samples };
}

function syntheticRenderedTrace(contract, reverse = false) {
  const trace = syntheticTrace(contract, reverse);
  trace.samples = trace.samples.map((sample, index) => ({ ...sample, radius_m: contract.radius,
    time_ms: index * 1000 / 120, simulated_seconds: index / 120, step_seconds: index ? 1 / 120 : 0,
    flying: false, pointer_locked: true, forward_key: true }));
  trace.rendered_frames = trace.samples.flatMap((sample, index) => index && (index % 5 === 0 || index === trace.samples.length - 1)
    ? [{ ...sample, time_ms: sample.time_ms + .01, sample_index: index }] : []);
  trace.keyboard_events = [{ type: "keydown", code: "KeyW", trusted: true, time_ms: 0 },
    { type: "keyup", code: "KeyW", trusted: true, time_ms: trace.samples.at(-1).time_ms + 1 }];
  return { ...trace, rendering_during_substeps: true, fixed_step_simulation: false, completed: true, error: null,
    input_method: "trusted-keyboard-and-requestAnimationFrame" };
}

test("synthetic rendered evidence validates both directions without claiming an actual browser run", () => {
  const { initial, study, navigation } = fixture(4, ENTRY_NAVIGATION);
  const contract = validateArchitecturalInputs(initial, study, navigation, 4);
  for (const reverse of [false, true]) {
    const report = validateRenderedCapsuleTrace(syntheticRenderedTrace(contract, reverse), contract, reverse);
    assert.ok(report.rendered_frames > 300);
    assert.ok(report.elapsed_seconds > 14);
  }
});

for (const defect of [null, "missing-gate", "intrusion", "envelope", "unknown"]) {
  test(`paired room appearance binds its envelope and extra clear-interior gate: ${defect || "valid"}`, () => {
    const { initial, study, navigation } = fixture(4, ENTRY_NAVIGATION);
    study.result.appearance_method = ROOM_APPEARANCE;
    study.result.gates.authored_room_interior_clear = true;
    navigation.gates.authored_room_interior_clear = true;
    study.clip = { appearance_method: ROOM_APPEARANCE, room_lower: [-1.65, .005, 1.925], room_upper: [1.65, 2.85, 5.075] };
    if (defect === "missing-gate") delete study.result.gates.authored_room_interior_clear;
    if (defect === "intrusion") study.result.gates.authored_room_interior_clear = false;
    if (defect === "envelope") study.clip.room_upper[0] += .1;
    if (defect === "unknown") study.result.appearance_method = "unknown";
    if (defect) assert.throws(() => validateArchitecturalInputs(initial, study, navigation, 4));
    else validateArchitecturalInputs(initial, study, navigation, 4);
  });
}

for (const defect of ["simulation", "no-render", "untrusted-key", "no-key-up", "flying-frame", "unlocked-step",
  "resized-step", "large-step", "render-position", "render-order", "incomplete-render", "late-start", "instantaneous", "error"]) {
  test(`rendered evidence refuses ${defect}`, () => {
    const { initial, study, navigation } = fixture(4, ENTRY_NAVIGATION);
    const contract = validateArchitecturalInputs(initial, study, navigation, 4);
    const trace = syntheticRenderedTrace(contract);
    if (defect === "simulation") trace.fixed_step_simulation = true;
    if (defect === "no-render") trace.rendered_frames = [];
    if (defect === "untrusted-key") trace.keyboard_events[0].trusted = false;
    if (defect === "no-key-up") trace.keyboard_events.pop();
    if (defect === "flying-frame") trace.rendered_frames[20].flying = true;
    if (defect === "unlocked-step") trace.samples[20].pointer_locked = false;
    if (defect === "resized-step") trace.samples[20].radius_m = .1;
    if (defect === "large-step") trace.samples[20].step_seconds = .1;
    if (defect === "render-position") trace.rendered_frames[20].position = [0, 0, 0];
    if (defect === "render-order") trace.rendered_frames.reverse();
    if (defect === "incomplete-render") trace.rendered_frames.pop();
    if (defect === "late-start") trace.rendered_frames.shift();
    if (defect === "instantaneous") {
      trace.samples.forEach(sample => { sample.time_ms /= 100; });
      trace.rendered_frames.forEach(frame => { frame.time_ms /= 100; });
    }
    if (defect === "error") trace.error = "Lost graphics context";
    assert.throws(() => validateRenderedCapsuleTrace(trace, contract));
  });
}

test("synthetic receipt fixture establishes a dimensioned three-lane contract, not actual geometry proof", () => {
  const { initial, study, navigation } = fixture();
  const contract = validateArchitecturalInputs(initial, study, navigation, 4);
  assert.equal(contract.lane.length * 3, navigation.route.length);
  assert.equal(contract.floor[0], -1);
  for (const reverse of [false, true]) {
    const report = validateCapsuleTrace(syntheticTrace(contract, reverse), contract, reverse);
    assert.equal(report.grounded_fraction, 1);
  }
});

for (const depth of [0.1, 4, 6]) for (const method of [LEGACY_NAVIGATION, ENTRY_NAVIGATION]) {
  test(`a ${depth} m vestibule retains the ${method} three-lane and two-direction physical contract`, () => {
    const { initial, study, navigation } = fixture(depth, method);
    const contract = validateArchitecturalInputs(initial, study, navigation, 4);
    assert.equal(contract.lane.length * 3, navigation.route.length);
    assert.equal(contract.radius, 0.22);
    assert.equal(contract.height, 1.7);
    for (const reverse of [false, true]) validateCapsuleTrace(syntheticTrace(contract, reverse), contract, reverse);
  });
}

for (const defect of ["method", "version", "parallel-route", "floor-step", "center-ray", "support-lift"]) {
  test(`shared entry refuses a ${defect} contradiction rather than reinterpreting the old recipe`, () => {
    const { initial, study, navigation } = fixture(4, ENTRY_NAVIGATION);
    if (defect === "method") study.result.navigation_method = "unknown";
    if (defect === "version") navigation.v = 1;
    if (defect === "parallel-route") navigation.route[0] = architecturalWorld(study.spec, [-0.27, 0, -2.6]);
    if (defect === "floor-step") navigation.floor_y[0] = -0.95;
    if (defect === "center-ray") navigation.center_floor_y[0] = null;
    if (defect === "support-lift") {
      navigation.center_floor_y.fill(-1.07);
      navigation.floor_y.fill(-0.98);
    }
    assert.throws(() => validateArchitecturalInputs(initial, study, navigation, 4));
  });
}

for (const depth of [0.099, 6.001]) {
  test(`refuses an out-of-bounds ${depth} m vestibule`, () => {
    const { initial, study, navigation } = fixture(depth);
    assert.throws(() => validateArchitecturalInputs(initial, study, navigation, 4), /Invalid cut_depth/);
  });
}

for (const defect of ["generation", "sources", "units", "stale", "base", "verdict", "gate", "missing-gate", "radius",
  "height", "route", "floor", "frame", "yaw", "dimensions", "rows"]) {
  test(`refuses ${defect} mismatch before browser mutation`, () => {
    const { initial, study, navigation } = fixture();
    if (defect === "generation") initial.active.generation++;
    if (defect === "sources") initial.legacy_sources_changed = true;
    if (defect === "units") initial.state.viewer.units = "scene-units";
    if (defect === "stale") study.stale = true;
    if (defect === "base") study.base.revision_id = "other";
    if (defect === "verdict") study.result.verdict = "FAIL_ARCHITECTURAL_EDIT";
    if (defect === "gate") study.result.gates.portal_volume_clear = 1;
    if (defect === "missing-gate") delete study.result.gates.protected_elements_unchanged;
    if (defect === "radius") navigation.capsule_radius_m = 0.1;
    if (defect === "height") navigation.capsule_height_m = 0.9;
    if (defect === "route") navigation.route[0][0] += 0.1;
    if (defect === "floor") navigation.floor_y[0] = null;
    if (defect === "frame") navigation.frame = "camera";
    if (defect === "yaw") study.spec.yaw_degrees = 360;
    if (defect === "dimensions") study.spec.opening_width = true;
    if (defect === "rows") navigation.route.pop();
    assert.throws(() => validateArchitecturalInputs(initial, study, navigation, 4));
  });
}

for (const defect of ["flying", "unlocked", "shortened", "scaled", "teleport", "unsupported", "feet", "incomplete", "sideways", "height-change", "method"]) {
  test(`rejects ${defect} navigation evidence`, () => {
    const { initial, study, navigation } = fixture();
    const contract = validateArchitecturalInputs(initial, study, navigation, 4);
    const trace = syntheticTrace(contract);
    if (defect === "flying") trace.flying = true;
    if (defect === "unlocked") trace.pointer_locked = false;
    if (defect === "shortened") trace.capsule_height_m = 0.9;
    if (defect === "scaled") trace.units_per_metre = 0.9;
    if (defect === "teleport") trace.samples.splice(10, 80);
    if (defect === "unsupported") trace.samples.forEach(sample => { sample.grounded = false; });
    if (defect === "feet") trace.samples[10].position[1] -= 0.15;
    if (defect === "incomplete") trace.samples.splice(-100);
    if (defect === "sideways") trace.samples[10].position[0] += 1;
    if (defect === "height-change") trace.samples[10].capsule_height_m = 1;
    if (defect === "method") trace.physics_method = "teleport camera";
    assert.throws(() => validateCapsuleTrace(trace, contract));
  });
}
