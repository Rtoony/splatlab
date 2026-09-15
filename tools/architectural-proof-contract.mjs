import assert from "node:assert/strict";

export const ARCHITECTURAL_GATES = ["base_watertight", "cut_watertight", "positive_bounded_removal",
  "portal_volume_clear", "outside_surface_preserved", "approach_floor_continuous", "room_floor_continuous",
  "capsule_route_clear", "protected_elements_unchanged", "replacement_region_contained"];
export const LEGACY_NAVIGATION = "parallel-center-ray/v1";
export const ENTRY_NAVIGATION = "shared-entry-footprint/v1";
export const ROOM_APPEARANCE = "cut-and-authored-room/v1";
export const LEGACY_APPEARANCE = "cut-only/v1";
export const LEGACY_GEOMETRY = "nominal-floor/v1";
export const FINISHED_GEOMETRY = "raised-floor-finish/v1";
export const JOINED_GEOMETRY = "joined-floor-envelope/v1";

const vector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const close = (first, second, tolerance = 1e-6) => Math.abs(first - second) <= tolerance;

export function validateArchitecturalRoomTriangles(study, renderedTriangles) {
  const expected = study.result?.metrics?.room_triangles;
  assert.ok(Number.isSafeInteger(expected) && expected > 0, "Missing sealed authored triangle count");
  assert.equal(renderedTriangles, expected, "Rendered room must match the sealed authored geometry count");
}

export function architecturalWorld(spec, local) {
  const angle = spec.yaw_degrees * Math.PI / 180;
  return [spec.origin[0] + Math.cos(angle) * local[0] + Math.sin(angle) * local[2],
    spec.origin[1] + local[1], spec.origin[2] - Math.sin(angle) * local[0] + Math.cos(angle) * local[2]];
}

export function architecturalLocal(spec, world) {
  const angle = spec.yaw_degrees * Math.PI / 180;
  const relative = world.map((value, axis) => value - spec.origin[axis]);
  return [Math.cos(angle) * relative[0] - Math.sin(angle) * relative[2], relative[1],
    Math.sin(angle) * relative[0] + Math.cos(angle) * relative[2]];
}

export function validateArchitecturalInputs(initial, study, navigation, expectedGeneration) {
  assert.equal(initial.active?.generation, expectedGeneration, "The private generation changed");
  assert.equal(initial.legacy_sources_changed, false, "Captured source identities changed");
  assert.equal(initial.state?.viewer?.units, "meters", "The fixture must use calibrated world metres");
  assert.ok(study && !study.stale, "A fresh built architectural study is required");
  assert.deepEqual(study.base, initial.active, "Architectural evidence addresses another active revision");
  assert.match(study.architecture_id, /^architecture_[a-f0-9]{24}$/);
  assert.equal(study.result?.architecture_id, study.architecture_id);
  assert.equal(study.result?.method, "authored-portal-and-connected-room/v1");
  assert.equal(study.result?.verdict, "PASS_ARCHITECTURAL_EDIT");
  assert.match(study.result?.sha256 || "", /^[a-f0-9]{64}$/);
  const geometry = study.geometry_method === undefined ? LEGACY_GEOMETRY : study.geometry_method;
  assert.ok([LEGACY_GEOMETRY, FINISHED_GEOMETRY, JOINED_GEOMETRY].includes(geometry), "Unknown architectural geometry method");
  assert.equal(study.result.geometry_method === undefined ? LEGACY_GEOMETRY : study.result.geometry_method, geometry,
    "Result geometry method must match preparation");
  if (geometry !== LEGACY_GEOMETRY)
    assert.equal(study.result.metrics?.floor_finish_m, .016, "Finished slabs require the retained 16 mm datum");
  else
    assert.ok(study.result.metrics?.floor_finish_m === undefined || study.result.metrics.floor_finish_m === 0,
      "Nominal geometry cannot claim a raised floor");
  const appearance = study.result.appearance_method ?? LEGACY_APPEARANCE;
  assert.ok([LEGACY_APPEARANCE, ROOM_APPEARANCE].includes(appearance), "Unknown architectural appearance method");
  const gates = [...ARCHITECTURAL_GATES, ...(appearance === ROOM_APPEARANCE ? ["authored_room_interior_clear"] : []),
    ...(geometry === JOINED_GEOMETRY ? ["authored_room_watertight"] : [])];
  assert.deepEqual(Object.keys(study.result.gates).sort(), gates.sort());
  assert.ok(gates.every(name => study.result.gates[name] === true), "All local geometry gates must pass");
  assert.equal(navigation.architecture_id, study.architecture_id);
  assert.equal(navigation.frame, "world-y-up-metres");
  assert.deepEqual(navigation.gates, study.result.gates);
  assert.equal(navigation.capsule_radius_m, 0.22);
  assert.equal(navigation.capsule_height_m, 1.7);
  const method = study.result.navigation_method ?? LEGACY_NAVIGATION;
  assert.ok([LEGACY_NAVIGATION, ENTRY_NAVIGATION].includes(method), "Unknown navigation method");
  assert.equal(navigation.navigation_method ?? LEGACY_NAVIGATION, method);
  assert.equal(navigation.v, method === ENTRY_NAVIGATION ? 2 : 1);
  const spec = study.spec;
  assert.ok(vector(spec.origin) && spec.origin.every(value => Math.abs(value) <= 100));
  for (const [name, minimum, maximum] of [["yaw_degrees", -180, 180], ["opening_width", 0.8, 3],
    ["opening_height", 1.9, 3.5], ["cut_depth", 0.1, 6], ["room_width", 1.5, 8], ["room_depth", 1.5, 8],
    ["room_height", 2.1, 5], ["wall_thickness", 0.08, 0.4]])
    assert.ok(Number.isFinite(spec[name]) && spec[name] >= minimum && spec[name] <= maximum, `Invalid ${name}`);
  assert.ok(spec.room_width >= spec.opening_width + 0.3 && spec.room_height >= spec.opening_height + 0.15);
  if (appearance === ROOM_APPEARANCE) {
    assert.equal(study.clip?.appearance_method, ROOM_APPEARANCE);
    assert.deepEqual(study.clip.room_lower, [-spec.room_width / 2 - spec.wall_thickness, .005, spec.cut_depth / 2 - spec.wall_thickness / 2]);
    assert.deepEqual(study.clip.room_upper, [spec.room_width / 2 + spec.wall_thickness, spec.room_height + spec.wall_thickness,
      spec.cut_depth / 2 + spec.room_depth + spec.wall_thickness / 2]);
  }
  const lateral = spec.opening_width / 2 - navigation.capsule_radius_m - 0.06;
  const spacing = method === ENTRY_NAVIGATION ? 0.02 / Math.hypot(1, lateral / 0.6) : 0.04;
  const count = Math.ceil((spec.cut_depth + spec.room_depth + 0.1) / spacing) + 1;
  assert.equal(navigation.route.length, 3 * count);
  assert.equal(navigation.floor_y.length, navigation.route.length);
  const centers = method === ENTRY_NAVIGATION ? navigation.center_floor_y : navigation.floor_y;
  assert.ok(Array.isArray(centers) && centers.length === navigation.route.length, "Missing retained center-floor rays");
  const start = -spec.cut_depth / 2 - 0.6;
  const end = spec.cut_depth / 2 + spec.room_depth - 0.5;
  for (const [index, point] of navigation.route.entries()) {
    assert.ok(vector(point), "Invalid retained route point");
    const local = architecturalLocal(spec, point);
    const along = start + (end - start) * (index % count) / (count - 1);
    const factor = method === ENTRY_NAVIGATION ? Math.max(0, Math.min(1, (along + spec.cut_depth / 2 + 0.3) / 0.6)) : 1;
    const expected = [[-lateral, 0, lateral][Math.floor(index / count)] * factor, 0, along];
    assert.ok(local.every((value, axis) => close(value, expected[axis])), "Retained route does not match the dimensioned frame");
    assert.ok(Number.isFinite(navigation.floor_y[index]) && Math.abs(navigation.floor_y[index] - spec.origin[1]) <= 0.08,
      "Retained floor is absent or outside the local tolerance");
    assert.ok(Number.isFinite(centers[index]) && Math.abs(centers[index] - spec.origin[1]) <= 0.08,
      "Retained center-floor support is absent or outside the local tolerance");
    assert.ok(navigation.floor_y[index] >= centers[index] - 1e-6 && navigation.floor_y[index] - centers[index] <= 0.08,
      "Footprint support exceeds the unchanged floor tolerance");
    if (index % count) for (const levels of [navigation.floor_y, centers])
      assert.ok(Math.abs(levels[index] - levels[index - 1]) <= 0.04, "Retained floor jumps between route samples");
  }
  return { spec, lane: navigation.route.slice(count, 2 * count), floor: navigation.floor_y.slice(count, 2 * count),
    radius: navigation.capsule_radius_m, height: navigation.capsule_height_m };
}

export function validateCapsuleTrace(trace, contract, reverse = false) {
  assert.equal(trace.step_seconds, 1 / 120);
  return validatePathSamples(trace, contract, reverse, 5000);
}

export function validateRenderedCapsuleTrace(trace, contract, reverse = false) {
  assert.equal(trace.rendering_during_substeps, true);
  assert.equal(trace.fixed_step_simulation, false);
  assert.equal(trace.input_method, "trusted-keyboard-and-requestAnimationFrame");
  assert.equal(trace.completed, true);
  assert.equal(trace.error, null);
  const report = validatePathSamples(trace, contract, reverse, 30000);
  const keyboard = trace.keyboard_events;
  assert.ok(Array.isArray(keyboard) && keyboard.length === 2);
  assert.deepEqual(keyboard.map(event => [event.type, event.code, event.trusted]), [["keydown", "KeyW", true], ["keyup", "KeyW", true]]);
  const frames = trace.rendered_frames;
  assert.ok(Array.isArray(frames) && frames.length >= 30 && frames.length <= 6000, "Missing actual rendered walking frames");
  let previous = null;
  for (const [index, sample] of trace.samples.entries()) {
    assert.equal(sample.forward_key, true);
    assert.equal(sample.flying, false);
    assert.equal(sample.pointer_locked, true);
    assert.equal(sample.radius_m, contract.radius);
    assert.ok(Number.isFinite(sample.step_seconds) && sample.step_seconds >= 0 && sample.step_seconds <= .020001);
    assert.ok(Number.isFinite(sample.time_ms) && sample.time_ms >= keyboard[0].time_ms && sample.time_ms <= keyboard[1].time_ms);
    if (index) assert.ok(close(sample.simulated_seconds - trace.samples[index - 1].simulated_seconds, sample.step_seconds));
  }
  for (const frame of frames) {
    assert.equal(frame.flying, false);
    assert.equal(frame.pointer_locked, true);
    assert.equal(frame.forward_key, true);
    assert.equal(frame.radius_m, contract.radius);
    assert.equal(frame.capsule_height_m, trace.capsule_height_m);
    assert.ok(Number.isInteger(frame.sample_index) && frame.sample_index > 0 && frame.sample_index < trace.samples.length);
    assert.deepEqual(frame.position, trace.samples[frame.sample_index].position, "Render and physics positions differ");
    assert.ok(Number.isFinite(frame.time_ms) && frame.time_ms >= trace.samples[frame.sample_index].time_ms && frame.time_ms <= keyboard[1].time_ms);
    if (previous) assert.ok(frame.time_ms > previous.time_ms && frame.sample_index > previous.sample_index);
    previous = frame;
  }
  assert.equal(frames.at(-1).sample_index, trace.samples.length - 1);
  assert.ok(frames[0].sample_index <= 5);
  const elapsed = (frames.at(-1).time_ms - frames[0].time_ms) / 1000;
  const distance = Math.hypot(report.end[0] - report.start[0], report.end[2] - report.start[2]);
  assert.ok(elapsed >= distance / .5 - .25 && elapsed <= 60, "Rendered walking did not take physical elapsed time");
  return { ...report, rendered_frames: frames.length, elapsed_seconds: elapsed };
}

function validatePathSamples(trace, contract, reverse, maximumSamples) {
  assert.equal(trace.flying, false, "A flying camera is not navigation evidence");
  assert.equal(trace.pointer_locked, true, "Real pointer-lock input must reach the walker");
  assert.equal(trace.physics_method, "WorldWalker.stepPlayer");
  assert.equal(trace.radius_m, contract.radius);
  assert.ok(close(trace.capsule_height_m + trace.radius_m, contract.height), "Automatic capsule shortening cannot pass adult clearance");
  assert.equal(trace.units_per_metre, 1);
  assert.equal(trace.walk_speed_mps, 0.5);
  assert.ok(Array.isArray(trace.samples) && trace.samples.length > 10 && trace.samples.length <= maximumSamples);
  const lane = reverse ? [...contract.lane].reverse() : contract.lane;
  const floors = reverse ? [...contract.floor].reverse() : contract.floor;
  const start = architecturalLocal(contract.spec, lane[0]);
  const end = architecturalLocal(contract.spec, lane.at(-1));
  const sign = Math.sign(end[2] - start[2]);
  let previous = null, grounded = 0;
  for (const sample of trace.samples) {
    assert.ok(vector(sample.position), "Invalid capsule sample");
    assert.equal(sample.capsule_height_m, trace.capsule_height_m, "Capsule height changed during traversal");
    const local = architecturalLocal(contract.spec, sample.position);
    const fraction = (local[2] - start[2]) / (end[2] - start[2]);
    assert.ok(fraction >= -0.01 && fraction <= 1.01 && Math.abs(local[0]) <= 0.08, "The capsule escaped its tested corridor");
    const floor = floors[Math.min(floors.length - 1, Math.max(0, Math.round(fraction * (floors.length - 1))))];
    assert.ok(Math.abs(sample.position[1] - trace.capsule_height_m - floor) <= 0.08, "Feet fell through or left the tested floor");
    if (previous) {
      assert.ok(Math.hypot(sample.position[0] - previous.position[0], sample.position[2] - previous.position[2]) <= 0.015,
        "Navigation teleported rather than following bounded capsule substeps");
      assert.ok(Math.abs(sample.position[1] - previous.position[1]) <= 0.04, "Navigation jumped between floor levels");
      assert.ok(sign * (local[2] - architecturalLocal(contract.spec, previous.position)[2]) >= -0.01, "Unexpected route reversal");
    }
    grounded += Number(sample.grounded === true);
    previous = sample;
  }
  const first = architecturalLocal(contract.spec, trace.samples[0].position);
  const last = architecturalLocal(contract.spec, trace.samples.at(-1).position);
  assert.ok(Math.abs(first[2] - start[2]) <= 0.02, "The traversal did not start at the approach");
  assert.ok(Math.abs(last[2] - end[2]) <= 0.025, "The capsule did not reach the connected room");
  assert.ok(grounded / trace.samples.length >= 0.95, "The capsule did not remain supported");
  return { samples: trace.samples.length, grounded_fraction: grounded / trace.samples.length,
    start: trace.samples[0].position, end: trace.samples.at(-1).position, reverse };
}
