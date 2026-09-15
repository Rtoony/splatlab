export function beginRenderedArchitecturalLane({ lane, radius, height, reverse = false }) {
  const walker = window.__sceneStudioWalker;
  if (!walker?.bvh || !walker.collider || walker.isFlying || !walker.controls.isLocked
      || document.pointerLockElement !== walker.renderer.domElement || window.__architecturalRenderProbe)
    throw new Error("Rendered walking requires admitted collision, actual pointer capture and no existing probe");
  if (walker.params.bodySizing !== "fixed-metric" || walker.params.unitsPerMetre !== 1
      || Math.abs(walker.capsuleRadius - radius) > 1e-9 || Math.abs(walker.capsuleHeight + radius - height) > 1e-9)
    throw new Error("Rendered walking must retain the requested physical body");
  const path = reverse ? [...lane].reverse() : lane;
  const origin = path[0], target = path.at(-1);
  const distance = Math.hypot(target[0] - origin[0], target[2] - origin[2]);
  if (!Number.isFinite(distance) || distance < .5 || distance > 15) throw new Error("Unbounded rendered route");
  walker.stop();
  walker.camera.lookAt(target[0], walker.camera.position.y, target[2]);
  walker.keys.clear();
  walker.velocity.set(0, 0, 0);
  const probe = { samples: [], rendered_frames: [], snapshots: [], keyboard_events: [], done: false, error: null,
    originalStep: walker.stepPlayer, originalRender: walker.renderer.render, previousParams: { ...walker.params },
    previousAutoUpdate: walker.spark?.autoUpdate, simulatedSeconds: 0, started: performance.now() };
  const sample = () => ({ time_ms: performance.now(), simulated_seconds: probe.simulatedSeconds,
    position: walker.camera.position.toArray(), grounded: walker.grounded, radius_m: walker.capsuleRadius,
    capsule_height_m: walker.capsuleHeight, flying: walker.isFlying,
    pointer_locked: walker.controls.isLocked && document.pointerLockElement === walker.renderer.domElement,
    forward_key: walker.keys.has("KeyW") });
  const captureControlledFrame = () => {
    if (walker.scene.visible !== true) throw new Error("Walking scene is already hidden");
    const png = walker.renderer.domElement.toDataURL("image/png");
    const state = () => JSON.stringify({ position: walker.camera.position.toArray(),
      quaternion: walker.camera.quaternion.toArray(), projection: walker.camera.projectionMatrix.toArray(),
      radius: walker.capsuleRadius, height: walker.capsuleHeight, params: walker.params });
    const before = state(), collider = walker.bvh, autoUpdate = walker.spark?.autoUpdate;
    let hidden, restored;
    try {
      if (walker.spark) walker.spark.autoUpdate = false;
      walker.scene.visible = false;
      probe.originalRender.call(walker.renderer, walker.scene, walker.camera);
      hidden = walker.renderer.domElement.toDataURL("image/png");
    } finally {
      walker.scene.visible = true;
      try {
        probe.originalRender.call(walker.renderer, walker.scene, walker.camera);
        restored = walker.renderer.domElement.toDataURL("image/png");
      } finally {
        if (walker.spark) walker.spark.autoUpdate = autoUpdate;
      }
    }
    if (state() !== before || walker.bvh !== collider)
      throw new Error("Walking render control changed camera, body or collision");
    if (png !== restored) probe.error = "Walking render control did not restore exact pixels";
    if (png === hidden) probe.error = "Walking image has no rendered scene contribution";
    return { png, render_control: { method: "scene-visibility-ab/v1", same_camera_body_and_collision: true,
      scene_visible_after: walker.scene.visible, hidden: { png: hidden }, restored: { png: restored } } };
  };
  probe.keyListener = event => {
    if (event.code === "KeyW") probe.keyboard_events.push({ type: event.type, code: event.code, trusted: event.isTrusted, time_ms: performance.now() });
  };
  document.addEventListener("keydown", probe.keyListener);
  document.addEventListener("keyup", probe.keyListener);
  walker.stepPlayer = function(delta) {
    const recording = walker.keys.has("KeyW") && !probe.done;
    if (recording && !probe.samples.length) probe.samples.push({ ...sample(), step_seconds: 0 });
    probe.originalStep.call(walker, delta);
    if (recording) {
      probe.simulatedSeconds += delta;
      probe.samples.push({ ...sample(), step_seconds: delta });
    }
  };
  walker.renderer.render = function(...args) {
    if (args[0] !== walker.scene || !walker.keys.has("KeyW") || probe.done)
      return probe.originalRender.apply(walker.renderer, args);
    const fraction = ((walker.camera.position.x - origin[0]) * (target[0] - origin[0])
      + (walker.camera.position.z - origin[2]) * (target[2] - origin[2])) / distance ** 2;
    const captureFrame = probe.snapshots.length < 3 && fraction >= [0, .5, .99][probe.snapshots.length];
    const autoUpdate = walker.spark?.autoUpdate;
    if (captureFrame && walker.spark) walker.spark.autoUpdate = false;
    try {
      probe.originalRender.apply(walker.renderer, args);
      const current = sample();
      probe.rendered_frames.push({ ...current, sample_index: probe.samples.length - 1, fraction });
      if (captureFrame) {
        try {
          probe.snapshots.push({ ...current, fraction, frame_index: probe.rendered_frames.length - 1,
            ...captureControlledFrame() });
        } catch (error) {
          probe.error = error.message;
        }
      }
      const graphics = walker.renderer.getContext();
      if (walker.isFlying || !current.pointer_locked || graphics.isContextLost() || graphics.getError() !== graphics.NO_ERROR)
        probe.error = "Rendered walking lost collision, pointer capture or its graphics context";
      if (probe.samples.length > 30000 || probe.rendered_frames.length > 6000 || performance.now() - probe.started > 60000)
        probe.error = "Rendered walking exceeded its bounded time or sample budget";
      if (fraction >= 1 - .005 / distance || probe.error) {
        probe.done = true;
        walker.keys.clear();
        walker.stop();
      }
    } finally {
      if (captureFrame && walker.spark) walker.spark.autoUpdate = autoUpdate;
    }
  };
  walker.setParams({ walkSpeedMps: .5 });
  if (walker.spark) walker.spark.autoUpdate = true;
  window.__architecturalRenderProbe = probe;
  walker.start();
}

export function finishRenderedArchitecturalLane() {
  const walker = window.__sceneStudioWalker, probe = window.__architecturalRenderProbe;
  if (!walker || !probe) throw new Error("No rendered architectural trace is active");
  walker.stop();
  walker.keys.clear();
  walker.stepPlayer = probe.originalStep;
  walker.renderer.render = probe.originalRender;
  document.removeEventListener("keydown", probe.keyListener);
  document.removeEventListener("keyup", probe.keyListener);
  if (walker.spark) walker.spark.autoUpdate = probe.previousAutoUpdate;
  const trace = { physics_method: "WorldWalker.stepPlayer", rendering_during_substeps: true, fixed_step_simulation: false,
    input_method: "trusted-keyboard-and-requestAnimationFrame", flying: walker.isFlying, pointer_locked: walker.controls.isLocked,
    units_per_metre: walker.params.unitsPerMetre, radius_m: walker.capsuleRadius, capsule_height_m: walker.capsuleHeight,
    walk_speed_mps: walker.params.walkSpeedMps, completed: probe.done, error: probe.error,
    samples: probe.samples, rendered_frames: probe.rendered_frames, snapshots: probe.snapshots, keyboard_events: probe.keyboard_events };
  walker.setParams(probe.previousParams);
  walker.velocity.set(0, 0, 0);
  delete window.__architecturalRenderProbe;
  return trace;
}

export function traceArchitecturalLane({ lane, floor, radius, height, reverse = false }) {
  const walker = window.__sceneStudioWalker;
  if (!walker?.bvh || !walker.collider || !walker.controls.isLocked)
    throw new Error("Navigation needs the loaded paired collider and actual pointer lock");
  const path = reverse ? [...lane].reverse() : lane;
  const floors = reverse ? [...floor].reverse() : floor;
  if (path.length < 2 || path.length !== floors.length) throw new Error("Invalid retained navigation lane");
  const previous = { ...walker.params };
  const wasFlying = walker.isFlying;
  const oldKeys = [...walker.keys];
  const timeStep = 1 / 120;
  const speed = 0.5;
  const distance = Math.hypot(path.at(-1)[0] - path[0][0], path.at(-1)[2] - path[0][2]);
  const count = Math.ceil(distance / (speed * timeStep));
  if (!Number.isFinite(count) || count < 10 || count > 4900) throw new Error("Navigation lane exceeds its bounded trace budget");
  try {
    walker.stop();
    walker.setParams({ unitsPerMetre: 1, radiusM: radius, eyeHeightM: height - radius, walkSpeedMps: speed });
    if (Math.abs(walker.capsuleRadius - radius) > 1e-9 || Math.abs(walker.capsuleHeight + radius - height) > 1e-9)
      throw new Error("Automatic capsule shortening cannot establish the requested clearance");
    walker.keys.clear();
    walker.velocity.set(0, 0, 0);
    walker.camera.position.set(path[0][0], floors[0] + walker.capsuleHeight + 0.01, path[0][2]);
    walker.camera.up.set(0, 1, 0);
    walker.camera.lookAt(path.at(-1)[0], walker.camera.position.y, path.at(-1)[2]);
    walker.setFlying(false);
    if (walker.isFlying) throw new Error("The actual fixed-size walking start was refused by collision admission");
    for (let step = 0; step < 30; step++) walker.stepPlayer(timeStep);
    walker.keys.add("KeyW");
    const samples = [{ position: walker.camera.position.toArray(), grounded: walker.grounded, capsule_height_m: walker.capsuleHeight }];
    for (let step = 0; step < count; step++) {
      walker.stepPlayer(timeStep);
      samples.push({ position: walker.camera.position.toArray(), grounded: walker.grounded, capsule_height_m: walker.capsuleHeight });
    }
    return { physics_method: "WorldWalker.stepPlayer", fixed_step_simulation: true, rendering_during_substeps: false,
      pointer_locked: walker.controls.isLocked, flying: walker.isFlying, units_per_metre: walker.params.unitsPerMetre,
      radius_m: walker.capsuleRadius, capsule_height_m: walker.capsuleHeight,
      step_seconds: timeStep, walk_speed_mps: speed, samples };
  } finally {
    walker.keys.clear();
    for (const key of oldKeys) walker.keys.add(key);
    walker.setParams(previous);
    walker.setFlying(wasFlying);
    walker.velocity.set(0, 0, 0);
  }
}
