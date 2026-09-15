import assert from "node:assert/strict";
import { test } from "node:test";
import { beginRenderedArchitecturalLane, finishRenderedArchitecturalLane } from "./architectural-navigation-proof.mjs";

function controlledSnapshot(defect = null) {
  const previousWindow = globalThis.window, previousDocument = globalThis.document;
  const position = { values: [0, 1.48, 0], toArray() { return [...this.values]; },
    get x() { return this.values[0]; }, get y() { return this.values[1]; }, get z() { return this.values[2]; } };
  let hiddenRendered = false, renders = 0, accumulator = 0;
  const updateStates = [];
  const walker = {
    bvh: {}, collider: {}, isFlying: false, controls: { isLocked: true },
    params: { bodySizing: "fixed-metric", unitsPerMetre: 1, walkSpeedMps: 1 }, capsuleRadius: .22, capsuleHeight: 1.48,
    camera: { position, lookAt() {}, quaternion: { toArray: () => [0, 0, 0, 1] }, projectionMatrix: { toArray: () => [1, 0, 0, 1] } },
    scene: { visible: true }, spark: { autoUpdate: true }, keys: new Set(), velocity: { set() {} },
    stop() {}, start() {}, setParams(params) { Object.assign(this.params, params); },
    stepPlayer() { position.values[2] += .1; },
  };
  walker.renderer = {
    domElement: { toDataURL() {
      const visible = defect === "no-contribution" || walker.scene.visible;
      return `data:image/png;base64,${visible ? defect === "bad-restore" && hiddenRendered ? "changed" : "scene" : "empty"}-${accumulator}`;
    } },
    getContext: () => ({ NO_ERROR: 0, isContextLost: () => false, getError: () => 0 }),
    render() {
      renders++;
      updateStates.push(walker.spark.autoUpdate);
      if (walker.spark.autoUpdate) accumulator++;
      if (walker.scene.visible && hiddenRendered && defect === "restore-error") throw new Error("Synthetic restoration render failure");
      if (!walker.scene.visible) {
        hiddenRendered = true;
        if (defect === "render-error") throw new Error("Synthetic control render failure");
        if (defect === "moved-camera") position.values[0]++;
        if (defect === "changed-body") walker.capsuleRadius = .1;
        if (defect === "changed-collider") walker.bvh = {};
      }
    },
  };
  const originalRender = walker.renderer.render, originalStep = walker.stepPlayer;
  globalThis.window = { __sceneStudioWalker: walker };
  globalThis.document = { pointerLockElement: walker.renderer.domElement, addEventListener() {}, removeEventListener() {} };
  try {
    beginRenderedArchitecturalLane({ lane: [[0, 0, 0], [0, 0, 2]], radius: .22, height: 1.7 });
    walker.keys.add("KeyW");
    walker.stepPlayer(.01);
    walker.renderer.render(walker.scene, walker.camera);
    assert.equal(walker.scene.visible, true);
    assert.equal(walker.spark.autoUpdate, true);
    const trace = finishRenderedArchitecturalLane();
    assert.equal(walker.renderer.render, originalRender);
    assert.equal(walker.stepPlayer, originalStep);
    assert.equal(walker.params.walkSpeedMps, 1);
    return { trace, renders, updateStates };
  } finally {
    globalThis.window = previousWindow;
    globalThis.document = previousDocument;
  }
}

test("visibility A/B captures the same walking frame and exact restoration without adding physics samples", () => {
  const { trace, renders, updateStates } = controlledSnapshot();
  assert.equal(renders, 3);
  assert.equal(trace.samples.length, 2);
  assert.equal(trace.rendered_frames.length, 1);
  assert.equal(trace.error, null);
  assert.equal(trace.snapshots.length, 1);
  assert.deepEqual(updateStates, [false, false, false]);
  const snapshot = trace.snapshots[0];
  assert.notEqual(snapshot.png, snapshot.render_control.hidden.png);
  assert.equal(snapshot.png, snapshot.render_control.restored.png);
  assert.deepEqual(snapshot.position, trace.rendered_frames[0].position);
});

for (const defect of ["render-error", "restore-error", "no-contribution", "bad-restore", "moved-camera", "changed-body", "changed-collider"])
  test(`visibility control restores scene/renderer and refuses ${defect}`, () => {
    const { trace } = controlledSnapshot(defect);
    assert.equal(trace.completed, true);
    assert.equal(trace.snapshots.length, ["no-contribution", "bad-restore"].includes(defect) ? 1 : 0);
    assert.equal(typeof trace.error, "string");
    assert.ok(trace.error.length > 0);
  });
