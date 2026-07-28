// Physics module tests. The -compat Rapier build embeds its WASM, so the
// REAL engine runs under node — these are genuine simulation tests, not
// stub theatre: a box falls onto a ground trimesh, settles, and two
// identical runs agree bit-for-bit (Rapier's determinism contract, which
// the fixed-step accumulator in step() exists to preserve).

import { describe, expect, it } from "vitest";
import * as THREE from "three";

import {
  FIXED_DT,
  WorldPhysics,
  loadRapier,
  reoriginObject,
  validTransform,
  type PropTransforms,
} from "./world-physics";

function groundGeometry(size = 20): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(size, size, 1, 1);
  g.rotateX(-Math.PI / 2); // plane faces +Y: a floor
  return g;
}

function cubePoints(half = 0.5): Float32Array {
  const pts: number[] = [];
  for (const x of [-half, half])
    for (const y of [-half, half])
      for (const z of [-half, half]) pts.push(x, y, z);
  return new Float32Array(pts);
}

function propObject(centre: THREE.Vector3, half = 0.5): THREE.Object3D {
  // Mimic the exporter contract: geometry baked in world space, identity
  // node transform.
  const geometry = new THREE.BoxGeometry(half * 2, half * 2, half * 2);
  geometry.translate(centre.x, centre.y, centre.z);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const root = new THREE.Object3D();
  root.add(mesh);
  return root;
}

const defaultCamera = new THREE.PerspectiveCamera();

function settle(
  physics: WorldPhysics,
  eye: THREE.Vector3,
  seconds: number,
  camera: THREE.Camera = defaultCamera,
): void {
  const frames = Math.round(seconds / FIXED_DT);
  for (let i = 0; i < frames; i++) physics.step(FIXED_DT, eye, 1.7, camera);
}

describe("validTransform", () => {
  it("accepts finite unit-quaternion poses and rejects everything else", () => {
    expect(validTransform({ position: [1, 2, 3], quaternion: [0, 0, 0, 1] })).toBe(true);
    expect(validTransform({ position: [1, 2], quaternion: [0, 0, 0, 1] })).toBe(false);
    expect(validTransform({ position: [1, 2, NaN], quaternion: [0, 0, 0, 1] })).toBe(false);
    expect(validTransform({ position: [1, 2, 3], quaternion: [0, 0, 0, 2] })).toBe(false);
    expect(validTransform(null)).toBe(false);
    expect(validTransform("pose")).toBe(false);
  });
});

describe("reoriginObject", () => {
  it("moves the origin to the AABB centre without moving the world-space shape", () => {
    const centre = new THREE.Vector3(3, 1, -2);
    const object = propObject(centre);
    const before = new THREE.Box3().setFromObject(object);

    const reported = reoriginObject(object);
    expect(reported.distanceTo(centre)).toBeLessThan(1e-6);
    expect(object.position.distanceTo(centre)).toBeLessThan(1e-6);

    const after = new THREE.Box3().setFromObject(object);
    expect(after.min.distanceTo(before.min)).toBeLessThan(1e-6);
    expect(after.max.distanceTo(before.max)).toBeLessThan(1e-6);

    // Idempotent: a second call must not shift anything.
    reoriginObject(object);
    const again = new THREE.Box3().setFromObject(object);
    expect(again.min.distanceTo(before.min)).toBeLessThan(1e-6);
  });
});

describe("WorldPhysics (real Rapier)", () => {
  it("drops a prop onto the shell, reports it disturbed, and round-trips poses", async () => {
    const rapier = await loadRapier();
    const physics = new WorldPhysics(rapier, 1);
    physics.addStaticShell(groundGeometry());

    const start = new THREE.Vector3(0, 3, 0);
    const object = propObject(start);
    reoriginObject(object);
    const prop = physics.addProp("crate", object, [cubePoints()]);
    expect(prop).not.toBeNull();
    prop!.body.wakeUp(); // sleeping-by-default is the perf posture; wake to drop

    const eye = new THREE.Vector3(50, 2, 50); // far away: player must not interfere
    physics.installPlayer(0.32, 1.7);
    settle(physics, eye, 3);

    // Settled on the floor: cube half-height above y=0, not still at y=3.
    expect(object.position.y).toBeLessThan(1.0);
    expect(object.position.y).toBeGreaterThan(0.2);

    const poses = physics.disturbedTransforms();
    expect(Object.keys(poses)).toEqual(["crate"]);
    expect(validTransform(poses.crate)).toBe(true);

    // Restore into a fresh world: the pose applies to body AND mesh.
    const physics2 = new WorldPhysics(rapier, 1);
    physics2.addStaticShell(groundGeometry());
    const object2 = propObject(start);
    reoriginObject(object2);
    physics2.addProp("crate", object2, [cubePoints()]);
    physics2.applyTransforms(poses as PropTransforms);
    expect(object2.position.y).toBeCloseTo(object.position.y, 5);

    physics.dispose();
    physics2.dispose();
  });

  it("is deterministic for identical runs", async () => {
    const rapier = await loadRapier();
    const run = () => {
      const physics = new WorldPhysics(rapier, 1);
      physics.addStaticShell(groundGeometry());
      const object = propObject(new THREE.Vector3(0.3, 2.5, -0.2));
      reoriginObject(object);
      const prop = physics.addProp("crate", object, [cubePoints()]);
      prop!.body.wakeUp();
      settle(physics, new THREE.Vector3(50, 2, 50), 1.5);
      const pose = physics.disturbedTransforms().crate;
      physics.dispose();
      return pose;
    };
    const a = run();
    const b = run();
    expect(a).toEqual(b);
  });

  it("carry follows the camera and release drops back to dynamic", async () => {
    const rapier = await loadRapier();
    const physics = new WorldPhysics(rapier, 1);
    physics.addStaticShell(groundGeometry());
    const object = propObject(new THREE.Vector3(0, 0.5, -2));
    reoriginObject(object);
    physics.addProp("crate", object, [cubePoints()]);

    const cam = new THREE.PerspectiveCamera();
    cam.position.set(0, 1.7, 0);
    cam.lookAt(0, 1.7, -1);
    cam.updateMatrixWorld(true);

    expect(physics.pickUp("crate")).toBe(true);
    expect(physics.carrying).toBe("crate");
    expect(physics.pickUp("crate")).toBe(false); // hands already full

    settle(physics, cam.position, 0.5, cam);
    // Carried prop hovers near the hold point in front of the camera.
    expect(object.position.z).toBeLessThan(0);
    expect(object.position.z).toBeGreaterThan(-2);
    expect(object.position.y).toBeGreaterThan(1.0);

    expect(physics.release(cam, true)).toBe("crate");
    expect(physics.carrying).toBeNull();
    settle(physics, new THREE.Vector3(50, 2, 50), 2);
    // Thrown: it leaves the hold point and ends up resting on the floor.
    expect(object.position.y).toBeLessThan(1.0);
    physics.dispose();
  });
});
