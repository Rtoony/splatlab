import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { MeshBVH } from "three-mesh-bvh";
import { findWalkingStart, inspectWalkingPose, walkingBodyError, type WalkingBody } from "./walking-admission";
import { DEFAULT_WALK_PARAMS, WorldWalker } from "./world-walker";

const body = (): WalkingBody => ({ radiusM: 0.32, totalHeightM: 2.02, unitsPerMetre: 1 });

function room(ceilingY = 3, wall = false, units = 1) {
  const shapes = [
    new THREE.BoxGeometry(6, 0.2, 6).translate(0, -0.1, 0),
    new THREE.BoxGeometry(6, 0.2, 6).translate(0, ceilingY + 0.1, 0),
  ];
  if (wall) shapes.push(new THREE.BoxGeometry(0.2, ceilingY, 6).translate(0.2, ceilingY / 2, 0));
  const geometry = mergeGeometries(shapes)!;
  geometry.scale(units, units, units);
  geometry.computeBoundingBox();
  const bvh = new MeshBVH(geometry);
  const collider = new THREE.Mesh(geometry);
  const walker = Object.create(WorldWalker.prototype) as WorldWalker;
  Object.assign(walker, {
    bvh,
    collider,
    params: { ...DEFAULT_WALK_PARAMS, physicsProps: false },
    camera: new THREE.PerspectiveCamera(),
    velocity: new THREE.Vector3(),
    grounded: false,
    flying: true,
    controls: { isLocked: true },
    keys: new Set(),
    sceneBox: geometry.boundingBox!.clone(),
    spawn: new THREE.Vector3(),
    spawnSeed: new THREE.Vector3(0, 0.3, 0),
    spawnFloorY: 0,
    spawnTopY: 1.5,
    onWalkingAdmission: vi.fn(),
    onFlyChange: vi.fn(),
  });
  walker.camera.position.set(0, 1.8 * units, 0);
  return {
    bvh,
    collider,
    walker,
    dispose: () => {
      geometry.dispose();
      shapes.forEach((shape) => shape.dispose());
    },
  };
}

describe("fixed-metric walking admission", () => {
  it("reports rejected mouse capture without leaving an active walking body", async () => {
    const fixture = room();
    try {
      const walker = fixture.walker;
      const failure = vi.fn();
      const request = vi.fn().mockRejectedValue(new Error("Browser focus required"));
      walker.setParams({ bodySizing: "fixed-metric" });
      walker.beginWalking();
      Object.assign(walker, { onControlsError: failure });
      Object.assign(walker.controls, {
        isLocked: false,
        domElement: { isConnected: true, requestPointerLock: request },
      });
      await walker.requestLock();
      expect(walker.isFlying).toBe(true);
      expect(request).toHaveBeenCalledWith({ unadjustedMovement: false });
      expect(failure).toHaveBeenCalledWith(expect.stringContaining("Focus this tab"));
    } finally {
      fixture.dispose();
    }
  });

  it("releases real controls and held movement keys without altering body dimensions", () => {
    const fixture = room();
    try {
      const walker = fixture.walker;
      const unlock = vi.fn();
      const keys = new Set(["KeyW"]);
      Object.assign(walker, { keys });
      Object.assign(walker.controls, { unlock });
      walker.setParams({ bodySizing: "fixed-metric" });
      const profile = walker.walkingBody;
      walker.releaseControls();
      expect(unlock).toHaveBeenCalledOnce();
      expect(keys.size).toBe(0);
      expect(walker.walkingBody).toEqual(profile);
    } finally {
      fixture.dispose();
    }
  });

  it.each([
    { radiusM: 0 },
    { radiusM: 0.6 },
    { totalHeightM: NaN },
    { totalHeightM: 3 },
    { unitsPerMetre: 0 },
    { unitsPerMetre: Infinity },
  ])("rejects invalid dimensions %j", (patch) => {
    expect(walkingBodyError({ ...body(), ...patch })).not.toBeNull();
  });

  it.each([1, 2])("admits a full-size body on actual geometry at units=%s", (units) => {
    const fixture = room(3, false, units);
    try {
      const profile = { ...body(), unitsPerMetre: units };
      const result = findWalkingStart(fixture.bvh, fixture.collider, fixture.walker.camera.position, profile);
      expect(result.ok).toBe(true);
      expect(result.position?.[1]).toBeCloseTo(1.71 * units, 6);
      expect(result.body).toEqual(profile);
      expect(fixture.walker.camera.position.y).toBe(1.8 * units);
    } finally {
      fixture.dispose();
    }
  });

  it.each([
    { ceiling: 1.9, wall: false },
    { ceiling: 3, wall: true },
  ])("refuses real obstruction %j without a smaller body", (config) => {
    const fixture = room(config.ceiling, config.wall);
    try {
      const result = findWalkingStart(fixture.bvh, fixture.collider, fixture.walker.camera.position, body());
      expect(result.ok).toBe(false);
      expect(result.message).toContain("not shortened");
      expect(result.body.totalHeightM).toBe(2.02);
    } finally {
      fixture.dispose();
    }
  });

  it("refuses missing geometry, falling poses and unverified collider transforms", () => {
    const fixture = room();
    try {
      expect(findWalkingStart(null, null, new THREE.Vector3(), body()).ok).toBe(false);
      expect(inspectWalkingPose(fixture.bvh, fixture.collider, new THREE.Vector3(0, 2.5, 0), body()).ok).toBe(false);
      expect(findWalkingStart(fixture.bvh, fixture.collider, new THREE.Vector3(20, 2, 0), body()).ok).toBe(false);
      fixture.collider.matrixWorld.makeScale(2, 2, 2);
      expect(findWalkingStart(fixture.bvh, fixture.collider, new THREE.Vector3(0, 2, 0), body()).message).toContain(
        "world-baked",
      );
    } finally {
      fixture.dispose();
    }
  });

  it("does not raise a camera from the floor through an arbitrary obstacle", () => {
    const fixture = room();
    try {
      expect(findWalkingStart(fixture.bvh, fixture.collider, new THREE.Vector3(0, 0.3, 0), body()).ok).toBe(false);
      expect(findWalkingStart(fixture.bvh, fixture.collider, new THREE.Vector3(0, -0.05, 0), body()).ok).toBe(false);
    } finally {
      fixture.dispose();
    }
  });

  it("uses full dimensions despite the legacy source-height estimate, then invokes real capsule physics", () => {
    const fixture = room();
    try {
      const walker = fixture.walker;
      walker.setParams({ bodySizing: "fixed-metric" });
      const result = walker.beginWalking();
      expect(result.ok).toBe(true);
      expect(walker.isFlying).toBe(false);
      const internals = walker as unknown as {
        capsuleHeight: number;
        capsuleRadius: number;
        stepPlayer: (seconds: number) => void;
        keys: Set<string>;
      };
      expect(internals.capsuleHeight + internals.capsuleRadius).toBe(2.02);
      internals.keys.add("KeyW");
      for (let step = 0; step < 30; step++) internals.stepPlayer(1 / 120);
      expect(walker.camera.position.z).toBeLessThan(-0.8);
      expect(walker.camera.position.y).toBeCloseTo(1.7, 3);
      expect(internals.capsuleHeight + internals.capsuleRadius).toBe(2.02);
    } finally {
      fixture.dispose();
    }
  });

  it("refuses F/setFlying bypass and stops safely when dimensions or scale change", () => {
    const fixture = room();
    try {
      const walker = fixture.walker;
      walker.setParams({ bodySizing: "fixed-metric" });
      walker.camera.position.y = 2.5;
      walker.setFlying(false);
      expect(walker.isFlying).toBe(true);
      walker.camera.position.y = 1.8;
      expect(walker.beginWalking().ok).toBe(true);
      walker.setParams({ radiusM: 0.3 });
      expect(walker.isFlying).toBe(true);
      expect(() => walker.setParams({ eyeHeightM: 5 })).toThrow("total height");
      expect(walker.params.eyeHeightM).toBe(1.7);
      walker.beginWalking();
      walker.setParams({ unitsPerMetre: 2 });
      expect(walker.isFlying).toBe(true);
      walker.setParams({ unitsPerMetre: 1 });
      expect(walker.beginWalking().ok).toBe(true);
      walker.setParams({ bodySizing: "legacy-fit" });
      expect(walker.isFlying).toBe(true);
    } finally {
      fixture.dispose();
    }
  });

  it("respawns in inspection mode rather than ejecting or shrinking inside a low ceiling", () => {
    const fixture = room(1.9);
    try {
      const walker = fixture.walker;
      walker.setParams({ bodySizing: "fixed-metric" });
      walker.respawn();
      expect(walker.isFlying).toBe(true);
      expect(walker.walkingBody.totalHeightM).toBe(2.02);
      expect(walker.camera.position.y).toBeCloseTo(1.8);
      expect(walker.beginWalking().ok).toBe(false);
    } finally {
      fixture.dispose();
    }
  });

  it("keeps candidate inspection read-only and refuses the legacy sizing mode", () => {
    const fixture = room();
    try {
      const walker = fixture.walker;
      expect(walker.assessWalkingStart().ok).toBe(false);
      walker.setParams({ bodySizing: "fixed-metric" });
      const position = walker.camera.position.clone();
      const params = { ...walker.params };
      expect(walker.assessWalkingStart(new THREE.Vector3(0.5, 1.8, 0)).ok).toBe(true);
      expect(walker.camera.position.equals(position)).toBe(true);
      expect(walker.params).toEqual(params);
      expect(walker.isFlying).toBe(true);
    } finally {
      fixture.dispose();
    }
  });
});
