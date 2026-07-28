// Rapier physics for world props — P1 of the playable-worlds roadmap.
//
// Scope is deliberate: the PLAYER keeps the proven capsule-vs-BVH controller
// in world-walker.ts (shell collision unchanged); Rapier owns the PROPS. The
// player is mirrored into the physics world as a kinematic capsule so walking
// into a box shoves it — the player is never blocked by a prop, which both
// avoids a two-solver fight over the camera and feels right for pickable
// objects. Props rest on the same watertight collision shell the capsule
// walks on, loaded here as a static trimesh.
//
// Prop geometry arrives world-space-baked with identity node transforms (the
// trimesh exporter contract, world-walker.ts:12), so a prop must be
// RE-ORIGINED before it can tumble: bake -centre into the geometry once and
// carry the centre in the object's position. reoriginObject() does this
// in-place and is idempotent.
//
// Determinism note for tests: Rapier is deterministic for a fixed timestep
// and insertion order; step() uses a fixed 1/60 accumulator (capped to avoid
// the spiral of death on tab-switch resumes).

import * as THREE from "three";
import type RAPIER_NS from "@dimforge/rapier3d-compat";

export type Rapier = typeof RAPIER_NS;

let rapierModule: Rapier | null = null;

/** Load + init the WASM once per page. Safe to call repeatedly. */
export async function loadRapier(): Promise<Rapier> {
  if (rapierModule) return rapierModule;
  const mod = await import("@dimforge/rapier3d-compat");
  await mod.init();
  rapierModule = mod;
  return mod;
}

export const FIXED_DT = 1 / 60;
/** Never simulate more than this many substeps per frame (tab-switch gaps). */
const MAX_SUBSTEPS = 8;
/** Linear damping keeps captured-clutter props from sliding like hockey pucks. */
const PROP_LINEAR_DAMPING = 0.4;
const PROP_ANGULAR_DAMPING = 0.8;
/** Hold-point for a carried prop, metres in front of the camera. */
const CARRY_DISTANCE_M = 1.2;
const THROW_SPEED_MPS = 7.0;

export interface PropTransform {
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

export type PropTransforms = Record<string, PropTransform>;

export interface PhysicsProp {
  slug: string;
  object: THREE.Object3D;
  body: RAPIER_NS.RigidBody;
  carried: boolean;
  /** True once the prop has ever moved from its authored pose. */
  disturbed: boolean;
}

/**
 * Bake the object's geometry origin to its AABB centre so position/rotation
 * mean what physics expects. Idempotent via a userData flag.
 */
export function reoriginObject(object: THREE.Object3D): THREE.Vector3 {
  if (object.userData.reorigined) {
    return object.position.clone();
  }
  const box = new THREE.Box3().setFromObject(object);
  const centre = box.getCenter(new THREE.Vector3());
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const geometry = mesh.geometry as THREE.BufferGeometry;
    geometry.translate(-centre.x, -centre.y, -centre.z);
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  });
  object.position.copy(centre);
  object.userData.reorigined = true;
  object.updateWorldMatrix(true, true);
  return centre;
}

/** Deduplicated vertex positions of an object, in its LOCAL (re-origined) frame. */
export function collectLocalPoints(object: THREE.Object3D, maxPoints = 4096): Float32Array {
  const points: number[] = [];
  object.updateWorldMatrix(true, true);
  const inv = new THREE.Matrix4().copy(object.matrixWorld).invert();
  const v = new THREE.Vector3();
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const pos = (mesh.geometry as THREE.BufferGeometry).getAttribute("position");
    if (!pos) return;
    const stride = Math.max(1, Math.floor(pos.count / maxPoints));
    for (let i = 0; i < pos.count; i += stride) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld).applyMatrix4(inv);
      points.push(v.x, v.y, v.z);
    }
  });
  return new Float32Array(points);
}

export class WorldPhysics {
  private readonly rapier: Rapier;
  private readonly world: RAPIER_NS.World;
  private readonly props = new Map<string, PhysicsProp>();
  private playerBody: RAPIER_NS.RigidBody | null = null;
  private accumulator = 0;
  private unitsPerMetre = 1;
  private carriedSlug: string | null = null;
  /** Scratch — per-frame allocations are banned in the loop. */
  private readonly _q = new THREE.Quaternion();
  private readonly _v = new THREE.Vector3();
  private readonly _fwd = new THREE.Vector3();

  constructor(rapier: Rapier, unitsPerMetre: number) {
    this.rapier = rapier;
    this.unitsPerMetre = Math.max(1e-6, unitsPerMetre);
    // Gravity in SCENE units: 9.81 m/s^2 * units-per-metre.
    this.world = new rapier.World({ x: 0, y: -9.81 * this.unitsPerMetre, z: 0 });
  }

  setUnitsPerMetre(unitsPerMetre: number): void {
    this.unitsPerMetre = Math.max(1e-6, unitsPerMetre);
    this.world.gravity = { x: 0, y: -9.81 * this.unitsPerMetre, z: 0 };
  }

  /** The watertight walkable solid as fixed collision (props rest on it). */
  addStaticShell(geometry: THREE.BufferGeometry): void {
    const pos = geometry.getAttribute("position");
    if (!pos) return;
    const vertices = new Float32Array(pos.array as ArrayLike<number>);
    let indices: Uint32Array;
    if (geometry.index) {
      indices = new Uint32Array(geometry.index.array as ArrayLike<number>);
    } else {
      indices = new Uint32Array(pos.count);
      for (let i = 0; i < pos.count; i++) indices[i] = i;
    }
    const body = this.world.createRigidBody(this.rapier.RigidBodyDesc.fixed());
    this.world.createCollider(
      this.rapier.ColliderDesc.trimesh(vertices, indices),
      body,
    );
  }

  /**
   * Register a prop as a dynamic rigid body. `hullPoints` should come from
   * the prop's CoACD UCX hulls when available (each hull = one convex
   * collider); falls back to one convex hull of the render mesh otherwise.
   * The object must already be re-origined.
   */
  addProp(
    slug: string,
    object: THREE.Object3D,
    hullPoints: Float32Array[],
  ): PhysicsProp | null {
    if (this.props.has(slug)) return this.props.get(slug) ?? null;
    const desc = this.rapier.RigidBodyDesc.dynamic()
      .setTranslation(object.position.x, object.position.y, object.position.z)
      .setRotation({
        x: object.quaternion.x, y: object.quaternion.y,
        z: object.quaternion.z, w: object.quaternion.w,
      })
      .setLinearDamping(PROP_LINEAR_DAMPING)
      .setAngularDamping(PROP_ANGULAR_DAMPING)
      .setCcdEnabled(true);
    const body = this.world.createRigidBody(desc);
    let colliders = 0;
    for (const points of hullPoints) {
      if (points.length < 12) continue; // a hull needs 4+ points
      const collider = this.rapier.ColliderDesc.convexHull(points);
      if (collider) {
        this.world.createCollider(collider.setDensity(200), body);
        colliders++;
      }
    }
    if (!colliders) {
      this.world.removeRigidBody(body);
      return null;
    }
    body.sleep(); // undisturbed props cost nothing until touched
    const prop: PhysicsProp = { slug, object, body, carried: false, disturbed: false };
    this.props.set(slug, prop);
    return prop;
  }

  hasProp(slug: string): boolean {
    return this.props.has(slug);
  }

  /** Mirror the player capsule so contact shoves props. */
  installPlayer(radius: number, height: number): void {
    if (this.playerBody) {
      this.world.removeRigidBody(this.playerBody);
    }
    this.playerBody = this.world.createRigidBody(
      this.rapier.RigidBodyDesc.kinematicPositionBased(),
    );
    const half = Math.max(1e-4, (height - 2 * radius) / 2);
    this.world.createCollider(
      this.rapier.ColliderDesc.capsule(half, radius),
      this.playerBody,
    );
  }

  /**
   * Advance simulation with a fixed-step accumulator and write body poses
   * back to the prop objects. `eyePosition` is the camera (top of capsule);
   * the kinematic mirror is positioned so its centre matches the capsule the
   * walker sweeps.
   */
  step(dt: number, eyePosition: THREE.Vector3, eyeHeight: number, camera: THREE.Camera): void {
    if (this.playerBody) {
      this.playerBody.setNextKinematicTranslation({
        x: eyePosition.x,
        y: eyePosition.y - eyeHeight / 2,
        z: eyePosition.z,
      });
    }
    this.accumulator = Math.min(this.accumulator + dt, FIXED_DT * MAX_SUBSTEPS);
    while (this.accumulator >= FIXED_DT) {
      if (this.carriedSlug) this.updateCarriedTarget(camera);
      this.world.timestep = FIXED_DT;
      this.world.step();
      this.accumulator -= FIXED_DT;
    }
    for (const prop of this.props.values()) {
      if (prop.body.isSleeping() && !prop.carried) continue;
      const t = prop.body.translation();
      const r = prop.body.rotation();
      prop.object.position.set(t.x, t.y, t.z);
      prop.object.quaternion.set(r.x, r.y, r.z, r.w);
      prop.disturbed = true;
    }
  }

  /* ---------------- carry / throw ---------------- */

  get carrying(): string | null {
    return this.carriedSlug;
  }

  pickUp(slug: string): boolean {
    const prop = this.props.get(slug);
    if (!prop || this.carriedSlug) return false;
    prop.body.setBodyType(this.rapier.RigidBodyType.KinematicPositionBased, true);
    prop.carried = true;
    this.carriedSlug = slug;
    return true;
  }

  /** Release the carried prop; with `throwIt` it leaves along the camera ray. */
  release(camera: THREE.Camera, throwIt: boolean): string | null {
    const slug = this.carriedSlug;
    if (!slug) return null;
    const prop = this.props.get(slug);
    this.carriedSlug = null;
    if (!prop) return null;
    prop.carried = false;
    prop.body.setBodyType(this.rapier.RigidBodyType.Dynamic, true);
    if (throwIt) {
      camera.getWorldDirection(this._fwd);
      const speed = THROW_SPEED_MPS * this.unitsPerMetre;
      prop.body.setLinvel(
        { x: this._fwd.x * speed, y: this._fwd.y * speed + 0.5, z: this._fwd.z * speed },
        true,
      );
      prop.body.setAngvel({ x: 0.5, y: 0.3, z: 0.5 }, true);
    } else {
      prop.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    }
    return slug;
  }

  private updateCarriedTarget(camera: THREE.Camera): void {
    const prop = this.carriedSlug ? this.props.get(this.carriedSlug) : null;
    if (!prop) return;
    camera.getWorldDirection(this._fwd);
    camera.getWorldPosition(this._v);
    this._v.addScaledVector(this._fwd, CARRY_DISTANCE_M * this.unitsPerMetre);
    prop.body.setNextKinematicTranslation({ x: this._v.x, y: this._v.y, z: this._v.z });
    this._q.copy((camera as THREE.PerspectiveCamera).quaternion);
    prop.body.setNextKinematicRotation({
      x: this._q.x, y: this._q.y, z: this._q.z, w: this._q.w,
    });
  }

  /* ---------------- persistence ---------------- */

  /** Poses of every prop that has moved from its authored placement. */
  disturbedTransforms(): PropTransforms {
    const out: PropTransforms = {};
    for (const prop of this.props.values()) {
      if (!prop.disturbed) continue;
      const t = prop.body.translation();
      const r = prop.body.rotation();
      out[prop.slug] = {
        position: [t.x, t.y, t.z],
        quaternion: [r.x, r.y, r.z, r.w],
      };
    }
    return out;
  }

  /** Restore persisted poses (a reload's saved game). */
  applyTransforms(transforms: PropTransforms): void {
    for (const [slug, pose] of Object.entries(transforms)) {
      const prop = this.props.get(slug);
      if (!prop || !validTransform(pose)) continue;
      const [x, y, z] = pose.position;
      const [qx, qy, qz, qw] = pose.quaternion;
      prop.body.setTranslation({ x, y, z }, false);
      prop.body.setRotation({ x: qx, y: qy, z: qz, w: qw }, false);
      prop.object.position.set(x, y, z);
      prop.object.quaternion.set(qx, qy, qz, qw);
      prop.disturbed = true;
    }
  }

  dispose(): void {
    this.world.free();
    this.props.clear();
    this.playerBody = null;
    this.carriedSlug = null;
  }
}

export function validTransform(pose: unknown): pose is PropTransform {
  if (typeof pose !== "object" || pose === null) return false;
  const p = (pose as PropTransform).position;
  const q = (pose as PropTransform).quaternion;
  return (
    Array.isArray(p) && p.length === 3 && p.every((n) => Number.isFinite(n)) &&
    Array.isArray(q) && q.length === 4 && q.every((n) => Number.isFinite(n)) &&
    Math.abs(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2 - 1) < 0.01
  );
}
