// The ground probe, against a REAL MeshBVH — three-mesh-bvh is pure JS and
// needs no renderer, so these are the actual raycasts the walker performs.
//
// The geometry mirrors what a solidified outdoor world really looks like
// (measured on the Stump): a thick terrain SLAB with an underside a metre
// below its walkable top, plus canopy sheets overhead. Both naive picks —
// first hit from the sky, lowest hit in the column — choose wrong here.

import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { MeshBVH } from "three-mesh-bvh";
import { GLTFLoader, type GLTF } from "three/addons/loaders/GLTFLoader.js";
import {
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatMesh,
} from "@sparkjsdev/spark";
import { WorldWalker, DEFAULT_WALK_PARAMS, collisionScaleToWorld, type CurtainParams } from "./world-walker";

describe("collision capture-to-world units", () => {
  it("uses the explicit conversion, without guessing from player scale", () => {
    expect(collisionScaleToWorld({ meters_per_unit: 1, collision_shell: { scale_to_world: 0.94975 } })).toBe(0.94975);
    expect(collisionScaleToWorld({ meters_per_unit: 2 })).toBe(1);
  });

  it.each([0, -1, NaN, Infinity])("refuses an invalid collision scale %s", scale => {
    expect(() => collisionScaleToWorld({ collision_shell: { scale_to_world: scale } })).toThrow("invalid");
  });
});

describe("architectural threshold collider picking", () => {
  it("returns the real nearest BVH face in world coordinates, never a hidden surface behind a wall", () => {
    const floor = new THREE.BoxGeometry(4, 0.2, 4).translate(3, -1.1, 4);
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, { bvh: new MeshBVH(floor) });
    const ray = new THREE.Ray(new THREE.Vector3(3, 2, 4), new THREE.Vector3(0, -1, 0));
    const hit = walker.pickCollisionSurface(ray)!;
    expect(hit.point.y).toBeCloseTo(-1);
    expect(hit.normal.toArray()).toEqual([0, 1, 0]);
    const side = walker.pickCollisionSurface(new THREE.Ray(new THREE.Vector3(0, -1.1, 4), new THREE.Vector3(1, 0, 0)))!;
    expect(side.point.x).toBeCloseTo(1);
    expect(side.normal.y).toBe(0);
    hit.point.y = 123;
    expect(walker.pickCollisionSurface(ray)!.point.y).toBeCloseTo(-1);
    expect(ray.origin.toArray()).toEqual([3, 2, 4]);
    floor.dispose();
  });
  it("refuses missing collision, malformed rays and hits beyond the bounded pick range", () => {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    const ray = new THREE.Ray(new THREE.Vector3(0, 200, 0), new THREE.Vector3(0, -1, 0));
    Object.assign(walker, { bvh: null });
    expect(walker.pickCollisionSurface(ray)).toBeNull();
    const floor = new THREE.BoxGeometry(4, 0.2, 4);
    Object.assign(walker, { bvh: new MeshBVH(floor) });
    expect(walker.pickCollisionSurface(ray)).toBeNull();
    ray.origin.y = 2;
    ray.direction.y = -2;
    expect(walker.pickCollisionSurface(ray)).toBeNull();
    ray.direction.y = NaN;
    expect(walker.pickCollisionSurface(ray)).toBeNull();
    floor.dispose();
  });
});

describe("revision-pinned captured visibility", () => {
  it("waits for captured appearance before applying row selections", async () => {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    let ready!: () => void;
    const initialized = new Promise<void>(resolve => { ready = resolve; });
    const pluck = vi.fn(() => true);
    Object.assign(walker, { backdrop: { initialized }, pluckElement: pluck });
    const applied = walker.applyCapturedVisibility(["chair", "box"]);
    expect(pluck).not.toHaveBeenCalled();
    ready();
    await applied;
    expect(pluck.mock.calls).toEqual([["chair"], ["box"]]);
  });

  it("prepares the masked splat renderer before reporting readiness", async () => {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    const events: string[] = [];
    Object.assign(walker, { backdrop: { initialized: Promise.resolve() },
      pluckElement: () => { events.push("mask"); return true; },
      spark: { update: async () => { events.push("renderer-ready"); } } });
    await walker.applyCapturedVisibility(["box"]);
    expect(events).toEqual(["mask", "renderer-ready"]);
  });

  it("reports a failed backdrop even when there are no hidden rows", async () => {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, { backdrop: { initialized: Promise.reject(new Error("load failed")) } });
    await expect(walker.applyCapturedVisibility([])).rejects.toThrow("load failed");
  });

  it("refuses missing backdrops and invalid row addressing", async () => {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, { backdrop: null });
    await expect(walker.applyCapturedVisibility(["chair"])).rejects.toThrow("backdrop");
    Object.assign(walker, { backdrop: { initialized: Promise.resolve() }, pluckElement: () => false });
    await expect(walker.applyCapturedVisibility(["chair"])).rejects.toThrow("Cannot safely hide");
  });
});

/** A slab whose top is at `topY` and underside at `topY - thickness`. */
function slab(topY: number, thickness: number, size = 20): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(size, thickness, size);
  g.translate(0, topY - thickness / 2, 0);
  return g;
}

function sheet(y: number, size = 20): THREE.BufferGeometry {
  const g = new THREE.PlaneGeometry(size, size);
  g.rotateX(-Math.PI / 2);
  g.translate(0, y, 0);
  return g;
}

/** A walker with only the fields the probe touches — no canvas, no GL. */
function probeRig(geoms: THREE.BufferGeometry[]) {
  const merged = new THREE.BufferGeometry();
  const positions: number[] = [];
  for (const g of geoms) {
    const nonIndexed = g.index ? g.toNonIndexed() : g;
    positions.push(...Array.from(nonIndexed.getAttribute("position").array));
  }
  merged.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  merged.computeBoundingBox();
  const walker = Object.create(WorldWalker.prototype) as WorldWalker;
  Object.assign(walker, {
    bvh: new MeshBVH(merged, { maxLeafTris: 8 }),
    sceneBox: merged.boundingBox!.clone(),
  });
  return walker;
}

describe("architectural capsule proof against actual walker physics", () => {
  async function proofTools() {
    const navigationModule = new URL("../../../tools/architectural-navigation-proof.mjs", import.meta.url).href;
    const contractModule = new URL("../../../tools/architectural-proof-contract.mjs", import.meta.url).href;
    return { ...await import(navigationModule), ...await import(contractModule) };
  }
  function doorway(open: boolean, headroom = 3) {
    const geometries = [new THREE.BoxGeometry(8, 0.2, 8).translate(0, -0.1, 0),
      new THREE.BoxGeometry(2, 3, 0.2).translate(-1.55, 1.5, 0),
      new THREE.BoxGeometry(2, 3, 0.2).translate(1.55, 1.5, 0),
      new THREE.BoxGeometry(1.1, 0.9, 0.2).translate(0, 2.55, 0)];
    if (!open) geometries.push(new THREE.BoxGeometry(1.1, 2.1, 0.2).translate(0, 1.05, 0));
    const walker = probeRig(geometries);
    const geometry = (walker as unknown as { bvh: MeshBVH }).bvh.geometry;
    Object.assign(walker, { collider: new THREE.Mesh(geometry), camera: new THREE.PerspectiveCamera(),
      params: { ...DEFAULT_WALK_PARAMS, physicsProps: false }, controls: { isLocked: true },
      keys: new Set(), velocity: new THREE.Vector3(), grounded: false, flying: true,
      spawnFloorY: 0, spawnTopY: headroom, stop: vi.fn() });
    const contract = { spec: { origin: [0, 0, 0], yaw_degrees: 0 }, radius: 0.22, height: 1.7,
      lane: [[0, 0, -0.9], [0, 0, 2.8]], floor: [0, 0] };
    return { walker, contract, dispose: () => { geometry.dispose(); geometries.forEach(item => item.dispose()); } };
  }
  it.each([false, true])("walks an actual open fixture continuously with reverse=%s and restores configuration", async reverse => {
    const { traceArchitecturalLane, validateCapsuleTrace } = await proofTools();
    const fixture = doorway(true);
    vi.stubGlobal("window", { __sceneStudioWalker: fixture.walker });
    try {
      for (const profile of [{ radius: 0.22, height: 1.7 }, { radius: 0.32, height: 2.02 }]) {
        const contract = { ...fixture.contract, ...profile };
        const trace = traceArchitecturalLane({ ...contract, reverse });
        expect(validateCapsuleTrace(trace, contract, reverse).grounded_fraction).toBeGreaterThan(0.95);
        expect(fixture.walker.params.eyeHeightM).toBe(DEFAULT_WALK_PARAMS.eyeHeightM);
        expect(fixture.walker.isFlying).toBe(true);
      }
    } finally { vi.unstubAllGlobals(); fixture.dispose(); }
  });
  it("cannot pass by driving the camera through an uncut wall", async () => {
    const { traceArchitecturalLane, validateCapsuleTrace } = await proofTools();
    const fixture = doorway(false);
    vi.stubGlobal("window", { __sceneStudioWalker: fixture.walker });
    try {
      const trace = traceArchitecturalLane(fixture.contract);
      expect(() => validateCapsuleTrace(trace, fixture.contract)).toThrow("did not reach");
    } finally { vi.unstubAllGlobals(); fixture.dispose(); }
  });
  it("refuses auto-shortened capsules before traversal and restores their settings", async () => {
    const { traceArchitecturalLane } = await proofTools();
    const fixture = doorway(true, 1.5);
    vi.stubGlobal("window", { __sceneStudioWalker: fixture.walker });
    try {
      expect(() => traceArchitecturalLane(fixture.contract)).toThrow("shortening");
      expect(fixture.walker.params.eyeHeightM).toBe(DEFAULT_WALK_PARAMS.eyeHeightM);
      expect(fixture.walker.isFlying).toBe(true);
    } finally { vi.unstubAllGlobals(); fixture.dispose(); }
  });
});

describe("ground probe (real BVH)", () => {
  // Turf top at 0, slab underside at -1, canopy at +4 — the Stump's shape.
  const world = () => probeRig([slab(0, 1), sheet(4)]);

  // Probe off-centre: a ray straight down the box's exact centre grazes the
  // shared diagonal of two triangles and reports each surface twice. Real
  // behaviour, harmless to the pickers (they compare, not count) — but it
  // makes a raw hit-list assertion misleading.
  const PX = 1.3, PZ = -2.1;

  it("columnHits sees every face, lowest first", () => {
    const hits = world().columnHits(PX, PZ);
    expect(hits.map((y) => Math.round(y * 100) / 100)).toEqual([-1, 0, 4]);
  });

  it("duplicate grazing hits do not confuse the pickers", () => {
    const w = world();
    expect(w.columnHits(0, 0).length).toBeGreaterThan(3); // the degenerate ray
    expect(w.surfaceNear(0, 0, 0.1, 1.6)).toBeCloseTo(0, 5);
  });

  it("surfaceNear picks the turf, not the canopy or the slab underside", () => {
    const w = world();
    // Reference = the navmesh's graded floor (0.1 off, as a real grade is).
    expect(w.surfaceNear(PX, PZ, 0.1, 1.6)).toBeCloseTo(0, 5);
    // Both naive rules would be wrong here, and the test says so out loud:
    expect(Math.max(...w.columnHits(PX, PZ))).toBe(4); // highest = canopy
    expect(Math.min(...w.columnHits(PX, PZ))).toBe(-1); // lowest = underside
  });

  it("surfaceNear follows the grade: nearest face wins, band is the limit", () => {
    // A ledge at +1.2 over the same terrain — two candidate surfaces.
    const w = probeRig([slab(0, 1), slab(1.2, 0.2)]);
    expect(w.surfaceNear(PX, PZ, 0, 1.6)).toBeCloseTo(0, 5); // graded low: turf
    expect(w.surfaceNear(PX, PZ, 1.3, 1.6)).toBeCloseTo(1.2, 5); // graded high: ledge
    // A band tight around the ledge excludes the turf entirely...
    expect(w.surfaceNear(PX, PZ, 1.3, 0.5)).toBeCloseTo(1.2, 5);
    // ...and a band that reaches neither surface yields nothing at all.
    expect(w.surfaceNear(PX, PZ, 3, 0.5)).toBeNull();
  });

  it("surfaceNear returns null off the collider so callers keep their floor", () => {
    expect(world().surfaceNear(500, 500, 0, 1.6)).toBeNull();
  });

  it("groundAt (spawn placement) takes the highest face under its cap", () => {
    const w = world();
    expect(w.groundAt(PX, PZ, 10)).toBe(4); // uncapped: the canopy — the roof bug
    expect(w.groundAt(PX, PZ, 2)).toBeCloseTo(0, 5); // capped: the turf
    expect(w.groundAt(PX, PZ, -5)).toBeNull(); // nothing that low
  });

  it("a collider-less walker probes to null rather than throwing", () => {
    const w = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(w, { bvh: null, sceneBox: new THREE.Box3() });
    expect(w.columnHits(PX, PZ)).toEqual([]);
    expect(w.surfaceNear(PX, PZ, 0, 1)).toBeNull();
    expect(w.groundAt(0, 0)).toBeNull();
  });
});

describe("seeded respawn vs low headroom (real BVH + collisions)", () => {
  // The Truck (A9) numbers: an uncalibrated capture guessed at 1.9245 u/m
  // makes the 1.7 m player 3.27 units tall inside a proven headroom of only
  // 2.27 units. The seeded respawn clamped the EYE into the headroom but the
  // physics capsule kept its full height, so its feet sat ~1.9 units inside
  // the floor slab and the depenetration pass hurled the spawn out of the
  // world — an eject/fall/respawn loop the game read as "no reachable spawn
  // ground". The capsule must be clamped to fit the headroom too.
  const FLOOR = -0.3241;
  const TOP = 1.9502;
  const SEED = new THREE.Vector3(0.4176, -0.0241, -0.4565);

  function room(): THREE.BufferGeometry[] {
    const geoms: THREE.BufferGeometry[] = [slab(FLOOR, 1)]; // thick floor slab
    const ceiling = new THREE.PlaneGeometry(20, 20);
    ceiling.rotateX(Math.PI / 2); // faces down
    ceiling.translate(0, TOP, 0);
    geoms.push(ceiling);
    for (const [dx, dz, ry] of [[10, 0, Math.PI / 2], [-10, 0, -Math.PI / 2], [0, 10, Math.PI], [0, -10, 0]]) {
      const wall = new THREE.PlaneGeometry(20, 8);
      wall.rotateY(ry);
      wall.translate(dx, TOP - 4, dz);
      geoms.push(wall);
    }
    return geoms;
  }

  function spawnRig() {
    const walker = probeRig(room());
    const merged = (walker as unknown as { bvh: MeshBVH }).bvh.geometry;
    const collider = new THREE.Mesh(merged);
    collider.updateMatrixWorld(true);
    Object.assign(walker, {
      collider,
      camera: new THREE.PerspectiveCamera(),
      velocity: new THREE.Vector3(),
      spawn: new THREE.Vector3(),
      grounded: false,
      spawnSeed: SEED.clone(),
      spawnFloorY: FLOOR,
      spawnTopY: TOP,
      params: { eyeHeightM: 1.7, radiusM: 0.32, unitsPerMetre: 1.9245 },
    });
    return walker as WorldWalker & { camera: THREE.PerspectiveCamera };
  }

  it("the spawned capsule FITS the proven headroom: feet never below the floor", () => {
    // The actual failure contract. Unfixed, the capsule kept the full 3.27u
    // eye height, so at the clamped spawn eye its feet reached 1.9u below
    // the proven floor — inside the collision solid — and the depenetration
    // pass ejected the player. (The eject itself needs the voxel shell's
    // thousands of jittered faces and is not reproducible with clean box
    // geometry, so the test pins the geometric contract instead.)
    const w = spawnRig();
    w.respawn();
    const internals = w as unknown as {
      capsuleHeight: number; capsuleRadius: number; spawn: THREE.Vector3;
    };
    const feet = internals.spawn.y - internals.capsuleHeight;
    expect(feet).toBeGreaterThanOrEqual(FLOOR - 0.02);
    // The head may graze the ceiling by at most a whisker of the radius.
    const head = internals.spawn.y + internals.capsuleRadius;
    expect(head).toBeLessThanOrEqual(TOP + internals.capsuleRadius * 0.1);
  });

  it("spawns SETTLED: extra collision passes barely move the camera", () => {
    const w = spawnRig();
    w.respawn();
    // The bug's signature was a spawn the collider still disagreed with:
    // every subsequent pass kept displacing the capsule (in the wild, out of
    // the world). Settled means another pass finds ~no penetration to fix.
    const resolve = w as unknown as { resolveCollisions(dt: number): void };
    const before = w.camera.position.clone();
    let travelled = 0;
    for (let i = 0; i < 5; i++) {
      const prev = w.camera.position.clone();
      resolve.resolveCollisions(1 / 60);
      travelled += w.camera.position.distanceTo(prev);
    }
    expect(travelled).toBeLessThan(0.1);
    // And it is standing inside the room, near the seed, not ejected.
    const p = w.camera.position;
    expect(Math.abs(p.x - SEED.x)).toBeLessThan(1.5);
    expect(Math.abs(p.z - SEED.z)).toBeLessThan(1.5);
    expect(p.y).toBeGreaterThan(FLOOR);
    expect(p.y).toBeLessThan(TOP);
    expect(p.distanceTo(before)).toBeLessThan(0.1);
  });

  it("a world with real headroom keeps the full eye height", () => {
    const w = spawnRig();
    Object.assign(w, { spawnTopY: FLOOR + 6, params: { eyeHeightM: 1.7, radiusM: 0.32, unitsPerMetre: 1.0 } });
    const internals = w as unknown as { capsuleHeight: number; eyeHeight: number };
    expect(internals.capsuleHeight).toBeCloseTo(internals.eyeHeight, 6);
  });
});

describe("generated vertex-color delivery", () => {
  it.each([false, true])("preserves exported vertex colors when unlit is %s", async (unlit) => {
    const geometry = new THREE.BoxGeometry();
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(geometry.attributes.position.count * 3).fill(0.4), 3));
    const material = new THREE.MeshStandardMaterial({ vertexColors: true });
    const mesh = new THREE.Mesh(geometry, material);
    const root = new THREE.Group();
    root.add(mesh);
    const loader = vi.spyOn(GLTFLoader.prototype, "loadAsync").mockResolvedValue({ scene: root } as GLTF);
    try {
      const walker = Object.create(WorldWalker.prototype) as WorldWalker;
      Object.assign(walker, { params: { unlit } });
      await (walker as unknown as { loadGlb(source: { fileUrl: (name: string) => string }, url: string, dir: string): Promise<THREE.Object3D> })
        .loadGlb({ fileUrl: (name) => name }, "generated.glb", "");
      expect(mesh.userData.litMaterial.vertexColors).toBe(true);
      expect(mesh.userData.unlitMaterial.vertexColors).toBe(true);
      expect(mesh.material).toBe(unlit ? mesh.userData.unlitMaterial : mesh.userData.litMaterial);
      expect(mesh.geometry.getAttribute("color")).toBe(geometry.getAttribute("color"));
    } finally {
      loader.mockRestore();
    }
  });
});

describe("collider merge (environment role)", () => {
  // A BoxGeometry is 12 tris; positionOnly-style shells carry position+index
  // only, so the fixture shell mimics the load product's attribute shape.
  function bareBox(w: number, h: number, d: number): THREE.BufferGeometry {
    const g = new THREE.BoxGeometry(w, h, d);
    const out = new THREE.BufferGeometry();
    out.setAttribute("position", (g.getAttribute("position") as THREE.BufferAttribute).clone());
    out.setIndex(g.index!.clone());
    return out;
  }

  function mkEl(role: string, provenance: string | null, y = 0) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 0.2, 2));
    mesh.position.y = y;
    const object = new THREE.Group();
    object.add(mesh);
    return {
      slug: `${role}-${provenance ?? "captured"}-${y}`,
      role, provenance, object, collides: false, visible: true,
    };
  }

  function colliderRig(opts: { shell?: boolean; collideProps?: boolean;
                               elements: ReturnType<typeof mkEl>[] }) {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, {
      scene: new THREE.Scene(),
      params: { collideProps: opts.collideProps ?? false, showCollider: false },
      collisionShellGeom: opts.shell === false ? null : bareBox(20, 0.2, 20),
      elements: opts.elements,
      collider: null, colliderWire: null, bvh: null,
      sceneBox: new THREE.Box3(new THREE.Vector3(-10, -1, -10),
                               new THREE.Vector3(10, 5, 10)),
    });
    return walker;
  }

  const internals = (w: WorldWalker) =>
    w as unknown as { colliderTris: number; colliderSource: string };

  it.each(["authored", "generated"])("fast path merges %s environment/static into the shell BVH", (provenance) => {
    const env = mkEl("environment", provenance, 2);
    const authoredStatic = mkEl("static", provenance, 4);
    const w = colliderRig({ elements: [env, authoredStatic] });
    w.rebuildCollider();
    expect(internals(w).colliderTris).toBe(36); // shell 12 + 12 + 12
    expect(internals(w).colliderSource).toBe("collision_shell+authored");
    expect(env.collides).toBe(true);
    expect(authoredStatic.collides).toBe(true);
  });

  it("fast path does NOT double captured statics in (the shell already has them)", () => {
    const captured = mkEl("static", null, 1);
    const w = colliderRig({ elements: [captured] });
    w.rebuildCollider();
    expect(internals(w).colliderTris).toBe(12); // shell only
    expect(internals(w).colliderSource).toBe("collision_shell");
    // Truthful: the collision solid represents it, so collides stays true.
    expect(captured.collides).toBe(true);
  });

  it("studio retains the background solid while adding and removing prop collision", () => {
    const chair = mkEl("prop", null, 1);
    const table = mkEl("prop", null, 2);
    const walker = colliderRig({ elements: [chair, table] });
    walker.setStaticPropCollision(true);
    expect(internals(walker).colliderTris).toBe(36);
    expect(internals(walker).colliderSource).toBe("collision_shell+elements");
    expect(chair.collides).toBe(true);
    walker.elements.splice(0, 1);
    walker.rebuildCollider();
    expect(internals(walker).colliderTris).toBe(24);
    walker.setStaticPropCollision(false);
    expect(internals(walker).colliderTris).toBe(12);
  });

  it("fallback path (no collision shell) merges shell+static+environment", () => {
    const els = [mkEl("shell", null, 0), mkEl("static", null, 1),
                 mkEl("environment", "authored", 2), mkEl("prop", null, 3)];
    const w = colliderRig({ shell: false, elements: els });
    w.rebuildCollider();
    expect(internals(w).colliderTris).toBe(36); // prop excluded
    expect(internals(w).colliderSource).toBe("visual_shell");
    expect(els[3].collides).toBe(false);
  });

  it.each(["authored", "generated"])("a %s platform becomes REAL floor the probes can stand on", (provenance) => {
    const env = mkEl("environment", provenance, 2);
    const w = colliderRig({ elements: [env] });
    w.rebuildCollider();
    // Off-centre probes (the box-centre ray grazes shared diagonals) and a
    // ref above the slab midplane so top/bottom faces cannot tie.
    expect(w.surfaceNear(0.5, 0.3, 2.0, 1.0)).toBeCloseTo(2.1, 3);
    expect(w.surfaceNear(3.3, -2.1, 0.15, 0.5)).toBeCloseTo(0.1, 3); // shell top intact
  });

  it("Rapier gets the SAME solid: a prop settles on the authored platform", async () => {
    const { WorldPhysics, loadRapier, reoriginObject } = await import("./world-physics");
    const env = mkEl("environment", "authored", 2);
    const w = colliderRig({ elements: [env] });
    const built = (w as unknown as {
      buildStaticColliderGeometry(p: boolean): { geom: THREE.BufferGeometry } | null;
    }).buildStaticColliderGeometry(false);
    expect(built).not.toBeNull();
    const rapier = await loadRapier();
    const physics = new WorldPhysics(rapier, 1);
    physics.addStaticShell(built!.geom);
    const object = new THREE.Group();
    const box = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4));
    object.add(box);
    object.position.set(0, 4, 0);
    object.updateWorldMatrix(true, true);
    reoriginObject(object);
    const pts = new Float32Array([
      -0.2, -0.2, -0.2, 0.2, -0.2, -0.2, -0.2, 0.2, -0.2, -0.2, -0.2, 0.2,
      0.2, 0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
    ]);
    const prop = physics.addProp("crate", object, [pts]);
    expect(prop).not.toBeNull();
    prop!.body.wakeUp();
    const eye = new THREE.Vector3(50, 2, 50);
    const cam = new THREE.PerspectiveCamera();
    for (let i = 0; i < 240; i++) physics.step(1 / 60, eye, 1.7, cam);
    // Settled ON the platform (top 2.1 + half-height 0.2), not the shell.
    expect(object.position.y).toBeGreaterThan(2.0);
    expect(object.position.y).toBeLessThan(2.6);
    physics.dispose();
  });
});

describe("pluck (per-prop backdrop-splat rows)", () => {
  // Pluck ships through the mesh's worldModifier lane (mutating the packed
  // array after load is a visual no-op — the pipeline never re-reads it), so
  // the rig fakes a SplatMesh: numSplats + the modifier slot + the
  // updateGenerator call the module rule demands after every (re)assignment.
  const N = 6;
  function pluckRig() {
    const backdrop = {
      packedSplats: { numSplats: N },
      worldModifier: undefined as unknown,
      generatorCalls: 0,
      updateGenerator() { this.generatorCalls += 1; },
    };
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, {
      backdrop,
      elements: [],
      physics: null,
      pluckRows: new Map(), pluckNRows: 0, pluckedSlugs: new Set(),
      pluckMask: null, pluckClock: 0, pluckWarned: false,
      replacedSlugs: new Set(), visibilityOverrides: new Map(),
      restyleShowsMesh: false, restyle: null, interactions: new Map(),
    });
    return { walker, backdrop };
  }

  it("masks a prop's rows, leaves the rest, and clears on unpluck", () => {
    const { walker, backdrop } = pluckRig();
    walker.setPluckDoc({ n_rows: N, elements: { bike: { rows: [1, 4] }, table: { rows: [2] } } });
    expect(walker.pluckElement("bike")).toBe(true);
    const mask = (walker as unknown as { pluckMask: Uint8Array }).pluckMask;
    expect([...mask]).toEqual([0, 255, 0, 0, 255, 0]);
    expect(backdrop.worldModifier).toBeDefined();
    expect(backdrop.generatorCalls).toBe(1);
    expect(walker.pluckState().bike).toMatchObject({ rows: 2, plucked: true, sampleOpacity: 0 });
    expect(walker.pluckState().table.sampleOpacity).toBe(1); // untouched row
    walker.unpluckElement("bike");
    expect([...mask]).toEqual([0, 0, 0, 0, 0, 0]);
    // Last slug removed -> the modifier slot is handed back entirely.
    expect(backdrop.worldModifier).toBeUndefined();
    expect(backdrop.generatorCalls).toBe(2);
    expect(walker.pluckState().bike.plucked).toBe(false);
  });

  it("REFUSES when the backdrop row count mismatches the doc (fmt=web trap)", () => {
    const { walker, backdrop } = pluckRig();
    walker.setPluckDoc({ n_rows: 999, elements: { bike: { rows: [1] } } });
    expect(walker.pluckElement("bike")).toBe(false);
    expect(backdrop.worldModifier).toBeUndefined();
    expect(backdrop.generatorCalls).toBe(0);
  });

  it("isolates review rows without mutating removal state, then restores it", async () => {
    const { walker, backdrop } = pluckRig();
    walker.setPluckDoc({ n_rows: N, elements: { bike: { rows: [1, 4] } } });
    walker.pluckElement("bike");
    const state = walker.pluckState();
    const internal = walker as unknown as { captureInspectionMask: Uint8Array | null; pluckMask: Uint8Array };
    const original = [...internal.pluckMask];
    const update = vi.fn().mockResolvedValue(undefined);
    Object.assign(walker, { spark: { update } });
    await walker.inspectCapturedRows([2, 3]);
    expect([...internal.captureInspectionMask!]).toEqual([255, 255, 0, 0, 255, 255]);
    expect([...internal.pluckMask]).toEqual(original);
    expect(walker.pluckState()).toEqual(state);
    await walker.inspectCapturedRows(null);
    expect(internal.captureInspectionMask).toBeNull();
    expect([...internal.pluckMask]).toEqual(original);
    expect(backdrop.worldModifier).toBeDefined();
    expect(update).toHaveBeenCalledTimes(2);
  });

  it.each([[], [-1], [N], [1.5], [NaN]].map(rows => ({ rows })))("refuses invalid inspection rows $rows", async ({ rows }) => {
    const { walker, backdrop } = pluckRig();
    await expect(walker.inspectCapturedRows(rows)).rejects.toThrow("do not match");
    expect(backdrop.generatorCalls).toBe(0);
  });

  it("inspects remaining capture without resurrecting existing removals or changing their masks", async () => {
    const { walker, backdrop } = pluckRig();
    walker.setPluckDoc({ n_rows: N, elements: { bike: { rows: [1, 4] } } });
    walker.pluckElement("bike");
    const state = walker.pluckState();
    const internal = walker as unknown as { captureInspectionMask: Uint8Array | null; pluckMask: Uint8Array };
    const original = [...internal.pluckMask];
    await walker.inspectCapturedRows([2, 3], "remaining");
    expect([...internal.captureInspectionMask!]).toEqual([0, 255, 255, 255, 255, 0]);
    expect([...internal.pluckMask]).toEqual(original);
    expect(walker.pluckState()).toEqual(state);
    await walker.inspectCapturedRows(null);
    expect(internal.captureInspectionMask).toBeNull();
    expect([...internal.pluckMask]).toEqual(original);
    expect(backdrop.worldModifier).toBeDefined();
  });

  it("can switch remaining capture back to selected rows without persisting either diagnostic", async () => {
    const { walker, backdrop } = pluckRig();
    const internal = walker as unknown as { captureInspectionMask: Uint8Array | null; pluckMask: Uint8Array | null };
    await walker.inspectCapturedRows([2, 3], "remaining");
    expect([...internal.captureInspectionMask!]).toEqual([0, 0, 255, 255, 0, 0]);
    await walker.inspectCapturedRows([2, 3]);
    expect([...internal.captureInspectionMask!]).toEqual([255, 255, 0, 0, 255, 255]);
    await walker.inspectCapturedRows(null);
    expect(internal.captureInspectionMask).toBeNull();
    expect(internal.pluckMask).toBeNull();
    expect(backdrop.worldModifier).toBeUndefined();
  });

  it("waits for the backdrop before installing an inspection mask", async () => {
    const { walker, backdrop } = pluckRig();
    let ready!: () => void;
    Object.assign(backdrop, { initialized: new Promise<void>(resolve => { ready = resolve; }) });
    const pending = walker.inspectCapturedRows([1]);
    expect(backdrop.generatorCalls).toBe(0);
    ready();
    await pending;
    expect(backdrop.generatorCalls).toBe(1);
    await walker.inspectCapturedRows(null);
    expect(backdrop.worldModifier).toBeUndefined();
  });

  it("physics-disturbed props get plucked; undisturbed ones keep their ghosts", () => {
    const { walker } = pluckRig();
    walker.setPluckDoc({ n_rows: N, elements: { bike: { rows: [1] }, table: { rows: [2] } } });
    Object.assign(walker, {
      physics: { disturbedTransforms: () => ({ bike: { position: [0, 0, 0], quaternion: [0, 0, 0, 1] } }) },
    });
    (walker as unknown as { checkPluckDisturbed(): void }).checkPluckDisturbed();
    const state = walker.pluckState();
    expect(state.bike.plucked).toBe(true);
    expect(state.table.plucked).toBe(false);
  });
});

describe("baked look (setBakedLook)", () => {
  // After a bake the overlay document is EMPTY, so restyleShowsMesh would go
  // false and the splat backdrop — still the un-restyled photograph — would
  // hide the baked geometry. The baked flag keeps the mesh on top.
  function rig(withBackdrop = true) {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    const backdrop = { visible: true };
    Object.assign(walker, {
      restyle: null,
      restyleShowsMesh: false,
      bakedLook: false,
      elements: [],
      backdrop: withBackdrop ? backdrop : null,
    });
    return { walker, backdrop };
  }

  it("keeps the mesh in front of the photograph, and revert restores it", () => {
    const { walker, backdrop } = rig();
    walker.setBakedLook(true);
    expect(backdrop.visible).toBe(false);
    walker.setBakedLook(false); // a revert
    expect(backdrop.visible).toBe(true);
  });

  it("is idempotent and safe with no backdrop loaded", () => {
    const { walker } = rig(false);
    walker.setBakedLook(true);
    walker.setBakedLook(true);
    const internals = walker as unknown as { restyleShowsMesh: boolean };
    expect(internals.restyleShowsMesh).toBe(true);
  });
});

describe("curtain (SplatEdit SDF)", () => {
  function curtainRig() {
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, {
      scene: new THREE.Scene(),
      curtainEdit: null,
      curtainSdf: null,
    });
    return walker;
  }
  const params: CurtainParams = {
    enabled: true, shape: "sphere", center: [1, 2, 3],
    radius: 5, halfExtents: [5, 5, 5], softEdge: 1.2,
  };

  it("builds ONE scene-level SplatEdit: inverted zero-opacity SDF, MULTIPLY", () => {
    const w = curtainRig();
    w.setCurtain(params);
    const scene = (w as unknown as { scene: THREE.Scene }).scene;
    const edits = scene.children.filter((c) => c instanceof SplatEdit);
    expect(edits.length).toBe(1);
    const edit = edits[0] as SplatEdit;
    expect(edit.rgbaBlendMode).toBe(SplatEditRgbaBlendMode.MULTIPLY);
    expect(edit.softEdge).toBeCloseTo(1.2, 6);
    const sdf = (w as unknown as { curtainSdf: SplatEditSdf }).curtainSdf;
    expect(sdf.invert).toBe(true);
    expect(sdf.opacity).toBe(0);
    expect(sdf.type).toBe(SplatEditSdfType.SPHERE);
    expect(sdf.radius).toBe(5);
    expect(sdf.position.toArray()).toEqual([1, 2, 3]);
    // No SplatMesh ancestor: the edit is scene-global, in walker frame.
    let node: THREE.Object3D | null = sdf;
    let underSplatMesh = false;
    while (node) {
      if (node instanceof SplatMesh) underSplatMesh = true;
      node = node.parent;
    }
    expect(underSplatMesh).toBe(false);
  });

  it("mutates IN PLACE (identity preserved), box shape, disable, round-trip", () => {
    const w = curtainRig();
    w.setCurtain(params);
    const internals = w as unknown as { curtainEdit: SplatEdit; curtainSdf: SplatEditSdf };
    const edit1 = internals.curtainEdit;
    const sdf1 = internals.curtainSdf;
    w.setCurtain({ ...params, shape: "box", halfExtents: [2, 3, 4], softEdge: 0.5 });
    expect(internals.curtainEdit).toBe(edit1); // the no-rebuild contract
    expect(internals.curtainSdf).toBe(sdf1);
    const st = w.curtainState();
    expect(st).toMatchObject({
      enabled: true, shape: "box", halfExtents: [2, 3, 4],
    });
    expect(st!.softEdge).toBeCloseTo(0.5, 6);
    w.setCurtain(null);
    expect(w.curtainState()!.enabled).toBe(false); // visible flipped, kept
    w.setCurtain(params);
    expect(w.curtainState()!.enabled).toBe(true);
    expect(w.curtainState()!.shape).toBe("sphere");
  });
});

describe("photograph-first visibility (triage lane)", () => {
  function visRig(opts: { backdrop?: boolean; restyleShowsMesh?: boolean }) {
    const mk = (slug: string, role: string, provenance: string | null) => ({
      slug, role, provenance, visible: true, collides: false,
      object: new THREE.Group(),
    });
    const els = {
      shell: mk("shell", "shell", null),
      capturedProp: mk("bicycle", "prop", null),
      capturedStatic: mk("building", "static", null),
      authored: mk("torch", "environment", "authored"),
      generated: mk("generated-box", "environment", "generated"),
      lamp: mk("lamp-post", "prop", null),
    };
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, {
      elements: Object.values(els),
      backdrop: opts.backdrop === false ? null : { visible: true },
      restyleShowsMesh: opts.restyleShowsMesh ?? false,
      restyle: null,
      pluckedSlugs: new Set(),
      interactions: new Map([["lamp-post", { slug: "lamp-post", verb: "toggle" }]]),
      visibilityOverrides: new Map(),
      replacedSlugs: new Set(),
      pluckRows: new Map(), pluckNRows: 0, pluckMask: null,
      pluckClock: 0, pluckWarned: false,
    });
    return { walker, els };
  }

  it("switches native appearance and mesh without hiding semantics or rebuilding collision", () => {
    const { walker, els } = visRig({});
    const native = new THREE.Group();
    Object.assign(els.generated, { gaussianAppearance: native });
    const rebuild = vi.spyOn(walker, "rebuildCollider");
    walker.refreshElementVisibility();
    expect(native.visible).toBe(true);
    expect(els.generated.visible).toBe(true);
    expect(els.generated.object.visible).toBe(false);
    walker.setGeneratedSplats(false);
    expect(native.visible).toBe(false);
    expect(els.generated.object.visible).toBe(true);
    walker.setElementVisible(els.generated.slug, false);
    walker.setGeneratedSplats(true);
    expect(native.visible).toBe(false);
    expect(els.generated.object.visible).toBe(false);
    walker.setElementVisible(els.generated.slug, true);
    expect(native.visible).toBe(true);
    expect(els.generated.object.visible).toBe(false);
    expect(rebuild).not.toHaveBeenCalled();
    expect(native.matrix.equals(new THREE.Matrix4())).toBe(true);
  });

  it("uses the mesh for baked looks and restores the native layer when cleared", () => {
    const { walker, els } = visRig({ backdrop: false });
    const native = new THREE.Group();
    Object.assign(els.generated, { gaussianAppearance: native });
    walker.setBakedLook(true);
    expect(native.visible).toBe(false);
    expect(els.generated.object.visible).toBe(true);
    walker.setBakedLook(false);
    expect(native.visible).toBe(true);
    expect(els.generated.object.visible).toBe(false);
  });

  it("disposes the native layer and invalidates unfinished loads when a revision clears", () => {
    const { walker, els } = visRig({});
    const native = new THREE.Group();
    const dispose = vi.fn();
    Object.assign(native, { dispose });
    Object.assign(els.generated, { gaussianAppearance: native });
    const scene = new THREE.Scene();
    const worldGroup = new THREE.Group();
    scene.add(native, worldGroup);
    for (const element of Object.values(els)) worldGroup.add(element.object);
    Object.assign(walker, { scene, worldGroup, disposeCollider: vi.fn(), worldLoadEpoch: 7 });
    const internal = walker as unknown as { clearWorld(): void; worldLoadEpoch: number };
    internal.clearWorld();
    expect(dispose).toHaveBeenCalledOnce();
    expect(native.parent).toBeNull();
    expect(worldGroup.children).toHaveLength(0);
    expect(walker.elements).toHaveLength(0);
    expect(internal.worldLoadEpoch).toBe(8);
    internal.clearWorld();
    expect(dispose).toHaveBeenCalledOnce();
  });

  it("photograph showing: captured meshes hide; authored/interactive stay", () => {
    const { walker, els } = visRig({});
    walker.refreshElementVisibility();
    expect(els.shell.visible).toBe(false);
    expect(els.capturedProp.visible).toBe(false);
    expect(els.capturedStatic.visible).toBe(false);
    expect(els.authored.visible).toBe(true);      // mesh is all it has
    expect(els.generated.visible).toBe(true);
    expect(els.lamp.visible).toBe(true);          // a toggle earns its mesh
  });

  it("a PLUCKED prop shows its mesh — the photograph ghost is gone", () => {
    const { walker, els } = visRig({});
    (walker as unknown as { pluckedSlugs: Set<string> })
      .pluckedSlugs.add("bicycle");
    walker.refreshElementVisibility();
    expect(els.capturedProp.visible).toBe(true);
    expect(els.capturedStatic.visible).toBe(false);
  });

  it("eye-toggles override the computed default in both directions", () => {
    const { walker, els } = visRig({});
    walker.setElementVisible("bicycle", true);
    walker.setElementVisible("torch", false);
    walker.refreshElementVisibility();
    expect(els.capturedProp.visible).toBe(true);
    expect(els.authored.visible).toBe(false);
  });

  it("no backdrop, or a restyle showing the mesh world: everything visible", () => {
    const a = visRig({ backdrop: false });
    a.walker.refreshElementVisibility();
    expect(Object.values(a.els).every((e) => e.visible)).toBe(true);
    const b = visRig({ restyleShowsMesh: true });
    b.walker.refreshElementVisibility();
    expect(Object.values(b.els).every((e) => e.visible)).toBe(true);
  });

  it("a restyled element earns its mesh while the photograph shows", () => {
    const { walker, els } = visRig({});
    Object.assign(walker, {
      restyle: { v: 1, job_id: "j", elements: { building: { tint: "#ff0000" } },
                 lighting: { preset: "as-captured", intensity: 1 } },
    });
    walker.refreshElementVisibility();
    expect(els.capturedStatic.visible).toBe(true);
    expect(els.capturedProp.visible).toBe(false);
  });
});

describe("replace-with-asset (triage lane slice 2)", () => {
  it("a REPLACED captured slug never defaults visible — even earned", () => {
    const mk = (slug: string, provenance: string | null) => ({
      slug, role: "prop", provenance, visible: true, collides: false,
      object: new THREE.Group(),
    });
    const captured = mk("umbrella", null);
    const replacement = mk("re-umbrella", "authored");
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, {
      elements: [captured, replacement],
      backdrop: { visible: true }, restyleShowsMesh: false, restyle: null,
      pluckedSlugs: new Set(["umbrella"]),           // plucked would earn it…
      interactions: new Map([["umbrella", { slug: "umbrella", verb: "toggle" }]]),
      visibilityOverrides: new Map(),
      replacedSlugs: new Set(["umbrella"]),          // …but replaced wins
      pluckRows: new Map(), pluckNRows: 0, pluckMask: null,
      pluckClock: 0, pluckWarned: false,
    });
    walker.refreshElementVisibility();
    expect(captured.visible).toBe(false);
    expect(replacement.visible).toBe(true);
  });

  it("checkPluckDisturbed plucks a replaced slug's ghost without physics", () => {
    const N = 6;
    const backdrop = {
      packedSplats: { numSplats: N },
      worldModifier: undefined as unknown,
      updateGenerator() { /* counted elsewhere */ },
    };
    const walker = Object.create(WorldWalker.prototype) as WorldWalker;
    Object.assign(walker, {
      backdrop, elements: [], physics: null,
      pluckRows: new Map([["umbrella", [1, 2]]]), pluckNRows: N,
      pluckedSlugs: new Set(), pluckMask: null,
      pluckClock: 0, pluckWarned: false,
      replacedSlugs: new Set(["umbrella"]),
      visibilityOverrides: new Map(),
      restyleShowsMesh: false, restyle: null,
      interactions: new Map(),
      scene: new THREE.Scene(),
    });
    (walker as unknown as { checkPluckDisturbed(): void }).checkPluckDisturbed();
    expect(walker.pluckState().umbrella.plucked).toBe(true);
  });
});
