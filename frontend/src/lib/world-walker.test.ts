// The ground probe, against a REAL MeshBVH — three-mesh-bvh is pure JS and
// needs no renderer, so these are the actual raycasts the walker performs.
//
// The geometry mirrors what a solidified outdoor world really looks like
// (measured on the Stump): a thick terrain SLAB with an underside a metre
// below its walkable top, plus canopy sheets overhead. Both naive picks —
// first hit from the sky, lowest hit in the column — choose wrong here.

import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { MeshBVH } from "three-mesh-bvh";
import { WorldWalker } from "./world-walker";

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
