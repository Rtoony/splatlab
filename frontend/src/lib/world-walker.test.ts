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
import {
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatMesh,
} from "@sparkjsdev/spark";
import { WorldWalker, type CurtainParams } from "./world-walker";

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

  it("fast path merges AUTHORED environment/static into the shell BVH", () => {
    const env = mkEl("environment", "authored", 2);
    const authoredStatic = mkEl("static", "authored", 4);
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

  it("fallback path (no collision shell) merges shell+static+environment", () => {
    const els = [mkEl("shell", null, 0), mkEl("static", null, 1),
                 mkEl("environment", "authored", 2), mkEl("prop", null, 3)];
    const w = colliderRig({ shell: false, elements: els });
    w.rebuildCollider();
    expect(internals(w).colliderTris).toBe(36); // prop excluded
    expect(internals(w).colliderSource).toBe("visual_shell");
    expect(els[3].collides).toBe(false);
  });

  it("an authored platform becomes REAL floor the probes can stand on", () => {
    const env = mkEl("environment", "authored", 2); // top face at y=2.1
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

  it("photograph showing: captured meshes hide; authored/interactive stay", () => {
    const { walker, els } = visRig({});
    walker.refreshElementVisibility();
    expect(els.shell.visible).toBe(false);
    expect(els.capturedProp.visible).toBe(false);
    expect(els.capturedStatic.visible).toBe(false);
    expect(els.authored.visible).toBe(true);      // mesh is all it has
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
