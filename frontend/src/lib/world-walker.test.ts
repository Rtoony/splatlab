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
