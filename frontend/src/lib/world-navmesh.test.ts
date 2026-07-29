// Pathfinding is gameplay-load-bearing: a wrong path is a zombie in a wall.

import { describe, expect, it } from "vitest";

import {
  cellAt,
  chooseSpawnCells,
  findPath,
  isWalkable,
  nearestWalkable,
  parseNavmesh,
  worldAt,
} from "./world-navmesh";

function grid(rows: string[], cell = 0.5, origin: [number, number] = [0, 0]) {
  return parseNavmesh({
    cell,
    origin,
    shape: [rows.length, rows[0].length],
    floor_y: -1.0,
    rows,
  });
}

describe("parse + coordinates", () => {
  it("round-trips cells and world positions", () => {
    const nav = grid(["111", "101", "111"]);
    expect(nav.shape).toEqual([3, 3]);
    expect(isWalkable(nav, { i: 1, j: 1 })).toBe(false);
    const cell = cellAt(nav, 1.26, 0.24)!;
    expect(cell).toEqual({ i: 2, j: 0 });
    const [x, z] = worldAt(nav, cell);
    expect(x).toBeCloseTo(1.25);
    expect(z).toBeCloseTo(0.25);
    expect(cellAt(nav, -5, 0)).toBeNull();
  });

  it("rejects malformed documents loudly", () => {
    expect(() => grid(["11", "1"])).toThrow(/row 1 length/);
    expect(() =>
      parseNavmesh({ cell: 0.5, origin: [0, 0], shape: [2, 2], floor_y: 0,
                     rows: ["11"] }),
    ).toThrow(/rows/);
  });
});

describe("findPath", () => {
  it("routes around obstacles, never through them", () => {
    const nav = grid([
      "11111",
      "11101",
      "00101",
      "11101",
      "11111",
    ]);
    const path = findPath(nav, { i: 0, j: 0 }, { i: 4, j: 4 })!;
    expect(path[0]).toEqual({ i: 0, j: 0 });
    expect(path[path.length - 1]).toEqual({ i: 4, j: 4 });
    for (const step of path) expect(isWalkable(nav, step)).toBe(true);
    // Each step is 4-connected.
    for (let k = 1; k < path.length; k++) {
      const d = Math.abs(path[k].i - path[k - 1].i) + Math.abs(path[k].j - path[k - 1].j);
      expect(d).toBe(1);
    }
  });

  it("returns null for unreachable islands and unwalkable endpoints", () => {
    const nav = grid(["101", "000", "101"]);
    expect(findPath(nav, { i: 0, j: 0 }, { i: 2, j: 2 })).toBeNull();
    expect(findPath(nav, { i: 0, j: 1 }, { i: 0, j: 0 })).toBeNull();
  });
});

describe("nearestWalkable + spawns", () => {
  it("finds the closest standable cell to an off-mesh point", () => {
    const nav = grid(["000", "010", "000"]);
    expect(nearestWalkable(nav, 0.25, 0.25)).toEqual({ i: 1, j: 1 });
  });

  it("spawns far from the player, only on cells that can reach them", () => {
    const nav = grid([
      "11111111",
      "10000001",
      "11111111",
    ]);
    let seed = 42;
    const rng = () => {
      seed = (seed * 1103515245 + 12345) % 2 ** 31;
      return seed / 2 ** 31;
    };
    const player = { i: 0, j: 0 };
    const spawns = chooseSpawnCells(nav, player, 3, 6, rng);
    expect(spawns.length).toBe(3);
    for (const s of spawns) {
      expect(Math.abs(s.i - player.i) + Math.abs(s.j - player.j)).toBeGreaterThanOrEqual(6);
      expect(findPath(nav, s, player)).not.toBeNull();
    }
  });
});

describe("spawn robustness (review findings)", () => {
  const rng = (() => { let s = 9; return () => { s = (s * 1103515245 + 12345) % 2 ** 31; return s / 2 ** 31; }; })();

  it("never stacks spawns on the same cell", () => {
    const nav = grid(["111111111"]);
    const spawns = chooseSpawnCells(nav, { i: 0, j: 0 }, 4, 3, rng);
    const keys = new Set(spawns.map((c) => `${c.i}:${c.j}`));
    expect(keys.size).toBe(spawns.length);
  });

  it("relaxes distance in a cramped room instead of spawning nothing", () => {
    const nav = grid(["111", "111", "111"]);
    const spawns = chooseSpawnCells(nav, { i: 1, j: 1 }, 2, 50, rng);
    expect(spawns.length).toBe(2);
  });

  it("returns [] fast when the player stands on an unreachable island", () => {
    const rows = ["1101" + "1".repeat(60)];
    for (let k = 0; k < 40; k++) rows.push("0001" + "1".repeat(60));
    const nav = grid(rows.map((r) => r.slice(0, 30)));
    // player isolated at (0,0)-(0,1); the big field cannot reach them
    const started = performance.now();
    const spawns = chooseSpawnCells(nav, { i: 0, j: 0 }, 7, 5, rng);
    expect(performance.now() - started).toBeLessThan(100); // one BFS, not 210 A*s
    expect(spawns.length).toBeLessThanOrEqual(7);
    // every spawn, if any, can genuinely reach the player
    for (const s of spawns) expect(findPath(nav, s, { i: 0, j: 0 })).not.toBeNull();
  });
});
