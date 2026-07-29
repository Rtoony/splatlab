// The navmesh: world_shell's walkable-cell grid, loaded and pathed.
//
// Pure module (no three.js, no DOM) so every routine here is unit-tested:
// actors path over exactly the cells the world's acceptance gates graded as
// standable — the backend wrote them, this file only reads and searches.

export interface Navmesh {
  cell: number;
  origin: [number, number];
  shape: [number, number];
  floorY: number;
  /** walk[i * nz + j] — true when cell (i, j) is standable. */
  walk: Uint8Array;
}

export interface Cell {
  i: number;
  j: number;
}

export function parseNavmesh(doc: {
  cell: number;
  origin: number[];
  shape: number[];
  floor_y: number;
  rows: string[];
}): Navmesh {
  const [nx, nz] = doc.shape;
  if (!Number.isFinite(nx) || !Number.isFinite(nz) || nx < 1 || nz < 1) {
    throw new Error("navmesh shape is not usable");
  }
  if (doc.rows.length !== nx) {
    throw new Error(`navmesh rows (${doc.rows.length}) do not match shape nx (${nx})`);
  }
  const walk = new Uint8Array(nx * nz);
  for (let i = 0; i < nx; i++) {
    const row = doc.rows[i];
    if (row.length !== nz) {
      throw new Error(`navmesh row ${i} length ${row.length} != nz ${nz}`);
    }
    for (let j = 0; j < nz; j++) walk[i * nz + j] = row.charCodeAt(j) === 49 ? 1 : 0;
  }
  return {
    cell: doc.cell,
    origin: [doc.origin[0], doc.origin[1]],
    shape: [nx, nz],
    floorY: doc.floor_y,
    walk,
  };
}

export function cellAt(nav: Navmesh, x: number, z: number): Cell | null {
  const i = Math.floor((x - nav.origin[0]) / nav.cell);
  const j = Math.floor((z - nav.origin[1]) / nav.cell);
  if (i < 0 || j < 0 || i >= nav.shape[0] || j >= nav.shape[1]) return null;
  return { i, j };
}

export function worldAt(nav: Navmesh, cell: Cell): [number, number] {
  return [
    nav.origin[0] + (cell.i + 0.5) * nav.cell,
    nav.origin[1] + (cell.j + 0.5) * nav.cell,
  ];
}

export function isWalkable(nav: Navmesh, cell: Cell): boolean {
  return nav.walk[cell.i * nav.shape[1] + cell.j] === 1;
}

/** BFS ring search: the closest walkable cell to a world point. */
export function nearestWalkable(nav: Navmesh, x: number, z: number, maxRadius = 40): Cell | null {
  const start = cellAt(nav, x, z) ?? {
    i: Math.min(Math.max(0, Math.floor((x - nav.origin[0]) / nav.cell)), nav.shape[0] - 1),
    j: Math.min(Math.max(0, Math.floor((z - nav.origin[1]) / nav.cell)), nav.shape[1] - 1),
  };
  if (isWalkable(nav, start)) return start;
  for (let r = 1; r <= maxRadius; r++) {
    for (let di = -r; di <= r; di++) {
      for (const dj of Math.abs(di) === r ? range(-r, r) : [-r, r]) {
        const c = { i: start.i + di, j: start.j + dj };
        if (c.i < 0 || c.j < 0 || c.i >= nav.shape[0] || c.j >= nav.shape[1]) continue;
        if (isWalkable(nav, c)) return c;
      }
    }
  }
  return null;
}

function range(lo: number, hi: number): number[] {
  const out = [];
  for (let v = lo; v <= hi; v++) out.push(v);
  return out;
}

/**
 * A* over the 4-connected walkable grid. Returns cells from start to goal
 * inclusive, or null when unreachable. Budgeted: a search that expands more
 * than `maxExpand` nodes gives up (an actor with no path idles rather than
 * freezing the frame).
 */
export function findPath(
  nav: Navmesh,
  start: Cell,
  goal: Cell,
  maxExpand = 20000,
): Cell[] | null {
  const [nx, nz] = nav.shape;
  const index = (c: Cell) => c.i * nz + c.j;
  if (!isWalkable(nav, start) || !isWalkable(nav, goal)) return null;

  const g = new Float64Array(nx * nz).fill(Infinity);
  const from = new Int32Array(nx * nz).fill(-1);
  const closed = new Uint8Array(nx * nz);
  const h = (c: Cell) => Math.abs(c.i - goal.i) + Math.abs(c.j - goal.j);

  // Binary-heap-free open list: fine at grid sizes of ~30k cells with a
  // budget; keep it simple and measurable before optimizing.
  const open: Cell[] = [start];
  const f = new Float64Array(nx * nz).fill(Infinity);
  g[index(start)] = 0;
  f[index(start)] = h(start);
  let expanded = 0;

  while (open.length) {
    let bestAt = 0;
    for (let k = 1; k < open.length; k++) {
      if (f[index(open[k])] < f[index(open[bestAt])]) bestAt = k;
    }
    const current = open.splice(bestAt, 1)[0];
    const ci = index(current);
    if (closed[ci]) continue;
    closed[ci] = 1;
    if (current.i === goal.i && current.j === goal.j) {
      const path: Cell[] = [];
      let at = ci;
      while (at !== -1) {
        path.push({ i: Math.floor(at / nz), j: at % nz });
        at = from[at];
      }
      return path.reverse();
    }
    if (++expanded > maxExpand) return null;

    for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const next = { i: current.i + di, j: current.j + dj };
      if (next.i < 0 || next.j < 0 || next.i >= nx || next.j >= nz) continue;
      if (!isWalkable(nav, next)) continue;
      const ni = index(next);
      if (closed[ni]) continue;
      const tentative = g[ci] + 1;
      if (tentative < g[ni]) {
        g[ni] = tentative;
        f[ni] = tentative + h(next);
        from[ni] = ci;
        open.push(next);
      }
    }
  }
  return null;
}

/**
 * Spawn placement: walkable cells at least `minDistCells` from the player,
 * spread by rejection sampling over the walkable set (deterministic given
 * the provided rng). Cells that cannot REACH the player are rejected — a
 * zombie on an unreachable island is a wave that never ends.
 */
export function chooseSpawnCells(
  nav: Navmesh,
  playerCell: Cell,
  count: number,
  minDistCells: number,
  rng: () => number,
): Cell[] {
  const [nx, nz] = nav.shape;
  const candidates: Cell[] = [];
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      if (nav.walk[i * nz + j] !== 1) continue;
      const d = Math.abs(i - playerCell.i) + Math.abs(j - playerCell.j);
      if (d >= minDistCells) candidates.push({ i, j });
    }
  }
  const chosen: Cell[] = [];
  let guard = 0;
  while (chosen.length < count && candidates.length && guard++ < count * 30) {
    const pick = candidates[Math.floor(rng() * candidates.length) % candidates.length];
    if (findPath(nav, pick, playerCell) !== null) chosen.push(pick);
  }
  return chosen;
}
