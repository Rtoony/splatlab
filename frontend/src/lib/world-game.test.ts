// The whole game loop, simulated deterministically in node: three.js scene
// graph and raycasts work without a renderer, so spawn → chase → melee →
// shoot → wave-clear → win/lose are all REAL here, not mocked.

import { describe, expect, it } from "vitest";
import * as THREE from "three";

import { WorldGame, type GameHudState, type Scenario } from "./world-game";

const OPEN_ROOM = {
  cell: 0.5,
  origin: [0, 0],
  shape: [20, 20],
  floor_y: 0,
  rows: Array.from({ length: 20 }, () => "1".repeat(20)),
};

function scenario(overrides: Partial<Scenario> = {}): Scenario {
  return {
    name: "test waves",
    player: { health: 100, weapon: { damage: 100, range_m: 30, cooldown_s: 0.05 } },
    waves: [
      { actors: [{ type: "zombie", count: 2, health: 100, speed_mps: 2.0,
                   damage: 10, reach_m: 0.9, attack_cooldown_s: 0.5 }] },
      { actors: [{ type: "zombie", count: 1, health: 100, speed_mps: 2.0,
                   damage: 10, reach_m: 0.9, attack_cooldown_s: 0.5 }] },
    ],
    spawn: { policy: "far-walkable", min_distance_m: 3 },
    rest_between_waves_s: 0.5,
    ...overrides,
  };
}

function rig(s: Scenario = scenario()) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, 1);
  camera.position.set(5, 1.7, 5); // middle of the room
  camera.updateMatrixWorld(true);
  const game = new WorldGame({
    scene, camera, navmeshDoc: OPEN_ROOM, scenario: s, unitsPerMetre: 1, seed: 7,
  });
  let hud: GameHudState | null = null;
  game.onHud = (h) => { hud = h; };
  const step = (seconds: number) => {
    const dt = 1 / 60;
    for (let t = 0; t < seconds; t += dt) game.update(dt);
  };
  return { scene, camera, game, step, hud: () => hud! };
}

function zombieGroups(scene: THREE.Scene): THREE.Group[] {
  const out: THREE.Group[] = [];
  scene.traverse((o) => {
    if ((o.userData as { zombieIndex?: number }).zombieIndex !== undefined) {
      out.push(o as THREE.Group);
    }
  });
  return out;
}

describe("WorldGame", () => {
  it("spawns the wave after the rest and the zombies CHASE the player", () => {
    const { scene, camera, game, step, hud } = rig();
    game.start();
    expect(hud().phase).toBe("rest");
    step(2);
    expect(hud().phase).toBe("wave");
    const zombies = zombieGroups(scene);
    expect(zombies.length).toBe(2);
    const before = zombies.map((z) => z.position.distanceTo(camera.position));
    step(1.5);
    const after = zombies.map((z) => z.position.distanceTo(camera.position));
    for (let k = 0; k < zombies.length; k++) {
      expect(after[k]).toBeLessThan(before[k]);
    }
  });

  it("melee drains health and overruns the player; hits flash the callback", () => {
    const { game, step, hud } = rig(scenario({
      player: { health: 25, weapon: { damage: 100, range_m: 30, cooldown_s: 0.05 } },
    }));
    let hits = 0;
    game.onPlayerHit = () => { hits++; };
    game.start();
    step(20); // let them reach and chew
    expect(hud().phase).toBe("lost");
    expect(hits).toBeGreaterThanOrEqual(3); // 25 hp / 10 dmg
  });

  it("aimed shots kill, clear the wave, progress, and win", () => {
    const { scene, camera, game, step, hud } = rig();
    game.start();
    step(2); // wave 1 live
    for (let safety = 0; safety < 40 && hud().phase !== "won"; safety++) {
      const target = zombieGroups(scene).find((z) => z.visible);
      if (!target) { step(1); continue; }
      camera.lookAt(target.position.clone().setY(target.position.y + 0.8));
      camera.updateMatrixWorld(true);
      expect(game.shoot()).toBe(true); // click consumed by combat
      step(0.2); // cooldown + death animation progress
    }
    step(3); // final wave-clear bookkeeping + rest into next / win
    expect(hud().phase).toBe("won");
    expect(hud().kills).toBe(3); // 2 + 1 across both waves
    expect(zombieGroups(scene).length).toBe(0); // corpses cleaned up
  });

  it("does not consume clicks when idle (carry-throw keeps working)", () => {
    const { game } = rig();
    expect(game.shoot()).toBe(false);
  });
});

describe("review-finding regressions", () => {
  it("an unspawnable scenario goes back to idle with a notice — never a fake win", () => {
    const island = {
      cell: 0.5, origin: [0, 0], shape: [3, 3], floor_y: 0,
      rows: ["100", "000", "001"], // player island + one unreachable cell
    };
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1);
    camera.position.set(0.25, 1.7, 0.25); // on the lone island
    camera.updateMatrixWorld(true);
    const game = new WorldGame({
      scene, camera, navmeshDoc: island, scenario: scenario(), unitsPerMetre: 1, seed: 3,
    });
    let hud: GameHudState | null = null;
    let notice = "";
    game.onHud = (h) => { hud = h; };
    game.onNotice = (m) => { notice = m; };
    game.start();
    for (let t = 0; t < 10; t += 1 / 60) game.update(1 / 60);
    expect(hud!.phase).toBe("idle");
    expect(hud!.phase).not.toBe("won");
    expect(notice).toMatch(/could not spawn/);
    game.dispose();
  });

  it("keeps fading corpses and the tracer after the game ends", () => {
    const { scene, camera, game, step, hud } = rig(scenario({
      waves: [{ actors: [{ type: "zombie", count: 1, health: 50, speed_mps: 0.5,
                           damage: 100, reach_m: 0.9, attack_cooldown_s: 5 }] }],
    }));
    game.start();
    step(3.5); // rest + spawn-rise complete: the zombie is surfaced and shootable
    const target = zombieGroups(scene)[0];
    camera.lookAt(target.position.clone().setY(0.8));
    camera.updateMatrixWorld(true);
    game.shoot(); // kill -> last wave clears -> won while the corpse falls
    step(3);
    expect(hud().phase).toBe("won");
    expect(zombieGroups(scene).length).toBe(0); // corpse finished dying post-win
  });
});

describe("combat polish (W2-B, capped)", () => {
  it("a landed shot staggers the zombie — no steps, no bites, briefly", () => {
    const { scene, camera, game, step } = rig(scenario({
      player: { health: 100, weapon: { damage: 10, range_m: 30, cooldown_s: 0.05 } },
      waves: [{ actors: [{ type: "zombie", count: 1, health: 1000, speed_mps: 0.8,
                           damage: 10, reach_m: 0.9, attack_cooldown_s: 0.5 }] }],
      spawn: { policy: "far-walkable", min_distance_m: 4 },
    }));
    game.start();
    step(2.6); // rest + rise complete, chase underway but far from arrival
    const zombie = zombieGroups(scene)[0];
    camera.lookAt(zombie.position.clone().setY(0.8));
    camera.updateMatrixWorld(true);
    expect(game.shoot()).toBe(true);
    const frozen = zombie.position.clone();
    step(0.2); // inside the stagger window
    expect(zombie.position.distanceTo(frozen)).toBeLessThan(1e-6);
    step(1.0); // stagger over: it moves again
    expect(zombie.position.distanceTo(frozen)).toBeGreaterThan(0.05);
  });

  it("zombies rise from the ground and neither walk nor bite until surfaced", () => {
    const { scene, game, step, hud } = rig(scenario({
      player: { health: 100, weapon: { damage: 10, range_m: 30, cooldown_s: 0.05 } },
    }));
    game.start();
    step(1.6); // just past rest: spawn happened, rise in progress
    const zombie = zombieGroups(scene)[0];
    expect(zombie.position.y).toBeLessThan(0); // still surfacing
    expect(hud().health).toBe(100);
    step(1.5);
    expect(zombie.position.y).toBeGreaterThanOrEqual(0); // surfaced
  });

  it("per-zombie jitter varies size within the ±12% bound", () => {
    const { scene, game, step } = rig();
    game.start();
    step(2.5);
    const heights = zombieGroups(scene).map((z) => {
      const box = new THREE.Box3().setFromObject(z);
      return box.max.y - box.min.y;
    });
    expect(new Set(heights.map((h) => h.toFixed(3))).size).toBeGreaterThan(1);
    for (const h of heights) {
      expect(h).toBeGreaterThan(1.6 * 0.8);
      expect(h).toBeLessThan(1.6 * 1.25);
    }
  });
});

describe("outdoor terrain (measured live on the Stump)", () => {
  it("zombies stand on the probed ground, not the navmesh's flat floor_y", () => {
    // The navmesh says floor_y=0, but the real terrain slopes to ~2 units up.
    const caps: number[] = [];
    const terrain = (x: number, _z: number, belowY?: number) => {
      if (belowY !== undefined) caps.push(belowY);
      return 2 + 0.1 * x;
    };
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1);
    camera.position.set(5, 1.7 + terrain(5, 5), 5);
    camera.updateMatrixWorld(true);
    const game = new WorldGame({
      scene, camera, navmeshDoc: OPEN_ROOM, scenario: scenario(),
      unitsPerMetre: 1, seed: 7, groundAt: terrain,
    });
    game.start();
    const dt = 1 / 60;
    for (let t = 0; t < 4; t += dt) game.update(dt);
    const zombies = zombieGroups(scene);
    expect(zombies.length).toBeGreaterThan(0);
    for (const z of zombies) {
      const want = terrain(z.position.x, z.position.z);
      expect(Math.abs(z.position.y - want)).toBeLessThan(0.35);
      expect(z.position.y).toBeGreaterThan(1); // nowhere near flat floor_y=0
    }
    // The probe must be capped just above the walkable band so a canopy
    // face can never be picked: floor_y (0) + 1 m at 1 unit/m.
    expect(caps.length).toBeGreaterThan(0);
    for (const c of caps) expect(c).toBeCloseTo(1.0, 5);
    game.dispose();
  });

  it("carries its own actor light in unlit worlds, defers to scene lights", () => {
    const { scene, game, step } = rig();
    // The renderer honors hierarchical visibility, so measure the effective
    // flag (self AND every ancestor), not the light's own bit.
    const lit = (o: THREE.Object3D): boolean => {
      for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) return false;
      return true;
    };
    const lights = () => {
      const out: THREE.Light[] = [];
      scene.traverse((o) => { if ((o as THREE.Light).isLight) out.push(o as THREE.Light); });
      return out;
    };
    game.start();
    step(0.3);
    expect(lights().some(lit)).toBe(true); // dark world: ours is on
    const house = new THREE.DirectionalLight(0xffffff, 1);
    scene.add(house);
    step(0.3);
    const gameLights = lights().filter((l) => l !== house);
    expect(gameLights.every((l) => !lit(l))).toBe(true); // theirs wins
    game.stop();
    expect(gameLights.every((l) => !lit(l))).toBe(true);
  });
});
