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
