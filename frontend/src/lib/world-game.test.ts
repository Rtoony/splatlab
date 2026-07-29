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
    game.shoot(); // kill -> last wave clears -> won once the corpse is gone
    step(3);
    expect(hud().phase).toBe("won");
    expect(zombieGroups(scene).length).toBe(0); // corpse finished dying post-win
  });

  it("a corpse caught by the endgame keeps falling and fading, never freezes", () => {
    // Two slow zombies, a one-bite-fragile player: shoot the leader as it
    // closes, then be overrun while its corpse is still falling.
    const { scene, camera, game, hud } = rig(scenario({
      player: { health: 10, weapon: { damage: 100, range_m: 30, cooldown_s: 0.05 } },
      waves: [{ actors: [{ type: "zombie", count: 2, health: 50, speed_mps: 1,
                           damage: 10, reach_m: 1.2, attack_cooldown_s: 0.6 }] }],
      spawn: { policy: "far-walkable", min_distance_m: 3 },
    }));
    game.start();
    let victim: THREE.Group | undefined;
    for (let t = 0; t < 12 && hud().phase !== "lost"; t += 1 / 60) {
      game.update(1 / 60);
      if (!victim) {
        const near = zombieGroups(scene)
          .filter((z) => z.rotation.x >= 0 && z.position.y >= 0)
          .sort((a, b) => a.position.distanceTo(camera.position)
            - b.position.distanceTo(camera.position))[0];
        if (near && near.position.distanceTo(camera.position) < 2.2) {
          camera.lookAt(near.position.clone().setY(near.position.y + 0.8));
          camera.updateMatrixWorld(true);
          game.shoot();
          game.update(1 / 60);
          victim = zombieGroups(scene).find((z) => z.rotation.x < -1e-6);
        }
      }
    }
    expect(hud().phase).toBe("lost");
    expect(victim).toBeTruthy();
    const fallen = victim!.rotation.x;
    const mat = (victim!.children.find((c) => (c as THREE.Mesh).isMesh) as THREE.Mesh)
      .material as THREE.MeshStandardMaterial;
    for (let t = 0; t < 0.25; t += 1 / 60) game.update(1 / 60);
    expect(victim!.rotation.x).toBeLessThan(fallen - 0.05); // still falling
    expect(mat.opacity).toBeLessThan(1); // still fading
    for (let t = 0; t < 2; t += 1 / 60) game.update(1 / 60);
    // The corpse leaves once faded; the surviving zombie stays standing.
    expect(zombieGroups(scene)).not.toContain(victim);
    expect(zombieGroups(scene).length).toBe(1);
  });
});

describe("combat polish (W2-B, capped)", () => {
  it("a landed shot staggers the zombie — no steps, and it flashes NOW", () => {
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
    const torso = zombie.children.find(
      (c) => (c as THREE.Mesh).isMesh
        && ((c as THREE.Mesh).material as THREE.MeshStandardMaterial).emissive !== undefined
        && (c as THREE.Mesh).geometry.type === "CapsuleGeometry") as THREE.Mesh;
    step(0.1); // inside the stagger window
    expect(zombie.position.distanceTo(frozen)).toBeLessThan(1e-6);
    // The hit flash must burn DURING the stagger, not 0.35 s later when the
    // shamble resumes (review finding: feedback fired after the flinch).
    expect((torso.material as THREE.MeshStandardMaterial).emissive.getHex())
      .not.toBe(0x000000);
    step(1.0); // stagger over: it moves again
    expect(zombie.position.distanceTo(frozen)).toBeGreaterThan(0.05);
    expect((torso.material as THREE.MeshStandardMaterial).emissive.getHex())
      .toBe(0x000000);
  });

  it("a staggered zombie in reach cannot bite", () => {
    // The old assertion was vacuous: spawns are metres away, so health stayed
    // 100 whatever the guard did. Put one in the player's face instead.
    const { scene, camera, game, step, hud } = rig(scenario({
      player: { health: 100, weapon: { damage: 10, range_m: 30, cooldown_s: 0.05 } },
      waves: [{ actors: [{ type: "zombie", count: 1, health: 1000, speed_mps: 3,
                           damage: 5, reach_m: 2.5, attack_cooldown_s: 1.0 }] }],
      spawn: { policy: "far-walkable", min_distance_m: 2 },
    }));
    game.start();
    step(4); // it arrives and gets a couple of bites in
    expect(hud().health).toBeLessThan(100);
    const zombie = zombieGroups(scene)[0];
    expect(zombie.position.distanceTo(camera.position)).toBeLessThan(2.5);
    camera.lookAt(zombie.position.clone().setY(0.8));
    camera.updateMatrixWorld(true);
    expect(game.shoot()).toBe(true);
    const health = hud().health;
    step(0.3); // the whole stagger, point blank
    expect(hud().health).toBe(health); // no bites landed while flinching
  });

  it("zombies rise from the ground and do not travel until surfaced", () => {
    const { scene, game, step } = rig(scenario({
      player: { health: 100, weapon: { damage: 10, range_m: 30, cooldown_s: 0.05 } },
    }));
    game.start();
    step(1.6); // just past rest: spawn happened, rise in progress
    const zombie = zombieGroups(scene)[0];
    expect(zombie.position.y).toBeLessThan(0); // still surfacing
    // "Neither walks" pinned properly: XZ must not drift while rising.
    const xz = new THREE.Vector2(zombie.position.x, zombie.position.z);
    step(0.3);
    expect(zombie.position.y).toBeLessThan(0);
    expect(new THREE.Vector2(zombie.position.x, zombie.position.z).distanceTo(xz))
      .toBeLessThan(1e-6);
    step(1.5);
    expect(zombie.position.y).toBeGreaterThanOrEqual(0); // surfaced
    expect(new THREE.Vector2(zombie.position.x, zombie.position.z).distanceTo(xz))
      .toBeGreaterThan(0.05); // and then it walks
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
    // A built zombie is 1.52 x scale tall (head top), and scale carries the
    // jitter — so the honest band is 1.52 x [0.88, 1.12], not a loose guess
    // around 1.6 that a +/-20% regression would still satisfy.
    const nominal = 1.52 * (1.6 / 1.6);
    for (const h of heights) {
      expect(h).toBeGreaterThanOrEqual(nominal * 0.88 - 1e-6);
      expect(h).toBeLessThanOrEqual(nominal * 1.12 + 1e-6);
    }
  });
});

describe("outdoor terrain (measured live on the Stump)", () => {
  // The fake probe obeys walker.surfaceNear's REAL contract: nearest face to
  // the reference, and null when nothing is inside the band. A mock that
  // ignored the band would pin behaviour the walker can never produce.
  function terrainProbe(height: (x: number, z: number) => number) {
    const asked: { ref: number; band: number }[] = [];
    const probe = (x: number, z: number, ref: number, band: number) => {
      asked.push({ ref, band });
      const y = height(x, z);
      return Math.abs(y - ref) <= band ? y : null;
    };
    return { probe, asked };
  }

  function terrainRig(height: (x: number, z: number) => number) {
    const { probe, asked } = terrainProbe(height);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1);
    camera.position.set(5, 1.7 + height(5, 5), 5);
    camera.updateMatrixWorld(true);
    const game = new WorldGame({
      scene, camera, navmeshDoc: OPEN_ROOM, scenario: scenario(),
      unitsPerMetre: 1, seed: 7, groundAt: probe,
    });
    game.start();
    for (let t = 0; t < 4; t += 1 / 60) game.update(1 / 60);
    return { scene, game, asked };
  }

  it("zombies stand on the probed ground, not the navmesh's flat floor_y", () => {
    // Terrain undulating within the band around floor_y=0 — a real grade.
    const height = (x: number, _z: number) => 0.6 + 0.05 * x;
    const { scene, game, asked } = terrainRig(height);
    const zombies = zombieGroups(scene);
    expect(zombies.length).toBeGreaterThan(0);
    for (const z of zombies) {
      expect(Math.abs(z.position.y - height(z.position.x, z.position.z)))
        .toBeLessThan(0.35);
      expect(z.position.y).toBeGreaterThan(0.3); // not snapped to floor_y=0
    }
    // Asked around the graded floor, with a body-height band (1.6 m at 1 u/m):
    // wide enough for terrain, narrow enough to exclude a canopy overhead.
    expect(asked.length).toBeGreaterThan(0);
    for (const a of asked) {
      expect(a.ref).toBeCloseTo(0, 5);
      expect(a.band).toBeCloseTo(1.6, 5);
    }
    game.dispose();
  });

  it("ground outside the band leaves actors on the graded floor, not adrift", () => {
    // A mezzanine 8 units up is NOT this navmesh's storey: the probe returns
    // null and the actor keeps floor_y — the documented single-storey limit.
    const { scene, game } = terrainRig(() => 8);
    for (const z of zombieGroups(scene)) {
      expect(z.position.y).toBeCloseTo(0, 1);
    }
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
    // ...and back ON when the house light goes away (the walker's "unlit"
    // toggle does exactly this mid-game).
    scene.remove(house);
    step(0.3);
    expect(gameLights.some(lit)).toBe(true);
    game.stop();
    expect(gameLights.every((l) => !lit(l))).toBe(true);
  });

  it("the no-spawn bailout puts the actor light out too", () => {
    // update() early-returns on idle, so a light left on there never gets
    // reconciled again (review finding).
    const island = {
      cell: 0.5, origin: [0, 0], shape: [3, 3], floor_y: 0,
      rows: ["100", "000", "001"],
    };
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 1);
    camera.position.set(0.25, 1.7, 0.25);
    camera.updateMatrixWorld(true);
    const game = new WorldGame({
      scene, camera, navmeshDoc: island, scenario: scenario(),
      unitsPerMetre: 1, seed: 3,
    });
    game.start();
    for (let t = 0; t < 5; t += 1 / 60) game.update(1 / 60);
    const anyLit: boolean[] = [];
    scene.traverse((o) => {
      if (!(o as THREE.Light).isLight) return;
      let visible = true;
      for (let p: THREE.Object3D | null = o; p; p = p.parent) if (!p.visible) visible = false;
      anyLit.push(visible);
    });
    expect(anyLit.some(Boolean)).toBe(false);
    game.dispose();
  });
});
