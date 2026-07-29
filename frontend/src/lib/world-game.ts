// Game mode: the P3 thin slice — waves of zombies in a captured world.
//
// Deliberately data-driven and thin: the SCENARIO document says what plays
// (validated backend-side; this module trusts its bounds), the NAVMESH says
// where feet may go (the gates graded those cells), and the walker keeps
// owning movement/rendering. This class only adds actors, a hitscan, health
// and a wave machine on top — all disposable, leaving the world untouched.
//
// Zombies are built from primitives on purpose (no asset pipeline in the
// hot path): a capsule torso + box head, arms out, sway-and-bob animation.
// Ugly-charming beats blocking the playable loop on a character pipeline.

import * as THREE from "three";

import {
  type Cell,
  type Navmesh,
  cellAt,
  chooseSpawnCells,
  findPath,
  nearestWalkable,
  parseNavmesh,
  worldAt,
} from "./world-navmesh";

export interface ScenarioActor {
  type: string;
  count: number;
  health: number;
  speed_mps: number;
  damage: number;
  reach_m: number;
  attack_cooldown_s: number;
}

export interface Scenario {
  name: string;
  player: { health: number; weapon: { damage: number; range_m: number; cooldown_s: number } };
  waves: { actors: ScenarioActor[] }[];
  spawn: { policy: string; min_distance_m: number };
  rest_between_waves_s: number;
}

export type GamePhase = "idle" | "wave" | "rest" | "won" | "lost";

export interface GameHudState {
  phase: GamePhase;
  wave: number;
  totalWaves: number;
  health: number;
  maxHealth: number;
  kills: number;
  alive: number;
  restSeconds: number;
}

interface ZombieActor {
  group: THREE.Group;
  parts: { torso: THREE.Mesh; head: THREE.Mesh; armL: THREE.Mesh; armR: THREE.Mesh };
  health: number;
  stats: ScenarioActor;
  path: Cell[];
  pathAt: number;
  repathClock: number;
  attackClock: number;
  hitFlash: number;
  dying: number; // seconds into the death fall; -1 = alive
  seed: number;
}

const REPATH_SECONDS = 0.5;
const DEATH_SECONDS = 0.9;
const ZOMBIE_HEIGHT_M = 1.6;

/** Deterministic tiny rng (mulberry32) so replays and tests agree. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildZombie(scale: number): ZombieActor["parts"] & { group: THREE.Group } {
  const skin = new THREE.MeshStandardMaterial({ color: 0x5c7a4a, roughness: 0.9 });
  const cloth = new THREE.MeshStandardMaterial({ color: 0x3d3a45, roughness: 1.0 });
  const torso = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.22 * scale, 0.62 * scale, 4, 8), cloth);
  torso.position.y = 0.75 * scale;
  const head = new THREE.Mesh(
    new THREE.BoxGeometry(0.3 * scale, 0.32 * scale, 0.3 * scale), skin);
  head.position.y = 1.36 * scale;
  const eye = new THREE.MeshBasicMaterial({ color: 0xd8ff5a });
  for (const side of [-1, 1]) {
    const dot = new THREE.Mesh(
      new THREE.BoxGeometry(0.05 * scale, 0.05 * scale, 0.02 * scale), eye);
    dot.position.set(0.07 * side * scale, 0.03 * scale, 0.16 * scale);
    head.add(dot);
  }
  const armGeom = new THREE.BoxGeometry(0.09 * scale, 0.09 * scale, 0.5 * scale);
  const armL = new THREE.Mesh(armGeom, skin);
  const armR = new THREE.Mesh(armGeom, skin);
  armL.position.set(-0.3 * scale, 1.05 * scale, 0.25 * scale);
  armR.position.set(0.3 * scale, 1.05 * scale, 0.25 * scale);
  const group = new THREE.Group();
  group.add(torso, head, armL, armR);
  return { group, torso, head, armL, armR };
}

export class WorldGame {
  onHud: ((state: GameHudState) => void) | null = null;
  onPlayerHit: (() => void) | null = null;

  private readonly scene: THREE.Scene;
  private readonly camera: THREE.Camera;
  private readonly nav: Navmesh;
  private readonly scenario: Scenario;
  private readonly unitsPerMetre: number;
  private readonly rng: () => number;
  private readonly root = new THREE.Group();
  private readonly zombies: ZombieActor[] = [];
  private readonly _ray = new THREE.Raycaster();
  private readonly _fwd = new THREE.Vector3();
  private readonly _pos = new THREE.Vector3();

  private phase: GamePhase = "idle";
  private waveIndex = -1;
  private health = 100;
  private kills = 0;
  private restClock = 0;
  private shotClock = 0;
  private hudClock = 0;
  private tracer: THREE.Line | null = null;
  private tracerTtl = 0;

  constructor(options: {
    scene: THREE.Scene;
    camera: THREE.Camera;
    navmeshDoc: Parameters<typeof parseNavmesh>[0];
    scenario: Scenario;
    unitsPerMetre: number;
    seed?: number;
  }) {
    this.scene = options.scene;
    this.camera = options.camera;
    this.nav = parseNavmesh(options.navmeshDoc);
    this.scenario = options.scenario;
    this.unitsPerMetre = Math.max(1e-6, options.unitsPerMetre);
    this.rng = makeRng(options.seed ?? 1337);
    this.health = options.scenario.player.health;
    this.scene.add(this.root);
  }

  get active(): boolean {
    return this.phase === "wave" || this.phase === "rest";
  }

  start(): void {
    if (this.active) return;
    this.clearActors();
    this.health = this.scenario.player.health;
    this.kills = 0;
    this.waveIndex = -1;
    this.beginRest(1.5);
    this.emitHud(true);
  }

  stop(): void {
    this.phase = "idle";
    this.clearActors();
    this.emitHud(true);
  }

  /** Left-click while active: hitscan. Returns true when the shot was handled. */
  shoot(): boolean {
    if (this.phase !== "wave" && this.phase !== "rest") return false;
    if (this.shotClock > 0) return true; // cooling down still swallows the click
    this.shotClock = this.scenario.player.weapon.cooldown_s;

    const range = this.scenario.player.weapon.range_m * this.unitsPerMetre;
    this._ray.far = range;
    this._ray.setFromCamera(new THREE.Vector2(0, 0), this.camera);
    const targets = this.zombies.filter((z) => z.dying < 0).map((z) => z.group);
    const hits = this._ray.intersectObjects(targets, true);
    this.showTracer(range, hits[0]?.point ?? null);
    if (!hits.length) return true;

    let node: THREE.Object3D | null = hits[0].object;
    while (node && !node.userData.zombieIndex && node.userData.zombieIndex !== 0) {
      node = node.parent;
    }
    const index = node?.userData.zombieIndex as number | undefined;
    const zombie = index !== undefined ? this.zombies[index] : undefined;
    if (!zombie || zombie.dying >= 0) return true;

    zombie.health -= this.scenario.player.weapon.damage;
    zombie.hitFlash = 0.15;
    if (zombie.health <= 0) {
      zombie.dying = 0;
      this.kills++;
    }
    this.emitHud(true);
    return true;
  }

  update(dt: number): void {
    if (this.phase === "idle" || this.phase === "won" || this.phase === "lost") return;
    this.shotClock = Math.max(0, this.shotClock - dt);
    if (this.tracer && (this.tracerTtl -= dt) <= 0) this.killTracer();

    if (this.phase === "rest") {
      this.restClock -= dt;
      if (this.restClock <= 0) this.beginWave();
    }

    const upm = this.unitsPerMetre;
    this.camera.getWorldPosition(this._pos);
    const playerCell = nearestWalkable(this.nav, this._pos.x, this._pos.z);

    let alive = 0;
    for (const zombie of this.zombies) {
      if (zombie.dying >= 0) {
        zombie.dying += dt;
        zombie.group.rotation.x = -Math.min(1, zombie.dying / DEATH_SECONDS) * (Math.PI / 2);
        zombie.group.traverse((o) => {
          const mesh = o as THREE.Mesh;
          const mat = mesh.material as THREE.Material | undefined;
          if (mat && "opacity" in mat) {
            mat.transparent = true;
            (mat as THREE.MeshStandardMaterial).opacity =
              Math.max(0, 1 - zombie.dying / DEATH_SECONDS);
          }
        });
        continue;
      }
      alive++;

      // Chase: repath toward the player's cell on a slow clock.
      zombie.repathClock -= dt;
      if (zombie.repathClock <= 0 && playerCell) {
        zombie.repathClock = REPATH_SECONDS + this.rng() * 0.3;
        const from = nearestWalkable(this.nav, zombie.group.position.x, zombie.group.position.z);
        if (from) {
          const path = findPath(this.nav, from, playerCell, 8000);
          if (path && path.length > 1) {
            zombie.path = path;
            zombie.pathAt = 1;
          }
        }
      }

      // Follow the path; face the walking direction.
      const speed = zombie.stats.speed_mps * upm;
      if (zombie.pathAt < zombie.path.length) {
        const [tx, tz] = worldAt(this.nav, zombie.path[zombie.pathAt]);
        const dx = tx - zombie.group.position.x;
        const dz = tz - zombie.group.position.z;
        const dist = Math.hypot(dx, dz);
        if (dist < this.nav.cell * 0.4) {
          zombie.pathAt++;
        } else {
          zombie.group.position.x += (dx / dist) * speed * dt;
          zombie.group.position.z += (dz / dist) * speed * dt;
          zombie.group.rotation.y = Math.atan2(dx, dz);
        }
      }

      // Shamble animation + hit flash.
      zombie.seed += dt;
      const sway = Math.sin(zombie.seed * 6) * 0.06;
      zombie.group.position.y = this.nav.floorY + Math.abs(Math.sin(zombie.seed * 6)) * 0.03;
      zombie.group.rotation.z = sway * 0.4;
      zombie.parts.armL.rotation.x = -0.2 + sway;
      zombie.parts.armR.rotation.x = -0.2 - sway;
      const torsoMat = zombie.parts.torso.material as THREE.MeshStandardMaterial;
      if (zombie.hitFlash > 0) {
        zombie.hitFlash -= dt;
        torsoMat.emissive.setHex(0xff3333);
      } else {
        torsoMat.emissive.setHex(0x000000);
      }

      // Melee when in reach.
      zombie.attackClock -= dt;
      const reach = zombie.stats.reach_m * upm;
      const toPlayer = Math.hypot(
        this._pos.x - zombie.group.position.x,
        this._pos.z - zombie.group.position.z,
      );
      if (toPlayer <= reach && zombie.attackClock <= 0) {
        zombie.attackClock = zombie.stats.attack_cooldown_s;
        this.health -= zombie.stats.damage;
        this.onPlayerHit?.();
        this.emitHud(true);
        if (this.health <= 0) {
          this.phase = "lost";
          this.emitHud(true);
          return;
        }
      }
    }

    // Dead-and-faded actors leave the scene.
    for (let k = this.zombies.length - 1; k >= 0; k--) {
      if (this.zombies[k].dying >= DEATH_SECONDS) this.removeZombie(k);
    }

    if (this.phase === "wave" && alive === 0 && this.zombies.length === 0) {
      if (this.waveIndex + 1 >= this.scenario.waves.length) {
        this.phase = "won";
        this.emitHud(true);
      } else {
        this.beginRest(this.scenario.rest_between_waves_s);
      }
    }

    this.hudClock += dt;
    if (this.hudClock >= 0.25) this.emitHud(false);
  }

  dispose(): void {
    this.stop();
    this.scene.remove(this.root);
  }

  /* ---------------- internals ---------------- */

  private beginRest(seconds: number): void {
    this.phase = "rest";
    this.restClock = seconds;
    this.emitHud(true);
  }

  private beginWave(): void {
    this.waveIndex++;
    const wave = this.scenario.waves[this.waveIndex];
    if (!wave) {
      this.phase = "won";
      this.emitHud(true);
      return;
    }
    this.phase = "wave";
    this.camera.getWorldPosition(this._pos);
    const playerCell = nearestWalkable(this.nav, this._pos.x, this._pos.z)
      ?? { i: 0, j: 0 };
    const minDistCells = Math.max(
      2,
      Math.round((this.scenario.spawn.min_distance_m * this.unitsPerMetre) / this.nav.cell),
    );
    for (const stats of wave.actors) {
      const cells = chooseSpawnCells(this.nav, playerCell, stats.count, minDistCells, this.rng);
      for (const cell of cells) this.spawnZombie(stats, cell);
    }
    this.emitHud(true);
  }

  private spawnZombie(stats: ScenarioActor, cell: Cell): void {
    const scale = ZOMBIE_HEIGHT_M * this.unitsPerMetre / 1.6;
    const built = buildZombie(scale);
    const [x, z] = worldAt(this.nav, cell);
    built.group.position.set(x, this.nav.floorY, z);
    built.group.userData.zombieIndex = this.zombies.length;
    this.root.add(built.group);
    this.zombies.push({
      group: built.group,
      parts: built,
      health: stats.health,
      stats,
      path: [],
      pathAt: 0,
      repathClock: 0,
      attackClock: stats.attack_cooldown_s,
      hitFlash: 0,
      dying: -1,
      seed: this.rng() * 10,
    });
  }

  private removeZombie(index: number): void {
    const [zombie] = this.zombies.splice(index, 1);
    this.root.remove(zombie.group);
    zombie.group.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
    });
    // Reindex the survivors' hit lookups.
    this.zombies.forEach((z, at) => { z.group.userData.zombieIndex = at; });
  }

  private clearActors(): void {
    while (this.zombies.length) this.removeZombie(this.zombies.length - 1);
    this.killTracer();
  }

  private showTracer(range: number, hit: THREE.Vector3 | null): void {
    this.killTracer();
    this.camera.getWorldPosition(this._pos);
    this.camera.getWorldDirection(this._fwd);
    const start = this._pos.clone()
      .addScaledVector(this._fwd, 0.2)
      .add(new THREE.Vector3(0.06, -0.05, 0));
    const end = hit ?? this._pos.clone().addScaledVector(this._fwd, range);
    const geometry = new THREE.BufferGeometry().setFromPoints([start, end]);
    this.tracer = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0xffe9a8, transparent: true, opacity: 0.9 }),
    );
    this.root.add(this.tracer);
    this.tracerTtl = 0.07;
  }

  private killTracer(): void {
    if (!this.tracer) return;
    this.root.remove(this.tracer);
    this.tracer.geometry.dispose();
    (this.tracer.material as THREE.Material).dispose();
    this.tracer = null;
  }

  private emitHud(force: boolean): void {
    if (!force && this.hudClock < 0.25) return;
    this.hudClock = 0;
    this.onHud?.({
      phase: this.phase,
      wave: Math.max(0, this.waveIndex + (this.phase === "rest" ? 2 : 1)),
      totalWaves: this.scenario.waves.length,
      health: Math.max(0, Math.round(this.health)),
      maxHealth: this.scenario.player.health,
      kills: this.kills,
      alive: this.zombies.filter((z) => z.dying < 0).length,
      restSeconds: Math.max(0, Math.ceil(this.restClock)),
    });
  }
}
