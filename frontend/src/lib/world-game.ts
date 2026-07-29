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
  /** Seconds left of hit-stagger: no movement or attacks, a flinch instead. */
  stagger: number;
  /** Seconds into the spawn rise; >= RISE_SECONDS means fully surfaced. */
  rising: number;
  /** Smoothed terrain height under the actor (falls back to nav floor_y). */
  groundY: number;
  /** Latest ground sample; groundY eases toward it so slopes read as walking. */
  groundGoalY: number;
  /** Per-actor variety: scales size and speed (±12%), no new systems. */
  jitter: number;
}

const REPATH_SECONDS = 0.5;
const DEATH_SECONDS = 0.9;
const ZOMBIE_HEIGHT_M = 1.6;
const STAGGER_SECONDS = 0.35;
const RISE_SECONDS = 0.8;
/** How far from the navmesh's graded floor an actor's real ground may sit. */
const GROUND_BAND_M = 1.6;

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
  // Legs reach the group origin: without them the torso capsule ends 0.35
  // units up and every zombie looks like it hovers (measured on the Stump).
  const legGeom = new THREE.BoxGeometry(0.11 * scale, 0.42 * scale, 0.13 * scale);
  const legL = new THREE.Mesh(legGeom, cloth);
  const legR = new THREE.Mesh(legGeom, cloth);
  legL.position.set(-0.11 * scale, 0.21 * scale, 0);
  legR.position.set(0.11 * scale, 0.21 * scale, 0);
  const group = new THREE.Group();
  group.add(torso, head, armL, armR, legL, legR);
  return { group, torso, head, armL, armR };
}

export class WorldGame {
  onHud: ((state: GameHudState) => void) | null = null;
  onPlayerHit: (() => void) | null = null;
  /** Fired when a shot connects — the HUD flashes a hit-marker. */
  onHitMarker: (() => void) | null = null;
  /** Fired for player-facing problems (e.g. no spawnable ground). */
  onNotice: ((message: string) => void) | null = null;

  private readonly scene: THREE.Scene;
  private readonly camera: THREE.Camera;
  private readonly nav: Navmesh;
  private readonly scenario: Scenario;
  private readonly unitsPerMetre: number;
  private readonly rng: () => number;
  private readonly groundProbe:
    ((x: number, z: number, referenceY: number, band: number) => number | null) | null;
  private readonly root = new THREE.Group();
  private readonly gameLight = new THREE.Group();
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
    /** The collider surface at (x, z) nearest `referenceY` within `band`,
     *  or null. The navmesh's single flat floor_y floats actors over real
     *  outdoor ground (measured live on the Stump); this refines it.
     *  Contract per walker.surfaceNear — nearest wins, canopy excluded. */
    groundAt?: (x: number, z: number, referenceY: number, band: number) => number | null;
  }) {
    this.scene = options.scene;
    this.camera = options.camera;
    this.nav = parseNavmesh(options.navmeshDoc);
    this.scenario = options.scenario;
    this.unitsPerMetre = Math.max(1e-6, options.unitsPerMetre);
    this.rng = makeRng(options.seed ?? 1337);
    this.groundProbe = options.groundAt ?? null;
    this.health = options.scenario.player.health;
    // Unlit worlds (the faithful default — atlases bake their own light)
    // render standard-material actors pitch black. Carry our own light; it
    // only switches on when nothing else in the scene is lighting them.
    const hemi = new THREE.HemisphereLight(0xfff2df, 0x33402e, 2.2);
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(2, 5, 3);
    this.gameLight.add(hemi, key);
    this.gameLight.visible = false;
    this.root.add(this.gameLight);
    this.scene.add(this.root);
  }

  /** True when some OTHER visible light already reaches the actors. */
  private sceneHasLight(): boolean {
    let found = false;
    this.scene.traverse((o) => {
      if (found || !o.visible) return;
      if ((o as THREE.Light).isLight && !this.gameLight.children.includes(o as THREE.Light)) {
        let p = o.parent;
        let shaded = true;
        while (p) {
          if (!p.visible) { shaded = false; break; }
          p = p.parent;
        }
        if (shaded) found = true;
      }
    });
    return found;
  }

  private groundYAt(x: number, z: number): number {
    // Search a body-height band around the graded floor: wide enough for
    // real terrain undulation, narrow enough that overhead geometry is not
    // a candidate. No face in the band = keep the navmesh's floor.
    const band = GROUND_BAND_M * this.unitsPerMetre;
    return this.groundProbe?.(x, z, this.nav.floorY, band) ?? this.nav.floorY;
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
    this.gameLight.visible = !this.sceneHasLight();
    this.beginRest(1.5);
    this.emitHud(true);
  }

  stop(): void {
    this.phase = "idle";
    this.gameLight.visible = false;
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
    // The raycast must see where actors ARE, not where the last render left
    // their matrices — a shot fired before the next frame (or in a headless
    // test) would otherwise test against stale world transforms.
    for (const target of targets) target.updateMatrixWorld(true);
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
    zombie.stagger = STAGGER_SECONDS;
    this.onHitMarker?.();
    if (zombie.health <= 0) {
      zombie.dying = 0;
      this.kills++;
    }
    this.emitHud(true);
    return true;
  }

  update(dt: number): void {
    if (this.phase === "idle") return;
    // The endgame is not a freeze-frame: the final tracer still fades and
    // mid-fall corpses finish dying (review finding — they hung forever).
    if (this.tracer && (this.tracerTtl -= dt) <= 0) this.killTracer();
    if (this.phase === "won" || this.phase === "lost") {
      for (const zombie of this.zombies) {
        if (zombie.dying >= 0) this.advanceDeath(zombie, dt);
      }
      for (let k = this.zombies.length - 1; k >= 0; k--) {
        if (this.zombies[k].dying >= DEATH_SECONDS) this.removeZombie(k);
      }
      return;
    }
    this.shotClock = Math.max(0, this.shotClock - dt);

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
        this.advanceDeath(zombie, dt);
        continue;
      }
      alive++;

      // Spawn rise: surface from the ground before doing anything else — a
      // readable telegraph instead of zombies popping into existence.
      if (zombie.rising < RISE_SECONDS) {
        zombie.rising += dt;
        const t = Math.min(1, zombie.rising / RISE_SECONDS);
        const height = ZOMBIE_HEIGHT_M * this.unitsPerMetre * zombie.jitter;
        zombie.group.position.y = zombie.groundY - height * (1 - t * t);
        continue;
      }

      // Hit stagger: a landed shot interrupts — flinch, no steps, no bites.
      // The hit flash burns down HERE too: it used to be applied only in the
      // shamble section below, which stagger skips, so the red flash fired
      // 0.35 s AFTER the shot that caused it (review finding).
      if (zombie.stagger > 0) {
        zombie.stagger -= dt;
        zombie.group.rotation.x = -0.25 * (zombie.stagger / STAGGER_SECONDS);
        this.applyHitFlash(zombie, dt);
        continue;
      }
      zombie.group.rotation.x = 0;

      // Chase: repath toward the player's cell on a slow clock. The same
      // clock re-samples the terrain height so slopes are followed without
      // per-frame raycasts.
      zombie.repathClock -= dt;
      if (zombie.repathClock <= 0) {
        zombie.repathClock = REPATH_SECONDS + this.rng() * 0.3;
        zombie.groundGoalY = this.groundYAt(zombie.group.position.x, zombie.group.position.z);
        if (playerCell) {
          const from = nearestWalkable(this.nav, zombie.group.position.x, zombie.group.position.z);
          if (from) {
            const path = findPath(this.nav, from, playerCell, 8000);
            if (path && path.length > 1) {
              zombie.path = path;
              zombie.pathAt = 1;
            }
          }
        }
      }

      // Follow the path; face the walking direction.
      const speed = zombie.stats.speed_mps * upm * zombie.jitter;
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
      zombie.groundY += (zombie.groundGoalY - zombie.groundY) * Math.min(1, dt * 6);
      zombie.group.position.y = zombie.groundY + Math.abs(Math.sin(zombie.seed * 6)) * 0.03;
      zombie.group.rotation.z = sway * 0.4;
      zombie.parts.armL.rotation.x = -0.2 + sway;
      zombie.parts.armR.rotation.x = -0.2 - sway;
      this.applyHitFlash(zombie, dt);

      // Melee when in reach — in 3D: a zombie must not chew a player on a
      // balcony above it (review finding). The bite point is chest height.
      zombie.attackClock -= dt;
      const reach = zombie.stats.reach_m * upm;
      const chestY = zombie.group.position.y + 1.05 * (ZOMBIE_HEIGHT_M / 1.6) * upm;
      const toPlayer = Math.hypot(
        this._pos.x - zombie.group.position.x,
        this._pos.y - chestY,
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
    if (this.hudClock >= 0.25) {
      this.gameLight.visible = !this.sceneHasLight();
      this.emitHud(false);
    }
  }

  dispose(): void {
    this.stop();
    this.scene.remove(this.root);
  }

  /* ---------------- internals ---------------- */

  /** Burn down the hit flash. Called from both the stagger and shamble
   *  paths so the flash lands on the frame the shot did. */
  private applyHitFlash(zombie: ZombieActor, dt: number): void {
    const torsoMat = zombie.parts.torso.material as THREE.MeshStandardMaterial;
    if (zombie.hitFlash > 0) {
      zombie.hitFlash -= dt;
      torsoMat.emissive.setHex(0xff3333);
    } else {
      torsoMat.emissive.setHex(0x000000);
    }
  }

  /** One frame of dying: fall + fade. Shared so a corpse caught by the
   *  endgame finishes its animation instead of freezing and popping out
   *  (review finding — the endgame path advanced the clock only). */
  private advanceDeath(zombie: ZombieActor, dt: number): void {
    zombie.dying += dt;
    const t = Math.min(1, zombie.dying / DEATH_SECONDS);
    zombie.group.rotation.x = -t * (Math.PI / 2);
    zombie.group.traverse((o) => {
      const mat = (o as THREE.Mesh).material as THREE.Material | undefined;
      if (mat && "opacity" in mat) {
        mat.transparent = true;
        (mat as THREE.MeshStandardMaterial).opacity = Math.max(0, 1 - t);
      }
    });
  }

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
    if (!this.zombies.length) {
      // An empty wave must never quietly count as a survived one (review
      // finding: a cramped or islanded room fake-won in silence). Going idle
      // must also put the actor light out — update() early-returns on idle,
      // so nothing would ever reconcile it again (second review finding).
      this.phase = "idle";
      this.gameLight.visible = false;
      this.onNotice?.(
        "The scenario could not spawn any actors — no reachable spawn "
        + "ground from where you stand.");
      this.emitHud(true);
      return;
    }
    this.emitHud(true);
  }

  private spawnZombie(stats: ScenarioActor, cell: Cell): void {
    const jitter = 0.88 + this.rng() * 0.24; // ±12% size + speed variety
    const scale = (ZOMBIE_HEIGHT_M * this.unitsPerMetre / 1.6) * jitter;
    const built = buildZombie(scale);
    const [x, z] = worldAt(this.nav, cell);
    const groundY = this.groundYAt(x, z);
    // Born underground; the rise animation surfaces it.
    built.group.position.set(
      x, groundY - ZOMBIE_HEIGHT_M * this.unitsPerMetre * jitter, z);
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
      groundY,
      groundGoalY: groundY,
      stagger: 0,
      rising: 0,
      jitter,
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
