// world-walker — the three.js engine behind the first-person walkable world
// viewer (pages/world-view.tsx). Framework-agnostic on purpose: React owns the
// HUD, this class owns the canvas, the GLB loading, the BVH collider and the
// capsule character controller.
//
// WHY A SEPARATE FILE: the page is already a dense HUD; mixing an rAF loop,
// a loader pipeline and a physics solver into it makes both unreadable. This
// module has no React imports and can be unit-tested / reused headlessly.
//
// COORDINATE FRAME (verified against splat_aea04ab3/_world, 2026-07-26 —
// probed every GLB's node graph directly, not assumed):
//   - every GLB's node transforms are IDENTITY (trimesh exporter),
//   - POSITION accessor min/max place each prop at its own distinct offset
//     inside the shell AABB (props are NOT centred at the origin),
//   - shell AABB min=[-4.316,-2.049,-4.656] max=[3.232,0.402,4.169],
//     Y extent 2.451 is the short axis and world.json says up_axis "Y".
//   => importing every GLB with no transform assembles the scene in place.
//      `verifySharedFrame()` below re-checks this at runtime and reports
//      anything that disagrees instead of silently rendering a pile at 0,0,0.
//
// TWO OTHER MEASURED FACTS THAT DRIVE CODE HERE:
//   - the GLBs carry POSITION + TEXCOORD_0 only, NO NORMAL attribute, and
//     three's GLTFLoader does not synthesise one. Lit materials would render
//     black, so we computeVertexNormals() on load.
//   - baseColorTexture is embedded as a bufferView (not an external URI), so
//     no sidecar atlas fetch is required. The loader still installs a URL
//     modifier in case a future exporter emits external image URIs.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { MeshBVH } from "three-mesh-bvh";
import {
  WorldPhysics,
  loadRapier,
  reoriginObject,
  type PropTransforms,
} from "./world-physics";
import {
  RgbaArray,
  SparkRenderer,
  SplatEdit,
  SplatEditRgbaBlendMode,
  SplatEditSdf,
  SplatEditSdfType,
  SplatFileType,
  SplatMesh,
} from "@sparkjsdev/spark";
import { buildPluckModifier, packChannelsRgba } from "./spark-heatmap";
import {
  classTile,
  presetFor,
  triplanarMaterial,
  type RestyleDoc,
  type RestyleMaterial,
} from "./world-restyle";
import {
  INTERACT_KEY,
  type InteractionRecord,
  type TargetInfo,
  effectFor,
  isStateful,
  maxReachInSceneUnits,
  nextState as nextInteractionState,
  reachInSceneUnits,
  targetInfo,
} from "./world-interactions";

/* ------------------------------------------------------------------ *
 * Manifest types — declared LOCALLY on purpose.                       *
 * contracts.ts is being edited concurrently by another agent; this    *
 * module must not depend on (or race) that file.                      *
 * ------------------------------------------------------------------ */

export type WorldRole = "prop" | "static" | "environment";

export interface WorldCollisionBlock {
  ok?: boolean;
  strategy?: string;
  hulls?: number;
  /** Bare hull filenames inside _world/collision/ (raw-manifest shape). */
  files?: string[];
  hull_faces_total?: number;
  capped?: boolean;
  surface_coverage?: number;
  coverage_ok?: boolean;
  seconds?: number;
}

export interface WorldEntry {
  slug: string;
  label?: string;
  role: WorldRole;
  /**
   * Bare filename ("shell.glb"), resolved through WORLD_FILE_URL_TEMPLATE.
   * The live backend instead advertises already-resolved URLs in `files`, and
   * omits keys whose artifact is missing — so `glb` may be absent entirely.
   * Both shapes are supported; `files.glb` wins when present.
   */
  glb?: string | null;
  files?: { glb?: string | null; atlas?: string | null; report?: string | null } | null;
  faces?: number;
  extent?: [number, number, number];
  collision?: WorldCollisionBlock;
  classification?: string[];
  /** "authored" for placed set-dressing (world_placed registry); absent for
   *  capture-derived elements. */
  provenance?: string | null;
}

export interface WorldManifest {
  v?: number;
  job_id?: string;
  units?: string;
  /** null == uncalibrated scene units. Drives the default scale dial. */
  meters_per_unit?: number | null;
  scene_extent?: [number, number, number];
  shell?: WorldEntry | null;
  /**
   * The WALKABLE solid — a different mesh from the one you look at.
   * The visual shell is the textured TSDF surface: accurate to the capture but
   * a thin fragmented web (Bonsai: 645 components, 67% floor continuity) that
   * a capsule falls straight through. This one is voxel-solidified and
   * watertight. Render the shell, collide against this.
   */
  collision_shell?: {
    glb?: string | null;
    report?: string | null;
    gates?: Record<string, number | boolean> | null;
  } | null;
  elements?: WorldEntry[];
  counts?: Record<string, number>;
}

/* ------------------------------------------------------------------ *
 * Where the files come from.                                          *
 * ------------------------------------------------------------------ */

/**
 * Default backend route for `_world/` files. Another agent owns the actual
 * FastAPI handler; this is the single constant to edit if the shape changes.
 * `{jobId}` and `{name}` are substituted; `{name}` is URL-encoded.
 */
export const WORLD_FILE_URL_TEMPLATE = "/api/splat/jobs/{jobId}/world/file?name={name}";

/**
 * A fetchable URL for an entry's GLB, tolerating both manifest shapes.
 *
 * The backend resolves URLs itself and only advertises artifacts that exist on
 * disk, so `files.glb` is authoritative when present. A static directory served
 * for development has no such block and carries a bare filename instead.
 * Returns null when the entry has neither, which the caller reports rather than
 * silently skipping.
 */
export function entryGlbUrl(
  entry: WorldEntry,
  dir: string,
  source: { fileUrl: (name: string) => string },
): string | null {
  const direct = entry.files?.glb;
  if (direct) return direct;
  if (entry.glb) return source.fileUrl(`${dir}${entry.glb}`);
  return null;
}

/** Resolves a path relative to a job's `_world/` dir into a fetchable URL. */
export interface WorldSource {
  readonly kind: "api" | "local";
  readonly describe: string;
  /** @param name e.g. "world_manifest.json" | "shell.glb" | "elements/red-bicycle.glb" */
  fileUrl(name: string): string;
}

/** Files served by the backend route (production path). */
export function apiWorldSource(jobId: string, template = WORLD_FILE_URL_TEMPLATE): WorldSource {
  return {
    kind: "api",
    describe: template.replace("{jobId}", jobId).replace("{name}", "…"),
    fileUrl: (name) =>
      template.replace("{jobId}", encodeURIComponent(jobId)).replace("{name}", encodeURIComponent(name)),
  };
}

/**
 * Files served from a plain static directory (development path, e.g. a
 * symlink under frontend/public/). Works today, with no backend route.
 */
export function localWorldSource(dirUrl: string): WorldSource {
  const base = dirUrl.endsWith("/") ? dirUrl : `${dirUrl}/`;
  return {
    kind: "local",
    describe: `${base}…`,
    fileUrl: (name) => base + name.split("/").map(encodeURIComponent).join("/"),
  };
}

/* ------------------------------------------------------------------ *
 * Player parameters — every human dimension is in METRES and scaled   *
 * into scene units by `unitsPerMetre`, because the capture is         *
 * uncalibrated and the only honest fix is a live dial.                *
 * ------------------------------------------------------------------ */

export interface WalkParams {
  /** THE scale dial: how many scene units equal one real metre. */
  unitsPerMetre: number;
  eyeHeightM: number;
  radiusM: number;
  walkSpeedMps: number;
  sprintMultiplier: number;
  gravityMps2: number;
  jumpSpeedMps: number;
  /** Include role="prop" elements in the collider (statics always collide). */
  collideProps: boolean;
  /**
   * Props as Rapier rigid bodies: bump, carry, throw. Mutually exclusive with
   * collideProps (a prop cannot be both baked-static and dynamic); when
   * physics engages it wins and collideProps is forced off.
   */
  physicsProps: boolean;
  /** Baked atlases already contain lighting; unlit is the faithful default. */
  unlit: boolean;
  showCollider: boolean;
  fovDeg: number;
  mouseSensitivity: number;
}

export const DEFAULT_WALK_PARAMS: WalkParams = {
  unitsPerMetre: 1,
  eyeHeightM: 1.7,
  radiusM: 0.32,
  walkSpeedMps: 3.4,
  sprintMultiplier: 2.4,
  gravityMps2: 9.81,
  jumpSpeedMps: 4.4,
  collideProps: false,
  physicsProps: true,
  unlit: true,
  showCollider: false,
  fovDeg: 75,
  mouseSensitivity: 1,
};

/**
 * First scale guess for an uncalibrated scene: assume the shell's vertical
 * extent is a ~2.6 m storey. Deliberately a guess — the HUD says so and the
 * user dials it. If the manifest carries meters_per_unit we use that instead.
 */
export const ASSUMED_STOREY_METRES = 2.6;

export function guessUnitsPerMetre(manifest: WorldManifest, shellHeightUnits: number | null): number {
  const mpu = manifest.meters_per_unit;
  if (typeof mpu === "number" && Number.isFinite(mpu) && mpu > 0) return 1 / mpu;
  const h = shellHeightUnits ?? manifest.scene_extent?.[1] ?? null;
  if (h && Number.isFinite(h) && h > 0) return h / ASSUMED_STOREY_METRES;
  return 1;
}

/* ------------------------------------------------------------------ *
 * Loaded state surfaced to the HUD.                                   *
 * ------------------------------------------------------------------ */

export interface LoadedElement {
  slug: string;
  label: string;
  role: WorldRole | "shell";
  glb: string;
  faces: number;
  tris: number;
  /** World-space AABB after import with NO transform applied. */
  box: THREE.Box3;
  center: [number, number, number];
  size: [number, number, number];
  collides: boolean;
  visible: boolean;
  object: THREE.Object3D;
  /** "authored" for placed assets — the collider merges authored static/
   *  environment geometry into the BVH; captured statics are already
   *  represented by the collision shell and must not double in. */
  provenance: string | null;
  /** Populated when this GLB disagreed with the shared-frame assumption. */
  frameWarning: string | null;
}

/** Boundary of the photograph — walker-frame scene units (Y-up). Mirrors the
 *  server's dev.splatlab.world-curtain/v1 document. */
export interface CurtainParams {
  enabled: boolean;
  shape: "sphere" | "box";
  center: [number, number, number];
  radius: number;
  halfExtents: [number, number, number];
  softEdge: number;
}

export interface WalkerStats {
  fps: number;
  position: [number, number, number];
  grounded: boolean;
  colliderTris: number;
  drawnTris: number;
  locked: boolean;
}

export interface LoadProgress {
  loaded: number;
  total: number;
  label: string;
}

export interface WorldLoadResult {
  manifest: WorldManifest;
  elements: LoadedElement[];
  sceneBox: THREE.Box3;
  /** Empty when every GLB agreed with the shared-frame assumption. */
  frameWarnings: string[];
  colliderTris: number;
  /** Which geometry the BVH was built from — surfaced so a fallback to the
   *  fragmented visual shell is visible rather than silent, and so merged
   *  authored geometry (environment/static placements) is visible too. */
  colliderSource: "collision_shell" | "collision_shell+authored" | "visual_shell";
  suggestedUnitsPerMetre: number;
}

/* ------------------------------------------------------------------ *
 * Scratch objects — allocated once, the physics loop must not GC.     *
 * ------------------------------------------------------------------ */

const _seg = new THREE.Line3();
const _box = new THREE.Box3();
const _mat = new THREE.Matrix4();
const _triPoint = new THREE.Vector3();
const _capPoint = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _newPos = new THREE.Vector3();
const _delta = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _right = new THREE.Vector3();
const _move = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, "YXZ");
const _up = new THREE.Vector3(0, 1, 0);
const _ray = new THREE.Ray();
// Reused for interaction targeting. Allocated once beside the physics scratch
// for the same reason: neither loop may hand the GC work per frame.
const _interactRay = new THREE.Raycaster();
const _screenCentre = new THREE.Vector2(0, 0);
const _zero = new THREE.Vector3(0, 0, 0);

/**
 * Stride-sampled vertex positions of an object minus `offset`, for convex
 * hull building. Node transforms are identity by the exporter contract, so
 * raw POSITION values are already world-space; `offset` maps them into a
 * re-origined prop's local frame.
 */
function gatherPoints(object: THREE.Object3D, offset: THREE.Vector3, maxPoints = 2048): Float32Array {
  const out: number[] = [];
  object.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const pos = (mesh.geometry as THREE.BufferGeometry).getAttribute("position");
    if (!pos) return;
    const stride = Math.max(1, Math.floor(pos.count / maxPoints));
    for (let i = 0; i < pos.count; i += stride) {
      out.push(pos.getX(i) - offset.x, pos.getY(i) - offset.y, pos.getZ(i) - offset.z);
    }
  });
  return new Float32Array(out);
}

const MOVE_KEYS = new Set([
  "KeyW", "KeyA", "KeyS", "KeyD",
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "Space", "ShiftLeft", "ShiftRight", "KeyR", "BracketLeft", "BracketRight",
  "KeyF", "KeyC",
  INTERACT_KEY,
]);

export class WorldWalker {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: PointerLockControls;

  params: WalkParams = { ...DEFAULT_WALK_PARAMS };
  elements: LoadedElement[] = [];
  sceneBox = new THREE.Box3();

  /** Fired ~4x/second with HUD numbers. */
  onStats: ((s: WalkerStats) => void) | null = null;
  /** Fired when the engine itself changes params (the [ ] scale hotkeys). */
  onParams: ((p: WalkParams) => void) | null = null;
  onLockChange: ((locked: boolean) => void) | null = null;
  /** Fired ONLY when the crosshair target or its state changes — not per frame. */
  onTarget: ((target: TargetInfo | null) => void) | null = null;
  /** Fired when the player acts; the page persists it. */
  onInteract: ((slug: string, nextState: string) => void) | null = null;

  private interactions = new Map<string, InteractionRecord>();
  private interactionState = new Map<string, string>();
  private lastTargetKey: string | null = null;
  private targetClock = 0;
  /** Fly (noclip) — for inspecting a world you cannot yet walk. */
  private flying = false;
  private spark: SparkRenderer | null = null;
  /** world_shell's PROVEN-interior point, Y-up. See respawn(). */
  private spawnSeed: THREE.Vector3 | null = null;
  /** Proven floor/ceiling of the solid, Y-up — the headroom the seed sits in. */
  private spawnFloorY = 0;
  private spawnTopY = 0;
  private backdrop: SplatMesh | null = null;
  /** Per-prop backdrop-splat rows (`_world/pluck.json`) — see setPluckDoc. */
  private pluckRows = new Map<string, number[]>();
  private pluckNRows = 0;
  private pluckedSlugs = new Set<string>();
  /** 0/255 per backdrop row — the source array of the modifier's RgbaArray. */
  private pluckMask: Uint8Array | null = null;
  private pluckClock = 0;
  private pluckWarned = false;
  /** The curtain: a Spark SDF edit fading the photograph OUTSIDE a boundary. */
  private curtainEdit: SplatEdit | null = null;
  private curtainSdf: SplatEditSdf | null = null;
  /** Eye-toggle intent per slug — outranks the computed visibility default. */
  private visibilityOverrides = new Map<string, boolean>();
  /** Fired when fly mode toggles, so the HUD can say so. */
  onFlyChange: ((flying: boolean) => void) | null = null;

  private readonly canvas: HTMLCanvasElement;
  private readonly velocity = new THREE.Vector3();
  private readonly keys = new Set<string>();
  private readonly clock = new THREE.Clock();
  private readonly worldGroup = new THREE.Group();
  private readonly lights = new THREE.Group();
  /** The applied look, or null for "as captured". */
  private restyle: RestyleDoc | null = null;
  /** True when this world's atlases carry a permanently baked restyle. */
  private bakedLook = false;
  /** True while an active restyle is showing mesh in place of the splat. */
  private restyleShowsMesh = false;
  private readonly classTiles = new Map<string, THREE.DataTexture>();

  /** Geometry of the dedicated collision solid, when the manifest ships one. */
  private collisionShellGeom: THREE.BufferGeometry | null = null;
  private colliderSource: "collision_shell" | "collision_shell+authored" | "visual_shell" = "visual_shell";
  private collider: THREE.Mesh | null = null;
  private colliderWire: THREE.LineSegments | null = null;
  private bvh: MeshBVH | null = null;
  private colliderTris = 0;
  private drawnTris = 0;

  /** Rapier prop physics; null until enablePhysics() succeeds. */
  private physics: WorldPhysics | null = null;
  /** Fired when physics engages/disengages, so the HUD can say so. */
  onPhysicsChange: ((active: boolean, propCount: number) => void) | null = null;
  /** Per-frame hook for layered systems (game mode). Called after physics. */
  onFrame: ((dt: number) => void) | null = null;
  /** Consulted FIRST on left-click while locked; return true = click consumed
   *  (game mode's hitscan outranks the carry-throw). */
  combatClick: (() => boolean) | null = null;
  /** Fired when the carried prop changes (slug, or null on release). */
  onCarryChange: ((slug: string | null) => void) | null = null;

  private grounded = false;
  private rafId = 0;
  private running = false;
  private disposed = false;
  private resizeObserver: ResizeObserver | null = null;

  private frameCount = 0;
  private statsClock = 0;
  private fps = 0;
  private spawn = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x0b0b0e, 1);

    this.camera = new THREE.PerspectiveCamera(this.params.fovDeg, 1, 0.01, 5000);
    this.scene.add(this.worldGroup);

    // Lit mode rig. Unlit mode leaves these in the graph but unused.
    const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x2a2620, 2.0);
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(1, 2, 1);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-1.5, 1, -1.2);
    this.lights.add(hemi, key, fill, new THREE.AmbientLight(0xffffff, 0.35));
    this.scene.add(this.lights);

    this.controls = new PointerLockControls(this.camera, canvas);
    this.controls.addEventListener("lock", this.handleLock);
    this.controls.addEventListener("unlock", this.handleUnlock);

    window.addEventListener("keydown", this.handleKeyDown);
    window.addEventListener("keyup", this.handleKeyUp);
    window.addEventListener("blur", this.handleBlur);
    canvas.addEventListener("mousedown", this.handleMouseDown);

    if (typeof ResizeObserver !== "undefined") {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      if (canvas.parentElement) this.resizeObserver.observe(canvas.parentElement);
    }
    this.resize();
  }

  /* -------------------------------------------------------------- *
   * Loading                                                         *
   * -------------------------------------------------------------- */

  /** Fetch + parse the manifest. Throws with a URL-bearing message on failure. */
  static async fetchManifest(source: WorldSource, signal?: AbortSignal): Promise<WorldManifest> {
    const url = source.fileUrl("world_manifest.json");
    let res: Response;
    try {
      res = await fetch(url, { credentials: "same-origin", signal });
    } catch (err) {
      throw new Error(`Could not reach ${url} — ${(err as Error).message}`);
    }
    if (res.status === 401) throw new Error(`Not authorised for ${url} (sign in to SplatLab first).`);
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText} from ${url}`);
    let json: unknown;
    try {
      json = await res.json();
    } catch {
      throw new Error(`Response from ${url} was not JSON (is the world route wired up yet?)`);
    }
    const manifest = json as WorldManifest;
    if (!manifest || (!manifest.shell && !manifest.elements?.length)) {
      throw new Error(`Manifest at ${url} has neither a shell nor any elements.`);
    }
    return manifest;
  }

  async loadWorld(
    source: WorldSource,
    manifest: WorldManifest,
    opts: { signal?: AbortSignal; onProgress?: (p: LoadProgress) => void } = {},
  ): Promise<WorldLoadResult> {
    this.clearWorld();

    const entries: Array<{
      entry: WorldEntry; dir: string; role: WorldRole | "shell"; url: string;
    }> = [];
    const push = (entry: WorldEntry | null | undefined, dir: string, role: WorldRole | "shell") => {
      if (!entry) return;
      const url = entryGlbUrl(entry, dir, source);
      if (url) entries.push({ entry, dir, role, url });
    };
    push(manifest.shell, "", "shell");
    for (const el of manifest.elements ?? []) push(el, "elements/", el?.role ?? "prop");
    if (!entries.length) throw new Error("Manifest listed no loadable GLB files.");

    const loaded: LoadedElement[] = [];
    const warnings: string[] = [];

    for (let i = 0; i < entries.length; i++) {
      if (opts.signal?.aborted) throw new Error("aborted");
      const { entry, dir, role, url } = entries[i];
      opts.onProgress?.({ loaded: i, total: entries.length, label: entry.label || entry.slug });

      const object = await this.loadGlb(source, url, dir);
      const box = new THREE.Box3().setFromObject(object);
      let tris = 0;
      object.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        const g = mesh.geometry as THREE.BufferGeometry;
        tris += g.index ? g.index.count / 3 : (g.getAttribute("position")?.count ?? 0) / 3;
      });

      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      object.name = entry.slug;
      object.userData.slug = entry.slug;
      this.worldGroup.add(object);

      loaded.push({
        slug: entry.slug,
        label: entry.label || entry.slug,
        role,
        glb: entry.glb ?? url,
        faces: entry.faces ?? Math.round(tris),
        tris: Math.round(tris),
        box,
        center: [center.x, center.y, center.z],
        size: [size.x, size.y, size.z],
        collides: false, // filled in by rebuildCollider()
        visible: true,
        object,
        provenance: entry.provenance ?? null,
        frameWarning: null,
      });
    }
    opts.onProgress?.({ loaded: entries.length, total: entries.length, label: "assembling" });

    // Re-verify the shared-frame assumption at runtime rather than trusting it.
    const shellEntry = loaded.find((e) => e.role === "shell") ?? null;
    for (const w of verifySharedFrame(loaded, shellEntry)) {
      warnings.push(w.message);
      const target = loaded.find((e) => e.slug === w.slug);
      if (target) target.frameWarning = w.message;
    }

    this.elements = loaded;
    this.sceneBox = new THREE.Box3();
    for (const e of loaded) this.sceneBox.union(e.box);

    // The walkable solid, loaded but never added to the scene graph — it is
    // collision-only geometry and must not be drawn over the textured shell.
    // A failure here is reported, not fatal: the world still loads and falls
    // back to colliding against the visual shell, which is the pre-existing
    // (worse) behaviour rather than a blank screen.
    this.collisionShellGeom = null;
    // world_collision.py does NOT write collision_shell into world_manifest.json
    // — only the merged /world/manifest route synthesizes that key, and this
    // walker deliberately reads the raw file. So fall back to the conventional
    // path: world_shell.py always writes _world/collision_shell.glb. Without
    // this the watertight solid sits on disk unused and the player falls
    // through the lace visual shell, which is exactly what it exists to fix.
    const declaredCsUrl = manifest.collision_shell?.glb;
    const csUrl = declaredCsUrl ?? "collision_shell.glb";
    if (csUrl) {
      opts.onProgress?.({ loaded: entries.length, total: entries.length, label: "collision solid" });
      try {
        const obj = await this.loadGlb(source, csUrl, "");
        obj.updateWorldMatrix(true, true);
        const parts: THREE.BufferGeometry[] = [];
        obj.traverse((o) => {
          const mesh = o as THREE.Mesh;
          if (!mesh.isMesh) return;
          const g = positionOnly(mesh.geometry as THREE.BufferGeometry);
          g.applyMatrix4(mesh.matrixWorld);
          parts.push(g);
        });
        if (parts.length) {
          const solid = parts.length === 1 ? parts[0] : mergeGeometries(parts, false);
          if (parts.length > 1) for (const g of parts) g.dispose();
          if (solid) this.collisionShellGeom = solid;
        }
        disposeObject(obj);

        // world_shell searched for a point it PROVED is inside the solid and
        // recorded it as params.seed_yup, in the same Y-up frame as this GLB.
        // Without it the walker falls back to a downward ray from the scene
        // centre, which on a watertight shell lands on the LID — the
        // long-standing "you spawn on the roof" complaint. The seed is not in
        // world_manifest.json (only the merged route synthesises it), so read
        // the shell's own report.
        try {
          const rep = await fetch(source.fileUrl("collision_shell.json"),
                                  { credentials: "same-origin", signal: opts.signal });
          if (rep.ok) {
            const report = await rep.json();
            const seed = report?.params?.seed_yup;
            if (Array.isArray(seed) && seed.length === 3 && seed.every(Number.isFinite)) {
              this.spawnSeed = new THREE.Vector3(seed[0], seed[1], seed[2]);
              const probe = report?.probe ?? {};
              this.spawnFloorY = Number(probe.floor_level_y ?? seed[1]);
              this.spawnTopY = Number(probe.top_level_y ?? seed[1]);
            }
          }
        } catch {
          /* no report, or aborted — fall back to the ray */
        }
      } catch (err) {
        // A missing file on the conventional path just means this world has no
        // collision solid; only a DECLARED one failing to load is worth a
        // warning, otherwise every pre-shell world gets a scary banner.
        if (declaredCsUrl) {
          warnings.push(
            `Collision solid failed to load (${(err as Error).message || err}); ` +
            `colliding against the visual shell instead.`,
          );
        }
      }
    }

    const shellHeight = shellEntry ? shellEntry.size[1] : this.sceneBox.getSize(new THREE.Vector3()).y;
    const suggested = guessUnitsPerMetre(manifest, shellHeight);

    this.applyMaterialMode();

    // Prop physics — best-effort by design: a world without hulls, or a WASM
    // init failure, degrades to the pre-physics walker rather than failing
    // the load. Runs BEFORE rebuildCollider so the props-are-dynamic decision
    // can force collideProps off (a prop cannot be baked-static AND dynamic).
    if (this.params.physicsProps) {
      try {
        await this.enablePhysics(source, entries, opts.signal);
      } catch (err) {
        warnings.push(
          `Prop physics unavailable (${(err as Error).message || err}); ` +
          "props are display-only.",
        );
      }
    }

    this.rebuildCollider();
    this.respawn();

    // Far plane must comfortably clear the scene at any scale.
    const diag = this.sceneBox.getSize(new THREE.Vector3()).length();
    this.camera.far = Math.max(100, diag * 20);
    this.camera.near = Math.max(1e-4, diag / 20000);
    this.camera.updateProjectionMatrix();

    return {
      manifest,
      elements: loaded,
      sceneBox: this.sceneBox,
      frameWarnings: warnings,
      colliderTris: this.colliderTris,
      colliderSource: this.colliderSource,
      suggestedUnitsPerMetre: suggested,
    };
  }

  /**
   * Stand up Rapier for the loaded props. Best-effort: throws only for
   * whole-system failures (WASM init, no static geometry); a prop without
   * usable hulls falls back to a convex hull of its render mesh, and a prop
   * that yields no collider at all is simply left display-only.
   */
  private async enablePhysics(
    source: WorldSource,
    entries: Array<{ entry: WorldEntry; dir: string; role: WorldRole | "shell"; url: string }>,
    signal?: AbortSignal,
  ): Promise<void> {
    const props = this.elements.filter((el) => el.role === "prop");
    if (!props.length) return;
    const rapier = await loadRapier();
    if (signal?.aborted) throw new Error("aborted");

    this.physics?.dispose();
    this.physics = null;
    const physics = new WorldPhysics(rapier, this.params.unitsPerMetre);
    try {

    // Props rest on the same solid the capsule walks on — built by the same
    // builder the BVH uses (authored environment platforms included), so the
    // player's floor and the props' floor can never disagree. Never with
    // props merged in: a prop cannot be baked-static AND dynamic.
    const built = this.buildStaticColliderGeometry(false);
    if (!built) {
      physics.dispose();
      throw new Error("no static geometry for props to rest on");
    }
    physics.addStaticShell(built.geom);
    built.geom.dispose();

    let registered = 0;
    for (const el of props) {
      const entry = entries.find((e) => e.entry.slug === el.slug)?.entry;
      const files = entry?.collision?.files ?? [];
      // Geometry arrives world-space-baked with identity node transforms
      // (file-header contract); physics needs a local frame, so bake the
      // AABB centre out once. Idempotent, preserves the world pose.
      const centre = reoriginObject(el.object);
      const hulls: Float32Array[] = [];
      for (const file of files) {
        if (signal?.aborted) throw new Error("aborted");
        try {
          const hullObj = await this.loadGlb(
            source, source.fileUrl(`collision/${file}`), "collision/",
          );
          hulls.push(gatherPoints(hullObj, centre));
          disposeObject(hullObj);
        } catch {
          // One vanished hull is not fatal — the render-mesh fallback below
          // still gives the prop a body.
        }
      }
      if (!hulls.some((h) => h.length >= 12)) {
        hulls.length = 0;
        hulls.push(gatherPoints(el.object, _zero));
      }
      if (physics.addProp(el.slug, el.object, hulls)) registered++;
    }

    if (!registered || this.disposed) {
      // Zero usable props, or the walker was torn down while hulls loaded
      // (job switch, StrictMode remount): a World assigned now would never
      // be freed, since dispose() is once-only. Free it here instead.
      physics.dispose();
      return;
    }
    physics.installPlayer(this.capsuleRadius, this.eyeHeight);
    this.physics = physics;
    // Dynamic wins: a prop cannot also be baked into the static BVH.
    this.params.collideProps = false;
    this.onPhysicsChange?.(true, registered);
    } catch (err) {
      // Any throw on the way up (abort mid-hull-fetch, shell failure) must
      // free the WASM-side World — review finding: the abort path leaked it.
      physics.dispose();
      throw err;
    }
  }

  get physicsActive(): boolean {
    return this.physics !== null;
  }

  get carryingSlug(): string | null {
    return this.physics?.carrying ?? null;
  }

  /** Poses of props that moved from their authored placement (persistence). */
  getPhysicsPoses(): PropTransforms | null {
    return this.physics ? this.physics.disturbedTransforms() : null;
  }

  /** Restore persisted prop poses (a reload's saved game). */
  applyPhysicsPoses(poses: PropTransforms): void {
    this.physics?.applyTransforms(poses);
  }

  private async loadGlb(source: WorldSource, url: string, dir: string): Promise<THREE.Object3D> {
    // A URL modifier keeps external image URIs working through a query-param
    // API route. Embedded textures arrive as blob: URLs and must pass through.
    const manager = new THREE.LoadingManager();
    manager.setURLModifier((url) => {
      if (/^(blob:|data:|https?:|\/)/i.test(url)) return url;
      return source.fileUrl(dir + url);
    });
    const loader = new GLTFLoader(manager);
    loader.setPath("");

    let gltf;
    try {
      gltf = await loader.loadAsync(url);
    } catch (err) {
      throw new Error(`Failed to load ${url} — ${(err as Error).message || err}`);
    }

    const root = gltf.scene;
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const geometry = mesh.geometry as THREE.BufferGeometry;
      // MEASURED: these GLBs carry POSITION + TEXCOORD_0 only. Without this,
      // every lit material renders black.
      if (!geometry.getAttribute("normal")) geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      geometry.computeBoundingSphere();

      const src = (Array.isArray(mesh.material) ? mesh.material[0] : mesh.material) as THREE.MeshStandardMaterial;
      const map = (src?.map as THREE.Texture | null) ?? null;
      // Reconstructed shells are open surfaces — backface culling would make
      // the room invisible from inside. DoubleSide everywhere is the safe read.
      const lit = new THREE.MeshStandardMaterial({
        map,
        color: src?.color?.clone() ?? new THREE.Color(0xffffff),
        roughness: src?.roughness ?? 0.85,
        metalness: 0,
        side: THREE.DoubleSide,
      });
      const unlit = new THREE.MeshBasicMaterial({
        map,
        color: map ? new THREE.Color(0xffffff) : (src?.color?.clone() ?? new THREE.Color(0xffffff)),
        side: THREE.DoubleSide,
      });
      mesh.userData.litMaterial = lit;
      mesh.userData.unlitMaterial = unlit;
      mesh.userData.sourceMaterial = src;
      mesh.material = this.params.unlit ? unlit : lit;
      mesh.frustumCulled = true;
    });
    return root;
  }

  /* -------------------------------------------------------------- *
   * Collider                                                        *
   * -------------------------------------------------------------- */

  /**
   * Merge every colliding mesh into ONE position-only geometry and build a
   * MeshBVH over it. Shell + role="static" always collide; props are opt-in
   * (per the v1 brief — the manifest ships per-prop convex hulls we do not
   * need yet, because complex_as_simple shell collision is exact enough).
   */
  rebuildCollider(): void {
    this.disposeCollider();
    if (!this.elements.length) return;
    const useShell = !!this.collisionShellGeom && !this.params.collideProps;
    for (const el of this.elements) {
      // In the fast path a captured static is represented by the collision
      // solid itself, an authored static/environment by its merged triangles
      // — either way collides=true is the truth. Props opt in only on the
      // fallback path (physics forces collideProps off when it engages).
      el.collides = el.role === "shell" || el.role === "static"
        || el.role === "environment" || (!useShell && this.params.collideProps);
    }
    const built = this.buildStaticColliderGeometry(this.params.collideProps);
    if (!built) return;
    this.installCollider(built.geom, built.source);
  }

  /**
   * The static solid the capsule AND resting props collide with, built once
   * and shared by the BVH and Rapier so the two can never disagree.
   *
   * Fast path (a purpose-built collision shell exists): the shell solid PLUS
   * authored static/environment placements. Captured statics are already
   * inside the collision solid and must not double in — but a placed
   * platform or terrain skirt exists nowhere else, so its render triangles
   * ARE its collider (the environment role's whole point; this also ends the
   * era of authored statics reporting collides=true while contributing zero
   * triangles). Fallback path (no collision shell, or collideProps): the
   * merged visual geometry, the pre-existing degradation.
   */
  private buildStaticColliderGeometry(includeProps: boolean): {
    geom: THREE.BufferGeometry;
    source: "collision_shell" | "collision_shell+authored" | "visual_shell";
  } | null {
    const useShell = !!this.collisionShellGeom && !includeProps;
    const geoms: THREE.BufferGeometry[] = [];
    if (useShell) geoms.push(this.collisionShellGeom!.clone());
    let authored = 0;
    for (const el of this.elements) {
      const merge = useShell
        ? el.provenance === "authored"
          && (el.role === "static" || el.role === "environment")
        : el.role === "shell" || el.role === "static"
          || el.role === "environment" || includeProps;
      if (!merge) continue;
      if (useShell) authored += 1;
      el.object.updateWorldMatrix(true, true);
      el.object.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        const g = positionOnly(mesh.geometry as THREE.BufferGeometry);
        g.applyMatrix4(mesh.matrixWorld);
        geoms.push(g);
      });
    }
    if (!geoms.length) return null;
    const merged = geoms.length === 1 ? geoms[0] : mergeGeometries(geoms, false);
    if (geoms.length > 1) for (const g of geoms) g.dispose();
    if (!merged) {
      // mergeGeometries logs to console and returns null on mismatched attrs;
      // positionOnly() makes that impossible, but fail loud rather than
      // silently shipping a world with no floor.
      throw new Error("Collider merge failed — geometries had incompatible attributes.");
    }
    const source = useShell
      ? (authored ? "collision_shell+authored" as const : "collision_shell" as const)
      : "visual_shell" as const;
    return { geom: merged, source };
  }

  private installCollider(
    merged: THREE.BufferGeometry,
    source: "collision_shell" | "collision_shell+authored" | "visual_shell",
  ): void {
    this.colliderSource = source;
    merged.computeBoundingBox();
    this.bvh = new MeshBVH(merged, { maxLeafTris: 8 });
    this.colliderTris = merged.index ? merged.index.count / 3 : merged.getAttribute("position").count / 3;

    this.collider = new THREE.Mesh(merged, new THREE.MeshBasicMaterial({ visible: false }));
    this.collider.matrixAutoUpdate = false;
    this.collider.updateMatrixWorld(true);
    this.scene.add(this.collider);

    this.colliderWire = new THREE.LineSegments(
      new THREE.WireframeGeometry(merged),
      new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: 0.25, depthTest: true }),
    );
    this.colliderWire.visible = this.params.showCollider;
    this.scene.add(this.colliderWire);
  }

  private disposeCollider(): void {
    if (this.collider) {
      this.scene.remove(this.collider);
      this.collider.geometry.dispose();
      (this.collider.material as THREE.Material).dispose();
      this.collider = null;
    }
    if (this.colliderWire) {
      this.scene.remove(this.colliderWire);
      this.colliderWire.geometry.dispose();
      (this.colliderWire.material as THREE.Material).dispose();
      this.colliderWire = null;
    }
    this.bvh = null;
    this.colliderTris = 0;
  }

  /* -------------------------------------------------------------- *
   * Scale-relative derived dimensions                               *
   * -------------------------------------------------------------- */

  private get eyeHeight(): number {
    return Math.max(1e-4, this.params.eyeHeightM * this.params.unitsPerMetre);
  }

  private get capsuleRadius(): number {
    // A capsule needs radius < eye height; clamp instead of exploding when a
    // user dials the scale to something extreme.
    return Math.min(this.params.radiusM * this.params.unitsPerMetre, this.eyeHeight * 0.35);
  }

  private get capsuleHeight(): number {
    // The physics capsule's height in units: the eye height, clamped so the
    // whole capsule FITS the headroom world_shell proved at the spawn seed —
    // capsule extent is height + radius, hence the radius-and-margin cap. On
    // an uncalibrated capture the scale guess can make the player taller
    // than the interior, and a capsule that cannot fit has no penetration-
    // free pose: its feet ended ~2 units inside the floor solid and one
    // depenetration pass threw the player clean out of the world into an
    // eject/fall/respawn loop (Truck, A9). A wrong scale must cost a short-
    // feeling player, never a fall out of the world. Worlds without a probe,
    // and worlds whose headroom really fits the player, are untouched.
    const headroom = this.spawnTopY - this.spawnFloorY;
    if (!(headroom > 1e-4)) return this.eyeHeight;
    const fit = Math.max(this.capsuleRadius * 1.05, (headroom - this.capsuleRadius) * 0.95);
    return Math.min(this.eyeHeight, fit);
  }

  /* -------------------------------------------------------------- *
   * Spawning / teleporting                                          *
   * -------------------------------------------------------------- */

  /** Drop the player onto the floor near the middle of the world. */
  respawn(): void {
    // Prefer the seed world_shell proved is inside the solid. Casting down from
    // the scene centre finds the first surface below, and on a watertight shell
    // that is the roof.
    if (this.spawnSeed) {
      // The seed gives a proven-clear XZ; the height has to be chosen.
      //
      // Taking the seed's own y puts the camera in the grass — it is a
      // floor-level sample. Adding eyeHeight blindly launches the player
      // through the roof whenever the scale dial is off, which on an
      // uncalibrated capture it usually is: at 2.83 u/m a 1.7 m eye height is
      // 4.8 units, and the whole solid is 5.9 units tall, so the spawn landed
      // outside and fell straight out (ground=air).
      //
      // So stand feet-on-the-PROVEN-floor with the capsule capsuleHeight has
      // already clamped into the measured headroom (its ceiling margin keeps
      // the head clear too). A wrong scale then costs a short-feeling player,
      // never a fall out of the world. Anything less than the full capsule
      // height here puts the feet INSIDE the floor solid — an earlier eye
      // clamped to 0.6 × headroom while the capsule kept its full height left
      // the feet ~2 units deep in the slab, and one depenetration pass hurled
      // the spawn out of the world (the Truck A9 eject/fall/respawn loop).
      const seeded = this.spawnSeed.clone();
      seeded.y = this.spawnFloorY + this.capsuleHeight;
      this.spawn.copy(seeded);
      this.camera.position.copy(seeded);
      this.velocity.set(0, 0, 0);
      this.grounded = false;
      if (this.bvh) this.resolveCollisions(1 / 60);
      return;
    }
    const point = this.findStandingPoint(this.sceneBox.getCenter(new THREE.Vector3()));
    this.spawn.copy(point);
    this.camera.position.copy(point);
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    if (this.bvh) this.resolveCollisions(1 / 60);
  }

  /** Stand next to an element and look at it. */
  teleportTo(slug: string): void {
    const el = this.elements.find((e) => e.slug === slug);
    if (!el) return;
    const center = el.box.getCenter(new THREE.Vector3());
    const size = el.box.getSize(new THREE.Vector3());
    const worldCenter = this.sceneBox.getCenter(new THREE.Vector3());

    // Back off toward the middle of the world so we do not spawn in a wall.
    const away = new THREE.Vector3(worldCenter.x - center.x, 0, worldCenter.z - center.z);
    if (away.lengthSq() < 1e-9) away.set(1, 0, 0);
    away.normalize();
    const backoff = Math.max(size.x, size.z) * 0.5 + this.capsuleRadius * 3 + this.eyeHeight * 0.3;

    const target = center.clone().addScaledVector(away, backoff);
    this.camera.position.copy(this.findStandingPoint(target));
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    if (this.bvh) this.resolveCollisions(1 / 60);
    this.lookAt(center);
  }

  /** Every collider face in the column at (x, z), by height. Empty with no
   *  collider or nothing below. The primitive both surface picks share. */
  columnHits(x: number, z: number): number[] {
    if (!this.bvh) return [];
    const top = this.sceneBox.max.y + Math.max(1, this.sceneBox.getSize(new THREE.Vector3()).y);
    _ray.origin.set(x, top, z);
    _ray.direction.set(0, -1, 0);
    return this.bvh.raycast(_ray, THREE.DoubleSide)
      .map((h) => h.point.y)
      .sort((a, b) => a - b);
  }

  /**
   * The standing surface at (x, z) nearest `referenceY`, within `band`.
   *
   * Both naive picks are measurably wrong on a real outdoor world: on the
   * Stump's collider (600 sampled walkable columns, ~2.1 faces each),
   * highest-hit lands on the tree canopy in the top 5% of columns and
   * lowest-hit always lands on the terrain slab's UNDERSIDE. Choosing the
   * face NEAREST the navmesh's graded floor gave |error| p50 0.15 / p95
   * 0.76 units with no canopy tail at all — so that is the rule.
   *
   * The band is a deliberate single-storey contract, matching the navmesh:
   * this refines height AROUND the graded floor, it does not discover a
   * mezzanine. Null when no face falls in the band — callers keep using
   * `referenceY` itself, which is exactly the pre-probe behaviour.
   */
  surfaceNear(x: number, z: number, referenceY: number, band: number): number | null {
    let best: number | null = null;
    let bestGap = Infinity;
    for (const y of this.columnHits(x, z)) {
      const gap = Math.abs(y - referenceY);
      if (gap <= band && gap < bestGap) { best = y; bestGap = gap; }
    }
    return best;
  }

  /** The highest face at or below `belowY` — spawn placement, where there is
   *  no graded floor to refine around yet. Null when the column has none. */
  groundAt(x: number, z: number, belowY = Infinity): number | null {
    let best = -Infinity;
    for (const y of this.columnHits(x, z)) if (y <= belowY && y > best) best = y;
    return Number.isFinite(best) ? best : null;
  }

  /**
   * Ray-cast straight down onto the collider to find a floor, then place the
   * eye one eye-height above it. Falls back to a ring of nearby probes, then
   * to a mid-air drop, so an odd scene never leaves the player nowhere.
   */
  private findStandingPoint(nearXZ: THREE.Vector3): THREE.Vector3 {
    const radiusStep = this.sceneBox.getSize(new THREE.Vector3()).length() * 0.06;
    const probes: THREE.Vector3[] = [nearXZ.clone()];
    for (let ring = 1; ring <= 3; ring++) {
      for (let i = 0; i < 8; i++) {
        const a = (i / 8) * Math.PI * 2;
        probes.push(
          new THREE.Vector3(
            nearXZ.x + Math.cos(a) * radiusStep * ring,
            nearXZ.y,
            nearXZ.z + Math.sin(a) * radiusStep * ring,
          ),
        );
      }
    }

    if (this.bvh) {
      // Cap the probe below the upper third of the world so it lands on the
      // floor/turf, never the canopy or roof (the old shell-roof spawn bug).
      const cap = this.sceneBox.min.y + this.sceneBox.getSize(new THREE.Vector3()).y * 0.7;
      for (const p of probes) {
        const y = this.groundAt(p.x, p.z, cap);
        if (y !== null) return new THREE.Vector3(p.x, y + this.eyeHeight, p.z);
      }
    }
    // No floor found: hover at mid-height and let gravity sort it out.
    const c = this.sceneBox.getCenter(new THREE.Vector3());
    return new THREE.Vector3(c.x, this.sceneBox.max.y - this.eyeHeight * 0.5, c.z);
  }

  private lookAt(target: THREE.Vector3): void {
    this.camera.lookAt(target);
    // PointerLockControls re-derives its euler from the camera quaternion on
    // the next mouse move, so writing the quaternion is enough — but strip any
    // roll so the horizon stays level.
    _euler.setFromQuaternion(this.camera.quaternion, "YXZ");
    _euler.z = 0;
    this.camera.quaternion.setFromEuler(_euler);
  }

  /* -------------------------------------------------------------- *
   * Params / visibility                                             *
   * -------------------------------------------------------------- */

  setParams(patch: Partial<WalkParams>): void {
    const before = this.params;
    this.params = { ...before, ...patch };
    if (patch.fovDeg !== undefined && patch.fovDeg !== before.fovDeg) {
      this.camera.fov = this.params.fovDeg;
      this.camera.updateProjectionMatrix();
    }
    if (patch.mouseSensitivity !== undefined) this.controls.pointerSpeed = this.params.mouseSensitivity;
    if (patch.unlit !== undefined && patch.unlit !== before.unlit) this.applyMaterialMode();
    if (patch.showCollider !== undefined && this.colliderWire) {
      this.colliderWire.visible = this.params.showCollider;
    }
    if (patch.collideProps !== undefined && patch.collideProps !== before.collideProps) {
      // Dynamic props and baked-static props are mutually exclusive; while
      // physics is live the checkbox is a no-op (forced back off).
      if (this.physics && this.params.collideProps) this.params.collideProps = false;
      else this.rebuildCollider();
    }
    if (this.physics && patch.unitsPerMetre !== undefined && patch.unitsPerMetre !== before.unitsPerMetre) {
      this.physics.setUnitsPerMetre(this.params.unitsPerMetre);
      this.physics.installPlayer(this.capsuleRadius, this.eyeHeight);
    }
    if (this.physics && (patch.radiusM !== undefined || patch.eyeHeightM !== undefined)) {
      this.physics.installPlayer(this.capsuleRadius, this.eyeHeight);
    }
  }

  /**
   * Show the original splat behind the meshes.
   *
   * Why this exists, measured rather than assumed: a watertight shell built
   * over an object-centric orbit capture encloses the camera ring, so it can
   * never resemble the capture from any viewpoint (mesh_gate now reports
   * `encloses_capture`). And even the correctly-shaped ground TIN only covers
   * ~40% of a frame — the rest of what you actually SEE outdoors is trees,
   * hedge and sky, which no simplified mesh carries. The splat carries all of
   * it, at ~22 dB where the mesh scores ~12.
   *
   * So the mesh stops trying to be the whole world and becomes the part you
   * collide with and act on, while the splat is the part you look at.
   *
   * Frame: the splat is in the CAPTURE frame (Z-up, scene units); the world is
   * Y-up. A -90 degree rotation about X maps (x,y,z) -> (x, z, -y), which is
   * exactly the conversion object_texture applies on export, so the two land in
   * the same place. Scale matches the meshes: metres when calibrated, scene
   * units otherwise.
   */
  async setBackdrop(url: string | null, metersPerUnit: number | null): Promise<void> {
    this.clearBackdrop();
    if (!url) return;
    if (!this.spark) {
      this.spark = new SparkRenderer({ renderer: this.renderer });
      this.scene.add(this.spark);
    }
    const splat = new SplatMesh({ url, fileType: SplatFileType.PLY });
    splat.rotation.x = -Math.PI / 2;
    const scale = metersPerUnit && metersPerUnit > 0 ? metersPerUnit : 1;
    splat.scale.setScalar(scale);
    // Never collide with it and never let it capture a look — it is scenery.
    splat.renderOrder = -1;
    this.scene.add(splat);
    this.backdrop = splat;
    // Rows can only be masked once the splat data exists — and a reloaded
    // backdrop is a fresh photograph, so whatever was already plucked (or is
    // already disturbed) gets re-plucked against a fresh mask.
    void splat.initialized.then(() => {
      if (this.backdrop !== splat) return;
      const already = [...this.pluckedSlugs];
      this.pluckedSlugs.clear();
      this.pluckMask = null;
      this.pluckWarned = false;
      for (const slug of already) this.pluckElement(slug);
      this.checkPluckDisturbed();
    }).catch(() => { /* a failed load is reported by setBackdrop's caller */ });
    // An active restyle keeps the mesh world on top of the photograph.
    if (this.restyleShowsMesh) {
      splat.visible = false;
      this.refreshElementVisibility();
      return;
    }

    // Photograph-first: with the splat showing, the shell AND captured
    // element meshes hide behind their own photograph (crude tracings drawn
    // over a photoreal frame box the player in and read as garbage blobs).
    // Collision is untouched — rebuildCollider builds from collisionShellGeom
    // plus authored merges, none of which live in the visible scene graph.
    this.refreshElementVisibility();
  }

  clearBackdrop(): void {
    const had = this.backdrop !== null;
    if (this.backdrop) {
      this.scene.remove(this.backdrop);
      this.backdrop.dispose?.();
      this.backdrop = null;
    }
    // Meshes are the only visual now — everything computed-visible again.
    if (had) this.refreshElementVisibility();
  }

  get hasBackdrop(): boolean {
    return this.backdrop !== null;
  }

  /* -------------------------------------------------------------- *
   * Pluck — per-prop backdrop-splat rows                            *
   * -------------------------------------------------------------- */

  /**
   * Install the pluck doc (`_world/pluck.json`): which backdrop-splat rows
   * belong to each interactive prop, so a prop that moves can take its
   * photograph ghost with it. The page validates the doc's FRESHNESS via
   * GET /world/pluck before calling this; the walker still refuses to touch
   * rows unless the backdrop's row count matches the doc — a decimated
   * fmt=web backdrop or a stale doc would pluck the wrong gaussians, and
   * wrong is worse than none.
   */
  setPluckDoc(doc: { n_rows?: number; elements?: Record<string, { rows?: number[] }> } | null): void {
    this.pluckRows.clear();
    this.pluckNRows = Number(doc?.n_rows ?? 0);
    this.pluckWarned = false;
    for (const [slug, entry] of Object.entries(doc?.elements ?? {})) {
      if (Array.isArray(entry?.rows) && entry.rows.length) {
        this.pluckRows.set(slug, entry.rows);
      }
    }
  }

  /** Row count of the backdrop splat, or 0 when the guard fails. Mutating
   *  packedSplats.packedArray after load is a visual no-op (the pipeline
   *  never re-reads it — measured Δ0.00 through every dirty-flag combo), so
   *  pluck ships through the mesh's worldModifier lane instead; this guard
   *  only certifies the rows ADDRESS the right splat. The backdrop must be
   *  the row-identical fmt=langweb artifact, never the decimated fmt=web. */
  private pluckNumSplats(): number {
    const splat = this.backdrop;
    if (!splat) return 0;
    const numSplats = splat.packedSplats?.numSplats ?? 0;
    if (!numSplats) return 0;
    if (this.pluckNRows && numSplats !== this.pluckNRows) {
      if (!this.pluckWarned) {
        this.pluckWarned = true;
        console.warn(
          `pluck: backdrop has ${numSplats} splats but the doc maps `
          + `${this.pluckNRows} — the backdrop must be the row-identical `
          + `fmt=langweb artifact, not the decimated fmt=web. Not plucking.`);
      }
      return 0;
    }
    return numSplats;
  }

  /** Rebuild the backdrop's pluck modifier from the current mask. The
   *  worldModifier slot on the BACKDROP belongs to pluck (nothing else in the
   *  walker writes it); an empty mask clears it entirely. */
  private refreshPluckModifier(numSplats: number): void {
    const splat = this.backdrop;
    if (!splat) return;
    if (!this.pluckedSlugs.size || !this.pluckMask) {
      splat.worldModifier = undefined;
      splat.updateGenerator();
      return;
    }
    const maskArray = new RgbaArray({
      array: packChannelsRgba([this.pluckMask], numSplats),
      count: numSplats,
    });
    splat.worldModifier = buildPluckModifier({ maskArray });
    splat.updateGenerator();
  }

  /** Remove a prop's splats from the photograph (idempotent, non-destructive
   *  — a 0/255 mask byte per row drives a zero-opacity modifier). Returns
   *  whether the rows are now plucked. */
  pluckElement(slug: string): boolean {
    if (this.pluckedSlugs.has(slug)) return true;
    const rows = this.pluckRows.get(slug);
    const numSplats = this.pluckNumSplats();
    if (!rows || !numSplats) return false;
    if (!this.pluckMask || this.pluckMask.length !== numSplats) {
      this.pluckMask = new Uint8Array(numSplats);
    }
    for (const row of rows) {
      if (row >= 0 && row < numSplats) this.pluckMask[row] = 255;
    }
    this.pluckedSlugs.add(slug);
    this.refreshPluckModifier(numSplats);
    // Its photograph ghost is gone — the mesh IS the prop now.
    this.refreshElementVisibility();
    return true;
  }

  /** Put a plucked prop's splats back — clears its rows from the mask. */
  unpluckElement(slug: string): void {
    const had = this.pluckedSlugs.delete(slug);
    const rows = this.pluckRows.get(slug);
    const numSplats = this.pluckNumSplats();
    if (!had || !rows || !numSplats || !this.pluckMask) return;
    for (const row of rows) {
      if (row >= 0 && row < numSplats) this.pluckMask[row] = 0;
    }
    this.refreshPluckModifier(numSplats);
    this.refreshElementVisibility();
  }

  /** Structural receipt for tests and live proofs: mapped rows, pluck state,
   *  and the effective opacity of each slug's first row per the LIVE mask
   *  (the very array the modifier's RgbaArray is built from). */
  pluckState(): Record<string, { rows: number; plucked: boolean; sampleOpacity: number | null }> {
    const numSplats = this.pluckNumSplats();
    const out: Record<string, { rows: number; plucked: boolean; sampleOpacity: number | null }> = {};
    for (const [slug, rows] of this.pluckRows) {
      let sampleOpacity: number | null = null;
      const row = rows[0];
      if (numSplats && row !== undefined && row >= 0 && row < numSplats) {
        sampleOpacity = this.pluckMask && this.pluckMask[row] === 255 ? 0 : 1;
      }
      out[slug] = { rows: rows.length, plucked: this.pluckedSlugs.has(slug), sampleOpacity };
    }
    return out;
  }

  /** Physics-disturbed props lose their photograph ghosts. One-way by design:
   *  a knocked-over prop's ghost must not flicker back when the body sleeps —
   *  and `disturbed` itself is one-way, covering restored saved poses too. */
  private checkPluckDisturbed(): void {
    if (!this.pluckRows.size || !this.backdrop || !this.physics) return;
    for (const slug of Object.keys(this.physics.disturbedTransforms())) {
      if (this.pluckRows.has(slug) && !this.pluckedSlugs.has(slug)) {
        this.pluckElement(slug);
      }
    }
  }

  /* -------------------------------------------------------------- *
   * Curtain — where the photograph ends                             *
   * -------------------------------------------------------------- */

  /**
   * Fade the splat photograph OUTSIDE a boundary shape so authored
   * environment geometry takes over past the seam instead of competing with
   * the capture's floater fringe.
   *
   * Non-destructive: a Spark SplatEdit SDF layer (opacity 0 x MULTIPLY x
   * invert; softEdge = linear fade band in scene units) that runs at the
   * world-space edit stage — BEFORE worldModifiers, so it composes with the
   * pluck row mask without touching that slot's contract. Parented under
   * this.scene, NOT the backdrop: the backdrop carries a -90-degree X
   * rotation and the metres-per-unit scale, and a scene-parented SDF keeps
   * the authored params in plain walker-frame scene units. (Scene-level
   * SplatEdits apply to every SplatMesh; the walker owns exactly one.)
   * Param changes mutate the existing objects — no generator rebuild.
   */
  setCurtain(params: CurtainParams | null): void {
    if (!params || !params.enabled) {
      if (this.curtainEdit) this.curtainEdit.visible = false;
      return;
    }
    if (!this.curtainEdit || !this.curtainSdf) {
      // WHITE + opacity 0 under MULTIPLY: rgb is untouched through the fade
      // band while alpha runs to zero — a fade, never a tint.
      const sdf = new SplatEditSdf({
        invert: true, opacity: 0, color: new THREE.Color(1, 1, 1),
      });
      const edit = new SplatEdit({
        rgbaBlendMode: SplatEditRgbaBlendMode.MULTIPLY,
        sdfSmooth: 0,
        softEdge: params.softEdge,
      });
      edit.add(sdf);
      this.scene.add(edit);
      this.curtainEdit = edit;
      this.curtainSdf = sdf;
    }
    const sdf = this.curtainSdf;
    sdf.position.set(params.center[0], params.center[1], params.center[2]);
    if (params.shape === "box") {
      // BOX reads half-extents from scale; radius becomes corner rounding.
      sdf.type = SplatEditSdfType.BOX;
      sdf.scale.set(params.halfExtents[0], params.halfExtents[1], params.halfExtents[2]);
      sdf.radius = 0;
    } else {
      sdf.type = SplatEditSdfType.SPHERE;
      sdf.scale.set(1, 1, 1);
      sdf.radius = params.radius;
    }
    sdf.updateMatrixWorld(true);
    this.curtainEdit.softEdge = params.softEdge;
    this.curtainEdit.visible = true;
  }

  /** Live receipt, read back from the actual scene objects. */
  curtainState(): CurtainParams | null {
    const edit = this.curtainEdit;
    const sdf = this.curtainSdf;
    if (!edit || !sdf) return null;
    const shape = sdf.type === SplatEditSdfType.BOX ? "box" as const : "sphere" as const;
    return {
      enabled: edit.visible,
      shape,
      center: [sdf.position.x, sdf.position.y, sdf.position.z],
      radius: sdf.radius,
      halfExtents: [sdf.scale.x, sdf.scale.y, sdf.scale.z],
      softEdge: edit.softEdge,
    };
  }

  setElementVisible(slug: string, visible: boolean): void {
    const el = this.elements.find((e) => e.slug === slug);
    if (!el) return;
    // The USER door: an explicit eye-toggle outranks every computed default
    // until the world reloads.
    this.visibilityOverrides.set(slug, visible);
    el.visible = visible;
    el.object.visible = visible;
  }

  /**
   * Photograph until you touch it, mesh once it is yours.
   *
   * While the splat backdrop is the visual, a CAPTURED element's mesh is a
   * crude tracing drawn over its own photograph — the "blobs of garbage"
   * problem. So captured meshes default to hidden behind the photograph, and
   * an element earns its mesh by being: authored (the mesh is all it has),
   * plucked (its photograph ghost is gone — the mesh IS it now), restyled
   * (a look needs geometry to live on), or genuinely interactive (a toggle
   * with effects; inspect-only prompts don't earn a blob). With no backdrop,
   * or with a restyle showing the mesh world, everything is visible — the
   * mesh is the only visual there is. Eye-toggles override everything.
   */
  private defaultElementVisible(el: LoadedElement): boolean {
    const photographShowing = this.backdrop !== null && !this.restyleShowsMesh;
    if (!photographShowing) return true;
    if (el.role === "shell") return false; // the photograph IS the shell
    if (el.provenance === "authored") return true;
    if (this.pluckedSlugs.has(el.slug)) return true;
    if (this.restyle?.elements?.[el.slug]) return true;
    const record = this.interactions.get(el.slug);
    if (record && record.verb !== "inspect") return true;
    return false;
  }

  /** Re-apply the visibility rule to every element (overrides win). */
  refreshElementVisibility(): void {
    for (const el of this.elements) {
      const visible = this.visibilityOverrides.get(el.slug)
        ?? this.defaultElementVisible(el);
      el.visible = visible;
      el.object.visible = visible;
    }
  }

  /* ---------------------------------------------------------------- *
   * Restyle (W2-C2) — a look applied OVER the capture, never into it  *
   * ---------------------------------------------------------------- */

  /**
   * Apply a restyle document: per-element tint / class material, plus one
   * lighting preset. Idempotent and fully reversible — every restyled mesh
   * keeps its baked materials in userData, so an empty document restores
   * the capture exactly (that is what `Reset` and DELETE do).
   */
  applyRestyle(doc: RestyleDoc | null, materials: RestyleMaterial[]): void {
    this.restyle = doc;
    const byId = new Map(materials.map((m) => [m.id, m]));
    const entries = doc?.elements ?? {};

    // A restyle you cannot see is not a restyle. The splat backdrop is the
    // photograph — it cannot be re-surfaced or relit (its light is baked into
    // the gaussians), and it HIDES the shell it stands in for. So any active
    // restyle shows the reconstructed geometry instead. Measured: without
    // this, re-surfacing the shell changed the frame by 0.0 mean RGB.
    const relighting = (doc?.lighting?.preset ?? "as-captured") !== "as-captured";
    const restyled = Object.keys(entries);
    this.restyleShowsMesh = relighting || restyled.length > 0 || this.bakedLook;
    if (this.backdrop) this.backdrop.visible = !this.restyleShowsMesh;
    this.refreshElementVisibility();

    for (const el of this.elements) {
      const entry = entries[el.slug];
      const cls = entry?.material ? byId.get(entry.material) : undefined;
      const tint = entry?.tint ? new THREE.Color(entry.tint) : null;

      el.object.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (!mesh.isMesh) return;
        const lit = mesh.userData.litMaterial as THREE.MeshStandardMaterial | undefined;
        const unlit = mesh.userData.unlitMaterial as THREE.MeshBasicMaterial | undefined;

        // Restyled materials are OURS to dispose; baked ones never are.
        const previous = mesh.userData.restyleMaterial as THREE.Material | undefined;
        if (previous) {
          previous.dispose();
          mesh.userData.restyleMaterial = undefined;
        }

        if (cls) {
          const metresPerTile = entry?.material_scale ?? 1;
          const tile = this.classTile(cls);
          const mat = triplanarMaterial(
            tile, metresPerTile * this.params.unitsPerMetre, cls, tint);
          mesh.userData.restyleMaterial = mat;
          mesh.material = mat;
          return;
        }

        // No class swap: restore the baked material and re-apply any tint by
        // multiplying it into the material colour (never into the texture).
        for (const mat of [lit, unlit]) {
          if (!mat) continue;
          const baked = (mesh.userData[mat === lit ? "bakedLitColor" : "bakedUnlitColor"]
            ??= mat.color.clone()) as THREE.Color;
          mat.color.copy(tint ?? baked);
        }
        const base = this.params.unlit ? unlit : lit;
        if (base) mesh.material = base;
      });
    }

    this.applyLighting();
  }

  /**
   * Mark this world as carrying a permanently BAKED restyle. After a bake the
   * overlay document is empty, but the splat backdrop is still the
   * un-restyled photograph and it hides the shell it stands in for — so the
   * baked flag joins the restyleShowsMesh predicate and the reconstructed
   * geometry stays on top (the 0.0 mean-RGB lesson, permanent edition).
   */
  setBakedLook(baked: boolean): void {
    if (this.bakedLook === baked) return;
    this.bakedLook = baked;
    const relighting =
      (this.restyle?.lighting?.preset ?? "as-captured") !== "as-captured";
    const restyledCount = Object.keys(this.restyle?.elements ?? {}).length;
    this.restyleShowsMesh = relighting || restyledCount > 0 || this.bakedLook;
    if (this.backdrop) this.backdrop.visible = !this.restyleShowsMesh;
    this.refreshElementVisibility();
  }

  /** Cached per class id — one tile serves every element using that class. */
  private classTile(cls: RestyleMaterial): THREE.DataTexture {
    const cached = this.classTiles.get(cls.id);
    if (cached) return cached;
    const tile = classTile(cls);
    this.classTiles.set(cls.id, tile);
    return tile;
  }

  /** Drive the light rig (and the sky behind an open shell) from the preset. */
  private applyLighting(): void {
    const name = this.restyle?.lighting?.preset ?? "as-captured";
    const preset = presetFor(name, this.restyle?.lighting?.intensity ?? 1);
    const [hemi, key, fill, ambient] = this.lights.children as [
      THREE.HemisphereLight, THREE.DirectionalLight,
      THREE.DirectionalLight, THREE.AmbientLight];
    hemi.color.setHex(preset.sky);
    hemi.groundColor.setHex(preset.ground);
    hemi.intensity = preset.hemiIntensity;
    key.color.setHex(preset.key);
    key.intensity = preset.keyIntensity;
    fill.intensity = preset.keyIntensity * 0.4;
    ambient.intensity = preset.ambient;
    this.renderer.setClearColor(preset.background, 1);
    // A relight is only visible on LIT materials: choosing any preset other
    // than as-captured means "stop showing me the baked-in daylight".
    const wantLit = !preset.asCaptured;
    if (wantLit && this.params.unlit) this.setParams({ unlit: false });
    this.lights.visible = !this.params.unlit;
  }

  /* ---------------------------------------------------------------- *
   * Interactions                                                     *
   * ---------------------------------------------------------------- */

  /** Install the authored affordances and the resolved starting state. */
  setInteractions(records: InteractionRecord[], state: Record<string, string>): void {
    this.interactions = new Map(records.map((r) => [r.slug, r]));
    this.interactionState = new Map(Object.entries(state));
    for (const [slug, value] of this.interactionState) this.applyElementState(slug, value);
    this.lastTargetKey = null;
    // Interactivity earns a mesh (a toggleable lamp must be seen to be
    // toggled) — re-derive now that the walker knows the verbs.
    this.refreshElementVisibility();
    this.pollTarget();
  }

  /** Set one element's state locally (the page owns persistence). */
  setElementState(slug: string, value: string): void {
    if (!this.interactions.has(slug)) return;
    this.interactionState.set(slug, value);
    this.applyElementState(slug, value);
    this.lastTargetKey = null;
    this.pollTarget();
  }

  /**
   * Apply an effect to loaded geometry.
   *
   * Only `tint` is wired. `object.visible` already has exactly one owner — the
   * HUD's per-element eye — and a second writer would produce an object the
   * panel claims is hidden. The colour goes into BOTH cached materials because
   * the walker swaps between them for unlit mode, and MeshBasicMaterial.color
   * multiplies the baseColor map, so a tint reads correctly on a textured prop.
   */
  private applyElementState(slug: string, value: string): void {
    const record = this.interactions.get(slug);
    const el = this.elements.find((e) => e.slug === slug);
    if (!record || !el) return;

    const tint = effectFor(record, value).tint;
    el.object.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!(mesh as { isMesh?: boolean }).isMesh) return;
      for (const key of ["litMaterial", "unlitMaterial"] as const) {
        const material = mesh.userData?.[key] as THREE.MeshStandardMaterial | undefined;
        if (material?.color) material.color.set(tint ?? 0xffffff);
      }
    });
  }

  /** What the crosshair is on, or null. Only authored elements are candidates. */
  private targetUnderCrosshair(): { record: InteractionRecord; state: string } | null {
    if (!this.interactions.size) return null;

    const candidates = this.elements
      .filter((el) => el.visible && this.interactions.has(el.slug))
      .map((el) => el.object);
    if (!candidates.length) return null;

    // One ray for every candidate at the largest authored reach, then the
    // per-record range is checked on the hit — cheaper than a ray per element.
    _interactRay.far = maxReachInSceneUnits([...this.interactions.values()], this.params.unitsPerMetre);
    _interactRay.setFromCamera(_screenCentre, this.camera);

    for (const hit of _interactRay.intersectObjects(candidates, true)) {
      // The hit lands on a descendant Mesh; walk up to the element root, which
      // is where loadWorld stamped the slug.
      let node: THREE.Object3D | null = hit.object;
      while (node && !node.userData?.slug) node = node.parent;
      const slug = node?.userData?.slug as string | undefined;
      if (!slug) continue;

      const record = this.interactions.get(slug);
      if (!record) continue;
      if (hit.distance > reachInSceneUnits(record, this.params.unitsPerMetre)) continue;
      return { record, state: this.interactionState.get(slug) ?? record.initial };
    }
    return null;
  }

  /** Emit the prompt, but only when it actually changed. */
  private pollTarget(): void {
    if (!this.onTarget) return;
    const hit = this.controls.isLocked ? this.targetUnderCrosshair() : null;
    const key = hit ? `${hit.record.slug}:${hit.state}` : null;
    if (key === this.lastTargetKey) return;
    this.lastTargetKey = key;
    this.onTarget(hit ? targetInfo(hit.record, hit.state) : null);
  }

  private interact(): void {
    const hit = this.targetUnderCrosshair();
    if (!hit) return;
    // The pickup verb finally has its physical half: E lifts the prop into a
    // kinematic carry (left-click throws it, E again puts it down). The state
    // machine below still advances, so the authored held/placed state stays
    // the persisted source of truth.
    if (hit.record.verb === "pickup" && this.physics?.hasProp(hit.record.slug)) {
      if (this.physics.carrying === hit.record.slug) {
        this.physics.release(this.camera, false);
        this.onCarryChange?.(null);
      } else if (this.physics.carrying === null) {
        if (!this.physics.pickUp(hit.record.slug)) return;
        this.onCarryChange?.(hit.record.slug);
      } else {
        return; // hands full with a different prop
      }
    }
    if (!isStateful(hit.record)) return;
    const next = nextInteractionState(hit.record, hit.state);
    this.setElementState(hit.record.slug, next);
    this.onInteract?.(hit.record.slug, next);
  }

  /** Left-click while carrying = throw. Wired in the constructor. */
  private handleMouseDown = (e: MouseEvent): void => {
    if (!this.controls.isLocked || e.button !== 0) return;
    if (this.combatClick?.()) return; // game mode shot; carry-throw yields
    if (!this.physics) return;
    const slug = this.physics.carrying;
    if (!slug) return;
    this.physics.release(this.camera, true);
    this.onCarryChange?.(null);
    // A thrown prop is no longer held — flip the authored state back so the
    // prompt and the persisted record agree with what just happened.
    const record = this.interactions.get(slug);
    if (record && isStateful(record)) {
      this.setElementState(slug, record.initial);
      this.onInteract?.(slug, record.initial);
    }
  };

  /** Toggle noclip. Zeroes velocity so you do not inherit a fall on landing. */
  setFlying(flying: boolean): void {
    if (this.flying === flying) return;
    this.flying = flying;
    this.velocity.set(0, 0, 0);
    this.grounded = false;
    this.onFlyChange?.(flying);
  }

  get isFlying(): boolean {
    return this.flying;
  }

  /**
   * Free camera: no gravity, no collision, movement along the FULL look
   * direction (not the yaw-only ground vector) so looking down and pressing W
   * descends — which is the whole point when inspecting a world from outside.
   */
  private stepFly(dt: number): void {
    if (!this.controls.isLocked) return;

    this.camera.getWorldDirection(_forward);
    _right.copy(_forward).cross(_up).normalize();

    _move.set(0, 0, 0);
    if (this.keys.has("KeyW") || this.keys.has("ArrowUp")) _move.add(_forward);
    if (this.keys.has("KeyS") || this.keys.has("ArrowDown")) _move.sub(_forward);
    if (this.keys.has("KeyD") || this.keys.has("ArrowRight")) _move.add(_right);
    if (this.keys.has("KeyA") || this.keys.has("ArrowLeft")) _move.sub(_right);
    if (this.keys.has("Space")) _move.add(_up);
    if (this.keys.has("KeyC")) _move.sub(_up);

    if (_move.lengthSq() === 0) return;
    // Shift sprints here too; a scene you are inspecting is often large
    // relative to a walking pace.
    const sprint = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight")
      ? this.params.sprintMultiplier : 1;
    const speed = this.params.walkSpeedMps * this.params.unitsPerMetre * sprint;
    _move.normalize();
    this.camera.position.addScaledVector(_move, speed * dt);
  }

  private applyMaterialMode(): void {
    const unlit = this.params.unlit;
    this.worldGroup.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      // A restyled surface outranks the lit/unlit toggle: the class material
      // IS the surface now, and swapping it back would silently undo a
      // restyle every time the checkbox moved.
      if (mesh.userData.restyleMaterial) return;
      const next = unlit ? mesh.userData.unlitMaterial : mesh.userData.litMaterial;
      if (next) mesh.material = next as THREE.Material;
    });
    this.lights.visible = !unlit;
  }

  /* -------------------------------------------------------------- *
   * Input                                                           *
   * -------------------------------------------------------------- */

  requestLock(): void {
    if (!this.controls.isLocked) this.controls.lock();
  }

  private handleLock = () => this.onLockChange?.(true);
  private handleUnlock = () => {
    this.keys.clear();
    this.onLockChange?.(false);
  };
  private handleBlur = () => this.keys.clear();

  private handleKeyDown = (e: KeyboardEvent) => {
    if (!this.controls.isLocked) return;
    if (!MOVE_KEYS.has(e.code)) return;
    if (e.code === "Space") e.preventDefault();
    if (e.repeat) return;

    if (e.code === "KeyR") {
      this.respawn();
      return;
    }
    // Fly / noclip. A world whose collider is wrong is otherwise impossible to
    // inspect — you fall out of it before you can see what is wrong.
    if (e.code === "KeyF") {
      this.setFlying(!this.flying);
      return;
    }
    // Interact. Raycast on demand rather than reusing the throttled prompt
    // result, so the act is always against what is under the crosshair NOW.
    if (e.code === INTERACT_KEY) {
      this.interact();
      return;
    }
    // Live scale dialling without leaving pointer lock — the whole point of a
    // scale control on an uncalibrated capture.
    if (e.code === "BracketLeft" || e.code === "BracketRight") {
      const factor = e.code === "BracketRight" ? 1.1 : 1 / 1.1;
      this.setParams({ unitsPerMetre: clamp(this.params.unitsPerMetre * factor, 0.01, 1000) });
      this.onParams?.(this.params);
      return;
    }
    this.keys.add(e.code);
  };

  private handleKeyUp = (e: KeyboardEvent) => {
    this.keys.delete(e.code);
  };

  /* -------------------------------------------------------------- *
   * Loop                                                            *
   * -------------------------------------------------------------- */

  start(): void {
    if (this.running || this.disposed) return;
    this.running = true;
    this.clock.getDelta();
    const tick = () => {
      if (!this.running) return;
      this.rafId = requestAnimationFrame(tick);
      this.frame();
    };
    this.rafId = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
  }

  private frame(): void {
    const raw = this.clock.getDelta();
    const dt = Math.min(raw, 0.1);

    // Fly needs no collider — that is precisely when you want it, because a
    // world with a broken or missing collider is otherwise uninspectable.
    if (this.flying) {
      this.stepFly(dt);
    } else if (this.bvh) {
      const steps = 5;
      for (let i = 0; i < steps; i++) this.stepPlayer(dt / steps);
    }

    // Props simulate in fly mode too — a shoved box keeps tumbling while you
    // inspect it from the air.
    this.physics?.step(dt, this.camera.position, this.eyeHeight, this.camera);

    // A disturbed prop takes its photograph ghost with it. 4 Hz is plenty —
    // the ghost matters on the timescale of a shove, not a frame.
    this.pluckClock += raw;
    if (this.pluckClock >= 0.25) {
      this.pluckClock = 0;
      this.checkPluckDisturbed();
    }

    this.onFrame?.(dt);

    this.renderer.render(this.scene, this.camera);

    // Prompt polling is throttled deliberately. This repo has a measured prior
    // on per-frame raycasting: spark-scene-viewer hit 10 fps on a 102 ms
    // raycast until it stopped doing it every mouse move. 10 Hz is well inside
    // "feels instant" for a prompt.
    this.targetClock += raw;
    if (this.targetClock >= 0.1) {
      this.targetClock = 0;
      this.pollTarget();
    }

    this.frameCount++;
    this.statsClock += raw;
    if (this.statsClock >= 0.25) {
      this.fps = this.frameCount / this.statsClock;
      this.frameCount = 0;
      this.statsClock = 0;
      this.drawnTris = this.renderer.info.render.triangles;
      this.onStats?.({
        fps: this.fps,
        position: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
        grounded: this.grounded,
        colliderTris: this.colliderTris,
        drawnTris: this.drawnTris,
        locked: this.controls.isLocked,
      });
    }
  }

  /**
   * One physics substep. Capsule-vs-BVH sweep, following the shape of
   * three-mesh-bvh's characterMovement example: integrate, then push the
   * capsule segment out of every triangle it penetrates, then read the total
   * correction back as both the new position and the ground test.
   */
  private stepPlayer(dt: number): void {
    if (this.flying) {
      this.stepFly(dt);
      return;
    }
    const g = -this.params.gravityMps2 * this.params.unitsPerMetre;
    if (this.grounded) {
      this.velocity.y = dt * g; // keep a little downward bias so slopes stick
    } else {
      this.velocity.y += dt * g;
    }
    this.camera.position.addScaledVector(this.velocity, dt);

    if (this.controls.isLocked) {
      _euler.setFromQuaternion(this.camera.quaternion, "YXZ");
      const yaw = _euler.y;
      _forward.set(-Math.sin(yaw), 0, -Math.cos(yaw));
      _right.copy(_forward).cross(_up).normalize();

      _move.set(0, 0, 0);
      if (this.keys.has("KeyW") || this.keys.has("ArrowUp")) _move.add(_forward);
      if (this.keys.has("KeyS") || this.keys.has("ArrowDown")) _move.sub(_forward);
      if (this.keys.has("KeyD") || this.keys.has("ArrowRight")) _move.add(_right);
      if (this.keys.has("KeyA") || this.keys.has("ArrowLeft")) _move.sub(_right);

      if (_move.lengthSq() > 0) {
        const sprint = this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? this.params.sprintMultiplier : 1;
        const speed = this.params.walkSpeedMps * this.params.unitsPerMetre * sprint;
        _move.normalize();
        this.camera.position.addScaledVector(_move, speed * dt);
      }

      if (this.keys.has("Space") && this.grounded) {
        this.velocity.y = this.params.jumpSpeedMps * this.params.unitsPerMetre;
        this.grounded = false;
      }
    }

    this.resolveCollisions(dt);

    // Fell out of the world (thin floor, wild scale) — put them back.
    const drop = this.sceneBox.min.y - Math.max(this.eyeHeight * 20, 5);
    if (this.camera.position.y < drop) this.respawn();
  }

  private resolveCollisions(dt: number): void {
    const bvh = this.bvh;
    const collider = this.collider;
    if (!bvh || !collider) return;

    const radius = this.capsuleRadius;
    // Capsule in local space: player origin (camera) is the TOP sphere centre,
    // so the head clears by `radius` and the feet sit at eye - capsuleHeight
    // (the eye height clamped into the proven headroom — see capsuleHeight).
    const segLength = Math.max(1e-5, this.capsuleHeight - radius);

    _seg.start.copy(this.camera.position);
    _seg.end.copy(this.camera.position).addScaledVector(_up, -segLength);

    _mat.copy(collider.matrixWorld).invert();
    _seg.start.applyMatrix4(_mat);
    _seg.end.applyMatrix4(_mat);

    _box.makeEmpty();
    _box.expandByPoint(_seg.start);
    _box.expandByPoint(_seg.end);
    _box.min.addScalar(-radius);
    _box.max.addScalar(radius);

    bvh.shapecast({
      intersectsBounds: (box) => box.intersectsBox(_box),
      intersectsTriangle: (tri) => {
        const distance = tri.closestPointToSegment(_seg, _triPoint, _capPoint);
        if (distance < radius) {
          const depth = radius - distance;
          _dir.copy(_capPoint).sub(_triPoint);
          if (_dir.lengthSq() < 1e-20) return;
          _dir.normalize();
          _seg.start.addScaledVector(_dir, depth);
          _seg.end.addScaledVector(_dir, depth);
        }
      },
    });

    _newPos.copy(_seg.start).applyMatrix4(collider.matrixWorld);
    _delta.subVectors(_newPos, this.camera.position);

    // A correction that was mostly upward means we are standing on something.
    this.grounded = _delta.y > Math.abs(dt * this.velocity.y * 0.25);

    const offset = Math.max(0, _delta.length() - 1e-5);
    _delta.normalize().multiplyScalar(offset);
    this.camera.position.add(_delta);

    if (!this.grounded) {
      _delta.normalize();
      this.velocity.addScaledVector(_delta, -_delta.dot(this.velocity));
    } else {
      this.velocity.set(0, 0, 0);
    }
  }

  /* -------------------------------------------------------------- *
   * Lifecycle                                                       *
   * -------------------------------------------------------------- */

  resize(): void {
    const parent = this.canvas.parentElement;
    const w = Math.max(1, parent?.clientWidth || this.canvas.clientWidth || 1);
    const h = Math.max(1, parent?.clientHeight || this.canvas.clientHeight || 1);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private clearWorld(): void {
    this.disposeCollider();
    this.physics?.dispose();
    this.physics = null;
    for (const el of this.elements) {
      this.worldGroup.remove(el.object);
      disposeObject(el.object);
    }
    this.elements = [];
    this.sceneBox = new THREE.Box3();
    this.visibilityOverrides.clear();
    // The curtain belongs to the WORLD, not the walker: a new world's
    // curtain arrives from its own curtain.json, and the old one leaking
    // across loads would silently hide a fresh photograph.
    if (this.curtainEdit) {
      this.scene.remove(this.curtainEdit);
      this.curtainEdit = null;
      this.curtainSdf = null;
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.stop();
    window.removeEventListener("keydown", this.handleKeyDown);
    window.removeEventListener("keyup", this.handleKeyUp);
    window.removeEventListener("blur", this.handleBlur);
    this.canvas.removeEventListener("mousedown", this.handleMouseDown);
    this.controls.removeEventListener("lock", this.handleLock);
    this.controls.removeEventListener("unlock", this.handleUnlock);
    this.controls.disconnect();
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.clearWorld();
    // The splat backdrop is a separate lane from the world meshes: without
    // this, a disposed walker still holds the SplatMesh + SparkRenderer —
    // hundreds of MB of GPU buffers for a 1.2M-gaussian scene (review
    // finding, which the debug global made retainable).
    this.clearBackdrop();
    for (const tile of this.classTiles.values()) tile.dispose();
    this.classTiles.clear();
    this.renderer.dispose();
  }
}

/* ------------------------------------------------------------------ *
 * Helpers                                                             *
 * ------------------------------------------------------------------ */

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Strip a geometry down to position + index. mergeGeometries() refuses
 * mismatched attribute sets, and the collider needs nothing else — this makes
 * the merge total instead of "works until one GLB gains a vertex colour".
 */
function positionOnly(src: THREE.BufferGeometry): THREE.BufferGeometry {
  const out = new THREE.BufferGeometry();
  const pos = src.getAttribute("position");
  out.setAttribute("position", (pos as THREE.BufferAttribute).clone());
  if (src.index) {
    out.setIndex(src.index.clone());
  } else {
    const count = pos.count;
    const arr = count > 65535 ? new Uint32Array(count) : new Uint16Array(count);
    for (let i = 0; i < count; i++) arr[i] = i;
    out.setIndex(new THREE.BufferAttribute(arr, 1));
  }
  return out;
}

function disposeObject(root: THREE.Object3D): void {
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry?.dispose();
    for (const key of ["litMaterial", "unlitMaterial", "sourceMaterial"]) {
      const m = mesh.userData[key] as THREE.Material | undefined;
      if (!m) continue;
      const map = (m as THREE.MeshBasicMaterial).map;
      if (map) map.dispose();
      m.dispose();
    }
  });
}

export interface FrameWarning {
  slug: string;
  message: string;
}

/**
 * Probe the "all GLBs already share one frame" assumption at runtime.
 *
 * The failure mode we care about is an exporter that centres each element on
 * its own origin — then every prop's AABB centre lands at ~(0,0,0) and the
 * scene assembles as a pile. We flag: (a) an element centred at the origin
 * while the shell is not, and (b) an element whose centre falls outside the
 * shell's AABB (expanded by its own size, to tolerate props against a wall).
 */
export function verifySharedFrame(elements: LoadedElement[], shell: LoadedElement | null): FrameWarning[] {
  const out: FrameWarning[] = [];
  if (!shell) return out;

  const shellBox = shell.box.clone();
  const shellSize = shellBox.getSize(new THREE.Vector3());
  const shellCenteredAtOrigin = shellBox.getCenter(new THREE.Vector3()).length() < shellSize.length() * 0.01;

  for (const el of elements) {
    if (el.role === "shell") continue;
    const center = new THREE.Vector3(...el.center);
    const size = new THREE.Vector3(...el.size);

    if (!shellCenteredAtOrigin && center.length() < size.length() * 0.05) {
      out.push({
        slug: el.slug,
        message: `${el.slug}: AABB centre is at the origin — this GLB looks object-centred, not world-placed.`,
      });
      continue;
    }
    const tolerant = shellBox.clone().expandByVector(size);
    if (!tolerant.containsPoint(center)) {
      out.push({
        slug: el.slug,
        message: `${el.slug}: centre (${fmt3(center)}) lies outside the shell AABB — coordinate frames may differ.`,
      });
    }
  }
  return out;
}

function fmt3(v: THREE.Vector3): string {
  return `${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)}`;
}
