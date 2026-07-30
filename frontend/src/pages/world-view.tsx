// /world/:jobId — first-person walkable viewer for a job's `_world/` bundle.
//
// The scene solidifier emits a textured shell.glb plus one GLB per element, all
// already sharing a single coordinate frame. This page assembles them, builds a
// BVH collider from the static geometry and drops you inside with WASD + mouse
// look, gravity and jump.
//
// The capture is UNCALIBRATED (world_manifest.json ships meters_per_unit: null),
// so every human dimension here is expressed in METRES and converted to scene
// units through one dial: `unitsPerMetre`. Nothing is hardcoded in scene units.
// Press [ and ] while walking to dial it live until the walk feels right.
//
// Engine lives in @/lib/world-walker (three.js + three-mesh-bvh); this file is
// the HUD and the wiring.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { TargetInfo } from "@/lib/world-interactions";
import { Link, useRoute } from "wouter";
import { uploadPolishedWorldElement , bakeWorldRestyle, fetchWorldInteractions, fetchWorldNavmesh, fetchWorldPluck, fetchWorldRestyle, fetchWorldScenario, removePlacedWorldElement, resetWorldRestyle, revertWorldRestyleBake, setWorldElementState, setWorldPlayerPoses, setWorldRestyle, solidifyWorld, uploadPlacedWorldElement} from "@/lib/api";
import { LIGHT_PRESETS, emptyRestyle, type RestyleDoc, type RestyleEntry, type RestyleMaterial } from "@/lib/world-restyle";
import { WorldGame, type GameHudState, type Scenario } from "@/lib/world-game";
import type { parseNavmesh } from "@/lib/world-navmesh";
import { PolishUploadZone } from "@/components/workspace/polish-upload";
import {
  DEFAULT_WALK_PARAMS,
  WorldWalker,
  apiWorldSource,
  localWorldSource,
  type LoadedElement,
  type WalkParams,
  type WalkerStats,
  type WorldManifest,
  type WorldSource,
} from "@/lib/world-walker";
import {
  AlertTriangle,
  ArrowLeft,
  Box,
  Eye,
  EyeOff,
  Footprints,
  Gauge,
  Hammer,
  Layers,
  Loader2,
  MapPin,
  Ruler,
  Settings2,
  Palette,
  Shapes,
  UploadCloud,
  Wand2,
} from "lucide-react";

const DEV_SOURCE_KEY = "splatlab.world-view.devSource";

/** Snapshot of an engine element, safe to hold in React state. */
interface ElementRow {
  slug: string;
  label: string;
  role: string;
  faces: number;
  tris: number;
  size: [number, number, number];
  center: [number, number, number];
  collides: boolean;
  frameWarning: string | null;
}

interface SceneInfo {
  shellHeightUnits: number;
  sceneSize: [number, number, number];
  colliderTris: number;
  colliderSource: "collision_shell" | "collision_shell+authored" | "visual_shell";
  metersPerUnit: number | null;
}

type Phase = "idle" | "loading" | "ready" | "error";

function toRow(e: LoadedElement): ElementRow {
  return {
    slug: e.slug,
    label: e.label,
    role: e.role,
    faces: e.faces,
    tris: e.tris,
    size: e.size,
    center: e.center,
    collides: e.collides,
    frameWarning: e.frameWarning,
  };
}

export default function WorldViewPage() {
  const [, params] = useRoute("/world/:jobId");
  const routeJobId = params?.jobId ?? "";
  // Fallback so the page still finds its job if it gets mounted under a
  // different route pattern than the one documented.
  const jobId = routeJobId || window.location.pathname.split("/").filter(Boolean).pop() || "";

  // A local static directory overrides the API route — this is how the page is
  // usable today, before the backend /world/file handler lands.
  const [devSource, setDevSource] = useState<string>(() => {
    const q = new URLSearchParams(window.location.search).get("src");
    if (q) return q;
    return localStorage.getItem(DEV_SOURCE_KEY) ?? "";
  });
  const [pendingSource, setPendingSource] = useState(devSource);
  const [reloadNonce, setReloadNonce] = useState(0);

  const source: WorldSource = useMemo(
    () => (devSource.trim() ? localWorldSource(devSource.trim()) : apiWorldSource(jobId)),
    [devSource, jobId],
  );

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const walkerRef = useRef<WorldWalker | null>(null);
  const [target, setTarget] = useState<TargetInfo | null>(null);
  const [flying, setFlying] = useState(false);
  const [backdrop, setBackdrop] = useState(true);

  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ loaded: number; total: number; label: string } | null>(null);
  const [manifest, setManifest] = useState<WorldManifest | null>(null);
  const [rows, setRows] = useState<ElementRow[]>([]);
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());
  const [warnings, setWarnings] = useState<string[]>([]);
  const gameRef = useRef<WorldGame | null>(null);
  const gameDataRef = useRef<{
    navmeshDoc: Parameters<typeof parseNavmesh>[0];
    scenario: Scenario;
  } | null>(null);
  const [gameHud, setGameHud] = useState<GameHudState | null>(null);
  const [gameReady, setGameReady] = useState(false);
  const [restyle, setRestyle] = useState<RestyleDoc | null>(null);
  const [restyleMaterials, setRestyleMaterials] = useState<RestyleMaterial[]>([]);
  const [restyleBusy, setRestyleBusy] = useState(false);
  const [damageFlash, setDamageFlash] = useState(0);
  const [hitMarker, setHitMarker] = useState(0);
  const [sceneInfo, setSceneInfo] = useState<SceneInfo | null>(null);
  const [walkParams, setWalkParams] = useState<WalkParams>(DEFAULT_WALK_PARAMS);
  /** Slugs whose atlas carries a permanently baked restyle (server markers). */
  const [bakedElements, setBakedElements] = useState<string[]>([]);
  const [stats, setStats] = useState<WalkerStats>({
    fps: 0,
    position: [0, 0, 0],
    grounded: false,
    colliderTris: 0,
    drawnTris: 0,
    locked: false,
  });
  const [locked, setLocked] = useState(false);
  const [panelsOpen, setPanelsOpen] = useState(true);

  /* ---------------------------------------------------------------- *
   * Engine lifecycle. StrictMode double-invokes this in dev, so the   *
   * async load is guarded by `cancelled` and the walker is disposed.  *
   * ---------------------------------------------------------------- */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    const controller = new AbortController();
    const walker = new WorldWalker(canvas);
    walkerRef.current = walker;
    // Live-proof hook: tools/prove-*.py probe ground truth through this.
    (window as unknown as { __worldWalker?: WorldWalker }).__worldWalker = walker;
    walker.onStats = (s) => {
      if (!cancelled) setStats(s);
    };
    walker.onLockChange = (l) => {
      if (!cancelled) setLocked(l);
    };
    walker.onParams = (p) => {
      if (!cancelled) setWalkParams({ ...p });
    };
    walker.onTarget = (t) => {
      if (!cancelled) setTarget(t);
    };
    walker.onFlyChange = (f) => {
      if (!cancelled) setFlying(f);
    };
    // Persist optimistically: the walker has already applied it locally, so a
    // failed save must not leave the world showing a state the server rejected.
    walker.onInteract = (slug, next) => {
      if (source.kind !== "api") return;
      void setWorldElementState(jobId, slug, next).catch(() => {
        if (cancelled) return;
        setWarnings((w) => [...w, `Could not save the state of ${slug}.`]);
      });
    };

    setPhase("loading");
    setError(null);
    setWarnings([]);
    setProgress(null);

    (async () => {
      try {
        const m = await WorldWalker.fetchManifest(source, controller.signal);
        if (cancelled) return;
        setManifest(m);

        const result = await walker.loadWorld(source, m, {
          signal: controller.signal,
          onProgress: (p) => {
            if (!cancelled) setProgress(p);
          },
        });
        if (cancelled) return;

        // Adopt the scale the manifest implies (or the storey-height guess).
        walker.setParams({ unitsPerMetre: result.suggestedUnitsPerMetre });
        walker.respawn();
        setWalkParams({ ...walker.params });

        const shell = result.elements.find((e) => e.role === "shell");
        const { min, max } = result.sceneBox;
        const size: [number, number, number] = [max.x - min.x, max.y - min.y, max.z - min.z];
        setSceneInfo({
          shellHeightUnits: shell ? shell.size[1] : size[1],
          sceneSize: size,
          colliderTris: result.colliderTris,
          colliderSource: result.colliderSource,
          metersPerUnit: m.meters_per_unit ?? null,
        });
        setRows(result.elements.map(toRow));
        setHidden(new Set());
        setWarnings(result.frameWarnings);

        // Interactions are optional: a world with none authored must still
        // load, so every failure here is non-fatal.
        try {
          const authored = await fetchWorldInteractions(jobId, controller.signal);
          if (!cancelled && authored) {
            walker.setInteractions(
              authored.interactions?.elements ?? [],
              authored.resolved?.applied ?? {},
            );
            const lost = authored.resolved?.dropped ?? [];
            if (lost.length) {
              setWarnings((w) => [
                ...w,
                ...lost.map((d) => `Saved state for "${d.slug}" was dropped: ${d.reason}.`),
              ]);
            }
            if (authored.state_error) setWarnings((w) => [...w, authored.state_error]);
            // Physics save-game: put shoved props back where they were left.
            const poses = authored.resolved?.player?.physics?.poses;
            if (poses && Object.keys(poses).length) walker.applyPhysicsPoses(poses);
          }
        } catch {
          /* no interactions authored, or the route is unavailable */
        }

        // Persist disturbed prop poses: every 5 s when something changed, and
        // best-effort on teardown. Deliberately quiet — a failed save shows up
        // as a warning once, not a spam stream.
        if (source.kind === "api") {
          // "{}" not "": opening a world must never fire a gratuitous save —
          // on a duplicated job that first write would replace the copied
          // foreign save before the player touched anything (review finding).
          let lastSaved = "{}";
          let warned = false;
          const savePoses = () => {
            const poses = walker.getPhysicsPoses();
            if (!poses) return;
            const encoded = JSON.stringify(poses);
            if (encoded === lastSaved) return;
            void setWorldPlayerPoses(jobId, poses)
              .then(() => {
                // Commit the dedupe key only on SUCCESS: a transient failure
                // must retry on the next tick, not silently drop the pose
                // for the rest of the session (review finding).
                lastSaved = encoded;
              })
              .catch(() => {
                if (!warned) {
                  warned = true;
                  setWarnings((w) => [...w, "Prop positions could not be saved."]);
                }
              });
          };
          const poseTimer = window.setInterval(savePoses, 5000);
          const stopSaving = () => {
            window.clearInterval(poseTimer);
            savePoses();
          };
          controller.signal.addEventListener("abort", stopSaving, { once: true });
        }

        // Game data (optional): a scenario + navmesh make the world playable.
        if (source.kind === "api") {
          void Promise.all([
            fetchWorldScenario(jobId, controller.signal),
            fetchWorldNavmesh(jobId, controller.signal),
          ]).then(([scenarioPayload, navmeshDoc]) => {
            if (cancelled) return;
            const scenario = scenarioPayload.scenario as Scenario | null;
            if (scenario && navmeshDoc && Array.isArray((navmeshDoc as { rows?: unknown }).rows)) {
              gameDataRef.current = {
                navmeshDoc: navmeshDoc as Parameters<typeof parseNavmesh>[0],
                scenario,
              };
              setGameReady(true);
            }
          }).catch(() => { /* no scenario/navmesh: the world simply is not playable yet */ });

          // The stored look. A world with no restyle.json simply reads as
          // "as captured" — never a reason to fail the load.
          void fetchWorldRestyle(jobId, controller.signal).then((payload) => {
            if (cancelled) return;
            setRestyle(payload.restyle);
            setRestyleMaterials(payload.materials);
            const baked = payload.baked_elements ?? [];
            setBakedElements(baked);
            // Before the backdrop loads: a baked look must keep the mesh in
            // front of the un-restyled photograph.
            walker.setBakedLook(baked.length > 0);
            walker.applyRestyle(payload.restyle, payload.materials);
          }).catch(() => { /* no world yet, or restyle unavailable */ });
        }

        // The splat backdrop. The mesh is what you collide with and act on;
        // the splat is what you look at. Measured: a shell that encloses the
        // capture cameras cannot resemble it, and the ground TIN alone covers
        // ~40% of a frame — trees, hedge and sky come from the splat or from
        // nowhere.
        if (source.kind === "api") {
          // Pluck rows address the RAW splat's row order and fmt=web is
          // decimated — so the backdrop upgrades to fmt=langweb (row-identical;
          // the server falls back to the raw ply) only when a FRESH pluck doc
          // exists. A stale doc never chooses rows: wrong rows would pluck the
          // wrong gaussians, and wrong is worse than none.
          void fetchWorldPluck(jobId)
            .catch(() => null)
            .then((pluckRes) => {
              if (cancelled) return undefined;
              const fresh = pluckRes && pluckRes.ok && !pluckRes.stale ? pluckRes.pluck : null;
              if (fresh) walker.setPluckDoc(fresh);
              if (pluckRes?.stale) {
                setWarnings((w) => [...w,
                  "Pluck data is stale — moved props keep their splat ghosts (rebuild via POST /world/pluck)."]);
              }
              return walker.setBackdrop(
                `/api/splat/jobs/${encodeURIComponent(jobId)}/preview/file?fmt=${fresh ? "langweb" : "web"}`,
                m.meters_per_unit ?? null,
              );
            })
            .catch(() => {
              if (!cancelled) setWarnings((w) => [...w, "Splat backdrop failed to load."]);
            });
        }

        setPhase("ready");
        walker.start();
      } catch (err) {
        if (cancelled) return;
        setError((err as Error).message || String(err));
        setPhase("error");
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
      gameRef.current?.dispose();
      gameRef.current = null;
      setGameHud(null);
      setGameReady(false);
      walker.dispose();
      if (walkerRef.current === walker) walkerRef.current = null;
      const w = window as unknown as { __worldWalker?: WorldWalker };
      if (w.__worldWalker === walker) delete w.__worldWalker;
    };
  }, [source, reloadNonce]);

  /** Apply a restyle locally at once, then persist it. The walker is the
   *  preview; the document is the truth — a failed save says so and rolls
   *  the panel back to what the server still holds. */
  const commitRestyle = useCallback((next: RestyleDoc) => {
    const walker = walkerRef.current;
    setRestyle(next);
    walker?.applyRestyle(next, restyleMaterials);
    if (source.kind !== "api") return;
    setRestyleBusy(true);
    void setWorldRestyle(jobId, { elements: next.elements, lighting: next.lighting })
      .then((payload) => {
        setRestyle(payload.restyle);
        walker?.applyRestyle(payload.restyle, payload.materials);
      })
      .catch((e: unknown) => {
        setWarnings((w) => [...w,
          `Restyle not saved: ${e instanceof Error ? e.message : String(e)}`]);
        void fetchWorldRestyle(jobId).then((payload) => {
          setRestyle(payload.restyle);
          walker?.applyRestyle(payload.restyle, payload.materials);
        }).catch(() => undefined);
      })
      .finally(() => setRestyleBusy(false));
  }, [jobId, restyleMaterials, source.kind]);

  const clearRestyle = useCallback(() => {
    const walker = walkerRef.current;
    const empty = emptyRestyle(jobId);
    setRestyle(empty);
    walker?.applyRestyle(empty, restyleMaterials);
    if (source.kind !== "api") return;
    setRestyleBusy(true);
    void resetWorldRestyle(jobId)
      .then((payload) => {
        setRestyle(payload.restyle);
        walker?.applyRestyle(payload.restyle, payload.materials);
      })
      .catch(() => setWarnings((w) => [...w, "Could not clear the restyle."]))
      .finally(() => setRestyleBusy(false));
  }, [jobId, restyleMaterials, source.kind]);

  /** Bake is destructive-ish (priors are versioned server-side), so this is
   *  pessimistic on purpose — nothing applies locally; on success the world
   *  reloads and the walker fetches the re-baked GLBs from disk. */
  const bakeRestyle = useCallback(() => {
    if (source.kind !== "api") return;
    setRestyleBusy(true);
    void bakeWorldRestyle(jobId, { units_per_metre: walkParams.unitsPerMetre })
      .then(() => setReloadNonce((n) => n + 1))
      .catch((e: unknown) => setWarnings((w) => [...w,
        `Bake failed: ${e instanceof Error ? e.message : String(e)}`]))
      .finally(() => setRestyleBusy(false));
  }, [jobId, source.kind, walkParams.unitsPerMetre]);

  const revertBake = useCallback(() => {
    if (source.kind !== "api") return;
    setRestyleBusy(true);
    void revertWorldRestyleBake(jobId)
      .then(() => setReloadNonce((n) => n + 1))
      .catch((e: unknown) => setWarnings((w) => [...w,
        `Revert failed: ${e instanceof Error ? e.message : String(e)}`]))
      .finally(() => setRestyleBusy(false));
  }, [jobId, source.kind]);

  /** Land an authored asset through the create door, surface its warnings,
   *  and reload so the walker fetches the new element. Rejection messages
   *  come back verbatim for the upload zone to display. */
  const placeAsset = useCallback(async (
    slug: string, role: string, label: string, file: File,
    onProgress: (pct: number) => void,
  ) => {
    const result = await uploadPlacedWorldElement(
      jobId, slug, role, label, file, onProgress);
    if (result.warnings?.length) {
      setWarnings((w) => [...w,
        ...result.warnings.map((x) => `Placed '${slug}': ${x}`)]);
    }
    setReloadNonce((n) => n + 1);
  }, [jobId]);

  const removePlaced = useCallback((slug: string) => {
    void removePlacedWorldElement(jobId, slug)
      .then(() => setReloadNonce((n) => n + 1))
      .catch((e: unknown) => setWarnings((w) => [...w,
        `Remove failed: ${e instanceof Error ? e.message : String(e)}`]));
  }, [jobId]);

  const startGame = useCallback(() => {
    const walker = walkerRef.current;
    const data = gameDataRef.current;
    if (!walker || !data) return;
    // A fresh game per start: unitsPerMetre is baked into actor speeds and
    // reaches at construction, and the scale dial may have moved since the
    // last round (review finding) — rebuilding is idle-time-cheap.
    gameRef.current?.dispose();
    const game = new WorldGame({
      scene: walker.scene,
      camera: walker.camera,
      navmeshDoc: data.navmeshDoc,
      scenario: data.scenario,
      unitsPerMetre: walker.params.unitsPerMetre,
      groundAt: (x, z, ref, band) => walker.surfaceNear(x, z, ref, band),
    });
    game.onHud = (h) => {
      setGameHud(h);
      // The endgame dialog needs a cursor: release the pointer lock the
      // moment the game decides (review finding — the dialog was
      // unclickable behind a locked pointer).
      if (h.phase === "won" || h.phase === "lost") walker.controls.unlock();
    };
    game.onPlayerHit = () => {
      setDamageFlash(1);
      window.setTimeout(() => setDamageFlash(0), 220);
    };
    game.onHitMarker = () => {
      setHitMarker((n) => n + 1);
      window.setTimeout(() => setHitMarker(0), 130);
    };
    game.onNotice = (message) => setWarnings((w) => [...w, message]);
    walker.onFrame = (dt) => game.update(dt);
    walker.combatClick = () => game.shoot();
    gameRef.current = game;
    game.start();
    walker.requestLock();
  }, []);

  const stopGame = useCallback(() => {
    gameRef.current?.stop();
  }, []);

  /* ---------------------------------------------------------------- */

  const updateParams = useCallback((patch: Partial<WalkParams>) => {
    walkerRef.current?.setParams(patch);
    setWalkParams((prev) => ({ ...prev, ...patch }));
    if (patch.collideProps !== undefined) {
      const w = walkerRef.current;
      if (w) setRows(w.elements.map(toRow));
    }
  }, []);

  const toggleVisible = useCallback((slug: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      const nowHidden = !next.has(slug);
      if (nowHidden) next.add(slug);
      else next.delete(slug);
      walkerRef.current?.setElementVisible(slug, !nowHidden);
      return next;
    });
  }, []);

  const applyDevSource = useCallback(() => {
    const v = pendingSource.trim();
    setDevSource(v);
    if (v) localStorage.setItem(DEV_SOURCE_KEY, v);
    else localStorage.removeItem(DEV_SOURCE_KEY);
    setReloadNonce((n) => n + 1);
  }, [pendingSource]);

  const upm = walkParams.unitsPerMetre;
  const sceneHeightM = sceneInfo ? sceneInfo.shellHeightUnits / upm : null;
  const eyeUnits = walkParams.eyeHeightM * upm;

  // "This shell is N metres tall" -> solve for the dial. The most intuitive
  // calibration a human can give an uncalibrated capture.
  const calibrateByHeight = useCallback(
    (metres: number) => {
      if (!sceneInfo || !Number.isFinite(metres) || metres <= 0) return;
      updateParams({ unitsPerMetre: clamp(sceneInfo.shellHeightUnits / metres, 0.01, 1000) });
    },
    [sceneInfo, updateParams],
  );

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[#05070d] text-ink">
      <div className="absolute inset-0">
        <canvas ref={canvasRef} className="block h-full w-full outline-none" />
      </div>

      {/* Crosshair */}
      {phase === "ready" && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <div className="h-4 w-px bg-white/60" />
          <div className="absolute left-1/2 top-1/2 h-px w-4 -translate-x-1/2 -translate-y-1/2 bg-white/60" />
          {hitMarker > 0 && (
            <div className="absolute left-1/2 top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rotate-45">
              <div className="absolute left-0 top-0 h-1.5 w-1.5 border-l-2 border-t-2 border-red-400" />
              <div className="absolute right-0 top-0 h-1.5 w-1.5 border-r-2 border-t-2 border-red-400" />
              <div className="absolute bottom-0 left-0 h-1.5 w-1.5 border-b-2 border-l-2 border-red-400" />
              <div className="absolute bottom-0 right-0 h-1.5 w-1.5 border-b-2 border-r-2 border-red-400" />
            </div>
          )}
        </div>
      )}

      {phase === "ready" && flying && (
        <div className="pointer-events-none absolute left-1/2 top-4 -translate-x-1/2">
          <span className="rounded bg-cyan-500/20 px-2 py-0.5 font-mono text-[11px] text-cyan-200 ring-1 ring-cyan-400/40">
            FLY — no gravity, no collision · F to land
          </span>
        </div>
      )}

      {/* ---- Game mode HUD ---- */}
      {phase === "ready" && damageFlash > 0 && (
        <div className="pointer-events-none absolute inset-0 bg-red-600/25 transition-opacity" />
      )}
      {phase === "ready" && gameReady && (!gameHud || gameHud.phase === "idle") && (
        <div className="absolute inset-x-0 bottom-5 z-40 flex justify-center">
          <button
            type="button"
            onClick={startGame}
            className="rounded-xl border border-red-400/40 bg-red-950/70 px-4 py-2 text-sm font-semibold text-red-200 backdrop-blur hover:border-red-300 hover:text-white"
          >
            ▶ {gameDataRef.current?.scenario.name ?? "Play scenario"}
          </button>
        </div>
      )}
      {phase === "ready" && gameHud && gameHud.phase !== "idle" && (
        <div className="pointer-events-none absolute inset-x-0 bottom-5 flex justify-center">
          <div className="flex items-center gap-4 rounded-xl bg-black/70 px-4 py-2 font-mono text-xs text-zinc-200 backdrop-blur">
            <span>
              HP
              <span className="ml-2 inline-block h-2 w-28 overflow-hidden rounded bg-zinc-800 align-middle">
                <span
                  className={`block h-full ${gameHud.health / gameHud.maxHealth > 0.35 ? "bg-emerald-400" : "bg-red-500"}`}
                  style={{ width: `${(100 * gameHud.health) / gameHud.maxHealth}%` }}
                />
              </span>
              <span className="ml-2">{gameHud.health}</span>
            </span>
            <span>wave {Math.min(gameHud.wave, gameHud.totalWaves)}/{gameHud.totalWaves}</span>
            <span>kills {gameHud.kills}</span>
            {gameHud.phase === "rest" && <span className="text-cyan-300">next wave in {gameHud.restSeconds}s</span>}
            {gameHud.phase === "wave" && <span className="text-red-300">{gameHud.alive} undead</span>}
          </div>
        </div>
      )}
      {phase === "ready" && gameHud && (gameHud.phase === "won" || gameHud.phase === "lost") && (
        <div className="absolute inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="rounded-2xl border border-white/10 bg-zinc-950/90 px-8 py-6 text-center backdrop-blur">
            <div className={`text-2xl font-bold ${gameHud.phase === "won" ? "text-emerald-300" : "text-red-300"}`}>
              {gameHud.phase === "won" ? "YOU SURVIVED" : "OVERRUN"}
            </div>
            <div className="mt-1 font-mono text-xs text-zinc-400">
              {gameHud.kills} zombies down · wave {Math.min(gameHud.wave, gameHud.totalWaves)}/{gameHud.totalWaves}
            </div>
            <div className="mt-4 flex justify-center gap-3">
              <button
                type="button"
                onClick={startGame}
                className="rounded-lg border border-emerald-400/40 bg-emerald-950/60 px-4 py-1.5 text-sm font-semibold text-emerald-200 hover:text-white"
              >
                Play again
              </button>
              <button
                type="button"
                onClick={stopGame}
                className="rounded-lg border border-white/10 bg-zinc-900 px-4 py-1.5 text-sm text-zinc-300 hover:text-white"
              >
                Back to walking
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Interaction prompt — pointer-events-none so it cannot eat click-to-lock */}
      {phase === "ready" && target && (
        <div className="pointer-events-none absolute left-1/2 top-1/2 mt-8 -translate-x-1/2 text-center">
          <div className="inline-flex items-center gap-2 rounded-lg bg-black/60 px-3 py-1.5 text-sm text-white/90 backdrop-blur-sm">
            {target.nextState && <Kbd>E</Kbd>}
            <span>{target.prompt}</span>
          </div>
          {target.text && (
            <div className="mt-1.5 max-w-sm rounded-lg bg-black/50 px-3 py-1.5 text-xs text-white/70 backdrop-blur-sm">
              {target.text}
            </div>
          )}
        </div>
      )}

      {/* ---- Top bar ---- */}
      <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start justify-between gap-3 p-4">
        <div className="pointer-events-auto flex items-center gap-3">
          <Link
            href={`/view/${jobId}`}
            className="flex items-center gap-2 rounded-xl border border-white/10 bg-black/60 px-3 py-2 text-xs font-semibold text-zinc-300 backdrop-blur hover:border-accent/50 hover:text-white"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Scene
          </Link>
          <div className="rounded-xl border border-white/10 bg-black/60 px-3 py-2 backdrop-blur">
            <p className="text-[10px] uppercase tracking-[0.25em] text-zinc-500">Walkable world</p>
            <p className="font-mono text-xs text-white">{jobId || "—"}</p>
          </div>
        </div>

        {phase === "ready" && (
          <StatsHud stats={stats} upm={upm} sceneHeightM={sceneHeightM} elements={rows.length} />
        )}
      </div>

      {/* ---- Frame-assumption warnings ---- */}
      {warnings.length > 0 && (
        <div className="pointer-events-auto absolute left-1/2 top-20 w-[min(680px,92vw)] -translate-x-1/2 rounded-xl border border-amber-500/40 bg-amber-950/80 p-3 text-xs text-amber-100 backdrop-blur">
          <p className="mb-1 flex items-center gap-2 font-semibold">
            <AlertTriangle className="h-4 w-4" />
            Shared-coordinate-frame check flagged {warnings.length}{" "}
            {warnings.length === 1 ? "element" : "elements"}
          </p>
          <ul className="list-disc space-y-0.5 pl-5">
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ---- Panels ---- */}
      {phase === "ready" && (
        <>
          <button
            type="button"
            onClick={() => setPanelsOpen((v) => !v)}
            className="pointer-events-auto absolute bottom-4 left-4 z-20 flex items-center gap-2 rounded-xl border border-white/10 bg-black/70 px-3 py-2 text-xs font-semibold text-zinc-300 backdrop-blur hover:border-accent/50 hover:text-white"
          >
            <Settings2 className="h-3.5 w-3.5" />
            {panelsOpen ? "Hide panels" : "Show panels"}
          </button>

          {panelsOpen && (
            <div
              className={`absolute inset-y-0 left-0 flex w-[320px] flex-col gap-3 overflow-y-auto p-4 pb-16 pt-24 transition-opacity ${
                locked ? "pointer-events-none opacity-40" : "pointer-events-auto opacity-100"
              }`}
            >
              <ScalePanel
                params={walkParams}
                onChange={updateParams}
                sceneInfo={sceneInfo}
                sceneHeightM={sceneHeightM}
                eyeUnits={eyeUnits}
                onCalibrateByHeight={calibrateByHeight}
              />
              <MovementPanel params={walkParams} onChange={updateParams} />
              {restyle && (
                <RestylePanel
                  restyle={restyle}
                  materials={restyleMaterials}
                  rows={rows}
                  busy={restyleBusy}
                  api={source.kind === "api"}
                  baked={bakedElements.length > 0}
                  onChange={commitRestyle}
                  onClear={clearRestyle}
                  onBake={bakeRestyle}
                  onRevert={revertBake}
                />
              )}
              {source.kind === "api" && manifest && (
                <PlacedPanel
                  manifest={manifest}
                  onPlace={placeAsset}
                  onRemove={removePlaced}
                />
              )}
              <Panel icon={<Layers className="h-3.5 w-3.5" />} title="Backdrop">
                <label className="flex items-start gap-2 text-[11px] text-zinc-300">
                  <input
                    type="checkbox"
                    checked={backdrop}
                    onChange={(e) => {
                      setBackdrop(e.target.checked);
                      const w = walkerRef.current;
                      if (!w) return;
                      void w.setBackdrop(
                        e.target.checked
                          ? `/api/splat/jobs/${encodeURIComponent(jobId)}/preview/file?fmt=web`
                          : null,
                        sceneInfo?.metersPerUnit ?? null,
                      );
                    }}
                    className="mt-0.5"
                  />
                  <span>
                    Show the original splat
                    <span className="block text-[10px] leading-snug text-zinc-500">
                      The mesh is what you collide with; the splat is what you look
                      at. Trees, hedge and sky exist only in the splat. While this
                      is on, the shell is hidden — it is a blocky solid whose job
                      is collision, and drawing it just boxes you in. You still
                      cannot fall through it.
                    </span>
                  </span>
                </label>
              </Panel>
              <RebuildPanel jobId={jobId} onRebuilt={() => setReloadNonce((n) => n + 1)} />
            </div>
          )}

          {panelsOpen && (
            /* pt clears the taller stats HUD sitting in the top bar above it */
            <div
              className={`absolute inset-y-0 right-0 flex w-[320px] flex-col gap-3 overflow-y-auto p-4 pb-16 pt-[172px] transition-opacity ${
                locked ? "pointer-events-none opacity-40" : "pointer-events-auto opacity-100"
              }`}
            >
              <ElementsPanel
                rows={rows}
                hidden={hidden}
                upm={upm}
                colliderTris={sceneInfo?.colliderTris ?? 0}
                colliderSource={sceneInfo?.colliderSource ?? "visual_shell"}
                jobId={jobId}
                onToggle={toggleVisible}
                onGo={(slug) => walkerRef.current?.teleportTo(slug)}
                onPolished={() => setReloadNonce((n) => n + 1)}
              />
            </div>
          )}
        </>
      )}

      {/* ---- Click-to-walk ----
           Suppressed while the endgame dialog is up: at z-10 it painted OVER
           the dialog and swallowed the Play-again click (review finding). */}
      {phase === "ready" && !locked
        && !(gameHud && (gameHud.phase === "won" || gameHud.phase === "lost")) && (
        <button
          type="button"
          onClick={() => walkerRef.current?.requestLock()}
          className="absolute inset-x-0 top-1/2 z-10 mx-auto flex w-[min(440px,90vw)] -translate-y-1/2 flex-col items-center gap-3 rounded-2xl border border-white/10 bg-black/75 px-6 py-6 text-center backdrop-blur transition hover:border-accent/60"
        >
          <Footprints className="h-7 w-7 text-accent" />
          <p className="display text-lg font-black text-white">Click to walk</p>
          <div className="grid w-full grid-cols-2 gap-x-4 gap-y-1 text-[11px] text-zinc-400">
            <Key k="W A S D" v="move" />
            <Key k="Mouse" v="look" />
            <Key k="Shift" v="sprint" />
            <Key k="Space" v="jump" />
            <Key k="[ ]" v="scale down / up" />
            <Key k="R" v="respawn" />
          <Key k="F" v="fly / noclip" />
          <Key k="Space / C" v="fly up / down" />
            <Key k="Esc" v="release mouse" />
            <Key k="" v="" />
          </div>
        </button>
      )}

      {/* ---- Loading ---- */}
      {phase === "loading" && (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 bg-[#05070d]/90">
          <Loader2 className="h-7 w-7 animate-spin text-accent" />
          <p className="text-sm font-semibold text-white">
            {progress ? `Loading ${progress.label}…` : "Fetching world manifest…"}
          </p>
          {progress && (
            <>
              <div className="h-1 w-64 overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full bg-accent transition-[width]"
                  style={{ width: `${(progress.loaded / Math.max(1, progress.total)) * 100}%` }}
                />
              </div>
              <p className="font-mono text-[11px] text-zinc-500">
                {progress.loaded}/{progress.total} GLB
              </p>
            </>
          )}
          <p className="max-w-md text-center font-mono text-[10px] text-zinc-600">{source.describe}</p>
        </div>
      )}

      {/* ---- Error ---- */}
      {phase === "error" && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-[#05070d]/95 p-6">
          <div className="w-[min(620px,94vw)] rounded-2xl border border-red-500/40 bg-red-950/40 p-5">
            <p className="mb-2 flex items-center gap-2 text-sm font-black text-red-200">
              <AlertTriangle className="h-4 w-4" />
              Could not load this world
            </p>
            <p className="mb-3 break-words font-mono text-xs text-red-100/90">{error}</p>
            <p className="mb-3 text-xs text-zinc-400">
              The backend route <code className="font-mono text-zinc-300">{source.describe}</code> may not be
              wired up yet. Point the viewer at a static directory instead — anything served over HTTP that
              contains <code className="font-mono">world_manifest.json</code>,{" "}
              <code className="font-mono">shell.glb</code> and <code className="font-mono">elements/</code>.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={pendingSource}
                onChange={(e) => setPendingSource(e.target.value)}
                placeholder="/world-dev/   (or leave blank to use the API route)"
                className="min-w-[260px] flex-1 rounded-xl border border-white/10 bg-black/50 px-3 py-2 font-mono text-xs text-white outline-none focus:border-accent/60"
              />
              <button
                type="button"
                onClick={applyDevSource}
                className="rounded-xl bg-accent px-3 py-2 text-xs font-semibold text-accent-ink hover:bg-accent-hover"
              >
                Load
              </button>
              <button
                type="button"
                onClick={() => setReloadNonce((n) => n + 1)}
                className="rounded-xl border border-white/15 px-3 py-2 text-xs font-semibold text-zinc-300 hover:text-white"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ---- Dev source hint (visible once loaded from a local dir) ---- */}
      {phase === "ready" && devSource && !locked && (
        <p className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 rounded-lg border border-white/10 bg-black/60 px-3 py-1 font-mono text-[10px] text-zinc-500 backdrop-blur">
          local source: {devSource}
        </p>
      )}

      {manifest && phase === "ready" && manifest.meters_per_unit == null && !locked && (
        <p className="pointer-events-none absolute bottom-12 left-1/2 -translate-x-1/2 rounded-lg border border-amber-500/30 bg-amber-950/50 px-3 py-1 text-[10px] text-amber-200 backdrop-blur">
          Uncalibrated capture (meters_per_unit: null) — the scale below is a guess, dial it in.
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ *
 * HUD pieces                                                          *
 * ------------------------------------------------------------------ */

function StatsHud({
  stats,
  upm,
  sceneHeightM,
  elements,
}: {
  stats: WalkerStats;
  upm: number;
  sceneHeightM: number | null;
  elements: number;
}) {
  return (
    <div className="pointer-events-none rounded-xl border border-white/10 bg-black/60 px-3 py-2 text-right backdrop-blur">
      <dl className="grid grid-cols-[auto_auto] items-baseline gap-x-3 gap-y-0.5 font-mono text-[11px]">
        <dt className="text-zinc-500">fps</dt>
        <dd className={stats.fps < 30 ? "text-amber-300" : "text-white"}>{stats.fps.toFixed(0)}</dd>
        <dt className="text-zinc-500">pos</dt>
        <dd className="text-white">
          {stats.position[0].toFixed(2)}, {stats.position[1].toFixed(2)}, {stats.position[2].toFixed(2)}
        </dd>
        <dt className="text-zinc-500">scale</dt>
        <dd className="text-accent">{formatNum(upm)} u/m</dd>
        <dt className="text-zinc-500">scene</dt>
        <dd className="text-white">{sceneHeightM ? `${sceneHeightM.toFixed(2)} m tall` : "—"}</dd>
        <dt className="text-zinc-500">ground</dt>
        <dd className={stats.grounded ? "text-emerald-300" : "text-zinc-400"}>
          {stats.grounded ? "on" : "air"}
        </dd>
        <dt className="text-zinc-500">tris</dt>
        <dd className="text-zinc-300">
          {compact(stats.drawnTris)} drawn / {compact(stats.colliderTris)} collide
        </dd>
        <dt className="text-zinc-500">parts</dt>
        <dd className="text-zinc-300">{elements}</dd>
      </dl>
    </div>
  );
}

function ScalePanel({
  params,
  onChange,
  sceneInfo,
  sceneHeightM,
  eyeUnits,
  onCalibrateByHeight,
}: {
  params: WalkParams;
  onChange: (p: Partial<WalkParams>) => void;
  sceneInfo: SceneInfo | null;
  sceneHeightM: number | null;
  eyeUnits: number;
  onCalibrateByHeight: (m: number) => void;
}) {
  const [heightDraft, setHeightDraft] = useState("");
  useEffect(() => {
    if (sceneHeightM) setHeightDraft(sceneHeightM.toFixed(2));
  }, [sceneHeightM]);

  // Log slider so one drag spans 0.02 -> 50 units per metre.
  const logMin = Math.log10(0.02);
  const logMax = Math.log10(50);
  const sliderValue = (Math.log10(clamp(params.unitsPerMetre, 0.02, 50)) - logMin) / (logMax - logMin);

  return (
    <Panel icon={<Ruler className="h-3.5 w-3.5" />} title="Scale">
      <p className="mb-2 text-[11px] leading-snug text-zinc-500">
        The capture has no real-world units. Everything below is set in metres and converted with this one
        dial. Press <Kbd>[</Kbd> / <Kbd>]</Kbd> while walking to nudge it ±10%.
      </p>

      <label className="mb-1 block text-[10px] uppercase tracking-[0.2em] text-zinc-500">
        Scene units per metre
      </label>
      <div className="mb-2 flex items-center gap-2">
        <input
          type="number"
          step="0.01"
          min="0.01"
          value={round(params.unitsPerMetre, 4)}
          onChange={(e) => {
            const v = Number(e.target.value);
            if (Number.isFinite(v) && v > 0) onChange({ unitsPerMetre: clamp(v, 0.01, 1000) });
          }}
          className="w-28 rounded-lg border border-white/10 bg-black/50 px-2 py-1 font-mono text-xs text-white outline-none focus:border-accent/60"
        />
        <button
          type="button"
          onClick={() => onChange({ unitsPerMetre: clamp(params.unitsPerMetre / 1.1, 0.01, 1000) })}
          className="rounded-lg border border-white/10 px-2 py-1 font-mono text-xs text-zinc-300 hover:text-white"
        >
          −
        </button>
        <button
          type="button"
          onClick={() => onChange({ unitsPerMetre: clamp(params.unitsPerMetre * 1.1, 0.01, 1000) })}
          className="rounded-lg border border-white/10 px-2 py-1 font-mono text-xs text-zinc-300 hover:text-white"
        >
          +
        </button>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.001}
        value={sliderValue}
        onChange={(e) => {
          const t = Number(e.target.value);
          onChange({ unitsPerMetre: Math.pow(10, logMin + t * (logMax - logMin)) });
        }}
        className="mb-3 w-full accent-cyan-400"
      />

      {sceneInfo && (
        <>
          <label className="mb-1 block text-[10px] uppercase tracking-[0.2em] text-zinc-500">
            …or say how tall the shell really is
          </label>
          <div className="mb-2 flex items-center gap-2">
            <input
              type="number"
              step="0.1"
              min="0.1"
              value={heightDraft}
              onChange={(e) => setHeightDraft(e.target.value)}
              onBlur={() => onCalibrateByHeight(Number(heightDraft))}
              onKeyDown={(e) => {
                if (e.key === "Enter") onCalibrateByHeight(Number(heightDraft));
              }}
              className="w-24 rounded-lg border border-white/10 bg-black/50 px-2 py-1 font-mono text-xs text-white outline-none focus:border-accent/60"
            />
            <span className="text-[11px] text-zinc-500">metres tall</span>
            <button
              type="button"
              onClick={() => onCalibrateByHeight(Number(heightDraft))}
              className="rounded-lg border border-white/10 px-2 py-1 text-[11px] font-semibold text-zinc-300 hover:text-white"
            >
              Apply
            </button>
          </div>

          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[10px] text-zinc-500">
            <dt>1 unit</dt>
            <dd className="text-zinc-300">{formatNum(1 / params.unitsPerMetre)} m</dd>
            <dt>shell</dt>
            <dd className="text-zinc-300">
              {formatNum(sceneInfo.shellHeightUnits)} u = {sceneHeightM?.toFixed(2)} m
            </dd>
            <dt>footprint</dt>
            <dd className="text-zinc-300">
              {(sceneInfo.sceneSize[0] / params.unitsPerMetre).toFixed(1)} ×{" "}
              {(sceneInfo.sceneSize[2] / params.unitsPerMetre).toFixed(1)} m
            </dd>
            <dt>eye</dt>
            <dd className="text-zinc-300">
              {params.eyeHeightM} m = {formatNum(eyeUnits)} u
            </dd>
          </dl>
        </>
      )}
    </Panel>
  );
}

/**
 * Rebuilding the shell was API-only until now, so tuning how the world LOOKS
 * meant a curl and a page reload. Shell-only is the safe knob: it patches the
 * shell and its counts and never touches element records, so it can be re-run
 * freely while dialling detail in.
 */
function RebuildPanel({ jobId, onRebuilt }: { jobId: string; onRebuilt: () => void }) {
  const [faces, setFaces] = useState(60000);
  const [tex, setTex] = useState(2048);
  const [source, setSource] = useState<"voxel" | "tsdf">("voxel");
  const [route, setRoute] = useState<"auto" | "voxel">("auto");
  const [dropUnobserved, setDropUnobserved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await solidifyWorld(jobId, {
        shell_only: true,
        shell_source: source,
        shell_route: route,
        drop_unobserved: dropUnobserved,
        shell_faces: faces,
        shell_texture_size: tex,
        run_collision: false,
        run_gate: true,
      });
      // A world that fails its gates is still returned, with the verdict — that
      // is the point of the gates, so report it rather than only "done".
      setResult(r.gate ? (r.gate.passed ? "Rebuilt — gates passed" : "Rebuilt — GATES FAILED") : "Rebuilt");
      onRebuilt();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Rebuild failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel icon={<Hammer className="h-3.5 w-3.5" />} title="Rebuild shell">
      <p className="text-[10px] leading-snug text-zinc-400">
        Re-bakes the shell only; props and their placements are untouched.
      </p>
      <label className="flex items-center justify-between text-[11px] text-zinc-300">
        Source
        <select
          value={source}
          onChange={(e) => setSource(e.target.value as "voxel" | "tsdf")}
          className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[11px]"
        >
          <option value="voxel">voxel (watertight)</option>
          <option value="tsdf">tsdf (accurate, lacy)</option>
        </select>
      </label>
      <label className="flex items-center justify-between text-[11px] text-zinc-300">
        Detail
        <select
          value={route}
          onChange={(e) => setRoute(e.target.value as "auto" | "voxel")}
          className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[11px]"
        >
          <option value="auto">walkable (auto)</option>
          <option value="voxel">detailed (voxel)</option>
        </select>
      </label>
      <p className="text-[10px] leading-snug text-zinc-500">
        `auto` ranks shells on walkability alone and ignores detail — measured, it
        picked a 24k-tri shell over a 120k-tri one. `detailed` is ~5x the geometry
        and a tighter box, but can fail the floor-continuity gate.
      </p>
      <label className="flex items-center justify-between text-[11px] text-zinc-300">
        Faces
        <input
          type="number" min={1000} max={200000} step={1000} value={faces}
          onChange={(e) => setFaces(Number(e.target.value))}
          className="w-24 rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-right text-[11px]"
        />
      </label>
      <label className="flex items-center justify-between text-[11px] text-zinc-300">
        Texture
        <select
          value={tex}
          onChange={(e) => setTex(Number(e.target.value))}
          className="rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[11px]"
        >
          {[512, 1024, 2048, 4096].map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
      <label className="flex items-start gap-2 text-[11px] text-zinc-300">
        <input
          type="checkbox"
          checked={dropUnobserved}
          onChange={(e) => setDropUnobserved(e.target.checked)}
          className="mt-0.5"
        />
        <span>
          Don&rsquo;t draw unseen surface
          <span className="block text-[10px] leading-snug text-zinc-500">
            An outdoor capture has no walls or ceiling, but a watertight shell
            invents them and the bake smears colour over them. This drops those
            faces (~21% here). Collision is unaffected — you still can&rsquo;t
            fall out.
          </span>
        </span>
      </label>
      <button
        type="button"
        onClick={run}
        disabled={busy}
        className="mt-1 w-full rounded-lg bg-cyan-500/90 px-2 py-1.5 text-[11px] font-semibold text-[#04121a] disabled:opacity-50"
      >
        {busy ? "Rebuilding…" : "Rebuild shell"}
      </button>
      {result && (
        <p className={`text-[10px] leading-snug ${result.includes("FAILED") ? "text-amber-300" : "text-emerald-300/90"}`}>
          {result} — reload to see it.
        </p>
      )}
      {error && <p className="text-[10px] leading-snug text-rose-300/90">{error}</p>}
    </Panel>
  );
}

function MovementPanel({
  params,
  onChange,
}: {
  params: WalkParams;
  onChange: (p: Partial<WalkParams>) => void;
}) {
  return (
    <Panel icon={<Gauge className="h-3.5 w-3.5" />} title="Player & physics">
      <Slider
        label="Eye height"
        unit="m"
        min={0.3}
        max={2.6}
        step={0.05}
        value={params.eyeHeightM}
        onChange={(v) => onChange({ eyeHeightM: v })}
      />
      <Slider
        label="Body radius"
        unit="m"
        min={0.05}
        max={0.9}
        step={0.01}
        value={params.radiusM}
        onChange={(v) => onChange({ radiusM: v })}
      />
      <Slider
        label="Walk speed"
        unit="m/s"
        min={0.2}
        max={10}
        step={0.1}
        value={params.walkSpeedMps}
        onChange={(v) => onChange({ walkSpeedMps: v })}
      />
      <Slider
        label="Sprint ×"
        unit=""
        min={1}
        max={6}
        step={0.1}
        value={params.sprintMultiplier}
        onChange={(v) => onChange({ sprintMultiplier: v })}
      />
      <Slider
        label="Gravity"
        unit="m/s²"
        min={0}
        max={30}
        step={0.1}
        value={params.gravityMps2}
        onChange={(v) => onChange({ gravityMps2: v })}
      />
      <Slider
        label="Jump speed"
        unit="m/s"
        min={0}
        max={12}
        step={0.1}
        value={params.jumpSpeedMps}
        onChange={(v) => onChange({ jumpSpeedMps: v })}
      />
      <Slider
        label="Field of view"
        unit="°"
        min={45}
        max={110}
        step={1}
        value={params.fovDeg}
        onChange={(v) => onChange({ fovDeg: v })}
      />
      <Slider
        label="Mouse speed"
        unit="×"
        min={0.2}
        max={3}
        step={0.05}
        value={params.mouseSensitivity}
        onChange={(v) => onChange({ mouseSensitivity: v })}
      />

      <div className="mt-2 space-y-1.5 border-t border-white/10 pt-2">
        <Toggle
          label="Props are solid"
          hint="v1 collides with the shell + static elements only"
          checked={params.collideProps}
          onChange={(v) => onChange({ collideProps: v })}
        />
        <Toggle
          label="Unlit (baked textures)"
          hint="atlases already contain lighting"
          checked={params.unlit}
          onChange={(v) => onChange({ unlit: v })}
        />
        <Toggle
          label="Show collider wireframe"
          hint="what the capsule actually hits"
          checked={params.showCollider}
          onChange={(v) => onChange({ showCollider: v })}
        />
      </div>
    </Panel>
  );
}

function ElementsPanel({
  rows,
  hidden,
  upm,
  colliderTris,
  colliderSource,
  jobId,
  onToggle,
  onGo,
  onPolished,
}: {
  rows: ElementRow[];
  hidden: Set<string>;
  upm: number;
  colliderTris: number;
  colliderSource: "collision_shell" | "collision_shell+authored" | "visual_shell";
  jobId: string;
  onToggle: (slug: string) => void;
  onGo: (slug: string) => void;
  onPolished: () => void;
}) {
  // One expanded polish zone at a time — the HUD column is narrow.
  const [polishSlug, setPolishSlug] = useState<string | null>(null);
  return (
    <Panel icon={<Shapes className="h-3.5 w-3.5" />} title={`Loaded elements (${rows.length})`}>
      <p className="mb-2 font-mono text-[10px] text-zinc-500">
        collider: {compact(colliderTris)} tris{" "}
        <span
          title={
            colliderSource.startsWith("collision_shell")
              ? "Walking on the watertight voxel solid (collision_shell.glb)."
              : "No collision solid in the manifest — falling back to the visual shell, which is fragmented and can be fallen through. Run world_shell.py."
          }
          style={{
            color: colliderSource.startsWith("collision_shell") ? "#4ade80" : "#fbbf24",
            fontWeight: 600,
          }}
        >
          [{colliderSource === "collision_shell+authored" ? "SOLID+AUTHORED"
            : colliderSource === "collision_shell" ? "SOLID" : "VISUAL — may fall through"}]
        </span>{" "}
        ·{" "}
        {rows.filter((r) => r.collides).length}/{rows.length} solid
      </p>
      <ul className="space-y-1.5">
        {rows.map((r) => {
          const isHidden = hidden.has(r.slug);
          return (
            <li
              key={r.slug}
              className={`rounded-lg border p-2 transition ${
                r.frameWarning ? "border-amber-500/40 bg-amber-950/20" : "border-white/10 bg-white/[0.02]"
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-xs font-semibold text-white">{r.label}</p>
                  <p className="truncate font-mono text-[10px] text-zinc-500">{r.slug}</p>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <button
                    type="button"
                    onClick={() => onToggle(r.slug)}
                    title={isHidden ? "Show" : "Hide"}
                    className="rounded-md border border-white/10 p-1 text-zinc-400 hover:text-white"
                  >
                    {isHidden ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                  </button>
                  <button
                    type="button"
                    onClick={() => onGo(r.slug)}
                    title="Stand next to it"
                    className="flex items-center gap-1 rounded-md border border-white/10 px-1.5 py-1 text-[10px] font-semibold text-zinc-300 hover:border-accent/50 hover:text-white"
                  >
                    <MapPin className="h-3 w-3" />
                    Go
                  </button>
                  <button
                    type="button"
                    onClick={() => setPolishSlug(polishSlug === r.slug ? null : r.slug)}
                    title="Replace this element with a Blender-polished .glb"
                    className={`rounded-md border p-1 transition ${
                      polishSlug === r.slug
                        ? "border-cyan-400/50 text-cyan-200"
                        : "border-white/10 text-zinc-400 hover:text-white"
                    }`}
                  >
                    <UploadCloud className="h-3 w-3" />
                  </button>
                </div>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-1">
                <Tag tone={r.role === "shell" ? "accent" : r.role === "static" ? "emerald" : r.role === "environment" ? "cyan" : "zinc"}>
                  {r.role}
                </Tag>
                <Tag tone={r.collides ? "emerald" : "zinc"}>{r.collides ? "solid" : "pass-through"}</Tag>
                <Tag tone="zinc">
                  <Box className="mr-1 inline h-2.5 w-2.5" />
                  {compact(r.tris)} tris
                </Tag>
                <Tag tone="zinc">
                  {(r.size[0] / upm).toFixed(2)}×{(r.size[1] / upm).toFixed(2)}×{(r.size[2] / upm).toFixed(2)} m
                </Tag>
              </div>
              {r.frameWarning && <p className="mt-1 text-[10px] text-amber-300">{r.frameWarning}</p>}
              {polishSlug === r.slug && (
                <PolishUploadZone
                  title="Drop a polished .glb or click"
                  hint="Replaces this element's GLB (the old one is versioned server-side); the world reloads with it."
                  confirmLabel={`Sure? Replaces ${r.slug} in the world`}
                  onUpload={async (file, onPct) => {
                    await uploadPolishedWorldElement(jobId, r.slug, file, onPct);
                    setPolishSlug(null);
                    onPolished();
                  }}
                />
              )}
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}

/* ------------------------------------------------------------------ *
 * Small presentational primitives (kept local — @/components/ui is    *
 * being restructured by another agent right now).                     *
 * ------------------------------------------------------------------ */

function RestylePanel({
  restyle,
  materials,
  rows,
  busy,
  api,
  baked,
  onChange,
  onClear,
  onBake,
  onRevert,
}: {
  restyle: RestyleDoc;
  materials: RestyleMaterial[];
  rows: ElementRow[];
  busy: boolean;
  api: boolean;
  baked: boolean;
  onChange: (next: RestyleDoc) => void;
  onClear: () => void;
  onBake: () => void;
  onRevert: () => void;
}) {
  const [slug, setSlug] = useState<string>(rows[0]?.slug ?? "");
  const target = rows.find((r) => r.slug === slug) ?? rows[0];
  const entry: RestyleEntry = (target && restyle.elements[target.slug]) || {};
  const styled = Object.keys(restyle.elements).length;
  // Armed like every other destructive control (edit-lane idiom): first
  // click arms, second fires, 4 s to change your mind.
  const [bakeArmed, setBakeArmed] = useState(false);
  const [revertArmed, setRevertArmed] = useState(false);
  useEffect(() => {
    if (!bakeArmed) return;
    const t = window.setTimeout(() => setBakeArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [bakeArmed]);
  useEffect(() => {
    if (!revertArmed) return;
    const t = window.setTimeout(() => setRevertArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [revertArmed]);
  const canBake = api && styled > 0;

  const patch = (next: RestyleEntry | null) => {
    if (!target) return;
    const elements = { ...restyle.elements };
    if (!next || (!next.tint && !next.material)) delete elements[target.slug];
    else elements[target.slug] = next;
    onChange({ ...restyle, elements });
  };

  return (
    <Panel icon={<Palette className="h-3.5 w-3.5" />} title="Restyle">
      <p className="mb-2 text-[11px] leading-relaxed text-zinc-400">
        A look applied over the capture — nothing on disk is rewritten, so
        Reset always returns the world exactly as it was photographed.
      </p>

      <label className="mb-1 block text-[10px] uppercase tracking-wider text-zinc-500">
        Lighting
      </label>
      <select
        className="mb-2 w-full rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] text-zinc-200"
        value={restyle.lighting.preset}
        onChange={(e) => onChange({
          ...restyle,
          lighting: { ...restyle.lighting, preset: e.target.value },
        })}
      >
        {Object.keys(LIGHT_PRESETS).map((name) => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>
      <Slider
        label="Light level"
        unit="×"
        min={0}
        max={3}
        step={0.05}
        value={restyle.lighting.intensity}
        onChange={(v) => onChange({
          ...restyle, lighting: { ...restyle.lighting, intensity: v },
        })}
      />

      {target && (
        <>
          <label className="mb-1 mt-3 block text-[10px] uppercase tracking-wider text-zinc-500">
            Element
          </label>
          <select
            className="mb-2 w-full rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] text-zinc-200"
            value={target.slug}
            onChange={(e) => setSlug(e.target.value)}
          >
            {rows.map((r) => (
              <option key={r.slug} value={r.slug}>
                {r.label || r.slug}{restyle.elements[r.slug] ? " ●" : ""}
              </option>
            ))}
          </select>

          <div className="mb-2 flex items-center gap-2">
            <input
              type="color"
              aria-label="tint"
              className="h-7 w-10 cursor-pointer rounded border border-white/10 bg-black/60"
              value={entry.tint ?? "#ffffff"}
              onChange={(e) => patch({ ...entry, tint: e.target.value })}
            />
            <span className="text-[11px] text-zinc-400">Tint</span>
            {entry.tint && (
              <button
                type="button"
                className="ml-auto text-[10px] text-zinc-500 underline"
                onClick={() => patch({ ...entry, tint: undefined })}
              >
                clear
              </button>
            )}
          </div>

          <label className="mb-1 block text-[10px] uppercase tracking-wider text-zinc-500">
            Surface
          </label>
          <select
            className="w-full rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] text-zinc-200"
            value={entry.material ?? ""}
            onChange={(e) => patch({
              ...entry, material: e.target.value || undefined,
            })}
          >
            <option value="">as captured</option>
            {materials.map((m) => (
              <option key={m.id} value={m.id}>{m.display}</option>
            ))}
          </select>
          {entry.material && (
            <Slider
              label="Tile size"
              unit="m"
              min={0.1}
              max={5}
              step={0.1}
              value={entry.material_scale ?? 1}
              onChange={(v) => patch({ ...entry, material_scale: v })}
            />
          )}
        </>
      )}

      <div className="mt-3 flex items-center justify-between">
        <span className="text-[10px] text-zinc-500">
          {busy ? "saving…" : `${styled} element${styled === 1 ? "" : "s"} restyled`}
        </span>
        <button
          type="button"
          className="rounded-md border border-white/10 px-2 py-1 text-[10px] text-zinc-300 hover:bg-white/5"
          onClick={onClear}
        >
          Reset to capture
        </button>
      </div>

      {(canBake || (api && baked)) && (
        <div className="mt-2 flex items-center justify-between gap-2 border-t border-white/5 pt-2">
          <span className="text-[10px] leading-snug text-zinc-500">
            {canBake
              ? "Make it permanent — survives Blender/UE export"
              : "This world carries a baked look"}
          </span>
          {canBake ? (
            <button
              type="button"
              disabled={busy}
              className={`shrink-0 rounded-md border px-2 py-1 text-[10px] ${
                bakeArmed
                  ? "border-amber-400/60 bg-amber-500/15 text-amber-300"
                  : "border-white/10 text-zinc-300 hover:bg-white/5"
              } disabled:opacity-50`}
              onClick={() => {
                if (bakeArmed) {
                  setBakeArmed(false);
                  onBake();
                } else {
                  setBakeArmed(true);
                }
              }}
            >
              {bakeArmed
                ? `Sure? Rewrites ${styled} GLB${styled === 1 ? "" : "s"} (versioned)`
                : "Bake this look"}
            </button>
          ) : (
            <button
              type="button"
              disabled={busy}
              className={`shrink-0 rounded-md border px-2 py-1 text-[10px] ${
                revertArmed
                  ? "border-amber-400/60 bg-amber-500/15 text-amber-300"
                  : "border-white/10 text-zinc-300 hover:bg-white/5"
              } disabled:opacity-50`}
              onClick={() => {
                if (revertArmed) {
                  setRevertArmed(false);
                  onRevert();
                } else {
                  setRevertArmed(true);
                }
              }}
            >
              {revertArmed ? "Sure? Restore the capture" : "Revert bake"}
            </button>
          )}
        </div>
      )}
    </Panel>
  );
}

function PlacedPanel({
  manifest,
  onPlace,
  onRemove,
}: {
  manifest: WorldManifest;
  onPlace: (slug: string, role: string, label: string, file: File,
            onProgress: (pct: number) => void) => Promise<void>;
  onRemove: (slug: string) => void;
}) {
  const placed = (manifest.elements ?? []).filter(
    (e) => e.provenance === "authored");
  const [slug, setSlug] = useState("");
  const [role, setRole] = useState<"static" | "prop" | "environment">("static");
  // Armed like every other destructive control: first click arms, second
  // fires, 4 s to change your mind.
  const [armedRemove, setArmedRemove] = useState<string | null>(null);
  useEffect(() => {
    if (armedRemove === null) return;
    const t = window.setTimeout(() => setArmedRemove(null), 4000);
    return () => window.clearTimeout(t);
  }, [armedRemove]);
  // Mirror of the server's placement slug grammar (world_placed.SLUG_RE);
  // the door is the real gate — this only saves a round trip.
  const slugOk = /^[a-z0-9][a-z0-9-]{0,39}$/.test(slug) && slug !== "shell";
  const uncalibrated = manifest.meters_per_unit == null;

  return (
    <Panel icon={<Wand2 className="h-3.5 w-3.5" />} title="Placed assets">
      <p className="mb-2 text-[11px] leading-relaxed text-zinc-400">
        Fantasy set-dressing authored in Blender, landed as first-class world
        elements. They survive rebuilds; Remove tombstones the file.
      </p>

      {placed.length > 0 ? (
        <div className="mb-2 space-y-1">
          {placed.map((entry) => (
            <div
              key={entry.slug}
              className="flex items-center gap-2 rounded-md border border-white/5 bg-black/40 px-2 py-1"
            >
              <span className="min-w-0 flex-1 truncate text-[11px] text-zinc-200">
                {entry.label || entry.slug}
                <span className="ml-1 text-[9px] uppercase tracking-wider text-zinc-500">
                  {entry.role}
                </span>
                {entry.extent && (
                  <span className="ml-1 text-[10px] text-zinc-500">
                    {entry.extent.map((v) => v.toFixed(2)).join("×")}
                    {uncalibrated ? " u" : " m"}
                  </span>
                )}
              </span>
              <button
                type="button"
                className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] ${
                  armedRemove === entry.slug
                    ? "border-amber-400/60 bg-amber-500/15 text-amber-300"
                    : "border-white/10 text-zinc-400 hover:bg-white/5"
                }`}
                onClick={() => {
                  if (armedRemove === entry.slug) {
                    setArmedRemove(null);
                    onRemove(entry.slug);
                  } else {
                    setArmedRemove(entry.slug);
                  }
                }}
              >
                {armedRemove === entry.slug ? "Sure? Tombstone it" : "Remove"}
              </button>
            </div>
          ))}
        </div>
      ) : (
        <p className="mb-2 text-[10px] text-zinc-500">
          Nothing placed yet — the world is exactly as captured.
        </p>
      )}

      <label className="mb-1 block text-[10px] uppercase tracking-wider text-zinc-500">
        Place new asset
      </label>
      <div className="mb-1 flex items-center gap-2">
        <input
          type="text"
          value={slug}
          onChange={(e) => setSlug(e.target.value.trim())}
          placeholder="new-slug (kebab-case)"
          className="min-w-0 flex-1 rounded-md border border-white/10 bg-black/60 px-2 py-1 font-mono text-[11px] text-zinc-200 placeholder:text-zinc-600"
        />
        <select
          value={role}
          onChange={(e) => setRole(e.target.value as "static" | "prop" | "environment")}
          className="rounded-md border border-white/10 bg-black/60 px-2 py-1 text-[11px] text-zinc-200"
        >
          <option value="static">static</option>
          <option value="prop">prop</option>
          <option value="environment">environment</option>
        </select>
      </div>
      {slug && !slugOk && (
        <p className="mb-1 text-[10px] text-amber-300/90">
          Slug must be lowercase kebab-case (max 40 chars), not “shell”.
        </p>
      )}
      {uncalibrated && (
        <p className="mb-1 text-[10px] leading-snug text-amber-300/80">
          World scale is a guess — a metre-modelled asset lands at a guessed
          size. Calibrate via the Measure tab for honest sizing.
        </p>
      )}
      <PolishUploadZone
        title="Drop a world-frame GLB to place"
        hint={"Author in Blender: import_asset → transform_object → export with bake_world_transform=true. The slug must be NEW — placement invents, polish replaces. environment = player-walkable architecture (requires baked identity transforms, like props)."}
        confirmLabel={slugOk ? `Place as '${slug}' (${role})` : "Enter a slug first"}
        disabled={!slugOk}
        onUpload={(file, onProgress) => onPlace(slug, role, slug, file, onProgress)}
      />
    </Panel>
  );
}

function Panel({
  icon,
  title,
  children,
}: {
  icon: ReactNode;
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-white/10 bg-black/70 p-3 backdrop-blur">
      <h2 className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.25em] text-zinc-400">
        {icon}
        {title}
      </h2>
      {children}
    </section>
  );
}

function Slider({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="mb-1.5 block">
      <span className="flex items-baseline justify-between text-[11px] text-zinc-400">
        {label}
        <span className="font-mono text-[10px] text-zinc-300">
          {round(value, 2)}
          {unit && ` ${unit}`}
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-cyan-400"
      />
    </label>
  );
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint?: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-2">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 h-3.5 w-3.5 accent-cyan-400"
      />
      <span className="leading-tight">
        <span className="block text-[11px] text-zinc-200">{label}</span>
        {hint && <span className="block text-[10px] text-zinc-500">{hint}</span>}
      </span>
    </label>
  );
}

function Tag({ children, tone }: { children: ReactNode; tone: "accent" | "emerald" | "cyan" | "zinc" }) {
  const cls =
    tone === "accent"
      ? "border-accent/40 text-accent"
      : tone === "emerald"
        ? "border-emerald-500/40 text-emerald-300"
        : tone === "cyan"
          ? "border-cyan-400/40 text-cyan-300"
          : "border-white/10 text-zinc-400";
  return (
    <span className={`rounded border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide ${cls}`}>
      {children}
    </span>
  );
}

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-white/15 bg-white/5 px-1 font-mono text-[10px] text-zinc-200">
      {children}
    </kbd>
  );
}

function Key({ k, v }: { k: string; v: string }) {
  if (!k) return <span />;
  return (
    <span className="flex items-center justify-between gap-2">
      <Kbd>{k}</Kbd>
      <span>{v}</span>
    </span>
  );
}

/* ------------------------------------------------------------------ */

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function round(v: number, places: number): number {
  const f = Math.pow(10, places);
  return Math.round(v * f) / f;
}

function formatNum(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(2);
  return v.toFixed(4);
}

function compact(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return `${Math.round(n)}`;
}
