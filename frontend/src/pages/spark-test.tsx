import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { dyno, RgbaArray, readRgbaArray, SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { apiRequest } from "@/lib/api";
import type { SplatJob, SplatStatusResponse } from "@/lib/contracts";
import { Button, Card, SectionLabel } from "@/components/ui";
import { ArrowLeft, Box, Compass, Crosshair, Flame, Gauge, Loader2, RotateCcw, Search, Sun } from "lucide-react";

// SPIKE PAGE — proves the Spark (sparkjs.dev) viewer mechanism ahead of a possible
// migration off @mkkellogg/gaussian-splats-3d (see splat-viewer.tsx). Standalone route;
// the shipped viewer is untouched. Two things under test:
//  1) Can a fake per-splat scalar drive a GPU-side worldModifier (LUT tint + opacity
//     spotlight) the way a real language-field relevancy score eventually would?
//  2) Basic nav (reset/presets/pivot-on-double-click) against Spark's own raycasting.

// Same Z-up framing as the shipped viewer (splat-viewer.tsx) — nerfstudio splatfacto
// scenes come out Z-up; replicate the exact initial pose for a fair side-by-side feel.
const INITIAL_CAMERA_POSITION = new THREE.Vector3(0, -3, 1.4);
const INITIAL_CAMERA_LOOK_AT = new THREE.Vector3(0, 0, 0.2);
const INITIAL_CAMERA_UP = new THREE.Vector3(0, 0, 1);

type PresetView = "top" | "front" | "iso";
type PivotMode = "none" | "raycast" | "ground-plane";

export default function SparkTestPage() {
  const { data: status } = useQuery({
    queryKey: ["status"],
    queryFn: () => apiRequest<SplatStatusResponse>("/api/splat/status"),
    refetchInterval: 5000,
  });

  const scenes = useMemo<SplatJob[]>(
    () => (status?.jobs ?? []).filter((j) => j.preview_available && (j.preview_web_url || j.preview_view_url)),
    [status],
  );

  const [jobId, setJobId] = useState<string | null>(null);
  useEffect(() => {
    if (jobId && scenes.some((s) => s.job_id === jobId)) return;
    if (scenes.length > 0) setJobId(scenes[0].job_id);
  }, [scenes, jobId]);

  const job = scenes.find((s) => s.job_id === jobId) ?? null;
  // Langfield scenes load the full-count langweb variant so splat index i ==
  // gauss_emb row i == relevancy byte i. web.ply is decimated + REORDERED by
  // splat-transform, so its indices can never be zipped onto relevancy data.
  const url = job
    ? `/api/splat/jobs/${job.job_id}/preview/file?fmt=${job.langfield_available ? "langweb" : "web"}`
    : null;

  return (
    <div className="flex h-screen flex-col bg-[#05070d] text-zinc-100">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Link href="/" className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200">
            <ArrowLeft className="h-4 w-4" /> Splat Lab
          </Link>
          <span className="text-white/20">/</span>
          <div className="flex min-w-0 items-center gap-2">
            <Compass className="h-4 w-4 shrink-0 text-cyan-300" />
            <span className="truncate text-sm font-semibold">Spark viewer spike</span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <SectionLabel className="text-[10px]">Scene</SectionLabel>
          <select
            value={jobId ?? ""}
            onChange={(e) => setJobId(e.target.value || null)}
            className="h-9 max-w-[16rem] rounded-xl border border-white/12 bg-white/5 px-3 text-sm text-zinc-100 focus:border-cyan-400/40 focus:outline-none"
          >
            {scenes.length === 0 && <option value="">No previewable scenes</option>}
            {scenes.map((s) => (
              <option key={s.job_id} value={s.job_id}>
                {sceneLabel(s)}
              </option>
            ))}
          </select>
        </div>
      </header>
      <main className="relative flex-1 overflow-hidden">
        {url && job ? (
          <SparkViewport key={job.job_id} url={url} job={job} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-zinc-500">
            No previewable scene available yet.
          </div>
        )}
      </main>
    </div>
  );
}

function sceneLabel(job: SplatJob): string {
  const name = job.input_path?.split("/").pop() || job.job_id;
  const gaussians = job.stats?.gaussians;
  return gaussians ? `${name} (${gaussians.toLocaleString()} splats)` : name;
}

function SparkViewport({ url, job }: { url: string; job: SplatJob }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<SplatMesh | null>(null);
  const tweenCancelRef = useRef<(() => void) | null>(null);

  // Live dyno uniforms for the heatmap modifier — mutated in place from UI handlers so
  // toggling doesn't rebuild the GPU program (mirrors Spark's own splat-painter example).
  const heatmapEnabledRef = useRef<ReturnType<typeof dyno.dynoBool> | null>(null);
  const spotlightEnabledRef = useRef<ReturnType<typeof dyno.dynoBool> | null>(null);
  const spotlightThresholdRef = useRef<ReturnType<typeof dyno.dynoFloat> | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [fps, setFps] = useState(0);
  const [heatmapOn, setHeatmapOn] = useState(false);
  const [spotlightOn, setSpotlightOn] = useState(false);
  const [spotlightThreshold, setSpotlightThreshold] = useState(0.4);
  const [pivotMode, setPivotMode] = useState<PivotMode>("none");
  const [query, setQuery] = useState("");
  const [queryBusy, setQueryBusy] = useState(false);
  const [queryStatus, setQueryStatus] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [realScalarLive, setRealScalarLive] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    setError(null);
    setReady(false);
    setSplatCount(null);
    setPivotMode("none");
    setQueryStatus(null);
    setQueryError(null);
    setRealScalarLive(false);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.up.copy(INITIAL_CAMERA_UP);
    camera.position.copy(INITIAL_CAMERA_POSITION);
    camera.lookAt(INITIAL_CAMERA_LOOK_AT);
    cameraRef.current = camera;

    // antialias:false per Spark's own guidance — WebGL MSAA doesn't help splat
    // rendering and costs real perf.
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    const spark = new SparkRenderer({ renderer });
    scene.add(spark);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(INITIAL_CAMERA_LOOK_AT);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();
    controlsRef.current = controls;

    function resize() {
      const el = containerRef.current;
      if (!el) return;
      const width = Math.max(el.clientWidth, 1);
      const height = Math.max(el.clientHeight, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    const heatmapEnabled = dyno.dynoBool(heatmapOn);
    const spotlightEnabled = dyno.dynoBool(spotlightOn);
    const spotlightThresholdDyno = dyno.dynoFloat(spotlightThreshold);
    heatmapEnabledRef.current = heatmapEnabled;
    spotlightEnabledRef.current = spotlightEnabled;
    spotlightThresholdRef.current = spotlightThresholdDyno;

    // The preview URL has no file extension (/preview/file?fmt=web) so Spark's own
    // extension-sniffing (getSplatFileTypeFromPath) can't detect it — must pass
    // fileType explicitly, same reason the shipped mkkellogg viewer passes `format`.
    const mesh = new SplatMesh({
      url,
      fileType: SplatFileType.PLY,
      raycastable: true,
      minRaycastOpacity: 0.1,
    });
    scene.add(mesh);
    meshRef.current = mesh;

    let raf = 0;
    const clock = new THREE.Clock();
    let frames = 0;
    let fpsAccum = 0;
    function animate() {
      raf = requestAnimationFrame(animate);
      const dt = clock.getDelta();
      frames += 1;
      fpsAccum += dt;
      if (fpsAccum >= 0.4) {
        setFps(Math.round(frames / fpsAccum));
        frames = 0;
        fpsAccum = 0;
      }
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    mesh.initialized
      .then(() => {
        if (disposed) return;
        const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
        setSplatCount(numSplats);
        if (numSplats > 0) {
          const scalarArray = new RgbaArray({ array: buildFakeScalarArray(numSplats), count: numSplats });
          mesh.worldModifier = buildHeatmapModifier({
            scalarArray,
            heatmapEnabled,
            spotlightEnabled,
            spotlightThreshold: spotlightThresholdDyno,
          });
          mesh.updateGenerator();
        }
        setReady(true);
      })
      .catch((cause: unknown) => {
        if (disposed) return;
        setError(cause instanceof Error ? cause.message : "Could not load Spark preview.");
      });

    function onDoubleClick(event: MouseEvent) {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, camera);

      let hit: THREE.Vector3 | null = null;
      let mode: PivotMode = "ground-plane";
      if (mesh.raycastable) {
        const intersects: { distance: number; point: THREE.Vector3; object: THREE.Object3D }[] = [];
        mesh.raycast(raycaster, intersects);
        if (intersects.length > 0) {
          intersects.sort((a, b) => a.distance - b.distance);
          hit = intersects[0].point.clone();
          mode = "raycast";
        }
      }
      if (!hit) {
        const dir = raycaster.ray.direction;
        if (Math.abs(dir.z) > 1e-6) {
          const t = -raycaster.ray.origin.z / dir.z;
          if (t > 0) hit = raycaster.ray.origin.clone().addScaledVector(dir, t);
        }
      }
      if (hit) {
        tweenCancelRef.current?.();
        tweenCancelRef.current = null;
        controls.target.copy(hit);
        camera.lookAt(controls.target);
        controls.update();
        setPivotMode(mode);
      }
    }
    renderer.domElement.addEventListener("dblclick", onDoubleClick);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      tweenCancelRef.current?.();
      tweenCancelRef.current = null;
      renderer.domElement.removeEventListener("dblclick", onDoubleClick);
      resizeObserver.disconnect();
      controls.dispose();
      mesh.dispose();
      spark.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === container) container.removeChild(renderer.domElement);
      cameraRef.current = null;
      controlsRef.current = null;
      meshRef.current = null;
      heatmapEnabledRef.current = null;
      spotlightEnabledRef.current = null;
      spotlightThresholdRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  function resetView() {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    tweenCancelRef.current?.();
    tweenCancelRef.current = null;
    camera.position.copy(INITIAL_CAMERA_POSITION);
    camera.up.copy(INITIAL_CAMERA_UP);
    controls.target.copy(INITIAL_CAMERA_LOOK_AT);
    camera.lookAt(controls.target);
    controls.update();
    setPivotMode("none");
  }

  function goToPreset(preset: PresetView) {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    tweenCancelRef.current?.();
    const pivot = controls.target.clone();
    const distance = Math.max(camera.position.distanceTo(pivot), 0.5);
    let dir: THREE.Vector3;
    let up: THREE.Vector3;
    if (preset === "top") {
      dir = new THREE.Vector3(0, 0, 1);
      up = new THREE.Vector3(0, 1, 0);
    } else if (preset === "front") {
      dir = new THREE.Vector3(0, -1, 0.35).normalize();
      up = new THREE.Vector3(0, 0, 1);
    } else {
      dir = new THREE.Vector3(-0.6, -0.6, 0.55).normalize();
      up = new THREE.Vector3(0, 0, 1);
    }
    const targetPosition = pivot.clone().addScaledVector(dir, distance);
    tweenCancelRef.current = animateCameraTo(camera, controls, targetPosition, up, pivot);
  }

  // Real relevancy → the SAME modifier mechanism the fake proof uses, fed by the
  // uint8 vector from /langfield/relevancy (row i == langweb splat i). Refuses to
  // apply on any row/splat count mismatch — a silent misalignment would tint the
  // wrong splats, which is worse than an error.
  async function runRealQuery() {
    const mesh = meshRef.current;
    const heatmapEnabled = heatmapEnabledRef.current;
    const spotlightEnabled = spotlightEnabledRef.current;
    const spotlightThresholdDyno = spotlightThresholdRef.current;
    const text = query.trim();
    if (!job || !mesh || !heatmapEnabled || !spotlightEnabled || !spotlightThresholdDyno || !text) return;
    setQueryBusy(true);
    setQueryError(null);
    const started = performance.now();
    try {
      const res = await fetch(`/api/splat/jobs/${job.job_id}/langfield/relevancy`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (res.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`${res.status}: ${detail.slice(0, 300)}`);
      }
      const bytes = new Uint8Array(await res.arrayBuffer());
      const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
      if (bytes.length !== numSplats) {
        throw new Error(
          `relevancy rows (${bytes.length.toLocaleString()}) != loaded splats (${numSplats.toLocaleString()}) — scene not loaded as langweb?`,
        );
      }
      const rgba = new Uint8Array(numSplats * 4);
      for (let i = 0; i < numSplats; i += 1) {
        const b = bytes[i];
        const o = i * 4;
        rgba[o] = b;
        rgba[o + 1] = b;
        rgba[o + 2] = b;
        rgba[o + 3] = 255;
      }
      const scalarArray = new RgbaArray({ array: rgba, count: numSplats });
      mesh.worldModifier = buildHeatmapModifier({
        scalarArray,
        heatmapEnabled,
        spotlightEnabled,
        spotlightThreshold: spotlightThresholdDyno,
      });
      mesh.updateGenerator();
      setRealScalarLive(true);
      setHeatmapOn(true);
      heatmapEnabled.value = true;
      const relMin = res.headers.get("X-Min");
      const relMax = res.headers.get("X-Max");
      let matchCount: number | null = null;
      const matchesHeader = res.headers.get("X-Matches");
      if (matchesHeader) {
        try {
          const parsed = JSON.parse(matchesHeader);
          if (Array.isArray(parsed)) matchCount = parsed.length;
        } catch {
          matchCount = null;
        }
      }
      const ms = Math.round(performance.now() - started);
      setQueryStatus(
        `"${text}" · ${bytes.length.toLocaleString()} rows · rel ${relMin ?? "?"}–${relMax ?? "?"}` +
          (matchCount !== null ? ` · ${matchCount} instance${matchCount === 1 ? "" : "s"}` : "") +
          ` · ${ms}ms`,
      );
    } catch (cause) {
      setQueryError(cause instanceof Error ? cause.message : "Relevancy request failed.");
    } finally {
      setQueryBusy(false);
    }
  }

  function onHeatmapToggle(next: boolean) {
    setHeatmapOn(next);
    if (heatmapEnabledRef.current) heatmapEnabledRef.current.value = next;
  }
  function onSpotlightToggle(next: boolean) {
    setSpotlightOn(next);
    if (spotlightEnabledRef.current) spotlightEnabledRef.current.value = next;
  }
  function onThresholdChange(next: number) {
    setSpotlightThreshold(next);
    if (spotlightThresholdRef.current) spotlightThresholdRef.current.value = next;
  }

  return (
    <div className="relative h-full w-full overflow-hidden bg-black/70">
      <div ref={containerRef} className="h-full w-full" />

      {!ready && !error && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-black/70 px-4 py-2 text-xs text-zinc-300">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading Spark preview…
          </div>
        </div>
      )}
      {error && (
        <div className="absolute inset-4 flex items-center justify-center rounded-2xl border border-red-500/25 bg-red-500/10 p-4 text-sm text-red-200">
          {error}
        </div>
      )}

      <Card className="absolute left-3 top-3 z-20 w-72 space-y-3 p-4 text-xs">
        <div className="flex items-center justify-between">
          <SectionLabel>Nav prototype</SectionLabel>
          <div className="flex items-center gap-1 text-zinc-400">
            <Gauge className="h-3.5 w-3.5" /> {fps} fps
          </div>
        </div>
        <div className="flex items-center gap-1.5 text-zinc-400">
          <Box className="h-3.5 w-3.5" /> {splatCount === null ? "…" : splatCount.toLocaleString()} splats
        </div>
        <div className="flex flex-wrap gap-1.5">
          <Button type="button" variant="outline" size="sm" onClick={resetView} title="Reset camera">
            <RotateCcw className="h-3.5 w-3.5" /> Reset
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={() => goToPreset("top")}>
            Top
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={() => goToPreset("front")}>
            Front
          </Button>
          <Button type="button" variant="outline" size="sm" onClick={() => goToPreset("iso")}>
            Iso
          </Button>
        </div>
        <div className="flex items-center gap-1.5 text-zinc-400">
          <Crosshair className="h-3.5 w-3.5" />
          {pivotMode === "none" && "Double-click a splat to set the orbit pivot"}
          {pivotMode === "raycast" && "Pivot set — raycast hit a splat"}
          {pivotMode === "ground-plane" && "Pivot set — ground-plane fallback (no splat hit)"}
        </div>

        <div className="h-px bg-white/10" />
        <SectionLabel>Language heatmap (real relevancy)</SectionLabel>
        {job?.langfield_available ? (
          <>
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void runRealQuery();
              }}
            >
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder='Try "chair", "trash can"…'
                className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-600 focus:border-cyan-400/50 focus:outline-none"
              />
              <Button type="submit" size="sm" disabled={queryBusy || !query.trim()}>
                {queryBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
              </Button>
            </form>
            {queryStatus && <p className="text-[10px] leading-snug text-emerald-300/90">{queryStatus}</p>}
            {queryError && <p className="text-[10px] leading-snug text-rose-300/90">{queryError}</p>}
          </>
        ) : (
          <p className="text-[10px] leading-snug text-zinc-500">
            This scene has no language field — run a job with the language field enabled
            to drive the heatmap with real relevancy. The fake-scalar proof below still works.
          </p>
        )}

        <div className="h-px bg-white/10" />
        <SectionLabel>{realScalarLive ? "Heatmap (REAL relevancy live)" : "Heatmap mechanism proof (fake data)"}</SectionLabel>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={heatmapOn} onChange={(e) => onHeatmapToggle(e.target.checked)} />
          <Flame className="h-3.5 w-3.5 text-orange-300" />{" "}
          {realScalarLive ? "Tint by query relevancy" : "Tint by fake per-splat scalar"}
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={spotlightOn} onChange={(e) => onSpotlightToggle(e.target.checked)} />
          <Sun className="h-3.5 w-3.5 text-amber-200" /> Spotlight (fade below threshold)
        </label>
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={spotlightThreshold}
            onChange={(e) => onThresholdChange(Number(e.target.value))}
            className="w-full"
          />
          <span className="w-9 shrink-0 text-right text-zinc-400">{spotlightThreshold.toFixed(2)}</span>
        </div>
        <p className="text-[10px] leading-snug text-zinc-500">
          {realScalarLive
            ? "Tint + spotlight now read the real per-splat relevancy for the last query (turbo-style ramp: purple = low, yellow = high)."
            : "Scalar is fake (sin-of-index, not a real relevancy score) — this only proves the per-splat-index → texture → shader wiring for the language-field heatmap."}
        </p>
      </Card>
    </div>
  );
}

// Fake per-splat scalar (0..1), driven purely by load-order index — a stand-in for a
// real relevancy score. ~6 bands across the whole scene regardless of splat count, so
// the tint is visible whether the scene has 50k or 3M splats.
function buildFakeScalarArray(numSplats: number): Uint8Array {
  const array = new Uint8Array(numSplats * 4);
  const cycles = 6;
  const freq = (2 * Math.PI * cycles) / Math.max(numSplats, 1);
  for (let i = 0; i < numSplats; i += 1) {
    const t = Math.sin(i * freq) * 0.5 + 0.5;
    const byte = Math.max(0, Math.min(255, Math.round(t * 255)));
    const o = i * 4;
    array[o] = byte;
    array[o + 1] = byte;
    array[o + 2] = byte;
    array[o + 3] = 255;
  }
  return array;
}

// GPU worldModifier: reads the fake per-splat scalar back out of the RgbaArray texture
// (Spark's splat-painter example mechanism — readRgbaArray keyed by splat index),
// tints rgb through a 5-stop viridis-ish ramp, and fades opacity below a threshold.
function buildHeatmapModifier({
  scalarArray,
  heatmapEnabled,
  spotlightEnabled,
  spotlightThreshold,
}: {
  scalarArray: RgbaArray;
  heatmapEnabled: ReturnType<typeof dyno.dynoBool>;
  spotlightEnabled: ReturnType<typeof dyno.dynoBool>;
  spotlightThreshold: ReturnType<typeof dyno.dynoFloat>;
}) {
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    if (!gsplat) throw new Error("heatmap modifier: no gsplat input");
    const { rgb, opacity, index } = dyno.splitGsplat(gsplat).outputs;
    const raw = readRgbaArray(scalarArray.dyno, index);
    // NOTE: dyno's own .d.ts types `swizzle`'s single-component selector incorrectly
    // (a template-literal type that never matches a bare "x"/"w"), so single-component
    // reads go through `split(...).outputs.x` instead — same result, correctly typed.
    const t = dyno.split(raw).outputs.x;

    const stop0 = dyno.dynoVec3(new THREE.Vector3(0.267, 0.005, 0.329));
    const stop1 = dyno.dynoVec3(new THREE.Vector3(0.229, 0.322, 0.545));
    const stop2 = dyno.dynoVec3(new THREE.Vector3(0.128, 0.567, 0.551));
    const stop3 = dyno.dynoVec3(new THREE.Vector3(0.369, 0.789, 0.383));
    const stop4 = dyno.dynoVec3(new THREE.Vector3(0.993, 0.906, 0.144));
    let ramp = dyno.mix(stop0, stop1, dyno.smoothstep(dyno.dynoFloat(0.0), dyno.dynoFloat(0.25), t));
    ramp = dyno.mix(ramp, stop2, dyno.smoothstep(dyno.dynoFloat(0.25), dyno.dynoFloat(0.5), t));
    ramp = dyno.mix(ramp, stop3, dyno.smoothstep(dyno.dynoFloat(0.5), dyno.dynoFloat(0.75), t));
    ramp = dyno.mix(ramp, stop4, dyno.smoothstep(dyno.dynoFloat(0.75), dyno.dynoFloat(1.0), t));

    const heatmapMask = dyno.float(heatmapEnabled);
    const tintedRgb = dyno.mix(rgb, ramp, heatmapMask);

    const belowThreshold = dyno.lessThan(t, spotlightThreshold);
    const spotlightActive = dyno.and(spotlightEnabled, belowThreshold);
    const dimmedOpacity = dyno.mul(opacity, dyno.dynoFloat(0.04));
    const newOpacity = dyno.select(spotlightActive, dimmedOpacity, opacity);

    const outGsplat = dyno.combineGsplat({ gsplat, rgb: tintedRgb, opacity: newOpacity });
    return { gsplat: outGsplat };
  });
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

// One-shot camera tween (position + up + orbit pivot) for the preset-view buttons.
// Returns a cancel function so a new preset (or unmount) can interrupt cleanly.
function animateCameraTo(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  targetPosition: THREE.Vector3,
  targetUp: THREE.Vector3,
  pivot: THREE.Vector3,
): () => void {
  const startPosition = camera.position.clone();
  const startUp = camera.up.clone();
  const startTarget = controls.target.clone();
  const duration = 550;
  const startTime = performance.now();
  let cancelled = false;
  let raf = 0;
  function tick() {
    if (cancelled) return;
    const elapsed = performance.now() - startTime;
    const t = Math.min(1, elapsed / duration);
    const eased = easeInOutCubic(t);
    camera.position.lerpVectors(startPosition, targetPosition, eased);
    camera.up.lerpVectors(startUp, targetUp, eased).normalize();
    controls.target.lerpVectors(startTarget, pivot, eased);
    camera.lookAt(controls.target);
    controls.update();
    if (t < 1) raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);
  return () => {
    cancelled = true;
    cancelAnimationFrame(raf);
  };
}
