import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { dyno, RgbaArray, SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { apiRequest } from "@/lib/api";
import { buildHeatmapModifier, fetchRelevancy, packRelevancyRgba } from "@/lib/spark-heatmap";
import type { SplatJob } from "@/lib/contracts";
import { Button, SectionLabel } from "@/components/ui";
import { Flame, Loader2, Ruler, Search, Sun, Trash2 } from "lucide-react";

// SPARK BETA viewer for the /view page — the Wave-2 cutover surface. Carries
// the capabilities the classic mkkellogg viewer cannot: real per-splat language
// heatmap + spotlight (langfield scenes), and the survey lane's two-point
// measure with scale calibration (Spark raycasting makes picking native).
// Opt-in beside the classic viewer; camera overlays/search fly-to stay
// classic-only until the full 2.4 cutover.

const INITIAL_CAMERA_POSITION = new THREE.Vector3(0, -3, 1.4);
const INITIAL_CAMERA_LOOK_AT = new THREE.Vector3(0, 0, 0.2);
const INITIAL_CAMERA_UP = new THREE.Vector3(0, 0, 1);

const M_PER_FT = 0.3048;
const UNIT_TO_M: Record<string, number> = { m: 1, ft: M_PER_FT, in: M_PER_FT / 12 };

function formatReal(meters: number): string {
  const feet = meters / M_PER_FT;
  return `${meters.toFixed(3)} m · ${feet.toFixed(2)} ft`;
}

export function SparkSceneViewer({ job }: { job: SplatJob }) {
  // Langfield scenes MUST load the langweb variant: relevancy rows are served
  // in exported-ply order and langweb preserves it; web.ply is decimated +
  // reordered and can never carry per-splat data.
  const url = `/api/splat/jobs/${job.job_id}/preview/file?fmt=${job.langfield_available ? "langweb" : "web"}`;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<SplatMesh | null>(null);
  const measureGroupRef = useRef<THREE.Group | null>(null);
  const measurePointsRef = useRef<THREE.Vector3[]>([]);
  const measureModeRef = useRef(false);

  const heatmapEnabledRef = useRef<ReturnType<typeof dyno.dynoBool> | null>(null);
  const spotlightEnabledRef = useRef<ReturnType<typeof dyno.dynoBool> | null>(null);
  const spotlightThresholdRef = useRef<ReturnType<typeof dyno.dynoFloat> | null>(null);

  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [fps, setFps] = useState(0);

  const [query, setQuery] = useState("");
  const [queryBusy, setQueryBusy] = useState(false);
  const [queryStatus, setQueryStatus] = useState<string | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [heatmapLive, setHeatmapLive] = useState(false);
  const [heatmapOn, setHeatmapOn] = useState(false);
  const [spotlightOn, setSpotlightOn] = useState(false);
  const [spotlightThreshold, setSpotlightThreshold] = useState(0.5);

  const [measureMode, setMeasureMode] = useState(false);
  const [sceneDist, setSceneDist] = useState<number | null>(null);
  const [metersPerUnit, setMetersPerUnit] = useState<number | null>(job.meters_per_unit ?? null);
  const [calibLen, setCalibLen] = useState("");
  const [calibUnit, setCalibUnit] = useState<"m" | "ft" | "in">("ft");
  const [savingScale, setSavingScale] = useState(false);
  const [scaleError, setScaleError] = useState<string | null>(null);

  useEffect(() => {
    measureModeRef.current = measureMode;
  }, [measureMode]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    setError(null);
    setReady(false);
    setSplatCount(null);
    setQueryStatus(null);
    setQueryError(null);
    setHeatmapLive(false);
    setSceneDist(null);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.up.copy(INITIAL_CAMERA_UP);
    camera.position.copy(INITIAL_CAMERA_POSITION);
    camera.lookAt(INITIAL_CAMERA_LOOK_AT);
    cameraRef.current = camera;

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

    const measureGroup = new THREE.Group();
    scene.add(measureGroup);
    measureGroupRef.current = measureGroup;
    measurePointsRef.current = [];

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

    heatmapEnabledRef.current = dyno.dynoBool(false);
    spotlightEnabledRef.current = dyno.dynoBool(false);
    spotlightThresholdRef.current = dyno.dynoFloat(0.5);

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
      if (fpsAccum >= 0.5) {
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
        setSplatCount(mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0);
        setReady(true);
      })
      .catch((cause: unknown) => {
        if (disposed) return;
        setError(cause instanceof Error ? cause.message : "Could not load Spark preview.");
      });

    function raycastAt(clientX: number, clientY: number): THREE.Vector3 | null {
      const el = containerRef.current;
      if (!el || !mesh.raycastable) return null;
      const rect = el.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, camera);
      const intersects: { distance: number; point: THREE.Vector3; object: THREE.Object3D }[] = [];
      mesh.raycast(raycaster, intersects);
      if (intersects.length === 0) return null;
      intersects.sort((a, b) => a.distance - b.distance);
      return intersects[0].point.clone();
    }

    function redrawMeasure() {
      const group = measureGroupRef.current;
      if (!group) return;
      group.clear();
      const pts = measurePointsRef.current;
      for (const p of pts) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.02, 16, 12),
          new THREE.MeshBasicMaterial({ color: 0x22d3ee }),
        );
        marker.position.copy(p);
        group.add(marker);
      }
      if (pts.length === 2) {
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        group.add(new THREE.Line(geom, new THREE.LineBasicMaterial({ color: 0x22d3ee })));
      }
    }

    // Two-point picking: 'click' with a drag guard so orbit drags never place
    // points. Third click starts a fresh measurement.
    let downX = 0;
    let downY = 0;
    function onPointerDown(e: PointerEvent) {
      downX = e.clientX;
      downY = e.clientY;
    }
    function onClick(e: MouseEvent) {
      if (!measureModeRef.current) return;
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
      const hit = raycastAt(e.clientX, e.clientY);
      if (!hit) return;
      let pts = measurePointsRef.current;
      pts = pts.length >= 2 ? [hit] : [...pts, hit];
      measurePointsRef.current = pts;
      redrawMeasure();
      setSceneDist(pts.length === 2 ? pts[0].distanceTo(pts[1]) : null);
    }

    function onDoubleClick(event: MouseEvent) {
      if (measureModeRef.current) return; // measure owns clicks while active
      const hit = raycastAt(event.clientX, event.clientY);
      if (hit) {
        controls.target.copy(hit);
        camera.lookAt(controls.target);
        controls.update();
      }
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("click", onClick);
    renderer.domElement.addEventListener("dblclick", onDoubleClick);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("click", onClick);
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
      measureGroupRef.current = null;
      heatmapEnabledRef.current = null;
      spotlightEnabledRef.current = null;
      spotlightThresholdRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  async function runQuery() {
    const mesh = meshRef.current;
    const heatmapEnabled = heatmapEnabledRef.current;
    const spotlightEnabled = spotlightEnabledRef.current;
    const spotlightThresholdDyno = spotlightThresholdRef.current;
    const text = query.trim();
    if (!mesh || !heatmapEnabled || !spotlightEnabled || !spotlightThresholdDyno || !text) return;
    setQueryBusy(true);
    setQueryError(null);
    try {
      const result = await fetchRelevancy(job.job_id, text);
      const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
      if (result.bytes.length !== numSplats) {
        throw new Error(
          `relevancy rows (${result.bytes.length.toLocaleString()}) != loaded splats (${numSplats.toLocaleString()}) — scene not loaded as langweb?`,
        );
      }
      const scalarArray = new RgbaArray({ array: packRelevancyRgba(result.bytes), count: numSplats });
      mesh.worldModifier = buildHeatmapModifier({
        scalarArray,
        heatmapEnabled,
        spotlightEnabled,
        spotlightThreshold: spotlightThresholdDyno,
      });
      mesh.updateGenerator();
      setHeatmapLive(true);
      setHeatmapOn(true);
      heatmapEnabled.value = true;
      mesh.updateVersion();
      setQueryStatus(
        `"${text}" · ${result.bytes.length.toLocaleString()} splats` +
          (result.matchCount !== null ? ` · ${result.matchCount} instance${result.matchCount === 1 ? "" : "s"}` : "") +
          ` · ${result.ms}ms`,
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
    meshRef.current?.updateVersion();
  }
  function onSpotlightToggle(next: boolean) {
    setSpotlightOn(next);
    if (spotlightEnabledRef.current) spotlightEnabledRef.current.value = next;
    meshRef.current?.updateVersion();
  }
  function onThresholdChange(next: number) {
    setSpotlightThreshold(next);
    if (spotlightThresholdRef.current) spotlightThresholdRef.current.value = next;
    meshRef.current?.updateVersion();
  }

  function clearMeasure() {
    measurePointsRef.current = [];
    measureGroupRef.current?.clear();
    setSceneDist(null);
  }

  async function saveScale() {
    if (!sceneDist || sceneDist <= 0) return;
    const len = Number(calibLen);
    if (!Number.isFinite(len) || len <= 0) {
      setScaleError("Enter the reference's real length first.");
      return;
    }
    setSavingScale(true);
    setScaleError(null);
    try {
      const mpu = (len * UNIT_TO_M[calibUnit]) / sceneDist;
      const resp = await apiRequest<{ meters_per_unit: number | null }>(
        `/api/splat/jobs/${job.job_id}/scale`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ meters_per_unit: mpu }),
        },
      );
      setMetersPerUnit(resp.meters_per_unit);
    } catch (cause) {
      setScaleError(cause instanceof Error ? cause.message : "Could not save scale.");
    } finally {
      setSavingScale(false);
    }
  }

  return (
    <div className="relative h-full w-full overflow-hidden bg-black/70">
      <div ref={containerRef} className="absolute inset-0 [&>canvas]:h-full [&>canvas]:w-full" />

      {!ready && !error && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-400">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading Spark preview…
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="absolute left-3 top-3 z-20 w-72 space-y-3 rounded-xl border border-white/10 bg-black/70 p-3 text-xs text-zinc-200 shadow backdrop-blur-md">
        <div className="flex items-center justify-between">
          <span className="font-semibold uppercase tracking-widest text-cyan-300/90">Spark beta</span>
          <span className="text-zinc-400">
            {splatCount !== null ? `${splatCount.toLocaleString()} splats` : "…"} · {fps} fps
          </span>
        </div>

        {job.langfield_available ? (
          <>
            <SectionLabel>Language heatmap</SectionLabel>
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void runQuery();
              }}
            >
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search this scene…"
                className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-600 focus:border-cyan-400/50 focus:outline-none"
              />
              <Button type="submit" size="sm" disabled={queryBusy || !query.trim()}>
                {queryBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
              </Button>
            </form>
            {queryStatus && <p className="text-[10px] leading-snug text-emerald-300/90">{queryStatus}</p>}
            {queryError && <p className="text-[10px] leading-snug text-rose-300/90">{queryError}</p>}
            {heatmapLive && (
              <>
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={heatmapOn} onChange={(e) => onHeatmapToggle(e.target.checked)} />
                  <Flame className="h-3.5 w-3.5 text-orange-300" /> Heatmap tint
                </label>
                <label className="flex items-center gap-2">
                  <input type="checkbox" checked={spotlightOn} onChange={(e) => onSpotlightToggle(e.target.checked)} />
                  <Sun className="h-3.5 w-3.5 text-amber-200" /> Spotlight matches
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
              </>
            )}
          </>
        ) : (
          <p className="text-[10px] leading-snug text-zinc-500">
            No language field on this scene — re-run with the language field enabled to search it.
          </p>
        )}

        <div className="h-px bg-white/10" />
        <SectionLabel>Measure</SectionLabel>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant={measureMode ? "primary" : "outline"}
            onClick={() => setMeasureMode((v) => !v)}
            title="Click two points on the scene to measure between them"
          >
            <Ruler className="h-3.5 w-3.5" /> {measureMode ? "Measuring…" : "Measure"}
          </Button>
          <Button type="button" size="sm" variant="outline" onClick={clearMeasure} title="Clear measurement">
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
        {measureMode && sceneDist === null && (
          <p className="text-[10px] leading-snug text-zinc-500">
            Click the two endpoints (double-click pivot is paused while measuring).
          </p>
        )}
        {sceneDist !== null && (
          <div className="space-y-1">
            <p className="font-semibold text-cyan-200">
              {sceneDist.toFixed(4)} scene units
              {metersPerUnit ? <span className="text-emerald-300"> = {formatReal(sceneDist * metersPerUnit)}</span> : null}
            </p>
            {!metersPerUnit && (
              <p className="text-[10px] leading-snug text-zinc-500">
                Scene is unscaled — calibrate below with a known length (tape a yardstick into the capture).
              </p>
            )}
            <div className="flex items-center gap-1.5">
              <input
                value={calibLen}
                onChange={(e) => setCalibLen(e.target.value)}
                placeholder="known length"
                inputMode="decimal"
                className="w-24 rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-600 focus:border-cyan-400/50 focus:outline-none"
              />
              <select
                value={calibUnit}
                onChange={(e) => setCalibUnit(e.target.value as "m" | "ft" | "in")}
                className="rounded border border-white/10 bg-white/5 px-1.5 py-1 text-xs text-zinc-100 focus:outline-none"
              >
                <option value="ft">ft</option>
                <option value="in">in</option>
                <option value="m">m</option>
              </select>
              <Button type="button" size="sm" onClick={() => void saveScale()} disabled={savingScale}>
                {savingScale ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Set scale"}
              </Button>
            </div>
            {scaleError && <p className="text-[10px] leading-snug text-rose-300/90">{scaleError}</p>}
          </div>
        )}
        {metersPerUnit && (
          <p className="text-[10px] leading-snug text-zinc-500">
            Scale: 1 scene unit = {metersPerUnit.toFixed(4)} m (stored on the scene).
          </p>
        )}
      </div>
    </div>
  );
}
