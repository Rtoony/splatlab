import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { dyno, RgbaArray, SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { apiRequest } from "@/lib/api";
import {
  buildOverlayModifier,
  fetchRelevancy,
  packChannelsRgba,
  rampCssGradient,
  RAMPS,
  type OverlayMode,
} from "@/lib/spark-heatmap";
import type { SplatJob } from "@/lib/contracts";
import { Button, SectionLabel } from "@/components/ui";
import { Loader2, Plus, Ruler, Search, Trash2, X } from "lucide-react";

// SPARK BETA viewer for the /view page — the Wave-2 cutover surface.
// - Multi-query language overlay: up to 4 simultaneous text searches, one
//   color each, packed into a single RgbaArray (R/G/B/A channels) and
//   composited by a mode-baked dyno modifier (tint ramp / highlight on
//   natural / isolate / spotlight) with a live legend.
// - Survey dimensions: any number of two-point measurements, draggable
//   endpoints, floating labels, sessionStorage persistence per scene, and
//   known-length scale calibration stored on the scene (meters_per_unit).

const INITIAL_CAMERA_POSITION = new THREE.Vector3(0, -3, 1.4);
const INITIAL_CAMERA_LOOK_AT = new THREE.Vector3(0, 0, 0.2);
const INITIAL_CAMERA_UP = new THREE.Vector3(0, 0, 1);

const M_PER_FT = 0.3048;
const UNIT_TO_M: Record<string, number> = { m: 1, ft: M_PER_FT, in: M_PER_FT / 12 };
const CHANNEL_PALETTE = ["#facc15", "#f472b6", "#22d3ee", "#a3e635"];
const MODES: { key: OverlayMode; label: string; hint: string }[] = [
  { key: "highlight", label: "Highlight", hint: "natural scene, matches take their query color" },
  { key: "isolate", label: "Isolate", hint: "only matches visible, natural colors" },
  { key: "spotlight", label: "Spotlight", hint: "matches colored, rest dimmed" },
  { key: "tint", label: "Ramp", hint: "first query as a full scientific color ramp" },
];

interface QueryChannel {
  text: string;
  color: string; // hex
  bytes: Uint8Array;
  matchCount: number | null;
  relMin: string | null;
  relMax: string | null;
  enabled: boolean;
}

interface Dim {
  id: number;
  a: [number, number, number];
  b: [number, number, number];
}

function hexToRgb01(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

function dimLength(d: Dim): number {
  return Math.hypot(d.a[0] - d.b[0], d.a[1] - d.b[1], d.a[2] - d.b[2]);
}

function formatReal(meters: number): string {
  return `${meters.toFixed(3)} m · ${(meters / M_PER_FT).toFixed(2)} ft`;
}

export function SparkSceneViewer({ job }: { job: SplatJob }) {
  // Langfield scenes MUST load langweb: relevancy rows are exported-ply order
  // and langweb preserves it; web.ply is decimated + reordered.
  const url = `/api/splat/jobs/${job.job_id}/preview/file?fmt=${job.langfield_available ? "langweb" : "web"}`;
  const dimsStorageKey = `splatlab.dims.${job.job_id}`;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const labelsRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<SplatMesh | null>(null);

  const channelsRef = useRef<QueryChannel[]>([]);
  const enabledDynosRef = useRef<ReturnType<typeof dyno.dynoBool>[]>([]);
  const thresholdDynoRef = useRef<ReturnType<typeof dyno.dynoFloat> | null>(null);
  const applyOverlayRef = useRef<() => void>(() => {});

  const dimsRef = useRef<Dim[]>([]);
  const dimGroupRef = useRef<THREE.Group | null>(null);
  const redrawDimsRef = useRef<() => void>(() => {});
  const measureArmRef = useRef(false);
  const pendingPointRef = useRef<[number, number, number] | null>(null);
  const dimLabelEls = useRef<Map<number, HTMLDivElement>>(new Map());

  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [splatCount, setSplatCount] = useState<number | null>(null);
  const [fps, setFps] = useState(0);

  const [channels, setChannels] = useState<QueryChannel[]>([]);
  const [query, setQuery] = useState("");
  const [queryBusy, setQueryBusy] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [mode, setMode] = useState<OverlayMode>("highlight");
  const [rampName, setRampName] = useState("viridis");
  const [threshold, setThreshold] = useState(0.75);

  const [dims, setDims] = useState<Dim[]>([]);
  const [measureArm, setMeasureArm] = useState(false);
  const [hasPending, setHasPending] = useState(false);
  const [metersPerUnit, setMetersPerUnit] = useState<number | null>(job.meters_per_unit ?? null);
  const [calibDimId, setCalibDimId] = useState<number | null>(null);
  const [calibLen, setCalibLen] = useState("");
  const [calibUnit, setCalibUnit] = useState<"m" | "ft" | "in">("ft");
  const [savingScale, setSavingScale] = useState(false);
  const [scaleError, setScaleError] = useState<string | null>(null);

  useEffect(() => {
    measureArmRef.current = measureArm;
    if (!measureArm) {
      pendingPointRef.current = null;
      setHasPending(false);
    }
  }, [measureArm]);

  // ---- dims persistence -------------------------------------------------
  function syncDims(next: Dim[]) {
    dimsRef.current = next;
    setDims([...next]);
    try {
      sessionStorage.setItem(dimsStorageKey, JSON.stringify(next));
    } catch {
      /* storage full/blocked — dims stay session-local in memory */
    }
    redrawDimsRef.current();
  }

  function deleteDim(id: number) {
    syncDims(dimsRef.current.filter((d) => d.id !== id));
    if (calibDimId === id) setCalibDimId(null);
  }

  // ---- overlay (multi-query heatmap) -------------------------------------
  function rebuildOverlay(nextChannels: QueryChannel[], nextMode: OverlayMode, nextRamp: string) {
    const mesh = meshRef.current;
    if (!mesh) return;
    channelsRef.current = nextChannels;
    if (nextChannels.length === 0) {
      mesh.worldModifier = undefined;
      mesh.updateGenerator();
      return;
    }
    const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
    const scalarArray = new RgbaArray({
      array: packChannelsRgba(nextChannels.map((c) => c.bytes), numSplats),
      count: numSplats,
    });
    enabledDynosRef.current = nextChannels.map((c) => dyno.dynoBool(c.enabled));
    if (!thresholdDynoRef.current) thresholdDynoRef.current = dyno.dynoFloat(threshold);
    mesh.worldModifier = buildOverlayModifier({
      scalarArray,
      channelCount: nextChannels.length,
      channelColors: nextChannels.map((c) => hexToRgb01(c.color)),
      channelEnabled: enabledDynosRef.current,
      mode: nextMode,
      ramp: nextRamp,
      threshold: thresholdDynoRef.current,
    });
    mesh.updateGenerator();
  }
  applyOverlayRef.current = () => rebuildOverlay(channelsRef.current, mode, rampName);

  async function addQuery() {
    const mesh = meshRef.current;
    const text = query.trim();
    if (!mesh || !text || channels.length >= 4) return;
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
      const next: QueryChannel[] = [
        ...channelsRef.current,
        {
          text,
          color: CHANNEL_PALETTE[channelsRef.current.length % CHANNEL_PALETTE.length],
          bytes: result.bytes,
          matchCount: result.matchCount,
          relMin: result.relMin,
          relMax: result.relMax,
          enabled: true,
        },
      ];
      setChannels(next);
      rebuildOverlay(next, mode, rampName);
      setQuery("");
    } catch (cause) {
      setQueryError(cause instanceof Error ? cause.message : "Relevancy request failed.");
    } finally {
      setQueryBusy(false);
    }
  }

  function removeQuery(idx: number) {
    const next = channelsRef.current.filter((_, i) => i !== idx);
    setChannels(next);
    rebuildOverlay(next, mode, rampName);
  }

  function toggleQuery(idx: number, enabled: boolean) {
    const next = channelsRef.current.map((c, i) => (i === idx ? { ...c, enabled } : c));
    channelsRef.current = next;
    setChannels(next);
    const dynoBool = enabledDynosRef.current[idx];
    if (dynoBool) {
      dynoBool.value = enabled;
      meshRef.current?.updateVersion();
    }
  }

  function recolorQuery(idx: number, color: string) {
    const next = channelsRef.current.map((c, i) => (i === idx ? { ...c, color } : c));
    setChannels(next);
    rebuildOverlay(next, mode, rampName); // colors are baked -> rebuild
  }

  function changeMode(next: OverlayMode) {
    setMode(next);
    rebuildOverlay(channelsRef.current, next, rampName);
  }

  function changeRamp(next: string) {
    setRampName(next);
    if (mode === "tint") rebuildOverlay(channelsRef.current, mode, next);
  }

  function changeThreshold(next: number) {
    setThreshold(next);
    if (thresholdDynoRef.current) {
      thresholdDynoRef.current.value = next;
      meshRef.current?.updateVersion();
    }
  }

  // ---- scale calibration --------------------------------------------------
  async function saveScale() {
    const dim = dimsRef.current.find((d) => d.id === calibDimId) ?? dimsRef.current[dimsRef.current.length - 1];
    if (!dim) return;
    const sceneDist = dimLength(dim);
    const len = Number(calibLen);
    if (!Number.isFinite(len) || len <= 0 || sceneDist <= 0) {
      setScaleError("Pick a dimension and enter its real length.");
      return;
    }
    setSavingScale(true);
    setScaleError(null);
    try {
      const mpu = (len * UNIT_TO_M[calibUnit]) / sceneDist;
      const resp = await apiRequest<{ meters_per_unit: number | null }>(
        `/api/splat/jobs/${job.job_id}/scale`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ meters_per_unit: mpu }) },
      );
      setMetersPerUnit(resp.meters_per_unit);
    } catch (cause) {
      setScaleError(cause instanceof Error ? cause.message : "Could not save scale.");
    } finally {
      setSavingScale(false);
    }
  }

  // ---- three.js scene ------------------------------------------------------
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    setError(null);
    setReady(false);
    setSplatCount(null);

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

    const dimGroup = new THREE.Group();
    scene.add(dimGroup);
    dimGroupRef.current = dimGroup;

    // restore this scene's dimensions from the session
    try {
      const saved = sessionStorage.getItem(dimsStorageKey);
      if (saved) {
        const parsed = JSON.parse(saved) as Dim[];
        if (Array.isArray(parsed)) {
          dimsRef.current = parsed.filter((d) => d && Array.isArray(d.a) && Array.isArray(d.b));
          setDims([...dimsRef.current]);
        }
      }
    } catch {
      /* corrupted storage — start empty */
    }

    function redrawDims() {
      const group = dimGroupRef.current;
      if (!group) return;
      group.clear();
      for (const d of dimsRef.current) {
        const a = new THREE.Vector3(...d.a);
        const b = new THREE.Vector3(...d.b);
        for (const p of [a, b]) {
          const marker = new THREE.Mesh(
            new THREE.SphereGeometry(0.018, 16, 12),
            new THREE.MeshBasicMaterial({ color: 0x22d3ee }),
          );
          marker.position.copy(p);
          marker.userData = { dimId: d.id };
          group.add(marker);
        }
        group.add(
          new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([a, b]),
            new THREE.LineBasicMaterial({ color: 0x22d3ee }),
          ),
        );
      }
      const pending = pendingPointRef.current;
      if (pending) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.018, 16, 12),
          new THREE.MeshBasicMaterial({ color: 0xfacc15 }),
        );
        marker.position.set(...pending);
        group.add(marker);
      }
    }
    redrawDimsRef.current = redrawDims;
    redrawDims();

    function resize() {
      const el = containerRef.current;
      if (!el) return;
      renderer.setSize(Math.max(el.clientWidth, 1), Math.max(el.clientHeight, 1), false);
      camera.aspect = Math.max(el.clientWidth, 1) / Math.max(el.clientHeight, 1);
      camera.updateProjectionMatrix();
    }
    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);

    const mesh = new SplatMesh({ url, fileType: SplatFileType.PLY, raycastable: true, minRaycastOpacity: 0.1 });
    scene.add(mesh);
    meshRef.current = mesh;

    // floating dimension labels: DOM nodes moved imperatively every frame
    const proj = new THREE.Vector3();
    function positionLabels() {
      const el = containerRef.current;
      if (!el) return;
      const w = el.clientWidth;
      const h = el.clientHeight;
      for (const d of dimsRef.current) {
        const div = dimLabelEls.current.get(d.id);
        if (!div) continue;
        proj.set((d.a[0] + d.b[0]) / 2, (d.a[1] + d.b[1]) / 2, (d.a[2] + d.b[2]) / 2).project(camera);
        if (proj.z > 1) {
          div.style.display = "none";
          continue;
        }
        div.style.display = "block";
        div.style.transform = `translate(-50%, -130%) translate(${((proj.x + 1) / 2) * w}px, ${((1 - proj.y) / 2) * h}px)`;
      }
    }

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
      positionLabels();
      renderer.render(scene, camera);
    }
    animate();

    mesh.initialized
      .then(() => {
        if (disposed) return;
        setSplatCount(mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0);
        setReady(true);
        applyOverlayRef.current(); // re-apply overlay after a reload
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
      intersects.sort((x, y) => x.distance - y.distance);
      return intersects[0].point.clone();
    }

    // endpoint screen-space hit test (px) for drag-to-move
    function endpointAt(clientX: number, clientY: number): { dimId: number; end: "a" | "b" } | null {
      const el = containerRef.current;
      if (!el) return null;
      const rect = el.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      let best: { dimId: number; end: "a" | "b"; px: number } | null = null;
      for (const d of dimsRef.current) {
        for (const end of ["a", "b"] as const) {
          proj.set(...d[end]).project(camera);
          if (proj.z > 1) continue;
          const sx = ((proj.x + 1) / 2) * w + rect.left;
          const sy = ((1 - proj.y) / 2) * h + rect.top;
          const px = Math.hypot(sx - clientX, sy - clientY);
          if (px < 14 && (!best || px < best.px)) best = { dimId: d.id, end, px };
        }
      }
      return best ? { dimId: best.dimId, end: best.end } : null;
    }

    let downX = 0;
    let downY = 0;
    let drag: { dimId: number; end: "a" | "b" } | null = null;

    function onPointerDown(e: PointerEvent) {
      downX = e.clientX;
      downY = e.clientY;
      if (e.button !== 0) return;
      const hit = endpointAt(e.clientX, e.clientY);
      if (hit) {
        drag = hit;
        controls.enabled = false;
        renderer.domElement.setPointerCapture(e.pointerId);
      }
    }
    function onPointerMove(e: PointerEvent) {
      if (!drag) return;
      const p = raycastAt(e.clientX, e.clientY);
      if (!p) return;
      const d = dimsRef.current.find((x) => x.id === drag!.dimId);
      if (!d) return;
      d[drag.end] = [p.x, p.y, p.z];
      redrawDims();
    }
    function onPointerUp(e: PointerEvent) {
      if (!drag) return;
      drag = null;
      controls.enabled = true;
      try {
        renderer.domElement.releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
      // commit: sync list UI + storage
      dimsRef.current = [...dimsRef.current];
      setDims([...dimsRef.current]);
      try {
        sessionStorage.setItem(dimsStorageKey, JSON.stringify(dimsRef.current));
      } catch {
        /* ignore */
      }
    }
    function onClick(e: MouseEvent) {
      if (!measureArmRef.current) return;
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
      const hit = raycastAt(e.clientX, e.clientY);
      if (!hit) return;
      const pending = pendingPointRef.current;
      if (!pending) {
        pendingPointRef.current = [hit.x, hit.y, hit.z];
        setHasPending(true);
        redrawDims();
      } else {
        pendingPointRef.current = null;
        setHasPending(false);
        const id = Date.now();
        dimsRef.current = [...dimsRef.current, { id, a: pending, b: [hit.x, hit.y, hit.z] }];
        setDims([...dimsRef.current]);
        try {
          sessionStorage.setItem(dimsStorageKey, JSON.stringify(dimsRef.current));
        } catch {
          /* ignore */
        }
        redrawDims();
      }
    }
    function onDoubleClick(event: MouseEvent) {
      if (measureArmRef.current) return; // measuring owns clicks
      const hit = raycastAt(event.clientX, event.clientY);
      if (hit) {
        controls.target.copy(hit);
        camera.lookAt(controls.target);
        controls.update();
      }
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("click", onClick);
    renderer.domElement.addEventListener("dblclick", onDoubleClick);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
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
      dimGroupRef.current = null;
      enabledDynosRef.current = [];
      thresholdDynoRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  const calibDim = dims.find((d) => d.id === calibDimId) ?? dims[dims.length - 1] ?? null;

  return (
    <div className="relative h-full w-full overflow-hidden bg-black/70">
      <div ref={containerRef} className="absolute inset-0 [&>canvas]:h-full [&>canvas]:w-full" />

      {/* floating dimension labels */}
      <div ref={labelsRef} className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
        {dims.map((d) => {
          const len = dimLength(d);
          return (
            <div
              key={d.id}
              ref={(el) => {
                if (el) dimLabelEls.current.set(d.id, el);
                else dimLabelEls.current.delete(d.id);
              }}
              className="absolute left-0 top-0 whitespace-nowrap rounded-md border border-cyan-300/30 bg-black/75 px-1.5 py-0.5 text-[10px] font-semibold text-cyan-200 shadow backdrop-blur-sm"
            >
              {metersPerUnit ? formatReal(len * metersPerUnit) : `${len.toFixed(3)} u`}
            </div>
          );
        })}
      </div>

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

      {/* control panel */}
      <div className="absolute left-3 top-3 z-20 max-h-[calc(100%-1.5rem)] w-80 space-y-3 overflow-y-auto rounded-xl border border-white/10 bg-black/70 p-3 text-xs text-zinc-200 shadow backdrop-blur-md">
        <div className="flex items-center justify-between">
          <span className="font-semibold uppercase tracking-widest text-cyan-300/90">Spark beta</span>
          <span className="text-zinc-400">
            {splatCount !== null ? `${splatCount.toLocaleString()} splats` : "…"} · {fps} fps
          </span>
        </div>

        {job.langfield_available ? (
          <>
            <SectionLabel>Language overlay</SectionLabel>
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void addQuery();
              }}
            >
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={channels.length >= 4 ? "4 query limit reached" : "Add a search…"}
                disabled={channels.length >= 4}
                className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-600 focus:border-cyan-400/50 focus:outline-none disabled:opacity-50"
              />
              <Button type="submit" size="sm" disabled={queryBusy || !query.trim() || channels.length >= 4}>
                {queryBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
              </Button>
            </form>
            {queryError && <p className="text-[10px] leading-snug text-rose-300/90">{queryError}</p>}

            {channels.map((c, i) => (
              <div key={`${c.text}-${i}`} className="flex items-center gap-2">
                <input type="checkbox" checked={c.enabled} onChange={(e) => toggleQuery(i, e.target.checked)} />
                <input
                  type="color"
                  value={c.color}
                  onChange={(e) => recolorQuery(i, e.target.value)}
                  className="h-5 w-6 shrink-0 cursor-pointer rounded border-0 bg-transparent p-0"
                  title="Query color"
                />
                <span className="min-w-0 flex-1 truncate" title={c.text}>
                  {c.text}
                </span>
                <span className="shrink-0 text-[10px] text-zinc-500">
                  {c.matchCount !== null ? `${c.matchCount}×` : ""}
                </span>
                <button type="button" onClick={() => removeQuery(i)} className="shrink-0 text-zinc-500 hover:text-rose-300">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}

            {channels.length > 0 && (
              <>
                <div className="flex flex-wrap gap-1">
                  {MODES.map((m) => (
                    <button
                      key={m.key}
                      type="button"
                      title={m.hint}
                      onClick={() => changeMode(m.key)}
                      className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
                        mode === m.key
                          ? "border-cyan-300/60 bg-cyan-300/20 text-cyan-200"
                          : "border-white/10 bg-white/5 text-zinc-400 hover:text-zinc-200"
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
                {mode === "tint" ? (
                  <select
                    value={rampName}
                    onChange={(e) => changeRamp(e.target.value)}
                    className="w-full rounded border border-white/10 bg-white/5 px-1.5 py-1 text-xs text-zinc-100 focus:outline-none"
                  >
                    {Object.keys(RAMPS).map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                ) : (
                  <div className="flex items-center gap-2">
                    <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">match ≥</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={threshold}
                      onChange={(e) => changeThreshold(Number(e.target.value))}
                      className="w-full"
                    />
                    <span className="w-9 shrink-0 text-right text-zinc-400">{threshold.toFixed(2)}</span>
                  </div>
                )}
              </>
            )}
          </>
        ) : (
          <p className="text-[10px] leading-snug text-zinc-500">
            No language field on this scene — re-run with the language field enabled to search it.
          </p>
        )}

        <div className="h-px bg-white/10" />
        <SectionLabel>Dimensions</SectionLabel>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant={measureArm ? "primary" : "outline"}
            onClick={() => setMeasureArm((v) => !v)}
            title="Click two points per dimension; drag endpoints to adjust"
          >
            <Ruler className="h-3.5 w-3.5" /> {measureArm ? (hasPending ? "Pick 2nd point…" : "Adding…") : "Add dimension"}
          </Button>
          {dims.length > 0 && (
            <Button type="button" size="sm" variant="outline" onClick={() => syncDims([])} title="Delete all dimensions">
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
        {measureArm && (
          <p className="text-[10px] leading-snug text-zinc-500">
            Click two points on the scene. Drag any endpoint later to adjust; dimensions persist for this
            browser session.
          </p>
        )}
        {dims.map((d, i) => {
          const len = dimLength(d);
          return (
            <div key={d.id} className={`flex items-center gap-2 ${calibDim?.id === d.id ? "text-cyan-200" : ""}`}>
              <button
                type="button"
                className="min-w-0 flex-1 truncate text-left hover:text-cyan-200"
                title="Use this dimension for scale calibration"
                onClick={() => setCalibDimId(d.id)}
              >
                #{i + 1} · {len.toFixed(3)} u{metersPerUnit ? ` = ${formatReal(len * metersPerUnit)}` : ""}
              </button>
              <button type="button" onClick={() => deleteDim(d.id)} className="shrink-0 text-zinc-500 hover:text-rose-300">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          );
        })}
        {dims.length > 0 && (
          <div className="space-y-1">
            <p className="text-[10px] leading-snug text-zinc-500">
              Calibrate: select a dimension of known length{calibDim ? ` (#${dims.findIndex((d) => d.id === calibDim.id) + 1})` : ""}, enter it, set scale.
            </p>
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
              <Button type="button" size="sm" onClick={() => void saveScale()} disabled={savingScale || !calibDim}>
                {savingScale ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Set scale"}
              </Button>
            </div>
            {scaleError && <p className="text-[10px] leading-snug text-rose-300/90">{scaleError}</p>}
            {metersPerUnit && (
              <p className="text-[10px] leading-snug text-zinc-500">
                Scale: 1 scene unit = {metersPerUnit.toFixed(4)} m (stored on the scene).
              </p>
            )}
          </div>
        )}
      </div>

      {/* legend — updates live with queries, colors, mode, and threshold */}
      {channels.length > 0 && (
        <div className="absolute bottom-3 right-3 z-20 w-64 space-y-2 rounded-xl border border-white/10 bg-black/70 p-3 text-xs text-zinc-200 shadow backdrop-blur-md">
          <SectionLabel>Legend</SectionLabel>
          {mode === "tint" ? (
            <>
              <div className="h-3 w-full rounded" style={{ background: rampCssGradient(rampName) }} />
              <div className="flex justify-between text-[10px] text-zinc-400">
                <span>{channels[0].relMin ? Number(channels[0].relMin).toFixed(2) : "low"}</span>
                <span className="truncate px-2 text-zinc-300">“{channels[0].text}” relevancy</span>
                <span>{channels[0].relMax ? Number(channels[0].relMax).toFixed(2) : "high"}</span>
              </div>
              {channels.length > 1 && (
                <p className="text-[10px] text-zinc-500">Ramp mode shows the first query only.</p>
              )}
            </>
          ) : (
            <>
              {channels.map((c, i) => (
                <div key={`${c.text}-legend-${i}`} className={`flex items-center gap-2 ${c.enabled ? "" : "opacity-40"}`}>
                  <span className="h-3 w-3 shrink-0 rounded-sm" style={{ background: c.color }} />
                  <span className="min-w-0 flex-1 truncate">“{c.text}”</span>
                  <span className="shrink-0 text-[10px] text-zinc-500">
                    {c.matchCount !== null ? `${c.matchCount} found` : ""}
                  </span>
                </div>
              ))}
              <p className="text-[10px] text-zinc-500">
                {mode === "highlight" && `Colored = relevancy above ${threshold.toFixed(2)}; rest natural.`}
                {mode === "isolate" && `Only matches above ${threshold.toFixed(2)} visible, natural colors.`}
                {mode === "spotlight" && `Matches above ${threshold.toFixed(2)} colored; rest dimmed.`}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
