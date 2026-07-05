import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { dyno, RgbaArray, SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { apiRequest } from "@/lib/api";
import {
  buildOverlayModifier,
  cutoffForTopPercent,
  fetchRelevancy,
  packChannelsRgba,
  rampCssGradient,
  RAMPS,
  type OverlayMode,
} from "@/lib/spark-heatmap";
import type { SplatJob } from "@/lib/contracts";
import { Button, SectionLabel } from "@/components/ui";
import { Loader2, Paintbrush, Plus, Ruler, Trash2, Undo2, X } from "lucide-react";

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

// chunked btoa — a big selection is a few MB of uint32s and one giant binary
// string would blow the btoa argument path
function b64FromUint32(arr: Uint32Array): string {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

interface OverrideRecord {
  id: string;
  label: string;
  aliases: string[];
  op: string;
  count: number;
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
  const cutoffDynosRef = useRef<ReturnType<typeof dyno.dynoFloat>[]>([]);
  const applyOverlayRef = useRef<() => void>(() => {});
  const refreshModifierRef = useRef<() => void>(() => {});

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
  // "top X%" per query — relevancy is per-query min-max normalized, so a raw
  // shared threshold means something different for every query; a percentile
  // is the semantics a human expects ("light up this much of the scene").
  const [topPct, setTopPct] = useState(2);
  const topPctRef = useRef(2);

  // paint-the-embeddings (langfield scenes only)
  const paintModeRef = useRef(false);
  const strokeAtRef = useRef<(p: THREE.Vector3) => void>(() => {});
  const selSetRef = useRef<Set<number>>(new Set());
  const strokesRef = useRef<Uint32Array[]>([]);
  const [paintMode, setPaintMode] = useState(false);
  const [brushRadius, setBrushRadius] = useState(0.1);
  const brushRadiusRef = useRef(0.1);
  const [selCount, setSelCount] = useState(0);
  const [strokeBusy, setStrokeBusy] = useState(false);
  const [paintLabel, setPaintLabel] = useState("");
  const [paintAliases, setPaintAliases] = useState("");
  const [paintOp, setPaintOp] = useState<"assign" | "boost" | "suppress">("assign");
  const [limitToQuery, setLimitToQuery] = useState(false);
  const [paintBusy, setPaintBusy] = useState(false);
  const [paintError, setPaintError] = useState<string | null>(null);
  const [paintNeedsForce, setPaintNeedsForce] = useState(false);
  const [paintNotice, setPaintNotice] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<OverrideRecord[]>([]);

  const [dims, setDims] = useState<Dim[]>([]);
  const [measureArm, setMeasureArm] = useState(false);
  const [hasPending, setHasPending] = useState(false);
  const [metersPerUnit, setMetersPerUnit] = useState<number | null>(job.meters_per_unit ?? null);
  const [calibDimId, setCalibDimId] = useState<number | null>(null);
  const [calibLen, setCalibLen] = useState("");
  const [calibUnit, setCalibUnit] = useState<"m" | "ft" | "in">("ft");
  const [savingScale, setSavingScale] = useState(false);
  const [scaleError, setScaleError] = useState<string | null>(null);
  const [recalibrateArmed, setRecalibrateArmed] = useState(false);

  useEffect(() => {
    if (!recalibrateArmed) return;
    const t = window.setTimeout(() => setRecalibrateArmed(false), 3000);
    return () => window.clearTimeout(t);
  }, [recalibrateArmed]);

  useEffect(() => {
    measureArmRef.current = measureArm;
    if (!measureArm) {
      pendingPointRef.current = null;
      setHasPending(false);
    }
  }, [measureArm]);

  useEffect(() => {
    paintModeRef.current = paintMode;
    if (paintMode) setMeasureArm(false); // one click-owner at a time
    setPaintError(null); // never carry a stale error across a mode flip
    setPaintNeedsForce(false);
    // entering/leaving paint swaps the modifier (selection preview <-> queries)
    refreshModifierRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paintMode]);

  useEffect(() => {
    brushRadiusRef.current = brushRadius;
  }, [brushRadius]);

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
    if (calibDimId === id) {
      setCalibDimId(null);
      setRecalibrateArmed(false);
    }
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
    cutoffDynosRef.current = nextChannels.map((c) =>
      dyno.dynoFloat(cutoffForTopPercent(c.bytes, topPctRef.current)),
    );
    const firstEnabled = Math.max(0, nextChannels.findIndex((c) => c.enabled));
    mesh.worldModifier = buildOverlayModifier({
      scalarArray,
      channelCount: nextChannels.length,
      channelColors: nextChannels.map((c) => hexToRgb01(c.color)),
      channelEnabled: enabledDynosRef.current,
      mode: nextMode,
      ramp: nextRamp,
      channelCutoffs: cutoffDynosRef.current,
      tintChannel: firstEnabled,
    });
    mesh.updateGenerator();
  }
  applyOverlayRef.current = () => refreshModifierRef.current();

  // ---- paint-the-embeddings ----------------------------------------------
  function previewSelection() {
    const mesh = meshRef.current;
    if (!mesh) return;
    const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
    if (numSplats === 0) return;
    const bytes = new Uint8Array(numSplats);
    for (const i of selSetRef.current) if (i < numSplats) bytes[i] = 255;
    const scalarArray = new RgbaArray({ array: packChannelsRgba([bytes], numSplats), count: numSplats });
    mesh.worldModifier = buildOverlayModifier({
      scalarArray,
      channelCount: 1,
      channelColors: [[0.13, 0.83, 0.93]], // selection cyan
      channelEnabled: [dyno.dynoBool(true)],
      mode: "highlight",
      ramp: rampName,
      channelCutoffs: [dyno.dynoFloat(0.5)],
    });
    mesh.updateGenerator();
  }

  function refreshModifier() {
    if (paintModeRef.current && selSetRef.current.size > 0) previewSelection();
    else rebuildOverlay(channelsRef.current, mode, rampName);
  }
  refreshModifierRef.current = refreshModifier;

  async function strokeAt(p: THREE.Vector3) {
    setStrokeBusy(true);
    setPaintError(null);
    try {
      const res = await fetch(`/api/splat/jobs/${job.job_id}/langfield/select/sphere`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ center: [p.x, p.y, p.z], radius: brushRadiusRef.current }),
      });
      if (res.status === 404 || res.status === 405) {
        throw new Error(
          "Paint backend not deployed yet — it goes live on the next splatlab restart (waiting for the running job to finish).",
        );
      }
      if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
      const idx = new Uint32Array(await res.arrayBuffer());
      // optional hygiene: clip the sloppy sphere to the current query's matches
      let clipped: Uint32Array | number[] = idx;
      if (limitToQuery) {
        const ch = channelsRef.current.find((c) => c.enabled);
        if (ch) {
          const cut = Math.round(cutoffForTopPercent(ch.bytes, topPctRef.current) * 255);
          const keep: number[] = [];
          for (const i of idx) if (ch.bytes[i] >= cut) keep.push(i);
          clipped = keep;
        }
      }
      const delta: number[] = [];
      for (const i of clipped) {
        if (!selSetRef.current.has(i)) {
          selSetRef.current.add(i);
          delta.push(i);
        }
      }
      if (delta.length) strokesRef.current.push(Uint32Array.from(delta));
      setSelCount(selSetRef.current.size);
      refreshModifierRef.current();
    } catch (cause) {
      setPaintError(cause instanceof Error ? cause.message : "Brush stroke failed.");
    } finally {
      setStrokeBusy(false);
    }
  }
  strokeAtRef.current = (p) => void strokeAt(p);

  function undoStroke() {
    const last = strokesRef.current.pop();
    if (!last) return;
    for (const i of last) selSetRef.current.delete(i);
    setSelCount(selSetRef.current.size);
    refreshModifierRef.current();
  }

  function clearSelection() {
    selSetRef.current.clear();
    strokesRef.current = [];
    setSelCount(0);
    setPaintNeedsForce(false);
    refreshModifierRef.current();
  }

  async function loadOverridesList() {
    try {
      const data = await apiRequest<{ overrides: OverrideRecord[] }>(
        `/api/splat/jobs/${job.job_id}/langfield/overrides`,
      );
      setOverrides(data.overrides ?? []);
    } catch {
      /* list is a convenience — leave whatever we had */
    }
  }

  async function commitPaint(force = false) {
    const label = paintLabel.trim();
    const n = selSetRef.current.size;
    if (!label || n === 0) return;
    setPaintBusy(true);
    setPaintError(null);
    setPaintNotice(null);
    try {
      const resp = await fetch(`/api/splat/jobs/${job.job_id}/langfield/overrides`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label,
          aliases: paintAliases.split(",").map((s) => s.trim()).filter(Boolean),
          op: paintOp,
          indices_b64: b64FromUint32(Uint32Array.from(selSetRef.current)),
          force,
        }),
      });
      if (!resp.ok) {
        const detail = String(
          ((await resp.json().catch(() => null)) as { detail?: string } | null)?.detail ?? "commit failed",
        );
        setPaintNeedsForce(detail.includes("force"));
        throw new Error(detail);
      }
      setPaintNeedsForce(false);
      setPaintNotice(
        `${paintOp === "suppress" ? "Suppressed" : "Pinned"} ${n.toLocaleString()} splats as “${label}”. ` +
          "Searches respect it now (the first query re-composes the scene, ~10-30s). " +
          "Colored queries added BEFORE this paint are stale — re-add them to refresh.",
      );
      setPaintLabel("");
      setPaintAliases("");
      clearSelection();
      void loadOverridesList();
    } catch (cause) {
      setPaintError(cause instanceof Error ? cause.message : "Commit failed.");
    } finally {
      setPaintBusy(false);
    }
  }

  async function removeOverride(oid: string) {
    try {
      const resp = await fetch(`/api/splat/jobs/${job.job_id}/langfield/overrides/${oid}`, {
        method: "DELETE",
        credentials: "same-origin",
      });
      if (!resp.ok) throw new Error(`${resp.status}`);
      setPaintNotice("Override reverted — the field is back to its unpainted state for that region.");
      void loadOverridesList();
    } catch (cause) {
      setPaintError(cause instanceof Error ? cause.message : "Delete failed.");
    }
  }

  useEffect(() => {
    if (job.langfield_available) void loadOverridesList();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.job_id]);

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
    if (mode === "tint") {
      // the ramp's source channel is BAKED as the first enabled query
      rebuildOverlay(next, mode, rampName);
      return;
    }
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
    setQueryError(null);
    rebuildOverlay(channelsRef.current, next, rampName);
  }

  function changeRamp(next: string) {
    setRampName(next);
    if (mode === "tint") rebuildOverlay(channelsRef.current, mode, next);
  }

  function changeTopPct(next: number) {
    setTopPct(next);
    topPctRef.current = next;
    const chans = channelsRef.current;
    for (let i = 0; i < chans.length; i += 1) {
      const cut = cutoffDynosRef.current[i];
      if (cut) cut.value = cutoffForTopPercent(chans[i].bytes, next);
    }
    meshRef.current?.updateVersion();
  }

  // ---- scale calibration --------------------------------------------------
  async function saveScale() {
    // meters_per_unit is ONE scalar shared by every dimension in the scene
    // (line ~1124: `len * metersPerUnit` on each label) — recalibrating
    // silently changes every previously-placed dimension's displayed
    // real-world length. Require a second click, same two-step confirm
    // idiom as scene delete (splat.tsx confirmDel), instead of overwriting
    // an existing calibration with no warning.
    if (metersPerUnit !== null && !recalibrateArmed) {
      setRecalibrateArmed(true);
      return;
    }
    setRecalibrateArmed(false);
    const dim = dimsRef.current.find((d) => d.id === calibDimId);
    if (!dim) {
      setScaleError("Pick a dimension and enter its real length.");
      return;
    }
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
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
      if (paintModeRef.current) {
        const hit = raycastAt(e.clientX, e.clientY);
        if (hit) strokeAtRef.current(hit);
        return;
      }
      if (!measureArmRef.current) return;
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
      if (measureArmRef.current || paintModeRef.current) return; // click-owners active
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
      cutoffDynosRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // No fallback to "the last dimension" here: calibDimId is only ever set by an
  // explicit click on a dimension (line ~1122) or cleared by deleteDim() when its
  // target is removed. Silently re-targeting calibration onto whatever happens
  // to be last in the list, with no prompt, let a user recalibrate against the
  // wrong reference without noticing.
  const calibDim = dims.find((d) => d.id === calibDimId) ?? null;

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
                    <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">top</span>
                    <input
                      type="range"
                      min={0.5}
                      max={25}
                      step={0.5}
                      value={topPct}
                      onChange={(e) => changeTopPct(Number(e.target.value))}
                      className="w-full"
                    />
                    <span className="w-10 shrink-0 text-right text-zinc-400">{topPct}%</span>
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

        {job.langfield_available && (
          <>
            <div className="h-px bg-white/10" />
            <SectionLabel>Paint the field</SectionLabel>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant={paintMode ? "primary" : "outline"}
                onClick={() => setPaintMode((v) => !v)}
                title="Brush splats with a sphere, then pin them to a label — corrects or extends the AI's understanding. Fully revertible."
              >
                {strokeBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Paintbrush className="h-3.5 w-3.5" />}{" "}
                {paintMode ? "Painting…" : "Paint"}
              </Button>
              {selCount > 0 && (
                <>
                  <Button type="button" size="sm" variant="outline" onClick={undoStroke} title="Undo last stroke">
                    <Undo2 className="h-3.5 w-3.5" />
                  </Button>
                  <Button type="button" size="sm" variant="outline" onClick={clearSelection} title="Clear selection">
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </>
              )}
            </div>
            {paintMode && (
              <>
                <div className="flex items-center gap-2">
                  <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">brush</span>
                  <input
                    type="range"
                    min={0.02}
                    max={0.4}
                    step={0.01}
                    value={brushRadius}
                    onChange={(e) => setBrushRadius(Number(e.target.value))}
                    className="w-full"
                  />
                  <span className="w-16 shrink-0 text-right text-zinc-400">
                    {metersPerUnit ? `${(brushRadius * metersPerUnit).toFixed(2)} m` : `${brushRadius.toFixed(2)} u`}
                  </span>
                </div>
                {channels.length > 0 && (
                  <label className="flex items-center gap-2 text-[10px] text-zinc-400">
                    <input type="checkbox" checked={limitToQuery} onChange={(e) => setLimitToQuery(e.target.checked)} />
                    clip strokes to the top {topPct}% of “{channels.find((c) => c.enabled)?.text ?? channels[0].text}”
                    (snaps sloppy strokes to the object)
                  </label>
                )}
                <p className="text-[10px] leading-snug text-zinc-500">
                  Click the scene to add sphere strokes ({selCount.toLocaleString()} splats selected, shown cyan).
                </p>
                {selCount > 0 && (
                  <div className="space-y-1.5">
                    <input
                      value={paintLabel}
                      onChange={(e) => setPaintLabel(e.target.value)}
                      placeholder='Label — anything, e.g. "dad’s corner"'
                      className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-600 focus:border-cyan-400/50 focus:outline-none"
                    />
                    <input
                      value={paintAliases}
                      onChange={(e) => setPaintAliases(e.target.value)}
                      placeholder="Aliases, comma-separated (optional)"
                      className="w-full rounded border border-white/10 bg-white/5 px-2 py-1 text-xs text-zinc-100 placeholder:text-zinc-600 focus:border-cyan-400/50 focus:outline-none"
                    />
                    {overrides.some((o) => o.label.trim().toLowerCase() === paintLabel.trim().toLowerCase()) && (
                      <p className="text-[10px] leading-snug text-amber-300/90">
                        A paint with this exact label already exists — both will answer to it. Delete the old one
                        below if this is a re-do.
                      </p>
                    )}
                    <div className="flex items-center gap-1.5">
                      <select
                        value={paintOp}
                        onChange={(e) => setPaintOp(e.target.value as typeof paintOp)}
                        className="rounded border border-white/10 bg-white/5 px-1.5 py-1 text-xs text-zinc-100 focus:outline-none"
                        title="Pin = trust your label fully · Boost = nudge/confirm the AI · Not this = push the label away"
                      >
                        <option value="assign">Pin label</option>
                        <option value="boost">Boost</option>
                        <option value="suppress">Not this</option>
                      </select>
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => void commitPaint(false)}
                        disabled={paintBusy || !paintLabel.trim()}
                      >
                        {paintBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : `Commit ${selCount.toLocaleString()}`}
                      </Button>
                      {paintNeedsForce && (
                        <Button type="button" size="sm" variant="outline" onClick={() => void commitPaint(true)}>
                          Force
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </>
            )}
            {paintError && <p className="text-[10px] leading-snug text-rose-300/90">{paintError}</p>}
            {paintNotice && <p className="text-[10px] leading-snug text-emerald-300/90">{paintNotice}</p>}
            {overrides.length > 0 && (
              <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wide text-zinc-500">Painted labels</p>
                {overrides.map((o) => (
                  <div key={o.id} className="flex items-center gap-2">
                    <span className="min-w-0 flex-1 truncate" title={[o.label, ...(o.aliases ?? [])].join(", ")}>
                      {o.op === "suppress" ? "🚫 " : "📌 "}
                      {o.label}
                    </span>
                    <span className="shrink-0 text-[10px] text-zinc-500">{o.count.toLocaleString()}</span>
                    <button
                      type="button"
                      onClick={() => void removeOverride(o.id)}
                      className="shrink-0 text-zinc-500 hover:text-rose-300"
                      title="Revert this paint (fully non-destructive)"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </>
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
                onClick={() => {
                  setCalibDimId(d.id);
                  setRecalibrateArmed(false);
                }}
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
              <Button
                type="button"
                size="sm"
                variant={recalibrateArmed ? "primary" : "outline"}
                onClick={() => void saveScale()}
                disabled={savingScale || !calibDim}
                title={recalibrateArmed ? "This replaces the scale every dimension's length is shown in — click again to confirm" : undefined}
              >
                {savingScale ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : recalibrateArmed ? "Sure? (replaces all)" : "Set scale"}
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
        <div className="absolute bottom-16 right-3 z-20 w-64 space-y-2 rounded-xl border border-white/10 bg-black/70 p-3 text-xs text-zinc-200 shadow backdrop-blur-md">
          <SectionLabel>Legend</SectionLabel>
          {mode === "tint" ? (
            (() => {
              const tc = channels.find((c) => c.enabled) ?? channels[0];
              return (
                <>
                  <div className="h-3 w-full rounded" style={{ background: rampCssGradient(rampName) }} />
                  <div className="flex justify-between text-[10px] text-zinc-400">
                    <span>{tc.relMin ? Number(tc.relMin).toFixed(2) : "low"}</span>
                    <span className="truncate px-2 text-zinc-300">“{tc.text}” relevancy</span>
                    <span>{tc.relMax ? Number(tc.relMax).toFixed(2) : "high"}</span>
                  </div>
                  {channels.length > 1 && (
                    <p className="text-[10px] text-zinc-500">Ramp shows the first ENABLED query.</p>
                  )}
                </>
              );
            })()
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
                {mode === "highlight" && `Colored = each query's top ${topPct}%; rest natural.`}
                {mode === "isolate" && `Only each query's top ${topPct}% visible, natural colors.`}
                {mode === "spotlight" && `Each query's top ${topPct}% colored; rest dimmed.`}
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
