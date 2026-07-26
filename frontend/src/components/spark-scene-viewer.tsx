import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { dyno, RgbaArray, SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import { apiRequest, applyEditOps, rebuildLangfield, revertEdit } from "@/lib/api";
import {
  buildOverlayModifier,
  buildClassColorModifier,
  buildTestRelevancy,
  cutoffForTopPercent,
  fetchRelevancy,
  packChannelsRgba,
  packClassColorsRgba,
  rampCssGradient,
  RAMPS,
  shouldUseTestRelevancy,
  type RelevancyResult,
  type OverlayMode,
} from "@/lib/spark-heatmap";
import type { SplatJob } from "@/lib/contracts";
import type {
  ViewerCameraNodeTarget,
  ViewerCameraOverlay,
  ViewerCameraPose,
  ViewerCameraViewTarget,
  ViewerHighlight,
  ViewerOverlay,
  ViewerPoint,
} from "@/components/viewer-types";
import { Button, Input, SectionLabel } from "@/components/ui";
import { Box, Download, Loader2, Paintbrush, Plus, Ruler, Scissors, Trash2, Undo2, X } from "lucide-react";

// SPARK BETA viewer for the /view page — the Wave-2 cutover surface.
// - Multi-query language overlay: up to 4 simultaneous text searches, one
//   color each, packed into a single RgbaArray (R/G/B/A channels) and
//   composited by a mode-baked dyno modifier (tint ramp / highlight on
//   natural / isolate / spotlight) with a live legend.
// - Survey dimensions: any number of two-point measurements, draggable
//   endpoints, floating labels, server-persisted per scene (dimensions.json,
//   sessionStorage as an offline fallback only), known-length scale
//   calibration stored on the scene (meters_per_unit), CSV export.
// - Crop to sphere: pick a center + radius, permanently discard every splat
//   outside it (POST /edit/apply -> splat-transform -S, already-proven
//   backend, this is pure UI wiring). Snapshot-versioned; undoable.

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
  source: "real" | "test";
  detail?: string;
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

export function SparkSceneViewer({
  job,
  safeMode = false,
  computeReason = "",
  onViewerError,
  focus = null,
  overlay = null,
  highlights = [],
  cameraOverlay = null,
  viewCamera = null,
  cameraNodeTarget = null,
  resetViewToken = 0,
  showShortcutLegend = false,
  panelSections = null,
  reloadToken = 0,
  onEditPendingChange,
  onLangfieldRebuilt,
  onPickMatch,
  onPickCamera,
}: {
  job: SplatJob;
  safeMode?: boolean;
  computeReason?: string;
  onViewerError?: (message: string) => void;
  // Classic-viewer parity contract (same names/types as SplatViewer) so the
  // host page can drive either viewer identically.
  // When set (from a language search hit), fly the camera to this 3D point.
  focus?: ViewerPoint | null;
  // When set, draw a highlight marker on each 3D match + a label on the active one.
  overlay?: ViewerOverlay;
  // Multiple colored object groups highlighted + labeled simultaneously (legend toggles).
  highlights?: ViewerHighlight[];
  // Capture camera positions/directions projected over the current viewer camera.
  cameraOverlay?: ViewerCameraOverlay;
  // One-shot request to view the scene from an original capture camera pose.
  viewCamera?: ViewerCameraViewTarget;
  // One-shot request to inspect a camera marker from just behind its original pose.
  cameraNodeTarget?: ViewerCameraNodeTarget;
  resetViewToken?: number;
  // Which tool-panel sections the host page's active tab wants: "measure" =
  // search/paint/dimensions, "edit" = crop-sphere + crop-box, null/undefined
  // = no panel at all. Panel chrome (splat count + fps) renders for both.
  // Overlays/viewer unaffected.
  panelSections?: "measure" | "edit" | null;
  showShortcutLegend?: boolean;
  // Bump after an EXTERNAL edit (the Edit-lane ops) to make the viewer reload
  // the splat file — routes into the same reloadNonce/url mechanism the crop
  // tools already use internally, NOT a second loader.
  reloadToken?: number;
  onPickMatch?: (i: number) => void;
  onPickCamera?: (camera: ViewerCameraPose) => void;
  // Fires when a viewer-initiated edit (crop apply/undo) starts or settles —
  // drives the host page's GLOBAL progress rail, which survives tab switches
  // and this component's own url-keyed teardown.
  onEditPendingChange?: (pending: boolean) => void;
  // Fires after a successful language-field rebuild so the host refreshes the
  // job payload (langfield_stale flips off on the next status fetch).
  onLangfieldRebuilt?: () => void;
}) {
  // Bumped after every crop apply/revert: the preview URL string doesn't
  // otherwise change even though the file on disk did, so the mesh-load
  // effect (keyed on `url` below) would never re-run and the viewer would
  // keep showing stale pre-edit geometry.
  const [reloadNonce, setReloadNonce] = useState(0);
  // Langfield scenes MUST load langweb: relevancy rows are exported-ply order
  // and langweb preserves it; web.ply is decimated + reordered.
  const url = `/api/splat/jobs/${job.job_id}/preview/file?fmt=${job.langfield_available ? "langweb" : "web"}&v=${reloadNonce}`;
  const dimsStorageKey = `splatlab.dims.${job.job_id}`;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const labelsRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const meshRef = useRef<SplatMesh | null>(null);
  // Default FOV to restore on reset (viewCamera can override cam.fov).
  const defaultFovRef = useRef<number | null>(null);

  // Classic-parity screen-space annotations (ported from splat-viewer.tsx):
  // screen positions of each 3D match, updated every frame from the camera.
  const [markers, setMarkers] = useState<{ x: number; y: number; front: boolean }[]>([]);
  // Screen positions of each highlight group's points (colored legend highlights).
  const [hlMarkers, setHlMarkers] = useState<{ label: string; color: string; pts: { x: number; y: number; front: boolean }[] }[]>([]);
  const [cameraMarkers, setCameraMarkers] = useState<
    {
      index: number;
      name: string;
      x: number;
      y: number;
      front: boolean;
      nose: { x: number; y: number; front: boolean };
      left: { x: number; y: number; front: boolean };
      right: { x: number; y: number; front: boolean };
      top: { x: number; y: number; front: boolean };
      camera: ViewerCameraPose;
    }[]
  >([]);

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

  // crop to sphere (destructive, snapshot-versioned server-side already)
  const cropModeRef = useRef(false);
  const cropGroupRef = useRef<THREE.Group | null>(null);
  const redrawCropRef = useRef<() => void>(() => {});
  const cropCenterRef = useRef<[number, number, number] | null>(null);
  const cropRadiusRef = useRef(0.5);
  const cropPreviewTimeoutRef = useRef<number | null>(null);
  const [cropMode, setCropMode] = useState(false);
  const [cropCenter, setCropCenter] = useState<[number, number, number] | null>(null);
  const [cropRadius, setCropRadius] = useState(0.5);
  const [cropRemovedCount, setCropRemovedCount] = useState<number | null>(null);
  const [cropConfirmArmed, setCropConfirmArmed] = useState(false);
  const [cropBusy, setCropBusy] = useState(false);
  const [cropError, setCropError] = useState<string | null>(null);
  const [cropLastVersion, setCropLastVersion] = useState<number | null>(null);
  const [cropUndoBusy, setCropUndoBusy] = useState(false);
  // Langfield staleness is read from `job.langfield_stale` (status poll —
  // STALE-marker truth, same source as the backend's 409 guard), never
  // tracked locally: it survives reloads and sees edits from other tools.

  // crop to box (sibling of crop-to-sphere; same destructive/versioned flow).
  // Backend semantics: crop_box → splat-transform -B, which KEEPS everything
  // inside [min,max] and removes everything OUTSIDE — so the red preview
  // tints the outside, exactly like the sphere tool tints outside its sphere.
  const boxModeRef = useRef(false);
  const boxGroupRef = useRef<THREE.Group | null>(null);
  const redrawBoxRef = useRef<() => void>(() => {});
  const boxCenterRef = useRef<[number, number, number] | null>(null);
  const boxExtentsRef = useRef<[number, number, number]>([0.5, 0.5, 0.5]);
  const boxPreviewTimeoutRef = useRef<number | null>(null);
  const [boxMode, setBoxMode] = useState(false);
  const [boxCenter, setBoxCenter] = useState<[number, number, number] | null>(null);
  // Half-extents per axis: the box spans center ± extent on x/y/z.
  const [boxExtents, setBoxExtents] = useState<[number, number, number]>([0.5, 0.5, 0.5]);
  const [boxRemovedCount, setBoxRemovedCount] = useState<number | null>(null);
  const [boxConfirmArmed, setBoxConfirmArmed] = useState(false);
  const [boxBusy, setBoxBusy] = useState(false);
  const [boxError, setBoxError] = useState<string | null>(null);
  const [boxLastVersion, setBoxLastVersion] = useState<number | null>(null);
  const [boxUndoBusy, setBoxUndoBusy] = useState(false);
  // ---- class painting (grass/dirt/... — the generative labels) -------------
  // Same brush, second target: "field" paints search embeddings (overrides),
  // "class" paints surface classes the generative stages consume.
  const [paintTarget, setPaintTarget] = useState<"field" | "class">("field");
  const [taxonomy, setTaxonomy] = useState<
    { id: string; display: string; color: string; category: string }[] | null
  >(null);
  const [selectedClassId, setSelectedClassId] = useState<string | null>(null);
  const [classRecords, setClassRecords] = useState<
    { id: string; class_id: string; count: number; invalid_reason?: string }[]
  >([]);
  const [classBusy, setClassBusy] = useState(false);
  const [classError, setClassError] = useState<string | null>(null);
  const [classNeedsForce, setClassNeedsForce] = useState(false);
  const [classNotice, setClassNotice] = useState<string | null>(null);
  const [showClassLayer, setShowClassLayer] = useState(false);

  async function loadClassTaxonomy() {
    try {
      const data = await apiRequest<{ classes: { id: string; display: string; color: string; category: string }[] }>(
        `/api/splat/jobs/${job.job_id}/class-taxonomy`,
      );
      setTaxonomy(data.classes);
      setSelectedClassId((prev) => prev ?? data.classes[0]?.id ?? null);
    } catch {
      setClassError("Could not load the class taxonomy.");
    }
  }

  async function loadClassRecords() {
    try {
      const data = await apiRequest<{ records: { id: string; class_id: string; count: number; invalid_reason?: string }[] }>(
        `/api/splat/jobs/${job.job_id}/class-labels`,
      );
      setClassRecords(data.records);
    } catch {
      /* list is advisory */
    }
  }

  async function refreshClassLayer() {
    const mesh = meshRef.current;
    if (!mesh || !taxonomy) return;
    const resp = await fetch(`/api/splat/jobs/${job.job_id}/class-labels/map`, {
      credentials: "same-origin",
    });
    if (!resp.ok) throw new Error(`class map HTTP ${resp.status}`);
    const order = JSON.parse(resp.headers.get("X-Class-Order") ?? "[]") as string[];
    const bytes = new Int16Array(await resp.arrayBuffer());
    const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
    if (bytes.length !== numSplats) {
      throw new Error(
        `class map rows (${bytes.length.toLocaleString()}) != loaded splats (${numSplats.toLocaleString()})`,
      );
    }
    const byId = new Map(taxonomy.map((c) => [c.id, c]));
    const palette = order.map((cid) => {
      const hex = byId.get(cid)?.color ?? "#ffffff";
      return [
        parseInt(hex.slice(1, 3), 16) / 255,
        parseInt(hex.slice(3, 5), 16) / 255,
        parseInt(hex.slice(5, 7), 16) / 255,
      ] as [number, number, number];
    });
    const colorArray = new RgbaArray({
      array: packClassColorsRgba(bytes, palette, numSplats),
      count: numSplats,
    });
    mesh.worldModifier = buildClassColorModifier({ colorArray });
    mesh.updateGenerator();
  }

  async function toggleClassLayer(on: boolean) {
    setShowClassLayer(on);
    setClassError(null);
    try {
      if (on) await refreshClassLayer();
      else rebuildOverlay(channelsRef.current, mode, rampName); // restore query overlay
    } catch (cause) {
      setShowClassLayer(false);
      setClassError(cause instanceof Error ? cause.message : "Class layer failed.");
    }
  }

  async function commitClassPaint(force = false) {
    const n = selSetRef.current.size;
    if (!selectedClassId || n === 0) return;
    setClassBusy(true);
    setClassError(null);
    setClassNotice(null);
    try {
      const resp = await fetch(`/api/splat/jobs/${job.job_id}/class-labels`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          class_id: selectedClassId,
          indices_b64: b64FromUint32(Uint32Array.from(selSetRef.current)),
          force,
        }),
      });
      if (!resp.ok) {
        const detail = String(
          ((await resp.json().catch(() => null)) as { detail?: string } | null)?.detail ?? "commit failed",
        );
        setClassNeedsForce(detail.includes("force"));
        throw new Error(detail);
      }
      setClassNeedsForce(false);
      const display = taxonomy?.find((c) => c.id === selectedClassId)?.display ?? selectedClassId;
      setClassNotice(`Painted ${n.toLocaleString()} splats as ${display}. Survives crops (positions are snapshotted).`);
      clearSelection();
      void loadClassRecords();
      if (showClassLayer) void refreshClassLayer().catch(() => undefined);
    } catch (cause) {
      setClassError(cause instanceof Error ? cause.message : "Commit failed.");
    } finally {
      setClassBusy(false);
    }
  }

  async function removeClassRecord(rid: string) {
    try {
      await fetch(`/api/splat/jobs/${job.job_id}/class-labels/${rid}`, {
        method: "DELETE",
        credentials: "same-origin",
      });
      void loadClassRecords();
      if (showClassLayer) void refreshClassLayer().catch(() => undefined);
    } catch {
      /* record list refresh will show the truth */
    }
  }

  // Post-edit language-field rebuild: the one-click cure for a STALE field.
  // Success is confirmed by the status poll flipping langfield_stale off; the
  // notice shows the receipt's carried-label counts until then.
  const [rebuildBusy, setRebuildBusy] = useState(false);
  const [rebuildNotice, setRebuildNotice] = useState<string | null>(null);
  const [rebuildError, setRebuildError] = useState<string | null>(null);
  async function runLangfieldRebuild() {
    setRebuildBusy(true);
    setRebuildError(null);
    setRebuildNotice(null);
    try {
      const resp = await rebuildLangfield(job.job_id);
      const records = resp.receipt.records ?? [];
      const kept = records.reduce((a, r) => a + (r.kept || 0), 0);
      const dropped = records.reduce((a, r) => a + (r.dropped || 0), 0);
      setRebuildNotice(
        records.length
          ? `Language field rebuilt — ${kept.toLocaleString()} painted gaussians carried across${dropped ? `, ${dropped.toLocaleString()} were in removed geometry` : ""}.`
          : "Language field rebuilt.",
      );
      void loadOverridesList();
      onLangfieldRebuilt?.();
    } catch (cause) {
      setRebuildError(cause instanceof Error ? cause.message : "Rebuild failed.");
    } finally {
      setRebuildBusy(false);
    }
  }

  // Crop-slider ranges derived from the loaded scene's bbox (set once per
  // load in mesh.initialized); null = bounds not measured yet → legacy range.
  const [sliderScale, setSliderScale] = useState<{
    sphereMax: number;
    boxMax: [number, number, number];
  } | null>(null);
  // Non-fatal regen failures from the last crop/undo (backend `warnings[]`) —
  // previously discarded, so a scene silently degrading to raw-.ply serving
  // looked like success. Cleared by the next successful apply/undo.
  const [editWarnings, setEditWarnings] = useState<string[]>([]);
  // Armed-tool click that hit nothing (raycast miss on faint splats) —
  // previously a silent no-op that read as "the tool is broken".
  const [placementMiss, setPlacementMiss] = useState(false);
  const placementMissTimer = useRef<number | null>(null);
  function flagPlacementMiss() {
    setPlacementMiss(true);
    if (placementMissTimer.current) window.clearTimeout(placementMissTimer.current);
    placementMissTimer.current = window.setTimeout(() => setPlacementMiss(false), 2500);
  }
  // The click handler lives inside the url-keyed three.js effect; a ref keeps
  // it pointed at the current closure without re-running that effect.
  const flagPlacementMissRef = useRef(flagPlacementMiss);
  flagPlacementMissRef.current = flagPlacementMiss;

  // Host page's global progress rail rides these — it survives tab switches
  // and this component's own reload teardown.
  const editPending = cropBusy || boxBusy || cropUndoBusy || boxUndoBusy;
  useEffect(() => {
    onEditPendingChange?.(editPending);
  }, [editPending, onEditPendingChange]);

  // External-edit reload (Edit lane): reuse the crop tools' exact reload path
  // — bump reloadNonce so `url` changes and the mesh-load effect re-runs.
  // Token 0 is the initial mount, never a reload.
  useEffect(() => {
    if (reloadToken) setReloadNonce((n) => n + 1);
  }, [reloadToken]);

  useEffect(() => {
    if (!recalibrateArmed) return;
    const t = window.setTimeout(() => setRecalibrateArmed(false), 3000);
    return () => window.clearTimeout(t);
  }, [recalibrateArmed]);

  useEffect(() => {
    if (!cropConfirmArmed) return;
    const t = window.setTimeout(() => setCropConfirmArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [cropConfirmArmed]);

  useEffect(() => {
    if (!boxConfirmArmed) return;
    const t = window.setTimeout(() => setBoxConfirmArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [boxConfirmArmed]);

  useEffect(() => {
    measureArmRef.current = measureArm;
    if (!measureArm) {
      pendingPointRef.current = null;
      setHasPending(false);
    }
    if (measureArm) {
      setCropMode(false); // one click-owner at a time
      setBoxMode(false);
    }
  }, [measureArm]);

  useEffect(() => {
    paintModeRef.current = paintMode;
    if (paintMode) {
      setMeasureArm(false); // one click-owner at a time
      setCropMode(false);
      setBoxMode(false);
    }
    setPaintError(null); // never carry a stale error across a mode flip
    setPaintNeedsForce(false);
    // entering/leaving paint swaps the modifier (selection preview <-> queries)
    refreshModifierRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paintMode]);

  useEffect(() => {
    cropModeRef.current = cropMode;
    if (cropMode) {
      setMeasureArm(false); // one click-owner at a time
      setPaintMode(false);
      setBoxMode(false);
    } else {
      cropCenterRef.current = null;
      setCropCenter(null);
      setCropRemovedCount(null);
      setCropConfirmArmed(false);
    }
    setCropError(null); // never carry a stale error across a mode flip
    redrawCropRef.current();
    refreshModifierRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cropMode]);

  useEffect(() => {
    boxModeRef.current = boxMode;
    if (boxMode) {
      setMeasureArm(false); // one click-owner at a time
      setPaintMode(false);
      setCropMode(false);
    } else {
      boxCenterRef.current = null;
      setBoxCenter(null);
      setBoxRemovedCount(null);
      setBoxConfirmArmed(false);
    }
    setBoxError(null); // never carry a stale error across a mode flip
    redrawBoxRef.current();
    refreshModifierRef.current();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boxMode]);

  // Leaving the Edit tab hides the crop sections — ALSO disarm both crop
  // click-owners, or an invisible armed tool would keep eating scene clicks
  // (same reset discipline as the mode-flip effects above; their off-branches
  // do the center/preview cleanup).
  useEffect(() => {
    if (panelSections !== "edit") {
      setCropMode(false);
      setBoxMode(false);
    }
  }, [panelSections]);

  useEffect(() => {
    cropRadiusRef.current = cropRadius;
    redrawCropRef.current();
    scheduleCropPreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cropRadius]);

  useEffect(() => {
    boxExtentsRef.current = boxExtents;
    redrawBoxRef.current();
    scheduleBoxPreview();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boxExtents]);

  useEffect(() => {
    brushRadiusRef.current = brushRadius;
  }, [brushRadius]);

  // ---- dims persistence ---------------------------------------------------
  // Server is authoritative (dimensions.json on the job); sessionStorage is
  // kept only as a best-effort offline cache/fallback, never the source of
  // truth — costs nothing extra since the write calls already existed here.
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

  async function postDimension(d: Dim) {
    try {
      await fetch(`/api/splat/jobs/${job.job_id}/dimensions`, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: String(d.id), a: d.a, b: d.b }),
      });
    } catch {
      /* best-effort — sessionStorage already has this dim locally */
    }
  }

  async function deleteDimensionServer(id: number) {
    try {
      await fetch(`/api/splat/jobs/${job.job_id}/dimensions/${id}`, { method: "DELETE", credentials: "same-origin" });
    } catch {
      /* best-effort */
    }
  }

  function deleteDim(id: number) {
    syncDims(dimsRef.current.filter((d) => d.id !== id));
    void deleteDimensionServer(id);
    if (calibDimId === id) {
      setCalibDimId(null);
      setRecalibrateArmed(false);
    }
  }

  function deleteAllDims() {
    const prev = dimsRef.current;
    syncDims([]);
    setCalibDimId(null);
    setRecalibrateArmed(false);
    for (const d of prev) void deleteDimensionServer(d.id);
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
    if (cropModeRef.current && cropCenterRef.current) previewCropRemoval();
    else if (boxModeRef.current && boxCenterRef.current) previewBoxRemoval();
    else if (paintModeRef.current && selSetRef.current.size > 0) previewSelection();
    else rebuildOverlay(channelsRef.current, mode, rampName);
  }
  refreshModifierRef.current = refreshModifier;

  // ---- crop preview (shared by sphere + box) ------------------------------
  // One pass over live geometry tinting every splat `shouldRemove` says goes
  // red — exactly what "Apply crop" is about to permanently delete — and
  // returning the removed count (null if the mesh isn't ready). Reuses the
  // same overlay machinery as previewSelection(), just computed from a
  // predicate instead of a stored Set.
  function previewRemoval(shouldRemove: (x: number, y: number, z: number) => boolean): number | null {
    const mesh = meshRef.current;
    const packed = mesh?.packedSplats;
    if (!mesh || !packed) return null;
    const numSplats = packed.numSplats ?? 0;
    if (numSplats === 0) return null;
    const bytes = new Uint8Array(numSplats);
    let removed = 0;
    packed.forEachSplat((i, c) => {
      if (shouldRemove(c.x, c.y, c.z)) {
        bytes[i] = 255;
        removed += 1;
      }
    });
    const scalarArray = new RgbaArray({ array: packChannelsRgba([bytes], numSplats), count: numSplats });
    mesh.worldModifier = buildOverlayModifier({
      scalarArray,
      channelCount: 1,
      channelColors: [[0.97, 0.44, 0.44]], // to-be-removed red
      channelEnabled: [dyno.dynoBool(true)],
      mode: "highlight",
      ramp: rampName,
      channelCutoffs: [dyno.dynoFloat(0.5)],
    });
    mesh.updateGenerator();
    return removed;
  }

  // Sphere: crop_sphere → splat-transform -S KEEPS the inside ("remove
  // outside"), so red = outside the sphere. Same predicate math as the
  // original inline loop (squared distance vs r²).
  function previewCropRemoval() {
    const center = cropCenterRef.current;
    if (!center) return;
    const r2 = cropRadiusRef.current * cropRadiusRef.current;
    const [cx, cy, cz] = center;
    const removed = previewRemoval((x, y, z) => {
      const dx = x - cx;
      const dy = y - cy;
      const dz = z - cz;
      return dx * dx + dy * dy + dz * dz > r2;
    });
    if (removed !== null) setCropRemovedCount(removed);
  }

  // Box: crop_box → splat-transform -B KEEPS everything inside [min,max]
  // ("Remove Gaussians outside box (min, max corners)" — CLI README), so the
  // red to-be-removed tint covers everything OUTSIDE the box.
  function previewBoxRemoval() {
    const center = boxCenterRef.current;
    if (!center) return;
    const [cx, cy, cz] = center;
    const [ex, ey, ez] = boxExtentsRef.current;
    const removed = previewRemoval(
      (x, y, z) => Math.abs(x - cx) > ex || Math.abs(y - cy) > ey || Math.abs(z - cz) > ez,
    );
    if (removed !== null) setBoxRemovedCount(removed);
  }

  // Debounced: recomputing over every splat on each slider tick/drag frame
  // would be wasteful — settle briefly before the (still cheap, one-pass)
  // count + overlay rebuild runs.
  function scheduleCropPreview() {
    if (cropPreviewTimeoutRef.current !== null) window.clearTimeout(cropPreviewTimeoutRef.current);
    cropPreviewTimeoutRef.current = window.setTimeout(() => {
      cropPreviewTimeoutRef.current = null;
      if (cropModeRef.current && cropCenterRef.current) previewCropRemoval();
    }, 120);
  }

  function scheduleBoxPreview() {
    if (boxPreviewTimeoutRef.current !== null) window.clearTimeout(boxPreviewTimeoutRef.current);
    boxPreviewTimeoutRef.current = window.setTimeout(() => {
      boxPreviewTimeoutRef.current = null;
      if (boxModeRef.current && boxCenterRef.current) previewBoxRemoval();
    }, 120);
  }

  function setCropCenterFromHit(p: [number, number, number]) {
    cropCenterRef.current = p;
    setCropCenter(p);
    redrawCropRef.current();
    scheduleCropPreview();
  }

  function setBoxCenterFromHit(p: [number, number, number]) {
    boxCenterRef.current = p;
    setBoxCenter(p);
    redrawBoxRef.current();
    scheduleBoxPreview();
  }

  // ---- LOCAL brush selection --------------------------------------------
  // Strokes never touch the server: a one-time spatial grid over the loaded
  // splat positions answers sphere queries in-browser, so sweeps are instant
  // and NOTHING is sent until the commit button ("save"). The commit already
  // posts final indices — the per-stroke /select/sphere round-trip (the
  // choppiness source) was never actually needed.
  const paintIndexRef = useRef<{
    positions: Float32Array;
    grid: Map<string, number[]>;
    cellSize: number;
    count: number;
  } | null>(null);

  function ensurePaintIndex(): boolean {
    if (paintIndexRef.current) return true;
    const mesh = meshRef.current;
    const packed = mesh?.packedSplats;
    const count = packed?.numSplats ?? 0;
    if (!packed || !count) return false;
    const positions = new Float32Array(count * 3);
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    packed.forEachSplat((i: number, c: THREE.Vector3) => {
      positions[i * 3] = c.x;
      positions[i * 3 + 1] = c.y;
      positions[i * 3 + 2] = c.z;
      if (c.x < lo[0]) lo[0] = c.x;
      if (c.y < lo[1]) lo[1] = c.y;
      if (c.z < lo[2]) lo[2] = c.z;
      if (c.x > hi[0]) hi[0] = c.x;
      if (c.y > hi[1]) hi[1] = c.y;
      if (c.z > hi[2]) hi[2] = c.z;
    });
    const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const cellSize = Math.max(diag / 128, 1e-3);
    const grid = new Map<string, number[]>();
    for (let i = 0; i < count; i += 1) {
      const key = `${Math.floor(positions[i * 3] / cellSize)},${Math.floor(positions[i * 3 + 1] / cellSize)},${Math.floor(positions[i * 3 + 2] / cellSize)}`;
      const bucket = grid.get(key);
      if (bucket) bucket.push(i);
      else grid.set(key, [i]);
    }
    paintIndexRef.current = { positions, grid, cellSize, count };
    return true;
  }

  function localSelectSphere(p: THREE.Vector3, radius: number): number[] {
    const index = paintIndexRef.current;
    if (!index) return [];
    const { positions, grid, cellSize } = index;
    const r2 = radius * radius;
    const out: number[] = [];
    const x0 = Math.floor((p.x - radius) / cellSize);
    const x1 = Math.floor((p.x + radius) / cellSize);
    const y0 = Math.floor((p.y - radius) / cellSize);
    const y1 = Math.floor((p.y + radius) / cellSize);
    const z0 = Math.floor((p.z - radius) / cellSize);
    const z1 = Math.floor((p.z + radius) / cellSize);
    for (let ix = x0; ix <= x1; ix += 1) {
      for (let iy = y0; iy <= y1; iy += 1) {
        for (let iz = z0; iz <= z1; iz += 1) {
          const bucket = grid.get(`${ix},${iy},${iz}`);
          if (!bucket) continue;
          for (const i of bucket) {
            const dx = positions[i * 3] - p.x;
            const dy = positions[i * 3 + 1] - p.y;
            const dz = positions[i * 3 + 2] - p.z;
            if (dx * dx + dy * dy + dz * dz <= r2) out.push(i);
          }
        }
      }
    }
    return out;
  }

  // Selection-tint refresh is the expensive part of a stroke (full-scene
  // modifier rebuild) — coalesce to a 120ms trailing update during sweeps.
  const selPreviewTimer = useRef<number | null>(null);
  function scheduleSelectionPreview() {
    if (selPreviewTimer.current !== null) return;
    selPreviewTimer.current = window.setTimeout(() => {
      selPreviewTimer.current = null;
      refreshModifierRef.current();
    }, 120);
  }

  function strokeAt(p: THREE.Vector3) {
    if (!ensurePaintIndex()) {
      setPaintError("Scene not fully loaded yet — try the stroke again in a second.");
      return;
    }
    setPaintError(null);
    const idx = localSelectSphere(p, brushRadiusRef.current);
    // optional hygiene: clip the sloppy sphere to the current query's matches
    let clipped: number[] = idx;
    if (limitToQuery) {
      const ch = channelsRef.current.find((c) => c.enabled);
      if (ch) {
        const cut = Math.round(cutoffForTopPercent(ch.bytes, topPctRef.current) * 255);
        clipped = idx.filter((i) => ch.bytes[i] >= cut);
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
    scheduleSelectionPreview();
  }
  strokeAtRef.current = (p) => strokeAt(p);

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
      const numSplats = mesh.packedSplats?.numSplats ?? mesh.numSplats ?? 0;
      let result: RelevancyResult;
      try {
        result = job.langfield_available && !safeMode
          ? await fetchRelevancy(job.job_id, text)
          : buildTestRelevancy(
              numSplats,
              text,
              job.langfield_available ? "hardware gate active; using safe test search" : "no language field on this scene",
            );
      } catch (cause) {
        if (!shouldUseTestRelevancy(cause)) throw cause;
        result = buildTestRelevancy(numSplats, text, "language worker unavailable in safe browse mode");
      }
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
          source: result.source,
          detail: result.detail,
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

  // ---- crop to sphere: apply / undo ---------------------------------------
  async function applyCrop() {
    const center = cropCenterRef.current;
    if (!center) return;
    setCropConfirmArmed(false);
    setCropBusy(true);
    setCropError(null);
    try {
      const resp = await applyEditOps(job.job_id, [
        { type: "crop_sphere", center, radius: cropRadiusRef.current },
      ]);
      setCropLastVersion(resp.version_before);
      setEditWarnings(resp.warnings ?? []);
      setCropMode(false);
      setReloadNonce((n) => n + 1);
    } catch (cause) {
      setCropError(cause instanceof Error ? cause.message : "Crop failed.");
    } finally {
      setCropBusy(false);
    }
  }

  async function undoCrop() {
    if (cropLastVersion === null) return;
    setCropUndoBusy(true);
    setCropError(null);
    try {
      await revertEdit(job.job_id, cropLastVersion);
      setCropLastVersion(null);
      setEditWarnings([]);
      setReloadNonce((n) => n + 1);
    } catch (cause) {
      setCropError(cause instanceof Error ? cause.message : "Undo failed.");
    } finally {
      setCropUndoBusy(false);
    }
  }

  // ---- crop to box: apply / undo -------------------------------------------
  // Same versioned flow as the sphere, via the typed lib helpers. crop_box
  // KEEPS what's inside min/max (verified: splat-transform -B "Remove
  // Gaussians outside box"), so min/max = center ∓/± half-extents.
  async function applyBoxCrop() {
    const center = boxCenterRef.current;
    if (!center) return;
    setBoxConfirmArmed(false);
    setBoxBusy(true);
    setBoxError(null);
    try {
      const [cx, cy, cz] = center;
      const [ex, ey, ez] = boxExtentsRef.current;
      const resp = await applyEditOps(job.job_id, [
        { type: "crop_box", min: [cx - ex, cy - ey, cz - ez], max: [cx + ex, cy + ey, cz + ez] },
      ]);
      setBoxLastVersion(resp.version_before);
      setEditWarnings(resp.warnings ?? []);
      setBoxMode(false);
      setReloadNonce((n) => n + 1);
    } catch (cause) {
      setBoxError(cause instanceof Error ? cause.message : "Crop failed.");
    } finally {
      setBoxBusy(false);
    }
  }

  async function undoBoxCrop() {
    if (boxLastVersion === null) return;
    setBoxUndoBusy(true);
    setBoxError(null);
    try {
      await revertEdit(job.job_id, boxLastVersion);
      setBoxLastVersion(null);
      setReloadNonce((n) => n + 1);
    } catch (cause) {
      setBoxError(cause instanceof Error ? cause.message : "Undo failed.");
    } finally {
      setBoxUndoBusy(false);
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

    function reportViewerError(cause: unknown, fallback: string) {
      const message = cause instanceof Error ? cause.message : fallback;
      setError(message);
      onViewerError?.(message);
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1000);
    camera.up.copy(INITIAL_CAMERA_UP);
    camera.position.copy(INITIAL_CAMERA_POSITION);
    camera.lookAt(INITIAL_CAMERA_LOOK_AT);
    cameraRef.current = camera;
    defaultFovRef.current = camera.fov;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      container.appendChild(renderer.domElement);
    } catch (cause) {
      reportViewerError(cause, "WebGL could not start the Spark viewer.");
      return;
    }

    let spark: SparkRenderer;
    try {
      spark = new SparkRenderer({ renderer });
      scene.add(spark);
    } catch (cause) {
      renderer.dispose();
      if (renderer.domElement.parentElement === container) container.removeChild(renderer.domElement);
      reportViewerError(cause, "Spark could not initialize this scene.");
      return;
    }

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(INITIAL_CAMERA_LOOK_AT);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.update();
    controlsRef.current = controls;

    const dimGroup = new THREE.Group();
    scene.add(dimGroup);
    dimGroupRef.current = dimGroup;

    // Restore this scene's dimensions: sessionStorage first (instant, no
    // network wait), then the server (authoritative) once it responds —
    // avoids an empty-then-populated flash on a fast reload while still
    // ending up correct if the two ever disagree (e.g. a different device).
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
    fetch(`/api/splat/jobs/${job.job_id}/dimensions`, { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : null))
      .then((data: { dimensions?: { id: string; a: number[]; b: number[] }[] } | null) => {
        if (disposed || !data?.dimensions) return;
        const restored: Dim[] = data.dimensions
          .filter((d) => Array.isArray(d.a) && Array.isArray(d.b))
          .map((d) => ({
            id: Number(d.id) || Date.now(),
            a: d.a as [number, number, number],
            b: d.b as [number, number, number],
          }));
        dimsRef.current = restored;
        setDims([...restored]);
        try {
          sessionStorage.setItem(dimsStorageKey, JSON.stringify(restored));
        } catch {
          /* ignore */
        }
        redrawDimsRef.current();
      })
      .catch(() => {
        /* server unreachable — keep whatever the sessionStorage restore gave us */
      });

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

    const cropGroup = new THREE.Group();
    scene.add(cropGroup);
    cropGroupRef.current = cropGroup;

    // Brush cursor: a unit wireframe sphere scaled to the live brush radius,
    // following the raycast hit while a paint mode is armed. Without it the
    // operator aims blind (real complaint, 2026-07-26). Unit geometry +
    // scale means slider changes never rebuild geometry.
    const brushCursor = new THREE.Mesh(
      new THREE.SphereGeometry(1, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x22d3ee, wireframe: true, transparent: true, opacity: 0.5 }),
    );
    brushCursor.visible = false;
    scene.add(brushCursor);
    let brushRayAt = 0;
    let paintDrag = false;
    let lastStroke: THREE.Vector3 | null = null;
    function updateToolCursor(clientX: number, clientY: number): THREE.Vector3 | null {
      // One cursor, three owners: cyan brush while painting; a red preview
      // sphere while a crop tool is armed but UNPLACED (the "why is nothing
      // showing" moment — the crop sphere only used to appear after the first
      // click). ~14 Hz raycast throttle so a 2M-splat scene doesn't hitch.
      const paintArmed = paintModeRef.current;
      const cropPlacing = cropModeRef.current && !cropCenterRef.current;
      const boxPlacing = boxModeRef.current && !boxCenterRef.current;
      if (!paintArmed && !cropPlacing && !boxPlacing) {
        brushCursor.visible = false;
        return null;
      }
      const now = performance.now();
      if (now - brushRayAt < 70) return null;
      brushRayAt = now;
      const hit = raycastAt(clientX, clientY);
      if (hit) {
        brushCursor.position.copy(hit);
        const r = paintArmed
          ? brushRadiusRef.current
          : cropPlacing
            ? cropRadiusRef.current
            : Math.min(...boxExtentsRef.current);
        brushCursor.scale.setScalar(Math.max(r, 1e-4));
        (brushCursor.material as THREE.MeshBasicMaterial).color.setHex(
          paintArmed ? 0x22d3ee : 0xf87171,
        );
        brushCursor.visible = true;
      } else {
        brushCursor.visible = false;
      }
      return hit;
    }
    function enqueueStroke(hit: THREE.Vector3) {
      // Strokes are LOCAL (spatial grid) — instant, so a sweep just needs
      // spacing + path interpolation for full coverage.
      const spacing = brushRadiusRef.current * 0.6;
      if (lastStroke && lastStroke.distanceTo(hit) < spacing) return;
      if (lastStroke) {
        const steps = Math.min(12, Math.floor(lastStroke.distanceTo(hit) / spacing));
        for (let s = 1; s < steps; s += 1) {
          strokeAtRef.current(lastStroke.clone().lerp(hit, s / steps));
        }
      }
      strokeAtRef.current(hit.clone());
      lastStroke = hit.clone();
    }

    function redrawCrop() {
      const group = cropGroupRef.current;
      if (!group) return;
      group.clear();
      const center = cropCenterRef.current;
      if (!center) return;
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(cropRadiusRef.current, 32, 24),
        new THREE.MeshBasicMaterial({ color: 0xf87171, wireframe: true, transparent: true, opacity: 0.55 }),
      );
      sphere.position.set(...center);
      group.add(sphere);
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xf87171 }),
      );
      marker.position.set(...center);
      group.add(marker);
    }
    redrawCropRef.current = redrawCrop;
    redrawCrop();

    const boxGroup = new THREE.Group();
    scene.add(boxGroup);
    boxGroupRef.current = boxGroup;

    function redrawBox() {
      const group = boxGroupRef.current;
      if (!group) return;
      group.clear();
      const center = boxCenterRef.current;
      if (!center) return;
      const [ex, ey, ez] = boxExtentsRef.current;
      const box = new THREE.Mesh(
        new THREE.BoxGeometry(ex * 2, ey * 2, ez * 2),
        new THREE.MeshBasicMaterial({ color: 0xf87171, wireframe: true, transparent: true, opacity: 0.55 }),
      );
      box.position.set(...center);
      group.add(box);
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(0.018, 16, 12),
        new THREE.MeshBasicMaterial({ color: 0xf87171 }),
      );
      marker.position.set(...center);
      group.add(marker);
    }
    redrawBoxRef.current = redrawBox;
    redrawBox();

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
      // Chrome/Firefox already throttle a backgrounded tab's RAF to ~1Hz, but
      // that's still real render work every second, forever, for a scene
      // nobody's looking at. Skip it outright; resumes instantly on refocus
      // since the RAF loop itself keeps ticking.
      if (document.hidden) return;
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
        paintIndexRef.current = null; // rows/positions changed — rebuild lazily
        // One bbox pass so crop sliders scale to THIS scene: the old fixed
        // 0.05–5 range could never enclose a large capture and dwarfed tiny
        // ones. Floater-inflated bounds only lengthen the slider — harmless.
        const packed = mesh.packedSplats;
        if (packed?.numSplats) {
          const lo = [Infinity, Infinity, Infinity];
          const hi = [-Infinity, -Infinity, -Infinity];
          packed.forEachSplat((_i: number, c: THREE.Vector3) => {
            if (c.x < lo[0]) lo[0] = c.x;
            if (c.y < lo[1]) lo[1] = c.y;
            if (c.z < lo[2]) lo[2] = c.z;
            if (c.x > hi[0]) hi[0] = c.x;
            if (c.y > hi[1]) hi[1] = c.y;
            if (c.z > hi[2]) hi[2] = c.z;
          });
          const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
          if (ext.every((v) => Number.isFinite(v) && v > 0)) {
            const diagonal = Math.hypot(ext[0], ext[1], ext[2]);
            setSliderScale({
              sphereMax: Math.max(0.5, 0.75 * diagonal),
              boxMax: [
                Math.max(0.5, 1.5 * (ext[0] / 2)),
                Math.max(0.5, 1.5 * (ext[1] / 2)),
                Math.max(0.5, 1.5 * (ext[2] / 2)),
              ] as [number, number, number],
            });
          }
        }
        setReady(true);
        applyOverlayRef.current(); // re-apply overlay after a reload
      })
      .catch((cause: unknown) => {
        if (disposed) return;
        reportViewerError(cause, "Could not load Spark preview.");
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
      // Armed brush owns left-drag: hold and sweep to paint (disarm Painting
      // to orbit again). Single clicks still stroke via onClick.
      if (paintModeRef.current) {
        paintDrag = true;
        lastStroke = null;
        controls.enabled = false;
        renderer.domElement.setPointerCapture(e.pointerId);
        return;
      }
      const hit = endpointAt(e.clientX, e.clientY);
      if (hit) {
        drag = hit;
        controls.enabled = false;
        renderer.domElement.setPointerCapture(e.pointerId);
      }
    }
    function onPointerMove(e: PointerEvent) {
      if (!drag) {
        const hit = updateToolCursor(e.clientX, e.clientY);
        // Drag-to-paint: strokes enqueue (with path interpolation) while the
        // primary button is held — see enqueueStroke/pumpStrokes.
        if (paintModeRef.current && paintDrag && hit) enqueueStroke(hit);
        if (paintModeRef.current || cropModeRef.current || boxModeRef.current) return;
      }
      if (!drag) return;
      const p = raycastAt(e.clientX, e.clientY);
      if (!p) return;
      const d = dimsRef.current.find((x) => x.id === drag!.dimId);
      if (!d) return;
      d[drag.end] = [p.x, p.y, p.z];
      redrawDims();
    }
    function onPointerUp(e: PointerEvent) {
      if (paintDrag) {
        paintDrag = false;
        lastStroke = null;
        controls.enabled = true;
        try {
          renderer.domElement.releasePointerCapture(e.pointerId);
        } catch {
          /* already released */
        }
        return;
      }
      if (!drag) return;
      const draggedId = drag.dimId;
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
      const moved = dimsRef.current.find((d) => d.id === draggedId);
      if (moved) void postDimension(moved);
    }
    function onClick(e: MouseEvent) {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
      if (cropModeRef.current) {
        const hit = raycastAt(e.clientX, e.clientY);
        if (hit) setCropCenterFromHit([hit.x, hit.y, hit.z]);
        else flagPlacementMissRef.current();
        return;
      }
      if (boxModeRef.current) {
        const hit = raycastAt(e.clientX, e.clientY);
        if (hit) setBoxCenterFromHit([hit.x, hit.y, hit.z]);
        else flagPlacementMissRef.current();
        return;
      }
      if (paintModeRef.current) {
        const hit = raycastAt(e.clientX, e.clientY);
        if (hit) strokeAtRef.current(hit);
        else flagPlacementMissRef.current();
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
        const newDim: Dim = { id, a: pending, b: [hit.x, hit.y, hit.z] };
        dimsRef.current = [...dimsRef.current, newDim];
        setDims([...dimsRef.current]);
        try {
          sessionStorage.setItem(dimsStorageKey, JSON.stringify(dimsRef.current));
        } catch {
          /* ignore */
        }
        redrawDims();
        void postDimension(newDim);
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

    // Classic-viewer keyboard parity. The classic component's shortcuts came
    // from the mkkellogg viewer it wrapped: WASD = OrbitControls key-pan
    // (keys {LEFT:KeyA, UP:KeyW, RIGHT:KeyD, BOTTOM:KeyS}, keyPanSpeed 7px,
    // screen-space pan) and ArrowLeft/Right = roll camera.up around the view
    // axis by ±PI/128. Both reimplemented here with the same math. Guarded on
    // the event target so typing in the control panel's inputs never moves
    // the camera (classic relied on the lone search input stopping
    // propagation; Spark's panel has many inputs).
    const KEY_PAN_SPEED = 7;
    const panVecA = new THREE.Vector3();
    const panVecB = new THREE.Vector3();
    function panCamera(deltaX: number, deltaY: number) {
      // OrbitControls._pan for a perspective camera with screenSpacePanning:
      // half the visible height at the target distance, scaled by px/clientHeight.
      const targetDistance =
        camera.position.distanceTo(controls.target) * Math.tan(((camera.fov / 2) * Math.PI) / 180.0);
      const el = renderer.domElement;
      panVecA
        .setFromMatrixColumn(camera.matrix, 0) // panLeft: X column
        .multiplyScalar((-2 * deltaX * targetDistance) / el.clientHeight);
      panVecB
        .setFromMatrixColumn(camera.matrix, 1) // panUp (screen space): Y column
        .multiplyScalar((2 * deltaY * targetDistance) / el.clientHeight);
      panVecA.add(panVecB);
      camera.position.add(panVecA);
      controls.target.add(panVecA);
      controls.update();
    }
    const rollForward = new THREE.Vector3();
    const rollMatrix = new THREE.Matrix4();
    function onKeyDown(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      switch (e.code) {
        case "KeyW":
          panCamera(0, KEY_PAN_SPEED);
          e.preventDefault();
          break;
        case "KeyS":
          panCamera(0, -KEY_PAN_SPEED);
          e.preventDefault();
          break;
        case "KeyA":
          panCamera(KEY_PAN_SPEED, 0);
          e.preventDefault();
          break;
        case "KeyD":
          panCamera(-KEY_PAN_SPEED, 0);
          e.preventDefault();
          break;
        case "ArrowLeft":
        case "ArrowRight":
          rollForward.set(0, 0, -1).transformDirection(camera.matrixWorld);
          rollMatrix.makeRotationAxis(rollForward, e.code === "ArrowLeft" ? Math.PI / 128 : -Math.PI / 128);
          camera.up.transformDirection(rollMatrix);
          break;
      }
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("click", onClick);
    renderer.domElement.addEventListener("dblclick", onDoubleClick);
    window.addEventListener("keydown", onKeyDown);

    return () => {
      disposed = true;
      cancelAnimationFrame(raf);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("click", onClick);
      renderer.domElement.removeEventListener("dblclick", onDoubleClick);
      window.removeEventListener("keydown", onKeyDown);
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
      cropGroupRef.current = null;
      boxGroupRef.current = null;
      if (cropPreviewTimeoutRef.current !== null) window.clearTimeout(cropPreviewTimeoutRef.current);
      if (boxPreviewTimeoutRef.current !== null) window.clearTimeout(boxPreviewTimeoutRef.current);
      enabledDynosRef.current = [];
      cutoffDynosRef.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url]);

  // ---- classic-viewer parity: camera targeting + overlay projection -------
  // Near-verbatim ports from splat-viewer.tsx, adapted from the mkkellogg
  // viewer handle (untyped viewer.camera/viewer.controls + Vec3-constructor
  // tricks) to Spark's typed THREE camera/controls refs. Keep the math in
  // lockstep with the classic viewer until it is deleted.

  // Restore the viewer's default camera/orbit. Parent state clears overlays separately.
  useEffect(() => {
    const cam = cameraRef.current;
    const ctr = controlsRef.current;
    if (!resetViewToken || !cam || !ctr) return;
    if (typeof defaultFovRef.current === "number") {
      cam.fov = defaultFovRef.current;
      cam.updateProjectionMatrix();
    }
    cam.position.copy(INITIAL_CAMERA_POSITION);
    cam.up.copy(INITIAL_CAMERA_UP);
    ctr.target.copy(INITIAL_CAMERA_LOOK_AT);
    cam.lookAt(ctr.target);
    ctr.update();
  }, [resetViewToken]);

  // Fly the camera to a search hit: recenter on the 3D point, keeping the current
  // viewing angle, pulled back to frame the match (radius-scaled distance).
  useEffect(() => {
    const cam = cameraRef.current;
    const ctr = controlsRef.current;
    if (!focus || !cam || !ctr) return;
    const [fx, fy, fz] = focus.point;
    // Stand back to frame the target by its extent: small item -> close, large space ->
    // zoom-to-extents. FLY_MAX caps big legend objects (a floor) so we don't fly outside
    // the scene. (Search matches are small, so their feel is unchanged.)
    const FLY_ZOOM = 8;
    const FLY_MIN = 1.8;
    const FLY_MAX = 16;
    const dist = Math.min(Math.max(focus.radius * FLY_ZOOM, FLY_MIN), FLY_MAX);
    let dx = cam.position.x - fx;
    let dy = cam.position.y - fy;
    let dz = cam.position.z - fz;
    let len = Math.hypot(dx, dy, dz);
    if (len < 1e-3) {
      dx = 0; dy = -1; dz = 0.6; len = Math.hypot(dx, dy, dz);
    }
    ctr.target.set(fx, fy, fz);
    cam.position.set(fx + (dx / len) * dist, fy + (dy / len) * dist, fz + (dz / len) * dist);
    ctr.update();
  }, [focus]);

  // Tight camera-node zoom: unlike generic search/object focus, this aligns the
  // viewer behind the original camera and keeps the orbit pivot on the node.
  useEffect(() => {
    const cam = cameraRef.current;
    const ctr = controlsRef.current;
    if (!cameraNodeTarget || !cam || !ctr) return;
    const p = cameraNodeTarget.camera.position;
    const f = cameraNodeTarget.camera.forward;
    const u = cameraNodeTarget.camera.up;
    const node = new THREE.Vector3(p[0], p[1], p[2]);
    const forward = new THREE.Vector3(f[0], f[1], f[2]);
    if (forward.lengthSq() > 1e-10) forward.normalize();
    else forward.set(0, -1, 0);
    const up = new THREE.Vector3(u[0], u[1], u[2]);
    if (up.lengthSq() > 1e-10) up.normalize();
    else up.copy(INITIAL_CAMERA_UP);
    const distance = Math.max(cameraNodeTarget.distance ?? 0.35, 0.05);
    const pos = node.clone().add(forward.multiplyScalar(-distance));

    if (typeof defaultFovRef.current === "number") {
      cam.fov = defaultFovRef.current;
      cam.updateProjectionMatrix();
    }
    cam.position.copy(pos);
    cam.up.copy(up);
    ctr.target.copy(node);
    cam.lookAt(node);
    ctr.update();
  }, [cameraNodeTarget]);

  // Jump to an original capture camera pose. The target token lets the parent
  // request the same camera repeatedly.
  useEffect(() => {
    const cam = cameraRef.current;
    const ctr = controlsRef.current;
    if (!viewCamera || !cam || !ctr) return;
    const p = viewCamera.camera.position;
    const f = viewCamera.camera.forward;
    const u = viewCamera.camera.up;
    const pos = new THREE.Vector3(p[0], p[1], p[2]);
    const forward = new THREE.Vector3(f[0], f[1], f[2]).normalize();
    const up = new THREE.Vector3(u[0], u[1], u[2]).normalize();
    const targetDistance = Math.max(viewCamera.distance ?? 1.2, 0.2);
    const target = pos.clone().add(forward.multiplyScalar(targetDistance));

    if (typeof viewCamera.camera.fov_y_degrees === "number" && Number.isFinite(viewCamera.camera.fov_y_degrees)) {
      cam.fov = viewCamera.camera.fov_y_degrees;
      cam.updateProjectionMatrix();
    }
    cam.position.copy(pos);
    cam.up.copy(up);
    ctr.target.copy(target);
    cam.lookAt(target);
    ctr.update();
  }, [viewCamera]);

  // Project each 3D match to screen every frame so the highlight markers + label
  // track the camera as you orbit.
  useEffect(() => {
    if (!overlay || overlay.matches.length === 0) {
      setMarkers([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const cam = cameraRef.current;
      const root = containerRef.current;
      if (cam && root) {
        const rect = root.getBoundingClientRect();
        const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
        setMarkers(
          overlay.matches.map((m) => {
            const dir = new THREE.Vector3(m.point[0] - cam.position.x, m.point[1] - cam.position.y, m.point[2] - cam.position.z);
            const front = dir.dot(fwd) > 0;
            const p = new THREE.Vector3(m.point[0], m.point[1], m.point[2]);
            p.project(cam);
            return { x: (p.x * 0.5 + 0.5) * rect.width, y: (-p.y * 0.5 + 0.5) * rect.height, front };
          }),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [overlay]);

  // Project every highlight group's 3D points to screen each frame (legend toggles).
  // Separate from the search overlay so both can be shown at once without interfering.
  useEffect(() => {
    if (!highlights || highlights.length === 0) {
      setHlMarkers([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const cam = cameraRef.current;
      const root = containerRef.current;
      if (cam && root) {
        const rect = root.getBoundingClientRect();
        const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
        setHlMarkers(
          highlights.map((h) => ({
            label: h.label,
            color: h.color,
            pts: h.points.map((pt) => {
              const dir = new THREE.Vector3(pt[0] - cam.position.x, pt[1] - cam.position.y, pt[2] - cam.position.z);
              const front = dir.dot(fwd) > 0;
              const p = new THREE.Vector3(pt[0], pt[1], pt[2]);
              p.project(cam);
              return { x: (p.x * 0.5 + 0.5) * rect.width, y: (-p.y * 0.5 + 0.5) * rect.height, front };
            }),
          })),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [highlights]);

  // Project capture camera poses into screen-space. These are annotations only; they
  // don't modify the underlying Spark camera or controls. `frame` ("viewer" vs
  // "source") affects only the drawn opacity, exactly as in the classic viewer.
  useEffect(() => {
    if (!cameraOverlay || cameraOverlay.cameras.length === 0) {
      setCameraMarkers([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const cam = cameraRef.current;
      const root = containerRef.current;
      if (cam && root) {
        const rect = root.getBoundingClientRect();
        const viewForward = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
        const scale = Math.max(cameraOverlay.displayScale || 0.08, 0.01);
        const add = (a: [number, number, number], b: [number, number, number], m: number): [number, number, number] => [
          a[0] + b[0] * m,
          a[1] + b[1] * m,
          a[2] + b[2] * m,
        ];
        const project = (pt: [number, number, number]) => {
          const dir = new THREE.Vector3(pt[0] - cam.position.x, pt[1] - cam.position.y, pt[2] - cam.position.z);
          const front = dir.dot(viewForward) > 0;
          const p = new THREE.Vector3(pt[0], pt[1], pt[2]);
          p.project(cam);
          return { x: (p.x * 0.5 + 0.5) * rect.width, y: (-p.y * 0.5 + 0.5) * rect.height, front };
        };

        setCameraMarkers(
          cameraOverlay.cameras.map((c) => {
            const nose = add(c.position, c.forward, scale * 1.8);
            const left = add(nose, c.right, -scale * 0.55);
            const right = add(nose, c.right, scale * 0.55);
            const top = add(nose, c.up, scale * 0.45);
            const center = project(c.position);
            return {
              index: c.index,
              name: c.image_name,
              ...center,
              nose: project(nose),
              left: project(left),
              right: project(right),
              top: project(top),
              camera: c,
            };
          }),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [cameraOverlay]);

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

      {/* classic-viewer parity overlays: ported markup from splat-viewer.tsx.
          These are DOM/SVG siblings ABOVE the canvas — clicks on the marker
          buttons never reach the renderer's own pointer handlers, so they
          cannot double-fire crop/paint/measure picking. */}
      {showShortcutLegend && <ShortcutLegend />}
      {overlay &&
        markers.map((mk, i) =>
          mk.front ? (
            <button
              key={i}
              type="button"
              onClick={() => onPickMatch?.(i)}
              style={{ left: `${mk.x}px`, top: `${mk.y}px` }}
              title={`${overlay.label} — ${i + 1}/${overlay.matches.length}`}
              className={`pointer-events-auto absolute z-20 -translate-x-1/2 -translate-y-1/2 rounded-full transition ${
                i === overlay.active
                  ? "h-6 w-6 border-2 border-cyan-300 bg-cyan-300/25 shadow-[0_0_14px_rgba(34,211,238,0.75)]"
                  : "h-4 w-4 border-2 border-white/60 bg-white/10 hover:border-cyan-200 hover:bg-cyan-200/20"
              }`}
            />
          ) : null,
        )}
      {overlay && markers[overlay.active]?.front && (
        <div
          style={{ left: `${markers[overlay.active].x}px`, top: `${markers[overlay.active].y - 20}px` }}
          className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-md bg-black/75 px-2 py-0.5 text-xs font-semibold text-cyan-100 shadow backdrop-blur-sm"
        >
          {overlay.label}
        </div>
      )}
      {/* Legend highlights: a soft AREA wash — many translucent screen-blended dots over
          the object's own gaussians build up into a colored region — + one label at the
          region's centroid. "tile" paints the whole floor, not a pin. */}
      {hlMarkers.map((h, gi) => {
        const front = h.pts.filter((p) => p.front);
        if (!front.length) return null;
        const cx = front.reduce((s, p) => s + p.x, 0) / front.length;
        const cy = front.reduce((s, p) => s + p.y, 0) / front.length;
        return (
          <div key={gi}>
            {front.map((pt, pi) => (
              <div
                key={pi}
                style={{ left: `${pt.x}px`, top: `${pt.y}px`, backgroundColor: `${h.color}4d`, mixBlendMode: "screen" }}
                className="pointer-events-none absolute z-20 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
              />
            ))}
            <div
              style={{ left: `${cx}px`, top: `${cy}px`, color: h.color, borderColor: `${h.color}80` }}
              className="pointer-events-none absolute z-30 -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-md border bg-black/80 px-1.5 py-0.5 text-[11px] font-bold shadow backdrop-blur-sm"
            >
              {h.label}
            </div>
          </div>
        );
      })}
      {cameraOverlay && cameraMarkers.length > 0 && (
        <svg className="pointer-events-none absolute inset-0 z-20 h-full w-full" aria-hidden="true">
          {cameraMarkers.map((mk) =>
            mk.front ? (
              <g key={mk.index} opacity={cameraOverlay.frame === "viewer" ? 0.82 : 0.45}>
                <line x1={mk.x} y1={mk.y} x2={mk.nose.x} y2={mk.nose.y} stroke="#fbbf24" strokeWidth="1.3" strokeLinecap="round" />
                <line x1={mk.x} y1={mk.y} x2={mk.left.x} y2={mk.left.y} stroke="#f59e0b" strokeWidth="0.9" strokeLinecap="round" />
                <line x1={mk.x} y1={mk.y} x2={mk.right.x} y2={mk.right.y} stroke="#f59e0b" strokeWidth="0.9" strokeLinecap="round" />
                <line x1={mk.x} y1={mk.y} x2={mk.top.x} y2={mk.top.y} stroke="#fde68a" strokeWidth="0.8" strokeLinecap="round" />
              </g>
            ) : null,
          )}
        </svg>
      )}
      {cameraMarkers.map((mk) =>
        mk.front ? (
          <button
            key={mk.index}
            type="button"
            title={`Zoom to ${mk.name} · camera ${mk.index + 1}`}
            onClick={() => onPickCamera?.(mk.camera)}
            style={{ left: `${mk.x}px`, top: `${mk.y}px` }}
            className="pointer-events-auto absolute z-30 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-amber-100/80 bg-amber-300/75 shadow-[0_0_10px_rgba(251,191,36,0.65)] transition hover:h-4 hover:w-4 hover:bg-cyan-200"
          />
        ) : null,
      )}
      {cameraOverlay && cameraMarkers.length > 0 && (
        <div className="pointer-events-none absolute right-3 top-24 z-20 rounded-full border border-amber-300/25 bg-black/55 px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.2em] text-amber-100 backdrop-blur">
          {cameraMarkers.length} cameras
        </div>
      )}

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

      {/* control panel — top-left column on desktop, bottom drawer under
          1024px. Sections follow the host tab (panelSections: "measure" =
          search/paint/dimensions, "edit" = crop tools); chrome renders for
          both. Under lg on the Edit tab the Edit lane also overlays the right
          edge, so the drawer gets a lower max-height there — a cheap CSS-only
          mitigation that keeps the two from fully covering each other. */}
      {panelSections && (
      <div
        className={`absolute left-3 top-3 z-20 max-h-[calc(100%-1.5rem)] w-80 space-y-3 overflow-y-auto rounded-xl border border-white/10 bg-[#0a0f1a] p-3 text-xs text-zinc-200 shadow max-lg:bottom-2 max-lg:left-2 max-lg:right-2 max-lg:top-auto max-lg:w-auto ${
          panelSections === "edit" ? "max-lg:max-h-[30vh]" : "max-lg:max-h-[45vh]"
        }`}
      >
        <div className="flex items-center justify-between">
          <span className="font-semibold uppercase tracking-widest text-cyan-300/90">Spark beta</span>
          <span className="text-zinc-400">
            {splatCount !== null ? `${splatCount.toLocaleString()} splats` : "…"} · {fps} fps
          </span>
        </div>
        {placementMiss && (
          <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
            That click missed the splats — click on visible geometry (very faint splats don't register).
          </p>
        )}

        {panelSections === "measure" && (
        <>
        <>
            <SectionLabel>{job.langfield_available ? "Language overlay" : "Test search overlay"}</SectionLabel>
            {safeMode && (
              <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
                Real language search is blocked by the current hardware gate. Spark search will use a deterministic
                test pattern until the LangField worker is available.
              </p>
            )}
            {!job.langfield_available && (
              <p className="rounded-lg border border-cyan-300/15 bg-cyan-300/10 px-2 py-1.5 text-[10px] leading-snug text-cyan-100/80">
                This scene has no language field, so searches use a test pattern to verify overlay controls.
              </p>
            )}
            {/* Stale field: strokes/searches would 409 server-side — say so
                and offer the one-click cure instead of harvesting errors. */}
            {job.langfield_available && job.langfield_stale && (
              <div className="space-y-1.5 rounded-lg border border-amber-300/25 bg-amber-300/10 px-2 py-1.5">
                <p className="text-[10px] leading-snug text-amber-100/85">
                  The language field no longer matches this scene's geometry — it was edited after the field
                  was built. Search and paint are disabled until it's rebuilt (seconds; painted labels are
                  carried across).
                </p>
                <Button
                  type="button"
                  size="sm"
                  variant="primary"
                  className="w-full"
                  onClick={() => void runLangfieldRebuild()}
                  disabled={rebuildBusy}
                >
                  {rebuildBusy ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Rebuilding…
                    </>
                  ) : (
                    "Rebuild language field"
                  )}
                </Button>
              </div>
            )}
            {rebuildNotice && <p className="text-[10px] leading-snug text-emerald-300/90">{rebuildNotice}</p>}
            {rebuildError && <p className="text-[10px] leading-snug text-rose-300/90">{rebuildError}</p>}
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                void addQuery();
              }}
            >
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={channels.length >= 4 ? "4 query limit reached" : "Add a search…"}
                disabled={channels.length >= 4 || Boolean(job.langfield_available && job.langfield_stale)}
                size="xs"
                className="disabled:opacity-50"
              />
              <Button
                type="submit"
                size="sm"
                disabled={
                  queryBusy || !query.trim() || channels.length >= 4 ||
                  Boolean(job.langfield_available && job.langfield_stale)
                }
              >
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
                <span
                  className={
                    c.source === "test"
                      ? "shrink-0 rounded border border-amber-300/40 bg-amber-300/15 px-1 text-[9px] font-bold uppercase text-amber-200"
                      : "shrink-0 text-[10px] text-zinc-500"
                  }
                  title={c.source === "test" ? "Deterministic fake pattern — NOT a real search. The splatlab-langfield worker service is down (start it: systemctl --user start splatlab-langfield) or the hardware gate is active." : undefined}
                >
                  {c.source === "test" ? "test pattern" : c.matchCount !== null ? `${c.matchCount}×` : ""}
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

        {job.langfield_available && !safeMode && !job.langfield_stale && (
          <>
            <div className="h-px bg-white/10" />
            <SectionLabel>Paint the field</SectionLabel>
            {/* One brush, two targets: "field" biases search embeddings,
                "class" assigns surface classes the generative stages use. */}
            <div className="flex items-center gap-1.5">
              <Button
                type="button"
                size="sm"
                variant={paintTarget === "field" ? "primary" : "outline"}
                onClick={() => setPaintTarget("field")}
                title="Pin/boost/suppress a search label on the brushed splats"
              >
                Label
              </Button>
              <Button
                type="button"
                size="sm"
                variant={paintTarget === "class" ? "primary" : "outline"}
                onClick={() => {
                  setPaintTarget("class");
                  if (!taxonomy) void loadClassTaxonomy();
                  void loadClassRecords();
                }}
                title="Assign a surface class (grass/dirt/pavement/…) — feeds the generative stages"
              >
                Class
              </Button>
              <label className="ml-auto flex items-center gap-1 text-[10px] text-zinc-400">
                <input
                  type="checkbox"
                  checked={showClassLayer}
                  onChange={(e) => void toggleClassLayer(e.target.checked)}
                />
                Show class layer
              </label>
            </div>
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
                    min={sliderScale ? sliderScale.sphereMax / 400 : 0.02}
                    max={sliderScale ? sliderScale.sphereMax / 8 : 0.4}
                    step={sliderScale ? sliderScale.sphereMax / 400 : 0.01}
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
                  The cyan wireframe sphere is your brush. <b>Click</b> for one stroke or{" "}
                  <b>hold and sweep</b> to paint ({selCount.toLocaleString()} splats selected, shown
                  cyan). Everything stays local until you press the commit button — that's the save.
                  Camera orbit is paused while Painting is armed — toggle it off to navigate.
                </p>
                {paintTarget === "class" && (
                  <div className="space-y-1.5">
                    {taxonomy ? (
                      <div className="flex flex-wrap gap-1">
                        {taxonomy.map((c) => (
                          <button
                            key={c.id}
                            type="button"
                            onClick={() => setSelectedClassId(c.id)}
                            title={`${c.display} (${c.category})`}
                            className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-semibold transition ${
                              selectedClassId === c.id
                                ? "border-white/60 text-white"
                                : "border-white/15 text-zinc-400 hover:text-zinc-200"
                            }`}
                          >
                            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: c.color }} />
                            {c.display}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="text-[10px] text-zinc-500">Loading classes…</p>
                    )}
                    {selCount > 0 && selectedClassId && (
                      <div className="flex items-center gap-1.5">
                        <Button
                          type="button"
                          size="sm"
                          onClick={() => void commitClassPaint(false)}
                          disabled={classBusy}
                        >
                          {classBusy ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            `Paint ${selCount.toLocaleString()} as ${taxonomy?.find((c) => c.id === selectedClassId)?.display ?? selectedClassId}`
                          )}
                        </Button>
                        {classNeedsForce && (
                          <Button type="button" size="sm" variant="outline" onClick={() => void commitClassPaint(true)}>
                            Force
                          </Button>
                        )}
                      </div>
                    )}
                    {classNotice && <p className="text-[10px] leading-snug text-emerald-300/90">{classNotice}</p>}
                    {classError && <p className="text-[10px] leading-snug text-rose-300/90">{classError}</p>}
                    {classRecords.length > 0 && (
                      <div className="space-y-1">
                        {classRecords.map((r) => (
                          <div key={r.id} className="flex items-center gap-2 text-[10px]">
                            <span
                              className="h-2 w-2 shrink-0 rounded-full"
                              style={{ backgroundColor: taxonomy?.find((c) => c.id === r.class_id)?.color ?? "#fff" }}
                            />
                            <span className="min-w-0 flex-1 truncate text-zinc-300">
                              {taxonomy?.find((c) => c.id === r.class_id)?.display ?? r.class_id} ·{" "}
                              {r.count.toLocaleString()} splats
                              {r.invalid_reason && <span className="text-amber-300"> · {r.invalid_reason}</span>}
                            </span>
                            <button
                              type="button"
                              onClick={() => void removeClassRecord(r.id)}
                              className="shrink-0 text-zinc-500 hover:text-rose-300"
                              title="Delete this class paint (fully reverts it)"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                {paintTarget === "field" && selCount > 0 && (
                  <div className="space-y-1.5">
                    <Input
                      value={paintLabel}
                      onChange={(e) => setPaintLabel(e.target.value)}
                      placeholder='Label — anything, e.g. "dad’s corner"'
                      size="xs"
                    />
                    <Input
                      value={paintAliases}
                      onChange={(e) => setPaintAliases(e.target.value)}
                      placeholder="Aliases, comma-separated (optional)"
                      size="xs"
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
        {job.langfield_available && safeMode && (
          <>
            <div className="h-px bg-white/10" />
            <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
              Paint and semantic edits are disabled while the hardware-maintenance gate is active
              {computeReason ? `: ${computeReason}` : "."}
            </p>
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
            <>
              <a
                href={`/api/splat/jobs/${job.job_id}/dimensions/export?fmt=csv`}
                title="Download all dimensions as a CSV file"
                className="flex h-8 shrink-0 items-center gap-1 rounded-xl border border-white/15 bg-white/5 px-2 text-[11px] font-semibold text-zinc-200 hover:bg-white/10"
              >
                <Download className="h-3.5 w-3.5" /> CSV
              </a>
              <Button type="button" size="sm" variant="outline" onClick={deleteAllDims} title="Delete all dimensions">
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </>
          )}
        </div>
        {measureArm && (
          <p className="text-[10px] leading-snug text-zinc-500">
            Click two points on the scene. Drag any endpoint later to adjust; dimensions are saved with the
            scene.
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
              <Input
                value={calibLen}
                onChange={(e) => setCalibLen(e.target.value)}
                placeholder="known length"
                inputMode="decimal"
                size="xs"
                className="w-24"
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
        </>
        )}

        {panelSections === "edit" && !safeMode && (
          <>
            <SectionLabel>Crop to sphere</SectionLabel>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant={cropMode ? "primary" : "outline"}
                onClick={() => setCropMode((v) => !v)}
                title="Set a sphere around the subject; splats outside it are permanently removed. Undoable."
              >
                <Scissors className="h-3.5 w-3.5" /> {cropMode ? "Cropping…" : "Crop to sphere"}
              </Button>
              {cropLastVersion !== null && (
                <Button type="button" size="sm" variant="outline" onClick={() => void undoCrop()} disabled={cropUndoBusy} title="Undo the last crop">
                  {cropUndoBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Undo2 className="h-3.5 w-3.5" />} Undo crop
                </Button>
              )}
            </div>
            {cropMode && (
              <>
                <p className="text-[10px] leading-snug text-zinc-500">
                  Click the scene to place the sphere's center. Drag the radius slider to size it — everything
                  shown red will be permanently removed.
                </p>
                <div className="flex items-center gap-2">
                  <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">radius</span>
                  <input
                    type="range"
                    min={sliderScale ? sliderScale.sphereMax / 200 : 0.05}
                    max={sliderScale?.sphereMax ?? 5}
                    step={sliderScale ? sliderScale.sphereMax / 200 : 0.05}
                    value={cropRadius}
                    onChange={(e) => setCropRadius(Number(e.target.value))}
                    className="w-full"
                  />
                  <span className="w-16 shrink-0 text-right text-zinc-400">
                    {metersPerUnit ? `${(cropRadius * metersPerUnit).toFixed(2)} m` : `${cropRadius.toFixed(2)} u`}
                  </span>
                </div>
                {cropCenter && (
                  <>
                    <p className="text-[10px] leading-snug text-zinc-400">
                      {cropRemovedCount !== null
                        ? `${job.langfield_available ? "" : "≈"}${cropRemovedCount.toLocaleString()} of ${(splatCount ?? 0).toLocaleString()} loaded splats would be removed (${splatCount ? Math.round((cropRemovedCount / splatCount) * 100) : 0}%).${job.langfield_available ? "" : " Preview estimate — the full scene has more splats than this decimated preview."}`
                        : "Counting…"}
                    </p>
                    {job.langfield_available && (
                      <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
                        This scene has a language field. Cropping changes the gaussian count, so search and paint
                        pause until the field is rebuilt — one click afterwards, painted labels carry across.
                      </p>
                    )}
                    <Button
                      type="button"
                      size="sm"
                      variant={cropConfirmArmed ? "primary" : "outline"}
                      onClick={() => (cropConfirmArmed ? void applyCrop() : setCropConfirmArmed(true))}
                      disabled={cropBusy}
                      className={cropConfirmArmed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}
                    >
                      {cropBusy ? (
                        <>
                          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Applying…
                        </>
                      ) : cropConfirmArmed ? (
                        `Sure? Removes ${(cropRemovedCount ?? 0).toLocaleString()} splats permanently`
                      ) : (
                        "Apply crop"
                      )}
                    </Button>
                  </>
                )}
              </>
            )}
            {cropError && <p className="text-[10px] leading-snug text-rose-300/90">{cropError}</p>}

            <div className="h-px bg-white/10" />
            <SectionLabel>Crop to box</SectionLabel>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant={boxMode ? "primary" : "outline"}
                onClick={() => setBoxMode((v) => !v)}
                title="Set a box around the subject; splats outside it are permanently removed. Undoable."
              >
                <Box className="h-3.5 w-3.5" /> {boxMode ? "Cropping…" : "Crop to box"}
              </Button>
              {boxLastVersion !== null && (
                <Button type="button" size="sm" variant="outline" onClick={() => void undoBoxCrop()} disabled={boxUndoBusy} title="Undo the last box crop">
                  {boxUndoBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Undo2 className="h-3.5 w-3.5" />} Undo crop
                </Button>
              )}
            </div>
            {boxMode && (
              <>
                <p className="text-[10px] leading-snug text-zinc-500">
                  Click the scene to place the box's center, then size each side — everything shown red
                  (outside the box) will be permanently removed.
                </p>
                {(["x", "y", "z"] as const).map((axis, ai) => (
                  <div key={axis} className="flex items-center gap-2">
                    <span className="w-8 shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">±{axis}</span>
                    <input
                      type="range"
                      min={sliderScale ? sliderScale.boxMax[ai] / 200 : 0.05}
                      max={sliderScale?.boxMax[ai] ?? 5}
                      step={sliderScale ? sliderScale.boxMax[ai] / 200 : 0.05}
                      value={boxExtents[ai]}
                      onChange={(e) =>
                        setBoxExtents((prev) => {
                          const next = [...prev] as [number, number, number];
                          next[ai] = Number(e.target.value);
                          return next;
                        })
                      }
                      className="w-full"
                    />
                    <span className="w-16 shrink-0 text-right text-zinc-400">
                      {metersPerUnit ? `${(boxExtents[ai] * metersPerUnit).toFixed(2)} m` : `${boxExtents[ai].toFixed(2)} u`}
                    </span>
                  </div>
                ))}
                {boxCenter && (
                  <>
                    <p className="text-[10px] leading-snug text-zinc-400">
                      {boxRemovedCount !== null
                        ? `${job.langfield_available ? "" : "≈"}${boxRemovedCount.toLocaleString()} of ${(splatCount ?? 0).toLocaleString()} loaded splats would be removed (${splatCount ? Math.round((boxRemovedCount / splatCount) * 100) : 0}%).${job.langfield_available ? "" : " Preview estimate — the full scene has more splats than this decimated preview."}`
                        : "Counting…"}
                    </p>
                    {job.langfield_available && (
                      <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
                        This scene has a language field. Cropping changes the gaussian count, so search and paint
                        pause until the field is rebuilt — one click afterwards, painted labels carry across.
                      </p>
                    )}
                    <Button
                      type="button"
                      size="sm"
                      variant={boxConfirmArmed ? "primary" : "outline"}
                      onClick={() => (boxConfirmArmed ? void applyBoxCrop() : setBoxConfirmArmed(true))}
                      disabled={boxBusy}
                      className={boxConfirmArmed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}
                    >
                      {boxBusy ? (
                        <>
                          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Applying…
                        </>
                      ) : boxConfirmArmed ? (
                        `Sure? Removes ${(boxRemovedCount ?? 0).toLocaleString()} splats permanently`
                      ) : (
                        "Apply crop"
                      )}
                    </Button>
                  </>
                )}
              </>
            )}
            {boxError && <p className="text-[10px] leading-snug text-rose-300/90">{boxError}</p>}
            {/* Non-fatal regen failures from the last crop/undo — a scene
                silently degrading to raw-.ply serving used to look like
                success. Survives section collapse; cleared on next apply. */}
            {editWarnings.length > 0 && (
              <div className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5">
                <p className="text-[10px] font-semibold text-amber-100/90">
                  The crop applied, but some derived copies failed to rebuild:
                </p>
                {editWarnings.map((w, i) => (
                  <p key={i} className="mt-0.5 text-[10px] leading-snug text-amber-100/75">
                    {w}
                  </p>
                ))}
              </div>
            )}
            {/* Polled truth, one shared block for both crop tools — shows for
                ANY stale-making edit (this tab, the Edit lane, another tab)
                and offers the one-click rebuild right where the crop landed. */}
            {job.langfield_stale && (
              <div className="space-y-1.5 rounded-lg border border-amber-300/25 bg-amber-300/10 px-2 py-1.5">
                <p className="text-[10px] leading-snug text-amber-100/85">
                  Language field is stale — the scene was edited after the field was built. Rebuild it in one
                  click (seconds; painted labels are carried across the edit).
                </p>
                <Button
                  type="button"
                  size="sm"
                  variant="primary"
                  className="w-full"
                  onClick={() => void runLangfieldRebuild()}
                  disabled={rebuildBusy}
                >
                  {rebuildBusy ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" /> Rebuilding…
                    </>
                  ) : (
                    "Rebuild language field now"
                  )}
                </Button>
                {rebuildError && <p className="text-[10px] leading-snug text-rose-300/90">{rebuildError}</p>}
              </div>
            )}
            {rebuildNotice && <p className="text-[10px] leading-snug text-emerald-300/90">{rebuildNotice}</p>}
          </>
        )}
        {panelSections === "edit" && safeMode && (
          <p className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[10px] leading-snug text-amber-100/85">
            Crop edits are disabled while the hardware-maintenance gate is active
            {computeReason ? `: ${computeReason}` : "."}
          </p>
        )}
      </div>
      )}

      {/* query legend — updates live with queries, colors, mode, and threshold */}
      {channels.length > 0 && (
        <div className="absolute bottom-16 right-3 z-20 w-64 space-y-2 rounded-xl border border-white/10 bg-[#0a0f1a] p-3 text-xs text-zinc-200 shadow">
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

// Floating shortcut reference — no background, tucked top-right, non-interactive.
// Ported from the classic viewer, trimmed to the shortcuts Spark actually
// implements: F/G focal, =/- splat scale, and I/C/P/O were mkkellogg-library
// internals (focalAdjustment, info panel, mesh cursor, point-cloud, ortho)
// with no Spark equivalent — advertising them here would be lying.
function ShortcutLegend() {
  const k = "text-white/75";
  return (
    <div className="pointer-events-none absolute right-3 top-3 z-10 select-none text-right font-mono text-[10px] leading-[1.55] text-white/40">
      <div><span className={k}>drag</span> orbit · <span className={k}>scroll</span> zoom</div>
      <div><span className={k}>W A S D</span> pan · <span className={k}>← →</span> roll</div>
      <div><span className={k}>dbl-click</span> set orbit pivot</div>
    </div>
  );
}
