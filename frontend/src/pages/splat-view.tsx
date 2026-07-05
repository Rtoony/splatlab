import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useRoute } from "wouter";
import { apiRequest, fetchLangfieldInventory, fetchSplatCameras, queryLangfield } from "@/lib/api";
import { setSplatlabFeedbackContext } from "@/lib/feedback-context";
import type {
  LangfieldInventoryItem,
  LangfieldQueryResult,
  SplatCameraPose,
  SplatJob,
  SplatStatusResponse,
} from "@/lib/contracts";
import { SplatViewer, type ViewerCameraNodeTarget, type ViewerCameraViewTarget, type ViewerHighlight, type ViewerOverlay } from "@/components/splat-viewer";
import { Button, Card, Input, SectionLabel } from "@/components/ui";
import { ArrowLeft, Camera, ChevronDown, ChevronUp, Compass, Crosshair, Download, Eye, EyeOff, Layers, Loader2, Orbit, RotateCcw, Search, SlidersHorizontal, Sparkles, X } from "lucide-react";
import { SparkSceneViewer } from "@/components/spark-scene-viewer";

// Distinct colors handed out to toggled inventory objects (stable per item index).
const HL_PALETTE = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#fb7185", "#facc15", "#4ade80", "#c084fc"];

export default function SplatViewPage() {
  const [, params] = useRoute("/view/:jobId");
  const jobId = params?.jobId ?? "";

  const { data: status, isLoading } = useQuery({
    queryKey: ["status"],
    queryFn: () => apiRequest<SplatStatusResponse>("/api/splat/status"),
    refetchInterval: 4000,
  });

  const job: SplatJob | undefined = useMemo(
    () => status?.jobs.find((j) => j.job_id === jobId),
    [status, jobId],
  );
  const viewUrl = job?.preview_web_url ?? job?.preview_view_url ?? null;
  const title = job ? job.input_path?.split("/").pop() || job.job_id : jobId;
  const [cameraOverlayOn, setCameraOverlayOn] = useState(false);
  const [inventoryOpen, setInventoryOpen] = useState(false);
  const [cameraShotsOpen, setCameraShotsOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [shortcutLegendOpen, setShortcutLegendOpen] = useState(false);
  const [resetViewToken, setResetViewToken] = useState(0);
  // Wave-2 Spark cutover, opt-in: the beta viewer carries the real language
  // heatmap + measure/scale tools; the classic viewer keeps overlays/search
  // fly-to until the full 2.4 port. Sticky per browser.
  const [sparkBeta, setSparkBeta] = useState(() => localStorage.getItem("splatlab.sparkBeta") === "1");
  function toggleSparkBeta() {
    setSparkBeta((v) => {
      localStorage.setItem("splatlab.sparkBeta", v ? "0" : "1");
      return !v;
    });
  }

  const { data: cameras, isFetching: camerasLoading, error: camerasError } = useQuery({
    queryKey: ["cameras", jobId],
    queryFn: () => fetchSplatCameras(jobId),
    enabled: cameraOverlayOn && Boolean(jobId && viewUrl),
    staleTime: Infinity,
    retry: false,
  });
  const cameraOverlay = useMemo(
    () =>
      cameraOverlayOn && cameras
        ? { cameras: cameras.cameras, displayScale: cameras.display_scale, frame: cameras.frame }
        : null,
    [cameraOverlayOn, cameras],
  );
  const [selectedCameraIndex, setSelectedCameraIndex] = useState<number | null>(null);
  const [cameraViewTarget, setCameraViewTarget] = useState<ViewerCameraViewTarget>(null);
  const [cameraNodeTarget, setCameraNodeTarget] = useState<ViewerCameraNodeTarget>(null);

  function zoomToCamera(camera: SplatCameraPose) {
    setCameraOverlayOn(true);
    setCameraShotsOpen(true);
    setSelectedCameraIndex(camera.index);
    setFlyTarget(null);
    setCameraViewTarget(null);
    setCameraNodeTarget((prev) => ({
      camera,
      token: (prev?.token ?? 0) + 1,
      distance: Math.max((cameras?.display_scale ?? 0.08) * 4, 0.24),
    }));
  }

  function viewFromCamera(camera: SplatCameraPose) {
    setCameraOverlayOn(true);
    setCameraShotsOpen(true);
    setSelectedCameraIndex(camera.index);
    setFlyTarget(null);
    setCameraNodeTarget(null);
    setCameraViewTarget((prev) => ({
      camera,
      token: (prev?.token ?? 0) + 1,
      distance: Math.max((cameras?.display_scale ?? 0.08) * 14, 1.1),
    }));
  }

  function resetToDefaultView() {
    setCameraOverlayOn(false);
    setInventoryOpen(false);
    setCameraShotsOpen(false);
    setSearchOpen(false);
    setActiveLabels(new Set());
    setResult(null);
    setActiveIdx(0);
    setHighlightOn(false);
    setShortcutLegendOpen(false);
    setSelectedCameraIndex(null);
    setCameraViewTarget(null);
    setCameraNodeTarget(null);
    setFlyTarget(null);
    setResetViewToken((token) => token + 1);
  }

  function enableAdvancedView() {
    setCameraOverlayOn(true);
    setInventoryOpen(true);
    setCameraShotsOpen(true);
    setSearchOpen(true);
    setShortcutLegendOpen(true);
  }

  // Search-result navigation: the matched instances, which one is active (flown-to
  // + emphasized), and whether the highlight/label overlay is shown.
  const [result, setResult] = useState<LangfieldQueryResult | null>(null);
  const [activeIdx, setActiveIdx] = useState(0);
  const [highlightOn, setHighlightOn] = useState(false);

  const matches = result?.matches ?? [];
  const activeMatch = matches[activeIdx];
  // Camera fly target — set by a search hit OR a legend "zoom to". Flying recenters the
  // orbit pivot on the object and frames it by its extent (see splat-viewer).
  const [flyTarget, setFlyTarget] = useState<FocusTarget | null>(null);
  useEffect(() => {
    if (activeMatch) setFlyTarget({ point: activeMatch.focus, radius: activeMatch.radius });
  }, [activeMatch]);
  const overlay = useMemo<ViewerOverlay>(
    () =>
      highlightOn && matches.length && result
        ? { matches: matches.map((m) => ({ point: m.focus, radius: m.radius })), active: activeIdx, label: result.query }
        : null,
    [highlightOn, matches, activeIdx, result],
  );

  // Scene object inventory (top-N auto-detected objects) + which ones are toggled on.
  const { data: inventory, isLoading: invLoading } = useQuery({
    queryKey: ["inventory", jobId],
    queryFn: () => fetchLangfieldInventory(jobId),
    enabled: Boolean(jobId && job?.langfield_available),
    staleTime: Infinity,
    retry: false,
  });
  const invItems = inventory?.items ?? [];
  const [activeLabels, setActiveLabels] = useState<Set<string>>(new Set());
  const colorFor = (label: string) => HL_PALETTE[Math.max(0, invItems.findIndex((i) => i.label === label)) % HL_PALETTE.length];

  // Each toggled-on inventory object becomes a colored highlight group (all its instances).
  const highlights = useMemo<ViewerHighlight[]>(
    () =>
      invItems
        .filter((it) => activeLabels.has(it.label))
        // area highlight: the object's own gaussian spread (fallback to instance centroids)
        .map((it) => ({
          label: it.label,
          color: colorFor(it.label),
          points: it.points?.length ? it.points : it.matches.map((m) => m.focus),
        })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [invItems, activeLabels],
  );

  useEffect(() => {
    setSplatlabFeedbackContext({
      page: "splat-viewer",
      job_id: job?.job_id ?? jobId,
      job_status: job?.status ?? null,
      job_stage: job?.stage ?? null,
      preview_available: Boolean(job?.preview_available),
      language_field_available: Boolean(job?.langfield_available),
      active_search: result?.query ?? null,
      active_match_index: result ? activeIdx : null,
      visible_inventory_labels: Array.from(activeLabels),
      capture_cameras_visible: cameraOverlayOn,
      capture_camera_count: cameras?.count ?? null,
      capture_camera_frame: cameras?.frame ?? null,
      camera_shots_panel_open: cameraShotsOpen,
      inventory_panel_open: inventoryOpen,
      search_panel_open: searchOpen,
      shortcut_legend_open: shortcutLegendOpen,
      selected_capture_camera_index: selectedCameraIndex,
      selected_capture_camera_name: cameras?.cameras.find((camera) => camera.index === selectedCameraIndex)?.image_name ?? null,
      camera_node_zoom_active: Boolean(cameraNodeTarget),
    });
    return () => setSplatlabFeedbackContext(null);
  }, [activeIdx, activeLabels, cameraNodeTarget, cameraOverlayOn, cameraShotsOpen, cameras, inventoryOpen, job, jobId, result, searchOpen, selectedCameraIndex, shortcutLegendOpen]);

  return (
    <div className="flex h-screen flex-col bg-[#05070d] text-zinc-100">
      <header className="flex items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Link href="/" className="flex shrink-0 items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200">
            <ArrowLeft className="h-4 w-4" /> Splat Lab
          </Link>
          <span className="text-white/20">/</span>
          <div className="flex min-w-0 items-center gap-2">
            <Orbit className="h-4 w-4 shrink-0 text-cyan-300" />
            <span className="truncate text-sm font-semibold">{title}</span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {job && viewUrl && (
            <Button
              type="button"
              variant={sparkBeta ? "primary" : "outline"}
              size="sm"
              onClick={toggleSparkBeta}
              title="Spark beta viewer: real language heatmap on the splats + measure/scale tools"
              className={sparkBeta ? "bg-cyan-300 text-zinc-950 hover:bg-cyan-200" : ""}
            >
              <Compass className="h-3.5 w-3.5" /> {sparkBeta ? "Spark beta ON" : "Spark beta"}
            </Button>
          )}
          {job && viewUrl && !sparkBeta && (
            <>
              <Button type="button" variant="outline" size="sm" onClick={resetToDefaultView} title="Reset camera and collapse viewer extras">
                <RotateCcw className="h-3.5 w-3.5" /> Reset
              </Button>
              <Button type="button" variant="outline" size="sm" onClick={enableAdvancedView} title="Open advanced search, scene, and camera tools">
                <SlidersHorizontal className="h-3.5 w-3.5" /> Advanced
              </Button>
              <Button
                type="button"
                variant={cameraOverlayOn ? "primary" : "outline"}
                size="sm"
                onClick={() => {
                  setCameraOverlayOn((v) => !v);
                  setCameraShotsOpen((v) => (!cameraOverlayOn ? true : v));
                }}
                title={camerasError ? "Camera poses are unavailable for this scene" : "Toggle capture camera locations"}
                className={cameraOverlayOn ? "bg-amber-300 text-zinc-950 hover:bg-amber-200" : ""}
              >
                {camerasLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Camera className="h-3.5 w-3.5" />}
                {cameraOverlayOn && cameras ? `${cameras.count}/${cameras.total} cameras` : "Cameras"}
              </Button>
            </>
          )}
          {job?.preview_file_url && (
            <a
              href={job.preview_file_url}
              download={`${jobId}.ply`}
              className="flex shrink-0 items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-white/10"
            >
              <Download className="h-3.5 w-3.5" /> Full-quality .ply
            </a>
          )}
          {job?.preview_file_url && (
            <a
              href={`/supersplat/?load=${encodeURIComponent(job.preview_file_url)}&filename=${encodeURIComponent(`${jobId}.ply`)}`}
              target="_blank"
              rel="noreferrer"
              title="Open this scene in the SuperSplat editor (select, crop, transform, export)"
              className="flex shrink-0 items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-white/10"
            >
              <Sparkles className="h-3.5 w-3.5" /> Edit in SuperSplat
            </a>
          )}
        </div>
      </header>
      <main className="relative flex-1 overflow-hidden">
        {isLoading && !job ? (
          <Centered>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading…
          </Centered>
        ) : !job ? (
          <Centered>
            <div className="text-center">
              <p className="font-semibold text-zinc-200">Scene not found</p>
              <Link href="/" className="mt-2 inline-block text-cyan-300 hover:underline">
                Back to Splat Lab
              </Link>
            </div>
          </Centered>
        ) : !viewUrl || !job.preview_available ? (
          <Centered>
            <div className="text-center">
              <Loader2 className="mx-auto mb-2 h-5 w-5 animate-spin text-cyan-300" />
              <p className="font-semibold text-zinc-200">
                {job.status === "completed" ? "Preparing preview…" : `Scene is ${job.status}…`}
              </p>
            </div>
          </Centered>
        ) : sparkBeta ? (
          <SparkSceneViewer key={job.job_id} job={job} />
        ) : (
          <>
            <SplatViewer
              url={viewUrl}
              format="ply"
              fill
              focus={flyTarget}
              overlay={overlay}
              highlights={highlights}
              cameraOverlay={cameraOverlay}
              viewCamera={cameraViewTarget}
              cameraNodeTarget={cameraNodeTarget}
              resetViewToken={resetViewToken}
              showShortcutLegend={shortcutLegendOpen}
              onPickMatch={setActiveIdx}
              onPickCamera={zoomToCamera}
            />
            {cameraOverlayOn && camerasError && (
              <div className="pointer-events-none absolute right-3 top-20 z-30 max-w-xs rounded-xl border border-amber-300/25 bg-black/70 px-3 py-2 text-xs text-amber-100 shadow backdrop-blur-md">
                Camera poses are unavailable for this scene. Generated single-image splats and unfinished captures usually do not have SfM cameras.
              </div>
            )}
            {cameraOverlayOn && cameras && (
              <CameraShotsLegend
                cameras={cameras.cameras}
                total={cameras.total}
                sampled={cameras.sampled}
                activeIndex={selectedCameraIndex}
                open={cameraShotsOpen}
                onOpenChange={setCameraShotsOpen}
                onZoom={zoomToCamera}
                onView={viewFromCamera}
              />
            )}
            {job.langfield_available && (invItems.length > 0 || invLoading) && (
              <InventoryLegend
                items={invItems}
                loading={invLoading}
                active={activeLabels}
                open={inventoryOpen}
                onOpenChange={setInventoryOpen}
                colorFor={colorFor}
                onToggle={(label) =>
                  setActiveLabels((prev) => {
                    const next = new Set(prev);
                    next.has(label) ? next.delete(label) : next.add(label);
                    return next;
                  })
                }
                onZoom={(it) => setFlyTarget({ point: it.focus, radius: it.radius })}
                onClear={() => setActiveLabels(new Set())}
              />
            )}
            {job.langfield_available && (
              <LangfieldSearch
                jobId={job.job_id}
                result={result}
                activeIdx={activeIdx}
                highlightOn={highlightOn}
                open={searchOpen}
                onOpenChange={setSearchOpen}
                onResult={(r) => {
                  setResult(r);
                  setActiveIdx(0);
                }}
                onPick={setActiveIdx}
                onToggleHighlight={() => setHighlightOn((v) => !v)}
                onClear={() => setResult(null)}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center text-sm text-zinc-400">{children}</div>;
}

// Left-side "what's in this scene" legend: the top-N auto-detected objects with a
// presence bar; toggle any (multiple at once) to highlight + label all their instances
// on screen in a distinct color. Collapsible so it can get out of the way.
function InventoryLegend({
  items,
  loading,
  active,
  open,
  onOpenChange,
  colorFor,
  onToggle,
  onZoom,
  onClear,
}: {
  items: LangfieldInventoryItem[];
  loading: boolean;
  active: Set<string>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  colorFor: (label: string) => string;
  onToggle: (label: string) => void;
  onZoom: (it: LangfieldInventoryItem) => void;
  onClear: () => void;
}) {
  const maxP = Math.max(...items.map((i) => i.presence), 0.0001);
  return (
    <div className="pointer-events-none absolute left-3 top-3 z-20 w-56">
      <Card className="pointer-events-auto border-white/12 bg-[#070b14]/85 p-2 backdrop-blur-md">
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          className="flex w-full items-center justify-between gap-2 px-1 py-0.5 text-left"
        >
          <span className="flex items-center gap-1.5">
            <Layers className="h-3.5 w-3.5 text-cyan-200" />
            <SectionLabel>In this scene</SectionLabel>
          </span>
          <span className="flex items-center gap-1 text-[11px] text-zinc-500">
            {active.size > 0 && <span className="text-cyan-300">{active.size} on</span>}
            {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </span>
        </button>
        {open &&
          (loading ? (
            <p className="flex items-center gap-1.5 px-1 py-2 text-xs text-zinc-400">
              <Loader2 className="h-3 w-3 animate-spin" /> detecting objects…
            </p>
          ) : (
            <div className="mt-1.5">
              <div className="max-h-[52vh] space-y-0.5 overflow-y-auto pr-1">
                {items.map((it) => {
                  const on = active.has(it.label);
                  const color = colorFor(it.label);
                  return (
                    <div
                      key={it.label}
                      className={`flex w-full items-center gap-1 rounded-md pl-1.5 pr-0.5 transition ${on ? "bg-white/10" : "hover:bg-white/5"}`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          const wasOn = on;
                          onToggle(it.label);
                          if (!wasOn) onZoom(it); // activating -> also fly there
                        }}
                        title={`${(it.presence * 100).toFixed(1)}% of scene · reliability ${it.reliability.toFixed(2)} · ${it.count ?? 0} gaussians`}
                        className="flex min-w-0 flex-1 items-center gap-2 py-1 text-left"
                      >
                        <span
                          className="h-3 w-3 shrink-0 rounded-full border-2"
                          style={{ borderColor: color, backgroundColor: on ? color : "transparent", boxShadow: on ? `0 0 8px ${color}` : "none" }}
                        />
                        <span className={`w-[68px] shrink-0 truncate text-xs ${on ? "text-zinc-100" : "text-zinc-300"}`}>{it.label}</span>
                        <span className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/10">
                          <span
                            className="absolute inset-y-0 left-0 rounded-full"
                            style={{ width: `${(it.presence / maxP) * 100}%`, backgroundColor: color, opacity: on ? 1 : 0.55 }}
                          />
                        </span>
                        <span className="w-7 shrink-0 text-right text-[10px] tabular-nums text-zinc-500">{(it.presence * 100).toFixed(0)}%</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => onZoom(it)}
                        title="Zoom to & center on this"
                        className="shrink-0 rounded p-0.5 text-zinc-500 transition hover:bg-white/10 hover:text-cyan-200"
                      >
                        <Crosshair className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
              {active.size > 0 && (
                <button
                  type="button"
                  onClick={onClear}
                  className="mt-1 flex w-full items-center justify-center gap-1 rounded-md py-1 text-[11px] text-zinc-400 transition hover:bg-white/5 hover:text-zinc-200"
                >
                  <X className="h-3 w-3" /> clear highlights
                </button>
              )}
            </div>
          ))}
      </Card>
    </div>
  );
}

function CameraShotsLegend({
  cameras,
  total,
  sampled,
  activeIndex,
  open,
  onOpenChange,
  onZoom,
  onView,
}: {
  cameras: SplatCameraPose[];
  total: number;
  sampled: boolean;
  activeIndex: number | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onZoom: (camera: SplatCameraPose) => void;
  onView: (camera: SplatCameraPose) => void;
}) {
  return (
    <div className="pointer-events-none absolute right-3 top-20 z-20 w-[21rem] max-w-[calc(100vw-1.5rem)]">
      <Card className="pointer-events-auto border-amber-300/20 bg-[#100d08]/85 p-2 backdrop-blur-md">
        <button
          type="button"
          onClick={() => onOpenChange(!open)}
          className="flex w-full items-center justify-between gap-2 px-1 py-0.5 text-left"
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <Camera className="h-3.5 w-3.5 shrink-0 text-amber-200" />
            <SectionLabel className="text-amber-100/75">Camera shots</SectionLabel>
          </span>
          <span className="flex items-center gap-1 text-[11px] text-amber-100/60">
            {cameras.length}/{total}
            {sampled && <span>sampled</span>}
            {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </span>
        </button>
        {open && (
          <div className="mt-1.5">
            <p className="px-1 pb-1 text-[11px] leading-snug text-amber-100/55">
              Names come from the original source images. Crosshair zooms tight behind the camera node; View uses the capture pose and FOV.
            </p>
            <div className="max-h-[48vh] space-y-0.5 overflow-y-auto pr-1">
              {cameras.map((camera) => {
                const active = camera.index === activeIndex;
                return (
                  <div
                    key={camera.index}
                    className={`grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-1 rounded-md px-1 py-0.5 transition ${
                      active ? "bg-amber-300/15 ring-1 ring-amber-300/25" : "hover:bg-white/5"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => onView(camera)}
                      title={camera.file_path || camera.image_name}
                      className="min-w-0 py-1 text-left"
                    >
                      <span className={`block truncate text-xs font-semibold ${active ? "text-amber-100" : "text-zinc-200"}`}>
                        {camera.image_name || `camera-${camera.index + 1}`}
                      </span>
                      <span className="block text-[10px] text-zinc-500">#{camera.index + 1}</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => onZoom(camera)}
                      title="Zoom tight to this camera node"
                      className="rounded-md border border-white/10 bg-white/5 px-2 py-1 text-[11px] font-semibold text-zinc-300 transition hover:border-cyan-300/40 hover:text-cyan-100"
                    >
                      <Crosshair className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => onView(camera)}
                      title="View from this original camera pose"
                      className="rounded-md border border-amber-300/20 bg-amber-300/10 px-2 py-1 text-[11px] font-semibold text-amber-100 transition hover:bg-amber-300/20"
                    >
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

// The 3D point (viewer frame) + spread the viewer should fly to on a search hit.
export type FocusTarget = { point: [number, number, number]; radius: number };

// Text-search panel for scenes that carry an opt-in Language Field. Floats over
// the bottom of the viewer. A query renders a server-side relevancy heatmap strip
// AND returns the distinct 3D matches (via onResult). This panel is CONTROLLED —
// the page owns which match is active + whether highlights are on — so the
// in-viewer markers, the "fly to", and these chips all stay in sync.
function LangfieldSearch({
  jobId,
  result,
  activeIdx,
  highlightOn,
  open,
  onOpenChange,
  onResult,
  onPick,
  onToggleHighlight,
  onClear,
}: {
  jobId: string;
  result: LangfieldQueryResult | null;
  activeIdx: number;
  highlightOn: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onResult: (r: LangfieldQueryResult) => void;
  onPick: (i: number) => void;
  onToggleHighlight: () => void;
  onClear: () => void;
}) {
  const [text, setText] = useState("");
  const search = useMutation<LangfieldQueryResult, Error, string>({
    mutationFn: (q: string) => queryLangfield(jobId, q),
    onSuccess: (r) => onResult(r),
  });

  function submit() {
    const q = text.trim();
    if (q) search.mutate(q);
  }

  const matches = result?.matches ?? [];

  if (!open) {
    return (
      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-30 flex justify-center p-3 sm:p-4">
        <button
          type="button"
          onClick={() => onOpenChange(true)}
          className="pointer-events-auto flex items-center gap-2 rounded-full border border-cyan-300/20 bg-[#070b14]/80 px-3 py-2 text-xs font-bold uppercase tracking-[0.18em] text-cyan-100 shadow backdrop-blur-md transition hover:border-cyan-200/50 hover:bg-[#0a1724]"
        >
          <Sparkles className="h-3.5 w-3.5" /> Search this scene
        </button>
      </div>
    );
  }

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-30 flex justify-center p-3 sm:p-4">
      <Card className="pointer-events-auto w-full max-w-lg border-white/12 bg-[#070b14]/85 p-3 backdrop-blur-md">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-cyan-200" />
            <SectionLabel>Search this scene</SectionLabel>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              onClick={onToggleHighlight}
              title={highlightOn ? "Hide highlights & labels" : "Show highlights & labels"}
              className={`flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold transition ${
                highlightOn ? "bg-cyan-400/15 text-cyan-200" : "bg-white/5 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {highlightOn ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />} highlights
            </button>
            <button
              type="button"
              onClick={() => onOpenChange(false)}
              title="Collapse search"
              className="rounded-full p-1 text-zinc-500 transition hover:bg-white/10 hover:text-zinc-200"
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type what you’re looking for…"
            autoComplete="off"
            // The mkkellogg viewer listens for WASD/etc. on window (bubble phase),
            // which hijacks typing. Stop key events from reaching it while typing.
            onKeyDown={(e) => e.stopPropagation()}
            onKeyUp={(e) => e.stopPropagation()}
          />
          <Button type="submit" disabled={!text.trim() || search.isPending} className="shrink-0">
            {search.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            Search
          </Button>
        </form>

        {search.isPending && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-zinc-400">
            <Loader2 className="h-3 w-3 animate-spin" /> Searching…
          </p>
        )}
        {search.isError && (
          <p className="mt-2 text-xs text-red-300">{search.error.message || "Search failed."}</p>
        )}
        {result && !search.isPending && (
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <p className="truncate text-xs text-zinc-400">
                Relevancy for <span className="text-zinc-200">“{result.query}”</span>
                {matches.length > 1 && <span className="text-cyan-200"> · {matches.length} found</span>}
                {!result.ready && " · still building…"}
              </p>
              <button
                type="button"
                onClick={() => {
                  setText("");
                  search.reset();
                  onClear();
                }}
                className="flex shrink-0 items-center gap-1 text-xs text-zinc-400 transition hover:text-zinc-100"
              >
                <X className="h-3.5 w-3.5" /> Clear
              </button>
            </div>
            {/* Clickable per-instance result thumbnails (each framed on that match).
                Click = fly to + emphasize it. Falls back to the stitched heatmap on
                cold/legacy responses that carry no per-match thumbnails. */}
            {matches.some((m) => m.thumb) ? (
              <div className="flex gap-2 overflow-x-auto pb-1">
                {matches.map((m, i) =>
                  m.thumb ? (
                    <button
                      key={i}
                      type="button"
                      onClick={() => onPick(i)}
                      title={`Instance ${i + 1} · ${m.count} gaussians`}
                      className={`relative shrink-0 overflow-hidden rounded-lg border-2 transition ${
                        i === activeIdx
                          ? "border-cyan-400 shadow-[0_0_12px_rgba(34,211,238,0.6)]"
                          : "border-white/10 hover:border-white/40"
                      }`}
                    >
                      <img
                        src={`/api/splat/jobs/${jobId}/langfield/heatmap/${m.thumb}`}
                        alt={`Match ${i + 1}`}
                        className="h-20 w-20 object-cover"
                      />
                      <span className="absolute left-1 top-1 rounded bg-black/70 px-1.5 text-[10px] font-bold text-white">
                        {i + 1}
                      </span>
                    </button>
                  ) : null,
                )}
              </div>
            ) : result.heatmap_url ? (
              <img
                src={result.heatmap_url}
                alt={`Relevancy heatmap for "${result.query}"`}
                className="w-full rounded-xl border border-white/10"
              />
            ) : null}
          </div>
        )}
      </Card>
    </div>
  );
}
