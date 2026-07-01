import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Link, useRoute } from "wouter";
import { apiRequest, fetchLangfieldInventory, queryLangfield } from "@/lib/api";
import type {
  LangfieldInventoryItem,
  LangfieldQueryResult,
  SplatJob,
  SplatStatusResponse,
} from "@/lib/contracts";
import { SplatViewer, type ViewerHighlight, type ViewerOverlay } from "@/components/splat-viewer";
import { Button, Card, Input, SectionLabel } from "@/components/ui";
import { ArrowLeft, ChevronDown, ChevronUp, Crosshair, Download, Eye, EyeOff, Layers, Loader2, Orbit, Search, Sparkles, X } from "lucide-react";

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

  // Search-result navigation: the matched instances, which one is active (flown-to
  // + emphasized), and whether the highlight/label overlay is shown.
  const [result, setResult] = useState<LangfieldQueryResult | null>(null);
  const [activeIdx, setActiveIdx] = useState(0);
  const [highlightOn, setHighlightOn] = useState(true);

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
        {job?.preview_file_url && (
          <a
            href={job.preview_file_url}
            download={`${jobId}.ply`}
            className="flex shrink-0 items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-3 py-1.5 text-xs font-semibold text-zinc-200 hover:bg-white/10"
          >
            <Download className="h-3.5 w-3.5" /> Full-quality .ply
          </a>
        )}
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
        ) : (
          <>
            <SplatViewer url={viewUrl} format="ply" fill focus={flyTarget} overlay={overlay} highlights={highlights} onPickMatch={setActiveIdx} />
            {job.langfield_available && (invItems.length > 0 || invLoading) && (
              <InventoryLegend
                items={invItems}
                loading={invLoading}
                active={activeLabels}
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
  colorFor,
  onToggle,
  onZoom,
  onClear,
}: {
  items: LangfieldInventoryItem[];
  loading: boolean;
  active: Set<string>;
  colorFor: (label: string) => string;
  onToggle: (label: string) => void;
  onZoom: (it: LangfieldInventoryItem) => void;
  onClear: () => void;
}) {
  const [open, setOpen] = useState(true);
  const maxP = Math.max(...items.map((i) => i.presence), 0.0001);
  return (
    <div className="pointer-events-none absolute left-3 top-3 z-20 w-56">
      <Card className="pointer-events-auto border-white/12 bg-[#070b14]/85 p-2 backdrop-blur-md">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
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
  onResult,
  onPick,
  onToggleHighlight,
  onClear,
}: {
  jobId: string;
  result: LangfieldQueryResult | null;
  activeIdx: number;
  highlightOn: boolean;
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

  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 z-30 flex justify-center p-3 sm:p-4">
      <Card className="pointer-events-auto w-full max-w-lg border-white/12 bg-[#070b14]/85 p-3 backdrop-blur-md">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-cyan-200" />
            <SectionLabel>Search this scene</SectionLabel>
          </div>
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
