import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import type {
  SplatJob,
  SplatStatusResponse,
  SplatTransferEntry,
  SplatTransfersResponse,
  SplatUploadResult,
} from "@/lib/contracts";
import { Badge, Button, Card, SectionLabel } from "@/components/ui";
import { SplatViewer } from "@/components/splat-viewer";
import {
  AlertTriangle,
  Box,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Download,
  FolderOpen,
  Loader2,
  Orbit,
  Pin,
  RefreshCw,
  Sparkles,
  Square,
  UploadCloud,
  Wand2,
  X,
} from "lucide-react";

// ── pipeline metadata ────────────────────────────────────────────────────────
const STAGE_ORDER = ["stitch", "process", "train", "export", "compress", "webopt"];
const STAGE_HUMAN: Record<string, string> = {
  stitch: "Unwrapping 360 footage",
  process: "Finding camera positions",
  train: "Building the 3D scene",
  export: "Finishing the scene",
  compress: "Compressing",
  webopt: "Preparing web viewer",
};
const STAGE_SHORT: Record<string, string> = {
  stitch: "Stitch",
  process: "Process",
  train: "Train",
  export: "Export",
  compress: "Compress",
  webopt: "Web",
};
const QUALITY = {
  draft: { label: "Draft", iterations: 7000, blurb: "~2 min" },
  standard: { label: "Standard", iterations: 30000, blurb: "~6 min" },
  high: { label: "High detail", iterations: 50000, blurb: "~10 min" },
} as const;
type QualityKey = keyof typeof QUALITY;

function humanizeStage(job: SplatJob): string {
  if (job.status === "starting") return "Getting started…";
  return job.stage ? STAGE_HUMAN[job.stage] || job.stage : "Working…";
}
function relTime(value: string | null): string {
  if (!value) return "—";
  const d = new Date(value).getTime();
  const s = Math.max(0, (Date.now() - d) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

// ── page ──────────────────────────────────────────────────────────────────────
export default function SplatLabPage() {
  const qc = useQueryClient();
  const [uploaded, setUploaded] = useState<SplatUploadResult | null>(null);
  const [preset, setPreset] = useState<QualityKey>("standard");
  const [toast, setToast] = useState<{ msg: string; bad?: boolean } | null>(null);

  const { data: status } = useQuery({
    queryKey: ["status"],
    queryFn: () => apiRequest<SplatStatusResponse>("/api/splat/status"),
    refetchInterval: 2500,
  });
  const { data: transfers, refetch: refetchTransfers, isFetching: transfersFetching } = useQuery({
    queryKey: ["transfers"],
    queryFn: () => apiRequest<SplatTransfersResponse>("/api/splat/transfers"),
    refetchInterval: 15000,
  });

  const jobs = status?.jobs ?? [];
  const activeJob = jobs.find((j) => j.status === "running" || j.status === "starting") || null;
  const completed = jobs.filter((j) => j.status === "completed");
  const gpu = status?.gpu;
  const engineReady = Boolean(status?.engines?.ns_train_available && status?.engines?.colmap_available);

  function flash(msg: string, bad = false) {
    setToast({ msg, bad });
    window.setTimeout(() => setToast(null), 4000);
  }

  const startMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      apiRequest<SplatJob>("/api/splat/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      setUploaded(null);
      qc.invalidateQueries({ queryKey: ["status"] });
      flash("Scene building started.");
    },
    onError: (e) => flash(e instanceof Error ? e.message : "Could not start", true),
  });

  const stopMutation = useMutation({
    mutationFn: (id: string) => apiRequest(`/api/splat/jobs/${id}/stop`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });

  function createFrom(input: SplatUploadResult) {
    startMutation.mutate({
      mode: "3d",
      input_path: input.path,
      output_dir: "outputs/3d",
      capture_format: input.is_insv ? "equirectangular360" : "standard",
      images_per_equirect: input.is_insv ? 8 : undefined,
      crop_bottom: input.is_insv ? 0.15 : undefined,
      num_frames_target: input.is_insv ? 75 : 300,
      max_num_iterations: QUALITY[preset].iterations,
      insv_fov: input.is_insv ? 204 : undefined,
    });
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      {/* hero */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="rounded-[22px] border border-cyan-400/30 bg-cyan-400/10 p-3.5 text-cyan-200">
            <Orbit className="h-7 w-7" />
          </div>
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.32em] text-cyan-200/80">Spatial Pipeline</p>
            <h1 className="display text-3xl font-black tracking-tight text-white">Splat Lab</h1>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold ${
              engineReady
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}
          >
            <Cpu className="h-3.5 w-3.5" /> Engine {engineReady ? "ready" : "warming"}
          </span>
          <Badge className="border-cyan-500/30 bg-cyan-500/10 text-cyan-200">
            {status?.active_jobs ?? 0} active
          </Badge>
        </div>
      </header>

      {toast && (
        <div
          className={`mb-4 rounded-xl border px-4 py-2.5 text-sm ${
            toast.bad ? "border-red-500/30 bg-red-500/10 text-red-200" : "border-cyan-500/30 bg-cyan-500/10 text-cyan-100"
          }`}
        >
          {toast.msg}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(380px,0.9fr)]">
        {/* left: create */}
        <Card className="p-5">
          <h2 className="text-base font-semibold text-white">Make a 3D scene</h2>
          <p className="mb-4 mt-0.5 text-sm text-zinc-400">A video, a 360 clip, or a zip of photos.</p>
          <UploadBox onUploaded={setUploaded} onError={(m) => flash(m, true)} current={uploaded} />
          <TransfersPicker
            entries={transfers?.entries ?? []}
            selectedPath={uploaded?.path ?? null}
            onSelect={(e) =>
              setUploaded({
                path: e.path,
                name: e.name,
                kind: e.kind === "images" || e.kind === "dataset" ? "directory" : "file",
                is_insv: e.is_insv,
                detail: `From Transfers · ${e.detail}`,
              })
            }
            onRefresh={() => refetchTransfers()}
            refreshing={transfersFetching}
          />
          <div className="mt-5 space-y-2">
            <SectionLabel>Quality</SectionLabel>
            <div className="grid grid-cols-3 gap-2">
              {(Object.keys(QUALITY) as QualityKey[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setPreset(k)}
                  className={`rounded-2xl border p-3 text-left transition-all ${
                    preset === k
                      ? "border-cyan-400/40 bg-cyan-400/10"
                      : "border-white/10 bg-white/[0.02] hover:border-cyan-500/20"
                  }`}
                >
                  <p className="text-sm font-semibold text-white">{QUALITY[k].label}</p>
                  <p className="mt-0.5 text-xs text-zinc-400">{QUALITY[k].blurb}</p>
                </button>
              ))}
            </div>
          </div>
          <Button
            size="lg"
            className="mt-5 w-full"
            disabled={!uploaded || startMutation.isPending || !!activeJob}
            onClick={() => uploaded && createFrom(uploaded)}
          >
            {startMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
            {activeJob ? "A scene is already building…" : "Create 3D Scene"}
          </Button>
          {uploaded?.is_insv && (
            <p className="mt-2 text-center text-xs text-cyan-200/80">360 footage detected — auto-unwrapped.</p>
          )}
        </Card>

        {/* right: live status / viewer */}
        <div className="space-y-4">
          {gpu?.locked && gpu.lane && gpu.lane !== "splat" && (
            <div className="flex items-center gap-2 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
              <Loader2 className="h-4 w-4 animate-spin" />
              Waiting for the RTX 5090 — held by <span className="font-semibold">{gpu.lane}</span>. Your run is queued.
            </div>
          )}
          {activeJob ? (
            <ActiveJobPanel job={activeJob} onStop={() => stopMutation.mutate(activeJob.job_id)} stopping={stopMutation.isPending} />
          ) : (
            <Card className="flex h-[200px] flex-col items-center justify-center p-6 text-center">
              <Box className="mb-2 h-7 w-7 text-zinc-600" />
              <p className="text-sm text-zinc-400">Your scene will build here. Pick a file and press Create.</p>
            </Card>
          )}
        </div>
      </div>

      {/* gallery */}
      <ResultsGallery jobs={completed} />
    </div>
  );
}

// ── upload ────────────────────────────────────────────────────────────────────
function UploadBox({
  onUploaded,
  onError,
  current,
}: {
  onUploaded: (r: SplatUploadResult) => void;
  onError: (m: string) => void;
  current: SplatUploadResult | null;
}) {
  const [pct, setPct] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function upload(file: File) {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/splat/upload");
    xhr.upload.onprogress = (e) => e.lengthComputable && setPct(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => {
      setPct(null);
      if (xhr.status >= 200 && xhr.status < 300) onUploaded(JSON.parse(xhr.responseText));
      else onError(`Upload failed (${xhr.status}). Files >100 MB? Use Transfers below.`);
    };
    xhr.onerror = () => {
      setPct(null);
      onError("Upload failed. For large captures, drop into ~/transfers below.");
    };
    setPct(0);
    xhr.send(form);
  }

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
      }}
      onClick={() => inputRef.current?.click()}
      className="cursor-pointer rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-6 text-center transition-colors hover:border-cyan-400/40"
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.insv,.zip"
        className="hidden"
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      {pct !== null ? (
        <div className="space-y-2">
          <Loader2 className="mx-auto h-6 w-6 animate-spin text-cyan-300" />
          <div className="mx-auto h-1.5 w-2/3 overflow-hidden rounded-full bg-white/10">
            <div className="h-full bg-cyan-400 transition-all" style={{ width: `${pct}%` }} />
          </div>
          <p className="text-xs text-zinc-400">Uploading… {pct}%</p>
        </div>
      ) : current && !current.detail.startsWith("From Transfers") ? (
        <div className="flex items-center justify-center gap-2 text-sm text-emerald-300">
          <CheckCircle2 className="h-5 w-5" /> {current.name}
        </div>
      ) : (
        <div className="space-y-1">
          <UploadCloud className="mx-auto h-7 w-7 text-zinc-500" />
          <p className="text-sm font-medium text-zinc-200">Drop a file or click to browse</p>
          <p className="text-xs text-zinc-500">Video · 360 .insv · .zip of photos · up to ~100 MB over the web</p>
        </div>
      )}
    </div>
  );
}

// ── transfers ─────────────────────────────────────────────────────────────────
function TransfersPicker({
  entries,
  selectedPath,
  onSelect,
  onRefresh,
  refreshing,
}: {
  entries: SplatTransferEntry[];
  selectedPath: string | null;
  onSelect: (e: SplatTransferEntry) => void;
  onRefresh: () => void;
  refreshing: boolean;
}) {
  return (
    <div className="mt-4 space-y-2">
      <div className="flex items-center justify-between">
        <SectionLabel>Or pick from Transfers</SectionLabel>
        <button
          onClick={onRefresh}
          className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200"
          title="Refresh"
        >
          <RefreshCw className={`h-3 w-3 ${refreshing ? "animate-spin" : ""}`} /> no size limit
        </button>
      </div>
      <p className="text-xs text-zinc-500">
        Sync a capture into <code className="rounded bg-white/10 px-1">~/transfers</code> (Syncthing /
        pulse-share) — it skips the 100&nbsp;MB upload cap.
      </p>
      {entries.length > 0 ? (
        <div className="max-h-48 space-y-1.5 overflow-y-auto pr-1">
          {entries.map((e) => {
            const sel = selectedPath === e.path;
            return (
              <button
                key={e.path}
                onClick={() => onSelect(e)}
                className={`flex w-full items-center gap-3 rounded-xl border p-2.5 text-left transition-all ${
                  sel ? "border-cyan-400/40 bg-cyan-400/10" : "border-white/10 bg-white/[0.02] hover:border-cyan-500/20"
                }`}
              >
                <FolderOpen className={`h-4 w-4 shrink-0 ${sel ? "text-cyan-200" : "text-zinc-500"}`} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-zinc-100">{e.name}</span>
                  <span className="block truncate text-xs text-zinc-500">{e.detail}</span>
                </span>
                <Badge>{e.kind}</Badge>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-white/10 px-3 py-4 text-center text-xs text-zinc-500">
          Nothing splat-ready in Transfers yet. Drop a video, a 360 .insv, a .zip of photos, or a folder of JPGs.
        </div>
      )}
    </div>
  );
}

// ── active job: stage rail + humanized stage + log ────────────────────────────
function StageRail({ job }: { job: SplatJob }) {
  const planned = job.stages_planned?.length ? job.stages_planned : STAGE_ORDER;
  const done = new Set(job.stages_completed ?? []);
  return (
    <div className="flex items-center gap-1.5">
      {planned.map((s) => {
        const isDone = done.has(s);
        const isCurrent = job.stage === s && !isDone;
        return (
          <div key={s} className="flex flex-1 flex-col items-center gap-1">
            <div
              className={`h-1.5 w-full rounded-full ${
                isDone ? "bg-emerald-400" : isCurrent ? "bg-cyan-400 nx-breath" : "bg-white/10"
              }`}
            />
            <span
              className={`text-[9px] font-semibold uppercase tracking-wide ${
                isDone ? "text-emerald-300/80" : isCurrent ? "text-cyan-200" : "text-zinc-600"
              }`}
            >
              {STAGE_SHORT[s] || s}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function ActiveJobPanel({ job, onStop, stopping }: { job: SplatJob; onStop: () => void; stopping: boolean }) {
  const logRef = useRef<HTMLPreElement | null>(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job.log_lines]);
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-3">
        <Loader2 className="h-5 w-5 animate-spin text-cyan-300" />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-white">{humanizeStage(job)}</p>
          <p className="truncate font-mono text-[11px] text-zinc-500">
            {job.stage || "starting"} · {job.input_path.split("/").pop()}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={onStop} disabled={stopping}>
          <Square className="h-3.5 w-3.5" /> Cancel
        </Button>
      </div>
      <StageRail job={job} />
      <pre
        ref={logRef}
        className="mt-4 max-h-44 overflow-y-auto whitespace-pre-wrap rounded-xl border border-white/10 bg-black/40 p-3 font-mono text-[11px] leading-relaxed text-zinc-400"
      >
        {(job.log_lines ?? []).slice(-80).join("\n") || "Starting…"}
      </pre>
      <p className="mt-2 text-xs text-zinc-500">You can leave this page — it keeps running on the GPU.</p>
    </Card>
  );
}

// ── results gallery ───────────────────────────────────────────────────────────
function ResultsGallery({ jobs }: { jobs: SplatJob[] }) {
  const previewable = jobs.filter((j) => j.preview_available);
  const [featured, setFeatured] = useState<string | null>(null);
  const featuredJob = previewable.find((j) => j.job_id === featured) || previewable[0] || null;

  if (jobs.length === 0) return null;

  return (
    <section className="mt-8">
      <div className="mb-3 flex items-center justify-between">
        <SectionLabel>Your scenes</SectionLabel>
        <Badge>{jobs.length} completed</Badge>
      </div>

      {featuredJob?.preview_web_url && (
        <Card className="mb-4 overflow-hidden">
          <SplatViewer url={featuredJob.preview_web_url} format="ply" />
          <div className="flex items-center justify-between gap-2 p-3">
            <p className="truncate text-sm text-zinc-300">{featuredJob.input_path.split("/").pop()}</p>
            <div className="flex items-center gap-2">
              <a href={`/view/${featuredJob.job_id}`} target="_blank" rel="noreferrer">
                <Button size="sm">
                  <Orbit className="h-3.5 w-3.5" /> Fullscreen
                </Button>
              </a>
              <DownloadMenu job={featuredJob} />
            </div>
          </div>
        </Card>
      )}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {jobs.map((j) => (
          <SceneCard key={j.job_id} job={j} active={j.job_id === featuredJob?.job_id} onFeature={() => setFeatured(j.job_id)} />
        ))}
      </div>
    </section>
  );
}

function SceneCard({ job, active, onFeature }: { job: SplatJob; active: boolean; onFeature: () => void }) {
  return (
    <Card className={`p-3 transition-colors ${active ? "border-cyan-400/40" : "hover:border-white/20"}`}>
      <button onClick={onFeature} className="block w-full text-left">
        <div className="mb-2 flex aspect-video items-center justify-center rounded-xl border border-white/10 bg-gradient-to-br from-cyan-500/5 to-orange-500/5">
          <Orbit className="h-7 w-7 text-cyan-300/50" />
        </div>
        <p className="truncate text-sm font-medium text-zinc-100">{job.input_path.split("/").pop()}</p>
        <div className="mt-1 flex items-center gap-2 text-[11px] text-zinc-500">
          <span>{relTime(job.completed_at)}</span>
          {job.capture_format === "equirectangular360" && <Badge>360</Badge>}
          {job.pinned && <Pin className="h-3 w-3 text-cyan-300" />}
        </div>
      </button>
      <div className="mt-2 flex items-center gap-2">
        <a href={`/view/${job.job_id}`} target="_blank" rel="noreferrer" className="flex-1">
          <Button size="sm" variant="outline" className="w-full">
            <Orbit className="h-3.5 w-3.5" /> Open
          </Button>
        </a>
        <DownloadMenu job={job} />
      </div>
    </Card>
  );
}

// ── download-format menu (Q5) ─────────────────────────────────────────────────
function DownloadMenu({ job }: { job: SplatJob }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  const opts = [
    { url: job.preview_web_url, label: "Web .ply", hint: "small · for sharing/viewing" },
    { url: job.preview_spz_url, label: "Compressed .spz", hint: "smallest · modern viewers" },
    { url: job.preview_file_url, label: "Full .ply", hint: "full quality · for editing" },
  ].filter((o) => o.url);

  return (
    <div ref={ref} className="relative">
      <Button size="sm" variant="outline" onClick={() => setOpen((v) => !v)}>
        <Download className="h-3.5 w-3.5" /> <ChevronDown className="h-3 w-3" />
      </Button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-56 overflow-hidden rounded-xl border border-white/10 bg-[#0a0f1a] shadow-2xl">
          {opts.map((o) => (
            <a
              key={o.label}
              href={o.url!}
              download={`${job.job_id}.${o.label.includes("spz") ? "spz" : "ply"}`}
              onClick={() => setOpen(false)}
              className="block px-3 py-2 hover:bg-white/5"
            >
              <p className="text-sm font-medium text-zinc-100">{o.label}</p>
              <p className="text-[11px] text-zinc-500">{o.hint}</p>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

// keep lint happy about imported icons reserved for near-term use
void AlertTriangle;
void Sparkles;
void X;
