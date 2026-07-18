import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { setSplatlabFeedbackContext } from "@/lib/feedback-context";
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
  Camera,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Download,
  FolderOpen,
  Loader2,
  MapPin,
  Orbit,
  Pin,
  RefreshCw,
  Rocket,
  Trash2,
  Sparkles,
  Square,
  UploadCloud,
  Wand2,
  X,
} from "lucide-react";

// ── pipeline metadata ────────────────────────────────────────────────────────
const STAGE_ORDER = ["stitch", "process", "train", "langfield", "export", "health", "compress", "webopt"];
const STAGE_HUMAN: Record<string, string> = {
  stitch: "Unwrapping 360 footage",
  process: "Finding camera positions",
  glomap_sfm: "Re-solving with global SfM",
  rig_sfm: "Solving 360 rig geometry",
  mast3r_sfm: "Re-solving with MASt3R (pose-free)",
  train: "Building the 3D scene",
  langfield: "Building the language field",
  export: "Finishing the scene",
  health: "Checking capture health",
  compress: "Compressing",
  webopt: "Preparing web viewer",
};
const STAGE_SHORT: Record<string, string> = {
  stitch: "Stitch",
  process: "Process",
  glomap_sfm: "Global SfM",
  rig_sfm: "360 rig",
  mast3r_sfm: "MASt3R",
  train: "Train",
  langfield: "Language field",
  export: "Export",
  health: "Health",
  compress: "Compress",
  webopt: "Web",
};
// An auto-fallback solver's process step is named "reprocess<n>" on the backend so
// it never collides with the original "process" stage key — label it like Process.
function stageShort(s: string): string {
  return STAGE_SHORT[s] || (s.startsWith("reprocess") ? "Process" : s);
}
function stageHuman(s: string): string {
  return STAGE_HUMAN[s] || (s.startsWith("reprocess") ? "Finding camera positions" : s);
}
const QUALITY = {
  draft: { label: "Draft", iterations: 7000, blurb: "~2 min" },
  standard: { label: "Standard", iterations: 30000, blurb: "~6 min" },
  high: { label: "High detail", iterations: 50000, blurb: "~10 min" },
} as const;
type QualityKey = keyof typeof QUALITY;

function humanizeStage(job: SplatJob): string {
  if (job.status === "starting") return "Getting started…";
  return job.stage ? stageHuman(job.stage) : "Working…";
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

const MIN_ITERS = 1000;
const MAX_ITERS = 50000;
// Rough training-time estimate (5090): ~1 min overhead + ~1 min / 5k iters.
function trainMinutes(iters: number): number {
  return Math.max(2, Math.round(1 + iters / 5000));
}
function presetForIters(iters: number): QualityKey | null {
  return (Object.keys(QUALITY) as QualityKey[]).find((k) => QUALITY[k].iterations === iters) ?? null;
}
function sceneHue(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) % 360;
  return h;
}

// Compact count: 1284773 -> "1.3M", 608501 -> "609k".
function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

// ── page ──────────────────────────────────────────────────────────────────────
export default function SplatLabPage() {
  const qc = useQueryClient();
  const [uploaded, setUploaded] = useState<SplatUploadResult | null>(null);
  const [iters, setIters] = useState<number>(QUALITY.standard.iterations);
  const [showCustom, setShowCustom] = useState(false);
  // Opt-in: build a text-searchable Language Field alongside the scene.
  const [languageField, setLanguageField] = useState(false);
  // Safe default: raw .insv evaluations start as a bounded Test Flight. A full
  // capture build now requires the operator to deliberately turn this off.
  const [testFlight, setTestFlight] = useState(true);
  // Opt-in: "Few Photos (AI poses)" — MASt3R dense-seed sparse-view mode (InstantSplat).
  const [sparseMode, setSparseMode] = useState(false);
  // Opt-in: "Imagine a Splat" — TripoSplat single-image generative lane.
  const [generativeMode, setGenerativeMode] = useState(false);
  const [toast, setToast] = useState<{ msg: string; bad?: boolean } | null>(null);
  // Dismissed failed-job notice (by job_id) so a newer failure still shows.
  const [dismissedFailed, setDismissedFailed] = useState<string | null>(null);

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
  // Newest failed job (with a guidance message) so a doomed capture's
  // "why it failed + what to do" surfaces in the Simple UI instead of
  // vanishing silently. Sorted by completed_at (fallback created_at) desc.
  const latestFailed = jobs
    .filter((j) => j.status === "failed" && !!j.error_message)
    .sort((a, b) =>
      (b.completed_at ?? b.created_at).localeCompare(a.completed_at ?? a.created_at),
    )[0] ?? null;
  const gpu = status?.gpu;
  const engineReady = Boolean(status?.engines?.ns_train_available && status?.engines?.colmap_available);
  // Whether the Language Field toolchain exists — gates the opt-in toggle.
  const langfieldEngineAvailable = Boolean(status?.engines?.langfield_available);
  // Whether the MASt3R toolchain exists — gates the "Few Photos (AI poses)" mode.
  const mast3rEngineAvailable = Boolean(status?.engines?.mast3r_available);
  // Whether the TripoSplat toolchain exists — gates the "Imagine a Splat" mode.
  const triposplatEngineAvailable = Boolean(status?.engines?.triposplat_available);
  const compute = status?.compute;
  const computeBlocked = Boolean(compute && !compute.enabled);
  const computeArmed = Boolean(compute?.enabled && compute.mode === "supervised" && compute.supervised_unlock?.active);
  const computeControlsDisabled = !status || computeBlocked;
  const computeReason = !status ? "Waiting for SplatLab status." : compute?.reason ?? "SplatLab compute is disabled.";

  useEffect(() => {
    setSplatlabFeedbackContext({
      page: "splatlab-gallery",
      active_job_id: activeJob?.job_id ?? null,
      active_job_status: activeJob?.status ?? null,
      active_job_stage: activeJob?.stage ?? null,
      completed_count: completed.length,
      failed_count: jobs.filter((j) => j.status === "failed").length,
      selected_upload: uploaded
        ? {
            name: uploaded.name,
            kind: uploaded.kind,
            is_insv: uploaded.is_insv,
          }
        : null,
      settings: {
        iterations: iters,
        language_field: languageField,
        sparse_mode: sparseMode,
        generative_mode: generativeMode,
        test_flight: testFlight,
      },
      gpu_lane: gpu?.lane ?? null,
      gpu_locked: Boolean(gpu?.locked),
      compute_enabled: compute?.enabled ?? null,
      compute_reason: compute?.reason ?? null,
    });
    return () => setSplatlabFeedbackContext(null);
  }, [activeJob, completed.length, compute?.enabled, compute?.reason, generativeMode, gpu?.lane, gpu?.locked, iters, jobs, languageField, sparseMode, testFlight, uploaded]);

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

  const createDisabled = computeControlsDisabled || !uploaded || startMutation.isPending || !!activeJob || !engineReady;

  const stopMutation = useMutation({
    mutationFn: (id: string) => apiRequest(`/api/splat/jobs/${id}/stop`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });

  const pinMutation = useMutation({
    mutationFn: (job: SplatJob) =>
      apiRequest(`/api/splat/jobs/${job.job_id}/${job.pinned ? "unpin" : "pin"}`, { method: "POST" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }),
    onError: (e) => flash(e instanceof Error ? e.message : "Pin failed", true),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiRequest(`/api/splat/jobs/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["status"] });
      flash("Scene deleted.");
    },
    onError: (e) => flash(e instanceof Error ? e.message : "Delete failed", true),
  });

  function createFrom(input: SplatUploadResult) {
    // Test Flight (insv only): centered 30s trim window at the PROVEN ~3fps
    // equirect frame density (90 frames; 1.76fps triangulated ZERO points on
    // splat_9da9dff4b2, 3fps gave 99.8% reg + 105k points on the same window)
    // + glomap-first (skips the COLMAP rung that burned 84 min on 07-04) +
    // draft iterations + no language field. ~20 min proof of capture/settings.
    const flight = testFlight && input.is_insv;
    startMutation.mutate({
      mode: "3d",
      input_path: input.path,
      output_dir: "outputs/3d",
      capture_format: input.is_insv ? "equirectangular360" : "standard",
      images_per_equirect: input.is_insv ? 8 : undefined,
      crop_bottom: input.is_insv ? 0.15 : undefined,
      num_frames_target: flight ? 90 : input.is_insv ? 75 : 300,
      max_num_iterations: flight ? Math.min(iters, QUALITY.draft.iterations) : iters,
      insv_fov: input.is_insv ? 204 : undefined,
      trim_duration_s: flight ? 30 : undefined,
      // glomap-first only when the engine is actually there — otherwise let the
      // server default (colmap + auto-escalation) run; the trim still bounds cost.
      // insv: omit the backend so the server's default-flip resolves to the rig
      // lane (the old glomap-first flight override would BYPASS it). Only force
      // glomap when the rig toolchain is missing on this host.
      sfm_backend: flight && glomapAvailable && !rigAvailable ? "glomap" : undefined,
      language_field: flight ? false : languageField,
      // "Few Photos (AI poses)": dense-seed MASt3R sparse-view path (no COLMAP).
      capture_mode: sparseMode && !input.is_insv ? "sparse" : undefined,
      // "Imagine a Splat": single-image generative lane (skips SfM/train entirely).
      source_type: generativeMode && !input.is_insv ? "generative-image" : undefined,
    });
  }

  // Re-run a finished/failed scene with its own params (optionally at higher quality).
  // Forwards the scene's OWN persisted num_frames_target/sfm_backend rather than
  // leaving them unset — an unset sfm_backend falls back to the request default
  // ("colmap"), which would silently contradict a scene that was actually built
  // on glomap (the same "hybrid" trap Promote to full build exists to fix).
  function rerun(job: SplatJob, multiplier = 1) {
    if (activeJob) {
      flash("A scene is already building — wait for it to finish.", true);
      return;
    }
    const base = job.max_num_iterations || QUALITY.standard.iterations;
    startMutation.mutate({
      mode: "3d",
      input_path: job.input_path,
      output_dir: "outputs/3d",
      capture_format: job.capture_format,
      num_frames_target: job.num_frames_target,
      sfm_backend: job.sfm_backend,
      max_num_iterations: Math.min(MAX_ITERS, Math.round(base * multiplier)),
      language_field: Boolean(job.language_field),
    });
  }

  // Re-run a failed capture with COLMAP 4.x global SfM, which registers far more
  // frames on the low-overlap captures incremental COLMAP gives up on.
  function retryGlomap(job: SplatJob) {
    if (activeJob) {
      flash("A scene is already building — wait for it to finish.", true);
      return;
    }
    startMutation.mutate({
      mode: "3d",
      input_path: job.input_path,
      output_dir: "outputs/3d",
      capture_format: job.capture_format,
      num_frames_target: job.num_frames_target,
      max_num_iterations: job.max_num_iterations || QUALITY.standard.iterations,
      sfm_backend: "glomap",
      language_field: Boolean(job.language_field),
    });
  }

  // Promote a Test Flight (trimmed) proof scene to a full build: same insv
  // input with the trim dropped (full clip), num_frames_target reset to the
  // full-run default (the backend's duration-aware 3fps rule takes over from
  // here — see PLAN.md §1D′), the SfM rung the flight actually proved works
  // (persisted sfm_backend — glomap for every flight, so this never re-enters
  // the ~84min doomed COLMAP rung), and the CURRENTLY selected quality preset
  // instead of the flight's draft iteration cap.
  function promoteToFullBuild(job: SplatJob) {
    if (activeJob) {
      flash("A scene is already building — wait for it to finish.", true);
      return;
    }
    startMutation.mutate({
      mode: "3d",
      input_path: job.input_path,
      output_dir: "outputs/3d",
      capture_format: job.capture_format,
      images_per_equirect: job.capture_format === "equirectangular360" ? 8 : undefined,
      crop_bottom: job.capture_format === "equirectangular360" ? 0.15 : undefined,
      insv_fov: job.capture_format === "equirectangular360" ? 204 : undefined,
      num_frames_target: 300,
      max_num_iterations: iters,
      sfm_backend: job.sfm_backend ?? "glomap",
      language_field: languageField,
    });
  }
  const glomapAvailable = Boolean(status?.engines?.glomap_available);
  const rigAvailable = Boolean(status?.engines?.rig_available);

  return (
    <div className="mx-auto max-w-[1880px] px-4 py-8 sm:px-6 xl:px-10">
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
              !computeBlocked && engineReady
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}
          >
            <Cpu className="h-3.5 w-3.5" /> {computeBlocked ? "Safe browse mode" : computeArmed ? "Supervised compute" : `Engine ${engineReady ? "ready" : "warming"}`}
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

      <ComputeGateBanner compute={compute} />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(380px,0.9fr)] 2xl:grid-cols-[minmax(460px,640px)_minmax(0,1fr)]">
        {/* left: create */}
        <Card className="p-5">
          <h2 className="text-base font-semibold text-white">Make a 3D scene</h2>
          <p className="mb-4 mt-0.5 text-sm text-zinc-400">A video, a 360 clip, or a zip of photos.</p>
          <UploadBox onUploaded={setUploaded} onError={(m) => flash(m, true)} current={uploaded} disabled={computeControlsDisabled} disabledReason={computeReason} />
          <TransfersPicker
            entries={transfers?.entries ?? []}
            selectedPath={uploaded?.path ?? null}
            onSelect={(e) => {
              // A single image can only go through the generative lane; sync the toggle
              // to the picked input so "Imagine a Splat" turns on/off automatically.
              setGenerativeMode(e.kind === "image");
              setUploaded({
                path: e.path,
                name: e.name,
                kind: e.kind === "images" || e.kind === "dataset" ? "directory" : "file",
                is_insv: e.is_insv,
                detail: `From Transfers · ${e.detail}`,
              });
            }}
            onRefresh={() => refetchTransfers()}
            refreshing={transfersFetching}
            disabled={computeControlsDisabled}
          />
          <div className="mt-5 space-y-2">
            <div className="flex items-center justify-between">
              <SectionLabel>Quality</SectionLabel>
              <button onClick={() => setShowCustom((v) => !v)} className="text-[11px] text-zinc-400 hover:text-cyan-200">
                {showCustom ? "Hide custom" : "Customize"}
              </button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {(Object.keys(QUALITY) as QualityKey[]).map((k) => (
                <button
                  key={k}
                  onClick={() => setIters(QUALITY[k].iterations)}
                  className={`rounded-2xl border p-3 text-left transition-all ${
                    presetForIters(iters) === k
                      ? "border-cyan-400/40 bg-cyan-400/10"
                      : "border-white/10 bg-white/[0.02] hover:border-cyan-500/20"
                  }`}
                >
                  <p className="text-sm font-semibold text-white">{QUALITY[k].label}</p>
                  <p className="mt-0.5 text-xs text-zinc-400">{QUALITY[k].blurb}</p>
                </button>
              ))}
            </div>
            {showCustom && (
              <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-zinc-400">Training iterations</span>
                  <span className="font-mono font-semibold text-cyan-200">{iters.toLocaleString()}</span>
                </div>
                <input
                  type="range"
                  min={MIN_ITERS}
                  max={MAX_ITERS}
                  step={1000}
                  value={iters}
                  onChange={(e) => setIters(Number(e.target.value))}
                  className="mt-2 w-full accent-cyan-400"
                />
                <p className="mt-1 text-[11px] text-zinc-500">
                  More iterations = sharper detail, longer build · est. ~{trainMinutes(iters)} min.
                </p>
              </div>
            )}
          </div>

          {uploaded?.is_insv && (
            <button
              type="button"
              onClick={() => setTestFlight((v) => !v)}
              disabled={computeControlsDisabled}
              className={`mt-3 flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition-all ${
                testFlight
                  ? "border-emerald-400/40 bg-emerald-400/10"
                  : "border-white/10 bg-white/[0.02] hover:border-emerald-500/20"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors ${
                  testFlight ? "border-emerald-400 bg-emerald-400 text-[#04121a]" : "border-white/20 bg-white/5"
                }`}
              >
                {testFlight && <CheckCircle2 className="h-3.5 w-3.5" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-sm font-semibold text-white">
                  <Rocket className="h-3.5 w-3.5 text-emerald-200" /> Test flight (~20 min)
                </span>
                <span className="mt-0.5 block text-xs text-zinc-400">
                  Trims a short window from the middle of the clip and runs the whole pipeline on it —
                  prove the capture and settings before committing to the full multi-hour build.
                </span>
              </span>
            </button>
          )}

          {langfieldEngineAvailable && (
            <button
              type="button"
              onClick={() => setLanguageField((v) => !v)}
              disabled={computeControlsDisabled}
              className={`mt-3 flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition-all ${
                languageField
                  ? "border-cyan-400/40 bg-cyan-400/10"
                  : "border-white/10 bg-white/[0.02] hover:border-cyan-500/20"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors ${
                  languageField ? "border-cyan-400 bg-cyan-400 text-[#04121a]" : "border-white/20 bg-white/5"
                }`}
              >
                {languageField && <CheckCircle2 className="h-3.5 w-3.5" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-sm font-semibold text-white">
                  <Sparkles className="h-3.5 w-3.5 text-cyan-200" /> Language search (text-searchable)
                </span>
                <span className="mt-0.5 block text-xs text-zinc-400">
                  Build a language field so you can search the finished scene by typing what you’re looking for. Adds
                  some build time.
                </span>
              </span>
            </button>
          )}

          {mast3rEngineAvailable && (
            <button
              type="button"
              onClick={() => setSparseMode((v) => !v)}
              disabled={computeControlsDisabled}
              className={`mt-3 flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition-all ${
                sparseMode
                  ? "border-amber-400/40 bg-amber-400/10"
                  : "border-white/10 bg-white/[0.02] hover:border-amber-500/20"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors ${
                  sparseMode ? "border-amber-400 bg-amber-400 text-[#04121a]" : "border-white/20 bg-white/5"
                }`}
              >
                {sparseMode && <CheckCircle2 className="h-3.5 w-3.5" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-sm font-semibold text-white">
                  <Camera className="h-3.5 w-3.5 text-amber-200" /> Few Photos (AI poses)
                </span>
                <span className="mt-0.5 block text-xs text-zinc-400">
                  Only a handful of shots (2–12)? Skip COLMAP — MASt3R infers camera poses + a dense AI depth prior so
                  a few photos still make a splat. Geometry in unseen/untextured areas is AI-inferred, not measured.
                </span>
              </span>
            </button>
          )}

          {triposplatEngineAvailable && (
            <button
              type="button"
              onClick={() => setGenerativeMode((v) => !v)}
              disabled={computeControlsDisabled}
              className={`mt-3 flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition-all ${
                generativeMode
                  ? "border-fuchsia-400/40 bg-fuchsia-400/10"
                  : "border-white/10 bg-white/[0.02] hover:border-fuchsia-500/20"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors ${
                  generativeMode ? "border-fuchsia-400 bg-fuchsia-400 text-[#04121a]" : "border-white/20 bg-white/5"
                }`}
              >
                {generativeMode && <CheckCircle2 className="h-3.5 w-3.5" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-sm font-semibold text-white">
                  <Wand2 className="h-3.5 w-3.5 text-fuchsia-200" /> Imagine a Splat (1 image)
                </span>
                <span className="mt-0.5 block text-xs text-zinc-400">
                  Just ONE picture — TripoSplat generates a full 3D object in ~10s (no capture needed). The back and
                  sides you didn't photograph are invented by the model, not real.
                </span>
              </span>
            </button>
          )}

          {uploaded && (
            <div className="mt-4 rounded-2xl border border-cyan-500/20 bg-cyan-500/[0.06] p-3 text-xs">
              <div className="flex items-center gap-2 text-cyan-100">
                <CheckCircle2 className="h-4 w-4 shrink-0 text-cyan-300" />
                <span className="truncate font-medium">{uploaded.name}</span>
              </div>
              <div className="mt-1.5 space-y-0.5 text-zinc-400">
                <p>{uploaded.is_insv ? "360 footage — auto-unwrapped." : uploaded.detail}</p>
                <p>
                  Estimated build: <span className="text-zinc-200">~{trainMinutes(iters)} min</span> training
                  {uploaded.kind === "file" ? " + a few min to find camera positions" : ""}.
                </p>
              </div>
            </div>
          )}

          <Button
            size="lg"
            className="mt-4 w-full"
            disabled={createDisabled}
            onClick={() => uploaded && createFrom(uploaded)}
          >
            {startMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
            {computeBlocked
              ? "GPU compute blocked"
              : !!status && !engineReady
              ? "Engine warming up…"
              : activeJob
                ? "A scene is already building…"
                : uploaded
                  ? `Create 3D Scene · ~${trainMinutes(iters)} min`
                  : "Create 3D Scene"}
          </Button>
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
            <>
              {latestFailed && dismissedFailed !== latestFailed.job_id && (
                <Card className="relative border-amber-500/40 bg-amber-500/10 p-4">
                  <button
                    type="button"
                    aria-label="Dismiss"
                    onClick={() => setDismissedFailed(latestFailed.job_id)}
                    className="absolute right-3 top-3 text-amber-300/70 transition hover:text-amber-200"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <div className="flex items-start gap-3 pr-6">
                    <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
                    <div className="space-y-1">
                      <p className="text-sm font-semibold text-amber-100">
                        Scene couldn’t be built — {latestFailed.input_path.split("/").pop()}
                      </p>
                      <p className="text-sm text-amber-200/90">{latestFailed.error_message}</p>
                      {glomapAvailable && (
                        <div className="pt-1.5">
                          <Button size="sm" disabled={!!activeJob || computeControlsDisabled} onClick={() => retryGlomap(latestFailed)}>
                            <RefreshCw className="h-3.5 w-3.5" /> Retry with global SfM
                          </Button>
                          <p className="mt-1 text-[11px] text-amber-200/60">
                            Re-registers the same footage with a stronger solver — rescues most low-overlap captures.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              )}
              <Card className="flex h-[200px] flex-col items-center justify-center p-6 text-center">
                <Box className="mb-2 h-7 w-7 text-zinc-600" />
                <p className="text-sm text-zinc-400">Your scene will build here. Pick a file and press Create.</p>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* gallery */}
      <ResultsGallery
        jobs={completed}
        onRerun={rerun}
        onPromote={promoteToFullBuild}
        busy={!!activeJob}
        computeBlocked={computeControlsDisabled}
        onPin={(j) => pinMutation.mutate(j)}
        onDelete={(id) => deleteMutation.mutate(id)}
      />
    </div>
  );
}

function ComputeGateBanner({ compute }: { compute: SplatStatusResponse["compute"] }) {
  if (!compute) return null;
  if (compute.enabled && compute.mode === "supervised" && compute.supervised_unlock?.active) {
    const minutes = Math.max(0, Math.ceil((compute.supervised_unlock.seconds_remaining ?? 0) / 60));
    return (
      <Card className="mb-5 border-emerald-500/30 bg-emerald-500/10 p-4">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div className="flex min-w-0 gap-3">
            <Cpu className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />
            <div className="min-w-0">
              <p className="text-sm font-semibold text-emerald-100">Supervised compute is armed</p>
              <p className="mt-1 text-sm text-emerald-100/80">
                One SplatLab GPU job can run during this test window. Expires in about {minutes} minute{minutes === 1 ? "" : "s"}.
              </p>
              <p className="mt-1 truncate font-mono text-[11px] text-emerald-100/50">{compute.unlock_path || compute.supervised_unlock.path}</p>
            </div>
          </div>
          <div className="grid gap-2 text-xs text-emerald-100/80 sm:grid-cols-2 md:w-[520px]">
            <div className="rounded-xl border border-emerald-400/20 bg-black/20 p-2.5">
              <p className="mb-1 font-semibold text-emerald-100">Available now</p>
              <p>{compute.safe_capabilities.slice(0, 3).join(" · ")}</p>
            </div>
            <div className="rounded-xl border border-emerald-400/20 bg-black/20 p-2.5">
              <p className="mb-1 font-semibold text-emerald-100">Still held back</p>
              <p>{compute.blocked_capabilities.slice(0, 3).join(" · ")}</p>
            </div>
          </div>
        </div>
      </Card>
    );
  }
  if (compute.enabled) return null;
  return (
    <Card className="mb-5 border-amber-500/30 bg-amber-500/10 p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-amber-100">SplatLab is open in safe browse mode</p>
            <p className="mt-1 text-sm text-amber-100/80">
              GPU generation and scene mutations are blocked by the hardware-maintenance gate:{" "}
              {compute.reason || "hardware maintenance is active."}
            </p>
            <p className="mt-1 truncate font-mono text-[11px] text-amber-100/50">{compute.marker_path}</p>
          </div>
        </div>
        <div className="grid gap-2 text-xs text-amber-100/80 sm:grid-cols-2 md:w-[520px]">
          <div className="rounded-xl border border-amber-400/20 bg-black/20 p-2.5">
            <p className="mb-1 font-semibold text-amber-100">Available now</p>
            <p>{compute.safe_capabilities.slice(0, 3).join(" · ")}</p>
          </div>
          <div className="rounded-xl border border-amber-400/20 bg-black/20 p-2.5">
            <p className="mb-1 font-semibold text-amber-100">Paused</p>
            <p>{compute.blocked_capabilities.slice(0, 3).join(" · ")}</p>
          </div>
        </div>
      </div>
    </Card>
  );
}

// ── upload ────────────────────────────────────────────────────────────────────
function UploadBox({
  onUploaded,
  onError,
  current,
  disabled,
  disabledReason,
}: {
  onUploaded: (r: SplatUploadResult) => void;
  onError: (m: string) => void;
  current: SplatUploadResult | null;
  disabled: boolean;
  disabledReason: string;
}) {
  const [pct, setPct] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function upload(file: File) {
    if (disabled) {
      onError(disabledReason);
      return;
    }
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
        if (!disabled && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-6 text-center transition-colors ${
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-cyan-400/40"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.insv,.zip"
        className="hidden"
        disabled={disabled}
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      {disabled ? (
        <div className="space-y-1">
          <AlertTriangle className="mx-auto h-6 w-6 text-amber-300" />
          <p className="text-sm font-medium text-amber-100">New capture uploads blocked</p>
          <p className="text-xs text-amber-100/60">Existing scenes and downloads remain available.</p>
        </div>
      ) : pct !== null ? (
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
  disabled,
}: {
  entries: SplatTransferEntry[];
  selectedPath: string | null;
  onSelect: (e: SplatTransferEntry) => void;
  onRefresh: () => void;
  refreshing: boolean;
  disabled: boolean;
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
                disabled={disabled}
                className={`flex w-full items-center gap-3 rounded-xl border p-2.5 text-left transition-all ${
                  sel ? "border-cyan-400/40 bg-cyan-400/10" : "border-white/10 bg-white/[0.02] hover:border-cyan-500/20"
                } disabled:cursor-not-allowed disabled:opacity-50`}
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
              {stageShort(s)}
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

// ── capture health (report-only fog gate, Capture Coach) ──────────────────────
const HEALTH_STYLE = {
  FOG: { pill: "bg-amber-400/20 text-amber-200", label: "likely fog" },
  HEALTHY: { pill: "bg-emerald-400/20 text-emerald-200", label: "healthy" },
  UNCERTAIN: { pill: "bg-zinc-400/20 text-zinc-300", label: "unverified" },
} as const;

function HealthBadge({ job }: { job: SplatJob }) {
  const verdict = job.health?.fog?.verdict;
  if (!verdict) return null;
  const s = HEALTH_STYLE[verdict];
  return (
    <span
      title={
        verdict === "FOG"
          ? "Capture health (report-only): depth reads as a fog cocoon around the camera path — this reconstruction probably failed. See the health card under the featured viewer."
          : verdict === "HEALTHY"
            ? "Capture health (report-only): depth shows real scene structure at the probed cameras."
            : "Capture health (report-only): mixed signals at the probed cameras — judge the receipts yourself."
      }
      className={`flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold backdrop-blur-sm ${s.pill}`}
    >
      {verdict === "FOG" ? <AlertTriangle className="h-3 w-3" /> : verdict === "HEALTHY" ? <CheckCircle2 className="h-3 w-3" /> : null}
      {s.label}
    </span>
  );
}

function CaptureHealthCard({ job }: { job: SplatJob }) {
  const fog = job.health?.fog;
  if (!fog) return null;
  const s = HEALTH_STYLE[fog.verdict];
  const m = fog.summary;
  return (
    <Card className="mb-4 p-4">
      <div className="flex items-start gap-3">
        {fog.verdict === "FOG" ? (
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-300" />
        ) : fog.verdict === "HEALTHY" ? (
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-300" />
        ) : (
          <Box className="mt-0.5 h-5 w-5 shrink-0 text-zinc-400" />
        )}
        <div className="min-w-0 flex-1 space-y-1.5">
          <p className="text-sm font-semibold text-zinc-100">
            Capture health: <span className={`rounded px-1.5 py-0.5 ${s.pill}`}>{s.label}</span>
            <span className="ml-2 text-[11px] font-normal text-zinc-500">report-only — never blocks or changes your scene</span>
          </p>
          <p className="text-sm text-zinc-400">
            {fog.verdict === "FOG"
              ? "This reconstruction likely failed — the depth renders below should show scene structure, not a uniform haze. Reshoot guidance: walk overlapping parallel passes through the space (not an orbit around yourself), slow down, and vary camera height."
              : fog.verdict === "HEALTHY"
                ? "Depth shows real scene structure at the probed training cameras."
                : "Mixed signals — some cameras see structure, some sit inside a fog cocoon. Judge the receipts below yourself."}
          </p>
          <p className="text-[11px] tabular-nums text-zinc-500">
            {fog.summary && (
              <>
                {m.n_fog}/{m.n_counted} cameras read as fog · median shell {m.median_shell_frac ?? "—"} · spread{" "}
                {m.median_spread ?? "—"} · depth p50 {m.median_p50 ?? "—"}
              </>
            )}
          </p>
          {fog.receipts?.length ? (
            <div className="flex gap-2 overflow-x-auto pt-1">
              {fog.receipts.map((name) => (
                <img
                  key={name}
                  src={`/api/splat/jobs/${job.job_id}/health/receipt/${name}`}
                  alt={name}
                  title={`${name} — left: what the scene renders, right: its depth (turbo). Fog = mushy smear next to a flat blob.`}
                  loading="lazy"
                  className="h-24 shrink-0 rounded-lg border border-white/10"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = "none";
                  }}
                />
              ))}
            </div>
          ) : null}
        </div>
      </div>
    </Card>
  );
}

// ── results gallery ───────────────────────────────────────────────────────────
function ResultsGallery({
  jobs,
  onRerun,
  onPromote,
  busy,
  computeBlocked,
  onPin,
  onDelete,
}: {
  jobs: SplatJob[];
  onRerun: (job: SplatJob, mult?: number) => void;
  onPromote: (job: SplatJob) => void;
  busy: boolean;
  computeBlocked: boolean;
  onPin: (job: SplatJob) => void;
  onDelete: (id: string) => void;
}) {
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
      {featuredJob && <CaptureHealthCard job={featuredJob} />}

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
        {jobs.map((j) => (
          <SceneCard
            key={j.job_id}
            job={j}
            active={j.job_id === featuredJob?.job_id}
            onFeature={() => setFeatured(j.job_id)}
            onRerun={onRerun}
            onPromote={onPromote}
            busy={busy}
            computeBlocked={computeBlocked}
            onPin={onPin}
            onDelete={onDelete}
          />
        ))}
      </div>
    </section>
  );
}

function SceneCard({
  job,
  active,
  onFeature,
  onRerun,
  onPromote,
  busy,
  computeBlocked,
  onPin,
  onDelete,
}: {
  job: SplatJob;
  active: boolean;
  onFeature: () => void;
  onRerun: (job: SplatJob, mult?: number) => void;
  onPromote: (job: SplatJob) => void;
  busy: boolean;
  computeBlocked: boolean;
  onPin: (job: SplatJob) => void;
  onDelete: (id: string) => void;
}) {
  const [confirmDel, setConfirmDel] = useState(false);
  useEffect(() => {
    if (!confirmDel) return;
    const t = window.setTimeout(() => setConfirmDel(false), 3000);
    return () => window.clearTimeout(t);
  }, [confirmDel]);

  return (
    <Card className={`group relative p-3 transition-colors ${active ? "border-cyan-400/40" : "hover:border-white/20"}`}>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onPin(job);
        }}
        disabled={computeBlocked}
        className={`absolute right-4 top-4 z-10 rounded-md bg-black/40 p-1 transition-opacity ${
          job.pinned ? "text-cyan-300 opacity-100" : "text-zinc-400 opacity-0 hover:text-zinc-100 group-hover:opacity-100"
        } disabled:cursor-not-allowed disabled:opacity-40`}
        title={job.pinned ? "Unpin" : "Pin (protect from auto-cleanup)"}
      >
        <Pin className={`h-3.5 w-3.5 ${job.pinned ? "fill-current" : ""}`} />
      </button>

      <button onClick={onFeature} className="block w-full text-left">
        <div
          className="relative mb-2 flex aspect-video items-center justify-center overflow-hidden rounded-xl border border-white/10"
          style={{
            // per-scene tint shows through while the thumbnail loads / if missing
            background: `linear-gradient(135deg, hsl(${sceneHue(job.job_id)} 55% 13%), hsl(${(sceneHue(job.job_id) + 45) % 360} 50% 8%))`,
          }}
        >
          <Orbit className="absolute h-7 w-7" style={{ color: `hsl(${sceneHue(job.job_id)} 70% 62% / 0.55)` }} />
          {job.preview_available && (
            <img
              src={`/api/splat/jobs/${job.job_id}/thumbnail`}
              alt=""
              loading="lazy"
              className="relative h-full w-full object-cover"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
            />
          )}
          {job.stats?.gaussians ? (
            <span className="absolute bottom-1 left-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-cyan-100 backdrop-blur-sm">
              {fmtCount(job.stats.gaussians)} splats
            </span>
          ) : null}
          <span className="absolute bottom-1 right-1 flex items-center gap-1">
            {job.geo ? (
              <span
                title={`Located at ${job.geo.lat.toFixed(5)}, ${job.geo.lon.toFixed(5)} · heading ${job.geo.heading_deg.toFixed(0)}° — open the scene to see it on the map`}
                className="flex items-center gap-0.5 rounded bg-emerald-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-200 backdrop-blur-sm"
              >
                <MapPin className="h-3 w-3" /> located
              </span>
            ) : null}
            <HealthBadge job={job} />
            {job.langfield_available ? (
              <span className="rounded bg-cyan-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-cyan-200 backdrop-blur-sm">
                searchable
              </span>
            ) : (
              (() => {
                const lfFail = job.stages_failed?.find((f) => f.stage === "langfield");
                return lfFail ? (
                  <span
                    title={`Language field build failed — scene isn't text-searchable: ${lfFail.reason}`}
                    className="flex items-center gap-1 rounded bg-amber-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-200 backdrop-blur-sm"
                  >
                    <AlertTriangle className="h-3 w-3" /> field failed
                  </span>
                ) : null;
              })()
            )}
          </span>
          {job.source_type === "generative-image" ? (
            <span
              title="Generated from a single image — the back and sides are invented by the model, not photographed."
              className="absolute left-1 top-1 flex items-center gap-1 rounded bg-fuchsia-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-fuchsia-200 backdrop-blur-sm"
            >
              <Wand2 className="h-3 w-3" /> Generated
            </span>
          ) : job.capture_mode === "sparse" ? (
            <span
              title="Built from a few photos — camera poses + some geometry are AI-inferred, not measured."
              className="absolute left-1 top-1 flex items-center gap-1 rounded bg-amber-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-200 backdrop-blur-sm"
            >
              <Camera className="h-3 w-3" /> AI poses
            </span>
          ) : null}
        </div>
        <p className="truncate text-sm font-medium text-zinc-100">{job.input_path.split("/").pop()}</p>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-zinc-500">
          <span>{relTime(job.completed_at)}</span>
          {job.capture_format === "equirectangular360" && <Badge>360</Badge>}
          {job.max_num_iterations ? <span>{(job.max_num_iterations / 1000).toFixed(0)}k iters</span> : null}
          {job.stats?.width && job.stats?.height ? <span>{job.stats.width}×{job.stats.height}</span> : null}
          {job.stats?.images ? <span>{job.stats.images} imgs</span> : null}
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
      <div className="mt-1.5 flex items-center gap-2">
        {job.trim_duration_s != null ? (
          // A Test Flight proof scene: Re-run/↑Quality would silently drop the
          // trim and inherit the flight's draft settings (the F2 "hybrid" trap —
          // full clip + draft iters + whatever sfm_backend the request defaults
          // to). Promote to full build is the one correct next action here.
          <Button
            size="sm"
            variant="ghost"
            className="flex-1 border border-emerald-400/30 text-xs text-emerald-200 hover:bg-emerald-400/10"
            disabled={busy || computeBlocked}
            onClick={() => onPromote(job)}
            title="Build the full clip at full quality, using the settings this proof validated"
          >
            <Rocket className="h-3.5 w-3.5" /> Promote to full build
          </Button>
        ) : (
          <>
            <Button size="sm" variant="ghost" className="flex-1 text-xs" disabled={busy || computeBlocked} onClick={() => onRerun(job)} title="Re-run with the same settings">
              <RefreshCw className="h-3.5 w-3.5" /> Re-run
            </Button>
            <Button size="sm" variant="ghost" className="flex-1 text-xs" disabled={busy || computeBlocked} onClick={() => onRerun(job, 2)} title="Re-run at ~2x iterations">
              ↑ Quality
            </Button>
          </>
        )}
        <Button
          size="sm"
          variant="ghost"
          className={`text-xs ${confirmDel ? "text-red-300" : "text-zinc-500 hover:text-red-300"}`}
          onClick={() => (confirmDel ? onDelete(job.job_id) : setConfirmDel(true))}
          disabled={computeBlocked}
          title="Delete scene"
        >
          <Trash2 className="h-3.5 w-3.5" /> {confirmDel ? "Sure?" : ""}
        </Button>
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
