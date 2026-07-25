import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { setSplatlabFeedbackContext } from "@/lib/feedback-context";
import { ComputeGateBanner } from "@/components/compute-gate-banner";
import { UploadBox } from "@/components/create/upload-box";
import { TransfersPicker } from "@/components/create/transfers-picker";
import { ActiveJobPanel } from "@/components/jobs/active-job-panel";
import { RerouteChips } from "@/components/jobs/reroute-chips";
import { ResultsGallery } from "@/components/gallery/results-gallery";
import type {
  SplatJob,
  SplatPrecheckResult,
  SplatStatusResponse,
  SplatTransfersResponse,
  SplatUploadResult,
} from "@/lib/contracts";
import { Badge, Button, Card, SectionLabel, useToast } from "@/components/ui";
import {
  MAX_ITERS,
  MIN_ITERS,
  QUALITY,
  presetForIters,
  trainMinutes,
} from "@/lib/stage-meta";
import type { QualityKey } from "@/lib/stage-meta";
import {
  AlertTriangle,
  Box,
  Camera,
  CheckCircle2,
  Cpu,
  Loader2,
  Orbit,
  RefreshCw,
  Rocket,
  Sparkles,
  Wand2,
  X,
} from "lucide-react";

// ── page ──────────────────────────────────────────────────────────────────────
export default function SplatLabPage() {
  const qc = useQueryClient();
  const [uploaded, setUploaded] = useState<SplatUploadResult | null>(null);
  // Tier-0 capture screen (Capture Coach Phase 2): advisory strings for the
  // freshly picked input. Fire-and-forget; failures just clear the panel —
  // Create is NEVER disabled by this.
  const [precheck, setPrecheck] = useState<SplatPrecheckResult | null>(null);
  useEffect(() => {
    setPrecheck(null);
    if (!uploaded) return;
    let cancelled = false;
    apiRequest<SplatPrecheckResult>("/api/splat/precheck", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input_path: uploaded.path }),
    })
      .then((result) => {
        if (!cancelled) setPrecheck(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [uploaded]);
  const [iters, setIters] = useState<number>(QUALITY.standard.iterations);
  const [showCustom, setShowCustom] = useState(false);
  // Opt-in: build a text-searchable Language Field alongside the scene.
  const [languageField, setLanguageField] = useState(false);
  // Opt-in: export a triangle mesh (Digital Twin kernel — Blender/CAD/print).
  const [meshExport, setMeshExport] = useState(false);
  // Safe default: raw .insv evaluations start as a bounded Test Flight. A full
  // capture build now requires the operator to deliberately turn this off.
  const [testFlight, setTestFlight] = useState(true);
  // Opt-in: "Few Photos (AI poses)" — MASt3R dense-seed sparse-view mode (InstantSplat).
  const [sparseMode, setSparseMode] = useState(false);
  // Opt-in: "Imagine a Splat" — TripoSplat single-image generative lane.
  const [generativeMode, setGenerativeMode] = useState(false);
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
  const meshEngineAvailable = Boolean(status?.engines?.mesh_available);
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
        mesh_export: meshExport,
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
  }, [activeJob, completed.length, compute?.enabled, compute?.reason, generativeMode, gpu?.lane, gpu?.locked, iters, jobs, languageField, meshExport, sparseMode, testFlight, uploaded]);

  const pushToast = useToast();
  function flash(msg: string, bad = false) {
    pushToast(msg, bad ? "error" : "info");
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
      // insv full runs OMIT the target — the server's §1D′ duration-aware
      // 3 fps rule computes the real value (the old literal 75 was a
      // below-density-floor vestige the server had to override every time).
      // Flight's 90 = 3 fps x 30 s, byte-identical to what the server derives.
      num_frames_target: flight ? 90 : input.is_insv ? undefined : 300,
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
      mesh_export: flight ? false : meshExport,
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
      mesh_export: Boolean(job.mesh_export),
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
      mesh_export: Boolean(job.mesh_export),
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
      mesh_export: meshExport,
    });
  }
  const glomapAvailable = Boolean(status?.engines?.glomap_available);
  const rigAvailable = Boolean(status?.engines?.rig_available);

  return (
    <div className="mx-auto max-w-[1880px] px-4 py-8 sm:px-6 xl:px-10">
      {/* hero */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="rounded-2xl border border-cyan-400/30 bg-cyan-400/10 p-3.5 text-cyan-200">
            <Orbit className="h-7 w-7" />
          </div>
          <div>
            <p className="text-[11px] font-bold uppercase tracking-[0.32em] text-cyan-200/80">Spatial Pipeline</p>
            <h1 className="display text-3xl font-black tracking-tight text-white">SplatLab</h1>
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
          {precheck && precheck.advisories.length > 0 && (
            <div className="mt-3 rounded-2xl border border-amber-400/25 bg-amber-400/5 p-3">
              <p className="text-xs font-semibold text-amber-200">
                Capture check
                <span className="ml-2 font-normal text-amber-200/60">advisory only — you can still create</span>
              </p>
              <ul className="mt-1 space-y-0.5 text-[12px] text-amber-100/80">
                {precheck.advisories.map((tip) => (
                  <li key={tip}>· {tip}</li>
                ))}
              </ul>
            </div>
          )}
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
                  testFlight ? "border-emerald-400 bg-emerald-400 text-accent-ink" : "border-white/20 bg-white/5"
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
                  languageField ? "border-cyan-400 bg-cyan-400 text-accent-ink" : "border-white/20 bg-white/5"
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

          {meshEngineAvailable && (
            <button
              type="button"
              onClick={() => setMeshExport((v) => !v)}
              disabled={computeControlsDisabled}
              className={`mt-3 flex w-full items-start gap-3 rounded-2xl border p-3 text-left transition-all ${
                meshExport
                  ? "border-violet-400/40 bg-violet-400/10"
                  : "border-white/10 bg-white/[0.02] hover:border-violet-500/20"
              } disabled:cursor-not-allowed disabled:opacity-50`}
            >
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md border transition-colors ${
                  meshExport ? "border-violet-400 bg-violet-400 text-accent-ink" : "border-white/20 bg-white/5"
                }`}
              >
                {meshExport && <CheckCircle2 className="h-3.5 w-3.5" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-1.5 text-sm font-semibold text-white">
                  <Box className="h-3.5 w-3.5 text-violet-200" /> Triangle mesh (Blender/CAD)
                </span>
                <span className="mt-0.5 block text-xs text-zinc-400">
                  Export a solid triangle mesh of the scene (.ply/.glb) for Blender, CAD, and 3D printing. Adds a
                  couple of minutes after training.
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
                  sparseMode ? "border-amber-400 bg-amber-400 text-accent-ink" : "border-white/20 bg-white/5"
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
                  generativeMode ? "border-fuchsia-400 bg-fuchsia-400 text-accent-ink" : "border-white/20 bg-white/5"
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
                {uploaded.is_insv && (
                  <p className="text-amber-200/80">
                    360 tip: hold the camera OVERHEAD on a stick and keep walking — if you're visible
                    anywhere but straight down, the scan fails. Selfie-style captures don't reconstruct.
                  </p>
                )}
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
                      <RerouteChips job={latestFailed} />
                      {latestFailed.health?.probe?.coaching?.length ? (
                        <ul className="space-y-0.5 pt-1 text-[12px] text-amber-200/80">
                          {latestFailed.health.probe.coaching.map((tip) => (
                            <li key={tip}>· {tip}</li>
                          ))}
                        </ul>
                      ) : null}
                      {glomapAvailable && (
                        <div className="pt-1.5">
                          <Button
                            size="sm"
                            disabled={!!activeJob || computeControlsDisabled || (latestFailed.sfm_tried ?? []).includes("glomap")}
                            onClick={() => retryGlomap(latestFailed)}
                          >
                            <RefreshCw className="h-3.5 w-3.5" /> Retry with global SfM
                          </Button>
                          <p className="mt-1 text-[11px] text-amber-200/60">
                            {(latestFailed.sfm_tried ?? []).includes("glomap")
                              ? "Global SfM already ran on this capture (see the fallback history above) — a retry would repeat the same result. Recapture instead."
                              : "Re-registers the same footage with a stronger solver — rescues most low-overlap captures."}
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

