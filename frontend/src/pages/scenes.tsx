// Scenes — the gallery landing page. Owns job status polling, the active-job
// strip, the newest-failure card, and gallery actions (rerun / promote / pin /
// delete). The create flow lives on /new (new-capture.tsx); state split from
// the old single-page splat.tsx along those lines, query keys unchanged.
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { apiRequest } from "@/lib/api";
import { setSplatlabFeedbackContext } from "@/lib/feedback-context";
import { ComputeGateBanner } from "@/components/compute-gate-banner";
import { ActiveJobPanel } from "@/components/jobs/active-job-panel";
import { RerouteChips } from "@/components/jobs/reroute-chips";
import { ResultsGallery } from "@/components/gallery/results-gallery";
import type { SplatJob, SplatStatusResponse } from "@/lib/contracts";
import { Badge, Button, Card, EmptyState, Skeleton, useToast } from "@/components/ui";
import { MAX_ITERS, QUALITY } from "@/lib/stage-meta";
import { AlertTriangle, Box, Cpu, Loader2, Plus, RefreshCw, X } from "lucide-react";

export default function ScenesPage() {
  const qc = useQueryClient();
  const pushToast = useToast();
  const flash = (msg: string, bad = false) => pushToast(msg, bad ? "error" : "info");
  // Dismissed failed-job notice (by job_id) so a newer failure still shows.
  const [dismissedFailed, setDismissedFailed] = useState<string | null>(null);

  const { data: status, isError, refetch } = useQuery({
    queryKey: ["status"],
    queryFn: () => apiRequest<SplatStatusResponse>("/api/splat/status"),
    refetchInterval: 2500,
  });

  const jobs = status?.jobs ?? [];
  const activeJob = jobs.find((j) => j.status === "running" || j.status === "starting") || null;
  const completed = jobs.filter((j) => j.status === "completed");
  const latestFailed = jobs
    .filter((j) => j.status === "failed" && !!j.error_message)
    .sort((a, b) => (b.completed_at ?? b.created_at).localeCompare(a.completed_at ?? a.created_at))[0] ?? null;
  const gpu = status?.gpu;
  const engineReady = Boolean(status?.engines?.ns_train_available && status?.engines?.colmap_available);
  const glomapAvailable = Boolean(status?.engines?.glomap_available);
  const compute = status?.compute;
  const computeBlocked = Boolean(compute && !compute.enabled);
  const computeArmed = Boolean(compute?.enabled && compute.mode === "supervised" && compute.supervised_unlock?.active);
  const computeControlsDisabled = !status || computeBlocked;

  useEffect(() => {
    setSplatlabFeedbackContext({
      page: "splatlab-gallery",
      active_job_id: activeJob?.job_id ?? null,
      active_job_status: activeJob?.status ?? null,
      active_job_stage: activeJob?.stage ?? null,
      completed_count: completed.length,
      failed_count: jobs.filter((j) => j.status === "failed").length,
      gpu_lane: gpu?.lane ?? null,
      gpu_locked: Boolean(gpu?.locked),
      compute_enabled: compute?.enabled ?? null,
      compute_reason: compute?.reason ?? null,
    });
    return () => setSplatlabFeedbackContext(null);
  }, [activeJob, completed.length, compute?.enabled, compute?.reason, gpu?.lane, gpu?.locked, jobs]);

  const startMutation = useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      apiRequest<SplatJob>("/api/splat/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["status"] });
      flash("Scene building started.");
    },
    onError: (e) => flash(e instanceof Error ? e.message : "Could not start", true),
  });

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

  // Re-run a finished/failed scene with its own persisted params (optionally at
  // higher quality). Forwards the scene's OWN num_frames_target/sfm_backend —
  // an unset sfm_backend would fall back to the request default and silently
  // contradict a scene actually built on glomap.
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

  // Re-run a failed capture with COLMAP 4.x global SfM.
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

  // Promote a Test Flight proof scene to a full build. Quality options here are
  // the same fresh-load defaults the old combined page used (standard preset,
  // no language field, no mesh) — pick different options by re-running from /new.
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
      max_num_iterations: QUALITY.standard.iterations,
      sfm_backend: job.sfm_backend ?? "glomap",
      language_field: false,
      mesh_export: false,
    });
  }

  return (
    <div className="mx-auto max-w-[1880px] px-4 py-8 sm:px-6 xl:px-10">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="display text-3xl font-black tracking-tight text-white">Scenes</h1>
          <p className="mt-1 text-sm text-zinc-400">Everything you've captured, built, and measured.</p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold ${
              !computeBlocked && engineReady
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}
          >
            <Cpu className="h-3.5 w-3.5" />{" "}
            {computeBlocked ? "Safe browse mode" : computeArmed ? "Supervised compute" : `Engine ${engineReady ? "ready" : "warming"}`}
          </span>
          <Badge className="border-cyan-500/30 bg-cyan-500/10 text-cyan-200">{status?.active_jobs ?? 0} active</Badge>
        </div>
      </header>

      <ComputeGateBanner compute={compute} />

      {isError && (
        <Card className="mb-4 flex items-center justify-between gap-3 border-red-400/25 bg-red-400/5 p-4">
          <p className="text-sm text-red-200">Couldn't reach SplatLab — the status feed is failing.</p>
          <Button size="sm" variant="outline" onClick={() => refetch()}>
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </Button>
        </Card>
      )}

      {!status && !isError && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <Skeleton className="h-64" />
          <Skeleton className="h-64" />
          <Skeleton className="hidden h-64 xl:block" />
        </div>
      )}

      {gpu?.locked && gpu.lane && gpu.lane !== "splat" && (
        <div className="mb-4 flex items-center gap-2 rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          <Loader2 className="h-4 w-4 animate-spin" />
          Waiting for the RTX 5090 — held by <span className="font-semibold">{gpu.lane}</span>. Queued work resumes automatically.
        </div>
      )}

      {activeJob && (
        <div className="mb-6">
          <ActiveJobPanel job={activeJob} onStop={() => stopMutation.mutate(activeJob.job_id)} stopping={stopMutation.isPending} />
        </div>
      )}

      {latestFailed && dismissedFailed !== latestFailed.job_id && (
        <Card className="relative mb-6 border-amber-500/40 bg-amber-500/10 p-4">
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
                Scene couldn't be built — {latestFailed.input_path.split("/").pop()}
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

      {status && completed.length === 0 && !activeJob && (
        <Card>
          <EmptyState
            icon={<Box className="h-10 w-10" />}
            title="Create your first scene"
            hint="Upload a video, a 360 clip, or a zip of photos — SplatLab turns it into an explorable 3D Gaussian splat you can measure, edit, and export."
            action={
              <Link
                href="/new"
                className="inline-flex items-center gap-1.5 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-accent-ink hover:bg-accent-hover"
              >
                <Plus className="h-4 w-4" /> New capture
              </Link>
            }
          />
        </Card>
      )}

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
