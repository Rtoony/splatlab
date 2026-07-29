import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import type { SplatJob } from "@/lib/contracts";
import { apiRequest } from "@/lib/api";
import { Badge, Button, Card, useToast } from "@/components/ui";
import { DownloadMenu } from "@/components/gallery/download-menu";
import { HealthBadge } from "@/components/jobs/health-badge";
import { fmtCount, relTime, sceneHue } from "@/lib/format";
import {
  AlertTriangle,
  Camera,
  Layers,
  Loader2,
  MapPin,
  Mountain,
  Orbit,
  Pin,
  RefreshCw,
  Rocket,
  ShieldCheck,
  Trash2,
  Wand2,
} from "lucide-react";

export function SceneCard({
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

  // One-click into playable mode, straight from the gallery — same ladder the
  // scene page's Make walkable runs (resumable; skips existing stages).
  const [, navigate] = useLocation();
  const pushToast = useToast();
  const [worldPrep, setWorldPrep] = useState(false);
  const startWorldPrep = () => {
    if (worldPrep) return;
    setWorldPrep(true);
    pushToast("Preparing walkable world — heavy stages can take many minutes. Progress lives in Activity.", "info");
    apiRequest(`/api/splat/jobs/${encodeURIComponent(job.job_id)}/world/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    }).then(() => {
      pushToast("World ready — walking in.", "success");
      navigate(`/world/${job.job_id}`);
    }).catch((e: unknown) => {
      pushToast(`World preparation failed: ${e instanceof Error ? e.message : String(e)}`, "error");
    }).finally(() => setWorldPrep(false));
  };

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
            {job.world_available ? (
              <span
                title="A solidified, walkable world exists — Walk takes you straight in"
                className="flex items-center gap-0.5 rounded bg-cyan-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-cyan-200 backdrop-blur-sm"
              >
                <Mountain className="h-3 w-3" /> walkable
              </span>
            ) : null}
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
            {job.scene?.assemble?.state === "approved" ? (
              <span
                title="Scene reassembly approved — scene.glb/scene.blend available in the download menu"
                className="flex items-center gap-0.5 rounded bg-emerald-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-200 backdrop-blur-sm"
              >
                <ShieldCheck className="h-3 w-3" /> scene approved
              </span>
            ) : job.scene?.assemble?.state === "built" ? (
              <span
                title="Scene reassembled but not yet approved — open Scene in the viewer to review and approve"
                className="flex items-center gap-0.5 rounded bg-amber-400/20 px-1.5 py-0.5 text-[10px] font-semibold text-amber-200 backdrop-blur-sm"
              >
                <Layers className="h-3 w-3" /> scene assembled
              </span>
            ) : null}
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
        {job.world_available ? (
          <a href={`/world/${job.job_id}`} target="_blank" rel="noreferrer" className="flex-1">
            <Button
              size="sm"
              variant="outline"
              className="w-full border-cyan-300/40 text-cyan-200"
              title="Walk this scene in first person — zombie waves live here"
            >
              <Mountain className="h-3.5 w-3.5" /> Walk
            </Button>
          </a>
        ) : job.status === "completed" ? (
          <Button
            size="sm"
            variant="outline"
            className="flex-1 border-cyan-300/40 text-cyan-200"
            disabled={worldPrep || computeBlocked}
            onClick={startWorldPrep}
            title="Build the walkable world right from here — mesh through scenario in one go; takes minutes, resumes after a failure"
          >
            {worldPrep
              ? (<><Loader2 className="h-3.5 w-3.5 animate-spin" /> Preparing…</>)
              : (<><Mountain className="h-3.5 w-3.5" /> Make walkable</>)}
          </Button>
        ) : null}
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
