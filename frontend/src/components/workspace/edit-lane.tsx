// Edit lane — the /view workspace Edit tab. v1 is the mount point contract
// for the native edit workbench (wave 5: quick ops wiring edit_ops.py's nine
// operations, semantic edit, version timeline). Until then it hosts the
// SuperSplat escape hatch and points at the crop tool that already exists.
import type { SplatJob } from "@/lib/contracts";
import { SectionLabel } from "@/components/ui";
import { Crosshair, ExternalLink, Sparkles, Wrench } from "lucide-react";

export function EditLane({ job }: { job: SplatJob }) {
  const superSplatHref = job.preview_file_url
    ? `/supersplat/?load=${encodeURIComponent(job.preview_file_url)}&filename=${encodeURIComponent(`${job.job_id}.ply`)}`
    : null;
  return (
    <div className="space-y-4 p-4">
      <div className="flex items-center gap-2">
        <Wrench className="h-4 w-4 text-cyan-300" />
        <SectionLabel>Edit</SectionLabel>
      </div>

      <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
        <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
          <Crosshair className="h-3.5 w-3.5 text-cyan-200" /> Crop to sphere
        </p>
        <p className="mt-1 text-xs leading-relaxed text-zinc-400">
          Available now in the <span className="text-zinc-200">Measure</span> tab's tool panel — pick a center,
          preview exactly what gets removed, apply with undo.
        </p>
      </div>

      {superSplatHref && (
        <a
          href={superSplatHref}
          target="_blank"
          rel="noreferrer"
          className="flex items-start gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-3 transition hover:border-cyan-400/30"
        >
          <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-cyan-200" />
          <span className="min-w-0">
            <span className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
              Edit in SuperSplat <ExternalLink className="h-3 w-3 text-zinc-500" />
            </span>
            <span className="mt-0.5 block text-xs leading-relaxed text-zinc-400">
              The full manual editor — select with rect/brush/picker, delete gaussians, transform, export. Opens in
              a new tab with this scene loaded.
            </span>
          </span>
        </a>
      )}

      <p className="text-[11px] leading-relaxed text-zinc-600">
        Coming next: one-click floater cleanup, box crop, decimate, and transforms — with preview-before-apply and
        a restore-point timeline. The engine for it is already built server-side.
      </p>
    </div>
  );
}
