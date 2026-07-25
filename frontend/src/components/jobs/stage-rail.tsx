import type { SplatJob } from "@/lib/contracts";
import { STAGE_ORDER, stageShort } from "@/lib/stage-meta";

// ── active job: stage rail + humanized stage + log ────────────────────────────
export function StageRail({ job }: { job: SplatJob }) {
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
