import { useEffect, useRef } from "react";
import type { SplatJob } from "@/lib/contracts";
import { Button, Card } from "@/components/ui";
import { humanizeStage } from "@/lib/stage-meta";
import { StageRail } from "@/components/jobs/stage-rail";
import { RerouteChips } from "@/components/jobs/reroute-chips";
import { Loader2, Square } from "lucide-react";

export function ActiveJobPanel({ job, onStop, stopping }: { job: SplatJob; onStop: () => void; stopping: boolean }) {
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
      <RerouteChips job={job} />
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
