import type { SplatJob } from "@/lib/contracts";
import { RefreshCw } from "lucide-react";

// Auto-fallback history chips: one per reroute, e.g. "colmap → glomap · 10.3%
// registered". Rendered under the stage rail (live) and on the failed card so
// the WHY of a solver climb is visible without digging through logs.
export function RerouteChips({ job }: { job: SplatJob }) {
  const reroutes = job.sfm_reroutes ?? [];
  if (!reroutes.length) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {reroutes.map((r, i) => (
        <span
          key={`${r.to_solver}-${i}`}
          title={
            r.registered != null && r.extracted != null
              ? `Only ${r.registered}/${r.extracted} frames registered — auto-retried with ${r.to_solver}`
              : `Auto-retried with ${r.to_solver}`
          }
          className="inline-flex items-center gap-1 rounded-full border border-amber-400/25 bg-amber-400/10 px-2 py-0.5 text-[10px] font-semibold text-amber-200"
        >
          <RefreshCw className="h-2.5 w-2.5" />
          {r.from_solver} → {r.to_solver} · {r.pct} registered
        </span>
      ))}
    </div>
  );
}
