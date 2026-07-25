import type { SplatJob } from "@/lib/contracts";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

// ── capture health (report-only fog gate, Capture Coach) ──────────────────────
export const HEALTH_STYLE = {
  FOG: { pill: "bg-amber-400/20 text-amber-200", label: "likely fog" },
  HEALTHY: { pill: "bg-emerald-400/20 text-emerald-200", label: "healthy" },
  UNCERTAIN: { pill: "bg-zinc-400/20 text-zinc-300", label: "unverified" },
} as const;

export function HealthBadge({ job }: { job: SplatJob }) {
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
