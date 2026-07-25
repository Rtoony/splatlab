import type { SplatJob } from "@/lib/contracts";
import { Card } from "@/components/ui";
import { HEALTH_STYLE } from "@/components/jobs/health-badge";
import { ProbeSection } from "@/components/jobs/probe-section";
import { AlertTriangle, Box, CheckCircle2 } from "lucide-react";

export function CaptureHealthCard({ job }: { job: SplatJob }) {
  const fog = job.health?.fog;
  if (!fog && !job.health?.probe) return null;
  if (!fog) {
    return (
      <Card className="mb-4 p-4">
        <ProbeSection job={job} />
      </Card>
    );
  }
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
          <ProbeSection job={job} />
        </div>
      </div>
    </Card>
  );
}
