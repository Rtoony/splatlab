import type { SplatJob } from "@/lib/contracts";

export const PROBE_STYLE: Record<string, { label: string; pill: string }> = {
  GOOD: { label: "good capture", pill: "bg-emerald-500/15 text-emerald-300" },
  MARGINAL: { label: "marginal capture", pill: "bg-amber-500/15 text-amber-300" },
  POOR: { label: "poor capture", pill: "bg-red-500/15 text-red-300" },
};

// Pre-train probe section (Capture Coach Phase 1) — coaching computed from the
// SfM artifacts before any GPU training. Report-only, like the fog verdict.
export function ProbeSection({ job }: { job: SplatJob }) {
  const probe = job.health?.probe;
  if (!probe) return null;
  const style = PROBE_STYLE[probe.verdict] ?? PROBE_STYLE.MARGINAL;
  const metrics = probe.metrics ?? {};
  return (
    <div className="mt-2 border-t border-white/5 pt-2">
      <p className="text-sm font-semibold text-zinc-100">
        Capture probe: <span className={`rounded px-1.5 py-0.5 ${style.pill}`}>{style.label}</span>
        {metrics.capture_shape && (
          <span className="ml-2 text-[11px] font-normal text-zinc-500">
            {metrics.capture_shape === "orbit" ? "orbit-style capture" : "walkthrough-style capture"}
          </span>
        )}
      </p>
      {probe.coaching.length > 0 && (
        <ul className="mt-1 space-y-0.5 text-sm text-zinc-400">
          {probe.coaching.map((tip) => (
            <li key={tip}>· {tip}</li>
          ))}
        </ul>
      )}
      <p className="mt-1 text-[11px] tabular-nums text-zinc-500">
        {metrics.n_posed ?? "—"} posed · {metrics.n_points ?? "—"} map points
        {metrics.traj_cloud_ratio != null && <> · path/map ratio {metrics.traj_cloud_ratio}</>}
        {metrics.registration_ratio != null && <> · registration {Math.round(metrics.registration_ratio * 100)}%</>}
      </p>
    </div>
  );
}
