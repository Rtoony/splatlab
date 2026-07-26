// Stepped edit-progress strip — shown under whichever edit op is running.
// Server truth first: the ["activity"] poll's edit_progress entry for this
// job drives the stage dots + current-step label; the per-job `editing` flag
// (or the caller's `active` = a local mutation is pending) keeps the strip up
// even when the deployed backend is older and sends no edit_progress at all —
// in that degraded shape we show an indeterminate bar + elapsed instead of
// pretending to know the step. All activity access is optional-chained on
// purpose: the live server may predate the edit_progress contract.
import { useEffect, useState } from "react";
import { useActivity } from "@/lib/use-activity";

// Human labels for the backend step vocabulary (apply/upload = snapshot →
// apply → compress → webopt → langweb → finalize; semantic prepends "match").
const STEP_HUMAN: Record<string, string> = {
  match: "Matching gaussians",
  snapshot: "Saving restore point",
  apply: "Applying the edit",
  compress: "Rebuilding compressed copy",
  webopt: "Rebuilding web preview",
  langweb: "Rebuilding search overlay",
  finalize: "Finishing up",
};

function stepHuman(step: string): string {
  return STEP_HUMAN[step] ?? step;
}

function fmtElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

// Same tiny dot visual as scene-regen's StageDot (copied, not imported — that
// one is private to the P6 modal and keyed on different semantics).
function StepDot({ running, done, title }: { running: boolean; done: boolean; title: string }) {
  const color = running ? "bg-cyan-300 animate-pulse" : done ? "bg-emerald-400" : "bg-zinc-600";
  return <span title={title} className={`h-1.5 w-1.5 shrink-0 rounded-full ${running ? "ring-2 ring-cyan-300/40" : ""} ${color}`} />;
}

export function EditProgress({ jobId, active }: { jobId: string; active: boolean }) {
  // Fast-poll (2s) while OUR mutation is pending; server-observed edits from
  // elsewhere ride the normal 4s cadence.
  const activity = useActivity(active);
  const progress = activity.data?.edit_progress?.[jobId];
  const editing = Boolean(activity.data?.jobs?.[jobId]?.editing);
  const visible = active || Boolean(progress) || editing;

  // 1s elapsed clock, only while visible. One state object carries BOTH the
  // local fallback start (the moment this strip first became visible — used
  // when the server sends no started_at) and the current tick, updated only
  // from timer callbacks (never synchronously in the effect body).
  const [clock, setClock] = useState<{ start: number; now: number } | null>(null);
  useEffect(() => {
    if (!visible) {
      const t = window.setTimeout(() => setClock(null), 0);
      return () => window.clearTimeout(t);
    }
    const tick = () => setClock((prev) => ({ start: prev?.start ?? Date.now(), now: Date.now() }));
    const t0 = window.setTimeout(tick, 0);
    const t = window.setInterval(tick, 1000);
    return () => {
      window.clearTimeout(t0);
      window.clearInterval(t);
    };
  }, [visible]);

  if (!visible) return null;

  // Prefer the server's started_at, but only when it parses AND lands in a
  // sane window (0–6h ago) — a timezone-naive ISO string would otherwise pin
  // the ticker to 0s or an absurd value. Outside that window, fall back to
  // when we first saw the edit locally. Before the first tick lands (one
  // microtask after mount) we honestly show 0s.
  const serverStart = progress?.started_at ? Date.parse(progress.started_at) : NaN;
  let elapsed = "0s";
  if (clock) {
    const serverSane = Number.isFinite(serverStart) && clock.now - serverStart >= 0 && clock.now - serverStart < 6 * 3600_000;
    elapsed = fmtElapsed(clock.now - (serverSane ? serverStart : clock.start));
  }

  const labels = progress?.labels ?? [];
  const stepIndex = progress?.step_index ?? 0;
  const label = progress ? stepHuman(progress.step) : "Edit in progress…";

  return (
    <div className="mt-2 rounded-xl border border-cyan-300/20 bg-cyan-400/[0.06] px-3 py-2">
      <div className="flex items-center gap-2">
        {labels.length > 0 && (
          <span className="flex shrink-0 items-center gap-1">
            {labels.map((l, i) => (
              <StepDot key={`${l}-${i}`} done={i < stepIndex} running={i === stepIndex} title={stepHuman(l)} />
            ))}
          </span>
        )}
        <span className="min-w-0 flex-1 truncate text-xs font-semibold text-cyan-100">{label}</span>
        <span className="shrink-0 font-mono text-[11px] tabular-nums text-cyan-200/80">{elapsed}</span>
      </div>
      {labels.length === 0 && (
        // Degraded shape (no edit_progress from the server): indeterminate bar.
        <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div className="h-full w-1/3 animate-pulse rounded-full bg-cyan-400/70" />
        </div>
      )}
      <p className="mt-1 text-[10px] leading-snug text-zinc-500">
        Usually 10–30 s — large scenes take longer. Safe to stay on this tab.
      </p>
    </div>
  );
}
