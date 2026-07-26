// Semantic edit panel — type what to change, the server does the rest.
// Front-end for POST /jobs/{id}/edit/semantic (backend/edit_ops.py
// SemanticEditRequest): scores every gaussian against the text query via the
// language field, then row-masks the scene (delete/isolate) or copies the
// match into a new derived scene (extract).
//
// Deliberate scope cut: NO client-side match preview in this iteration — the
// relevancy-channel preview machinery is a much bigger integration, and the
// server recomputes matches at apply time anyway (it 409s if the scene
// changed under it). The copy below says exactly that instead of pretending.
//
// Gating is polled truth, never local state: the panel requires
// job.langfield_available && !job.langfield_stale, and since delete/isolate
// mark the field stale server-side, the next status poll flips
// job.langfield_stale and this panel disables itself with the stale reason.
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link } from "wouter";
import { semanticEdit } from "@/lib/api";
import type { SplatJob, SplatSemanticEditRequest } from "@/lib/contracts";
import { Button, Input, useToast } from "@/components/ui";
import { EditProgress } from "@/components/edit-progress";
import { ArrowRight, Loader2, ScanSearch } from "lucide-react";

// Backend defaults, read from edit_ops.py SemanticEditRequest — keep in sync.
const DEFAULT_THRESHOLD = 0.5; // Field(default=0.5, ge=0.0, le=1.0)
const DEFAULT_CLEANUP = true; // cleanup: bool = True

type SemanticMode = SplatSemanticEditRequest["mode"];

const MODES: { key: SemanticMode; label: string; blurb: string }[] = [
  { key: "delete", label: "Delete", blurb: "Remove everything that matches; keep the rest of the scene." },
  { key: "isolate", label: "Isolate", blurb: "Keep only what matches; remove everything else." },
  { key: "extract", label: "Extract", blurb: "Copy the match into a new scene — this scene stays untouched." },
];

export function SemanticEditPanel({
  job,
  onEdited,
  onBusyChange,
}: {
  job: SplatJob;
  onEdited?: () => void;
  // Reports this panel's own mutation state up so the host lane's
  // "edit started elsewhere" server-truth banner doesn't misfire on it.
  onBusyChange?: (busy: boolean) => void;
}) {
  const toast = useToast();
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [mode, setMode] = useState<SemanticMode>("delete");
  const [cleanup, setCleanup] = useState(DEFAULT_CLEANUP);
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  // Backend {detail} verbatim (409 stale, 422 no match, 503 worker/arbiter).
  const [error, setError] = useState<string | null>(null);
  // Non-fatal derived-artifact regen warnings from the response.
  const [warnings, setWarnings] = useState<string[]>([]);
  // Last successful extract → the "Open new scene" link.
  const [extracted, setExtracted] = useState<{ id: string; query: string } | null>(null);

  useEffect(() => {
    if (!armed) return;
    const t = window.setTimeout(() => setArmed(false), 4000);
    return () => window.clearTimeout(t);
  }, [armed]);

  // Mirror busy up to the host (cleanup also clears it on unmount so a
  // mid-flight tab switch can't leave the lane thinking we're still busy).
  useEffect(() => {
    onBusyChange?.(busy);
    return () => onBusyChange?.(false);
  }, [busy, onBusyChange]);

  const gated = !job.langfield_available || Boolean(job.langfield_stale);

  async function run() {
    const q = text.trim();
    if (!q) return;
    setArmed(false);
    setBusy(true);
    setError(null);
    setWarnings([]);
    try {
      const resp = await semanticEdit(job.job_id, { text: q, threshold, mode, cleanup });
      setWarnings(resp.warnings ?? []);
      if (resp.mode === "extract" && resp.new_job_id) {
        setExtracted({ id: resp.new_job_id, query: q });
        toast(
          `Extracted ${resp.matched.toLocaleString()} matched splats into a new scene (${(resp.rows_after_cleanup ?? resp.kept).toLocaleString()} after cleanup)`,
          "success",
        );
        // The new derived job shows up in the status poll's job list.
        void qc.invalidateQueries({ queryKey: ["status"] });
      } else {
        toast(
          resp.mode === "delete"
            ? `Deleted ${resp.matched.toLocaleString()} matched splats · restore point v${resp.version_before} saved`
            : `Isolated ${resp.matched.toLocaleString()} matched splats (removed the rest) · restore point v${resp.version_before} saved`,
          "success",
        );
        // Status poll picks up the new gaussian count AND langfield_stale=true
        // (which disables this panel with the honest stale reason).
        void qc.invalidateQueries({ queryKey: ["status"] });
        void qc.invalidateQueries({ queryKey: ["edit-versions", job.job_id] });
        onEdited?.(); // reload the viewer — this scene's geometry changed
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Semantic edit failed.");
    } finally {
      setBusy(false);
    }
  }

  const activeMode = MODES.find((m) => m.key === mode)!;

  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-3">
      <p className="flex items-center gap-1.5 text-sm font-semibold text-zinc-100">
        <ScanSearch className="h-3.5 w-3.5 text-cyan-200" /> Semantic edit
      </p>

      {gated ? (
        <>
          {/* Honest disabled card — the same two reasons the backend 422s/409s on. */}
          <p className="mt-1 text-xs leading-relaxed text-zinc-400">
            Type what to delete, isolate, or extract — needs a language field that still matches this scene.
          </p>
          <p className="mt-2 rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[11px] leading-snug text-amber-100/85">
            {job.langfield_stale
              ? "Language field is stale — the scene's geometry was edited after the field was built, so text matches would land on the wrong splats."
              : "No language field built for this scene, so there's nothing to match text against."}{" "}
            Re-run this scene with Language search on to rebuild it (retrains the scene).
          </p>
          {extracted && (
            <ExtractedLink id={extracted.id} query={extracted.query} />
          )}
        </>
      ) : (
        <div className="mt-1 space-y-2">
          <p className="text-xs leading-relaxed text-zinc-400">
            Type what to change and pick what happens to the matches. Matches are computed server-side when you
            apply; save point is created first, Undo restores it.
          </p>

          <Input
            size="xs"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              setArmed(false);
            }}
            placeholder="the fire hydrant, power lines, …"
            autoComplete="off"
            maxLength={200}
            disabled={busy}
            // The Spark viewer listens for WASD/etc. on window — stop key
            // events from reaching it while typing (same fix as the search panel).
            onKeyDown={(e) => e.stopPropagation()}
            onKeyUp={(e) => e.stopPropagation()}
          />

          <div className="flex items-center gap-2">
            <span className="shrink-0 text-[10px] uppercase tracking-wide text-zinc-500">threshold</span>
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={threshold}
              onChange={(e) => {
                setThreshold(Number(e.target.value));
                setArmed(false);
              }}
              className="w-full"
              disabled={busy}
            />
            <span className="w-10 shrink-0 text-right text-xs tabular-nums text-zinc-400">{threshold.toFixed(2)}</span>
          </div>
          <p className="text-[10px] leading-snug text-zinc-500">
            Lower matches more (looser), higher matches less (stricter). If nothing matches, the edit refuses
            and changes nothing.
          </p>

          <div className="flex gap-1 rounded-xl border border-white/10 bg-black/20 p-1">
            {MODES.map((m) => (
              <button
                key={m.key}
                type="button"
                onClick={() => {
                  setMode(m.key);
                  setArmed(false);
                }}
                disabled={busy}
                className={`flex-1 rounded-lg py-1.5 text-xs font-semibold transition ${
                  mode === m.key ? "bg-cyan-400/20 text-cyan-200" : "text-zinc-400 hover:text-zinc-200"
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
          <p className="text-[10px] leading-snug text-zinc-500">{activeMode.blurb}</p>

          <label className="flex items-start gap-1.5 text-[11px] leading-snug text-zinc-400">
            <input
              type="checkbox"
              checked={cleanup}
              onChange={(e) => setCleanup(e.target.checked)}
              className="mt-0.5 accent-cyan-300"
              disabled={busy}
            />
            Also remove boundary halos (turn off for wispy subjects: wires, foliage, smoke)
          </label>

          <Button
            type="button"
            size="sm"
            variant={armed ? "primary" : "outline"}
            className={`w-full ${armed && mode !== "extract" ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}`}
            onClick={() => (armed ? void run() : setArmed(true))}
            disabled={busy || !text.trim()}
          >
            {busy ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" /> Applying…
              </>
            ) : armed ? (
              mode === "delete" ? (
                "Sure? Permanently removes the matches"
              ) : mode === "isolate" ? (
                "Sure? Permanently removes everything else"
              ) : (
                "Sure? Builds a new scene from the matches"
              )
            ) : !text.trim() ? (
              "Apply (type a query first)"
            ) : (
              `${activeMode.label} “${text.trim().length > 24 ? `${text.trim().slice(0, 24)}…` : text.trim()}”`
            )}
          </Button>
          {/* Semantic edits prepend a "match" step to the stepped pipeline. */}
          {busy && <EditProgress jobId={job.job_id} active />}

          {error && (
            <p className="rounded-lg border border-red-400/25 bg-red-400/10 px-2 py-1.5 text-[11px] leading-snug text-red-200">
              {error}
            </p>
          )}
          {warnings.length > 0 && (
            <div className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-2 py-1.5 text-[11px] leading-snug text-amber-100/85">
              {warnings.map((w, i) => (
                <p key={i}>{w}</p>
              ))}
            </div>
          )}
          {extracted && <ExtractedLink id={extracted.id} query={extracted.query} />}
        </div>
      )}
    </div>
  );
}

// Persistent link to the derived scene an extract produced (survives the
// panel disabling itself if a later delete/isolate marks the field stale).
function ExtractedLink({ id, query }: { id: string; query: string }) {
  return (
    <Link
      href={`/view/${id}`}
      className="mt-2 flex items-center justify-between gap-2 rounded-lg border border-cyan-300/25 bg-cyan-400/10 px-2.5 py-1.5 text-xs text-cyan-100 transition hover:bg-cyan-400/20"
    >
      <span className="min-w-0 truncate">
        Extracted <span className="font-semibold">“{query}”</span>
      </span>
      <span className="flex shrink-0 items-center gap-1 font-semibold">
        Open new scene <ArrowRight className="h-3 w-3" />
      </span>
    </Link>
  );
}
