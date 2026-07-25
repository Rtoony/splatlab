import type { SplatTransferEntry } from "@/lib/contracts";
import { Badge, SectionLabel } from "@/components/ui";
import { FolderOpen, RefreshCw } from "lucide-react";

// ── transfers ─────────────────────────────────────────────────────────────────
export function TransfersPicker({
  entries,
  selectedPath,
  onSelect,
  onRefresh,
  refreshing,
  disabled,
}: {
  entries: SplatTransferEntry[];
  selectedPath: string | null;
  onSelect: (e: SplatTransferEntry) => void;
  onRefresh: () => void;
  refreshing: boolean;
  disabled: boolean;
}) {
  return (
    <div className="mt-4 space-y-2">
      <div className="flex items-center justify-between">
        <SectionLabel>Or pick from Transfers</SectionLabel>
        <button
          onClick={onRefresh}
          className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200"
          title="Refresh"
        >
          <RefreshCw className={`h-3 w-3 ${refreshing ? "animate-spin" : ""}`} /> no size limit
        </button>
      </div>
      <p className="text-xs text-zinc-500">
        Sync a capture into <code className="rounded bg-white/10 px-1">~/transfers</code> (Syncthing /
        pulse-share) — it skips the 100&nbsp;MB upload cap.
      </p>
      {entries.length > 0 ? (
        <div className="max-h-48 space-y-1.5 overflow-y-auto pr-1">
          {entries.map((e) => {
            const sel = selectedPath === e.path;
            return (
              <button
                key={e.path}
                onClick={() => onSelect(e)}
                disabled={disabled}
                className={`flex w-full items-center gap-3 rounded-xl border p-2.5 text-left transition-all ${
                  sel ? "border-cyan-400/40 bg-cyan-400/10" : "border-white/10 bg-white/[0.02] hover:border-cyan-500/20"
                } disabled:cursor-not-allowed disabled:opacity-50`}
              >
                <FolderOpen className={`h-4 w-4 shrink-0 ${sel ? "text-cyan-200" : "text-zinc-500"}`} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-zinc-100">{e.name}</span>
                  <span className="block truncate text-xs text-zinc-500">{e.detail}</span>
                </span>
                <Badge>{e.kind}</Badge>
              </button>
            );
          })}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed border-white/10 px-3 py-4 text-center text-xs text-zinc-500">
          Nothing splat-ready in Transfers yet. Drop a video, a 360 .insv, a .zip of photos, or a folder of JPGs.
        </div>
      )}
    </div>
  );
}
