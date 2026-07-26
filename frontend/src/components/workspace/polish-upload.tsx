// The polish round-trip's return leg, as a reusable drop zone: a GLB polished
// in Blender (or any DCC) lands back as a first-class versioned asset. Mirrors
// the Import-edited-.ply card's discipline — drag/drop or browse, client-side
// extension preflight, two-click arm/confirm, XHR progress — but stays
// artifact-agnostic: the caller supplies the actual upload call.

import { useRef, useState } from "react";
import { Button } from "@/components/ui";
import { Loader2, UploadCloud } from "lucide-react";

function fmtFileSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(2)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1000))} KB`;
}

export function PolishUploadZone({
  title,
  hint,
  confirmLabel,
  disabled,
  onUpload,
}: {
  title: string;
  hint: string;
  /** Shown on the armed confirm button — say what gets replaced. */
  confirmLabel: string;
  disabled?: boolean;
  /** Perform the upload; resolve on success, reject with a user-readable error. */
  onUpload: (file: File, onProgress: (pct: number) => void) => Promise<void>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [armed, setArmed] = useState(false);
  const [pct, setPct] = useState<number | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const busy = disabled || uploading;

  const choose = (candidate: File | null) => {
    setError(null);
    setArmed(false);
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".glb")) {
      setError(`'${candidate.name}' is not a .glb — export binary glTF from your DCC tool.`);
      setFile(null);
      return;
    }
    setFile(candidate);
  };

  const run = async () => {
    if (!file) return;
    setUploading(true);
    setPct(0);
    setError(null);
    try {
      await onUpload(file, setPct);
      setFile(null);
      setArmed(false);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setUploading(false);
      setPct(null);
    }
  };

  return (
    <div className="mt-2">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!busy && e.dataTransfer.files[0]) choose(e.dataTransfer.files[0]);
        }}
        onClick={() => !busy && inputRef.current?.click()}
        className={`rounded-xl border border-dashed border-white/15 bg-white/[0.02] px-3 py-2.5 text-center transition-colors ${
          busy ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-cyan-400/40"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".glb"
          className="hidden"
          disabled={busy}
          onChange={(e) => {
            choose(e.target.files?.[0] ?? null);
            e.target.value = ""; // re-picking the same file must re-fire onChange
          }}
        />
        {uploading ? (
          <div className="space-y-1.5">
            <Loader2 className="mx-auto h-4 w-4 animate-spin text-cyan-300" />
            {pct !== null && (
              <>
                <div className="mx-auto h-1.5 w-2/3 overflow-hidden rounded-full bg-white/10">
                  <div className="h-full bg-cyan-400 transition-all" style={{ width: `${pct}%` }} />
                </div>
                <p className="text-[11px] text-zinc-400">Uploading… {pct}%</p>
              </>
            )}
          </div>
        ) : file ? (
          <p className="text-xs text-zinc-200">
            <span className="font-semibold">{file.name}</span>
            <span className="text-zinc-500"> · {fmtFileSize(file.size)}</span>
          </p>
        ) : (
          <p className="flex items-center justify-center gap-1.5 text-xs text-zinc-500">
            <UploadCloud className="h-3.5 w-3.5" /> {title}
          </p>
        )}
      </div>
      {!file && !uploading && <p className="mt-1 text-[10px] leading-snug text-zinc-500">{hint}</p>}
      {error && <p className="mt-1 text-[11px] leading-snug text-red-300">{error}</p>}
      {file && !uploading && (
        <Button
          type="button"
          size="sm"
          variant={armed ? "primary" : "outline"}
          className={`mt-1.5 w-full ${armed ? "border-red-400/50 bg-red-400/20 text-red-100" : ""}`}
          onClick={() => (armed ? void run() : setArmed(true))}
          disabled={busy}
        >
          {armed ? confirmLabel : `Upload ${file.name}`}
        </Button>
      )}
    </div>
  );
}
