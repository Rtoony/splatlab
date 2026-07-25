import { useRef, useState } from "react";
import type { SplatUploadResult } from "@/lib/contracts";
import { AlertTriangle, CheckCircle2, Loader2, UploadCloud } from "lucide-react";

// ── upload ────────────────────────────────────────────────────────────────────
export function UploadBox({
  onUploaded,
  onError,
  current,
  disabled,
  disabledReason,
}: {
  onUploaded: (r: SplatUploadResult) => void;
  onError: (m: string) => void;
  current: SplatUploadResult | null;
  disabled: boolean;
  disabledReason: string;
}) {
  const [pct, setPct] = useState<number | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  function upload(file: File) {
    if (disabled) {
      onError(disabledReason);
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/splat/upload");
    xhr.upload.onprogress = (e) => e.lengthComputable && setPct(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => {
      setPct(null);
      if (xhr.status >= 200 && xhr.status < 300) onUploaded(JSON.parse(xhr.responseText));
      else onError(`Upload failed (${xhr.status}). Files >100 MB? Use Transfers below.`);
    };
    xhr.onerror = () => {
      setPct(null);
      onError("Upload failed. For large captures, drop into ~/transfers below.");
    };
    setPct(0);
    xhr.send(form);
  }

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        if (!disabled && e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`rounded-2xl border border-dashed border-white/15 bg-white/[0.02] p-6 text-center transition-colors ${
        disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:border-cyan-400/40"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.insv,.zip"
        className="hidden"
        disabled={disabled}
        onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
      />
      {disabled ? (
        <div className="space-y-1">
          <AlertTriangle className="mx-auto h-6 w-6 text-amber-300" />
          <p className="text-sm font-medium text-amber-100">New capture uploads blocked</p>
          <p className="text-xs text-amber-100/60">Existing scenes and downloads remain available.</p>
        </div>
      ) : pct !== null ? (
        <div className="space-y-2">
          <Loader2 className="mx-auto h-6 w-6 animate-spin text-cyan-300" />
          <div className="mx-auto h-1.5 w-2/3 overflow-hidden rounded-full bg-white/10">
            <div className="h-full bg-cyan-400 transition-all" style={{ width: `${pct}%` }} />
          </div>
          <p className="text-xs text-zinc-400">Uploading… {pct}%</p>
        </div>
      ) : current && !current.detail.startsWith("From Transfers") ? (
        <div className="flex items-center justify-center gap-2 text-sm text-emerald-300">
          <CheckCircle2 className="h-5 w-5" /> {current.name}
        </div>
      ) : (
        <div className="space-y-1">
          <UploadCloud className="mx-auto h-7 w-7 text-zinc-500" />
          <p className="text-sm font-medium text-zinc-200">Drop a file or click to browse</p>
          <p className="text-xs text-zinc-500">Video · 360 .insv · .zip of photos · up to ~100 MB over the web</p>
        </div>
      )}
    </div>
  );
}
