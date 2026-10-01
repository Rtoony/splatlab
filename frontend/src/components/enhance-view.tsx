// "Enhance this view" (owner go 2026-10-01): grab what the viewer is showing, send it with the camera pose (in the
// splat's own frame) to the server, which runs NVIDIA Difix guided by the nearest real capture photo, and show the
// result as a before/after slider. The result is a STILL, clearly badged AI-enhanced — the scene never changes.
import { useCallback, useRef, useState } from "react";
import { Download, Loader2, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui";
import { enhanceView, type EnhanceResult } from "@/lib/api";

export interface EnhanceCapture {
  image: string;
  camera: { position: number[]; forward: number[] } | null;
}

const UPLOAD_LONG_SIDE = 1280;

/** Downscale a canvas data URL so a 4K, DPR-2 viewport doesn't upload 10 MB; JPEG keeps it small. */
async function shrink(dataUrl: string): Promise<string> {
  const img = new Image();
  img.src = dataUrl;
  await img.decode();
  const s = Math.min(1, UPLOAD_LONG_SIDE / Math.max(img.width, img.height));
  const c = document.createElement("canvas");
  c.width = Math.max(8, Math.round(img.width * s));
  c.height = Math.max(8, Math.round(img.height * s));
  c.getContext("2d")?.drawImage(img, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.95);
}

export function EnhanceViewButton({ jobId, capture, disabled, className = "" }: {
  jobId: string;
  capture: () => EnhanceCapture | null;
  disabled?: boolean;
  className?: string;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EnhanceResult | null>(null);

  const run = useCallback(async () => {
    if (busy) return;
    const shot = capture();
    if (!shot) {
      setError("The view isn't ready yet.");
      return;
    }
    setBusy(true); setError(null);
    try {
      setResult(await enhanceView(jobId, await shrink(shot.image), shot.camera));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [busy, capture, jobId]);

  return (
    <>
      <div className={`flex flex-col items-start gap-1 ${className}`}>
        <Button type="button" size="sm" variant="outline" disabled={disabled || busy} onClick={() => void run()}
          title="Make an AI-enhanced still of exactly this view (NVIDIA Difix, guided by your nearest real photo). The scene itself is not changed.">
          {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
          {busy ? "Enhancing… (~10 s)" : "Enhance this view"}
        </Button>
        {error && <p className="max-w-xs rounded bg-black/70 px-2 py-1 text-[10px] text-rose-300">{error}</p>}
      </div>
      {result && <EnhanceModal result={result} onClose={() => setResult(null)} />}
    </>
  );
}

function EnhanceModal({ result, onClose }: { result: EnhanceResult; onClose: () => void }) {
  const [split, setSplit] = useState(50);
  const box = useRef<HTMLDivElement | null>(null);
  const ref = result.reference;
  const refNote = ref?.used
    ? `Guided by your real photo ${ref.source ?? ""} (${ref.distance_m} m away).`
    : ref?.why_not ?? "No real photo to guide it (plain Difix).";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
      <div className="w-full max-w-6xl rounded-2xl border border-white/10 bg-[#0a0f1a] p-3 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-2 flex items-center gap-2">
          <span className="rounded-full border border-fuchsia-300/40 bg-fuchsia-400/15 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-fuchsia-200">
            AI-enhanced
          </span>
          <span className="text-xs text-zinc-400">Drag the slider: left = what the splat shows, right = enhanced. Generated detail, not a capture.</span>
          <button type="button" onClick={onClose} className="ml-auto rounded p-1 text-zinc-400 hover:bg-white/10" aria-label="Close"><X className="h-4 w-4" /></button>
        </div>
        <div ref={box} className="relative w-full select-none overflow-hidden rounded-lg bg-black">
          <img src={result.after_url} alt="AI-enhanced view" className="block w-full" draggable={false} />
          <div className="absolute inset-0 overflow-hidden" style={{ width: `${split}%` }}>
            <img src={result.before_url} alt="Splat view as rendered" className="block h-full max-w-none object-cover object-left"
              style={{ width: box.current ? `${box.current.clientWidth}px` : "100%" }} draggable={false} />
          </div>
          <div className="pointer-events-none absolute inset-y-0" style={{ left: `${split}%` }}>
            <div className="h-full w-0.5 -translate-x-1/2 bg-white/80 shadow" />
          </div>
          <span className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-zinc-200">splat</span>
          <span className="absolute right-2 top-2 rounded bg-fuchsia-500/70 px-1.5 py-0.5 text-[10px] text-white">AI-enhanced</span>
          <input type="range" min={0} max={100} value={split} onChange={(e) => setSplit(Number(e.target.value))}
            aria-label="Before / after" className="absolute inset-x-0 bottom-2 mx-auto w-2/3 accent-fuchsia-400" />
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-zinc-400">
          <span>{refNote}</span>
          <span>{result.model} · {result.seconds}s</span>
          <a href={result.after_url} download className="ml-auto">
            <Button type="button" size="sm" variant="outline"><Download className="h-3.5 w-3.5" /> Download enhanced</Button>
          </a>
          {result.ref_url && (
            <a href={result.ref_url} target="_blank" rel="noreferrer" className="text-cyan-300/80 hover:underline">see the real photo</a>
          )}
        </div>
      </div>
    </div>
  );
}
