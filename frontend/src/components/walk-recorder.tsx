// "Record walk" (owner go 2026-10-01): record exactly what the walker's 3D canvas draws while you walk (the HUD is
// DOM, so it is not in the clip), save it as an MP4 beside the scene, and optionally run nexus-film's "realism
// finish" (SeedVR2: steadier and crisper, invents nothing) in the background. V starts/stops, even with the mouse
// captured; recordings stop themselves at MAX_SECONDS.
import { useCallback, useEffect, useRef, useState } from "react";
import { Circle, Download, Film, Loader2, Square, Wand2, X } from "lucide-react";
import { Button } from "@/components/ui";
import { finishWalk, getWalk, uploadWalk, type WalkRecord } from "@/lib/api";

const MAX_SECONDS = 60;
const CAPTURE_FPS = 30;
const MIME_TYPES = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];

type Phase = "idle" | "recording" | "saving";

function clock(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function minutes(s: number): string {
  return s < 90 ? `~${Math.max(1, Math.round(s / 10) * 10)} s` : `~${Math.round(s / 60)} min`;
}

export function WalkRecorderButton({ jobId, getCanvas, onStart, className = "" }: {
  jobId: string;
  getCanvas: () => HTMLCanvasElement | null;
  onStart?: () => void;
  className?: string;
}) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [walk, setWalk] = useState<WalkRecord | null>(null);
  const [open, setOpen] = useState(false);
  const rec = useRef<MediaRecorder | null>(null);
  const timer = useRef<number | null>(null);
  const phaseRef = useRef<Phase>("idle");
  phaseRef.current = phase;

  const stop = useCallback(() => {
    if (timer.current !== null) { window.clearInterval(timer.current); timer.current = null; }
    if (rec.current && rec.current.state !== "inactive") {
      rec.current.stop();
      // Stopping with V mid-walk leaves the mouse captured: release it so the pop-up can be clicked straight away.
      if (document.pointerLockElement) document.exitPointerLock();
    }
  }, []);

  const start = useCallback(() => {
    if (phaseRef.current !== "idle") return;
    const canvas = getCanvas();
    if (!canvas || typeof canvas.captureStream !== "function" || typeof MediaRecorder === "undefined") {
      setError("This browser can't record the canvas.");
      return;
    }
    const mimeType = MIME_TYPES.find((t) => MediaRecorder.isTypeSupported(t));
    if (!mimeType) { setError("This browser can't record WebM video."); return; }
    const stream = canvas.captureStream(CAPTURE_FPS);
    const r = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 16_000_000 });
    const chunks: Blob[] = [];
    r.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    r.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      rec.current = null;
      setPhase("saving");
      uploadWalk(jobId, new Blob(chunks, { type: "video/webm" }))
        .then((w) => { setWalk(w); setOpen(true); })
        .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
        .finally(() => setPhase("idle"));
    };
    rec.current = r;
    r.start(1000);
    const t0 = performance.now();
    setElapsed(0); setError(null); setPhase("recording");
    timer.current = window.setInterval(() => {
      const s = (performance.now() - t0) / 1000;
      setElapsed(s);
      if (s >= MAX_SECONDS) stop();
    }, 250);
    onStart?.();
  }, [getCanvas, jobId, onStart, stop]);

  // V toggles recording, also while the mouse is captured (the button can't be clicked then).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "KeyV" || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName))) return;
      if (phaseRef.current === "recording") stop();
      else if (phaseRef.current === "idle") start();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [start, stop]);

  useEffect(() => () => stop(), [stop]);

  // Poll a running finish even with the pop-up closed, so "Last walk" shows when it is done.
  const finishing = walk?.finish.status === "running";
  useEffect(() => {
    if (!walk || !finishing) return;
    const id = window.setInterval(() => {
      getWalk(jobId, walk.walk_id).then(setWalk).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(id);
  }, [finishing, jobId, walk]);

  return (
    <>
      <div className={`flex flex-col items-start gap-1 ${className}`}>
        <div className="flex items-center gap-2">
          {phase === "recording" ? (
            <Button type="button" size="sm" variant="outline" onClick={stop} title="Stop recording (V)">
              <Square className="h-3.5 w-3.5 fill-rose-400 text-rose-400" /> Stop · {clock(elapsed)}
            </Button>
          ) : (
            <Button type="button" size="sm" variant="outline" disabled={phase === "saving"} onClick={start}
              title={`Record what you see while you walk (up to ${MAX_SECONDS} s). Press V to start and stop, even while walking.`}>
              {phase === "saving" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Circle className="h-3.5 w-3.5 fill-rose-500 text-rose-500" />}
              {phase === "saving" ? "Saving walk…" : "Record walk (V)"}
            </Button>
          )}
          {walk && !open && phase === "idle" && (
            <Button type="button" size="sm" variant="outline" onClick={() => setOpen(true)}>
              {finishing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Film className="h-3.5 w-3.5" />} Last walk
            </Button>
          )}
        </div>
        {error && <p className="max-w-xs rounded bg-black/70 px-2 py-1 text-[10px] text-rose-300">{error}</p>}
      </div>
      {phase === "recording" && (
        <div className="pointer-events-none fixed left-1/2 top-4 z-40 flex -translate-x-1/2 items-center gap-2 rounded-full border border-rose-400/40 bg-black/70 px-3 py-1 text-xs font-semibold text-rose-200 backdrop-blur">
          <span className="h-2 w-2 animate-pulse rounded-full bg-rose-500" />
          REC {clock(elapsed)} / {clock(MAX_SECONDS)} · press V to stop
        </div>
      )}
      {walk && open && <WalkModal jobId={jobId} walk={walk} onChange={setWalk} onClose={() => setOpen(false)} />}
    </>
  );
}

function WalkModal({ jobId, walk, onChange, onClose }: {
  jobId: string;
  walk: WalkRecord;
  onChange: (w: WalkRecord) => void;
  onClose: () => void;
}) {
  const done = walk.finish.status === "done" && !!walk.realism_url;
  const [show, setShow] = useState<"walk" | "realism">(done ? "realism" : "walk");
  const [error, setError] = useState<string | null>(null);
  const running = walk.finish.status === "running";
  const ran = running && walk.finish.started ? Date.now() / 1000 - walk.finish.started : 0;
  const src = show === "realism" && walk.realism_url ? walk.realism_url : walk.video_url;

  useEffect(() => { if (done) setShow("realism"); }, [done]);

  const finish = () => {
    setError(null);
    finishWalk(jobId, walk.walk_id).then(onChange).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4" onClick={onClose}>
      <div className="w-full max-w-5xl rounded-2xl border border-white/10 bg-[#0a0f1a] p-3 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <span className="text-sm font-semibold text-white">Your walk</span>
          <span className="text-xs text-zinc-400">{walk.seconds}s · {walk.width}×{walk.height}</span>
          <div className="ml-2 flex overflow-hidden rounded-lg border border-white/10 text-[11px]">
            <button type="button" onClick={() => setShow("walk")}
              className={`px-2 py-1 ${show === "walk" ? "bg-white/15 text-white" : "text-zinc-400 hover:bg-white/5"}`}>As recorded</button>
            <button type="button" disabled={!done} onClick={() => setShow("realism")}
              className={`px-2 py-1 ${show === "realism" ? "bg-fuchsia-500/25 text-fuchsia-100" : "text-zinc-500"} disabled:opacity-40`}>Realism finish</button>
          </div>
          <button type="button" onClick={onClose} className="ml-auto rounded p-1 text-zinc-400 hover:bg-white/10" aria-label="Close"><X className="h-4 w-4" /></button>
        </div>
        <div className="relative overflow-hidden rounded-lg bg-black">
          <video key={src} src={src} controls autoPlay loop muted playsInline className="block max-h-[70vh] w-full" aria-label="Recorded walk" />
          {show === "realism" && (
            <span className="absolute right-2 top-2 rounded bg-fuchsia-500/70 px-1.5 py-0.5 text-[10px] text-white">AI polish (SeedVR2)</span>
          )}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-zinc-400">
          {walk.finish.status === "none" || walk.finish.status === "failed" || walk.finish.status === "interrupted" ? (
            <>
              <Button type="button" size="sm" variant="outline" disabled={!walk.finish_available} onClick={finish}
                title="SeedVR2 restoration via nexus-film: steadier and crisper, invents nothing. Runs in the background.">
                <Wand2 className="h-3.5 w-3.5" /> Realism finish ({minutes(walk.finish_estimate_s)})
              </Button>
              {walk.finish.status !== "none" && <span className="text-rose-300">Last finish {walk.finish.status}: {walk.finish.error}</span>}
              {!walk.finish_available && <span>nexus-film isn't installed here.</span>}
            </>
          ) : running ? (
            <span className="flex items-center gap-1.5">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Realism finish running… {clock(ran)} of {minutes(walk.finish.estimate_s ?? walk.finish_estimate_s)}.
              You can close this and keep walking.
            </span>
          ) : (
            <span>Realism finish done in {walk.finish.seconds}s: crisper and steadier, nothing invented.</span>
          )}
          {error && <span className="text-rose-300">{error}</span>}
          <a href={walk.video_url} download className="ml-auto">
            <Button type="button" size="sm" variant="outline"><Download className="h-3.5 w-3.5" /> Walk MP4</Button>
          </a>
          {done && walk.realism_url && (
            <a href={walk.realism_url} download>
              <Button type="button" size="sm" variant="outline"><Download className="h-3.5 w-3.5" /> Realism MP4</Button>
            </a>
          )}
        </div>
      </div>
    </div>
  );
}
