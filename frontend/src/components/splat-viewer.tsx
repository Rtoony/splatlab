import { useEffect, useRef, useState } from "react";

// In-browser Gaussian-splat viewer (mkkellogg). Ported from the portal; renders
// a .ply (raw or web-optimized) or .spz. `fill` makes it fill its parent.
export function SplatViewer({
  url,
  format = "ply",
  fill = false,
  focus = null,
}: {
  url: string;
  format?: "ply" | "spz";
  fill?: boolean;
  // When set (from a language search hit), fly the camera to this 3D point.
  focus?: { point: [number, number, number]; radius: number } | null;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let viewer: {
      start: () => void;
      addSplatScene: (path: string, options?: object) => Promise<unknown>;
      dispose: () => Promise<void>;
    } | null = null;

    async function boot() {
      if (!rootRef.current) return;
      setError(null);
      try {
        const GaussianSplats3D = await import("@mkkellogg/gaussian-splats-3d");
        if (!mounted || !rootRef.current) return;
        viewer = new GaussianSplats3D.Viewer({
          rootElement: rootRef.current,
          // Nerfstudio splatfacto scenes come out Z-up (verified: avg camera-up
          // ≈ [0,0,1] on every scene). The old [0,-1,-0.6] was a wrong tilted up
          // that rendered scenes upside-down. Z-up + a front-elevated start frame.
          cameraUp: [0, 0, 1],
          initialCameraPosition: [0, -3, 1.4],
          initialCameraLookAt: [0, 0, 0.2],
          // The page isn't cross-origin-isolated (no COOP/COEP headers), so
          // SharedArrayBuffer is disabled in the browser. The sort worker defaults
          // to it and hangs forever on "Processing splats…". Use the non-shared-
          // memory path — slightly slower sort, but it actually works.
          sharedMemoryForWorkers: false,
        });
        await viewer.addSplatScene(url, {
          showLoadingUI: true,
          progressiveLoad: true,
          format: format === "spz" ? GaussianSplats3D.SceneFormat.Spz : GaussianSplats3D.SceneFormat.Ply,
        });
        if (!mounted) {
          await viewer.dispose();
          return;
        }
        viewer.start();
        viewerRef.current = viewer;
      } catch (cause) {
        if (!mounted) return;
        setError(cause instanceof Error ? cause.message : "Could not load splat preview.");
      }
    }

    void boot();
    return () => {
      mounted = false;
      viewerRef.current = null;
      if (viewer) void viewer.dispose();
    };
  }, [url, format]);

  // Fly the camera to a search hit: recenter on the 3D point, keeping the current
  // viewing angle, pulled back to frame the match (radius-scaled distance).
  useEffect(() => {
    const v = viewerRef.current;
    if (!focus || !v || !v.camera || !v.controls) return;
    const [fx, fy, fz] = focus.point;
    const dist = Math.max(focus.radius * 3.5, 0.7);
    const cam = v.camera;
    const ctr = v.controls;
    let dx = cam.position.x - fx;
    let dy = cam.position.y - fy;
    let dz = cam.position.z - fz;
    let len = Math.hypot(dx, dy, dz);
    if (len < 1e-3) {
      dx = 0; dy = -1; dz = 0.6; len = Math.hypot(dx, dy, dz);
    }
    ctr.target.set(fx, fy, fz);
    cam.position.set(fx + (dx / len) * dist, fy + (dy / len) * dist, fz + (dz / len) * dist);
    ctr.update();
  }, [focus]);

  return (
    <div
      className={
        fill
          ? "relative h-full w-full overflow-hidden bg-black/70"
          : "relative h-[460px] overflow-hidden rounded-[24px] border border-white/10 bg-black/70"
      }
    >
      <div ref={rootRef} className="h-full w-full" />
      <ShortcutLegend />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/75 to-transparent px-4 py-3">
        <p className="text-[10px] font-bold uppercase tracking-[0.22em] text-white/65">Drag to orbit. Scroll to zoom.</p>
      </div>
      {error && (
        <div className="absolute inset-4 flex items-center justify-center rounded-2xl border border-red-500/25 bg-red-500/10 p-4 text-sm text-red-200">
          {error}
        </div>
      )}
    </div>
  );
}

// Floating shortcut reference — no background, tucked top-right, non-interactive.
function ShortcutLegend() {
  const k = "text-white/75";
  return (
    <div className="pointer-events-none absolute right-3 top-3 z-10 select-none text-right font-mono text-[10px] leading-[1.55] text-white/40">
      <div><span className={k}>drag</span> orbit · <span className={k}>scroll</span> zoom</div>
      <div><span className={k}>W A S D</span> pan · <span className={k}>← →</span> roll</div>
      <div><span className={k}>F</span>/<span className={k}>G</span> focal · <span className={k}>=</span>/<span className={k}>−</span> scale</div>
      <div><span className={k}>I</span> info · <span className={k}>C</span> cursor · <span className={k}>P</span> points · <span className={k}>O</span> ortho</div>
    </div>
  );
}
