import { useEffect, useRef, useState } from "react";

// In-browser Gaussian-splat viewer (mkkellogg). Ported from the portal; renders
// a .ply (raw or web-optimized) or .spz. `fill` makes it fill its parent.
export type ViewerPoint = { point: [number, number, number]; radius: number };
export type ViewerOverlay = { matches: ViewerPoint[]; active: number; label: string } | null;
// A named, colored group of 3D points to highlight+label all at once (inventory legend).
export type ViewerHighlight = { label: string; color: string; points: [number, number, number][] };

export function SplatViewer({
  url,
  format = "ply",
  fill = false,
  focus = null,
  overlay = null,
  highlights = [],
  onPickMatch,
}: {
  url: string;
  format?: "ply" | "spz";
  fill?: boolean;
  // When set (from a language search hit), fly the camera to this 3D point.
  focus?: ViewerPoint | null;
  // When set, draw a highlight marker on each 3D match + a label on the active one.
  overlay?: ViewerOverlay;
  // Multiple colored object groups highlighted + labeled simultaneously (legend toggles).
  highlights?: ViewerHighlight[];
  onPickMatch?: (i: number) => void;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  const [error, setError] = useState<string | null>(null);
  // Screen positions of each 3D match, updated every frame from the camera.
  const [markers, setMarkers] = useState<{ x: number; y: number; front: boolean }[]>([]);
  // Screen positions of each highlight group's points (colored legend highlights).
  const [hlMarkers, setHlMarkers] = useState<{ label: string; color: string; pts: { x: number; y: number; front: boolean }[] }[]>([]);

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
    // Stand back far enough to see the match IN CONTEXT (its surroundings), not a
    // tight close-up. Bump FLY_ZOOM / FLY_MIN up for more room, down for tighter.
    const FLY_ZOOM = 8;
    const FLY_MIN = 1.8;
    const dist = Math.max(focus.radius * FLY_ZOOM, FLY_MIN);
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

  // Project each 3D match to screen every frame so the highlight markers + label
  // track the camera as you orbit. Uses the camera's own THREE math (no `three` import).
  useEffect(() => {
    if (!overlay || overlay.matches.length === 0) {
      setMarkers([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const v = viewerRef.current;
      const root = rootRef.current;
      if (v && v.camera && root) {
        const cam = v.camera;
        const rect = root.getBoundingClientRect();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const Vec3: any = cam.position.constructor;
        const fwd = new Vec3(0, 0, -1).applyQuaternion(cam.quaternion);
        setMarkers(
          overlay.matches.map((m) => {
            const dir = new Vec3(m.point[0] - cam.position.x, m.point[1] - cam.position.y, m.point[2] - cam.position.z);
            const front = dir.dot(fwd) > 0;
            const p = new Vec3(m.point[0], m.point[1], m.point[2]);
            p.project(cam);
            return { x: (p.x * 0.5 + 0.5) * rect.width, y: (-p.y * 0.5 + 0.5) * rect.height, front };
          }),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [overlay]);

  // Project every highlight group's 3D points to screen each frame (legend toggles).
  // Separate from the search overlay so both can be shown at once without interfering.
  useEffect(() => {
    if (!highlights || highlights.length === 0) {
      setHlMarkers([]);
      return;
    }
    let raf = 0;
    const tick = () => {
      const v = viewerRef.current;
      const root = rootRef.current;
      if (v && v.camera && root) {
        const cam = v.camera;
        const rect = root.getBoundingClientRect();
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const Vec3: any = cam.position.constructor;
        const fwd = new Vec3(0, 0, -1).applyQuaternion(cam.quaternion);
        setHlMarkers(
          highlights.map((h) => ({
            label: h.label,
            color: h.color,
            pts: h.points.map((pt) => {
              const dir = new Vec3(pt[0] - cam.position.x, pt[1] - cam.position.y, pt[2] - cam.position.z);
              const front = dir.dot(fwd) > 0;
              const p = new Vec3(pt[0], pt[1], pt[2]);
              p.project(cam);
              return { x: (p.x * 0.5 + 0.5) * rect.width, y: (-p.y * 0.5 + 0.5) * rect.height, front };
            }),
          })),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [highlights]);

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
      {overlay &&
        markers.map((mk, i) =>
          mk.front ? (
            <button
              key={i}
              type="button"
              onClick={() => onPickMatch?.(i)}
              style={{ left: `${mk.x}px`, top: `${mk.y}px` }}
              title={`${overlay.label} — ${i + 1}/${overlay.matches.length}`}
              className={`pointer-events-auto absolute z-20 -translate-x-1/2 -translate-y-1/2 rounded-full transition ${
                i === overlay.active
                  ? "h-6 w-6 border-2 border-cyan-300 bg-cyan-300/25 shadow-[0_0_14px_rgba(34,211,238,0.75)]"
                  : "h-4 w-4 border-2 border-white/60 bg-white/10 hover:border-cyan-200 hover:bg-cyan-200/20"
              }`}
            />
          ) : null,
        )}
      {overlay && markers[overlay.active]?.front && (
        <div
          style={{ left: `${markers[overlay.active].x}px`, top: `${markers[overlay.active].y - 20}px` }}
          className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-md bg-black/75 px-2 py-0.5 text-xs font-semibold text-cyan-100 shadow backdrop-blur-sm"
        >
          {overlay.label}
        </div>
      )}
      {/* Legend highlights: colored dots on every instance of each toggled object,
          plus one label per object at its first on-screen instance. */}
      {hlMarkers.map((h, gi) => (
        <div key={gi}>
          {h.pts.map((pt, pi) =>
            pt.front ? (
              <div
                key={pi}
                style={{ left: `${pt.x}px`, top: `${pt.y}px`, backgroundColor: `${h.color}40`, borderColor: h.color, boxShadow: `0 0 10px ${h.color}` }}
                className="pointer-events-none absolute z-20 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2"
              />
            ) : null,
          )}
          {(() => {
            const lead = h.pts.find((p) => p.front);
            return lead ? (
              <div
                style={{ left: `${lead.x}px`, top: `${lead.y - 14}px`, color: h.color, borderColor: `${h.color}80` }}
                className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-full whitespace-nowrap rounded-md border bg-black/80 px-1.5 py-0.5 text-[11px] font-bold shadow backdrop-blur-sm"
              >
                {h.label}
              </div>
            ) : null;
          })()}
        </div>
      ))}
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
