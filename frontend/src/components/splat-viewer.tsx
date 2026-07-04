import { useEffect, useRef, useState } from "react";

// In-browser Gaussian-splat viewer (mkkellogg). Ported from the portal; renders
// a .ply (raw or web-optimized) or .spz. `fill` makes it fill its parent.
export type ViewerPoint = { point: [number, number, number]; radius: number };
export type ViewerOverlay = { matches: ViewerPoint[]; active: number; label: string } | null;
// A named, colored group of 3D points to highlight+label all at once (inventory legend).
export type ViewerHighlight = { label: string; color: string; points: [number, number, number][] };
export type ViewerCameraPose = {
  index: number;
  image_name: string;
  position: [number, number, number];
  forward: [number, number, number];
  up: [number, number, number];
  right: [number, number, number];
  fov_y_degrees?: number | null;
};
export type ViewerCameraOverlay = { cameras: ViewerCameraPose[]; displayScale: number; frame: "viewer" | "source" } | null;
export type ViewerCameraViewTarget = { camera: ViewerCameraPose; token: number; distance?: number } | null;
export type ViewerCameraNodeTarget = { camera: ViewerCameraPose; token: number; distance?: number } | null;

const INITIAL_CAMERA_POSITION: [number, number, number] = [0, -3, 1.4];
const INITIAL_CAMERA_LOOK_AT: [number, number, number] = [0, 0, 0.2];
const INITIAL_CAMERA_UP: [number, number, number] = [0, 0, 1];

export function SplatViewer({
  url,
  format = "ply",
  fill = false,
  focus = null,
  overlay = null,
  highlights = [],
  cameraOverlay = null,
  viewCamera = null,
  cameraNodeTarget = null,
  resetViewToken = 0,
  showShortcutLegend = false,
  onPickMatch,
  onPickCamera,
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
  // Capture camera positions/directions projected over the current viewer camera.
  cameraOverlay?: ViewerCameraOverlay;
  // One-shot request to view the scene from an original capture camera pose.
  viewCamera?: ViewerCameraViewTarget;
  // One-shot request to inspect a camera marker from just behind its original pose.
  cameraNodeTarget?: ViewerCameraNodeTarget;
  resetViewToken?: number;
  showShortcutLegend?: boolean;
  onPickMatch?: (i: number) => void;
  onPickCamera?: (camera: ViewerCameraPose) => void;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  const defaultFovRef = useRef<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Screen positions of each 3D match, updated every frame from the camera.
  const [markers, setMarkers] = useState<{ x: number; y: number; front: boolean }[]>([]);
  // Screen positions of each highlight group's points (colored legend highlights).
  const [hlMarkers, setHlMarkers] = useState<{ label: string; color: string; pts: { x: number; y: number; front: boolean }[] }[]>([]);
  const [cameraMarkers, setCameraMarkers] = useState<
    {
      index: number;
      name: string;
      x: number;
      y: number;
      front: boolean;
      nose: { x: number; y: number; front: boolean };
      left: { x: number; y: number; front: boolean };
      right: { x: number; y: number; front: boolean };
      top: { x: number; y: number; front: boolean };
      camera: ViewerCameraPose;
    }[]
  >([]);

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
          cameraUp: INITIAL_CAMERA_UP,
          initialCameraPosition: INITIAL_CAMERA_POSITION,
          initialCameraLookAt: INITIAL_CAMERA_LOOK_AT,
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
        if (viewer.camera && typeof viewer.camera.fov === "number") defaultFovRef.current = viewer.camera.fov;
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

  // Restore the viewer's default camera/orbit. Parent state clears overlays separately.
  useEffect(() => {
    const v = viewerRef.current;
    if (!resetViewToken || !v || !v.camera || !v.controls) return;
    const cam = v.camera;
    const ctr = v.controls;
    if (typeof defaultFovRef.current === "number") {
      cam.fov = defaultFovRef.current;
      if (typeof cam.updateProjectionMatrix === "function") cam.updateProjectionMatrix();
    }
    cam.position.set(...INITIAL_CAMERA_POSITION);
    cam.up.set(...INITIAL_CAMERA_UP);
    ctr.target.set(...INITIAL_CAMERA_LOOK_AT);
    if (typeof cam.lookAt === "function") cam.lookAt(ctr.target);
    ctr.update();
  }, [resetViewToken]);

  // Fly the camera to a search hit: recenter on the 3D point, keeping the current
  // viewing angle, pulled back to frame the match (radius-scaled distance).
  useEffect(() => {
    const v = viewerRef.current;
    if (!focus || !v || !v.camera || !v.controls) return;
    const [fx, fy, fz] = focus.point;
    // Stand back to frame the target by its extent: small item -> close, large space ->
    // zoom-to-extents. FLY_MAX caps big legend objects (a floor) so we don't fly outside
    // the scene. (Search matches are small, so their feel is unchanged.)
    const FLY_ZOOM = 8;
    const FLY_MIN = 1.8;
    const FLY_MAX = 16;
    const dist = Math.min(Math.max(focus.radius * FLY_ZOOM, FLY_MIN), FLY_MAX);
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

  // Tight camera-node zoom: unlike generic search/object focus, this aligns the
  // viewer behind the original camera and keeps the orbit pivot on the node.
  useEffect(() => {
    const v = viewerRef.current;
    if (!cameraNodeTarget || !v || !v.camera || !v.controls) return;
    const cam = v.camera;
    const ctr = v.controls;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Vec3: any = cam.position.constructor;
    const p = cameraNodeTarget.camera.position;
    const f = cameraNodeTarget.camera.forward;
    const u = cameraNodeTarget.camera.up;
    const node = new Vec3(p[0], p[1], p[2]);
    const forward = new Vec3(f[0], f[1], f[2]);
    if (typeof forward.lengthSq === "function" && forward.lengthSq() > 1e-10) forward.normalize();
    else forward.set(0, -1, 0);
    const up = new Vec3(u[0], u[1], u[2]);
    if (typeof up.lengthSq === "function" && up.lengthSq() > 1e-10) up.normalize();
    else up.set(...INITIAL_CAMERA_UP);
    const distance = Math.max(cameraNodeTarget.distance ?? 0.35, 0.05);
    const pos = node.clone().add(forward.multiplyScalar(-distance));

    if (typeof defaultFovRef.current === "number") {
      cam.fov = defaultFovRef.current;
      if (typeof cam.updateProjectionMatrix === "function") cam.updateProjectionMatrix();
    }
    cam.position.copy ? cam.position.copy(pos) : cam.position.set(pos.x, pos.y, pos.z);
    cam.up.copy ? cam.up.copy(up) : cam.up.set(up.x, up.y, up.z);
    ctr.target.copy ? ctr.target.copy(node) : ctr.target.set(p[0], p[1], p[2]);
    if (typeof cam.lookAt === "function") cam.lookAt(node);
    ctr.update();
  }, [cameraNodeTarget]);

  // Jump to an original capture camera pose. The target token lets the parent
  // request the same camera repeatedly.
  useEffect(() => {
    const v = viewerRef.current;
    if (!viewCamera || !v || !v.camera || !v.controls) return;
    const cam = v.camera;
    const ctr = v.controls;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const Vec3: any = cam.position.constructor;
    const p = viewCamera.camera.position;
    const f = viewCamera.camera.forward;
    const u = viewCamera.camera.up;
    const pos = new Vec3(p[0], p[1], p[2]);
    const forward = new Vec3(f[0], f[1], f[2]).normalize();
    const up = new Vec3(u[0], u[1], u[2]).normalize();
    const targetDistance = Math.max(viewCamera.distance ?? 1.2, 0.2);
    const target = pos.clone().add(forward.multiplyScalar(targetDistance));

    if (typeof viewCamera.camera.fov_y_degrees === "number" && Number.isFinite(viewCamera.camera.fov_y_degrees)) {
      cam.fov = viewCamera.camera.fov_y_degrees;
      if (typeof cam.updateProjectionMatrix === "function") cam.updateProjectionMatrix();
    }
    cam.position.copy ? cam.position.copy(pos) : cam.position.set(p[0], p[1], p[2]);
    cam.up.copy ? cam.up.copy(up) : cam.up.set(u[0], u[1], u[2]);
    ctr.target.copy ? ctr.target.copy(target) : ctr.target.set(target.x, target.y, target.z);
    if (typeof cam.lookAt === "function") cam.lookAt(target);
    ctr.update();
  }, [viewCamera]);

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

  // Project capture camera poses into screen-space. These are annotations only; they
  // don't modify the underlying GaussianSplats3D camera or controls.
  useEffect(() => {
    if (!cameraOverlay || cameraOverlay.cameras.length === 0) {
      setCameraMarkers([]);
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
        const viewForward = new Vec3(0, 0, -1).applyQuaternion(cam.quaternion);
        const scale = Math.max(cameraOverlay.displayScale || 0.08, 0.01);
        const add = (a: [number, number, number], b: [number, number, number], m: number): [number, number, number] => [
          a[0] + b[0] * m,
          a[1] + b[1] * m,
          a[2] + b[2] * m,
        ];
        const project = (pt: [number, number, number]) => {
          const dir = new Vec3(pt[0] - cam.position.x, pt[1] - cam.position.y, pt[2] - cam.position.z);
          const front = dir.dot(viewForward) > 0;
          const p = new Vec3(pt[0], pt[1], pt[2]);
          p.project(cam);
          return { x: (p.x * 0.5 + 0.5) * rect.width, y: (-p.y * 0.5 + 0.5) * rect.height, front };
        };

        setCameraMarkers(
          cameraOverlay.cameras.map((c) => {
            const nose = add(c.position, c.forward, scale * 1.8);
            const left = add(nose, c.right, -scale * 0.55);
            const right = add(nose, c.right, scale * 0.55);
            const top = add(nose, c.up, scale * 0.45);
            const center = project(c.position);
            return {
              index: c.index,
              name: c.image_name,
              ...center,
              nose: project(nose),
              left: project(left),
              right: project(right),
              top: project(top),
              camera: c,
            };
          }),
        );
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [cameraOverlay]);

  return (
    <div
      className={
        fill
          ? "relative h-full w-full overflow-hidden bg-black/70"
          : "relative h-[460px] overflow-hidden rounded-[24px] border border-white/10 bg-black/70"
      }
    >
      <div ref={rootRef} className="h-full w-full" />
      {showShortcutLegend && <ShortcutLegend />}
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
      {/* Legend highlights: a soft AREA wash — many translucent screen-blended dots over
          the object's own gaussians build up into a colored region — + one label at the
          region's centroid. "tile" paints the whole floor, not a pin. */}
      {hlMarkers.map((h, gi) => {
        const front = h.pts.filter((p) => p.front);
        if (!front.length) return null;
        const cx = front.reduce((s, p) => s + p.x, 0) / front.length;
        const cy = front.reduce((s, p) => s + p.y, 0) / front.length;
        return (
          <div key={gi}>
            {front.map((pt, pi) => (
              <div
                key={pi}
                style={{ left: `${pt.x}px`, top: `${pt.y}px`, backgroundColor: `${h.color}4d`, mixBlendMode: "screen" }}
                className="pointer-events-none absolute z-20 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
              />
            ))}
            <div
              style={{ left: `${cx}px`, top: `${cy}px`, color: h.color, borderColor: `${h.color}80` }}
              className="pointer-events-none absolute z-30 -translate-x-1/2 -translate-y-1/2 whitespace-nowrap rounded-md border bg-black/80 px-1.5 py-0.5 text-[11px] font-bold shadow backdrop-blur-sm"
            >
              {h.label}
            </div>
          </div>
        );
      })}
      {cameraOverlay && cameraMarkers.length > 0 && (
        <svg className="pointer-events-none absolute inset-0 z-20 h-full w-full" aria-hidden="true">
          {cameraMarkers.map((mk) =>
            mk.front ? (
              <g key={mk.index} opacity={cameraOverlay.frame === "viewer" ? 0.82 : 0.45}>
                <line x1={mk.x} y1={mk.y} x2={mk.nose.x} y2={mk.nose.y} stroke="#fbbf24" strokeWidth="1.3" strokeLinecap="round" />
                <line x1={mk.x} y1={mk.y} x2={mk.left.x} y2={mk.left.y} stroke="#f59e0b" strokeWidth="0.9" strokeLinecap="round" />
                <line x1={mk.x} y1={mk.y} x2={mk.right.x} y2={mk.right.y} stroke="#f59e0b" strokeWidth="0.9" strokeLinecap="round" />
                <line x1={mk.x} y1={mk.y} x2={mk.top.x} y2={mk.top.y} stroke="#fde68a" strokeWidth="0.8" strokeLinecap="round" />
              </g>
            ) : null,
          )}
        </svg>
      )}
      {cameraMarkers.map((mk) =>
        mk.front ? (
          <button
            key={mk.index}
            type="button"
            title={`Zoom to ${mk.name} · camera ${mk.index + 1}`}
            onClick={() => onPickCamera?.(mk.camera)}
            style={{ left: `${mk.x}px`, top: `${mk.y}px` }}
            className="pointer-events-auto absolute z-30 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-amber-100/80 bg-amber-300/75 shadow-[0_0_10px_rgba(251,191,36,0.65)] transition hover:h-4 hover:w-4 hover:bg-cyan-200"
          />
        ) : null,
      )}
      {cameraOverlay && cameraMarkers.length > 0 && (
        <div className="pointer-events-none absolute right-3 top-24 z-20 rounded-full border border-amber-300/25 bg-black/55 px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.2em] text-amber-100 backdrop-blur">
          {cameraMarkers.length} cameras
        </div>
      )}
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
