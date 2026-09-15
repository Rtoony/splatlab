import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useLocation } from "wouter";
import * as THREE from "three";
import { ArrowLeft, ArrowRight, MapPin, Pause, Play, RotateCcw } from "lucide-react";
import { apiRequest } from "@/lib/api";

interface Capture {
  capture_id: string;
  duration_s: number;
  source: { name: string };
  camera_model: string | null;
  gps_count: number;
}

interface Point {
  index: number;
  time_s: number;
  status: string;
  location?: {
    latitude: number;
    longitude: number;
    alignment_verified: boolean;
  } | null;
}

interface RouteSummary {
  route_id: string;
  name: string;
  status: string;
  points: number;
  ready: number;
}

interface RouteDocument {
  route_id: string;
  name: string;
  status: string;
  points: Point[];
  error?: string;
  warnings: string[];
  reconstruction_prepared?: boolean;
}

function Panorama({ url }: { url: string }) {
  const host = useRef<HTMLDivElement>(null);
  const orientation = useRef({ yaw: 0, pitch: 0, fov: 75 });
  const [error, setError] = useState("");
  useEffect(() => {
    const container = host.current;
    if (!container) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch {
      container.textContent = "The panorama viewer needs WebGL. You can still open the image.";
      return () => {
        container.textContent = "";
      };
    }
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(orientation.current.fov, 1, 0.1, 100);
    const geometry = new THREE.SphereGeometry(10, 64, 32);
    geometry.scale(-1, 1, 1);
    const material = new THREE.MeshBasicMaterial();
    scene.add(new THREE.Mesh(geometry, material));
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    let disposed = false;
    new THREE.TextureLoader().load(
      url,
      (texture) => {
        if (disposed) {
          texture.dispose();
          return;
        }
        texture.colorSpace = THREE.SRGBColorSpace;
        material.map = texture;
        material.needsUpdate = true;
        setError("");
      },
      undefined,
      () => {
        if (!disposed) setError("This viewpoint could not be loaded. Resume the route to recover it.");
      },
    );
    let yaw = orientation.current.yaw;
    let pitch = orientation.current.pitch;
    let dragging = false;
    let previousX = 0;
    let previousY = 0;
    const down = (event: PointerEvent) => {
      dragging = true;
      previousX = event.clientX;
      previousY = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
    };
    const move = (event: PointerEvent) => {
      if (!dragging) return;
      yaw -= (event.clientX - previousX) * 0.003;
      pitch = THREE.MathUtils.clamp(pitch + (event.clientY - previousY) * 0.003, -1.45, 1.45);
      previousX = event.clientX;
      previousY = event.clientY;
    };
    const up = () => {
      dragging = false;
    };
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      camera.fov = THREE.MathUtils.clamp(camera.fov + event.deltaY * 0.04, 35, 100);
      camera.updateProjectionMatrix();
    };
    const resize = new ResizeObserver(() => {
      camera.aspect = container.clientWidth / Math.max(1, container.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    });
    resize.observe(container);
    renderer.domElement.style.touchAction = "none";
    renderer.domElement.addEventListener("pointerdown", down);
    renderer.domElement.addEventListener("pointermove", move);
    renderer.domElement.addEventListener("pointerup", up);
    renderer.domElement.addEventListener("pointercancel", up);
    renderer.domElement.addEventListener("wheel", wheel, { passive: false });
    renderer.setAnimationLoop(() => {
      camera.lookAt(Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw));
      renderer.render(scene, camera);
    });
    return () => {
      orientation.current = { yaw, pitch, fov: camera.fov };
      disposed = true;
      resize.disconnect();
      renderer.setAnimationLoop(null);
      renderer.domElement.removeEventListener("pointerdown", down);
      renderer.domElement.removeEventListener("pointermove", move);
      renderer.domElement.removeEventListener("pointerup", up);
      renderer.domElement.removeEventListener("pointercancel", up);
      renderer.domElement.removeEventListener("wheel", wheel);
      geometry.dispose();
      material.map?.dispose();
      material.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [url]);
  return (
    <div className="relative">
      <div
        ref={host}
        className="h-[60vh] min-h-80 w-full cursor-grab overflow-hidden rounded-2xl bg-black"
        aria-label="Drag to look around the captured panorama"
      />
      <p className="absolute left-4 top-4 rounded-lg bg-black/65 px-3 py-2 text-xs text-white">
        {error || "Drag to look around · scroll to zoom"}
      </p>
    </div>
  );
}

function RouteMap({
  points,
  selected,
  select,
}: {
  points: Point[];
  selected: number;
  select: (index: number) => void;
}) {
  const located = points.filter((point) => point.location);
  if (!located.length) return <p className="text-sm text-zinc-500">No GPS fixes aligned to these viewpoints.</p>;
  const latitude = located.reduce((sum, point) => sum + point.location!.latitude, 0) / located.length;
  const projected = located.map((point) => ({
    point,
    east: point.location!.longitude * Math.cos((latitude * Math.PI) / 180),
    north: point.location!.latitude,
  }));
  const west = Math.min(...projected.map((item) => item.east));
  const south = Math.min(...projected.map((item) => item.north));
  const extent = Math.max(
    0.00001,
    ...projected.map((item) => item.east - west),
    ...projected.map((item) => item.north - south),
  );
  const coordinates = projected.map((item) => ({
    ...item,
    horizontal: 20 + ((item.east - west) / extent) * 200,
    vertical: 220 - ((item.north - south) / extent) * 200,
  }));
  return (
    <svg viewBox="0 0 240 240" className="h-52 w-full rounded-xl bg-white/5" aria-label="GPS route, north up">
      <polyline
        points={coordinates.map((item) => item.horizontal + "," + item.vertical).join(" ")}
        fill="none"
        stroke="#22d3ee"
        strokeWidth="2"
      />
      {coordinates.map((item) => (
        <circle
          key={item.point.index}
          cx={item.horizontal}
          cy={item.vertical}
          r={item.point.index === selected ? 7 : 4}
          fill={item.point.index === selected ? "white" : "#22d3ee"}
          tabIndex={0}
          role="button"
          aria-label={"Viewpoint " + (item.point.index + 1)}
          onClick={() => select(item.point.index)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") select(item.point.index);
          }}
        />
      ))}
      <text x="220" y="18" fill="#a1a1aa" fontSize="12">
        N ↑
      </text>
    </svg>
  );
}

export default function RoutesPage() {
  const [location, navigate] = useLocation();
  const routeId = location.startsWith("/routes/") ? location.split("/")[2] : "";
  const queryClient = useQueryClient();
  const [captureId, setCaptureId] = useState("");
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(30);
  const [interval, setIntervalValue] = useState(5);
  const [pitch, setPitch] = useState(0);
  const [roll, setRoll] = useState(0);
  const [index, setIndex] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const captures = useQuery({
    queryKey: ["spatial-captures"],
    queryFn: () => apiRequest<{ captures: Capture[] }>("/api/splat/captures"),
    refetchInterval: 5000,
  });
  const list = useQuery({
    queryKey: ["spatial-routes"],
    queryFn: () => apiRequest<{ routes: RouteSummary[] }>("/api/splat/routes"),
    refetchInterval: 3000,
  });
  const atlases = useQuery({
    queryKey: ["capture-atlases"],
    queryFn: () =>
      apiRequest<{ atlases: { atlas_id: string; title: string; summary: { clips: number; review_views: number } }[] }>(
        "/api/splat/capture-atlases",
      ),
  });
  const route = useQuery({
    queryKey: ["spatial-route", routeId],
    queryFn: () => apiRequest<RouteDocument>("/api/splat/routes/" + routeId),
    enabled: Boolean(routeId),
    refetchInterval: 2000,
  });
  const transfers = useQuery({
    queryKey: ["spatial-transfers"],
    queryFn: () => apiRequest<{ entries: { path: string; name: string }[] }>("/api/splat/transfers"),
  });
  const ready = route.data?.points.filter((point) => point.status === "ready") || [];
  const current = ready.find((point) => point.index === index) || ready[0];
  const action = async (path: string, body?: unknown) => {
    setError("");
    setBusy(true);
    try {
      const result = await apiRequest<{ route_id?: string }>(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      if (result.route_id) {
        setIndex(0);
        navigate("/routes/" + result.route_id);
      }
      await queryClient.invalidateQueries({ queryKey: ["spatial-routes"] });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Operation failed");
    } finally {
      setBusy(false);
    }
  };
  const button =
    "inline-flex items-center gap-2 rounded-xl border border-white/15 px-3 py-2 text-sm disabled:opacity-40 hover:bg-white/10";
  const input = "min-w-0 w-full max-w-full rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm";
  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 sm:p-8">
      <div>
        <h1 className="display text-3xl font-black text-white">Explore your captures</h1>
        <p className="mt-2 text-sm text-zinc-400">
          Local 360 viewpoints linked to the original recording and GPS route.
        </p>
      </div>
      {(error || captures.error || list.error || route.error) && (
        <p role="alert" className="rounded-xl bg-red-500/10 p-4 text-sm text-red-300">
          {error || String(captures.error || list.error || route.error)}
        </p>
      )}
      {atlases.data?.atlases.map((atlas) => (
        <a
          key={atlas.atlas_id}
          href={`/api/splat/capture-atlases/${atlas.atlas_id}/index.html`}
          className="block rounded-2xl border border-cyan-400/30 bg-cyan-400/5 p-5 hover:bg-cyan-400/10"
        >
          <h2 className="text-lg font-bold text-cyan-200">{atlas.title}</h2>
          <p className="mt-2 text-sm text-zinc-400">
            {atlas.summary.clips} recordings · {atlas.summary.review_views} review views · GPS footprint and 360
            explorer →
          </p>
          <p className="mt-1 text-xs text-zinc-500">
            Private capture context, not calibrated reconstruction or model registration.
          </p>
        </a>
      ))}
      {routeId && route.data ? (
        <section className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <button className={button} onClick={() => navigate("/routes")}>
              <ArrowLeft size={15} />
              All routes
            </button>
            <span className="text-sm text-zinc-400">
              {route.data.status} · {ready.length}/{route.data.points.length} viewpoints
            </span>
            <div className="flex gap-2">
              <button
                className={button}
                disabled={busy || route.data.status === "running"}
                onClick={() => action("/api/splat/routes/" + routeId + "/resume")}
              >
                <Play size={15} />
                Resume
              </button>
              <button
                className={button}
                disabled={busy || route.data.status !== "running"}
                onClick={() => action("/api/splat/routes/" + routeId + "/stop")}
              >
                <Pause size={15} />
                Stop
              </button>
            </div>
          </div>
          {route.data.error && <p className="text-sm text-amber-300">{route.data.error}</p>}
          <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
            <button
              className={button}
              disabled={busy || route.data.status === "running" || ready.length < 3}
              onClick={() => action(`/api/splat/routes/${routeId}/reconstruction/prepare`, {})}
            >
              Prepare reconstruction inputs
            </button>
            {route.data.reconstruction_prepared && (
              <a
                href={`/api/splat/routes/${routeId}/reconstruction`}
                target="_blank"
                rel="noreferrer"
                className="text-cyan-300"
              >
                Open prepared plan
              </a>
            )}
            <span>Groups complete timestamps; excludes the lower 50° nadir cap. Does not start training.</span>
          </div>
          <details className="text-xs text-amber-200">
            <summary className="cursor-pointer">Capture limitations</summary>
            <ul className="mt-2 space-y-1">
              {route.data.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </details>
          <div className="grid gap-5 lg:grid-cols-[1fr_260px]">
            <div className="space-y-3">
              {current ? (
                <Panorama url={"/api/splat/routes/" + routeId + "/panoramas/" + current.index} />
              ) : (
                <div className="flex h-80 items-center justify-center rounded-2xl bg-white/5 text-zinc-400">
                  Preparing the first viewpoint…
                </div>
              )}
              {current && (
                <div className="flex items-center gap-3">
                  <button
                    aria-label="Previous viewpoint"
                    className={button}
                    disabled={current === ready[0]}
                    onClick={() => setIndex(ready[ready.indexOf(current) - 1].index)}
                  >
                    <ArrowLeft size={16} />
                  </button>
                  <input
                    aria-label="Viewpoint"
                    type="range"
                    min={0}
                    max={Math.max(0, ready.length - 1)}
                    value={ready.indexOf(current)}
                    onChange={(event) => setIndex(ready[Number(event.target.value)].index)}
                    className="min-w-0 flex-1"
                  />
                  <button
                    aria-label="Next viewpoint"
                    className={button}
                    disabled={current === ready[ready.length - 1]}
                    onClick={() => setIndex(ready[ready.indexOf(current) + 1].index)}
                  >
                    <ArrowRight size={16} />
                  </button>
                  <span className="text-xs text-zinc-400">{current.time_s.toFixed(1)}s</span>
                </div>
              )}
            </div>
            <aside className="space-y-4">
              <h2 className="flex items-center gap-2 font-semibold">
                <MapPin size={16} />
                Captured route
              </h2>
              <RouteMap points={ready} selected={current?.index ?? 0} select={setIndex} />
              <p className="text-xs leading-relaxed text-amber-200">
                {current?.location?.alignment_verified
                  ? "Clock alignment verified."
                  : "GPS alignment is approximate. Camera direction is independent of travel direction."}
              </p>
              {current?.location && (
                <p className="font-mono text-xs text-zinc-400">
                  {current.location.latitude.toFixed(6)}, {current.location.longitude.toFixed(6)}
                </p>
              )}
              {current && (
                <a
                  className="block text-xs text-cyan-300"
                  href={"/api/splat/routes/" + routeId + "/panoramas/" + current.index}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open panorama image
                </a>
              )}
              <p className="text-xs text-zinc-500">
                These viewpoints preserve captured content. Continuous 3D sections are not yet available for this route.
              </p>
            </aside>
          </div>
        </section>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
          <section className="min-w-0 space-y-4 rounded-2xl border border-white/10 p-5">
            <h2 className="text-lg font-bold">Build a route</h2>
            <label className="block min-w-0 space-y-1 text-xs text-zinc-400">
              Capture
              <select className={input} value={captureId} onChange={(event) => setCaptureId(event.target.value)}>
                <option value="">Select an inspected capture</option>
                {captures.data?.captures.map((capture) => (
                  <option key={capture.capture_id} value={capture.capture_id}>
                    {capture.source.name} · {Math.round(capture.duration_s)}s · {capture.gps_count} GPS fixes
                  </option>
                ))}
              </select>
            </label>
            <div className="grid grid-cols-2 gap-3">
              {[
                ["Start (seconds)", start, setStart],
                ["End (seconds)", end, setEnd],
                ["Spacing (seconds)", interval, setIntervalValue],
                ["Horizon pitch (°)", pitch, setPitch],
                ["Horizon roll (°)", roll, setRoll],
              ].map(([label, value, setter]) => (
                <label key={String(label)} className="space-y-1 text-xs text-zinc-400">
                  {String(label)}
                  <input
                    type="number"
                    className={input}
                    value={Number(value)}
                    onChange={(event) => (setter as (value: number) => void)(Number(event.target.value))}
                  />
                </label>
              ))}
            </div>
            <button
              className={button}
              disabled={busy || !captureId}
              onClick={() =>
                action("/api/splat/routes", {
                  capture_id: captureId,
                  start_s: start,
                  end_s: end,
                  interval_s: interval,
                  pitch_deg: pitch,
                  roll_deg: roll,
                })
              }
            >
              <Play size={15} />
              Build preview
            </button>
            <p className="text-xs text-zinc-500">
              Start with 30 seconds. Original video is preserved. Completed viewpoints survive interruptions.
            </p>
            <label className="block space-y-2 border-t border-white/10 pt-4 text-xs text-zinc-400">
              Inspect a Transfers video
              <select
                className={input}
                value=""
                disabled={busy}
                onChange={(event) => {
                  if (event.target.value)
                    void action("/api/splat/captures/inspect", {
                      input_path: event.target.value,
                    });
                }}
              >
                <option value="">Choose a file…</option>
                {transfers.data?.entries
                  ?.filter((entry) => /\.(insv|mp4|mov|mkv)$/i.test(entry.name))
                  .map((entry) => (
                    <option key={entry.path} value={entry.path}>
                      {entry.name}
                    </option>
                  ))}
              </select>
            </label>
          </section>
          <section className="space-y-3">
            <h2 className="text-lg font-bold">Your routes</h2>
            {list.data?.routes.map((item) => (
              <button
                key={item.route_id}
                onClick={() => {
                  setIndex(0);
                  navigate("/routes/" + item.route_id);
                }}
                className="flex w-full items-center gap-4 rounded-2xl border border-white/10 p-5 text-left hover:bg-white/5"
              >
                <MapPin className="shrink-0 text-cyan-300" />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-semibold">{item.name}</p>
                  <p className="mt-1 text-xs text-zinc-400">
                    {item.ready}/{item.points} viewpoints · {item.status}
                  </p>
                </div>
                <ArrowRight size={16} />
              </button>
            ))}
            {!list.data?.routes.length && (
              <p className="rounded-2xl bg-white/5 p-8 text-sm text-zinc-500">Build a preview to begin exploring.</p>
            )}
            <button
              className={button}
              onClick={() =>
                queryClient.invalidateQueries({
                  queryKey: ["spatial-captures"],
                })
              }
            >
              <RotateCcw size={14} />
              Refresh captures
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
