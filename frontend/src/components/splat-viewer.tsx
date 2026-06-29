import { useEffect, useRef, useState } from "react";

// In-browser Gaussian-splat viewer (mkkellogg). Ported from the portal; renders
// a .ply (raw or web-optimized) or .spz. `fill` makes it fill its parent.
export function SplatViewer({
  url,
  format = "ply",
  fill = false,
}: {
  url: string;
  format?: "ply" | "spz";
  fill?: boolean;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
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
          cameraUp: [0, -1, -0.6],
          initialCameraPosition: [-1, -4, 6],
          initialCameraLookAt: [0, 2, 0],
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
      } catch (cause) {
        if (!mounted) return;
        setError(cause instanceof Error ? cause.message : "Could not load splat preview.");
      }
    }

    void boot();
    return () => {
      mounted = false;
      if (viewer) void viewer.dispose();
    };
  }, [url, format]);

  return (
    <div
      className={
        fill
          ? "relative h-full w-full overflow-hidden bg-black/70"
          : "relative h-[460px] overflow-hidden rounded-[24px] border border-white/10 bg-black/70"
      }
    >
      <div ref={rootRef} className="h-full w-full" />
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
