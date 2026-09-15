import { useEffect, useState, type RefObject } from "react";
import { apiRequest } from "@/lib/api";
import type { SplatGeoFootprint } from "@/lib/contracts";
import { Button, Card } from "@/components/ui";

type Control = { scene: [number, number]; lat: number; lon: number };
type Fit = {
  meters_per_unit: number;
  geo: { heading_deg: number };
  rms_m: number;
  residuals_m: number[];
  redundant_control_check: boolean;
  base_scale_generation: number;
};

export default function GeoControlsPanel({
  jobId,
  footprint,
  mapPickRef,
  onApplied,
}: {
  jobId: string;
  footprint: SplatGeoFootprint;
  mapPickRef: RefObject<((latitude: number, longitude: number) => void) | null>;
  onApplied: () => void;
}) {
  const [enabled, setEnabled] = useState(false);
  const [controls, setControls] = useState<Control[]>([]);
  const [pending, setPending] = useState<[number, number] | null>(null);
  const [fit, setFit] = useState<Fit | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [force, setForce] = useState(false);

  useEffect(() => {
    mapPickRef.current = enabled
      ? (latitude, longitude) => {
          if (!pending || controls.length >= 20) return;
          setControls((previous) => [
            ...previous,
            { scene: pending, lat: latitude, lon: longitude },
          ]);
          setPending(null);
          setFit(null);
        }
      : null;
    return () => {
      mapPickRef.current = null;
    };
  }, [mapPickRef, enabled, pending, controls.length]);

  const propose = async () => {
    setBusy(true);
    setError("");
    setFit(null);
    try {
      setFit(
        await apiRequest<Fit>(`/api/splat/jobs/${jobId}/geo/controls/propose`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ controls }),
        }),
      );
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  };

  const apply = async () => {
    if (!fit) return;
    setBusy(true);
    setError("");
    try {
      await apiRequest(`/api/splat/jobs/${jobId}/geo/controls/apply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          controls,
          expected_scale_generation: fit.base_scale_generation,
          force,
        }),
      });
      onApplied();
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  };

  if (
    !footprint.url ||
    footprint.x0 == null ||
    footprint.x1 == null ||
    footprint.y0 == null ||
    footprint.y1 == null
  )
    return null;
  const { x0, x1, y0, y1 } = footprint;
  const markers = [
    ...controls.map((control) => control.scene),
    ...(pending ? [pending] : []),
  ];
  return (
    <Card className="space-y-2 border-cyan-400/20 bg-cyan-400/5 p-2 text-xs">
      <label className="flex items-center gap-2 font-semibold">
        <input
          type="checkbox"
          checked={enabled}
          disabled={busy}
          onChange={(event) => setEnabled(event.target.checked)}
        />
        Align matching points
      </label>
      {enabled && (
        <>
          <p className="text-zinc-400">
            Pick a recognizable point in this unrotated footprint, then its
            matching location on the map. Use three or more well-spaced pairs to
            check consistency.
          </p>
          <div className="relative">
            <img
              src={footprint.url}
              alt="Scene footprint: click to select a control point"
              className="w-full cursor-crosshair rounded bg-black/40"
              draggable={false}
              onClick={(event) => {
                if (busy || controls.length >= 20) return;
                const bounds = event.currentTarget.getBoundingClientRect();
                setPending([
                  x0 +
                    ((event.clientX - bounds.left) / bounds.width) * (x1 - x0),
                  y1 -
                    ((event.clientY - bounds.top) / bounds.height) * (y1 - y0),
                ]);
                setFit(null);
              }}
            />
            {markers.map((point, index) => (
              <span
                key={index}
                className="pointer-events-none absolute grid h-5 w-5 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-cyan-300 text-[10px] font-bold text-black"
                style={{
                  left: `${((point[0] - x0) / (x1 - x0)) * 100}%`,
                  top: `${((y1 - point[1]) / (y1 - y0)) * 100}%`,
                }}
              >
                {index + 1}
              </span>
            ))}
          </div>
          <p aria-live="polite" className="text-cyan-200">
            {pending
              ? "Now click the matching point on the map."
              : `${controls.length} matched pairs. Pick the next scene point.`}
          </p>
          {controls.map((control, index) => (
            <div
              key={index}
              className="flex items-center justify-between gap-1 text-[10px] tabular-nums"
            >
              <span>
                {index + 1}: {control.lat.toFixed(6)}, {control.lon.toFixed(6)}
                {fit
                  ? ` · ${fit.residuals_m[index].toFixed(2)} m residual`
                  : ""}
              </span>
              <button
                type="button"
                disabled={busy}
                aria-label={`Remove control ${index + 1}`}
                onClick={() => {
                  setControls((previous) =>
                    previous.filter((_, item) => item !== index),
                  );
                  setFit(null);
                }}
              >
                Remove
              </button>
            </div>
          ))}
          <Button
            type="button"
            disabled={busy || !!pending || controls.length < 2}
            onClick={propose}
            className="w-full"
          >
            Preview alignment
          </Button>
          {fit && (
            <>
              <p>
                Scale {fit.meters_per_unit.toFixed(4)} m/unit · heading{" "}
                {fit.geo.heading_deg.toFixed(2)}° · RMS {fit.rms_m.toFixed(2)} m
              </p>
              <p className="text-amber-200">
                {fit.redundant_control_check
                  ? "Residuals measure agreement, not survey accuracy."
                  : "Two pairs fit exactly: add a third to check consistency."}{" "}
                Horizontal alignment only; elevation is unchanged.
              </p>
              <label className="flex items-start gap-2 text-zinc-400">
                <input
                  type="checkbox"
                  checked={force}
                  onChange={(event) => setForce(event.target.checked)}
                />
                Allow replacing stronger scale evidence
              </label>
              <Button
                type="button"
                disabled={busy}
                onClick={apply}
                className="w-full"
              >
                Apply point alignment
              </Button>
            </>
          )}
          {error && (
            <p role="alert" className="text-red-300">
              {error}
            </p>
          )}
        </>
      )}
    </Card>
  );
}
