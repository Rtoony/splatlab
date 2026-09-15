import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { Dialog } from "@/components/ui/dialog";
import { nearestSupportFeature, type SupportFeature, type SupportAnchors } from "@/lib/support-anchor-picking";

type Observations = {
  evidence_sha256: string;
  image_id: number;
  photo_sha256: string;
  width: number;
  height: number;
  features: SupportFeature[];
  cameras: { image_id: number; name: string; regional_features: number }[];
};

export function SupportSurfacePicker({
  jobId,
  generation,
  selected,
  disabled,
  anchors,
  onChange,
}: {
  jobId: string;
  generation: number;
  selected: string;
  disabled: boolean;
  anchors: SupportAnchors | null;
  onChange: (value: SupportAnchors | null) => void;
}) {
  const [imageId, setImageId] = useState<number | null>(null);
  const [message, setMessage] = useState("");
  const [typedPoint, setTypedPoint] = useState("");
  const [loadedPhoto, setLoadedPhoto] = useState("");
  const [expanded, setExpanded] = useState(false);
  const base = `/api/splat/jobs/${jobId}/studio/recovery-support`;
  const parameters = new URLSearchParams({ selected_slug: selected, expected_generation: String(generation) });
  const query = useQuery({
    queryKey: ["support-observations", jobId, generation, selected, imageId],
    queryFn: () => apiRequest<Observations>(`${base}?${parameters}${imageId ? `&image_id=${imageId}` : ""}`),
    enabled: Boolean(selected),
    retry: (failures, error) => failures < 2 && error.message.includes("background recovery is already running"),
    retryDelay: 750,
    staleTime: Infinity,
  });
  const data = query.data;
  const picks = anchors?.points || [];
  const stale = Boolean(data && anchors && data.evidence_sha256 !== anchors.evidence_sha256);
  const add = (feature: SupportFeature | null) => {
    if (!feature || !data || disabled || stale || loadedPhoto !== data.photo_sha256) {
      setMessage(
        "Wait for the photo, then choose a tracked dot within 12 screen pixels; arbitrary photo pixels have no verified depth.",
      );
      return;
    }
    if (picks.some((point) => point.point_id === feature.point_id)) {
      setMessage("That tracked point is already selected.");
      return;
    }
    if (picks.length >= 6) {
      setMessage("Use at most six anchors; remove one before choosing another.");
      return;
    }
    onChange({
      evidence_sha256: data.evidence_sha256,
      points: [...picks, { point_id: feature.point_id, image_id: data.image_id, photo_sha256: data.photo_sha256 }],
    });
    setMessage("");
  };
  const content = (
    <section className="mx-auto w-full max-w-[1100px] space-y-3 rounded-xl border border-sky-300/20 p-3">
      <h4 className="text-sm font-semibold">Choose the support surface from photos</h4>
      <p className="text-xs text-zinc-400">
        Pick three to six well-spread tracked dots on the same visible floor or support surface, not on packaging or
        furniture. This anchors a fitted plane; hidden geometry remains an extrapolation.
      </p>
      {query.isFetching && <p className="text-xs text-zinc-400">Loading tracked observations…</p>}
      {query.error && (
        <p role="alert" className="text-xs text-red-300">
          {String(query.error)}
        </p>
      )}
      {data && (
        <>
          <label className="block text-xs text-zinc-400">
            Support reference photograph
            <select
              aria-label="Support reference photograph"
              className="mt-1 w-full rounded-lg bg-zinc-950 p-2"
              disabled={disabled}
              value={data.image_id}
              onChange={(event) => {
                setImageId(Number(event.target.value));
                setMessage("");
              }}
            >
              {data.cameras.map((camera) => (
                <option key={camera.image_id} value={camera.image_id}>
                  {camera.name} · {camera.regional_features} nearby tracks
                </option>
              ))}
            </select>
          </label>
          <svg
            aria-label="Tracked support observations"
            role="img"
            viewBox={`0 0 ${data.width} ${data.height}`}
            className="w-full cursor-crosshair rounded-lg bg-zinc-900"
            onClick={(event) => {
              const rectangle = event.currentTarget.getBoundingClientRect();
              add(
                nearestSupportFeature(
                  data.features,
                  [
                    ((event.clientX - rectangle.left) / rectangle.width) * data.width,
                    ((event.clientY - rectangle.top) / rectangle.height) * data.height,
                  ],
                  data.width / rectangle.width,
                ),
              );
            }}
          >
            <image
              href={`${base}/${data.image_id}/photo?${parameters}&photo_sha256=${data.photo_sha256}`}
              width={data.width}
              height={data.height}
              onLoad={() => setLoadedPhoto(data.photo_sha256)}
              onError={() => {
                setLoadedPhoto("");
                setMessage("The source photo could not load. Reload before choosing anchors.");
              }}
            />
            {data.features.map((feature) => (
              <circle
                key={feature.point_id}
                cx={feature.pixel[0]}
                cy={feature.pixel[1]}
                r={picks.some((point) => point.point_id === feature.point_id) ? 8 : 2.5}
                fill={picks.some((point) => point.point_id === feature.point_id) ? "#facc15" : "#38bdf8"}
                fillOpacity={0.8}
              >
                <title>{`Point ${feature.point_id} · Y ${feature.world[1].toFixed(3)} m · reprojection ${feature.reprojection_px.toFixed(2)} px`}</title>
              </circle>
            ))}
          </svg>
          <div className="flex gap-2">
            <input
              aria-label="Tracked support point ID"
              placeholder="Tracked point ID"
              value={typedPoint}
              disabled={disabled}
              onChange={(event) => setTypedPoint(event.target.value)}
              className="min-w-0 flex-1 rounded-lg bg-zinc-950 p-2 text-xs"
            />
            <button
              type="button"
              disabled={disabled || stale || loadedPhoto !== data.photo_sha256}
              className="rounded-lg border border-white/15 px-3 py-2 text-xs"
              onClick={() => add(data.features.find((feature) => feature.point_id === typedPoint.trim()) || null)}
            >
              Add tracked point
            </button>
          </div>
          <p className="text-xs text-sky-200">
            {picks.length} support anchors selected · {data.features.length} verified photo observations shown
          </p>
          <div className="flex flex-wrap gap-2">
            {picks.map((point, index) => (
              <button
                type="button"
                key={point.point_id}
                disabled={disabled}
                className="rounded-lg border border-yellow-300/30 px-2 py-1 text-xs"
                aria-label={`Remove support anchor ${point.point_id}`}
                onClick={() =>
                  onChange({
                    evidence_sha256: anchors!.evidence_sha256,
                    points: picks.filter((item) => item.point_id !== point.point_id),
                  })
                }
              >
                {index + 1}: {point.point_id} · photo {point.image_id} ×
              </button>
            ))}
          </div>
        </>
      )}
      {(message || stale) && (
        <p role="alert" className="text-xs text-amber-200">
          {stale ? "The reconstruction changed. Clear the anchors and reload observations." : message}
        </p>
      )}
      <button
        type="button"
        disabled={disabled}
        className="text-xs text-zinc-400 underline"
        onClick={() => {
          onChange(null);
          setMessage("");
          void query.refetch();
        }}
      >
        Clear anchors and reload
      </button>
    </section>
  );
  return (
    <div className="space-y-2">
      <button
        type="button"
        disabled={disabled}
        aria-expanded={expanded}
        onClick={() => setExpanded(true)}
        className="rounded-lg border border-sky-300/30 px-3 py-2 text-xs text-sky-200"
      >
        Expand photo picker
      </button>
      {expanded ? (
        <Dialog open onOpenChange={setExpanded} title="Choose support surface" layout="screen">
          <div className="overflow-y-auto p-4 sm:p-8">{content}</div>
        </Dialog>
      ) : (
        content
      )}
    </div>
  );
}
