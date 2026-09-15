import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { GeneratedSplatInspection } from "./generated-splat-inspection";
import type { GeneratedGaussianChoice } from "@/lib/generated-objects";

type Review = GeneratedGaussianChoice & {
  gaussians?: number;
  error?: string;
  mesh_color?: { policy: string };
  views: { image_id: number; split: string; splat_mask_iou: number; mesh_splat_silhouette_iou: number }[];
};

export function GeneratedGaussianPanel({
  jobId,
  objectId,
  canPlace,
  canReplace,
  onPropose,
}: {
  jobId: string;
  objectId: string;
  canPlace: boolean;
  canReplace: boolean;
  onPropose: (replace: boolean, gaussian: GeneratedGaussianChoice) => void;
}) {
  const base = `/api/splat/jobs/${jobId}/studio/generated-objects/${objectId}`;
  const [shown, setShown] = useState("");
  const query = useQuery({
    queryKey: ["generated-gaussians", jobId, objectId],
    queryFn: () => apiRequest<{ reviews: Review[] }>(base + "/gaussians"),
  });
  const reviews = query.data?.reviews || [];
  const review = reviews.find((item) => item.gaussians_id === shown) || reviews.at(-1);
  const artifact = (name: string) =>
    `${base}/gaussians/${review?.gaussians_id}/artifact?name=${encodeURIComponent(name)}`;
  return (
    <section aria-label="Native Gaussian frame review" className="space-y-3 border-t border-white/10 pt-3">
      <h4 className="text-sm font-semibold">Native Gaussian frame review</h4>
      <p className="text-xs text-amber-200">
        This separately retained derivative rotates Gaussian covariances and scales, not only centers. Historical review
        does not activate it in the room or validate collision. Check masks are separate from pose optimization, not all
        upstream support evidence.
      </p>
      <button className="rounded-lg border border-white/15 px-3 py-2 text-xs" onClick={() => void query.refetch()}>
        Refresh Gaussian reviews
      </button>
      {query.error && <p role="alert">{query.error.message}</p>}
      {!review && (
        <p className="text-xs text-zinc-400">
          No retained native-splat placement yet; the agent runs the bounded local verification worker.
        </p>
      )}
      {review && (
        <>
          <select
            aria-label="Retained Gaussian frame review"
            className="w-full min-w-0 rounded bg-zinc-900 p-2 text-xs"
            value={review.gaussians_id}
            onChange={(event) => setShown(event.target.value)}
          >
            {reviews.map((item) => (
              <option key={item.gaussians_id} value={item.gaussians_id}>
                {item.gaussians_id.slice(-8)} · {item.status}
              </option>
            ))}
          </select>
          {review.error && <p className="break-words text-xs text-amber-200">{review.error}</p>}
          {review.status === "needs-review" && (
            <>
              <p className="text-xs text-zinc-400">
                {review.gaussians?.toLocaleString()} native rows retained. Placement is Y-up metres; do not apply
                another capture-frame rotation.
              </p>
              <GeneratedSplatInspection
                key={review.gaussians_id}
                splatUrl={artifact("placed-splat.ply")}
                meshUrl={
                  review.mesh_color
                    ? artifact("appearance-delivery.glb")
                    : `${base}/artifact?name=delivery.glb&result=true`
                }
              />
              <div className="flex flex-wrap gap-2">
                <button
                  className="rounded-lg border border-white/15 px-3 py-2 text-xs disabled:opacity-40"
                  disabled={!canPlace || !review.mesh_color}
                  onClick={() => onPropose(false, review)}
                >
                  Inspect native splat placement
                </button>
                <button
                  className="rounded-lg border border-white/15 px-3 py-2 text-xs disabled:opacity-40"
                  disabled={!canReplace || !review.mesh_color}
                  onClick={() => onPropose(true, review)}
                >
                  Propose native splat replacement + background
                </button>
              </div>
              <p className="text-xs text-zinc-400">
                Only current evidence can create a proposal. Native appearance, mesh collision, captured removal and
                observed background share one reviewed revision and undo. Historical studies remain read-only.
              </p>
              <a className="block text-xs text-teal-300" href={artifact("placed-splat.ply")} download>
                Placed full native Gaussian master (Y-up metres)
              </a>
              {review.mesh_color && (
                <>
                  <p className="text-xs text-zinc-400">
                    Mesh colors use explicit SAM image-sRGB interpretation in standard linear-float glTF attributes.
                    Original exports remain unchanged.
                  </p>
                  <a className="block text-xs text-teal-300" href={artifact("appearance-delivery.glb")} download>
                    Color-interpreted delivery GLB (unchanged geometry)
                  </a>
                </>
              )}
              {review.views.map((view) => (
                <details className="text-xs text-zinc-400" key={view.image_id}>
                  <summary>
                    Gaussian camera {view.image_id} · {view.split} · mask {(view.splat_mask_iou * 100).toFixed(1)}% ·
                    mesh/splat {(view.mesh_splat_silhouette_iou * 100).toFixed(1)}%
                  </summary>
                  {["reference", "native", "placed", "mesh", "alpha"].map((kind) => (
                    <img
                      key={kind}
                      className="mt-2 w-full rounded"
                      src={artifact(`review/${kind}-${view.image_id}.png`)}
                      alt={`${kind} Gaussian comparison camera ${view.image_id}`}
                      loading="lazy"
                    />
                  ))}
                </details>
              ))}
            </>
          )}
          <a className="block text-xs text-teal-300" href={`${base}/gaussians/${review.gaussians_id}`}>
            Full Gaussian frame and runtime receipt
          </a>
        </>
      )}
    </section>
  );
}
