import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";

export type CaptureInspection = {
  base: { revision_id: string; generation: number };
  selected_slug: string;
  rows: number[];
  n_rows: number;
  label: string;
  mode?: "selected" | "remaining";
};

type Review = {
  review_id: string;
  label: string;
  selected_slug: string;
  core_count: number;
  stale: boolean;
  cameras: { image_id: number; image_key: string; photo: string; centers_overlay: string; split: string }[];
  result: null | {
    candidate_count: number;
    added_count: number;
    scope: string;
    recipe?: { visibility_method?: string; spatial_margin_m?: number };
    views: {
      image_id: number;
      reason: string | null;
      selected_mask: number | null;
      core_mask_coverage?: number;
      candidate_mask_coverage?: number;
      candidate_outside_mask_fraction?: number;
      candidate_inside_alpha_mass_fraction?: number;
      candidate_outside_alpha_mass_fraction?: number;
    }[];
  };
};

type Proposal = {
  proposal_id: string;
  preview_revision: string;
  base: { revision_id: string; generation: number };
  operation: { kind: string; slug?: string };
};

type Collision = {
  collision_id: string;
  selection_review_id: string;
  stale: boolean;
  result: null | {
    verdict: string;
    candidate: { triangles: number };
    local_occupancy?: { selected_centers_occupied_after?: number; maximum_outside_distance_m?: number };
    scope: string;
  };
};

export function SelectionReviewPanel({
  jobId,
  generation,
  elements,
  disabled,
  inspection,
  onInspect,
  onProposal,
}: {
  jobId: string;
  generation: number;
  elements: Record<string, { label: string; provenance: string; active: boolean }>;
  disabled: boolean;
  inspection: CaptureInspection | null;
  onInspect: (value: CaptureInspection | null) => void;
  onProposal: (value: Proposal) => void;
}) {
  const base = `/api/splat/jobs/${jobId}/studio/selection-reviews`;
  const query = useQuery({
    queryKey: ["selection-reviews", jobId, generation],
    queryFn: () => apiRequest<{ reviews: Review[] }>(base),
  });
  const collisions = useQuery({
    queryKey: ["selection-collisions", jobId, generation],
    queryFn: () => apiRequest<{ collisions: Collision[] }>(`/api/splat/jobs/${jobId}/studio/selection-collisions`),
  });
  const [selected, setSelected] = useState("");
  const [shown, setShown] = useState("");
  const [shownCollision, setShownCollision] = useState("");
  const [layer, setLayer] = useState("photo");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const reviews = query.data?.reviews || [];
  const review = reviews.find((item) => item.review_id === shown) || reviews.at(-1);
  const availableCollisions = (collisions.data?.collisions || []).filter(
    (item) => item.selection_review_id === review?.review_id,
  );
  const preparedCollision =
    availableCollisions.find((item) => item.collision_id === shownCollision) || availableCollisions.at(-1);
  const button = "rounded-xl border border-white/15 px-3 py-2 text-sm hover:bg-white/10 disabled:opacity-40";
  const input = "rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm";
  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };
  const inspect = (candidate: boolean) =>
    run(async () => {
      if (!review) return;
      const value = await apiRequest<CaptureInspection>(`${base}/${review.review_id}/rows?candidate=${candidate}`);
      onInspect({ ...value, label: candidate ? "Refined selection only" : "Current selection only" });
    });
  return (
    <section className="space-y-4 rounded-2xl border border-white/10 p-4">
      <h2 className="font-bold">Inspect and refine captured selection</h2>
      <p className="text-xs text-zinc-400">
        A correct row address is not proof of a complete object. Compare captured photos, inferred masks and visible
        splat contribution before expanding a removal.
      </p>
      <div className="flex flex-wrap gap-3">
        <select
          aria-label="Captured instance to inspect"
          className={input + " min-w-0 max-w-full"}
          value={selected}
          onChange={(event) => setSelected(event.target.value)}
        >
          <option value="">Choose a captured instance</option>
          {Object.entries(elements)
            .filter(([, entry]) => entry.active && !["authored", "generated"].includes(entry.provenance))
            .map(([slug, entry]) => (
              <option key={slug} value={slug}>
                {entry.label} · {slug}
              </option>
            ))}
        </select>
        <button
          className={button}
          disabled={disabled || busy || !selected}
          onClick={() =>
            run(async () => {
              const prepared = await apiRequest<{ review_id: string }>(base, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ selected_slug: selected, expected_generation: generation }),
              });
              setShown(prepared.review_id);
              await query.refetch();
            })
          }
        >
          Prepare selection views
        </button>
        <button
          className={button}
          disabled={busy}
          onClick={() => {
            void query.refetch();
            void collisions.refetch();
          }}
        >
          Refresh selection results
        </button>
      </div>
      {(error || query.error) && (
        <p role="alert" className="text-sm text-red-300">
          {error || String(query.error)}
        </p>
      )}
      {review && (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <select
              aria-label="Retained selection review"
              className={input}
              value={review.review_id}
              onChange={(event) => setShown(event.target.value)}
            >
              {reviews.map((item) => (
                <option key={item.review_id} value={item.review_id}>
                  {item.selected_slug} · {item.review_id.slice(-8)}
                  {item.stale ? " · stale" : ""}
                </option>
              ))}
            </select>
            <select
              aria-label="Selection evidence layer"
              className={input}
              value={layer}
              onChange={(event) => setLayer(event.target.value)}
            >
              <option value="photo">Source photographs</option>
              <option value="centers">Projected centers (not visibility)</option>
              {review.result && (
                <>
                  <option value="current">Current visible splats</option>
                  <option value="mask">Predicted instance mask</option>
                  <option value="candidate">Refined visible splats</option>
                </>
              )}
            </select>
          </div>
          <p className="text-xs text-zinc-400">
            {review.core_count.toLocaleString()} current rows
            {review.result
              ? ` → ${review.result.candidate_count.toLocaleString()} candidate rows (${review.result.added_count.toLocaleString()} added)`
              : "; mask refinement runs through the bounded agent CLI, not this CPU preparation button."}
          </p>
          {review.result?.recipe?.visibility_method === "alpha-compositing-color-gradient/v1" && (
            <p className="text-xs text-zinc-400">
              Experimental footprint refinement: integrated visible alpha, two independent fit groups, mixed-footprint
              abstention and a {((review.result.recipe.spatial_margin_m ?? 0.05) * 100).toFixed(1)} cm spatial margin.
              This is not proof that every part of the object is selected.
            </p>
          )}
          <div className="flex flex-wrap gap-3">
            <button className={button} disabled={disabled || busy || review.stale} onClick={() => inspect(false)}>
              Inspect current selection in 3D
            </button>
            <button
              className={button}
              disabled={disabled || busy || review.stale || !review.result}
              onClick={() => inspect(true)}
            >
              Inspect refined selection in 3D
            </button>
            {inspection && (
              <>
                <label className="flex items-center gap-2 text-xs text-zinc-300">
                  <input
                    type="checkbox"
                    checked={inspection.mode === "remaining"}
                    onChange={(event) =>
                      onInspect({ ...inspection, mode: event.target.checked ? "remaining" : "selected" })
                    }
                  />
                  Show remaining captured appearance
                </label>
                <button className={button} onClick={() => onInspect(null)}>
                  Exit selection inspection
                </button>
              </>
            )}
          </div>
          <p className="text-xs text-amber-200">
            Review only. Predicted masks and rendered depth are not measurements. Refined rows cannot be applied until
            their collision and removal proposal are rebuilt together. Walking and apply review are disabled during 3D
            isolation.
          </p>
          {review.result && (
            <section className="space-y-3 rounded-xl border border-white/10 p-3">
              <h3 className="text-sm font-bold">Selection-aware collision</h3>
              <p className="text-xs text-zinc-400">
                A passing old room collider can still contain the removed object. Prepare a local clearance study, run
                its bounded mesh worker, then review geometry and captured appearance together. No GPU work starts from
                this button.
              </p>
              <button
                className={button}
                disabled={disabled || busy || review.stale}
                onClick={() =>
                  run(async () => {
                    const value = await apiRequest<{ collision_id: string }>(
                      `/api/splat/jobs/${jobId}/studio/selection-collisions`,
                      {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          expected_generation: generation,
                          selection_review_id: review.review_id,
                        }),
                      },
                    );
                    setShownCollision(value.collision_id);
                    await collisions.refetch();
                  })
                }
              >
                Prepare collision inputs
              </button>
              {preparedCollision && (
                <>
                  <select
                    aria-label="Retained selection collision"
                    className={input}
                    value={preparedCollision.collision_id}
                    onChange={(event) => setShownCollision(event.target.value)}
                  >
                    {availableCollisions.map((item) => (
                      <option key={item.collision_id} value={item.collision_id}>
                        {item.collision_id.slice(-8)} ·{" "}
                        {item.stale ? "stale" : item.result?.verdict || "no completed result"}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-zinc-400">
                    {preparedCollision.result
                      ? `${preparedCollision.result.verdict}: ${preparedCollision.result.candidate.triangles?.toLocaleString() || "unavailable"} candidate triangles. ${preparedCollision.result.scope}`
                      : "No sealed result. This build may be unstarted, running or interrupted; inspect the worker before retrying. A completed passing result is required for a proposal."}
                  </p>
                  {preparedCollision.result?.local_occupancy?.selected_centers_occupied_after != null && (
                    <p className="text-xs text-zinc-400">
                      Selected centers still colliding:{" "}
                      {preparedCollision.result.local_occupancy.selected_centers_occupied_after}. Outside-surface
                      maximum displacement:{" "}
                      {((preparedCollision.result.local_occupancy.maximum_outside_distance_m || 0) * 1000).toFixed(3)}{" "}
                      mm.
                    </p>
                  )}
                  <button
                    className={button}
                    disabled={
                      disabled ||
                      busy ||
                      Boolean(inspection) ||
                      preparedCollision.stale ||
                      preparedCollision.result?.verdict !== "PASS_LOCAL_EDIT"
                    }
                    onClick={() =>
                      run(async () => {
                        const proposal = await apiRequest<Proposal>(`/api/splat/jobs/${jobId}/studio/proposals`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            expected_generation: generation,
                            instruction:
                              "Review refined captured removal and its fitted local collision clearance; no complete background is claimed.",
                            operation: {
                              kind: "remove",
                              selected_slug: review.selected_slug,
                              selection_collision_id: preparedCollision.collision_id,
                            },
                          }),
                        });
                        onInspect(null);
                        onProposal(proposal);
                      })
                    }
                  >
                    Propose refined removal
                  </button>
                </>
              )}
              {collisions.error && (
                <p role="alert" className="text-xs text-red-300">
                  {String(collisions.error)}
                </p>
              )}
              <p className="text-xs text-amber-200">
                Local clearance is not whole-scene navigation acceptance. The floor boundary is fitted; background
                appearance and unnamed nearby content remain separate review concerns.
              </p>
            </section>
          )}
          <div className="grid gap-3 md:grid-cols-2">
            {review.cameras.map((camera) => {
              const result = review.result?.views.find((item) => item.image_id === camera.image_id);
              const available = review.result && (layer !== "mask" || result?.selected_mask != null);
              const name =
                layer === "centers"
                  ? camera.centers_overlay
                  : available && layer === "current"
                    ? `visibility/overlay-${camera.image_id}.png`
                    : available && layer === "candidate"
                      ? `candidate-visibility/overlay-${camera.image_id}.png`
                      : available && layer === "mask"
                        ? `mask-overlay-${camera.image_id}.png`
                        : camera.photo;
              return (
                <figure
                  key={camera.image_id}
                  className="space-y-2 overflow-hidden rounded-xl border border-white/10 p-2"
                >
                  <img
                    className="aspect-[3/2] w-full object-contain"
                    loading="lazy"
                    src={`${base}/${review.review_id}/artifact?name=${encodeURIComponent(name)}`}
                    alt={`${layer} selection evidence from ${camera.image_key}`}
                  />
                  <figcaption className="space-y-1 text-xs text-zinc-400">
                    <p>
                      {camera.image_key.split("/").at(-1)} ·{" "}
                      {camera.split === "check" ? "CHECK: excluded from row voting" : "FIT view"}
                    </p>
                    {result?.reason && <p className="text-amber-200">{result.reason}</p>}
                    {result?.candidate_mask_coverage != null && (
                      <p>
                        Inferred-mask coverage: {((result.core_mask_coverage || 0) * 100).toFixed(1)}% →{" "}
                        {(result.candidate_mask_coverage * 100).toFixed(1)}%. Outside-mask contribution:{" "}
                        {((result.candidate_outside_mask_fraction || 0) * 100).toFixed(1)}%.
                      </p>
                    )}
                    {result?.candidate_inside_alpha_mass_fraction != null && (
                      <p>
                        Captured alpha mass inside mask retained by selection:{" "}
                        {(result.candidate_inside_alpha_mass_fraction * 100).toFixed(1)}%. Selected alpha mass outside
                        mask: {((result.candidate_outside_alpha_mass_fraction || 0) * 100).toFixed(1)}%. Baseline
                        occlusion only; newly exposed content is not evaluated by this metric.
                      </p>
                    )}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}
