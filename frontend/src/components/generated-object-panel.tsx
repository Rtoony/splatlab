import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { GeneratedGaussianPanel } from "./generated-gaussian-panel";
import {
  generatedObjectOperation,
  type GeneratedObject,
  type GeneratedCollision,
  type GeneratedGaussianChoice,
} from "@/lib/generated-objects";

type Proposal = {
  proposal_id: string;
  preview_revision: string;
  base: { revision_id: string; generation: number };
  operation: { kind: string };
};

export function GeneratedObjectPanel({
  jobId,
  generation,
  recoveryId,
  selectedSlug,
  disabled,
  onProposal,
}: {
  jobId: string;
  generation: number;
  recoveryId: string;
  selectedSlug: string;
  disabled: boolean;
  onProposal: (proposal: Proposal) => void;
}) {
  const base = `/api/splat/jobs/${jobId}/studio`;
  const client = useQueryClient();
  const [shown, setShown] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const query = useQuery({
    queryKey: ["generated-objects", jobId, generation],
    queryFn: () => apiRequest<{ objects: GeneratedObject[] }>(base + "/generated-objects"),
  });
  const evidence = useQuery({
    queryKey: ["generated-object-evidence", jobId, generation],
    queryFn: async () => {
      const [reviews, collisions] = await Promise.all([
        apiRequest<{ reviews: { review_id: string; selected_slug: string; stale: boolean; result: unknown }[] }>(
          base + "/selection-reviews",
        ),
        apiRequest<{ collisions: GeneratedCollision[] }>(base + "/selection-collisions"),
      ]);
      return { ...reviews, ...collisions };
    },
  });
  const items = (query.data?.objects || []).filter((item) => item.recovery_id === recoveryId);
  const object = items.find((item) => item.generated_object_id === shown) || items.at(-1);
  const review = evidence.data?.reviews
    .filter((item) => !item.stale && item.selected_slug === selectedSlug && item.result)
    .at(-1);
  const collision = evidence.data?.collisions
    .filter(
      (item) =>
        !item.stale &&
        item.selected_slug === selectedSlug &&
        item.selection_review_id === object?.selection_review_id &&
        item.result?.verdict === "PASS_LOCAL_EDIT",
    )
    .at(-1);
  const ready =
    !disabled &&
    !busy &&
    object &&
    !object.stale &&
    object.result?.status === "needs-review" &&
    object.result.placement_resolved;
  const button = "rounded-xl border border-white/15 px-3 py-2 text-xs hover:bg-white/10 disabled:opacity-40";
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
  const asset = (name: string, result = false) =>
    `${base}/generated-objects/${object?.generated_object_id}/artifact?name=${encodeURIComponent(name)}&result=${result}`;
  const propose = (replace: boolean, gaussian?: GeneratedGaussianChoice) =>
    run(async () => {
      if (!object) return;
      const proposal = await apiRequest<Proposal>(base + "/proposals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_generation: generation,
          instruction:
            "Review local generated geometry, source agreement, inferred support and delivery loss. This is not captured or measured truth.",
          operation: generatedObjectOperation(object, replace, collision, gaussian),
        }),
      });
      onProposal(proposal);
      await client.invalidateQueries({ queryKey: ["scene-studio", jobId] });
    });
  return (
    <section aria-label="Local generated object" className="space-y-3 border-t border-white/10 pt-4">
      <h3 className="text-sm font-bold">Generate a replacement object locally</h3>
      <p className="text-xs text-amber-200">
        SAM-3D invents unseen shape and appearance from one masked reference. Check masks never enter generation or pose
        optimization; shared SfM/support evidence is not independent ground truth. Preparing starts no model; the agent
        runs the bounded local worker. Raw mesh/splat masters remain separate from delivery.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          className={button}
          disabled={disabled || busy || !review}
          onClick={() =>
            run(async () => {
              if (!review) return;
              const prepared = await apiRequest<{ generated_object_id: string }>(base + "/generated-objects", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  expected_generation: generation,
                  selection_review_id: review.review_id,
                  recovery_id: recoveryId,
                }),
              });
              setShown(prepared.generated_object_id);
              await query.refetch();
            })
          }
        >
          Prepare local model input
        </button>
        <button
          className={button}
          disabled={busy}
          onClick={() =>
            run(async () => {
              await Promise.all([query.refetch(), evidence.refetch()]);
            })
          }
        >
          Refresh generated evidence
        </button>
      </div>
      {!review && <p className="text-xs text-zinc-400">Complete current multi-view selection evidence first.</p>}
      {object && (
        <div className="space-y-3">
          <label className="block text-xs text-zinc-400">
            Retained generated object
            <select
              aria-label="Retained generated object"
              value={object.generated_object_id}
              onChange={(event) => setShown(event.target.value)}
              className="w-full min-w-0 rounded-lg border border-white/15 bg-zinc-950 p-2"
            >
              {items.map((item) => (
                <option key={item.generated_object_id} value={item.generated_object_id}>
                  {item.generated_object_id.slice(-8)} ·{" "}
                  {item.stale ? "stale" : item.result?.status || "awaiting local worker"}
                </option>
              ))}
            </select>
          </label>
          <img
            className="max-h-64 w-full rounded-lg bg-zinc-900 object-contain"
            src={asset("input.png")}
            alt={`Masked generation input from source camera ${object.input_image_id}`}
          />
          {object.result?.error && <p className="break-words text-xs text-amber-200">{object.result.error}</p>}
          {object.result?.views && (
            <>
              <p className="text-xs text-zinc-400">
                {object.result.master_triangles?.toLocaleString()} master triangles →{" "}
                {object.result.delivery_triangles?.toLocaleString()} delivery triangles. Silhouette overlap is against
                inferred masks, not measured geometry.
              </p>
              {object.result.views.map((view) => (
                <details key={view.image_id} className="text-xs text-zinc-400">
                  <summary>
                    Camera {view.image_id} · {view.split} · master {(view.master_mask_iou * 100).toFixed(1)}% / delivery{" "}
                    {(view.delivery_mask_iou * 100).toFixed(1)}%
                  </summary>
                  {["reference", "master", "delivery"].map((kind) => (
                    <img
                      key={kind}
                      className="mt-2 w-full rounded"
                      src={asset(`review/${kind}-${view.image_id}.png`, true)}
                      alt={`${kind} review at camera ${view.image_id}`}
                      loading="lazy"
                    />
                  ))}
                </details>
              ))}
              <a className="block text-xs text-teal-300" href={asset("master.glb", true)} download>
                Retained generated master (native frame)
              </a>
              <a className="block text-xs text-teal-300" href={asset("placed-master.glb", true)} download>
                Full-detail mesh in the fitted scene frame
              </a>
              <a className="block text-xs text-teal-300" href={asset("master-splat.ply", true)} download>
                Original generated splat (native decoder frame)
              </a>
              <a
                className="block text-xs text-teal-300"
                href={`${base}/generated-objects/${object.generated_object_id}?result=true`}
              >
                Model, placement and delivery receipt
              </a>
              <GeneratedGaussianPanel
                key={object.generated_object_id}
                jobId={jobId}
                objectId={object.generated_object_id}
                canPlace={Boolean(ready)}
                canReplace={Boolean(ready && collision)}
                onPropose={propose}
              />
            </>
          )}
          <div className="flex flex-wrap gap-2">
            <button className={button} disabled={!ready} onClick={() => propose(false)}>
              Inspect generated placement
            </button>
            <button className={button} disabled={!ready || !collision} onClick={() => propose(true)}>
              Propose generated replacement + background
            </button>
          </div>
          <p className="text-xs text-zinc-500">
            Unplaced/failed outputs cannot apply. Inferred plane contact and local clearance do not establish full
            navigation. Use the native Gaussian review's proposal controls to pair verified splat appearance with mesh
            collision; mesh-only proposals remain available.
          </p>
        </div>
      )}
      {(error || query.error || evidence.error) && (
        <p role="alert" className="break-words text-xs text-red-300">
          {error || String(query.error || evidence.error)}
        </p>
      )}
    </section>
  );
}
