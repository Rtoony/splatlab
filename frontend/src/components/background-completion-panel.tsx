import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import {
  completionOperation,
  completionUpload,
  type Completion,
  type CompletionCollision,
} from "@/lib/background-completion";

type Proposal = {
  proposal_id: string;
  preview_revision: string;
  base: { revision_id: string; generation: number };
  operation: { kind: string };
};

export function BackgroundCompletionPanel({
  jobId,
  generation,
  recoveryId,
  disabled,
  onProposal,
}: {
  jobId: string;
  generation: number;
  recoveryId: string;
  disabled: boolean;
  onProposal: (proposal: Proposal) => void;
}) {
  const base = `/api/splat/jobs/${jobId}/studio`;
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["background-completions", jobId, generation],
    queryFn: () => apiRequest<{ completions: Completion[] }>(base + "/completions"),
  });
  const collisions = useQuery({
    queryKey: ["selection-collisions", jobId, generation],
    queryFn: () => apiRequest<{ collisions: CompletionCollision[] }>(base + "/selection-collisions"),
  });
  const [shown, setShown] = useState("");
  const [prompt, setPrompt] = useState("");
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const items = (query.data?.completions || []).filter((item) => item.recovery_id === recoveryId);
  const completion = items.find((item) => item.completion_id === shown) || items.at(-1);
  const collision = (collisions.data?.collisions || [])
    .filter(
      (item) =>
        !item.stale && item.selected_slug === completion?.selected_slug && item.result?.verdict === "PASS_LOCAL_EDIT",
    )
    .at(-1);
  const button = "rounded-xl border border-white/15 px-3 py-2 text-sm hover:bg-white/10 disabled:opacity-40";
  const input = "w-full rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm";
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
  const artifact = (name: string, result = false) =>
    `${base}/completions/${completion?.completion_id}/artifact?name=${encodeURIComponent(name)}&result=${result}`;
  const propose = (replace: boolean) =>
    run(async () => {
      if (!completion) return;
      const proposal = await apiRequest<Proposal>(base + "/proposals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_generation: generation,
          instruction: replace
            ? "Review captured removal with local clearance and explicitly invented support completion. Unseen material is not measured evidence."
            : "Inspect observed and invented material on a fitted support plane without removing the captured object.",
          operation: completionOperation(completion, replace, collision),
        }),
      });
      onProposal(proposal);
      await client.invalidateQueries({ queryKey: ["scene-studio", jobId] });
    });
  return (
    <section aria-label="Unknown-region completion" className="space-y-3 border-t border-white/10 pt-4">
      <h3 className="text-sm font-bold">Complete unknown support cells</h3>
      <p className="text-xs text-amber-200">
        Creative material import, not automatic reconstruction. Observed atlas bytes and triangles stay intact; a
        separate material covers only unknown cells. The extended plane is invented, not measured. Preparing or
        importing runs no model and sends no reference images to a cloud provider.
      </p>
      <label className="block space-y-1 text-xs text-zinc-400">
        Intended unknown-region material
        <textarea
          className={input}
          value={prompt}
          maxLength={2000}
          rows={3}
          disabled={disabled || busy}
          placeholder="Describe what may be invented, what must remain, and any uncertain material boundaries."
          onChange={(event) => setPrompt(event.target.value)}
        />
      </label>
      <div className="flex flex-wrap gap-2">
        <button
          className={button}
          disabled={disabled || busy || !prompt.trim()}
          onClick={() =>
            run(async () => {
              const receipt = await apiRequest<{ completion_id: string }>(base + "/completions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ expected_generation: generation, recovery_id: recoveryId, prompt }),
              });
              setShown(receipt.completion_id);
              await client.invalidateQueries({ queryKey: ["background-completions", jobId] });
            })
          }
        >
          Prepare creative completion
        </button>
        <button
          className={button}
          disabled={busy}
          onClick={() =>
            run(async () => {
              await Promise.all([query.refetch(), collisions.refetch()]);
            })
          }
        >
          Refresh completion evidence
        </button>
      </div>
      {completion && (
        <div className="space-y-3">
          <label className="block space-y-1 text-xs text-zinc-400">
            Retained completion
            <select
              aria-label="Retained completion"
              className={input}
              value={completion.completion_id}
              disabled={busy}
              onChange={(event) => setShown(event.target.value)}
            >
              {items.map((item) => (
                <option key={item.completion_id} value={item.completion_id}>
                  {item.completion_id.slice(-8)} ·{" "}
                  {item.stale ? "stale" : item.result ? "needs review" : "awaiting material"}
                </option>
              ))}
            </select>
          </label>
          <p className="break-words text-xs text-zinc-300">Recorded intent: {completion.prompt}</p>
          <p className="text-xs text-teal-200">
            {(completion.observed_fraction * 100).toFixed(1)}% observed ·{" "}
            {(completion.generated_fraction * 100).toFixed(1)}% designated for invention
          </p>
          <div className="flex flex-wrap gap-3 text-xs text-sky-200">
            <a href={artifact("atlas.png")} download="observed-atlas.png">
              Download observed atlas
            </a>
            <a href={artifact("support.png")} download="support-mask.png">
              Download coverage mask
            </a>
          </div>
          <p className="text-xs text-zinc-400">
            Supply an opaque square PNG/JPEG, 64–2048 px and at most 16 MiB, aligned to the complete atlas UV layout.
            Teal mask cells are observed; magenta cells may be invented. Do not upload the diagnostic mask as a
            material. Generate externally only with your chosen local or explicitly approved cloud workflow.
          </p>
          {!completion.result && (
            <div className="space-y-2">
              <label className="block space-y-1 text-xs text-zinc-400">
                Material provider
                <input
                  className={input}
                  maxLength={120}
                  value={provider}
                  disabled={disabled || busy || completion.stale}
                  onChange={(event) => setProvider(event.target.value)}
                />
              </label>
              <label className="block space-y-1 text-xs text-zinc-400">
                Model or source label
                <input
                  className={input}
                  maxLength={120}
                  value={model}
                  disabled={disabled || busy || completion.stale}
                  onChange={(event) => setModel(event.target.value)}
                />
              </label>
              <label className="block space-y-1 text-xs text-zinc-400">
                UV-aligned completion image
                <input
                  className="block w-full min-w-0 text-xs"
                  type="file"
                  accept="image/png,image/jpeg"
                  disabled={disabled || busy || completion.stale}
                  onChange={(event) => setFile(event.target.files?.[0] || null)}
                />
              </label>
              <button
                className={button}
                disabled={disabled || busy || completion.stale || !file || !provider.trim() || !model.trim()}
                onClick={() =>
                  run(async () => {
                    await apiRequest(`${base}/completions/${completion.completion_id}/image`, {
                      method: "POST",
                      body: completionUpload(file, provider, model),
                    });
                    await client.invalidateQueries({ queryKey: ["background-completions", jobId] });
                  })
                }
              >
                Import completion material
              </button>
            </div>
          )}
          {completion.result && (
            <div className="space-y-2">
              <img
                className="w-full rounded-lg"
                src={artifact(completion.result.generated_image_name, true)}
                alt="Imported material before observed-cell protection; this image is not captured evidence"
              />
              <p className="text-xs text-zinc-400">
                Uploader-reported: {completion.result.generator.provider} / {completion.result.generator.model}.
                Identity is not independently verified. {completion.result.triangles.toLocaleString()}{" "}
                fitted/extrapolated triangles. Texture seams, material boundaries and moving-camera coherence still need
                review.
              </p>
              <div className="flex flex-wrap gap-2">
                <button
                  className={button}
                  disabled={disabled || busy || completion.stale}
                  onClick={() => propose(false)}
                >
                  Preview completed support
                </button>
                <button
                  className={button}
                  disabled={disabled || busy || completion.stale || !collision}
                  onClick={() => propose(true)}
                >
                  Propose removal + completion
                </button>
              </div>
              {completion.result.material_diagnostics && (
                <details className="text-xs text-zinc-400">
                  <summary>Texture boundary diagnostics (not a quality verdict)</summary>
                  <p>
                    At {completion.result.material_diagnostics.boundary_pairs.toLocaleString()} observed/unknown edges,
                    the model disagrees with observed boundary colors by{" "}
                    {completion.result.material_diagnostics.boundary_anchor_mae_rgb8.toFixed(1)} / 255 on average. The
                    protected-material boundary jump is{" "}
                    {completion.result.material_diagnostics.boundary_jump_mae_rgb8.toFixed(1)}, versus{" "}
                    {completion.result.material_diagnostics.generated_boundary_jump_mae_rgb8.toFixed(1)} in the model
                    image alone. Real material edges also create jumps; inspect seams in multiple views. These numbers
                    cannot certify the hidden floor or automatically approve a candidate.
                  </p>
                </details>
              )}
              {!collision && (
                <p className="text-xs text-amber-200">
                  Removal is unavailable until current, matching selection-aware collision passes. A filled grid does
                  not certify safe walking.
                </p>
              )}
            </div>
          )}
          {completion.stale && (
            <p role="status" className="text-xs text-amber-200">
              This completion is stale. Build a current anchored recovery and prepare a new candidate; old evidence is
              retained for review.
            </p>
          )}
        </div>
      )}
      {(error || query.error || collisions.error) && (
        <p role="alert" className="text-xs text-red-300">
          {error || String(query.error || collisions.error)}
        </p>
      )}
    </section>
  );
}
