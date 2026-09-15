import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { SupportSurfacePicker } from "@/components/support-surface-picker";
import { BackgroundCompletionPanel } from "@/components/background-completion-panel";
import { GeneratedObjectPanel } from "@/components/generated-object-panel";
import type { SupportAnchors } from "@/lib/support-anchor-picking";

type Recovery = {
  recovery_id: string;
  selected_slug: string;
  stale: boolean;
  report: {
    supported_fraction: number;
    unknown_fraction: number;
    support_points: number;
    texture_support_points?: number;
    plane: { rms_m: number; selection?: { method?: string; anchors: unknown[]; anchor_to_target_distance_m: number } };
    source_image_ids: number[];
    appearance_checks: { image_id: number; pixels: number; mean_rgb_absolute_error: number }[];
    visibility_qualification?: {
      parent_supported_fraction: number;
      scope: string;
    };
    visibility_comparison?: {
      rejected_parent_cells: number;
      newly_supported_cells: number;
      paired_appearance_checks: {
        image_id: number;
        shared_pixels: number;
        parent_mean_rgb_absolute_error: number | null;
        qualified_mean_rgb_absolute_error: number | null;
      }[];
    };
  };
  evidence: { median_reprojection_px: number };
};
type Proposal = {
  proposal_id: string;
  preview_revision: string;
  base: { revision_id: string; generation: number };
  operation: { kind: string };
};

export function BackgroundRecoveryPanel({
  jobId,
  generation,
  elements,
  disabled,
  onProposal,
}: {
  jobId: string;
  generation: number;
  elements: Record<string, { label: string; provenance: string; active: boolean }>;
  disabled: boolean;
  onProposal: (proposal: Proposal) => void;
}) {
  const base = `/api/splat/jobs/${jobId}/studio`;
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["background-recoveries", jobId, generation],
    queryFn: () => apiRequest<{ recoveries: Recovery[] }>(base + "/recoveries"),
  });
  const [selected, setSelected] = useState("");
  const [shown, setShown] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [anchorMode, setAnchorMode] = useState(false);
  const [anchors, setAnchors] = useState<SupportAnchors | null>(null);
  const [anchorGeneration, setAnchorGeneration] = useState(generation);
  const validAnchors = anchorGeneration === generation ? anchors : null;
  const recoveries = query.data?.recoveries || [];
  const recovery = recoveries.find((item) => item.recovery_id === shown) || recoveries.at(-1);
  const button = "rounded-xl border border-white/15 px-3 py-2 text-sm hover:bg-white/10 disabled:opacity-40";
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
  const propose = (replace: boolean) =>
    run(async () => {
      if (!recovery) return;
      const proposal = await apiRequest<Proposal>(base + "/proposals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_generation: generation,
          instruction: replace
            ? "Remove selected capture and review the observed background cells; unobserved areas remain unresolved."
            : "Inspect photo-backed fitted background cells without removing the captured object.",
          operation: {
            kind: replace ? "replace" : "place",
            selected_slug: replace ? recovery.selected_slug : undefined,
            slug: `background-${recovery.recovery_id.slice(-8)}`,
            label: "Photo-backed fitted background",
            recovery_id: recovery.recovery_id,
          },
        }),
      });
      onProposal(proposal);
      await client.invalidateQueries({ queryKey: ["scene-studio", jobId] });
    });
  return (
    <section className="space-y-3 border-t border-white/10 pt-4">
      <h3 className="text-sm font-bold">Recover observed background</h3>
      <p className="text-xs text-zinc-400">
        Fit a nearby horizontal support surface from tracked features and sample real photos. Hidden pixels are not
        invented. Recovery is CPU-only; optional agent-run depth reviews use the existing captured splat.
      </p>
      <label className="block space-y-1 text-xs text-zinc-400">
        Captured element for background recovery
        <select
          className="w-full rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm"
          value={selected}
          onChange={(event) => {
            setSelected(event.target.value);
            setAnchors(null);
          }}
        >
          <option value="">Choose a captured object</option>
          {Object.entries(elements)
            .filter(([, entry]) => entry.active && !["authored", "generated"].includes(entry.provenance))
            .map(([slug, entry]) => (
              <option key={slug} value={slug}>
                {entry.label}
              </option>
            ))}
        </select>
      </label>
      <label className="flex items-center gap-2 text-xs text-sky-200">
        <input
          type="checkbox"
          checked={anchorMode}
          disabled={disabled || busy}
          onChange={(event) => setAnchorMode(event.target.checked)}
        />
        Choose support plane from tracked photo points
      </label>
      {anchorMode && selected && (
        <SupportSurfacePicker
          key={`${jobId}:${generation}:${selected}`}
          jobId={jobId}
          generation={generation}
          selected={selected}
          disabled={disabled || busy}
          anchors={validAnchors}
          onChange={(value) => {
            setAnchors(value);
            setAnchorGeneration(generation);
          }}
        />
      )}
      {!anchorMode && (
        <p className="text-xs text-amber-200">
          Automatic fitting chooses a strong nearby horizontal plane, not a verified floor. Use tracked photo points to
          choose the intended surface.
        </p>
      )}
      <button
        className={button}
        disabled={disabled || busy || !selected || (anchorMode && (!validAnchors || validAnchors.points.length < 3))}
        onClick={() =>
          run(async () => {
            const result = await apiRequest<Recovery>(base + "/recoveries", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                selected_slug: selected,
                expected_generation: generation,
                texture_size: 128,
                support_anchors: anchorMode ? validAnchors : undefined,
              }),
            });
            setShown(result.recovery_id);
            await client.invalidateQueries({ queryKey: ["background-recoveries", jobId] });
          })
        }
      >
        {busy ? "Working on recovery…" : "Recover surface observations"}
      </button>
      {(error || query.error) && (
        <p role="alert" className="text-xs text-red-300">
          {error || String(query.error)}
        </p>
      )}
      {recovery && (
        <div className="space-y-3">
          <label className="block space-y-1 text-xs text-zinc-400">
            Retained recovery
            <select
              aria-label="Retained recovery"
              className="w-full rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm"
              value={recovery.recovery_id}
              onChange={(event) => setShown(event.target.value)}
            >
              {recoveries.map((item) => (
                <option key={item.recovery_id} value={item.recovery_id}>
                  {item.selected_slug} · {item.recovery_id.slice(-8)}
                  {item.report.visibility_qualification ? " · depth-qualified" : ""}
                  {item.stale ? " · stale" : ""}
                </option>
              ))}
            </select>
          </label>
          <div className="grid grid-cols-2 gap-2">
            <img
              className="w-full rounded-lg bg-zinc-800"
              src={`${base}/recoveries/${recovery.recovery_id}/artifacts/atlas.png`}
              alt="Recovered photo samples; transparent areas are unknown"
            />
            <img
              className="w-full rounded-lg"
              src={`${base}/recoveries/${recovery.recovery_id}/artifacts/support.png`}
              alt="Coverage: teal is supported, magenta is unknown"
            />
          </div>
          <p className="text-xs text-teal-200">
            {(recovery.report.supported_fraction * 100).toFixed(1)}% photo-supported ·{" "}
            {(recovery.report.unknown_fraction * 100).toFixed(1)}% unresolved
          </p>
          <p className="text-xs text-zinc-400">
            {recovery.report.support_points.toLocaleString()} tracked features ·{" "}
            {(recovery.report.plane.rms_m * 1000).toFixed(1)} mm plane-fit RMS ·{" "}
            {recovery.report.source_image_ids.length} photo sources
          </p>
          {recovery.report.texture_support_points != null && (
            <p className="text-xs text-zinc-400">
              {recovery.report.texture_support_points.toLocaleString()} tracked support points lie near the recovery
              footprint; source views are ranked using those tracks.
            </p>
          )}
          <p className="text-xs text-zinc-500">
            Plane-fit RMS is not survey accuracy. Visibility is conservatively inferred, not verified by dense depth.
            Separate appearance checks: {recovery.report.appearance_checks.length} views.
          </p>
          {recovery.report.visibility_qualification && (
            <div className="space-y-2 rounded-lg border border-sky-300/20 bg-sky-300/5 p-3 text-xs">
              <h4 className="font-semibold text-sky-200">Inferred-depth qualification</h4>
              <p className="text-zinc-300">
                Photo-supported coverage:{" "}
                {(recovery.report.visibility_qualification.parent_supported_fraction * 100).toFixed(1)}% original →{" "}
                {(recovery.report.supported_fraction * 100).toFixed(1)}% qualified.
              </p>
              <p className="text-amber-200">
                {recovery.report.visibility_qualification.scope}. Wrong splat depth or an extrapolated plane can reject
                valid photos. Unknown areas remain empty; this is not proof of a clean floor.
              </p>
              {recovery.report.visibility_comparison && (
                <>
                  <p className="text-zinc-400">
                    {recovery.report.visibility_comparison.rejected_parent_cells.toLocaleString()} previously supported
                    cells rejected; {recovery.report.visibility_comparison.newly_supported_cells.toLocaleString()} newly
                    supported after rechecking source agreement.
                  </p>
                  <details>
                    <summary className="cursor-pointer text-sky-200">Same-pixel appearance checks</summary>
                    <p className="mt-2 text-zinc-400">
                      Original → qualified RGB error (0–255), using the same check pixels and delivered atlases. These
                      views do not supply texture; changed coverage is not a quality gain by itself.
                    </p>
                    {recovery.report.visibility_comparison.paired_appearance_checks.map((check) => (
                      <p key={check.image_id} className="mt-1 text-zinc-300">
                        View {check.image_id} · {check.shared_pixels.toLocaleString()} shared pixels ·{" "}
                        {check.parent_mean_rgb_absolute_error == null
                          ? "no overlap"
                          : `${(check.parent_mean_rgb_absolute_error * 255).toFixed(2)} → ${((check.qualified_mean_rgb_absolute_error ?? 0) * 255).toFixed(2)}`}
                      </p>
                    ))}
                  </details>
                </>
              )}
            </div>
          )}
          {recovery.report.plane.selection && (
            <p className="text-xs text-sky-200">
              Anchored by {recovery.report.plane.selection.anchors.length} tracked photo points · target is{" "}
              {recovery.report.plane.selection.anchor_to_target_distance_m.toFixed(2)} m from anchor center. This is an
              extrapolated support plane, not a semantic floor guarantee.
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <button className={button} disabled={disabled || busy || recovery.stale} onClick={() => propose(false)}>
              Preview recovered cells
            </button>
            <button className={button} disabled={disabled || busy || recovery.stale} onClick={() => propose(true)}>
              Propose removal + recovery
            </button>
          </div>
          <p className="text-xs text-amber-200">
            Unobserved cells have no texture or geometry. This is not a complete background fill. Removal still requires
            fresh splat-row and background-collision evidence.
          </p>
          {recovery.report.plane.selection?.method === "photo-linked-sfm-anchors" && (
            <BackgroundCompletionPanel
              key={`${jobId}:${generation}:${recovery.recovery_id}`}
              jobId={jobId}
              generation={generation}
              recoveryId={recovery.recovery_id}
              disabled={disabled || busy || recovery.stale}
              onProposal={onProposal}
            />
          )}
          {recovery.report.plane.selection?.method === "photo-linked-sfm-anchors" && (
            <GeneratedObjectPanel
              key={`generated:${jobId}:${generation}:${recovery.recovery_id}`}
              jobId={jobId}
              generation={generation}
              recoveryId={recovery.recovery_id}
              selectedSlug={recovery.selected_slug}
              disabled={disabled || busy || recovery.stale}
              onProposal={onProposal}
            />
          )}
        </div>
      )}
    </section>
  );
}
