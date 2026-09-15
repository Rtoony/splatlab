import { useState, type Dispatch, type SetStateAction } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import {
  architecturalFailureGuidance,
  architecturalNavigationDescription,
  architecturalAppearanceDescription,
  architecturalCaptureImpactDescription,
  architecturalGeometryDescription,
  architecturalResultCanPreview,
  type ArchitecturalResult,
} from "@/lib/architectural-feedback";
import {
  architecturalDimensions,
  architecturalSpecError,
  type ArchitecturalSpec,
  type ArchitecturalPickMode,
} from "@/lib/architectural-draft";

type Study = {
  architecture_id: string;
  geometry_method?: string;
  stale: boolean;
  spec: ArchitecturalSpec;
  instruction: string;
  scope: string;
  replaced_elements: { slug: string; rows: number }[];
  result: ArchitecturalResult | null;
};
type Proposal = {
  proposal_id: string;
  preview_revision: string;
  base: { revision_id: string; generation: number };
  operation: { kind: string; slug?: string };
};

export function ArchitecturalEditPanel({
  jobId,
  generation,
  elements,
  disabled,
  spec,
  setSpec,
  draftMode,
  canPosition,
  onDraftMode,
  onProposal,
}: {
  jobId: string;
  generation: number;
  elements: { slug: string; label?: string; role?: string; provenance?: string | null }[];
  disabled: boolean;
  spec: ArchitecturalSpec;
  setSpec: Dispatch<SetStateAction<ArchitecturalSpec>>;
  draftMode: ArchitecturalPickMode | null;
  canPosition: boolean;
  onDraftMode: (mode: ArchitecturalPickMode | null) => void;
  onProposal: (proposal: Proposal) => void;
}) {
  const [instruction, setInstruction] = useState("");
  const [replacedSlugs, setReplacedSlugs] = useState<string[]>([]);
  const [geometryMethod, setGeometryMethod] = useState("nominal-floor/v1");
  const [shown, setShown] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const specError = architecturalSpecError(spec);
  const base = `/api/splat/jobs/${jobId}/studio`;
  const query = useQuery({
    queryKey: ["architectural-edits", jobId, generation],
    queryFn: () => apiRequest<{ edits: Study[] }>(base + "/architectural-edits"),
  });
  const edits = query.data?.edits || [];
  const study = edits.find((item) => item.architecture_id === shown) || edits.at(-1);
  const button = "rounded-lg border border-white/15 px-3 py-2 text-xs hover:bg-white/10 disabled:opacity-40";
  const input = "w-full min-w-0 rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm";
  const post = async <Value,>(url: string, value: unknown): Promise<Value> =>
    apiRequest<Value>(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(value),
    });
  const prepare = async () => {
    setBusy(true);
    setError("");
    try {
      const prepared = await post<Study>(base + "/architectural-edits", {
        expected_generation: generation,
        spec,
        instruction,
        replaced_slugs: replacedSlugs,
        geometry_method: geometryMethod,
      });
      setShown(prepared.architecture_id);
      onDraftMode(null);
      await query.refetch();
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  };
  const preview = async () => {
    if (!study) return;
    setBusy(true);
    setError("");
    try {
      onProposal(
        await post<Proposal>(base + "/proposals", {
          expected_generation: generation,
          instruction: study.instruction,
          operation: {
            kind: "extend-room",
            architecture_id: study.architecture_id,
            slug: "extension-" + study.architecture_id.slice(-8),
            label: "Connected authored room",
          },
        }),
      );
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="space-y-3 rounded-xl border border-white/10 p-4">
      <h3 className="font-bold">Doorway and connected room</h3>
      <p className="text-xs text-zinc-400">
        Author a threshold frame; do not mistake it for a measured wall. Local geometry checks protect named objects,
        floor continuity and a capsule route. Preview pairs captured appearance clipping with the edited collider and
        room. Apply and undo remain separate.
      </p>
      <div className="flex flex-wrap gap-2">
        <button
          className={button}
          disabled={disabled || busy || (!draftMode && (!canPosition || Boolean(specError)))}
          onClick={() => onDraftMode(draftMode ? null : "inspect")}
        >
          {draftMode ? "Hide positioning guide" : "Show positioning guide"}
        </button>
        <button
          className={button}
          disabled={disabled || busy || !canPosition || Boolean(specError)}
          onClick={() => onDraftMode("position")}
        >
          Pick threshold in scene
        </button>
        <button
          className={button}
          disabled={disabled || busy || !canPosition || Boolean(specError)}
          onClick={() => onDraftMode("direction")}
        >
          Aim outward in scene
        </button>
      </div>
      <p className="text-xs text-zinc-400">
        Amber marks the proposed cut; cyan marks room interior and outward direction; green marks the bridge floor.
        Guides show through occluders and change no geometry or collision. A picked collider surface is inferred, not a
        measured floor; check the threshold height and all dimensions before preparing. Cut depth can author a vestibule
        up to 6 m; it is not a measured wall thickness. Inspect the entire cut and the room beyond it for captured
        objects, including unnamed furniture. Longer depth does not waive floor or player-clearance checks.
      </p>
      {specError && (
        <p role="alert" className="text-sm text-amber-200">
          {specError}
        </p>
      )}
      {(error || query.error) && (
        <p role="alert" className="text-sm text-red-300">
          {error || String(query.error)}
        </p>
      )}
      <details className="space-y-3">
        <summary className="cursor-pointer text-sm text-sky-200">Dimensioned portal frame</summary>
        <p className="text-xs text-zinc-400">
          Origin is the threshold floor in world metres. Y is up; yaw zero extends toward +Z, positive yaw toward +X.
          Preparing does not launch a model or change the scene.
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {["X", "Y", "Z"].map((axis, ordinal) => (
            <label key={axis} className="block space-y-1 text-xs text-zinc-400">
              Threshold {axis} (m)
              <input
                aria-label={`Threshold ${axis} (m)`}
                className={input}
                type="number"
                step="0.01"
                min="-100"
                max="100"
                value={spec.origin[ordinal]}
                onChange={(event) =>
                  setSpec((previous) => ({
                    ...previous,
                    origin: previous.origin.map((value, index) =>
                      index === ordinal ? Number(event.target.value) : value,
                    ),
                  }))
                }
              />
            </label>
          ))}
          {architecturalDimensions.map(([key, label, minimum, maximum]) => (
            <label key={key} className="block space-y-1 text-xs text-zinc-400">
              {label}
              <input
                aria-label={label}
                className={input}
                type="number"
                step="0.01"
                min={minimum}
                max={maximum}
                value={spec[key]}
                onChange={(event) => setSpec((previous) => ({ ...previous, [key]: Number(event.target.value) }))}
              />
            </label>
          ))}
        </div>
        <label className="block space-y-1 text-xs text-zinc-400">
          Authored floor recipe
          <select
            aria-label="Authored floor recipe"
            className={input}
            value={geometryMethod}
            onChange={(event) => setGeometryMethod(event.target.value)}
          >
            <option value="nominal-floor/v1">Nominal threshold floor</option>
            <option value="raised-floor-finish/v1">Experimental 16 mm paired floor finish</option>
            <option value="joined-floor-envelope/v1">Experimental joined boundary + 16 mm floor finish</option>
          </select>
        </label>
        {geometryMethod !== "nominal-floor/v1" && (
          <p className="text-xs text-amber-200">
            Raises both authored floor slabs and collision, not captured clipping or player size. Finished clear
            opening: {(spec.opening_height - 0.016).toFixed(3)} m; room height: {(spec.room_height - 0.016).toFixed(3)}{" "}
            m. Requires a new build and review, not a cosmetic toggle.
            {geometryMethod === "joined-floor-envelope/v1" &&
              " Removes overlapping authored faces and extends the roof over the wall tops within the reserved frame. Adds a closed-boundary check; visual seams still require review."}
          </p>
        )}
        <label className="block space-y-1 text-xs text-zinc-400">
          Architectural intent
          <textarea
            aria-label="Architectural intent"
            className={input}
            rows={2}
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            placeholder="Which wall region should open, and what must remain untouched?"
          />
        </label>
        <fieldset className="space-y-1 text-xs text-amber-200">
          <legend>Optional included removals — not moved or silently preserved</legend>
          <p>
            Only include a captured prop if the portal fully contains its pinned mesh and selected centers. Its scene
            element and captured rows are removed in the same reversible edit.
          </p>
          {elements
            .filter(
              (element) => element.role === "prop" && !["authored", "generated"].includes(element.provenance || ""),
            )
            .map((element) => (
              <label key={element.slug} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  aria-label={`Include removal of ${element.label || element.slug}`}
                  checked={replacedSlugs.includes(element.slug)}
                  onChange={(event) =>
                    setReplacedSlugs((previous) =>
                      event.target.checked
                        ? [...previous, element.slug]
                        : previous.filter((slug) => slug !== element.slug),
                    )
                  }
                />
                {element.label || element.slug}
              </label>
            ))}
        </fieldset>
        <button
          className={button}
          disabled={disabled || busy || !instruction.trim() || Boolean(specError)}
          onClick={() => void prepare()}
        >
          Prepare dimensioned room
        </button>
      </details>
      <button className={button} disabled={busy || query.isFetching} onClick={() => void query.refetch()}>
        Refresh architectural evidence
      </button>
      {study && (
        <>
          <label className="block space-y-1 text-xs text-zinc-400">
            Retained architectural edit
            <select
              aria-label="Retained architectural edit"
              className={input}
              value={study.architecture_id}
              onChange={(event) => setShown(event.target.value)}
            >
              {edits.map((item) => (
                <option key={item.architecture_id} value={item.architecture_id}>
                  {item.architecture_id.slice(-8)} · {item.result?.verdict || "awaiting contained geometry build"}
                  {item.stale ? " · stale" : ""}
                </option>
              ))}
            </select>
          </label>
          <p className="text-xs text-zinc-400">{study.instruction}</p>
          <p className="text-sm text-sky-200">
            {study.spec.opening_width.toFixed(2)} × {study.spec.opening_height.toFixed(2)} m opening ·{" "}
            {study.spec.room_width.toFixed(2)} × {study.spec.room_depth.toFixed(2)} m extension
          </p>
          <button
            className={button}
            disabled={disabled || busy}
            onClick={() => {
              setSpec(structuredClone(study.spec));
              setInstruction(study.instruction);
              setReplacedSlugs(study.replaced_elements.map((item) => item.slug));
              setGeometryMethod(study.geometry_method || "nominal-floor/v1");
              if (canPosition) onDraftMode("inspect");
            }}
          >
            Adjust as a new candidate
          </button>
          {Boolean(study.replaced_elements.length) && (
            <p className="text-sm text-amber-200">
              This edit removes:{" "}
              {study.replaced_elements
                .map((item) => `${item.slug} (${item.rows.toLocaleString()} captured rows)`)
                .join(", ")}
              . Undo restores them together with the original wall and collision.
            </p>
          )}
          {study.stale && (
            <p className="text-xs text-amber-200">
              Historical edit: prepare again against the current scene before proposing.
            </p>
          )}
          {!study.result && (
            <p className="break-words text-xs text-amber-200">
              Ask the agent to run the contained architectural builder for {study.architecture_id}, then refresh. No
              background worker is launched automatically.
            </p>
          )}
          {study.result && (
            <>
              <div className="flex flex-wrap gap-2 text-xs">
                {Object.entries(study.result.gates).map(([name, passed]) => (
                  <span key={name} className={passed ? "text-emerald-300" : "text-red-300"}>
                    {passed ? "✓" : "×"} {name.replaceAll("_", " ")}
                  </span>
                ))}
              </div>
              <p className="text-xs text-zinc-400">
                {study.result.metrics.removed_volume_m3.toFixed(3)} m³ removed · {study.result.metrics.route_samples}{" "}
                route samples · minimum capsule-center clearance{" "}
                {study.result.metrics.minimum_capsule_clearance_m.toFixed(3)} m.
              </p>
              <p className="text-xs text-zinc-400">{architecturalNavigationDescription(study.result)}</p>
              <p className="text-xs text-zinc-400">{architecturalAppearanceDescription(study.result)}</p>
              <p aria-label="Architectural floor recipe" className="text-xs text-zinc-400">
                {architecturalGeometryDescription(study.result)}
              </p>
              <p
                aria-label="Captured content affected by architecture"
                className="rounded-lg border border-amber-400/20 p-3 text-xs text-amber-100"
              >
                {architecturalCaptureImpactDescription(study.result)}
              </p>
              {!architecturalResultCanPreview(study.result) && (
                <div
                  aria-label="Architectural adjustments needed"
                  className="space-y-2 rounded-lg border border-amber-400/20 p-3 text-xs text-amber-100"
                >
                  <p className="font-semibold">Adjust before another build; this attempt stays retained.</p>
                  <ul className="list-disc space-y-2 pl-4">
                    {architecturalFailureGuidance(study.result).map((item) => (
                      <li key={item.gate}>{item.message}</li>
                    ))}
                  </ul>
                </div>
              )}
              <button
                className={button}
                disabled={disabled || busy || study.stale || !architecturalResultCanPreview(study.result)}
                onClick={() => void preview()}
              >
                Preview paired doorway and room
              </button>
              <p className="text-xs text-zinc-500">
                Local probe success is not whole-scene navigation or visual acceptance. Gaussian clipping uses displayed
                fragment depth; review oblique views and boundary artifacts.
              </p>
            </>
          )}
        </>
      )}
    </section>
  );
}
