import { useCallback, useEffect, useRef, useState } from "react";
import { useRoute, Link } from "wouter";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";
import { WorldWalker, type WorldManifest } from "@/lib/world-walker";
import { walkingBodyError } from "@/lib/walking-admission";
import { architecturalEntry, enterArchitecturalWalk, type ArchitecturalEntry } from "@/lib/architectural-walking";
import { BackgroundRecoveryPanel } from "@/components/background-recovery-panel";
import { StructuralSurfacePanel } from "@/components/structural-surface-panel";
import { ArchitecturalEditPanel } from "@/components/architectural-edit-panel";
import {
  architecturalSpecError,
  createArchitecturalGuide,
  currentArchitecturalDraft,
  defaultArchitecturalSpec,
  installArchitecturalPicking,
  type ArchitecturalDraft,
  type ArchitecturalPickMode,
  type ArchitecturalSpec,
} from "@/lib/architectural-draft";
import { SelectionReviewPanel, type CaptureInspection } from "@/components/selection-review-panel";
import type { Completion, CompletionCollision } from "@/lib/background-completion";
import { replacementBackgrounds, replacementClearance, type RecoveryChoice } from "@/lib/scene-replacement";

type Pointer = { revision_id: string; generation: number };
type RevisionState = {
  viewer: WorldManifest;
  selections: Parameters<WorldWalker["setPluckDoc"]>[0];
  hidden_capture_slugs: string[];
  architectural_portals?: Parameters<WorldWalker["setArchitecturalPortals"]>[0];
  selection_warnings: string[];
  semantics: Record<string, { label: string; provenance: string; active: boolean }>;
};
type Studio = {
  active: Pointer | null;
  history: { revision_id: string; parent: string | null; activated: boolean; operation: { kind: string } }[];
  state: RevisionState | null;
  legacy_sources_changed: boolean;
  blender_exports: { filename: string; bytes: number }[];
  proposals: (Proposal & { stale: boolean; instruction: string })[];
};
type Proposal = {
  proposal_id: string;
  preview_revision: string;
  base: Pointer;
  operation: { kind: string; slug?: string; background?: { slug: string } };
};
type Revision = {
  revision_id: string;
  viewer: WorldManifest;
  state: RevisionState;
  backdrop_url: string | null;
  backdrop_meters_per_unit: number | null;
};

function frameCapturedSelection(walker: WorldWalker, slug: string) {
  const element = walker.elements.find((item) => item.slug === slug);
  if (!element) return;
  const center = walker.camera.position.clone().set(...element.center);
  const direction = walker.camera.position.clone().sub(center);
  if (direction.lengthSq() < 0.0001) direction.set(1, 0.6, 1);
  const radius = Math.max(Math.hypot(...element.size) / 2, 0.05);
  const halfAngle = Math.atan(Math.tan((walker.camera.fov * Math.PI) / 360) * Math.min(walker.camera.aspect, 1));
  walker.camera.position.copy(center).addScaledVector(direction.normalize(), (radius / Math.sin(halfAngle)) * 1.25);
  walker.camera.lookAt(center);
}

function RevisionViewer({
  jobId,
  revisionId,
  onReady,
  candidateSlug,
  backgroundSlug,
  inspection,
  architecturalDraft,
  architecturalSpec,
  onArchitecturePick,
}: {
  jobId: string;
  revisionId: string;
  onReady: (revision: string | null) => void;
  candidateSlug?: string;
  backgroundSlug?: string;
  inspection: CaptureInspection | null;
  architecturalDraft: ArchitecturalDraft | null;
  architecturalSpec: ArchitecturalSpec;
  onArchitecturePick: (spec: ArchitecturalSpec, mode: ArchitecturalPickMode) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const walkerRef = useRef<WorldWalker | null>(null);
  const cameraPoseRef = useRef<{ position: number[]; quaternion: number[] } | null>(null);
  const [message, setMessage] = useState("Loading revision…");
  const [error, setError] = useState("");
  const [captured, setCaptured] = useState(
    () => new URLSearchParams(window.location.search).get("appearance") !== "mesh",
  );
  const [walking, setWalking] = useState(false);
  const [walkingBody, setWalkingBody] = useState({ totalHeightM: 2.02, radiusM: 0.32 });
  const walkingBodyRef = useRef(walkingBody);
  const [walkingCalibrated, setWalkingCalibrated] = useState(false);
  const [walkingMessage, setWalkingMessage] = useState("");
  const [walkingEntries, setWalkingEntries] = useState<(ArchitecturalEntry & { label: string })[]>([]);
  const [entryError, setEntryError] = useState("");
  const bodyError = walkingBodyError({ ...walkingBody, unitsPerMetre: 1 });
  const [showCollider, setShowCollider] = useState(false);
  const [generatedSplats, setGeneratedSplats] = useState(true);
  const generatedSplatsRef = useRef(generatedSplats);
  const showColliderRef = useRef(showCollider);
  const [isolated, setIsolated] = useState(false);
  const walkingRef = useRef(walking);
  const [loadedRevision, setLoadedRevision] = useState<string | null>(null);
  const [draftError, setDraftError] = useState<{
    spec: ArchitecturalSpec;
    mode: ArchitecturalPickMode;
    message: string;
  } | null>(null);
  const reviewableRevisionRef = useRef<string | null>(null);
  const drafting = Boolean(architecturalDraft) && !inspection && !isolated;
  const specError = architecturalSpecError(architecturalSpec);
  const shownDraftError =
    specError ||
    (draftError?.spec === architecturalSpec && draftError.mode === architecturalDraft?.mode ? draftError.message : "");

  useEffect(() => {
    walkingRef.current = walking;
    if (drafting || !walking) walkerRef.current?.setFlying(true);
  }, [walking, drafting]);

  useEffect(() => {
    walkingBodyRef.current = walkingBody;
    const walker = walkerRef.current;
    if (!walker || loadedRevision !== revisionId) return;
    walker.setFlying(true);
    if (!bodyError && walkingCalibrated)
      walker.setParams({
        bodySizing: "fixed-metric",
        radiusM: walkingBody.radiusM,
        eyeHeightM: walkingBody.totalHeightM - walkingBody.radiusM,
      });
  }, [walkingBody, bodyError, walkingCalibrated, loadedRevision, revisionId]);

  useEffect(() => {
    const walker = walkerRef.current;
    const canvas = canvasRef.current;
    if (!architecturalDraft || !drafting || !walker || !canvas || loadedRevision !== revisionId) return;
    let guide: ReturnType<typeof createArchitecturalGuide> | undefined;
    onReady(null);
    walker.controls.unlock();
    walker.setFlying(true);
    if (!specError) {
      guide = createArchitecturalGuide(architecturalSpec);
      walker.scene.add(guide.object);
    }
    const stopPicking = installArchitecturalPicking({
      canvas,
      keyboard: window,
      camera: walker.camera,
      spec: architecturalSpec,
      mode: architecturalDraft.mode,
      isPointerLocked: () => walker.controls.isLocked,
      pickSurface: (ray) => walker.pickCollisionSurface(ray),
      onPick: onArchitecturePick,
      onError: (message) =>
        setDraftError(message ? { spec: architecturalSpec, mode: architecturalDraft.mode, message } : null),
    });
    return () => {
      stopPicking();
      guide?.dispose();
      if (walkerRef.current === walker) {
        walker.setFlying(!walkingRef.current);
        onReady(reviewableRevisionRef.current);
      }
    };
  }, [
    architecturalDraft,
    architecturalSpec,
    specError,
    drafting,
    loadedRevision,
    revisionId,
    onArchitecturePick,
    onReady,
  ]);

  useEffect(() => {
    showColliderRef.current = showCollider;
    walkerRef.current?.setParams({ showCollider });
  }, [showCollider]);

  useEffect(() => {
    generatedSplatsRef.current = generatedSplats;
    walkerRef.current?.setGeneratedSplats(generatedSplats);
  }, [generatedSplats]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let disposed = false;
    let walker: WorldWalker;
    let loaded = false;
    let visible = true;
    const visibility = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (!loaded || disposed) return;
      if (visible) walker.start();
      else walker.stop();
    });
    visibility.observe(canvas);
    const controller = new AbortController();
    const load = async () => {
      try {
        setError("");
        setWalking(false);
        walkingRef.current = false;
        setWalkingCalibrated(false);
        setWalkingMessage("");
        setWalkingEntries([]);
        setEntryError("");
        setLoadedRevision(null);
        reviewableRevisionRef.current = null;
        onReady(null);
        setMessage("Loading pinned scene revision…");
        walker = new WorldWalker(canvas, { antialias: false });
        walkerRef.current = walker;
        walker.onFlyChange = (flying) => {
          if (!disposed) {
            walkingRef.current = !flying;
            setWalking(!flying);
          }
        };
        walker.onWalkingAdmission = (admission) => {
          if (!disposed) {
            setWalkingMessage(admission.message);
            if (!admission.ok) setWalking(false);
          }
        };
        walker.onControlsError = (message) => {
          if (!disposed) setWalkingMessage(message);
        };
        walker.setGeneratedSplats(generatedSplatsRef.current);
        const revision = await apiRequest<Revision>(`/api/splat/jobs/${jobId}/studio/revisions/${revisionId}`, {
          signal: controller.signal,
        });
        if (disposed) return;
        if (
          inspection &&
          (inspection.base.revision_id !== revisionId || inspection.n_rows !== revision.state.selections?.n_rows)
        )
          throw new Error("Exit the stale selection inspection before viewing this revision.");
        const unitsPerMetre = revision.viewer.units === "meters" ? 1 : 1 / (revision.viewer.meters_per_unit || 1);
        walker.setParams({ unitsPerMetre, physicsProps: false, showCollider: showColliderRef.current });
        walker.setStaticPropCollision(true);
        const source = {
          kind: "local" as const,
          describe: revisionId,
          fileUrl: (name: string) =>
            `/api/splat/jobs/${jobId}/studio/revisions/${revisionId}/artifact?key=${encodeURIComponent("_world/" + name)}`,
        };
        const result = await walker.loadWorld(source, revision.viewer, {
          signal: controller.signal,
          onProgress: (progress) => {
            if (!disposed) setMessage(`${progress.loaded}/${progress.total}: ${progress.label}`);
          },
        });
        if (disposed) return;
        if (
          (revision.state.hidden_capture_slugs.length || revision.state.architectural_portals?.length) &&
          result.colliderSource === "visual_shell"
        ) {
          throw new Error(
            "Captured removal requires its pinned background collider; refusing a visual-shell fallback.",
          );
        }
        walker.setFlying(true);
        const calibrated = revision.viewer.units === "meters" && !revision.viewer.calibration?.stale;
        setWalkingCalibrated(calibrated);
        if (calibrated && !walkingBodyError({ ...walkingBodyRef.current, unitsPerMetre: 1 }))
          walker.setParams({
            bodySizing: "fixed-metric",
            radiusM: walkingBodyRef.current.radiusM,
            eyeHeightM: walkingBodyRef.current.totalHeightM - walkingBodyRef.current.radiusM,
          });
        walker.respawn();
        if (cameraPoseRef.current) {
          walker.camera.position.fromArray(cameraPoseRef.current.position);
          walker.camera.quaternion.fromArray(cameraPoseRef.current.quaternion);
        }
        walker.setPluckDoc(revision.state.selections);
        if (revision.backdrop_url && (captured || inspection) && !isolated) {
          await walker.setBackdrop(revision.backdrop_url, revision.backdrop_meters_per_unit);
          if (disposed) return;
          await walker.applyCapturedVisibility(revision.state.hidden_capture_slugs);
        } else if (
          captured &&
          !isolated &&
          (revision.state.hidden_capture_slugs.length || revision.state.architectural_portals?.length)
        ) {
          throw new Error("Cannot preview removed captured objects without their splat backdrop.");
        }
        if (disposed) return;
        walker.setArchitecturalPortals(revision.state.architectural_portals || []);
        const entries: (ArchitecturalEntry & { label: string })[] = [];
        for (const portal of revision.state.architectural_portals || []) {
          try {
            const key = `_studio/architectural-edits/${portal.architecture_id}/navmesh.json`;
            const navigation = await apiRequest<unknown>(
              `/api/splat/jobs/${jobId}/studio/revisions/${revisionId}/artifact?key=${encodeURIComponent(key)}`,
              { signal: controller.signal },
            );
            if (disposed) return;
            entries.push({
              ...architecturalEntry(navigation, revisionId, portal.architecture_id),
              label: revision.state.semantics[portal.slug]?.label || portal.slug,
            });
          } catch (reason) {
            if (disposed) return;
            setEntryError(
              `Walking entry unavailable for ${portal.slug}: ${reason instanceof Error ? reason.message : String(reason)}`,
            );
          }
        }
        if (disposed) return;
        setWalkingEntries(entries);
        if (inspection) {
          await walker.inspectCapturedRows(inspection.rows, inspection.mode);
          if (disposed) return;
          for (const element of result.elements) walker.setElementVisible(element.slug, false);
          if (inspection.mode !== "remaining") frameCapturedSelection(walker, inspection.selected_slug);
        }
        if (isolated && result.elements.some((element) => element.slug === candidateSlug)) {
          for (const element of result.elements)
            walker.setElementVisible(element.slug, element.slug === candidateSlug || element.slug === backgroundSlug);
        }
        loaded = true;
        if (visible) walker.start();
        reviewableRevisionRef.current =
          inspection ||
          isolated ||
          (!captured &&
            (revision.state.hidden_capture_slugs.length > 0 || Boolean(revision.state.architectural_portals?.length)))
            ? null
            : revisionId;
        setLoadedRevision(revisionId);
        onReady(reviewableRevisionRef.current);
        setMessage(
          `${result.elements.length} elements · ${result.colliderTris.toLocaleString()} collision triangles · pinned ${revisionId.slice(-8)}` +
            (!captured && (revision.state.hidden_capture_slugs.length || revision.state.architectural_portals?.length)
              ? " · Enable captured appearance to review the paired captured edit."
              : ""),
        );
        (window as unknown as { __sceneStudioWalker?: WorldWalker }).__sceneStudioWalker = walker;
      } catch (reason) {
        if (!disposed) setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    void load();
    return () => {
      disposed = true;
      visibility.disconnect();
      controller.abort();
      if (walker) {
        cameraPoseRef.current = {
          position: walker.camera.position.toArray(),
          quaternion: walker.camera.quaternion.toArray(),
        };
        walker.dispose();
      }
      walkerRef.current = null;
      delete (window as unknown as { __sceneStudioWalker?: WorldWalker }).__sceneStudioWalker;
    };
  }, [jobId, revisionId, captured, isolated, candidateSlug, backgroundSlug, onReady, inspection]);

  return (
    <section className="min-w-0 space-y-3">
      <div className="relative h-[65vh] min-h-80 overflow-hidden rounded-2xl bg-black">
        <canvas
          ref={canvasRef}
          className={`h-full w-full ${drafting && architecturalDraft?.mode !== "inspect" ? "cursor-crosshair touch-none" : ""}`}
          aria-label="Revision-pinned 3D scene"
        />
        {drafting && (
          <p
            className="pointer-events-none absolute inset-x-3 bottom-3 rounded-lg bg-zinc-950/90 p-3 text-xs text-amber-100"
            role="status"
          >
            {architecturalDraft?.mode === "position"
              ? "Click an upward-facing collision surface to position the threshold."
              : architecturalDraft?.mode === "direction"
                ? "Click on the threshold-height plane to aim the room outward."
                : "Positioning guide only: no wall is cut and no collision changes. Adjust dimensions or pick again."}{" "}
            Esc finishes picking. X-ray guides are authored/inferred; walking and review/apply are disabled.
          </p>
        )}
        {error && (
          <p role="alert" className="absolute inset-x-4 top-4 rounded-xl bg-red-950/90 p-4 text-sm text-red-200">
            {error}
          </p>
        )}
      </div>
      {drafting && shownDraftError && (
        <p role="alert" className="text-sm text-amber-200">
          {shownDraftError}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-4 text-xs text-zinc-400">
        <button
          onClick={() => void walkerRef.current?.requestLock()}
          disabled={drafting && architecturalDraft?.mode !== "inspect"}
          className="rounded-lg border border-white/15 px-3 py-2"
        >
          Explore with WASD · Esc to release
        </button>
        <label>
          <input
            type="checkbox"
            checked={captured || Boolean(inspection)}
            disabled={Boolean(inspection)}
            onChange={(event) => setCaptured(event.target.checked)}
          />{" "}
          Captured appearance
        </label>
        <label>
          <input
            type="checkbox"
            checked={generatedSplats}
            onChange={(event) => setGeneratedSplats(event.target.checked)}
          />{" "}
          Native generated appearance (mesh collision unchanged)
        </label>
        <label>
          <input
            type="checkbox"
            checked={walking && !inspection && !drafting}
            disabled={
              isolated ||
              Boolean(inspection) ||
              drafting ||
              !walkingCalibrated ||
              Boolean(bodyError) ||
              loadedRevision !== revisionId
            }
            onChange={(event) => {
              if (event.target.checked) walkerRef.current?.beginWalking();
              else walkerRef.current?.setFlying(true);
            }}
          />{" "}
          Walking / collision
        </label>
        <label>
          <input type="checkbox" checked={showCollider} onChange={(event) => setShowCollider(event.target.checked)} />{" "}
          Collision wireframe
        </label>
        {inspection && (
          <button
            className="rounded-lg border border-white/15 px-3 py-2"
            onClick={() => {
              if (walkerRef.current) frameCapturedSelection(walkerRef.current, inspection.selected_slug);
            }}
          >
            Frame selected splats
          </button>
        )}
        {candidateSlug && (
          <>
            <label>
              <input
                type="checkbox"
                checked={isolated}
                onChange={(event) => {
                  setIsolated(event.target.checked);
                  if (event.target.checked) setWalking(false);
                }}
              />{" "}
              Candidate only (inspection)
            </label>
            <button
              className="rounded-lg border border-white/15 px-3 py-2"
              onClick={() => {
                const walker = walkerRef.current;
                const element = walker?.elements.find((item) => item.slug === candidateSlug);
                if (!walker || !element) return;
                const span = Math.max(...element.size, 0.5);
                const [centerX, centerY, centerZ] = element.center;
                walker.camera.position.set(centerX + span * 0.4, centerY + span * 1.1, centerZ + span * 0.9);
                walker.camera.lookAt(centerX, centerY, centerZ);
              }}
            >
              Frame candidate
            </button>
          </>
        )}
      </div>
      <fieldset
        className="flex flex-wrap items-end gap-3 rounded-lg border border-white/10 p-3 text-xs text-zinc-300"
        disabled={loadedRevision !== revisionId || drafting || Boolean(inspection) || isolated}
      >
        <legend className="px-1 text-zinc-400">Explicit player dimensions</legend>
        <label className="grid gap-1">
          Total player height (m)
          <input
            type="number"
            aria-label="Total player height (m)"
            min={1.2}
            max={2.5}
            step={0.01}
            value={Number.isFinite(walkingBody.totalHeightM) ? walkingBody.totalHeightM : ""}
            onChange={(event) =>
              setWalkingBody((previous) => ({ ...previous, totalHeightM: event.target.valueAsNumber }))
            }
            className="w-28 rounded border border-white/15 bg-zinc-950 p-2"
          />
        </label>
        <label className="grid gap-1">
          Player radius (m)
          <input
            type="number"
            aria-label="Player radius (m)"
            min={0.15}
            max={0.5}
            step={0.01}
            value={Number.isFinite(walkingBody.radiusM) ? walkingBody.radiusM : ""}
            onChange={(event) => setWalkingBody((previous) => ({ ...previous, radiusM: event.target.valueAsNumber }))}
            className="w-28 rounded border border-white/15 bg-zinc-950 p-2"
          />
        </label>
        <p className="max-w-lg text-zinc-400">
          No automatic body shortening. Changing dimensions stops walking and requires a fresh collision check.
        </p>
      </fieldset>
      <p role="status" aria-label="Walking admission" className="text-xs text-amber-100">
        {walking ? "Walking with the stated body dimensions. " : "Free camera / inspection; walking is off. "}
        {bodyError || (!walkingCalibrated ? "Current metre calibration is required before walking." : walkingMessage)}
      </p>
      {walkingEntries.map((entry) => (
        <fieldset
          key={entry.architectureId}
          aria-label={`Walking entry ${entry.architectureId}`}
          disabled={
            loadedRevision !== revisionId ||
            entry.revisionId !== revisionId ||
            drafting ||
            Boolean(inspection) ||
            isolated ||
            !walkingCalibrated ||
            Boolean(bodyError)
          }
          className="flex flex-wrap items-center gap-3 rounded-lg border border-white/10 p-3 text-xs"
        >
          <legend className="px-1 text-zinc-400">Walk: {entry.label}</legend>
          {(["captured", "extension"] as const).map((side) => (
            <button
              key={side}
              className="rounded-lg border border-white/15 px-3 py-2"
              onClick={() => {
                const walker = walkerRef.current;
                if (
                  !walker ||
                  loadedRevision !== revisionId ||
                  entry.revisionId !== revisionId ||
                  drafting ||
                  inspection ||
                  isolated ||
                  !walkingCalibrated ||
                  bodyError
                )
                  return;
                try {
                  const admission = enterArchitecturalWalk(walker, entry, revisionId, side);
                  setWalkingMessage(
                    admission.ok
                      ? `${admission.message} Repositioned to the ${side === "captured" ? "captured side" : "extension"}; click Explore with WASD to walk.`
                      : admission.message,
                  );
                } catch (reason) {
                  setWalkingMessage(reason instanceof Error ? reason.message : String(reason));
                }
              }}
            >
              {side === "captured" ? "Start at captured side" : "Start in extension"}
            </button>
          ))}
          <p className="text-zinc-400">
            Explicit reposition, not a traversal. Checks the current collider with your stated body; no automatic
            walking or body resizing.
          </p>
        </fieldset>
      ))}
      {entryError && (
        <p role="alert" className="text-xs text-amber-200">
          {entryError}
        </p>
      )}
      {isolated && (
        <p className="text-xs text-amber-200">
          Inspection only: surrounding geometry is hidden, not removed. Return to context before review/apply.
        </p>
      )}
      {inspection && (
        <p className="text-xs text-amber-200">
          {inspection.mode === "remaining"
            ? `Remaining captured appearance: ${inspection.rows.length.toLocaleString()} selected splats hidden temporarily. No background fill or collision edit is applied; walking and review/apply remain disabled.`
            : `${inspection.label}: ${inspection.rows.length.toLocaleString()} splats. Surrounding geometry is hidden only for inspection.`}
        </p>
      )}
      <p className="text-xs text-zinc-500" aria-live="polite">
        {message}
      </p>
    </section>
  );
}

export default function SceneStudioPage() {
  const [, params] = useRoute("/studio/:jobId");
  const jobId = params?.jobId || "";
  const base = `/api/splat/jobs/${jobId}/studio`;
  const queryClient = useQueryClient();
  const studio = useQuery({
    queryKey: ["scene-studio", jobId],
    queryFn: () => apiRequest<Studio>(base),
    enabled: Boolean(jobId),
  });
  const [proposal, setProposal] = useState<Proposal | null>(null);
  const [compareBase, setCompareBase] = useState(false);
  const [reviewed, setReviewed] = useState(false);
  const [readyRevision, setReadyRevision] = useState<string | null>(null);
  const [inspection, setInspection] = useState<CaptureInspection | null>(null);
  const [architecturalSpec, setArchitecturalSpec] = useState(defaultArchitecturalSpec);
  const [architecturalDraft, setArchitecturalDraft] = useState<ArchitecturalDraft | null>(null);
  const [operation, setOperation] = useState<"place" | "remove" | "replace">("place");
  const [selection, setSelection] = useState("");
  const [exportName, setExportName] = useState("");
  const [backgroundKey, setBackgroundKey] = useState("");
  const [slug, setSlug] = useState("room-extension");
  const [instruction, setInstruction] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const current = studio.data?.active;
  const activeArchitecturalDraft = currentArchitecturalDraft(
    architecturalDraft,
    jobId,
    current,
    busy ||
      Boolean(proposal) ||
      Boolean(inspection) ||
      Boolean(studio.data?.legacy_sources_changed) ||
      studio.data?.state?.viewer.units !== "meters",
  );
  const pickArchitecture = useCallback(
    (spec: ArchitecturalSpec, mode: ArchitecturalPickMode) => {
      if (!activeArchitecturalDraft) return;
      setArchitecturalSpec(spec);
      setArchitecturalDraft({ ...activeArchitecturalDraft, mode });
    },
    [activeArchitecturalDraft],
  );
  const replacementEvidence = useQuery({
    queryKey: ["replacement-evidence", jobId, current?.generation, operation],
    enabled: Boolean(current) && operation !== "place",
    queryFn: async () => {
      const [recoveries, completions, collisions] = await Promise.all([
        apiRequest<{ recoveries: RecoveryChoice[] }>(base + "/recoveries"),
        apiRequest<{ completions: Completion[] }>(base + "/completions"),
        apiRequest<{ collisions: CompletionCollision[] }>(base + "/selection-collisions"),
      ]);
      return { ...recoveries, ...completions, ...collisions };
    },
  });
  const backgrounds = replacementBackgrounds(
    selection,
    replacementEvidence.data?.recoveries || [],
    replacementEvidence.data?.completions || [],
  );
  const background = backgrounds.find((item) => item.key === backgroundKey);
  const clearance = replacementClearance(selection, replacementEvidence.data?.collisions || []);
  const capturedSelection = Boolean(
    selection && !["authored", "generated"].includes(studio.data?.state?.semantics[selection]?.provenance || ""),
  );
  const shown = proposal ? (compareBase ? proposal.base.revision_id : proposal.preview_revision) : current?.revision_id;
  const button = "rounded-xl border border-white/15 px-3 py-2 text-sm hover:bg-white/10 disabled:opacity-40";
  const input = "w-full min-w-0 rounded-lg border border-white/15 bg-zinc-950 p-2 text-sm";
  const run = async (path: string, body: unknown, onSuccess?: (value: any) => void) => {
    setInspection(null);
    setArchitecturalDraft(null);
    setBusy(true);
    setError("");
    try {
      const result = await apiRequest(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      onSuccess?.(result);
      await queryClient.invalidateQueries({ queryKey: ["scene-studio", jobId] });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="mx-auto max-w-[1800px] space-y-5 p-5 sm:p-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="display text-3xl font-black">Creative scene studio</h1>
          <p className="mt-2 text-sm text-zinc-400">
            Select, propose, compare, apply, undo. The captured source stays untouched.
          </p>
        </div>
        <Link href={`/world/${jobId}`} className={button}>
          Original world
        </Link>
      </div>
      {(error || studio.error) && (
        <p role="alert" className="rounded-xl bg-red-500/10 p-4 text-sm text-red-300">
          {error || String(studio.error)}
        </p>
      )}
      {studio.data?.legacy_sources_changed && current && (
        <section className="space-y-3 rounded-xl bg-amber-500/10 p-4 text-sm text-amber-200">
          <p>
            The capture, scale or selection changed. Preview a fresh capture baseline before editing again. This
            replaces studio edits rather than rescaling them silently; prior revisions remain available for undo.
          </p>
          <button
            className={button}
            disabled={busy}
            onClick={() =>
              run(base + "/refresh-proposal", { expected_generation: current.generation }, (value) => {
                setProposal(value as Proposal);
                setCompareBase(false);
                setReviewed(false);
              })
            }
          >
            Preview fresh capture baseline
          </button>
        </section>
      )}
      {studio.data && !current ? (
        <section className="rounded-2xl border border-white/10 p-6">
          <h2 className="mb-3 font-bold">Create a captured baseline</h2>
          <p className="mb-4 max-w-2xl text-sm text-zinc-400">
            Freeze geometry, splats, selection evidence, collision, calibration and authored metadata into a private
            content-addressed scene bundle. This does not change the original world or start training.
          </p>
          <button className={button} disabled={busy} onClick={() => run(base + "/initialize", {})}>
            Open revisioned scene
          </button>
        </section>
      ) : (
        current && (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
            <div className="min-w-0 space-y-3">
              <div className="flex flex-wrap gap-3 text-xs text-zinc-400">
                <span>Active generation {current.generation}</span>
                {proposal && (
                  <>
                    <button className={button} onClick={() => setCompareBase((previous) => !previous)}>
                      {compareBase ? "Show proposal" : "Compare original"}
                    </button>
                    <span className="self-center text-amber-200">
                      {compareBase ? "Base revision" : "Unapplied proposal"} · camera preserved
                    </span>
                  </>
                )}
              </div>
              {shown && (
                <RevisionViewer
                  key={jobId}
                  jobId={jobId}
                  revisionId={shown}
                  candidateSlug={proposal?.operation.slug}
                  backgroundSlug={proposal?.operation.background?.slug}
                  onReady={setReadyRevision}
                  inspection={inspection}
                  architecturalDraft={activeArchitecturalDraft}
                  architecturalSpec={architecturalSpec}
                  onArchitecturePick={pickArchitecture}
                />
              )}
              <SelectionReviewPanel
                jobId={jobId}
                generation={current.generation}
                elements={studio.data?.state?.semantics || {}}
                disabled={busy || Boolean(proposal) || Boolean(studio.data?.legacy_sources_changed)}
                inspection={inspection}
                onInspect={(value) => {
                  setArchitecturalDraft(null);
                  setInspection(value);
                }}
                onProposal={(value) => {
                  setInspection(null);
                  setProposal(value);
                  setCompareBase(false);
                  setReviewed(false);
                }}
              />
              <StructuralSurfacePanel jobId={jobId} generation={current.generation} />
              <ArchitecturalEditPanel
                key={`${jobId}:${current.generation}`}
                jobId={jobId}
                generation={current.generation}
                elements={studio.data?.state?.viewer.elements || []}
                disabled={
                  busy || Boolean(proposal) || Boolean(inspection) || Boolean(studio.data?.legacy_sources_changed)
                }
                spec={architecturalSpec}
                setSpec={setArchitecturalSpec}
                draftMode={activeArchitecturalDraft?.mode || null}
                canPosition={
                  studio.data?.state?.viewer.units === "meters" &&
                  (readyRevision === current.revision_id || Boolean(activeArchitecturalDraft))
                }
                onDraftMode={(mode) => {
                  setArchitecturalDraft(mode ? { jobId, base: { ...current }, mode } : null);
                  setReviewed(false);
                }}
                onProposal={(value) => {
                  setArchitecturalDraft(null);
                  setInspection(null);
                  setProposal(value);
                  setCompareBase(false);
                  setReviewed(false);
                }}
              />
            </div>
            <aside className="min-w-0 space-y-4 rounded-2xl border border-white/10 p-4">
              <h2 className="font-bold">Describe the next edit</h2>
              <BackgroundRecoveryPanel
                jobId={jobId}
                generation={current.generation}
                elements={studio.data?.state?.semantics || {}}
                disabled={busy || Boolean(studio.data?.legacy_sources_changed)}
                onProposal={(value) => {
                  setInspection(null);
                  setProposal(value);
                  setCompareBase(false);
                  setReviewed(false);
                }}
              />
              <label className="block space-y-1 text-xs text-zinc-400">
                Operation
                <select
                  aria-label="Operation"
                  className={input}
                  value={operation}
                  onChange={(event) => setOperation(event.target.value as typeof operation)}
                >
                  <option value="place">Place authored architecture</option>
                  <option value="remove">Remove selected element</option>
                  <option value="replace">Replace selected element</option>
                </select>
              </label>
              {operation !== "place" && (
                <label className="block space-y-1 text-xs text-zinc-400">
                  Selected element
                  <select
                    aria-label="Selected element"
                    className={input}
                    value={selection}
                    onChange={(event) => setSelection(event.target.value)}
                  >
                    <option value="">Choose an element</option>
                    {Object.entries(studio.data?.state?.semantics || {})
                      .filter(([, record]) => record.active)
                      .map(([name, record]) => (
                        <option key={name} value={name}>
                          {record.label} · {record.provenance}
                        </option>
                      ))}
                  </select>
                </label>
              )}
              {operation !== "remove" && (
                <>
                  <label className="block space-y-1 text-xs text-zinc-400">
                    Versioned Blender export
                    <select
                      aria-label="Versioned Blender export"
                      className={input}
                      value={exportName}
                      onChange={(event) => setExportName(event.target.value)}
                    >
                      <option value="">Choose a reviewed export</option>
                      {studio.data?.blender_exports.map((item) => (
                        <option key={item.filename}>{item.filename}</option>
                      ))}
                    </select>
                  </label>
                  <label className="block space-y-1 text-xs text-zinc-400">
                    New element name
                    <input
                      aria-label="New element name"
                      className={input}
                      value={slug}
                      onChange={(event) => setSlug(event.target.value)}
                    />
                  </label>
                  <p className="text-xs text-zinc-500">
                    The agent authors dimensions and placement with typed Blender tools. This form does not interpret
                    prose as geometry yet.
                  </p>
                </>
              )}
              {operation !== "place" && capturedSelection && (
                <div className="space-y-2 text-xs text-zinc-400">
                  <p>
                    {clearance
                      ? "Current local clearance is attached."
                      : "Build current selection-aware clearance before proposing this removal."}{" "}
                    Local clearance is not full navigation acceptance.
                  </p>
                  {operation === "replace" && (
                    <label className="block space-y-1">
                      Companion background
                      <select
                        aria-label="Companion background"
                        className={input}
                        value={backgroundKey}
                        onChange={(event) => setBackgroundKey(event.target.value)}
                      >
                        <option value="">No background patch (gaps may remain)</option>
                        {backgrounds.map((item) => (
                          <option key={item.key} value={item.key}>
                            {item.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  <p>
                    Replacement and background remain separate editable elements; preview, apply and undo move both
                    together.
                  </p>
                  {replacementEvidence.error && <p role="alert">{String(replacementEvidence.error)}</p>}
                  <button
                    className={button}
                    disabled={replacementEvidence.isFetching}
                    onClick={() => void replacementEvidence.refetch()}
                  >
                    Refresh replacement evidence
                  </button>
                </div>
              )}
              <label className="block space-y-1 text-xs text-zinc-400">
                Intent / review note
                <textarea
                  aria-label="Intent / review note"
                  rows={3}
                  className={input}
                  value={instruction}
                  onChange={(event) => setInstruction(event.target.value)}
                  placeholder="What should change, and what must stay faithful?"
                />
              </label>
              <button
                className={button}
                disabled={
                  busy ||
                  !instruction.trim() ||
                  studio.data?.legacy_sources_changed ||
                  (operation !== "remove" && (!exportName || !slug)) ||
                  (operation !== "place" && (!selection || (capturedSelection && !clearance))) ||
                  (operation === "replace" && capturedSelection && Boolean(backgroundKey) && !background)
                }
                onClick={() =>
                  run(
                    base + "/proposals",
                    {
                      expected_generation: current.generation,
                      instruction,
                      operation: {
                        kind: operation,
                        selected_slug: selection || undefined,
                        slug: operation !== "remove" ? slug : undefined,
                        blender_export: operation !== "remove" ? exportName : undefined,
                        selection_collision_id:
                          operation !== "place" && capturedSelection ? clearance?.collision_id : undefined,
                        background: operation === "replace" && capturedSelection ? background?.operation : undefined,
                      },
                    },
                    (value) => {
                      setProposal(value as Proposal);
                      setCompareBase(false);
                      setReviewed(false);
                    },
                  )
                }
              >
                Build review proposal
              </button>
              {proposal && (
                <div className="space-y-3 border-t border-white/10 pt-4">
                  {proposal.operation.kind === "refresh-baseline" && (
                    <p className="text-xs text-amber-200">
                      Baseline refresh: applying replaces the current studio edits with the latest captured state.
                      Review the comparison first.
                    </p>
                  )}
                  <label className="flex items-start gap-2 text-xs text-zinc-300">
                    <input
                      type="checkbox"
                      checked={reviewed}
                      disabled={readyRevision !== proposal.preview_revision}
                      onChange={(event) => setReviewed(event.target.checked)}
                    />
                    I reviewed the geometry, appearance and collision implications.
                  </label>
                  <div className="flex gap-2">
                    <button
                      className={button}
                      disabled={
                        busy ||
                        !reviewed ||
                        Boolean(activeArchitecturalDraft) ||
                        readyRevision !== proposal.preview_revision ||
                        current.generation !== proposal.base.generation
                      }
                      onClick={() =>
                        run(
                          base + `/proposals/${proposal.proposal_id}/apply`,
                          { expected_generation: proposal.base.generation, reviewed: true },
                          () => setProposal(null),
                        )
                      }
                    >
                      Apply together
                    </button>
                    <button className={button} disabled={busy} onClick={() => setProposal(null)}>
                      Keep unapplied
                    </button>
                  </div>
                </div>
              )}
              {Boolean(studio.data?.proposals.length) && (
                <div className="space-y-2 border-t border-white/10 pt-4">
                  <h3 className="text-sm font-bold">Saved proposals</h3>
                  {studio.data?.proposals
                    .slice(-5)
                    .reverse()
                    .map((item) => (
                      <button
                        key={item.proposal_id}
                        className={button + " block w-full text-left"}
                        disabled={busy}
                        onClick={() => {
                          setInspection(null);
                          setProposal(item);
                          setCompareBase(false);
                          setReviewed(false);
                        }}
                      >
                        {item.operation.kind} · {item.proposal_id.slice(-8)}
                        {item.stale ? " · stale, view only" : " · unapplied"}
                      </button>
                    ))}
                </div>
              )}
              <div className="space-y-2 border-t border-white/10 pt-4">
                <h3 className="text-sm font-bold">Undo to a complete revision</h3>
                {studio.data?.history
                  .filter((item) => item.activated && item.revision_id !== current.revision_id)
                  .slice(-5)
                  .reverse()
                  .map((item) => (
                    <button
                      key={item.revision_id}
                      className={button + " block w-full text-left"}
                      disabled={busy}
                      onClick={() =>
                        run(
                          base + "/restore",
                          { revision_id: item.revision_id, expected_generation: current.generation },
                          () => setProposal(null),
                        )
                      }
                    >
                      {item.operation.kind} · {item.revision_id.slice(-8)}
                    </button>
                  ))}
              </div>
              <p className="text-xs text-zinc-500">
                Captured-object removal refuses missing splat-row or collision evidence rather than leaving a ghost.
                Authored edits remain marked authored.
              </p>
            </aside>
          </div>
        )
      )}
    </div>
  );
}
