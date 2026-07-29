import type { RestyleDoc, RestyleEntry, RestyleMaterial } from "./world-restyle";
import type {
  LangfieldInventoryResult,
  LangfieldQueryResult,
  SplatActivityResponse,
  SceneAssembleReport,
  SceneGroundReport,
  SceneInventoryReport,
  SceneIsolateReport,
  SceneProxyReport,
  SplatCamerasResponse,
  SplatCollisionArtifact,
  SplatCollisionRequest,
  SplatContoursRequest,
  SplatContoursResponse,
  SplatDuplicateResponse,
  SplatEditApplyResponse,
  SplatEditOp,
  SplatEditRevertResponse,
  SplatEditUploadResponse,
  SplatPolishUploadResponse,
  SplatEditVersionsResponse,
  SplatExportBuildRequest,
  SplatJob,
  SplatMeshBuildRequest,
  SplatMeshBuildResponse,
  SplatObjectIsolateRequest,
  SplatObjectIsolateResponse,
  SplatObjectListing,
  SplatSemanticEditRequest,
  SplatSemanticEditResponse,
  SplatExportManifest,
  SplatUnrealBundle,
  SplatUnrealBundleRequest,
} from "@/lib/contracts";
import { recordFailedApiCall } from "@/lib/feedback-context";

// Same-origin fetch helper. splatlab owns /api/splat directly (the portal
// proxy era ended with Phase 2); the browser only ever sees same-origin.
export async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const started = performance.now();
  let res: Response;
  try {
    res = await fetch(path, { credentials: "same-origin", ...init });
  } catch (error) {
    recordFailedApiCall(path, init, "network_error", performance.now() - started);
    throw error;
  }
  if (res.status === 401) {
    window.location.href = "/login";
    throw new Error("unauthorized");
  }
  if (!res.ok) {
    recordFailedApiCall(path, init, res.status, performance.now() - started);
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

// Run a text query against a scene's opt-in Language Field. Returns a
// server-rendered relevancy heatmap (PNG url) plus the normalized query.
export function queryLangfield(jobId: string, text: string): Promise<LangfieldQueryResult> {
  return apiRequest<LangfieldQueryResult>(`/api/splat/jobs/${jobId}/langfield/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

// Fetch a scene's auto-detected object inventory (top-N by presence) for the
// toggle-to-highlight legend. Warm-worker only; 503 -> caller hides the legend.
export function fetchLangfieldInventory(jobId: string): Promise<LangfieldInventoryResult> {
  return apiRequest<LangfieldInventoryResult>(`/api/splat/jobs/${jobId}/langfield/inventory`);
}

// Live activity snapshot: which jobs have an operation in flight right now
// (server-truth from the backend's in-process locks, not client state) plus
// the GPU lease holder. Cheap read-only poll — see useActivity().
export function fetchActivity(): Promise<SplatActivityResponse> {
  return apiRequest<SplatActivityResponse>("/api/splat/activity");
}

export function fetchSplatCameras(jobId: string, limit = 500): Promise<SplatCamerasResponse> {
  return apiRequest<SplatCamerasResponse>(`/api/splat/jobs/${jobId}/cameras?limit=${limit}`);
}

// P6 scene-regen lane: on-demand detail reports behind each stage's poll-safe
// meta["scene"] summary (SplatSceneSummary). Same fmt=report convention every
// scene/* file route shares.
export function fetchSceneInventory(jobId: string): Promise<SceneInventoryReport> {
  return apiRequest<SceneInventoryReport>(`/api/splat/jobs/${jobId}/scene/inventory/file?fmt=report`);
}

export function fetchSceneIsolate(jobId: string): Promise<SceneIsolateReport> {
  return apiRequest<SceneIsolateReport>(`/api/splat/jobs/${jobId}/scene/isolate/file?fmt=report`);
}

export function fetchSceneProxy(jobId: string): Promise<SceneProxyReport> {
  return apiRequest<SceneProxyReport>(`/api/splat/jobs/${jobId}/scene/proxy/file?fmt=report`);
}

export function fetchSceneGround(jobId: string): Promise<SceneGroundReport> {
  return apiRequest<SceneGroundReport>(`/api/splat/jobs/${jobId}/scene/ground/file?fmt=report`);
}

export function fetchSceneAssemble(jobId: string): Promise<SceneAssembleReport> {
  return apiRequest<SceneAssembleReport>(`/api/splat/jobs/${jobId}/scene/assemble/file?fmt=report`);
}

export function fetchPortableExports(jobId: string): Promise<SplatExportManifest> {
  return apiRequest<SplatExportManifest>(`/api/splat/jobs/${jobId}/exports`);
}

// FastAPI-error-aware POST for the build/edit routes. Their failures
// (langfield-stale 409, export/edit-lock 409/423, GPU-arbiter 503, tool
// exit != 0 with a log tail) carry a human-readable {detail} — surface it
// verbatim, same helper pattern as scene-regen.tsx's postJSON. apiRequest()
// would throw the raw JSON blob. A non-string detail (FastAPI 422 validation
// arrays) is stringified rather than becoming "[object Object]".
async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const payload = (await resp.json().catch(() => null)) as { detail?: unknown } | null;
    const detail =
      payload?.detail === undefined
        ? `HTTP ${resp.status}`
        : typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail);
    throw new Error(detail);
  }
  return (await resp.json()) as T;
}

// Build/refresh the portable formats. Long blocking POST (partial failures
// come back as 200 with result.failures; all-failed is a 500 whose detail
// joins the per-format errors). Default sog_iterations stays LOW here too —
// the backend's own default (10) blew the 60-min timeout live on a 1.32M-
// gaussian scene (2026-07-25), so the quick-build path must not inherit it.
export function buildPortableExports(
  jobId: string,
  request: SplatExportBuildRequest = { sog_iterations: 2 },
): Promise<SplatExportManifest> {
  return postJSON<SplatExportManifest>(`/api/splat/jobs/${jobId}/exports`, request);
}

// GPU voxelization → collision/scene.voxel.json + .bin + scene.collision.glb.
// The result is recorded in the exports manifest under its top-level
// `collision` key — refetch GET /exports to pick up voxel_url/mesh_url.
export function buildCollision(jobId: string, request: SplatCollisionRequest): Promise<SplatCollisionArtifact> {
  return postJSON<SplatCollisionArtifact>(`/api/splat/jobs/${jobId}/collision`, request);
}

// Post-hoc splat→mesh export (Digital Twin kernel). Idempotent without
// finetune (an existing mesh.ply returns cached=true instantly).
export function buildSplatMesh(jobId: string, request: SplatMeshBuildRequest): Promise<SplatMeshBuildResponse> {
  return postJSON<SplatMeshBuildResponse>(`/api/splat/jobs/${jobId}/mesh`, request);
}

// Ground contours: mesh/semantic ground → PNEZD points → cdt TIN + DXF on
// office layers. Loud 409s for missing scale calibration / geo anchor.
export function buildGroundContours(jobId: string, request: SplatContoursRequest): Promise<SplatContoursResponse> {
  return postJSON<SplatContoursResponse>(`/api/splat/jobs/${jobId}/geo/contours`, request);
}

// Enumerate every built object for a scene (object.json receipts + an
// on-disk files{} map). Cheap read-only scan.
export function fetchSplatObjects(jobId: string): Promise<SplatObjectListing> {
  return apiRequest<SplatObjectListing>(`/api/splat/jobs/${jobId}/objects`);
}

// Multi-minute GPU build: language-field isolation + optional per-object
// mesh/twin finish/proxy. Loud 409s: missing/stale language field, another
// mesh/object build already running for the job.
export function isolateSplatObject(
  jobId: string,
  request: SplatObjectIsolateRequest,
): Promise<SplatObjectIsolateResponse> {
  return postJSON<SplatObjectIsolateResponse>(`/api/splat/jobs/${jobId}/objects`, request);
}

// Apply 1-32 destructive edit ops in pipeline order. The backend snapshots
// _preview/ first (max 5 kept) — pass the returned version_before to
// revertEdit() for single-step undo.
export function applyEditOps(jobId: string, ops: SplatEditOp[]): Promise<SplatEditApplyResponse> {
  return postJSON<SplatEditApplyResponse>(`/api/splat/jobs/${jobId}/edit/apply`, { ops });
}

// Full working copy of a completed scene — edit the copy, keep the original
// pristine. Really copies every byte, so it is slow-ish and costs real disk;
// the response reports how much. Details worth surfacing verbatim: 409 the
// scene isn't completed or an edit is mid-flight, 507 not enough disk.
export function duplicateScene(jobId: string): Promise<SplatDuplicateResponse> {
  return postJSON<SplatDuplicateResponse>(`/api/splat/jobs/${jobId}/duplicate`, {});
}

// Text-driven edit: delete/isolate rewrite this scene (snapshot-versioned,
// marks the language field stale); extract copies the match into a NEW
// derived scene. Failure details worth surfacing verbatim: 409 stale field,
// 422 no language field / nothing matched, 503 relevancy worker missing.
export function semanticEdit(jobId: string, req: SplatSemanticEditRequest): Promise<SplatSemanticEditResponse> {
  return postJSON<SplatSemanticEditResponse>(`/api/splat/jobs/${jobId}/edit/semantic`, req);
}

// SuperSplat roundtrip: POST an externally-edited canonical .ply back as a
// first-class edit version (snapshot first, derived artifacts regenerated,
// language field marked stale). XHR instead of fetch purely for upload
// progress — same pattern as create/upload-box.tsx. Failures surface the
// backend {detail} verbatim (400 bad PLY, 409 edit lock, 413 >2GB, 503).
export function uploadEditedPly(
  jobId: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<SplatEditUploadResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/splat/jobs/${jobId}/edit/upload`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as SplatEditUploadResponse);
        } catch {
          reject(new Error("Import finished but the server response was unreadable."));
        }
        return;
      }
      let detail = `HTTP ${xhr.status}`;
      try {
        const payload = JSON.parse(xhr.responseText) as { detail?: unknown };
        if (typeof payload.detail === "string") detail = payload.detail;
        else if (payload.detail !== undefined) detail = JSON.stringify(payload.detail);
      } catch {
        // keep the bare HTTP status fallback
      }
      reject(new Error(detail));
    };
    xhr.onerror = () => reject(new Error("Import failed — network error before the server answered."));
    xhr.send(form);
  });
}

// Restore a snapshot version (itself snapshotted first, so revert is undoable).
export function revertEdit(jobId: string, version: number): Promise<SplatEditRevertResponse> {
  return postJSON<SplatEditRevertResponse>(`/api/splat/jobs/${jobId}/edit/revert`, { version });
}

// Post-edit language-field rebuild (realignment): clears STALE, carries
// painted labels across the edit, receipt says what survived. 422 = geometry
// was transformed (retrain is the only cure); 409 = an edit is running.
export function rebuildLangfield(jobId: string): Promise<{
  ok: boolean;
  receipt: {
    ply_rows?: number;
    dropped_ckpt_rows?: number;
    records?: { label?: string | null; kept: number; dropped: number; invalid?: string }[];
  };
  warnings: string[];
  job: SplatJob;
}> {
  return postJSON(`/api/splat/jobs/${jobId}/langfield/rebuild`, {});
}

// Polish round-trip return leg: a Blender/UE-polished GLB lands back as a
// first-class versioned asset. Same XHR-for-progress pattern as
// uploadEditedPly; failures surface the backend {detail} verbatim
// (400 bad GLB, 404 unknown slug, 409 build running, 413 >2GB).
function uploadGlbTo<T>(
  url: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress?.(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch {
          reject(new Error("Upload finished but the server response was unreadable."));
        }
        return;
      }
      let detail = `HTTP ${xhr.status}`;
      try {
        const payload = JSON.parse(xhr.responseText) as { detail?: unknown };
        if (typeof payload.detail === "string") detail = payload.detail;
        else if (payload.detail !== undefined) detail = JSON.stringify(payload.detail);
      } catch {
        // keep the bare HTTP status fallback
      }
      reject(new Error(detail));
    };
    xhr.onerror = () =>
      reject(new Error("Upload failed — network error before the server answered."));
    xhr.send(form);
  });
}

export function uploadPolishedObject(
  jobId: string,
  slug: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<SplatPolishUploadResponse> {
  return uploadGlbTo(
    `/api/splat/jobs/${jobId}/objects/${encodeURIComponent(slug)}/polish`,
    file,
    onProgress,
  );
}

export function uploadPolishedWorldElement(
  jobId: string,
  slug: string,
  file: File,
  onProgress?: (pct: number) => void,
): Promise<SplatPolishUploadResponse> {
  return uploadGlbTo(
    `/api/splat/jobs/${jobId}/world/elements/${encodeURIComponent(slug)}/polish`,
    file,
    onProgress,
  );
}

// List the job's edit restore points (newest-first ordering is the caller's job).
export function fetchEditVersions(jobId: string): Promise<SplatEditVersionsResponse> {
  return apiRequest<SplatEditVersionsResponse>(`/api/splat/jobs/${jobId}/edit/versions`);
}

// Package the UE 5.6 handoff. Requires current portable exports (409
// otherwise). include_zip defaults true here — the download row needs it.
export function buildUnrealBundle(
  jobId: string,
  request: SplatUnrealBundleRequest = { include_zip: true },
): Promise<SplatUnrealBundle> {
  return postJSON<SplatUnrealBundle>(`/api/splat/jobs/${jobId}/unreal-bundle`, request);
}

/* ------------------------------------------------------------------ *
 * Walkable-world interactions (R2)                                    *
 * ------------------------------------------------------------------ */

import type { InteractionRecord } from "@/lib/world-interactions";

export type WorldInteractionsPayload = {
  job_id: string;
  interactions: { elements: InteractionRecord[] } | null;
  state: { elements: Record<string, string> } | null;
  resolved: {
    applied: Record<string, string>;
    dropped: { slug: string; saved: string; reason: string }[];
    world_rebuilt: boolean;
    player?: {
      physics?: { poses?: Record<string, WorldPropPose> };
    };
  } | null;
  state_error: string;
  known_slugs: string[];
};

export type WorldPropPose = {
  position: [number, number, number];
  quaternion: [number, number, number, number];
};

/** Authored affordances plus the resolved player state, in one round trip. */
export function fetchWorldInteractions(
  jobId: string,
  signal?: AbortSignal,
): Promise<WorldInteractionsPayload> {
  return apiRequest<WorldInteractionsPayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/interactions`,
    { signal },
  );
}

export type WorldScenarioPayload = {
  job_id: string;
  scenario: Record<string, unknown> | null;
  default: Record<string, unknown> | null;
};

/** The scenario document (validated backend-side) + the navmesh grid. */
export function fetchWorldScenario(
  jobId: string,
  signal?: AbortSignal,
): Promise<WorldScenarioPayload> {
  return apiRequest<WorldScenarioPayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/scenario`,
    { signal },
  );
}

export function fetchWorldNavmesh(
  jobId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return apiRequest<Record<string, unknown>>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/file?name=navmesh.json`,
    { signal },
  );
}

/** Persist physics-disturbed prop poses under the reserved player seam. */
export function setWorldPlayerPoses(
  jobId: string,
  poses: Record<string, WorldPropPose>,
): Promise<WorldInteractionsPayload> {
  return apiRequest<WorldInteractionsPayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/player`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ poses }),
    },
  );
}

/** Persist one element's new state. */
export function setWorldElementState(
  jobId: string,
  slug: string,
  state: string,
): Promise<WorldInteractionsPayload> {
  return apiRequest<WorldInteractionsPayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/state`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ slug, state }),
    },
  );
}

export type WorldSolidifyResult = {
  ok: boolean;
  op_id: string;
  gate: { passed: boolean; report?: Record<string, unknown> } | null;
  collision: { exit_code: number; error?: string } | null;
  uncalibrated: boolean;
};

/**
 * Rebuild a job's walkable world. `shell_only` patches the shell and its counts
 * and never touches element records, which is what makes it safe to re-run
 * while tuning the shell's look.
 */
export function solidifyWorld(
  jobId: string,
  body: Record<string, unknown>,
): Promise<WorldSolidifyResult> {
  return apiRequest<WorldSolidifyResult>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/solidify`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

/* ------------------------------------------------------------------ *
 * Restyle (W2-C2): the look applied over a captured world.            *
 * ------------------------------------------------------------------ */

export interface WorldRestylePayload {
  job_id: string;
  restyle: RestyleDoc;
  known_slugs: string[];
  materials: RestyleMaterial[];
  presets: string[];
  /** Slugs whose atlas carries a permanently BAKED restyle — the walker must
   *  keep the reconstructed geometry in front of the splat photograph. */
  baked_elements?: string[];
}

export function fetchWorldRestyle(
  jobId: string,
  signal?: AbortSignal,
): Promise<WorldRestylePayload> {
  return apiRequest<WorldRestylePayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/restyle`,
    { signal },
  );
}

export function setWorldRestyle(
  jobId: string,
  doc: { elements: Record<string, RestyleEntry>; lighting: RestyleDoc["lighting"] },
): Promise<WorldRestylePayload> {
  return apiRequest<WorldRestylePayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/restyle`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(doc),
    },
  );
}

export function resetWorldRestyle(jobId: string): Promise<WorldRestylePayload> {
  return apiRequest<WorldRestylePayload>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/restyle`,
    { method: "DELETE" },
  );
}

export interface RestyleBakeResult {
  ok: boolean;
  job_id: string;
  op_id: string;
  uncalibrated?: boolean;
  lighting_baked?: boolean;
  elements: { slug: string; verbs?: string[]; versioned_as?: string }[];
  restyle: WorldRestylePayload;
}

/** Make the current restyle permanent — the server re-bakes each restyled
 *  element's atlas and versions the priors. The world must be reloaded after
 *  (the GLB files on disk changed). */
export function bakeWorldRestyle(
  jobId: string,
  body: { units_per_metre: number },
): Promise<RestyleBakeResult> {
  return apiRequest<RestyleBakeResult>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/restyle/bake`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export function revertWorldRestyleBake(jobId: string): Promise<RestyleBakeResult> {
  return apiRequest<RestyleBakeResult>(
    `/api/splat/jobs/${encodeURIComponent(jobId)}/world/restyle/bake/revert`,
    { method: "POST" },
  );
}
