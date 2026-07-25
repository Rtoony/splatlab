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
  SplatEditApplyResponse,
  SplatEditOp,
  SplatEditRevertResponse,
  SplatEditVersionsResponse,
  SplatExportBuildRequest,
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

// Text-driven edit: delete/isolate rewrite this scene (snapshot-versioned,
// marks the language field stale); extract copies the match into a NEW
// derived scene. Failure details worth surfacing verbatim: 409 stale field,
// 422 no language field / nothing matched, 503 relevancy worker missing.
export function semanticEdit(jobId: string, req: SplatSemanticEditRequest): Promise<SplatSemanticEditResponse> {
  return postJSON<SplatSemanticEditResponse>(`/api/splat/jobs/${jobId}/edit/semantic`, req);
}

// Restore a snapshot version (itself snapshotted first, so revert is undoable).
export function revertEdit(jobId: string, version: number): Promise<SplatEditRevertResponse> {
  return postJSON<SplatEditRevertResponse>(`/api/splat/jobs/${jobId}/edit/revert`, { version });
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
