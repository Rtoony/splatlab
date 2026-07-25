import type {
  LangfieldInventoryResult,
  LangfieldQueryResult,
  SceneAssembleReport,
  SceneGroundReport,
  SceneInventoryReport,
  SceneIsolateReport,
  SceneProxyReport,
  SplatCamerasResponse,
  SplatExportManifest,
  SplatUnrealBundle,
} from "@/lib/contracts";
import { recordFailedApiCall } from "@/lib/feedback-context";

// Same-origin fetch helper. The splatlab backend proxies /api/* to the portal
// splat API with the bearer injected, so the browser only ever sees same-origin.
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

export function buildPortableExports(jobId: string): Promise<SplatExportManifest> {
  return apiRequest<SplatExportManifest>(`/api/splat/jobs/${jobId}/exports`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export function buildUnrealBundle(jobId: string): Promise<SplatUnrealBundle> {
  return apiRequest<SplatUnrealBundle>(`/api/splat/jobs/${jobId}/unreal-bundle`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_zip: true }),
  });
}
