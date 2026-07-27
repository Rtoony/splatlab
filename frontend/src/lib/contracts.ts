// Splat API contracts (mirrors the portal's server/routes/splat.py payloads).

// One auto-fallback reroute: the registration result that triggered it and the
// solver rung the job climbed to. Persisted by the backend on every reroute.
export interface SfmReroute {
  from_solver: string;
  to_solver: string;
  registered: number | null;
  extracted: number | null;
  pct: string;
  at: string;
}

export interface SplatJob {
  job_id: string;
  mode: "3d" | "4d";
  capture_format: "standard" | "equirectangular360";
  input_path: string;
  output_dir: string;
  command: string[];
  status: "starting" | "running" | "completed" | "failed" | "stopped";
  stage: string | null;
  stages_planned: string[];
  stages_completed: string[];
  // Best-effort/optional stages (compress, webopt, webopt-langweb, health,
  // langfield) that ran but FAILED. Since 2026-07-18 a failed optional stage
  // lands ONLY here — it no longer also claims a slot in stages_completed
  // (the stage rail shows it untraversed). Job status stays "completed";
  // the splat itself is unaffected. Absent on jobs with no failures.
  stages_failed?: { stage: string; reason: string }[];
  pid: number | null;
  exit_code: number | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  stop_requested: boolean;
  pinned: boolean;
  max_num_iterations?: number | null;
  log_lines: string[];
  preview_available?: boolean;
  preview_file_url?: string | null;
  preview_compressed?: boolean;
  preview_spz_url?: string | null;
  preview_view_url?: string | null;
  preview_web_url?: string | null;
  // Opt-in text-searchable "Language Field" — both flags optional so existing
  // scenes (which never carry them) keep deserializing unchanged.
  language_field?: boolean;
  langfield_available?: boolean;
  // Navigable world (_world/ solidify lane). Optional: pre-world payloads
  // never carry them.
  world_available?: boolean;
  world_manifest_url?: string | null;
  // A geometry edit invalidated the built field (on-disk STALE marker — the
  // exact same truth source the backend's 409 guard checks). Polled via
  // /status, so the UI never has to track "did I just make it stale" locally.
  langfield_stale?: boolean;
  // Opt-in splat→mesh export (Digital Twin kernel) — optional for the same reason.
  mesh_export?: boolean;
  mesh_file_url?: string | null;
  mesh_glb_url?: string | null;
  twin_glb_url?: string | null;
  // Survey export (mesh + scale + geo anchor → grid-placed CAD deliverables).
  survey_dxf_url?: string | null;
  survey_landxml_url?: string | null;
  // Ground contours (cdt-drawn contour DXF + the PNEZD points behind it).
  contours_dxf_url?: string | null;
  ground_points_url?: string | null;
  // Standard surface views: cross-sections + isometric TIN (auto receipts).
  sections_url?: string | null;
  surface_iso_url?: string | null;
  // Persisted SfM/frame-density params (backend already returns these via the
  // meta spread; declared here so "Promote to full build" can read a scene's
  // own settings instead of falling back to request defaults that could
  // contradict how the scene was actually built).
  num_frames_target?: number;
  sfm_backend?: "colmap" | "glomap";
  // Escalation visibility (2026-07-18): the RESOLVED starting solver (default
  // flips applied — may differ from the requested sfm_backend), every solver
  // already run, and the structured auto-fallback history. All optional so
  // pre-existing scenes deserialize unchanged.
  sfm_start_solver?: string | null;
  sfm_tried?: string[];
  reroute_count?: number;
  sfm_reroutes?: SfmReroute[];
  // Test Flight trim window. Non-null trim_duration_s marks a scene as a
  // trimmed proof build (see the gallery card's "Promote to full build" action).
  trim_start_s?: number | null;
  trim_duration_s?: number | null;
  // Survey-lane scale calibration: meters per scene unit (nerfstudio scenes are
  // non-metric). Set from the viewer's measure tool; absent/null = uncalibrated.
  meters_per_unit?: number | null;
  // Locate-in-the-world anchor: pins the scene to real WGS84 coordinates.
  // heading_deg = compass bearing (deg CW from true north) of the scene's +Y
  // ground axis; anchor_scene = the scene-unit ground point at (lat, lon).
  // Set from the map modal; absent/null = not located yet.
  geo?: SplatGeoAnchor | null;
  // "sparse" when built via "Few Photos (AI poses)" (MASt3R dense-seed) — poses/geometry
  // are partly AI-inferred, so the card badges it as such. Absent/"standard" otherwise.
  capture_mode?: "standard" | "sparse";
  // "generative-image" when built from a SINGLE image via TripoSplat ("Imagine a Splat")
  // — the whole object is generated; the card badges it "Generated".
  source_type?: "capture" | "generative-image";
  // Cheap per-scene stats for the gallery card (present once the scene is finished).
  stats?: {
    gaussians?: number;
    width?: number;
    height?: number;
    images?: number;
  } | null;
  // Capture-health verdict (report-only fog gate, Capture Coach). Written by the
  // post-train "health" stage or the backfill CLI; absent on scenes never checked.
  // enforced stays false until the doctrine flip — the UI must present verdicts
  // as advisory, never as a hard state.
  health?: {
    v: number;
    fog?: {
      verdict: "FOG" | "HEALTHY" | "UNCERTAIN";
      checked_at: string;
      runtime_s?: number;
      cameras: {
        cam: number;
        counted?: boolean;
        valid_px?: number;
        acc_mean?: number;
        p5?: number;
        p50?: number;
        p95?: number;
        spread?: number;
        shell_frac?: number;
        fog?: boolean;
        healthy?: boolean;
      }[];
      summary: {
        n_cams: number;
        n_counted: number;
        n_fog: number;
        n_healthy: number;
        median_shell_frac: number | null;
        median_spread: number | null;
        median_p50: number | null;
        median_acc: number | null;
      };
      receipts: string[];
      enforced?: boolean;
    };
    // Capture Coach Phase 1: pre-train probe from the SfM artifacts (patched
    // by the A1 gate on pass AND fail paths). Report-only, like fog.
    probe?: {
      v: number;
      verdict: "GOOD" | "MARGINAL" | "POOR";
      findings: string[];
      coaching: string[];
      metrics: {
        n_posed?: number;
        path_bbox_diag?: number;
        mean_step?: number;
        registration_ratio?: number;
        n_points?: number;
        cloud_bbox_diag?: number;
        traj_cloud_ratio?: number;
        inward_frac?: number;
        capture_shape?: "orbit" | "walkthrough";
      };
      caveat: string;
      enforced?: boolean;
    };
  } | null;
  // P6 scene-regen lane: small, poll-safe per-stage summaries merged into
  // meta["scene"] by each POST /scene/* route. Absent until a stage has run;
  // each sub-key absent until THAT stage has run. Full per-stage detail
  // (instances/receipts/etc.) is fetched on demand via the matching
  // fetchScene*() call, same reasoning as langfield inventory.
  scene?: SplatSceneSummary;
}

export interface SplatExportFile {
  path: string;
  media_type: string;
  bytes: number;
  sha256: string;
}

export interface SplatExportArtifact {
  status: "ready" | "stale" | "skipped" | "failed";
  built_at?: string;
  reason?: string;
  error?: string;
  url?: string;
  primary_file?: string;
  files?: SplatExportFile[];
  parameters?: Record<string, unknown>;
}

export interface SplatUnrealBundle {
  status: "ready";
  built_at: string;
  directory: string;
  manifest: string;
  files: number;
  manifest_url?: string;
  zip_url?: string;
  zip?: SplatExportFile | null;
}

// ── Export Center request/response shapes ────────────────────────────────────
// Mirrors backend/export_route.py's pydantic models EXACTLY (read from source
// 2026-07-25). Every field is optional client-side: omitted fields take the
// backend defaults noted per line.

export type SplatExportFormat = "spz" | "sog" | "streamed-sog" | "gltf";

// POST /jobs/{id}/exports (ExportRequest, export_route.py ~line 52).
export interface SplatExportBuildRequest {
  formats?: SplatExportFormat[]; // default: all four, deduped
  spz_version?: 3 | 4; // default 4
  // 1-100, backend default 10 — but the UI defaults LOW (2): CPU SOG at 10
  // iterations exceeded the 60-min conversion timeout on a 1.32M-gaussian
  // scene in the first live run (STATUS.md 2026-07-25).
  sog_iterations?: number;
  lod_chunk_count_k?: number; // 32-4096, default 512 (streamed-SOG only)
  lod_chunk_extent?: number; // >0, ≤10000, default 16.0 (streamed-SOG only)
  force_streamed_sog?: boolean; // default false (auto-skip below 1M gaussians)
  overwrite?: boolean; // default false (current ready artifacts are cached)
}

// POST /jobs/{id}/collision (CollisionRequest, export_route.py ~line 71).
export interface SplatCollisionRequest {
  mode?: "interior" | "exterior" | "raw"; // default "exterior"
  seed_position?: [number, number, number]; // default [0,0,0]
  voxel_size?: number; // >0, ≤10, default 0.05 (scene units)
  opacity_threshold?: number; // 0-1, default 0.1
  cluster_resolution?: number; // >0, ≤100, default 1.0
  cluster_opacity?: number; // 0-1, default 0.999
  cluster_min_contribution?: number; // 0-1, default 0.1
  fill_size?: number; // >0, ≤100, default 1.6 (external/floor fill)
  carve?: boolean; // default false
  carve_height?: number; // >0, ≤20, default 1.6
  carve_radius?: number; // >0, ≤10, default 0.2
  mesh_style?: "smooth" | "faces"; // default "smooth"
}

// manifest["collision"] — a TOP-LEVEL manifest key beside `artifacts` (NOT an
// entry inside it). _public_manifest adds voxel_url/mesh_url when ready; the
// scene.voxel.bin sidecar has no direct route (it ships in the UE bundle).
export interface SplatCollisionArtifact {
  status: "ready";
  built_at: string;
  mode: "interior" | "exterior" | "raw";
  parameters?: Record<string, unknown>;
  files?: SplatExportFile[];
  command?: string[];
  voxel_url?: string; // /exports/file/collision-voxel (scene.voxel.json)
  mesh_url?: string; // /exports/file/collision-mesh (scene.collision.glb)
}

// POST /jobs/{id}/unreal-bundle (UnrealBundleRequest, export_route.py ~86).
export interface SplatUnrealBundleRequest {
  include_zip?: boolean; // default false — the UI always sends true
  include_canonical_ply?: boolean; // default true
  include_survey?: boolean; // default true
  require_current_exports?: boolean; // default true
}

// POST /jobs/{id}/mesh (MeshExportBody, splat_route.py ~line 4261).
export interface SplatMeshBuildRequest {
  finetune?: boolean; // DN escalation, ~10-15 min GPU, REBUILDS (bypasses cache)
  finish?: boolean; // twin finish — needs the exported splat.ply (preview)
  gate?: boolean; // mesh-fidelity gate — PSNR/SSIM vs 6 capture photos
}

export interface SplatMeshBuildResponse {
  job_id: string;
  mesh: Record<string, unknown> | null;
  mesh_file_url: string;
  mesh_glb_url: string | null;
  twin_glb_url: string | null;
  cached: boolean; // true when an existing mesh.ply short-circuited the build
}

// POST /jobs/{id}/geo/contours (ContoursBody = GroundSampleParams + intervals,
// geo_route.py ~line 89/107). Prereqs enforced server-side with loud 409s:
// meters_per_unit (scale calibration) AND meta.geo (Locate anchor), plus a
// mesh OR a language field (semantic=null AUTO falls back to mesh-slope).
export interface SplatContoursRequest {
  epsg?: number; // default 2226 (NAD83 / CA zone 2, US survey foot)
  cell_m?: number; // 0.05-5.0, default 0.25 — ground-sampling grid (meters)
  max_slope_deg?: number; // default 40
  spike_tol_m?: number; // default 0.5
  semantic?: boolean | null; // default null = AUTO (semantic when field exists)
  semantic_thresh?: number; // default 0.5
  minor_ft?: number; // default 0.5; 0 < minor ≤ major ≤ 100
  major_ft?: number; // default 2.5
  tin_faces?: boolean; // default false — also draw the TIN as review linework
}

export interface SplatContoursResponse {
  job_id: string;
  contours: Record<string, unknown>;
  contours_dxf_url: string;
  ground_points_url: string;
  receipt_url: string | null;
  sections_url: string | null;
  surface_iso_url: string | null;
}

// ── Objects lane (single-object isolation, splat_route.py P5b/P5c) ──────────
// Shapes mirrored from source 2026-07-25: ObjectIsolateBody (~line 4501),
// _OBJECT_FILES (~4776), the objects listing route, and object_isolate.py's
// object.json receipt.

export type SplatObjectFileFormat =
  | "splat"
  | "ply"
  | "glb"
  | "receipt"
  | "twin"
  | "twin-top"
  | "twin-oblique"
  | "textured"
  | "textured-atlas"
  | "proxy"
  | "proxy-preview"
  | "polished"
  | "polish-receipt";

// POST /jobs/{id}/objects. Needs a built, non-stale language field (409s
// otherwise); finish requires mesh (400).
export interface SplatObjectIsolateRequest {
  query: string; // 2-80 chars, must not start with whitespace or "-"
  cluster?: number; // 0-5, default 0 — which candidate cluster to take
  expand?: number; // 1.0-3.0, default 1.6 (graded on the garden table)
  rel_floor?: number; // (0,1) exclusive, default 0.30
  mesh?: boolean; // default true — tight fine-voxel object mesh
  proxy?: boolean; // default false — TripoSplat clean replacement (render/VR only)
  finish?: boolean; // default false — twin finish (color + decimate + Y-up GLB)
  finish_target_faces?: number; // 1000-100000, default 10000 (~hero-prop range)
  smooth?: boolean; // default false — curvature-adaptive smoothing in the finish
  smooth_iterations?: number; // 1-10, default 2
  smooth_feature_deg?: number; // 5-90, default 40
  // Textured asset lane: Poisson refit + decimate + UV unwrap + a colour map
  // baked from the gaussians. Independent of `finish`; requires mesh (400).
  texture?: boolean; // default false
  texture_target_faces?: number; // 500-200000, default 8000
  texture_size?: number; // 256-4096, default 1024
  texture_crop?: boolean; // default true — cut the TSDF-fused ground away
}

// The object.json isolation receipt (mesh/object_isolate.py). The listing
// serves it verbatim minus `artifacts`; the POST response's copy may also
// carry in-memory mesh/twin/proxy sub-reports (object.json on disk does not,
// so they're absent from later listings of the same object).
export interface SplatObjectReceipt {
  query: string;
  cluster: number;
  clusters_found: number;
  pool_members: number; // tight core-cluster members (pre-expansion)
  expanded_members: number; // final member count after expand/rel_floor
  expand: number;
  rel_floor: number;
  focus_scene: [number, number, number];
  radius_scene: number;
  bbox_scene: { min: [number, number, number]; max: [number, number, number] };
  bbox_tight?: { min: [number, number, number]; max: [number, number, number] };
  mesh?: Record<string, unknown>;
  mesh_voxel?: number;
  twin?: Record<string, unknown>;
  proxy?: Record<string, unknown>;
}

// Bake-off summary on the objects listing: verdict winner + ranked paired
// PSNR only. `error` replaces the rest when bakeoff.json is unreadable.
export interface SplatObjectBakeoff {
  winner?: string | null;
  reason?: string | null;
  ranked?: { name: string | null; median_psnr_paired: number | null }[];
  error?: string;
}

// One entry from GET /jobs/{id}/objects: receipt + slug + a files{} map of
// only the formats whose artifacts exist on disk RIGHT NOW (same per-format
// URLs the /objects/{slug}/file route serves).
export interface SplatObjectEntry extends SplatObjectReceipt {
  slug: string;
  files: Partial<Record<SplatObjectFileFormat, string>>;
  bakeoff?: SplatObjectBakeoff | null;
}

export interface SplatObjectListing {
  job_id: string;
  objects: SplatObjectEntry[];
}

// POST /jobs/{id}/objects response (splat_route.py ~line 4756). The *_url
// fields reflect what THIS build produced; the listing's files{} map is the
// durable on-disk truth.
export interface SplatObjectIsolateResponse {
  job_id: string;
  slug: string;
  object: SplatObjectReceipt;
  splat_url: string;
  mesh_ply_url: string | null;
  mesh_glb_url: string | null;
  receipt_url: string | null;
  twin_glb_url: string | null;
  proxy_url: string | null;
  proxy_preview_url: string | null;
}

export interface SplatExportManifest {
  schema: "dev.splatlab.exports/v1";
  job_id: string;
  status?: "not-built" | "ready" | "stale";
  created_at?: string;
  updated_at?: string;
  manifest_url?: string;
  source?: {
    gaussian_count?: number;
    sh_degree?: number | null;
    sha256?: string;
    bounds_scene?: { min: number[]; max: number[]; extent: number[] } | null;
  };
  artifacts: Record<string, SplatExportArtifact>;
  collision?: SplatCollisionArtifact;
  unreal_bundle?: SplatUnrealBundle;
  warnings?: string[];
  result?: {
    built: string[];
    cached: string[];
    skipped: string[];
    failures: string[];
  };
}

// meta["scene"] (P6a-P6f). Each key is written by its own route and merged,
// never replaced (review finding 2026-07-23 fixed a real merge bug) — so any
// subset can be present depending on which stages have run.
export interface SplatSceneSummary {
  inventory?: {
    n_instances: number;
    n_vetoed: number;
    conservation?: unknown;
    built_at: string;
  };
  isolate?: {
    n_built: number;
    n_skipped: number;
    sanity: { n_gaussians: number; n_claimed: number; n_background: number; sanity_sum_ok: boolean };
    built_at: string;
  };
  proxy?: {
    n_built: number;
    n_skipped: number;
    built_at: string;
  };
  ground?: {
    ground_points: number | null;
    triangles: number | null;
    built_at: string;
  };
  assemble?: {
    state: "building" | "built" | "approved";
    n_built: number;
    n_flagged: number;
    mode: "faithful" | "styled";
    built_at: string;
  };
}

// P6b instance inventory (GET /jobs/{id}/scene/inventory/file?fmt=report).
export interface SceneInventoryInstance {
  id: string;
  label: string;
  slug: string;
  n_members: number;
  n_views: number;
  views_seen: number[];
  vote_threshold: number;
  mean_score: number;
  centroid_scene: [number, number, number];
  bbox_tight_scene: { min: [number, number, number]; max: [number, number, number] };
  best_view: number | null;
  best_view_box: [number, number, number, number] | null;
  // Present only when this instance overlaps a known P5b _objects/ reference
  // by >5% IoU — informative only, never gates anything.
  regression?: Record<string, { iou: number; recall: number; precision: number }> | null;
}

export interface SceneInventoryReport {
  job_id: string;
  things: string[];
  instances: SceneInventoryInstance[];
  // Nouns that never became an instance — stuff-classified, slug-collided,
  // or SAM3 found nothing — each with a human-readable reason.
  vetoed: { noun: string | null; reason: string }[];
  conservation?: unknown;
  consolidated?: unknown;
}

// P6c batch isolation (GET /jobs/{id}/scene/isolate/file?fmt=report).
export interface SceneIsolateInstance {
  slug: string;
  label: string;
  status: "built" | "SKIPPED:indices-file-missing" | "SKIPPED:too-few-members-after-dedup";
  n_members_original?: number;
  n_overlap_removed?: number;
  n_members_final?: number;
  recall_expand?: { candidates_n: number; kept_n: number } | null;
}

export interface SceneIsolateReport {
  // Absent on the GET (fmt=report serves batch_isolate.json verbatim, which
  // batch_isolate.py never stamps with job_id) -- present on the raw POST
  // response (the route adds it in-memory before returning). Confirmed live
  // against garden 2026-07-23.
  job_id?: string;
  n_gaussians: number;
  recall_expand: boolean;
  instances: SceneIsolateInstance[];
  sanity: { n_gaussians: number; n_claimed: number; n_background: number; sanity_sum_ok: boolean };
}

// P6d batch proxy (GET /jobs/{id}/scene/proxy/file?fmt=report).
export interface SceneProxyInstance {
  slug: string;
  label: string;
  status: "built" | "SKIPPED:crop-failed" | "SKIPPED:generation-failed" | "SKIPPED:registration-failed";
  icp_fitness?: number | null;
  icp_rmse?: number | null;
  total_scale?: number | null;
  transform_4x4?: number[][] | null;
}

export interface SceneProxyReport {
  job_id: string;
  instances: SceneProxyInstance[];
  n_built: number;
  n_skipped: number;
}

// P6e ground mesh (GET /jobs/{id}/scene/ground/file?fmt=report).
export interface SceneGroundReport {
  job_id: string;
  ground_points: number;
  triangles: number;
  finish?: Record<string, unknown>;
}

// P6f assembled scene manifest element (mirrors scene_assemble.py's dict).
export interface SceneAssembleElement {
  slug: string;
  provenance: "captured" | "proxy" | "ground-derived";
  files: Record<string, string>;
  transform_4x4?: number[][] | null;
  registration?: { icp_fitness?: number | null; icp_rmse?: number | null; total_scale?: number | null } | null;
  selection: {
    mode: "faithful" | "styled";
    chosen: string;
    available: string[];
    reason: string;
    label?: string;
  };
}

// P6f scene assembly (GET /jobs/{id}/scene/assemble/file?fmt=report).
export interface SceneAssembleReport {
  job_id: string;
  mode: "faithful" | "styled";
  overrides: Record<string, "captured" | "proxy">;
  manifest: {
    state: "building" | "built" | "approved";
    elements: SceneAssembleElement[];
    [key: string]: unknown;
  };
  assemble: {
    n_elements_total: number;
    n_built: number;
    built: string[];
    n_flagged: number;
    flagged: string[];
    contamination_gate: { ok: boolean; errors: string[] };
    seconds: number;
  };
}

// Tier-0 upload-time capture screen (POST /api/splat/precheck). Advisory-only:
// the Create button is NEVER disabled by this.
export interface SplatPrecheckResult {
  v: number;
  capture_type: "video" | "photo-folder" | "photo-zip" | "unknown";
  advisories: string[];
  metrics: {
    n_frames?: number;
    median_edge_energy?: number;
    median_dark_frac?: number;
    median_bright_frac?: number;
    static_pair_ratio?: number;
  };
  note?: string;
}

// A scene's real-world anchor (meta["geo"], written by POST /jobs/{id}/geo).
export interface SplatGeoAnchor {
  v: number;
  lat: number;
  lon: number;
  alt_m?: number | null;
  heading_deg: number;
  anchor_scene?: [number, number] | null;
  source?: "map" | "exif" | "manual";
  set_at?: string;
}

// Bootstrap for the Locate map modal (GET /jobs/{id}/geo/footprint).
export interface SplatGeoFootprint {
  job_id: string;
  available: boolean;
  reason?: string;
  url?: string;
  width?: number;
  height?: number;
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
  units_per_px?: number;
  center?: [number, number];
  meters_per_unit?: number | null;
  geo?: SplatGeoAnchor | null;
}

// One GPS candidate from the capture source (GET /jobs/{id}/geo/suggest).
export interface SplatGeoSuggestion {
  lat: number;
  lon: number;
  alt_m?: number | null;
  source: string;
  detail: string;
}

// Result of a Language Field text query: a server-rendered 3-view relevancy
// heatmap strip (PNG) plus the normalized query and a readiness flag.
export interface LangfieldMatch {
  focus: [number, number, number];
  radius: number;
  score: number;
  count: number;
  // Per-instance result thumbnail FILENAME (rendered from the camera that best frames
  // this match, cropped to the object); served via the same heatmap route for this job.
  // Absent on cold/legacy responses.
  thumb?: string;
}

export interface LangfieldQueryResult {
  query: string;
  heatmap_url: string;
  ready: boolean;
  // 3D centroid of the primary match (viewer frame) + its spread, for "fly to".
  focus?: [number, number, number];
  radius?: number;
  // Distinct clustered instances of the match (multiple references), for the
  // clickable results + per-instance highlight overlay.
  matches?: LangfieldMatch[];
}

// One auto-detected object in a scene's inventory: a label, how much of the scene it
// occupies (presence 0..1), a peak-confidence reliability (0..1), and its clustered
// instances (for the toggle-to-highlight legend).
export interface LangfieldInventoryItem {
  label: string;
  presence: number;
  reliability: number;
  focus: [number, number, number];
  radius: number;
  count?: number;
  // A spread of the object's own matching gaussians, for an AREA highlight (not a pin).
  points: [number, number, number][];
  matches: LangfieldMatch[];
}

export interface LangfieldInventoryResult {
  job_id: string;
  items: LangfieldInventoryItem[];
}

export interface SplatCameraPose {
  index: number;
  image_name: string;
  file_path: string;
  position: [number, number, number];
  forward: [number, number, number];
  up: [number, number, number];
  right: [number, number, number];
  fov_y_degrees?: number | null;
}

export interface SplatCamerasResponse {
  job_id: string;
  count: number;
  total: number;
  sampled: boolean;
  frame: "viewer" | "source";
  source: "dataparser_transforms" | "transforms_json";
  display_scale: number;
  image_size: { width: number | null; height: number | null };
  cameras: SplatCameraPose[];
}

export interface SplatGpuHolder {
  lane: string | null;
  job_id: string | null;
  since: string | null;
  locked: boolean;
}

// GET /api/splat/activity (backend/activity_route.py): live "what is busy
// RIGHT NOW" snapshot from the backend's own in-process locks + the GPU lease.
// Sparse by contract: only HELD flags appear, and only busy jobs appear in
// `jobs` — a job with nothing in flight is simply absent.
export interface SplatActivityJobFlags {
  editing?: boolean; // mutating scene edit (apply/semantic/merge/restore)
  preview_exporting?: boolean; // ns-export preview regeneration
  meshing?: boolean; // mesh, object-isolate, AND scene-lane (P6) builds — one shared lock
  exporting?: boolean; // portable SPZ/SOG/glTF + collision + Unreal bundle
}

// Server-side stepped progress for one in-flight scene edit. Step vocabulary:
// apply/upload edits run ["snapshot","apply","compress","webopt","langweb",
// "finalize"]; semantic edits prepend a leading "match". Entries exist only
// while an edit is running, and only on newer backends — the deployed server
// may omit the edit_progress key entirely, so all access must stay
// optional-chained.
export interface SplatEditProgress {
  step: string;
  step_index: number;
  steps: number;
  labels: string[];
  started_at: string;
}

export interface SplatActivityResponse {
  gpu: {
    // null when nobody holds the GPU lease.
    holder: { lane: string; job_id: string | null; since: string | null } | null;
  };
  jobs: Record<string, SplatActivityJobFlags>;
  // Optional by contract: older deployed backends don't send this key at all.
  edit_progress?: Record<string, SplatEditProgress>;
}

export interface SplatComputeStatus {
  enabled: boolean;
  maintenance_active: boolean;
  reason: string | null;
  marker_path: string;
  unlock_path?: string;
  mode?: "safe-browse" | "supervised" | "normal";
  supervised_unlock?: {
    active: boolean;
    path: string;
    schema: string;
    mode: string | null;
    reason: string | null;
    operator: string | null;
    created_at: string | null;
    expires_at: string | null;
    seconds_remaining: number;
    max_active_jobs: number;
    detail: string | null;
  };
  safe_capabilities: string[];
  blocked_capabilities: string[];
}

export interface SplatStatusResponse {
  workspace: {
    root: string;
    data_dir: string;
    outputs_dir: string;
    conda_env_bin: string;
  };
  engines: {
    ns_train_available: boolean;
    ns_train_path: string | null;
    ns_process_data_available: boolean;
    colmap_available: boolean;
    ffmpeg_available: boolean;
    insv_stitch_available: boolean;
    four_d_engine_ready: boolean;
    // Whether the Language Field toolchain exists on this host (opt-in feature gate).
    langfield_available?: boolean;
    mesh_available?: boolean;
    [k: string]: unknown;
  };
  compute?: SplatComputeStatus;
  media_samples: Array<{ name: string; path: string; kind: "file" | "directory" }>;
  jobs: SplatJob[];
  active_jobs: number;
  gpu: SplatGpuHolder;
  notes: string[];
}

// ── Native edit ops (POST /jobs/{id}/edit/apply) ─────────────────────────────
// Mirrors backend/edit_ops.py's pydantic models EXACTLY (discriminated on
// `type`). Field-name gotchas verified against the backend source 2026-07-25:
// decimate is `n`/`pct` (NOT count/percent), translate/rotate are scalar
// x/y/z fields (NOT an offset/degrees vector).

// KEEPS everything inside [min, max]; removes everything outside. Maps to
// splat-transform `-B min,max` — "Remove Gaussians outside box (min, max
// corners)" per the CLI's own README.
export interface SplatCropBoxOp {
  type: "crop_box";
  min: [number, number, number];
  max: [number, number, number];
}

// KEEPS everything inside the sphere (splat-transform -S: "remove outside").
export interface SplatCropSphereOp {
  type: "crop_sphere";
  center: [number, number, number];
  radius: number; // > 0
}

export interface SplatFilterValueOp {
  type: "filter_value";
  name: "opacity" | "scale_0" | "scale_1" | "scale_2" | "x" | "y" | "z";
  min?: number; // at least one of min/max required
  max?: number;
}

// All-optional: {} uses the splat-transform CLI defaults. If any of
// size/op/min is given, ALL THREE must be (backend model_validator).
export interface SplatFilterFloatersOp {
  type: "filter_floaters";
  size?: number;
  op?: number;
  min?: number;
}

export interface SplatFilterClusterOp {
  type: "filter_cluster";
  res?: number; // res/op/min all together, or none
  op?: number;
  min?: number;
  seed_pos?: [number, number, number];
}

// Exactly ONE of n (absolute target count) or pct (percentage of gaussians
// to KEEP, 0 < pct <= 100 — splat-transform -F "n% to keep a percentage").
export interface SplatDecimateOp {
  type: "decimate";
  n?: number;
  pct?: number;
}

export interface SplatTranslateOp {
  type: "translate";
  x: number;
  y: number;
  z: number;
}

// Euler degrees per axis, each within ±360.
export interface SplatRotateOp {
  type: "rotate";
  x: number;
  y: number;
  z: number;
}

export interface SplatScaleOp {
  type: "scale";
  factor: number; // 0 < factor <= 1000
}

export type SplatEditOp =
  | SplatCropBoxOp
  | SplatCropSphereOp
  | SplatFilterValueOp
  | SplatFilterFloatersOp
  | SplatFilterClusterOp
  | SplatDecimateOp
  | SplatTranslateOp
  | SplatRotateOp
  | SplatScaleOp;

// apply_edit_ops response (edit_ops.py ~line 878). `job` is the refreshed
// _job_payload — its stats cache is cleared by the edit and recomputed inside
// this very payload, so job.stats?.gaussians IS the fresh post-edit count.
export interface SplatEditApplyResponse {
  ok: boolean;
  version_before: number; // snapshot seq to pass to /edit/revert for undo
  warnings: string[];
  job: SplatJob;
}

// POST /jobs/{id}/duplicate — a full working copy of a completed scene, so
// destructive edits land on the copy and the original stays pristine. Every
// byte is really copied (no hardlinks: colmap's SQLite db and the langfield
// npz are both written in place, and a shared inode would corrupt the
// original), minus the source's restore points. `bytes` is what it occupies.
export interface SplatDuplicateResponse {
  ok: boolean;
  new_job_id: string;
  bytes: number;
  job: SplatJob;
}

// upload_edited_ply response (edit_ops.py POST /jobs/{id}/edit/upload) —
// apply's shape plus the gaussian count read from the validated PLY header.
export interface SplatEditUploadResponse extends SplatEditApplyResponse {
  gaussians: number;
}

// Polish upload response (polish_route.py POST .../polish, both object and
// world-element variants): serving URL for the landed GLB + the provenance
// receipt (schema dev.splatlab.polish-receipt/v1).
export interface SplatPolishUploadResponse {
  ok: boolean;
  slug: string;
  url: string;
  receipt: Record<string, unknown>;
}

// revert_version response (edit_ops.py ~line 410).
export interface SplatEditRevertResponse {
  ok: boolean;
  reverted_to: number;
  restored_files: string[];
  job: SplatJob;
}

// ── Semantic edit (POST /jobs/{id}/edit/semantic) ────────────────────────────
// Mirrors backend/edit_ops.py SemanticEditRequest exactly (verified against
// source 2026-07-25): text 1-200 chars, threshold 0..1 (backend default 0.5),
// mode delete|isolate|extract, optional name (derived-scene title for extract,
// max 80), cleanup default true (chains a -G floater pass after the row-mask
// cut — kills boundary halos, but guts legitimately wispy subjects).
export interface SplatSemanticEditRequest {
  text: string;
  threshold: number;
  mode: "delete" | "isolate" | "extract";
  name?: string | null;
  cleanup: boolean;
}

// semantic_edit response (edit_ops.py ~line 1200/1256). Two shapes by mode:
// delete/isolate rewrite THIS scene (preview snapshot first → version_before
// for /edit/revert; language_field_stale flips true and the STALE marker
// lands on disk), extract builds a NEW derived job (new_job_id) and leaves
// this scene — and its language field — untouched.
export interface SplatSemanticEditResponse {
  ok: boolean;
  mode: "delete" | "isolate" | "extract";
  matched: number; // gaussians scoring >= threshold for the query
  kept: number; // rows kept by the mask (delete: total−matched; isolate/extract: matched)
  cleanup: boolean;
  rows_before_cleanup: number; // == kept (pre-cleanup row count)
  rows_after_cleanup: number | null; // final row count after optional -G pass
  // delete/isolate only:
  version_before?: number; // snapshot seq to pass to /edit/revert for undo
  warnings?: string[]; // derived-artifact regen failures (non-fatal)
  language_field_stale?: boolean; // always true on the delete/isolate response
  // extract only:
  new_job_id?: string;
  // delete/isolate: the refreshed SOURCE job payload; extract: the NEW derived job.
  job: SplatJob;
}

// Snapshot manifest entries from GET /edit/versions (edit_ops.py _snapshot manifest).
export interface SplatEditVersion {
  seq: number;
  ts: string;
  op: string;
  params?: Record<string, unknown> | null;
  files?: string[];
}

export interface SplatEditVersionsResponse {
  job_id: string;
  versions: SplatEditVersion[];
  max_versions: number;
}

export interface SplatUploadResult {
  path: string;
  name: string;
  kind: "file" | "directory";
  is_insv: boolean;
  detail: string;
}

export interface SplatTransferEntry {
  name: string;
  path: string;
  kind: "video" | "insv" | "zip" | "images" | "image" | "dataset";
  is_insv: boolean;
  size_bytes: number;
  detail: string;
}

export interface SplatTransfersResponse {
  dir: string;
  entries: SplatTransferEntry[];
}
