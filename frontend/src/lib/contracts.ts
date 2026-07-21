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
  // Opt-in splat→mesh export (Digital Twin kernel) — optional for the same reason.
  mesh_export?: boolean;
  mesh_file_url?: string | null;
  mesh_glb_url?: string | null;
  // Survey export (mesh + scale + geo anchor → grid-placed CAD deliverables).
  survey_dxf_url?: string | null;
  survey_landxml_url?: string | null;
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
