// Splat API contracts (mirrors the portal's server/routes/splat.py payloads).

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
  // Best-effort/optional stages (currently: langfield) that ran but failed —
  // parallel to stages_completed, never instead of it. A failed optional
  // stage still lands in stages_completed (the stage rail treats it as
  // traversed) AND here (so the failure itself stays visible). Absent on
  // jobs with no optional-stage failures.
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
  // Persisted SfM/frame-density params (backend already returns these via the
  // meta spread; declared here so "Promote to full build" can read a scene's
  // own settings instead of falling back to request defaults that could
  // contradict how the scene was actually built).
  num_frames_target?: number;
  sfm_backend?: "colmap" | "glomap";
  // Test Flight trim window. Non-null trim_duration_s marks a scene as a
  // trimmed proof build (see the gallery card's "Promote to full build" action).
  trim_start_s?: number | null;
  trim_duration_s?: number | null;
  // Survey-lane scale calibration: meters per scene unit (nerfstudio scenes are
  // non-metric). Set from the viewer's measure tool; absent/null = uncalibrated.
  meters_per_unit?: number | null;
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
    [k: string]: unknown;
  };
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
