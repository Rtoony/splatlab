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
  matches: LangfieldMatch[];
}

export interface LangfieldInventoryResult {
  job_id: string;
  items: LangfieldInventoryItem[];
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
  kind: "video" | "insv" | "zip" | "images" | "dataset";
  is_insv: boolean;
  size_bytes: number;
  detail: string;
}

export interface SplatTransfersResponse {
  dir: string;
  entries: SplatTransferEntry[];
}
