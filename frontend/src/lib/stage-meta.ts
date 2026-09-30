import type { SplatJob } from "@/lib/contracts";

// ── pipeline metadata ────────────────────────────────────────────────────────
export const STAGE_ORDER = ["spirula_trim", "spirula_extract", "spirula_mask", "spirula_person", "spirula_sfm", "spirula_train", "spirula_publish", "stitch", "process", "train", "langfield", "export", "health", "compress", "webopt", "mesh"];
export const STAGE_HUMAN: Record<string, string> = {
  spirula_trim: "Cutting the test-flight window",
  spirula_extract: "Pulling frames from both lenses",
  spirula_mask: "Masking the lens borders",
  spirula_person: "Masking out the camera operator",
  spirula_sfm: "Solving the 360 camera path",
  spirula_train: "Building the 3D scene (Spirula)",
  spirula_publish: "Publishing the scene",
  stitch: "Unwrapping 360 footage",
  process: "Finding camera positions",
  glomap_sfm: "Re-solving with global SfM",
  rig_sfm: "Solving 360 rig geometry",
  mast3r_sfm: "Re-solving with MASt3R (pose-free)",
  train: "Building the 3D scene",
  langfield: "Building the language field",
  export: "Finishing the scene",
  health: "Checking capture health",
  compress: "Compressing",
  webopt: "Preparing web viewer",
  mesh: "Extracting triangle mesh",
};
export const STAGE_SHORT: Record<string, string> = {
  spirula_trim: "Trim",
  spirula_extract: "Frames",
  spirula_mask: "Mask",
  spirula_person: "Operator",
  spirula_sfm: "360 path",
  spirula_train: "Train",
  spirula_publish: "Publish",
  stitch: "Stitch",
  process: "Process",
  glomap_sfm: "Global SfM",
  rig_sfm: "360 rig",
  mast3r_sfm: "MASt3R",
  train: "Train",
  langfield: "Language field",
  export: "Export",
  health: "Health",
  compress: "Compress",
  webopt: "Web",
  mesh: "Mesh",
};
// An auto-fallback solver's process step is named "reprocess<n>" on the backend so
// it never collides with the original "process" stage key — label it like Process.
export function stageShort(s: string): string {
  return STAGE_SHORT[s] || (s.startsWith("reprocess") ? "Process" : s);
}
export function stageHuman(s: string): string {
  return STAGE_HUMAN[s] || (s.startsWith("reprocess") ? "Finding camera positions" : s);
}
export const QUALITY = {
  draft: { label: "Draft", iterations: 7000, blurb: "~2 min" },
  standard: { label: "Standard", iterations: 30000, blurb: "~6 min" },
  high: { label: "High detail", iterations: 50000, blurb: "~10 min" },
} as const;
export type QualityKey = keyof typeof QUALITY;

export function humanizeStage(job: SplatJob): string {
  if (job.status === "starting") return "Getting started…";
  return job.stage ? stageHuman(job.stage) : "Working…";
}

export const MIN_ITERS = 1000;
export const MAX_ITERS = 50000;
// Rough training-time estimate (5090): ~1 min overhead + ~1 min / 5k iters.
export function trainMinutes(iters: number): number {
  return Math.max(2, Math.round(1 + iters / 5000));
}
export function presetForIters(iters: number): QualityKey | null {
  return (Object.keys(QUALITY) as QualityKey[]).find((k) => QUALITY[k].iterations === iters) ?? null;
}
