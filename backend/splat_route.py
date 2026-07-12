"""Splat Lab — 3D Gaussian Splatting pipeline orchestration.

The route drives the Nerfstudio binaries in the `splatops` conda env directly
(ns-process-data -> ns-train splatfacto -> ns-export gaussian-splat) as a
single staged job per launch.

Persistence model (mirrors three_d.py): every job dir under outputs/3d/<id>/
holds a meta.json (atomic tmp+replace) that is the source of truth for the
job board, plus a job.log tail flushed on stage transitions. The in-memory
JOBS dict holds only live handles (process, runner task, log ring buffer) —
a portal restart keeps every finished job listable and previewable.

GPU: the train stage serialises against TRELLIS inference through
server/lib/gpu_arbiter.HEAVY_GPU_LOCK and asks the nexus-gpu-orchestrator to
evict resident services for VRAM headroom before training.

The 4D pipeline is deferred until the 3D path is validated end-to-end.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import gpu_arbiter
from operator_audit import audit_operator_event

log = logging.getLogger(__name__)

router = APIRouter()

# Standalone app: point straight at the shared splatcli tree (the portal reached
# it via a symlink; we use the real absolute path). .resolve() so user paths that
# resolve through it compare against the same root.
SPLAT_ROOT = Path("/home/rtoony/projects/splatcli").resolve()
DATA_DIR = SPLAT_ROOT / "data"
# Novice uploads land here (gitignored under data/). Each upload gets its own
# token dir so concurrent uploads never collide.
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = SPLAT_ROOT / "outputs"
DEFAULT_3D_ROOT = OUTPUTS_DIR / "3d"
# Off-LAN handoff folder. Files synced here via Syncthing (the ~/Sync mesh) or
# `pulse-share`/rsync over SSH are NOT subject to the ~100MB Cloudflare
# request-body cap that limits browser uploads through the public tunnel, so
# this is the path for splatting large captures from a remote device. The
# /transfers endpoint lists splat-ready inputs found here; the operator picks
# one and /train consumes its absolute path directly (no browser upload).
# Scoped to a per-app subfolder (~/transfers/splatlab) so Splat Lab only sees ITS
# own drop folder, not the whole shared transfers tree. Override via env if needed.
TRANSFERS_DIR = Path(
    os.environ.get("SPLAT_TRANSFERS_DIR", "").strip() or str(Path.home() / "transfers" / "splatlab")
)
TRANSFERS_DIR.mkdir(parents=True, exist_ok=True)
CONDA_ENV_BIN = Path.home() / "miniconda3" / "envs" / "splatops" / "bin"
# COLMAP lives in its own env: conda-forge colmap pins a cudatoolkit that
# conflicts with the nvcc 12.8 the splatops env needs for sm_120 (RTX 5090)
# gsplat kernel compilation. Must stay 3.11.x — colmap 4.x renamed CLI flags
# nerfstudio 1.1.5 depends on.
COLMAP_ENV_BIN = Path.home() / "miniconda3" / "envs" / "colmap" / "bin"
# COLMAP 4.x lives in its OWN env (colmap4), kept apart from the 3.11.x env
# above precisely because 4.x renamed CLI flags nerfstudio's incremental driver
# depends on. We never let ns-process-data call this binary; the glomap backend
# drives it directly (feature_extractor / sequential_matcher / global_mapper),
# then hands the resulting sparse model to ns-process-data via --skip-colmap.
# global_mapper (global SfM) registers far more frames on low-overlap captures.
COLMAP4_BIN = Path.home() / "miniconda3" / "envs" / "colmap4" / "bin" / "colmap"
# MASt3R-SfM rung (the strongest SfM escalation target). Deep-learning dense
# matcher that solves poses where COLMAP/glomap fail entirely (it needs no
# repeatable SIFT keypoints). It runs in its OWN persistent conda env
# (mast3r-spike) with its own torch 2.10+cu128 build and a 2.6GB ViT-Large
# checkpoint — kept apart from splatops so neither env's CUDA pins collide.
# The runner emits an in-memory SparseGA scene as poses.npz/points3D.npz; a
# converter then reproduces nerfstudio 1.1.5's EXACT colmap_to_json transform
# convention (verified to 4.4e-16) to write transforms.json + images/ +
# sparse_pc.ply directly — so the mast3r `process` stage produces a finished
# Nerfstudio dataset and never invokes ns-process-data at all.
MAST3R_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "mast3r-spike" / "bin" / "python"
MAST3R_RUNNER = Path.home() / "tools" / "mast3r-spike" / "run_mast3r_sfm.py"
MAST3R_CONVERTER = Path.home() / "tools" / "mast3r-spike" / "mast3r_to_nerfstudio.py"
MAST3R_CHECKPOINT = (
    Path.home()
    / "tools"
    / "mast3r-spike"
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
# swin-5-noncyclic gives a linear (not N^2) pair count — the validated default
# for sequences; 'complete' is full N^2 for small sets.
MAST3R_SCENE_GRAPH = "swin-5-noncyclic"
# "Few Photos (AI poses)" caps the input count so the complete (all-pairs) scene-graph
# stays cheap; 24 photos = 276 pairs. Above this a movie/large folder is evenly sampled.
SPARSE_MAX_IMAGES = 24

# TripoSplat generative lane ("Imagine a Splat"): one image -> a Z-up 3DGS .ply, no SfM.
TRIPOSPLAT_DIR = Path.home() / "tools" / "triposplat-spike"
TRIPOSPLAT_RUNNER = TRIPOSPLAT_DIR / "run_triposplat.sh"
TRIPOSPLAT_CKPT = TRIPOSPLAT_DIR / "ckpts" / "diffusion_models" / "triposplat_fp16.safetensors"
TRIPOSPLAT_VRAM_MB = 20_000  # model weights (~14GB) + working set; reserve generously
# splat-transform (PlayCanvas, MIT) ships in the self-hosted SuperSplat
# checkout. Used to compress the raw export into a .spz the in-page viewer
# loads ~10x faster over the tunnel. Optional: a missing binary just skips
# the compress stage, leaving the raw .ply.
SPLAT_TRANSFORM_BIN = Path.home() / "projects" / "supersplat" / "node_modules" / ".bin" / "splat-transform"
MAX_LOG_LINES = 400
MAX_SAMPLE_MEDIA = 8
MAX_LISTED_JOBS = 24
TRAIN_VRAM_MB = 16_000
# MASt3R-SfM ViT-Large inference peaks ~3.46 GB measured; reserve headroom so the
# arbiter frees enough before the heavy step (it serialises against TRELLIS).
MAST3R_VRAM_MB = 6_000
# ── Language Field (opt-in, additive) ────────────────────────────────────────────
# Text-searchable splats: SAM 2.1 masks + SigLIP 2 region features lifted onto the
# trained gaussians (training-free). ALL heavy work is SUBPROCESSED into the
# langfield-spike / sam2 conda envs by absolute path — NOTHING is added to splatlab's
# own venv (the proven MASt3R/colmap pattern). The build is a BEST-EFFORT post-train
# stage (modelled on compress/webopt): a failure logs + continues, never fails the
# splat job. Pinned scripts live in backend/langfield/ (not the scratch ~/tools dir).
LANGFIELD_DIR = Path(__file__).resolve().parent / "langfield"
LANGFIELD_RUNNER = LANGFIELD_DIR / "run_langfield.sh"
LANGFIELD_QUERY_RUNNER = LANGFIELD_DIR / "run_query.sh"
LANGFIELD_QUERY_SCRIPT = LANGFIELD_DIR / "query_render_v2.py"
LANGFIELD_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "langfield-spike" / "bin" / "python"
SAM2_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "sam2" / "bin" / "python"
LANGFIELD_DIRNAME = "_langfield"          # per-job artifact dir (sibling of _preview)
LANGFIELD_VRAM_MB = 10_000                # SAM2.1 ~5GB + SigLIP2 ~2.3GB + gsplat render
QUERY_VRAM_MB = 4_000                     # per-query render reserve (lock held ~ms)
LANGFIELD_WORKER_URL = os.environ.get("SPLAT_LANGFIELD_WORKER_URL", "http://127.0.0.1:3417")
# ── Capture-health fog gate (report-only, Capture Coach Phase 0.5) ───────────────
# Post-train reconstruction-health verdict (backend/health/fog_gate.py, calibrated
# 2026-07-11 vs RToony-graded scenes — tools/gates/gate_p0_fog_calibration.sh).
# Best-effort stage in the langfield mold: subprocessed into the langfield-spike
# env, failure never fails the job, verdict lands in meta["health"] REPORT-ONLY
# (enforcement is a later per-gate opt-in). Kill-switch: SPLAT_HEALTH_GATE=0.
HEALTH_DIR = Path(__file__).resolve().parent / "health"
HEALTH_RUNNER = HEALTH_DIR / "run_health.sh"
HEALTH_DIRNAME = "_health"                # per-job artifact dir (sibling of _langfield)
HEALTH_VRAM_MB = 4_000                    # checkpoint load + 2 rasterizations at 640px
# ── Rig-constrained 360 SfM (opt-in via sfm_backend="rig") ───────────────────────
# Root cause of the 360 fog cocoons (probe 2026-07-11, probe-operator-mask/STATUS.md):
# the legacy fan-out solves each frame's crops as FREE cameras — same-frame centers
# scatter a median 5.1 units where truth is 0 → the trajectory blows up to 12× the
# scene → normalization collapses real geometry into the fog fingerprint. This lane
# renders the stitched sphere into a RIG of virtual views (backend/rig/render_rig.py)
# and solves with COLMAP 4.x rig constraints (shared center, fixed relative
# rotations, rig-verified matching, rig+intrinsics held fixed in BA).
RIG_LANE_DIR = Path(__file__).resolve().parent / "rig"
RIG_RENDER_SCRIPT = RIG_LANE_DIR / "render_rig.py"
COLMAP4_ENV_PYTHON = Path.home() / "miniconda3" / "envs" / "colmap4" / "bin" / "python"
KEEP_UNPINNED_COMPLETED = 10
FAILED_RETENTION_HOURS = 24
PREVIEW_DIRNAME = "_preview"
STOP_GRACE_SECONDS = 10
# Web-viewer splat budget: the shareable /splat/view page loads a lightweight
# copy (spherical harmonics dropped + decimated to this many splats) so a scene
# that exports at hundreds of MB streams in ~12x smaller and orbits smoothly on
# a laptop. Best-effort: if it can't be made, the viewer falls back to the raw .ply.
WEB_DECIMATE_TARGET = 1_200_000
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
# Raw Insta360 footage: dual-fisheye in an MP4 container. Needs an ffmpeg
# v360 stitch into equirectangular before nerfstudio can reproject it.
INSV_EXTENSIONS = {".insv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# A novice uploads a folder of photos as a .zip; the server extracts it.
ARCHIVE_EXTENSIONS = {".zip"}
UPLOAD_VIDEO_EXTS = VIDEO_EXTENSIONS | INSV_EXTENSIONS
# Transfers pick-list tunables: a folder needs at least this many images to
# read as a photo set, and we list at most this many entries (newest first).
MIN_TRANSFER_IMAGES = 3
MAX_TRANSFER_ENTRIES = 60
# 2GB cap: phone clips and .insv files run hundreds of MB; disk has ~1TB free.
# NOTE: Cloudflare proxies cap request bodies at ~100MB on non-Enterprise
# plans, so uploads through splat.roonytoony.dev are limited by CF, not this —
# the LAN origin (192.168.87.34:3300) has no such cap.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
# Registration-quality gate: COLMAP/nerfstudio's "process" stage extracts N
# frames but only registers the ones it can solve camera poses for. A capture
# with low texture, motion blur, or thin overlap registers almost none — e.g.
# the backyard job solved 2 of 311 (0.6%) and would have wasted ~10 min of GPU
# training producing garbage. Below this ratio we fail FAST with guidance and
# skip training. Tunable; raise to be stricter, lower to be more permissive.
MIN_REGISTRATION_RATIO = 0.30
# 5.7K equirectangular (2:1) — the native sphere resolution of X3/X4-class
# cameras; nerfstudio downscales from here as needed.
STITCH_WIDTH = 5760
STITCH_HEIGHT = 2880
# The guidance every unsupported-.insv failure points at ("Entry B" in
# docs/360-ingestion.md): Insta360 Studio's optical-flow stitch is both the
# quality path and the universal fallback for layouts we can't auto-compose.
INSV_ENTRY_B_HINT = (
    "Export an equirectangular MP4 from Insta360 Studio and upload that instead "
    "(360 mode, no stitch stage needed)."
)
# ── Post-stitch sanity gate (R0) ────────────────────────────────────────────
# After the ffmpeg stitch, two frames are extracted and checked for the KNOWN
# corrupt signature of a mis-composed dual-fisheye (job splat_ec1b984ffb:
# COLMAP registered 2/624 because half the pano was structurally garbage).
# Thresholds are deliberately conservative — only blatant corruption trips
# them; a dark night pano or a normal v360 lens seam must never false-fail.
# Skippable via SPLAT_STITCH_SANITY=0.
STITCH_SANITY_ANALYSIS_SIZE = (512, 256)  # downscale for analysis: fast + denoising
STITCH_SANITY_NEAR_BLACK = 6              # 0-255 luminance considered "black"
STITCH_SANITY_BLACK_COL_FRAC = 0.97       # column is "dead" if >=97% of its upper half is black
STITCH_SANITY_WEDGE_MIN_FRAC = 0.30       # dead wedge must span >=30% of the width
STITCH_SANITY_CONTEXT_MIN_LUMA = 20.0     # rest of the upper half must be non-dark (night guard)
STITCH_SANITY_SPIKE_MIN = 60.0            # hard vertical cut: adjacent column-mean jump (0-255)
STITCH_SANITY_SPIKE_DOMINANCE = 12.0      # ... and >=12x the median column-to-column change
# The cross-frame "same column in both frames" seam check only has power when
# content MOVES between the two sampled frames. On a static capture (tripod,
# test pattern) every content edge sits at the same column in both frames, so
# the check false-fails healthy stitches (proved: a static test pattern AND a
# healthy hstack+v360 static dual-stream stitch both tripped it). Below this
# mean-absolute-delta floor the frames are treated as static and a seam
# finding alone may NOT fail the gate — only the wedge signature (the proven
# corruption mode) stays fatal.
STITCH_SANITY_STATIC_DELTA_FLOOR = 3.0    # mean |Δ| (0-255) between the two analysis frames
# v360 dfisheye->e places the two lens boundaries at fixed relative x positions
# (~0.25 and ~0.75 of the width for an hstack'd dual fisheye). A discontinuity
# within tolerance of those columns is an EXPECTED lens seam, never corruption.
STITCH_SANITY_LENS_COL_FRACS = (0.25, 0.75)  # expected lens-boundary columns, fraction of width
STITCH_SANITY_LENS_COL_TOL_FRAC = 0.02       # ± tolerance around a lens column, fraction of width
# nerfstudio 1.1.5 does NOT expose COLMAP's SequentialMatching.overlap (its
# colmap_utils builds "<method>_matcher" with no overlap flag), so sequential
# matching only pairs images <=10 positions apart (COLMAP default). The
# equirect fan-out names crops <frame>_<k>.jpg grouped per frame, so same-view
# temporal neighbors are exactly images_per_equirect apart: sequential works
# for 8 crops/frame (8<=10) and misses ALL temporal pairs for 14 (14>10).
NS_SEQUENTIAL_DEFAULT_OVERLAP = 10
# SfM rungs that understand the equirect fan-out. MASt3R is deliberately
# excluded: it is a perspective-pairwise matcher — pointing it at hundreds of
# reprojected crops is the wrong tool (wrong scene graph, O(n^2) pairs).
EQUIRECT_CAPABLE_SOLVERS = {"colmap", "glomap", "rig"}
# Solvers that ONLY make sense on equirect captures (rig renders virtual views
# from the stitched sphere) — escalation skips them for flat captures.
EQUIRECT_ONLY_SOLVERS = {"rig"}
# AUTO-FALLBACK solver escalation. When the A1 registration gate trips on a
# video/image-folder capture, the pipeline climbs this chain automatically
# (zero clicks) instead of failing: it reroutes to the NEXT solver after the
# current one that is AVAILABLE and not yet tried, rebuilds the SfM stage(s)
# mid-run, and re-runs the gate. Ordered weakest -> strongest.
#   - "colmap": the validated COLMAP 3.11.1 incremental path nerfstudio drives.
#   - "glomap": COLMAP 4.x global SfM (global_mapper) — registers far more
#     frames on low-overlap captures (proved 311/311 vs 2/311 incremental).
#   - "mast3r": MASt3R-SfM (deep dense matcher) — the strongest rung; solves
#     poses on captures with no repeatable SIFT features where both COLMAP
#     paths register almost nothing (proved 39 frames on the backyard).
# To add a rung later: append its name here AND add a branch in
# _sfm_stage_commands + an availability key in SFM_SOLVER_AVAILABILITY. Nothing
# in the escalation loop hardcodes the number of rungs.
#   - "rig": rig-constrained 360 SfM — equirect-only (EQUIRECT_ONLY_SOLVERS);
#     FIRST in the chain because the unrigged fan-out fog-cocoons every 360
#     capture (2026-07-11 root cause) — for equirect it is the strongest rung,
#     and flat captures skip it entirely.
SFM_ESCALATION = ["rig", "colmap", "glomap", "mast3r"]
# Per-solver availability key in _engine_availability(). A solver is only a
# valid escalation target when its key is truthy. "colmap" is always present
# (validated by _plan_3d_job up front), so it has no gating key.
SFM_SOLVER_AVAILABILITY = {
    "glomap": "glomap_available",
    "mast3r": "mast3r_available",
    # InstantSplat sparse-view rung: same MASt3R toolchain, dense-seed + low-iter path.
    # NOT in SFM_ESCALATION (it's a user-selected capture mode, not an auto-fallback rung).
    "mast3r-sparse": "mast3r_available",
    # Rig-constrained 360 SfM: user-selected, equirect-only, NOT an auto-fallback rung
    # (yet — default-flip is a separate decision after RToony grades real runs).
    "rig": "rig_available",
}
_JOB_ID_RE = re.compile(r"^splat_[0-9a-f]{6,32}$")

for directory in (DATA_DIR, UPLOADS_DIR, DEFAULT_3D_ROOT, OUTPUTS_DIR / "4d"):
    directory.mkdir(parents=True, exist_ok=True)


class SplatTrainRequest(BaseModel):
    mode: Literal["3d", "4d"]
    input_path: str = Field(..., min_length=1)
    output_dir: str | None = None
    capture_format: Literal["standard", "equirectangular360"] = "standard"
    # Nerfstudio only accepts 8 or 14 planar crops per equirectangular frame.
    images_per_equirect: Literal[8, 14] = 8
    crop_bottom: float = Field(default=0.15, ge=0.0, lt=1.0)
    num_frames_target: int = Field(default=300, ge=10, le=2000)
    max_num_iterations: int = Field(default=30000, ge=100, le=100000)
    # Per-lens field of view for the ffmpeg v360 dual-fisheye -> equirectangular
    # stitch of raw Insta360 .insv. ~204 fits X2/X3; X4/X5 may want 206-210.
    insv_fov: float = Field(default=204.0, ge=160.0, le=280.0)
    # Structure-from-Motion backend. "colmap" (default) is the validated COLMAP
    # 3.11.1 incremental path nerfstudio drives via ns-process-data — unchanged.
    # "glomap" is an OPT-IN rescue: COLMAP 4.x global SfM (global_mapper), which
    # registers far more frames on hard captures (proved 311/311 on a backyard
    # clip vs 2/311 for incremental). It runs feature_extractor + sequential
    # matching + global_mapper itself, then hands the model to ns-process-data
    # via --skip-colmap. Applied to video / image-folder inputs including
    # equirectangular 360 (the stage reproduces nerfstudio's perspective
    # fan-out before COLMAP); a pre-processed dataset falls back to the
    # default path silently.
    # "rig" is an OPT-IN equirect-only lane: renders the stitched sphere into a rig
    # of virtual views and solves with COLMAP 4.x rig constraints — fixes the
    # unrigged-crop pose scatter that fog-cocoons every 360 capture (2026-07-11
    # root cause). Falls back to colmap silently on non-equirect inputs or when
    # the toolchain is missing (same convention as an unavailable glomap).
    sfm_backend: Literal["colmap", "glomap", "rig"] = "colmap"
    # OPT-IN: build a text-searchable language field after training (SAM 2.1 +
    # SigLIP 2, training-free lift). Default off → the pipeline is byte-identical.
    # Best-effort: a build failure never fails the splat job.
    language_field: bool = False
    # OPT-IN capture mode. "standard" (default) → the pipeline is byte-identical.
    # "sparse" = InstantSplat "Few Photos (AI poses)": force the MASt3R rung with a
    # DENSE pointmap seed + complete scene-graph + low iterations, so a handful of
    # ordinary photos (2-12) yield a usable splat with no COLMAP. Requires the MASt3R
    # toolchain; only applies to image-folder / video inputs (not equirect / pre-processed).
    capture_mode: Literal["standard", "sparse"] = "standard"
    # OPT-IN source. "capture" (default) = photos/video -> the normal pipeline. "generative-
    # image" = "Imagine a Splat": a SINGLE image -> TripoSplat -> a 3DGS .ply, NO SfM/train.
    source_type: Literal["capture", "generative-image"] = "capture"
    # OPT-IN "Test Flight" trim window: stitch (and therefore the whole
    # pipeline) consumes only [trim_start_s, trim_start_s + trim_duration_s]
    # of the source .insv, proving capture + settings in minutes before a
    # multi-hour full run. trim_duration_s alone centers the window in the
    # clip. Only valid for raw .insv inputs (the stitch is the trim point);
    # any other input type is rejected loudly rather than silently ignored.
    trim_start_s: float | None = Field(default=None, ge=0.0)
    trim_duration_s: float | None = Field(default=None, ge=5.0, le=600.0)


@dataclass
class SplatJob:
    """Live handles for an in-flight job. Durable state lives in meta.json."""

    job_id: str
    output_dir: str
    input_path: str
    process: asyncio.subprocess.Process | None = None
    runner_task: asyncio.Task[None] | None = None
    pid: int | None = None
    stop_requested: bool = False
    stage_commands: dict[str, list[str]] = field(default_factory=dict)
    stages_planned: list[str] = field(default_factory=list)
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    # AUTO-FALLBACK solver escalation state. sfm_tried records every SfM solver
    # this job has already run (seeded with the starting sfm_backend) so the
    # gate never re-runs the same solver and the chain terminates. reroute_count
    # caps total reroutes at len(SFM_ESCALATION) as a belt-and-suspenders guard
    # against any infinite loop. sfm_context carries the plan-time inputs the
    # gate needs to rebuild SfM stage commands for a different solver mid-run;
    # it is only populated for escalation-eligible jobs (video / image-folder,
    # incl. equirect — NOT a pre-processed dataset).
    sfm_tried: set[str] = field(default_factory=set)
    reroute_count: int = 0
    sfm_context: dict[str, Any] | None = None
    # The original train request, kept so the gate can rebuild SfM stage commands
    # for a fallback solver mid-run (it needs num_frames_target / images-per-
    # equirect / crop-bottom). Only set for escalation-eligible jobs.
    sfm_req: SplatTrainRequest | None = None


JOBS: dict[str, SplatJob] = {}
JOBS_LOCK = asyncio.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_job_id(job_id: str) -> bool:
    return bool(_JOB_ID_RE.match(job_id))


def _job_dir(job_id: str) -> Path:
    return DEFAULT_3D_ROOT / job_id


# ---------------------------------------------------------------------------
# Meta state helpers (pattern from three_d.py)
# ---------------------------------------------------------------------------


def _new_meta(job_id: str, req: SplatTrainRequest, input_path: Path, job_dir: Path, stages: list[str]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "mode": req.mode,
        "status": "starting",
        "stage": None,
        "stages_planned": stages,
        "stages_completed": [],
        # Parallel to stages_completed: best-effort/optional stages (compress,
        # webopt, webopt-langweb, langfield) that ran but failed land here too,
        # so a failed optional stage stays distinguishable from a genuinely
        # successful one. See _record_stage_failure.
        "stages_failed": [],
        "input_path": str(input_path),
        "output_dir": str(job_dir),
        "capture_format": req.capture_format,
        # 360 sub-params, persisted so Re-run/Retry can rebuild the EXACT
        # request (they were previously dropped: a 360 re-run silently reverted
        # to the defaults). Present on every job for a uniform payload shape;
        # harmless no-ops for standard captures.
        "images_per_equirect": req.images_per_equirect,
        "crop_bottom": req.crop_bottom,
        "insv_fov": req.insv_fov,
        "max_num_iterations": req.max_num_iterations,
        # Remaining request knobs, persisted for the same reason as the 360
        # sub-params above: a Re-run/"promote Test Flight to full" must know
        # exactly what the original job ran with.
        "num_frames_target": req.num_frames_target,
        "sfm_backend": req.sfm_backend,
        "language_field": req.language_field,
        "trim_start_s": req.trim_start_s,
        "trim_duration_s": req.trim_duration_s,
        # Record the ACTUAL engaged capture mode: "sparse" only when the plan starts with
        # the dense-seed MASt3R rung (the sole plan-time producer of a leading mast3r_sfm
        # stage), so a sparse request that didn't apply (equirect/dataset) isn't mis-badged.
        "capture_mode": "sparse" if (req.capture_mode == "sparse" and stages[:1] == ["mast3r_sfm"]) else "standard",
        "source_type": req.source_type,
        "command": [],
        "created_at": _utc_now(),
        "started_at": None,
        "completed_at": None,
        "pid": None,
        "exit_code": None,
        "error_message": None,
        "stop_requested": False,
        "pinned": False,
    }


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _read_meta(job_id: str) -> dict[str, Any] | None:
    path = _meta_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _write_meta(job_id: str, meta: dict[str, Any]) -> None:
    path = _meta_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(path)


def _patch_meta(job_id: str, **fields: Any) -> dict[str, Any] | None:
    meta = _read_meta(job_id)
    if meta is None:
        return None
    meta.update(fields)
    _write_meta(job_id, meta)
    return meta


def _record_stage_failure(job_id: str, stage: str, reason: str) -> None:
    """Append a {stage, reason} record to meta['stages_failed'] — parallel to,
    never instead of, stages_completed.

    For a best-effort/optional stage (compress, webopt, webopt-langweb,
    langfield) a non-zero exit or a caught exception must NEVER flip the job
    to "failed" — the splat itself already succeeded independent of it. But
    before this helper existed, the
    only bookkeeping was appending the stage name to stages_completed
    unconditionally, which made a failed optional stage indistinguishable from
    a successful one in job meta / the API payload / the frontend. This is the
    fix: stages_completed keeps its existing meaning (semantics unchanged,
    final_status still "completed"), and stages_failed makes the failure
    durably visible alongside it.
    """
    failed = (_read_meta(job_id) or {}).get("stages_failed", [])
    _patch_meta(job_id, stages_failed=[*failed, {"stage": stage, "reason": reason[:300]}])


def _all_metas() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not DEFAULT_3D_ROOT.is_dir():
        return out
    for d in DEFAULT_3D_ROOT.iterdir():
        if not d.is_dir() or not _safe_job_id(d.name):
            continue
        meta = _read_meta(d.name)
        if meta:
            out.append(meta)
    return out


def _flush_log(job: SplatJob) -> None:
    """Persist the in-memory log ring buffer to <job_dir>/job.log."""
    try:
        log_path = Path(job.output_dir) / "job.log"
        log_path.write_text("\n".join(job.log_lines) + "\n")
    except OSError:
        pass


def _log_tail_from_disk(job_id: str) -> list[str]:
    log_path = _job_dir(job_id) / "job.log"
    if not log_path.is_file():
        return []
    try:
        return log_path.read_text().splitlines()[-MAX_LOG_LINES:]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Engine detection
# ---------------------------------------------------------------------------


def _resolve_input_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = DATA_DIR / candidate
    return candidate.resolve()


def _resolve_output_root(mode: Literal["3d", "4d"], raw_path: str | None) -> Path:
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = SPLAT_ROOT / candidate
        return candidate.resolve()
    return (OUTPUTS_DIR / mode).resolve()


def _tool_path(binary: str, env_var: str) -> str | None:
    override = os.environ.get(env_var, "").strip()
    if override:
        return override
    for env_bin in (CONDA_ENV_BIN, COLMAP_ENV_BIN):
        candidate = env_bin / binary
        if candidate.is_file():
            return str(candidate)
    return shutil.which(binary)


def _colmap4_path() -> str | None:
    """Locate the COLMAP 4.x binary used by the opt-in glomap (global SfM)
    backend. Deliberately NOT on the splatops/colmap-3.11 PATH so ns-process-data
    never picks it up; we only invoke it directly. Override via SPLAT_COLMAP4_BIN."""
    override = os.environ.get("SPLAT_COLMAP4_BIN", "").strip()
    if override:
        return override
    if COLMAP4_BIN.is_file():
        return str(COLMAP4_BIN)
    return None


def _mast3r_availability() -> dict:
    """Locate the MASt3R-SfM rung: its env python, runner, converter, and the
    2.6GB checkpoint. All four must exist for "mast3r" to be a valid escalation
    target. Each path is independently overridable via an env var so the runner
    can be relocated without a code change."""
    python = os.environ.get("SPLAT_MAST3R_PYTHON", "").strip() or str(MAST3R_ENV_PYTHON)
    runner = os.environ.get("SPLAT_MAST3R_RUNNER", "").strip() or str(MAST3R_RUNNER)
    converter = os.environ.get("SPLAT_MAST3R_CONVERTER", "").strip() or str(MAST3R_CONVERTER)
    checkpoint = os.environ.get("SPLAT_MAST3R_CHECKPOINT", "").strip() or str(MAST3R_CHECKPOINT)
    available = all(Path(p).is_file() for p in (python, runner, converter, checkpoint))
    return {
        "mast3r_available": available,
        "mast3r_python": python,
        "mast3r_runner": runner,
        "mast3r_converter": converter,
        "mast3r_checkpoint": checkpoint,
    }


def _triposplat_availability() -> dict:
    """Locate the TripoSplat generative lane: the 2-env runner + the fp16 checkpoint.
    Both must exist for source_type='generative-image' to be offered."""
    runner = os.environ.get("SPLAT_TRIPOSPLAT_RUNNER", "").strip() or str(TRIPOSPLAT_RUNNER)
    available = Path(runner).is_file() and TRIPOSPLAT_CKPT.is_file()
    return {"triposplat_available": available, "triposplat_runner": runner}


def _langfield_available() -> bool:
    """True iff the Language-Field toolchain is present: the build wrapper + both
    conda-env pythons (langfield-spike = SigLIP+lift, sam2 = masks). The heavy deps
    live in those envs; splatlab's own venv is never touched."""
    lf_py = os.environ.get("SPLAT_LANGFIELD_PYTHON", "").strip() or str(LANGFIELD_ENV_PYTHON)
    sam_py = os.environ.get("SPLAT_SAM2_PYTHON", "").strip() or str(SAM2_ENV_PYTHON)
    return all(Path(p).is_file() for p in (LANGFIELD_RUNNER, lf_py, sam_py))


def _health_available() -> bool:
    """True iff the fog-gate toolchain is present: the runner + the langfield-spike
    env python (fog_gate.py renders via nerfstudio+gsplat there). Deliberately NOT
    _langfield_available() — that also demands the sam2 env, which the gate never uses."""
    hf_py = os.environ.get("SPLAT_HEALTH_PYTHON", "").strip() or str(LANGFIELD_ENV_PYTHON)
    return all(Path(p).is_file() for p in (HEALTH_RUNNER, hf_py))


def _append_health_stage(stages: list[str]) -> None:
    """Append the report-only capture-health stage when the toolchain is present
    and the kill-switch (SPLAT_HEALTH_GATE=0) isn't set. Called exactly once by
    _plan_3d_job, right after train/export — never by the generative lane."""
    if _health_available() and os.environ.get("SPLAT_HEALTH_GATE", "").strip() != "0":
        stages.append("health")


_V360_CACHE: bool | None = None


def _ffmpeg_has_v360(ffmpeg: str | None) -> bool:
    """Cache whether this ffmpeg build ships the v360 (360-projection) filter."""
    global _V360_CACHE
    if _V360_CACHE is not None:
        return _V360_CACHE
    if not ffmpeg:
        _V360_CACHE = False
        return False
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=10)
        _V360_CACHE = " v360 " in out.stdout
    except (OSError, subprocess.SubprocessError):
        _V360_CACHE = False
    return _V360_CACHE


def _probe_video_streams(ffprobe: str, src: Path) -> dict:
    """Return {streams, width, height, dims} for the video streams, best-effort.

    width/height describe the FIRST stream (legacy callers); dims is the
    per-stream [(w, h), ...] list the .insv layout classifier needs to tell a
    dual-lens X4/X5 file (two equal square streams) from anything else.
    """
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=width,height", "-of", "json", str(src)],
            capture_output=True, text=True, timeout=20,
        )
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams", [])
        first = streams[0] if streams else {}
        return {
            "streams": len(streams),
            "width": first.get("width"),
            "height": first.get("height"),
            "dims": [(s.get("width"), s.get("height")) for s in streams],
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {"streams": 0, "width": None, "height": None, "dims": []}


def _stitch_layout(info: dict) -> tuple[str, str | None]:
    """Classify a probed .insv stream layout for the stitch planner.

    Returns ("single", None)  — one video stream: the file already carries
                                side-by-side dual-fisheye; use the legacy -vf form.
            ("dual", None)    — exactly two streams of equal square dims (X4/X5
                                one-lens-per-stream): hstack them first.
            ("error", msg)    — anything else. NO warn-and-limp: a mis-read
                                layout produces a structurally corrupt equirect
                                that wastes a full COLMAP run (splat_ec1b984ffb
                                registered 2/624), so refuse loudly instead.
    """
    n = info.get("streams", 0)
    dims = info.get("dims") or []
    if n == 1:
        return "single", None
    if n == 2 and len(dims) == 2:
        (w0, h0), (w1, h1) = dims
        if w0 and h0 and (w0, h0) == (w1, h1) and w0 == h0:
            return "dual", None
        return "error", (
            f".insv has 2 video streams but not the expected equal square fisheyes "
            f"({w0}x{h0} + {w1}x{h1}) — the auto-stitch would mis-compose them. "
            f"{INSV_ENTRY_B_HINT}"
        )
    if n == 0:
        return "error", (
            f"Could not probe any video stream in this .insv — cannot determine the "
            f"lens layout to stitch. {INSV_ENTRY_B_HINT}"
        )
    return "error", (
        f".insv has {n} video streams — an unrecognized layout the auto-stitch does "
        f"not support. {INSV_ENTRY_B_HINT}"
    )


def _engine_availability() -> dict:
    ns_train = _tool_path("ns-train", "SPLAT_NS_TRAIN_BIN")
    ns_process_data = _tool_path("ns-process-data", "SPLAT_NS_PROCESS_DATA_BIN")
    ns_export = _tool_path("ns-export", "SPLAT_NS_EXPORT_BIN")
    colmap = _tool_path("colmap", "SPLAT_COLMAP_BIN")
    colmap4 = _colmap4_path()
    ffmpeg = _tool_path("ffmpeg", "SPLAT_FFMPEG_BIN")
    return {
        "ns_train_available": bool(ns_train),
        "ns_train_path": ns_train,
        "ns_process_data_available": bool(ns_process_data),
        "ns_process_data_path": ns_process_data,
        "ns_export_available": bool(ns_export),
        "ns_export_path": ns_export,
        "colmap_available": bool(colmap),
        "colmap_path": colmap,
        # COLMAP 4.x global-SfM backend (opt-in via sfm_backend="glomap").
        "glomap_available": bool(colmap4),
        "glomap_path": colmap4,
        # Rig-constrained 360 SfM (opt-in via sfm_backend="rig", equirect only):
        # colmap4 binary + the rig render script + its env python (cv2/scipy).
        "rig_available": bool(colmap4) and RIG_RENDER_SCRIPT.is_file() and COLMAP4_ENV_PYTHON.is_file(),
        # MASt3R-SfM rung (strongest escalation target): env python + runner +
        # converter + 2.6GB checkpoint. Spread into the dict so the gate sees
        # mast3r_available / mast3r_* alongside the COLMAP keys.
        **_mast3r_availability(),
        # TripoSplat generative lane (opt-in): one image -> 3DGS .ply.
        **_triposplat_availability(),
        # Language Field (opt-in): SAM 2.1 + SigLIP 2 toolchain present?
        "langfield_available": _langfield_available(),
        "ffmpeg_available": bool(ffmpeg),
        "ffmpeg_path": ffmpeg,
        # Insta360 .insv auto-stitch needs ffmpeg's v360 filter.
        "insv_stitch_available": _ffmpeg_has_v360(ffmpeg),
        # 4D is deferred: the engine checkout exists but its CUDA deps were
        # never installed and the data-prep stage does not exist yet.
        "four_d_engine_ready": False,
        "four_d_engine_path": str(SPLAT_ROOT / "engines" / "4d_engine" / "train.py"),
    }


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    # ns-process-data resolves `colmap` via PATH, so both env bins must be on it.
    env["PATH"] = f"{CONDA_ENV_BIN}:{COLMAP_ENV_BIN}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    # gsplat JIT-compiles its CUDA kernels on first use (and after any
    # ~/.cache/torch_extensions wipe). The nvidia-channel CUDA headers/libs
    # live under targets/x86_64-linux, which nvcc's host compiler does not
    # search by default — without these the build dies on cuda_runtime.h.
    splatops_env = CONDA_ENV_BIN.parent
    env["CPATH"] = str(splatops_env / "targets" / "x86_64-linux" / "include")
    env["LIBRARY_PATH"] = f"{splatops_env / 'targets' / 'x86_64-linux' / 'lib'}:{splatops_env / 'lib'}"
    # nerfstudio 1.1.5 predates torch>=2.6's weights_only=True default and
    # cannot load its own checkpoints without this. Safe here: the pipeline
    # only loads checkpoints it just wrote to its own job directory.
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return env


def _sample_media_entries() -> list[dict]:
    entries: list[dict] = []
    for path in sorted(DATA_DIR.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if len(entries) >= MAX_SAMPLE_MEDIA:
            break
        if path.name.startswith("."):
            continue
        entries.append(
            {
                "name": path.name,
                "path": str(path),
                "kind": "directory" if path.is_dir() else "file",
            }
        )
    return entries


def _human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def _classify_transfer(path: Path) -> tuple[str, str] | None:
    """Return (kind, human-detail) if path is a usable splat input, else None.

    Deliberately strict: only surface things the pipeline can actually consume,
    so the operator never picks something that fails in COLMAP. Raw camera files
    (e.g. .CR2) are intentionally excluded — they must be developed to JPG first.
    """
    if path.is_file():
        ext = path.suffix.lower()
        if ext in INSV_EXTENSIONS:
            return "insv", f"360 footage · {_human_size(path.stat().st_size)}"
        if ext in VIDEO_EXTENSIONS:
            return "video", f"Video · {_human_size(path.stat().st_size)}"
        if ext in ARCHIVE_EXTENSIONS:
            return "zip", f"Photo zip · {_human_size(path.stat().st_size)}"
        # A single image can't be photogrammetry-captured, but it IS a valid input for
        # the generative "Imagine a Splat" lane — surface it so it's pickable there.
        if ext in IMAGE_EXTENSIONS:
            return "image", f"Single image · Imagine a Splat · {_human_size(path.stat().st_size)}"
        return None
    if path.is_dir():
        if (path / "transforms.json").is_file():
            return "dataset", "Processed Nerfstudio dataset"
        images = 0
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                images += 1
                if images >= 500:
                    break
        if images >= MIN_TRANSFER_IMAGES:
            return "images", f"{images}{'+' if images >= 500 else ''} photos"
        return None
    return None


def _transfers_entries() -> list[dict]:
    """Splat-ready inputs sitting in ~/transfers, newest first."""
    root = TRANSFERS_DIR.resolve()
    if not root.is_dir():
        return []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)
    except OSError:
        return []
    entries: list[dict] = []
    for path in children:
        if len(entries) >= MAX_TRANSFER_ENTRIES:
            break
        if path.name.startswith("."):
            continue
        try:
            classified = _classify_transfer(path)
        except OSError:
            continue
        if not classified:
            continue
        resolved = path.resolve()
        # Containment guard: never expose a path a symlink points outside ~/transfers to.
        if not resolved.is_relative_to(root):
            continue
        kind, detail = classified
        entries.append(
            {
                "name": path.name,
                "path": str(resolved),
                "kind": kind,
                "is_insv": kind == "insv",
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "detail": detail,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------


def _preview_dir_path(output_dir: Path) -> Path:
    return output_dir / PREVIEW_DIRNAME


def _preview_file_path(output_dir: Path) -> Path:
    return _preview_dir_path(output_dir) / "splat.ply"


def _preview_spz_path(output_dir: Path) -> Path:
    return _preview_dir_path(output_dir) / "splat.spz"


def _preview_web_path(output_dir: Path) -> Path:
    # Lightweight raw .ply (SH-stripped + decimated) for the web viewer.
    return _preview_dir_path(output_dir) / "web.ply"


def _preview_langweb_path(output_dir: Path) -> Path:
    # Full-count SH-stripped .ply for the CLIENT-SIDE language heatmap: its row
    # order matches gauss_emb.npz (== the raw splat.ply export order), unlike
    # web.ply which is decimated AND merge-reordered (index-incompatible).
    return _preview_dir_path(output_dir) / "langweb.ply"


def _langweb_command(transform: str, output_dir: Path) -> list[str]:
    """splat-transform argv for the langweb artifact: harmonics stripped, NO
    decimation. Probe-verified (2026-07-04, splat-transform 2.7.1, 487k-gaussian
    splatfacto export): --filter-harmonics 0 alone preserves row order bit-exactly,
    so langweb.ply row i == splat.ply row i == gauss_emb.npz row i. Adding
    --decimate would break that correspondence — never add it here."""
    return [
        transform,
        str(_preview_file_path(output_dir)),
        "--filter-harmonics",
        "0",
        str(_preview_langweb_path(output_dir)),
    ]


def _splat_transform_path() -> str | None:
    override = os.environ.get("SPLAT_TRANSFORM_BIN", "").strip()
    if override:
        return override
    if SPLAT_TRANSFORM_BIN.is_file():
        return str(SPLAT_TRANSFORM_BIN)
    return shutil.which("splat-transform")


STATS_VERSION = 1


def _ply_vertex_count(ply_path: Path) -> int | None:
    """Gaussian count = the .ply vertex count, from the header only (cheap)."""
    try:
        with ply_path.open("rb") as f:
            hdr = b""
            while b"end_header" not in hdr and len(hdr) < 8192:
                chunk = f.read(512)
                if not chunk:
                    break
                hdr += chunk
        m = re.search(rb"element vertex (\d+)", hdr)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _scene_stats(job_id: str, output_dir: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    """Cheap per-scene stats (gaussian count, source resolution, image count) for the
    gallery card. Computed once from the finished artifacts, then cached in meta.json."""
    cached = meta.get("stats")
    if isinstance(cached, dict) and cached.get("v") == STATS_VERSION:
        return cached
    splat = _preview_file_path(output_dir)
    if not splat.is_file():
        return None  # scene not finished — compute on a later poll
    stats: dict[str, Any] = {"v": STATS_VERSION}
    gc = _ply_vertex_count(splat)
    if gc:
        stats["gaussians"] = gc
    for cand in output_dir.rglob("transforms.json"):
        try:
            d = json.loads(cand.read_text())
            if "w" in d and "h" in d:
                stats["width"], stats["height"] = int(d["w"]), int(d["h"])
            if isinstance(d.get("frames"), list):
                stats["images"] = len(d["frames"])
        except Exception:
            pass
        break
    try:
        _patch_meta(job_id, stats=stats)
    except Exception:
        pass
    return stats


def _job_payload(meta: dict[str, Any], live: SplatJob | None = None) -> dict:
    job_id = meta["job_id"]
    output_dir = Path(meta["output_dir"])
    preview_file = _preview_file_path(output_dir)
    preview_spz = _preview_spz_path(output_dir)
    langfield_built = (output_dir / LANGFIELD_DIRNAME / "gauss_emb.npz").is_file()
    if live is not None:
        log_lines = list(live.log_lines)
        pid = live.pid
    else:
        log_lines = _log_tail_from_disk(job_id)
        pid = None
    return {
        **meta,
        "pid": pid,
        "log_lines": log_lines,
        "preview_available": preview_file.is_file(),
        # Raw .ply: in-page viewer, download, SuperSplat, engine interchange.
        "preview_file_url": f"/api/splat/jobs/{job_id}/preview/file" if preview_file.is_file() else None,
        # A .spz copy exists (10x smaller) for download / modern viewers, BUT
        # the in-page mkkellogg 0.4.7 viewer cannot decode splat-transform's
        # newer SPZ container (non-gzip "NG" header -> decompression error), so
        # the viewer always loads the .ply. preview_compressed is informational.
        "preview_compressed": preview_spz.is_file(),
        "preview_spz_url": f"/api/splat/jobs/{job_id}/preview/file?fmt=spz" if preview_spz.is_file() else None,
        "preview_view_url": f"/api/splat/jobs/{job_id}/preview/file" if preview_file.is_file() else None,
        # Lightweight copy for the shareable /splat/view page; fmt=web falls back
        # to the raw .ply server-side, so this is offered whenever a preview exists.
        "preview_web_url": f"/api/splat/jobs/{job_id}/preview/file?fmt=web" if preview_file.is_file() else None,
        # Opt-in language field: a built per-gaussian feature sidecar exists -> the
        # scene is text-searchable (the viewer shows the query UI when this is true).
        "langfield_available": langfield_built,
        # Client-side heatmap splat: SH-stripped, FULL count, row order == gauss_emb.npz.
        # fmt=langweb falls back to the raw .ply server-side (same order), so this is
        # offered whenever the scene has both a preview and a built language field.
        "preview_langweb_url": (
            f"/api/splat/jobs/{job_id}/preview/file?fmt=langweb"
            if langfield_built and preview_file.is_file()
            else None
        ),
        # Cheap per-scene stats for the gallery card (gaussian count, resolution, images).
        "stats": _scene_stats(job_id, output_dir, meta),
    }


def _find_latest_config(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("config.yml"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_scene_transforms(output_dir: Path) -> Path | None:
    preferred = output_dir / "processed" / "transforms.json"
    if preferred.is_file():
        return preferred
    candidates = sorted(output_dir.rglob("transforms.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _find_dataparser_transforms(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("dataparser_transforms.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _mat3x4_apply(mat: list[list[float]], pose: list[list[float]]) -> list[list[float]]:
    """Apply Nerfstudio's saved dataparser 3x4 transform to a camera-to-world pose."""
    out: list[list[float]] = []
    for r in range(3):
        row: list[float] = []
        for c in range(4):
            value = mat[r][3] if c == 3 else 0.0
            for k in range(3):
                value += mat[r][k] * pose[k][c]
            row.append(float(value))
        out.append(row)
    return out


def _mat4_from_3x4(mat: list[list[float]]) -> list[list[float]]:
    return [[float(mat[r][c]) for c in range(4)] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _mat4_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]


def _mat4_inverse_affine(mat: list[list[float]]) -> list[list[float]]:
    a, b, c = mat[0][0], mat[0][1], mat[0][2]
    d, e, f = mat[1][0], mat[1][1], mat[1][2]
    g, h, i = mat[2][0], mat[2][1], mat[2][2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("applied_transform is not invertible")
    inv_det = 1.0 / det
    inv3 = [
        [(e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det],
        [(f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det],
        [(d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det],
    ]
    t = [mat[0][3], mat[1][3], mat[2][3]]
    inv_t = [-sum(inv3[r][k] * t[k] for k in range(3)) for r in range(3)]
    return [inv3[0] + [inv_t[0]], inv3[1] + [inv_t[1]], inv3[2] + [inv_t[2]], [0.0, 0.0, 0.0, 1.0]]


def _compose_saved_pose_transform(
    dataparser_transform: list[list[float]], applied_transform: Any | None
) -> list[list[float]]:
    """Return saved-transforms.json pose -> trained/viewer-frame transform.

    Nerfstudio saves dataparser_transforms.json as original-data -> output frame.
    The frame transform_matrix entries in processed/transforms.json are already in
    the saved dataset frame, so undo the dataset applied_transform before applying
    the saved dataparser transform. Without this, camera paths get drawn on the
    wrong up axis.
    """
    mat = _mat4_from_3x4(dataparser_transform)
    if isinstance(applied_transform, list) and len(applied_transform) == 3:
        applied = _mat4_from_3x4([[float(applied_transform[r][c]) for c in range(4)] for r in range(3)])
        mat = _mat4_multiply(mat, _mat4_inverse_affine(applied))
    return [mat[0], mat[1], mat[2]]


def _vec_norm(v: list[float]) -> list[float]:
    mag = sum(x * x for x in v) ** 0.5
    if mag <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x / mag) for x in v]


def _vec_distance(a: list[float], b: list[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _camera_display_scale(positions: list[list[float]]) -> float:
    if len(positions) < 2:
        return 0.08
    mins = [min(p[i] for p in positions) for i in range(3)]
    maxs = [max(p[i] for p in positions) for i in range(3)]
    diag = _vec_distance(mins, maxs)
    return max(0.025, min(diag * 0.035, 0.18))


def _export_command(ns_export: str, config_path: Path, output_dir: Path) -> list[str]:
    preview_dir = _preview_dir_path(output_dir)
    return [
        ns_export,
        "gaussian-splat",
        "--load-config",
        str(config_path),
        "--output-dir",
        str(preview_dir),
        "--output-filename",
        _preview_file_path(output_dir).name,
    ]


def _stitched_path(job_dir: Path) -> Path:
    return job_dir / "stitched" / "equirect.mp4"


def _stitch_cpu_leash() -> list[str]:
    """argv prefix bounding the stitch's CPU footprint (taskset + nice).

    2026-07-04 hard reset: firmware logged a FATAL CrashLog (BERT) the instant
    the stitch's all-core x264 encode launched — an idle→full-package power
    step with the 5090 still in its post-train power state. Confining the
    encode to half the cores makes that step smaller and gentler, and keeps
    the box responsive during stitches. taskset and nice both exec through,
    so job.pid still lands on ffmpeg and stop/kill semantics are unchanged.

    SPLAT_STITCH_CPUS: max CPUs for the stitch (default: half the cores,
    floor 4). 0 disables the leash entirely.
    """
    raw = os.environ.get("SPLAT_STITCH_CPUS", "").strip()
    default = max(4, (os.cpu_count() or 8) // 2)
    try:
        count = int(raw) if raw else default
    except ValueError:
        count = default
    if count <= 0:
        return []
    prefix: list[str] = []
    taskset = shutil.which("taskset")
    if taskset:
        prefix += [taskset, "-c", f"0-{count - 1}"]
    nice = shutil.which("nice")
    if nice:
        prefix += [nice, "-n", "10"]
    return prefix


def _probe_video_duration(ffprobe: str, src: Path) -> float | None:
    """Container duration in seconds, best-effort (None on any failure)."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(src)],
            capture_output=True, text=True, timeout=20,
        )
        raw = json.loads(out.stdout or "{}").get("format", {}).get("duration")
        return float(raw) if raw is not None else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _stitch_command(
    ffmpeg: str, src: Path, dst: Path, fov: float, dual_stream: bool = False,
    trim_start: float | None = None, trim_duration: float | None = None,
) -> list[str]:
    """ffmpeg v360: Insta360 dual-fisheye -> equirectangular MP4 (open-source,
    no SDK). Seams are visible at the lens boundary but splatfacto tolerates
    them; for seamless output, export equirect from the Insta360 app instead.

    dual_stream=False (legacy, byte-for-byte the original command): the single
    video stream already carries both fisheyes side-by-side; -vf v360 reads it
    directly.

    dual_stream=True (X4/X5): the .insv carries TWO video streams, one square
    fisheye per lens. Plain -vf reads ONLY stream 0 -> a structurally corrupt
    equirect (proved: COLMAP registered 2/624 on splat_ec1b984ffb). So hstack
    the two streams into one side-by-side dual-fisheye first, THEN v360 it —
    validated on the real X4 capture (two clean circles -> coherent pano).

    Both paths run under _stitch_cpu_leash() — the all-core x264 launch is
    the proven trigger of the 07-04 hardware reset.
    """
    v360 = (
        f"v360=input=dfisheye:output=e:ih_fov={fov}:iv_fov={fov}"
        f":w={STITCH_WIDTH}:h={STITCH_HEIGHT}:interp=lanczos"
    )
    leash = _stitch_cpu_leash()
    # Test Flight trim: INPUT-side -ss/-t is a fast container seek that bounds
    # every mapped stream of the single input (both lenses of a dual-stream
    # X4/X5 file), so the expensive 5760x2880 x264 encode only ever sees the
    # window. Output-side placement would decode the whole clip — don't.
    trim: list[str] = []
    if trim_start is not None or trim_duration is not None:
        trim = ["-ss", f"{trim_start or 0.0:.3f}"]
        if trim_duration is not None:
            trim += ["-t", f"{trim_duration:.3f}"]
    if dual_stream:
        filter_complex = f"[0:v:0][0:v:1]hstack=inputs=2[d];[d]{v360}[eq]"
        return [
            *leash, ffmpeg, "-y", *trim, "-i", str(src),
            "-filter_complex", filter_complex, "-map", "[eq]",
            "-c:v", "libx264", "-crf", "18", "-an", str(dst),
        ]
    return [*leash, ffmpeg, "-y", *trim, "-i", str(src), "-vf", v360, "-c:v", "libx264", "-crf", "18", "-an", str(dst)]


def _equirect_frame_corruption(rows: list[list[int]]) -> tuple[str, int] | None:
    """Detect the blatant corrupt-stitch signature in ONE analysis frame.

    rows = grayscale pixel rows (already downscaled to the analysis size).
    Returns (kind, column) or None:
      - ("wedge", start_col): a large contiguous near-black wedge in the UPPER
        half (wrap-aware — panoramas are circular) while the rest of the upper
        half has real content. This is what reading only lens 0 produces.
      - ("seam", col): a hard vertical discontinuity — one adjacent-column
        luminance jump that towers over the frame's normal column-to-column
        variation. Content edges move between frames; a projection fault sits
        at a FIXED column, which the caller cross-checks across two frames.
    Pure python on a ~512x256 grid (no numpy: the runtime venv ships Pillow
    only). Thresholds are conservative by design — catch only blatant
    corruption, never a healthy pano (incl. dark night scenes: the wedge check
    requires the REST of the sky to be non-dark before it may fire).
    """
    height = len(rows)
    width = len(rows[0]) if height else 0
    if height < 8 or width < 16:
        return None
    half = height // 2

    # ── Wedge: per-column near-black fraction over the upper half ──
    dead_cols: list[bool] = []
    col_upper_sums: list[float] = []
    for c in range(width):
        black = 0
        total = 0
        for r in range(half):
            v = rows[r][c]
            total += v
            if v <= STITCH_SANITY_NEAR_BLACK:
                black += 1
        dead_cols.append(black / half >= STITCH_SANITY_BLACK_COL_FRAC)
        col_upper_sums.append(total / half)

    # Longest contiguous dead run, wrap-aware (scan the doubled array, cap at width).
    best_run, best_start, run, run_start = 0, 0, 0, 0
    for i in range(2 * width):
        if dead_cols[i % width]:
            if run == 0:
                run_start = i % width
            run += 1
            if min(run, width) > best_run:
                best_run, best_start = min(run, width), run_start
        else:
            run = 0
    if best_run >= STITCH_SANITY_WEDGE_MIN_FRAC * width:
        alive = [col_upper_sums[c] for c in range(width) if not dead_cols[c]]
        if alive and (sum(alive) / len(alive)) > STITCH_SANITY_CONTEXT_MIN_LUMA:
            return ("wedge", best_start)

    # ── Seam: adjacent full-height column-mean jump, magnitude + dominance ──
    col_means = [sum(rows[r][c] for r in range(height)) / height for c in range(width)]
    diffs = [abs(col_means[c + 1] - col_means[c]) for c in range(width - 1)]
    if diffs:
        spike = max(diffs)
        spike_col = diffs.index(spike)
        median = sorted(diffs)[len(diffs) // 2]
        if spike >= STITCH_SANITY_SPIKE_MIN and spike >= STITCH_SANITY_SPIKE_DOMINANCE * max(median, 1.0):
            return ("seam", spike_col)
    return None


def _load_analysis_rows(image_path: Path) -> list[list[int]] | None:
    """Load a frame as a downscaled grayscale row grid for the corruption check.
    Returns None (caller skips the check) if Pillow is unavailable or the read fails."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            g = im.convert("L").resize(STITCH_SANITY_ANALYSIS_SIZE)
            w, h = g.size
            data = list(g.getdata())
        return [data[r * w:(r + 1) * w] for r in range(h)]
    except Exception:
        return None


def _frames_mean_abs_delta(a: list[list[int]], b: list[list[int]]) -> float:
    """Mean absolute per-pixel luminance delta between two analysis-row grids.

    Near-zero means the two sampled frames are (near-)identical — a static
    capture — where the cross-frame same-column seam check has no power.
    """
    total = 0
    count = 0
    for row_a, row_b in zip(a, b):
        for va, vb in zip(row_a, row_b):
            total += abs(va - vb)
            count += 1
    return total / count if count else 0.0


def _is_lens_boundary_column(col: int, width: int) -> bool:
    """True when `col` sits within tolerance of a v360 dfisheye->e lens seam.

    For an hstack'd dual fisheye stitched to equirect the lens boundaries land
    at fixed relative x positions (~0.25 and ~0.75 of the width). A hard
    vertical discontinuity there is an expected lens seam — splatfacto
    tolerates it (see _stitch_command) — never the corruption signature.
    """
    if width <= 0:
        return False
    tol = max(2, int(STITCH_SANITY_LENS_COL_TOL_FRAC * width))
    return any(abs(col - frac * width) <= tol for frac in STITCH_SANITY_LENS_COL_FRACS)


def _stitch_sanity_check(job: "SplatJob", stitched: Path) -> str | None:
    """R0 gate: cheaply validate the stitched equirect BEFORE COLMAP burns time.

    Checks, in order (any tooling hiccup logs a note and skips — the gate must
    never fail a healthy job because ffprobe/Pillow misbehaved):
      1. 2:1 aspect (an equirect that isn't 2:1 is structurally wrong).
      2. Extract 2 frames (~25% / ~75% of the clip) next to the stitched file
         (kept on disk as receipts) and run _equirect_frame_corruption on each.
         The job only fails when BOTH frames show the same corruption class —
         and for "seam", at (near) the same column, since a projection fault is
         position-fixed while real content edges move with the camera.
    Returns an actionable STITCH-scoped error message, or None when sane.
    Skippable via SPLAT_STITCH_SANITY=0.
    """
    if os.environ.get("SPLAT_STITCH_SANITY", "").strip() == "0":
        job.log_lines.append("[stitch] sanity gate skipped (SPLAT_STITCH_SANITY=0).")
        return None
    ffprobe = _tool_path("ffprobe", "SPLAT_FFPROBE_BIN")
    ffmpeg = _tool_path("ffmpeg", "SPLAT_FFMPEG_BIN")
    if not ffprobe or not ffmpeg:
        job.log_lines.append("[stitch] sanity gate skipped (ffprobe/ffmpeg not found).")
        return None

    info = _probe_video_streams(ffprobe, stitched)
    width, height = info.get("width"), info.get("height")
    if not width or not height:
        job.log_lines.append("[stitch] sanity gate skipped (could not probe stitched output).")
        return None
    if abs(width - 2 * height) > 4:
        return (
            f"Stitched output is {width}x{height}, not the 2:1 equirectangular shape — "
            f"the stitch mis-composed this file. This is a stitching problem, not a "
            f"capture problem. {INSV_ENTRY_B_HINT}"
        )

    duration = 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(stitched)],
            capture_output=True, text=True, timeout=20,
        )
        duration = float((out.stdout or "0").strip() or 0.0)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    timestamps = [duration * 0.25, duration * 0.75] if duration > 2.0 else [0.0, max(duration - 0.1, 0.0)]

    findings: list[tuple[str, int] | None] = []
    frame_rows: list[list[list[int]]] = []
    for idx, ts in enumerate(timestamps):
        frame_path = stitched.parent / f"sanity_{idx}.png"
        try:
            proc = subprocess.run(
                [ffmpeg, "-y", "-ss", f"{ts:.3f}", "-i", str(stitched), "-frames:v", "1", str(frame_path)],
                capture_output=True, text=True, timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is None or proc.returncode != 0 or not frame_path.is_file():
            job.log_lines.append(f"[stitch] sanity gate: could not extract frame @{ts:.1f}s; check skipped.")
            return None
        rows = _load_analysis_rows(frame_path)
        if rows is None:
            job.log_lines.append("[stitch] sanity gate skipped (Pillow unavailable or unreadable frame).")
            return None
        frame_rows.append(rows)
        findings.append(_equirect_frame_corruption(rows))

    first, second = findings[0], findings[1]
    corrupt = False
    reason = ""
    if first and second and first[0] == second[0]:
        if first[0] == "wedge":
            # The near-black wedge is the PROVEN corruption mode (reading only
            # lens 0 of a dual-stream .insv) — fatal on its own, motion or not.
            corrupt = True
            reason = "a large near-black wedge covers the upper half in both sampled frames"
        elif abs(first[1] - second[1]) <= max(8, int(0.02 * STITCH_SANITY_ANALYSIS_SIZE[0])):
            seam_col = first[1]
            analysis_width = STITCH_SANITY_ANALYSIS_SIZE[0]
            frames_delta = _frames_mean_abs_delta(frame_rows[0], frame_rows[1])
            if _is_lens_boundary_column(seam_col, analysis_width):
                # Expected v360 lens seam (~0.25/0.75 of width) — never corruption.
                job.log_lines.append(
                    f"[stitch] sanity gate: vertical discontinuity at column ~{seam_col} is a known "
                    f"v360 lens-boundary position (~0.25/0.75 of width) — expected lens seam, not corruption."
                )
            elif frames_delta < STITCH_SANITY_STATIC_DELTA_FLOOR:
                # Static capture: content doesn't move between the sampled frames,
                # so "same column in both frames" carries no signal. A seam finding
                # alone must NOT fail the gate here (proved false-positive on a
                # static test pattern AND a healthy static dual-stream stitch);
                # only wedge-scale corroboration (handled above) stays fatal.
                job.log_lines.append(
                    f"[stitch] sanity gate: seam candidate at column ~{seam_col} on near-identical frames "
                    f"(mean |Δ|={frames_delta:.2f} < {STITCH_SANITY_STATIC_DELTA_FLOOR}) — static capture, "
                    f"cross-frame check has no power; not failing on the seam signal alone."
                )
            else:
                corrupt = True
                reason = f"a hard vertical discontinuity sits at the same column (~{seam_col}) in both sampled frames"
    if corrupt:
        return (
            f"Stitched 360 panorama failed the coherence check: {reason}. The stitch "
            f"mis-composed this capture — this is a stitching problem, not a capture-"
            f"technique problem, and COLMAP would register almost nothing from it. "
            f"{INSV_ENTRY_B_HINT} (Set SPLAT_STITCH_SANITY=0 to bypass this gate.)"
        )
    job.log_lines.append("[stitch] sanity gate passed (2:1 aspect, both sampled frames coherent).")
    return None


# ---------------------------------------------------------------------------
# Pipeline planning
# ---------------------------------------------------------------------------


def _colmap_image_dir(job_dir: Path) -> Path:
    """Frames/photos COLMAP 4.x runs SfM on, also the --data ns-process-data
    copies+downscales from (same basenames => model image names line up)."""
    return job_dir / "colmap" / "images"


def _glomap_sfm_command(
    colmap4: str,
    ffmpeg: str,
    job_dir: Path,
    processed_dir: Path,
    process_input: Path,
    is_video: bool,
    num_frames_target: int,
    is_equirect: bool = False,
    images_per_equirect: int = 8,
    crop_bottom: float = 0.0,
) -> list[str]:
    """One self-contained shell command for the opt-in global-SfM backend.

    Runs, in order, all writing under <job_dir>/colmap/:
      0. (video only) ffmpeg extracts ~num_frames_target evenly-spaced frames
         into colmap/images/ — so COLMAP and ns-process-data see identical
         filenames. (image input: the frames are symlinked/copied in as-is.)
      0b. (equirect only) the raw frames land in colmap/equirect_frames/ instead,
         then nerfstudio's OWN equirect_utils (splatops env python) fans each
         one out into images_per_equirect perspective crops — the exact fan-out
         (naming <frame>_<k>.jpg, crop_factor=(0, crop_bottom, 0, 0)) that
         ns-process-data --camera-type equirectangular performs — into
         colmap/images/. COLMAP 4.x then solves the CROPS, and because we drive
         its CLI directly (nerfstudio 1.1.5 exposes no overlap flag) we set
         SequentialMatching.overlap to 2x images_per_equirect so same-view
         temporal neighbors (exactly images_per_equirect apart in filename
         order) are always paired. The fan-out's torch remap runs on the GPU
         briefly (~tens of MB working set) — lockless like the other light SfM
         stages, per the documented precedent.
      1. colmap feature_extractor  (single shared camera; SIFT on GPU)
      2. colmap sequential_matcher (loop closure ON — same temporal-ordering
         intent as the incremental path's --matching-method sequential)
      3. colmap global_mapper      -> colmap/sparse/0/{cameras,images,points3D}.bin

    Emitted as `bash -c` so the existing single-exec stage runner runs it
    unchanged; failure of any step aborts the stage (set -e) and the A1
    registration gate on the following `process` stage still applies.
    """
    img_dir = _colmap_image_dir(job_dir)
    db_path = job_dir / "colmap" / "database.db"
    sparse_dir = job_dir / "colmap" / "sparse"
    model_dir = sparse_dir / "0"
    equirect_dir = job_dir / "colmap" / "equirect_frames"
    # Raw ingest target: perspective inputs go straight to the COLMAP image
    # dir; equirect frames land in a staging dir and are fanned out from there.
    frames_dst = equirect_dir if is_equirect else img_dir

    if is_video:
        # -vf fps drops the clip to ~num_frames_target frames over its length;
        # COLMAP solves more frames than it can register, so over-sample lightly.
        # We compute the fps from duration so the count tracks the request even
        # for clips of unknown length: take every Nth frame via select.
        extract = (
            f'ffmpeg -y -i "{process_input}" '
            f'-vf "select=not(mod(n\\,$STRIDE))" -vsync vfr '
            f'-q:v 2 "{frames_dst}/frame_%05d.jpg"'
        )
        # STRIDE = max(1, total_frames / target). Probe frame count with ffprobe;
        # fall back to stride 1 (keep all) if probing fails.
        prelude = (
            f'FFPROBE="$(dirname "{ffmpeg}")/ffprobe"; '
            f'[ -x "$FFPROBE" ] || FFPROBE=ffprobe; '
            f'TOTAL="$("$FFPROBE" -v error -count_frames -select_streams v:0 '
            f'-show_entries stream=nb_read_frames -of csv=p=0 "{process_input}" 2>/dev/null)"; '
            f'case "$TOTAL" in ""|*[!0-9]*) TOTAL=0;; esac; '
            f'if [ "$TOTAL" -gt {num_frames_target} ]; then '
            f'STRIDE=$(( TOTAL / {num_frames_target} )); else STRIDE=1; fi; '
        )
        ingest = prelude + extract
    else:
        # Image folder: copy the source images in so COLMAP and ns-process-data
        # share one directory of identical basenames.
        ingest = (
            f'shopt -s nullglob nocaseglob; '
            f'for f in "{process_input}"/*.jpg "{process_input}"/*.jpeg '
            f'"{process_input}"/*.png "{process_input}"/*.webp '
            f'"{process_input}"/*.bmp "{process_input}"/*.tif "{process_input}"/*.tiff; '
            f'do cp -n "$f" "{frames_dst}/"; done'
        )

    if is_equirect:
        # Fan the staged equirect frames out into perspective crops with
        # nerfstudio's own equirect_utils (splatops env), reproducing what
        # ns-process-data --camera-type equirectangular does before COLMAP:
        # same crop naming (<frame>_<k>.jpg), same resolution heuristic, same
        # crop_factor mapping (crop_bottom -> (0, cb, 0, 0)). The crops are
        # moved into img_dir so feature_extractor + ns-process-data
        # --skip-colmap both see one directory of identical basenames.
        splatops_python = CONDA_ENV_BIN / "python"
        fanout = (
            f'"{splatops_python}" - <<\'PYEOF\'\n'
            f'import shutil\n'
            f'from pathlib import Path\n'
            f'from nerfstudio.process_data import equirect_utils\n'
            f'src = Path("{equirect_dir}")\n'
            f'dst = Path("{img_dir}")\n'
            f'size = equirect_utils.compute_resolution_from_equirect(src, {images_per_equirect})\n'
            f'equirect_utils.generate_planar_projections_from_equirectangular(\n'
            f'    src, size, {images_per_equirect}, crop_factor=(0.0, {crop_bottom}, 0.0, 0.0)\n'
            f')\n'
            f'dst.mkdir(parents=True, exist_ok=True)\n'
            f'for p in sorted((src / "planar_projections").iterdir()):\n'
            f'    shutil.move(str(p), dst / p.name)\n'
            f'shutil.rmtree(src / "planar_projections", ignore_errors=True)\n'
            # The trailing `true` gives the outer single-line command chain a
            # token to attach its `; ` to after the heredoc terminator line
            # (a line starting with `;` would be a bash syntax error).
            f'PYEOF\n'
            f'true'
        )
        ingest = ingest + '; ' + fanout
        # Clear BOTH the staging dir and img_dir on entry: an auto-fallback
        # reroute from colmap would otherwise mix stale crops into the fan-out.
        extra_clear = f' "{equirect_dir}" "{img_dir}"'
        mkdirs = f'"{equirect_dir}" "{img_dir}" "{sparse_dir}"'
        # Same-view temporal neighbors are exactly images_per_equirect apart in
        # filename order (crops are grouped per frame); COLMAP's default
        # overlap of 10 misses them for 14 crops/frame. 2x covers the whole
        # next frame either side — cheap next to global_mapper.
        overlap_flag = f'--SequentialMatching.overlap {2 * images_per_equirect} '
    else:
        extra_clear = ''
        mkdirs = f'"{img_dir}" "{sparse_dir}"'
        overlap_flag = ''

    script = (
        f'set -euo pipefail; '
        # rm the sparse dir first so a stale sparse/0 from an aborted prior run
        # can't satisfy the final test -f and feed nerfstudio a partial model.
        # ALSO rm processed_dir: on a colmap->glomap auto-fallback reroute the
        # earlier colmap `process` already populated it (transforms.json + images/
        # + images_2/4/8); ns-process-data does NOT guarantee a clean overwrite, so
        # the A1 registration gate could otherwise measure a stale colmap/glomap
        # mix. Mirrors the mast3r path's clear. (Fresh glomap-start jobs: no-op.)
        f'rm -rf "{sparse_dir}" "{processed_dir}"{extra_clear}; mkdir -p {mkdirs}; '
        f'{ingest}; '
        f'"{colmap4}" feature_extractor '
        f'--database_path "{db_path}" --image_path "{img_dir}" '
        # COLMAP 4.x renamed the GPU toggle: SiftExtraction.use_gpu (3.x) ->
        # FeatureExtraction.use_gpu (4.x). GPU is the default; we set it
        # explicitly so a config change can't silently drop us to CPU.
        f'--ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1; '
        f'"{colmap4}" sequential_matcher '
        f'--database_path "{db_path}" '
        # 4.x: SiftMatching.use_gpu -> FeatureMatching.use_gpu.
        f'{overlap_flag}--SequentialMatching.loop_detection 1 --FeatureMatching.use_gpu 1; '
        f'"{colmap4}" global_mapper '
        f'--database_path "{db_path}" --image_path "{img_dir}" '
        f'--output_path "{sparse_dir}"; '
        # global_mapper writes its first reconstruction to sparse/0/ already;
        # assert the expected model exists so the stage fails loud if SfM produced
        # nothing (rather than letting ns-process-data choke downstream).
        f'test -f "{model_dir}/cameras.bin" || exit 1; '
        # And assert it triangulated actual 3D points: glomap can pose every
        # camera yet emit ZERO points ("Cannot run bundle adjustment: no 3D
        # points to optimize") and still exit 0 — an empty points3D.bin is
        # exactly its 8-byte count header, and ns-process-data downstream
        # einsum-crashes on the empty array while ALSO exiting 0 (proven live,
        # splat_9da9dff4b2). Fail HERE with the real reason instead.
        f'PTS=$(stat -c%s "{model_dir}/points3D.bin" 2>/dev/null || echo 0); '
        f'if [ "$PTS" -le 8 ]; then '
        f'echo "[glomap] FATAL: cameras were posed but 0 3D points were triangulated '
        f'(points3D.bin=$PTS bytes). The footage window may lack camera movement or '
        f'frame density is too low." >&2; exit 1; fi'
    )
    return ["bash", "-c", script]


def _mast3r_image_dir(job_dir: Path) -> Path:
    """Frames MASt3R-SfM runs ViT inference on. For video we extract frames here
    first; for an image folder we copy the originals in. The converter reads
    THIS same dir as --src-images so its --image-mode resized/fullres logic sees
    identical basenames to the model's image names."""
    return job_dir / "mast3r" / "images"


def _mast3r_sfm_command(
    availability: dict,
    ffmpeg: str,
    job_dir: Path,
    processed_dir: Path,
    process_input: Path,
    is_video: bool,
    num_frames_target: int,
    dense: bool = False,
    scene_graph: str = MAST3R_SCENE_GRAPH,
    max_images: int = 0,
) -> list[str]:
    """One self-contained `bash -c` command for the MASt3R-SfM rung.

    Runs, in order:
      0. (video only) ffmpeg extracts ~num_frames_target evenly-spaced frames
         into mast3r/images/. (image input: originals copied in as-is.)
      1. run_mast3r_sfm.py  -> mast3r/out/{poses.npz, points3D.npz, ...}
         (ViT-Large dense matching on the GPU; the heavy step.)
      2. mast3r_to_nerfstudio.py  -> writes transforms.json + images/ +
         sparse_pc.ply DIRECTLY into <processed_dir>, reproducing nerfstudio
         1.1.5's colmap_to_json convention (verified to 4.4e-16). There is NO
         ns-process-data call: the converter output IS the finished dataset, so
         the downstream `train` stage consumes processed_dir unchanged.

    GPU NOTE (reviewer flag): like the colmap/glomap process stages, the MASt3R
    ViT inference here runs OUTSIDE gpu_arbiter.HEAVY_GPU_LOCK — only the `train`
    stage takes that lock. This matches existing SfM-stage behavior; the run is
    short and modest (~3.46GB peak VRAM measured), but it is NOT serialised
    against TRELLIS inference. Consistent with colmap/glomap, flagged for review.
    """
    img_dir = _mast3r_image_dir(job_dir)
    run_out = job_dir / "mast3r" / "out"

    if is_video:
        # Identical frame-extraction approach to the glomap path: stride-sample
        # ~num_frames_target evenly-spaced frames so MASt3R and the converter
        # both see one directory of identical basenames.
        extract = (
            f'ffmpeg -y -i "{process_input}" '
            f'-vf "select=not(mod(n\\,$STRIDE))" -vsync vfr '
            f'-q:v 2 "{img_dir}/frame_%05d.jpg"'
        )
        prelude = (
            f'FFPROBE="$(dirname "{ffmpeg}")/ffprobe"; '
            f'[ -x "$FFPROBE" ] || FFPROBE=ffprobe; '
            f'TOTAL="$("$FFPROBE" -v error -count_frames -select_streams v:0 '
            f'-show_entries stream=nb_read_frames -of csv=p=0 "{process_input}" 2>/dev/null)"; '
            f'case "$TOTAL" in ""|*[!0-9]*) TOTAL=0;; esac; '
            f'if [ "$TOTAL" -gt {num_frames_target} ]; then '
            f'STRIDE=$(( TOTAL / {num_frames_target} )); else STRIDE=1; fi; '
        )
        ingest = prelude + extract
    else:
        ingest = (
            f'shopt -s nullglob nocaseglob; '
            f'for f in "{process_input}"/*.jpg "{process_input}"/*.jpeg '
            f'"{process_input}"/*.png "{process_input}"/*.webp '
            f'"{process_input}"/*.bmp "{process_input}"/*.tif "{process_input}"/*.tiff; '
            f'do cp -n "$f" "{img_dir}/"; done'
        )

    script = (
        f'set -euo pipefail; '
        # Clear stale run output + the destination dataset so a partial dataset
        # from an aborted prior run can't satisfy the downstream check.
        f'rm -rf "{run_out}" "{processed_dir}"; '
        f'mkdir -p "{img_dir}" "{run_out}"; '
        f'{ingest}; '
        # ViT-Large dense matching -> in-memory SparseGA -> poses.npz/points3D.npz
        # (+ dense3D.npz when --dense, for the InstantSplat sparse-view seed).
        f'"{availability["mast3r_python"]}" "{availability["mast3r_runner"]}" '
        f'--images "{img_dir}" --out "{run_out}" '
        f'--ckpt "{availability["mast3r_checkpoint"]}" '
        f'--scene-graph {scene_graph}{" --dense" if dense else ""}'
        f'{f" --max-images {max_images}" if max_images else ""}; '
        # Reproduce nerfstudio's colmap_to_json convention -> processed_dir
        # (dense seed -> ply_file_path=dense_pc.ply when --seed dense).
        f'"{availability["mast3r_python"]}" "{availability["mast3r_converter"]}" '
        f'--mast3r-out "{run_out}" --src-images "{img_dir}" '
        f'--out "{processed_dir}" --image-mode resized{" --seed dense" if dense else ""}; '
        # Assert the converter produced a usable Nerfstudio dataset so the stage
        # fails loud (rather than letting the A1 gate / train choke downstream).
        f'test -f "{processed_dir}/transforms.json"'
    )
    return ["bash", "-c", script]


def _rig_sfm_command(
    *,
    colmap4: str,
    ffmpeg: str,
    job_dir: Path,
    processed_dir: Path,
    process_input: Path,
    num_frames_target: int,
) -> list[str]:
    """Rig-constrained 360 SfM (equirect video only): extract equirect frames,
    render them into a rig of virtual perspective views (render_rig.py: 12 views,
    ownership masks, rig_config.json), then colmap4 with the rig applied —
    feature_extractor (per-folder pinhole) -> rig_configurator -> rig-verified
    sequential matching -> global_mapper with rig + intrinsics held fixed.
    Spike receipt (probe-operator-mask/STATUS.md): same capture, unrigged lane
    same-frame pose scatter 5.1 units -> rig lane 1080/1080 registered and the
    first recognizable insv reconstruction in the program."""
    rig_dir = job_dir / "rig"
    frames_dir = rig_dir / "equirect_frames"
    db = rig_dir / "database.db"
    sparse = rig_dir / "sparse"
    script = (
        f'set -euo pipefail; '
        f'rm -rf "{rig_dir}" "{processed_dir}"; '
        f'mkdir -p "{frames_dir}" "{sparse}"; '
        # Evenly-spaced frame sampling — same STRIDE recipe as the glomap lane.
        f'FFPROBE="$(dirname "{ffmpeg}")/ffprobe"; [ -x "$FFPROBE" ] || FFPROBE=ffprobe; '
        f'TOTAL="$("$FFPROBE" -v error -count_frames -select_streams v:0 '
        f'-show_entries stream=nb_read_frames -of csv=p=0 "{process_input}" 2>/dev/null)"; '
        f'case "$TOTAL" in ""|*[!0-9]*) TOTAL=0;; esac; '
        f'if [ "$TOTAL" -gt {num_frames_target} ]; then STRIDE=$(( TOTAL / {num_frames_target} )); else STRIDE=1; fi; '
        f'"{ffmpeg}" -y -i "{process_input}" -vf "select=not(mod(n\\,$STRIDE))" -vsync vfr -q:v 2 '
        f'"{frames_dir}/frame_%05d.jpg"; '
        f'"{COLMAP4_ENV_PYTHON}" "{RIG_RENDER_SCRIPT}" "{frames_dir}" "{rig_dir}"; '
        f'"{colmap4}" feature_extractor --database_path "{db}" --image_path "{rig_dir}/images" '
        f'--ImageReader.single_camera_per_folder 1 --ImageReader.camera_model SIMPLE_PINHOLE '
        f'--ImageReader.mask_path "{rig_dir}/masks" --FeatureExtraction.use_gpu 1; '
        f'"{colmap4}" rig_configurator --database_path "{db}" '
        f'--rig_config_path "{rig_dir}/rig_config.json"; '
        f'"{colmap4}" sequential_matcher --database_path "{db}" '
        f'--SequentialMatching.loop_detection 1 --FeatureMatching.rig_verification 1 '
        f'--FeatureMatching.skip_image_pairs_in_same_frame 1 --FeatureMatching.use_gpu 1; '
        f'"{colmap4}" global_mapper --database_path "{db}" --image_path "{rig_dir}/images" '
        f'--output_path "{sparse}" --GlobalMapper.refine_sensor_from_rig 0 '
        f'--GlobalMapper.ba_refine_focal_length 0 --GlobalMapper.ba_refine_extra_params 0; '
        f'test -f "{sparse}/0/cameras.bin" || {{ echo "[rig] global mapper produced no model — '
        f'the capture may need the glomap or Studio-stitch path" >&2; exit 1; }}; '
        f'PTS=$(stat -c%s "{sparse}/0/points3D.bin" 2>/dev/null || echo 0); '
        f'if [ "$PTS" -le 8 ]; then echo "[rig] global mapper posed cameras but triangulated '
        f'0 points" >&2; exit 1; fi'
    )
    return ["bash", "-c", script]


def _sfm_stage_commands(
    solver: str,
    req: SplatTrainRequest,
    availability: dict,
    job_dir: Path,
    processed_dir: Path,
    process_input: Path,
    subcommand: str,
    is_equirect: bool,
) -> dict[str, list[str]]:
    """Build the SfM pre-stage(s) + `process` command(s) for ONE solver.

    Returns an ordered {stage_name: command} dict — e.g. for "colmap" just
    {"process": [...]}, for "glomap" {"glomap_sfm": [...], "process": [...]}.
    This is the single source of truth for SfM command construction so the
    AUTO-FALLBACK gate can rebuild the stages for a different solver mid-run,
    exactly as the planner builds them up front. To add a rung (MASt3R), add an
    `elif solver == "mast3r":` branch here that emits its own pre-stage(s) + a
    `process` that consumes the model — no other call site changes.
    """
    if solver == "rig":
        # Rig-constrained 360 SfM. The `rig_sfm` stage does everything up to a
        # posed sparse model in <job_dir>/rig/sparse/0; `process` hands it to
        # ns-process-data via --skip-colmap (nested pano_camera*/ names proven
        # to convert cleanly in the arm-R spike run).
        rig_sfm = _rig_sfm_command(
            colmap4=availability["glomap_path"],
            ffmpeg=availability["ffmpeg_path"],
            job_dir=job_dir,
            processed_dir=processed_dir,
            process_input=process_input,
            num_frames_target=req.num_frames_target,
        )
        process_cmd = [
            availability["ns_process_data_path"],
            "images",
            "--data",
            str(job_dir / "rig" / "images"),
            "--output-dir",
            str(processed_dir),
            "--skip-colmap",
            "--colmap-model-path",
            os.path.relpath(job_dir / "rig" / "sparse" / "0", processed_dir),
            "--num-downscales",
            "3",
        ]
        return {"rig_sfm": rig_sfm, "process": process_cmd}

    if solver == "glomap":
        # COLMAP 4.x global SfM: we drive feature_extractor + sequential
        # matching + global_mapper ourselves into <job_dir>/colmap/sparse/0,
        # then ns-process-data reads that model via --skip-colmap. For an
        # equirect job the stage additionally reproduces nerfstudio's fan-out
        # (each frame -> images_per_equirect perspective crops) BEFORE COLMAP,
        # so global SfM solves the same crops the incremental path would; the
        # downstream `process` then consumes those crops as plain images
        # (nerfstudio itself flips camera_type to "perspective" post-fan-out).
        glomap_sfm = _glomap_sfm_command(
            colmap4=availability["glomap_path"],
            ffmpeg=availability["ffmpeg_path"],
            job_dir=job_dir,
            processed_dir=processed_dir,
            process_input=process_input,
            is_video=(subcommand == "video"),
            num_frames_target=req.num_frames_target,
            is_equirect=is_equirect,
            images_per_equirect=req.images_per_equirect,
            crop_bottom=req.crop_bottom,
        )
        process_cmd = [
            availability["ns_process_data_path"],
            "images",
            "--data",
            str(_colmap_image_dir(job_dir)),
            "--output-dir",
            str(processed_dir),
            "--skip-colmap",
            # --colmap-model-path is resolved RELATIVE to --output-dir.
            "--colmap-model-path",
            os.path.relpath(job_dir / "colmap" / "sparse" / "0", processed_dir),
            "--num-downscales",
            "3",
        ]
        return {"glomap_sfm": glomap_sfm, "process": process_cmd}

    if solver == "mast3r":
        # MASt3R-SfM rung. The `mast3r_sfm` stage runs the runner (ViT inference
        # -> poses.npz/points3D.npz) AND the converter, which writes a finished
        # Nerfstudio dataset (transforms.json + images/ + sparse_pc.ply) DIRECTLY
        # into processed_dir — reproducing nerfstudio 1.1.5's colmap_to_json
        # convention (verified to 4.4e-16), with NO ns-process-data call. The
        # `process` stage is therefore a no-op assert: it re-confirms the dataset
        # exists, then the existing A1 registration gate (which counts frames in
        # processed_dir/transforms.json vs processed_dir/images) validates the
        # mast3r result exactly as it does for colmap/glomap. If mast3r ALSO
        # under-registers, the chain is exhausted (no rung after it) and the gate
        # fails with guidance.
        mast3r_sfm = _mast3r_sfm_command(
            availability=availability,
            ffmpeg=availability["ffmpeg_path"],
            job_dir=job_dir,
            processed_dir=processed_dir,
            process_input=process_input,
            is_video=(subcommand == "video"),
            num_frames_target=req.num_frames_target,
        )
        # Cheap, env-independent guard so the registration gate has a `process`
        # stage to run and gate on (the real work already happened in mast3r_sfm).
        process_cmd = ["test", "-f", str(processed_dir / "transforms.json")]
        return {"mast3r_sfm": mast3r_sfm, "process": process_cmd}

    if solver == "mast3r-sparse":
        # InstantSplat "Few Photos (AI poses)" rung: identical to the mast3r rung but
        # with a DENSE pointmap seed (ply_file_path=dense_pc.ply) + the `complete`
        # scene-graph (all image pairs — right for 2-12 photos, unlike swin windows).
        # The converter still writes a standard Nerfstudio dataset, so the `process`
        # assert + the A1 registration gate + the `train` stage are all unchanged.
        # "Few Photos" caps the input at SPARSE_MAX_IMAGES (evenly sampled) so the
        # complete (all-pairs) scene-graph stays O(cap^2) even if a movie or a big DSLR
        # folder is fed in — a many-frame input degrades gracefully instead of hanging.
        mast3r_sfm = _mast3r_sfm_command(
            availability=availability,
            ffmpeg=availability["ffmpeg_path"],
            job_dir=job_dir,
            processed_dir=processed_dir,
            process_input=process_input,
            is_video=(subcommand == "video"),
            num_frames_target=min(req.num_frames_target, SPARSE_MAX_IMAGES),
            dense=True,
            scene_graph="complete",
            max_images=SPARSE_MAX_IMAGES,
        )
        process_cmd = ["test", "-f", str(processed_dir / "transforms.json")]
        return {"mast3r_sfm": mast3r_sfm, "process": process_cmd}

    # solver == "colmap" (default): the validated COLMAP 3.11.1 incremental
    # path nerfstudio drives through ns-process-data — byte-for-byte the
    # original command. Reaching here with anything else is a programming error.
    if solver != "colmap":
        raise ValueError(f"Unknown SfM solver: {solver}")
    process_cmd = [
        availability["ns_process_data_path"],
        subcommand,
        "--data",
        str(process_input),
        "--output-dir",
        str(processed_dir),
    ]
    if subcommand == "video":
        process_cmd.extend(["--num-frames-target", str(req.num_frames_target)])
        # Sequential matching (loop closure) chains temporally-ordered frames
        # far better than the default unordered-collection vocab_tree.
        #
        # Equirect fan-out (decided from the nerfstudio 1.1.5 SOURCE, not the
        # docs): equirect_utils names crops <frame>_<k>.jpg GROUPED PER FRAME,
        # so in COLMAP's lexicographic ingest order same-view temporal
        # neighbors sit exactly images_per_equirect positions apart, and
        # adjacent-yaw crops of one frame (30° overlap at fov 120 / 90° step)
        # are 1-3 apart. nerfstudio exposes NO SequentialMatching.overlap flag
        # (colmap_utils builds "<method>_matcher" bare), so COLMAP's default
        # overlap of 10 applies: sequential covers the pairs that matter for 8
        # crops/frame (8 <= 10) but misses EVERY temporal pair for 14
        # (14 > 10, loop detection off) — so 14 keeps the vocab_tree default.
        if not is_equirect:
            process_cmd.extend(["--matching-method", "sequential"])
        elif req.images_per_equirect <= NS_SEQUENTIAL_DEFAULT_OVERLAP:
            process_cmd.extend(["--matching-method", "sequential"])
    if is_equirect:
        process_cmd.extend(
            [
                "--camera-type",
                "equirectangular",
                "--images-per-equirect",
                str(req.images_per_equirect),
                "--crop-bottom",
                str(req.crop_bottom),
            ]
        )
    return {"process": process_cmd}


def _seed_sfm_tried(sfm_context: dict[str, Any] | None, stages: list[str]) -> set[str]:
    """The initial job.sfm_tried set, honest about the solver a job actually
    started on — used both by the escalation gate's "not yet tried" check and
    by exhaustion error messages ("Auto-fallback tried {tried_label}").

    Escalation-eligible jobs (sfm_context is set) seed with the planner's
    start_solver. Sparse ("Few Photos") jobs are NOT escalation-eligible —
    sfm_context is deliberately None for them — but they still start on a real
    solver (mast3r-sparse), detected the same way _new_meta detects a
    genuinely-engaged sparse plan (stages[:1] == ["mast3r_sfm"]). Leaving this
    empty for sparse jobs made a failed job's error message fall back to a
    hardcoded "colmap" default even though COLMAP never ran.
    """
    if sfm_context:
        return {sfm_context["start_solver"]}
    if stages[:1] == ["mast3r_sfm"]:
        return {"mast3r-sparse"}
    return set()


def _next_sfm_solver(tried: set[str], availability: dict, is_equirect: bool = False) -> str | None:
    """The next solver in SFM_ESCALATION that is AVAILABLE and not yet tried.

    Walks the whole chain from the start (not from a fixed index) so it works
    regardless of which rung the job started on, and so an unavailable rung is
    transparently skipped to the next available one. Returns None when the
    chain is exhausted.

    Equirect jobs only climb to rungs in EQUIRECT_CAPABLE_SOLVERS (glomap yes,
    MASt3R no — a perspective-pairwise matcher is the wrong tool for hundreds
    of reprojected fan-out crops).
    """
    for candidate in SFM_ESCALATION:
        if candidate in tried:
            continue
        if is_equirect and candidate not in EQUIRECT_CAPABLE_SOLVERS:
            continue
        if not is_equirect and candidate in EQUIRECT_ONLY_SOLVERS:
            continue
        avail_key = SFM_SOLVER_AVAILABILITY.get(candidate)
        if avail_key is not None and not availability.get(avail_key):
            continue
        return candidate
    return None


def _plan_3d_job(
    req: SplatTrainRequest, availability: dict, job_dir: Path, input_path: Path
) -> tuple[list[str], dict[str, list[str]], dict[str, Any] | None]:
    """Build the ordered stage list, per-stage commands, and escalation context.

    The third return value (sfm_context) is non-None only for escalation-eligible
    inputs (video / image-folder, incl. equirect — but not a pre-processed dataset);
    it carries the inputs the runner's A1 gate needs to rebuild SfM stages for a
    fallback solver (equirect jobs only climb to the glomap rung).

    Inputs already containing a Nerfstudio dataset (transforms.json) skip the
    ns-process-data stage and train directly on the input.
    """
    # Test Flight trim is a stitch-time feature: only raw .insv inputs have a
    # stitch. Reject every other lane HERE, before any early return (the
    # generative-image branch below would otherwise swallow it silently).
    if (req.trim_start_s is not None or req.trim_duration_s is not None) and (
        req.source_type == "generative-image"
        or input_path.suffix.lower() not in INSV_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="Test Flight trim (trim_start_s/trim_duration_s) only applies to raw .insv inputs.",
        )

    # ── Generative image-to-3D ("Imagine a Splat"): one image -> TripoSplat -> a Z-up
    #    .ply. Skips the ENTIRE capture pipeline (no SfM, no ns-train, no capture tools),
    #    then reuses compress/webopt to produce the viewer/download artifacts. ──
    if req.source_type == "generative-image":
        if not availability.get("triposplat_available"):
            raise HTTPException(status_code=400, detail="TripoSplat toolchain is not available.")
        preview_dir = _preview_dir_path(job_dir)
        gen_cmd = ["bash", str(availability["triposplat_runner"]), str(input_path), str(preview_dir)]
        stages = ["generate"]
        commands = {"generate": gen_cmd}
        if _splat_transform_path():
            stages += ["compress", "webopt"]  # .spz + decimated web.ply from splat.ply
        return stages, commands, None

    missing = [
        name
        for name, key in (
            ("ns-train", "ns_train_available"),
            ("ns-process-data", "ns_process_data_available"),
            ("ns-export", "ns_export_available"),
            ("colmap", "colmap_available"),
            ("ffmpeg", "ffmpeg_available"),
        )
        if not availability[key]
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing tools for the 3D pipeline: {', '.join(missing)}. Expected in {CONDA_ENV_BIN} / {COLMAP_ENV_BIN}.",
        )

    stages: list[str] = []
    commands: dict[str, list[str]] = {}
    processed_dir = job_dir / "processed"
    # Escalation context, populated only for escalation-eligible inputs below.
    # A pre-processed dataset (transforms.json) leaves it None, so the runner's
    # gate never reroutes those. Equirect jobs get a context too; the gate
    # limits them to the equirect-capable rungs (glomap, never MASt3R).
    sfm_context: dict[str, Any] | None = None

    # Raw Insta360 .insv: stitch dual-fisheye -> equirectangular MP4 first, then
    # treat the result as a 360 video. Forces equirectangular semantics
    # regardless of the requested capture_format.
    is_insv = input_path.suffix.lower() in INSV_EXTENSIONS
    if is_insv:
        if not availability["insv_stitch_available"]:
            raise HTTPException(
                status_code=400,
                detail="ffmpeg with the v360 filter is required to stitch .insv. Export an equirectangular MP4 from the Insta360 app instead.",
            )
        # Stream-count-aware stitch planning: X4/X5 .insv carry TWO square HEVC
        # fisheye streams; the legacy single-input v360 command reads only
        # stream 0 and produces a structurally corrupt equirect
        # (splat_ec1b984ffb: 2/624 registered). Probe the layout NOW and pick
        # the right command. A POSITIVE detection of an unsupported layout
        # (3+ streams, mismatched dims) refuses loudly before any GPU/COLMAP
        # time is spent — but a failure of the PROBE ITSELF (ffprobe missing/
        # timeout -> streams:0) proves nothing about the file, so it falls
        # back to the legacy single-stream stitch (the pre-probe behavior)
        # rather than 400-ing a single-stream-capable capture; the post-stitch
        # sanity gate still catches a mis-composed result downstream.
        ffprobe = _tool_path("ffprobe", "SPLAT_FFPROBE_BIN")
        probe_info = (
            _probe_video_streams(ffprobe, input_path)
            if ffprobe
            else {"streams": 0, "width": None, "height": None, "dims": []}
        )
        layout, layout_err = _stitch_layout(probe_info)
        if layout == "error":
            if not probe_info.get("streams"):
                # Probe error (not a layout verdict): cannot determine the
                # layout at all. Warn loudly and keep the legacy behavior.
                log.warning(
                    "[stitch] .insv layout probe failed for %s (%s) — falling back to the "
                    "legacy single-stream v360 stitch; the post-stitch sanity gate will "
                    "catch a mis-composed equirect.",
                    input_path,
                    "ffprobe not found" if not ffprobe else "no probeable video stream",
                )
                layout = "single"
            else:
                # Positive detection of a layout the auto-stitch cannot compose.
                raise HTTPException(status_code=400, detail=layout_err)
        # Test Flight trim window. trim_duration_s alone -> center the window
        # (skips the start/stop fumbling at the capture ends). If the probe
        # can't get a duration, fall back to start=0 rather than failing a
        # job the stitch itself can still run; if the clip is shorter than
        # the window, trim is a no-op (drop it entirely).
        trim_start = req.trim_start_s
        trim_duration = req.trim_duration_s
        full_duration = _probe_video_duration(ffprobe, input_path) if ffprobe else None
        if trim_start is not None or trim_duration is not None:
            duration = full_duration
            if duration is not None and trim_start is not None and trim_start >= duration:
                raise HTTPException(
                    status_code=400,
                    detail=f"trim_start_s ({trim_start:g}s) is beyond the end of the clip ({duration:.1f}s).",
                )
            if trim_duration is not None:
                if duration is not None and duration <= trim_duration:
                    trim_start = None
                    trim_duration = None  # window >= clip: trim is a no-op
                elif trim_start is None:
                    trim_start = max(0.0, ((duration or 0.0) - trim_duration) / 2)

        # Frame-density rule (the pipeline's ONLY proven-good 360 config —
        # empirically confirmed both directions on the same window:
        # splat_9da9dff4b2 @1.76fps -> 599 cameras posed, ZERO 3D points;
        # splat_5177f8d99a @3.0fps, same clip -> 1078/1080 registered, 105k
        # points). Test Flight already sends a correct 3fps-derived value
        # client-side, but a full (non-flight) insv run has no way to know
        # the clip's actual duration, so the UI can only ever hardcode a
        # flat guess — wrong for anything but one specific clip length.
        # Compute it here from the REAL (post-trim) window instead of
        # trusting the client, capped so num_frames_target * images_per_equirect
        # never exceeds the crop-count guard the /train endpoint enforces.
        # A trimmed request lands on the exact same number the client already
        # computed (same rule, same window), so this is a no-op override for
        # Test Flight and a real fix for full runs. Probe failure (no ffprobe,
        # unreadable container) leaves num_frames_target as the client sent it
        # rather than guessing — mirrors the layout-probe's own fail-open policy.
        density_window_s = trim_duration if trim_duration is not None else full_duration
        if density_window_s is not None and density_window_s > 0:
            cap = 4000 // req.images_per_equirect
            req.num_frames_target = min(math.ceil(3.0 * density_window_s), cap)

        stitched = _stitched_path(job_dir)
        commands["stitch"] = _stitch_command(
            availability["ffmpeg_path"], input_path, stitched, req.insv_fov,
            dual_stream=(layout == "dual"),
            trim_start=trim_start, trim_duration=trim_duration,
        )
        stages.append("stitch")
        process_input = stitched
        is_video = True
        is_equirect = True
    else:
        process_input = input_path
        is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
        is_equirect = req.capture_format == "equirectangular360"

    sparse_mode = False  # set True below when the InstantSplat few-photos rung engages
    if not is_insv and input_path.is_dir() and (input_path / "transforms.json").is_file():
        train_data = input_path
    else:
        if is_insv or is_video:
            subcommand = "video"
        elif input_path.is_dir():
            subcommand = "images"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported input: {input_path}. Provide an image directory, a video file, a raw Insta360 .insv, or a processed Nerfstudio dataset.",
            )

        # SfM solver selection + AUTO-FALLBACK eligibility. The escalation chain
        # engages for video / image-folder inputs, INCLUDING equirectangular —
        # the gate itself restricts equirect jobs to the equirect-capable rungs
        # (glomap; never MASt3R) via sfm_context["is_equirect"]. A pre-processed
        # dataset (handled above) never reroutes.
        escalation_eligible = subcommand in ("video", "images")

        # Resolve the starting solver. The requested sfm_backend is honored when
        # available; an opt-in glomap that isn't actually present silently falls
        # back to colmap (byte-for-byte the original behavior). MASt3R etc. slot
        # in here automatically via SFM_SOLVER_AVAILABILITY.
        start_solver = req.sfm_backend
        avail_key = SFM_SOLVER_AVAILABILITY.get(start_solver)
        if avail_key is not None and not availability.get(avail_key):
            start_solver = "colmap"
        # Rig SfM renders virtual views from the stitched SPHERE — meaningless for
        # flat captures, and v1 only implements the video input path. A rig request
        # outside that falls back silently (same convention as unavailable-glomap).
        if start_solver == "rig" and (not is_equirect or subcommand != "video"):
            start_solver = "colmap"
        # DEFAULT-FLIP (2026-07-11, RToony's go): equirect VIDEO captures start on
        # the rig lane whenever its toolchain is present — the unrigged fan-out
        # fog-cocooned every 360 capture ever run (probe-operator-mask/STATUS.md;
        # live receipts splat_3885b68e54). The A1 gate still escalates through the
        # legacy rungs if rig under-registers, so the old behavior remains the
        # fallback, not the default.
        if (start_solver == "colmap" and is_equirect and subcommand == "video"
                and availability.get("rig_available")):
            start_solver = "rig"

        # InstantSplat "Few Photos (AI poses)": override to the dense-seed MASt3R rung.
        # A handful of photos won't COLMAP, so there is NO auto-fallback (escalation off);
        # if MASt3R isn't available the request degrades to the standard path silently.
        sparse_mode = (
            req.capture_mode == "sparse"
            and subcommand in ("video", "images")
            and not is_equirect
            and bool(availability.get("mast3r_available"))
        )
        if sparse_mode:
            start_solver = "mast3r-sparse"
            escalation_eligible = False

        sfm_cmds = _sfm_stage_commands(
            solver=start_solver,
            req=req,
            availability=availability,
            job_dir=job_dir,
            processed_dir=processed_dir,
            process_input=process_input,
            subcommand=subcommand,
            is_equirect=is_equirect,
        )
        for stage_name, cmd in sfm_cmds.items():
            stages.append(stage_name)
            commands[stage_name] = cmd
        train_data = processed_dir

        if escalation_eligible:
            # Stash the plan-time inputs the runner's gate needs to rebuild SfM
            # stage commands for a different solver mid-run. Paths are stringified
            # so the context is plain JSON-friendly data on the live job handle.
            sfm_context = {
                "start_solver": start_solver,
                "processed_dir": str(processed_dir),
                "process_input": str(process_input),
                "subcommand": subcommand,
                "is_equirect": is_equirect,
            }
        else:
            sfm_context = None

    # train_data may point at the input dataset (skip-process branch) or
    # processed_dir; in the escalation case the gate re-runs `process` into the
    # same processed_dir, so train_data stays valid across a reroute.
    # Sparse-view (few photos) trains at LOW iterations — the geometry is already
    # seeded by the dense MASt3R cloud and few images overfit fast; 30k just bakes in
    # artifacts. Cap the default; an explicit lower request is still honored.
    train_iters = req.max_num_iterations
    if sparse_mode and train_iters > 8000:
        train_iters = 7000
    commands["train"] = [
        availability["ns_train_path"],
        "splatfacto",
        "--data",
        str(train_data),
        "--output-dir",
        str(job_dir),
        "--max-num-iterations",
        str(train_iters),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    stages.extend(["train", "export"])
    # Capture-health fog gate: REPORT-ONLY verdict right after export, before the
    # optional tail (compress/webopt/langfield) — so the verdict exists even when
    # those are skipped, and a later opt-in enforcement flip can skip them on FOG.
    _append_health_stage(stages)
    # The export command needs the config.yml path that only exists after
    # training; the runner builds it when the stage starts. The compress
    # stage (best-effort .spz for fast preview) is appended only when the
    # tool is present, and its failure never fails the job.
    if _splat_transform_path():
        stages.append("compress")
        # webopt: lightweight .ply for the shareable web viewer. Best-effort,
        # runs after compress, failure never fails the job.
        stages.append("webopt")
    # OPT-IN language field, appended OUTSIDE the splat-transform guard (it needs
    # only the nerfstudio checkpoint, not compress/webopt). Runs DEAD LAST; the
    # runner builds its command from the job dir at stage time (like export).
    if req.language_field and _langfield_available():
        stages.append("langfield")
    return stages, commands, sfm_context


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


async def _consume_stream(job: SplatJob, stream: asyncio.StreamReader | None, label: str) -> None:
    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except (ValueError, asyncio.LimitOverrunError):
            # Over-long line (rich progress output); skip it rather than dying.
            continue
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            job.log_lines.append(f"[{label}] {text}")


async def _run_stage(job: SplatJob, stage: str, command: list[str]) -> int:
    _patch_meta(job.job_id, stage=stage, command=command)
    job.log_lines.append(f"[{stage}] $ {' '.join(command)}")
    _flush_log(job)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(SPLAT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_subprocess_env(),
        start_new_session=True,
    )
    job.process = process
    job.pid = process.pid

    stdout_task = asyncio.create_task(_consume_stream(job, process.stdout, stage))
    stderr_task = asyncio.create_task(_consume_stream(job, process.stderr, f"{stage} stderr"))
    return_code = await process.wait()
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    job.process = None
    _patch_meta(job.job_id, exit_code=return_code)
    _flush_log(job)
    return return_code


async def _run_locked_stage(job: SplatJob, stage: str, command: list[str], vram_mb: int) -> int:
    """Run a heavy-GPU stage under the cross-route lock (gpu_arbiter.HEAVY_GPU_LOCK).

    Used by `train` and the MASt3R-SfM rung. MASt3R loads a 2.6 GB ViT-Large and
    does dense matching on the GPU (~3.46 GB peak measured), so — unlike the light
    COLMAP/GLOMAP SfM stages — it MUST serialise against the portal's TRELLIS lane
    on the shared 5090, or a concurrent TRELLIS run can OOM (or be OOM'd by) it.
    """
    if gpu_arbiter.HEAVY_GPU_LOCK.locked():
        holder = gpu_arbiter.holder_info()
        job.log_lines.append(
            f"[{stage}] Waiting for GPU — currently held by {holder.get('lane')} job {holder.get('job_id')}."
        )
        _flush_log(job)
    async with gpu_arbiter.HEAVY_GPU_LOCK:
        if job.stop_requested:
            return -1
        gpu_arbiter.set_holder("splat", job.job_id)
        try:
            ok, msg = await gpu_arbiter.acquire_gpu(vram_mb)
            job.log_lines.append(f"[{stage}] GPU arbiter: {msg}")
            if not ok:
                _patch_meta(job.job_id, error_message=f"GPU acquire failed: {msg}")
                return -1
            return await _run_stage(job, stage, command)
        finally:
            gpu_arbiter.clear_holder()


async def _run_train_stage(job: SplatJob, command: list[str]) -> int:
    """Run the train stage under the cross-route heavy-GPU lock."""
    return await _run_locked_stage(job, "train", command, TRAIN_VRAM_MB)


def _langfield_clean_text(text: str) -> str:
    """Sanitize a query into a SigLIP-safe + filename-safe string (word chars, spaces,
    hyphens; collapsed; capped). The renderer writes q_<clean.replace(' ','_')>.png, so
    the endpoint computes the heatmap URL deterministically."""
    t = re.sub(r"[^\w\s-]+", " ", text).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:80].strip() or "query"


def _langfield_heatmap_name(clean_text: str) -> str:
    return f"q_{clean_text.replace(' ', '_')}.png"


async def _langfield_worker_query(config_path: str, lfdir: str, clean_text: str) -> dict | None:
    """Try the warm worker (resident SigLIP + cached scene = sub-second). Returns the
    worker's JSON (incl. the 3D match `focus`/`radius`) on success, else None so the
    caller falls back to the cold path."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=1.5)) as client:
            resp = await client.post(
                f"{LANGFIELD_WORKER_URL}/query",
                json={"config": config_path, "lfdir": lfdir, "text": clean_text},
            )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# The worker's /relevancy response headers the proxy passes through to the client:
# X-Count/X-Min/X-Max = dequantization params for the uint8 body, X-Matches = the
# clustered 3D instances JSON (same shape as /query's, minus the rendered thumbs).
RELEVANCY_FORWARD_HEADERS = ("x-count", "x-min", "x-max", "x-matches", "x-label-hit")


async def _langfield_worker_relevancy(config_path: str, lfdir: str, clean_text: str):
    """Ask the warm worker for the raw per-gaussian relevancy vector (uint8-quantized
    binary body + X-* headers). Returns the httpx.Response on HTTP 200, else None.
    WARM-WORKER ONLY — the caller turns None into a 503 (no cold-subprocess fallback:
    this is an interactive client path and the cold path renders nothing useful for it).
    The generous read timeout covers a first-touch scene build waiting behind a
    training run on HEAVY_GPU_LOCK; the compute itself is lockless and sub-second."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=1.5)) as client:
            resp = await client.post(
                f"{LANGFIELD_WORKER_URL}/relevancy",
                json={"config": config_path, "lfdir": lfdir, "text": clean_text},
            )
        return resp if resp.status_code == 200 else None
    except Exception:
        return None


async def _langfield_worker_json(path: str, payload: dict):
    """POST a JSON request to the warm worker; returns the httpx.Response or
    None on transport failure (caller maps None -> 503). Worker-side HTTPErrors
    (4xx) pass through so guardrail messages reach the client verbatim."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=1.5)) as client:
            return await client.post(f"{LANGFIELD_WORKER_URL}{path}", json=payload)
    except Exception:
        return None


async def _langfield_worker_inventory(config_path: str, lfdir: str) -> dict | None:
    """Ask the warm worker for the scene's top-N object inventory (cached per scene).
    Returns the worker JSON on success, else None (no cold fallback — inventory is a
    warm-worker-only convenience)."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=1.5)) as client:
            resp = await client.post(
                f"{LANGFIELD_WORKER_URL}/inventory",
                json={"config": config_path, "lfdir": lfdir},
            )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


async def ensure_hero_thumb(output_dir: Path) -> Path | None:
    """A REAL rendered thumbnail for scenes with a language field: ask the warm worker
    to render one clean hero view (cached at <lfdir>/hero.webp). Returns the path, or
    None (no field / worker down / render failed) so the caller falls back to the cheap
    CPU point-cloud thumbnail."""
    lfdir = output_dir / LANGFIELD_DIRNAME
    if not (lfdir / "gauss_emb.npz").is_file():
        return None
    hero = lfdir / "hero.webp"
    if hero.is_file():
        return hero
    config_path = _find_latest_config(output_dir)
    if config_path is None:
        return None
    import httpx
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=1.5)) as client:
            resp = await client.post(
                f"{LANGFIELD_WORKER_URL}/hero",
                json={"config": str(config_path), "lfdir": str(lfdir)},
            )
        if resp.status_code == 200 and hero.is_file():
            return hero
    except Exception:
        pass
    return None


async def _langfield_query_cold(scene_id: str, config_path: str, lfdir: str, clean_text: str) -> bool:
    """Fallback when the worker is down: cold subprocess render under HEAVY_GPU_LOCK
    (loads SigLIP + the pipeline, ~20-40s) so it serialises with train/TRELLIS and can
    never OOM them. The lock releases the moment the render finishes."""
    command = ["bash", str(LANGFIELD_QUERY_RUNNER), config_path, lfdir, clean_text]
    async with gpu_arbiter.HEAVY_GPU_LOCK:
        gpu_arbiter.set_holder("langfield", scene_id)
        try:
            ok, _msg = await gpu_arbiter.acquire_gpu(QUERY_VRAM_MB)
            if not ok:
                return False
            proc = await asyncio.create_subprocess_exec(
                *command, cwd=str(SPLAT_ROOT),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=_subprocess_env(), start_new_session=True,
            )
            await proc.communicate()
            return proc.returncode == 0
        finally:
            gpu_arbiter.clear_holder()


def _maybe_escalate_sfm(
    job: SplatJob, process_index: int, registered: int, extracted: int, pct: str
) -> bool:
    """AUTO-FALLBACK: reroute a low-registration job to the next solver, in place.

    Returns True iff it set up a reroute (caller should `continue` so the loop
    re-reads job.stages_planned). Returns False when the job is not eligible or
    the solver chain is exhausted (caller should fail with guidance).

    Infinite-loop safety, in order:
      - only escalation-eligible jobs reroute (sfm_context is set => video /
        image-folder, incl. equirect; a pre-processed dataset leaves it None);
      - equirect jobs only climb to EQUIRECT_CAPABLE_SOLVERS (glomap, not mast3r);
      - reroute_count is capped at len(SFM_ESCALATION);
      - the chosen solver is added to job.sfm_tried BEFORE rerouting, and
        _next_sfm_solver skips anything already tried — so no solver runs twice.
    """
    ctx = job.sfm_context
    if ctx is None:
        # Pre-processed dataset: not eligible. Caller fails as before.
        return False
    if job.reroute_count >= len(SFM_ESCALATION):
        # Hard cap: even if availability lies, never reroute more than there are
        # rungs in the chain.
        return False

    availability = _engine_availability()
    next_solver = _next_sfm_solver(job.sfm_tried, availability, is_equirect=bool(ctx.get("is_equirect")))
    if next_solver is None:
        # Chain exhausted: every available solver after the current one is tried.
        return False

    job.log_lines.append(
        f"[process] only {pct} registered ({registered}/{extracted}) — "
        f"auto-retrying with {next_solver}…"
    )

    # Mark tried + count the reroute BEFORE mutating state, so any later failure
    # can never re-pick this solver.
    job.sfm_tried.add(next_solver)
    job.reroute_count += 1

    # Rebuild the SfM stage(s) + a fresh `process` for the new solver, using the
    # same plan-time inputs (paths rehydrated from the stashed context).
    if job.sfm_req is None:
        # Defensive: sfm_context implies sfm_req was set at plan time. If somehow
        # not, fail safe (no reroute) rather than crash mid-pipeline.
        return False
    new_cmds = _sfm_stage_commands(
        solver=next_solver,
        req=job.sfm_req,
        availability=availability,
        job_dir=Path(job.output_dir),
        processed_dir=Path(ctx["processed_dir"]),
        process_input=Path(ctx["process_input"]),
        subcommand=ctx["subcommand"],
        is_equirect=ctx["is_equirect"],
    )

    # Rename the new solver's `process` to a unique "reprocess<n>" so
    # stages_planned never holds two identical "process" entries — duplicate names
    # collide React keys on the stage rail AND green BOTH process bars when the
    # reroute's process completes. The gate dispatches any "reprocess*" name
    # exactly like "process" (same A1 registration check).
    reprocess_name = f"reprocess{job.reroute_count}"
    new_cmds = {
        (reprocess_name if name == "process" else name): cmd
        for name, cmd in new_cmds.items()
    }

    # Register the new commands (the SfM pre-stage + the uniquely-named reprocess;
    # the stale already-run process/glomap_sfm entries stay behind us in the loop).
    job.stage_commands.update(new_cmds)

    # Inject the new solver's stages immediately AFTER the current process stage
    # so they run next, ahead of the unchanged train/export/compress/webopt tail.
    new_stage_names = list(new_cmds.keys())
    insert_at = process_index + 1
    job.stages_planned[insert_at:insert_at] = new_stage_names
    _patch_meta(job.job_id, stages_planned=job.stages_planned)
    return True


async def _run_pipeline(job: SplatJob) -> None:
    meta = _patch_meta(job.job_id, status="running", started_at=_utc_now()) or {}
    job_dir = Path(job.output_dir)
    final_status = "completed"
    error_message: str | None = None

    try:
        # Index-aware iteration: the AUTO-FALLBACK gate may insert a fallback
        # solver's SfM stage(s) + a fresh `process` into job.stages_planned
        # *immediately after the current process stage* and then `continue`.
        # enumerate over the live list reflects in-place insertions, so the very
        # next loop step picks up the injected stages ahead of the unchanged
        # train/export/... tail. stage_index gives the gate the insert anchor.
        for stage_index, stage in enumerate(job.stages_planned):
            if job.stop_requested:
                break

            if stage == "stitch":
                # The stream layout was probed and classified AT PLAN TIME
                # (_stitch_layout): single-stream -> legacy -vf v360, dual
                # square streams -> hstack both lenses first, anything else
                # was rejected before the job existed. Log the layout here
                # for the receipt trail, then run the pre-built command.
                stitched = _stitched_path(job_dir)
                stitched.parent.mkdir(parents=True, exist_ok=True)
                ffprobe = _tool_path("ffprobe", "SPLAT_FFPROBE_BIN")
                if ffprobe:
                    info = _probe_video_streams(ffprobe, Path(job.input_path))
                    dual = "-filter_complex" in job.stage_commands["stitch"]
                    job.log_lines.append(
                        f"[stitch] source: {info['streams']} video stream(s), "
                        f"{info['width']}x{info['height']} — "
                        f"{'dual-lens hstack compose' if dual else 'single-stream dfisheye'}"
                    )
                return_code = await _run_stage(job, stage, job.stage_commands["stitch"])
                if return_code == 0 and not job.stop_requested:
                    # R0 sanity gate: validate the equirect BEFORE COLMAP burns
                    # ~10 min on a structurally corrupt pano. Failure here is a
                    # STITCH problem and says so (never the misleading capture-
                    # technique guidance). Skippable via SPLAT_STITCH_SANITY=0.
                    # Off the event loop: the gate runs sync subprocess.run
                    # (ffprobe + 2 frame extractions, ~280s worst case) plus
                    # pure-python pixel analysis — thread-safe (no shared
                    # mutable state beyond job.log_lines appends, which are
                    # atomic under the GIL), so to_thread keeps the API
                    # responsive instead of blocking every request.
                    sanity_error = await asyncio.to_thread(_stitch_sanity_check, job, stitched)
                    if sanity_error:
                        final_status = "failed"
                        error_message = sanity_error
                        job.log_lines.append(error_message)
                        break
            elif stage == "export":
                config_path = _find_latest_config(job_dir)
                if config_path is None:
                    final_status = "failed"
                    error_message = f"Training finished but no config.yml was found under {job_dir}."
                    break
                availability = _engine_availability()
                command = _export_command(availability["ns_export_path"], config_path, job_dir)
                return_code = await _run_stage(job, stage, command)
            elif stage == "train":
                return_code = await _run_train_stage(job, job.stage_commands["train"])
            elif stage == "generate":
                # TripoSplat generative lane — heavy GPU (14B-ish weights + working set);
                # take the shared lock so it serialises against train/MASt3R/TRELLIS.
                return_code = await _run_locked_stage(
                    job, stage, job.stage_commands[stage], TRIPOSPLAT_VRAM_MB
                )
            elif stage == "mast3r_sfm":
                # MASt3R-SfM ViT inference is heavy GPU work — run it under the
                # shared lock so it serialises against the portal's TRELLIS lane.
                # (The light COLMAP/GLOMAP SfM stages stay lockless via `else`.)
                return_code = await _run_locked_stage(
                    job, stage, job.stage_commands[stage], MAST3R_VRAM_MB
                )
            elif stage == "compress":
                # Best-effort: compress the exported .ply into a viewer-native
                # .spz. A missing tool or non-zero exit is logged and skipped —
                # the raw .ply preview already works.
                transform = _splat_transform_path()
                ply = _preview_file_path(job_dir)
                if transform and ply.is_file():
                    command = [transform, str(ply), str(_preview_spz_path(job_dir))]
                    rc = await _run_stage(job, stage, command)
                    if rc != 0:
                        job.log_lines.append("[compress] .spz compression failed; keeping raw .ply preview.")
                        _record_stage_failure(job.job_id, stage, f"exit code {rc}")
                else:
                    job.log_lines.append("[compress] skipped (tool or .ply unavailable).")
                completed = (_read_meta(job.job_id) or {}).get("stages_completed", [])
                _patch_meta(job.job_id, stages_completed=[*completed, stage])
                continue
            elif stage == "webopt":
                # Best-effort: a lightweight web-viewer .ply (spherical harmonics
                # dropped + decimated) so the shareable /splat/view page loads
                # ~12x smaller than the raw export. Failure is logged and the
                # viewer falls back to the raw .ply (fmt=web -> .ply server-side).
                transform = _splat_transform_path()
                ply = _preview_file_path(job_dir)
                if transform and ply.is_file():
                    command = [
                        transform,
                        str(ply),
                        "--filter-harmonics",
                        "0",
                        "--decimate",
                        str(WEB_DECIMATE_TARGET),
                        str(_preview_web_path(job_dir)),
                    ]
                    rc = await _run_stage(job, stage, command)
                    if rc != 0:
                        job.log_lines.append("[webopt] web-optimized .ply failed; viewer falls back to the raw .ply.")
                        _record_stage_failure(job.job_id, stage, f"exit code {rc}")
                    # langweb: full-count SH-stripped copy for the CLIENT-SIDE language
                    # heatmap. No --decimate, so its row order matches gauss_emb.npz
                    # (probe-verified — see _langweb_command). Built only for jobs that
                    # will build a language field; best-effort like web.ply (fmt=langweb
                    # falls back to the raw .ply server-side).
                    # Distinct label: reusing "webopt" here would clobber the real
                    # webopt receipt in meta (stage/command/exit_code would show the
                    # langweb run under the webopt name). A label outside
                    # stages_planned is safe: _run_stage only patches meta.stage/
                    # command/exit_code, stages_completed still gets "webopt" below,
                    # and the stage rail simply highlights no bar for the brief
                    # secondary run (same degrade-to-raw-string path as reprocess<n>).
                    if "langfield" in job.stages_planned:
                        rc = await _run_stage(job, "webopt-langweb", _langweb_command(transform, job_dir))
                        if rc != 0:
                            job.log_lines.append(
                                "[webopt] langweb .ply failed; the client heatmap falls back to the raw .ply."
                            )
                            _record_stage_failure(job.job_id, "webopt-langweb", f"exit code {rc}")
                else:
                    job.log_lines.append("[webopt] skipped (tool or .ply unavailable).")
                completed = (_read_meta(job.job_id) or {}).get("stages_completed", [])
                _patch_meta(job.job_id, stages_completed=[*completed, stage])
                continue
            elif stage == "health":
                # Best-effort capture-health fog gate (REPORT-ONLY, Capture Coach).
                # Renders depth at 6 training cameras in the langfield-spike env,
                # writes FOG/HEALTHY/UNCERTAIN + receipt images into _health/, then
                # meta["health"]. Same never-fail contract as langfield: the whole
                # body is wrapped; any failure is bookkeeping, never job state.
                try:
                    config_path = _find_latest_config(job_dir)
                    if config_path is None or not _health_available():
                        job.log_lines.append("[health] skipped (no config or toolchain unavailable).")
                    else:
                        hdir = job_dir / HEALTH_DIRNAME
                        hdir.mkdir(parents=True, exist_ok=True)
                        command = ["bash", str(HEALTH_RUNNER), str(config_path), str(hdir)]
                        rc = await _run_locked_stage(job, stage, command, HEALTH_VRAM_MB)
                        fog_json = hdir / "fog.json"
                        if rc == 0 and fog_json.is_file():
                            fog = json.loads(fog_json.read_text())
                            fog["enforced"] = False  # report-only until the doctrine flip
                            _patch_meta(job.job_id, health={"v": 1, "fog": fog})
                            summ = fog.get("summary", {})
                            job.log_lines.append(
                                f"[health] verdict: {fog.get('verdict')} — "
                                f"{summ.get('n_fog')}/{summ.get('n_counted')} cameras read as fog cocoon "
                                f"(median shell {summ.get('median_shell_frac')}). Report-only."
                            )
                        else:
                            job.log_lines.append("[health] fog gate failed; the splat is unaffected.")
                            _record_stage_failure(job.job_id, stage, f"exit code {rc}")
                except Exception as exc:  # noqa: BLE001 — best-effort: never fail the splat
                    job.log_lines.append(f"[health] skipped (error: {exc}); the splat is unaffected.")
                    _record_stage_failure(job.job_id, stage, f"error: {exc}")
                completed = (_read_meta(job.job_id) or {}).get("stages_completed", [])
                _patch_meta(job.job_id, stages_completed=[*completed, stage])
                continue
            elif stage == "langfield":
                # Best-effort OPT-IN: build the text-searchable language field (SAM 2.1
                # masks + SigLIP 2 region features, training-free lift). Heavy GPU work,
                # so it takes HEAVY_GPU_LOCK ONCE around the whole ~7-min build (the
                # wrapper runs the sam2 + langfield-spike passes back-to-back so TRELLIS
                # can't wedge between them). A missing config or non-zero exit is logged
                # and skipped — it NEVER fails the splat job; the splat is already done.
                # The whole body is wrapped so even an OSError (mkdir / config read /
                # arbiter) is logged, not propagated — provably cannot flip the job to
                # failed (one step beyond the compress/webopt best-effort precedent).
                try:
                    config_path = _find_latest_config(job_dir)
                    if config_path is None or not _langfield_available():
                        job.log_lines.append("[langfield] skipped (no config or toolchain unavailable).")
                    else:
                        lfdir = job_dir / LANGFIELD_DIRNAME
                        lfdir.mkdir(parents=True, exist_ok=True)
                        command = ["bash", str(LANGFIELD_RUNNER), str(config_path), str(lfdir)]
                        rc = await _run_locked_stage(job, stage, command, LANGFIELD_VRAM_MB)
                        if rc != 0:
                            job.log_lines.append(
                                "[langfield] build failed; the splat is unaffected (no language search for this scene)."
                            )
                            # Bookkeeping only — never flips final_status. Without this,
                            # stages_completed below makes a failed optional stage look
                            # identical to a successful one in job meta / the API / the UI.
                            _record_stage_failure(job.job_id, stage, f"exit code {rc}")
                except Exception as exc:  # noqa: BLE001 — best-effort: never fail the splat
                    job.log_lines.append(f"[langfield] skipped (build error: {exc}); the splat is unaffected.")
                    _record_stage_failure(job.job_id, stage, f"error: {exc}")
                completed = (_read_meta(job.job_id) or {}).get("stages_completed", [])
                _patch_meta(job.job_id, stages_completed=[*completed, stage])
                continue
            elif stage == "process" or stage.startswith("reprocess"):
                # Registration-quality gate. The process (COLMAP/SfM) stage
                # extracts frames and solves camera poses; only the registered
                # frames land in transforms.json. A doomed capture registers
                # almost nothing — training it just burns ~10 min of GPU on
                # garbage. So: run process as normal, then check the ratio and
                # FAIL FAST with guidance before the train stage if it's too low.
                # A "reprocess<n>" stage is an auto-fallback solver's process step
                # (uniquely named so it never collides with the original "process"
                # on the stage rail); it runs through the exact same gate.
                return_code = await _run_stage(job, stage, job.stage_commands[stage])
                if job.stop_requested:
                    break
                if return_code != 0:
                    final_status = "failed"
                    if error_message is None:
                        error_message = f"Stage '{stage}' exited with code {return_code}."
                    job.log_lines.append(error_message)
                    break

                # Robustness: any parse/IO problem must NEVER abort a good job —
                # on doubt we log a note and fall through to normal behavior.
                processed_dir = job_dir / "processed"
                registered: int | None = None
                extracted: int | None = None
                try:
                    transforms = processed_dir / "transforms.json"
                    with transforms.open() as fh:
                        registered = len(json.load(fh).get("frames", []))
                    images_dir = processed_dir / "images"
                    extracted = sum(
                        1
                        for p in images_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                    ) if images_dir.is_dir() else 0
                except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
                    # FAIL CLOSED (2026-07-05): a missing/unreadable transforms.json
                    # after a zero-exit process stage means SfM output was unusable
                    # (proven live: glomap posed 599 images with 0 3D points ->
                    # ns-process-data einsum'd, printed the error, exited 0, wrote
                    # nothing -> train crashed on the missing file). Falling
                    # through here sent a doomed job to the GPU; instead treat it
                    # exactly like a 0% registration: escalate solvers if
                    # possible, else fail with the real reason.
                    job.log_lines.append(
                        f"[process] transforms.json missing/unreadable after process ({exc})."
                    )
                    if _maybe_escalate_sfm(job, stage_index, 0, extracted or 0, "0.0%"):
                        continue
                    final_status = "failed"
                    tried_label = ", ".join(sorted(job.sfm_tried)) or "colmap"
                    error_message = (
                        "Camera solving produced no usable output (no transforms.json) — "
                        f"the SfM model was empty or degenerate. Auto-fallback tried {tried_label}. "
                        "If this was a Test Flight, the sampled window may lack camera movement "
                        "(standing still) or enough frames — try a different window or a denser "
                        "sample. Training was skipped to save GPU time."
                    )
                    job.log_lines.append(error_message)
                    _patch_meta(job.job_id, status=final_status, stage=None, error_message=error_message)
                    break

                if registered is not None and extracted:
                    ratio = registered / extracted
                    pct = f"{ratio * 100:.1f}%"
                    job.log_lines.append(
                        f"[process] registration: {registered}/{extracted} frames ({pct})."
                    )
                    if ratio < MIN_REGISTRATION_RATIO:
                        # AUTO-FALLBACK: try to climb the solver chain (zero
                        # clicks) before giving up. _maybe_escalate_sfm rebuilds
                        # the next solver's SfM stage(s), mutates stages_planned
                        # to inject them ahead of the remaining train/export/...,
                        # and returns True iff a reroute was set up. Eligibility,
                        # infinite-loop safety (sfm_tried + reroute cap), and the
                        # equirect/dataset exclusion all live inside it.
                        if _maybe_escalate_sfm(job, stage_index, registered, extracted, pct):
                            # Loop re-reads job.stages_planned from the top, which
                            # now begins with the new solver's pre-stage(s) + a
                            # fresh `process`. The current low-reg run is dropped.
                            continue
                        final_status = "failed"
                        tried_label = ", ".join(sorted(job.sfm_tried)) or "colmap"
                        error_message = (
                            f"Only {registered} of {extracted} frames registered ({pct}). "
                            "The capture likely has low texture, motion blur, or not enough "
                            f"overlap. Auto-fallback tried {tried_label}; recapture with slow, "
                            "heavily-overlapping sweeps of a smaller area. "
                            "Training was skipped to save GPU time."
                        )
                        job.log_lines.append(error_message)
                        _patch_meta(
                            job.job_id,
                            status=final_status,
                            stage=None,
                            error_message=error_message,
                        )
                        break
                elif registered is not None and not extracted:
                    job.log_lines.append(
                        "[process] registration check skipped (no extracted images found)."
                    )

                completed = (_read_meta(job.job_id) or {}).get("stages_completed", [])
                _patch_meta(job.job_id, stages_completed=[*completed, stage])
                continue
            else:
                return_code = await _run_stage(job, stage, job.stage_commands[stage])

            if job.stop_requested:
                break
            if return_code != 0:
                final_status = "failed"
                if error_message is None:
                    error_message = f"Stage '{stage}' exited with code {return_code}."
                job.log_lines.append(error_message)
                break
            completed = (_read_meta(job.job_id) or {}).get("stages_completed", [])
            _patch_meta(job.job_id, stages_completed=[*completed, stage])
        else:
            job.log_lines.append("Pipeline completed. Preview is ready in the viewer panel.")

        if job.stop_requested:
            final_status = "stopped"
            job.log_lines.append("Pipeline stopped by operator request.")
    except Exception as exc:  # noqa: BLE001 — surface anything to the job log
        import traceback
        final_status = "failed"
        error_message = f"Pipeline crashed: {exc}"
        job.log_lines.append(error_message)
        job.log_lines.append("[traceback] " + traceback.format_exc().replace("\n", " | "))
    finally:
        meta = _patch_meta(
            job.job_id,
            status=final_status,
            stage=None,
            completed_at=_utc_now(),
            error_message=error_message,
            stop_requested=job.stop_requested,
        ) or {}
        _flush_log(job)
        try:
            pruned = _prune_old_jobs()
            if pruned:
                job.log_lines.append(f"Pruned {pruned} old unpinned job(s).")
                _flush_log(job)
        except Exception:
            pass

    await audit_operator_event(
        request=None,
        title=f"Splat 3D job {final_status}",
        description=f"{Path(job.input_path).name} -> {job.output_dir}",
        variant="success" if final_status == "completed" else "destructive" if final_status == "failed" else "default",
        action="splat.train",
        target="3d",
        metadata={
            "job_id": job.job_id,
            "status": final_status,
            "stages_completed": meta.get("stages_completed", []),
            # Optional-stage failures (e.g. langfield) — job status stays
            # "completed", but the audit trail shouldn't hide that one failed.
            "stages_failed": meta.get("stages_failed", []),
            "output_dir": job.output_dir,
        },
    )


def _kill_job_process(job: SplatJob, sig: signal.Signals) -> None:
    process = job.process
    if process is None:
        return
    try:
        # start_new_session puts each stage in its own process group, so this
        # also reaches grandchildren (colmap/ffmpeg workers under ns-*).
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(sig)


# ---------------------------------------------------------------------------
# Pruning + startup recovery (pattern from three_d.py)
# ---------------------------------------------------------------------------


def _delete_job_files(job_id: str) -> None:
    if not _safe_job_id(job_id):
        return
    job_dir = _job_dir(job_id)
    if job_dir.is_dir():
        shutil.rmtree(job_dir, ignore_errors=True)


def _iso_to_epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def _prune_old_jobs() -> int:
    """Delete unpinned completed jobs beyond the cap, plus stale failures.

    Splat job dirs are multi-GB (processed images + checkpoints + preview),
    so the unpinned cap is deliberately low. Pin anything worth keeping.
    """
    metas = _all_metas()
    pruned = 0

    cutoff_failed = datetime.now(timezone.utc).timestamp() - FAILED_RETENTION_HOURS * 3600
    for m in metas:
        if m.get("status") in ("failed", "stopped") and not m.get("pinned"):
            ts = _iso_to_epoch(m.get("created_at"))
            if ts and ts < cutoff_failed and m["job_id"] not in JOBS:
                _delete_job_files(m["job_id"])
                pruned += 1

    completed_unpinned = [
        m for m in metas
        if m.get("status") == "completed" and not m.get("pinned") and m["job_id"] not in JOBS
    ]
    completed_unpinned.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    for m in completed_unpinned[KEEP_UNPINNED_COMPLETED:]:
        _delete_job_files(m["job_id"])
        pruned += 1

    return pruned


# Resume-on-start: how recent an orphaned job must be to earn an auto-restart
# (after a long outage the input may be gone / the user has moved on), and how
# many auto-restarts a single job gets before we stop believing in it (guards
# against a job that somehow crash-loops the service).
RESUME_MAX_AGE_HOURS = float(os.environ.get("SPLAT_RESUME_MAX_AGE_H", "12"))
RESUME_MAX_RESTARTS = 2


def _req_from_meta(meta: dict[str, Any]) -> SplatTrainRequest | None:
    """Rebuild the original training request from persisted meta (every request
    knob is persisted by _new_meta precisely so re-runs can be exact)."""
    keys = ("mode", "input_path", "capture_format", "images_per_equirect",
            "crop_bottom", "num_frames_target", "max_num_iterations", "insv_fov",
            "sfm_backend", "language_field", "capture_mode", "source_type",
            "trim_start_s", "trim_duration_s")
    fields = {k: meta[k] for k in keys if meta.get(k) is not None}
    try:
        return SplatTrainRequest(output_dir="outputs/3d", **fields)
    except Exception:  # noqa: BLE001 — legacy/hand-edited meta: not restartable
        return None


def _restart_job(meta: dict[str, Any], req: SplatTrainRequest) -> None:
    """Re-plan and relaunch an orphaned job under its ORIGINAL job_id. Stage
    scripts are self-cleaning (each rm -rfs its own outputs), so restarting from
    the first stage is safe; prior escalation state is deliberately dropped
    (worst case a rung is retried). Must run inside a live event loop."""
    job_id = meta["job_id"]
    availability = _engine_availability()
    input_path = _resolve_input_path(req.input_path)
    job_dir = _job_dir(job_id)
    stages, commands, sfm_context = _plan_3d_job(req, availability, job_dir, input_path)
    job = SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path=str(input_path),
        stages_planned=stages,
        stage_commands=commands,
        sfm_tried=_seed_sfm_tried(sfm_context, stages),
        sfm_context=sfm_context,
        sfm_req=req if sfm_context else None,
    )
    fresh = _new_meta(job_id, req, input_path, job_dir, stages)
    fresh["created_at"] = meta.get("created_at") or fresh["created_at"]
    fresh["pinned"] = bool(meta.get("pinned"))
    fresh["restart_count"] = int(meta.get("restart_count") or 0) + 1
    fresh["restarted_at"] = _utc_now()
    _write_meta(job_id, fresh)
    JOBS[job_id] = job
    job.log_lines.append(
        "Auto-restarted after a service restart (in-flight work does not survive "
        "the restart; stages are self-cleaning). Planned stages: " + " -> ".join(stages)
    )
    job.runner_task = asyncio.create_task(_run_pipeline(job))


async def resume_orphan_jobs() -> int:
    """On startup, AUTO-RESTART the newest orphaned in-flight job instead of
    just declaring it dead (the pre-2026-07-11 behavior, kept as the fallback
    for everything else and behind SPLAT_RESUME_ON_START=0).

    Only the newest orphan restarts — one job runs at a time on the 5090 —
    and only when it is fresh (RESUME_MAX_AGE_HOURS), below the restart cap,
    and its input still exists. Every other orphan gets the honest failed
    marker exactly as before. Any error falls back to mark-failed: resume must
    never be able to wedge startup."""
    if os.environ.get("SPLAT_RESUME_ON_START", "").strip() == "0":
        return 0 if cleanup_orphan_jobs() >= 0 else 0
    orphans = [m for m in _all_metas() if m.get("status") in ("starting", "running")]
    orphans.sort(key=lambda m: m.get("started_at") or m.get("created_at") or "", reverse=True)
    restarted = 0
    for idx, meta in enumerate(orphans):
        job_id = meta["job_id"]
        reason = "splatlab restarted while job was active"
        if idx == 0 and restarted == 0:
            req = _req_from_meta(meta)
            born = meta.get("started_at") or meta.get("created_at")
            fresh_enough = False
            with contextlib.suppress(Exception):
                age_h = (datetime.now(timezone.utc)
                         - datetime.fromisoformat(born)).total_seconds() / 3600
                fresh_enough = age_h <= RESUME_MAX_AGE_HOURS
            if (req is not None and fresh_enough
                    and int(meta.get("restart_count") or 0) < RESUME_MAX_RESTARTS
                    and _resolve_input_path(req.input_path).exists()):
                try:
                    _restart_job(meta, req)
                    restarted += 1
                    continue
                except Exception as exc:  # noqa: BLE001 — never wedge startup
                    reason = f"auto-restart after service restart failed ({exc})"
        _patch_meta(
            job_id,
            status="failed",
            stage=None,
            error_message=reason,
            completed_at=_utc_now(),
        )
    return restarted


def cleanup_orphan_jobs() -> int:
    """On portal start, mark jobs stuck in starting/running as failed.

    Note: a training process launched in its own session survives a portal
    restart; we lose the handle, so the job is marked failed even though the
    process may still be finishing on the GPU.
    """
    n = 0
    for meta in _all_metas():
        if meta.get("status") in ("starting", "running"):
            _patch_meta(
                meta["job_id"],
                status="failed",
                stage=None,
                error_message="portal restarted while job was active",
                completed_at=_utc_now(),
            )
            n += 1
    return n


def migrate_legacy_metas() -> int:
    """Backfill meta.json for job dirs created before persistence existed.

    Completed if a Nerfstudio config.yml exists anywhere under the dir,
    failed otherwise; preview availability is derived from disk at payload
    time. Legacy jobs are pinned to protect them from auto-prune.
    """
    migrated = 0
    if not DEFAULT_3D_ROOT.is_dir():
        return 0
    for d in DEFAULT_3D_ROOT.iterdir():
        if not d.is_dir() or not _safe_job_id(d.name) or (d / "meta.json").exists():
            continue
        ts_iso = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc).isoformat()
        has_config = next(d.rglob("config.yml"), None) is not None
        meta = {
            "job_id": d.name,
            "mode": "3d",
            "status": "completed" if has_config else "failed",
            "stage": None,
            "stages_planned": [],
            "stages_completed": [],
            "input_path": "(pre-persistence job)",
            "output_dir": str(d),
            "capture_format": "standard",
            "max_num_iterations": None,
            "command": [],
            "created_at": ts_iso,
            "started_at": ts_iso,
            "completed_at": ts_iso,
            "pid": None,
            "exit_code": None,
            "error_message": None if has_config else "legacy job missing training output",
            "stop_requested": False,
            "pinned": True,
        }
        _write_meta(d.name, meta)
        migrated += 1
    return migrated


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status")
async def get_splat_status():
    availability = _engine_availability()
    metas = sorted(_all_metas(), key=lambda m: m.get("created_at", ""), reverse=True)[:MAX_LISTED_JOBS]
    async with JOBS_LOCK:
        live = dict(JOBS)

    jobs = [_job_payload(meta, live.get(meta["job_id"])) for meta in metas]
    return {
        "workspace": {
            "root": str(SPLAT_ROOT),
            "data_dir": str(DATA_DIR),
            "outputs_dir": str(OUTPUTS_DIR),
            "conda_env_bin": str(CONDA_ENV_BIN),
        },
        "engines": availability,
        "media_samples": _sample_media_entries(),
        "jobs": jobs,
        "active_jobs": sum(1 for job in jobs if job["status"] in {"starting", "running"}),
        "gpu": gpu_arbiter.holder_info(),
        "notes": [
            "Relative input paths resolve from splatcli/data; relative output paths resolve from splatcli/.",
            "Each job gets its own directory under the output root: <output>/<job_id>/.",
            "Pipeline stages: [stitch] -> ns-process-data (COLMAP) -> ns-train splatfacto -> ns-export -> compress.",
            "Inputs that already contain transforms.json skip the processing stage.",
            "Input types: image folder, video (.mp4/.mov/...), raw Insta360 .insv (auto-stitched via ffmpeg v360), or a processed dataset.",
            "Raw .insv stitches with ffmpeg v360 (seams visible). For seamless output, export an equirectangular MP4 from the Insta360 app/Studio and feed that in 360 mode.",
            "360 capture: orbit slowly, keep the camera moving for parallax, and use crop-bottom to drop the operator/nadir.",
            "Training shares the RTX 5090 with TRELLIS through a single GPU lock — queued jobs wait their turn.",
            f"Unpinned completed jobs beyond the newest {KEEP_UNPINNED_COMPLETED} are pruned (job dirs are multi-GB). Pin anything worth keeping.",
            "4D training is deferred until the 3D path is validated.",
        ],
    }


@router.get("/transfers")
async def list_transfer_inputs():
    """List splat-ready inputs synced into ~/transfers.

    The Transfers folder is reachable from off-LAN devices via Syncthing or
    `pulse-share` (rsync over SSH) — neither limited by the ~100MB Cloudflare
    request-body cap on browser uploads through splat.roonytoony.dev. Dropping
    a video / .insv / .zip-of-photos / folder-of-images there lets the operator
    splat large captures without uploading through the tunnel.
    """
    return {
        "dir": str(TRANSFERS_DIR),
        "entries": _transfers_entries(),
    }


def _safe_upload_name(filename: str | None) -> str:
    """Strip any path components and keep a conservative basename."""
    base = Path(filename or "").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._") or "upload"
    return cleaned[:120]


def _extract_image_zip(zip_path: Path, dest: Path) -> int:
    """Extract image files from a zip into dest (flattened), zip-slip safe."""
    dest.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            # Flatten to a sanitized basename; this also defeats zip-slip
            # (no original path is honored, so "../" can't escape dest).
            out_name = _safe_upload_name(Path(info.filename).name)
            out_path = dest / out_name
            stem, dot, ext = out_name.rpartition(".")
            n = 1
            while out_path.exists():
                out_path = dest / f"{stem}_{n}{dot}{ext}"
                n += 1
            with zf.open(info) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


@router.post("/upload")
async def upload_splat_input(file: UploadFile):
    """Accept a novice upload (video, 360 .insv, or a .zip of photos) and stage
    it under data/uploads/<token>/. Returns a path the /train endpoint can use
    directly. Gated by the portal auth middleware like every other endpoint."""
    suffix = Path(file.filename or "").suffix.lower()
    is_zip = suffix == ".zip"
    if not is_zip and suffix not in UPLOAD_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file '{file.filename}'. Upload a video ({', '.join(sorted(VIDEO_EXTENSIONS))}), "
            f"a 360 .insv, or a .zip of photos.",
        )

    token = secrets.token_hex(5)
    upload_dir = UPLOADS_DIR / token
    upload_dir.mkdir(parents=True, exist_ok=True)
    staged = upload_dir / _safe_upload_name(file.filename)

    written = 0
    try:
        with staged.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds 2 GB.")
                fh.write(chunk)
    except HTTPException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    if written == 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    if is_zip:
        images_dir = upload_dir / "images"
        try:
            count = _extract_image_zip(staged, images_dir)
        except zipfile.BadZipFile:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="Not a valid .zip file.")
        staged.unlink(missing_ok=True)
        if count < 2:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"The .zip held {count} image(s). A splat needs many overlapping photos of the subject.",
            )
        result_path = images_dir
        kind = "directory"
        detail = f"{count} images"
    else:
        result_path = staged
        kind = "file"
        detail = f"{written / (1024 * 1024):.1f} MB"

    return {
        "path": str(result_path),
        "name": Path(file.filename or staged.name).name,
        "kind": kind,
        "is_insv": suffix in INSV_EXTENSIONS,
        "detail": detail,
    }


@router.post("/train")
async def start_splat_training(request: Request, req: SplatTrainRequest):
    if req.mode == "4d":
        raise HTTPException(
            status_code=501,
            detail="The 4D pipeline is deferred while the 3D path is rebuilt. Its engine deps are not installed yet.",
        )

    availability = _engine_availability()
    input_path = _resolve_input_path(req.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Input path not found: {input_path}")

    # In 360 mode each extracted frame becomes images_per_equirect perspective
    # views, so COLMAP's image count is the product. Guard against accidental
    # OOM-scale jobs (nerfstudio #2006) with a clear error rather than a crash.
    is_insv = input_path.suffix.lower() in INSV_EXTENSIONS
    if (req.capture_format == "equirectangular360" or is_insv):
        perspective_images = req.num_frames_target * req.images_per_equirect
        if perspective_images > 4000:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"360 mode reprojects each frame into {req.images_per_equirect} views, so this would feed COLMAP "
                    f"~{perspective_images} images ({req.num_frames_target}x{req.images_per_equirect}). Lower num_frames_target "
                    f"(~60-100 is plenty for 360) to keep it under ~4000."
                ),
            )

    output_root = _resolve_output_root(req.mode, req.output_dir)
    if output_root != DEFAULT_3D_ROOT:
        raise HTTPException(
            status_code=400,
            detail=f"Custom output roots are no longer supported — jobs persist under {DEFAULT_3D_ROOT}.",
        )

    job_id = f"splat_{secrets.token_hex(5)}"
    job_dir = _job_dir(job_id)
    stages, commands, sfm_context = _plan_3d_job(req, availability, job_dir, input_path)

    job = SplatJob(
        job_id=job_id,
        output_dir=str(job_dir),
        input_path=str(input_path),
        stages_planned=stages,
        stage_commands=commands,
        # Seed sfm_tried with the solver this job is actually starting on — see
        # _seed_sfm_tried for why sparse jobs need special handling too.
        sfm_tried=_seed_sfm_tried(sfm_context, stages),
        sfm_context=sfm_context,
        sfm_req=req if sfm_context else None,
    )

    async with JOBS_LOCK:
        active = [jid for jid, item in JOBS.items() if item.runner_task and not item.runner_task.done()]
        if active:
            raise HTTPException(
                status_code=409,
                detail=f"Job {active[0]} is already running. One splat job runs at a time on the RTX 5090.",
            )
        job_dir.mkdir(parents=True, exist_ok=True)
        _write_meta(job_id, _new_meta(job_id, req, input_path, job_dir, stages))
        JOBS[job_id] = job
        # Drop live handles for finished jobs; their state lives in meta.json.
        for jid in [j for j, item in JOBS.items() if item.runner_task and item.runner_task.done()]:
            del JOBS[jid]

    job.log_lines.append(f"Planned stages: {' -> '.join(stages)}")
    job.runner_task = asyncio.create_task(_run_pipeline(job))

    await audit_operator_event(
        request=request,
        title="Started Splat 3D job",
        description=f"{input_path.name} -> {job_dir}",
        variant="loading",
        action="splat.train",
        target=req.mode,
        metadata={"job_id": job_id, "input_path": str(input_path), "output_dir": str(job_dir), "stages": stages},
    )
    meta = _read_meta(job_id) or {}
    return _job_payload(meta, job)


@router.post("/jobs/{job_id}/preview")
async def generate_splat_preview(request: Request, job_id: str):
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    availability = _engine_availability()
    meta = _read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    if meta["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Preview export requires a completed job. Current status: {meta['status']}")
    if not availability["ns_export_available"]:
        raise HTTPException(status_code=400, detail="`ns-export` is not available in the splatops environment.")

    output_dir = Path(meta["output_dir"])
    config_path = _find_latest_config(output_dir)
    if config_path is None:
        raise HTTPException(status_code=404, detail=f"No Nerfstudio config.yml found under {output_dir}")

    preview_dir = _preview_dir_path(output_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_file = _preview_file_path(output_dir)

    command = _export_command(availability["ns_export_path"], config_path, output_dir)
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(SPLAT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_subprocess_env(),
        start_new_session=True,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0 or not preview_file.is_file():
        tail = "\n".join((stderr.decode("utf-8", errors="replace")).splitlines()[-10:])
        raise HTTPException(status_code=500, detail=f"Preview export failed (exit {process.returncode}): {tail}")

    await audit_operator_event(
        request=request,
        title="Exported Splat preview",
        description=f"{Path(meta['input_path']).name} -> {preview_file}",
        variant="success",
        action="splat.preview",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id, "preview_file": str(preview_file)},
    )
    return {
        "job_id": job_id,
        "preview_file_path": str(preview_file),
        "preview_file_url": f"/api/splat/jobs/{job_id}/preview/file",
    }


@router.get("/jobs/{job_id}/cameras")
async def get_splat_cameras(job_id: str, limit: int = 500):
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = _read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")

    output_dir = Path(meta["output_dir"])
    transforms_path = _find_scene_transforms(output_dir)
    if transforms_path is None:
        raise HTTPException(status_code=404, detail="No capture camera transforms found for this scene")

    try:
        transforms = json.loads(transforms_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read camera transforms: {exc}") from exc

    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise HTTPException(status_code=404, detail="No capture cameras found for this scene")

    dataparser_path = _find_dataparser_transforms(output_dir)
    dataparser_transform: list[list[float]] | None = None
    dataparser_scale = 1.0
    if dataparser_path is not None:
        try:
            dataparser = json.loads(dataparser_path.read_text())
            raw_transform = dataparser.get("transform")
            if isinstance(raw_transform, list) and len(raw_transform) == 3:
                raw_dataparser_transform = [[float(raw_transform[r][c]) for c in range(4)] for r in range(3)]
                dataparser_transform = _compose_saved_pose_transform(raw_dataparser_transform, transforms.get("applied_transform"))
                dataparser_scale = float(dataparser.get("scale", 1.0))
        except Exception:
            dataparser_transform = None
            dataparser_scale = 1.0

    limit = max(1, min(int(limit or 500), 1000))
    step = max(1, (len(frames) + limit - 1) // limit)
    cameras: list[dict[str, Any]] = []
    positions: list[list[float]] = []

    for index, frame in enumerate(frames):
        if index % step != 0 or not isinstance(frame, dict):
            continue
        raw_pose = frame.get("transform_matrix")
        if not isinstance(raw_pose, list) or len(raw_pose) < 3:
            continue
        try:
            pose = [[float(raw_pose[r][c]) for c in range(4)] for r in range(3)]
        except Exception:
            continue

        if dataparser_transform is not None:
            pose = _mat3x4_apply(dataparser_transform, pose)
            for r in range(3):
                pose[r][3] *= dataparser_scale

        position = [float(pose[0][3]), float(pose[1][3]), float(pose[2][3])]
        right = _vec_norm([pose[0][0], pose[1][0], pose[2][0]])
        up = _vec_norm([pose[0][1], pose[1][1], pose[2][1]])
        forward = _vec_norm([-pose[0][2], -pose[1][2], -pose[2][2]])
        positions.append(position)
        file_path = str(frame.get("file_path") or "")
        frame_height = frame.get("h", transforms.get("h"))
        frame_fy = frame.get("fl_y", transforms.get("fl_y"))
        fov_y_degrees = None
        if isinstance(frame_height, int | float) and isinstance(frame_fy, int | float) and frame_fy > 0:
            fov_y_degrees = math.degrees(2.0 * math.atan(float(frame_height) / (2.0 * float(frame_fy))))
        cameras.append({
            "index": index,
            "image_name": Path(file_path).name if file_path else f"camera-{index + 1}",
            "file_path": file_path,
            "position": position,
            "forward": forward,
            "up": up,
            "right": right,
            "fov_y_degrees": fov_y_degrees,
        })

    if not cameras:
        raise HTTPException(status_code=404, detail="No valid capture cameras found for this scene")

    width = transforms.get("w")
    height = transforms.get("h")
    return {
        "job_id": job_id,
        "count": len(cameras),
        "total": len(frames),
        "sampled": step > 1,
        "frame": "viewer" if dataparser_transform is not None else "source",
        "source": "dataparser_transforms" if dataparser_transform is not None else "transforms_json",
        "display_scale": _camera_display_scale(positions),
        "image_size": {
            "width": int(width) if isinstance(width, int | float) else None,
            "height": int(height) if isinstance(height, int | float) else None,
        },
        "cameras": cameras,
    }


@router.get("/jobs/{job_id}/preview/file")
async def get_splat_preview_file(job_id: str, fmt: Literal["ply", "spz", "web", "langweb"] = "ply"):
    # Resolved purely from disk so preview URLs survive portal restarts.
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    if fmt == "spz":
        preview_file = _preview_spz_path(_job_dir(job_id))
        suffix = "spz"
    elif fmt == "langweb":
        # Full-count SH-stripped copy whose row order matches gauss_emb.npz, for
        # the client-side language heatmap. Falls back to the raw .ply — identical
        # row order, just heavier — for jobs built before this artifact existed.
        preview_file = _preview_langweb_path(_job_dir(job_id))
        if not preview_file.is_file():
            preview_file = _preview_file_path(_job_dir(job_id))
        suffix = "ply"
    elif fmt == "web":
        # Lightweight copy for the web viewer; fall back to the raw .ply when the
        # web-optimized file hasn't been generated (older jobs, or webopt skipped).
        preview_file = _preview_web_path(_job_dir(job_id))
        if not preview_file.is_file():
            preview_file = _preview_file_path(_job_dir(job_id))
        suffix = "ply"
    else:
        preview_file = _preview_file_path(_job_dir(job_id))
        suffix = "ply"
    if not preview_file.is_file():
        raise HTTPException(status_code=404, detail="Preview file not generated yet")
    return FileResponse(str(preview_file), media_type="application/octet-stream", filename=f"{job_id}.{suffix}")


def _langfield_stale_guard(lfdir: Path) -> None:
    """A geometry edit (edit_ops) row-masked splat.ply after the field was built, so
    gauss_emb.npz rows no longer match the scene. Refuse to serve misaligned results."""
    if (lfdir / "STALE").is_file():
        raise HTTPException(
            status_code=409,
            detail="Language field is stale (the scene was edited after the field was "
            "built) — re-run the language field for this scene to search it again.",
        )


@router.post("/jobs/{job_id}/langfield/query")
async def langfield_query(job_id: str, payload: dict[str, Any]):
    """Text-search a built language field -> a server-rendered relevancy heatmap.
    Prefers the warm worker (sub-second); falls back to a cold subprocess under the
    GPU lock. 404 if this scene has no built field. (Auth via the router mount.)"""
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    raw = str(payload.get("text", "")).strip()
    if not raw:
        raise HTTPException(status_code=400, detail="empty query")
    clean = _langfield_clean_text(raw)
    job_dir = _job_dir(job_id)
    lfdir = job_dir / LANGFIELD_DIRNAME
    if not (lfdir / "gauss_emb.npz").is_file():
        raise HTTPException(status_code=404, detail="Language field not built for this scene")
    _langfield_stale_guard(lfdir)
    config_path = _find_latest_config(job_dir)
    if config_path is None:
        raise HTTPException(status_code=409, detail="Scene checkpoint missing")
    name = _langfield_heatmap_name(clean)
    worker_result = await _langfield_worker_query(str(config_path), str(lfdir), clean)
    focus: dict[str, Any] = {}
    if worker_result is not None:
        rendered = True
        # 3D centroid(s) of the match(es), for the viewer to fly to / highlight
        # (worker path only). `matches` = distinct clustered instances; each carries a
        # `thumb` filename the UI turns into a served heatmap URL (same job, same route).
        focus = {k: worker_result[k] for k in ("focus", "radius", "matches") if k in worker_result}
    else:
        rendered = await _langfield_query_cold(job_id, str(config_path), str(lfdir), clean)
    if not rendered or not (lfdir / name).is_file():
        raise HTTPException(status_code=500, detail="Language query render failed")
    return {
        "query": clean,
        "heatmap_url": f"/api/splat/jobs/{job_id}/langfield/heatmap/{name}",
        "ready": True,
        **focus,
    }


@router.post("/jobs/{job_id}/langfield/relevancy")
async def langfield_relevancy(job_id: str, payload: dict[str, Any]):
    """Raw per-gaussian relevancy for the CLIENT-SIDE heatmap: a uint8-quantized
    vector whose row i corresponds to gauss_emb.npz row i == the raw splat.ply /
    fmt=langweb row i. Body = N bytes; X-Count/X-Min/X-Max dequantize it
    (rel = min + q/255*(max-min)); X-Matches = the clustered 3D instances JSON.
    WARM-WORKER ONLY: no PNG render, no GPU lock — a down worker is a fast 503,
    never the ~30s cold-subprocess fallback (this is an interactive path).
    (Auth via the router mount.)"""
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    raw = str(payload.get("text", "")).strip()
    if not raw:
        raise HTTPException(status_code=400, detail="empty query")
    clean = _langfield_clean_text(raw)
    job_dir = _job_dir(job_id)
    lfdir = job_dir / LANGFIELD_DIRNAME
    if not (lfdir / "gauss_emb.npz").is_file():
        raise HTTPException(status_code=404, detail="Language field not built for this scene")
    _langfield_stale_guard(lfdir)
    config_path = _find_latest_config(job_dir)
    if config_path is None:
        raise HTTPException(status_code=409, detail="Scene checkpoint missing")
    resp = await _langfield_worker_relevancy(str(config_path), str(lfdir), clean)
    if resp is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Language-field worker is not running; the client-side heatmap needs "
                "the warm worker (systemd unit splatlab-langfield on :3417)."
            ),
        )
    headers = {k: v for k, v in resp.headers.items() if k.lower() in RELEVANCY_FORWARD_HEADERS}
    headers["Cache-Control"] = "no-store"
    return Response(content=resp.content, media_type="application/octet-stream", headers=headers)


def _langfield_paint_context(job_id: str) -> tuple[str, str]:
    """Shared resolution + guards for the paint endpoints: returns
    (config_path, lfdir) or raises. Same checks as /relevancy."""
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    job_dir = _job_dir(job_id)
    lfdir = job_dir / LANGFIELD_DIRNAME
    if not (lfdir / "gauss_emb.npz").is_file():
        raise HTTPException(status_code=404, detail="Language field not built for this scene")
    _langfield_stale_guard(lfdir)
    config_path = _find_latest_config(job_dir)
    if config_path is None:
        raise HTTPException(status_code=409, detail="Scene checkpoint missing")
    return str(config_path), str(lfdir)


@router.post("/jobs/{job_id}/langfield/select/sphere")
async def langfield_select_sphere(job_id: str, payload: dict[str, Any]):
    """Paint-brush stroke: rows (exported-ply order, LE uint32 binary) within a
    3D sphere. Proxied to the warm worker (GPU distance test on the resident
    positions — no numpy in this venv)."""
    config_path, lfdir = _langfield_paint_context(job_id)
    center = payload.get("center")
    radius = payload.get("radius")
    if not (isinstance(center, list) and len(center) == 3) or not isinstance(radius, (int, float)):
        raise HTTPException(status_code=400, detail="body must be {center:[x,y,z], radius}")
    resp = await _langfield_worker_json(
        "/select_sphere", {"config": config_path, "lfdir": lfdir, "center": center, "radius": radius}
    )
    if resp is None:
        raise HTTPException(status_code=503, detail="Language-field worker is not running (splatlab-langfield on :3417).")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "worker error"))
    return Response(
        content=resp.content,
        media_type="application/octet-stream",
        headers={"X-Count": resp.headers.get("X-Count", "0"), "Cache-Control": "no-store"},
    )


@router.get("/jobs/{job_id}/langfield/overrides")
async def langfield_overrides_list(job_id: str):
    """The scene's paint-override manifest (labels, ops, counts). Read straight
    from disk — works even when the worker is down."""
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    manifest = _job_dir(job_id) / LANGFIELD_DIRNAME / "overrides.json"
    if not manifest.is_file():
        return {"overrides": []}
    try:
        items = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        items = []
    return {"overrides": items if isinstance(items, list) else []}


@router.post("/jobs/{job_id}/langfield/overrides")
async def langfield_overrides_add(job_id: str, payload: dict[str, Any]):
    """Commit a paint action (label + indices). Guardrails (min count, max
    scene fraction, bounds) are enforced worker-side; their human-readable 400s
    pass through so the UI can offer force=true."""
    config_path, lfdir = _langfield_paint_context(job_id)
    body = {
        "config": config_path,
        "lfdir": lfdir,
        "label": str(payload.get("label", "")),
        "aliases": payload.get("aliases", []) if isinstance(payload.get("aliases"), list) else [],
        "op": str(payload.get("op", "assign")),
        "alpha": payload.get("alpha"),
        "indices_b64": str(payload.get("indices_b64", "")),
        "force": bool(payload.get("force", False)),
        "note": str(payload.get("note", "")),
    }
    resp = await _langfield_worker_json("/overrides_add", body)
    if resp is None:
        raise HTTPException(status_code=503, detail="Language-field worker is not running (splatlab-langfield on :3417).")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "worker error"))
    return resp.json()


@router.delete("/jobs/{job_id}/langfield/overrides/{override_id}")
async def langfield_overrides_delete(job_id: str, override_id: str):
    config_path, lfdir = _langfield_paint_context(job_id)
    if not re.fullmatch(r"[0-9a-f]{8}", override_id):
        raise HTTPException(status_code=404, detail="override not found")
    resp = await _langfield_worker_json(
        "/overrides_delete", {"config": config_path, "lfdir": lfdir, "override_id": override_id}
    )
    if resp is None:
        raise HTTPException(status_code=503, detail="Language-field worker is not running (splatlab-langfield on :3417).")
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.json().get("detail", "worker error"))
    return resp.json()


@router.get("/jobs/{job_id}/langfield/inventory")
async def langfield_inventory(job_id: str):
    """Top-N objects auto-detected in this scene (open-vocab), for the toggle-to-highlight
    legend. Warm-worker only + cached per scene; 404 if no field, 503 if the worker is
    down (the UI just hides the legend). (Auth via the router mount.)"""
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    job_dir = _job_dir(job_id)
    lfdir = job_dir / LANGFIELD_DIRNAME
    if not (lfdir / "gauss_emb.npz").is_file():
        raise HTTPException(status_code=404, detail="Language field not built for this scene")
    _langfield_stale_guard(lfdir)
    config_path = _find_latest_config(job_dir)
    if config_path is None:
        raise HTTPException(status_code=409, detail="Scene checkpoint missing")
    result = await _langfield_worker_inventory(str(config_path), str(lfdir))
    if result is None:
        raise HTTPException(status_code=503, detail="Inventory worker unavailable")
    return {"job_id": job_id, "items": result.get("items", [])}


@router.get("/jobs/{job_id}/langfield/heatmap/{name}")
async def langfield_heatmap(job_id: str, name: str):
    # name is constrained to the renderer's q_<safe>.png shape — no path traversal.
    if not _safe_job_id(job_id) or not re.match(r"^q_[\w-]+\.png$", name):
        raise HTTPException(status_code=404, detail="not found")
    heat = _job_dir(job_id) / LANGFIELD_DIRNAME / name
    if not heat.is_file():
        raise HTTPException(status_code=404, detail="heatmap not rendered yet")
    return FileResponse(str(heat), media_type="image/png")


@router.get("/jobs/{job_id}/health/receipt/{name}")
async def health_receipt(job_id: str, name: str):
    # name is constrained to the fog gate's fog_*.webp/png shape — no path traversal.
    if not _safe_job_id(job_id) or not re.match(r"^fog_[\w.-]+\.(png|webp)$", name):
        raise HTTPException(status_code=404, detail="not found")
    receipt = _job_dir(job_id) / HEALTH_DIRNAME / name
    if not receipt.is_file():
        raise HTTPException(status_code=404, detail="receipt not rendered")
    media = "image/webp" if name.endswith(".webp") else "image/png"
    return FileResponse(str(receipt), media_type=media)


@router.post("/jobs/{job_id}/stop")
async def stop_splat_training(request: Request, job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    meta = _read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    if job is None or meta["status"] not in {"starting", "running"}:
        raise HTTPException(status_code=409, detail=f"Job is already {meta['status']}")

    job.stop_requested = True
    job.log_lines.append("Stop requested by operator.")
    _kill_job_process(job, signal.SIGTERM)

    if job.runner_task:
        try:
            await asyncio.wait_for(asyncio.shield(job.runner_task), timeout=STOP_GRACE_SECONDS)
        except asyncio.TimeoutError:
            job.log_lines.append("Process did not terminate in time. Sending SIGKILL.")
            _kill_job_process(job, signal.SIGKILL)
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(job.runner_task), timeout=STOP_GRACE_SECONDS)

    await audit_operator_event(
        request=request,
        title="Stopped Splat 3D job",
        description=f"{Path(meta['input_path']).name} -> {meta['output_dir']}",
        variant="default",
        action="splat.stop",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id},
    )
    return _job_payload(_read_meta(job_id) or meta, job)


@router.delete("/jobs/{job_id}")
async def delete_splat_job(request: Request, job_id: str):
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    meta = _read_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    async with JOBS_LOCK:
        live = JOBS.get(job_id)
        if live and live.runner_task and not live.runner_task.done():
            raise HTTPException(status_code=409, detail="Stop the job before deleting it.")
        JOBS.pop(job_id, None)
    _delete_job_files(job_id)
    await audit_operator_event(
        request=request,
        title="Deleted Splat job",
        description=f"{job_id} ({meta.get('input_path', '?')})",
        variant="default",
        action="splat.delete",
        target=meta.get("mode", "3d"),
        metadata={"job_id": job_id},
    )
    return {"ok": True, "job_id": job_id}


@router.post("/jobs/{job_id}/pin")
async def pin_splat_job(job_id: str):
    meta = _patch_meta(job_id, pinned=True) if _safe_job_id(job_id) else None
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return {"ok": True, "job_id": job_id, "pinned": True}


@router.post("/jobs/{job_id}/scale")
async def set_splat_scale(job_id: str, payload: dict[str, Any]):
    """Survey-lane scale calibration: METERS PER SCENE UNIT, set by measuring a
    reference of known real-world length in the viewer. nerfstudio scenes are
    non-metric (poses auto-normalized to a unit box), so this stored factor is
    the only bridge from scene units to real distances — measure/DXF/LandXML
    all hang off it. Body {"meters_per_unit": <float>} sets; null clears.
    Reaches clients via the **meta spread in _job_payload."""
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    raw = payload.get("meters_per_unit")
    if raw is None:
        meta = _patch_meta(job_id, meters_per_unit=None)
    else:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="meters_per_unit must be a number")
        if not (math.isfinite(value) and 0.0 < value < 1e6):
            raise HTTPException(status_code=400, detail="meters_per_unit must be finite and > 0")
        meta = _patch_meta(job_id, meters_per_unit=value)
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return {"ok": True, "job_id": job_id, "meters_per_unit": meta.get("meters_per_unit")}


@router.post("/jobs/{job_id}/unpin")
async def unpin_splat_job(job_id: str):
    meta = _patch_meta(job_id, pinned=False) if _safe_job_id(job_id) else None
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return {"ok": True, "job_id": job_id, "pinned": False}
