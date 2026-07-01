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

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import gpu_arbiter
from operator_audit import audit_operator_event

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
TRANSFERS_DIR = Path.home() / "transfers"
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
SFM_ESCALATION = ["colmap", "glomap", "mast3r"]
# Per-solver availability key in _engine_availability(). A solver is only a
# valid escalation target when its key is truthy. "colmap" is always present
# (validated by _plan_3d_job up front), so it has no gating key.
SFM_SOLVER_AVAILABILITY = {
    "glomap": "glomap_available",
    "mast3r": "mast3r_available",
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
    # via --skip-colmap. Only applied to video / image-folder inputs (not a
    # pre-processed dataset, not equirectangular — for now); otherwise it falls
    # back to the default path silently.
    sfm_backend: Literal["colmap", "glomap"] = "colmap"
    # OPT-IN: build a text-searchable language field after training (SAM 2.1 +
    # SigLIP 2, training-free lift). Default off → the pipeline is byte-identical.
    # Best-effort: a build failure never fails the splat job.
    language_field: bool = False


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
    # NOT equirect / pre-processed-dataset).
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
        "input_path": str(input_path),
        "output_dir": str(job_dir),
        "capture_format": req.capture_format,
        "max_num_iterations": req.max_num_iterations,
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


def _langfield_available() -> bool:
    """True iff the Language-Field toolchain is present: the build wrapper + both
    conda-env pythons (langfield-spike = SigLIP+lift, sam2 = masks). The heavy deps
    live in those envs; splatlab's own venv is never touched."""
    lf_py = os.environ.get("SPLAT_LANGFIELD_PYTHON", "").strip() or str(LANGFIELD_ENV_PYTHON)
    sam_py = os.environ.get("SPLAT_SAM2_PYTHON", "").strip() or str(SAM2_ENV_PYTHON)
    return all(Path(p).is_file() for p in (LANGFIELD_RUNNER, lf_py, sam_py))


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
    """Return {streams, width, height} for the first video stream, best-effort."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries",
             "stream=width,height", "-of", "json", str(src)],
            capture_output=True, text=True, timeout=20,
        )
        data = json.loads(out.stdout or "{}")
        streams = data.get("streams", [])
        first = streams[0] if streams else {}
        return {"streams": len(streams), "width": first.get("width"), "height": first.get("height")}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {"streams": 0, "width": None, "height": None}


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
        # MASt3R-SfM rung (strongest escalation target): env python + runner +
        # converter + 2.6GB checkpoint. Spread into the dict so the gate sees
        # mast3r_available / mast3r_* alongside the COLMAP keys.
        **_mast3r_availability(),
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


def _splat_transform_path() -> str | None:
    override = os.environ.get("SPLAT_TRANSFORM_BIN", "").strip()
    if override:
        return override
    if SPLAT_TRANSFORM_BIN.is_file():
        return str(SPLAT_TRANSFORM_BIN)
    return shutil.which("splat-transform")


def _job_payload(meta: dict[str, Any], live: SplatJob | None = None) -> dict:
    job_id = meta["job_id"]
    output_dir = Path(meta["output_dir"])
    preview_file = _preview_file_path(output_dir)
    preview_spz = _preview_spz_path(output_dir)
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
        "langfield_available": (output_dir / LANGFIELD_DIRNAME / "gauss_emb.npz").is_file(),
    }


def _find_latest_config(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.rglob("config.yml"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


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


def _stitch_command(ffmpeg: str, src: Path, dst: Path, fov: float) -> list[str]:
    """ffmpeg v360: Insta360 dual-fisheye -> equirectangular MP4 (open-source,
    no SDK). Seams are visible at the lens boundary but splatfacto tolerates
    them; for seamless output, export equirect from the Insta360 app instead."""
    vf = (
        f"v360=input=dfisheye:output=e:ih_fov={fov}:iv_fov={fov}"
        f":w={STITCH_WIDTH}:h={STITCH_HEIGHT}:interp=lanczos"
    )
    return [ffmpeg, "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-crf", "18", "-an", str(dst)]


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
) -> list[str]:
    """One self-contained shell command for the opt-in global-SfM backend.

    Runs, in order, all writing under <job_dir>/colmap/:
      0. (video only) ffmpeg extracts ~num_frames_target evenly-spaced frames
         into colmap/images/ — so COLMAP and ns-process-data see identical
         filenames. (image input: the frames are symlinked/copied in as-is.)
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

    if is_video:
        # -vf fps drops the clip to ~num_frames_target frames over its length;
        # COLMAP solves more frames than it can register, so over-sample lightly.
        # We compute the fps from duration so the count tracks the request even
        # for clips of unknown length: take every Nth frame via select.
        extract = (
            f'ffmpeg -y -i "{process_input}" '
            f'-vf "select=not(mod(n\\,$STRIDE))" -vsync vfr '
            f'-q:v 2 "{img_dir}/frame_%05d.jpg"'
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
            f'do cp -n "$f" "{img_dir}/"; done'
        )

    script = (
        f'set -euo pipefail; '
        # rm the sparse dir first so a stale sparse/0 from an aborted prior run
        # can't satisfy the final test -f and feed nerfstudio a partial model.
        # ALSO rm processed_dir: on a colmap->glomap auto-fallback reroute the
        # earlier colmap `process` already populated it (transforms.json + images/
        # + images_2/4/8); ns-process-data does NOT guarantee a clean overwrite, so
        # the A1 registration gate could otherwise measure a stale colmap/glomap
        # mix. Mirrors the mast3r path's clear. (Fresh glomap-start jobs: no-op.)
        f'rm -rf "{sparse_dir}" "{processed_dir}"; mkdir -p "{img_dir}" "{sparse_dir}"; '
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
        f'--SequentialMatching.loop_detection 1 --FeatureMatching.use_gpu 1; '
        f'"{colmap4}" global_mapper '
        f'--database_path "{db_path}" --image_path "{img_dir}" '
        f'--output_path "{sparse_dir}"; '
        # global_mapper writes its first reconstruction to sparse/0/ already;
        # assert the expected model exists so the stage fails loud if SfM produced
        # nothing (rather than letting ns-process-data choke downstream).
        f'test -f "{model_dir}/cameras.bin"'
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
        # ViT-Large dense matching -> in-memory SparseGA -> poses.npz/points3D.npz.
        f'"{availability["mast3r_python"]}" "{availability["mast3r_runner"]}" '
        f'--images "{img_dir}" --out "{run_out}" '
        f'--ckpt "{availability["mast3r_checkpoint"]}" '
        f'--scene-graph {MAST3R_SCENE_GRAPH}; '
        # Reproduce nerfstudio's colmap_to_json convention -> processed_dir.
        f'"{availability["mast3r_python"]}" "{availability["mast3r_converter"]}" '
        f'--mast3r-out "{run_out}" --src-images "{img_dir}" '
        f'--out "{processed_dir}" --image-mode resized; '
        # Assert the converter produced a usable Nerfstudio dataset so the stage
        # fails loud (rather than letting the A1 gate / train choke downstream).
        f'test -f "{processed_dir}/transforms.json"'
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
    if solver == "glomap":
        # COLMAP 4.x global SfM: we drive feature_extractor + sequential
        # matching + global_mapper ourselves into <job_dir>/colmap/sparse/0,
        # then ns-process-data reads that model via --skip-colmap.
        glomap_sfm = _glomap_sfm_command(
            colmap4=availability["glomap_path"],
            ffmpeg=availability["ffmpeg_path"],
            job_dir=job_dir,
            processed_dir=processed_dir,
            process_input=process_input,
            is_video=(subcommand == "video"),
            num_frames_target=req.num_frames_target,
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
        # far better than the default unordered-collection vocab_tree. Skip for
        # equirect: each frame fans out into N perspective views, not a sequence.
        if not is_equirect:
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


def _next_sfm_solver(tried: set[str], availability: dict) -> str | None:
    """The next solver in SFM_ESCALATION that is AVAILABLE and not yet tried.

    Walks the whole chain from the start (not from a fixed index) so it works
    regardless of which rung the job started on, and so an unavailable rung is
    transparently skipped to the next available one. Returns None when the
    chain is exhausted.
    """
    for candidate in SFM_ESCALATION:
        if candidate in tried:
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
    inputs (video / image-folder, not equirect / pre-processed dataset); it carries
    the inputs the runner's A1 gate needs to rebuild SfM stages for a fallback solver.

    Inputs already containing a Nerfstudio dataset (transforms.json) skip the
    ns-process-data stage and train directly on the input.
    """
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
    # A pre-processed dataset (transforms.json) or equirect job leaves it None,
    # so the runner's gate never reroutes those.
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
        stitched = _stitched_path(job_dir)
        commands["stitch"] = _stitch_command(availability["ffmpeg_path"], input_path, stitched, req.insv_fov)
        stages.append("stitch")
        process_input = stitched
        is_video = True
        is_equirect = True
    else:
        process_input = input_path
        is_video = input_path.suffix.lower() in VIDEO_EXTENSIONS
        is_equirect = req.capture_format == "equirectangular360"

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
        # only engages for plain video / image-folder inputs — NOT equirectangular
        # (each 360 frame fans out into N perspective views, not one temporal
        # sequence) and NOT a pre-processed dataset (handled above). For those
        # excluded cases we run the requested solver once with no reroute.
        escalation_eligible = (not is_equirect) and subcommand in ("video", "images")

        # Resolve the starting solver. The requested sfm_backend is honored when
        # available; an opt-in glomap that isn't actually present silently falls
        # back to colmap (byte-for-byte the original behavior). MASt3R etc. slot
        # in here automatically via SFM_SOLVER_AVAILABILITY.
        start_solver = req.sfm_backend
        avail_key = SFM_SOLVER_AVAILABILITY.get(start_solver)
        if avail_key is not None and not availability.get(avail_key):
            start_solver = "colmap"

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
    commands["train"] = [
        availability["ns_train_path"],
        "splatfacto",
        "--data",
        str(train_data),
        "--output-dir",
        str(job_dir),
        "--max-num-iterations",
        str(req.max_num_iterations),
        "--viewer.quit-on-train-completion",
        "True",
    ]
    stages.extend(["train", "export"])
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
        image-folder; equirect / pre-processed-dataset leave it None);
      - reroute_count is capped at len(SFM_ESCALATION);
      - the chosen solver is added to job.sfm_tried BEFORE rerouting, and
        _next_sfm_solver skips anything already tried — so no solver runs twice.
    """
    ctx = job.sfm_context
    if ctx is None:
        # Equirect / pre-processed dataset: not eligible. Caller fails as before.
        return False
    if job.reroute_count >= len(SFM_ESCALATION):
        # Hard cap: even if availability lies, never reroute more than there are
        # rungs in the chain.
        return False

    availability = _engine_availability()
    next_solver = _next_sfm_solver(job.sfm_tried, availability)
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
                # Pre-flight ffprobe: X4/X5 store dual H.265 streams in one
                # .insv and ffmpeg sees only the first, so warn (don't fail) if
                # the layout looks wrong. Single side-by-side dual-fisheye is
                # ~2:1; a near-1:1 or multi-stream file likely needs an app
                # export instead.
                stitched = _stitched_path(job_dir)
                stitched.parent.mkdir(parents=True, exist_ok=True)
                ffprobe = _tool_path("ffprobe", "SPLAT_FFPROBE_BIN")
                if ffprobe:
                    info = _probe_video_streams(ffprobe, Path(job.input_path))
                    job.log_lines.append(
                        f"[stitch] source: {info['streams']} video stream(s), "
                        f"{info['width']}x{info['height']}"
                    )
                    if info["streams"] > 1:
                        job.log_lines.append(
                            "[stitch] WARNING: multiple video streams — this model likely stores "
                            "dual-stream fisheye; ffmpeg sees only the first. If the stitch looks "
                            "half-empty, export an equirectangular MP4 from the Insta360 app instead."
                        )
                return_code = await _run_stage(job, stage, job.stage_commands["stitch"])
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
                else:
                    job.log_lines.append("[webopt] skipped (tool or .ply unavailable).")
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
                except Exception as exc:  # noqa: BLE001 — best-effort: never fail the splat
                    job.log_lines.append(f"[langfield] skipped (build error: {exc}); the splat is unaffected.")
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
                    job.log_lines.append(
                        f"[process] registration check skipped (could not read transforms/images: {exc})."
                    )

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
        final_status = "failed"
        error_message = f"Pipeline crashed: {exc}"
        job.log_lines.append(error_message)
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
        # Seed sfm_tried with the solver this job is actually starting on (the
        # planner may have downgraded an unavailable glomap request to colmap),
        # not the raw requested backend — so the gate's "not yet tried" check is
        # honest about what really ran. sfm_context carries the rebuild inputs.
        sfm_tried={sfm_context["start_solver"]} if sfm_context else set(),
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


@router.get("/jobs/{job_id}/preview/file")
async def get_splat_preview_file(job_id: str, fmt: Literal["ply", "spz", "web"] = "ply"):
    # Resolved purely from disk so preview URLs survive portal restarts.
    if not _safe_job_id(job_id):
        raise HTTPException(status_code=404, detail="Splat job not found")
    if fmt == "spz":
        preview_file = _preview_spz_path(_job_dir(job_id))
        suffix = "spz"
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


@router.get("/jobs/{job_id}/langfield/heatmap/{name}")
async def langfield_heatmap(job_id: str, name: str):
    # name is constrained to the renderer's q_<safe>.png shape — no path traversal.
    if not _safe_job_id(job_id) or not re.match(r"^q_[\w-]+\.png$", name):
        raise HTTPException(status_code=404, detail="not found")
    heat = _job_dir(job_id) / LANGFIELD_DIRNAME / name
    if not heat.is_file():
        raise HTTPException(status_code=404, detail="heatmap not rendered yet")
    return FileResponse(str(heat), media_type="image/png")


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


@router.post("/jobs/{job_id}/unpin")
async def unpin_splat_job(job_id: str):
    meta = _patch_meta(job_id, pinned=False) if _safe_job_id(job_id) else None
    if not meta:
        raise HTTPException(status_code=404, detail="Splat job not found")
    return {"ok": True, "job_id": job_id, "pinned": False}
