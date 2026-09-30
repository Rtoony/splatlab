"""Spirula 360 lane: a raw dual-fisheye Insta360 .insv -> Spirula Studio -> a Z-up splat.

Why (2026-09-29 bake-off, ~/reports/2026-09-29-spirula-bakeoff/): SplatLab's rig lane stitches the two
lenses into an equirect, cuts 12 pinhole views and runs COLMAP + splatfacto; the stitch/resample softens
the footage and splatfacto stops growing (181k splats on the storage room). Spirula trains on the RAW
fisheye with its own rig-aware SfM (6.5 min vs ~60) and was visibly far sharper. Owner decision
2026-09-29: default lane for .insv, view + walk first (no nerfstudio checkpoint, so langfield / mesh /
isolate / world stay on rig-lane jobs), quality `high`.

Spirula is GPL-3.0: it runs as a subprocess from ~/tools and is never vendored. Guards that are NOT
optional (all measured): no `academic-baseline` preset (diverged on 360 views), no `--resume`
(re-evaluation unfaithful / size-mismatch crash), `sfm auto` gets the IMAGE dir (the dataset dir also
enumerates masks/), Vulkan pinned to the NVIDIA ICD (an Intel iGPU also enumerates), and production
runs hold nothing out (`--eval-mode all`) so the eval-writer race cannot bite.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SPIRULA_BIN = Path.home() / "tools" / "spirula" / "2026.9.24" / "spirula"
# Spirula's own SAM 3 (sam3.cpp GGML conversion of Meta's SAM 3, 1.84 GB) — the same cache path its app uses.
SAM3_MODEL = Path.home() / ".cache" / "spirula-studio" / "models" / "sam3-f16.ggml"
NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
WORKDIR = "_spirula"

INSTANTS_PER_SECOND = 3.3          # the storage-room proof: 410 instants from 123 s (stride 9 at 29.97 fps)
MAX_INSTANTS = 1400
MIN_STRIDE = 3
CPU_CACHE_BUDGET_BYTES = 40e9      # decoded RGB frames kept in RAM; the service cgroup is MemoryHigh 64G
SCALES = (1.0, 0.75, 0.5)          # extraction scale ladder when the full-res frames would not fit
FOCAL_PER_WIDTH = 0.27             # ~204° equidistant lens: f ≈ r/θmax (measured 518.7 px at 1920)
MIN_REGISTERED_FRACTION = 0.8

EXTRACT_VRAM_MB = 4_000
PERSON_VRAM_MB = 8_000
MASK_VRAM_MB = 4_000
SFM_VRAM_MB = 8_000
TRAIN_VRAM_MB = 16_000

STAGES = ("spirula_trim", "spirula_extract", "spirula_mask", "spirula_person", "spirula_sfm", "spirula_train",
          "spirula_publish")


def spirula_bin() -> str:
    return os.environ.get("SPLAT_SPIRULA_BIN", "").strip() or str(SPIRULA_BIN)


def availability() -> dict:
    binary = spirula_bin()
    ok = Path(binary).is_file() and os.access(binary, os.X_OK) and Path(NVIDIA_ICD).is_file()
    return {"spirula_available": ok, "spirula_path": binary}


def sam3_model() -> str | None:
    """The SAM 3 checkpoint for operator masking, or None (then the person stage is skipped)."""
    path = os.environ.get("SPLAT_SPIRULA_SAM3", "").strip() or str(SAM3_MODEL)
    return path if Path(path).is_file() else None


def person_mask_enabled() -> bool:
    """Kill-switch: SPLAT_SPIRULA_PERSON_MASK=0 trains without masking the camera operator."""
    return os.environ.get("SPLAT_SPIRULA_PERSON_MASK", "").strip() != "0"


def lane_enabled() -> bool:
    """Kill-switch: SPLAT_SPIRULA_LANE=0 sends every .insv back to the rig lane."""
    return os.environ.get("SPLAT_SPIRULA_LANE", "").strip() != "0"


def quality() -> str:
    q = os.environ.get("SPLAT_SPIRULA_QUALITY", "").strip() or "high"
    return q if q in ("medium", "high", "ultra") else "high"


def web_decimate_target() -> int:
    """Spirula jobs carry 3M+ splats; the walker's default 1.2M decimation throws most of them away."""
    try:
        return max(100_000, int(os.environ.get("SPLAT_SPIRULA_WEB_DECIMATE", "") or 3_000_000))
    except ValueError:
        return 3_000_000


# ---------------------------------------------------------------------------- probe + settings

def probe(ffprobe: str, path: Path) -> dict:
    """Video streams + duration of an .insv (dual-fisheye X4/X5 = two square HEVC streams)."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type,width,height,nb_frames,r_frame_rate:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        data = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {"video_streams": 0}
    vids = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    if not vids:
        return {"video_streams": 0}
    num, _, den = str(vids[0].get("r_frame_rate", "30/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    duration = float(data.get("format", {}).get("duration") or 0.0)
    frames = int(vids[0].get("nb_frames") or 0) or int(round(duration * fps))
    return {"video_streams": len(vids), "width": int(vids[0].get("width") or 0),
            "height": int(vids[0].get("height") or 0), "fps": fps, "duration": duration, "frames": frames,
            "same_size": len({(v.get("width"), v.get("height")) for v in vids}) == 1}


def is_dual_fisheye(info: dict) -> bool:
    return (info.get("video_streams") == 2 and info.get("same_size")
            and info.get("width") and info.get("width") == info.get("height"))


@dataclass
class LaneSettings:
    stride: int
    instants: int
    scale: float
    cache: str
    focal: float
    width: int
    window_s: float
    quality: str


def settings(info: dict, trim_duration_s: float | None = None, quality_override: str | None = None) -> LaneSettings:
    """Frame stride, extraction scale and image cache for one clip.

    ~3.3 instants per second (the proven storage-room density), at most MAX_INSTANTS, never denser than
    every MIN_STRIDE-th frame. Full resolution when the decoded frames fit the RAM cache budget; otherwise
    the largest scale that does, and only past the ladder a disk cache.
    """
    fps = info.get("fps") or 30.0
    window = float(trim_duration_s) if trim_duration_s else float(info.get("duration") or 0.0)
    frames = max(1, int(round(window * fps)) if window else int(info.get("frames") or 1))
    target = max(1, min(math.ceil(INSTANTS_PER_SECOND * (frames / fps)), MAX_INSTANTS))
    stride = max(MIN_STRIDE, frames // target)
    instants = math.ceil(frames / stride)
    w, h = int(info.get("width") or 3840), int(info.get("height") or 3840)
    scale, cache = SCALES[-1], "disk"
    for s in SCALES:
        if instants * 2 * (w * s) * (h * s) * 3 <= CPU_CACHE_BUDGET_BYTES:
            scale, cache = s, "cpu"
            break
    width = int(round(w * scale))
    return LaneSettings(stride=stride, instants=instants, scale=scale, cache=cache,
                        focal=round(FOCAL_PER_WIDTH * width, 1), width=width, window_s=round(window, 2),
                        quality=quality_override if quality_override in ("high", "ultra") else quality())


# ---------------------------------------------------------------------------- commands

def workdir(job_dir: Path) -> Path:
    return Path(job_dir) / WORKDIR


def commands(binary: str, input_path: Path, job_dir: Path, s: LaneSettings, ffmpeg: str | None = None,
             trim_start_s: float | None = None, trim_duration_s: float | None = None) -> dict[str, list[str]]:
    ws = workdir(job_dir)
    data, images = ws / "data", ws / "data" / "images"
    run = ["env", f"VK_ICD_FILENAMES={NVIDIA_ICD}", binary]
    cmds: dict[str, list[str]] = {}
    source = Path(input_path)
    if trim_start_s is not None or trim_duration_s is not None:
        if not ffmpeg:
            raise ValueError("a trimmed .insv needs ffmpeg")
        source = ws / "trimmed.insv"
        cut = []
        if trim_start_s is not None:
            cut += ["-ss", f"{trim_start_s:g}"]
        if trim_duration_s is not None:
            cut += ["-t", f"{trim_duration_s:g}"]
        # Stream copy keeps both lens tracks; the cut lands on keyframes (fine for a test flight).
        cmds["spirula_trim"] = ["bash", "-c", f'mkdir -p "{ws}" && "{ffmpeg}" -y -loglevel error '
                                + " ".join(cut) + f' -i "{input_path}" -map 0:v -c copy "{source}"']
    cmds["spirula_extract"] = [*run, "sam", "extract", str(source), "--sync", "--skip", str(s.stride),
                               "--keep", "0", *(["--scale", f"{s.scale:g}"] if s.scale < 1 else []),
                               "-o", str(images)]
    cmds["spirula_mask"] = [*run, "sam", "mask", str(images)]
    model = sam3_model()
    if model and person_mask_enabled():
        # The camera operator (on the stick) moves WITH the camera, so robust losses cannot drop them and they train
        # into a smoky ghost (office A/B 09-30: --distraction-robustness mild/strong left it untouched). SAM 3 masks
        # "person" per lens; combine_person_masks() then ANDs them into masks/ so SfM and training both ignore them.
        person = ws / "data" / "person"
        cmds["spirula_person"] = ["bash", "-c", " && ".join(
            f'env VK_ICD_FILENAMES={NVIDIA_ICD} "{binary}" sam track --model "{model}" --frames "{images}/{cam}" '
            f'--text person --out "{person}/{cam}"' for cam in ("cam0", "cam1"))]
    # Up (and metric scale) from the camera's own IMU: the .insv trailer carries a 1 kHz gyro+accelerometer log.
    # Without it Spirula GUESSES up from the largest level-looking plane, and in a shelf-lined aisle it picked a wall
    # (storage room 2026-09-29, published sideways). A trimmed copy has no trailer (ffmpeg drops it), so a test
    # flight falls back to the guess and publish flags it.
    telemetry = [] if source != Path(input_path) else ["--telemetry", str(input_path)]
    cmds["spirula_sfm"] = [*run, "sfm", "auto", str(images), "-o", str(data), "--data-type", "video",
                           "--rig", "dual-fisheye=cam0,cam1", "--sequence", "cam0,cam1",
                           "--camera-model", "opencv-fisheye", "--focal", f"{s.focal:g}", *telemetry]
    cmds["spirula_train"] = [*run, "train", "360-camera", "--data", str(data), "--data-format", "colmap",
                             "--output-dir-prefix", str(ws), "--output-dir-name", "train",
                             "--quality", s.quality, "--floater-suppression", "mild",
                             "--cache-images", s.cache, "--eval-mode", "all", "--disable-viewer", "1"]
    return cmds


# ---------------------------------------------------------------------------- COLMAP binary readers

def read_images_bin(path: Path) -> list[dict]:
    """COLMAP images.bin -> [{id, qvec(wxyz), tvec, camera_id, name}] (2D points skipped)."""
    out = []
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        for _ in range(n):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<I7dI", fh.read(64))
            name = bytearray()
            while (c := fh.read(1)) not in (b"\x00", b""):
                name += c
            (npts,) = struct.unpack("<Q", fh.read(8))
            fh.seek(npts * 24, 1)
            out.append({"id": iid, "qvec": (qw, qx, qy, qz), "tvec": (tx, ty, tz), "camera_id": cid,
                        "name": name.decode()})
    return out


def _rotmat(q) -> list[list[float]]:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def c2w_gl(qvec, tvec) -> list[list[float]]:
    """COLMAP world->camera (OpenCV) → nerfstudio-style camera->world (OpenGL: x right, y up, z back)."""
    r = _rotmat(qvec)
    rt = [[r[j][i] for j in range(3)] for i in range(3)]                     # R^T = camera->world rotation
    centre = [-sum(rt[i][k] * tvec[k] for k in range(3)) for i in range(3)]
    return [[rt[i][0], -rt[i][1], -rt[i][2], centre[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


# ---------------------------------------------------------------------------- checks + publish

def sfm_check(job_dir: Path) -> tuple[bool, str]:
    """The run fails unless the main model registered MIN_REGISTERED_FRACTION of the extracted frames."""
    data = workdir(job_dir) / "data"
    images_bin = data / "sparse" / "0" / "images.bin"
    extracted = sum(1 for p in (data / "images").rglob("*.jpg"))
    if not images_bin.is_file():
        return False, "[spirula_sfm] no model at sparse/0 — structure from motion failed."
    with open(images_bin, "rb") as fh:
        (registered,) = struct.unpack("<Q", fh.read(8))
    frac = registered / extracted if extracted else 0.0
    others = sorted(p.name for p in (data / "sparse").iterdir() if p.is_dir() and p.name != "0")
    msg = (f"[spirula_sfm] registered {registered}/{extracted} fisheye images ({frac:.0%})"
           + (f"; {len(others)} smaller disconnected model(s) ignored" if others else ""))
    if frac < MIN_REGISTERED_FRACTION:
        return False, msg + f" — below {MIN_REGISTERED_FRACTION:.0%}; the capture did not hold together."
    return True, msg


def combine_person_masks(job_dir: Path) -> str:
    """AND SAM 3's per-frame person masks (frame_NNNNN.png = the Nth image in sorted order, white = keep) into the
    lens-border masks in data/masks/. Idempotent. Returns a one-line receipt."""
    import numpy as np
    from PIL import Image
    data = workdir(job_dir) / "data"
    pristine = data / "masks-border"            # the lens-border masks as `sam mask` wrote them, kept for re-runs
    if not pristine.is_dir():
        shutil.copytree(data / "masks", pristine)
    total = kept = frames = 0
    for cam in ("cam0", "cam1"):
        for i, img in enumerate(sorted((data / "images" / cam).glob("*.jpg"))):
            border_path = data / "masks" / cam / f"{img.stem}.png"
            source_path = pristine / cam / f"{img.stem}.png"
            person_path = data / "person" / cam / f"frame_{i:05d}.png"
            if not source_path.is_file() or not person_path.is_file():
                continue
            b = np.asarray(Image.open(source_path).convert("L")) >= 128
            p = np.asarray(Image.open(person_path).convert("L").resize(b.shape[::-1])) >= 128
            m = b & p
            Image.fromarray((m * 255).astype(np.uint8)).save(border_path)
            total += int(b.sum()); kept += int(m.sum()); frames += 1
    share = 100.0 * (1 - kept / total) if total else 0.0
    return f"[spirula_person] masked the camera operator in {frames} frames ({share:.2f} % of in-lens pixels)"


def latest_splat(job_dir: Path) -> Path | None:
    plys = sorted((workdir(job_dir) / "train").glob("step-*.ckpt/splat.ply"),
                  key=lambda p: int(re.sub(r"\D", "", p.parent.name) or 0))
    return plys[-1] if plys else None


def _copy_with_up_comment(src: Path, dst: Path) -> int:
    """Copy a binary PLY, adding SplatLab's `Vertical Axis: z` comment; returns the vertex count."""
    with open(src, "rb") as fin:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = fin.readline()
            if not line:
                raise ValueError(f"{src}: no end_header")
            header += line
        text = header.decode("ascii")
        count = int(re.search(r"element vertex (\d+)", text).group(1))
        if "Vertical Axis" not in text:
            # After the `format` line: strict PLY readers require `format` to follow `ply` directly.
            text = re.sub(r"(format [^\n]*\n)", r"\1comment Vertical Axis: z\n", text, count=1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".ply.tmp")
        with open(tmp, "wb") as fout:
            fout.write(text.encode("ascii"))
            shutil.copyfileobj(fin, fout, 16 << 20)
        os.replace(tmp, dst)
    return count


def read_gauge(path: Path) -> dict[str, str]:
    """sparse/N/gauge.txt -> {oriented, metric, up, scale}: whether +Z-up / metre units were MEASURED, and by what."""
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2 and not line.startswith("#"):
                out[parts[0]] = parts[1].strip()
    return out


def publish(job_dir: Path, preview_ply: Path) -> dict:
    """Spirula's splat → the job's preview + viewer cameras. Returns the receipt stored in meta["spirula"].

    Spirula writes an oriented model (+Z up, ground at z=0: SplatLab's viewer frame), so no rotation and no SH
    rotation. `up` in gauge.txt says what set it: the camera's IMU (measured, via --telemetry) or `ground`
    (a GUESS — in a shelf-lined aisle it levelled on a wall, 2026-09-29). A guessed up is published but flagged;
    a model that is not oriented at all is refused rather than shown at an arbitrary tilt.
    """
    ws = workdir(job_dir)
    sparse = ws / "data" / "sparse" / "0"
    g = read_gauge(sparse / "gauge.txt")
    if g.get("oriented") != "1":
        raise ValueError(f"Spirula model is not oriented (gauge.txt: {g!r}); refusing to publish a splat whose up "
                         "axis is unknown.")
    ply = latest_splat(job_dir)
    if ply is None:
        raise FileNotFoundError(f"no trained splat under {ws / 'train'}")
    count = _copy_with_up_comment(ply, preview_ply)

    images = read_images_bin(sparse / "images.bin")
    frames = [{"file_path": f"images/{im['name']}", "transform_matrix": c2w_gl(im["qvec"], im["tvec"])}
              for im in sorted(images, key=lambda i: i["name"])]
    cams = ws / "cameras"
    cams.mkdir(parents=True, exist_ok=True)
    (cams / "transforms.json").write_text(json.dumps(
        {"camera_model": "OPENCV_FISHEYE", "note": "Spirula SfM poses in the published splat's frame",
         "frames": frames}))
    # Identity: the cameras already live in the viewer frame (the /cameras route draws them solid).
    (cams / "dataparser_transforms.json").write_text(json.dumps(
        {"transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], "scale": 1.0}))
    zs = sorted(f["transform_matrix"][2][3] for f in frames)
    up_source = g.get("up", "unknown")
    return {"splats": count, "source_ply": str(ply), "cameras": len(frames),
            "camera_height_range": [round(zs[0], 4), round(zs[-1], 4)] if zs else None,
            "gauge": g, "up_source": up_source, "metric": g.get("metric") == "1",
            "orientation_warning": (None if up_source not in ("ground", "cameras", "unknown") else
                                    f"up was guessed ({up_source}), not measured by the camera's IMU — the scene may "
                                    "be tilted or on its side"),
            "binary": spirula_bin()}


def settings_dict(s: LaneSettings) -> dict:
    return asdict(s)


# --- Walk (owner 2026-09-30 "Yes build next!"): a first-person world with no nerfstudio checkpoint ---------
# The rig lane's world ladder (mesh -> inventory -> isolate -> ground -> solidify) needs a language field and
# a checkpoint. A walk needs neither: the walker LOOKS at the splat (setBackdrop) and COLLIDES with
# world_shell.py's voxel solid, which it builds from the splat PLY alone. Measured on the storage room
# (splat_e5b31df394): voxel route, PASS, 1 component, watertight, floor continuity 0.987, 42 s on CPU.
# The shell entry points at the collision solid too; the walker hides every shell while the photograph shows.
WALK_PLAYER_HEIGHT_M = 1.7
WALK_PLAYER_RADIUS_M = 0.32
WALK_OK_VERDICTS = ("PASS", "WALKABLE", "WALKABLE_NOT_WATERTIGHT")
# The walker downloads the visual shell AND the collision solid; a plain copy doubled a 298 MB download on the
# pool long take and it never loaded. The stand-in is hidden while the splat shows, so it only needs to be light.
WALK_VISUAL_MAX_FACES = 200_000
# The browser must build a BVH over the collision solid. 2.9M triangles (condo frontage, 37 x 40 m) loads at 60 fps;
# 16.6M (pool long take, 104 x 128 m at the 0.07 voxel) never finished loading in 240 s. Triangles scale ~1/voxel^2,
# so an oversized solid is rebuilt once at the voxel that lands near WALK_TARGET_COLLIDE_TRIS.
WALK_BASE_VOXEL = 0.07
WALK_MAX_COLLIDE_TRIS = 4_000_000
WALK_TARGET_COLLIDE_TRIS = 3_000_000


def is_spirula_job(meta: dict) -> bool:
    return meta.get("trainer_resolved") == "spirula" or bool((meta.get("spirula") or {}).get("published"))


def walk_seed_yup(job_dir: Path) -> list[float] | None:
    """The capture camera nearest the median camera position, mapped Z-up (x,y,z) -> Y-up (x, z, -y).

    The operator stood there, so it is free space with ground under it — a better flood-fill seed than the
    cloud-median probe in a narrow room. It must be a REAL camera: the component-wise median of a loop around a
    pool (pool long take, splat_aab7a913bc) sits over the water, and the walker dropped into the basin."""
    try:
        frames = json.loads((workdir(job_dir) / "cameras" / "transforms.json").read_text())["frames"]
        pos = [[float(f["transform_matrix"][r][3]) for r in range(3)] for f in frames]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    if not pos:
        return None
    med = [sorted(p[i] for p in pos)[len(pos) // 2] for i in range(3)]
    near = min(pos, key=lambda p: sum((p[i] - med[i]) ** 2 for i in range(3)))
    return [round(near[0], 4), round(near[2], 4), round(-near[1], 4)]


# world_shell.py ranks candidates BEFORE its crumb drop re-grades the winner, so the auto pick can land just under
# the floor gate (office, splat_b30d3965e8: voxel 0.9524 -> 0.9496 after the drop) while the splat-transform
# candidate scored 1.0. The walk retries that route once rather than refusing a walkable room.
WALK_FALLBACK_ROUTE = "splat-transform"


def walk_shell_command(python: str, script: Path, job_dir: Path, seed: list[float] | None,
                       route: str = "auto", voxel: float = WALK_BASE_VOXEL) -> list[str]:
    cmd = [python, str(script), str(job_dir), "--source", str(job_dir / "_preview" / "splat.ply"),
           "--player-height", str(WALK_PLAYER_HEIGHT_M), "--player-radius", str(WALK_PLAYER_RADIUS_M),
           "--route", route, "--voxel-size", f"{voxel:.4g}", "--json"]
    if seed:
        # `--seed=`: a seed starting with "-" would otherwise parse as an option (argparse exit 2).
        cmd.append("--seed=" + ",".join(f"{v:.6g}" for v in seed))
    return cmd


def walk_visual_shell_command(python: str, script: Path, job_dir: Path) -> list[str]:
    return [python, str(script), str(job_dir), "--max-faces", str(WALK_VISUAL_MAX_FACES)]


def walk_coarser_voxel(job_dir: Path, voxel: float = WALK_BASE_VOXEL) -> float | None:
    """The voxel to rebuild at when the collision solid is too heavy for the browser, else None."""
    try:
        tris = json.loads((job_dir / "_world" / "collision_shell.json").read_text())["artifact"]["triangles_written"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not tris or tris <= WALK_MAX_COLLIDE_TRIS:
        return None
    return round(voxel * math.sqrt(tris / WALK_TARGET_COLLIDE_TRIS), 3)


def walk_verdict(job_dir: Path) -> str | None:
    try:
        return json.loads((job_dir / "_world" / "collision_shell.json").read_text()).get("verdict")
    except (OSError, ValueError):
        return None


def write_walk_world(job_dir: Path, job_id: str, meters_per_unit: float | None) -> dict:
    """world.json + world_manifest.json + shell.glb around world_shell.py's output. Refuses a shell whose
    verdict is not walkable — a world you fall through is worse than no world."""
    world = job_dir / "_world"
    report = json.loads((world / "collision_shell.json").read_text())
    verdict = report.get("verdict")
    if verdict not in WALK_OK_VERDICTS:
        raise ValueError(f"collision shell verdict {verdict!r} is not walkable")
    if not (world / "navmesh.json").is_file():
        raise ValueError("world_shell.py wrote no navmesh.json")
    if not (world / "shell.glb").is_file():  # walk_visual_shell.py normally wrote a decimated stand-in
        shutil.copyfile(world / "collision_shell.glb", world / "shell.glb")
    probe = report.get("probe") or {}
    mpu = float(meters_per_unit) if meters_per_unit else None
    shell = {"built": True, "glb": "shell.glb", "source": "collision_shell", "texture": None,
             "faces": (report.get("artifact") or {}).get("triangles_written")}
    (world / "world.json").write_text(json.dumps({
        "v": 1, "job_id": job_id, "kind": "spirula-walk",
        # Metric only when the IMU measured it; otherwise the walker keeps its scene-unit dial.
        "units": "meters" if mpu else "scene-units",
        "meters_per_unit": 1.0 if mpu else None,
        "calibrated_from_meters_per_unit": mpu,
        "up_axis": "Y", "shell": shell, "elements": [],
        "note": "Raw 360 walk: look at the splat, collide with the voxel solid (no checkpoint, no elements)"},
        indent=2))
    (world / "world_manifest.json").write_text(json.dumps({
        "v": 1, "units": "meters" if mpu else "scene-units", "meters_per_unit": 1.0 if mpu else None,
        "shell": {"slug": "shell", "role": "static", "glb": "shell.glb"}, "elements": [],
        "counts": {"elements": 0}}, indent=2))
    return {"verdict": verdict, "gates": report.get("gates"), "route": report.get("route_used"),
            "floor_y": probe.get("floor_level_y"), "top_y": probe.get("top_level_y"),
            "seed_yup": (report.get("params") or {}).get("seed_yup"), "seconds": report.get("seconds")}
