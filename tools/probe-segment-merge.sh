#!/usr/bin/env bash
# Phase 3.1 probe: prove the segmented-pipeline join (Architecture A, SfM-level)
# before touching production code. Stitches a raw .insv once, extracts ALL
# frames into ONE shared database, then runs colmap4's global_mapper TWICE —
# bounded to two overlapping time windows via --GlobalMapper.image_list_path —
# and joins the two resulting models with model_merger -> bundle_adjuster,
# gating on the registration ratio + post-BA reprojection error.
#
# See ~/reports/splatlab-360-sample-segment-plan-2026-07-05/PLAN.md Phase 3.1
# and evidence/segment-merge-feasibility.md for the architecture rationale.
# feature_extractor/global_mapper invocations mirror what backend/splat_route.py's
# _glomap_sfm_command already runs in production (same tool paths, same crop
# fan-out, same overlap flag).
#
# ARCHITECTURE NOTE (learned the hard way, run 1 of this script): model_merger's
# alignment estimator (colmap/estimators/alignment.cc, ReconstructionAlignmentEstimator
# ::Estimate) hard-requires THROW_CHECK_EQ(src_images[i]->ImageId(), tgt_images[i]
# ->ImageId()) for every common-by-NAME image pair. Two segments built from
# INDEPENDENT `feature_extractor`/database.db runs get INDEPENDENTLY assigned
# numeric IDs for the same-named overlap frames, so that check throws
# std::invalid_argument even though FindCommonRegImageIds() correctly matched
# them by name first — this is colmap's actual "merge disconnected sub-models
# of ONE run" use case (doc/faq.rst:315), not "merge two independently-solved
# reconstructions". database_merger is NOT a workaround either: Database::Merge
# (scene/database.cc) explicitly THROW_CHECKs that the two DBs share NO image
# names, the opposite of what an overlap-based join needs.
# THE FIX: one SHARED database + image_path for feature_extractor and
# sequential_matcher (run once, over the union of frames), then TWO bounded
# global_mapper calls via --GlobalMapper.image_list_path restricting each to
# one segment's crop names (confirmed in source: PostParse() reads the list
# into GlobalMapperOptions.image_names, which global_pipeline.cc feeds straight
# into DatabaseCacheOptions.image_names — a real input-scope restriction, not a
# post-hoc filter). Since both segment models are loaded from ONE database, a
# common-name image gets the IDENTICAL numeric ID in both, so
# ReconstructionAlignmentEstimator's check holds and model_merger succeeds.
# This still gives each segment an independent, checkpointable reconstruction
# step (global_mapper is the expensive/crash-prone part — matching is cheap,
# see STATUS.md timing receipts) — the restart-survival property Phase 3.2
# wants is preserved, just implemented via list-restricted runs against a
# shared DB instead of physically-separate per-segment databases.
#
# Usage: tools/probe-segment-merge.sh [path/to/source.insv]
#   Defaults to the 90s dev clip (fast iteration loop) documented in STATUS.md.
#   Override segment windows via SEG1_START/SEG1_END/SEG2_START/SEG2_END env vars.
set -euo pipefail

SPLATLAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$SPLATLAB_ROOT/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
SRC_INSV="${1:-$HOME/transfers/splatlab/VID_20260514_064632_first90s.insv}"
OUT="$SPLATLAB_ROOT/tools/probe-segment-merge-output"

COLMAP4="$HOME/miniconda3/envs/colmap4/bin/colmap"
SPLATOPS_PY="$HOME/miniconda3/envs/splatops/bin/python"
FFMPEG="$(command -v ffmpeg || true)"
FFPROBE="$(command -v ffprobe || true)"

# Segment geometry (PLAN.md Phase 3.1 step 3): two 40s windows with a 10s
# overlap by default — override via env vars to retarget a different part of
# the source clip. Frame density: 3.0fps — the ONLY proven-good density for
# this pipeline (STATUS.md G3 attempt 1: 1.76fps -> 599 posed cameras but ZERO
# 3D points; 3.0fps, same window -> 1078/1080 registered, 105k points). SfM
# needs actual camera translation to triangulate — a correctly-dense window
# can still 0-point if the operator held still during it ("pick a different
# trim_start_s, don't crank frames"); this bit run 1 of this script (see
# STATUS.md Phase 3.1 attempt 1). images_per_equirect=8 matches the
# flight/full-run default.
SEG1_START="${SEG1_START:-0}"; SEG1_END="${SEG1_END:-40}"
SEG2_START="${SEG2_START:-30}"; SEG2_END="${SEG2_END:-70}"
FPS=3
IMAGES_PER_EQUIRECT=8
CROP_BOTTOM=0.15
INSV_FOV=204.0

# Acceptance gate (PLAN.md Phase 3.1 step 7).
MIN_REGISTRATION_RATIO=0.80
MAX_MEAN_REPROJ_PX=1.5

log() { echo "[probe-segment-merge] $*"; }
fail() { echo "[probe-segment-merge] FATAL: $*" >&2; exit 1; }

# ── 0. Preflight — fail loud before spending any GPU/CPU time ───────────────
[ -x "$COLMAP4" ] || fail "colmap4 not found at $COLMAP4 (never modify this env — read-only)"
[ -x "$SPLATOPS_PY" ] || fail "splatops python not found at $SPLATOPS_PY"
[ -n "$FFMPEG" ] || fail "ffmpeg not on PATH"
[ -n "$FFPROBE" ] || fail "ffprobe not on PATH"
[ -f "$SRC_INSV" ] || fail "source .insv not found: $SRC_INSV"
if pgrep -af 'ns-process|ns-train|colmap|glomap|run_langfield' | grep -v grep | grep -v probe-segment-merge >/dev/null 2>&1; then
  fail "a splat/colmap/glomap process is already running — this probe would contend for the single 5090. Check: pgrep -af 'ns-process|ns-train|colmap|glomap'"
fi
"$SPLATOPS_PY" -c "from nerfstudio.process_data import equirect_utils" \
  || fail "nerfstudio equirect_utils not importable in the splatops env"
log "preflight OK — source: $SRC_INSV"

rm -rf "$OUT"
mkdir -p "$OUT"/{stitched,all_frames,shared,seg1,seg2,merged}

# ── 1. Stitch once (auto-detect single vs dual fisheye stream, mirrors
#      backend/splat_route.py _stitch_command) ──────────────────────────────
STREAMS=$("$FFPROBE" -v error -select_streams v -show_entries stream=index -of csv=p=0 "$SRC_INSV" | wc -l)
log "detected $STREAMS video stream(s) in source"
CPU_COUNT=$(( $(nproc) / 2 )); [ "$CPU_COUNT" -ge 4 ] || CPU_COUNT=4
LEASH=(taskset -c "0-$((CPU_COUNT - 1))" nice -n 10)
V360="v360=input=dfisheye:output=e:ih_fov=${INSV_FOV}:iv_fov=${INSV_FOV}:w=5760:h=2880:interp=lanczos"
STITCHED="$OUT/stitched/equirect.mp4"
if [ "$STREAMS" -ge 2 ]; then
  log "stitching (dual-stream hstack)..."
  "${LEASH[@]}" "$FFMPEG" -y -i "$SRC_INSV" \
    -filter_complex "[0:v:0][0:v:1]hstack=inputs=2[d];[d]${V360}[eq]" -map "[eq]" \
    -c:v libx264 -crf 18 -an "$STITCHED"
else
  log "stitching (single-stream)..."
  "${LEASH[@]}" "$FFMPEG" -y -i "$SRC_INSV" -vf "$V360" -c:v libx264 -crf 18 -an "$STITCHED"
fi
STITCHED_DURATION=$("$FFPROBE" -v error -show_entries format=duration -of csv=p=0 "$STITCHED")
log "stitched: $STITCHED ($STITCHED_DURATION s)"
awk -v d="$STITCHED_DURATION" -v need="$SEG2_END" 'BEGIN { if (d+0 < need+0) { exit 1 } }' \
  || fail "stitched clip is only ${STITCHED_DURATION}s, need >= ${SEG2_END}s for the seg2 window"

# ── 2. Extract ALL frames ONCE at a fixed 3fps, named by SEQUENTIAL index.
#      fps=3 gives deterministic uniform timestamps (frame i -> t=(i-1)/3.0s),
#      so segment membership below is derivable from the filename alone —
#      no duplication needed since both segments now share ONE frame pool. ──
log "extracting all frames at ${FPS}fps..."
"${LEASH[@]}" "$FFMPEG" -y -i "$STITCHED" -vf "fps=${FPS}" -q:v 2 "$OUT/all_frames/frame_%08d.jpg"
TOTAL_FRAMES=$(ls "$OUT/all_frames" | wc -l)
log "extracted $TOTAL_FRAMES frames"

# ── 3. 8-crop fan-out ONCE on the full frame pool (reuses nerfstudio's own
#      equirect_utils — identical call to backend/splat_route.py
#      _glomap_sfm_command) into ONE shared images dir. ─────────────────────
log "fanning out equirect frames -> $IMAGES_PER_EQUIRECT crops/frame (shared)..."
"$SPLATOPS_PY" - "$OUT/all_frames" "$OUT/shared/images" "$IMAGES_PER_EQUIRECT" "$CROP_BOTTOM" <<'PYEOF'
import shutil, sys
from pathlib import Path
from nerfstudio.process_data import equirect_utils
src, dst, n_crops, crop_bottom = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
size = equirect_utils.compute_resolution_from_equirect(src, n_crops)
equirect_utils.generate_planar_projections_from_equirectangular(
    src, size, n_crops, crop_factor=(0.0, crop_bottom, 0.0, 0.0)
)
dst.mkdir(parents=True, exist_ok=True)
for p in sorted((src / "planar_projections").iterdir()):
    shutil.move(str(p), dst / p.name)
shutil.rmtree(src / "planar_projections", ignore_errors=True)
PYEOF
TOTAL_CROPS=$(ls "$OUT/shared/images" | wc -l)
log "$TOTAL_CROPS crops ready (shared)"

# ── 4. Per-segment CROP-NAME LISTS by timestamp (no file copying — both
#      segments read from the ONE shared image dir/database; the [30,40)s-
#      style overlap window just means the SAME crop names appear in BOTH
#      lists, which is exactly what --GlobalMapper.image_list_path needs). ──
"$SPLATOPS_PY" - "$OUT/all_frames" "$OUT/shared/images" "$OUT/seg1/image_list.txt" "$OUT/seg2/image_list.txt" \
  "$FPS" "$IMAGES_PER_EQUIRECT" "$SEG1_START" "$SEG1_END" "$SEG2_START" "$SEG2_END" <<'PYEOF'
import sys
from pathlib import Path
frames_dir, images_dir, seg1_list, seg2_list, fps, n_crops, s1a, s1b, s2a, s2b = sys.argv[1:]
frames_dir, images_dir = Path(frames_dir), Path(images_dir)
fps, n_crops = float(fps), int(n_crops)
s1a, s1b, s2a, s2b = float(s1a), float(s1b), float(s2a), float(s2b)

crop_names = {p.name for p in images_dir.iterdir()}
seg1_names, seg2_names = [], []
n1_frames = n2_frames = overlap_frames = 0
for p in sorted(frames_dir.iterdir()):
    idx = int(p.stem.split("_")[-1])  # frame_00000042.jpg -> 42 (1-based)
    t = (idx - 1) / fps
    in1, in2 = s1a <= t < s1b, s2a <= t < s2b
    if not (in1 or in2):
        continue
    for k in range(n_crops):
        crop = f"{p.stem}_{k}.jpg"
        assert crop in crop_names, f"expected crop {crop} missing from fan-out output"
        if in1:
            seg1_names.append(crop)
        if in2:
            seg2_names.append(crop)
    n1_frames += in1
    n2_frames += in2
    overlap_frames += in1 and in2

Path(seg1_list).write_text("\n".join(seg1_names) + "\n")
Path(seg2_list).write_text("\n".join(seg2_names) + "\n")
print(f"seg1={n1_frames} frames ({len(seg1_names)} crops), "
      f"seg2={n2_frames} frames ({len(seg2_names)} crops), "
      f"overlap={overlap_frames} frames (shared crop names, present in both lists)")
assert overlap_frames > 0, "no overlap frames were produced — segment windows must intersect"
PYEOF

# ── 5. ONE feature_extractor + ONE sequential_matcher over the shared DB —
#      matching is the cheap step (STATUS.md receipts: ~4-5min vs
#      global_mapper's ~10min); no benefit to bounding it per segment, and
#      doing it once avoids re-processing the physically-duplicated overlap
#      crops the old (broken) per-segment-database design required. ────────
SHARED_DB="$OUT/shared/database.db"
SHARED_IMG="$OUT/shared/images"
log "feature_extractor (shared)..."
"$COLMAP4" feature_extractor \
  --database_path "$SHARED_DB" --image_path "$SHARED_IMG" \
  --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1

log "sequential_matcher (shared, loop_detection on, overlap=$((2 * IMAGES_PER_EQUIRECT)))..."
"$COLMAP4" sequential_matcher \
  --database_path "$SHARED_DB" \
  --SequentialMatching.overlap "$((2 * IMAGES_PER_EQUIRECT))" \
  --SequentialMatching.loop_detection 1 --FeatureMatching.use_gpu 1

# ── 6. Per segment: BOUNDED global_mapper via --GlobalMapper.image_list_path
#      against the ONE shared database — this is the expensive, crash-prone
#      step (STATUS.md receipts: ~10min), and it's what actually needs to be
#      independently checkpointable for Phase 3.2's restart-survival story.
#      Sequential, not parallel: one 5090, avoid contending with itself. ────
run_segment() {
  local seg_dir="$1"
  local image_list="$seg_dir/image_list.txt"
  local sparse="$seg_dir/sparse"
  mkdir -p "$sparse"

  local n_crops
  n_crops=$(wc -l < "$image_list")
  log "[$seg_dir] global_mapper (bounded to $n_crops crops via image_list_path)..."
  "$COLMAP4" global_mapper \
    --database_path "$SHARED_DB" --image_path "$SHARED_IMG" --output_path "$sparse" \
    --GlobalMapper.image_list_path "$image_list"

  [ -f "$sparse/0/cameras.bin" ] || fail "[$seg_dir] global_mapper produced no model (cameras.bin missing)"
  local pts_bytes
  pts_bytes=$(stat -c%s "$sparse/0/points3D.bin" 2>/dev/null || echo 0)
  [ "$pts_bytes" -gt 8 ] || fail "[$seg_dir] global_mapper posed cameras but triangulated ZERO 3D points (points3D.bin=${pts_bytes}B) — same failure class as splat_9da9dff4b2, see STATUS.md G3 attempt 1"
  log "[$seg_dir] SfM OK — $(stat -c%s "$sparse/0/cameras.bin")B cameras.bin, ${pts_bytes}B points3D.bin"
}

run_segment "$OUT/seg1"
run_segment "$OUT/seg2"

# ── 7. Join: model_merger (pairwise) -> point_filtering -> bundle_adjuster.
#      Both segment models were loaded from the SAME shared database, so a
#      common-by-name image now carries the IDENTICAL numeric ImageId in both
#      — satisfying ReconstructionAlignmentEstimator's equality check (see the
#      header note; this is the actual fix for run 1's crash). ─────────────
#      model_merger's default --max_reproj_error (64px!) is a loose RANSAC
#      inlier threshold for the alignment sim3, not a point-quality filter —
#      run 2 of the fixed script merged clean (ratio 1.000) but then
#      bundle_adjuster diverged to NO_CONVERGENCE with an astronomical mean
#      reprojection error (one bad correspondence admitted at the loose
#      threshold, left unconstrained enough to run away during BA). Fix
#      (validated by hand against these exact seg1/seg2 models before landing
#      here): tighten the merge threshold to 8px, then run colmap's own
#      point_filtering (defaults: max_reproj_error=4, min_track_len=2) to
#      strip bad observations before bundle_adjuster ever sees them. Result:
#      pre-BA 2.67px -> 1.32px (tight merge alone) -> 0.99px (+ filtering) ->
#      0.89px stable (post-BA), registration ratio unchanged at 1.000.
log "model_merger (seg1 + seg2 -> merged, max_reproj_error=8)..."
mkdir -p "$OUT/merged/model_merger_out"
"$COLMAP4" model_merger \
  --input_path1 "$OUT/seg1/sparse/0" --input_path2 "$OUT/seg2/sparse/0" \
  --output_path "$OUT/merged/model_merger_out" --max_reproj_error 8

log "point_filtering on the merged model (strip bad observations before BA)..."
mkdir -p "$OUT/merged/filtered"
"$COLMAP4" point_filtering \
  --input_path "$OUT/merged/model_merger_out" --output_path "$OUT/merged/filtered" \
  --max_reproj_error 4 --min_track_len 2

log "bundle_adjuster on the filtered merged model..."
mkdir -p "$OUT/merged/ba_out"
"$COLMAP4" bundle_adjuster \
  --input_path "$OUT/merged/filtered" --output_path "$OUT/merged/ba_out"

# ── 8. Receipts: per-segment + merged registered counts (via model_analyzer),
#      registration ratio vs the UNION of per-segment registered image NAMES
#      (not a naive sum — the shared overlap frames must count once), and the
#      post-BA mean reprojection error. TXT export only to read image names;
#      the BIN models (what nerfstudio/ns-process-data would actually consume)
#      are untouched. ────────────────────────────────────────────────────────
analyze() { "$COLMAP4" model_analyzer --path "$1" 2>&1; }

log "=== seg1 model_analyzer ==="
SEG1_STATS=$(analyze "$OUT/seg1/sparse/0"); echo "$SEG1_STATS"
log "=== seg2 model_analyzer ==="
SEG2_STATS=$(analyze "$OUT/seg2/sparse/0"); echo "$SEG2_STATS"
log "=== merged (pre-filter) model_analyzer ==="
analyze "$OUT/merged/model_merger_out"
log "=== merged (post-filter, pre-BA) model_analyzer ==="
analyze "$OUT/merged/filtered"
log "=== merged (post-BA) model_analyzer ==="
export MERGED_STATS
MERGED_STATS=$(analyze "$OUT/merged/ba_out"); echo "$MERGED_STATS"

mkdir -p "$OUT/merged/ba_out_txt" "$OUT/seg1/sparse_txt" "$OUT/seg2/sparse_txt"
"$COLMAP4" model_converter --input_path "$OUT/seg1/sparse/0" --output_path "$OUT/seg1/sparse_txt" --output_type TXT >/dev/null
"$COLMAP4" model_converter --input_path "$OUT/seg2/sparse/0" --output_path "$OUT/seg2/sparse_txt" --output_type TXT >/dev/null
"$COLMAP4" model_converter --input_path "$OUT/merged/ba_out" --output_path "$OUT/merged/ba_out_txt" --output_type TXT >/dev/null

log "=== gate ==="
set +e  # capture the gate's real exit code below instead of letting errexit kill the script on FAIL
"$SPLATOPS_PY" - "$OUT/seg1/sparse_txt/images.txt" "$OUT/seg2/sparse_txt/images.txt" \
  "$OUT/merged/ba_out_txt/images.txt" "$MIN_REGISTRATION_RATIO" "$MAX_MEAN_REPROJ_PX" <<'PYEOF'
import re, sys
from pathlib import Path

def registered_names(images_txt):
    # Image-info lines have exactly 10 whitespace fields (IMAGE_ID QW QX QY QZ
    # TX TY TZ CAMERA_ID NAME); the following POINTS2D line is always a
    # multiple of 3 tokens (X Y POINT3D_ID triplets) and can never total 10 —
    # classifying by field count is robust even if a zero-observation image
    # writes a blank POINTS2D line (no assumption of strict even/odd pairing).
    names = set()
    for line in Path(images_txt).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 10:
            names.add(fields[-1])
    return names

seg1_p, seg2_p, merged_p, min_ratio, max_reproj = sys.argv[1:]
seg1 = registered_names(seg1_p)
seg2 = registered_names(seg2_p)
merged = registered_names(merged_p)
union = seg1 | seg2
ratio = len(merged) / len(union) if union else 0.0

# Mean reprojection error from the merged model_analyzer stdout, captured by
# the caller into $MERGED_STATS and passed via env (avoids re-shelling out).
import os
merged_stats = os.environ.get("MERGED_STATS", "")
m = re.search(r"Mean reprojection error:\s*([0-9.]+)px", merged_stats)
mean_reproj = float(m.group(1)) if m else None

print(f"seg1 registered: {len(seg1)}")
print(f"seg2 registered: {len(seg2)}")
print(f"union (distinct names, seg1|seg2): {len(union)}")
print(f"merged (post-BA) registered: {len(merged)}")
print(f"registration ratio (merged/union): {ratio:.3f} (gate: >= {float(min_ratio):.2f})")
print(f"mean reprojection error: {mean_reproj}px (gate: <= {float(max_reproj):.2f}px)" if mean_reproj is not None
      else "mean reprojection error: COULD NOT PARSE from model_analyzer output")

ok_ratio = ratio >= float(min_ratio)
ok_reproj = mean_reproj is not None and mean_reproj <= float(max_reproj)
print(f"\nGATE: {'PASS' if (ok_ratio and ok_reproj) else 'FAIL'} "
      f"(ratio {'OK' if ok_ratio else 'FAIL'}, reproj {'OK' if ok_reproj else 'FAIL'})")
sys.exit(0 if (ok_ratio and ok_reproj) else 1)
PYEOF
GATE_RC=$?
set -e

log "probe complete — output under $OUT"
exit "$GATE_RC"
