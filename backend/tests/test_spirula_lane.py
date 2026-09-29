"""Spirula 360 lane: settings rule, command guards, plan routing (auto / explicit / kill-switch /
fallback), the SfM registration check, and publish (up-axis refusal, PLY header, camera poses).
CPU-only: no binary runs; probes and tools are monkeypatched."""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spirula_lane as sl  # noqa: E402
import splat_route  # noqa: E402

DUAL_4K = {"video_streams": 2, "width": 3840, "height": 3840, "fps": 29.97, "duration": 123.09, "frames": 3689,
           "same_size": True}


# ---------------------------------------------------------------- settings rule

def test_settings_reproduce_the_proven_storage_room_run():
    s = sl.settings(DUAL_4K)
    assert (s.stride, s.instants, s.scale, s.cache) == (9, 410, 1.0, "cpu")    # the 09-29 P3full run
    assert s.focal == pytest.approx(1036.8, abs=0.1) and s.quality == "high"


def test_long_clips_trade_resolution_for_ram_never_density():
    walk = sl.settings({**DUAL_4K, "duration": 277.4, "frames": 8313})
    assert walk.stride == 9 and walk.scale == 0.5 and walk.cache == "cpu"
    assert walk.instants * 2 * walk.width ** 2 * 3 <= sl.CPU_CACHE_BUDGET_BYTES
    pool = sl.settings({**DUAL_4K, "duration": 433.9, "frames": 13003})
    assert pool.instants <= sl.MAX_INSTANTS + 60 and pool.scale == 0.5


def test_small_lenses_and_short_clips():
    small = sl.settings({**DUAL_4K, "width": 1920, "height": 1920, "duration": 66.9, "frames": 2004})
    assert small.scale == 1.0 and small.focal == pytest.approx(518.4, abs=0.1)
    tiny = sl.settings({**DUAL_4K, "duration": 1.2, "frames": 36})
    assert tiny.stride >= sl.MIN_STRIDE and tiny.instants == math.ceil(36 / tiny.stride) == 4
    trimmed = sl.settings(DUAL_4K, trim_duration_s=20)
    assert trimmed.window_s == 20 and trimmed.instants == math.ceil(round(20 * 29.97) / trimmed.stride)


def test_quality_env(monkeypatch):
    monkeypatch.setenv("SPLAT_SPIRULA_QUALITY", "ultra")
    assert sl.settings(DUAL_4K).quality == "ultra"
    monkeypatch.setenv("SPLAT_SPIRULA_QUALITY", "academic-baseline")      # never a preset, never a typo
    assert sl.settings(DUAL_4K).quality == "high"


# ---------------------------------------------------------------- commands

def test_commands_carry_the_measured_guards(tmp_path):
    cmds = sl.commands("/bin/spirula", Path("/in/VID_x.insv"), tmp_path, sl.settings(DUAL_4K))
    assert list(cmds) == ["spirula_extract", "spirula_mask", "spirula_sfm", "spirula_train"]
    for c in cmds.values():
        assert c[:2] == ["env", f"VK_ICD_FILENAMES={sl.NVIDIA_ICD}"]           # never the Intel iGPU
    images = str(tmp_path / "_spirula" / "data" / "images")
    assert cmds["spirula_sfm"][3:6] == ["sfm", "auto", images]                                     # IMAGE dir, not the dataset dir
    assert cmds["spirula_extract"][-1] == images and "--scale" not in cmds["spirula_extract"]
    train = cmds["spirula_train"]
    assert train[3:5] == ["train", "360-camera"]
    assert "academic-baseline" not in train and "--resume" not in train
    assert train[train.index("--eval-mode") + 1] == "all" and "--save-eval-images" not in train
    assert train[train.index("--quality") + 1] == "high"
    assert train[train.index("--floater-suppression") + 1] == "mild"


def test_trim_prepends_a_stream_copy(tmp_path):
    cmds = sl.commands("/bin/spirula", Path("/in/v.insv"), tmp_path, sl.settings(DUAL_4K, 20),
                       ffmpeg="/bin/ffmpeg", trim_start_s=40.0, trim_duration_s=20.0)
    assert list(cmds)[0] == "spirula_trim"
    assert "-ss 40 -t 20" in cmds["spirula_trim"][2] and "-map 0:v -c copy" in cmds["spirula_trim"][2]
    assert cmds["spirula_extract"][5] == str(tmp_path / "_spirula" / "trimmed.insv")


# ---------------------------------------------------------------- plan routing

AVAIL = {"spirula_available": True, "spirula_path": "/bin/spirula", "ffmpeg_path": "/bin/ffmpeg",
         "ns_train_available": True, "ns_train_path": "/bin/ns-train", "ns_process_data_available": True,
         "ns_process_data_path": "/bin/ns-process-data", "ns_export_available": True, "ns_export_path": "/bin/ns-export",
         "colmap_available": True, "colmap_path": "/bin/colmap", "glomap_available": True, "glomap_path": "/bin/colmap4",
         "rig_available": True, "mast3r_available": False, "ffmpeg_available": True, "insv_stitch_available": True,
         "triposplat_available": False, "langfield_available": False}


@pytest.fixture
def plan(monkeypatch):
    monkeypatch.setattr(splat_route, "_splat_transform_path", lambda: "/bin/splat-transform")
    monkeypatch.setattr(splat_route, "_health_available", lambda: False)
    monkeypatch.setattr(splat_route, "_eval_available", lambda: False)
    monkeypatch.setattr(splat_route, "_tool_path", lambda binary, env: f"/bin/{binary}")
    monkeypatch.setattr(splat_route, "_probe_video_streams",
                        lambda f, p: {"streams": 2, "width": 3840, "height": 3840, "dims": [(3840, 3840)] * 2})
    monkeypatch.setattr(splat_route, "_probe_video_duration", lambda f, p: 123.09)
    monkeypatch.setattr(sl, "probe", lambda ffprobe, path: dict(DUAL_4K))

    def run(input_path="/in/VID_x.insv", avail=AVAIL, **kw):
        req = splat_route.SplatTrainRequest(mode="3d", input_path=input_path, **kw)
        return splat_route._plan_3d_job(req, avail, Path("/jobs/splat_t"), Path(input_path))
    return run


def test_auto_routes_dual_fisheye_insv_to_spirula(plan):
    stages, cmds, ctx = plan()
    assert stages == ["spirula_extract", "spirula_mask", "spirula_sfm", "spirula_train", "spirula_publish",
                      "compress", "webopt"]
    assert ctx is None                                  # never escalation-eligible (no COLMAP reroutes)
    meta = splat_route._new_meta("splat_t", splat_route.SplatTrainRequest(mode="3d", input_path="/in/VID_x.insv"),
                                 Path("/in/VID_x.insv"), Path("/jobs/splat_t"), stages, ctx)
    assert meta["trainer_resolved"] == "spirula" and meta["spirula"]["stride"] == 9
    assert meta["trainer"] == "auto"


def test_fallbacks_keep_the_rig_lane(plan, monkeypatch):
    rig = lambda stages: stages[0] == "stitch"  # noqa: E731
    assert rig(plan(trainer="splatfacto")[0])
    assert rig(plan(avail={**AVAIL, "spirula_available": False})[0])
    monkeypatch.setenv("SPLAT_SPIRULA_LANE", "0")
    assert rig(plan()[0])
    monkeypatch.delenv("SPLAT_SPIRULA_LANE")
    monkeypatch.setattr(sl, "probe", lambda ffprobe, path: {**DUAL_4K, "video_streams": 1})
    assert rig(plan()[0])                               # single-stream .insv: rig lane, silently


def test_explicit_spirula_that_cannot_run_is_a_400(plan, monkeypatch):
    with pytest.raises(HTTPException) as e:
        plan(input_path="/in/clip.mp4", trainer="spirula")
    assert e.value.status_code == 400
    with pytest.raises(HTTPException):
        plan(trainer="spirula", avail={**AVAIL, "spirula_available": False})
    monkeypatch.setattr(sl, "probe", lambda ffprobe, path: {**DUAL_4K, "video_streams": 1})
    with pytest.raises(HTTPException, match="dual-fisheye"):
        plan(trainer="spirula")


def test_trim_window_is_centred_and_planned(plan):
    stages, cmds, _ = plan(trim_duration_s=20.0)
    assert stages[0] == "spirula_trim"
    start = (123.09 - 20) / 2
    assert f"-ss {start:g} -t 20" in cmds["spirula_trim"][2]


def test_non_insv_is_untouched(plan):
    assert plan(input_path="/in/clip.mp4")[0][:1] == ["process"]


# ---------------------------------------------------------------- COLMAP readers, sfm check, publish

def _write_images_bin(path: Path, images):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(images)))
        for iid, q, t, name in images:
            fh.write(struct.pack("<I7dI", iid, *q, *t, 1) + name.encode() + b"\x00")
            fh.write(struct.pack("<Q", 2) + struct.pack("<ddq", 1.0, 2.0, -1) * 2)


def _quat(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = math.radians(deg) / 2
    return (math.cos(h), *(math.sin(h) * axis))


def _job(tmp_path, n=10, registered=10, gauge="oriented 1\nmetric 0\nup ground\nscale none\n"):
    ws = tmp_path / "_spirula"
    for cam in ("cam0", "cam1"):
        (ws / "data" / "images" / cam).mkdir(parents=True)
        for i in range(n // 2):
            (ws / "data" / "images" / cam / f"{i * 9:05d}.jpg").write_bytes(b"x")
    sparse = ws / "data" / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "gauge.txt").write_text("# What this model's frame means\n" + gauge)
    ims = []
    rng = np.random.default_rng(0)
    for i in range(registered):
        q = _quat(rng.normal(size=3), rng.uniform(0, 180))
        centre = np.array([0.1 * i, 0.2, 0.5 + 0.01 * i])
        r = np.array(sl._rotmat(q))
        ims.append((i + 1, q, tuple(-r @ centre), f"cam{i % 2}/{(i // 2) * 9:05d}.jpg"))
    _write_images_bin(sparse / "images.bin", ims)
    ckpt = ws / "train" / "step-000050000.ckpt"
    ckpt.mkdir(parents=True)
    body = np.arange(3 * 17, dtype="<f4").tobytes()
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
              + "".join(f"property float p{k}\n" for k in range(17)) + "end_header\n")
    (ckpt / "splat.ply").write_bytes(header.encode() + body)
    return ims, body


def test_images_bin_round_trip_and_c2w(tmp_path):
    ims, _ = _job(tmp_path)
    got = sl.read_images_bin(tmp_path / "_spirula" / "data" / "sparse" / "0" / "images.bin")
    assert [g["name"] for g in got] == [i[3] for i in ims]
    for g, (_, q, t, _) in zip(got, ims):
        m = np.array(sl.c2w_gl(g["qvec"], g["tvec"]))
        r = np.array(sl._rotmat(q))
        assert np.allclose(m[:3, 3], -r.T @ np.array(t))                        # camera centre
        assert np.allclose(m[:3, 2], -(r.T @ [0, 0, 1]))                         # GL z = -forward


def test_sfm_check_threshold(tmp_path):
    _job(tmp_path, n=10, registered=10)
    assert sl.sfm_check(tmp_path)[0]
    other = tmp_path / "b"
    _job(other, n=10, registered=7)
    ok, msg = sl.sfm_check(other)
    assert not ok and "7/10" in msg


def test_publish_writes_preview_and_viewer_cameras(tmp_path):
    ims, body = _job(tmp_path)
    preview = tmp_path / "_preview" / "splat.ply"
    receipt = sl.publish(tmp_path, preview)
    raw = preview.read_bytes()
    head, _, data = raw.partition(b"end_header\n")
    assert b"comment Vertical Axis: z" in head and data == body and receipt["splats"] == 3
    lines = head.decode().splitlines()
    assert lines[:3] == ["ply", "format binary_little_endian 1.0", "comment Vertical Axis: z"]
    tf = json.loads((tmp_path / "_spirula" / "cameras" / "transforms.json").read_text())
    dp = json.loads((tmp_path / "_spirula" / "cameras" / "dataparser_transforms.json").read_text())
    assert len(tf["frames"]) == len(ims) == receipt["cameras"] and dp["scale"] == 1.0
    assert tf["frames"][0]["file_path"].startswith("images/cam")


def test_publish_refuses_an_unlevelled_model(tmp_path):
    _job(tmp_path, gauge="oriented 0\nmetric 0\nup none\n")
    with pytest.raises(ValueError, match="up axis is unknown"):
        sl.publish(tmp_path, tmp_path / "_preview" / "splat.ply")
    assert not (tmp_path / "_preview" / "splat.ply").exists()


def test_default_output_root_survives_a_symlinked_outputs_dir(tmp_path, monkeypatch):
    """Regression: outputs/ -> /mnt/storage/... (2026-09-07) made /train refuse every job."""
    real = tmp_path / "raid" / "outputs"
    (real / "3d").mkdir(parents=True)
    link = tmp_path / "splatcli" / "outputs"
    link.parent.mkdir()
    link.symlink_to(real)
    monkeypatch.setattr(splat_route, "DEFAULT_3D_ROOT", link / "3d")
    assert splat_route._is_default_3d_root((link / "3d").resolve())
    assert not splat_route._is_default_3d_root((tmp_path / "elsewhere").resolve())
