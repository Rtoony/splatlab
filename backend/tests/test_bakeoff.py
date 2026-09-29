"""Trainer bake-off helpers: the split mirror must hold out exactly the named
photos, the scorer must tie every render to its photo (and refuse anything
partial or ambiguous), and the frame move must carry whole gaussians — centre,
size and orientation — through the known dataparser similarity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bakeoff import ply_frame, scoring, split  # noqa: E402
from isolate.splat_ply import write_splat_ply  # noqa: E402

RNG = np.random.default_rng(3)


# ---------- split mirror ----------

def _dataset(tmp: Path, names: list[str]) -> Path:
    proc = tmp / "processed"
    frames = []
    for n in names:
        p = proc / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpg:" + n.encode())
        frames.append({"file_path": n, "transform_matrix": np.eye(4).tolist()})
    (proc / "sparse_pc.ply").write_bytes(b"ply")
    (proc / "transforms.json").write_text(json.dumps({"camera_model": "OPENCV", "ply_file_path": "sparse_pc.ply",
                                                      "frames": frames}))
    return proc


def test_mirror_tags_every_frame_and_links_not_copies(tmp_path):
    names = [f"images/IMG_{i:03d}.JPG" for i in range(10)]
    proc = _dataset(tmp_path, names)
    receipt = split.build_mirror(proc, names[1::4], tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "transforms.json").read_text())
    tagged = [Path(f["file_path"]).name for f in meta["frames"]]
    assert sorted(t for t in tagged if t.startswith("eval_")) == ["eval_IMG_001.JPG", "eval_IMG_005.JPG",
                                                                    "eval_IMG_009.JPG"]
    assert sum(t.startswith("train_") for t in tagged) == 7
    assert (receipt["train"], receipt["eval"]) == (7, 3)
    link = tmp_path / "m" / "images" / "eval_IMG_005.JPG"
    assert link.is_symlink() and link.read_bytes() == b"jpg:images/IMG_005.JPG"
    assert (tmp_path / "m" / "sparse_pc.ply").is_symlink()
    assert json.loads((proc / "transforms.json").read_text())["frames"][0]["file_path"] == names[0]  # source untouched


def test_mirror_flattens_subfolders_without_collisions(tmp_path):
    names = ["images/pano_camera0/f_0001.jpg", "images/pano_camera1/f_0001.jpg"]
    proc = _dataset(tmp_path, names)
    split.build_mirror(proc, [names[1]], tmp_path / "m")
    got = sorted(p.name for p in (tmp_path / "m" / "images").iterdir())
    assert got == ["eval_pano_camera1__f_0001.jpg", "train_pano_camera0__f_0001.jpg"]


@pytest.mark.parametrize("bad", ["images/retrain_01.jpg", "images/Evaluation.jpg"])
def test_mirror_refuses_names_filename_mode_would_misread(tmp_path, bad):
    proc = _dataset(tmp_path, ["images/a.jpg", bad])
    with pytest.raises(ValueError, match="misread"):
        split.build_mirror(proc, ["images/a.jpg"], tmp_path / "m")


def test_mirror_refuses_unknown_eval_name(tmp_path):
    proc = _dataset(tmp_path, ["images/a.jpg"])
    with pytest.raises(ValueError, match="not frames"):
        split.build_mirror(proc, ["images/zzz.jpg"], tmp_path / "m")


# ---------- scorer matching ----------

def _photos(n=4, h=40, w=60):
    return {f"images/p{i}.jpg": RNG.random((h, w, 3), dtype=np.float32) for i in range(n)}


def _fake_metrics(gt, pred):
    return {"psnr": scoring.psnr_np(gt, pred)}


def test_scorer_ties_shuffled_renders_to_their_photos():
    photos = _photos()
    names = list(photos)
    order = [2, 0, 3, 1]
    pairs = []
    for k, i in enumerate(order):
        gt = np.clip(photos[names[i]] + 1 / 255, 0, 1)            # decoder-level drift is tolerated
        pred = np.clip(photos[names[i]] + 0.01 * (i + 1), 0, 1)   # view i gets its own error
        pairs.append(scoring.Pair(f"k{k}", gt, pred))
    res = scoring.score(pairs, photos, _fake_metrics)
    assert res["views"] == 4
    by_photo = {r["photo"]: r for r in res["per_view"]}
    for i, name in enumerate(names):
        assert by_photo[name]["arm_key"] == f"k{order.index(i)}"
    assert by_photo[names[0]]["psnr"] > by_photo[names[3]]["psnr"]


def test_scorer_refuses_partial_or_foreign_or_duplicate_renders():
    photos = _photos()
    names = list(photos)
    ok = [scoring.Pair(n, photos[n], photos[n]) for n in names]
    with pytest.raises(ValueError, match="no render"):
        scoring.score(ok[:3], photos, _fake_metrics)
    foreign = RNG.random(photos[names[0]].shape, dtype=np.float32)
    with pytest.raises(ValueError, match="not one of the held-out"):
        scoring.score(ok[:3] + [scoring.Pair("x", foreign, foreign)], photos, _fake_metrics)
    with pytest.raises(ValueError, match="both match"):
        scoring.score(ok + [scoring.Pair("dup", photos[names[0]], photos[names[0]])], photos, _fake_metrics)
    small = scoring.Pair("s", photos[names[0]][:20], photos[names[0]][:20])
    with pytest.raises(ValueError, match="size"):
        scoring.score(ok[1:] + [small], photos, _fake_metrics)


def test_psnr_known_noise():
    a = np.full((64, 64, 3), 0.5, np.float32)
    assert scoring.psnr_np(a, a) == float("inf")
    assert scoring.psnr_np(a, a + 0.1) == pytest.approx(20.0, abs=1e-4)   # mse 0.01 → 20 dB


def test_layout_readers(tmp_path):
    from PIL import Image
    gt = (RNG.random((8, 10, 3)) * 255).astype(np.uint8)
    pr = (RNG.random((8, 10, 3)) * 255).astype(np.uint8)
    Image.fromarray(np.concatenate([gt, pr], axis=1)).save(tmp_path / "eval_img_0000.png")
    Image.fromarray(gt).save(tmp_path / "eval-gt-00000.png")
    Image.fromarray(pr).save(tmp_path / "eval-render-00000.png")
    (ns,) = scoring.pairs_nerfstudio(tmp_path)
    (sp,) = scoring.pairs_spirula(tmp_path)
    for p in (ns, sp):
        assert np.array_equal((p.gt * 255).round().astype(np.uint8), gt)
        assert np.array_equal((p.pred * 255).round().astype(np.uint8), pr)


# ---------- frame move ----------

def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def _cov(q, log_s):
    w, x, y, z = q / np.linalg.norm(q)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    s = np.diag(np.exp(log_s))
    return r @ s @ s @ r.T


def test_quat_from_matrix_round_trips():
    for axis, deg in (([0, 0, 1], 30), ([1, 0, 0], 179), ([1, 1, 0], -120), ([0.3, -1, 2], 200)):
        r = _rot(axis, deg)
        q = ply_frame.quat_from_matrix(r)
        assert np.allclose(_cov(q, np.zeros(3)), np.eye(3), atol=1e-9)
        assert np.allclose(_cov(q, np.log([2.0, 1.0, 1.0])), r @ np.diag([4.0, 1, 1]) @ r.T, atol=1e-9)


def test_to_ns_frame_moves_whole_gaussians(tmp_path):
    r, t, s = _rot([0.2, 1, -0.4], 73), np.array([0.5, -2.0, 1.25]), 0.2211
    (tmp_path / "dp.json").write_text(json.dumps({"transform": np.hstack([r, t[:, None]]).tolist(), "scale": s}))
    n = 50
    xyz = RNG.normal(size=(n, 3))
    q = RNG.normal(size=(n, 4))
    log_s = RNG.normal(size=(n, 3)) - 3
    ply = tmp_path / "in.ply"
    write_splat_ply(ply, xyz, RNG.random((n, 3)), RNG.random(n), log_s, q)
    cols = ply_frame.read_vertex_ply(ply)
    R, T, S = ply_frame.load_dataparser_transform(tmp_path / "dp.json")
    xyz2, _, _, log_s2, q2 = ply_frame.to_ns_frame(cols, R, T, S)
    assert np.allclose(xyz2, s * (xyz @ r.T + t), atol=1e-5)
    for i in range(n):  # covariance must transform as (sR) Σ (sR)^T
        want = (s * r) @ _cov(q[i], log_s[i]) @ (s * r).T
        assert np.allclose(_cov(q2[i], log_s2[i]), want, rtol=1e-4, atol=1e-10)


def test_dataparser_transform_must_be_a_rotation(tmp_path):
    (tmp_path / "dp.json").write_text(json.dumps({"transform": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                                                  "scale": 1.0}))
    with pytest.raises(ValueError, match="proper rotation"):
        ply_frame.load_dataparser_transform(tmp_path / "dp.json")
