import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from health import spark_agreement as sa


def test_psnr_identity_and_known_value():
    a = np.zeros((4, 4, 3)); b = np.full((4, 4, 3), 0.1)
    assert sa.psnr(a, a) == 100.0
    assert abs(sa.psnr(a, b) - 20.0) < 1e-9          # mse 0.01 -> 20 dB


def test_camera_to_three_matches_the_viewer_contract():
    c2w = np.eye(4)[:3]                               # camera at origin, OpenGL axes
    row = {"c2w": c2w.tolist(), "w": 640, "h": 480, "fy": 240.0, "file": "x.png"}
    cam = sa.camera_to_three(row)
    assert cam["position"] == [0.0, 0.0, 0.0] and cam["up"] == [0.0, 1.0, 0.0] and cam["forward"] == [0.0, 0.0, -1.0]
    assert abs(cam["fov_y_degrees"] - 90.0) < 1e-9 and abs(cam["aspect"] - 4 / 3) < 1e-9
    rot = np.array([[0, 0, -1, 5.0], [0, 1, 0, 6.0], [1, 0, 0, 7.0]], dtype=float)   # column 2 = (-1,0,0) -> forward +x
    cam = sa.camera_to_three({"c2w": rot.tolist(), "w": 10, "h": 10, "fy": 5.0, "file": "y"})
    assert cam["position"] == [5.0, 6.0, 7.0] and np.allclose(cam["forward"], [1.0, 0.0, 0.0])


def _write(p: Path, val: float, size=(8, 6)):
    Image.fromarray(np.full((size[1], size[0], 3), int(val * 255), dtype=np.uint8)).save(p)


def _arm(tmp: Path, name: str, spark_val: float, ns_val: float, photo: Path) -> Path:
    arm = tmp / name; (arm / "_renders").mkdir(parents=True); (arm / "_spark").mkdir()
    rows = []
    for cam in (0, 1):
        f = f"cam{cam:04d}_eval.png"
        _write(arm / "_renders" / f, ns_val); _write(arm / "_spark" / f, spark_val)
        rows.append({"file": f, "cam": cam, "variant": "eval", "photo": str(photo), "w": 8, "h": 6,
                     "c2w": np.eye(4)[:3].tolist(), "fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0})
    rows.append({"file": "cam0000_yaw+12.png", "cam": 0, "variant": "yaw+12", "photo": str(photo), "w": 8, "h": 6})
    (arm / "_renders" / "cams.json").write_text(json.dumps({"rendered": rows, "rasterize_mode": name}))
    return arm


def test_compare_and_verdict(tmp_path: Path):
    photo = tmp_path / "photo.png"; _write(photo, 0.5, size=(16, 12))   # different size: resized on load
    base = _arm(tmp_path, "classic", spark_val=0.5, ns_val=0.5, photo=photo)      # viewer == trained == photo
    cand = _arm(tmp_path, "antialiased", spark_val=0.6, ns_val=0.5, photo=photo)  # viewer drifts from trained
    b = sa.compare_arm(base); c = sa.compare_arm(cand)
    assert b["n"] == 2 and b["psnr_spark_vs_ns"] == 100.0          # identical images cap at 100 dB
    assert abs(c["psnr_spark_vs_ns"] - 20.0) < 0.3 and abs(c["psnr_spark_vs_photo"] - 20.0) < 0.3   # 8-bit quantisation
    v = sa.verdict(b, c)
    assert v["viewer_safe"] is False
    same = _arm(tmp_path, "same", spark_val=0.5, ns_val=0.5, photo=photo)
    assert sa.verdict(b, sa.compare_arm(same))["viewer_safe"] is True


def test_contact_sheet_layout(tmp_path: Path):
    photo = tmp_path / "photo.png"; _write(photo, 0.3)
    a = _arm(tmp_path, "a", 0.3, 0.3, photo); b = _arm(tmp_path, "b", 0.4, 0.3, photo)
    out = sa.contact_sheet([a, b], tmp_path / "sheet.png", max_cams=2, tile_w=64)
    im = Image.open(out)
    assert im.size[0] == 64 * 5 and im.size[1] == 2 * (48 + 18)
