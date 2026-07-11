"""Inject per-frame mask_path into a freshly generated transforms.json and copy
the sequential-named masks into <processed>/masks/. nerfstudio requires
all-or-none mask_path across frames — asserts every registered frame has one.

  python3 inject_masks.py <processed_dir> <seq_masks_dir>
"""
import json
import shutil
import sys
from pathlib import Path

processed = Path(sys.argv[1])
seq_masks = Path(sys.argv[2])
tj = processed / "transforms.json"
data = json.loads(tj.read_text())

mask_dir = processed / "masks"
mask_dir.mkdir(exist_ok=True)
n = 0
for frame in data["frames"]:
    name = Path(frame["file_path"]).name           # frame_00042.jpg
    src = seq_masks / f"{name}.png"
    assert src.exists(), f"missing mask for registered frame {name}"
    shutil.copy2(src, mask_dir / f"{name}.png")
    frame["mask_path"] = f"masks/{name}.png"
    n += 1
tj.write_text(json.dumps(data, indent=4))
print(f"[inject] mask_path set on {n} frames -> {mask_dir}", flush=True)
