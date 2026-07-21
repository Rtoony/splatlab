#!/usr/bin/env python3
"""Copy a trained splatfacto run keeping ONLY the gaussians named in an
indices npz — so gs-mesh can export a tight, object-only mesh (P5b).

Adapted 2026-07-21 from mesh-trial/filter_checkpoint.py, which solved the two
recorded config-rewrite bugs: (1) nerfstudio yaml stores paths as PosixPath
component lists so text replacement silently no-ops — rewrite the OBJECT;
(2) method_name must stay real for the eval_setup registry. Verification is by
resolved gaussian count, never by filename.

Usage: checkpoint_subset.py <config.yml> <out_dir> --keep-npz object_indices.npz
Prints the subset config path on the LAST stdout line.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out_dir")
    ap.add_argument("--keep-npz", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    run_dir = cfg_path.parent
    models_dir = run_dir / "nerfstudio_models"
    ckpts = sorted(models_dir.glob("*.ckpt"))
    if not ckpts:
        print(f"FATAL: no checkpoint under {models_dir}", file=sys.stderr)
        return 2
    ckpt_path = ckpts[-1]

    indices = np.load(args.keep_npz)["indices"]
    try:
        with torch.serialization.safe_globals(
            [np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
        ):
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pipeline = sd["pipeline"]
    opac_key = next((k for k in pipeline if k.endswith("gauss_params.opacities")), None)
    if opac_key is None:
        print("FATAL: no gauss_params.opacities in checkpoint", file=sys.stderr)
        return 2
    n_total = pipeline[opac_key].shape[0]
    if indices.max() >= n_total:
        print(f"FATAL: index {indices.max()} out of range for {n_total} gaussians", file=sys.stderr)
        return 2
    keep = torch.zeros(n_total, dtype=torch.bool)
    keep[torch.tensor(indices)] = True

    prefix = opac_key.rsplit("gauss_params.", 1)[0] + "gauss_params."
    filtered = 0
    for k, v in pipeline.items():
        if k.startswith(prefix) and torch.is_tensor(v) and v.shape[:1] == (n_total,):
            pipeline[k] = v[keep].clone()
            filtered += 1
    if filtered == 0:
        print("FATAL: no gauss_params tensors matched for filtering", file=sys.stderr)
        return 2

    import yaml

    # nerfstudio's own !!python/object config graph, self-produced on this box.
    cfg_obj = yaml.load(cfg_path.read_text(), Loader=yaml.Loader)
    out_dir = Path(args.out_dir).resolve()
    cfg_obj.output_dir = out_dir
    cfg_obj.load_dir = None
    real_base = Path(cfg_obj.get_base_dir()).resolve()
    if not str(real_base).startswith(str(out_dir)):
        print(f"FATAL: rewritten base_dir {real_base} escapes {out_dir}", file=sys.stderr)
        return 2
    out_models = real_base / "nerfstudio_models"
    out_models.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_models / ckpt_path.name)
    (real_base / "config.yml").write_text(yaml.dump(cfg_obj))
    for aux in ("dataparser_transforms.json",):
        src = run_dir / aux
        if src.exists():
            shutil.copy2(src, real_base / aux)

    print(f"subset: kept {int(keep.sum())}/{n_total} gaussians ({filtered} tensors)", file=sys.stderr)
    print(real_base / "config.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
