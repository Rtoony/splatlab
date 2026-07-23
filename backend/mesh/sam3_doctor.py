"""Fail-loud preflight for the SAM 3 enumeration lane (P6b).

Shallow by default (files + env present, no GPU); --deep loads the checkpoint
onto the GPU via the repo's own smoke test (call it only under the compute gate).

Usage: sam3_doctor.py [--deep]
Exit 0 = healthy, non-zero = a numbered, actionable failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SAM3_ROOT = Path("/home/rtoony/projects/ml/sam3")
SAM3_CKPT = SAM3_ROOT / "checkpoints" / "sam3.1_multiplex.pt"
SAM3_BPE = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
SAM3_PYTHON = Path("/home/rtoony/miniconda3/envs/sam3/bin/python")
SAM3_SMOKE = SAM3_ROOT / "scripts" / "local_smoke_test.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true",
                    help="load the checkpoint on the GPU (run under the compute gate)")
    args = ap.parse_args()

    checks = [
        ("sam3 repo", SAM3_ROOT.is_dir()),
        ("sam3.1 checkpoint (~3.3G)", SAM3_CKPT.is_file() and SAM3_CKPT.stat().st_size > 2e9),
        ("bpe tokenizer asset", SAM3_BPE.is_file()),
        ("sam3 conda env python", SAM3_PYTHON.is_file()),
        ("smoke test script", SAM3_SMOKE.is_file()),
    ]
    ok = True
    for name, passed in checks:
        print(f"{'OK  ' if passed else 'FAIL'} {name}")
        ok = ok and passed
    if not ok:
        print("sam3_doctor: FAILED (fix the FAIL lines above)", file=sys.stderr)
        return 1

    if args.deep:
        proc = subprocess.run(
            [str(SAM3_PYTHON), str(SAM3_SMOKE)],
            env={"PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=300)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            print("sam3_doctor: deep load FAILED", file=sys.stderr)
            return 2
        print("OK   deep checkpoint load on GPU")
    print("sam3_doctor: healthy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
