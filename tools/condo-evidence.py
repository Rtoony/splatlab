#!/usr/bin/env python3
"""2286 Chanate wrapper around tools/building-evidence.py (the building-agnostic exporter):
fills in the condo's review target, datums and the scale-check target; every other flag passes through."""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULTS = ["--building-target", "building:2286-chanate", "--scale-check-target", "opening:garage-door", "--floor-datums", "garage slab=0,studio FF=0.3048"]
args = sys.argv[1:]
for flag in ("--building-target", "--scale-check-target", "--floor-datums"):
    if flag in args:                                     # an explicit flag wins over the condo default
        i = DEFAULTS.index(flag); del DEFAULTS[i:i + 2]
os.execv(sys.executable, [sys.executable, str(HERE / "building-evidence.py"), *DEFAULTS, *args])
