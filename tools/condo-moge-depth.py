#!/usr/bin/env python3
"""Compatibility wrapper: use tools/moge-depth-views.py (building-agnostic)."""
import os, sys
from pathlib import Path
os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve().parent / "moge-depth-views.py"), *sys.argv[1:]])
