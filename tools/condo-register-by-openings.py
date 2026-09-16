#!/usr/bin/env python3
"""Compatibility wrapper: use tools/register-by-openings.py (building-agnostic)."""
import os, sys
from pathlib import Path
os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve().parent / "register-by-openings.py"), *sys.argv[1:]])
