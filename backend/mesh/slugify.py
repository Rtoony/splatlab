"""The ONE noun -> filesystem-slug function shared across every P6b/P6c
script that must agree on it (review finding 2026-07-23: scene_sam3_masks.py
and instance_lift.py each had their own character-for-character-identical
copy — safe today, a silent trap the moment either drifts).

Pure stdlib on purpose, same pattern as provenance.py: importable from the
FastAPI venv, the sam3 env, and the langfield-spike env, none of which share
third-party deps.
"""
from __future__ import annotations


def slug(noun: str) -> str:
    # Per-char substitution (NOT a collapsing regex) -- matches the two
    # duplicated implementations this replaces exactly, so existing on-disk
    # slugs ("fire-hydrant", "round-wooden-table", ...) don't shift.
    return "".join(c if c.isalnum() else "-" for c in noun.lower()).strip("-")[:40] or "thing"
