#!/usr/bin/env python3
"""Scheduled log-maintenance: prune the operator-audit and op-registry stores.

Runs the two retention prunes that nothing else schedules (operator_audit
daily JSONL files; opregistry finished rows). The heavyweight third prune —
_prune_old_jobs, which deletes multi-GB job DIRECTORIES — is deliberately NOT
here: it already runs opportunistically in the job lifecycle
(splat_route.py, "Pruned N old unpinned job(s)"), and unattended deletion of
job trees is an operator-approval class of action.

Invoked by splatlab-prune.timer (daily). Safe to run by hand any time:

  python3 tools/prune-maintenance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import operator_audit  # noqa: E402
import opregistry  # noqa: E402


def main() -> int:
    audit_removed = operator_audit.prune()
    registry_removed = opregistry.prune()
    print(json.dumps({
        "audit_files_removed": audit_removed,
        "registry_rows_removed": registry_removed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
