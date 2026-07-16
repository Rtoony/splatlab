# Splatlab Agent Notes

## Project Status And Operator Intent (AUTHORITATIVE 2026-07-14)

- This project and its hardware-evaluation ladder are paused indefinitely.
- SplatLab is a hobby/research interest. The depth of its implementation and safety documentation does not make it a strategic priority over the user's work, infrastructure, or other projects.
- Do not proactively resume development or testing, run hardware acceptance, install or start Flight A units, launch GPU work, or remove/archive the maintenance marker.
- Resume only after a new explicit user request naming SplatLab or its hardware-acceptance process. Prior interest and prior authorization do not carry forward.
- Read `/home/rtoony/reports/2026-07-14-system-intent-and-indefinite-splatlab-pause.md` and the exact pause checkpoint before taking any action.

## Feedback Loop Workflow

- Treat `/api/feedback` records as the source of truth before guessing from chat.
- Start by listing non-terminal feedback: `New`, `Triaged`, `Planned`, `In Progress`, `Needs Info`, `Ready to Test`, and `Fixed`.
- Read `title`, `body`, `feedback_type`, `priority`, `status`, `page_url`, `page_path`, `page_tab`, `component_label`, `tags_json`, and `resolution_notes` first.
- Parse `context_json` for route, Splatlab scene/job id, selected viewer state, active search query, last click, browser/viewport, recent JS errors, failed API calls, and app commit.
- Inspect attachments when the feedback mentions visual layout, confusing UI, screenshots, or viewer behavior.
- Move records to `In Progress` while working, then to `Ready to Test` or `Fixed` with clear `resolution_notes` and machine-readable `resolution_metadata_json`.
- Leave `Accepted`, `Closed`, `Won't Fix`, and `Archived` for explicit user confirmation unless instructed otherwise.

## Safety

- Do not create `.env` files or write secrets to disk.
- Feedback context must not store cookies, auth headers, request bodies, localStorage dumps, or raw secrets.
- Attachments are local runtime data under `data/` and are intentionally gitignored.

## GPU Hardware Interlock

- `/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf` is the canonical, non-secret hardware-maintenance marker. Its presence blocks all SplatLab GPU work; never rename, bypass, remove, or archive it manually. Only the reviewed Flight A supervisor may manage its temporary, exact-one transition after all attended acceptance gates pass.
- Do not invoke `ns-train`, `ns-export`, `gs-mesh`, CUDA Python entry points, GPU-enabled COLMAP, LangField build/query scripts, or mesh/probe runners directly.
- Every manual GPU-capable command must be launched through `/home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh --run COMMAND [ARG ...]`. The wrapper fails closed and confines allowed work to `splatlab.slice`, CPUs 8-15, 400% CPU, 32/48 GiB memory, 8 GiB swap, and 512 tasks. Flight A is not a manual command and must not be nested through this wrapper; it may run only through the generated `splatlab-flight-a@.service` template.
- Every user service that starts GPU work directly must have `ExecCondition=/usr/bin/bash /home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh`, use `splatlab.slice`, and carry the same CPU affinity and limits. A timer inherits this rule through its paired service; timers cannot carry `ExecCondition` themselves.
- `splatlab.service` stays available for read-only status and browsing during maintenance, so it intentionally has no unit-level `ExecCondition`. Its GPU and mutation endpoints must remain fail-closed through `SPLAT_TRAINING_DISABLED_REASON` from the canonical marker.
- `/home/rtoony/projects/splatlab/tools/gpu-hardware-acceptance.py` may only produce `PASS_PRE_FLIGHT_A`; it never removes the marker and its result is not permission to run GPU work.
- A verified `PASS_PRE_FLIGHT_A` may be consumed once only by `/home/rtoony/projects/splatlab/tools/splatlab-flight-a-supervisor.py`, through the generated systemd template and an exact receipt basename. The supervisor has one immutable payload, performs at most one API POST, treats an ambiguous POST as consumed without retry, monitors the complete service cgroup, and restores or retains an authoritative marker before result review on every recovery path.
- The Flight A supervisor and recovery units are source-staged only. Do not install, enable, start, or invoke them directly until the attended physical/firmware/MemTest gate, a fresh-boot acceptance receipt, an explicit operator authorization, and a separate controlled deployment review all pass. Permanent release and every later ladder rung remain separate explicit decisions.
