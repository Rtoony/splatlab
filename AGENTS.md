# Splatlab Agent Notes

## Project Status And Operator Intent (AUTHORITATIVE 2026-07-19)

- SplatLab is explicitly reopened for development, test design, and manually selected experiments.
- The former hardware-acceptance and Flight A ladder is historical, not the current development path. Do not resume, install, or run it merely because SplatLab is unlocked.
- The old hardware-maintenance marker was retired and archived under `/home/rtoony/backups`; the marker-reasserting watcher and nightly autoresearch timer are disabled.
- The project remains a hobby/research interest rather than a strategic priority. Follow the user's selected development direction without inferring a roadmap from the old safety documentation.
- Unlocking the project does not itself authorize an unattended or heavy GPU run. Launch GPU work only when it is part of the user's current task, and keep the existing bounded compute path.
- Read `/home/rtoony/reports/2026-07-14-system-intent-and-indefinite-splatlab-pause.md` for the current handoff and treat pre-2026-07-19 acceptance documents as historical references.

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

## GPU Development Controls

- No production hardware-maintenance marker should be present. Do not recreate it from an old acceptance document; a future hold needs a new explicit operator decision.
- Do not invoke `ns-train`, `ns-export`, `gs-mesh`, CUDA Python entry points, GPU-enabled COLMAP, LangField build/query scripts, or mesh/probe runners directly.
- Every manual GPU-capable command must be launched through `/home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh --run COMMAND [ARG ...]`. The wrapper confines work to `splatlab.slice`, CPUs 8-15, 400% CPU, 32/48 GiB memory, 8 GiB swap, and 512 tasks, then uses the shared GPU coordinator for VRAM admission.
- Every user service that starts GPU work directly must retain `ExecCondition=/usr/bin/bash /home/rtoony/projects/splatlab/tools/splatlab-compute-gate.sh`, use `splatlab.slice`, and carry the same CPU affinity and limits.
- Keep Redis serialization, backup exclusion, VRAM admission, process-group cleanup, and resource boundaries in place unless the user explicitly asks to redesign them.
- `nexus-gpu-health-watch.timer` and `splatlab-mesh-autoresearch.timer` are intentionally disabled. Do not enable them as a side effect of development work.
- Flight A supervisor/recovery source and historical receipts remain for reference, but the installed runtime units and boot dependency are retired. Reusing that ladder requires a new explicit design decision.
