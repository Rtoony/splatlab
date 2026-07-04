# Splatlab Agent Notes

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
