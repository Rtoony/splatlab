# Claude Code Handoff - SplatLab Feedback And Camera Pass

Date: 2026-07-02
Owner context: Rtoony is handing active SplatLab viewer development back to Claude Code after a Codex pass.

## Current State

- Public app: `https://splatlab.roonytoony.dev/`
- Local service: `splatlab.service`
- Local health: `http://127.0.0.1:3416/healthz`
- Working directory: `/home/rtoony/projects/splatlab`
- Feedback database: `/home/rtoony/projects/splatlab/data/feedback/feedback.sqlite3`
- Feedback records `#3`, `#4`, `#5`, and `#6` were fixed, verified, then closed at user request.
- At handoff time there were no non-terminal feedback records.

## What Shipped

- Added a structured in-app feedback loop.
- Added backend feedback storage, comments, attachments, status/resolution fields, and safe context capture.
- Added frontend feedback widget, feedback API/contracts/context helpers, and a management page at `/feedback`.
- Added project-level agent notes in `AGENTS.md` so future agents start from feedback records before guessing.
- Added capture camera metadata endpoint at `/api/splat/jobs/{job_id}/cameras`.
- Camera metadata includes original image names when available, original file paths, position/forward/up/right vectors, optional FOV, count/sample metadata, and viewer-frame transform handling.
- Added fullscreen viewer camera overlays with visible camera nodes/frustums.
- Added a `Camera shots` panel listing original capture image names.
- Added separate camera actions:
  - Crosshair / visible node click: tight camera-node inspection.
  - Eye / image-name click: view from the original capture camera pose and FOV.
- Added `Reset` and `Advanced` viewer controls.
- Default viewer state is quieter: search, inventory, camera shots, shortcut legend, and highlights start collapsed/off.
- `Reset` hides overlays/panels, clears search/selections/highlights, and restores default camera/orbit/FOV.
- `Advanced` opens the optional search, inventory, camera, and shortcut panels without enabling all object highlights.
- Fixed feedback text-entry hotkey conflicts by making feedback modal/input focus suppress viewer hotkeys.

## Closed Feedback Records

- `#3`: Add camera shots legend with original image names, zoom-to camera shot, and view-from-camera action.
- `#4`: Make visible camera nodes clickable and route them through the same zoom/orbit behavior as the camera panel.
- `#5`: Add default/reset behavior and make advanced viewer extras opt-in rather than all visible by default.
- `#6`: Make camera panel crosshair zoom match visible-node click, but tighter and aligned behind the source camera direction.

Useful query:

```bash
sqlite3 data/feedback/feedback.sqlite3 "SELECT id,status,title FROM feedback_items ORDER BY id;"
```

## Verification Already Run

```bash
cd /home/rtoony/projects/splatlab/frontend && npm run build
cd /home/rtoony/projects/splatlab && python3 -m py_compile backend/splat_route.py backend/main.py backend/feedback.py
cd /home/rtoony/projects/splatlab && pytest backend/tests/test_splat_cameras.py backend/tests/test_feedback.py -q
systemctl --user restart splatlab.service
systemctl --user is-active splatlab.service
curl -fsS http://127.0.0.1:3416/healthz
cd /home/rtoony && just preflight-app splatlab
```

Observed verification result:

- Frontend build passed.
- Backend compile passed.
- Tests passed: `6 passed`.
- `splatlab.service` was active.
- Health endpoint returned `{"ok":true,"service":"splatlab",...}`.
- `just preflight-app splatlab` passed.

## Key Files

- `backend/feedback.py`: feedback storage/API helpers.
- `backend/main.py`: feedback route registration and app wiring.
- `backend/splat_route.py`: camera metadata endpoint and camera transform handling.
- `backend/tests/test_feedback.py`: feedback backend tests.
- `backend/tests/test_splat_cameras.py`: camera endpoint/transform tests.
- `frontend/src/components/feedback-widget.tsx`: floating feedback capture UI.
- `frontend/src/components/splat-viewer.tsx`: Gaussian splat viewer overlays, camera nodes, reset/view/zoom effects.
- `frontend/src/lib/feedback-api.ts`: feedback frontend API helpers.
- `frontend/src/lib/feedback-context.ts`: safe contextual feedback state.
- `frontend/src/lib/feedback-contracts.ts`: feedback frontend types.
- `frontend/src/lib/api.ts`: camera endpoint client.
- `frontend/src/lib/contracts.ts`: camera response contracts.
- `frontend/src/pages/feedback.tsx`: feedback management page.
- `frontend/src/pages/splat-view.tsx`: fullscreen viewer controls, camera shots panel, reset/advanced behavior.

## Known Caveats And Follow-Up

- The current repo has many uncommitted tracked and untracked changes from the feedback/camera work. Review before committing.
- `frontend/dist/` changes are build output; decide whether to commit or ignore based on the repo's deployment convention.
- Camera alignment depends on Nerfstudio transform metadata. The latest fix corrected the known garden-scene axis issue, but new datasets should still be spot-checked visually.
- Camera-node zoom distance is scaled from `display_scale` with a minimum distance. If Rtoony says it is still too close/far, tune `distance: Math.max((cameras?.display_scale ?? 0.08) * 4, 0.24)` in `frontend/src/pages/splat-view.tsx`.
- The visible camera overlay currently samples via endpoint `limit=500`; for very large scenes, consider UI-side filtering or a lower default.
- The feedback database lives under `data/` and should stay gitignored.
- Do not add `.env` files. Secrets are vault/injected only per `/home/rtoony/AGENTS.md`.

## Suggested Next Claude Code Pass

1. Start by running the feedback query above and reading any new `New`, `Triaged`, `In Progress`, `Ready to Test`, or `Fixed` records.
2. Run `git status --short` and separate intended SplatLab changes from unrelated local edits.
3. Review the viewer manually on `splat_32d926d9` because that was the main camera-feedback test scene.
4. If committing, make a focused commit that includes feedback system + camera viewer work, excluding runtime data and unrelated artifacts.
5. Continue using feedback records as the source of truth. Move records to `In Progress` while working, then `Fixed` or `Closed` with concrete `resolution_notes`.
