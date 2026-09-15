# Splat Lab — standalone app (extraction from portal)

> Source of truth for this build. Read before resuming; update after every step.

## 2026-09-09 17:04 PDT — native condo surface pipeline executes; live mesh inspection and handoffs verified

- The owner's fresh 60-minute authorization supersedes the earlier no-progress/native-training blocker. Three actual native 2DGS runs finish: unanchored 3,000 steps, source-anchored 3,000 and source-anchored 6,000. The latter has 407,131 surfels and 19.9762 dB masked validation PSNR. Saved native parameters reproduce all 24 validation renders byte-for-byte in a new process; a real 120-frame/eight-second model-rendered camera tour is delivered.
- **Open http://127.0.0.1:32856/**. The final 559-file gallery lets the owner switch between the live radiance approximation and both actual triangle meshes at the same source camera. It includes the native movie, 24 source/render/error triplets, three controlled surface comparisons, both Blender downloads and two Condo Lab-compatible mesh manifests. It serves `data/spatial/condo-surfel-showcase-2026-09-09-07` until approximately 23:03 PDT; older previews are preserved.
- Shared static-region ray coverage rises 87.06% for the appearance mesh to 95.03/95.20/95.79% for the three surface variants. This is **coverage, not accuracy**. Source anchoring improves reserved-point mean depth agreement .849 → .642 arbitrary units at 3,000 steps; 6,000 worsens it to .787 with fewer reserved ray hits, despite better RGB quality. Keep both anchored candidates; neither is accepted architectural geometry, metric registration or a condo replacement.
- Both Blender scenes independently reopen with 72 packed photos/cameras and 1,949 sparse points. New handoff GLBs pass exact float32 source-coordinate and complete oriented-triangle checks (304,513 / 328,869 triangles). Portable unlit vertex-color materials fix an actual Blender emission-export mismatch without changing any binary geometry/color buffers. Final handoffs are `condo-surface-external-reference-2026-09-09-03` and `...-04`, with registration null; both validate against the real Condo Lab schema. No condo code, correction acceptance or canonical geometry changed.
- Final NVIDIA Chromium proof passes real parallax, both live meshes, unchanged source camera, reset, five sampled views, 72 comparison images, mobile/idle checks, native video, both hash-verified mesh downloads and simulated GPU-context-loss fallback. Proof: `/home/rtoony/reports/2026-09-09-condo-surfel-browser-proof-08/proof.json`. Video decoding is CPU-only in this verification launcher after hardware-decoding context loss; WebGL remains NVIDIA. Earlier failed proofs are preserved. **142 focused Python tests pass**, with syntax/whitespace and Node checks; no full app regression claimed.
- GPU jobs are finished; the RAM-only override is removed at **17:04 PDT**, before the original 17:19 expiry. Persistent policy, other capabilities and resource limits remain intact. No cloud spending/downloads, live deployment, hardware acceptance or source-media mutation. The broader registered real-to-condo-detailing Goal remains incomplete; do not replace it with this successful pipeline milestone. Full commands, receipts, tradeoffs and next phase: `/home/rtoony/reports/2026-09-09-splatlab-condo-native-surfaces.md`.

## 2026-09-09 15:59:45 PDT — Goal blocked after three consecutive no-progress audits

- The 15:58, 15:59 and current read-only audits confirm the same elapsed GPU launch cutoff, unchanged lease and absence of an owned reconstruction worker or fresh clearance. The Goal is marked **blocked, not complete**. The pending 30-minute GPU-window request is the next required operator action; automatic continuations are not authorization to renew the lease.
- The entire real-images/video → recreated 3D → reviewed condo-detailing objective is retained. Appearance delivery and structural diagnostics work, but acceptable structural geometry/registration and native 2DGS execution remain incomplete. Do not substitute the 127 passing CPU tests or private previews for that end state.
- No new compute, code feature, condo edit, service restart, acceptance or policy change occurs in this audit. Existing private previews remain bounded and all source artifacts are retained. Resume from the staged `tools/train-capture-surfels.py` experiment after fresh clearance; read `/home/rtoony/reports/2026-09-09-splatlab-condo-structural-study.md` first.

## 2026-09-09 15:57 PDT — staged 2DGS input/export integration corrected, CPU-only

- **15:59 PDT second consecutive no-progress audit:** authoritative process inspection again finds no owned reconstruction worker; no new clearance or lease change has arrived. The same expired launch cutoff remains the blocker. No new code, compute or acceptance is justified by this audit; the full Goal remains active/incomplete pending the third-audit threshold.
- **15:58 PDT first no-progress audit:** the preceding turn changed and validated the staged runner. Fresh process inspection now finds no owned training, fusion or Blender worker. Nexus has no new clearance; the lease still has its elapsed 15:45 launch cutoff and unchanged 16:00 expiry. Typed capability remains allowed until expiry, but that is not a renewal. No substantive safe next action is identified before GPU validation. This is not a verified worker wait; keep the full Goal active/incomplete for this first no-progress audit.
- Previous Goal turn made substantive progress (actual structural comparison, Blender/browser proof and new gallery); this continuation also advances the staged reconstruction path rather than claiming a GPU wait. No training worker is live and the pending GPU-window request is unanswered.
- The new loader now rejects omitted transform/sparse-point hashes and per-frame calibration that the fixed-intrinsics renderer would otherwise ignore. A tested preview adapter inverts solver normalization, preserves tangent scales/quaternions/SH/opacity, emits the required float32 PLY parameters, and leaves native 2D surface parameters untouched. Sparse initialization rejects fewer than four points rather than producing infinite neighbor scales.
- **127 focused tests pass in 0.62 seconds**, including a real PLY write from converted synthetic surface parameters; syntax and whitespace checks pass. The actual condo dataset still verifies all 244 sealed artifacts and its 72/24/24 split. This is CPU input/export validation, **not native 2DGS training or a new condo reconstruction**.
- Existing previews remain unchanged. The geometry-directed GPU run and structural acceptance remain incomplete; do not renew the 16:00 expiry or infer new GPU permission from this automatic continuation.

## 2026-09-09 15:51 PDT — actual condo structural comparison, not an architectural pass

- Local SAM3 processes 24 real central training photos. Corrected seven-prompt study retains garage doors; 168 mask files, no cloud calls/downloads. Source-track voting excludes 43 ambiguous original SfM points without editing originals. There are 72 supported façade landmarks but **no passing façade plane**; one pavement candidate has 23/30 reserved point checks near its plane, not surveyed accuracy.
- Matched 24-view TSDF control versus semantic-filtered extraction executes. Triangles drop 711,081 → 107,485 and components 71,578 → 8,672, but shared static-surface ray coverage drops **89.36% → 79.74%**. Visual comparisons confirm holes. **Do not call this a better architectural model or promote a floor/wall/collider.**
- Open **http://127.0.0.1:32853/** for the actual structural review, all 24 four-column comparisons and the independent Blender download. Existing live appearance splat remains at **http://127.0.0.1:32852/**. The new six-hour loopback server starts approximately 15:48 PDT; its 129-file package persists in `data/spatial/condo-structure-showcase-2026-09-09-01`.
- New Blender save/reopen passes: 90,894 vertices, 107,485 triangles, 72 packed source photos/cameras. Actual static Chromium proof passes 96 comparison images, five overview images, mobile/download/security checks and no external requests, with GPU/WebGL disabled. **121 focused Python tests**, syntax compilation, whitespace checks and actual CPU verification of 244 frozen input files pass. No full regression, metric registration, owner acceptance, accepted condo edit or live deployment is claimed.
- Native 2DGS normal/distortion training is **staged, not run** in `tools/train-capture-surfels.py`. CPU preparation/projection tests pass, but CUDA training/output compatibility remains unverified. The existing launch cutoff passed at **15:45**; the GPU window expires **16:00 PDT**. No new GPU work starts or lease extension occurs. A fresh bounded window is needed for the next geometry-directed experiment.
- Goal active/incomplete. Details, source citations, actual negative result and next experiment: `/home/rtoony/reports/2026-09-09-splatlab-condo-structural-study.md`. The accepted condo model and older appearance showcase remain unchanged.

## 2026-09-09 14:18 PDT — real condo splat, live Spark viewer and Blender delivery

- Fresh owner GPU/Chromium authorization produces an actual **2,000-step / 392,249-Gaussian condo exterior baseline**. Training succeeds; the original combined job exits 1 on legacy `ns-export`/PyTorch weights-only compatibility. A new safe-load export/evaluation runner succeeds without retraining, unrestricted pickle fallback or checkpoint-row filtering. Do not relabel the original failed export as passed.
- Actual frozen-geometry localization passes **16/16 physical held-out images** in 5.05 seconds. Posed crops preserve **72 train / 24 validation / 24 test** splits. Experimental geometric rim/nadir masks are visually reviewed, not semantically accepted. Real validation renders score **20.0381 dB masked / 19.3906 dB full-image PSNR**. The 24 crops represent four timestamps; test appearance is unused, and neither score establishes metric geometry accuracy.
- **Open http://127.0.0.1:32852/** for actual interactive 3D plus all 24 source/render/error comparisons. It serves `data/spatial/condo-reconstruction-showcase-2026-09-09-02` until approximately **20:17 PDT**, command session `65135`. Actual Chromium/RTX 5090 WebGL proof passes: camera parallax changes pixels, reset, five sampled views, all 72 comparison images, mobile, no external requests and idle rendering pause. Proof: `/home/rtoony/reports/2026-09-09-condo-reconstruction-browser-proof-02/proof.json`. Host/traversal negative controls return 403/404. No public deployment.
- Blender 4.0.2 actually imports **1,949 points + 72 source cameras**, passes **376 on-image projection checks**, packs all 72 source photos, renders, saves and **reopens successfully**. File: `data/spatial/condo-blender-review-2026-09-09-04/condo-capture-review.blend`; reopen proof: `/home/rtoony/reports/2026-09-09-condo-blender-reopen-proof.json`. Fixed basis `x,y,z → x,-z,y` is documented, not claimed as physical up or condo registration. Accepted condo model is never opened or modified.
- **Architectural geometry is NOT ready:** inferred training-depth TSDF has 968,438 vertices / 1,100,381 triangles, 110,472 components, 11.93% largest-component triangle membership and 625,421 boundary/nonmanifold edges. Retain this fragmented mesh for diagnosis, not architecture/collision/fabrication/survey promotion. First finer extraction fails its bounded nonempty/size gate; coarser `.15` arbitrary-unit output is not quality acceptance. Next: source-supported structural geometry, not blind appearance-only training or a return to Bonsai.
- **140 focused Python tests + 3 direct Node cases pass**, including complete synthetic input preparation with frozen-source/split/pose/hash checks; actual Blender and GPU browser proofs are separate. No full backend/frontend regression, cloud call, paid spend, owner acceptance, architectural registration or live app change is claimed. All compute/proof jobs are terminal; only private preview PID `3367858` is intentionally retained. Temporary SplatLab-only RAM window expires **16:00 PDT**, no starts after **15:45**, existing GPU/cgroup/backup guards retained. Initial training admission automatically evicts `maximus-tts` and `muse-glimmer`; no manual resident killing occurs.
- Goal active/incomplete: appearance and delivery work; usable structural geometry/detailing remain. Read `/home/rtoony/reports/2026-09-09-splatlab-condo-reconstruction.md`. Earlier “localizer unexecuted / GPU permission outstanding” notes are superseded.

## 2026-09-09 — NEW real-to-model Goal: condo raw-lens projection bridge

- The Goal tool now explicitly names real images/videos → 3D model recreation → independent condo/Blender detailing. This supersedes the old indoor doorway objective below; do not resume Bonsai placement as the default queue. Finish studies remain a separate design feature, not reconstruction acceptance.
- New CPU-only preparation creates 120 rectilinear crops from 40 real raw-lens images in `data/spatial/condo-rectified-reference-2026-09-09-01`: 72 training views with inherited poses, 24 validation and 24 test views with deliberately unknown poses. Three crops share each source optical center; the two physical lens centers are not merged. Gallery, source/mask hashes, PTS lineage, camera transforms, unchanged existing sparse PLY and independent condo handoff are written. No new dense mesh or Gaussian training is claimed.
- Actual native-projection check: 8,041 supported training observations, 38 outside the 85-degree cone; mean/median/p95 residual 0.566019/0.385606/1.744113 pixels. This verifies the estimated projection path, not held-out appearance or physical accuracy. Rectification uses estimated calibration, binary projection masks only, arbitrary scale/up, and null architectural registration.
- Staged next runner localizes held-out frames using a separate consistent database copy and frozen 3D tracks/intrinsics. Correspondence identity, ambiguity, uniqueness and direct-launch guards are unit-tested; the actual pycolmap estimation path has not run. All 70 focused new/adjacent tests pass. Existing training and condo geometry remain unchanged.
- Live condo studies list is empty; save a study to enable its board downloads. Accepted hash remains `21c8ef2ea8d5676e81762fd4ed3f6a07fbee0cc56ad254b0d464ed218efd2d1a`, while unrelated saved architectural exports remain stale. No condo code, owner acceptance, secrets, routing, services or exports are changed.
- `splatlab.cuda` remains denied by operator hold. Fresh 30-minute compute permission was requested, not assumed; no GPU/reconstruction launch or policy change occurs. Next: gated held-out localization, reviewed masks, one exterior baseline/source-render comparison, then separate Blender reference/correction handoff. Goal active and incomplete. Runbook: `docs/real-to-model-pipeline.md`; report: `/home/rtoony/reports/2026-09-09-splatlab-real-to-model-inputs.md`.

## 2026-09-09 12:52 PDT — active connected-space Goal: source-height audit advances diagnosis

- The active Goal still names the captured-room connection; the owner's separately verified live condo finishes do not complete it. Current private scene is freshly verified at generation 48, `scene_436ca21966e33dd3920b228f`, unchanged and unapplied. Fresh typed policy denies `splatlab.cuda`; no expired window is renewed.
- A new CPU-only audit reviews all eight retained placement leads against 605,391 pinned Gaussian centers and 579 retained source guard points. All eight avoid named centers and the guard, but affect 4,062–19,922 unassigned centers. This is not complete object/footprint preservation. The existing room-interior reservation is already present; no missing-envelope bug is claimed.
- **New reason not to promote those leads:** their two sampled starts sit above nearby triangulated source observations. Start `[-0.35,-1.425619,1.3]` has 493 quality points nearby, zero within 8 cm vertically; nearest source point is 28.45 cm lower. Start `[-0.55,-1.619351,0.5]` has 2,958 nearby points, only two within 8 cm; nearest is 10.73 cm lower. Native original-photo inspection places the projected starts over wood/carpet, not the bonsai tabletop, but does not independently certify visibility/depth. Reconcile source-supported floor and collision before another placement search or GPU traversal; never pass by resizing/flying the player.
- Added `tools/review-architectural-candidates.py` and five focused tests. New plus adjacent volume tests: 25 passed; actual audit completes exit 0 and preserves scene/source hashes. No new geometry activation, GPU render, apply/reload/undo, secret retrieval, condo mutation, deployment or overall acceptance is claimed.
- Report and next bounded experiment: `/home/rtoony/reports/2026-09-09-splatlab-connected-space-audit/README.md`; final evidence `candidate-source-review-03.json`. This is a progress turn, not a live-worker wait or another blocked-status audit. Full Goal remains active and incomplete.

## 2026-09-09 — owner-selected GPS360 exterior atlas and camera diagnostics

- Owner assigns SplatLab's GPS/360 surroundings work here while a separate session owns condo-model corrections and public deployment. Condo write scope is released in Nexus368; no condo/routing/accepted geometry is changed.
- New private atlas `atlas_b4d85d5b90b8439c` assembles **eight original recordings, 617.867 seconds, 20 review panoramas and 5,998 GPS fixes**. It provides actual perspective look-around, separate track/map selection, cross-clip contacts, keyboard/mobile controls and provenance/GeoJSON exports. Authenticated API and Routes-page integration are staged and privately built, not deployed.
- **Clip002 clock mismatch:** first GPS fix is361.09seconds after container start, outside the277.377-second recording interval. Its nine views remain unplaced; eleven others have tentative GPS locations. Four stale reference files remain explicitly missing. No guessed clock correction, camera orientation, terrain, architectural transform or physical scale is invented.
- Existing training-only raw-lens model receives read-only rig diagnostics:12pairs/24images, median/max relative-rotation deviation0.1210°/0.1925°, but estimated lens separation0.01424–0.14458arbitraryunits. This is not calibrated-rig acceptance. Frozen-geometry held-out localization and projection validation remain before a small Gaussian baseline.
- Validation:32atlas/server/route Python tests,4rig Python tests,4Node cases,339frontend tests/22files, TypeScript/privatebuild/targetedlint pass. Both actual CPU browser proofs cover all20distinct views, look-around/exactreset/mapkeyboard/390px/zeroexternalrequests and unchanged snapshot hashes. Initial socket-restricted sandbox HTTP tests time out; the host retry passes. No full backend regression or new reconstruction success is claimed.
- Private preview: `http://127.0.0.1:32851/`, existing process1933830 with an eight-hour budget from approximately12:22PDT. Only this loopback review server remains; no GPU job, cloud call, paid spend, hardware acceptance or Goal-state change occurs. Durable report: `/home/rtoony/reports/2026-09-09-splatlab-capture-atlas.md`; implementation/commands: `docs/capture-atlas.md`.

## 2026-09-09 02:30 UTC — owner-directed condo-first priority reset

- Owner questions over-investment in the cluttered Bonsai fixture and asks whether newer captures should lead. **Make the retained condo exterior pilot the primary demonstration; stop the Bonsai doorway-placement search.** Keep its eight unaccepted leads and incomplete preservation/transaction requirements as deferred evidence, not an automatic work queue.
- Existing generated-object/native-Gaussian results are real but mostly share the same source scene. The showcase now exposes earlier generative evidence and the revised roadmap. Candidate plane detection is not complete observed floor coverage or trustworthy collision; do not erase uncertainty to make a demo pass.
- Next bounded work: inspect existing condo camera/rig estimates and validation-view localization, verify the actual projection path, then produce a small exterior Gaussian baseline and source/render comparison. First input diagnosis is capped at 90 active minutes; one baseline and at most one evidence-directed refinement precede a visible checkpoint or input switch. No more open-ended Bonsai floor polishing.
- New controlling handoff: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/condo-first-development-reset-2026-09-08.md`. Nexus 346 announces read-only shared media and separate SplatLab derivative scope; the other condo-model and print sessions remain untouched. This turn changes documentation/gallery only; no compute, scene activation, upload or policy change. The previous Goal is not complete and its old work ordering must not override this user-directed reset.

## 2026-09-09 02:05 UTC — overnight source-preserving approach leads and owner showcase

- User explicitly authorizes another 6–12 hours of shared work and requests a visual progress update. First overnight RAM segment is active until **00:45 PDT**, no new starts after **00:30**; only our evening override is replaced. Thirteen helper tests pass. Initial activation briefly encounters connection refusal during the orchestrator restart; readiness is rechecked and activation succeeds before any compute. Existing shared-resource/backup/power safeguards are unchanged.
- New bounded CPU footprint search is terminal exit 0: **210 starts, nine sampled-admissible starts, 216 headings, 23 sampled-clear approaches and eight layout survivors**. Actual scene remains generation 48. The candidates are not accepted geometry; especially inspect higher starting surfaces against original photos so furniture is not mistaken for floor. Full source/named-Gaussian protection, geometry and GPU walking remain next.
- Owner-facing summary and actual-image gallery: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/progress-summary-2026-09-08.md` and `progress-showcase-2026-09-08.html` beside it. The gallery separates the proven indoor preview, remaining preservation/transaction checks, preliminary condo input work and longer-term creative editor direction. No generated illustration, new render or live deployment is substituted for actual retained evidence.
- Current search evidence: `preserving-approach-search-01/preserving-approach-search.json`. Nexus 345 releases all owned compute while preparing the user's update. Coordinate with the separate 2286 condo-model and print-waiting sessions; do not assume zero registered Nexus sessions means no competing work. The Goal tool still reports blocked and needs client-side resume for automatic continuation; this turn has made new concrete progress.

## 2026-09-09 01:47 UTC — owner-authorized shared evening window and preservation studies

- Fresh evening GPU authorization is implemented as a separate SplatLab-only RAM window until **22:00 PDT**, no new starts after **21:45**. Independent expiry is verified active/waiting; persistent policy and all unrelated capabilities remain unchanged. Old window expired cleanly. Twelve helper tests pass; existing power/cgroup/Redis/backup/VRAM guards stay in place. Use short sequential bursts, coordinate in Nexus, and never extend the window from an automatic continuation.
- Actual CPU-only screening leaves resident GPU work alone: 120 layouts tested against 579 guarded source points and named centers; 52 survive that limited point screen. All 52 fail a further conservative uncut-approach screen, and 25 also intersect sampled room occupancy. These are bounded feasibility observations, not complete object protection or whole-route acceptance.
- A full diagnostic joined-floor build at yaw165/depth6, `architecture_3b1c7898e49d4e6592d395db`, confirms genuine failure beyond the prefilter: **exit 2, FAIL_ARCHITECTURAL_EDIT, 9/12 gates**. Approach/room floor continuity and capsule clearance fail; minimum clearance 0.1711 m and maximum floor error 0.09533 m. No threshold is relaxed and no failed candidate is previewed, rendered or activated. Current successful preview remains intact; active generation 48 unchanged.
- Next: inspect a different approach start and actual floor footprint, not just rotate this entrance or shrink the player. Tripod/boxes stay protected by default; the user granted compute, not removal. Full acceptance still needs reviewed preservation, a usable candidate and real apply/reload/undo.
- Report: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/evening-sharing-2026-09-08.md`. The persistent Goal still reports its earlier `blocked` status and needs client-side resume for automatic continuation; this fresh user-directed turn makes concrete progress and is not another blocked audit. No new Goal, deployment or cloud expenditure is introduced.

## 2026-09-09 01:20 UTC — Goal blocked after three consecutive owner-clearance audits

- The two preceding audit turns were no progress, not live-worker waits. A third fresh read confirms no new owner decision, no new Nexus message beyond resource release 334, the same closed GPU new-start window and unchanged private scene generation 48. The persistent Goal is now **blocked, not complete**, with its full objective retained.
- Resume requires a new bounded GPU authorization for actual remaining validation. The tripod/stacked-box area remains preserve-by-default unless the owner explicitly permits this bounded replacement; a preserving alternative needs its own build and rendered evidence. The requested 30-minute window and object decision have not been granted by automatic continuations or read-only command approvals.
- The independent expiry timer is actually `active / waiting`, triggers `splatlab-gpu-window-expiry-20260908.service` at **18:27:49 PDT**, and remains unchanged. No owned SplatLab compute is running; no worker, service, policy override or scene edit is started by this checkpoint.
- Accepted evidence remains the preview-only `walking-entry-gpu-preview-01/` and private `walking-entry-ui-dist-2026-09-08-01`: four rendered fixed-body walks, 339 frontend tests and independent image/source audit pass. Threshold review and this recipe's real apply/reload/undo remain incomplete. Review `/home/rtoony/reports/2026-09-06-splatlab-next-phases/walking-entry-2026-09-08.md` before resuming; do not relabel preview dismissal as transaction undo.

## 2026-09-09 01:19 UTC — first owner-clearance blocked audit, no new work started

- **01:20 UTC second consecutive audit:** previous turn was no progress. Read-only host status confirms the identical lease, elapsed new-start cutoff and unchanged expiry; Nexus still ends at release message 334 and scene generation 48 is unchanged. No owner decision or fresh GPU authorization has arrived. The sandbox socket check failed, then the host status check succeeded; this is not evidence of a stopped service or a reason to restart it. This turn also makes no substantive progress; leave the Goal active until the third consecutive audit if the same impasse remains.
- The preceding walking-entry turn made concrete progress. This continuation is a **no-progress / first blocked-audit turn**, not a verified wait: no owned worker remains live, and no new GPU launch is allowed after 18:12:49 PDT. The existing RAM override still reports its unchanged 18:27:49 expiry; it is not renewed.
- Read-only review confirms the product already reports unassigned capture impact and explicitly distinguishes named-object gates from unnamed furniture preservation. Adding another generic warning or rerunning synthetic tests would not resolve this candidate's known tripod/box conflict or prove the pending real transaction. No duplicate protection feature is added.
- Current private scene is still generation 48, revision `scene_436ca21966e33dd3920b228f`. The selected recipe's actual apply/reload/undo remains unproven. A preserving alternative would also need fresh geometry and rendered validation; no permission-free completion path is established under the closed compute window.
- Ask the owner whether this private doorway may replace the tripod/stacked boxes left of the bonsai, or must instead move to preserve them, and request a new bounded GPU window for the remaining real checks. Do not infer either decision from automatic Goal continuation. Keep the full Goal active; the three-consecutive-blocked-turn threshold is not yet met.
- Feedback discovery returns HTTP 401; this shell lacks injected `PORTAL_TOKEN`. No credentials are exposed/fetched/persisted, and no feedback queue contents or approvals are inferred. Nexus 334 remains the terminal resource-release message. No source behavior, scene, media, service or policy is changed in this audit.

## 2026-09-09 01:13 UTC — real UI walking entry passes private GPU preview

- New per-connection “Start at captured side” and “Start in extension” controls use revision-pinned navigation and fresh current-collider admission with the user's explicit body. Stale/inspection/draft states disable entry; failed final admission restores the camera. Entry is explicitly repositioning, never counted as walking.
- `walking-entry-gpu-preview-01/` is terminal exit 0 with a passing independent audit: all four actual rendered traversals now start via the UI buttons, not console camera placement. 3,405 rendered frames / 17,029 physical positions; both fixed body profiles and both directions; no page/shader errors; active generation 48 unchanged. Private build: `data/spatial/walking-entry-ui-dist-2026-09-08-01`.
- New validation: 339 frontend tests / 22 files, TypeScript, private build, targeted lint/format, 93 Node cases and syntax/whitespace pass. Prior full backend remains 2,103 passed / 17 skipped; unchanged backend was not rerun in this milestone.
- Report: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/walking-entry-2026-09-08.md`. Goal remains incomplete: threshold review, unnamed-object protection decision or preserving placement, then this recipe's authorized apply/reload/undo. No activation, deployment or cloud upload occurred.
- GPU command completed before the existing 18:12:49 PDT new-start cutoff; all compute released in Nexus 334. No more GPU starts under this window and no extension of its 18:27:49 expiry. CPU-light documentation/source review may continue; do not treat automatic Goal continuation as fresh resource or object-removal consent.

## 2026-09-09 00:29 UTC — captured wall specks isolated and corrected on GPU

- Same-camera real controls distinguish two errors: unstable normalized-depth unprojection and ordinary 24-bit depth ties at the actual 400,000 near/far ratio. Fragment coordinates now use viewport plus homogeneous W; WorldWalker uses the existing Three/Spark logarithmic-depth path. No cut enlargement, camera clipping-plane change, geometry change or hidden player shrinking occurs.
- The 18-PNG comparison actually passes with exact old/new restorations. Old controls reproduce earlier PNG hashes; projection alone retains the specks, while logarithmic depth removes the isolated wall/jamb components. Three small source-floor/threshold components remain documented, not claimed pixel-perfect.
- The new private renderer passes actual walking and independent image/source audit: 3,405 rendered frames / 17,029 physics positions, four traversals at both fixed body profiles. Active scene remains generation 48, unapplied. New private bundle: `data/spatial/precise-depth-ui-dist-2026-09-08-01`.
- Current checks: new full backend regression 2,103 passed / 17 skipped / four existing warnings in 119.80 seconds; 320 frontend tests, 45 focused Python proof tests, 93 individual Node cases, TypeScript, private build, lint/format/syntax pass. The first requested comparison ran ordinary inspection because an environment flag was dropped; explicit validated CLI admission fixes that, and the incomplete comparison is retained honestly.
- Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/fragment-depth-repair-2026-09-08.md`. Goal remains active/incomplete pending tripod/box protection or preserving placement, user visual review and this recipe's apply/reload/undo. All resources released in Nexus 325; expiry and new-start cutoff unchanged.

## 2026-09-08 23:57 UTC — joined-room GPU preview and authored seam repair verified

- Actual NVIDIA preview `joined-room-gpu-preview-04/` and independent audit both finish exit 0. Candidate `architecture_a104d911bfc7447686edb208` passes 12 geometry gates and four actual traversals, outward and back at two fixed body sizes: 3,406 rendered frames / 17,034 physical positions. Twelve same-pose hidden-scene/exact-restored controls verify real scene contribution without rejecting plain walls for having few colors. Attempts 01–03 remain failed evidence, not relabeled passes.
- The new 15-frame appearance inspection also finishes exit 0. Authored jamb/lintel stripes are gone and inspected floors remain clean, including with captured appearance disabled. Small capture-dependent entrance specks remain; this is not whole-scene photographic acceptance.
- Current backend regression: 2,102 passed, 17 skipped, four existing warnings. The unchanged frontend bundle has 312 passing tests and a passing TypeScript/Vite build. Node proof/diagnostic/control packets pass 92 individual cases. Source and original artifacts remain intact; active scene stays generation 48 and no edit is applied.
- Review `/home/rtoony/reports/2026-09-06-splatlab-next-phases/connected-space-visual-review.html`; detailed evidence and failures: `joined-room-envelope-2026-09-08.md` beside it. Next: resolve tripod/stacked-box protection or preserving placement, diagnose residual capture-edge marks, then this recipe's own apply/reload/undo proof. Goal remains active/incomplete.
- All resources released to other sessions in Nexus 313. Existing 18:27:49 PDT GPU expiry and 18:12:49 new-start cutoff stay unchanged; no service/deployment/cloud/policy change. Condo inputs remain local with no removable-drive dependency.

## 2026-09-08 23:49 UTC — joined room built; render audit correction staged

- New candidate `architecture_a104d911bfc7447686edb208` actually builds in 1.426 seconds and passes 12/12 geometry gates with 656 authored triangles. The full pre-audit-change backend suite passes 2,083 tests / 17 skipped, frontend 312, TypeScript and private Vite build. Scene generation 48 remains untouched.
- First GPU attempt stops on a stale hardcoded triangle count; that verifier now binds to the sealed master. Second attempt completes browser walking but its enclosing command fails the old color-diversity audit on genuinely flat walls. Neither failed command is relabeled an end-to-end pass.
- A same-frame hidden-scene / exact-restored-image control replaces that walking heuristic, without camera, body, geometry or material changes. Focused synthetic tests pass; the latest complete regression and a fresh controlled GPU preview remain next after Manifold's short resource slot. Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/joined-room-envelope-2026-09-08.md`.
- Preview-only remains mandatory pending the unnamed tripod/stacked-box decision or a preserving placement. No new recipe apply/reload/undo, deployment, cloud upload or overall Goal completion is claimed.

## 2026-09-08 23:20 UTC — joined-room boundary source and tests ready

- Returning from the condo input pilot to the actual indoor Goal. New opt-in `joined-floor-envelope/v1` removes overlapping authored faces, retains both16mm floors and all clipping/body/protection reservations, and exports flat face normals. Topology tests additionally expose edge-only roof/wall contact; the new roof covers the wall tops within the existing reserved frame. This adds explicit corner-cap material, not an exactly volume-preserving retessellation.
- New deterministic1nm boundary construction has27solid tests; the latest architecture/solid packet passes148non-API cases and separate approved API tests pass6. Earlier broader packet181passes overlaps these counts.20frontend/83individualNode cases, TypeScript, targeted lint/format and syntax checks pass. The additional `authored_room_watertight` gate supplements all existing checks.
- No new real candidate or GPU proof yet. Manifold's final full regression remains live and holds the CPU window; wait for explicit terminal release before the prepared bounded build and private render. Source/report checkpoint: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/joined-room-envelope-2026-09-08.md`.
- Unnamed tripod/stacked-box replacement approval is requested; keep preview-only until a clear answer or a preserving placement. Scene48, source captures and existing artifacts are unchanged. Goal remains active/incomplete; existing18:27:49PDT GPU expiry is not extended.

## 2026-09-08 23:04 UTC — first condo camera reconstruction actually succeeds

- The coordinated CPU-only pycolmap4.1 probe is terminal **exit0**,6.532seconds: **all24training images join one sparse model**, with12views from each lens and all12training timestamp pairs represented. It estimates1,9493Dpoints, mean track length4.145 and training reprojection error0.555pixels. This is not held-out quality, metric scale or final optical/rig calibration.
- Independent read-only SQLite/model checks confirm the exact training set and **all16held-out images absent**. Source-preserving preparer/diagnostic now pass **44 focused tests**; syntax/whitespace checks pass. No full-suite pass is claimed for the earlier interrupted combined command.
- First pilot data:40raw-lens images,20pairedgroups,0–19.019seconds. Local visual review: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/condo-input-visual-review.html`. Full results, hashes and next gate: `condo-raw-lens-pilot-2026-09-08.md` alongside it.
- CPU slot released to Manifold in Nexus291. No owned worker remains, no GPU computation/cloud call/install/deployment occurs, and indoor scene48 is unchanged. Next: assess the estimated physical-lens relation and localize held-out views against frozen training geometry before undistortion or Gaussian training. The indoor Goal remains active/incomplete; no replacement or expiry extension.

## 2026-09-08 22:51 UTC — local condo raw-lens pilot actually decoded

- Owner's explicit “proceed” ends the input-discussion pause. The removable SSD is no longer used. Clip003 independently rehashes to its original handoff identity; registration retains934 GPS fixes with unverified clock alignment.
- New bounded raw-lens preparation actually exits0: **40 images /20 paired timestamp groups**,1536-square,0–19.019seconds, in59.857seconds. Integer container PTS match exactly across both lenses; no stitch, resampling or automatic rotation is applied. Exposure synchronization and calibration remain unverified.
- New source-preserving CLI preparer and training-only COLMAP4.1 diagnostic have **42 focused synthetic tests passing**. Twenty-four geometric feature masks are prepared and their contact sheet reviewed; these are not semantic or approved Gaussian-training masks. No actual SfM/Gaussian model exists yet at this checkpoint.
- CPU resources are explicitly released to Manifold for its short browser packet; the small registration probe waits its next terminal handoff. No GPU computation, cloud upload, SDK install, deployment or indoor-scene change. The indoor Goal remains active/incomplete and the existing GPU expiry remains18:27:49 PDT.
- Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/condo-raw-lens-pilot-2026-09-08.md`; usage in `docs/spatial-development.md`.

## 2026-09-08 21:34 UTC — owner selects condo exterior capture; handoff review only

- Owner chooses Terra's new condo exterior videos/data. Read-only review locates eight original dual-fisheye INSV clips (~10.3minutes), eight LRV companions,20 actual1920×960 review frames and six coarse GPS tracks. Raw byte sizes match the manifest; full raw hashes and SSD completion are not independently reverified here.
- Four removed provisional panorama files remain listed in both observed24-derivative manifests. Nexus261/263 requests Terra's final handoff and correction without overlapping its write scope. Stitches show visible seams; one reviewed clip is inverted. Review panoramas are not calibrated training inputs, and GPS is not solved camera pose or survey scale.
- No GPU, package modification, reconstruction or upload. Discussion now selects a bounded condo target and original-quality camera/projection path. The indoor Goal and its unresolved failures are preserved, not silently replaced or marked complete. Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/condo-input-review-2026-09-08.md`.

## 2026-09-08 21:29 UTC — owner pauses for better input-data discussion

- Owner asks to discuss better videos/images before continuing. No new GPU/build/cloud task starts. The untested seam-union draft is withdrawn, preserving the previously validated paired-floor paths; scene48 remains unchanged. All SplatLab resources are released in Nexus260; Manifold's separate work is not interrupted.
- Discuss existing user-owned captures, desired scenes and available devices before choosing the next dataset. Keep the current cluttered capture as regression evidence rather than erasing its failures. No data is selected, downloaded or uploaded yet.
- Checkpoint: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/input-data-discussion-checkpoint-2026-09-08.md`. The full Goal is incomplete; user should use `/goal pause` while discussing inputs. Do not simulate pause by marking complete/blocked. Existing GPU expiry is unchanged.

## 2026-09-08 21:05 UTC — actual paired-floor preview passes; seams remain

- Owner resumes the focused doorway plan. New `architecture_cd439008c5aa4370a0138e5b` builds the opt-in paired 16 mm floor finish and passes **11/11 geometry gates**, then the actual NVIDIA preview-only proof **exit 0**: four fixed-body traversals, **3,409 rendered frames**, independent verification of **86 artifacts**, exact preview dismissal, no page/shader errors. Recipe selection/reuse is browser-verified.
- Fifteen fresh appearance comparison PNGs show both floors clean in inspected views. Thin authored jamb/lintel seams remain without captured splats; source identifies overlapping coplanar faces as the next target. This is not final appearance or a new apply/reload/undo acceptance. Unnamed tripod/boxes remain preserved by default; scene48 never activates or changes.
- **2,000 backend passed /17 skipped /4 existing warnings**, **311 frontend**, **82 Node**; TypeScript/private build/syntax/whitespace pass. No model install, cloud call or deployment. All compute is terminal and released to Manifold in Nexus253; independent GPU expiry remains18:27:49 PDT, no new starts after18:12:49.
- New evidence and next work: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/paired-floor-rendered-findings-2026-09-08.md`; the static visual review includes the new floor comparison. Persistent Goal still records `blocked`; owner should enter `/goal resume` in this session for automatic continuation, without replacing the objective or extending permissions. This user-directed turn has made progress; the full objective remains incomplete.

## 2026-09-08 20:46 UTC — technology review; experimental floor finish source checkpoint

- Owner requests review of recent splat technology; new mesh/GPU work is deferred in Nexus239. Research and proposed priorities: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/technology-review-2026-09-08.md`.
- Actual24-image floor diagnostic completes without activation. A16mm bridge lift removes bridge speckles but not room-floor speckles; visual-only broader masking is not accepted. Wall seams remain.
- New opt-in `raised-floor-finish/v1` moves both authored slabs/collision together, keeps legacy nominal geometry as default, binds method evidence and exposes explicit API/CLI/UI selection.187 focused backend,19 frontend and78 Node checks pass; type/lint/format pass.
- No new floor-finish candidate, frontend bundle, GPU walking or transaction proof has been produced. Original capture and scene48 remain unchanged; unnamed tripod/boxes remain unapproved for replacement. Details: `floor-finish-source-checkpoint-2026-09-08.md` in the same report directory. No new model install, cloud spend, live deployment or goal-completion claim.

## 2026-09-08 19:44 UTC — fresh room-envelope GPU preview passes; floor review continues

- Fresh `architecture_2c7b4f81200f4c1b9eb932c3` passes11/11 actual geometry gates and the real NVIDIA preview-only command **exit0**, including independent audit. Four fixed-body keyboard traversals total **3,411 rendered frames**; 12 static/12 walking PNGs, four CPU/four rendered traces and86 artifacts verify. Scene48 never activates or changes.
- Actual new impact:13,722 unique captured centers inside cut/room, zero named protected/included; unassigned is not permission to remove objects. Tripod/box decision remains separate.
- Product room masking/lighting now really renders, but floor speckles and thin mesh seams remain visible. A bounded raised-bridge versus mask-only diagnostic is source-ready with **nine Node tests passing**, not yet run. No floor recipe or acceptance gate is changed.
- Evidence and exact scope: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/room-envelope-rendered-preview-findings.md`. SplatLab releases the resource slot; Manifold takes the next CPU packet. GPU authorization still expires18:27:49 PDT.

## 2026-09-08 19:33 UTC — owner authorizes resumed bounded GPU development

- Owner explicitly authorizes SplatLab and required services. RAM-only SplatLab policy override is active, other capabilities/persistent policy unchanged; independent expiry timer is verified for **18:27:49 PDT /01:27:49 UTC**. No new starts after18:12:49 PDT.
- Manifold remains CPU-only; finish its current regression before the next bounded SplatLab proof slot. No GPU proof starts at this checkpoint. Scene48 and original captures are unchanged; tripod/box replacement permission remains separate and unanswered.
- Scope, verified control state, eleven helper tests and cleanup: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/renewed-gpu-window-2026-09-08.md`. The full demonstration is still incomplete; work resumes under this authorization without weakening any acceptance gate.

## 2026-09-08 14:27 UTC — execution blocked pending policy permission and backup release

- Three consecutive goal turns confirm the same SplatLab CUDA hold; available source/test/review packets are retained, but the required fresh real render cannot proceed. Full objective stays unchanged and incomplete; execution is blocked, not accepted or finished.
- The hold is explicit in `~/.config/nexus-gpu/policy.json`; the running orchestrator reads it at startup. A targeted policy change and controlled orchestrator restart need approval, plus a bounded expiry/cleanup plan. No policy, service or maintenance marker is changed.
- Backup PID3695520 is confirmed live; do not restart or interrupt it. No owned worker remains and scene48 is unchanged. Preserve the tripod/boxes pending a separate decision. Details and precise resume prerequisites: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/room-envelope-and-coordination-findings.md`.

## 2026-09-08 14:10 UTC — room-envelope source ready; GPU policy and object review hold

- **14:20 source-only follow-up:** sealed worker output now records unique cut/room/protected/included/unassigned Gaussian-center counts and retained row memberships; UI distinguishes them from object counts and permission to remove furniture. **20 volume /14 guidance tests**, TypeScript and targeted checks pass. No fresh real build or GPU run; staged Vite bundle predates this follow-up. Private visual review: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/connected-space-visual-review.html`.
- Versioned cut-and-room appearance, authored-only lighting, named Gaussian-center protection and a whole-room collision-interior gate are implemented. Fresh `architecture_bf569438f71448a6a362f77c` passes **11/11 actual local gates**, but is not activated or newly GPU-verified.
- Source photos confirm the cut intersects an unnamed tripod/stacked-box area. Preserve it unless the owner explicitly allows private replacement; passing five named-object checks is insufficient.
- **1,945 backend passed / 17 skipped; 297 frontend / 62 Node passed**. Thirteen subsequent tamper tests bring the focused backend suite to **145 passing**; no second full suite after those test-only additions.
- Manifold agrees to CPU/software rendering through Nexus messages 142/143. Live typed policy denies `splatlab.cuda`; GPU launches are held and backups are not interrupted. A six-hour extension/policy authorization is requested, not confirmed. Existing window ends **16:01:59 UTC**.
- Private generation48 is unchanged. Preview-only proof mode is staged, not executed; no cloud call, deployment or original-capture change. **Goal active, incomplete.** Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/room-envelope-and-coordination-findings.md`.

## 2026-09-08 01:30 UTC — actual rendered doorway traversal and undo; visual integration open

- Fresh `architecture_fff48045af224ce2b474d697` passes **10/10** gates under the explicitly new `shared-entry-footprint/v1` procedure. Old parallel-route failures remain failed; body/margin/floor tolerances are not reduced.
- Actual NVIDIA keyboard/RAF traversal covers about **7.1 m both ways at 1.70m/0.22m and 2.02m/0.32m height/radius**: 3,411 rendered frames. Shader A/B, paired apply/reload/undo, exact PNG restoration and 390px layout pass. The original command exits 1 on a post-run audit variable-shadowing bug; a corrected independent re-audit verifies unchanged artifacts/sources. Do not call the original command an uninterrupted pass.
- Independent exact capsule/BVH checks at **23,879** saved positions find only the existing 10-micrometre contact offset, none above 20 micrometres. Restored **generation48**, `scene_436ca21966e33dd3920b228f`, has no proposal; state and all 86 artifacts exactly match generation46.
- Visual review still rejects room ghosting and flat appearance. GPU A/B isolates missing authored-room appearance clipping and unlit depth cues; larger clipping is diagnostic only. Unnamed-object protection and threshold remnants remain open. **Goal active, incomplete.** No new cloud call/deployment; all owned workers finish.
- **1,926 full backend / 117 latest focused backend / 288 frontend / 57 Node cases pass** (17 backend skips). Latest private build: `data/spatial/shared-entry-ui-dist-2026-09-08-v1`. Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/shared-entry-rendered-traversal-findings.md`.

## 2026-09-07 21:52 UTC — four-metre vestibule reaches nine geometry gates

- Actual private build `architecture_a6ed01a601c549f980efa139` passes **9/10** unchanged gates, including both floor checks; capsule clearance still refuses, so it is not applied. Actual CPU WorldWalker predictions traverse the center route approximately **7.1 m both ways at both fixed body profiles**. This is simulation, not GPU/input proof; several side starts still refuse.
- Source/UI/proof contracts support explicitly authored **0.1–6 m vestibules**, not measured wall thickness. **79 focused backend / 288 frontend / 33 individually executed Node tests pass**, plus TypeScript and targeted lint/formatting. Previous full-backend results remain historical; new UI is not yet built or browser-proven.
- Original photos and nearest contacts distinguish threshold/floor fragments from genuine unnamed chair/bag context. Adaptive semantic/floor pilots improve some metrics but do not establish an accepted repair. No gate is weakened or candidate promoted.
- Baseline46, original captures and cloud holds are unchanged. All known owned workers finish at the requested update break; a flush-threshold diagnostic is prepared but unrun. Goal remains active. Details: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/vestibule-and-floor-investigation-findings.md`.

## 2026-09-07 18:07 UTC — real fixed-size walking and causal floor-clearance diagnosis

- Studio now admits explicit physical body dimensions against the actual BVH; it does not silently shorten the player using the source-height estimate. Dimension/scale/revision changes stop walking. Mouse-capture errors are actionable and Escape releases held controls.
- **Actual NVIDIA proof passes** on staged `data/spatial/fixed-walking-ui-dist-2026-09-07-v3`: 24 rendered samples, **0.200098m** W-key movement with the explicitly selected **1.70m height / 0.22m radius**, exact baseline PNG restoration, unchanged source/artifact hashes, refusal controls and 390px layout. This is a short baseline walk, **not the doorway/extension goal**. All failed attempts are retained.
- Floor pilot02 loses 247 and gains 81 sampled free centers. Nearest surfaces identify **127 retained-observation-box cases and 119 hard patch-edge cases**, with no occupied body centers. Its improved floor residual is not navigation acceptance. The semantic check's 59 observations all land in view108, whose floor/rug detector returned no masks; view183 has no supported-patch observations. No protection or threshold is weakened.
- **1,887 backend passed / 17 skipped; 283 frontend passed; 28 Node contract tests passed**. TypeScript, targeted lint/formatting, syntax, whitespace and private build pass. Baseline46 remains unchanged; all owned workers finish; no new cloud spending, deployment or original-data change.
- Detailed evidence and next work: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/fixed-walking-and-floor-clearance-findings.md`. The connected-space goal stays **active and incomplete**, with current compute authority through **2026-09-08 16:01:59 UTC**.

## 2026-09-07 17:14 UTC — connected-space goal; actual geometry and floor diagnosis

- The user sets a focused goal: a real, privately staged captured-room → doorway/connected-space demonstration with actual walking, protected objects, synchronized appearance/collision and exact reload/undo. **Active, incomplete**; the renewed compute window continues through **2026-09-08 16:01:59 UTC**.
- Real architectural builds expose and fix cross-runtime room-coordinate determinism and malformed generative export metadata. Actual container/west-facing candidates still fail clearance/floor checks; neither is applied. Protected bounds now distinguish the narrow tunnel from the full room, with complete geometry-enclosure tests.
- Added actionable failure guidance and strict all-gates preview admission. **1,871 backend passed / 17 skipped; 265 frontend passed; 28 Node proof-contract tests passed**. TypeScript, targeted lint/formatting, syntax, whitespace and a new private build pass. Staged UI: `data/spatial/architectural-feedback-ui-dist-2026-09-07-v2`; mounted GPU proof remains pending.
- A source-bound tilted floor fit has **32,872 inliers / 8.21 mm RMS**, but both real repair pilots worsen sampled clearance despite better floor residuals. A 15.069-second installed local SAM floor/rug/table study has contradictory held-out masks. No pilot, closure setting or semantic model is promoted; no acceptance gate is weakened.
- Baseline remains **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, no active proposal. All owned workers finish; no cloud call, installation, deployment, original-data change or manual service pause occurs. Prior cloud holds stay **$7.013210 / $20**.
- Findings, retained evidence and next substantive work: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/connected-space-goal-checkpoint.md`. Next: explicit fixed-size navigation admission, actual collision/floor correction and complete paired geometry, then NVIDIA traversal and transaction proof—not more broad model benchmarking.

## 2026-09-07 16:01 UTC — explicit 24-hour development resumption

- The operator renews SplatLab work through **2026-09-08 16:01:59 UTC / 09:01:59 AM Pacific**, superseding the expired-compute blocker below. Resume actual bounded geometry/GPU validation, not the retired hardware/Flight A ladder.
- The original aggregate **$20 cloud allowance is not exhausted**: nine returned calls hold **$7.013210**. No increased total, new paid call or expired-ledger mutation is inferred. Prefer installed local capabilities; preserve budget evidence and all resource safeguards.
- First priority: fresh generation-bound architectural preparation, actual contained Boolean/room/navigation gates, then private NVIDIA shader/capsule/apply-reload-undo proof if those gates pass. The full phase 0–6 plan remains incomplete; previous validation results remain historical until rerun.
- Work-window authority and cleanup deadline: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/extended-work-window.md`.

## 2026-09-07 — staged architectural acceptance; renewed compute authorization required

- Added a private-only real-shader/capsule/apply-reload-undo harness with current geometry preflight, immutable input capture, owned rollback and independent saved-image/trace/source/artifact verification. Unsafe modes and reused output directories refuse. The harness is source-staged, **not run on the captured scene**.
- Actual production capsule/BVH physics passes synthetic open-doorway tests in both directions at **0.22m/1.70m** and **0.32m/2.02m** radius/total-height profiles. Closed walls block passage; insufficient headroom refuses rather than auto-shortening the capsule. This is not real captured-room navigation acceptance.
- **1,861 backend passed / 17 skipped / four existing warnings (108.93s); 260 frontend passed; 28 Node proof-contract tests passed**. TypeScript, targeted lint, syntax and whitespace pass. Application source/build is unchanged from the positioning tranche; no deployment occurs.
- Actual preparation remains unbuilt and stale under membership hardening. Baseline **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, is unchanged; all owned workers are stopped. Cloud remains expired with nine returned calls and **$7.013210 held**, not invoiced spend.
- The same expired compute boundary persists through three consecutive development checkpoints. No more source-only substitutes establish actual Boolean, exported-geometry, WebGL or captured-room traversal acceptance. The full phase 0–6 objective is **blocked pending fresh compute authorization**, not complete. No command is queued or scheduled. Resume instructions: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/architectural-acceptance-harness-findings.md`.

## 2026-09-07 — source-only doorway positioning guide

- Added a live dimensioned cut/room/bridge guide and two-click threshold/outward placement. Actual collision-BVH picking refuses walls/undersides; direction picking uses the threshold-height plane. All guides are labelled authored/inferred X-ray aids, not measured floors or completed cuts.
- Drafts bind job/revision/generation, do not enter collision or revision artifacts, and cannot grant review/apply. Escape, failure/retry, listener cleanup, independent ArrowHelper geometry ownership and blocked interaction/walking/scale shortcuts are tested. Numeric edits and picking share one spec.
- **256 frontend tests pass**, including **35 new draft/BVH tests**; TypeScript, targeted lint/format and private build pass. Vite's large-chunk advisory remains. Backend source is unchanged; the preceding **1,843 passed / 17 skipped** result is not a fresh run.
- Staged UI: `data/spatial/architectural-positioning-ui-dist-2026-09-07`. Actual mounted-browser rendering, touch/pointer-lock behavior, real portal geometry and capsule traversal remain unverified pending fresh compute authorization. No heavy work, paid call, source-capture change or deployment occurs; baseline **generation 46** remains unchanged.
- Findings and browser acceptance steps: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/architectural-positioning-findings.md`. The expired overnight allowance is not renewed. Full phase 0–6 remains incomplete.

## 2026-09-07 — paired architectural edit prototype; overnight window closed

- Implemented a dimensioned doorway/connected-room operation that pairs captured collision subtraction, Spark/mesh clipping, explicit included-prop removal, room geometry, semantics and exact revision undo. Preparation is CPU-only; the builder remains manually gated. Authored/inferred frames are never promoted to measured walls.
- Ten local geometry/navigation gates, source/membership binding and activation-time pairing checks refuse stale or mismatched evidence. Native Gaussian layers in either operation order and refined captured partitions refuse pending dedicated portal adapters. Later mesh placements cannot block the retained doorway.
- **1,843 backend passed / 17 skipped / four existing warnings; 221 frontend passed**. Focused architecture/revision/collision suite: **95 passed**. TypeScript, targeted lint/formatting, private Vite build and whitespace checks pass. Synthetic transaction tests do not prove real Boolean output, WebGL shader correctness or capsule navigation.
- Actual preparation `architecture_aae7b684216d49a68bf89a04` is **unbuilt** after VRAM and backup admission refusals, both before worker start. No result/edited geometry exists; all preparation blobs verify. Its older receipt lacks the new selection binding and must be preserved, not reused or re-signed. Actual doorway/browser/traversal acceptance is still pending.
- Active scene is unchanged: **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, no proposal. No owned model/browser workers remain; no original capture, live service, deployment, model default or hardware guard changes. Staged UI: `data/spatial/architectural-edit-ui-dist-2026-09-07`.
- Overnight compute and cloud authorization are **expired**. Nine returned requests retain **$7.013210 in conservative holds**, not invoiced charges; unused funds do not carry forward. No more heavy work or paid calls without a fresh window. Findings and concrete next acceptance steps: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/architectural-edit-findings.md`. Full phase 0–6 remains incomplete.

## 2026-09-07 — captured-wall evidence and local geometry priors (private)

- Added revision-bound semantic wall studies, verified bidirectional photo tracks, fit-only plane extraction and same-ray inferred-depth comparison. Studio now shows original/masked photos, support counts, missing evidence and retained membership. No wall opening or automatic geometry approval is implied.
- Actual SAM run finds only **two multi-view wall points / zero supported planes**. Existing Gaussian depth, installed MoGe and paired-view MASt3R produce candidates, but fit RMS hides poor/missing check support. Original camera/scale lineage stays fixed; inferred wall accuracy is not claimed. Installed-only GPU runs take approximately **0.958 / 7.116 / 8.094 s** respectively.
- Independent CPU audit checks **28 structural / 10 prior artifacts**, separate point-ID scale checks and actual resize/intrinsic contracts. Same-ray checks correct the earlier tangential-footprint interpretation and distinguish occlusion from unsupported inference. All failed attempts remain retained.
- **1,797 backend passed / 17 skipped; 211 frontend passed**, plus TypeScript, targeted lint/format, staged build and syntax/whitespace checks. Actual NVIDIA browser proof passes **16 image decodes**, membership download, reload, 390px layout, zero errors and unchanged active scene/state. Staged UI: `data/spatial/structural-wall-ui-dist`.
- Active baseline remains **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, no proposal. No paid call, install, deployment, source edit or manual service pause. Cloud holds **$7.013210** across nine returned calls. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/captured-wall-findings.md`. Next: a reviewed architectural frame, actual dimensioned portal, connected-room collision/navigation and reversible visual treatment—not more unsupported wall fitting. Full phase 0–6 remains open.

## 2026-09-07 — inferred-depth background qualification (private)

- Added optional revision-bound qualification of captured photo samples against the existing Gaussian model's metric expected depth and opacity. The final GPU worker replays **28 camera arrays exactly**, with fit/check separation, exact texture-grid binding, source/artifact checksums and separately retained recoveries. No new image generation or automatic scene activation.
- Real coverage changes **22.9614% → 11.3525%**: 1,903 previous cells rejected, 1,859 retained, one newly supported after source reconsensus. Delivered geometry changes **7,524 → 3,720 triangles**. Paired delivered-atlas appearance improves on three check views and worsens on one. Studio displays the uncertainty and same-pixel errors; projected book colors still remain, so this is not a clean-floor or ghost-free acceptance claim.
- Independent audit verifies **39 original/derived artifacts**, actual GLB cell membership/positions/embedded texture, fit-only sources and 28 calibrated camera registrations. Final NVIDIA proof passes six context/isolated views with matched A/B poses, four image decodes, exact baseline PNG restoration, 390px layout and zero page errors. Earlier harness failures are retained and diagnosed.
- **1,766 backend passed / 17 skipped; 211 frontend passed**, plus TypeScript, targeted lint/format, staged build and syntax/whitespace checks. Staged UI: `data/spatial/visibility-qualified-ui-dist`. Current compared recovery: `recovery_ea1e16cdaa994b468f442ce5`. Active scene remains **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, no active proposal. No source capture or live deployment changes.
- All workers exit; no manual service pauses. Cloud remains nine returned requests with **$7.013210 reserved** of $20. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/background-visibility-findings.md`. Full phase 0–6 remains open; this is progress, not completion.

## 2026-09-07 — footprint-aware selection and visible residual review (private)

- Added opt-in integrated Gaussian-footprint voting using installed SAM/gsplat, not more cloud generation. It handles extended contributions missed by projected-center/depth tests, retains source/mask/camera lineage and excludes check cameras from votes. Mixed/tiny/occluded footprints abstain; original core and other-instance boundaries remain protected.
- At the same conservative **5 cm** margin, candidate rows improve **469 → 628**; check-view mask alpha coverage improves **59.57/56.15% → 70.80/71.34%**. A distinct **710-row / 20 cm** candidate correctly fails named-instance collision clearance. No clearance threshold is reduced or neighboring object implicitly removed.
- Fresh 628-row collision passes all eight local gates in **3.510 s**, clears all selected centers/interior samples and preserves sampled outside surfaces exactly. This is local clearance, not whole-scene navigation or complete-object acceptance.
- Studio adds **Show remaining captured appearance**. The review-only diagnostic preserves existing removals and disables walking/proposals/apply. Actual 960×640 NVIDIA UI proof passes ten selected/framed/residual views, eight image decodes, exact baseline pixel restoration and zero page errors. Audit verifies **65 selection / five collision artifacts**. A retained earlier proof failure is fixed by awaiting Spark's scheduled sort, not weakening image checks.
- **1,742 backend passed / 17 skipped; 211 frontend passed**; TypeScript, targeted lint/formatting and staged build pass. Staged UI: `data/spatial/footprint-selection-ui-dist`. Active scene stays **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, no active proposal. Fresh selection `selection_c52c16ec23cb4d98ad8201e3` and collision `collision_05f693d7cdb1487fa8ef2960` remain available.
- Actual residual imagery still contains ghosts and a separate cup. More complete selection helps but does not solve hidden-background or support dependencies. No paid calls, installs, deployment, source-capture changes or manually paused services in this tranche. Nine cloud calls still hold **$7.013210** under the **$20** ceiling. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/selection-footprint-findings.md`. Full phase 0–6 remains open.
- Investigated SAM's four missing neck parameters: the image adapter discards that fourth feature level. A two-fit-frame in-memory intervention changes its actual features while detector outputs stay identical. Initial ineffective interventions are retained; disabling autocast's weight cache only for the causal test resolves the audit issue. The installed checkpoint remains unchanged. This explains that specific warning, not general model compatibility or mask quality.

## 2026-09-07 — actual cloud materials in the captured room (private)

- Kimi K3, Gemini 3.8 Flash and GLM-5.3-Flash inspect the same two approved fit-only reference crops. Gemini Flash Image and Pro Image generate two actual 1024px floor materials. This is task-specific local/cloud comparison, not a new model default; Kimi's ambiguous cardboard suggestion is rejected rather than blindly executed.
- Fixed PNG-only cloud result extraction for real JPEG responses. Original responses/images remain unchanged; hash-verified `collect` recovers local artifacts without another API call. Nine requests retain **$7.013210** of the **$20** ceiling; usage-priced estimate is approximately **$0.247**, not a provider invoice. No further image generation is needed for scene-bound retries.
- Both GLBs preserve the original observed atlas, **7,524 observed triangles/UVs** and **25,244 disjoint generated triangles**. Studio exposes advisory boundary disagreement rather than a misleading hidden-geometry quality score. Pro matches observed colors more closely; neither candidate is automatically selected or accepted.
- Actual NVIDIA browser A/B fixed/moving/isolated/collision review passes; Pro's private transaction hides **469 rows**, changes collision **250,108 → 280,610**, reloads and undoes **44 → 45 → 46**. Exact canvas PNG reload/undo, exact state/artifact-map undo, 390px layout and zero page errors pass. Three earlier failed attempts remain retained; rounded-corner screenshot composition is excluded by direct canvas encoding, not by loosening pixel tolerance.
- Final independent artifact audit **PASS**. Baseline restored at **generation 46**, `scene_7b39a3e4b6f9b450401cc20b`, no active proposal. Earlier generation-bound studies are stale. **1,706 backend passed / 17 skipped; 209 frontend passed**, plus TypeScript, targeted lint/formatting and staged build. No live deployment, original-capture changes or manually paused services.
- **Appearance acceptance remains open:** fuzzy residual captured imagery, the still-visible cup/support dependency and potentially occluder-contaminated projected texture need review. Better texture generation does not fix incomplete object selection. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/background-model-findings.md`; full phase 0–6, 1080p performance/navigation and connected architecture remain unfinished.
- Subsequent read-only **1080p baseline measurement** displays all 605,391 captured rows and measures **60.002/60.003 FPS**, P95 **16.8/16.7 ms**, with zero slow frames in two 360-frame stationary/moving scenarios after 60-frame warm-up. Actual application frame path, DPR1/NVIDIA identity, pose/row samples and valid GPU timer queries are retained. Baseline 46 stays unchanged. This meets the scoped baseline timing target, not native-replacement/large-route performance or navigation acceptance; see `spatial-performance-findings.md` in the same report directory.

## 2026-09-07 — overnight native room and cloud preparation (private)

- Native generated appearance and inferred mesh collision now share an atomic revision identity. Fresh selection/support/clearance and fitted placement reuse only byte-identical native inference; full **572,960-row** frame checks and color-correct mesh delivery remain enforced. Native/mesh visibility, cleanup and paired removal have regression coverage.
- Latest full source suites: **1,698 backend passed, 17 skipped; 209 frontend passed**. The private NVIDIA browser proof now passes paired removal/reload, exact native-state/pixel undo, exact baseline-state/pixel undo, zero page errors and 390px layout. Final independent artifact audit passes. Baseline restored at **generation 42**, `scene_8a5fa7829b3e7906fc1c15f3`; generation-26 preparation remains stale for new proposals.
- Installed Chromium uses actual NVIDIA RTX 5090 WebGL2 through ANGLE/Vulkan. One-frame captures disagreed even after sort settling; a render/update warm-up before readback fixes exact comparison without relaxing pixels. Failed attempts remain retained. This proves deterministic settled frames, not 1080p FPS, smooth navigation or good generated-object appearance; full-row scene totals **1,178,351** captured/generated splats.
- User authorizes an eight-hour window through **06:55 AM Pacific** and **$20 total** across existing Vault-backed cloud APIs. One non-resettable ledger expires at **06:40:25 AM Pacific** with per-request margins, conservative pre-call reservations and no automatic retries. **22 focused cloud tests pass**. Gemini 3.8 Flash, GLM-5.3-Flash, GLM-5.3 and **Kimi K3** answer the synthetic spatial check numerically correctly; Kimi uses nested column arrays, reinforcing schema validation. No project image has been uploaded at this checkpoint.
- Prefer local when adequate, but use better cloud models when they improve results; keep capability adapters replaceable for future local models. No broad default promotion, new installation, unrelated service stop, production deployment, original-capture edit or Flight A work. Read `/home/rtoony/reports/2026-09-06-splatlab-next-phases/extended-work-window.md`, `native-room-findings.md` and `cloud-model-and-budget-findings.md` before continuing.

## 2026-09-06 — native generated Gaussians and corrected mesh color (private)

- Solved the retained SAM-3D native-splat frame from actual SDK source, rather than bbox fitting. Full **572,960-row** export transforms positions, anisotropic covariance rotations and log scales; DC/opacity fields stay byte-exact. Six real GPU camera-invariance checks pass; mesh/splat silhouette IoU **0.9777–0.9849** supports the decoder-frame mapping.
- Added immutable Gaussian review/download APIs, 30 evidence images and interactive Spark/mesh comparison. Studio now pauses offscreen room rendering and resumes onscreen; two-angle comparison, all-row import, disposal, mobile width390 and zero page errors pass on the staged server.
- Fixed generated-mesh brightness at its export source: explicit model image-sRGB interpretation becomes float32 linear glTF color, retaining original mesh/color copies and geometry bytes. Actual two-angle browser mesh/splat MAE improves **~75→~8 RGB bytes**, with unchanged native screenshots. This is not measured albedo, relighting or universal renderer parity.
- Final study `gaussians_f8957134b77549b791851534` takes **1.987s** with cached kernels, versus **105.061s** initial JIT-inclusive run. GPU Torch allocation/reservation peaks **165/191 MB**. One default 24GB admission is refused; measured-small replay uses the existing 4GB per-command setting, with global guards unchanged. No new model inference or installation.
- **1,656 backend tests pass, 17 skip; 205 frontend tests pass**; TypeScript, targeted lint/formatting, staged build and syntax/whitespace pass. Existing unrelated full-lint error remains. Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/generated-gaussian-findings.md`.
- This is a **historical native-layer review/export**, not a room application. Private generation **26**, `scene_bc12c389991bf6a84234e0cb`, remains unchanged. Fresh revision-aware Gaussian/mesh/collision transactions, unknown background, connected architectural edits, navigation and other roadmap acceptance remain open. All workers exit; original captures/live services are untouched.

## 2026-09-06 — actual local generated replacement (private, not promoted)

- New revision-bound SAM-3D workflow prepares one masked source photo, retains **654,364-triangle mesh / 572,960-Gaussian native masters**, fits the mesh to multi-view/support evidence, and derives a separate **32k-triangle** delivery. Real inference takes **14.2s**, with **21.9s** model load. Native Gaussian placement is unsolved; the actual scene replacement uses the generated mesh.
- Fixed scaled-extrinsic ray origins; retained-master rederivation avoids repeating inference. Four-fit-view mean mesh/mask IoU **0.603739** and two check IoUs **0.673693 / 0.528139** admit review, not measured-shape acceptance. Twelve independent actual-GLB reload renders match retained depths, hit masks and RGB exactly.
- Studio offers masked input, six reference/master/delivery comparisons, receipts and master downloads. Generated replacement plus observed background share one reviewed revision. Browser testing catches and fixes missing generated collision/photograph visibility and discarded vertex colors; regression tests cover the actual shared walker paths.
- Successful private retry hides **469 rows**, adds **32,000 + 7,524 triangles**, and changes collision **250,108 → 287,366**. Fixed/moving/isolated/wireframe views, **19 image decodes**, apply/reload/undo **24 → 25 → 26**, exact state/artifact restoration and pixel-identical undo pass; zero page errors, 390px mobile width. Final revision **`scene_bc12c389991bf6a84234e0cb`**, no active proposal.
- **1,622 backend tests pass, 17 skip; 205 frontend tests pass**. TypeScript, targeted lint/formatting, staged build and syntax/whitespace pass. Full frontend lint still fails on unrelated `world-game.ts:18` unused `cellAt`; existing warnings remain. All workers exit. No source capture, live service, model default, cloud-upload policy or hardware guard changes.
- Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/generated-object-findings.md`; commands: `docs/spatial-development.md`. Unknown background remains **77.0386%**, nearby fuzzy content and whole-scene navigation remain unresolved. Generation-24 studies are stale after undo26; use fresh evidence for another edit. Actual background completion, splat registration and connected architectural editing remain substantive next work; the full roadmap is not complete.

## 2026-09-06 — compound authored replacement (private, not promoted)

- Studio now places a Blender replacement and a distinct observed/invented background in one reviewed revision, including rows, collision, registries and semantics. Same-object/current-evidence guards and exact undo remain; candidate-only inspection shows both assets without approving context.
- Fixed a real spatial bug: quaternion-mode glTF imports ignored typed Euler rotations. Explicit Euler actions now select XYZ mode; inspection reports effective rotation. Five opt-in headless Blender tests pass, including actual exported-bounds verification.
- Fresh private 469-row selection/local clearance plus four-anchor recovery prepare an authored chest with **728 unchanged triangles** and **0.177 µm** maximum numerical export error. This is local library set dressing, not new model inference; the background remains only **22.9614% observed**.
- Two fixed views, six source-camera frames, four interpolated frames, two-asset inspection and collision wireframe pass in the browser. **250,108 → 256,094** collision triangles; apply/reload/undo **22 → 23 → 24** restores exact state/artifact maps and a pixel-identical PNG. Zero page errors; mobile width **390px**. Current private revision `scene_a705529604585ecccfbe4b9e`, no active proposal.
- **1,597 backend tests pass, 17 skip; 198 frontend tests pass**; TypeScript, targeted lint/formatting, staged build and syntax/whitespace pass. Failed preparation/browser attempts are retained. All workers terminate. No original capture, production service, cloud-upload policy, model default or hardware guard changes.
- Read `/home/rtoony/reports/2026-09-06-splatlab-next-phases/compound-replacement-findings.md`. Disconnected observed texture islands, unknown background, nearby fuzzy captured content and navigation are visibly unresolved. Actual local reference-guided generation and revision-aware generated-object admission remain next; fresh evidence must bind generation **24**. The full roadmap remains open.

## 2026-09-06 — training-only inferred surfaces (private, not promoted)

- Added frozen TSDF surface studies from the three retained models: exact config/checkpoint/coordinate checks, all **243 training views**, **25 held-out views per method**, masks/opacity/crop/depth filters, retained samples and failure-stage lineage. No new training or scene mutation.
- Raw Splatfacto/DN-Splatter/AGS-Mesh outputs retain **4,234,654 / 3,228,887 / 3,547,886 triangles** and **134,823 / 63,605 / 95,548 components**. AGS gains **1.093 dB** crop-masked mesh photo PSNR over baseline but loses **1.47 percentage points** of crop-ray hit coverage. None is accepted as a clean architectural mesh or collider; model-depth agreement is not measured geometric accuracy.
- **75 native checkpoint images match exactly**. Independent exported-PLY checks reproduce **75 view hit masks and depths exactly**, with at most one RGB byte difference. All raw components/double positions and portable inferred-geometry survey refusal remain intact.
- The first raw-mesh cap failure is retained and diagnosed through sample replay; a fresh equal-recipe study preserves voxel detail with bounded output limits. The first gallery decode failure is retained; synchronized retry verifies **175 images**, zero page errors and **1440/390px** desktop/mobile layout.
- **1,580 backend tests pass, 16 skip**, including **35 new surface tests**; syntax/whitespace checks pass. All workers exit. Original data and private generation **22** remain unchanged; no model download/retraining, cloud upload, deployment, default promotion or hardware-policy/Flight A changes.
- Findings and actual review paths: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/surface-comparison-findings.md`; commands: `docs/spatial-development.md`. Advance the complete creative edit with separately fitted structure rather than automatically adopting raw TSDF geometry. The full roadmap remains active.

## 2026-09-06 — actual paired reconstruction and lossless review (private, not promoted)

- Added frozen reconstruction preparation, guarded fresh training, strict comparison and full-SH export/review commands. All **292 Bonsai photos** remain in a **243/24/25** train/validation/test split; initial colors use only training observations. No original checkpoint warm-start or active-scene mutation.
- All three methods finish **10,000 steps** and **25 held-out renders**. Splatfacto: **25.0866 dB / 0.832302 SSIM**, 826,004 Gaussians, 87.23s training. DN-Splatter RGB/depth-gradient variant: **22.6422 / 0.752682**, 336,183 rows, 623.51s. AGS-Mesh variant: **26.4450 / 0.856505**, 447,548 rows, 639.78s.
- AGS improves mean image quality **+1.3584 dB / +0.02420 SSIM** with **45.82% fewer Gaussians**, but costs **7.33× training time**, has lower rendered opacity coverage and a median axis ratio near **7.94e8**. These are not independently verified surfaces; geometric error remains null and no method default changes.
- Lossless float32/SH3 PLYs preserve every checkpoint row. Local read-only gallery decodes all **100 images**, has zero page errors and fits desktop/mobile widths **1440/390**. Original Bonsai and private scene generation **22** remain unchanged.
- Fixed future distorted-capture evaluation to cache/undistort before selecting cameras; all distortion coefficients in this completed study are zero, so its results are unaffected. Actual training harness snapshots and pilot failures remain retained. **1,545 backend tests pass, 16 skip**, including **40 focused comparison/export tests**; syntax and whitespace checks pass. Frontend app source is unchanged this tranche.
- Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/reconstruction-comparison-findings.md`; commands: `docs/spatial-development.md`. Full remove/recover/replace, real generated completion, stable surface extraction, moving-camera/navigation and broader roadmap acceptance remain open. No model download, deployment, cloud upload or hardware-policy/Flight A changes occur.

## 2026-09-06 — explicit unknown-cell material completion (staged, not deployed)

- Added current-recovery completion preparation, bounded PNG/JPEG import, a labelled completion panel and shared reviewed place/replace operations. No automatic model launch or cloud reference upload. Provider/model identity is uploader-reported.
- Two disjoint GLB material groups preserve the original observed atlas bytes, triangle positions and UVs; only unknown cells receive extrapolated geometry/imported material. Per-cell ownership, portable render/VR-only tags and immutable source lineage prevent generated appearance from masquerading as measured evidence.
- Private striped software-fixture proof preserves **7,524 observed triangles**, adds **25,244 unknown-material triangles**, and applies/reloads/undoes **20 → 21 → 22** with exact state/artifact restoration and zero page errors. It does not remove the box or evaluate an image model. Shared replacement/clearance is separately covered by controlled backend fixtures.
- Two browser invocations retain their failures: an exact-label selector timeout before import, then mobile overflow after successful undo. The label and captured-selector width are fixed. The separate read-only layout check passes at **390px** with no overflow, zero page errors and unchanged generation 22.
- Validation: **1,505 backend passed, 16 skipped**, **194 frontend passed**; TypeScript, targeted lint, Prettier, staged build, syntax and whitespace checks pass. Current private generation **22**, revision `scene_4d11cadefaccfc5e8cb4fb65`, no active proposal; older studies are stale for new edits.
- Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/completion-findings.md`; usage/API: `docs/spatial-development.md`. Real reference-guided generation, full remove/recover/replace, navigation and the broader roadmap remain open. No deployment, weight installation, power-policy or Flight A changes occur.

## 2026-09-06 — photo-linked support surfaces (staged, not deployed)

- Added an expandable source-photo picker for three to six verified SfM support anchors, with source masks, individual track/reprojection checks, string-safe point IDs, stale-input refusal and retained camera/photo/point lineage. Automatic fitting remains optional; chosen planes stay fitted/unclassified rather than becoming asserted ground truth.
- Recovery source views now rank by the actual texture footprint, not distant plane inliers. Same-plane/private-input comparison improves supported cells **18.2739% → 22.9614%**; **77.0386% remains unknown**, with no invented texture or geometry.
- Expanded photo selection and two-view mesh-support preview/apply/reload/undo pass, adding **7,524** triangles and restoring exact state/artifact maps **18 → 19 → 20** with zero page errors. No box removal, generated completion or navigation acceptance is claimed. Final private revision is `scene_242a940dc62f715d88c63100`; fresh unapplied recovery is `recovery_5bb7fb6bfb9a41ec8352edfa`.
- Validation: **1,486 backend passed, 16 skipped**; **176 frontend passed**. TypeScript, targeted lint, Prettier, staged build, syntax and whitespace checks pass. Earlier contention/automation failures are retained and described, not counted as passes.
- Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/support-surface-findings.md`. All work is local CPU/software rendering; no model installation/inference/training, deployment or hardware-policy changes occur. Unknown-region completion and the broader roadmap remain open.

## 2026-09-06 — selection-aware local collision (staged, not deployed)

- Found that the old background solid still contains the selected box because the global builder samples the unedited reconstruction mesh. Prior row/triangle/undo proofs do not establish object clearance. Both new and saved legacy removal proposals now require selection-aware clearance.
- Added sealed collision preparation, bounded local convex-volume subtraction and atomic refined rows/collider/visibility/semantics updates. Eight explicit local gates protect clearance and sampled outside preservation. Studio exposes collision study review, refined-removal proposals and a wireframe toggle; preparation launches no GPU work.
- Real private cut clears **469 selected centers and 813 interior samples** while removing **0.095727 m³**. Combined collision changes **250,108 → 247,842**. Preview/apply/reload/browser undo succeeds **16 → 17 → 18**, with exact state/artifact restoration, pixel-identical matched-camera undo and zero page errors.
- **PASS_LOCAL_EDIT is not full navigation acceptance.** The adjacent prop's open mesh remains within 0.498 mm of selected centers. Full-object appearance, unknown background, support semantics, complete merged collision and real capsule traversal remain open. Private generation 18 is restored; earlier generation-16 evidence is now stale for another edit.
- Validation: **1,466 backend tests passed, 16 skipped**; **173 frontend tests passed**. TypeScript, targeted lint, Prettier, staged build, Python compilation and whitespace checks pass. No deployment, model training/inference, power/resource-policy or Flight A changes occur.
- Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/selection-collision-findings.md`; commands and upgrade behavior: `docs/spatial-development.md`. Full phase 0–6 acceptance remains incomplete.

## 2026-09-06 — captured selection refinement (staged, not deployed)

- Added evidence-bound multi-view selection studies, inference with already installed SAM3.1 weights, occlusion-weighted captured-Gaussian visibility, conservative group voting, sealed lineage and read-only API/UI inspection. Photo/mask/contribution layers and automatically framed current/refined splats expose selection quality without changing active rows or collision.
- Real private box candidate expands **391 → 469 served rows**. Two cameras excluded from row voting improve inferred-mask coverage **79.2% → 88.7%** and **65.3% → 83.4%**; outside-mask fractions remain **8.0% / 8.4%**. Two fit views fail association. Complete physical-object membership is not established.
- Two recorded-camera comparisons plus close-up framing, eight decoded evidence images, inspection guards and exact context restoration pass in the staged software browser with zero page errors. Active private generation remains **16**; collision remains **250,108 triangles**. No selection-apply endpoint is exposed before rebuilding its matching object/background collision.
- Full backend regression: **1,445 passed, 16 skipped**, 38.06 seconds; **23 new selection tests**. Frontend: **173 passed**. TypeScript, targeted lint, staged build and whitespace checks pass. Existing deprecation/chunk warnings remain; the SAM loader's four missing-key warnings remain a separate model-compatibility review item.
- Read `/home/rtoony/reports/2026-09-06-splatlab-next-phases/selection-refinement-findings.md` and `docs/spatial-development.md`. Results remain private/unapplied; no model installation/training, deployment, power/cgroup changes or Flight A work occurs. Selection-specific collision, unknown-background completion and the broader phase acceptance gates remain open.

## 2026-09-06 — verified captured-instance removal (staged, not deployed)

- Re-materialized retained checkpoint memberships on the private clone through the compute gate; five instance selections verify against actual served coordinates. Malformed maps, changed evidence, stale calibration and failed/ungraded collision candidates cannot authorize captured removal.
- Corrected raw collision-to-metre-world conversion and pinned backdrop scale. Studio retains static collision for remaining props while removing selected prop geometry and adding supported recovery geometry. Captured rendering is prepared before review; mesh-only removal views cannot acknowledge full review.
- Successful recorded-camera browser proof hides exactly 391 selected rows, changes collision 250,108 − 2,570 + 7,308 = 254,846, preserves removal on reload, and undoes through the UI. Final private generation is 16 with exact state/artifact-map and matched-camera pixel undo. Full-object recall, the unresolved 77.7% background and human-size navigation are not accepted.
- Validation: 1,422 backend tests pass (16 skipped), 166 frontend tests pass; TypeScript, targeted lint, staged build and whitespace checks pass. Failed software-rendering attempts and cleanup receipts remain retained separately from the completed proof.
- Findings: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/captured-removal-findings.md`. No new model installation/training, production restart, publication or hardware-policy change occurred. This is progress on phase 3, not completion of the full roadmap.

## 2026-09-06 — photo-backed background recovery (staged, not deployed)

- Added retained COLMAP/pose/image evidence loading and a CPU photo-backed support-surface recovery panel. Original photos, recorded intrinsics and masks are checked; Gaussian centres are not treated as measured surface samples.
- Partial recovery is explicit: the private box fixture uses 24 photos and 4,086 tracked support points, supports 22.3% of the local footprint and leaves 77.7% unresolved. Unknown cells have neither fabricated texture nor mesh geometry. Fitted GLBs carry portable render/VR-only provenance.
- Recovered candidates use the shared scene transaction, retaining camera/source/point/texel evidence and checking source hashes before apply. Added candidate-only inspection and framing; inspection disables walking and full-context review acknowledgement.
- Multi-angle browser inspection and contextual apply/undo pass on the private clone; exact baseline state is restored at generation 9. Source capture/live services remain unchanged. Full removal, dense visibility, unknown-region completion and connected-room acceptance remain unfinished.
- Final checks: 1,409 backend tests pass (16 skipped), 159 frontend tests pass; TypeScript, targeted ESLint, staged build and whitespace checks pass. Survey provenance checks now recognize copied tagged GLBs and refuse unverifiable GLB metadata.
- Findings and final test ledger: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/background-recovery-findings.md`, `implementation-status.md`, and `docs/spatial-development.md`.

## 2026-09-06 — creative scene revisions (staged, not deployed)

- Added `/studio/:jobId`: versioned Blender placement/removal/replacement proposals, camera-preserving comparison, explicit review/apply, saved proposal recovery and complete revision undo.
- Studio uses immutable content-addressed scene bundles and a generation-checked active pointer. Source/calibration/selection drift refuses stale edits; reviewed baseline refresh replaces studio state without discarding undo history. Legacy world/export consumers remain unchanged.
- CPU Blender + staged browser proof on a private Bonsai-derived clone adds an adjacent authored room, advances collision triangles 217,140 → 217,224, and restores exact state on undo. Saved-proposal reload, auth and mobile overflow checks pass with zero page errors. The retained fixture ends at generation 5.
- Final full regression: 1,397 backend tests pass (16 skipped), including 27 scene-revision tests; 159 frontend tests pass. TypeScript, targeted studio ESLint, staged build and `git diff --check` pass.
- Evidence and remaining work: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/implementation-status.md` and `docs/spatial-development.md`. No captured-wall cutting, multi-view background recovery, connected navigation or photoreal acceptance is claimed. No live restart, new GPU/model workload or publication occurred.

## 2026-09-06 — spatial workspace development tranche (staged, not deployed)

- Interviewed proposal and implementation ledger: `/home/rtoony/reports/2026-09-06-splatlab-next-phases/`.
- Added local INSV/GPS capture records, resumable panorama routes and `/routes` viewer; six-viewpoint X5 pilot and 72 masked rig views generated using CPU-only processing.
- Added whole-timestamp reconstruction splits, overlapping section plans, calibration-aware derived-artifact/candidate freshness and retained generated masters with explicit delivery profiles.
- Added four typed Blender architectural actions and multi-point Locate UI for feedback #7. Real synthetic Blender doorway/room/export/undo proof passes; captured-room integration is not yet implemented.
- Validation: 1,370 backend tests pass (16 skipped), 156 frontend tests pass, TypeScript/build pass, staged browser navigation/mobile/auth proof passes. New components lint clean; existing warnings remain.
- Production services/build are unchanged; no new models or GPU workloads launched. This is not completion of the full phase 0–6 research roadmap. Next review and remaining work are explicit in the ledger and `docs/spatial-development.md`.

## Goal
Extract Splat Lab from the Nexus portal into its own standalone app at
**splatlab.roonytoony.dev**, with a fresh improved GUI. **PHASED** (RToony's call):
- Phase 1 (NOW): standalone app (own frontend + auth) reusing the proven splat
  backend in-place via a proxy. Zero risk to the GPU arbiter / TRELLIS / pipeline.
- Phase 2: extract the backend into its own service + cross-process GPU lock (Redis).
- Phase 3: retire /splat from the portal (redirect splat.roonytoony.dev -> splatlab).

## Architecture (Phase 1)
- `backend/main.py` — FastAPI on :3416. Auth (PORTAL_TOKEN -> signed
  `splatlab_session` cookie). Streaming reverse-proxy of `/api/*` and
  `/supersplat/*` -> portal 127.0.0.1:3300 with the portal bearer injected
  server-side. Serves the SPA from `frontend/dist`.
- Frontend: `frontend/` (Vite React TS) — TODO.
- Port 3416 (verified free). Tunnel nexus-ai. Token reused = PORTAL_TOKEN.

## Phase 1 — DONE & LIVE (2026-06-28), commit 8c6f204
- [x] Backend (backend/main.py): auth + streaming reverse-proxy to portal :3300.
      Verified: /healthz, login 303, proxied /api/splat/status (8 jobs), 401 unauth.
- [x] Frontend (frontend/, Vite/React): improved GUI shipped — stage timeline (Q1),
      GPU-queue banner (Q3), humanized stages (Q2), results gallery + featured
      viewer (M1), download-format menu (Q5), Transfers picker + refresh, quality
      presets. Routes: / and /view/:jobId. Lean (no portal hooks). Built clean.
- [x] Deployed: splatlab.service (systemd --user, vault-injected PORTAL_TOKEN,
      enabled+active on :3416). apps-registry/apps/splatlab.toml (published, protected;
      added to protected-hostnames.txt). cloudflared config generated+synced; DNS
      CNAME added. Manifest add-app + log.
- [x] Verified LIVE: https://splatlab.roonytoony.dev/healthz -> 200; root -> 303
      login; headless render of the public URL shows the full GUI. Portal /splat
      untouched and still serving.

## Phase 2 — IN PROGRESS
- [x] **Keystone: cross-process Redis GPU arbiter** (`backend/gpu_arbiter.py`).
      Drop-in for the portal's arbiter (HEAVY_GPU_LOCK async-with + .locked(),
      set/clear/holder_info, gpu_status/evict/acquire_gpu). Redis SET-NX lock +
      TTL(45s)+heartbeat(15s); holder in a Redis hash. **FAIL-OPEN**: Redis down →
      degrades to in-process asyncio.Lock + local holder (= old behavior), never
      deadlocks. redis-py installed in BOTH venvs (portal + splatlab).
      VERIFIED in isolation: 2-process mutual exclusion (no overlap); fail-open on
      dead Redis (no hang); locked()/holder_info correct. Test keys cleaned.
- [x] **Ported splat pipeline -> splatlab backend.** `backend/splat_route.py`
      (copy of portal splat.py; swapped to local `gpu_arbiter`, `operator_audit`
      stub, SPLAT_ROOT=/home/rtoony/projects/splatcli). `main.py` mounts the router
      at /api/splat (auth-gated via require_auth dep) + runs migrate/cleanup on
      lifespan; /api proxy REMOVED (splatlab owns it now), /supersplat still proxied.
      splatlab.service injects "Rtoony Portal" + "Infrastructure" (REDIS_PASSWORD).
      → The FRONTEND is implicitly cut over: it calls same-origin /api/splat which
        now hits splatlab's own backend (no code change needed).
- [x] **Ran a REAL job through splatlab's own backend** (splat_bf25300429, 128-frame
      processed dataset, 3000-iter): train->export->compress->webopt all completed;
      produced splat.ply(65M)+spz(4.3M)+web.ply(17.9M); **Redis GPU lock taken
      (holder lane=splat) during, released+cleared after**. Job visible in status,
      preview_available. (Smoke scene left in the gallery — RToony can delete.)
- [x] **Portal arbiter SWAPPED to Redis** (portal commit 5cdcb2c, pushed main).
      Backed up old -> server/lib/gpu_arbiter.py.bak.inprocess.* (one-cp rollback).
      Hardened per a 5-agent adversarial swarm (verdict fix-first): TTL 45->90s,
      cancel-safe __aenter__/__aexit__ (no local-lock leak), socket_timeout 0.5,
      broadened excepts. Re-verified: 2-proc mutual exclusion, fail-open, cancel-
      mid-acquire releases local lock.
      Post-swap gates ALL PASS: 87 vars (no stale-BW 0-var), REDIS_PASSWORD present,
      /api/3d/queue 200 (TRELLIS alive), and COORDINATION ENGAGED — external Redis
      lock flips the portal's gpu.locked to True (reads the shared lock, not per-proc).
- [x] Frontend cutover: done (splatlab calls its own /api/splat). All three lanes
      (splatlab-splat, portal-splat, portal-TRELLIS) now serialize on the 5090 via
      the shared Redis lock. **The Phase-1->2 coordination gap is CLOSED.**

## PHASE 2 COMPLETE (2026-06-29).
## PHASE 3 COMPLETE (2026-06-29, portal commit 716f3be pushed). EXTRACTION DONE.
- [x] splat.roonytoony.dev -> 307 splatlab (in auth_middleware, PRE-auth, so old
      bookmarks skip the portal login wall). portal /splat + /splat/view -> client
      redirect to splatlab (deep link preserved). Sidebar "Splat Lab" -> external.
      Launch Bay: splat="moved", splatlab=the studio. Verified all; portal+splatlab
      unaffected. Portal splat.py backend left DORMANT (still coordinated via the
      shared Redis arbiter); deleting it is optional cleanup.

## GUI backlog
- [x] Capture confidence (commit 78cd649): Customize iterations slider + live time
      estimate; preflight summary card; engine-ready gate on Create.
- [x] Retry-with-params: Re-run + ↑Quality (2x) on scene cards (re-POST /train with
      the job's params). Standard scenes faithful; 360 sub-params (images_per_equirect
      /crop_bottom/insv_fov) not persisted on SplatJob meta so a 360 re-run uses
      defaults — fine for standard, note for 360.
- [x] Scene pin + two-click delete on gallery cards (commit 477d2c8); per-scene
      color tint so cards are distinguishable; 30k-iters badge.
- [x] Cleanups: smoke scene splat_bf25300429 deleted; Launch Bay deduped (portal
      commit ecc8e13: projects.py hides superseded "splat", canonical card ->
      splatlab); redis declared in portal pyproject.
- [x] Real gallery thumbnails — DONE (different approach than the deferred ones).
      `backend/thumb.py`: a point-cloud projection of the splat's .ply, sampled by
      SEEKING (CPU-only, ~50ms even on millions of points), colored by SH-DC, drawn
      with Pillow, cached to _preview/thumb.webp. Endpoint GET /api/splat/jobs/{id}/
      thumbnail (auth-gated, off-thread). SceneCard shows it with the per-scene
      gradient+icon as the fallback (scenes without web.ply). Pillow added to venv +
      requirements. (Avoided the GPU-rasterizer / preserveDrawingBuffer paths.)
- [ ] Optional: delete portal dormant splat.py + splat*.tsx (harmless dead code,
      cross-referenced — risky to remove unattended).
- [ ] optional cleanup: delete the portal's now-dormant splat.py + splat*.tsx;
      declare `redis` in portal pyproject deps; dedupe the splat/splatlab Launch
      Bay cards; delete the splat_bf25300429 smoke scene if unwanted.
- [ ] Minor: declare `redis` in portal pyproject.toml deps (installed in venv now,
      not yet in the manifest — matters only on a clean rebuild).
- [ ] Phase 3: redirect splat.roonytoony.dev -> splatlab; remove the /splat page +
      nav entry from the portal (leave a redirect). Continue GUI backlog: capture-
      confidence (M3 preset sliders, M5 preflight validation), real gallery
      thumbnails, retry-with-params (M2, needs 360 sub-params persisted on SplatJob).

## Invariants (do NOT break)
- Do NOT touch the portal's gpu_arbiter / three_d.py / splat.py pipeline in Phase 1.
- Portal /splat keeps working until Phase 3.
- Reuse PORTAL_TOKEN; pull via vault (nexus-svc-inject), never write to disk.
- PROCESS MGMT: never broad-pkill `uvicorn backend.main:app` — many Nexus apps
  share that cmdline (nexus-vicinity :3404, etc.). Kill by exact port/cmdline only.

## CAPTURE RELIABILITY — "every capture just works" (2026-06-30, commit e560c41)
- [x] A1 registration GATE: splat_route.py — after `process`, ratio = registered
      (transforms.json frames) / extracted (processed/images); < MIN_REGISTRATION_RATIO
      (0.30) → fail fast pre-train with an actionable message (no GPU wasted).
      Additive/reversible/parse-safe; default COLMAP path byte-for-byte unchanged.
      Frontend: amber failure card surfaces the message + a "Retry with global SfM" button.
      VERIFIED fires on backyard (2/311=0.6%), passes good jobs (128/128).
- [x] A3 global-SfM rescue: opt-in `sfm_backend="glomap"` → glomap_sfm stage runs
      COLMAP 4.x feature_extractor + sequential_matcher + global_mapper, then
      ns-process-data --skip-colmap --colmap-model-path ../colmap/sparse/0 (RELATIVE).
      **DEP: conda env `colmap4` = COLMAP 4.1.0 built from source (CUDA 12.8, sm_120,
      -DCMAKE_CUDA_ARCHITECTURES=120). Binary: ~/miniconda3/envs/colmap4/bin/colmap.
      ISOLATED — the working `colmap` (3.11.1) + `splatops` envs are untouched.**
      Build needed 4 patches (CHOLMOD target, glog version macros, cuda_runtime.h
      include, Eigen config-mode) — see workflow output if rebuilding.
      PROVEN end-to-end: global_mapper 311/311 on backyard (vs 2); nerfstudio 1.1.5
      reads the 4.x model → transforms.json 311 frames; 4.x renamed the GPU flags
      (SiftExtraction→FeatureExtraction.use_gpu, SiftMatching→FeatureMatching.use_gpu).
- [x] **AUTO-FALLBACK (zero-click "just works")** — the A1 gate no longer just fails
      on low reg: it climbs the solver chain `SFM_ESCALATION = [colmap, glomap, mast3r]`
      automatically. `_maybe_escalate_sfm` rebuilds the next available solver's SfM
      pre-stage + a uniquely-named `reprocess<n>` and injects them into the live
      stages_planned ahead of train; the loop (enumerate over the live list) picks them
      up next. Only fails the job with guidance once the chain is exhausted. Manual
      "Retry with global SfM" button preserved; default COLMAP success path byte-for-byte
      unchanged (verified: colmap planner emits only `['process']`, no `--skip-colmap`).
      Loop-safety (no solver twice via sfm_tried; reroute cap = len(chain); equirect/
      dataset excluded via sfm_context=None) — 22/22 unit checks PASS.
- [x] **Phase B (pose-free MASt3R-SfM fallback) — WIRED & TESTED.** Terminal rung of the
      chain. `mast3r_sfm` stage runs the runner (`~/tools/mast3r-spike/run_mast3r_sfm.py`,
      ViT-Large dense matching → poses.npz/points3D.npz) then a DIRECT converter
      (`mast3r_to_nerfstudio.py`) that reproduces nerfstudio 1.1.5's colmap_to_json
      convention (proven identical to 4.4e-16) → writes transforms.json + images/ +
      sparse_pc.ply straight into processed_dir (NO ns-process-data). END-TO-END TESTED:
      39 backyard frames → 39/39 finite poses, 88.6s, 3.46GB peak → converter → full
      ns-train splatfacto 100-iter smoke EXIT=0 (seeded from the MASt3R ply, random_init=
      False). Coordinate gotcha handled (OpenCV c2w → OpenGL; world permute; applied_
      transform on the cloud too). 4 path constants env-overridable; `mast3r_available`
      True only if all 4 (env python + runner + converter + 2.6GB ckpt) exist —
      VERIFIED live True. **DEP: conda env `mast3r-spike` + checkpoint (2.6GB) at
      ~/tools/mast3r-spike/. CC-BY-NC-SA (non-commercial).**
- Review fixes folded in before commit (3-issue adversarial pass): (#1, ship-blocker)
  glomap `process` now `rm -rf processed_dir` so a colmap→glomap reroute can't measure
  a stale colmap/glomap mix; (#4) `mast3r_sfm` runs under HEAVY_GPU_LOCK (6GB reserve)
  so its ViT can't OOM the portal's TRELLIS lane (light colmap/glomap SfM stay lockless);
  (#3) reroute process uniquely named `reprocess<n>` → no duplicate stage-rail key / no
  false double-green. Review CONFIRMED safe: infinite-loop guards, mid-run list mutation,
  no false-escalation of good captures, default path + manual button intact.

## WAVE 1 — 360 fix + heatmap backend + edit-ops + Spark spike (2026-07-04, IN FLIGHT)
Master plan: ~/reports/splatlab-ultra-plan-2026-07-04/PLAN.md (RToony GO'd waves; UE
parked, replaced by survey/scale/benchmark design — see reports dir).
- [x] Housekeeping: 07-02 feedback+camera pass committed (e2a8409 XC-1 gpu_arbiter alert,
      6628092 feedback+camera). Tree was clean before wave-1 agents started.
- [x] 360 ROOT CAUSE (probe receipts in PLAN.md Appendix A): X4 .insv = TWO HEVC streams
      (one square fisheye per lens); ffmpeg -i read only stream 0 -> corrupt equirect ->
      2/624 (0.3%) registration on splat_ec1b984ffb. Fix VALIDATED manually:
      hstack both streams -> v360 dfisheye -> coherent panorama (scratchpad receipts).
- [x] BUILT + REVIEWED + FIXED + COMMITTED (workflow wf_aa28b8d5-7f4 + 2 fix agents;
      commits 9e565a3 360-fix / 48e701a langfield-heatmap-backend / 7174809 edit-ops /
      7d75ea0 spark-spike+supersplat-link; full suite 127 passed; adversarial reviews:
      backend SHIP w/ 4 findings fixed, edit-ops FIX_FIRST w/ all 10 fixed incl.
      dequantization blocker; splat-transform bumped 2.5.1->2.7.1 by the supersplat
      2.28.1 install — compress/webopt argv SMOKE-TESTED OK):
      A1 splat_route.py 360 fix (hstack compose, fail-loud, sanity gate, equirect
      matcher, glomap escalation, 360-param persistence) then langfield relevancy
      backend (langweb artifact + worker /relevancy + app proxy);
      A2 backend/edit_ops.py NEW (snapshots/versions, splat-transform ops, text-select
      delete/isolate/extract, MERGE scenes) — orchestrator mounts router in main.py;
      A3 frontend Spark spike /spark-test (fake-scalar heatmap via dyno worldModifier,
      nav prototype: reset/presets/pivot);
      A4 ~/projects/supersplat bump 2.27.4->2.28.1 (NODE_ENV-unset build gotcha);
      A5 survey/scale/benchmark DESIGN -> ~/reports/splatlab-survey-scale-design-2026-07-04/.
- [x] DEPLOYED 2026-07-04 ~17:04: splatlab.service (36 vars, healthz ok) +
      splatlab-langfield.service (worker /relevancy live in openapi). edit_ops
      router mounted (5 routes) + langfield STALE guard added to query/relevancy/
      inventory.
- [ ] LIVE 360 VALIDATION IN FLIGHT: job splat_98095cb055 = the SAME .insv, SAME
      params as failed splat_ec1b984ffb. Acceptance: registration >=30% (target
      >50%) + coherent render. Watcher bg6s98ixl.
- Deferred to next wave: "Edit in SuperSplat" deep-link + heatmap/nav UI on the real
  viewer (blocked on Spark spike verdict); portal dead-code deletion (0.5).

### Live 360 validation findings (2026-07-04 evening)
- splat_98095cb055 (the office .insv, SAME params as the original failure):
  stitch = hstack compose RAN (both HEVC streams mapped — receipt in job.log),
  sanity gate PASSED (no false-positive on the static-ish capture), sequential
  matcher used... registration STILL 2/624 (0.3%) -> ROOT CAUSE #2 (visual
  receipts in scratchpad/motion/): the clip is a SELFIE — operator holds the X4
  at arm's length facing himself; face/torso/arm dominate the sphere and move
  WITH the camera (dynamic occluder, camera-stable features) -> geometric
  verification rejects nearly all matches. NOT a pipeline bug; SfM physics.
  The new auto-escalation then fired (glomap_sfm rung, COLMAP 4.1 global_mapper,
  overlap 16) — mechanics receipt regardless of its verdict on doomed data.
- FOUND proper validation capture: ~/transfers/splatlab/VID_20260514_073947_00_002.insv
  (1.65GB, 106s, 3197 frames, dual-stream) = OUTDOOR POOL FACILITY WALKTHROUGH,
  camera overhead on stick, operator only at nadir (crop_bottom trims), textured
  concrete/buildings. Frames: scratchpad/may14/. This is the real acceptance run.
  (Also VID_20260514_064632_00_001.insv, 6.7GB/434s — same site, longer.)
- FOLLOW-UP FEATURE (high value, next wave): operator auto-masking for 360 —
  SAM2.1 person segmentation on fan-out crops -> COLMAP ImageReader.mask_path;
  would make selfie-style/visible-operator captures reconstructable. We already
  have SAM2.1 + the sam2 env on disk.
- UI guidance follow-up: 360 upload card should say "hold the camera OVERHEAD
  on a stick — if you're visible anywhere but straight down, the scan fails".

## CRASH POST-MORTEM + CPU LEASH (2026-07-04 evening)
- 17:36:53 splat_98095cb055 (office selfie clip) COMPLETED end-to-end: hstack
  stitch + glomap escalation -> trained, 1.92M gaussians, artifacts in
  _preview/ (splat.ply 454MB, web.ply 78MB). Quality UNVETTED — selfie data;
  eyeball in the viewer before judging. thumb.webp is 0 bytes (crash cut it).
- 17:37:22 the REAL acceptance run (May-14 pool walkthrough, 1.65GB) started as
  splat_fdac9edaab; the PC HARD-RESET within seconds of its stitch launching.
  Forensics: NOT VRAM/GPU (vram 31%, 56C, xid 0, gpu-watch clean at 17:35:37),
  NOT mains power (UPS event log silent), NOT kernel (no oops/pstore; journal
  tail lost). Firmware BERT record = CPER severity FATAL, section GUID
  81212A96-09ED-4996-9471-8D729C8E69ED (Firmware Error Record Reference /
  Intel CrashLog) -> CPU-domain hardware fatal error at the instant the
  all-core x264 encode launched (idle->250W package step; RAPL PL1=PL2=250W;
  ASUS ROG MAXIMUS Z890 HERO BIOS 3002, 285K ucode 0x121). The orphaned job
  was auto-marked failed on restart ("portal restarted while job was active").
- MITIGATION 1 (this commit): `_stitch_cpu_leash()` — taskset to half the
  cores (floor 4) + nice 10 on BOTH stitch paths. SPLAT_STITCH_CPUS overrides;
  0 disables. taskset/nice exec through -> job.pid still ffmpeg. 131 tests pass.
- MITIGATION 2 (system level): RAPL power-limit guard staged as
  ~/scripts/aipc-cpu-power-guard.sh (dry-run default; --apply caps PL1/PL2 +
  installs a persistent boot unit). BIOS checklist in the crash report.
- GATE: re-dispatch the pool-walkthrough acceptance run ONLY after the power
  guard is applied (app leash alone shrinks the transient but the fault is
  hardware-marginal).

## WAVE 2 START — acceptance run + Spark real-relevancy wiring (2026-07-04 late)
- Power guard APPLIED by RToony (RAPL PL1=125W/PL2=177W verified; boot unit enabled).
- Acceptance run DISPATCHED: `splat_75ebbcddde` (May-14 pool walkthrough,
  language_field=true — langfield stage queued last, will be the FIRST langfield
  scene on disk). Leash receipt in job.log: `taskset -c 0-11 nice -n 10 ffmpeg`;
  package 75C under load; the crash scenario now runs safely.
- spark-test upgraded (Wave 2.3 wiring): real language query → POST
  /langfield/relevancy → uint8 vector → RgbaArray → the SAME dyno modifier as the
  fake proof. Langfield scenes load fmt=langweb (index alignment with gauss_emb);
  FAIL-LOUD on any rows≠splats mismatch. End-to-end test unblocks the moment
  splat_75ebbcddde's langfield lands.

## WAVE 2.1-2.3 PROVEN END-TO-END ON REAL DATA (2026-07-04 evening)
- **Index mismatch ROOT-CAUSED + FIXED**: gauss_emb rows follow the CHECKPOINT,
  but ns-export FILTERS gaussians (Garden: 1,326,611 ckpt -> 1,321,833 ply;
  4,778 dropped) — so even langweb order could never match raw gauss_emb.
  Fix = `backend/langfield_align.py`: byte-exact float32 xyz hash map
  (ply row -> ckpt row), built+cached lazily per scene by the worker
  (`_langfield/ply_index_map.npy`), applied to /relevancy BEFORE quantization.
  Receipt: X-Count 1326611 -> 1321833 after fix; worker log "ply->ckpt map
  ready"; 100% of ply rows matched. Legacy scenes fixed retroactively, no
  retrain. 137 backend tests pass (6 new).
- **Live browser receipt (Garden, real GPU)**: "flower vase" -> 1,321,833 rows,
  420ms warm -> REAL per-splat heatmap tint on the Spark viewer + spotlight
  fade of low-relevancy splats. Spark verdict = PASS (Z-up correct, 1.3M splats
  crisp; fps unmeasurable headless — rAF throttled in background windows).
- **Spark gotcha (proven live)**: mutating a dyno uniform does NOT re-run the
  generator — spotlight/threshold flips were visual no-ops until
  `mesh.updateVersion()` after each uniform write.
- Garden langweb.ply backfilled (86MB vs 328MB raw fallback; 0.8s). TODO:
  backfill the other 5 mip360 langfield scenes the same one-liner way.
- **Portal clobber bug FIXED (portal commit 87632b6, NOT yet restarted)**: the
  portal's dormant splat.py ran cleanup_orphan_jobs() on every deploy and
  marked LIVE splatlab jobs failed ("portal restarted while job was active" at
  18:21:24 = portal ActiveEnterTimestamp, receipt). Startup hook removed;
  takes effect next portal restart (deferred — another session is deploying
  portal). splat_75ebbcddde meta hand-restored to running; its pipeline never
  actually stopped.

## SURVEY v1 SLICE + SPARK BETA ON THE VIEW PAGE (2026-07-04 night)
- **Scale calibration shipped end-to-end**: POST /jobs/{id}/scale stores
  meters_per_unit in meta (validated, null clears; 9 tests). Viewer measure
  tool: Spark raycast two-point pick -> markers+line -> scene units ->
  calibrate with known length (m/ft/in) -> real units everywhere after.
  LIVE RECEIPT (Garden): 0.5235 units = 1.524m = 5.00ft; meta shows
  meters_per_unit=2.9113 (NB: test calibration with a made-up 5ft reference —
  clear via {"meters_per_unit": null} or recalibrate on a real reference).
- **Spark beta viewer on /view/:jobId** (spark-scene-viewer.tsx, opt-in header
  toggle, sticky localStorage): real language heatmap + spotlight + measure.
  Classic viewer untouched/default; overlays/search-flyto stay classic until
  the full 2.4 cutover. Shared machinery extracted to lib/spark-heatmap.ts
  (spike page refactored onto it — one implementation).
- ezdxf 1.4.4 installed in backend/.venv (survey exports dep, per DESIGN.md).
- ⚠️ **LESSON (cost us the first acceptance run): `systemctl --user restart
  splatlab` SIGTERMs the WHOLE cgroup — start_new_session does NOT protect
  job subprocesses from systemd (KillMode=control-group). splat_75ebbcddde
  died mid-mapper ("Stage 'process' exited with code -15"). RULE: never
  restart splatlab.service with a job in flight. BACKLOG: job resume-on-start
  (rehydrate running meta + stage checkpoints) — codev candidate.
- Acceptance run RE-DISPATCHED: **splat_192e4223fb** (same params,
  language_field=true); leash verified (taskset 0-11, nice 10).

## OVERLAY v2 + DIMENSIONS (2026-07-04 late night, all browser-verified on Garden)
- **Multi-query language overlay**: up to 4 simultaneous searches, one color
  each (editable via color picker), packed into ONE RgbaArray (R/G/B/A
  channels) + mode-baked dyno modifier in lib/spark-heatmap.ts
  (buildOverlayModifier). Modes: Highlight (natural + colored matches),
  Isolate (only matches visible), Spotlight (colored + rest dimmed), Ramp
  (single-query scientific ramp: viridis/turbo/magma/grayscale). Live legend
  (bottom-right) tracks queries/colors/mode/threshold; per-query enable
  toggles + shared match-threshold slider are live uniforms (updateVersion).
  Receipts: "ball"@0.91 highlight = just the ball yellow on natural scene;
  isolate = table floats alone; 2-query legend (ball/wooden table).
  NOTE: relevancy bytes are PER-QUERY min-max normalized -> threshold is
  relative (default 0.75); absolute calibration = future work.
- **Dimensions**: unlimited two-point dimensions; draggable endpoints
  (pointer-capture, orbit paused during drag); floating midpoint labels
  (imperative DOM, projected per frame); list with per-dim delete + clear-all;
  sessionStorage persistence per scene; calibration binds to a selected
  dimension. Receipt: patio dim "3.039 m · 9.97 ft" label live.
- **Embedding-paint designed** (RToony's idea): sidecar override model
  (never mutate gauss_emb), query-select/sphere/brush rungs, worker apply +
  CRUD -> ~/reports/splatlab-embedding-paint-design-2026-07-04/DESIGN.md.
  P1 unblocked by today's langfield_align work.

## PAINT-THE-EMBEDDINGS SHIPPED (2026-07-04 night) — RToony's feature
- Backend COMPLETE + worker-verified live on Garden; frontend brush UI built.
  ⚠️ app proxy endpoints (select/sphere, overrides CRUD) need a splatlab.service
  restart — GATED until splat_192e4223fb finishes (no restarts mid-job). The
  worker side (:3417) is already live.
- Mechanism: sidecar overrides (backend/langfield_overrides.py — manifest json
  + per-record uint32 npy in _langfield/, EXPORTED-PLY order; gauss_emb.npz
  NEVER touched). Worker composes at scene load (assign/boost = blend toward
  label embedding — a zero/unseen row BECOMES the label, which is what makes
  abstract "liberal" labels work; suppress = remove projection). Worker
  endpoints: /select_sphere (GPU sphere test on resident positions),
  /overrides_add (guardrails: min 10 splats, ≤30% of scene unless force=true,
  bounds check), /overrides_delete; scene cache invalidated on mutation.
- EXACT-LABEL RECALL: /relevancy pins a painted region to max relevancy when
  the query names its label OR alias (X-Label-Hit header) — deterministic for
  labels SigLIP can't ground ("lucky orb" verified).
- LIVE RECEIPTS (worker-direct, Garden): sphere stroke @ ball focus r=0.12 ->
  2,229 rows; committed label "lucky orb" alias "the special thing" -> both
  queries X-Label-Hit:1; delete -> hit:0, files gone, manifest empty (full
  revert). 154 backend tests (8 new for guardrails/roundtrip).
- UI (beta viewer "Paint the field"): brush radius slider (meters when scale
  set), stroke preview in cyan w/ live count, UNDO per stroke + clear,
  "clip strokes to <query> matches" hygiene toggle, duplicate-label warning,
  Pin/Boost/Not-this ops, force-flow for oversized selections, painted-labels
  list with one-click revert.

## SCREENSHOT-DRIVEN FIX PASS (2026-07-04 late night, from RToony's 8 captures)
- **Percentile thresholds**: raw "match ≥ 0.75" replaced by "top X%" per query
  (cutoffForTopPercent histogram → per-channel cutoff uniforms). Root cause of
  the all-yellow Spotlight/Kitchen shots: relevancy is per-query min-max
  normalized so raw thresholds are meaningless across queries. Default top 2%.
- **Ramp mode honors enables**: tint channel + legend = first ENABLED query
  (was hardwired to channel 0 — RToony had ch0 disabled, got nothing + wrong
  legend). Enable toggles rebuild in tint mode.
- Stale paint 405 error cleared on mode/paint flips; stroke 404/405 now says
  "paint backend deploys on next splatlab restart (waiting for running job)".
- Legend raised above the Feedback FAB (was overlapped/truncated).
- Verified live on Garden: ball @ top-2% highlight, legend copy, dim persisted.
- Paintbrush 405 itself = the KNOWN deploy gate (old app process; endpoints
  land with the post-job restart). No code change needed.

## Test Flight + widescreen + segmentation program (2026-07-05)
- Program pack (plan for post-Fable executors): ~/reports/splatlab-360-sample-segment-plan-2026-07-05/
  (PLAN.md phased w/ acceptance gates, STATUS.md spine, evidence/ = 5-agent ultracode map).
- SHIPPED: Test Flight trim window (trim_start_s/trim_duration_s on /train; input-side
  -ss/-t on stitch; auto-centered; 400 on non-insv; meta now persists num_frames_target/
  sfm_backend/language_field/trim_*). 56/56 stitch tests. Frontend toggle (insv-only) +
  widescreen pass (max-w-[1880px], 2xl grid split, gallery 5-col, viewer 2xl:h-560).
- SHIPPED: ~/bin/splatlab-safe-restart — the ONLY sanctioned way to restart the service
  (the raw restart killed 2 real jobs on 07-04 = the "-15" cards).
- NEXT: Phase 3 segmentation (SfM-level join via colmap4 model_merger + bundle_adjuster,
  probe script first), train-resume via --load-dir. See the pack's PLAN.md.

## Langfield optional-stage bookkeeping fix (2026-07-05, from HANDOFF-PLAYBOOK PACKET 7)
**Bug**: langfield is a deliberately best-effort/opt-in stage — its failure correctly
never flips the job to `failed` (the splat itself is already done). But the pipeline loop
unconditionally appended `"langfield"` to `stages_completed` on both the non-zero-exit path
and the catch-all `except Exception` path, with no other record that it actually failed —
so a job's meta made a failed optional stage look identical to a successful one.

**Fix** (`backend/splat_route.py`, langfield stage only — compress/webopt share the
identical pattern but were left untouched, out of scope per the brief):
- New `_new_meta()` key `"stages_failed": []`, parallel to `stages_completed`.
- New helper `_record_stage_failure(job_id, stage, reason)` — read-patch-write, same
  pattern as the existing `stages_completed` append.
- Called from both langfield failure paths: `rc != 0` → `f"exit code {rc}"`; caught
  `Exception` → `f"error: {exc}"`. The "no config / toolchain unavailable" **skip** path is
  deliberately NOT recorded as a failure (it's a normal no-op, not a stage that ran and failed).
- `stages_completed` still gets `"langfield"` appended unconditionally either way — job
  semantics UNCHANGED: `final_status` still ends up `"completed"`, splat still "done".
- Also threaded `stages_failed` into the `audit_operator_event` metadata dict (the audit
  trail was hiding the same failure).
- Confirmed visible end-to-end: `grep -rn "stages_completed\|stages_failed"` showed every
  consumer — `_job_payload()` spreads `**meta` verbatim into every job API response (status
  list, stop, preview endpoints), so `stages_failed` reaches the frontend with zero extra
  plumbing. Updated `frontend/src/lib/contracts.ts` (`SplatJob.stages_failed?`) and
  `frontend/src/pages/splat.tsx` (gallery card: amber "field failed" badge + reason tooltip
  when `langfield_available` is false AND a `stages_failed` entry names `"langfield"`).

**Verification (real receipts, backend/):**
```
$ ~/.local/bin/pytest backend/tests/test_langfield_stage_bookkeeping.py -v
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_nonzero_exit_does_not_fail_job PASSED
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_exception_does_not_fail_job PASSED
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_success_leaves_stages_failed_empty PASSED
backend/tests/test_langfield_stage_bookkeeping.py::test_langfield_skipped_no_config_is_not_recorded_as_failure PASSED
4 passed in 0.15s

$ ~/.local/bin/pytest backend/tests/ -q
FAILED backend/tests/test_scale_calibration.py::test_scale_rejects_garbage[nan]
FAILED backend/tests/test_scale_calibration.py::test_scale_rejects_garbage[inf]
2 failed, 169 passed, 4 warnings in 2.19s
```
The 2 failures are PRE-EXISTING and unrelated (JSON `nan`/`inf` encoding in the scale
calibration endpoint) — confirmed identical (165 passed / 2 failed) on a `git stash` of this
diff before making any change. No regressions from this fix; 4 new tests, 0 net new failures.

Frontend typecheck (`npx tsc --noEmit` in `frontend/`): 43 pre-existing errors, byte-identical
before and after this diff (all in `splat-viewer.tsx` / `feedback.tsx` / `feedback-api.ts` /
`splat-view.tsx` — none in the two files this fix touched).

**Out-of-scope note**: `compress` and `webopt` (lines ~2437-2517 as of this commit) have the
IDENTICAL bug — best-effort, log-only on failure, unconditional `stages_completed` append,
no `stages_failed` record. Left untouched per the brief ("Do NOT touch the compress/webopt
stages... just note in your report"). Same `_record_stage_failure` helper would cover them
if/when someone picks that up.

## §1E "Promote to full build" SHIPPED (2026-07-05, Sonnet 5 swarm session, from the handoff pack)
Fixes the F2 "hybrid" trap: on a completed Test Flight (trimmed) scene, the old **Re-run**
button silently dropped the trim (job.input_path is always the original full .insv — trim is
stitch-time only, never re-sent) but kept the flight's draft `max_num_iterations` and fell back
to the request default `sfm_backend="colmap"` — a multi-hour run at draft quality on the doomed
84-min COLMAP rung, neither a test nor a full build.

- `frontend/src/lib/contracts.ts`: exposed `num_frames_target`, `sfm_backend`, `trim_start_s`,
  `trim_duration_s` on `SplatJob` (backend already returned them via the meta spread; the type
  just didn't declare them).
- `frontend/src/pages/splat.tsx`: new `promoteToFullBuild(job)` — same insv input, trim omitted
  (full clip), `num_frames_target: 300` (the backend's §1D′ duration-aware rule overrides this
  for insv jobs once deployed), `sfm_backend: job.sfm_backend ?? "glomap"` (the rung the flight
  actually proved — every flight requests glomap directly, so the persisted value is reliable;
  no reroute history needs to be exposed), iterations from the **currently selected quality
  preset** (`iters`), `language_field` from the current toggle (not the flight's stored value).
  On scene cards where `trim_duration_s != null`, this one button replaces BOTH Re-run and
  ↑Quality — both call the same old `rerun()` and both inherit the identical hybrid bug on a
  trimmed scene, so leaving ↑Quality in place would leave the trap under a different label.
- `rerun()`/`retryGlomap()` (non-flight jobs) now also forward the scene's persisted
  `num_frames_target` (both) and `sfm_backend` (rerun only — retryGlomap's whole point is to
  override it to glomap) instead of leaving them unset and falling back to request defaults
  that could silently contradict how the scene was actually built.

**Verification:**
```
$ npx tsc --noEmit    (frontend/)
43 errors — byte-identical to the pre-existing baseline (splat-viewer.tsx/feedback.tsx/
feedback-api.ts/splat-view.tsx); zero in contracts.ts or splat.tsx.

$ npm run build       (frontend/)
✓ built in 2.31s — clean.
```
No component-test runner exists in this repo (no vitest/jest configured), so the gate's
"manual dispatch meta.json diff" path was done **without creating a live job** (no visible
gallery row, no audit event, no meta.json write) — imported `splat_route` directly and ran the
exact JSON.stringify(body) shape `promoteToFullBuild()` sends through `SplatTrainRequest` ->
`_plan_3d_job` -> `_new_meta`, using the real `VID_20260514_064632_first90s.insv` for duration:
```
1. SplatTrainRequest validated OK:
   trim_start_s=None trim_duration_s=None sfm_backend='glomap' num_frames_target=300
2. _plan_3d_job stages: ['stitch', 'glomap_sfm', 'process', 'train', 'export', 'compress', 'webopt']
   stitch argv: [...'-i', '.../VID_20260514_064632_first90s.insv', '-filter_complex', ...]
   (no -ss/-t in the argv — full clip, trim correctly dropped)
3. _new_meta persisted fields (would-be meta.json):
   num_frames_target = 300 / sfm_backend = 'glomap' / trim_start_s = None / trim_duration_s = None
OK: promoted payload validates, drops the trim, keeps glomap, no job/meta.json written.
```
Confirms: trim dropped, glomap rung kept (not colmap), stages plan correctly — proves the
exact defect (F2) is fixed without spending GPU time or creating visible state.

Deploy: frontend-only, `npm run build` already run above — **no service restart needed**.

Committed locally (not pushed): see git log.

## §1D′ duration-aware num_frames_target SHIPPED (2026-07-05, Sonnet 5 swarm session)
The pipeline's only proven-good 360 config is ~3.0fps equirect frame density
(splat_9da9dff4b2 @1.76fps: 599 posed, ZERO points vs splat_5177f8d99a @3.0fps: 1078/1080
registered, 105k points — same window, both directions confirmed). Test Flight already
computes `num_frames_target = 3 * trim_duration_s` client-side, but a full (non-flight)
insv run has no way to know the real clip duration — the UI can only hardcode a flat
guess (75), which is 0.7fps on a 106s clip and hits the exact same 0-point cliff on
anything longer than ~25s.

**Fix** (`backend/splat_route.py`, `_plan_3d_job`'s `is_insv` branch):
- Duration is now always probed (not just when a trim is requested) — `full_duration`.
- After trim resolution: `density_window_s = trim_duration if trim_duration is not None
  else full_duration`. When known and > 0: `req.num_frames_target =
  min(ceil(3.0 * density_window_s), 4000 // req.images_per_equirect)` — overrides
  whatever the client sent, self-capped so it can never trip the `/train` endpoint's
  existing `perspective_images > 4000` guard (backend/splat_route.py:3020, unchanged).
  Probe failure (no ffprobe / unreadable container) leaves the client's value alone
  rather than guessing — mirrors the layout-probe's existing fail-open policy.
- Test Flight is a **no-op** under this rule: same 3fps formula, same (trim) window,
  same result the client already sends (30s -> 90, matches exactly) — only full runs
  change behavior.
- `_new_meta` runs AFTER `_plan_3d_job` in the `/train` handler, so meta.json now shows
  the REAL computed value, not the raw client request — also fixes the SfM-escalation
  gate's rebuild path (`sfm_req`), which is captured after the override and so reroutes
  to a fallback solver using the correct density too.

**Verification:**
```
$ ~/.local/bin/pytest backend/tests/test_360_stitch.py -q
64 passed in 0.56s   (56 baseline + 5 new §1D′ cases + 3 already added post-langfield fix)

$ ~/.local/bin/pytest backend/tests/ -q
2 failed, 174 passed, 4 warnings in 2.16s
```
The 2 failures are the same PRE-EXISTING `test_scale_calibration.py` nan/inf cases (unrelated,
confirmed unchanged since the langfield commit). 5 new tests, 0 regressions:
- full run computes duration-aware target (80s -> 240, no cap needed)
- crop-count cap engages on a long clip (300s -> would be 900, capped to 500)
- cap scales with images_per_equirect (14-crop: 4000//14=285)
- Test Flight trim produces the SAME value the client already computes (30s -> 90 —
  proves this ships with zero behavior change for the already-correct lane)
- probe failure leaves the client's value alone (no crash, no guess)

Deploy: **DEPLOYED 2026-07-05 11:34 PDT** via `splatlab-safe-restart` (RToony's go-ahead,
post-Phase-3.1). First attempt showed healthz `token:false` (should be `true`) —
`nexus-svc-inject` had written 0 vars. Root cause: systemd user manager's global
`BW_SESSION` (`systemctl --user show-environment`) was STALE relative to the current
valid session in `/dev/shm/nexus_session` (confirmed: `bw` rejected it, prompted for
the master password) — this is the known `gotcha_stale_systemd_bw_session.md` pattern
("wrote 0 vars" is its exact signature). Fix: `systemctl --user set-environment
BW_SESSION=<value from /dev/shm/nexus_session>`, then re-ran `splatlab-safe-restart`.
Second attempt: `nexus-svc-inject: wrote 36 vars`, healthz `token:true`, service active.
§1D′ is now genuinely live for every new insv/equirect dispatch.

Committed locally (not pushed): see git log.

Committed locally (not pushed): see git log for the Problem/Fix/Verification/Risk message.

## Phase 3.1 segment-merge probe: GATE PASSED 2026-07-05 (Sonnet 5 swarm session)
`tools/probe-segment-merge.sh` proves Architecture A (SfM-level join) works, after 3
failed attempts that each surfaced a real, distinct problem — none of them fixed by
retrying, each fixed by understanding root cause:

**Attempt 1 — FAILED, non-architectural**: dev clip, SEG1=[0,40)/SEG2=[30,70) @3fps,
independent per-segment databases. seg1 succeeded (960/960, 4MB points); seg2 posed
all cameras but triangulated ZERO points — the [30,70)s window of that specific clip
lacks parallax somewhere past t=40s (operator likely held still). Same failure class
as G3 attempt 1. Fixed by switching to the pool clip
(`VID_20260514_073947_00_002.insv`) with SEG1=[15,55)/SEG2=[45,85), centered on the
window G3 already proved has strong parallax (`[30.837,75.837)`).

**Attempt 2 — per-segment SfM PASSED, model_merger CRASHED (real architecture bug)**:
both segments' independent SfM succeeded cleanly (959/960 @ 70k pts, 960/960 @ 113k
pts) — proves the per-segment SfM step is solid given real parallax. But
`model_merger` SIGABRT'd: `Check failed: src_images[i]->ImageId() ==
tgt_images[i]->ImageId()` (`estimators/alignment.cc:76`,
`ReconstructionAlignmentEstimator::Estimate`). Root cause, traced through
colmap4-src: model_merger's alignment estimator requires the SAME numeric ImageId for
a common-by-name image across both input models — true only when both models load
from ONE shared database (colmap's actual "merge disconnected sub-models of one run"
use case, doc/faq.rst:315), not two independently-run segments with independently
assigned IDs. `database_merger` is NOT a workaround: `Database::Merge`
(scene/database.cc:60) explicitly refuses to merge databases sharing any image name —
built for disjoint sets, the opposite of what an overlap join needs. **Fix**: one
shared database + one shared `feature_extractor`/`sequential_matcher` pass, then two
bounded `global_mapper --GlobalMapper.image_list_path <segN.txt>` calls (confirmed in
source — `option_manager.cc:1195` + `global_pipeline.cc:81-82` — this genuinely
restricts the DatabaseCache input, not a post-hoc filter). Script rewritten to this
design; `model_merger` succeeded (ratio 1.000) on the very next attempt.

**Attempt 3 — merge succeeded, bundle_adjuster DIVERGED (NO_CONVERGENCE, one runaway
point)**: registration ratio 1.000 (1679/1679), but post-BA mean reprojection error
was an astronomical garbage value (~1.2e149 px) — one degenerate correspondence
admitted at `model_merger`'s default `--max_reproj_error 64` (a loose RANSAC inlier
threshold for the alignment sim3, not a point-quality filter) ran away during BA.
Diagnosed and fixed WITHOUT re-running any SfM: reused the existing seg1/seg2 sparse
models on disk, tightened `model_merger --max_reproj_error` to 8, ran colmap's own
`point_filtering` (`--max_reproj_error 4 --min_track_len 2`) on the merged model
before `bundle_adjuster`. Pre-BA error 2.67px → 1.32px (tight merge) → 0.99px
(+filter) → 0.89px stable post-BA (verified by hand against the real attempt-2
models before landing in the script).

**Attempt 4 (FINAL, clean single-invocation run of the fully-fixed script) — GATE
PASS**:
```
seg1 registered: 959   seg2 registered: 960
union (distinct names, seg1|seg2): 1679
merged (post-BA) registered: 1679
registration ratio (merged/union): 1.000 (gate: >= 0.80)
mean reprojection error: 0.807224px (gate: <= 1.50px)
GATE: PASS (ratio OK, reproj OK)
```
Full log: `tools/probe-segment-merge-run4.log` (repo-root, gitignored via `*.log`).
Output artifacts: `tools/probe-segment-merge-output/` (gitignored).

**⚠️ Correction needed before Phase 3.2 implementation**: PLAN.md's Phase 3.2 §3.2 text
(and its Phase 3 preamble) describes independent per-segment databases — that's the
design attempt 2 disproved. Phase 3.2 must instead: one shared database per job,
`feature_extractor`+`sequential_matcher` run once over the full frame set,
per-segment `global_mapper --GlobalMapper.image_list_path` calls (this is still the
independently-checkpointable expensive step — the restart-survival property is
preserved), `model_merger --max_reproj_error 8` (not the default 64), then
`point_filtering --max_reproj_error 4 --min_track_len 2` before the final
`bundle_adjuster`. `tools/probe-segment-merge.sh` is the reference implementation for
all of this — Phase 3.2 should port its logic into `_plan_3d_job`/new pipeline stages,
not re-derive it.

Committed (script + this STATUS.md, NOT the gitignored output/log): see git log.

## Photo-capture reliability + survey polish pass (2026-07-05, RToony's call)
RToony deprioritized the 360-video segmentation work (Phase 3.2 on hold) in favor of
the photo/standard-capture path — "smaller, quicker, easier for a small site." Two
review agents audited the escalation chain and survey/measurement tools; RToony picked
the two quick fixes to land now (DXF export, escalation UX polish, and the two
dimension-bug fixes below are deferred/already-fixed — see below):

- **fix(splat) 4e1e4f3**: compress/webopt now use the existing `_record_stage_failure`
  helper (previously langfield-only) — a failed .spz/web.ply/langweb build is no longer
  indistinguishable from a success in job meta. 7 new tests.
- **test(scale) 0e01afe**: `test_scale_rejects_garbage[nan/inf]` — root cause was
  httpx's own request encoder refusing NaN/Infinity (RFC 8259 compliant), not the
  endpoint (which was always correct). Fixed by sending raw bytes for those two cases.
  **Full backend suite now 183/183 green — first fully-green run this program.**
- **fix(splat) a8662b4**: sparse ("Few Photos") jobs seeded `sfm_tried` as an empty set
  (not escalation-eligible), so a failed sparse job's error message fell back to
  claiming "Auto-fallback tried colmap" even though it ran mast3r-sparse and never
  touched COLMAP — directly misleading for the small-site/few-photos use case.
  Extracted `_seed_sfm_tried()`, 3 new tests.
- **fix(survey) 2ce5692**: scale calibration UI (`spark-scene-viewer.tsx`) — (1)
  `calibDim` no longer silently falls back to "the last dimension in the list" when
  nothing is explicitly picked (bit a user who deletes their calibration target); (2)
  recalibrating now requires a two-click confirm (same idiom as scene delete) since
  meters_per_unit is one scalar shared by every dimension's displayed length. Live-
  verified on the Garden scene (see commit for the verification detail — the browser
  automation tool's synthetic clicks didn't register on this specific button, a tool
  quirk; a real dispatched click confirmed both fixes end-to-end).

**Deferred (not started, real findings on record for later)**:
- DXF/LandXML export: `ezdxf` is installed and the scale endpoint's own docstring
  claims "measure/DXF/LandXML all hang off it," but no export code exists at all —
  dimensions are 100% client-side sessionStorage with no save/export path. Biggest
  gap for the civil-survey use case; real feature work, not a quick fix.
- Escalation UX polish: `sfm_tried`/`reroute_count` never reach the frontend (the
  "Retry with global SfM" button doesn't know if that solver was already exhausted);
  reroute reasons only appear in scrolling logs, not the stage rail; exhaustion
  guidance text is video-flavored regardless of actual capture type (photo vs video).

## CAPTURE COACH PHASE 0 — fog-fingerprint gate + calibration PASSED (2026-07-11)

**Goal**: score reconstruction health so fog scenes stop being discovered after hours
of GPU spend (the 07-10 root-cause finding, splat-geometry-health-gate memory). Plan:
~/.claude/plans/lets-brainstorm-my-next-functional-backus.md (Phases 0→2 + earned
enforcement). REPORT-ONLY per the metric-trust doctrine.

**Shipped (new files only, no server change, no restart)**:
- [x] `backend/health/fog_gate.py` — langfield-spike env; renders ED depth + RGB at 6
      spread training cameras (640px downscale); per-cam metrics over opaque px;
      writes `<job>/_health/fog.json` + side-by-side [RGB|turbo log-depth] receipts.
      Exit 0 = analysis ran (any verdict); non-zero = execution failure only.
- [x] `backend/health/run_health.sh` — run_langfield.sh clone minus SAM (env
      hardening: unset CPATH/LIBRARY_PATH, pin CUDA_HOME). `SPLAT_HEALTH_PYTHON` override.
- [x] `backend/health/backfill_fog.py` — stdlib CLI; REFUSES while any meta.json is
      starting/running or GPU free <6GB (--force); --write-meta patches meta["health"]
      (only safe because of that preflight); writes calibration report + summary.json.
- [x] `tools/gates/gate_p0_fog_calibration.sh` — executable acceptance gate.

**METRIC CHANGE (calibration finding)**: the raw 07-10 fingerprint (p95/p5 spread < 3)
failed on the MIXED selfie scene splat_98095cb055 — every camera has p5 pinned at the
near plane (cocoon contamination) but 3 cams punch through to real structure, inflating
p95 (spread up to 45 while still junk). Verdict now uses per-camera **shell fraction**
(share of opaque px with depth ≤ 0.03 = 3× near plane): fog cam = shell ≥ 50% @ acc ≥
.98; clean cam = shell ≤ 5% AND p50 ≥ 0.1; 2/3 camera majority (CAM_FRAC=0.66 — 0.67
rejects a legit 4/6). Spread still reported for context. All thresholds HEALTH_FOG_* env.

**GATE PASS (exit 0), full separation on graded scenes** (~4s/scene after JIT warm):
| scene | graded | verdict |
| splat_5177f8d99a | FOG (07-10) | FOG 6/6 |
| splat_98095cb055 | FOG (07-10) | FOG 4/6 (mixed — operator cocoon + real office) |
| splat_32d926d9 garden | HEALTHY | HEALTHY |
| kitchen/bonsai/counter | unlabeled | HEALTHY (matches langfield-verified geometry) |

**⚠️ FINDING — pool scene splat_192e4223fb is FOG**: depth pinned at the near plane
(spread 1.00, p50 0.0100) at ALL cameras; RGB receipt is a structureless smear. Its
"HEALTHY" label was an ungraded assumption (07-05 acceptance passed on 90% REGISTRATION
— registration ≠ reconstruction). Gate asserts only RToony-graded scenes; pool is a
pending-grade row. Receipts: ~/reports/2026-07-11-capture-coach-fog-calibration/index.md.

**Next (gated on RToony's receipt review)**: Phase 0.5 = wire `health` stage after
export (kill-switch SPLAT_HEALTH_GATE) + meta["health"] + SceneCard badge +
CaptureHealthCard; then Phase 1 capture probe, Phase 2 upload heuristics. Enforcement
stays opt-in-later per gate.

## CAPTURE COACH PHASE 0.5 — health stage WIRED + LIVE, report-only (2026-07-11)

RToony graded the Phase-0 calibration receipts ("receipts check out") → go.

**Backend (splat_route.py)**:
- [x] Constants + `_health_available()` (runner + langfield-spike python only — NOT
      `_langfield_available()`, which also demands sam2) + `_append_health_stage()`
      (kill-switch `SPLAT_HEALTH_GATE=0`; extracted as a helper so the guard is unit-testable).
- [x] `health` planned right after train/export, before compress/webopt/langfield;
      generative lane naturally excluded (early-returns before the append).
- [x] Runner branch cloned from the langfield best-effort contract: whole body wrapped,
      `_run_locked_stage` under HEAVY_GPU_LOCK (HEALTH_VRAM_MB=4000), verdict from
      `_health/fog.json` → `_patch_meta(health={"v":1,"fog":{...,"enforced":False}})`,
      failure = `_record_stage_failure` + continue, provably never flips final_status.
- [x] Receipt route `GET /jobs/{id}/health/receipt/{name}` (regex-guarded, webp/png).

**Frontend**: contracts.ts `health?` block (all-optional, old scenes deserialize
unchanged); STAGE_HUMAN/SHORT "Checking capture health"/"Health"; SceneCard verdict
pill (amber "likely fog" / green "healthy" / gray "unverified") sharing the bottom-right
corner with the searchable badge; `CaptureHealthCard` under the featured viewer —
verdict headline (report-only wording), reshoot coaching, per-camera receipt strip.

**Gates + deploy (all receipts real)**:
- `tools/gates/gate_p05_wiring.sh` → **PASS exit 0**: 7/7 pytest
  (test_health_stage_bookkeeping.py — non-fatal on failure/exception, meta persisted,
  plan guard + kill-switch), frontend build OK, live API serves 7 scenes with verdicts,
  receipt route returns image/webp with bearer auth.
- Deployed via `splatlab-safe-restart` (no jobs in flight); healthz OK.
- Backfill `--write-meta` patched all 7 calibration scenes → badges live in gallery.
- Live traversal proof: Test Flight `splat_7f3d29f3de` (pool clip, 30s trim, glomap)
  dispatched with `health` in stages_planned after export — verdict lands when it
  finishes (expected FOG per the Phase-0 finding on the full-clip scene).
- Gotcha (repeat offender): `python3 - <<'PY'` heredoc CLOBBERS a curl pipe into
  stdin — fetch inside the script. And never pipe a gate through `| tail` (masks exit).

**Enforcement stays OFF** (`enforced:false` everywhere). The flip
(`SPLAT_HEALTH_ENFORCE_FOG` skipping langfield/mesh) is a later, per-gate, revocable
opt-in after RToony grades real-run receipts. Next: Phase 1 capture probe, Phase 2
upload-time Tier-0 heuristics (see the capture-coach plan file).

## 360 FOG ROOT CAUSE + RIG LANE (2026-07-11, Capture Coach spin-off)

**ROOT CAUSE of every insv fog cocoon — pinned by probes, not vibes**
(full ledger: probe-operator-mask/STATUS.md):
- Masking arms (seam bands, person masks) moved NOTHING: FOG, shell 1.0.
- Geometry probes on the SfM output: camera-path bbox 1584 units vs point-cloud 129
  (12×); same-frame 8-crop camera centers (physically identical) solved median
  **5.1 units apart** vs true step 0.13 — the unrigged fan-out scatters poses, the
  trajectory explodes, normalization collapses real geometry to depth ~0.01 = the
  fog fingerprint. All 3 insv scenes FOG, all 4 pinhole scenes HEALTHY.
- **Arm R (colmap4 panorama_sfm rig)**: 1080/1080 registered, shell 0.997→0.23,
  first recognizable insv reconstruction (pool receipts in the arm_R/_health dir).

**Gate v2 (fog_gate.py)**: sky-pitch exemption (cams pitched >+20° up = no parallax,
near-shell is legal — env HEALTH_FOG_SKY_PITCH_DEG) + mask-aware stats (person-masked
px are unsupervised, excluded) + default 8 probe cams + pitch in every receipt label.
**Recalibrated: gate_p0 PASS exit 0** (graded verdicts unchanged; report at
~/reports/2026-07-11-capture-coach-fog-calibration-v2/). Arm RP under v2: honest
UNCERTAIN (draft 7k iters; real 30k + floater cleanup expected to improve).

**RIG LANE WIRED, OPT-IN (`sfm_backend="rig"`, equirect video only)**:
- `backend/rig/render_rig.py` (colmap4 env): sphere → 12 virtual views (4 yaw ×
  3 pitch, 90°) + per-pixel ownership masks + rig_config.json.
- `_rig_sfm_command`: ffmpeg STRIDE extract → render_rig → colmap4 feature_extractor
  (per-folder SIMPLE_PINHOLE + ownership masks, GPU) → **rig_configurator** →
  sequential_matcher (rig_verification + skip_same_frame + loop_detection) →
  global_mapper (refine_sensor_from_rig 0, focal/extra fixed) → guards → ns-process-data
  --skip-colmap. Stage name `rig_sfm` (frontend labels added).
- NOT in SFM_ESCALATION, NOT default — falls back to colmap silently on non-equirect/
  non-video/missing toolchain. Default-flip = RToony's call after graded real runs.
- NEW DEP: pycolmap 4.1.0 pip-installed in colmap4 env (render script needs only
  cv2/scipy/PIL; pycolmap used by the spike's panorama_sfm arm, kept for parity).
- Tests: test_rig_sfm_plan.py (+ suite green, 14/14). Live acceptance:
  splat_ff2b9dd395 dispatched via the pipeline with rig_sfm planned.

## RIG LANE LIVE ACCEPTANCE — PASS (2026-07-11 10:57)
- First flight splat_ff2b9dd395 FAILED in sequential_matcher (exit 134): rig-config
  camera_params was a comma STRING; colmap's parser iterates it as a JSON array →
  empty params → poisoned camera rows. Fixed b1a594a (array form).
- Retry **splat_3885b68e54 COMPLETED end-to-end**: stitch → rig_sfm → process → train
  → export → health → compress → webopt, stages_failed=[], **~11.5 min total** (vs
  ~14.5 min for the glomap flight — the rig lane is FASTER despite 12 views/frame).
- Health (gate v2, no person masks, draft 7k): UNCERTAIN — shell 0.555, spread 21.8,
  4 counted / 4 sky-exempt. Consistent with the spike arms; receipts in the gallery
  health card for RToony's grading.
- **Open for default-flip**: RToony grades the live receipts; then candidates =
  full-quality 30k run, person-mask training stage (masks proven, ~30s/720 crops),
  rig-first escalation for equirect. All opt-in until graded.

## DEFAULT-FLIP: 360 captures route to the rig lane (2026-07-11, RToony /goal)
- The problem: the rig fix only worked if you typed sfm_backend="rig" — a default
  insv job still took the fog-producing unrigged fan-out.
- Backend: SFM_ESCALATION = [rig, colmap, glomap, mast3r] with EQUIRECT_ONLY_SOLVERS
  guard (flat captures never route into rig); _plan_3d_job upgrades default-colmap
  equirect VIDEO to rig when rig_available; legacy rungs remain the A1-gate fallback.
- Frontend: flights no longer force glomap when the rig toolchain exists (that
  override would have bypassed the server flip); glomap kept when rig is missing.
- Tests: golden-snapshot helper pins _health_available False (goldens = SfM/stitch
  drift only; health has its own plan-guard tests). Suite 199/199.
- LIVE PROOF: splat_f4c9416afb dispatched with NO sfm_backend → planner routed it
  to rig_sfm (meta shows requested colmap default + rig_sfm planned).
- CONFIRMED 13:23: splat_f4c9416afb (default path, auto-routed) COMPLETED all 8
  stages, ~11.5 min — health 0/4 fog cams, median shell **0.014** (vs 0.997
  baseline / 0.555 first rig run), spread 8.96, 2 healthy cams. Best 360 result
  in the program; one cam short of formal HEALTHY. Grading datapoint for RToony.

## RESUME-ON-START SHIPPED (2026-07-11 evening, Lane B #1)
- `resume_orphan_jobs()` replaces mark-failed-only at startup: the NEWEST orphaned
  in-flight job is re-planned from its persisted meta (every request knob persists in
  _new_meta) and relaunched under the SAME job_id from stage 1 (stage scripts are
  self-cleaning). Guards: SPLAT_RESUME_ON_START=0 kill-switch, RESUME_MAX_AGE_HOURS
  (12, env), RESUME_MAX_RESTARTS=2 crash-loop cap, input-still-exists, one restart
  only (single-job GPU), any error → honest failed marker (never wedges startup).
  Older orphans still get the failed marker. restart_count/restarted_at in meta.
- Tests: test_resume_on_start.py (7 cases). Suite 205/205.
- ⚠️ NOT YET DEPLOYED — full pool run splat_7c369afbde in flight; deploy via
  splatlab-safe-restart after it completes.

## WORKSTATION SAFETY ENVELOPE (2026-07-11 crash follow-up)
- `splat_7c369afbde` triggered a second abrupt platform reset while its x264
  stitch used about 12 cores. Telemetry ruled out OOM, thermal, GPU, and storage
  exhaustion; the load exposed underlying platform instability.
- The reboot auto-deployed resume-on-start and relaunched the same job. Recovery
  is now opt-in only: `SPLAT_RESUME_ON_START=1`. Unset, invalid, and `0` values
  leave interrupted jobs stopped for manual review.
- `splatlab.service.d/60-safety-guard.conf` confines the full service tree to
  E-cores 8-15, a four-core average CPU quota, nice 10, low CPU weight,
  32G/48G memory thresholds, 8G swap, and 512 tasks. This covers stitch, COLMAP,
  training, export, and every descendant, not just ffmpeg.
- Raw Insta360 work now defaults to the existing 30-second Test Flight. Full
  capture builds require a deliberate toggle and staged promotion.
- The interrupted full run is preserved as failed with stitch complete. It must
  not be resumed until the hardware gate and bounded flight ladder in
  `~/reports/splatlab-safe-evaluation-2026-07-11/plan.md` pass.

## ROBUSTNESS WAVE (2026-07-18, RToony /plan: "hardware acceptance first, then robustness")

Gate: `tools/gates/gate_robustness_wave.sh` — **PASS exit 0** (43 wave tests, full suite
348, tsc 23-error baseline, build clean). Commits 5a26f54..f6cf297:
- [x] **Dirty-tree audit landed** (5a26f54 supervised compute-unlock backend/tools,
      b84f0ac gate-visibility UI) — audit caught + fixed a REAL TDZ render crash
      (createDisabled referenced startMutation before declaration, splat.tsx:170).
- [x] **glomap default for photo folders** (2f60824): _plan_3d_job upgrades default-colmap
      flat image dirs to glomap when colmap4 present (07-17 A/B: 29/29 vs 17/29).
      Video/equirect/sparse untouched; escalation keeps colmap as fallback rung.
- [x] **Escalation surfacing** (83f003c): meta gains sfm_start_solver (RESOLVED) +
      sfm_tried + reroute_count + structured sfm_reroutes; RerouteChips on stage rail +
      failed card; Retry-with-glomap disabled when already tried; _recapture_guidance()
      keys exhaustion advice to photo/video/360.
- [x] **Optional-stage bookkeeping** (4fee640): failed langfield/compress/webopt/health
      no longer ALSO append to stages_completed (stages_failed is the record; skips
      unchanged; final_status semantics untouched).
- [x] **Capture Coach Phase 1** (3d29117): backend/health/probe.py pre-train probe at the
      A1 gate (pass AND fail paths) — trajectory/cloud bbox ratio (the 07-11 12x
      fingerprint), map density, orbit/walkthrough shape → meta.health.probe
      (MERGES with fog; fog patch now merges too). Report-only per doctrine.
- [x] **Capture Coach Phase 2** (c8badfc): backend/health/precheck.py + POST
      /api/splat/precheck (NOT compute-gated) — pure-Pillow blur/exposure/static +
      density advisories at upload; amber advisory panel; Create NEVER disabled.
- [x] **Polish** (f6cf297): 360 overhead-stick tip on the upload card; insv full runs
      omit num_frames_target (server §1D′ rule is the single source of truth).
- **Pre-Flight-A**: MemTest 4-pass ACCEPTED 07-16 16:21 (operator photo clarification);
  staged `~/scripts/splatlab-preflight-a-run.sh` (dry-run default; --apply stops
  langfield/sam-video-lab aux units, runs tools/gpu-hardware-acceptance.py with the
  three boot-bound evidence records + all confirmations, verifies receipt+sidecar).
  MUST run from a plain terminal (script refuses inside an aipc-safe-run scope).
  STOP at PASS_PRE_FLIGHT_A — marker stays; Flight A is a separate explicit decision.

## LOCATE-IN-THE-WORLD SHIPPED (2026-07-15, RToony /goal — splatedit.app-inspired)
Pin any scene to real WGS84 coordinates (feature work only — GPU pause, maintenance
marker, and Flight A ladder all untouched; endpoints are metadata/CPU-only and
deliberately NOT behind require_heavy_work_admitted, same policy as /scale).
- **Backend (b3168ba)**: `geo_route.py` mounted like edit_ops. POST /jobs/{id}/geo →
  meta["geo"] = {lat, lon, alt_m, heading_deg (bearing of scene +Y), anchor_scene,
  source, set_at}; {"geo": null} clears. GET geo/suggest = GPS from photo EXIF
  (Pillow) / video tags (ffprobe ISO6709) / embedded tracks (exiftool -ee), fail-soft.
  GET geo/footprint[.webp] = transparent top-down plan projection (`geo_footprint.py`,
  thumb.py-style CPU sampling, cached _preview/footprint.{webp,json}) with exact
  scene-unit bounds. GET geo/export?fmt=geojson|kml. Scene→ENU transform documented in
  the module docstring. 21 new tests; suite 313/313.
- **Frontend (602d594)**: "Locate" header button on /view/:jobId → lazy Leaflet modal
  (Esri satellite + OSM, Nominatim search, "Use photo GPS", draggable anchor +
  footprint overlay rotated by heading / scaled by meters_per_unit, opacity). Save
  writes /geo; /scale only on deliberate adjustment (checkbox guard over an existing
  calibration). Emerald "located" gallery badge. tsc 23 errors == baseline.
- **Live receipts (splat_f4c9416afb)**: footprint 541×768 renders the pool deck;
  heading 405.5→45.5; geo round-trips /status; KML/GeoJSON correct; clear → geo null
  (no visible state left); unauth 401; suggest [] on the GPS-less pool .insv (23k
  embedded records, 0 GPS tags — camera GPS was off). Deployed via splatlab-safe-restart
  (no jobs in flight), healthz token:true.
- Backlog ideas (NOT started): compass/north arrow in the 3D viewer once located;
  geo-anchored DXF/LandXML export (ties into the deferred survey-export gap);
  batch-locate from a GPS-tagged capture at upload time.

## DIGITAL TWIN KERNEL — P0 through P6a (2026-07-21/22, backfilled 07-22)
Splat → simplified/georeferenced solids kernel; plan `~/.claude/plans/fuzzy-foraging-moore.md`
(approved 07-21). This entry condenses P0–P5 (all shipped+deployed 07-21) and closes out P6a
(07-22) — full detail lives in memory `digital-twin-kernel-program-2026-07-21.md` and
`~/reports/2026-07-21-twin-kernel-day-digest.md`; this is the durable on-repo summary that was
previously missing.

- **P0 (e3ad59d/a76202b)**: opt-in `mesh` stage, `POST /jobs/{id}/mesh` — champion TSDF recipe
  (`gs-mesh o3dtsdf --voxel-size 0.015 --sdf-trunc 0.045 --depth-trunc 6`, `TSDF_ALPHA_MIN=0.5`)
  ported from `dn-splatter-probe`. Garden reproduces 78.81% LCC.
- **P1 (30a3550/a2cc5d4)**: `POST /jobs/{id}/geo/contours` — ground extraction (Track C) → cdt
  `survey_to_surface` → real EX-CONT-MJR/MNR-layer contours. Semantic ground filter (P5a,
  174f498/bb9af02) via langfield relevancy; section/ISO receipts auto-generated.
  **GLB gotcha (658718f)**: open3d 0.19 `write_triangle_mesh(.glb)` silently corrupts + returns
  True — mesh_report.py writes GLB via trimesh + mandatory readback now.
- **P2 (4cc6705/d66f9e0)**: `POST /jobs/{id}/geo/export` — probe-derived grid calibration (never
  trust convergence-sign conventions, measure instead) → site.dxf / surface.xml (LandXML) /
  site.geojson.
- **P3**: blender-mcp wired (official ahujasid upstream) — Blender 4.5.11 LTS installed
  separately (system 4.0.2 can't GPU-render Blackwell/5090). Cockpit launcher `blender-cockpit`.
- **P5b/P5c (9edda81/b4c7891)**: `POST /jobs/{id}/objects {"query": "..."}` — name an object,
  get isolated splat + mesh + a TripoSplat-generated proxy ICP-registered onto it. Proven on the
  garden table (icp_fitness 1.0) and, 07-21 field capture, RToony's **fire hydrant**
  (Santa Rosa, 49 Canon T3i CR2s, all 9 stages green from one POST, object mesh 100% LCC).
- **WS1/WS2 gates (215630d/2fb5ad7)**: `mesh_gate.py` (PSNR/SSIM/coverage vs source photos) +
  `mesh_completeness.py` (solid-gaussian distances) — quality receipts, not narrative.
- **Fix wave (3c9a90b)**: 14 confirmed findings from a 22-agent autonomous review closed in one
  commit (config-poisoning, guard hoisting, cached-mesh 500s, subset-ckpt leaks, stale-langfield
  guard, atomic contour staging). 407 tests at close.
- **ETH3D validation (840b121 + solidify-probe)**: laser-truth harness proves render-health gates
  (fog/acc) CANNOT certify mesh geometry — courtyard splat HEALTHY yet mesh vs laser = 130-171cm
  median error (sparse-view regime). Track B (depth-regularized extractors) blocked on a
  dn-splatter fork `--load-depths` compat bug — needs a proper fork-compat pass, not inline
  patches. Garden-class dense captures (RToony's actual capture spec) are a different, better
  regime.

### P6 — Scene Regeneration Lane (SAM3/TRELLIS-class), opened 07-22
Plan `~/.claude/plans/snazzy-gathering-dahl.md` (approved 07-22): enumerate + batch-isolate +
batch-proxy + assemble a FULL scene (not just one named object), all-local (SAM 3 + optional
TRELLIS.2). Doctrine: regenerated scenes are plausible-not-faithful — render/VR lane only,
mechanically enforced (quarantine dir + manifest + in-file tags + a refusal gate the survey lane
must fail against).

- **Step 0 de-risk spike: PROVEN, GO for P6b** (`~/tools/scene-regen-spike/STATUS.md`). The one
  algorithmic bet — SAM 3 text-prompted masks lift cleanly to per-gaussian instance sets via the
  existing PASS-B depth-gated lift + cross-view majority vote — holds with recipe
  `--min-views 2 --vote-frac 0.3`. Garden "table"/"flower vase" both PASS (vase lift actually
  *fixes* a known P5b langfield-query miss). Hydrant misses the pre-registered IoU floor
  (0.4994 vs 0.5) but root-caused as **reference contamination** (P5b's spatial expansion swept
  in ground-disc/shadow debris the SAM mask correctly excludes), not a lift failure — precision
  vs the reference = 1.000.
- **P6a SHIPPED + gate CLOSED (bdce4d3, then closed out same evening)**: `backend/mesh/
  provenance.py` (stdlib-only tag + quarantine rules + fail-loud survey-refusal, importable from
  every env), `proxy_register.py` retrofit (writes `transform_4x4` + `crop_camera_id` +
  `crop_box` + in-file PLY tag — needs `--crop-json` from `object_crop.py`'s new crop.json
  side-file), `scene_manifest.py` (schema + atomic writer + fail-loud validator), `sam3_doctor.py`
  preflight, `tools/gates/gate_p6a_scene_rails.sh`. 427 backend tests (20 new).
  ⚠️ **Gotcha**: the commit landed while `splatlab.service` was already running (started 16:49,
  commit at 20:13) — `proxy_register.py`/`object_crop.py` picked up the new code immediately
  (subprocess-invoked, read fresh each call) but `splat_route.py`'s new `--crop-json` wiring did
  NOT take effect until `splatlab-safe-restart` ran (it's imported once at process start). A
  post-commit rebuild without the restart silently produced a proxy.json missing
  `crop_camera_id` — always restart after a `splat_route.py`-touching deploy, not just when the
  gate says so.
  **Runtime close-out**: pinned `splat_513e89171d` (hydrant) + `splat_32d926d9` (garden) as
  scene-lane sources; restarted the service; rebuilt the hydrant `fire-hydrant` proxy (icp_fitness
  held at 1.0, `crop_camera_id: 8`). `bash tools/gates/gate_p6a_scene_rails.sh` → **GATE_P6A: PASS**
  (8/8 checks) — first real structural receipt that P6a actually holds, not just compiles.
- **P6b SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-22 night)**: `POST /jobs/{id}/scene/
  inventory` — enumerates every object in a scene, not just one named query. New:
  `backend/mesh/scene_views.py` (K evenly-spaced views, adapted from the spike's select_views.py),
  `backend/mesh/noun_consolidate.py` (clean/dedupe candidate nouns via SigLIP-cosine + curated
  STUFF_TERMS ground/vegetation split — pure-stdlib classify_stuff/clean_nouns, heavy SigLIP import
  isolated inside siglip_dedupe() so the module imports with zero ML deps), `backend/mesh/
  scene_sam3_masks.py` (one SAM3 model load, loops every "thing" noun × view), `backend/mesh/
  instance_lift.py` (the productionized multi-noun PASS-B lift+vote, conservation bookkeeping,
  optional regression vs known `_objects/*/object_indices.npz`, receipts). Route mirrors `/objects`
  (lock, arbiter lanes `scene-views`/`scene-sam3`/`scene-lift`, 409 contracts, sam3_doctor preflight
  BEFORE any GPU work). `nouns` body field is an explicit override that skips Qwen3-VL/langfield
  auto-sourcing entirely — the cheap HITL safety valve. Real captured gaussians only, no generative
  tag needed (P6d proxies are where that applies). 442 tests (15 new). `tools/gates/
  gate_p6b_instance_inventory.sh` — PASS.
  **Live run on garden (`splat_32d926d9`, auto-sourced, no explicit nouns)**: Qwen3-VL proposed
  "round wooden table", "flower vase", "blue ball" (langfield-vocab worker was inactive this run —
  fail-soft, VL-only sourcing). Table: 38,327 members, 8/8 views, regression IoU 0.7656 vs the
  known reference (recall 0.8115/precision 0.9312) — matches the Step 0 spike's 0.764. **Flower
  vase: 1,198 members — EXACT match to the Step 0 spike's number**, strong cross-validation that
  production faithfully reproduces the proven mechanism. "Blue ball" (a real object under the
  table, confirmed in the crop receipt) correctly VETOED rather than hallucinated — SAM3/lift
  declined an unconfident detection instead of fabricating one. Conservation held (39,525 claimed,
  0 overlap). Receipts: `~/projects/splatcli/outputs/3d/splat_32d926d9/_scene/` (overlay +
  per-instance crops) — RToony's eyeball grade is the next checkpoint per the HITL doctrine.
- **P6c SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23 night)**: `POST /jobs/{id}/scene/
  isolate` — materializes P6b's already-determined instances into Blender-ready per-instance
  `object.ply` + a `background.ply` complement. New `backend/mesh/batch_isolate.py`: reuses
  P6b's SAM3-lifted membership directly (NOT a re-run of the older per-query DBSCAN
  `object_isolate.py` clustering — that's exactly the mechanism P6b was built to supersede,
  proven on the vase case) with first-claim-wins bookkeeping across instances (largest first,
  P6b's own size-sorted order), and the `write_splat_ply`/`sanity_sum_ok` convention lifted
  verbatim from the proven `~/tools/langfield-isolate-probe/isolate_export.py`. Never aborts on
  a thin instance (`SKIPPED:too-few-members-after-dedup`, batch continues). Receipt: a REAL
  gsplat RGB render with claimed gaussians' opacity zeroed (not a photo overlay) — genuinely
  shows the scene with instances removed. 446 tests (4 new). `tools/gates/
  gate_p6c_batch_isolate.sh` — PASS.
  **Live run on garden**: both instances built clean (table 38,327, vase 1,198, zero overlap,
  `sanity_sum_ok: true`, 39,525 claimed / 1,287,086 background of 1,326,611 total).
  ⚠️ **Honest finding from the background-removed receipt**: the ball (never an instance,
  correctly excluded) stays fully intact, and walls/grass/hedge render clean — but the table
  leaves a **ghostly translucent disc** rather than a clean hole, and part of the vase structure
  is still faintly visible. SAM3's tight masks are high-precision but don't claim every gaussian
  contributing to an object's visual footprint (unlike the old expand-based DBSCAN, which
  over-captures on purpose) — some residual low-opacity gaussians near the object's location
  remain. Not a bug; a real limitation to weigh for P6e/P6f (a hole-fill/backdrop-completion
  probe is already parked in P6x if graded unacceptable). Receipts:
  `~/Downloads/splatlab-scene-isolate-garden-background*.png`.

## HYBRID RECALL-EXPANSION — `--recall-expand` on batch_isolate.py (2026-07-23, autonomous day session)
RToony's question after seeing the ghost-disc finding: combine SAM3's precision with the
older DBSCAN+expand approach's recall? Probed in `~/tools/hybrid-recall-probe/` (full
writeup + both scene-class verdicts there). **Answer: yes, but scene-class-dependent —
shipped as opt-in, not a default.**
- **Mechanism**: SAM3 core (trusted identity) → KDTree radius query around the core's own
  members (shell dilation, radius = `dilation_mult` × the scene's typical nearest-neighbor
  spacing — scale-invariant, not object-size-based) → kept only if SigLIP relevancy to the
  instance's own label clears `rel_floor` (reuses `gauss_emb.npz`, no model reload for the
  lookup). Reuses `object_isolate.py`'s proven relevancy math.
- **PROVEN on garden** (multi-object, 971 dense views): `dilation_mult=10, rel_floor=0.30`
  — table core 38,327→41,273 (+7.7%), the large translucent ghost disc from the original
  P6c run substantially cleared; grass/pavement/hedge/ball stayed clean (no debris pulled
  in). Live-verified through the production route (`{"recall_expand": true}`), receipts:
  `~/Downloads/splatlab-p6c-recall-expand-garden-final*.png`.
- **PROVEN INSUFFICIENT on hydrant** (single-object, 49 tight close-up photos): core-only
  removal (14,295 members) left the hydrant visually almost fully intact; `dilation_mult=10`
  barely moved it (verified NOT a code bug — opacity zeroing measurably applied, `opac.sum()`
  dropped by 11,236.8). A tight close-up capture has far more locally-redundant/overlapping
  gaussians contributing to an object's visual mass than either SAM3's mask or a modest
  dilation reaches — a deeper, different problem than garden's ghost-disc, left unsolved
  and explicitly flagged rather than force-fit.
- **Separate honest finding (garden vase)**: the dried flower/frond sticking out of the vase
  was never claimed by SAM3's "flower vase" core at all (a masking-granularity question, not
  a recall-expansion bug) — stays visible in both before/after as real background content.
- Shipped: `batch_isolate.py --recall-expand --gauss-emb <path> [--dilation-mult 10]
  [--rel-floor 0.30]`; route body `{"recall_expand": bool, "dilation_mult": float,
  "rel_floor": float}` on `POST /scene/isolate` (needs a built language field). 448 tests
  (2 new). `gate_p6c_batch_isolate.sh` extended, still PASS.
- **Next**: P6d — batch proxy + gated registration (loop unchanged P5c per instance under ONE
  20GB TripoSplat lease; `SKIPPED:<reason>` degrades to `provenance:captured`, never aborts).
  P6e (ground/environment), P6f (assembly+Blender+contamination gate) follow in order, each
  behind its own gate/receipt/HITL checkpoint per the approved plan.

## P6d SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23, autonomous day session)
`POST /jobs/{id}/scene/proxy` — loops the unchanged, already-proven P5c crop→TripoSplat→ICP
chain over every P6c-built instance. New `backend/mesh/proxy_triptych.py` (capture crop |
generated proxy preview | registered overlay — top+front orthographic scatter, deliberately
simple matplotlib math over a full perspective render, to stay robust). Route phases: (A)
crops CPU-only, per instance, before any GPU; (B) TripoSplat generation for ALL instances
under ONE shared 20GB lease (not one lease per instance); (C) ICP-register + triptych,
CPU, per instance. Per-instance `SKIPPED:crop-failed|generation-failed|registration-failed`
never aborts the batch. 453 tests (5 new). `gate_p6d_batch_proxy.sh`: verifies every built
element carries registration numerics AND the in-file generative PLY tag (P6a provenance
rails) — PASS.
⚠️ **Gotcha**: `object_crop.py` needs an `object.json` with a bbox, which P6c's
`batch_isolate.py` didn't write until this same session — added it (writes bbox from the
ACTUAL final claimed points, not P6b's pre-dedup bbox). First live P6d attempt on garden
0-for-2 SKIPPED:crop-failed because `/scene/isolate` hadn't been re-run since that fix landed
— object.json genuinely didn't exist on disk yet. Re-ran P6c, then P6d succeeded. Lesson:
after ANY change to a script another phase's route depends on, re-run the UPSTREAM phase
before testing the downstream one, not just restart the service.
**Live-verified on garden**: both instances built, **icp_fitness 1.0 on both** (table
rmse 0.0232, vase rmse 0.0099). Triptych receipts show real, informative signal: the
table's proxy is a near-perfect top-down silhouette match (thin captured-blue rim visible
at the disc edge) with minor leg-shape difference in the front view; the vase's captured
point cloud is visibly more diffuse/scattered than its clean regenerated proxy (consistent
with recall-expand's high candidate-acceptance rate on that instance, noted honestly, not
hidden). Also honestly notable: TripoSplat regenerates whatever the crop photo shows, so
the "table" proxy includes the vase sitting on it too (crops include visible context, not
just the tightly-masked object) — a real characteristic of crop-based generation worth
knowing before P6f assembly. Receipts: `~/Downloads/splatlab-p6d-proxy-triptych-*.png`.
- **Next**: P6e — ground + environment (`semantic_ground.py` wired into the scene lane,
  persisted `ground_gaussians.npz` → TIN → `ground_mesh.glb`). Then P6f (assembly + Blender +
  contamination gate). A comprehensive multi-agent adversarial review pass over everything
  shipped today (hybrid recall-expand, P6d) plus a fresh look at P6a-c is still queued before
  end of day, per the approved autonomous-run plan.

## P6e SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23, autonomous day session)
`POST /jobs/{id}/scene/ground` — real captured ground gaussians → a scene-unit TIN →
splat-colored Y-up GLB. Render/VR lane only (`provenance:ground-derived`), deliberately NOT
the survey lane (`POST /geo/contours`/`ground_extract.py`, which needs a real CRS + geo
anchor this scene doesn't have — garden has scale but no anchor). New: `backend/mesh/
ground_mesh_build.py` (reuses `ground_extract.py`'s proven cell-bin + 15th-percentile-z +
spike-rejection + largest-connected-component algorithm VERBATIM, just staying in scene
units instead of transforming to ENU/CRS — no geo/scale requirement at all), `backend/
mesh/ground_mesh_receipt.py` (top+oblique renders, same Open3D EGL `OffscreenRenderer`
pattern as `mesh_report.py`, `defaultUnlit` shader to read the GLB's baked vertex colors).
Reused as-is: `semantic_ground.py` (already wired for P5a/geo_route, langfield-spike env),
`twin_finish.py` (6-NN color transfer from the scene's existing `_preview/splat.ply` — no
new export needed). 458 tests (5 new). `gate_p6e_ground.sh` — PASS.
**Live-verified on garden, worked cleanly first try** (no debugging detour, unlike P6d):
8,578 ground points, 17,139 triangles, real extent 9.25m × 11.51m footprint (0.95m
vertical range — correctly flat terrain). Receipts show genuinely legible, correctly
colored ground (green grass, tan pavement patch) in a real, irregular garden-boundary
shape. Honest finding: one small pyramidal spike artifact visible near center in the
oblique view — a likely single outlier cell the spike-rejection pass didn't catch: noted,
not chased further (timeboxed). Receipts: `~/Downloads/splatlab-p6e-ground-*.png`.

## COMPREHENSIVE REVIEW + FIX WAVE (2026-07-23, autonomous day session, 44-agent workflow)
5 parallel reviewers (correctness/security/resource-safety/test-quality/doctrine) over the full
`840b121..HEAD` diff (P6a close-out through P6e, 3,644 lines) → adversarial refute-by-default
verification (3 lenses per finding). **13/13 raw findings survived verification** — all real.
9 fixed this session (low-risk, well-scoped, matched "never break working things"); 4 left for
RToony's call (design-decision territory: DELETE/reset semantics for `/scene/*` artifact trees).

**Fixed:**
- **`scene_inventory` (P6b) now merges `meta["scene"]` instead of overwriting it** — this is the
  ONE route users are told to re-POST (the documented HITL correction flow), and it was silently
  erasing `isolate`/`proxy`/`ground` summaries every re-call even though the on-disk artifacts
  stayed untouched. **Live-verified on garden**: re-POSTing `/scene/inventory` gave `inventory` a
  fresh `built_at` while `isolate`/`proxy`/`ground` kept their original timestamps and data intact.
- `scene_inventory`'s `_work` scratch (frames/masks, can be multi-GB) now cleans up on EVERY exit
  (try/finally), not just the two success paths — every failure branch used to orphan it.
- `SceneInventoryBody.nouns: []` (explicit "there's nothing here") no longer collapses to the
  `None`/absent case and silently re-triggers Qwen3-VL + langfield auto-sourcing.
- `scene_inventory`'s zero-things short-circuit now still calls `audit_operator_event` — it used
  to return from inside the lock before reaching the audit call, silently dropping the record of
  a real GPU-time operation from the audit trail.
- `scene/isolate`'s `recall_expand=true` path now runs `_langfield_stale_guard()` before spending
  any GPU lease (it only checked `gauss_emb.is_file()`, which is true even when STALE) — restores
  the exact "preflight before any GPU work" doctrine this same diff's own comments cite as already
  fixed elsewhere, that `recall_expand` had quietly reintroduced.
- `scene/isolate`'s GPU-lease budget bumps to 10GB when `recall_expand=true` (was flat 6GB, didn't
  account for the extra SigLIP2 model load the flag adds on top of the checkpoint+render pass).
- **New `backend/mesh/slugify.py`** — the ONE noun→slug function, replacing two
  character-for-character-identical copies in `scene_sam3_masks.py`/`instance_lift.py` (pinned
  behavior-identical via a direct comparison test). **New slug-collision guard in
  `noun_consolidate.py`** (`dedupe_slugs`): two nouns differing only in punctuation (e.g. "Fire
  Hydrant" / "Fire-Hydrant") used to both reduce to slug `fire-hydrant` and silently clobber each
  other's SAM3 masks/instance files across a noun boundary — now the second is vetoed with a clear
  reason instead.
- `proxy_triptych.py`'s `_overlay_tmp.png` scratch file now cleans up on every exit (try/finally)
  instead of only after a fully successful run.
- Test-quality gaps closed: the ground route's `semantic_thresh`/`cell_units` now assert they
  reach `ground_mesh_build.py`'s argv (previously untested — a field-rename bug would've passed
  the whole suite); added coverage for the fail-soft `_langfield_worker_inventory() -> None` path
  STATUS.md already documented happening in production.
- 468 backend tests (up from 458; 10 new, all 5 gates re-verified PASS after a live re-run).

**Left for RToony (design-decision territory, not silently decided):** none of `/scene/isolate`,
`/scene/proxy`, or `/scene/ground` clear their prior output directory before a re-run with
different tuning knobs (e.g. raising `min_members`) — instances that stop qualifying leave stale
`object.ply`/proxy artifacts on disk indefinitely. No DELETE/reset route exists for any `/scene/*`
tree today. Fixing this means deciding what "re-POST with different params" should MEAN (wipe and
rebuild? keep both? explicit reset endpoint?) — a real product-contract call, not a bug fix.

## P6f SHIPPED + gate PASSING + LIVE-VERIFIED (2026-07-23 night) — the fidelity dial
`POST /jobs/{id}/scene/assemble` — the final P6 phase: assembles P6b-e's outputs into one
`scene.blend`/`scene.glb`. Built around RToony's explicit framing that this tool deliberately
straddles "digital twin" (perfect fidelity) and "reimagined 3D scenario" (creative AI
interpretation) — **that choice is now a real, explicit, user-controlled dial, not an implicit
default**: `mode: "faithful"|"styled"` (default) + an optional `overrides: {slug: "captured"|
"proxy"}` map layered on top (e.g. "styled scene but keep the hydrant real"). Every element
records WHY it ended up where it did (`selection: {mode, chosen, available, reason}`) — a
`faithful`-mode deliberate exclusion and an upstream proxy failure are otherwise indistinguishable
months later. An override naming an unbuilt/unknown slug is a hard fail (a deliberate creative
choice, not a best-effort batch op) — live-verified against real garden data before any Blender
work: `faithful`/`styled`/mixed-override modes all resolved correctly.

**OSS research done first** (RToony asked): Gaussian Grouping (ECCV 2024, Apache-2.0) validates
the identity-consistent-grouping direction conceptually but needs training-from-scratch in its
own pipeline, not nerfstudio — not adoptable now, worth remembering if the hydrant-class recall
problem is ever revisited from scratch. OpenUSD 26.03 (shipped THIS MONTH) added native 3DGS
support and its `purpose`/variant-set system is architecturally the industry answer to the same
fidelity duality — but tooling is a converter-script-only, no Blender/glTF-grade ecosystem yet.
Watch, don't build on, either.

**New**: `backend/mesh/scene_assemble.py` (pure-stdlib fidelity-dial resolution, no GPU/Blender),
`backend/mesh/blender_assemble.py` (headless build). **Retrofit** (additive): `scene_manifest.py`
gained an optional `selection` field on `add_element`.
- **Real bug found and fixed by testing against the actual Blender binary, not assumed**: a
  vertex-only (zero-face) point-cloud mesh is SILENTLY DROPPED by Blender's own glTF exporter
  ("Mesh has no primitives and will be omitted") — every splat element would have vanished from
  `scene.glb` while the script still reported success. Fixed by baking real triangulated geometry
  (Geometry Nodes Instance-on-Points → tiny cube → Realize Instances) before export instead of
  relying on a points-only representation.
- Per-element `try`/`except` lives INSIDE the one headless Blender process (the only place the
  "kill one instance, scene still completes, element flagged" requirement can be enforced, since
  the whole manifest builds in one `blender --background` call, unlike P6c/d's one-subprocess-
  per-element pattern) — **chaos-tested live**: a missing file for one element still produced a
  complete scene with the other 3 elements built and the bad one flagged with a clear reason.
- glTF export passes `export_extras=True` explicitly, followed by a **mandatory readback**
  (independently re-parse the written GLB's own JSON chunk) — same discipline as the P0 GLB fix
  (a writer that silently corrupts output while returning success). This readback IS the
  contamination gate: a `captured`/`ground-derived` node must NOT carry the generative tag, every
  `proxy` node MUST — verified by hand against the real exported GLB, not just the script's own
  self-report.
- Resumability = full idempotent rebuild from the manifest, never incremental `.blend` patching
  (matches how P6b-e are already safe to re-POST; sidesteps Blender's node-name collision
  suffixing on partial re-imports).
- The one mandatory HITL stays separate on purpose: `POST /scene/assemble/approve` is the only
  way `state` becomes `"approved"` — never automatic. Live-verified: build → `state: "built"` →
  approve call → `state: "approved"`.
- **Real bug found via the live run, not caught by mocked tests**: the route never actually
  patched the manifest's `state` from `"building"` to `"built"` after a successful assembly — the
  unit test's mock had accidentally hardcoded `"built"` directly, papering over the gap. Fixed
  (route now promotes state explicitly) and the test fixture corrected to write `"building"`
  (matching the real script's `new_manifest()` default) so the suite itself would catch a
  regression here going forward.
- 486 backend tests (18 new across 2 test files). `gate_p6f_assembly.sh` — PASS, including an
  independent glTF-extras re-verification (not trusting the app's own report).
- **A/B receipt** (the fidelity dial, literally visible): `faithful` vs `styled` render of garden
  from the same camera. Honest finding: the quick receipt-render script's first pass used a naive
  min/max bbox for camera framing and got blown out by far-field background outliers (whole scene
  reduced to a tiny fleck) — fixed with the same robust-percentile bbox `blender-receipt-views.py`
  already proved necessary for exactly this. The resulting renders show real, correctly-composed
  geometry with a genuine (if visually subtle at this render quality) difference between modes —
  the point-cloud-as-tiny-cubes visualization technique itself is a rough first pass, not tuned
  for real image quality, and would benefit from more work if a polished render is wanted later.
  The pipeline correctness (manifest, tags, gates) does not depend on this receipt looking good.
  Receipt: `~/Downloads/splatlab-p6f-ab-faithful-vs-styled.png`.

**Live-verified on garden end to end**: `styled` mode assembled all 4 elements (background,
table+proxy, vase+proxy, ground) in ~4s, contamination gate passed, approved. A separate
`faithful`-mode build (scratch location, not clobbering the approved styled build — the open
re-run-semantics question from the review is still genuinely open, deliberately not resolved
here) also assembled cleanly for the A/B comparison.

- **Not done, deliberately**: adding new generative capability (e.g. "place an object that was
  never captured") — P6f only chooses among representations of what P6b-e already produced. Pure
  imagination is a future P6x-style capability, kept separate on purpose.
- This closes out the entire P6 Scene Regeneration Lane (P6a through P6f) as originally scoped.

## P6 CONTROL PANEL (GUI) SHIPPED (2026-07-23, RToony /plan: "word salad" -> GUI-first plan)

RToony: "the central space to control and understand what tools we are making is important for
my ongoing understanding and future reference" — P6a-f had been REST-only all day (six endpoints,
zero frontend). This closes that gap: a full front-end control panel for the whole scene-regen
lane, no backend changes (all six routes/reports were already exactly what the UI needed).

**New**: `frontend/src/components/scene-regen.tsx` (`SceneRegenModal`) — a lazy-loaded full-screen
modal mirroring `geo-locate.tsx`'s exact mount pattern (`job`/`onClose` props, `lazy(() =>
import(...))` + `<Suspense>` in `splat-view.tsx`). Six `useMutation`s (inventory/isolate/proxy/
ground/assemble/approve) live in the modal shell and share one `busy` flag that disables every
trigger across all five stage tabs at once — lifted to the parent (not per-panel) specifically so
`busy` survives switching tabs mid-build. A NEW tab strip, not `StageRail` — deliberate, matches
the plan's reasoning: `StageRail` is contractually tied to one job's auto-advancing single-process
`stage` field; P6 is six independently-triggered, freely re-runnable endpoints with no unifying
stage.
- **Data model**: `SplatSceneSummary` (+5 detail-report interfaces) added to `contracts.ts`,
  5 `fetchScene*()` GETs added to `api.ts` (`.../file?fmt=report` convention, matching every other
  scene-file route already shipped). Each panel prefers its own mutation's just-returned report
  over the polled GET (`mutation.data ?? reportQuery.data`) — shows the real result immediately
  instead of waiting on the `["status"]` poll to update `job.scene` before the on-demand query
  even enables. Caught live: without this, the Assemble panel's Approve button stayed wrongly
  disabled right after a passing build.
- **Live-verified the entire API contract layer against garden** (curl + `PORTAL_TOKEN`, not
  guessed from source reading alone — no interactive browser tool available in this environment,
  so this is the strongest verification achievable here; the app's own login-cookie auth is what
  the real browser uses for both JSON and `<img>` tag requests, same as the existing thumbnail
  pattern already in production). All 5 `fmt=report` GETs, all 6 image/model file endpoints (crop,
  overlay, isolate receipt, proxy triptych, ground top/oblique, assemble glb/blend) returned real
  200s with correct content-types against garden's actual P6a-f output.
- **Two real, if minor, type inaccuracies found by that live check and fixed**: (1)
  `SceneIsolateReport.job_id` was declared required but the GET path (which serves
  `batch_isolate.json` verbatim — `batch_isolate.py` never stamps it with `job_id`, only the POST
  route's in-memory response does) never actually has it; now `job_id?: string`. (2)
  `SceneInventoryInstance.regression` can be wire-`null` (not just absent) for an instance with no
  reference match — the panel's truthiness check (`inst.regression && ...`) already handled `null`
  correctly at runtime, but the type now says so honestly (`| null`).
- **Fidelity-dial UI**: segmented `faithful`/`styled` control + a per-object override table (one
  row per isolate-built instance; proxy option disabled+tooltipped with its skip reason when no
  proxy was built for that slug) — lifted to the modal shell so it survives tab switches, per plan.
  After a build, each row also shows its live `selection.reason` from the manifest. Approve gate
  is its own visually-separated bordered block (amber "not approved yet" vs. emerald "approved"),
  matching the one-mandatory-HITL doctrine P6f was built to enforce.
- **Staleness banner**: no server-side cascade invalidation exists (re-running inventory doesn't
  mark isolate/proxy/ground/assemble stale) — the UI is the only place that catches it, by
  comparing `built_at` across `job.scene.*`. Live-verified this actually fires correctly: garden's
  real data has `inventory.built_at` (14:31) newer than `isolate.built_at` (13:37) from an
  in-session re-run earlier today, and the Isolate panel correctly shows "Built from an earlier
  inventory — re-run isolate…".
- **Gallery integration**: `SceneCard` gets one more badge (amber "scene assembled" for
  `state:"built"`, emerald "scene approved" for `"approved"`); `DownloadMenu` gets `Scene .glb`/
  `Scene .blend` entries gated on `state==="approved"` specifically (not `"built"`) — surfacing an
  unapproved draft as a top-level gallery download would quietly defeat the one mandatory HITL
  gate. A preview download of the built-but-unapproved GLB/blend stays inside the Assemble panel.
- Error surface bypasses `api.ts`'s `apiRequest()` (which throws the raw response body, often
  FastAPI's `{"detail": "..."}` JSON) in favor of a local `postJSON()` that parses `.detail` —
  matches `spark-scene-viewer.tsx`'s existing precedent. The three known 409/503 causes
  (`require_heavy_work_admitted()`'s maintenance/backup interlock, `_mesh_export_lock`'s "already
  running", and each route's own prerequisite check) already write distinct, human-readable
  `detail` strings server-side — no client-side remapping needed, just don't mangle them into a
  JSON blob.
- **Verification**: `npx tsc --noEmit` — exactly 43 errors both before and after, none in the new
  files (matches the established pre-existing baseline). `npm run build` clean (one transient
  esbuild crash mid-session from the same session-scope task-pressure class already diagnosed for
  `gate_p6a_scene_rails.sh` earlier today — not a code issue, retry succeeded with identical output
  hashes). 486 backend tests still pass (confirm-untouched baseline; no backend changes this pass).
  Live API-contract verification against garden as described above. **Not done**: an actual
  interactive-browser click-through — no browser automation tool is available in this environment,
  so DOM rendering/interaction itself is unverified beyond the build succeeding and the contract
  layer matching exactly. RToony should click through it once against garden before trusting it.
- **Accepted for v1, not fixed**: no progress signal for the multi-minute blocking POSTs (proxy
  especially) — an indeterminate spinner + "don't close this tab" copy, honestly, per plan. A
  refreshed/closed tab mid-request can't distinguish "still running" from "crashed" until the next
  poll.
- **Deferred, not started**: Phase 2 (Reference CAD/DXF alignment) — scoped in the plan, explicitly
  not picked up until Phase 1 ships and gets real use.

## VIEWER RENDER-LOOP PAUSE-ON-HIDDEN (2026-07-23, RToony live-test feedback)

RToony, testing P6 live at splatlab.roonytoony.dev: reported a suspected "resource hog animation"
on visiting a scene. Real, confirmed: `viewer.start()` (`@mkkellogg/gaussian-splats-3d`, the
classic viewer) and the hand-rolled `animate()` loop (`spark-scene-viewer.tsx`, Spark beta) both
run an unconditional `requestAnimationFrame` loop for as long as `/view/:jobId` stays open — even
for a 100% static scene nobody's touching. Not a bug (real-time WebGL splat rendering needs a
continuous loop by nature), but a legitimate, low-risk improvement: browsers already throttle a
*backgrounded* tab's RAF to ~1Hz, but that's still real render work every second, forever.
- **Fix**: both viewers now pause on `document.visibilitychange` (`document.hidden`). Classic
  viewer uses the library's own `start()`/`stop()` pair (confirmed via source read: `stop()` just
  `cancelAnimationFrame`s the loop, no GPU buffer teardown — cheap, instant resume). Spark beta's
  own loop checks `document.hidden` at the top of `animate()` and skips the real work (still
  reschedules RAF so it resumes instantly on refocus).
- **Not done, deliberately**: true render-on-demand (only re-render when the camera actually
  moves, foreground-and-idle) — a materially bigger, riskier change (would need to audit every
  per-frame effect — highlight markers, camera overlay, dimension labels — for hidden
  continuous-update assumptions). RToony's own framing ("if not, nevermind") didn't call for it;
  revisit only if the pause-on-hidden fix proves insufficient.
- Verification: `tsc --noEmit` steady at the 43-error baseline, `npm run build` clean. No backend
  changes. Not yet live-clicked by RToony (report-and-fix cycle, not yet re-confirmed by him).

## OBJECT-LANE TWIN FINISH SHIPPED + LIVE-VERIFIED (2026-07-23 night, RToony /plan)

RToony wants to shoot small bounded objects (hydrant-class, ~5'×5') with 5-10x denser coverage
(250-500 photos vs the field-proof's 49) and get back "decent yet reduced voxel/3d face... more
recent and better detailed" — a clean, colored, art-directable Blender mesh, not a raw scan dump
and explicitly not the "PS2 graphics" blocky low-poly he first floated. Direct code + live-artifact
inspection found the geometry side (P5b `/objects` isolation → TSDF mesh) was already excellent
(hydrant: 100% LCC, fully connected) — the actual gap was entirely on the finishing side: the raw
object mesh shipped with **zero decimation and only pale TSDF-baked color** (confirmed: `mesh_
report.py`'s own `defaultLit`+sun-light receipt already shows faint color from TSDF integration
itself — "zero color" in the plan was slightly imprecise; "pale, unenhanced color" is accurate,
matching `twin_finish.py`'s own docstring). `twin_finish.py` (WS3 — 6-NN gaussian color transfer +
pymeshlab quadric decimation + Y-up vertex-colored GLB + mandatory readback) already solves exactly
this and was already shipped/proven on two other routes — it had just never been wired into
`/objects`.

- **New `ObjectIsolateBody` fields**: `finish: bool = False`, `finish_target_faces: int = 10_000`
  (1k-100k range, request-overridable). Color source is the object's OWN `object.ply` (not the
  whole-scene splat) — verified field-for-field compatible with `twin_finish.py`'s loader
  (`object_isolate.py` writes raw/pre-activation `f_dc`/`opacity`, matching what the loader
  decodes itself) — scoped, ~140x less data to KDTree-index than the whole scene, and structurally
  guarantees no background-gaussian color leak at boundary vertices.
- New block in `POST /jobs/{id}/objects`, after the existing mesh block: runs `twin_finish.py`
  (`mesh.ply` + `object.ply` → `mesh/twin.glb`), then best-effort colored receipts via
  `ground_mesh_receipt.py` reused **unmodified** (already fully generic on any Y-up vertex-colored
  GLB). New `_OBJECT_FILES` entries (`twin`/`twin-top`/`twin-oblique`) + `twin_glb_url` in the
  response. No changes needed to `twin_finish.py`, `object_isolate.py`, `checkpoint_subset.py`, or
  `mesh_report.py` — confirmed the subset-checkpoint's `finally: shutil.rmtree` cleanup (which runs
  before the new block) is a non-issue since `twin_finish.py` never touches the checkpoint, only
  plain mesh/splat file paths that already exist and are never deleted.
- **`finish_target_faces` default 10,000**: the scene-level 400k default would never even trigger
  decimation on an ~90k-tri object (silently a no-op) — 10k is a ~9x reduction against the
  hydrant's real 89,990 raw tris, in the conventional low/mid-poly hero-prop range, not a "PS2"
  cut (~500-2k) or a near-lossless pass-through.
- **Capture-density guidance**: extended the one existing, already-firing `_recapture_guidance()`
  photo-orbit string with a dense-single-object recommendation (250-500 photos, stage via
  `splatcli/inputs/<name>/` + `input_path` to skip the 2GB web-upload cap) — deliberately NOT a new
  precheck heuristic (Tier-0 has zero access to real-world scale pre-SfM; a wrong auto-fired
  advisory costs more trust than it's worth, per the metric-trust doctrine).
- **Verified**: smoke-tested `twin_finish.py` directly against the hydrant's real on-disk artifacts
  BEFORE touching any route code (cheapest possible proof of the one real risk) — exit 0, 89,990
  raw tris → exactly 10,000, genuine color (per-channel std ~46, confirmed programmatically, not
  eyeballed). 490 backend tests pass (486 baseline + 4 new: builds twin.glb with the right args/
  object.ply scoping, `finish=True` without `mesh` is a loud 400, `finish_target_faces` override
  lands in the command, a finish failure is a loud 500 that leaves the already-succeeded raw mesh
  artifacts intact). **Live end-to-end regression** against the real hydrant job through the actual
  GPU pipeline (`splatlab-safe-restart` deployed, 0 active jobs first): re-isolated + re-meshed +
  finished, numbers matched the smoke test exactly (6,919 verts / 10,000 faces / 19,659 solid
  gaussians / 231,708 GLB bytes), all three new file endpoints 200, `meta.json["objects"]
  ["fire-hydrant"]["twin"]` populated correctly. Receipts + both GLBs copied to `~/Downloads/
  splatlab-hydrant-twin*`/`splatlab-hydrant-raw-mesh.glb` for RToony's own Blender eyeball —
  the actual "does this look good" call is his, not scriptable.
- **Not done, deliberately** (per plan): no `mesh_gate.py` PSNR/SSIM wiring for objects (would be
  structurally misleading — scores against full uncropped photos, dominated by background pixels
  an object mesh never claims); no photo-count heuristic in precheck; no size-adaptive
  `finish_target_faces` formula; no 2DGS/SuGaR/MILo reconstruction swap (stays a documented future
  "watch" item). The actual dense-capture field test (250-500 photos, real subject) is RToony's own
  hands-on work — not scripted here.

## PRESENTATION TOOLKIT PHASE 1 SHIPPED (2026-07-24, RToony /plan) — crop-sphere + dimensions

RToony wants to turn SplatLab into something that produces real, shareable files: named the Spark
tools (Locate/Add dimension/Paint), an official site-section generator, a radius-sphere crop to
discard poorly-captured surrounding context, and a direct "am I asking too much?" about
semantic-driven mesh smoothing. Researched, planned as 3 phases; this ships Phase 1.

**Correction found by reading the code**: "Locate" isn't a Spark tool (it's the pre-existing
`GeoLocateModal`, unrelated). Of the two real Spark tools, Paint was already fully complete
(server-persisted). Add Dimension was fully functional but session-only (`sessionStorage`) — the
one real gap.

- **Crop-sphere tool** — the backend was already 100% done: `edit_ops.py`'s `CropSphereOp` maps
  directly onto `splat-transform -S`, whose own docs say exactly what was asked: "Remove Gaussians
  outside sphere." Zero backend changes. New UI in `spark-scene-viewer.tsx`: click-to-place center
  (reuses `raycastAt()`), a red wireframe sphere gizmo, a live "N of M splats would be removed"
  count computed client-side via `PackedSplats.forEachSplat` (debounced 120ms) and fed into the
  same overlay-tint machinery Paint already uses, a two-step confirm (reusing the in-file
  `recalibrateArmed` idiom), POST to the already-tested `/edit/apply`, and a minimal "Undo crop"
  button against `/edit/revert` using the apply response's `version_before`.
- **Real bug found and fixed as part of this work**: the mesh-load effect was keyed only on the
  preview URL string, which never changes after an edit even though the file on disk does — the
  viewer would keep showing stale geometry after a crop. Fixed with a `reloadNonce` query param
  bumped on every apply/revert.
- **Server-persisted dimensions**: new `backend/dimensions_route.py` (pure stdlib, no numpy —
  confirmed the main backend's own venv has neither, matching `edit_ops.py`'s documented
  constraint), mirroring `geo_route.py`'s "small CPU-only feature + own router" shape rather than
  `langfield_overrides.py` (wrong process, numpy/index-file id scheme this data doesn't need).
  `GET/POST/DELETE /jobs/{id}/dimensions` + `.../export?fmt=csv|json`. POST is a client-id-keyed
  upsert (not server-assigned) so drag-to-move reuses the same route. `dimensions.json` at the job
  root. Frontend: sessionStorage kept only as an instant-restore/offline fallback, server is
  authoritative; inline CSV download link next to "delete all," shipped now (not deferred) since
  it's what actually answers "generate a file I can share" for this tool.
- **Verified**: 499 backend tests pass (490 baseline + 9 new for `dimensions_route.py`). `tsc
  --noEmit` steady at the 43-error baseline, `npm run build` clean. Live: full dimensions
  CRUD+export round-trip via curl against the real hydrant job (add → list → CSV export with
  correct computed length → delete → confirm empty); confirmed the deployed bundle contains the
  new crop/dimensions strings at the exact served asset hash.
- **Not yet done**: an actual interactive click-through in a real browser (no browser automation
  tool available in this environment — same limitation as the P6 GUI phase). RToony should try
  placing/dragging dimensions across a refresh (proves server persistence) and running a real crop
  (proves the reload-nonce fix + undo) before trusting this fully.
- **Phase 2 (site sections) and Phase 3 (curvature-adaptive smoothing) not started** — fully
  scoped in the approved plan, queued next.

## PRESENTATION TOOLKIT PHASE 2 SHIPPED (2026-07-24) — official on-demand site sections

Closes the "make site sections an official tool" ask. Previously `surface_receipts.py` (two
principal-axis cross-sections + an isometric ground TIN) only ran as an always-on, best-effort
tail step buried inside the much heavier `/geo/contours` CAD/DXF pipeline, with zero frontend
trigger — the only UI was a passive download link that appeared IF someone had already called the
API directly.

- **`GroundSampleParams` refactor** in `geo_route.py`: factored `ContoursBody`'s shared
  ground-sampling fields (epsg/cell_m/max_slope_deg/spike_tol_m/semantic/semantic_thresh) into a
  base class, backward-compatible (all 35 pre-existing `test_contours_route.py` +
  `test_geo_route.py` tests pass unchanged, proving it). Extracted the semantic-AUTO-fallback
  decision tree + ground_extract.py subprocess call (previously inline in `build_ground_contours`)
  into a shared `_extract_ground_points()` helper, so the two routes that both need it
  (`/geo/contours` and the new `/geo/sections`) can't drift out of sync.
- **New `POST /jobs/{id}/geo/sections`**: a genuinely lighter route — only needs ground-extract +
  `surface_receipts.py` (no CDT venv, no DXF authoring), sharing `_mesh/geo/` and its lock with
  `/geo/export`/`/geo/contours` but promoting only the files it just produced (verified: running
  either route after the other leaves the other's exclusive files — `contours.dxf` etc. —
  untouched). Unlike `/geo/contours` (where these images are a best-effort bonus on top of the
  DXF), a `surface_receipts.py` failure here is a loud 500 — it's this route's entire deliverable.
  v1 keeps the existing fixed "two auto-picked principal-axis sections + two iso angles"
  algorithm — no new tunable knobs, matching this project's "ship the baseline" pattern.
- **Frontend**: extracted `ReceiptLightbox` out of `scene-regen.tsx` into its own shared
  `receipt-lightbox.tsx` (mechanical, its props were already fully generic) so both it and the
  new modal use the same component. New small single-purpose `site-sections.tsx` modal — one
  trigger button, two result thumbnails, the shared lightbox — deliberately NOT folded into
  `GeoLocateModal` (scoped to map/anchor concerns) or `SceneRegenModal` (a different lane, P6
  object decomposition). New "Sections" header button in `splat-view.tsx` between Locate and
  Scene, same visibility/styling convention as its siblings.
- **Verified**: 508 backend tests pass (499 baseline + 9 new for `test_sections_route.py`). `tsc
  --noEmit` steady at the 43-error baseline, `npm run build` clean (new `site-sections`/
  `receipt-lightbox` lazy chunks). **Live end-to-end, not just mocked**: set a temporary synthetic
  geo anchor on garden (already had scale+mesh+langfield, just missing an anchor), ran the real
  route — 260,036 ground gaussians identified out of 1,326,611 total, 1,065 ground points, ~29s —
  pulled the actual rendered PNG and visually confirmed it: the garden table's silhouette is
  clearly recognizable and the ground TIN correctly traces the real terrain including the
  pedestal notch. Cleared the synthetic anchor afterward (the real generated images were left in
  place — legitimate artifacts, not test scaffolding). Confirmed the deployed bundle serves the
  new code at the live asset hash. Receipts copied to `~/Downloads/splatlab-garden-sections.png` /
  `splatlab-garden-surface-iso.png`.
- **Not yet done**: an actual interactive click-through in a real browser (no browser automation
  tool available in this environment). RToony should try the "Sections" button on a scene that
  already has scale + a real Locate anchor set.

## PRESENTATION TOOLKIT PHASE 3 SHIPPED (2026-07-24) — curvature-adaptive mesh smoothing

The concrete, ship-now answer to "am I asking too much?" about semantic-driven surface smoothing.
Dedicated research this session found the semantic version (SAM-mask-driven region-specific mesh
edits) is real published work (Häne et al. 2017, PlanarGS NeurIPS 2025, MagicRoad July 2025) but
never shipped as a general, robust feature — the hard part is multi-view-consistent 2D→3D label
fusion, not labeling. The mature, no-semantics-needed answer: curvature-adaptive mesh denoising,
already one filter call away in `pymeshlab`, already a `twin_finish.py` dependency.

- **Filter**: `apply_coord_two_steps_smoothing`, confirmed directly against the installed
  pymeshlab (2025.7.post1) parameter surface. Feature-adaptive (`normalthr` degree threshold
  excludes sharp face-pairs from normal-averaging, so rims/bolt heads stay crisp) and
  shrink-resistant by construction (fits vertex positions to the smoothed normal field rather than
  averaging neighbor positions — the actual mechanism that makes plain Laplacian shrivel a mesh,
  which is why plain Laplacian was rejected). Defaults: `smooth_iterations=2`,
  `smooth_feature_deg=40°` (tighter than pymeshlab's own 60° default).
- **Pipeline-order bug caught by the mandated pre-wiring smoke test, not by RToony**: the approved
  plan called for smoothing AFTER decimation (cheaper, fewer verts). Built exactly as planned, then
  smoke-tested `twin_finish.py --smooth` against the real hydrant `mesh.ply`/`object.ply` per this
  session's own "smoke-test the real subprocess call before wiring the route" discipline — the
  rendered receipt showed clearly WORSE results than unsmoothed: blocky, faceted, crumpled patches.
  Root cause: `apply_coord_two_steps_smoothing` fits positions to a locally-averaged normal field,
  which needs enough face resolution to express a smoothly-varying field — the already-decimated
  10k-face mesh didn't have it. Fixed by reversing the order: smooth the full-resolution raw mesh
  (~47k verts/90k tris) first, then decimate the now-clean surface. Concrete before/after on the
  same hydrant mesh: surface-area change vs. the unsmoothed baseline went from **-6.47% (faceted,
  bad)** in the wrong order to **-3.56% (clean noise removal)** in the corrected order; verts/faces
  and extent stayed effectively unchanged in both. `transfer_attributes_per_vertex` (color
  re-projection from the pristine mesh-0 source) now fires whenever EITHER smoothing or decimation
  moved a vertex, not only on decimation.
- **Route wiring scoped to `/objects {"finish": true}` only, this pass** — not `/mesh` or
  `/scene/ground`. This is the one lane with a real, proven smoke-test artifact (the hydrant), and
  a single global feature-angle threshold is a safer bet on one isolated object than across a
  heterogeneous scene. `ObjectIsolateBody` gained `smooth`/`smooth_iterations`/
  `smooth_feature_deg`; `finish_cmd` in the `/objects` route passes them through as `--smooth
  --smooth-iterations N --smooth-feature-deg D` when requested. The script-level change is shared
  and cheap to extend to `/mesh`/`/scene/ground` later — a mechanical follow-up, not a redesign.
- **Verified**: 510 backend tests pass (508 baseline + 2 new wiring-correctness tests in
  `test_objects_route.py` — assert `--smooth`/`--smooth-iterations`/`--smooth-feature-deg` appear
  correctly when requested and are absent when not; the smoothing algorithm itself is proven by the
  hand-run smoke test above, not re-proven in the mocked-subprocess unit test). **Live end-to-end**:
  `POST /objects {"query": "fire hydrant", "mesh": true, "finish": true, "smooth": true,
  "smooth_iterations": 2, "smooth_feature_deg": 40.0}` against the real hydrant job through the
  full GPU pipeline (re-isolation + re-mesh + finish+smooth, 45.9s) returned
  `twin.smoothing: {"applied": true, "iterations": 2, "feature_deg": 40.0}`, verts=6921,
  faces=10000, extent=[0.47,0.47,0.47] — matching the smoke test exactly (reproducible). Pulled the
  live-route-produced GLB and both twin receipts (`fmt=twin`, `twin-top`, `twin-oblique` all 200
  OK) and visually confirmed: smooth cylindrical hydrant body, nozzle caps and rim edges still
  crisp, no faceting. Copied to `~/Downloads/splatlab-hydrant-twin-smoothed.glb` +
  `-oblique.png`/`-top.png`.
- **Not yet done**: RToony's own Blender/eyeball call on whether the smoothing defaults
  (2 iterations, 40°) look right on other object shapes beyond the hydrant — this was the one
  planned verification step explicitly deferred to his own hands-on judgment, same as the finish-
  stage verification earlier this session.
- **Explicitly deferred, not attempted**: semantic-mask-driven region-specific smoothing (genuine
  open research problem per this session's research, not a near-term deliverable) and RANSAC
  plane/cylinder primitive-fitting (mature for planes, confirmed unreliable for cylinders even in
  established libraries — directly relevant to the hydrant, so deliberately not attempted). Both
  remain documented "watch" items.

**All 3 phases of the Presentation & Editing Toolkit plan are now shipped.**

## AUTORESEARCH MARATHON — 4 workstreams built, real trials run (2026-07-24, RToony /plan)

RToony asked for a 6-8h automated test series using "the karpathy/autoresearch system" to compare
splat generation, twin-finish simplification, and DXF exports against source photos as ground
truth. Found RToony already runs a real local implementation of this pattern
(`~/projects/autoresearch-lab`, `splatlab-mesh-quality` project, 28 prior trials). Built 3 new
sibling autoresearch-lab projects + extended the existing one. **Actual wall-clock: 1h24m, not
6-8h** — flagged honestly below, not padded to look like more happened.

- **New `eval_splat_holdout.py`** (`~/tools/dn-splatter-probe/mesh-trial/`): fills a confirmed gap
  — no held-out-photo eval existed anywhere for the raw Gaussian splat. Uses nerfstudio's own
  first-party `eval_setup()` + `get_average_eval_image_metrics()` against the 10%-holdout test
  split nerfstudio already computes and silently discards every training run. Carries the same
  `torch.load(weights_only=False)` monkeypatch already proven in `mesh_gate.py`.
- **New `splatlab-splat-holdout` project**: `baseline-30k-reuse`/`fresh-7k`/`fresh-15k`/`fresh-50k`
  on hydrant + garden. Fresh trials write to an ISOLATED `trials/` dir, never the real job's
  `processed/splatfacto/` tree (a newer config.yml landing there could silently become "latest" for
  production lookups — checked and confirmed clean after the run).
- **New `splatlab-twin-finish-quality` project**: sweeps Phase 3's smoothing params against real
  photos via `mesh_gate.py` (unchanged) pointed at each trial's `twin.glb`. **Caught and fixed a
  real bug before any real trial ran**: `twin_finish.py` rotates its GLB output to glTF's Y-up
  convention (correct for Blender), but `mesh_gate.py`'s cameras expect the scene's native Z-up
  frame — first smoke test rendered the hydrant sideways. Fixed via `twin_gate_prep.py` (inverts
  the rotation before scoring only; production `/objects` output is untouched).
- **New `splatlab-dxf-quality` project**: sweeps `cell_m`/`max_slope_deg`/`spike_tol_m` via the
  real `ground_extract.py`/`contours_build.py` scripts directly. Set a clearly-labeled TEST geo
  anchor on garden (`source: manual`, placeholder lat/lon) since it had none — **flagged here for
  RToony to replace with a real anchor**.
- **Extended `splatlab-mesh-quality`** with 4 strategies from its own documented "Frontier"
  backlog (lambda sweep, dn-Poisson+outlier-removal, cropbox-to-reference) — all using stock
  `gs-mesh` CLI flags, zero `mesh_trial.sh` changes needed.
- **Incident, self-caught and fixed mid-run**: `lambda-lo-s012` thrashed `scene_5177f8d99a`'s
  memory envelope under the `splatlab.slice` cgroup (8G swap cap maxed, 83-90% iowait, ~26min
  negligible progress) — killed by hand, root-caused (forces a fresh full finetune, a combination
  never exercised on this scene before), scoped the 2 lambda strategies to the other scene, and
  rebuilt the marathon script to loop per-strategy (was per-project, which hid the stall entirely)
  with a 1h per-strategy runaway cap. Relaunched clean; finished in 1h24m with zero further issues.

### Real findings (measured, not narrated)
- **Splat holdout (garden/hydrant, PSNR/SSIM/LPIPS vs real held-out photos)**: existing 30k-iter
  checkpoints are already near-optimal — garden 30k ssim=0.8426 vs 50k ssim=0.8447 (~flat); hydrant
  30k ssim=0.6411 vs 50k ssim=0.6381 (no gain). 7k is measurably worse on both (garden 0.7741,
  hydrant 0.6513 — note hydrant's 7k number is *slightly above* its own 30k baseline, within noise
  at n=4 eval images). **Validates the current 30k default; no iteration-count win found.**
- **Twin-finish smoothing (SSIM vs real photos, 3 objects)**: no-smooth baseline scored *highest*
  (0.069, only `accepted=True`); all smoothed variants scored 0.060-0.069, with more aggressive
  decimation (5000 faces) hurting most. **This does not contradict Phase 3's shipped defaults** —
  it measures a different axis. Phase 3's own eyeball check judged geometric cleanliness (less
  faceting); this metric measures per-pixel photo similarity, which decimation/smoothing can
  slightly hurt even when the surface looks visually cleaner. Both are real, legitimate, and
  different questions.
- **Mesh-quality (TSDF trunc sweep)**: `tsdf-s020-reuse` (looser truncation, 0.20 vs the prior
  0.12 champion) is the new numeric champion on `scene_5177f8d99a` — 90.92% LCC @ 75% coverage vs
  the previous 55.3%/74% at trunc=0.12, in this run. **Important honesty check**: pulled the actual
  receipt renders for both — visually, BOTH are heavily spiky/jagged with TSDF fusion noise from
  every camera angle tried (top, interior). The LCC×coverage metric confirms "one big connected,
  room-sized blob," not "a clean, presentable mesh." Neither the old nor new best config produces
  something visually clean — a real, humbling limitation of the current v4 scoring metric, not
  a new regression. `scene_98095cb055`'s fragmentation problem remains open (27% LCC even at the
  new champion config; `cropbox-ref-s012-reuse` didn't fix it either, 0.2732).
- **DXF ground-sampling**: all 7 param configs produced non-degenerate, audit-clean output (no new
  warnings beyond the constant "watermarked provisional" flag every trial gets). `cell_m` has a
  large real effect on point density the score doesn't capture (0.1→10,783 points/615 contours,
  0.5→631 points/92 contours) — the per-trial score formula only varies with audit-warning count,
  which didn't change across configs; a real, honestly-noted scoring-design gap for this project.

### Honest gaps / what's NOT done
- Ran **1h24m of the requested 6-8h** — the infrastructure is real and reusable, but this was a
  single pass through modest grids, not a deep multi-hour search. Scaling up (larger grids, more
  seeds per fable5 anti-pattern #6 on single-seed noise, extending `--smooth` to `/mesh`/
  `/scene/ground`, a semantic-ground axis for W4) is queued, not done.
- `splatlab-dxf-quality`'s score doesn't discriminate parameter sensitivity (see above) — would
  need a real stability metric (e.g. contour-length variance across configs), not just audit
  warnings, to rank ground-sampling settings meaningfully.
- Pre-existing, unrelated bug found in passing: `nexus-notify --title/--message` (used by both the
  existing `overnight_runner.sh` and this session's new marathon script, copied verbatim) doesn't
  match the current `nexus-notify` CLI (Slack-only, positional message, no `--title` flag) — the
  completion digest silently failed to send. Not fixed (out of scope tonight), flagged here.
- All 4 autoresearch-lab projects + `generator.py`/`eval_splat_holdout.py`/marathon script changes
  are uncommitted in `~/projects/autoresearch-lab` (separate repo from splatlab) — not committed,
  per standing "only commit when asked" — RToony's call on whether/how to commit that repo.

## AUTORESEARCH RUN 2 (2026-07-24, same day, RToony back at the PC) — 1h48m, follow-up strategies

RToony explicitly did NOT want another 6-8h run while he was actively using the PC — asked to scale
up depth instead of duration, iterate in short bursts, and use the data to plan a longer sustained
12-24h run for tonight's ~12-24h absence. Added 10 new strategies across all 4 projects, each a
direct follow-up to Run 1's own findings (not a blind bigger grid). 49 total strategies (up from
39), finished in 1h48m (vs Run 1's 1h24m) with zero incidents — the per-strategy-loop fix and
memory-envelope scoping from Run 1 held.

- **W1 truncation curve extended past s020 — numbers climb, but it's still metric-gaming, not a
  real win.** `tsdf-s024-reuse` (0.6484) and `tsdf-s028-reuse` (0.7432) both now PASS the gate on
  BOTH scenes (up from s020's 0.4727, pass_rate 0.5 — only one scene passing). Pulled the actual
  renders for s028 on BOTH scenes before trusting this: **identical spiky, incoherent TSDF fusion
  noise as every prior "champion," on both `scene_5177f8d99a` and `scene_98095cb055`.** Looser
  truncation is fusing more noise into one bigger connected blob, satisfying LCC×coverage without
  producing anything visually usable. The v4 scoring metric (survivor of v1-v3's own documented
  gaming problems) is itself now gameable by truncation alone — **do not treat any of s020/s024/
  s028 as real recipe improvements.** Cleanup combined with s020 (`clean10/20-s020-reuse`,
  0.4727/0.4715) didn't change anything — cleanup doesn't touch this failure mode.
  scene_98095cb055's fragmentation problem remains genuinely unsolved.
- **W2 iteration curve now well-bracketed (7k/15k/20k/30k/40k/50k), and the two scenes tell
  different stories.** Garden (18 held-out images): clean, real diminishing-returns curve —
  7k=0.7738 → 15k=0.8311 → 20k=0.839 → 30k=0.8426 → 40k=0.8429 → 50k=0.8451. Flattens hard by
  15-20k; everything past that is a rounding error. Hydrant (4 held-out images): **not
  monotonic at all** — 0.6529/0.6547/0.6483/0.6411/0.6447/0.6365 across the same iteration range,
  a ~0.018 spread with no discernible trend. With only 4 eval images this is almost certainly
  measurement noise, not a real iteration-count effect — can't draw a hydrant-specific conclusion
  beyond "nothing catastrophic happens across this range." **Actionable**: 15-20k iterations
  captures nearly all the achievable held-out fidelity on richer captures; the current 30k default
  isn't wrong, just not obviously necessary either.
- **W3 smoothing gap is real but tiny, and lighter settings win.** `smooth1-f25-t20000` (1
  iteration, 25° feature threshold, 20k target faces) scored 0.0693 — the only smoothed variant to
  beat the no-smooth baseline's 0.0690, driven by a small flower-vase gain (0.049 vs 0.046) with
  fire-hydrant unchanged and round-wooden-table slightly lower. `smooth2-f15-t10000` tied baseline
  exactly (0.0690). **This is a noise-level difference at 3 objects, not a strong result** — but
  directionally, the lightest/most conservative smoothing settings are consistently the ones that
  don't hurt photo-fidelity, while heavier settings (more iterations, looser feature threshold,
  smaller target_faces) consistently cost a little. If Phase 3's defaults get revisited, lighter is
  the supported direction, not heavier.
- **W4 boundary strategies found real structural effects the score still doesn't capture.**
  `cellm-xloose` (cell_m=1.0): 184 points / 352 tris / 56 contours — a real, large drop from
  baseline's 2118/4214/169, genuinely approaching (not yet crossing) a too-sparse-to-trust regime.
  `spike-xtight` (spike_tol_m=0.1): 1718 points / 3418 tris / 105 contours — also real
  thinning (~19% fewer points, ~38% fewer contours than baseline), consistent with aggressive spike
  rejection discarding legitimate terrain variation as false positives. Neither triggered actual
  degeneracy (0 points, audit failure) — the boundary is further out than tonight's grid reached —
  but both confirm the score's blindness (still 0.85, identical to baseline) is a real scoring gap,
  not proof the params don't matter.

**Bottom line for planning the 12-24h overnight run**: the two workstreams worth real overnight
depth are W2 (splat-holdout — clean, trustworthy, real signal, room for a genuinely thorough sweep
including the previously-deferred SfM-backend comparison) and a redesigned W1 (the LCC×coverage
metric itself needs work before more truncation sweeping is worth running — a v5 metric or a
paired visual-receipt gate should probably be built before spending overnight compute chasing more
of the same gameable signal). W3's signal is real but small; W4 needs either a real stability
metric or to just keep pushing boundaries further out.

## PORTABLE PIPELINE + RESTRICTED BLENDER MCP SHIPPED (2026-07-24) — SPZ/SOG/GLB export, UE 5.6 handoff, research ledger

Delivered: checksummed export manifest + portable-format API (`backend/artifact_manifest.py`,
`backend/export_route.py` 1164 lines — SPZ v4, CPU SOG, streamed SOG, KHR_gaussian_splatting GLB,
collision, caching, stale-source detection, lock-guarded rebuild), wired into the live app at
`main.py` (`app.include_router(export_route.router, prefix="/api/splat", ...)`). Frontend DownloadMenu
(`frontend/src/pages/splat.tsx`) gained a "Portable pipeline" section (Build formats / Package UE 5.6),
new `SplatExportManifest`/`SplatUnrealBundle` types in `contracts.ts`, new
`fetchPortableExports`/`buildPortableExports`/`buildUnrealBundle` calls in `api.ts`.

**NEW restricted Blender MCP** (`backend/dcc/blender_mcp_server.py`, port **9877**, `127.0.0.1`
streamable-http): exactly **8** `@mcp.tool` functions (inspect_job, inspect_blend,
list_blender_versions, snapshot_blend, toggle_collection, transform_object, restore_blender_version,
open_blender) — no exec/Python/URL-fetch tool, no arbitrary code execution. Backed by
`blender_workflow.py`: atomic `.building-` → final staged writes, symlink rejection, env-allowlisted
subprocesses. **This is a separate, distinct server from the EXISTING general-purpose Blender MCP on
port 9876** (official `ahujasid/blender-mcp` addon, GUI-attended via `blender-cockpit`, documented
2026-07-21 P3 COCKPIT). Do not conflate the two: 9876 = general, GUI-attended, full addon surface;
9877 = restricted, 8-tool, headless, no code exec. Currently a script
(`integrations/blender/run-mcp.sh`) with its own `.venv` — no systemd unit today; would need a
`nexus-manifest.json` entry only if later turned into a persistent service (not done now).

Also shipped: `integrations/unreal/` — `bundle_tool.py` (symlink/zip-bomb guarded), verify/stage
PowerShell scripts, renderer probe table (NanoGS/MLSLabsRenderer/UnrealSplat),
`docs/portable-interchange.md` (format table + typed `POST /exports` API docs). `research/` ledger
(`README.md` "What Was Incorporated", `sources.json` with 17 license-reviewed candidates,
`benchmark.py`/`capability_probe.py` harness).

**Verified independently, not trusted from narrative:** `~/.local/bin/pytest` on the 8 new test files
→ **64 passed, 2 skipped, 0 failed** (skips are opt-in real-binary smoke tests, correctly not
exercised in a normal pass). `ruff check` on all new/modified `.py` files: clean. `tsc --noEmit`: 0 new
errors in the 3 touched frontend files (23 pre-existing errors elsewhere, unrelated). No GPU workload,
no backend restart, no UE launch/compile occurred during delivery or during this verification pass.
Windows UE 5.6 target workstation ("Triforce": i7-7700/64GB/RTX 2080 SUPER 8GB, Win10 Pro 22H2)
confirmed online and hardware-sufficient the same day, but Unreal Engine itself is not yet installed
there and remote access (SSH/WinRM/RDP) is not yet open — SMB only — so the UE-side half of this kit
remains unexercised end-to-end.

Two unrelated AUTORESEARCH MARATHON/RUN 2 sections landed in this same file from a separate,
still-in-progress session — left uncommitted on purpose, not swept into this delivery; RToony's call
on if/how to commit that content separately.

## PROFESSIONALIZATION WAVES 1-3 SHIPPED (2026-07-25) — hygiene, tokens/brand, IA shell + first real export proof

RToony approved the full professionalization plan (~/.claude/plans/dazzling-tumbling-horizon.md —
7 waves: GUI polish, native Edit mode wiring the existing 9 edit ops, Export Center, Spark
consolidation, self-hosted SuperSplat roundtrip). Decisions locked: converge on Spark then delete
classic; native-first editing with SuperSplat as escape hatch; tokens + à-la-carte Radix (no
shadcn); interleaved quick-wins sequencing. Waves 1-3 shipped today, all frontend deploys via dist
swap (no restarts except the one noted below):

- **Wave 1.1** (`b0f021b`): all 14 non-classic-viewer tsc errors fixed; flat ESLint
  (typescript-eslint + react-hooks, compiler-era rules demoted to warn) + Prettier + `npm run
  check`/`lint`. Remaining 9 tsc errors are all in splat-viewer.tsx (classic — deleted in wave 7);
  `check` stays red by design until then.
- **Wave 1.2** (`99d7b6d`): CSS-var design tokens (--surface/--ink/--accent*/--radius-*) mapped
  into Tailwind; radius drift collapsed to a 3-step scale; @fontsource self-hosted fonts (Google
  Fonts @import gone); favicon.svg + OG/theme-color; ALL user-visible naming standardized to
  "SplatLab" (incl. backend login page).
- **Wave 1.3** (`8b66845`): DownloadMenu extracted to components/gallery/download-menu.tsx and
  rendered on /view (export parity with home cards) + **first-ever real-data portable export run**
  (below).
- **Wave 2.1** (`e64df4c`..`11d8430`): components/ui/ primitive library — Radix Dialog/Tooltip/
  Tabs/DropdownMenu (radix-ui 1.6.7, React 19 OK) + hand-rolled ToastProvider + Skeleton +
  EmptyState; all three hand-built modals (site-sections, geo-locate, scene-regen) migrated onto
  Dialog (focus trap/Escape/aria); app-wide toasts replace the setTimeout div; spark panel inputs
  unified onto Input size="xs".
- **Wave 2.2** (`964cd77`, `3826975`): splat.tsx decomposed 1,458→689 lines by pure moves —
  lib/stage-meta.ts, lib/format.ts, components/create/*, components/jobs/*, components/gallery/*.
  Extraction verified byte-identical.
- **Wave 3** (`757f2a2`): AppShell nav (Scenes / New capture / ⋯→Feedback) wraps /, /new,
  /feedback; pages/splat.tsx DELETED, split into pages/scenes.tsx + pages/new-capture.tsx (query
  keys unchanged — in-flight jobs survive navigation); route ErrorBoundary, styled 404, first-run
  EmptyState, skeletons, status-feed Retry card. ⚠️ Behavior note: Promote-to-full-build now always
  uses fresh-load defaults (standard/no-langfield/no-mesh).

### ⚠️ FOUND LIVE: the 07-24 portable-export delivery was never running
The service last restarted 07-24 06:04; the export/UE/collision commits landed 18:00 — production
"Build formats" pointed at routes that didn't exist (SPA catch-all answered 405). Fixed with an
idle-window `splatlab-safe-restart` (no jobs in flight). **Deploy rule reaffirmed: any
backend/*.py change needs a safe-restart to go live — verify with /openapi.json, not git log.**

### First real-data export proof (splat_32d926d9, 1.32M gaussians, SH-3)
- **SPZ v4 ✅** 313MB→30MB in ~30s, sha256 verified against manifest `files[].sha256`.
- **GLB (KHR_gaussian_splatting) ✅** 317MB, valid glTF-2 header, sha256 verified.
- **UE 5.6 bundle ✅** built in 22s, 627MB zip, 14 entries (gaussian ply/spz/glb, twin/mesh glb,
  approved scene.glb/.blend, survey artifacts, receipts), zipfile integrity OK.
- **SOG ❌ REAL LIMIT**: CPU SOG at default sog_iterations=10 exceeds CONVERSION_TIMEOUT_S (60
  min) on 1.32M gaussians → `-1 conversion exceeded 3600 seconds`. Wave-6 Export Center must
  default SOG iterations low / warn on big scenes; consider GPU SOG or a bigger timeout as backend
  follow-up.
- **streamed-SOG ❌ UPSTREAM BUG**: splat-transform v2.7.1 `writeLod` throws "Missing lod
  assignment" with `--lod-chunk-count`/`--lod-chunk-extent` on this scene. Needs a minimal repro +
  upstream issue (or flag-order fix) before wave 6 exposes the knobs.
- Manifest gap for the UI: failed artifacts carry the log in `error`, no short `reason` — Export
  Center should render `error` tail when `reason` is absent.

Next: wave 4 (Spark feature port + /view workspace tabs), wave 5 (Edit mode), wave 6 (Export
Center), wave 7 (classic deletion + SuperSplat self-host, restart window 2).

## PROFESSIONALIZATION WAVES 4-7.2 SHIPPED (2026-07-25, same day) — Spark default, workspace tabs, Edit mode, Export Center, SuperSplat self-host

Continues the waves 1-3 section above. All shipped and LIVE; classic-viewer deletion (wave 7.1) is
the only remaining plan item — gated on RToony's live soak + explicit sign-off.

- **Wave 4.1** (`a4e7acd`): Spark viewer parity port — fly-to, camera-frusta overlay (DOM/SVG
  screen-projected like classic, so crop/paint picking can't double-fire), zoom-to/view-from-camera,
  WASD pan + arrow roll (OrbitControls/mkkellogg math mirrored), reset, trimmed shortcut legend.
  **Spark is now DEFAULT** (absent localStorage key ⇒ Spark; escape hatches: ⋯ menu toggle,
  `?viewer=classic` (unpersisted), `splatlab.sparkBeta=0`); classic remains the auto error-fallback.
  NOT click-tested headlessly — fly-to/frusta/WASD feel needs the live soak.
- **Wave 4.2** (`0b15ed6`): /view is a tabbed workspace — breadcrumb + status badge + one ⋯
  overflow menu (secondary path for every action, one release); tabs View | Measure | Objects |
  Edit | Export with per-mode toolbars. Viewer NEVER unmounts on tab switch (lanes overlay it).
  Spark tool panel = Measure-only via new `toolsVisible` prop; bottom drawer <1024px.
- **Wave 5.1** (`bdeab5e`): native Edit ops — crop BOX beside crop-sphere (shared predicate
  `previewRemoval()`, red-tint previews byte-identical for sphere; `-B` KEEPS inside, preview
  tints outside), floater cleanup / decimate (pct = KEPT) / batched T-R-S transforms in the Edit
  lane, all armed-confirm + verbatim {detail} errors + version_before undo. New `reloadToken`
  prop routes lane edits into the existing crop reload nonce (camera resets on reload — inherited).
- **Wave 5.2** (`00b8814`): restore-point timeline (GET /edit/versions) with per-version armed
  Restore; restore is itself undoable; 5-cap explained.
- **Wave 5.3** (`73b6e70` + `22553af`): Backend Batch A (restart window 1, taken 13:5x after
  0-jobs check): `langfield_stale` in _job_payload (same helper as the 409 guard — can never
  disagree), GET /api/splat/activity (arbiter holder + per-job lock flags editing/
  preview_exporting/meshing/exporting; restart-truthful by construction), GET /jobs/{id}/objects
  listing. Frontend truth layer: useActivity 4s poll, P6-modal "started elsewhere" banner (closes
  the STATUS.md:1304 gap), polled staleness replaces ALL local made-stale state, amber header
  badge, distinct "no field built" vs "field stale" gating copy.
- **Wave 5.4** (`bdff5ef`): semantic edit panel — text → threshold → delete|isolate|extract →
  armed confirm; real matched counts; extract links the derived scene. Request field is `text`;
  **extract does NOT mark the field stale** (only delete/isolate do — coded to source after my
  brief said otherwise). No client-side preview (deliberate scope cut).
- **Wave 6** (`3d5ab71`, `78d2e54`): Export Center (4 lanes: portable formats w/ knobs — SOG
  iterations DEFAULTS 2 everywhere incl. the DownloadMenu quick-build that previously posted {}
  ⇒ backend default 10, the exact config that timed out live; big-scene >5-iter warning; failed
  chips render the `error` log tail; collision surfaces as top-level manifest key w/ voxel_url +
  mesh_url — scene.voxel.bin only ships inside the UE bundle; post-hoc mesh + contours triggers
  with honest prerequisite gating) + Objects panel (GET listing + extract form w/ activity banner).
- **Backend follow-ups** (`e52eb5c`, `9e3bd31`): streamed-SOG "Missing lod assignment" was OUR
  bug, not upstream — lod-meta.json output requires LOD-tagged gaussians; `-l 0` injected after
  the input (repro: backend's exact shape fails in 0.04s, tagged succeeds; regression test pins
  the command). `/activity` gained the `surveying` flag (geo_route._GEO_EXPORT_LOCKS was the one
  missed busy-lane).
- **Wave 7.2** (`20537e6`, `e2940a1`, `020db47`): SuperSplat SELF-HOSTED — authed static serve of
  `SUPERSPLAT_DIST` (default ~/projects/supersplat/dist) at /supersplat/*, traversal-guarded,
  303→/login for unauthed humans (020db47 fixed the initial 401); **portal :3300 /supersplat proxy
  DELETED** (portal can now retire without breaking /view; PORTAL_ORIGIN survives only for
  healthz). POST /jobs/{id}/edit/upload: streamed multipart ingest of an externally-edited PLY as
  a first-class edit version (validate header → snapshot op="upload" → replace → regen → mark
  stale; 400/409/413 paths leave everything untouched). Edit lane "Import edited .ply" card +
  SuperSplat roundtrip guidance. **Restart window 2 taken + verified live** (303/200/200,
  edit/upload in openapi, proxy gone). nexus-manifest log entry added.

Tests: 540 → **567 passed / 2 skipped**. eslint 0 errors; tsc clean outside the classic viewer
(its 9 errors die with wave 7.1). Frontend deploys all dist-swap; two restart windows total,
both 0-jobs-verified via splatlab-safe-restart.

### Open / gated
- **Wave 7.1** (classic deletion + `tsc && vite build` gate + vitest smokes): NEEDS RToony's
  Spark soak verdict + sign-off. Until then `npm run check` stays red by design.
- Cloudflare ~100MB body cap applies to the new upload roundtrip through the public hostname —
  big edited PLYs need the LAN/Tailscale origin (same as the create lane).
- SuperSplat sw.js/PWA behavior under the new origin unverified until first live open.
- Multi-level LOD pyramids for streamed-SOG (current fix = single level 0, spatial chunking works).
- Collision/unreal_bundle have no per-artifact staleness (manifest-level only) — backend follow-up.

### Streamed-SOG fix PROVEN LIVE (2026-07-25 15:37) + throughput truth
`splat_30b75bc81f` (green-bottle, 289,115 gaussians), forced streamed-SOG at sog_iterations=2
through the live API post-restart: **status ready** — valid lod-meta.json (1 LOD level, chunk
tree with bounds), 9 files, 69MB PLY → 6.2MB, served through the relative-chunk route. The lane
went from instant "Missing lod assignment" crash to working artifacts.
**Throughput caveat (measured):** that build took ~44 minutes for 289k gaussians at iterations=2
(single chunk — whole scene fits one extent-16 chunk, so it's one big SOG encode). Extrapolating,
scenes ≳400k gaussians risk the 60-min CONVERSION_TIMEOUT_S even at low iterations. Streamed-SOG
is CORRECT now but SLOW — backend follow-ups if it matters: longer timeout for this format, GPU
SOG (`-g gpu`), or parallel per-chunk encoding. Export Center copy already warns builds can take
many minutes.

## WAVE 7.1 SHIPPED (2026-07-25) — classic viewer DELETED, builds type-checked. Program COMPLETE.

RToony signed off after the Spark soak ("proceed to the next wave"). `a5f43c9` (+ style commits
`013faaf`/`f91689c`: Spark tool panel + all three legends now SOLID #0a0f1a/kept hues, per his
call):
- DELETED: components/splat-viewer.tsx (mkkellogg classic), pages/spark-test.tsx (the 601-line
  spike route), mkkellogg.d.ts, @mkkellogg/gaussian-splats-3d dependency. Zero mkkellogg chunks
  in dist.
- Shared viewer prop types live in components/viewer-types.ts.
- /view renders Spark unconditionally; a viewer crash shows a recover card (no silent fallback);
  sparkBeta/localStorage/?viewer=classic/menu toggle all removed.
- Home featured pane = lazily-loaded Spark (3D chunk stays out of the landing bundle).
- `npm run build` = `tsc --noEmit && vite build` — gate PROVEN (injected type error → exit 2).
  tsc fully clean repo-wide; the grep-exclusion era is over. eslint 0 errors, vitest 9/9.

**The 7-wave professionalization program is COMPLETE.** Remaining ideas are backlog, not plan:
multi-level streamed-SOG pyramids / GPU SOG (throughput), per-artifact collision staleness,
SuperSplat save-back button (trigger: manual roundtrip proves annoying), merge-scenes GUI
(trigger: two geo-anchored scenes of one site), post-hoc langfield rebuild lane, light theme.

## EDIT TOOLS UNBROKEN + STEPPED PROGRESS (2026-07-25 evening) — first successful edits ever

RToony hit "Unsupported output file type: ...edit-tmp" on Clean up floaters (3× live 500s).
Root cause: `_edit_tmp_path` built temp names ending `.edit-tmp`; splat-transform v2.7.1
dispatches output format purely on suffix. FIVE sites shared it (apply + all three regens +
`.float-clean`); **no edit had ever landed on this machine** — the extension-blind test stub
kept 80 tests green against the wrong contract.

- **Fix** (`649b539`): temp shape `.edit-tmp.<token>.<real name>`; `.float-clean` same
  treatment. **Stub now enforces the extension contract** (falsification-proven: buggy helper +
  new stub fails the suite). Unit test on the helper.
- **Stepped progress** (`1c96bc9` + `a11815d`): in-memory EDIT_PROGRESS (begin/step/end at
  transaction boundaries; semantic adds leading "match"), additive `edit_progress` on
  GET /activity; frontend EditProgress strip (stage dots + human labels + 1s elapsed ticker +
  patience copy, indeterminate degrade), useActivity(fast) 2s poll during local edits,
  "Applying…" button text, server-truth "edit in progress (started elsewhere)" lock banner.
- **Crop lives in the Edit tab now** (`b948729`, RToony's call): `toolsVisible` →
  `panelSections: "measure"|"edit"|null` — Measure = search/paint/dims, Edit = crop sphere+box
  beside the op cards; leaving Edit disarms crop modes; <1024px drawer capped 30vh on edit.
- **Live proof (restart window taken, 0 jobs)**: keep-everything crop on the hydrant → HTTP 200
  in 8.7s, 0 warnings (all regens work), v1 snapshot; /activity showed apply(2/6)→compress(3/6)
  mid-flight. Revert v1 → 5 files restored. **Acceptance = RToony's real ask: Clean up floaters
  → 910,560 → 909,532 (−1,028) in 10.1s, GPU splat-edit lane visible as holder, restore point
  v3 left in place for undo.** Note: −1,028 is `-G` engine defaults — conservative; if streaks
  remain, the backend accepts tuning params the lane doesn't yet expose (backlog), and crop
  handles the rest.
- Tests 567 → **570 passed / 2 skipped**; frontend gates green (tsc-gated build, eslint 0,
  vitest 9/9).

## NAVIGABLE WORLD + CANDIDATE BAKE-OFF (2026-07-26) — splat → walkable three.js world, and generative candidates

13 commits on `object-calibration-staleness` (unpushed). Suite **626 passed, 2 skipped** throughout.

### What shipped
- **Texture bake** — `object_texture.py`: clean → simplify → xatlas unwrap → bake gaussian colour into a UV
  map. Decouples colour fidelity from face budget. Exposed as `texture:true` on POST /jobs/{id}/objects
  plus an Objects-panel control. Also takes a POINT CLOUD as geometry source (Poisson direct from
  gaussians) — the scene lane never meshes, so this is the normal path.
- **Whole-capture solidify** — `scene_solidify.py` (props + shell), `world_collision.py` (static/prop
  classification + CoACD `UCX_` hulls), `world_shell.py` (watertight walkable solid), `world_gate.py`
  (acceptance gates).
- **Walkable world** — `/world/:jobId`, three.js + three-mesh-bvh capsule collision, backend
  `/world/manifest` + `/world/file`. Renders the visual shell, COLLIDES against the solid.
- **Candidate bake-off** — `world_bakeoff.py` + `mesh_gate.py --transform`, registers candidates into one
  frame then scores against the real photos.
- **Parametric authoring** — `parametric_schema.json` + `parametric_build.py` (measure/validate/build).
- **Generative** — `object_generate.py` wrapping SAM 3D Objects (13 GB ckpts local).

### Findings that cost time — do not re-derive
- **Render geometry ≠ collision geometry.** Bonsai's TSDF shell is a lacy web (645 components, 67% floor
  continuity). NO decimation budget preserves connectivity (2,088 components even at 40%). The walkable
  solid must be voxelised separately — `world_shell.py`, 222k tris, watertight, floor continuity 1.0.
- **`/collision` was voxelising in the wrong frame AND uncropped.** Captures are Z-up, splat-transform is
  Y-up, so `--voxel-floor-fill` filled sideways; and no `--filter-box` meant 64×53×70 units of sky.
  Fixed: 6,826,202 tris → 138,556. splat-transform's contract is `input [ACTIONS] … output`, actions in
  order — the source must LEAD.
- **Open3D aborts the PROCESS** (uncatchable C++ terminate) on `enable_post_processing` for an image-less
  GLB, and on `np.asarray()` of an empty texture placeholder. Read the glTF JSON chunk first; call
  `tex.is_empty()` always.
- **pymeshlab screened Poisson never returns** (>240 s at depth 6 on 49k faces). Open3D: 1.5 s at depth 8.
- **xatlas is steeply superlinear**: 0.2 s @8k → 46 s @200k → 1564 s @400k. Measured 13× fix (chunked
  unwrap, 118 s) documented at the call site, NOT applied — it changes atlas layout.
- **Volume-based hull checks are meaningless** on density-trimmed Poisson meshes (not watertight). Use
  sampled surface coverage.
- **The object lane's `bbox_extent_m` is a MISLABEL** (fixed 2026-07-21). Older meshes carry scene units
  under that key. Bonsai is UNCALIBRATED.
- **PSNR ranks by alignment, not quality.** The operator-preferred generated hydrant scores LAST (9.83 dB
  vs 13.76) because its yaw is fitted. Only rank exactly-registered candidates. Fix = photometric pose
  refinement, not a different verdict.
- **DATA DEFECT: camera 0 of `splat_513e89171d` is badly posed** — ~6 dB shared dip across every
  candidate, object renders ~3× oversized. One bad frame in 45.

### Open
- Bonsai has no real scale (viewer guesses 0.9428 u/m from a 2.6 m storey; bicycle then reads 1.86 m).
- Visual shell still fails its own gates; chunked xatlas is the measured fix, pending a design call.
- `object_texture.py` reports coverage PRE-dilation (0.53) under the same name as the shipped atlas (0.98).
- 13 commits unpushed.
- **UE 5.6.1 installed on Triforce** (`G:\UE_5.6`, 25.76 GB, verified; Quixel Bridge + Fab plugin).
  Triforce: i7-7700 4c/8t, 64 GB, RTX 2080 SUPER 8 GB, Win10 19045, VS Build Tools 2022 present.
  C: only 27.8 GB free — keep engine work on G:. UE is a POLISH station; three.js stays the runtime.

## BLENDER+UE PIPELINE CONTINUATION WAVE (2026-07-26, second session) — UE first light staged, bundle catches up, polish round-trip

Plan: ~/.claude/plans/purring-coalescing-crane.md (approved). 7 commits eb11eea..850d727 on
`object-calibration-staleness`. Suite 626 → **666 passed / 5 skipped** (3 opt-in lanes run once
this session: mesh-env e2e ×2, real headless Blender ×1 — all green). Two safe-restart windows
taken (0-jobs verified), routes proven via /openapi.json.

- **B2 coverage honesty (eb11eea)**: `texture.coverage` now = the SHIPPED dilated atlas;
  `coverage_rasterized` = the pre-dilation fraction (was misfiled under one key: 0.53 vs 0.98).
  world_gate comparison reads the rasterized key for either report vintage. First-ever
  test_object_texture.py (12 tests incl. real-env e2e).
- **B1 bundle catch-up (080f7e6)**: UE bundle ships `World/` (shell, collision_shell, elements,
  UCX hulls w/ `collision_for`, combined per-prop `World/Props/<slug>.glb` — UCX binds only
  within ONE imported file; node-name preservation probe-proven) + `Objects/` (textured+atlas,
  bakeoff.json, **winner GLB with score metadata**; provenance: capture-native/ours →
  captured-derived, else generated=render-only; out-of-tree winners skipped visibly) + 4th
  actor child `WorldGeometry` + manifest `world` block + serve-time `world_current` flag.
  bundle_tool requires WorldGeometry only when world files present (skew-safe both ways).
  world_collision.py emits `UE_<slug>.glb` per prop (first CoACD-real file on next collision run).
- **A-lane: UE FIRST LIGHT STAGED ON TRIFORCE.** Preflight probed live (UE 5.6 CL-44394996 at
  G:\UE_5.6, MSVC 14.44, py 3.14, driver 566.03; **Windows SDK MISSING**, RDP disabled, also a
  UE_5.8 dir). `~/scripts/triforce-ue-station-prep.sh` (dry-run proven) = RToony's --apply gate
  for RDP+SDK. Staged on G:\splatlab-ue\: SplatLabUE56 project + PS scripts + **NanoGS v1.0.3
  prebuilt** (Plugins/NanoGS) + the REAL hydrant bundle (splat_513e89171d: exports built SPZ
  22.6MB/GLB 218MB, 398MB zip, byte-verified transfer) — probe/verify/stage receipts ALL GREEN
  on Triforce (staged_path SplatLabImports\splat_513e89171d\50a210ca9afe0019, 14 files, NanoGS
  selected; bundle carries Objects/fire-hydrant + winner-textured 13.76dB). Attended half =
  `integrations/unreal/first-light-runbook.md` (copy beside the kit on G:). Gotchas: Windows
  scp needs `-O` (sftp mode dies); CLIXML noise → grep -v; base64 -EncodedCommand for PS.
- **C1 polish round-trip (09949fd)**: `backend/glb_check.py` (stdlib GLB validation — Open3D
  BANNED here, hard-aborts on image-less GLBs) + `polish_route.py`:
  POST /jobs/{id}/objects/{slug}/polish + /world/elements/{slug}/polish (shell ok). edit/upload
  discipline: stream→validate→lock→version prior→atomic land→provenance receipt; every
  400/404/409/413 leaves the tree untouched (20 tests). Walker needs no change; world manifest
  gains additive `polished` marker; _OBJECT_FILES += polished/polish-receipt.
- **C2 UI (f1e3312)**: objects listing carries additive `bakeoff` verdict; "Walk world" button
  on /view when `world_available` (bonsai proven live); verdict line on object cards
  (formatBakeoffVerdict, vitest'd); Polished .glb in the downloads list.
  ⚠️ Route-decorator gotcha cost 12 tests: inserting a helper between @router.get and its
  handler registers the HELPER as the route (422s everywhere). Helpers go above the decorator.
- **D1 chunked xatlas (c7ac40d)**: `--unwrap-chunks N` (default 1 byte-identical; solidify
  passes 4 for the SHELL only). **Real bonsai shell @400k: unwrap 96s vs 1564s = 16×**, bake
  total 119s, shipped coverage 1.0, texture gate PASSES. Honest residual (scratch world_gate
  A/B, live tree untouched): shell_connectivity/floor_continuity still FAIL — 2,088 components,
  the EXACT recorded source-TSDF number → the blocker is source lacing, not unwrap cost; fix =
  bake onto the voxel-solidified shell (design call, backlog). Also seen: live world.json was
  rewritten 08:24 with shell:null (stale shell.glb on disk) and prop_integrity now 3/5 on the
  current tree — pre-existing drift, not this wave.
- **D2 typed export_glb (850d727)**: 9th allowlisted action on the restricted workflow
  (:9877 MCP tool `export_blend_glb`): zero free-form params, staged output, glb_check
  readback, receipt at _blender/exports/scene-vNNNN.{glb,json}. REAL headless Blender 4.5.11
  integration test run this session (2 passed). Loop closes: export_glb → polish upload.

**Awaiting RToony**: (1) `! bash ~/scripts/triforce-ue-station-prep.sh --apply` (RDP + Windows
SDK), then the attended first light per the runbook (screenshot receipt → G:\splatlab-ue\
receipts\); (2) branch is ~20 commits ahead, unpushed — push is his call; (3) visual-shell
voxel-bake design call.

## UE FIRST LIGHT ACHIEVED UNATTENDED (2026-07-26 ~11:55) — receipt ok:true
RToony ran prep --apply (RDP live; SDK add initially failed on MY Start-Process array-quoting
bug — path with spaces split into 3 tokens; fixed in the script, retried clean: Windows SDK
22621+26100 + NetFxSDK 4.8.1 installed). SplatLabUE56Editor + NanoGS compiled FROM SOURCE on
Triforce (UBT green, 419s; needed the NetFx SDK — Build Tools omits it). Headless first light:
`-run=pythonscript -nullrhi` imports WORK but actor spawn EXCEPTION_ACCESS_VIOLATIONs
(commandlet has no level-editor frame; recorded in firstlight.py docstring) → full-editor
`-ExecutePythonScript -RenderOffscreen` run SUCCEEDED: scene.scene + winner mesh imported,
FirstLight.umap assembled (SplatLabSceneRoot → GaussianRender + ConventionalGeometry), numeric
witness: dominant axis Z = UPRIGHT (NanoGS preserves +Z-up PLY, no axis correction — folded
into README); scale NOT auto-applied (bounds = whole cloud incl. background splats; operator
sets root scale). Receipt: G:\splatlab-ue\receipts\firstlight-splat_513e89171d.json. Vault:
`Nexus - Triforce Windows Login` (d550d669) added by RToony via vault-add-secret; verified
readable in-memory. Remaining attended: 30-second RDP eyeball of the FirstLight level + F9.

## BLENDER STUDIO LOOP — W1-W3 SHIPPED (2026-07-26 pm; plan purring-coalescing-crane v2)
- **W1 voxel shell (64423ba)**: scene_solidify `--shell-source voxel` + `--shell-only`
  (patches world.json in place). Bonsai: 41s bake, 120k faces → **world_gate shell gates
  FLIPPED to PASS** (1 component / 1.0 fraction / floor 0.94; was 2,088 components). Visual
  surface now coincides with the collider. prop_integrity still FAIL = pre-existing drift.
- **W2 polish UI (0af2e3f)**: PolishUploadZone (shared, .glb preflight/arm/progress) on
  object cards + per-row in the world Elements panel (incl. shell) with reloadNonce reload.
  All frontend gates green.
- **W3 Blender infra (b6979c5, 52b3c3f)**: `splatlab-blender-mcp.service` LIVE (user unit,
  loopback :9877; initialize handshake verified "SplatLab Blender") + registered as claude
  MCP `splatlab-blender` (✔ Connected; :9876 untouched) + manifest node/log +
  `~/bin/splatlab-blender` launcher (versions→_regen→GLB fallback; prints return path).
  **First-ever real _blender lane on splat_aea04ab3**: P6 assemble via API (6 built, gate
  ok) → snapshot v1 → transform v2 → export_glb 6 meshes/81.5MB validated. Caught+fixed:
  glTF export of GN-driven assembled scenes needs export_apply (readback gate flagged the
  zero-mesh export exactly as designed). Polish-upload leg of the loop deliberately awaits
  REAL polished content (W4) rather than landing test data in a live slot.
- **W4 pending RToony's capture** (scene w/ ground, 150-300 photos, taped distance, map
  pin — spec in the plan). Sequencing: derive (objects/P6/world) BEFORE destructive edits.

## SEMANTIC PAINT-TO-GENERATE PHASE — P1+P2 SHIPPED (2026-07-26 pm; plan purring-coalescing-crane v3)
Direction (RToony): crop → label → generative uses labels ("representative but faked").
Taxonomy fixed+extensible; first gen output = class-textured ground; testbed = bicycle.
- **P1 crop honesty (8b664db)**: "queued" lead step (host-lock waits visible — the recorded
  "takes forever" source), revert/upload rails, to_thread snapshots (event loop no longer
  freezes /activity), GLOBAL EditProgress under the tab row (survives tab switches/viewer
  teardown), off-by-one fixed (rail read done while running), warnings[] surfaced,
  bbox-derived slider ranges, placement-miss notice, honest preview-count copy. 668 tests.
- **P2 langfield rebuild (aa9c949)**: REALIGNMENT not re-lift — langfield_realign.py
  (exact-xyz map rebuild, total-or-nothing; painted records carried via xyz snapshots,
  now written on every paint commit; legacy via old-map chain) + POST /langfield/rebuild
  (edit-lock, rail, GPU lane only for one-time ckpt_xyz cache, worker /invalidate) + UI:
  stale banner + one-click Rebuild CTA in search AND crop surfaces, TEST PATTERN badge.
  681 tests. **LIVE PROOF on the hydrant** (stale from RToony's own edit session):
  rebuild in 16 s incl. first checkpoint load — 909,527 rows realigned, 1,610 edited-away
  rows detected, relevancy 200 w/ X-Count exact + hydrant found. Bit-exact xyz through
  splat-transform CONFIRMED on real edits (risk #1 retired).
- **Worker truth found**: splatlab-langfield.service binds :3425 (drop-in; :3417 code
  default is cerberus-studio's port), deliberately on-demand/disabled — language tools are
  down whenever nobody started it (`systemctl --user start splatlab-langfield`). Started
  this session with the new /invalidate + snapshot code.
- Open: P3 class-label painting → P4 class-textured ground/shell → bicycle E2E.

## PAINT-TO-GENERATE P3+P4 SHIPPED (2026-07-26 evening) — the loop is closed
- **P3 class painting (400aa3f)**: class_taxonomy.json (9 classes, each w/ generative
  meaning) + stdlib loader (extend-only job extras); class_labels.py store (overrides
  discipline + xyz snapshots DAY ONE); worker class_add/delete/map/summary; app routes
  (taxonomy validated before the worker; binary map w/ X-Class-Order); realign carries
  class records. Viewer: Label|Class brush toggle, palette chips, per-record delete,
  "Show class layer" via new direct-color modifier (no 4-channel limit). 692 tests.
- **P4 class-textured world (b5e2462)**: semantic_ground keeps class_rel [N,C] (+taxonomy
  queries, --live-map post-crop honesty, --class-labels = user paint ABSOLUTE precedence);
  ground_mesh_build per-cell vote → ground_class_cells.npz; class_textures.py (seeded
  procedural tiles; textures/<id>.png upgrade path) + ground_texture.py (planar-XY UV,
  world-space sampling, jittered boundaries, readback, provenance
  "captured-geometry/class-textured") FAIL-LOUD in /scene/ground (fmt=glb_classed/
  atlas_classed/report_classed); shell --class-map (xyz-keyed, composite post-dilation,
  per-class blend_capture) via scene_solidify's auto class-map. Survey rails proven
  untouched. 697 tests + probe-env e2e.
- **LIVE BICYCLE PROOF (machine half)**: POST /scene/ground on splat_3aaf8067 →
  10,689 cells auto-classed (grass 7,505 / dirt 1,360 / pavement 1,223 / gravel 601) →
  ground_classed.glb, texel fractions 61/17/13/9% — matches the scene's real lawn+path.
- **Awaiting RToony (attended half)**: paint pass on bicycle (Measure → Paint → Class:
  correct/override the auto classes, e.g. paint the gravel patch properly) → re-run
  /scene/ground (paint precedence) → optionally solidify --shell-source voxel for the
  classed shell + /world walk. Worker note: language tools need
  `systemctl --user start splatlab-langfield` (on-demand by design, :3425).

## SPARK EDITOR PROFESSIONALIZATION (2026-07-26 late; plan enchanted-shimmying-hollerith)
RToony: "polish the edit tools + make it a more professional software experience,
e.g. Esc should cancel paint mode." Frontend-only wave, `6081ee9..2fbb300`.
Backend untouched — no restart window, `npm run build` alone deployed it.

- **Baseline commit `6081ee9`**: the working tree already held an uncommitted first
  cut from 15:41 (log-scale sliders, class-coloured brush tint, first-cut Esc/[/]/Z).
  Landed alone as a rollback point before building on it. Also corrected a stale
  receipt: recent commits claimed "eslint 0 errors" but HEAD actually had 1 (unused
  `setStrokeBusy`) — proven by stashing the diff and re-running eslint.
- **`lib/viewer-shortcuts.ts` (`e9cb5da`)** — pure `resolveShortcut` / `escapeAction` /
  `stepRadius`. The input layer lives in ONE `useEffect([url])` and is unreachable
  from a test; there is no DOM test runner here. 21 tests, **falsification-proven**
  (inverted the Esc ladder + dropped the clamp → 3 failures; restored → green).
- **Esc is a ladder, and never eats work**: cancel a pending confirm → clear an
  unapplied placement → disarm. RToony's explicit call: an uncommitted paint
  selection is NEVER discarded by Esc. So the new **ToolHud** (top-centre, solid
  `#0a0f1a`) always shows `N splats selected · not committed [Discard]` even with no
  tool armed — that state was previously invisible-but-live.
- **Four disarm holes closed**: `measureArm` never disarmed paint (and `onClick`
  tests paint FIRST, so arming the ruler silently kept painting); leaving Measure
  left paint+ruler armed; the paint section could unmount on a stale langfield with
  `paintMode` still true; `pointercancel` was unhandled so a cancelled drag left
  `controls.enabled=false` **permanently** ("orbit broke"). Also: dblclick no longer
  places a crop centre AND yanks the orbit pivot; Ctrl+S reaches the browser again.
- **Keys**: `B`/`C`/`Shift+C`/`M` tools — and a tool key **switches to the tab that
  owns the tool**, so a shortcut can never arm something invisible. `[ ]` size
  (clamped to the slider's OWN bounds — one shared `brushBounds`/`cropBounds`/
  `boxBounds` definition), `Ctrl+Z`/`Ctrl+Shift+Z` (new redo stack, invalidated by
  any new stroke), `Enter` commits (arm-then-apply on destructive crops), `Del`
  discards, `1`–`9` pick a class (taxonomy is exactly 9), `?` help, `0` reset view,
  `H` hide chrome. Everything state-reading goes through a **ref trampoline** —
  a direct binding acts on whatever was true when the scene last reloaded.
- **ShortcutLegend rewritten**: it listed 3 camera lines and its comment claimed
  `F/G` + `=/-` bindings **that never existed in Spark**. Every row is now wired and
  tested. `Kbd` chip promoted out of world-view.
- **SizeControl**: log slider + a typed exact box — you cannot land on 0.37 m twice
  by dragging, and "same radius as last time" is a real need.
- **One progress rail (`2fbb300`)**: the page rail already covered everything but 8
  section-local rails were never removed → two identical rails at once. Deleting them
  naively would have regressed (the page rail was only instant for viewer crops), so
  `EditLane` now reports `onBusyChange` up and the page ORs it in; unmount reports
  false so a mid-op tab switch can't strand it. Tab also syncs to `?tab=`.
- **Gates**: eslint **1 error → 0**, vitest **14 → 35/35**, tsc-gated build green,
  backend suite untouched at 699 passed / 6 skipped. Deploy confirmed structurally
  (new UI strings present in the built bundle, serves HTTP 200).
- ⚠️ **NOT driven in a real browser** — the Chrome extension is disconnected and
  playwright has no chromium. The live pass is RToony's; see the plan file.
- ✅ **SHIPPED instead (`096bece`) — see the DUPLICATE section below.** The parked
  draft's hardlink optimisation was unsafe and was removed rather than repaired.

## LIVE-VERIFIED IN A REAL BROWSER (2026-07-26, same night) — the wave above is now proven
RToony: "connect with playwright and install chromium. I want this feature for you."
Installed `~/.cache/ms-playwright` (641 MB, chromium-headless-shell 148 + ffmpeg).
New gate: **`tools/verify-editor-live.py`** — drives the real app with real
keystrokes/clicks and asserts against the real DOM.

- **Results: `splat_6b2e82e5` (486,960 gaussians, live langfield) 31/31 ·
  `splat_7f98469203` (14k) 23/23.** Proven, not asserted: the Esc LADDER with a real
  placement (#1 cleared the centre and STAYED in the tool, #2 disarmed); `C` from View
  switching to `tab=edit` and arming; `[`/`]` clamping (90 presses bottomed at 0.0021,
  40 topped at 4.23); paint → **Esc → "834 splats selected / NOT COMMITTED / Discard"**
  (RToony's rule, verified); Ctrl+Z 834→591 and Ctrl+Shift+Z 591→834; Del discarding;
  1–9 class pick; `?` card rows; `H`; Ctrl+S NOT swallowed; typing "0.42" in the radius
  box firing no tool keys and committing on Enter; zero console/page errors.
- ⚙️ **GPU, not SwiftShader.** Software raster is NOT viable here: a 487k-gaussian scene
  pushed CDP round trips to **15 s** and a 900k one wedged the page outright (two runs
  hung, one for 12 min). `--use-gl=angle --use-angle=gl` binds the RTX 5090 headlessly
  (verified `ANGLE (NVIDIA ... RTX 5090, OpenGL 4.5.0)`) and the same page is instant.
  A responsiveness probe now fails fast instead of hanging.
- 🐛 **Three real bugs the browser caught that tsc/eslint could not:**
  1. `860afb9` — every tool's lower bound is max/2000, so the small end of the brush,
     crop-radius and box-extent ranges all printed as a flat **"0.00"**. Adaptive
     `preciseSize()`; the floor now reads 0.0021.
  2. `d39e9c1` — **the Edit/Export/Objects lane had NO background at all.** It was
     `bg-surface/95`, but the colour tokens are hex-valued CSS vars
     (`surface: "var(--surface)"`), so Tailwind cannot build their opacity variants —
     the built CSS has **zero** `.bg-surface\/95` rules. Only a backdrop-blur was
     rendering and the splat read straight through the floaters/decimate copy. Now
     solid `bg-surface`.
  3. `d39e9c1` — the new HUD was anchored top-centre, directly on top of the "In this
     scene" legend. Moved to bottom-centre above the search pill (it reads as a status
     bar there anyway).
- ✅ **RESOLVED same night (`3936714`) — the token-opacity failure was systemic and is
  now fixed at the root.** `bg-surface/85` (app nav), `bg-accent/30` (disabled primary
  buttons) and every `border-accent/NN` were also emitting nothing; `border-accent`
  never compiled at all. Tokens migrated to channel triplets +
  `rgb(var(--x) / <alpha-value>)`; the two raw `var(--x)` consumers in `body` became
  `rgb(var(--x))` (a bare var now resolves to the literal "5 7 13" and paints nothing);
  `surface-raised` stays bare on purpose (baked-alpha overlay, no `<alpha-value>` form).
  Guarded by `src/lib/tokens.test.ts`, **falsification-proven** three ways (token back
  to hex / config back to a bare var / a token consumed bare → exactly one failure
  each). That test reads index.css with **fs, not `?raw`** — vitest stubs CSS imports
  to `""`, and the first attempt asserted against an empty string and passed for the
  wrong reason. Receipts: built CSS now has
  `.bg-surface\/85{background-color:rgb(var(--surface) / .85)}` and
  `.disabled\:bg-accent\/30:disabled{...}`; in a real browser the nav computes
  `rgba(5,7,13,0.85)` (was transparent) while body is still `rgb(5,7,13)` — unchanged.
  vitest 35 → 49, live editor gate 23/23, screenshot reviewed.


## DUPLICATE A SCENE SHIPPED + LIVE-PROVEN (2026-07-26 night) — edit the copy, keep the original
`096bece`. RToony: "fix the hardlink hazard and wire up duplicate."

- **The hazard was real and the fix was to delete the optimisation, not repair it.** The
  parked draft hardlinked `processed/`, `colmap/` and `_langfield/gauss_emb.npz` on the
  claim that they are only ever added to. False where it matters: **colmap opens
  `colmap/database.db` read-write (SQLite, in place)** and
  `backend/langfield/langfield_v2.py:172` writes `gauss_emb.npz` with a plain
  `np.savez_compressed` — a **truncate-in-place** write. Either corrupts the ORIGINAL,
  the exact disaster the feature prevents. Note `rm -rf` on a hardlink is SAFE (it
  unlinks one name), so the reroute path that clears `processed/` was never the risk —
  **in-place writes were**. ext4 here has no reflink, so there is no COW middle ground.
- **The speed argument was hollow anyway**: real copy of splat_6b2e82e5 (486,960
  gaussians) = **0.7 s / 1.50 GiB** on NVMe.
- Also in the route: the **SOURCE's edit lock is held for the whole walk** (an
  `/edit/apply` landing mid-copy would otherwise give a torn tree — splat.ply from after
  the crop, derived artifacts from before); a **507 disk preflight** with a 20 GiB floor,
  estimated with the same skip rules as the copy so it cannot under-count; `versions/`
  skipped (restore points are the original's history — a working copy starts clean);
  meta keeps scene truth (scale calibration, capture format, langfield flags) and
  replaces only identity/lineage/lifecycle (`parents`, `duplicated_from`,
  `"<original> (copy)"`, `pinned:false`).
- **LIVE END-TO-END PROOF**: 0 shared inodes across 267 files, max `nlink` 1, and after
  applying a real `crop_sphere` **to the copy** the ORIGINAL's `splat.ply` sha256 was
  byte-identical (`a6fc2863eb75f267` before and after) while the copy's changed. Test
  scene deleted after; source intact, disk back to 228 G free.
- **Guard**: `test_duplicate_shares_no_inodes_with_source` is **falsification-proven** —
  putting `copy_function=os.link` back fails that test and only that test.
- **UI**: "Work on a copy" card FIRST in the Edit lane, above everything destructive,
  two-click armed, states the disk cost and that history isn't carried; on success it
  navigates to the copy (leaving you on the original is how the mistake happens).
- Gates: suite 699 → **703**, ruff clean, tsc-gated build green, eslint 0 errors,
  vitest 49/49, live editor gate 23/23, Duplicate card verified in a real browser
  (first click arms, does not copy). Restart window taken; route confirmed in
  `/openapi.json`.

## PAINT → GENERATIVE WORLD, END TO END (2026-07-26 night) — RToony's classed world walks
Long session, `6081ee9..0c96a33`, all pushed. The editor wave is in the section above;
this records the pipeline run and the infrastructure it broke loose.

### The loop closed
RToony painted classes on the bicycle scene (grass 156,454 · pavement 108,887 ·
vegetation 58,286 · dirt 4,774) and it drove the generative stages:
- **`/scene/ground` with his paint beat the machine.** vs the 15:08 auto-classification:
  pavement **1,223 → 2,893** cells (the machine badly under-detected the path), gravel
  **601 → 99** (it had hallucinated gravel), grass ~flat. **Mean confidence 0.202 →
  0.614.** Of the 270,247 gaussians clearing the class map's 0.6 floor, **270,115 are
  his one-hot paint** — the machine's own guesses don't clear it.
- **Walkable classed world**: shell `classed=true`, 24,268 tris, **4/4 collision gates**
  (1 component / watertight / floor 1.0 / no holes), auto-picked from 4 candidate routes
  (plain voxel scored 29 components, smooth 509; carve won). Props bicycle/bench-seat/
  bicycle-2 all built + textured. 67.1 s.
- **The chain is longer than the notes implied and nothing warns you up front**:
  `scene/inventory` (279 s) → `scene/isolate` (12 s) → `scene_solidify --shell-source
  voxel` (67 s) → `world_collision.py` (writes world_manifest.json). Solidify fails clean
  with "not isolated yet (no object.ply)"; the viewer 404s until world_collision runs.

### Three infrastructure defects this exposed, all fixed
1. **The ground lane declared 6 GB for a 10 GB job** (`c512962`). semantic_ground loads
   gauss_emb as a rows×dim FLOAT32 tensor — 2,351,565 × 1152 = **10.09 GiB** — against a
   flat `SCENE_GROUND_VRAM_MB = 6_000`. The arbiter admitted it against 6.4 GB free and
   it died with CUDA OOM. Now measured by peeking the .npz member header (a few hundred
   bytes; np.load would pull 3.3 GB through zlib for two integers) → 14,417 MB.
2. **`splatlab-langfield` was unregistered with the orchestrator** (`4c65454`). It had
   ballooned to **22 GB of 32** and the arbiter could see the VRAM gone but had nothing
   to evict. Now a catalog entry (priority 1, evictable) + splatlab **re-wakes it on
   demand** — registering without that would have turned every eviction into a 503.
3. **splatlab's arbiter ignored `evictable: false`** (`4c65454`). It drives its own
   eviction loop and the orchestrator's manual evict endpoint is operator-only, so a
   heavy splat job could have stopped RToony's **dictation**. The orchestrator now
   reports `evictable`/`enabled` (it previously did not, so remote callers *could not*
   have respected it) and splatlab filters on them.
   Proof: worker warm / 9,668 MB free → arbiter evicted it, build finished in 42 s;
   worker evicted → language query self-healed, 5 matches in 31 s, no 503.

### Open
- ⚠️ **You spawn on the shell's ROOF.** The shell is watertight, so the walker's respawn
  ray-cast lands on the lid: y=14.4 in a 12.56-unit shell, `0 drawn`. The Elements
  panel's **"Go" buttons** drop you inside (same frame → 42.6k drawn). `0c96a33` exposes
  `collision_shell.spawn` (world_shell's proven-interior seed) but the VIEWER half is
  NOT done: the seed is in the voxel-probe frame (floor −0.37 → top 3.04) while the
  loaded GLB is 12.56 units tall — it needs converting, not passing through. A walker
  change that just used it was reverted for having no effect.
- Props are PASS-THROUGH: **coacd not installed** in the solidify env.
- Scene still uncalibrated (`meters_per_unit: null`).
- The langfield worker has **no VRAM ceiling** — eviction now reclaims it, but nothing
  stops it reaching 22 GB again between evictions.

---

## 2026-07-27 — Trust pass from the Kimi k3 investigation (branch `kimi-report-hardening`)

Acting on `~/reports/2026-07-27-splatlab-investigation/` (5 reports, read-only audit)
plus the operator's own framing: **"your rails outrun your train"** — excellent
guardrails, receipts and safety infrastructure, with the actual numerical cores
untested and `meters_per_unit` the least evidenced artifact in the system. The
instruction was *finish loops, don't open lanes*. Nine commits, all on a branch, all
additive; **nothing under `~/.config/systemd` was touched and no service was
restarted.**

Test suite **708 → 933 passed**, 6 skipped. Baseline captured before the first change.

### The three trust gaps, closed
1. **Numerical cores now tested.** 44 CPU-only tests on `object_generate`'s gate
   (`test_object_gate.py`): selection-by-agreement (the bonsai-in-the-bicycle case),
   every refusal path, `passed` never diverging from `problems`, rotation helpers
   staying proper rotations, and `derive_placement` end-to-end on a synthetic camera.
   No extraction was needed — the module already imports under the test interpreter.
2. **Scale is the spine** (`scale_calibration.py`). `POST /scale` now accepts
   `{references:[{dimension_id, real_length_m}]}` and the SERVER derives the factor
   from stored `dimensions.json`. Multiple references average; spread is reported
   (stddev / relative / spread ratio / `disagreement` flag) with each reference's
   deviation, so the STATUS.md:294 made-up-5-ft case shows up and the outlier is
   identifiable. One reference reports uncertainty as **null, not 0.0**. Evidence is
   ranked `dimension > manual > map`: a map-eyeballed factor replacing a measured one
   now **409s** unless `force: true`. Every change bumps `scale_generation`.
   The legacy `{meters_per_unit}` body still works unchanged.
3. **`POST /jobs/{id}/world/solidify`** — solidify → collision → gate, behind the
   heavy-work gate and the per-job mesh lock. A world that FAILS its gates is returned
   with the verdict attached; suppressing it would hide what the gates exist to show.

### Also shipped
- **`opregistry.py`** — persistent operation registry. The design point is keeping the
  truthfulness rail persistence would otherwise cost: rows carry a per-process
  `runtime_id`, and a `running` row from a dead process reads as **abandoned**, never
  as running. `GET /ops`, `GET /ops/{id}`, `/activity.operations`; startup reconciles
  orphans and says how many.
- **Unseen gaussians scored exactly 0.5 against every query.** A zero embedding row
  makes both softmax logits equal. Four copies of the relevancy math existed;
  `object_isolate`/`batch_isolate` zeroed unseen rows, and the two lanes serving every
  interactive query did not. `langfield/relevancy_core.py` is now the one definition
  (also the single `SIGLIP_CKPT` — was in 6 files — and the LERF negatives, 4 files).
  Torch branch is operation-for-operation identical, verified against the numpy branch
  inside the langfield-spike env: max abs diff < 1e-6, CUDA never initialised.
- **Vault secrets no longer reach the model subprocesses.** `object_generate.run_worker`
  built its env from `dict(os.environ)`, handing SAM 3.1 / SA-3DAO / the nerfstudio
  probe every API key, `DATABASE_URL`, `REDIS_PASSWORD` and `BW_SESSION`. Now an
  allowlist. Worker success is also structural: `produces=` requires the artifact to
  exist, be non-empty and parse — the 4 KB done-token log-tail grep was the fallback
  and could fail a correct run whose token was pushed out by late CUDA warnings.
- **Blender MCP**: optional `SPLATLAB_MCP_TOKEN` bearer gate (unset = today's
  behaviour, byte-for-byte) + the **first protocol-level test** — a real server spawned
  from its isolated SDK venv: 401 without the token, then initialize → tools/list (all
  9) → tools/call, and `../../etc` still refused over the wire.
- **`operator_audit` is real** — 25 call sites were awaiting a no-op. Append-only JSONL,
  never raises, and a request contributes only path + client host, never credentials.
- **Auth**: per-IP `/login` throttle (8 / 5 min, blocks the correct token while tripped);
  future-dated cookies rejected (only `age > MAX_AGE` was checked — a validly-signed
  forever-session); unknown `/api/*` GETs return a JSON 404 instead of the SPA's HTML.
- **Live systemd units captured** into `deploy/systemd/` with a README. Scanned for
  secret values first — they reference the vault, never values.

### Corrections to the investigation
- The **langfield "port trio" is not three ports.** `90-supervised-port-3418.conf`
  *contains* `--port 3425`. It is one live port + a misleading filename + a stale code
  default. Both code defaults now say 3425; renaming the `.conf` is a live systemd
  mutation, left to the operator with commands in `deploy/systemd/README.md`.
- `pick_and_gate_mask`/`derive_placement` did **not** need extracting to be testable.
- The `seen` mask was already written to `gauss_emb.npz`; only the consumers ignored it.

### Inventory staleness — per scene, not fleet-wide (operator decision, same day)
`INVENTORY_VERSION` is deliberately **still 7**. It is a hard cache key: bumping it
would silently force a multi-minute GPU recompute of every scene the next time anyone
opened it. Instead `relevancy_core.RELEVANCY_GENERATION = 1` is a **soft** marker —
an inventory from an older generation is still served and reports `stale: true` with a
reason, and `?refresh=true` recomputes ONE scene.

Absence of the key IS the stale signal, so no migration pass has to walk the job tree;
every one of the 7 inventories on disk reads as out-of-date today.

Staged (not applied): `~/scripts/splatlab-refresh-three-inventories.sh` — dry-run by
default, `--apply` to act. It refuses to restart while heavy work is in flight, backs
up all 7 inventories first, restarts the services (the fix is NOT live until they
restart — recomputing against the running pre-fix code would just rewrite the same
wrong numbers), recomputes **bicycle `splat_3aaf8067`, garden `splat_32d926d9`,
fire-hydrant `splat_513e89171d`** through the app's gated route, then asserts exactly
those three sit at generation 1 and prints the rollback. Dry run verified.

**First `--apply` run (05:06) failed loudly at its own verify gate — two real findings:**
1. The script's token parse was wrong: `tr -d '\n' | sed 's/.*PORTAL_TOKEN=\([^ ]*\).*/\1/'`
   joined the whole env file into one line, so the greedy `.*` matched
   `MAXIMUS_PORTAL_TOKEN` and `[^ ]*` swallowed every following variable — a 1145-char
   blob instead of the 64-char token, so all three calls 401'd. Now line-anchored
   (`grep -m1 '^PORTAL_TOKEN='`), length/whitespace checked, and **proven against
   `GET /api/splat/status` before anything is restarted** rather than merely non-empty.
   The recompute step now reports the HTTP code and `detail`, not a bare `generation=None`.
2. **The bicycle `splat_3aaf8067` has a STALE language field** (marker
   `2026-07-27T04:57:20Z`, version `v2-20260727T045701Z`) — the scene was edited, so
   every language route 409s. That is the stale guard working, not a fault, and it
   predates this work. Its inventory cannot be refreshed until the field is realigned.
   The script now detects it in preflight, SKIPS that scene by default, exits **2** so a
   partial run is never mistaken for success, and `--realign` runs the documented cure
   (`POST /langfield/rebuild` — exact-xyz row remap, paints carried across) first.

The restart DID happen on that run, so the fix is live: garden and hydrant now return
`stale: true, relevancy_generation: null, current_relevancy_generation: 1` — served and
flagged, exactly as intended. The script now detects an already-fixed build and skips a
redundant bounce.

Noted in passing: `splat_aea04ab3` (Bonsai) is at `version: 3`, so it will hard-recompute
on its next read via the pre-existing INVENTORY_VERSION path regardless.

Cached `q_<query>.png` heatmaps are also pre-fix, but each is overwritten by the next
query for that term, so they self-heal rather than needing a sweep.

### Open / deliberately not done
- `mesh/object_isolate.py` and `mesh/batch_isolate.py` keep their own already-correct
  inline relevancy copies. Consolidating them touches working GPU code this suite
  cannot exercise.
- `opregistry.prune()` and `operator_audit.prune()` exist but nothing schedules them;
  same for `_prune_old_jobs` on a timer. No periodic-maintenance mechanism exists yet.
### World numerical cores — CLOSED (same day)
The last third of trust gap #3. 46 tests over `world_collision.classify`,
`world_shell` (`frame_check`, `room_box`, `build_probe`, `rank_key`) and ground-cell
binning. `ground_mesh_build`'s three binning stages moved verbatim into
`mesh/ground_binning.py` (numpy only, no open3d) so they are testable with nothing
stubbed; the receipt is a refactor-equivalence test comparing the extracted module
cell-for-cell against the pre-extraction inline algorithm over 6 randomised clouds.

Two behaviours were **discovered while testing, not assumed**: a high cell sharing a
square with ground is absorbed by the 15th percentile and never reaches spike
rejection (the stages compose), and a cloud too sparse to be a room yields ~no
footprint rather than a thin one that would pass the gates quietly — `build_probe`
requires ~12 points per column at cell=0.1, area-scaled so gates mean the same region
at any `--grid-res`.

- Not attempted: `splat_route.py` split / `jobstore.py`, langfield VRAM halving (needs a
  live GPU run to verify), the `mcp==2.0.0b2` beta pin, lift-crop context margins
  (a quality change that needs measurement, not a guess). `world_shell.evaluate`'s four
  gates still need open3d + trimesh raycasting, so they remain untested — the ranking
  and probe layers around them are now covered.
- **R1–R7 interactive-worlds work is untouched** — advisory, and explicitly lane-opening
  rather than loop-closing. See `~/projects/nexus-planning/05-interactive-worlds-vision.md`.

---

## 2026-07-27 — R2: world interactions and state (branch `r2-world-interactions`)

First rung of the interactive-worlds direction. The world was furniture you could
bump into; now elements can afford something and the world remembers what you did.

**Not a `scene_manifest` v2, and the reason is the HITL — not just which file the
walker reads.** `scene_assemble_approve` rewrites the manifest with
`state: "approved"`, which means a human graded that assembly. Putting authored
interactions in the same document forces a choice between "approved" quietly
degrading to "approved at some point, with unknown edits since", or making someone
re-approve geometry because they labelled a light switch. The slug spaces differ
too: `_regen/` has `ground` and no `shell`; `_world/` has `shell`, no `ground`, and
drops elements that failed to solidify. So interactions are **sidecars in the world
lane**: `_world/interactions.json` + `_world/state.json`.

### Design calls worth keeping
- **`open` is not a verb.** It is `toggle` with the states `closed`/`open`.
  Vocabulary is `inspect` / `toggle` / `pickup`; effect keys are a CLOSED set
  (`visible`, `tint`) and an unknown one is rejected, not ignored.
- **Staleness is per element, never a file stamp.** `world_collision.py` writes
  `seconds: time.time()-t0` through a non-atomic `write_text`, and
  `same_file_identity` compares `mtime_ns`/`inode` while ignoring `sha256` — so any
  identity check reports a change on EVERY rebuild whether or not anything a save
  depends on moved. Gating on it would mean never being able to rebuild a world
  without losing the save. The stamp is advisory (`world_rebuilt`); each entry is
  judged alone. Nineteen survivors are not discarded because one vanished.
- **`job_id` mismatch IS a hard refusal.** `_DUP_SKIP_DIRS` is only `{"versions"}`,
  so job duplication copies `_world/` wholesale — without the check a duplicated
  scene silently inherits its parent's save.
- **Only `tint` is wired, not `visible`.** `object.visible` already has one owner
  (the HUD's per-element eye); a second writer would produce an object the panel
  calls hidden.
- **`E`, raycast from screen centre, 10 Hz prompt.** Under pointer lock the cursor
  does not exist, and the canvas click already means "acquire lock". The prompt
  poll is throttled because spark-scene-viewer measured 10 fps on a per-move
  102 ms raycast. The interact raycasts on demand, so acting is always exact.
  Reach is authored in metres and scaled by `unitsPerMetre`.
- **`fetchManifest` still reads the raw `world_manifest.json`, deliberately.**
  Repointing it at `/world/manifest` would win `collision_shell`/`calibration`/
  `hull_urls`, but `role` is null there before the collision stage runs and the
  walker defaults to `"prop"` — silently turning every static element
  pass-through on an ungraded world. Left for a phase that can verify it.

### Verified
- `pytest backend/tests -q` → **1073 passed, 7 skipped** (was 990/6). Frontend
  `npm run test` → **85 passed** (was 49); `npm run check` → 0 errors, no new
  warnings; `npm run build` clean.
- **Live, against the real bicycle world `splat_3aaf8067`** (services restarted,
  nothing in flight): authored a toggle on `bench-seat` → set it `on` → fresh read
  returned `on` from disk. Undeclared state → 400 listing the legal ones; unknown
  slug → 400 listing the real ones. Simulated a rebuild that lost the element:
  dropped with a reason, `world_rebuilt: true`, not resurrected. `world_manifest.json`
  restored and both sidecars removed afterwards — the job dir is as found.

### Open
- `pickup` validates and persists but is not wired in the walker; the state
  document reserves `player: {}` as its seam (a player-owned container is a second
  state axis).
- No affordance authoring UI — `PUT /world/interactions` is the only door. That is
  R3 (LangField proposing affordances through the propose→gate→approve idiom).

---

## 2026-07-27 — Why the world does not look real yet (measured, not guessed)

### Fixed: the voxel shell was rotated twice and coloured from the wrong places
`build_shell --shell-source voxel` fed `collision_shell.glb` — which
`world_shell.py` writes in Y-up — into `object_texture.py`, which assumes
CAPTURE-frame input and applies the capture→Y-up rotation itself,
**unconditionally** (`object_texture.py:863-866`; only the SCALE depends on
`meters_per_unit`). So the shell was rotated a second time AND every texel's
colour was sampled from the wrong neighbourhood — the hazard
`object_texture.py:460` warns about.

Measured on `splat_3aaf8067`, thin-axis across the tree: capture cloud after
`to_yup` → Y, `collision_shell.glb` → Y, `elements/bicycle.glb` → Y,
`shell.glb` → **Z**. The shell was the only artifact in the wrong frame.
After the fix: `[10.52, 12.56, 7.37]` → `[10.52, 7.37, 12.56]`, and bbox IoU
against the collision solid **0.349 → 0.733**. TSDF path was always correct.

### Disproved: texture size is NOT the blotchiness lever
Sweep on the rebuilt shell, `coverage_rasterized` = fraction of atlas texels
that got a real rasterizer hit rather than dilation:

| texture | coverage_rasterized | build | atlas |
|---|---|---|---|
| 1024 | 0.5995 | 6.5 s | 1.2 MB |
| 2048 | 0.6001 | 18.3 s | 3.4 MB |
| 4096 | 0.6003 | 93.9 s | 9.6 MB |

**Flat at 0.60 across 4× resolution for 14× the cost.** Resolution sharpens the
observed 60% but never shrinks the invented 40%. So raising it does not touch
the smeared look — that was my hypothesis and the data killed it.

### The actual cause, and the lever
40% of a **watertight** shell is surface the capture never saw — the lid, back
and underside a solid must invent to close. Dilation fills it by smearing
neighbours, which is exactly the blotchy olive/white.

`class_textures.py` exists to fill precisely that with procedural material
(grass/dirt/gravel/pavement from the taxonomy's PBR params). It is barely
firing: `class_texel_fractions` = grass 0.002, dirt 0.0001, pavement 0.0009 —
**~0.3% of texels**, because the shell texturing reads `_langfield/class_labels.json`
(hand paints; this scene has 4 records) and nothing else.

**Next lever, highest value for "looks real":** feed `semantic_ground.py`'s
SigLIP auto-classification into the shell bake instead of relying on manual
paints, so the unobserved 40% gets an honest procedural material rather than a
smear. That is a pipeline change, not a tuning knob.

### Also shipped
- **Rebuild shell panel** in the world view (source / faces / texture), calling
  `/world/solidify` with `shell_only`. Surfaces the gate verdict in amber when
  gates fail rather than reporting a flat success. Rebuilding was API-only
  before, which is the loop you are in most while tuning appearance.
- **Fly mode** (`F`, `Space`/`C`) — a world you cannot walk was also a world you
  could not inspect. Runs ahead of the collider branch so it works with no BVH.
- **The walker never used the watertight collision shell.** It reads the raw
  `world_manifest.json`, which has no `collision_shell` key (only the merged
  route synthesizes it), so `csUrl` was always undefined and every world
  collided against the lace visual shell. Now falls back to the conventional
  `_world/collision_shell.glb`.
- **Calibrate now sends the measurement, not the answer.** The Measure tab
  divided client-side and POSTed a bare `meters_per_unit`; it now posts
  `references: [{dimension_id, real_length_m}]` so the server records provenance
  and uncertainty, and prints its own account of the result.

### Note
The backup interlock (`nexus-backup.service` activating) correctly refused a
rebuild mid-session — rails working. The bicycle shell is currently at 4096;
the new panel resets it to 2048 in a click.

### The appearance chain, traced (2026-07-27, later)
Two of my own hypotheses died on the way, which is the useful part.

1. **"The atlas is over-resolved."** Disproved: `coverage_rasterized` is flat at
   0.60 across 1024/2048/4096 for 14x the cost. Resolution sharpens the observed
   60%; it never shrinks the invented 40%.
2. **"The class match radius is too tight."** Disproved: median distance from
   shell surface to nearest classed gaussian is **3.44 scene units** in a ~6-unit
   scene. Radius 0.15 -> 0.9% of surface, 2.0 -> 9.4%. No radius rescues it.
   An auto-scale keyed to voxel size (0.07 -> 0.12) would have been below the
   existing 0.15 default: a no-op dressed as a fix. Written, measured, removed.

**The actual cause: the shell encloses far too much empty space.**

| | centre | size | volume |
|---|---|---|---|
| classed ground gaussians | `[0.05, 0.00, -0.38]` | `[3.69, 4.77, 0.99]` | 17.4 |
| shell (capture frame) | `[-1.40, 0.25, 2.09]` | `[10.52, 12.56, 7.37]` | 974 |

The shell wraps ~56x the ground's volume. Most of its surface is out where the
capture has neither colour nor class, so the bake dilates a smear over it — the
blotchy olive/white. `room_box` uses `_mesh/mesh.ply` p0.1-p99.9 (that file
exists here, 17 MB), which on an outdoor orbit capture still swallows far-field.

What tighter bounds would give, measured on this mesh:

| percentiles | shell size | volume |
|---|---|---|
| p0.1-p99.9 (current) | `[8.44, 9.89, 4.00]` | 334 |
| p1-p99 | `[7.83, 8.66, 3.52]` | 239 |
| p2-p98 | `[7.58, 8.37, 3.33]` | 211 |
| p5-p95 | `[6.76, 7.48, 2.48]` | 125 |

Note the on-disk shell (974) is ~3x even the p0.1-p99.9 box (334), so percentile
tightening is necessary but not sufficient — something downstream of `room_box`
(likely the voxel `close_radius: 2`) inflates it further. **That is the next
thing to measure**, before any tuning.

`--class-radius` is now reachable from scene_solidify for tuning, but the
measurement above says it is a knob, not the fix.

---

## 2026-07-27 — Why it does not look like reality: the cameras are inside the shell

Stopped guessing and scored it. `mesh_gate.py` renders the mesh through real
TRAIN cameras and compares against the actual photos — an executable realism
metric that already existed and was never being used on the world lane.

**Result on `splat_3aaf8067`: PSNR 11.96 dB, SSIM 0.163, coverage 0.835**, against
the recorded lab reference of 17.64 / 0.234 / 0.746.

The gate's side-by-side render explains it instantly, in a way no number did:
the left half is the bench-and-bicycle photo; the right half is a jumble of
white and green slabs at close range. **You are inside the geometry, looking at
its interior walls.**

Verified directly:

```
train cameras   [-0.88, -0.98, -0.27] .. [ 0.89, 1.00, 0.30]
shell bbox      [-6.71, -6.39, -1.45] .. [ 3.81, 6.17, 5.92]
CAMERAS INSIDE THE SHELL BBOX: 175 / 175  (100%)
```

The cameras occupy a ~2-unit region; the shell wraps a ~10-unit box around them.
Rendering from any capture viewpoint therefore looks at the inside of a closed
solid. **A watertight shell is the wrong primitive for an object-centric orbit
capture**: the cameras orbit OUTSIDE the subject, so any solid enclosing the
point cloud necessarily encloses the cameras too.

### Every knob is irrelevant while that holds
A/B on the realism metric, all through the gated route:

| config | PSNR | SSIM | coverage | faces |
|---|---|---|---|---|
| auto, no drop | 11.80 | **0.222** | 1.00 | 24,268 |
| auto + drop-unobserved | 11.96 | 0.163 | 0.835 | 19,040 |
| voxel route + drop | **12.03** | 0.192 | 0.907 | 103,700 |

**0.23 dB across 4x the geometry.** Noise. SSIM is actually best on the original
default. Four hypotheses tested this session — texture resolution, class radius,
unobserved-face dropping, shell-route detail — and none moves realism, because
none addresses the camera containment.

The world was restored to the best-SSIM configuration (auto route, no drop,
24,268 faces): "all measurable gates passed".

### What would actually move it
Not a knob. The world lane needs to stop building an enclosing solid for
object-centric captures and instead produce ground + objects viewed from
outside — which is R1's hybrid rendering (splat as the world skin, mesh only
where interaction demands it) rather than a mesh-everything shell. Detecting
the case is cheap and now has a test: are the train cameras inside the candidate
shell's bbox? If yes, the shell will never render like the capture.

### Also corrected here
`world_gate` graded `shell.glb` — the RENDER mesh — for `shell_connectivity` and
`floor_continuity`, which are walkability questions about the mesh the player
COLLIDES with. Since the walker was fixed to load `collision_shell.glb`, those
gates were measuring the wrong artifact; that is what made `--drop-unobserved`
look like a regression. They now grade the collision solid, fall back to the
visual shell when there is none, and record `measured_mesh` in the verdict.

---

## 2026-07-27 — SESSION CLOSED: the shell approach is a recorded dead end

RToony's verdict on the final screenshots: *"looks like nothing — green splats
that resemble nothing real."* He asked for **an entirely new approach, outside
this session's context**. Do not resume the tuning path.

**The finding: a watertight mesh shell cannot represent an object-centric ORBIT
capture.** 175/175 train cameras sit inside the shell bbox, so every render is a
view of its interior walls. `mesh_gate`: 11.96 dB / SSIM 0.163 vs a 17.64 / 0.234
reference. Four tuning hypotheses were disproved by measurement (texture
resolution flat at 0.60 coverage across 4x; class radius vs a 3.44-unit median
distance; unobserved-face dropping; shell-route detail) — **0.23 dB total spread
across 4x geometry.**

The most promising new direction is probably a **capture-methodology** change:
a walkable world wants a walk-THROUGH capture, not an orbit of an object.

Full detail, including every fixed bug worth keeping and the current blocker
(the uncalibrated scale suggestion assumes an indoor storey height and gives
2.83 u/m outdoors), is in memory:
`~/.claude/projects/-home-rtoony/memory/splatlab-world-appearance-dead-end-2026-07-27.md`
and `splatlab-r2-world-interactions-2026-07-27.md`.

State at close: branch `r2-world-interactions`, 20 commits, UNPUSHED, clean
tree; `main` level with origin; 1089 backend + 85 frontend tests passing; all
three services active; `splat_3aaf8067` shell restored to its gate-passing
config with a demo `interactions.json` still authored on it.

## 2026-07-28 — R3: THE POLISH RETURN-LEG FIRED (first real content through the loop)

Wave branch `r3-polish-return-leg` (off main after the r2 ff-merge+push, RToony-approved).

**G2 (01af54e, 9e40a77):** the restricted workflow gained the ops a real polish needs —
`import_world_element` (host-resolved, containment-checked; Blender never sees a free-form
path), `cleanup_mesh` (weld / decimate / debris-island drop with the largest island always
kept / smooth; receipts carry faces+verts before/after), and selective `export_glb
object_name=` with a suffixed stem so whole-scene exports are never clobbered. MCP :9877
now 11 tools (restarted, verified via tools/list). Real-Blender lane grew a polish-primitives
e2e test (glTF triangulates: factory cube imports as 12 faces — the fake runner can't see that).

**G1 — first-ever polish upload, bonsai `red-bicycle` (splat_aea04ab3):** snapshot v3 →
import v4 → cleanup v5 (8000→7884 faces: exactly the 116-face crumb; weld 6296→4008 verts;
the 502-face island deliberately KEPT — could be real detached geometry) → selective export
`scene-v0005-polish-red-bicycle.glb` (1 mesh, 1 material, **1 image — texture survived the
round-trip**) → POST /world/elements/red-bicycle/polish → 200. Receipts, all verified:
supersedes sha a3e03024 == prior file; `_world/versions/red-bicycle-v0001.glb` created
(first fire — the dir never existed); marker `red-bicycle.polish.json` v1 schema; manifest
`polished` non-null; served sha == uploaded sha d18fca12; audit row
`splat.polish_world_element` 13:45:03Z. Gate: frac 0.9227→**0.9363**, uv/tex TRUE, still PASS.
Rollback: `cp _world/versions/red-bicycle-v0001.glb _world/elements/red-bicycle.glb` + rm marker.

**⚠️ Finding (pre-existing, surfaced — NOT caused by the polish):** re-running world_gate
under the post-r2 code (610e93d grades what you STAND on) fails bonsai on
`shell_connectivity` (27 collision components vs ≤20) and `floor_continuity` (coverage
0.8975 vs ≥0.9). Proof of innocence: the PRE-r2 gate on the SAME post-polish tree fails only
prop_integrity, identical to the stored 07-26 report. The r2 wave only ever re-graded the
bicycle job (splat_3aaf8067). Bonsai's collision shell needs a walkability pass some day —
recorded as an open item, deliberately not chased this wave (not scoped; not appearance work).

## 2026-07-28 — G3: prop_integrity 3/5 → 5/5 (fixed the artifact, not the gate)

Root cause was structural: `place_generated()` shipped SAM-3D props as vertex-coloured
meshes with NO UV/texture (its `tex` parameter was never consumed), so every
--prefer-generated world failed `prop_integrity` by construction. New
`mesh/generated_texture.py` (2667b57) completes the artifact: measured island-drop
(smallest-first, only until largest/total ≥ 0.8, never any component >20% — bonsai
cardboard-box's secondaries all sit INSIDE the main bbox: interior generation shells,
not real parts) → xatlas unwrap → object_texture's k-NN bake fed by the mesh's own
vertex colours → dilate → textured GLB + fresh sidecar (cures the stale-atlas drift).
`--patch-elements` added: edits world.json in place; a bare `--only` run REWRITES it
with just the filtered slugs (the 07-26 drift class). Guarded: requires --only, refuses
--report, needs an existing world.json.

**Live re-derive (18.6 s, backup at `_world/_work/g3-backup-2026-07-28/`):**
cardboard-box frac 0.7107→**0.8151** uv/tex TRUE (6,976 faces; 546+460+18-face interior
fragments dropped, 1,290-face wall KEPT); orange-bike-bottle frac→**0.996** uv/tex TRUE.
Gate: **prop_integrity PASS 5/5**, texture_coverage PASS. Remaining failures are only the
pre-existing r2 walkability items (shell_connectivity 27>20, floor_continuity 0.8975<0.9)
recorded in the entry above. Tests: 6/6 incl. opt-in mesh-env bake e2e (equal-sized
components are refused by the max-single-drop guard — proven by the test's first draft).

## 2026-07-28 — Phase 3: polish recipes (the loop is now a product feature)

`dcc/polish_recipe.py` (81351e0): snapshot → import → cleanup → validated selective
export as ONE typed call, composite receipt, partial receipts preserved on failure.
MCP :9877 grew `run_polish_recipe` (**12 tools**). `tools/polish-element.py` closes the
loop: dry-run by default, `--upload` drives the audited polish route and verifies the
served identity TWICE (route-response sha + independent on-disk rehash vs the export
receipt). Ingestion deliberately stays outside the recipe — the polish route remains
the one audited door.

**Batch applied live (bonsai):** plastic-storage-container → 6283bfff (frac 0.9999),
cardboard-box-2 → 1ba9c3f5 (frac **1.0**, debris fully swept, 2,570 faces). Gate:
prop_integrity 5/5 HOLDS, every texture survived the Blender round-trip.
`_world/versions/` now archives rollback copies for red-bicycle,
plastic-storage-container, cardboard-box-2. Verdict fails only the pre-existing r2
walkability items (open item, recorded above).

## 2026-07-28 — Phase 4 spike: KIRI 3DGS Render — SPLATS are now Blender-editable headless

The research swarm's #1 (Apache-2.0, zero blockers) proven in a ~35-minute spike, far
inside the 90-min box. Blender **5.1.2** + extension `dgs_render_by_kiri_engine` v5.0.0 —
fully isolated from the 4.5.11 studio-loop toolchain (untouched).

**Receipts:** bonsai `_preview/splat.ply` → headless import **605,391 splats in 6.6 s**
(full 3DGS attribute set + 4 KIRI GN modifiers) → uniform-scale edit via the splat-aware
apply-transforms op → PLY export → **605,391 splats out, bbox ratio exactly [2,2,2],
canonical layout intact (SH3, all 45 f_rest coeffs)**. Re-proven end-to-end through the
checked-in driver at 0.5x.

**Two headless gotchas, now encoded in `integrations/blender/kiri_headless_driver.py`:**
(1) the add-on is Serpens-generated — operators live under `bpy.ops.sna.*` with hash
suffixes; (2) export DEFERS through bpy.app.timers and --background quits before timers
tick → `{'FINISHED'}` with **no file** (silent). Fix: monkeypatch timers.register, drain
the chain synchronously (2 ticks).

**⚠️ Open validation item (ledger-gated):** exported `scale_*` shifts 1.621 under 2x
(ln 2 = 0.693) — KIRI may reparametrize gaussian scale on export. Render-compare in Spark
before any user-facing use of EDITED splats. Export also adds harmless extras
(red/green/blue/Col_*/normals/Shadeless) beside the canonical fields.

**What this buys the pipeline:** the Blender loop handled MESHES only; SplatLab can now
programmatically edit the SPLAT itself (crop, attribute-select, paint, animate — the op
surface is scriptable) in a license-clean add-on. Graduating it into the audited
workflow is a deliberate future step, not part of this wave.

## 2026-07-28 — R3 WAVE CLOSED (automode run, ~2.5 h active)

13 commits on `r3-polish-return-leg`, PUSHED — merge to main awaits RToony. Suites:
backend 1102 / frontend 85 / real-Blender 3/3 / mesh-env 6/6, all green. Adversarial
review swarm (3 lenses → skeptic-per-finding): 7 confirmed, 0 refuted, all fixed
(28b1489) — headline: --patch-elements shell-clobber (HIGH), inert colour guard,
glb_check self-containment (security). Return digest:
`~/reports/2026-07-28-splatlab-r3-automode-digest.md`. Memory:
`splatlab-r3-polish-return-leg-2026-07-28`.

## 2026-07-28 pm — CALIBRATION + FRAME FIX + KIRI VALIDATED (RToony back, live session)

RToony's verdict on the walkable world: "Holy Crap. This is MUCH better than what we had
before." Then three follow-ups landed in one burst:

**Banner root-caused:** the viewer's shared-coordinate-frame warning was bonsai's visual
shell.glb — a 07-26 bake predating r2's double-rotation fix (height stored in the wrong
axis). The wave's Blender polishes were proven innocent (0.000 centre drift vs their
versioned originals); the G3 re-derive had actually CORRECTED the generated props (the old
place_generated skipped the Y-up rotation whenever mpu was None — its rotation lived
inside `if mpu:`; floor heights now agree across all props).

**Calibration LIVE:** RToony measured a 32" door (dimension 1785271325762, 0.856 u);
completed via POST /scale references → **meters_per_unit 0.94975, method "dimension",
scale_generation 1**. Sanity: room height 2.68 m, bicycle 1.66 m long.

**Full re-solidify at metre scale** (60 s; backup `_world/_work/pre-recal-backup-2026-07-28/`):
all 5 props + shell rebuilt with correct rotations → **all elements inside the shell AABB
(banner clears), scale_sanity PASS 6/6 (first time ever), prop_integrity 5/5,
texture_coverage 6/6.** The 3 polished elements were re-polished via recipes post-rebuild
(713aff85 / 0a48263d / d1254d6d, double-verified).

**Walkability:** collision rebuild reproduced voxel 27 comps; ONE principled cleanup
(largest-component keep, 5,464 crumb tris across 26 floating islands) → shell_connectivity
**PASS (1 comp, frac 1.0)**. floor_continuity stays marginally red (coverage 0.8968 vs 0.9,
gap 0.0308 vs 0.03) — deliberately NOT tuned further. ⚠️ **Design question for RToony:**
world_shell's own acceptance passes this exact mesh 4/4 (its floor metric reads 1.0) while
world_gate fails it at threshold-noise margins — the two gate systems disagree, and per the
metric-trust doctrine the threshold shouldn't drive work until it matches his lived grades
(he just walked the world happily).

**KIRI scale anomaly RESOLVED analytically:** positions EXACT (0.0 err at 2x, 605k splats);
per-axis scales differ because KIRI canonicalizes ellipsoid axis order with a compensating
rotation — sorted per-splat triples match orig+ln2 exactly for 99.98%. Edited-splat exports
are mathematically equivalent → ledger gate cleared; graduation into the audited workflow
is the open decision.

## 2026-07-28 pm — R4-A: floor metric reconciled → BONSAI IS THE FIRST ALL-GREEN WORLD

The two graders disagreed on the same mesh (world_shell 1.0 vs world_gate 0.8968) because
world_gate demanded every column's LOWEST surface in a ground band — counting standable
furniture over occluded floor as holes (diagnosed: the failing 10% hugged the room
perimeter at desk/shelf height, exactly where cameras never saw beneath). New
`mesh/floor_support.py` is THE measurement, consumed verbatim by BOTH graders: down-rays
from a head-height probe (0.6×span, unit-free); a hit is standable support, a miss is a
fall-through. No ceiling blind spot (rays start below it — the case the old design
existed for, pinned by synthetic tests: intact room 1.0 / hole-under-ceiling caught /
furniture supported / slab-over-hole = support). Old strict number survives as the
`ground_band_coverage` diagnostic. world_shell keeps its probe machinery for
capsule/walkability internals; its ACCEPTED number is now the shared one.

**Bonsai: `VERDICT: PASS — all gates passed`** (floor 0.9779, gap 0.0063; diag band
0.8968). **Bicycle regression: all measurable gates passed** (floor 1.0, unchanged).
Suite 1102 green. Per the metric-trust doctrine this is the direction a metric earns
gating rights: it now agrees with RToony's lived walk.

## 2026-07-28 pm — R4: SPLAT EDITING GRADUATES INTO THE AUDITED WORKFLOW

Branch `r4-splat-edit-lane` (gate reconciliation folded in per RToony's "Both").

**The lane** (c441fbe, 97b04b4): `mesh/splat_edit.py` — typed, attribute-preserving ops
on 3DGS PLYs (crop_box / clean / transform; opacity is a LOGIT, scales are LOGS —
thresholds taken in real units; uniform scale shifts log scales by ln s, the KIRI
validation lesson). Route layer `splat_edit_route.py`: POST /splat/edit lands immutable
`_splat/versions/splat-vNNNN.ply` + receipt + audit row, chaining latest-version-first;
GET /splat/versions; POST /splat/promote replaces the LIVE `_preview/splat.ply` with the
pristine original preserved to `_splat/original.ply` on first promote and a HARD GUARD:
index-keyed consumers (`_langfield/gauss_emb.npz`, `_scene/isolated`) 409 promotion
without force=true — their per-gaussian indices silently desynchronise otherwise.
`tools/splat-edit.py` CLI. Deliberately NOT MCP tools: live-adjacent mutations go
through the audited HTTP door only (polish-upload doctrine).

**Live proof on a real duplicate** (`splat_ccd678ed26`, 2.4 GB copy of bonsai):
clean(min_opacity 0.02, max_scale 2.0, max_dist 20) swept **6,949 debris gaussians** →
v0001; unforced promote correctly **409'd naming both consumers**; forced promote
preserved the original and landed live sha == version sha (2cd368ab). Audit rows:
splat.duplicate / splat.edit / splat.promote, 21:22Z.

## 2026-07-28 pm — R4 review swarm: 8 confirmed findings, 0 refuted — all fixed

Two HIGHs, both proven by execution before fixing:
1. **floor probe keyed off raw bbox span** — one 0.3-unit floater above the roof lifted
   the probe over the ceiling and flipped a holed room FAIL→false-PASS in BOTH graders.
   Fix: probe keys off a robust ceiling estimate (median of per-column tops over the
   interior); floater cases pinned in the synthetic suite. Bonsai stays all-green (0.9631).
2. **promote skipped the splat-mutation invalidation contract** — the viewer serves
   web.ply/langweb.ply/splat.spz preferentially, so promote "succeeded" while the viewer
   rendered PRE-EDIT geometry (proven stale on the live-proof duplicate by mtime). Fix:
   promote now runs edit_ops' regen-or-unlink contract + langfield STALE marker + thumb
   invalidation, warnings in the receipt. Re-promoted the duplicate: all three variants
   rebuilt fresh, zero warnings.
Also fixed: dual-lock (mesh + edit lanes no longer interleave on splat.ply), staged-file
try/finally on promote, status=completed admission (a running training's ns-export writes
splat.ply in place), sha256 hashing off the event loop. Suites: 1108 backend green.

## 2026-07-28 pm — P1: PROPS OBEY PHYSICS (playable-worlds roadmap, phase 1)

Branch `p1-physics`, per the approved macro roadmap
(`~/projects/nexus-planning/06-splatlab-playable-worlds-roadmap.md` — capture → inhabit →
play; flagship: playable game first).

**Architecture (aab639f):** the player keeps the proven capsule-vs-BVH controller; Rapier
(`@dimforge/rapier3d-compat` 0.19.3, WASM, deterministic) owns the PROPS. Each prop is
re-origined (geometry arrives world-space-baked, identity transforms — exporter contract)
and gets convex colliders from its CoACD `UCX_*.glb` hulls via the world file route
(raw-manifest `collision.files`), falling back to a render-mesh hull. Player mirrored as a
kinematic capsule → walking into a box SHOVES it. Props rest on the collision solid; fixed
1/60 accumulator; sleeping props cost nothing; dynamic-vs-`collideProps` mutual exclusion.
Best-effort: WASM/hull failure degrades to display-only with a visible warning.

**Pickup lands (7641a16):** E on an authored `pickup` prop lifts it into a kinematic carry
(hold-point in front of the camera), E puts down, left-click THROWS. Disturbed prop poses
persist via the reserved `player` seam: new `POST /world/player` (validated — finite
floats, ~unit quaternions, bounded count; element-state writes and pose writes never
clobber each other), page saves every 5 s + on teardown, restores on load. Carrying is
deliberately ephemeral. Physics tests run the REAL Rapier WASM in node: drop-settle,
bit-identical determinism, carry/throw, pose round-trip. Suites: frontend 90, backend 1111.

**Live proof (f737475, `tools/prove-physics-live.py`):** real GPU walker on bonsai, teleport
to the prop cluster, walk in — `plastic-storage-container` shoved with a real tumble
quaternion and **persisted server-side** at [1.835, -1.648, 1.016]; screenshots show the
cardboard box knocked over on the floor. First authored pickup record: `cardboard-box-2`.

**⚠️ Noticed (pre-existing, open for P2):** the walker's raw `world_manifest.json` carries
no `meters_per_unit`, so a CALIBRATED capture still walks on the storey-height GUESS
(bonsai: guess 1.032 u/m vs calibrated 1.053 — luckily close). The merged /world/manifest
route has it; the walker deliberately reads the raw file. Fold into P2.

## 2026-07-28 pm — P2: THE LIVING WORLD (branch p2-living-world)

**P2-a scale semantics (bd4bd04):** metre-baked worlds declared their CAPTURE factor as
mpu while their GLBs were metres — every consumer 5% wrong and the walker guessing from
storey height whenever the manifest went stale. Baked worlds now declare mpu 1.0 with
`calibrated_from_meters_per_unit` provenance; patch-mode bakes at the WORLD's factor;
/scale back-stamps scene-unit worlds only. Bonsai migrated live: HUD locked 1.00 u/m,
banner gone, exact-metre 1.5m rule + scale_sanity.

**P2-b regrade (bcdf858):** POST /world/regrade — sticky (regrades.json replayed on every
rebuild), audited, collision regenerated to match. Live: the bicycle regraded to prop
("a bicycle is movable; the 1.5 m heuristic misgraded it"), 8 CoACD hulls, shovable.

**P2-c affordances (66766ed):** POST /world/affordances/propose — rule-driven, explained,
through the SAME validation gate as PUT; authored records never clobbered. Applied live on
bonsai: water bottle + cardboard box → pickup, bicycle/container correctly "too large to
carry (1.66 m)".

**P2-d auto-generate-under-gates (fb50b0c):** captured props that trip a measured floor
(build failed / gate-failing frac / <1000 source gaussians) trigger object_generate during
solidify; used ONLY if its mask/placement gates pass; trigger+outcome recorded either way,
refusals in the gate's own words. Live on the bicycle scene: only the 240-gaussian
fragment triggered; SAM masked it but the alignment gate REFUSED at IoU 0.057 — correctly
declining to hallucinate a bicycle for a junk cluster; captured kept; verdict all-green.
Found+pinned en route: the UV-seam weld trap (frac 0.06 vs gate's 0.93 pre-fix).
**Recorded gap:** the big shard-bicycle passes every measured floor (frac 0.93, 25k
gaussians) while looking wrong — the trigger for THAT needs a render-agreement metric
(the UMI3D watch item), not a guessed threshold.

## 2026-07-28 pm — P3: THE WORLD IS PLAYABLE (branch p3-game-mode)

Zombie waves in the captured bonsai room, end to end: navmesh from the gate-graded
walkable cells (79038f2, + durable collision-crumb policy), the scenario as validated
data (2e3199b — closed vocabulary, bounded stats; PUT {} installs three growing waves),
and the game itself (09267f0): primitive-built shamblers with grid-A* chase, hitscan +
tracer + hit flash, melee + damage vignette, health, rest timers, win/lose + retry.
Left-click arbitrated: combat outranks carry-throw. The whole loop runs REAL and
deterministic under vitest (spawn/chase/melee/aimed-kills/waves/win — 4 tests) and the
live browser proof spawned 3 undead who chased and OVERRAN a blind playwright player at
61 fps. FE 102 / BE 1122 green.

## 2026-07-28 late — W2: SECOND WORLD + WORLD-BUILDING TOOLS (branch w2-world-tools)

**W2-A — the Stump is world #2 (virgin outdoor E2E).** Mip-NeRF360 Stump
(splat_56831407, zero edits): inventory found 5 instances, isolate 1.87M gaussians,
ground tolerated its known partial failure, solidify + gates + navmesh (21,193 walkable
cells) + affordances + scenario. Gate verdict honest: FAILED shell_connectivity only
(2 components, frac 0.9442 vs 0.95 — real disconnected outdoor geometry, not crumbs;
floor PASS, props 5/5 textured). Live playwright proof: 3 undead rose in the forest,
chased, died; 60 fps.

**W2-B — combat polish, one capped commit (c153a30).** Hit stagger, crosshair
hit-marker, spawn-rise telegraph, ±12% size/speed jitter. NO new mechanics, per the cap.

**W2-C1 — one-command world pipeline (8868e65).** POST /jobs/{id}/world/prepare runs
mesh → inventory → isolate → ground → solidify → collision/navmesh → gates →
affordances → scenario under ONE opregistry op, skip-if-exists per stage, per-stage
force, failures resume on re-POST. Surfaces: route + splatlab-make-walkable.sh
(rewritten onto it) + a Make-walkable button on the scene page. Live-proven on the
Stump: second run skipped mesh/inventory/isolate/ground, completed the rest.

**Measured on the way (the reason W2-A existed):** three real outdoor-world defects,
all fixed with tests + live proof: (1) actors floated at the navmesh's single flat
floor_y — ground probes now ask the collider; (2) BOTH naive probe semantics are wrong
on a voxel terrain slab (first-hit = canopy/shell roof — the old spawn bug — and
lowest-hit = slab underside); groundAt(x,z,belowY) returns the highest face at or below
the walkable band; (3) unlit worlds rendered actors pitch black — the game carries its
own light, deferring to any visible scene light. Plus: legless zombies read as hovering
→ legs; uncalibrated captures proposed picking up a tree stump → no size claims without
calibration (inspect + honest rationale).

**Wave review (8877948): 31 findings, 20 confirmed, all fixed.** Four adversarial
lenses + one skeptic per finding. The two that mattered most: `/world/prepare` could
report a forced ground rebuild as partial success when a stale ply from an older run
existed (a 503 before any work read as 200 ok), and a second `opregistry.finish` on the
failure path erased the stage+detail the registry exists to hold. The probe rule itself
was re-derived from measurement, not argument: sampling 600 walkable columns of the
Stump collider, "nearest face to the graded floor" beat "highest below a cap" (|error|
p50 0.15 / p95 0.76 vs naive-highest's p95 2.87) and removed the canopy tail entirely
(reproduce with `tools/measure-ground-probe.py <job> 600 <u/m>`). Test-honesty findings were
taken as seriously as the bugs — the terrain test's mock had been asserting behaviour the
real probe can never produce, and `walker.groundAt` (the wave's actual fix) had no direct
test; it now has 7 against a real MeshBVH. ⚠️ A review subagent edited the working tree
mid-run and reverted a committed P3 endgame fix; caught by `git status`, restored from
HEAD. Review agents must be read-only.

**W2-C2 — restyle (7a1025b).** The decorating / dungeon-ifying starter: a validated
`_world/restyle.json` applied OVER the capture — per-element tint, per-element taxonomy
class material (procedural tile from the SAME `class_taxonomy.json` the bakes read,
projected triplanarly in world space because each element's UVs are its own atlas chart
packing), and one scene lighting preset (as-captured / noon / golden-hour / overcast /
dusk / night / dungeon × intensity). Reversibility is measured, not asserted:
`tools/prove-restyle-live.py` compares the frame before / restyled / reset — the restyle
moved it 53.5 mean RGB and reset returned it to **0.0**. Found live and fixed: a restyle
you cannot see is not a restyle — the splat backdrop is the photograph (it cannot be
re-surfaced or relit, and it hides the shell it stands in for), so an active restyle now
shows the reconstructed geometry instead. 7 backend + 11 frontend tests.

**Wave totals:** backend 1140 green, frontend 134 green. Branch `w2-world-tools` pushed;
merge is RToony's call.

---

## 2026-07-28 — W3 Lane A: the restyle BAKES (fantasy looks survive export)

**Fantasy taxonomy (e14a40d).** Seven restyle-only surface classes — cobblestone,
mossy-stone, obsidian, lava, sandstone, timber-planks, thatch — as full generative
recipes in `class_taxonomy.json` (v2), category `"fantasy"` so they can never enter the
SigLIP ground lane (no queries; semantic_ground filters `category=="ground"`). The
preview dropdown, `tile_for`, and the bake all read the same numbers by construction.

**The baker (892b62e).** `mesh/restyle_bake.py` writes the walker's preview into the
element's OWN atlas, in the GLB's frame (no capture-frame inversion — deliberate):
triplanar planes X→(z,y)/Y→(x,z)/Z→(x,y) via `class_textures.sample_world_triplanar`,
`texels_per_unit = tile_px/(material_scale_m×unitsPerMetre)`, tint multiplied in LINEAR
space (what `material.color` actually does). Material bakes build a FRESH atlas so chart
gutters carry the new surface; painted texels floor-clamp to 1 so `texture_coverage`
never counts them as background. Geometry passes through behind a readback face-count
gate. All-or-nothing into staging.

**The landing (a6f70e8).** `POST /world/restyle/bake` (+`/revert`): mesh lock +
opregistry + audit; priors versioned as GLB+atlas PAIRS at one `-vNNNN` seq; marker
`<slug>.restyle.json` with `atlas_updated: true` (the opposite of polish's
`atlas_superseded` — the fresh sidecar DOES describe the file); `restyle_bake.json`
archives the full pre-bake overlay; the overlay clears (lighting mood preserved —
lighting is never baked, the atlas already carries the capture's light). Manifest gains
`restyled` beside `polished`; UE-bundle provenance = polished|restyled|captured-derived,
newer marker wins.

**The walker rule (ef6ba1d).** After a bake the overlay is empty, so the splat backdrop
— still the un-restyled photograph — would hide the baked shell. `setBakedLook` folds a
`baked_elements` signal (on the restyle payload) into `restyleShowsMesh`. RestylePanel:
armed "Bake this look" / "Revert bake".

**Live proof (`tools/prove-bake-live.py`, splat_aea04ab3, 5090 ANGLE/GL):** preview
moved the frame **28.2** mean RGB; the bake held **28.6** with the overlay cleared;
**preview↔bake agreed within 0.4**; revert restored the capture to **0.0**. Post-bake
object truth: 0 restyle-shader meshes of 6; the bicycle's tint lives in its atlas
(material colour ffffff) — and that bicycle was a POLISHED GLB (v0003), so Blender UV
charts bake correctly. Also proven: a fantasy class (cobblestone) end-to-end.

**Suites:** backend 1156 green (+16), frontend 136 green (+2), tsc clean.

---

## 2026-07-28 — W3 Lane B: fantasy set-dressing (authored assets enter the world)

**The registry (0882a73).** Nothing could ADD an object to a world — four refusal
points, and every rebuild erased anything not capture-derived. `world_placed.py` is the
sticky-regrades pattern for placed assets: `_world/placed.json`, written only by the
audited door, re-applied by `world_collision` every run (captured slugs win loudly via
`authored_conflicts`; missing GLBs mark unbuilt; authored props get real CoACD hulls on
the next run). The non-atomic `world_manifest.json` write is fixed (staged + os.replace).

**The frame gate (c85a009).** `glb_check.position_bounds`: POSITION min/max union +
node-transform identity (own or inherited), stdlib-only — the shared-frame contract,
checkable at the door.

**The door (2fdc32e).** `POST /world/elements` is polish's mirror image: it INVENTS and
refuses to replace (slug existing anywhere → 409). Same discipline: .building-* stage,
glb_check + position_bounds before anything live, mesh lock, `supersedes: null` receipt,
audit. Node-TRS props are 422ed; statics warn. DELETE tombstones to `_world/versions/`,
authored-only. One append point buys walker / world_slugs / affordances / restyle /
interactions / HUD.

**The library (808b109).** 3 committed self-validating starters (torch-sconce 0.44 m,
treasure-chest 0.48 m, wooden-signpost 1.10 m) from a deterministic Blender script:
metre-scale, origin bottom-centre, ONE joined mesh, identity nodes, <20 KB each.

**The authoring loop (a0f2cc8).** Blender MCP 12 → 14 tools: `list_asset_library`
(broken files reported, never hidden), `import_asset` (containment-checked against
assets/library; ONE-mesh contract; reports dimensions in metres), and
`export_blend_glb(bake_world_transform=True)` — matrix_world into the vertices, node
identitied, parented objects refused, in-memory only. **(fca6b60)** PlacedPanel in the
walker: rows + armed Remove + place-new via the house upload zone.

**Live proof (`tools/prove-dressing-live.py`, splat_aea04ab3, 5090):** all 8 receipts —
authored torch exported at the commanded floor point (x=0.30 z=-1.20 y=floor, identity);
landed 200/authored; same-slug re-POST **409 live**; visible (3 mesh primitives + DOM);
authored toggle tints it #ffd27f in the walker; state survives reload; a REAL
`world_collision` re-run preserves the entry (counts.authored=1); round-trips back into
Blender (276 faces); removed with tombstone, pixel delta 6.8. Found live and fixed: the
centered "Click to walk" card can cover a small asset — pixel receipts now measure an
overlay-free band.

**Suites:** backend 1195 green (+39 this lane), frontend 136 green, tsc clean.
`prove-game-live` (Stump) and `prove-restyle-live` re-run green — nothing regressed.

---

## 2026-07-29 — PW A9: LIVE RECEIPTS — Truck game proof GREEN, courtyard wall-fill CAUGHT A REAL GAP

Phase-0 handoff executed (Kimi review pass → unconfined re-run). Suites first:
**backend 1256 green** (+61 since W3), **frontend 139 green** (+3, see below), serve
restarted at HEAD with /scene/surfaces + tunables + pluck stage.

**PROOF 1 — Truck (splat_716a9122) game loop: GREEN, after finding two real bugs.**
The 07-29 timeout was NOT flakiness. Root cause chain, proven with a live walker dump
(`window.__worldWalker`): the uncalibrated scale guess (1.9245 u/m) made the 1.7 m
player 3.27 units tall inside the probe's 2.27-unit headroom; respawn clamped the EYE
but the physics capsule kept full height, its feet sat ~1.9 u inside the floor solid,
and ONE depenetration pass hurled the spawn from the proven seed (0.42, 1.04, -0.46)
to (2.12, -9.97, 8.46) — an eject/fall/respawn loop (ground=air forever), which the
game read as "no reachable spawn ground" and refused to start. **Fix (3005c6c):**
`capsuleHeight` clamps the capsule extent (height + radius) into the probe headroom;
respawn stands feet-on-the-proven-floor at that height. Contract test fails 2/12 with
the clamp disabled; worlds with real headroom byte-identical.
Second bug, in the proof tool itself: its receipts (kills/HP/OVERRUN) were read from
the FINAL frame, after the endgame dialog had vanished — and its own blind fight-clicks
are what dismissed the dialog ("kills 0 absent" then counted as a kill). The tool now
accumulates evidence DURING the fight and stops shooting the moment an endgame dialog
appears. Honest receipts this run: **wave 1 spawns 3 undead, ground=on at the seed,
HP 100→10 real damage, 60 fps, exit 0.**
Found, not yet fixed (cosmetic): game notices render under the "Shared-coordinate-frame
check flagged N element(s)" banner title (world-view.tsx wires game.onNotice into the
same warnings list) — the mislabel sent this debug down a frame-check rabbit hole.

**PROOF 2 — courtyard (splat_5c1db781) wall fill: pipeline lands, UNION DOESN'T — a
real, caught defect.** After two wall-args corrections (calibrated 1.0 m/u lives in
_world/world.json, NOT meta.json's stale 8.58; anchor the wall to room_box_scene, not
a raw-band quantile that painted 4,920 orbit floaters OUTSIDE the +x wall — that junk
record was deleted via DELETE /class-labels, receipt 8243944b): **PAINT 5,327 gaussians
→ 1 planar patch (rms 0.02508, 183 inliers, 0.365 u²) → GENERATIVE provenance tag
verified in the ply header.** Then `/world/prepare {force:[solidify]}` ran solidify
'done' — and **no artifact records the union** (shell surfaces_merged=None, collision
surface_patches_sampled absent). Why, precisely:
1. Both live proof worlds win EXTERIOR collision routes (Truck `exterior/faces`,
   courtyard `exterior/smooth+carve`); patch sampling only exists in the voxel
   candidate (`candidates[3].surface_patches_sampled` — present, loses the vote).
2. scene_solidify's voxel shell path REUSES an existing collision_shell.glb
   (route=auto + file exists → no rebuild), so a solid built at 06:07 — before the
   paint existed — was re-textured into shell.glb at 16:28 untouched by the patch.
3. `cut=None` on the voxel route means surfaces_merged can never exist there either
   (it's a tsdf/mesh-route receipt from build_shell_mesh).
**Decision needed (RToony):** where should painted patches reach the collision solid —
sample them in every route candidate, force the voxel route when patches exist, or
post-union patch voxels into whichever solid wins? Plus the staleness rule: solidify
must rebuild collision when a patch postdates it. Until then the A8 wall-fill is
UNREACHABLE on every current world, and prove-surfaces-live's 4c receipt stays red.

**Artifacts:** proof screenshots /tmp/splat_716a9122-paint-rescue-proof/ (01-wave1,
02-fight); courtyard receipts _scene/surfaces/ (patch_00.ply + receipt_top/oblique).
The corrected WALL_ARGS + mpu-source math lives in the phase0 handoff script.

---

## 2026-07-29 pm — PLUCK CONSUMER: a disturbed prop takes its photograph ghost with it

The walker now consumes `_world/pluck.json`. When physics marks a prop disturbed
(shove, carry, restored save — rapier's own one-way `disturbed` flag), its
backdrop-splat rows fade to zero through a dyno worldModifier (`RgbaArray` mask
read per splat index, `buildPluckModifier` in spark-heatmap.ts). world-view fetches
GET /world/pluck FIRST and upgrades the backdrop to **fmt=langweb** (row-identical;
the server falls back to the raw ply) only when the doc is fresh — fmt=web is
decimated and would address the wrong rows; the walker additionally refuses to
pluck on any row-count mismatch. One-way by design: the frame loop re-plucks a
still-disturbed prop, so a moved prop's ghost can never return on its own.

**The dead lane, measured so nobody walks it again:** mutating
`packedSplats.packedArray` after load is a visual no-op — the pipeline never
re-reads it. Δ0.00 through every dirty-flag combo (packed.needsUpdate,
source.needsUpdate, mesh.updateVersion, mesh.needsUpdate); the SAME mask through
the worldModifier lane wiped the whole photograph (Δ103.8). Also: `worldModifier`
is a setter-only property — reading it back is always undefined.

**Live receipts (`tools/prove-pluck-live.py`, Truck, 161 MB raw backdrop):** doc
fresh, 7 props mapped (pickup 27,712 rows); physics-disturbed umbrella
auto-plucked WITHOUT being asked; the yellow-umbrella ghost visibly vanished
(band Δ2.21 vs 0.00 measured drift, dead-still fly-cam, mesh eye-hidden so the
ghost stands alone); unpluck restored the bytes (opacity 1.000) and the frame
loop re-plucked the still-disturbed prop on its own. The pickup auto-plucks at
load (its body wakes on settle) — aggressive but correct under the contract.

**CAUGHT by the same proof on the courtyard (splat_5c1db781): its isolate claims
predate the 07-28 calibration.** `object_indices` select a 4,015-splat blob 9 cm
wide at (-0.62,-0.22,-0.07) — the table at 1/9.577 scale, the OLD normalized
frame — while the real table sits at (-6.1..-5.0). The index-map chain is
self-consistent (map verified point-exact, object.ply agrees with the indices),
so `world_pluck`'s freshness verdict CANNOT see the staleness, and re-running the
isolate stage reproduces it (the claims themselves are stale). **Decision needed
(RToony):** re-claim the courtyard props against the calibrated cloud, or teach
calibration to migrate isolate claims (scale transform + row remap) — and either
way the pluck freshness verdict should compare isolate claim revisions against
the checkpoint revision, not just artifact identities.

Frontend 142 green (3 new pluck tests), tsc clean. Suites + game proof re-run
green via `~/scripts/splatlab-pw-verify.sh` (10/10).

---

## 2026-07-30 — LANE 6 / E1+E2: THE ENVIRONMENT ROLE — authored geometry is real floor

RToony's redirect made concrete: worlds are built deliberately. New role
`environment` = authored, world-frame-baked, walked-on architecture (terrain
skirts, walls, platforms) through the existing create door.

**Backend (E1, ac21b18):** ROLES gains "environment"; the door 422s it on node
transforms like props (the recorded AABB must match the walked-on collider);
origin-distance warning relaxes to 4x the shell half-diagonal for environment
only; collision manifest records complex_as_simple with an honest
classification + counts.environment (also recomputed AT THE DOOR — an
environment piece is countable the moment it lands); affordances skip it with
"walked on, not interacted with". Captured elements can never become
environment. Backend 1259 green (+3).

**Walker (E2):** `buildStaticColliderGeometry()` — ONE builder shared by the
player BVH and Rapier so they can never disagree. Fast path = collision shell
PLUS authored static/environment placements (captured statics stay represented
by the shell and never double in); colliderSource gains
"collision_shell+authored" so the HUD says what the BVH contains. This also
ends the era of authored statics reporting collides=true while contributing
zero triangles (the pre-existing fast-path lie). enablePhysics uses the same
builder with props excluded — a prop cannot be baked-static AND dynamic.
Frontend 147 green (+5: merge rules, captured-static pin, surfaceNear finds an
authored platform as real floor, real-Rapier prop settles ON the platform).

**Live proof (`tools/prove-environment-live.py`, Truck):** baked 2x0.3x2
platform through the door as environment → walker loads it collides=true,
colliderSource=collision_shell+authored → the player DROPS ONTO IT and stands
at eye y=1.86 (platform top 0.28 + clamped capsule 1.575; captured floor would
be 1.26, the drop height was 2.60 — the first cut of this receipt read a stale
grounded flag at the drop height and false-passed; settled-stand polling fixed
it) → DELETE returns the collider to the bare shell, tris -12 exact.

---

## 2026-07-30 — LANE 6 / E3: THE CURTAIN — where the photograph ends

A sparse capture has a good core and a fringe of orbit floaters; the curtain
is the boundary. `_world/curtain.json` (dev.splatlab.world-curtain/v1 —
closed vocabulary, bounded finite numbers, fail-loud validator, GET/PUT
cloned from the scenario pair with the same invalid-stored-doc degradation)
describes a sphere/box in walker-frame scene units; the walker renders it as
ONE Spark SplatEdit SDF layer (opacity 0 x MULTIPLY x invert; softEdge = the
fade band; WHITE so the band fades, never tints). Scene-parented, NOT under
the backdrop — the backdrop carries -90degX + m/u scale and scene-parenting
keeps authored params in plain scene units. Runs at the world-space edit
stage BEFORE worldModifiers, so it composes with pluck without touching that
slot. Param changes mutate the same objects (no generator rebuild); the
curtain is removed on world clear so it can never leak across loads.
CurtainPanel in the walker: enable, shape, size + soft-edge sliders,
Center-on-me; live apply + debounced PUT.

**Live receipts (`tools/prove-curtain-live.py`, Truck, dead-still fly-cam,
drift 0.00):** tight sphere around the spawn -> OUTSIDE band faded Δ63.99
mean-RGB while the INSIDE band moved Δ5.49 — a 12:1 two-sided verdict that
this is a spatial boundary, not global dimming. Fresh page load reinstalls
from curtain.json (curtainState round-trip + outside Δ65.22 vs baseline).

Backend 1262 green (+3), frontend 149 green (+2), tsc clean.

---

## 2026-07-30 — LANE 6 / E4: CC0 FANTASY LIBRARY — 30 pieces, license-bookkept

`assets/library/intake_pack.py` (pinned Blender, headless): imports a pack
piece (glTF/GLB), refuses ARMATUREs (static environment only — rigged actors
wait for the actor lane), strips to geometry, joins to ONE mesh, applies
transforms, re-origins bottom-centre, exports Y-up with textures embedded,
and self-validates against the exact starter contract (glb_check structure,
single mesh, identity transforms, floor within 5 mm, 0.05–30 m extent). A
failed piece aborts loudly. Every export lands in
`assets/library/catalog.json` (dev.splatlab.asset-catalog/v1: pack,
source_url, license, license_url, imported_at) and `list_asset_library`
attaches the catalog entry per asset — a corrupt catalog is reported as
`catalog_error`, never swallowed.

**First batch, committed (KayKit, CC0, pulled scriptably from GitHub):**
18 dungeon pieces (walls incl. doorway/arched/broken/pillar/window, floor
tiles, stairs, column, torch, banner, chest, barrel) + 12 graveyard pieces
(arch-gate, crypt, fences, graves, gravestone, lanterns, shrine, dead tree)
— 30 assets, 20–75 KB each, palette-textured. The old 3 m "human-scale"
ceiling in the library contract test widened to the intake's 30 m band: the
library now holds architecture, not just hand props.

**Kenney Castle/Graveyard + Quaternius (RToony's other two picks):** their
downloads are browser-gated (no scrapeable zip URL, no API) — the intake
tool takes their zips unchanged whenever one click lands them in a dir:
`blender --background --python assets/library/intake_pack.py -- --pack ...
--source-dir <unzipped dir> --pieces ...`. Noted, not blocked on.

Backend 1294 green (+32: 30 auto-parametrized contract tests + 2 catalog).

---

## 2026-07-30 — LANE 6 / E5: THE INTEGRATION RECEIPT — a fantasy scene on a splat foundation

**`tools/prove-fantasy-live.py` (Truck), all receipts landed:** three CC0
library pieces authored through the REAL Blender loop (import_asset ->
transform_object -> export bake_world_transform) and landed through the door
as role=environment (raised dungeon platform, doorway wall, graveyard fence;
counts.environment=3); the walker builds collision_shell+authored (HUD:
SOLID+AUTHORED); the player drops onto authored stone and STANDS at eye
y=3.30 vs the captured floor's 1.26; the curtain wraps the capture core
(sphere r=2.5) while the fantasy geometry past the seam remains; the zombie
scenario still runs unchanged on the captured ground (min_hp=10, 60 fps);
cleanup tombstones everything and the manifest returns to
counts.environment=0. Prereq discovered en route: the Blender loop needs the
job's scene assembly — POST /scene/assemble {mode: faithful} built it for
the Truck.

**Composition note (not a defect):** KayKit metre-scale architecture lands
visually HUGE against the Truck's miniature scene units (1 u = 0.52 m) — the
demo pass wants intake --scale or per-piece transform tuning. The receipts
stand regardless.

`splatlab-pw-verify.sh` gains step 6/7: both lane-6 proofs (self-cleaning)
run in the safe legs.

Lane 6 layer 1 is COMPLETE: E1 role -> E2 collider -> E3 curtain -> E4
library -> E5 integration, five commits, each gated green.

---

## 2026-07-30 — TRIAGE LANE OPENS: photograph until you touch it, mesh once it's yours

Lane 6 merged to main (fast-forward 333ddc2 -> f3a013a, branch pruned). The
triage lane's first slice ships the redirect's quick win — STOP DRAWING MUSH.

**The rule (walker `defaultElementVisible` + `refreshElementVisibility`):**
while the splat backdrop is the visual, a CAPTURED element's mesh is a crude
tracing drawn over its own photograph — so captured meshes now default to
HIDDEN. An element earns its mesh by being: authored (the mesh is all it
has), plucked (its photograph ghost is gone — the mesh IS it now), restyled
(a look needs geometry to live on), or genuinely interactive (toggle verbs;
inspect-only prompts don't earn a blob). No backdrop, or a restyle showing
the mesh world: everything visible. Eye-toggles override everything, both
directions, world-scoped. All the old shell/restyle visibility juggling now
routes through the one computed rule (applyRestyle, setBackdrop,
clearBackdrop, setBakedLook, pluck transitions, setInteractions).

**Live receipts (Bicycle — the world that prompted the redirect):** all 5
captured meshes hidden behind the photograph on load (the crumpled-blob
scene is now a clean photograph), eye-toggle brings any of them back on
demand, collider untouched at SOLID 332.4k tris.

Frontend 154 green (+5), tsc clean. Backend untouched (1294).

---

## 2026-07-30 — TRIAGE SLICE 2: REPLACE-WITH-ASSET — the fire-hydrant pattern, two clicks

**One call replaces a captured element with a curated library asset.**
`POST /world/elements/{captured}/replace?asset=<name>`: the asset (metre
scale, bottom-centre origin — the library contract doing its job) is
uniform-scaled to the captured element's height and translated to its floor
point entirely server-side (`glb_transform.py` — stdlib POSITION-accessor
surgery, no Blender round-trip; sparse/external-buffer GLBs refused loudly),
then lands through the same registry/manifest path as the door carrying
`replaces: <captured>`. One replacement per captured element; authored
elements refuse; deleting the replacement frees everything.

**The walker acts on the link with no hand-holding:** the captured mesh can
never default visible while replaced (outranks plucked/interactive earns),
and its photograph ghost is plucked unconditionally the moment rows exist —
the authored asset IS the object now. `GET /assets/library` feeds the new
per-element "Swap" control in the ElementsPanel (captured, non-shell rows):
pick an asset, Replace, world reloads.

**Live receipts (`tools/prove-replace-live.py`, Truck):** dungeon-torch ->
umbrella at scale 0.327, standing at the umbrella's floor point (centre
within 0.3u); umbrella mesh hidden + ghost auto-plucked via the 4 Hz tick;
DELETE + reload restores the photograph untouched — perfectly reversible.

Backend 1298 green (+4), frontend 156 green (+2), tsc clean.

---

## 2026-07-30 — TRIAGE SLICE 3: PROPOSE-GENERATED — the gates speak, the human decides

The third treatment. `auto_generate` (solidify) auto-APPLIES; this lane never
decides. New `backend/generate_route.py`:

- `POST .../objects/{slug}/generate/propose` — runs SAM-3D **under the GPU
  arbiter** (a lease the solidify path never took — a 13 GB CUDA load now
  serializes like every other generative lane), own process group so a
  timeout kills the CUDA grandchild. The candidate lands in
  `_regen/objects/<slug>/`; NOTHING is applied; a gate refusal is a
  first-class verdict, not an error.
- `GET .../generate/candidate` + `GET .../generate/file?slug&fmt` — the
  review payload and the previously-unreachable candidate artifacts
  (preview/alignment/mesh/report), no-cache.
- `POST .../generate/promote` — the HITL apply: place_generated (mesh env:
  capture-frame transform + decimate to 8k + bake) into a staged GLB, then
  the prior element GLB+atlas pair is VERSIONED with exact filenames in a
  `<slug>.generated.json` marker; world.json reads geometry_source=generated.
- `POST .../generate/revert` — restores the exact versioned pair;
  `DELETE .../generate/candidate` — discard (refused while promoted).

Walker UI: a "Gen" zone per captured element — propose (with the ~2-4 min
GPU honesty label), verdict chips (PLACED / mask IoU), preview thumbnail,
Promote / Discard / Re-propose / Revert.

**Live receipts (`tools/prove-generated-live.py`, fire-hydrant — the
operator's own reference case):** fresh arbitrated SAM-3D run -> PLACED,
mask IoU 0.8869; preview served no-cache; PROMOTE: element became the
generated hydrant (8,000 faces, sha changed, prior versioned as
fire-hydrant-v0001.glb, world.json geometry_source=generated); REVERT:
**byte-exact restore** (sha 85ebfe10 both sides), marker gone. Found and
fixed en route: the mesh-env promote snippet passed a str where
_generated_candidate needs a Path; the report's IoU lives at
mask_alignment_gate.iou_vs_captured_object (all readers aligned).

All three triage treatments now exist: Photo-only (default) / Show mesh /
Replace-with-asset / Propose-generated — deliberate, per-object, reversible.

Backend 1302 green (+4), frontend 156 green, tsc clean.

## 2026-07-31 — FIRST OWN CAPTURE WALKABLE + the resilience overhaul it forced

RToony's first deliberate INSV capture (splat_f9f3b4fed4, storage room, 4920 frames —
the densest dataset this pipeline has ever seen) is WALKABLE: health GOOD (registration
1.0), voxel shell 52,294 faces (connectivity gate PASS), navmesh live, floor gate
report-only at 76.5% standable (the rest is real shelving). Six attempts to get there;
every failure bought a permanent fix, all on main, all deployed:

- Three-layer lease-death root cause: asyncio heartbeat starvation (5980fa0, thread
  heartbeat) → 30.6G uint8 cache vs 32G VRAM (fe0a5e6, cache-images cpu + TTL 180s) →
  TRUE cause: service cgroup MemoryHigh=32G quicksand, 35M throttle hits (raised to
  64G/80G in 60-safety-guard.conf — the canonical file, survives daemon-reload).
- New standing armor: train memory preflight (c9322e0), coordination-death flight
  recorder (185a12b), wait_backup_idle stage queueing (c198802), VRAM fence (f6add9a —
  first production kill same day: evicted qwen3.6:35b loaded mid-langfield), POST
  /jobs/{id}/resume with artifact-verified prefix skip (526b0a9, used twice same day),
  instance-free worlds via solidify --allow-empty-inventory (this scene's 12 candidates
  were ALL honestly vetoed; shell+ground worlds are legitimate).
- Suite-isolation fix (b5ef7c2): the long-recorded "transient pytest hang" was tests
  blocking on the LIVE GPU lock whenever a real job ran. Suite green mid-job now (1333).
- Dense-scene knobs discovered: LANGFIELD_DEPTH_TOL=0.15 (drop-in), ground
  semantic_thresh 0.35, voxel shell route for cluttered interiors. Langfield is now
  ~2h/40% of the pipeline at this density — frame subsampling is the top optimization.
- Ops: nexus-throttle-watch (5-min cgroup/PSI watchdog) + nexus-backup-reconciler
  (daily 12:15) timers live; redis-key-forensics.sh staged as a reusable tool.

Batch PARKED by RToony (system stays his): `bash ~/scripts/splatlab-hero-queue.sh --run`
= capture-2@30k → capture-1@100k → capture-2@100k → walk Test Flight. 100k-vs-60k
decision deferred until 30k/100k results compare side by side.

## 2026-09-14 — RESEARCH SWEEP (6 months of GS/radiance-field work) → first wave staged

Report + receipts: `~/reports/2026-09-14-splatlab-research-sweep/` (README = the approved plan; triage
receipts; commit lists). Catalog: `research/sources.json` grew 22 → 52 candidates, licenses verified via
the GitHub API, `research/tests` 17 passed.

- **Method:** 1,914 papers since 2026-03-14 (Semantic Scholar) → 1,645 abstracts scored against a
  SplatLab-specific rubric by deepseek-v4-flash ($0.44); Kimi validation sample of 300 (ρ = 0.889, 97 %
  within 2 points); GLM unusable. Fable verified the top ~80 by hand (code, license, README).
- **Version audit was the cheapest win:** gsplat 1.5.3 already has 3DGUT/fisheye/2DGS/MCMC; nerfstudio
  1.1.5 already exposes bilateral-grid / antialiased / camera-optimizer / absgrad / scale-reg — the train
  command uses none of them; MoGe-2 (metric depth, MIT) was already installed; Spark 2.2.0 was out.
- **Done today:** Spark 2.1.0 → 2.2.0 (vitest 339, tsc, build, live game proof 61 fps on the Stump).
  `backend/scale_estimate.py` + `tools/moge2-scale.py` (MoGe-2 metric-scale *proposal*, never writes
  meta.json; CPU dry-run green on all 8 worlds). `backend/health/render_views.py` + `scoda_gate.py` +
  `tools/scoda-gate.py` (SCODA, arXiv 2609.07346, MIT, no-reference render-agreement score; cloned at
  `~/tools/research/scoda`). Suites: backend **2392 passed / 17 skipped**, frontend 339, tsc clean.
- **Staged for RToony (dry-run default, rollback printed):** `~/scripts/splatlab-wip-commit-2026-09-14.sh`
  (8 topical commits for the 255 uncommitted paths — the Aug–Sep snapshot + today), `splatlab-moge2-scale-…`
  (gate: bonsai within ±5 % of 0.94975 m/u, then the 7 uncalibrated worlds), `splatlab-scoda-gate-…`
  (must rank splat_aea04ab3 above splat_3aaf8067), `splatlab-flag-ab-…` (5 splatfacto flags, 2 worlds,
  PSNR/LPIPS + fog + mesh LCC, ~3 h).
- **Next waves (his pick):** R5.1 isolate-by-reference (Seed2GS idea: SAM3 + frozen splat, seconds
  instead of the 2 h language field), R3.2 plane-scaffold architecture layer (TopoGS idea, the condo's
  missing geometry + shell connectivity), R2.2 FullCircle (Apache-2.0, raw dual-fisheye + 3DGRT,
  operator-robust) vs the 12-view rig, R2.1 VGGT-SLAM 2.0 rung. Watch: TurboGS (~100 s/train, code
  pending), VidSplat, GSCompleter, LightBridge.

## 2026-09-14 pm — first wave RAN (RToony): commit/push done, MoGe gate FAILED honestly, SCODA = no separation

- **Committed + pushed** (`splatlab-wip-commit-2026-09-14.sh --apply --push`): 8 commits d9038eb…71646a4,
  `origin/main` = 71646a4, tree clean. Six weeks of prior-session work is now on record.
- **R1 MoGe-2 gate FAILED, twice, for two different reasons — both recorded:**
  1. Raw run: 0.2518 m/unit vs measured 0.94975 → −73 %. Cause: **frame mismatch**, not depth error.
     `ns-export gaussian-splat` writes raw `model.means`, so the viewer PLY (and every calibration measured in
     it) is in nerfstudio's normalised frame = COLMAP units × `dataparser_transforms.json` scale (0.2211 here).
     MoGe compares against COLMAP-frame sparse depth. Fixed: the proposal now carries both frames
     (`meters_per_unit` = viewer frame, `meters_per_unit_colmap_frame`, `dataparser_scale`); `--rescore`
     rebuilds a receipt on CPU.
  2. Frame-corrected: **1.1386 m/unit vs 0.94975 → +19.9 %**, 12/12 frames used, relative MAD 9 %.
     Two candidate causes, neither proven: MoGe-2 zero-shot metric bias (its papers report 10–20 %), or the
     single click-measured reference (n = 1, no uncertainty). Staged fix for the known edge bias (SfM
     features sit on edges where nearest-pixel depth reads the background): `--window 3` local-minimum
     sampling + depth maps cached as `_scale/depth_*.npz` so variants rescore without the GPU. Re-gate with
     `! bash ~/scripts/splatlab-moge2-scale-2026-09-14.sh --apply` (27 s). Independent tie-breaker: a second
     measured dimension in the bonsai world via the Measure tab.
- **R4.1 SCODA: NEGATIVE RESULT as designed.** Renders (8 eval cams + ±12° yaw variants, 640 px) scored on
  the GPU env in ~45 s/world (the CPU venv needed >5 min/scene — runner switched to langfield-spike).
  Means: bonsai −283.3 vs bicycle −285.7 (higher = better) → "correct" by 2.4 on a within-world spread of
  ~150; **pairwise P(good view > bad view) = 0.50, Q_F 0.36, Q_R 0.45 — no separation.** SCODA scores a
  render against *its own scene's* photo manifold, so absolute values are not comparable across worlds;
  my cross-world gate design was wrong, the tool is fine. What it can still do: rank views *within* a
  world (bonsai eval max −198 vs novel min −343) — a per-viewpoint "off-manifold" signal for the walker.
  Cross-world render agreement should come from **held-out `ns-eval`** (PSNR/LPIPS vs the never-trained
  eval photos), which SplatLab never recorded — staged as `~/scripts/splatlab-eval-agreement-2026-09-14.sh`.
- Suites after the changes: backend green (see receipt in the sweep report), 19 new tests.

## 2026-09-14 late — MoGe edge fix refuted, held-out eval SEPARATES the pair, flag bundle HURTS, eval stage LIVE

- **MoGe-2 re-gate with `--window 3`**: 1.1361 m/u (+19.6 %) vs 1.1386 nearest-pixel — the edge-sampling
  hypothesis moved it 0.3 %, refuted. The +20 % stands as MoGe-2 zero-shot bias vs one click-measured
  reference; only a second measured dimension can arbitrate. Receipt `_scale/moge2-proposal.json` (+ v1, v2).
- **Held-out render agreement (ns-eval on the never-trained eval split) does what SCODA could not:**
  bonsai `splat_aea04ab3` **31.55 dB / SSIM 0.936 / LPIPS 0.143**; bicycle `splat_3aaf8067` ("looks
  wrong, passes every gate") **22.93 dB / 0.716 / 0.173**. Receipts `<job>/_health/eval.json`.
  ⚠️ ns-eval needs `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (nerfstudio 1.1.5 bare `torch.load` vs torch ≥ 2.6);
  the pipeline already exports it, the two new scripts now do.
- **`eval` is now a pipeline stage** (report-only, best-effort, after `health`; kill-switch
  `SPLAT_EVAL_GATE=0`; `EVAL_VRAM_MB = 8000`): `backend/splat_route.py` `_append_health_stage` + runner
  branch → `meta["health"]["eval"] = {psnr, psnr_std, ssim, lpips, checkpoint, checked_at, enforced: false}`.
  6 bookkeeping tests (`test_eval_stage_bookkeeping.py`), golden plan pins `_eval_available` off. Suite
  **2402 passed / 17 skipped**. Service restarted 20:04 (no jobs in flight; GPU coordinator is a separate
  process, pid 4157430) — back in 1 s. `splatlab-where-am-i.sh` now prints held-out PSNR + MoGe proposals.
- **Flag A/B on the bonsai, 15k iters (baseline trains in 2.5 min, 10 ms/iter — the 37-min estimate was the
  storage room's):** all five flags together = **PSNR 30.69 → 26.39, SSIM 0.932 → 0.844, LPIPS 0.150 → 0.161,
  mesh LCC 55.9 % → 73.9 %**. Confounded: the bilateral grid absorbs per-photo exposure (raw renders score low
  unless `color-corrected-metrics` is on), the camera optimizer refines train poses but not eval poses, and
  `use-absgrad` is already the 1.1.5 default (no-op). Single-flag ablation launched (`--ablate`, `--arms=`).
  **Bilateral grid arm ABORTED at 46 %: 58 → 213 ms/iter (6–20× baseline), 28 min remaining — cost alone
  rules it out on this stack.** antialiased / camopt / scalereg arms running; table appended when done.
- Experiment tree: `~/projects/splatcli/outputs/experiments/flag-ab-2026-09-14/` (rollback = delete it).

### Single-flag ablation result (bonsai, 15k iters, held-out eval split, one run each)

| arm | PSNR | SSIM | LPIPS | fog | TSDF LCC % |
|---|---|---|---|---|---|
| baseline (today's train command) | 30.69 | 0.932 | 0.150 | HEALTHY | 55.9 |
| `rasterize-mode antialiased` | **30.93** | **0.933** | **0.149** | HEALTHY | 37.2 |
| `camera-optimizer SO3xR3` | 28.19 | 0.880 | 0.162 | HEALTHY | 41.4 |
| `use-scale-regularization` | 30.63 | 0.931 | 0.153 | HEALTHY | **60.7** |
| all five together | 26.39 | 0.844 | 0.161 | HEALTHY | 73.9 |
| bilateral grid | aborted at 46 % (58 → 213 ms/iter) | | | | |

- **antialiased** is a free +0.24 dB (no training cost) — adopt for photometrics.
- **scale-regularization** is PSNR-neutral and +4.8 LCC points — adopt for mesh-bound worlds.
- **camera optimizer** costs 2.5 dB on held-out views; the metric is unfair to it (eval poses are not
  refined) but there is no evidence of benefit either — leave off unless the rig lane shows otherwise.
- **bilateral grid**: ruled out on cost. **absgrad**: already the default.
- ⚠️ TSDF LCC swings 37–74 % across arms from one run each — not stable evidence; the bundle's 73.9 % is
  driven by camopt and/or bilateral, not by the two flags worth adopting. Do not read LCC as a gate here.
- Not changed in production yet: the plan says "flags that win on **both** worlds". Storage-room arms
  (must run from a plain terminal, ~40 min each):
  `! bash ~/scripts/splatlab-flag-ab-2026-09-14.sh --apply --world=splat_f9f3b4fed4 --arms=baseline,antialiased,scalereg`

## 2026-09-15 — storage-room flag run STUCK: the compute gate still carried the July 32G guard

`tools/splatlab-compute-gate.sh` wraps manual runs in a transient scope with `MemoryHigh=32G /
MemoryMax=48G`. The 2026-07-31 fix raised splatlab.service and splatlab.slice to 64G/80G but not this
path. The storage-room baseline arm (4920 frames, ~63 GB cache) sat for an hour at 37 GB against a
34 GB ceiling: `memory.events high 380,681`, PSI some 85 %, swap at its 8G cap, trainer in D state, GPU
1 %. Fixed: gate scope now 64G/80G. Recovery: `~/scripts/splatlab-unstick-and-rerun.sh` (kills only a
scope with throttle events AND idle GPU, then relaunches). The pipeline's train memory preflight
(c9322e0) does not cover manual runs — a follow-up if manual trains keep growing.

### Storage-room confirmation (splat_f9f3b4fed4, 15k iters, held-out eval; run 2026-09-15 06:36–07:00 after the guard fix)

| arm | PSNR | SSIM | LPIPS | fog | TSDF LCC % |
|---|---|---|---|---|---|
| baseline | 18.05 | 0.752 | 0.637 | HEALTHY | 95.8 |
| `rasterize-mode antialiased` | 18.05 | 0.752 | 0.638 | HEALTHY | 95.9 |
| `use-scale-regularization` | 18.04 | 0.751 | 0.639 | HEALTHY | (see receipt) |

- Both flags are **neutral** on the dense walk-through (Δ ≤ 0.01 dB). With the bonsai: antialiased is the
  only flag with a measured win (+0.24 dB) and no loss anywhere; scale-reg is photometrically neutral on
  both worlds (its +4.8 LCC on the bonsai is single-run noise-level).
- Under the raised guard the storage room trained 15k in **2 min at 8.7 ms/iter** (cache 2 min) — the
  earlier hour was entirely the throttle. 18 dB / 180k gaussians at 15k is far from the 30k production
  fit; only the relative comparison is meaningful here.
- **Not flipped in production yet — one check first.** gsplat's `antialiased` mode applies an opacity
  compensation at render time; `ns-export` writes raw opacities, so SparkJS (which does not apply it)
  may render an antialiased-trained splat differently from nerfstudio. The fog gate and `render_views.py`
  already rasterize with `antialiased`, so the *health* path is consistent — the walker is the open
  question. Next: export both bonsai arms' PLYs and compare them in Spark (mean-RGB / alpha coverage
  delta, `prove-*` pattern) before changing `backend/splat_route.py`'s train command. The golden plan
  snapshot (`test_360_stitch.py`) pins the train command and will need the same one-line update.
- Compute-gate scope guard 32G/48G → 64G/80G (`tools/splatlab-compute-gate.sh`) is in commit d.

## 2026-09-15 — Spark side-by-side PASSED → `antialiased` adopted in the production train command

**The check** (`tools/prove-spark-agreement.py`, runner `~/scripts/splatlab-spark-agreement-2026-09-15.sh`):
export both bonsai arms (`ns-export`), render 8 held-out cameras in each arm's *own* rasterize mode
(`render_views.py --rasterize-mode auto`, which now records c2w + intrinsics), then render the same
cameras in a standalone SparkJS page that mirrors `spark-scene-viewer.tsx` exactly (SplatMesh PLY, no
flips, position / up / lookAt(forward), fov from fy) and score vs the real photo and vs nerfstudio.

| arm | Spark vs photo | nerfstudio vs photo | Spark vs nerfstudio | mean abs Δ /255 |
|---|---|---|---|---|
| baseline (classic) | 28.27 dB | 27.72 dB | 30.08 dB | 4.5 |
| antialiased | **28.68 dB** | 29.67 dB | **37.19 dB** | 2.1 |

- The feared opacity-compensation mismatch runs the *other* way: Spark reproduces antialiased-trained
  splats far more faithfully (37 vs 30 dB) — it applies the same kind of 2D anti-aliasing itself. In the
  walker, antialiased is +0.41 dB against photos, better on 7 of 8 cameras. Contact sheet + receipt:
  `~/reports/2026-09-14-splatlab-research-sweep/spark-agreement/`.
- **Adopted:** `--pipeline.model.rasterize-mode antialiased` in `backend/splat_route.py`'s train command
  (`_train_rasterize_mode()`; kill-switch `SPLAT_TRAIN_RASTERIZE_MODE=classic`). Golden plan snapshot
  updated with the flag. Applies to new training jobs after the next service restart.
- Evidence chain for this one flag: bonsai held-out +0.24 dB · storage room neutral · zero cost · Spark
  faithful. scale-reg stays off (neutral), camera optimizer off (−2.5 dB), bilateral grid out (20× slower).
- Trap hit on the way: Spark's ESM imports `three/addons/postprocessing/Pass.js` — a standalone page
  needs `"three/addons/": "/node_modules/three/examples/jsm/"` in its import map or the module dies
  silently before any hook is defined. The proof now reports page/console errors instead of timing out.
- The compute gate refuses heavy work while `nexus-backup.service` runs (07:00–07:3x daily; it mirrors
  splat outputs to the NAS, so experiment trees make it longer). `systemctl is-active` reports
  `activating` as inactive — wait on `ActiveState` instead.

## 2026-09-15 — R5.1 ISOLATE-BY-REFERENCE: first end-to-end, 61 s per object vs the 2 h language field

**What was built** (`backend/isolate/`, `tools/isolate-by-reference.py`, runner
`~/scripts/splatlab-isolate-by-reference-2026-09-15.sh`; Seed2GS idea, independently implemented):
1. **ground** — 6 evenly spaced training PHOTOS staged at render size → the existing `scene_sam3_masks.py`
   text grounding (`sam3` env). Reference = highest-score instance with a plausible mask fraction,
   discounted ×0.3 when its box touches the image edge (cut-off object).
2. **orbits** (`langfield-spike` env) — render expected depth at the reference camera, back-project the
   mask, voxel-select gaussians (2× the scene's median NN spacing, dilated) and keep those projecting
   into the dilated mask → **seed** (8.5k bicycle / 17.9k bonsai). 65 candidate cameras (8 azimuths ×
   4 elevations × 2 dolly rings) scored by seed visibility (seed-only vs full-scene expected depth);
   keep ≥ 0.3, top up to 8, cap 16, ordered as a nearest-camera path. Each kept view also writes the
   seed's own silhouette.
3. **track** (`sam3` env, SAM 3.1 multiplex predictor request API) — PER-FRAME prompts: the seed
   silhouette's box (+4 %) plus the concept text; keep the returned mask that best overlaps the
   silhouette (≥ 0.2 IoU). 16/16 frames found on both objects. Propagation-based tracking was tried
   first and lost the object on 12/16 frames: orbit views are not a video. `predictor.shutdown()` is
   mandatory (a worker keeps ~12 GB otherwise → claim 14 GB so the coordinator evicts residents).
4. **fit** — one foreground logit per gaussian (seed +1, rest −1), rendered as a colour through gsplat and
   fitted by BCE against the masks over valid frames (60 Adam steps, weak seed prior), threshold 0.5,
   then **prune**: behind-the-front-surface in ≥ 80 % of views (packed rasterization, 8 % depth tol),
   needles > 10 spacings, and the instance_lift silhouette vote (< 60 % inside over ≥ 3 views).
   Writes `<job>/_isolate/<slug>/object_indices.npz` (checkpoint order) + `object.ply` (batch_isolate's
   14-field layout) + `receipt.json` + `receipt_object.png` (full | object-only | baseline-only).

**Results on the bonsai job (`splat_aea04ab3`)**

| concept | seed → object | seed retained | mask IoU (mean/min) | frames | wall | verdict (receipt_object.png) |
|---|---|---|---|---|---|---|
| bonsai | 17,877 → 17,591 (prune −3,135) | 0.935 | 0.84 / 0.60 | 16/16 | **61 s** (20+11+17+12) | clean bonsai + pot + tray, two small residual blobs |
| red bicycle | 8,530 → 8,826 (prune −4,262) | 0.879 | 0.86 / 0.66 | 16/16 | 42 s from orbits | clean **rear wheel only** — every staged photo shows the bike cut off; the reference is partial, so the object is |

- vs the existing `_scene/isolated/red-bicycle` (instance_lift over training views, 6,715 gaussians):
  IoU 0.10 — not because either is wrong but because they hold different parts (mine the wheel, the
  baseline the frame tubes). There is no bonsai instance in the inventory to compare against.
- **Limits recorded:** the object is whatever the reference shows (next step: re-ground the text prompt
  on a pulled-back render around the seed so a full-object reference exists even when no photo has one);
  floaters hovering inside the object's volume survive every silhouette test (the antialiased training
  adopted today reduces them upstream); the fit uses one image-space channel, so thin far-side geometry
  can be lost.
- Tests: `backend/tests/test_isolate_orbits.py` (10: projection round-trip, voxel select, orbit
  geometry, view selection, path order, border penalty, visibility, prompt points, PLY subset).
  GPU steps are gated and run in their own envs; nothing new in the FastAPI venv.

## 2026-09-15 — ISOLATE-BY-REFERENCE RE-GROUNDS ON A PULLED-BACK RENDER + opt-in route

The object used to be whatever the best photo showed (red bicycle = one rear wheel). Fixed in three
measured steps on `splat_aea04ab3` (progress sheet, receipts and the 12-candidate contact sheets in
`~/reports/2026-09-14-splatlab-research-sweep/isolate-by-reference/`):

- **Re-ground stage** (`render_orbits.py --stage seed|orbit`, orchestrator steps `ground → seed → reground →
  orbits → track → fit`): after the photo seed, render 12 pulled-back views that frame the seed
  (3 azimuths × 2 elevations × 2 distances; framing distance from the photo mask's pixel extent, never the
  seed radius), run `scene_sam3_masks.py` on them, pick the instance that CONTAINS the seed silhouette and
  grows it the most — quality = score × containment × √growth, ×0.6 if it touches the frame edge, score
  floor 0.5 — then re-lift the seed from that render's depth and orbit from that camera. Rejected if the
  new seed keeps < 40 % of the initial one (different object) → falls back to the photo seed, receipt says why.
  The first picker (score × containment, hard ×0.3 border penalty) chose a rear-triangle mask over the
  whole bike two rows later on the candidate sheet; the growth term fixed that.
- **Depth-banded seed lift:** a SAM3 mask over a see-through object (spokes, leaves) contains pixels whose
  rendered depth is the wall behind it. Lifting only pixels within ±30 % of the mask's median depth removed
  the wall gaussians: the bonsai seed's 95th-percentile radius had been 2.65 units at a 1.2-unit reference
  distance (a 13-unit pull-back where SAM3 saw nothing); the bonsai reference distance itself moved
  1.21 → 0.91 once the centroid stopped being dragged toward the wall.
- **Results** (66 s per object, SAM3 twice + tracker): red bicycle 8,826 → **29,603 gaussians = the whole
  bike** (handlebars, bottle, both wheels), mask IoU 0.85 (min 0.73), recall vs the language-field instance
  0.21 → 0.93 (precision 0.21 — that instance holds 6,715 frame-tube gaussians only). Bonsai 17,591 →
  16,269 with IoU 0.84 → **0.89** (min 0.57 → 0.85): re-ground engages, same object, cleaner seed.
- **Route:** `POST /api/splat/jobs/{job_id}/isolate/reference {concept, views=6, iters=60, threshold=0.5,
  reground=true}` beside `/scene/isolate` — same lock/arbiter/audit pattern (lane `isolate-reference`,
  14 GB), runs `tools/isolate-by-reference.py --no-gate` because the route's lease is the claim; writes
  `meta.isolate_reference[slug]`; `GET …/isolate/reference/file?slug=&fmt=report|object|indices|receipt`.
  5 route tests (`test_isolate_reference_route.py`), 14 orbit tests. Nothing in the viewer yet.
- **Still open:** fog/floaters inside the object's volume (antialiased training reduces them upstream);
  the bicycle's see-through wheels still lift some wall — the fit's behind-front + vote prune removes most
  (3,194 + 5,099 gaussians this run). Pull-backs that leave the capture hull render fog; the candidate
  sheet shows SAM3 simply finds nothing there and the fallback holds.
- Backups of the one-pass artifacts: `<job>/_isolate/{red-bicycle,bonsai}.before-reground/`.

## 2026-09-15 — R3.2 ARCHITECTURE SCAFFOLD on the condo exterior: planes + openings + an independent scale check

`backend/architecture/scaffold_core.py` (8 tests) + `tools/condo-moge-depth.py` (MoGe-2 per view, scaled by each
view's own SfM tracks, gated GPU, 8 s) + `tools/architecture-scaffold.py` (planes, up/Manhattan frame, plumb walls,
track anchoring, coplanar merge, depth-free ray-cast openings, gap rectangles, reserved-track checks, overlays, PLY,
receipt). Report: `~/reports/2026-09-15-splatlab-condo-architecture-scaffold.md`. Wrapper:
`bash ~/scripts/splatlab-architecture-scaffold-2026-09-15.sh --apply`.

- Measured dead ends first: the 3,000-step 2DGS surfel normals are ~random (4 % within 12° of the pavement plane);
  the model's rendered depth fragments walls into layers 20–60 cm apart. MoGe-2 depth scaled by the tracks is
  view-consistent: street wall = one plane, 64k points, RMS 9.7 cm, anchor shift 0.000, reserved tracks 11 cm.
- **Independent scale:** MoGe-2 → 1.343 m/u vs the Condo Lab alignment's 1.357 m/u (from the photo-estimated
  4.8768 × 2.1336 m garage opening): ratio 0.990. Wall normal 1.9° from the model's −Y, up 2.8° from +Z.
- Openings, multi-view (5 views, spread ≤ 0.12 m): 2286 door leaf 4.56 × 2.31 m (alignment scale) / 4.51 × 2.29 m
  (MoGe scale) vs the model's 16 × 7 ft; upper-wall windows 1.16 × 1.89 and 1.05 × 1.89 m; street-wall window
  1.15 × 2.02 m — the model lists door/window sizes as an uncertainty, these are the first capture-derived values.
- Not measured: the 2286 wall length (the plane spans all coplanar units). Nothing accepted; registration provisional.
