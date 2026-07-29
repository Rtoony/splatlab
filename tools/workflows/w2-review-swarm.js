// Adversarial wave review — the shape used on W2 (2026-07-28): 4 find lenses,
// one skeptic per finding, default-refute. 31 raised, 20 confirmed, all fixed.
//
//   Workflow({scriptPath: "tools/workflows/w2-review-swarm.js"})
//
// Point DIFF at the wave's diff first (git diff <base>..HEAD > /tmp/wave.patch)
// and rewrite CONTEXT for the wave under review — the baseline-facts paragraph
// matters most: it stops reviewers re-reporting accepted design decisions.
//
// RAIL: reviewers must be READ-ONLY. On this run one subagent edited the working
// tree and reverted a committed fix. ALWAYS `git status` after the swarm.

export const meta = {
  name: 'w2-review-swarm',
  description: 'Adversarial review of the W2 world-tools wave diff',
  phases: [
    { title: 'Find', detail: '4 lenses over the wave diff' },
    { title: 'Verify', detail: 'one skeptic per finding' },
  ],
}

const DIFF = '/tmp/claude-1000/-home-rtoony/8780afb2-d9ba-4d00-b04b-407d0e8a96a3/scratchpad/w2-diff.patch'
const REPO = '/home/rtoony/projects/splatlab'

const CONTEXT = `You are reviewing one wave of changes on branch w2-world-tools of ${REPO}.
Read the diff at ${DIFF} first, then open the touched files in the repo for full context.
The wave contains: (1) combat polish in frontend/src/lib/world-game.ts (stagger, hit-marker
callback, spawn-rise, jitter, legs, self-carried game light, terrain ground probe);
(2) POST /jobs/{id}/world/prepare orchestrator in backend/splat_route.py + rewritten
~/scripts/splatlab-make-walkable.sh; (3) walker.groundAt(x,z,belowY) highest-hit-below-cap
probe + findStandingPoint change in frontend/src/lib/world-walker.ts; (4) uncalibrated-world
affordance guard in backend/world_affordances.py; (5) tests in backend/tests/test_world_prepare.py
and frontend/src/lib/world-game.test.ts. Baseline facts you must NOT re-report (already known,
accepted): scale is a user-dialed guess on uncalibrated captures; Stump shell_connectivity gate
fails honestly at frac 0.9442 (2 components); the navmesh is single-storey flat floor_y.`

const FINDINGS_SCHEMA = {
  type: 'object', required: ['findings'],
  properties: { findings: { type: 'array', items: {
    type: 'object', required: ['title', 'file', 'claim', 'severity'],
    properties: {
      title: { type: 'string' }, file: { type: 'string' },
      line_hint: { type: 'string' },
      claim: { type: 'string', description: 'the defect + concrete failure scenario' },
      severity: { enum: ['high', 'medium', 'low'] },
    } } } },
}
const VERDICT_SCHEMA = {
  type: 'object', required: ['isReal', 'reason'],
  properties: { isReal: { type: 'boolean' }, reason: { type: 'string' } },
}

const LENSES = [
  { key: 'correctness', prompt: `${CONTEXT}\n\nLens: CORRECTNESS. Hunt real bugs: wrong math (ground probe cap semantics, easing, jitter bounds), broken state machines (game phases, stage skip/force logic in /world/prepare), unit errors (metres vs scene units — this codebase has a history of double-calibration bugs), off-by-one in stage resume. Only report defects with a concrete failure scenario. Return findings.` },
  { key: 'contract', prompt: `${CONTEXT}\n\nLens: API/CONTRACT + concurrency. /world/prepare: op registry lifecycle (op left open on exceptions?), per-stage force list validation, lock discipline (per-job _mesh_export_lock held across GPU waits?), partial-ground tolerance masking real failures, re-POST resume semantics, HTTP codes. The shell script's parsing of the response. Race: two concurrent prepare POSTs on one job. Return findings.` },
  { key: 'frontend-state', prompt: `${CONTEXT}\n\nLens: FRONTEND runtime state. WorldGame: gameLight lifecycle across start/stop/dispose (leaked lights in scene?), sceneHasLight traverse cost per hud tick, groundY easing vs rise animation interaction, legs added but parts not disposed?, removeZombie/clearActors disposing new leg geometry, __worldWalker global leaking across page unmounts, groundAt called before collider ready. Return findings.` },
  { key: 'test-honesty', prompt: `${CONTEXT}\n\nLens: TEST HONESTY. Do the new tests actually pin the behavior they claim? world-game.test.ts terrain test tolerances (0.35 — does it actually catch the old floorY bug?), probe-cap assertion, light test hierarchy check, test_world_prepare.py mocks vs real handler behavior (does the skip-check test actually exercise _exists?), missing coverage: force-list stages, groundAt belowY = -Infinity edge, uncalibrated affordance path via /world/prepare. Return findings.` },
]

phase('Find')
const results = await pipeline(
  LENSES,
  l => agent(l.prompt, { label: `find:${l.key}`, phase: 'Find', schema: FINDINGS_SCHEMA }),
  (r, l) => parallel((r?.findings || []).map(f => () =>
    agent(`${CONTEXT}\n\nA reviewer claims (lens ${l.key}): [${f.severity}] ${f.title} in ${f.file} ${f.line_hint || ''}: ${f.claim}\n\nAdversarially VERIFY by reading the actual current code in ${REPO}. Try hard to REFUTE it: is the scenario actually reachable in the current code? Is it already handled elsewhere? Is it baseline-accepted behavior? Default isReal=false unless you can trace a concrete reachable failure.`,
      { label: `verify:${f.title.slice(0, 30)}`, phase: 'Verify', schema: VERDICT_SCHEMA })
      .then(v => ({ ...f, lens: l.key, verdict: v }))))
)
const all = results.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter(f => f.verdict?.isReal)
log(`${all.length} raw findings, ${confirmed.length} confirmed`)
return { confirmed, refuted: all.filter(f => !f.verdict?.isReal).map(f => ({ title: f.title, reason: f.verdict?.reason })) }