// Walkable-world interaction logic, kept as pure functions on purpose.
//
// Why a separate module: the walker's input layer lives inside WorldWalker,
// which owns a WebGL context and a physics loop — nothing in it is reachable
// from a test, and this project has no DOM test runner (vitest, pure-function
// tests only). Same reasoning as viewer-shortcuts.ts. So the decisions —
// which state comes next, what the prompt says, what a saved state still means
// after a rebuild — live here and are executable-checkable; the walker becomes
// a thin dispatch over the result.
//
// Mirrors backend/world_interactions.py. The server is authoritative; this is
// the optimistic local half, and `resolveState` deliberately reproduces the
// server's drop rules so the two cannot disagree about what a save means.

/** The closed vocabulary. `open` is a toggle with the states closed/open. */
export type Verb = "inspect" | "toggle" | "pickup";

/** Closed effect set — an unknown key is rejected server-side. */
export type Effect = {
  visible?: boolean;
  tint?: string | null;
};

export type InteractionRecord = {
  slug: string;
  verb: Verb;
  prompt: string;
  states: string[];
  initial: string;
  effects: Record<string, Effect>;
  range_m: number;
  text?: string;
};

export type ResolvedState = {
  applied: Record<string, string>;
  dropped: { slug: string; saved: string; reason: string }[];
  world_rebuilt: boolean;
};

/** What the walker hands the page when the crosshair is on something. */
export type TargetInfo = {
  slug: string;
  prompt: string;
  verb: Verb;
  state: string;
  /** Null for a read-only verb — nothing advances. */
  nextState: string | null;
  text?: string;
};

export const INTERACT_KEY = "KeyE";

/**
 * The state one interaction advances to.
 *
 * Stable on a single-state record (an `inspect` that only ever becomes "seen"),
 * and recovers to the authored initial when the current value is not one this
 * record declares — which happens when authoring changed under a live save.
 */
export function nextState(record: InteractionRecord, current: string | null): string {
  const index = current === null ? -1 : record.states.indexOf(current);
  if (index < 0) return record.initial;
  return record.states[(index + 1) % record.states.length];
}

/** The effect for a state, or an empty effect when none is authored. */
export function effectFor(record: InteractionRecord, state: string): Effect {
  return record.effects?.[state] ?? {};
}

/**
 * The HUD line. Deliberately says what pressing the key DOES, not what the
 * element currently is — "Turn on the lamp" beats "the lamp (off)".
 */
export function promptFor(record: InteractionRecord, state: string): string {
  if (record.verb === "inspect") return `Look at ${record.prompt}`;
  if (record.verb === "pickup") {
    return state === record.states[0] ? `Pick up ${record.prompt}` : `Put down ${record.prompt}`;
  }
  const next = nextState(record, state);
  // A two-state toggle reads naturally as "make it <next>".
  return `Turn ${next} ${record.prompt}`;
}

/** Whether this verb changes anything when acted on. */
export function isStateful(record: InteractionRecord): boolean {
  return record.states.length > 1;
}

export function targetInfo(record: InteractionRecord, state: string): TargetInfo {
  return {
    slug: record.slug,
    prompt: promptFor(record, state),
    verb: record.verb,
    state,
    nextState: isStateful(record) ? nextState(record, state) : null,
    ...(record.text ? { text: record.text } : {}),
  };
}

/**
 * Resolve a saved state against what the world currently contains.
 *
 * Reproduces backend/world_interactions.resolve_state: per-element survival,
 * never wholesale invalidation. A rebuild rewrites the world manifest every
 * time, so "the world changed" can never mean "your save is void" — the worst
 * stale entry here is a door that is shut again.
 *
 * `presentSlugs` is null on the local dev-static path, where there is no world
 * manifest to check against; the presence rule is then simply not applied.
 */
export function resolveState(
  records: InteractionRecord[],
  saved: Record<string, string> | null | undefined,
  presentSlugs: Set<string> | null = null,
): ResolvedState {
  const byslug = new Map(records.map((record) => [record.slug, record]));
  const applied: Record<string, string> = {};
  const dropped: ResolvedState["dropped"] = [];

  for (const record of records) {
    if (presentSlugs && !presentSlugs.has(record.slug)) continue;
    applied[record.slug] = record.initial;
  }

  for (const [slug, value] of Object.entries(saved ?? {})) {
    const record = byslug.get(slug);
    if (!record) {
      dropped.push({ slug, saved: value, reason: "no interaction is authored for this element" });
      continue;
    }
    if (presentSlugs && !presentSlugs.has(slug)) {
      dropped.push({ slug, saved: value, reason: "element is not in the rebuilt world" });
      delete applied[slug];
      continue;
    }
    if (!record.states.includes(value)) {
      dropped.push({ slug, saved: value, reason: `state '${value}' is no longer authored` });
      continue;
    }
    applied[slug] = value;
  }

  return { applied, dropped, world_rebuilt: false };
}

/**
 * Reach in SCENE UNITS for a raycast.
 *
 * Authored ranges are metres because that is the only unit that means the same
 * thing on two captures; the walker converts with the same unitsPerMetre it
 * uses for eye height and capsule radius. On an uncalibrated scene the dial is
 * the user's own, and this stays consistent with how they set it.
 */
export function reachInSceneUnits(record: InteractionRecord, unitsPerMetre: number): number {
  const scale = Number.isFinite(unitsPerMetre) && unitsPerMetre > 0 ? unitsPerMetre : 1;
  return record.range_m * scale;
}

/** The largest authored reach — one raycast can then serve every candidate. */
export function maxReachInSceneUnits(
  records: InteractionRecord[],
  unitsPerMetre: number,
): number {
  return records.reduce((max, r) => Math.max(max, reachInSceneUnits(r, unitsPerMetre)), 0);
}
