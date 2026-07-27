// Guards for the paint commit step.
//
// Why these exist: a paint pass meant to label ground CLASSES ended up pinning
// 186,710 splats of flat ground to a field label typed "bikr" — and the
// operator didn't register that he'd committed anything at all. Three things
// let that through, and each has a function here:
//   1. the commit button said "Commit 186,710" instead of naming the act;
//   2. nothing described WHAT was being labelled (a 0.04-thick ground slab
//      called "bike" is self-evidently wrong once you can see it);
//   3. "bikr" was one keystroke from "bike", already in the scene inventory.
//
// Pure on purpose — this repo has no DOM test runner, so the judgement lives
// where vitest can reach it.

/** Levenshtein, bailing out as soon as it can't beat `max`. */
export function editDistance(a: string, b: string, max = 3): number {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    const row = [i];
    let best = i;
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      row[j] = Math.min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost);
      if (row[j] < best) best = row[j];
    }
    if (best > max) return max + 1;
    prev = row;
  }
  return prev[b.length];
}

/**
 * The closest known label to `input`, or null when it's already known, too
 * short to guess at, or nothing is close enough. Deliberately conservative:
 * a wrong suggestion is worse than none, because the operator will trust it.
 */
export function suggestLabel(input: string, known: string[], maxDistance = 2): string | null {
  const q = input.trim().toLowerCase();
  if (q.length < 3) return null;
  let best: string | null = null;
  let bestD = maxDistance + 1;
  for (const raw of known) {
    const k = raw.trim().toLowerCase();
    if (!k) continue;
    if (k === q) return null; // already a real label — nothing to suggest
    // Short labels need a tighter bound or "bike"->"bikes"->"dirt" all collide.
    const limit = Math.min(maxDistance, Math.max(1, Math.floor(k.length / 3)));
    const d = editDistance(q, k, limit);
    if (d <= limit && d < bestD) {
      bestD = d;
      best = raw;
    }
  }
  return best;
}

export type SelectionShape = {
  /** true when one axis is far thinner than the others — a slab, not an object. */
  flat: boolean;
  /** "flat patch 1.30 × 1.54 × 0.04 m" */
  text: string;
};

/**
 * Describe what a selection actually IS, from its robust extent. Flatness is
 * the tell that separates "I painted the ground" from "I painted the bicycle",
 * and it is exactly the check that was missing.
 */
export function describeSelection(
  extent: [number, number, number],
  metersPerUnit: number | null,
): SelectionShape {
  const scale = metersPerUnit ?? 1;
  const unit = metersPerUnit ? "m" : "u";
  const scaled = extent.map((v) => v * scale) as [number, number, number];
  const sorted = [...scaled].sort((a, b) => b - a);
  // Thinnest axis under 15% of the largest = a slab. Guard against a zero
  // largest extent (a single splat) so a degenerate selection isn't "flat".
  const flat = sorted[0] > 0 && sorted[2] / sorted[0] < 0.15;
  const fmt = (v: number) => (v >= 10 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(3));
  return {
    flat,
    text: `${flat ? "flat patch" : "volume"} ${fmt(scaled[0])} × ${fmt(scaled[1])} × ${fmt(scaled[2])} ${unit}`,
  };
}
