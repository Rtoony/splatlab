import { describe, expect, it } from "vitest";
import { describeSelection, editDistance, suggestLabel } from "./paint-guards";

// The real case these exist for: 186,710 splats of flat ground, extent
// 1.30 x 1.54 x 0.04, committed under the label "bikr" while the scene
// inventory already contained "bike".
const INVENTORY = [
  "tree", "bush", "shadow", "grass", "branch", "pavement",
  "ground", "sidewalk", "bench", "path", "wheel", "trunk", "bike",
];

describe("editDistance", () => {
  it("is zero for identical strings and symmetric", () => {
    expect(editDistance("bike", "bike")).toBe(0);
    expect(editDistance("bikr", "bike")).toBe(editDistance("bike", "bikr"));
  });

  it("counts single edits", () => {
    expect(editDistance("bikr", "bike")).toBe(1); // substitution
    expect(editDistance("bik", "bike")).toBe(1); // deletion
    expect(editDistance("biike", "bike")).toBe(1); // insertion
  });

  it("bails out past the cap instead of computing a big distance", () => {
    expect(editDistance("bicycle", "x", 2)).toBeGreaterThan(2);
  });
});

describe("suggestLabel", () => {
  it("catches the typo that started this", () => {
    expect(suggestLabel("bikr", INVENTORY)).toBe("bike");
  });

  it("says nothing when the label is already real", () => {
    expect(suggestLabel("bike", INVENTORY)).toBeNull();
    expect(suggestLabel("  Grass ", INVENTORY)).toBeNull(); // case/space insensitive
  });

  it("stays quiet on genuinely new labels — a wrong suggestion is worse than none", () => {
    expect(suggestLabel("dad's corner", INVENTORY)).toBeNull();
    expect(suggestLabel("hydrant", INVENTORY)).toBeNull();
  });

  it("ignores inputs too short to guess at", () => {
    expect(suggestLabel("bi", INVENTORY)).toBeNull();
    expect(suggestLabel("", INVENTORY)).toBeNull();
  });

  it("does not collide short labels with each other", () => {
    // "path" and "bush" are both 4 chars; neither should suggest the other.
    expect(suggestLabel("path", INVENTORY)).toBeNull();
    expect(suggestLabel("bush", INVENTORY)).toBeNull();
  });
});

describe("describeSelection", () => {
  it("calls the real 'bikr' selection a flat patch", () => {
    const s = describeSelection([1.3, 1.54, 0.04], null);
    expect(s.flat).toBe(true);
    expect(s.text).toContain("flat patch");
    expect(s.text).toContain("u");
  });

  it("calls an upright object a volume", () => {
    expect(describeSelection([1.2, 0.6, 0.9], null).flat).toBe(false);
  });

  it("renders metres once the scene is calibrated", () => {
    const s = describeSelection([1, 1, 1], 2);
    expect(s.text).toContain("m");
    expect(s.text).toContain("2.00");
  });

  it("does not call a degenerate selection flat", () => {
    expect(describeSelection([0, 0, 0], null).flat).toBe(false);
  });
});
