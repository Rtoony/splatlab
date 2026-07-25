import { describe, expect, it } from "vitest";
import { MAX_ITERS, MIN_ITERS, QUALITY, presetForIters, stageHuman, stageShort, trainMinutes } from "./stage-meta";

describe("presetForIters", () => {
  it("roundtrips every quality preset", () => {
    for (const key of Object.keys(QUALITY) as (keyof typeof QUALITY)[]) {
      expect(presetForIters(QUALITY[key].iterations)).toBe(key);
    }
  });
  it("returns null off-preset", () => {
    expect(presetForIters(12345)).toBeNull();
  });
});

describe("trainMinutes", () => {
  it("never estimates under 2 minutes and grows with iterations", () => {
    expect(trainMinutes(MIN_ITERS)).toBeGreaterThanOrEqual(2);
    expect(trainMinutes(MAX_ITERS)).toBeGreaterThan(trainMinutes(MIN_ITERS));
  });
});

describe("stage labels", () => {
  it("labels auto-fallback reprocess<n> stages like Process", () => {
    expect(stageShort("reprocess2")).toBe("Process");
    expect(stageHuman("reprocess2")).toBe("Finding camera positions");
  });
  it("passes unknown stages through verbatim (never invents copy)", () => {
    expect(stageShort("mystery")).toBe("mystery");
    expect(stageHuman("mystery")).toBe("mystery");
  });
});
