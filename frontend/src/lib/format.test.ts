import { describe, expect, it } from "vitest";
import { fmtCount, relTime, sceneHue } from "./format";

describe("relTime", () => {
  it("dashes on null and floors recent times to 'just now'", () => {
    expect(relTime(null)).toBe("—");
    expect(relTime(new Date().toISOString())).toBe("just now");
  });
  it("buckets hours and days", () => {
    expect(relTime(new Date(Date.now() - 2 * 3600_000).toISOString())).toBe("2h ago");
    expect(relTime(new Date(Date.now() - 3 * 86400_000).toISOString())).toBe("3d ago");
  });
});

describe("fmtCount", () => {
  it("matches the documented examples", () => {
    expect(fmtCount(1_284_773)).toBe("1.3M");
    expect(fmtCount(608_501)).toBe("609k");
    expect(fmtCount(42)).toBe("42");
  });
});

describe("sceneHue", () => {
  it("is deterministic and in [0, 360)", () => {
    expect(sceneHue("splat_32d926d9")).toBe(sceneHue("splat_32d926d9"));
    for (const id of ["a", "splat_30b75bc81f", ""]) {
      const h = sceneHue(id);
      expect(h).toBeGreaterThanOrEqual(0);
      expect(h).toBeLessThan(360);
    }
  });
});
