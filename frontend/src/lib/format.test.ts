import { describe, expect, it } from "vitest";
import { fmtCount, formatBakeoffVerdict, relTime, sceneHue } from "./format";

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

describe("formatBakeoffVerdict", () => {
  it("names the winner with the margin over the runner-up", () => {
    expect(
      formatBakeoffVerdict({
        winner: "textured",
        ranked: [
          { name: "textured", median_psnr_paired: 13.76 },
          { name: "raw_tsdf", median_psnr_paired: 13.55 },
        ],
      }),
    ).toBe("Bake-off: textured wins (+0.21 dB vs raw_tsdf)");
  });

  it("handles a sole ranked candidate", () => {
    expect(
      formatBakeoffVerdict({
        winner: "textured",
        ranked: [{ name: "textured", median_psnr_paired: 12.5 }],
      }),
    ).toBe("Bake-off: textured wins (12.50 dB, sole ranked candidate)");
  });

  it("states when nothing was trustworthy", () => {
    expect(formatBakeoffVerdict({ winner: null, ranked: [] })).toBe(
      "Bake-off: no trustworthy winner",
    );
  });

  it("surfaces an unreadable report as-is", () => {
    expect(formatBakeoffVerdict({ error: "unreadable bakeoff.json" })).toBe(
      "Bake-off: unreadable bakeoff.json",
    );
  });

  it("degrades to a plain win when scores are missing", () => {
    expect(formatBakeoffVerdict({ winner: "twin", ranked: [{ name: "twin", median_psnr_paired: null }] })).toBe(
      "Bake-off: twin wins",
    );
  });
});
