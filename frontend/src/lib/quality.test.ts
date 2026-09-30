import { describe, expect, it } from "vitest";
import type { QualityTier } from "./contracts";
import { defaultTier, formatBytes, formatSplats, viewableTiers } from "./quality";

const tier = (id: QualityTier["id"], splats: number, bytes: number): QualityTier => ({
  id, label: id, splats, bytes, compressed: id !== "full", url: `/f?fmt=${id}`,
});
const ALL = [tier("full", 9_979_075, 2_474_812_157), tier("web", 3_000_000, 48_844_429), tier("lite", 1_000_000, 16_281_964)];

describe("viewing-quality tiers (full only on the Nexus PC)", () => {
  it("never offers full quality to a browser that is not the PC", () => {
    expect(viewableTiers(ALL, false).map((t) => t.id)).toEqual(["web", "lite"]);
    expect(viewableTiers(ALL, true).map((t) => t.id)).toEqual(["full", "web", "lite"]);
  });

  it("defaults: PC -> full, other desktop -> web, phone -> lite", () => {
    expect(defaultTier(viewableTiers(ALL, true), true, false, null)?.id).toBe("full");
    expect(defaultTier(viewableTiers(ALL, false), false, false, null)?.id).toBe("web");
    expect(defaultTier(viewableTiers(ALL, false), false, true, null)?.id).toBe("lite");
    // A phone on the home network is still a phone.
    expect(defaultTier(viewableTiers(ALL, true), true, true, null)?.id).toBe("lite");
  });

  it("a remembered choice wins only where it is viewable", () => {
    expect(defaultTier(viewableTiers(ALL, true), true, false, "lite")?.id).toBe("lite");
    expect(defaultTier(viewableTiers(ALL, false), false, false, "full")?.id).toBe("web");
  });

  it("falls back to what exists (legacy scenes have no lite copy)", () => {
    const legacy = [ALL[0], tier("web", 1_200_000, 80_000_000)];
    expect(defaultTier(viewableTiers(legacy, false), false, true, null)?.id).toBe("web");
    expect(defaultTier([], false, false, null)).toBeNull();
  });

  it("formats sizes and counts for the picker", () => {
    expect(formatBytes(2_474_812_157)).toBe("2.5 GB");
    expect(formatBytes(48_844_429)).toBe("49 MB");
    expect(formatSplats(9_979_075)).toBe("10M");
    expect(formatSplats(3_000_000)).toBe("3.0M");
    expect(formatSplats(null)).toBe("");
  });
});
