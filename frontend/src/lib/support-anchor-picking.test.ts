import { describe, expect, it } from "vitest";
import { nearestSupportFeature, type SupportFeature } from "./support-anchor-picking";

const features: SupportFeature[] = [
  { point_id: "9007199254740993", pixel: [100, 100], world: [0, 0, 0], reprojection_px: 0.2 },
  { point_id: "25", pixel: [130, 105], world: [1, 0, 0], reprojection_px: 0.3 },
];

describe("photo-linked support picking", () => {
  it("picks the nearest observation without rounding large point identifiers", () => {
    expect(nearestSupportFeature(features, [102, 102], 1)?.point_id).toBe("9007199254740993");
    expect(nearestSupportFeature(features, [128, 106], 1)?.point_id).toBe("25");
  });
  it("keeps the click tolerance in rendered screen pixels", () => {
    expect(nearestSupportFeature(features, [75, 100], 1)).toBeNull();
    expect(nearestSupportFeature(features, [75, 100], 3)?.point_id).toBe("9007199254740993");
  });
  it("does not invent a depth for empty or invalid clicks", () => {
    expect(nearestSupportFeature([], [100, 100], 1)).toBeNull();
    expect(nearestSupportFeature(features, [100, 100], 0)).toBeNull();
    expect(nearestSupportFeature(features, [NaN, 100], 1)).toBeNull();
    expect(nearestSupportFeature(features, [100, 100], Infinity)).toBeNull();
  });
});
