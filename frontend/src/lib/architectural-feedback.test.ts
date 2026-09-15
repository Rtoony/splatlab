import { describe, expect, it } from "vitest";
import {
  architecturalFailureGuidance,
  architecturalNavigationDescription,
  architecturalAppearanceDescription,
  architecturalCaptureImpactDescription,
  architecturalGeometryDescription,
  architecturalResultCanPreview,
  type ArchitecturalResult,
} from "./architectural-feedback";

it("requires the new closed boundary gate without weakening the finished floor contract", () => {
  const value = passingResult();
  value.geometry_method = "joined-floor-envelope/v1";
  value.metrics.floor_finish_m = 0.016;
  expect(architecturalResultCanPreview(value)).toBe(false);
  value.gates.authored_room_watertight = true;
  expect(architecturalResultCanPreview(value)).toBe(true);
  expect(architecturalGeometryDescription(value)).toContain("overlapping authored faces");
  value.gates.authored_room_watertight = false;
  expect(architecturalFailureGuidance(value)[0]?.message).toContain("closed, consistently oriented");
  value.gates.authored_room_watertight = true;
  value.metrics.floor_finish_m = 0;
  expect(architecturalResultCanPreview(value)).toBe(false);
});

it("keeps the experimental finish explicit without changing legacy floor evidence", () => {
  const value = passingResult();
  expect(architecturalGeometryDescription(value)).toContain("nominal threshold");
  expect(architecturalResultCanPreview(value)).toBe(true);
  value.geometry_method = "raised-floor-finish/v1";
  value.metrics.floor_finish_m = 0.016;
  expect(architecturalResultCanPreview(value)).toBe(true);
  expect(architecturalGeometryDescription(value)).toContain("collision together");
  expect(architecturalGeometryDescription(value)).toContain("16 mm below");
  expect(architecturalGeometryDescription(value)).toContain("player size");
});

it.each([
  { geometry_method: "raised-floor-finish/v1", floor_finish_m: undefined },
  { geometry_method: "raised-floor-finish/v1", floor_finish_m: 0.03 },
  { geometry_method: "nominal-floor/v1", floor_finish_m: 0.016 },
  { geometry_method: "raised-floor-finish/v2", floor_finish_m: 0.016 },
])("refuses inconsistent floor evidence: $geometry_method / $floor_finish_m", (fields) => {
  const value = passingResult();
  value.geometry_method = fields.geometry_method;
  value.metrics.floor_finish_m = fields.floor_finish_m;
  expect(architecturalResultCanPreview(value)).toBe(false);
  expect(architecturalFailureGuidance(value).at(-1)?.message).toContain("floor geometry recipe");
});

function result(): ArchitecturalResult {
  return {
    verdict: "FAIL_ARCHITECTURAL_EDIT",
    gates: {
      approach_floor_continuous: false,
      capsule_route_clear: false,
      protected_elements_unchanged: false,
      replacement_region_contained: false,
      cut_watertight: true,
    },
    metrics: {
      removed_volume_m3: 4.2995,
      minimum_capsule_clearance_m: 0.000219,
      maximum_floor_error_m: 0.3972,
      protected_conflicts: ["red-bicycle", "orange-bike-bottle"],
      route_samples: 387,
      replacement_samples_outside_portal: { "plastic-storage-container": 1, "contained-object": 0 },
    },
  };
}

function passingResult(): ArchitecturalResult {
  const value = result();
  value.verdict = "PASS_ARCHITECTURAL_EDIT";
  value.gates = Object.fromEntries(
    [
      "base_watertight",
      "cut_watertight",
      "positive_bounded_removal",
      "portal_volume_clear",
      "outside_surface_preserved",
      "approach_floor_continuous",
      "room_floor_continuous",
      "capsule_route_clear",
      "protected_elements_unchanged",
      "replacement_region_contained",
    ].map((gate) => [gate, true]),
  );
  return value;
}

describe("architectural failure guidance", () => {
  it("does not describe missing captured impact as zero affected objects", () => {
    const value = result();
    expect(architecturalCaptureImpactDescription(value)).toContain("unavailable");
    value.metrics.capture_impact = null;
    expect(architecturalCaptureImpactDescription(value)).toContain("No unnamed-object preservation claim");
  });

  it("reports overlapping center sets without calling unassigned content approved removal", () => {
    const value = passingResult();
    value.metrics.capture_impact = {
      basis: "captured-gaussian-centers/v1",
      total_rows: 5,
      cut_rows: 2,
      room_rows: 2,
      overlap_rows: 1,
      union_rows: 3,
      protected_rows: 1,
      included_rows: 1,
      unassigned_rows: 1,
    };
    const message = architecturalCaptureImpactDescription(value);
    expect(message).toContain("3 unique");
    expect(message).toContain("1 protected, 1 explicitly included, 1 outside both sets");
    expect(message).toContain("not approved removals");
    expect(message).toContain("not object counts or exact rendered-fragment removal");
    expect(architecturalResultCanPreview(value)).toBe(false);
    expect(architecturalFailureGuidance(value).at(-1)?.message).toContain("overlaps protected rows");
    value.metrics.capture_impact.protected_rows = 0;
    value.metrics.capture_impact.unassigned_rows = 2;
    expect(architecturalResultCanPreview(value)).toBe(true);
  });

  it.each([
    { basis: "unknown" },
    { union_rows: 4 },
    { protected_rows: 2 },
    { overlap_rows: 3 },
    { cut_rows: -1 },
    { room_rows: 2.5 },
    { total_rows: Number.NaN },
  ])("refuses inconsistent captured impact without changing its receipt: %j", (change) => {
    const value = passingResult();
    expect(architecturalResultCanPreview(value)).toBe(true);
    value.metrics.capture_impact = {
      basis: "captured-gaussian-centers/v1",
      total_rows: 5,
      cut_rows: 2,
      room_rows: 2,
      overlap_rows: 1,
      union_rows: 3,
      protected_rows: 1,
      included_rows: 1,
      unassigned_rows: 1,
      ...change,
    };
    const original = structuredClone(value);
    expect(architecturalCaptureImpactDescription(value)).toContain("inconsistent or unsupported");
    expect(architecturalFailureGuidance(value).at(-1)?.gate).toBe("capture_impact");
    expect(architecturalResultCanPreview(value)).toBe(false);
    expect(value).toEqual(original);
  });

  it("explains retained real-candidate failure shapes without hiding failed checks", () => {
    const feedback = architecturalFailureGuidance(result());
    expect(feedback).toHaveLength(4);
    expect(feedback.find((item) => item.gate === "approach_floor_continuous")?.message).toContain("39.7 cm");
    expect(feedback.find((item) => item.gate === "capsule_route_clear")?.message).toContain("do not silently shrink");
    expect(feedback.find((item) => item.gate === "protected_elements_unchanged")?.message).toContain(
      "red-bicycle, orange-bike-bottle",
    );
    const containment = feedback.find((item) => item.gate === "replacement_region_contained")?.message;
    expect(containment).toContain("plastic-storage-container: 1 samples");
    expect(containment).not.toContain("contained-object");
  });

  it("does not turn missing support into a zero-centimetre claim", () => {
    const missing = result();
    missing.metrics.maximum_floor_error_m = null;
    const text = architecturalFailureGuidance(missing)[0].message;
    expect(text).toContain("support may be missing");
    expect(text).not.toContain("0.0 cm");
  });

  it("retains unknown failed checks and does not explain passed checks as failures", () => {
    const value = result();
    value.gates = { future_check: false, portal_volume_clear: true };
    expect(architecturalFailureGuidance(value)).toEqual([
      {
        gate: "future_check",
        message: "This required geometry check failed. Inspect its retained evidence before preparing a new candidate.",
      },
    ]);
  });

  it("refuses a passing label with failed or missing required evidence", () => {
    const value = result();
    value.verdict = "PASS_ARCHITECTURAL_EDIT";
    expect(architecturalResultCanPreview(value)).toBe(false);
    value.gates = {};
    expect(architecturalResultCanPreview(value)).toBe(false);
  });

  it("allows preview only with the exact passing verdict and every required gate", () => {
    const value = passingResult();
    expect(architecturalResultCanPreview(value)).toBe(true);
    expect(architecturalFailureGuidance(value)).toEqual([]);
    value.navigation_method = "shared-entry-footprint/v1";
    expect(architecturalResultCanPreview(value)).toBe(true);
    expect(architecturalNavigationDescription(value)).toContain("Centered entry");
    expect(architecturalNavigationDescription(value)).toContain("not default-body");
    value.navigation_method = "unknown";
    expect(architecturalResultCanPreview(value)).toBe(false);
    expect(architecturalFailureGuidance(value)[0].message).toContain("Unknown navigation method");
    value.navigation_method = "parallel-center-ray/v1";
    expect(architecturalResultCanPreview(value)).toBe(true);
    expect(architecturalNavigationDescription(value)).toContain("Legacy parallel");
    value.appearance_method = "cut-and-authored-room/v1";
    expect(architecturalResultCanPreview(value)).toBe(false);
    value.gates.authored_room_interior_clear = true;
    expect(architecturalResultCanPreview(value)).toBe(true);
    expect(architecturalAppearanceDescription(value)).toContain("Unnamed objects");
    value.gates.authored_room_interior_clear = false;
    expect(architecturalFailureGuidance(value)[0].message).toContain("must not hide an obstacle");
    value.gates.authored_room_interior_clear = true;
    value.appearance_method = "unknown";
    expect(architecturalResultCanPreview(value)).toBe(false);
    expect(architecturalFailureGuidance(value)[0].message).toContain("Unknown appearance method");
    value.appearance_method = "cut-and-authored-room/v1";
    value.gates.future_check = false;
    expect(architecturalResultCanPreview(value)).toBe(false);
    delete value.gates.future_check;
    value.verdict = "FAIL_ARCHITECTURAL_EDIT";
    expect(architecturalResultCanPreview(value)).toBe(false);
  });
});
