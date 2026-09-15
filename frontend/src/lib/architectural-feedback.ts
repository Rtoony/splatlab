type CaptureImpact = {
  basis: string;
  total_rows: number;
  cut_rows: number;
  room_rows: number;
  overlap_rows: number;
  union_rows: number;
  protected_rows: number;
  included_rows: number;
  unassigned_rows: number;
};

export type ArchitecturalResult = {
  verdict: string;
  navigation_method?: string;
  appearance_method?: string;
  geometry_method?: string;
  gates: Record<string, boolean>;
  metrics: {
    removed_volume_m3: number;
    minimum_capsule_clearance_m: number;
    maximum_floor_error_m: number | null;
    protected_conflicts: string[];
    replacement_samples_outside_portal?: Record<string, number>;
    capture_impact?: CaptureImpact | null;
    route_samples: number;
    floor_finish_m?: number;
  };
};

const navigationMethods = ["parallel-center-ray/v1", "shared-entry-footprint/v1"];
const appearanceMethods = ["cut-only/v1", "cut-and-authored-room/v1"];
const geometryMethods = ["nominal-floor/v1", "raised-floor-finish/v1", "joined-floor-envelope/v1"];

function validGeometry(result: ArchitecturalResult): boolean {
  const method = result.geometry_method === undefined ? geometryMethods[0] : result.geometry_method;
  return (
    geometryMethods.includes(method) &&
    (method !== geometryMethods[0]
      ? result.metrics.floor_finish_m === 0.016
      : result.metrics.floor_finish_m === undefined || result.metrics.floor_finish_m === 0)
  );
}

export function architecturalGeometryDescription(result: ArchitecturalResult): string {
  if (!validGeometry(result))
    return "Unsupported or inconsistent floor geometry recipe. Build a fresh candidate before previewing.";
  if (result.geometry_method === geometryMethods[2])
    return "Experimental joined solid boundary removes overlapping authored faces and uses flat face normals. The roof extends over wall tops within the reserved frame, replacing edge-only joints. Both floors and collision keep the 16 mm finish; clear opening and room heights remain 16 mm below nominal. Captured clipping, protection and player dimensions are unchanged; visual and walking review are still required.";
  if (result.geometry_method === geometryMethods[1])
    return "Experimental 16 mm floor finish raises both authored slabs and their collision together. Clear opening and room heights are 16 mm below the nominal dimensions. Captured clipping, player size and walking limits are unchanged; visual and walking review are still required.";
  return "Authored bridge and room floors use the nominal threshold datum. No raised floor finish is applied.";
}

function validCaptureImpact(value: CaptureImpact): boolean {
  if (typeof value !== "object" || value.basis !== "captured-gaussian-centers/v1") return false;
  const counts = [
    value.total_rows,
    value.cut_rows,
    value.room_rows,
    value.overlap_rows,
    value.union_rows,
    value.protected_rows,
    value.included_rows,
    value.unassigned_rows,
  ];
  return (
    counts.every((count) => Number.isSafeInteger(count) && count >= 0 && count <= value.total_rows) &&
    value.overlap_rows <= Math.min(value.cut_rows, value.room_rows) &&
    value.union_rows === value.cut_rows + value.room_rows - value.overlap_rows &&
    value.union_rows === value.protected_rows + value.included_rows + value.unassigned_rows
  );
}

export function architecturalCaptureImpactDescription(result: ArchitecturalResult): string {
  const impact = result.metrics.capture_impact;
  if (impact == null)
    return "Captured-content impact is unavailable for this retained result. No unnamed-object preservation claim follows from passing named-object checks. Build a fresh candidate for center-membership evidence.";
  if (!validCaptureImpact(impact))
    return "Captured-content impact is inconsistent or unsupported. Retain this result for diagnosis and build a fresh candidate before previewing.";
  return (
    `Captured centers inside edit: ${impact.union_rows.toLocaleString()} unique (doorway ${impact.cut_rows.toLocaleString()}; room ${impact.room_rows.toLocaleString()}; overlap ${impact.overlap_rows.toLocaleString()}). ` +
    `${impact.protected_rows.toLocaleString()} protected, ${impact.included_rows.toLocaleString()} explicitly included, ${impact.unassigned_rows.toLocaleString()} outside both sets. ` +
    "Unassigned centers can include real unnamed furniture or walls; they are not approved removals. These are center memberships, not object counts or exact rendered-fragment removal. Review source photos and oblique views before applying."
  );
}

export function architecturalAppearanceDescription(result: ArchitecturalResult): string {
  if (result.appearance_method === appearanceMethods[1])
    return "Captured appearance is clipped in the doorway and authored room envelope. The full room interior must be clear of captured collision, and named captured rows remain protected. Unnamed objects and oblique boundaries still require visual review.";
  if (result.appearance_method === undefined || result.appearance_method === appearanceMethods[0])
    return "Legacy cut-only appearance: captured remnants can remain inside the authored room. Review the interior before applying.";
  return "Unknown appearance method. Prepare a fresh supported candidate; this result cannot be previewed.";
}

export function architecturalNavigationDescription(result: ArchitecturalResult): string {
  if (result.navigation_method === "shared-entry-footprint/v1") {
    return "Centered entry with three interior lanes and footprint floor support. Sampled body: 1.70 m height, 0.22 m radius; not default-body or whole-scene walking acceptance.";
  }
  if (result.navigation_method === undefined || result.navigation_method === navigationMethods[0]) {
    return "Legacy parallel approach lanes with center-ray floor support. Sampled body: 1.70 m height, 0.22 m radius; not default-body or whole-scene walking acceptance.";
  }
  return "Unknown navigation method. This result cannot be previewed; prepare a fresh candidate with a supported method.";
}

export function architecturalFailureGuidance(result: ArchitecturalResult): { gate: string; message: string }[] {
  const floorError = result.metrics.maximum_floor_error_m;
  const floorDetail =
    Number.isFinite(floorError) && floorError !== null
      ? ` The largest sampled floor offset is ${(floorError * 100).toFixed(1)} cm.`
      : " Some floor support may be missing; inspect the retained probes.";
  const protectedNames = result.metrics.protected_conflicts.join(", ");
  const outside = Object.entries(result.metrics.replacement_samples_outside_portal || {})
    .filter(([, count]) => count > 0)
    .map(([slug, count]) => `${slug}: ${count} samples`)
    .join(", ");
  const messages: Record<string, string> = {
    base_watertight:
      "The captured collider is not a valid closed solid. Repair its geometry before preparing another cut.",
    cut_watertight:
      "The Boolean result is not a valid closed solid. Keep this attempt for diagnosis; it cannot be previewed.",
    positive_bounded_removal:
      "The cut misses captured solid or removes an invalid volume. Check the frame against the collision view, not only the photograph.",
    portal_volume_clear:
      "Captured collision remains inside the proposed opening. Inspect the retained cut and collider before changing the frame.",
    outside_surface_preserved:
      "The result changes sampled captured surfaces outside the allowed cut. Reject this result rather than accepting collateral changes.",
    approach_floor_continuous:
      "The approach does not establish continuous supported floor. Inspect the actual floor height and obstacles before moving the threshold." +
      floorDetail,
    room_floor_continuous:
      "The doorway or extension does not establish continuous supported floor. Check the bridge elevation and whether the room intersects the captured collider." +
      floorDetail,
    authored_room_interior_clear:
      "Captured collision intrudes into the authored room interior. A room appearance mask must not hide an obstacle; inspect the retained intrusion geometry or reposition the extension.",
    authored_room_watertight:
      "The joined authored boundary is not a closed, consistently oriented solid. Retain this failed build for diagnosis; do not hide seams or weaken the geometry check.",
    capsule_route_clear:
      "The sampled player body is obstructed. Inspect the approach, jambs, overhead clearance and room interior; do not silently shrink the player to make the result pass.",
    protected_elements_unchanged: `Protected objects overlap the proposed edit${protectedNames ? `: ${protectedNames}` : ""}. Move or resize it rather than silently erasing them.`,
    replacement_region_contained: `An explicitly included object's geometry or captured centers extend outside the cut${outside ? ` (${outside})` : ""}. Reposition the frame or stop including that removal; partial containment is not enough.`,
  };
  const feedback = Object.entries(result.gates)
    .filter(([, passed]) => passed === false)
    .map(([gate]) => ({
      gate,
      message:
        messages[gate] ||
        "This required geometry check failed. Inspect its retained evidence before preparing a new candidate.",
    }));
  if (result.navigation_method !== undefined && !navigationMethods.includes(result.navigation_method)) {
    feedback.push({ gate: "navigation_method", message: architecturalNavigationDescription(result) });
  }
  if (result.appearance_method !== undefined && !appearanceMethods.includes(result.appearance_method)) {
    feedback.push({ gate: "appearance_method", message: architecturalAppearanceDescription(result) });
  }
  if (result.metrics.capture_impact != null && !validCaptureImpact(result.metrics.capture_impact)) {
    feedback.push({ gate: "capture_impact", message: architecturalCaptureImpactDescription(result) });
  } else if (
    result.metrics.capture_impact != null &&
    result.metrics.capture_impact.protected_rows > 0 &&
    result.gates.protected_elements_unchanged !== false
  ) {
    feedback.push({
      gate: "capture_impact",
      message:
        "Captured-center evidence overlaps protected rows despite the protection gate. Do not preview this inconsistent result; inspect the retained memberships and build a fresh candidate.",
    });
  }
  if (!validGeometry(result)) {
    feedback.push({ gate: "geometry_method", message: architecturalGeometryDescription(result) });
  }
  return feedback;
}

export function architecturalResultCanPreview(result: ArchitecturalResult): boolean {
  const required = [
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
  ];
  if (result.appearance_method === appearanceMethods[1]) required.push("authored_room_interior_clear");
  if (result.geometry_method === geometryMethods[2]) required.push("authored_room_watertight");
  return (
    result.verdict === "PASS_ARCHITECTURAL_EDIT" &&
    (result.navigation_method === undefined || navigationMethods.includes(result.navigation_method)) &&
    (result.appearance_method === undefined || appearanceMethods.includes(result.appearance_method)) &&
    validGeometry(result) &&
    (result.metrics.capture_impact == null ||
      (validCaptureImpact(result.metrics.capture_impact) && result.metrics.capture_impact.protected_rows === 0)) &&
    required.every((gate) => result.gates[gate] === true) &&
    Object.values(result.gates).every((passed) => passed === true)
  );
}
