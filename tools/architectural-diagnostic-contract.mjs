import assert from "node:assert/strict";
import { LEGACY_APPEARANCE, ROOM_APPEARANCE, LEGACY_GEOMETRY, FINISHED_GEOMETRY, JOINED_GEOMETRY } from "./architectural-proof-contract.mjs";

export function architecturalAppearanceModes(appearance = LEGACY_APPEARANCE, geometry = LEGACY_GEOMETRY, projectionComparison = false) {
  assert.ok([LEGACY_APPEARANCE, ROOM_APPEARANCE].includes(appearance), "Unknown diagnostic appearance method");
  assert.ok([LEGACY_GEOMETRY, FINISHED_GEOMETRY, JOINED_GEOMETRY].includes(geometry), "Unknown diagnostic geometry method");
  if (projectionComparison) {
    assert.equal(appearance, ROOM_APPEARANCE);
    assert.equal(geometry, JOINED_GEOMETRY);
    return ["legacy-render-diagnostic", "projection-only-diagnostic", "original", "no-capture", "legacy-render-restored", "original-restored"];
  }
  return ["original", "lit", "no-capture", "diagnostic-room-mask", "lit-room-mask",
    ...(appearance === ROOM_APPEARANCE && geometry === LEGACY_GEOMETRY
      ? ["threshold-mask-diagnostic", "raised-bridge-diagnostic", "raised-bridge-threshold-mask-diagnostic"] : [])];
}

export function legacyArchitecturalProjection(shader) {
  const block = /    vec2 architecturalNdc =[^]*?    for \(int portalIndex/g;
  assert.equal([...shader.matchAll(block)].length, 1, "Projection comparison requires the new fragment contract");
  assert.ok(shader.includes("architecturalRay / (-architecturalRay.z * gl_FragCoord.w)"));
  return shader.replace(block, `    vec4 architecturalView = architecturalInverseProjection * vec4(vNdc, 1.0);
    vec4 architecturalWorld = architecturalViewToWorld * vec4(architecturalView.xyz / architecturalView.w, 1.0);
    for (int portalIndex`);
}

export function architecturalProofReportName(names) {
  const reports = names.filter(name => ["architectural-browser-proof.json", "architectural-preview-proof.json"].includes(name));
  assert.equal(reports.length, 1, "Require exactly one retained full or preview-only proof report");
  return reports[0];
}

export function raisedBridgeFloor(vertices, spec, liftM) {
  assert.ok(Number.isFinite(liftM) && liftM >= 0 && liftM <= .03, "Diagnostic bridge lift must be between zero and 3 cm");
  assert.ok(vertices.length > 0 && vertices.length % 3 === 0 && Array.from(vertices).every(Number.isFinite));
  assert.ok(Array.isArray(spec.origin) && spec.origin.length === 3 && spec.origin.every(Number.isFinite));
  for (const field of ["yaw_degrees", "opening_width", "cut_depth", "wall_thickness"])
    assert.ok(Number.isFinite(spec[field]));
  assert.ok(spec.opening_width > 0 && spec.cut_depth > 0 && spec.wall_thickness > 0);
  const cosine = Math.cos(spec.yaw_degrees * Math.PI / 180);
  const sine = Math.sin(spec.yaw_degrees * Math.PI / 180);
  const halfWidth = (spec.opening_width + .1) / 2;
  const near = -spec.cut_depth / 2 - .3;
  const far = spec.cut_depth / 2;
  const positions = Array.from(vertices);
  const indices = [], corners = new Set();
  const close = (left, right) => Math.abs(left - right) <= .000005;
  for (let offset = 0; offset < vertices.length; offset += 3) {
    const worldX = vertices[offset] - spec.origin[0];
    const worldY = vertices[offset + 1] - spec.origin[1];
    const worldZ = vertices[offset + 2] - spec.origin[2];
    const localX = cosine * worldX - sine * worldZ;
    const localZ = sine * worldX + cosine * worldZ;
    if (close(Math.abs(localX), halfWidth) && (close(worldY, 0) || close(worldY, -spec.wall_thickness))
        && (close(localZ, near) || close(localZ, far))) {
      corners.add(`${localX > 0}:${close(worldY, 0)}:${close(localZ, far)}`);
      positions[offset + 1] += liftM;
      indices.push(offset / 3);
    }
  }
  assert.equal(corners.size, 8, "Diagnostic must identify all eight authored bridge corners, not infer arbitrary floor vertices");
  return { positions, indices, lift_m: liftM, unique_corners: corners.size };
}
