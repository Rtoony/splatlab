import * as THREE from "three";
import type { MeshBVH } from "three-mesh-bvh";

export type WalkingBody = { radiusM: number; totalHeightM: number; unitsPerMetre: number };
export type WalkingAdmission = {
  ok: boolean;
  message: string;
  body: WalkingBody;
  position?: [number, number, number];
  floorY?: number;
  blockingDistanceM?: number;
};

export function walkingBodyError(body: WalkingBody): string | null {
  if (!Number.isFinite(body.unitsPerMetre) || body.unitsPerMetre < 0.01 || body.unitsPerMetre > 1000)
    return "Walking requires a finite calibrated scene scale.";
  if (!Number.isFinite(body.radiusM) || body.radiusM < 0.15 || body.radiusM > 0.5)
    return "Player radius must be between 0.15 and 0.50 m.";
  if (
    !Number.isFinite(body.totalHeightM) ||
    body.totalHeightM < 1.2 ||
    body.totalHeightM > 2.5 ||
    body.totalHeightM <= 2 * body.radiusM
  )
    return "Player total height must be between 1.20 and 2.50 m and exceed twice the radius.";
  return null;
}

function prerequisites(
  bvh: MeshBVH | null,
  collider: THREE.Object3D | null,
  position: THREE.Vector3,
  body: WalkingBody,
): string | null {
  const problem = walkingBodyError(body);
  if (problem) return problem;
  if (!bvh || !collider) return "The pinned collision geometry is not ready for walking.";
  if (!position.toArray().every(Number.isFinite)) return "The camera position is invalid.";
  const identity = new THREE.Matrix4().elements;
  if (
    !collider.matrixWorld.elements.every(
      (value, index) => Number.isFinite(value) && Math.abs(value - identity[index]) < 1e-9,
    )
  )
    return "Walking admission requires the world-baked collider; transformed collision needs its own verified adapter.";
  return null;
}

export function inspectWalkingPose(
  bvh: MeshBVH | null,
  collider: THREE.Object3D | null,
  position: THREE.Vector3,
  body: WalkingBody,
): WalkingAdmission {
  const error = prerequisites(bvh, collider, position, body);
  if (error || !bvh) return { ok: false, message: error || "Collision unavailable.", body };
  const units = body.unitsPerMetre;
  const radius = body.radiusM * units;
  const eyeHeight = (body.totalHeightM - body.radiusM) * units;
  const feet = position.y - eyeHeight;
  const ray = new THREE.Ray(
    new THREE.Vector3(position.x, feet + 0.04 * units, position.z),
    new THREE.Vector3(0, -1, 0),
  );
  const floor = bvh.raycastFirst(ray, THREE.DoubleSide, 0, 0.12 * units);
  if (!floor?.face || floor.face.normal.y < 0.85 || Math.abs(floor.point.y - feet) > 0.08 * units)
    return {
      ok: false,
      message:
        "No supported, upward-facing floor under the requested player body. Move the free camera over a clear floor and try again.",
      body,
    };
  const segment = new THREE.Line3(
    position.clone(),
    position.clone().add(new THREE.Vector3(0, -(eyeHeight - radius), 0)),
  );
  const bounds = new THREE.Box3().setFromPoints([segment.start, segment.end]).expandByScalar(radius);
  const trianglePoint = new THREE.Vector3();
  const bodyPoint = new THREE.Vector3();
  let closest = Infinity;
  bvh.shapecast({
    intersectsBounds: (box) => box.intersectsBox(bounds),
    intersectsTriangle: (triangle) => {
      closest = Math.min(closest, triangle.closestPointToSegment(segment, trianglePoint, bodyPoint));
      return false;
    },
  });
  if ((!Number.isFinite(closest) && closest !== Infinity) || closest < radius - 1e-5 * units)
    return {
      ok: false,
      message: `The ${body.totalHeightM.toFixed(2)} m tall, ${(2 * body.radiusM).toFixed(2)} m wide player intersects collision here. Check nearby objects and overhead clearance; the body is not shortened.`,
      body,
      blockingDistanceM: Number.isFinite(closest) ? closest / units : undefined,
    };
  return {
    ok: true,
    message: `Fixed player: ${body.totalHeightM.toFixed(2)} m total height, ${(2 * body.radiusM).toFixed(2)} m width. Local start clearance passes; the route still needs review.`,
    body,
    position: position.toArray(),
    floorY: floor.point.y,
  };
}

export function findWalkingStart(
  bvh: MeshBVH | null,
  collider: THREE.Object3D | null,
  viewpoint: THREE.Vector3,
  body: WalkingBody,
): WalkingAdmission {
  const error = prerequisites(bvh, collider, viewpoint, body);
  if (error || !bvh) return { ok: false, message: error || "Collision unavailable.", body };
  const units = body.unitsPerMetre;
  const ray = new THREE.Ray(viewpoint.clone().add(new THREE.Vector3(0, 0.001 * units, 0)), new THREE.Vector3(0, -1, 0));
  const hit = bvh.raycastFirst(ray, THREE.DoubleSide, 0, 3 * units);
  if (!hit?.face || hit.face.normal.y < 0.85)
    return {
      ok: false,
      message:
        "No upward-facing floor within 3 m below this view. Move over a clear surface; collision will not be bypassed.",
      body,
    };
  const position = new THREE.Vector3(
    viewpoint.x,
    hit.point.y + (body.totalHeightM - body.radiusM + 0.01) * units,
    viewpoint.z,
  );
  if (position.y - viewpoint.y > 0.35 * units)
    return {
      ok: false,
      message:
        "This view is too close to a raised surface for the requested body. Move the free camera above a clear floor before entering walking mode.",
      body,
    };
  return inspectWalkingPose(bvh, collider, position, body);
}
