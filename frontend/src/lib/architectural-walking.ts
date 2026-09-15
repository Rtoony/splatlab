import { Vector3 } from "three";
import type { WorldWalker } from "./world-walker";
import { walkingBodyError, type WalkingAdmission } from "./walking-admission";

type Point = [number, number, number];
export type ArchitecturalEntry = {
  revisionId: string;
  architectureId: string;
  captured: Point;
  extension: Point;
};

export function architecturalEntry(document: unknown, revisionId: string, architectureId: string): ArchitecturalEntry {
  if (!document || typeof document !== "object") throw new Error("Missing pinned architectural navigation.");
  const navigation = document as Record<string, unknown>;
  const route = navigation.route;
  const floors = navigation.floor_y;
  const gates = navigation.gates;
  const legacy =
    navigation.v === 1 && (navigation.navigation_method ?? "parallel-center-ray/v1") === "parallel-center-ray/v1";
  const current = navigation.v === 2 && navigation.navigation_method === "shared-entry-footprint/v1";
  if (
    !/^scene_[a-f0-9]{24}$/.test(revisionId) ||
    !/^architecture_[a-f0-9]{24}$/.test(architectureId) ||
    navigation.architecture_id !== architectureId ||
    navigation.frame !== "world-y-up-metres" ||
    (!legacy && !current) ||
    !Array.isArray(route) ||
    route.length < 6 ||
    route.length > 30000 ||
    route.length % 3 !== 0 ||
    !Array.isArray(floors) ||
    floors.length !== route.length ||
    !gates ||
    typeof gates !== "object" ||
    Object.values(gates).some((value) => value !== true) ||
    !["approach_floor_continuous", "room_floor_continuous", "capsule_route_clear"].every(
      (name) => (gates as Record<string, unknown>)[name] === true,
    ) ||
    route.some((point) => !Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) ||
    !floors.every(Number.isFinite)
  )
    throw new Error("Walking entry requires matching, passed navigation from this pinned revision.");
  const count = route.length / 3;
  const pointAt = (index: number): Point => [route[index][0], floors[index], route[index][2]];
  const captured = pointAt(count);
  const extension = pointAt(2 * count - 1);
  const distance = Math.hypot(extension[0] - captured[0], extension[2] - captured[2]);
  if (distance < 0.5 || distance > 15)
    throw new Error("Architectural entry points do not form a bounded walking route.");
  return { revisionId, architectureId, captured, extension };
}

export function enterArchitecturalWalk(
  walker: Pick<
    WorldWalker,
    "camera" | "walkingBody" | "assessWalkingStart" | "beginWalking" | "setFlying" | "isFlying"
  >,
  entry: ArchitecturalEntry,
  revisionId: string,
  side: "captured" | "extension",
): WalkingAdmission {
  const body = walker.walkingBody;
  const problem = walkingBodyError(body);
  if (entry.revisionId !== revisionId || body.unitsPerMetre !== 1 || problem)
    return {
      ok: false,
      body,
      message: problem || "Reload the matching metre-calibrated revision before entering this route.",
    };
  const start = entry[side];
  const target = entry[side === "captured" ? "extension" : "captured"];
  const viewpoint = new Vector3(start[0], start[1] + body.totalHeightM - body.radiusM + 0.01, start[2]);
  const assessment = walker.assessWalkingStart(viewpoint);
  if (!assessment.ok || !assessment.position) return assessment;
  const position = walker.camera.position.clone();
  const quaternion = walker.camera.quaternion.clone();
  let accepted = false;
  try {
    walker.setFlying(true);
    walker.camera.position.fromArray(assessment.position);
    walker.camera.lookAt(target[0], walker.camera.position.y, target[2]);
    const admission = walker.beginWalking();
    accepted = admission.ok && !walker.isFlying;
    return accepted
      ? admission
      : { ...admission, ok: false, message: "Walking entry was refused by the current collider. " + admission.message };
  } finally {
    if (!accepted) {
      walker.setFlying(true);
      walker.camera.position.copy(position);
      walker.camera.quaternion.copy(quaternion);
    }
  }
}
