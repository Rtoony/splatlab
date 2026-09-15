import { describe, expect, it, vi } from "vitest";
import { PerspectiveCamera } from "three";
import { architecturalEntry, enterArchitecturalWalk } from "./architectural-walking";

const revisionId = "scene_" + "a".repeat(24);
const architectureId = "architecture_" + "b".repeat(24);
const navigation = () => ({
  v: 2,
  navigation_method: "shared-entry-footprint/v1",
  frame: "world-y-up-metres",
  architecture_id: architectureId,
  gates: { approach_floor_continuous: true, room_floor_continuous: true, capsule_route_clear: true },
  route: [
    [-1, 0, 0],
    [-1, 0, 3],
    [0, 0, 0],
    [0, 0, 3],
    [1, 0, 0],
    [1, 0, 3],
  ],
  floor_y: [0, 0.016, 0, 0.016, 0, 0.016],
});

function fixture() {
  const camera = new PerspectiveCamera();
  camera.position.set(20, 30, 40);
  const walkingBody = { totalHeightM: 2.02, radiusM: 0.32, unitsPerMetre: 1 };
  const walker = {
    camera,
    walkingBody,
    isFlying: true,
    assessWalkingStart: vi.fn((position) => ({
      ok: true,
      message: "Clear",
      body: walkingBody,
      position: position.toArray(),
    })),
    beginWalking: vi.fn(() => {
      walker.isFlying = false;
      return { ok: true, message: "Admitted", body: walkingBody };
    }),
    setFlying: vi.fn((flying: boolean) => {
      walker.isFlying = flying;
    }),
  };
  return walker;
}

describe("architectural walking entry", () => {
  it("uses the pinned center lane and retained floor, not the worker's body dimensions", () => {
    expect(architecturalEntry(navigation(), revisionId, architectureId)).toEqual({
      revisionId,
      architectureId,
      captured: [0, 0, 0],
      extension: [0, 0.016, 3],
    });
  });
  it.each([
    { architecture_id: "architecture_" + "c".repeat(24) },
    { frame: "unknown" },
    { v: 9 },
    { gates: {} },
    { gates: { ...navigation().gates, unrelated: false } },
    { floor_y: [0] },
    { floor_y: [0, 0, 0, NaN, 0, 0] },
    { route: [[0, 0, 0]] },
    { route: Array(6).fill([0, 0, 0]) },
    { navigation_method: "unrecognized" },
  ])("rejects incompatible or incomplete navigation: %j", (change) => {
    expect(() => architecturalEntry({ ...navigation(), ...change }, revisionId, architectureId)).toThrow();
  });
  it("supports retained legacy navigation explicitly", () => {
    expect(
      architecturalEntry(
        { ...navigation(), v: 1, navigation_method: "parallel-center-ray/v1" },
        revisionId,
        architectureId,
      ).extension,
    ).toEqual([0, 0.016, 3]);
  });
  it.each(["captured", "extension"] as const)("checks and admits the selected %s side without resizing", (side) => {
    const walker = fixture();
    const body = { ...walker.walkingBody };
    const entry = architecturalEntry(navigation(), revisionId, architectureId);
    expect(enterArchitecturalWalk(walker, entry, revisionId, side).ok).toBe(true);
    expect(walker.walkingBody).toEqual(body);
    expect(walker.assessWalkingStart).toHaveBeenCalledOnce();
    expect(walker.camera.position.toArray()).toEqual([0, entry[side][1] + 1.71, entry[side][2]]);
    expect(walker.camera.getWorldDirection(walker.camera.position.clone()).z).toBeCloseTo(side === "captured" ? 1 : -1);
    expect(walker.isFlying).toBe(false);
  });
  it("does not move or stop an existing walk when preflight refuses", () => {
    const walker = fixture();
    walker.isFlying = false;
    const position = walker.camera.position.clone();
    walker.assessWalkingStart.mockReturnValue({
      ok: false,
      body: walker.walkingBody,
      message: "Blocked",
      position: undefined,
    });
    expect(
      enterArchitecturalWalk(
        walker,
        architecturalEntry(navigation(), revisionId, architectureId),
        revisionId,
        "captured",
      ).ok,
    ).toBe(false);
    expect(walker.camera.position).toEqual(position);
    expect(walker.setFlying).not.toHaveBeenCalled();
    expect(walker.beginWalking).not.toHaveBeenCalled();
  });
  it.each(["reject", "throw", "still-flying"])("restores the original camera on final admission %s", (failure) => {
    const walker = fixture();
    const position = walker.camera.position.clone();
    const quaternion = walker.camera.quaternion.clone();
    walker.beginWalking.mockImplementation(() => {
      if (failure === "throw") throw new Error("Collider failed");
      return { ok: failure === "still-flying", body: walker.walkingBody, message: "Blocked" };
    });
    const enter = () =>
      enterArchitecturalWalk(
        walker,
        architecturalEntry(navigation(), revisionId, architectureId),
        revisionId,
        "extension",
      );
    if (failure === "throw") expect(enter).toThrow("Collider failed");
    else expect(enter().ok).toBe(false);
    expect(walker.camera.position).toEqual(position);
    expect(walker.camera.quaternion.toArray()).toEqual(quaternion.toArray());
    expect(walker.isFlying).toBe(true);
  });
  it("refuses stale revision and uncalibrated bodies before touching the walker", () => {
    const walker = fixture();
    const entry = architecturalEntry(navigation(), revisionId, architectureId);
    expect(enterArchitecturalWalk(walker, entry, "scene_" + "c".repeat(24), "captured").ok).toBe(false);
    walker.walkingBody.unitsPerMetre = 2;
    expect(enterArchitecturalWalk(walker, entry, revisionId, "captured").ok).toBe(false);
    expect(walker.assessWalkingStart).not.toHaveBeenCalled();
    expect(walker.setFlying).not.toHaveBeenCalled();
  });
});
