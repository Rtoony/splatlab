import { describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import {
  aimArchitecturalThreshold,
  architecturalSpecError,
  canvasPickRay,
  createArchitecturalGuide,
  currentArchitecturalDraft,
  defaultArchitecturalSpec,
  installArchitecturalPicking,
  positionArchitecturalThreshold,
  type ArchitecturalDraft,
} from "./architectural-draft";

describe("architectural pointer and keyboard controls", () => {
  function setup(mode: "position" | "direction" | "inspect" = "position") {
    const canvas = Object.assign(new EventTarget(), {
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 400 }),
    }) as unknown as HTMLCanvasElement;
    const keyboard = new EventTarget();
    const camera = new THREE.PerspectiveCamera(60, 2, 0.1, 100);
    camera.position.set(0, 2, -3);
    camera.lookAt(0, 0, 2);
    const onPick = vi.fn(),
      onError = vi.fn(),
      isPointerLocked = vi.fn(() => false);
    const pickSurface = vi.fn((): { point: THREE.Vector3; normal: THREE.Vector3 } | null => ({
      point: new THREE.Vector3(0, 0, 2),
      normal: new THREE.Vector3(0, 1, 0),
    }));
    const dispose = installArchitecturalPicking({
      canvas,
      keyboard,
      camera,
      spec: defaultArchitecturalSpec(),
      mode,
      isPointerLocked,
      pickSurface,
      onPick,
      onError,
    });
    const pointer = (button = 0) => {
      const event = Object.assign(new Event("pointerdown", { cancelable: true }), {
        button,
        clientX: 400,
        clientY: 200,
      });
      canvas.dispatchEvent(event);
      return event;
    };
    const escape = () => {
      const event = Object.assign(new Event("keydown", { cancelable: true }), { key: "Escape" });
      keyboard.dispatchEvent(event);
      return event;
    };
    return { canvas, keyboard, onPick, onError, isPointerLocked, pickSurface, dispose, pointer, escape };
  }
  it("advances a single threshold pick to direction and rejects duplicate events before the next mode binds", () => {
    const controls = setup();
    expect(controls.pointer(2).defaultPrevented).toBe(false);
    expect(controls.pointer().defaultPrevented).toBe(true);
    expect(controls.onPick).toHaveBeenCalledWith(expect.objectContaining({ origin: [0, 0, 2] }), "direction");
    controls.pointer();
    expect(controls.onPick).toHaveBeenCalledOnce();
    controls.dispose();
  });
  it("retains the draft after an invalid hit and allows a corrected retry", () => {
    const controls = setup();
    controls.pickSurface.mockReturnValueOnce(null);
    controls.pointer();
    expect(controls.onError).toHaveBeenCalledWith(expect.stringContaining("upward-facing"));
    expect(controls.onPick).not.toHaveBeenCalled();
    controls.pointer();
    expect(controls.onPick).toHaveBeenCalledOnce();
    controls.dispose();
  });
  it("aims on the threshold plane without asking a collider for hidden geometry", () => {
    const controls = setup("direction");
    controls.pointer();
    expect(controls.onPick).toHaveBeenCalledWith(expect.objectContaining({ yaw_degrees: 0 }), "inspect");
    expect(controls.pickSurface).not.toHaveBeenCalled();
    controls.dispose();
  });
  it("leaves pointer-lock exploration alone and permits Escape to exit picking after unlock", () => {
    const controls = setup();
    controls.isPointerLocked.mockReturnValue(true);
    expect(controls.pointer().defaultPrevented).toBe(false);
    expect(controls.escape().defaultPrevented).toBe(false);
    expect(controls.onPick).not.toHaveBeenCalled();
    controls.isPointerLocked.mockReturnValue(false);
    expect(controls.escape().defaultPrevented).toBe(true);
    expect(controls.onPick).toHaveBeenCalledWith(defaultArchitecturalSpec(), "inspect");
    controls.dispose();
  });
  it("does not consume inspection clicks and removes both listeners without disturbing other owners", () => {
    const controls = setup("inspect");
    const other = vi.fn();
    controls.canvas.addEventListener("pointerdown", other);
    expect(controls.pointer().defaultPrevented).toBe(false);
    expect(controls.escape().defaultPrevented).toBe(false);
    controls.dispose();
    controls.pointer();
    controls.escape();
    expect(other).toHaveBeenCalledTimes(2);
    expect(controls.onPick).not.toHaveBeenCalled();
    const picking = setup();
    picking.dispose();
    expect(picking.pointer().defaultPrevented).toBe(false);
    expect(picking.escape().defaultPrevented).toBe(false);
    expect(picking.onPick).not.toHaveBeenCalled();
  });
  it("blocks scene interaction, scale changes and walking shortcuts during draft exploration, not normal typing", () => {
    const controls = setup("inspect");
    const other = vi.fn();
    controls.keyboard.addEventListener("keydown", other);
    controls.isPointerLocked.mockReturnValue(true);
    for (const code of ["KeyF", "KeyE", "BracketLeft", "BracketRight"]) {
      const event = Object.assign(new Event("keydown", { cancelable: true }), { code });
      controls.keyboard.dispatchEvent(event);
      expect(event.defaultPrevented).toBe(true);
    }
    expect(other).not.toHaveBeenCalled();
    controls.isPointerLocked.mockReturnValue(false);
    const typing = Object.assign(new Event("keydown", { cancelable: true }), { code: "KeyF" });
    controls.keyboard.dispatchEvent(typing);
    expect(typing.defaultPrevented).toBe(false);
    expect(other).toHaveBeenCalledOnce();
    controls.dispose();
  });
});

describe("revision-bound architectural positioning", () => {
  const draft: ArchitecturalDraft = {
    jobId: "capture",
    base: { revision_id: "scene", generation: 4 },
    mode: "inspect",
  };
  it("shows a draft only for the same job, revision and generation without conflicting review", () => {
    expect(currentArchitecturalDraft(draft, "capture", draft.base, false)).toBe(draft);
    expect(currentArchitecturalDraft(draft, "other", draft.base, false)).toBeNull();
    expect(currentArchitecturalDraft(draft, "capture", { ...draft.base, generation: 5 }, false)).toBeNull();
    expect(currentArchitecturalDraft(draft, "capture", { ...draft.base, revision_id: "other" }, false)).toBeNull();
    expect(currentArchitecturalDraft(draft, "capture", draft.base, true)).toBeNull();
    expect(currentArchitecturalDraft(draft, "capture", null, false)).toBeNull();
    expect(currentArchitecturalDraft(null, "capture", draft.base, false)).toBeNull();
  });
  it.each(["coordinate", "width", "height", "clearance", "lintel", "yaw", "boolean", "depth"])(
    "refuses an invalid %s draft",
    (defect) => {
      const spec = defaultArchitecturalSpec();
      if (defect === "coordinate") spec.origin[0] = NaN;
      if (defect === "width") spec.opening_width = 0.2;
      if (defect === "height") spec.room_height = 7;
      if (defect === "clearance") spec.opening_width = spec.room_width;
      if (defect === "lintel") spec.opening_height = spec.room_height;
      if (defect === "yaw") spec.yaw_degrees = 181;
      if (defect === "boolean") spec.opening_width = true as unknown as number;
      if (defect === "depth") spec.cut_depth = 6.001;
      expect(architecturalSpecError(spec)).not.toBeNull();
      expect(() => createArchitecturalGuide(spec)).toThrow();
    },
  );
  it.each([0.1, 2, 4, 6])("keeps a %s m authored vestibule dimensioned without changing the body", (depth) => {
    const spec = {
      ...defaultArchitecturalSpec(),
      cut_depth: depth,
      yaw_degrees: 0,
      origin: [0, 0, 0] as [number, number, number],
    };
    expect(architecturalSpecError(spec)).toBeNull();
    const guide = createArchitecturalGuide(spec);
    expect(new THREE.Box3().setFromObject(guide.object).max.z).toBeGreaterThanOrEqual(depth / 2 + spec.room_depth);
    guide.dispose();
  });
  it("converts CSS viewport coordinates to the camera ray without a DPR assumption", () => {
    const camera = new THREE.PerspectiveCamera(60, 2, 0.1, 100);
    camera.position.set(1, 2, 3);
    camera.lookAt(1, 2, -3);
    const bounds = { left: 20, top: 30, width: 800, height: 400 };
    const ray = canvasPickRay(camera, bounds, 420, 230);
    expect(ray.origin.toArray()).toEqual([1, 2, 3]);
    expect(ray.direction.toArray()).toEqual([0, 0, -1]);
    expect(canvasPickRay(camera, bounds, 820, 30).direction.x).toBeGreaterThan(0);
    expect(canvasPickRay(camera, bounds, 820, 30).direction.y).toBeGreaterThan(0);
    expect(() => canvasPickRay(camera, bounds, 0, 30)).toThrow("inside");
    expect(() => canvasPickRay(camera, { ...bounds, width: 0 }, 20, 30)).toThrow("inside");
  });
  it("positions the threshold on the selected surface and initially aims away from the camera", () => {
    const spec = defaultArchitecturalSpec();
    const hit = { point: new THREE.Vector3(3, -1, 4), normal: new THREE.Vector3(0, 1, 0) };
    const next = positionArchitecturalThreshold(spec, hit, new THREE.Vector3(1, 0.7, 4));
    expect(next.origin).toEqual([3, -1, 4]);
    expect(next.yaw_degrees).toBeCloseTo(90);
    expect(spec.origin).toEqual([0, 0, 0]);
    expect(hit.point.toArray()).toEqual([3, -1, 4]);
  });
  it.each(["missing", "wall", "underside", "slope", "distant", "invalid"])("rejects a %s threshold hit", (defect) => {
    const hit = { point: new THREE.Vector3(), normal: new THREE.Vector3(0, 1, 0) };
    if (defect === "wall") hit.normal.set(1, 0, 0);
    if (defect === "underside") hit.normal.set(0, -1, 0);
    if (defect === "slope") hit.normal.set(1, 1, 0).normalize();
    if (defect === "distant") hit.point.x = 101;
    if (defect === "invalid") hit.point.z = Infinity;
    expect(() =>
      positionArchitecturalThreshold(
        defaultArchitecturalSpec(),
        defect === "missing" ? null : hit,
        new THREE.Vector3(0, 1, -1),
      ),
    ).toThrow();
  });
  it.each([0, 90, -90, 180])("aims outward at %s degrees on the exact threshold-height plane", (degrees) => {
    const spec = { ...defaultArchitecturalSpec(), origin: [3, -1, 4] };
    const target = new THREE.Vector3(
      Math.sin(THREE.MathUtils.degToRad(degrees)),
      0,
      Math.cos(THREE.MathUtils.degToRad(degrees)),
    )
      .multiplyScalar(2)
      .add(new THREE.Vector3().fromArray(spec.origin));
    const camera = target.clone().add(new THREE.Vector3(0, 3, 0));
    const next = aimArchitecturalThreshold(spec, new THREE.Ray(camera, new THREE.Vector3(0, -1, 0)));
    expect(Math.abs(next.yaw_degrees)).toBeCloseTo(Math.abs(degrees));
    expect(next.origin).toEqual(spec.origin);
    expect(next.opening_width).toBe(spec.opening_width);
  });
  it.each(["parallel", "behind", "too-close", "distant", "invalid"])("refuses a %s outward aim", (defect) => {
    const ray = new THREE.Ray(new THREE.Vector3(0, 2, 0), new THREE.Vector3(0, -1, 0));
    if (defect === "parallel") ray.direction.set(1, 0, 0);
    if (defect === "behind") ray.direction.set(0, 1, 0);
    if (defect === "distant") ray.origin.set(0, 101, 0);
    if (defect === "invalid") ray.origin.x = NaN;
    expect(() => aimArchitecturalThreshold(defaultArchitecturalSpec(), ray)).toThrow();
  });
  it("draws the authored cut, room interior and bridge in the backend's Y-up frame", () => {
    const spec = { ...defaultArchitecturalSpec(), origin: [3, -1, 4], yaw_degrees: 90 };
    const guide = createArchitecturalGuide(spec);
    guide.object.updateMatrixWorld(true);
    const cut = guide.object.getObjectByName("draft-cut-volume") as THREE.LineSegments;
    cut.geometry.computeBoundingBox();
    const bounds = cut.geometry.boundingBox!.clone().translate(cut.position);
    expect(bounds.min.x).toBeCloseTo(-spec.opening_width / 2);
    expect(bounds.min.y).toBeCloseTo(0.015);
    expect(bounds.min.z).toBeCloseTo(-spec.cut_depth / 2);
    expect(bounds.max.y).toBeCloseTo(spec.opening_height);
    const room = guide.object.getObjectByName("draft-room-interior")!;
    const worldCenter = room.getWorldPosition(new THREE.Vector3());
    expect(worldCenter.x).toBeCloseTo(3 + spec.cut_depth / 2 + spec.room_depth / 2);
    expect(worldCenter.y).toBeCloseTo(-1 + spec.room_height / 2);
    expect(worldCenter.z).toBeCloseTo(4);
    expect(guide.object.getObjectByName("draft-bridge-floor")!.position.y).toBeCloseTo(-spec.wall_thickness / 2);
    expect(guide.object.userData.scope).toContain("not a cut");
    guide.dispose();
  });
  it("disposes only its own overlay resources and does so once", () => {
    const scene = new THREE.Scene();
    const existing = new THREE.ArrowHelper();
    scene.add(existing);
    const guide = createArchitecturalGuide(defaultArchitecturalSpec());
    scene.add(guide.object);
    const arrow = guide.object.getObjectByName("draft-outward-direction") as THREE.ArrowHelper;
    expect(arrow.line.geometry).not.toBe(existing.line.geometry);
    expect(arrow.cone.geometry).not.toBe(existing.cone.geometry);
    const owned = vi.spyOn(arrow.cone.geometry, "dispose");
    const preserved = vi.spyOn(existing.cone.geometry, "dispose");
    guide.dispose();
    guide.dispose();
    expect(owned).toHaveBeenCalledOnce();
    expect(preserved).not.toHaveBeenCalled();
    expect(scene.children).toEqual([existing]);
    existing.dispose();
  });
});
