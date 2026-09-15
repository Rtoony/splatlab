import * as THREE from "three";

export type ArchitecturalSpec = {
  origin: number[];
  yaw_degrees: number;
  opening_width: number;
  opening_height: number;
  cut_depth: number;
  room_width: number;
  room_depth: number;
  room_height: number;
  wall_thickness: number;
};
export type ArchitecturalPickMode = "inspect" | "position" | "direction";
export type ArchitecturalDraft = {
  jobId: string;
  base: { revision_id: string; generation: number };
  mode: ArchitecturalPickMode;
};

export const architecturalDimensions: [Exclude<keyof ArchitecturalSpec, "origin">, string, number, number][] = [
  ["yaw_degrees", "Portal outward yaw (degrees)", -180, 180],
  ["opening_width", "Opening width (m)", 0.8, 3],
  ["opening_height", "Opening height (m)", 1.9, 3.5],
  ["cut_depth", "Authored cut / vestibule depth (m)", 0.1, 6],
  ["room_width", "Room width (m)", 1.5, 8],
  ["room_depth", "Room depth (m)", 1.5, 8],
  ["room_height", "Room height (m)", 2.1, 5],
  ["wall_thickness", "Authored wall thickness (m)", 0.08, 0.4],
];

export function defaultArchitecturalSpec(): ArchitecturalSpec {
  return {
    origin: [0, 0, 0],
    yaw_degrees: 0,
    opening_width: 1.1,
    opening_height: 2.1,
    cut_depth: 0.6,
    room_width: 3,
    room_depth: 3,
    room_height: 2.7,
    wall_thickness: 0.15,
  };
}

export function architecturalSpecError(spec: ArchitecturalSpec): string | null {
  if (
    !Array.isArray(spec.origin) ||
    spec.origin.length !== 3 ||
    spec.origin.some((value) => !Number.isFinite(value) || Math.abs(value) > 100)
  )
    return "Threshold coordinates must be finite world metres within ±100m.";
  for (const [key, label, minimum, maximum] of architecturalDimensions) {
    if (!Number.isFinite(spec[key]) || spec[key] < minimum || spec[key] > maximum)
      return `${label} must be between ${minimum} and ${maximum}.`;
  }
  if (spec.room_width < spec.opening_width + 0.3) return "The room must leave at least 0.3m beyond the opening width.";
  if (spec.room_height < spec.opening_height + 0.15) return "The room must leave at least 0.15m above the opening.";
  return null;
}

export function currentArchitecturalDraft(
  draft: ArchitecturalDraft | null,
  jobId: string,
  base: { revision_id: string; generation: number } | null | undefined,
  blocked: boolean,
): ArchitecturalDraft | null {
  return !blocked &&
    draft?.jobId === jobId &&
    draft.base.revision_id === base?.revision_id &&
    draft.base.generation === base?.generation
    ? draft
    : null;
}

export function canvasPickRay(
  camera: THREE.PerspectiveCamera,
  bounds: { left: number; top: number; width: number; height: number },
  clientX: number,
  clientY: number,
): THREE.Ray {
  if (
    ![bounds.left, bounds.top, bounds.width, bounds.height, clientX, clientY].every(Number.isFinite) ||
    bounds.width <= 0 ||
    bounds.height <= 0 ||
    clientX < bounds.left ||
    clientY < bounds.top ||
    clientX > bounds.left + bounds.width ||
    clientY > bounds.top + bounds.height
  )
    throw new Error("Pick inside the loaded scene viewport.");
  camera.updateMatrixWorld();
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(
    new THREE.Vector2(
      (2 * (clientX - bounds.left)) / bounds.width - 1,
      1 - (2 * (clientY - bounds.top)) / bounds.height,
    ),
    camera,
  );
  return raycaster.ray.clone();
}

export function positionArchitecturalThreshold(
  spec: ArchitecturalSpec,
  hit: { point: THREE.Vector3; normal: THREE.Vector3 } | null,
  cameraPosition: THREE.Vector3,
): ArchitecturalSpec {
  if (
    !hit ||
    ![...hit.point.toArray(), ...hit.normal.toArray(), ...cameraPosition.toArray()].every(Number.isFinite) ||
    hit.normal.lengthSq() < 0.5 ||
    hit.normal.clone().normalize().y < 0.85
  )
    throw new Error(
      "Pick a visible, upward-facing collision surface near the intended threshold, not a wall or underside.",
    );
  const outward = hit.point.clone().sub(cameraPosition).setY(0);
  const next = {
    ...spec,
    origin: hit.point.toArray(),
    yaw_degrees:
      outward.lengthSq() >= 0.04 ? THREE.MathUtils.radToDeg(Math.atan2(outward.x, outward.z)) : spec.yaw_degrees,
  };
  const error = architecturalSpecError(next);
  if (error) throw new Error(error);
  return next;
}

export function aimArchitecturalThreshold(spec: ArchitecturalSpec, ray: THREE.Ray): ArchitecturalSpec {
  const error = architecturalSpecError(spec);
  if (error) throw new Error(error);
  if (![...ray.origin.toArray(), ...ray.direction.toArray()].every(Number.isFinite) || Math.abs(ray.direction.y) < 1e-6)
    throw new Error("Aim at the threshold-height plane, away from the horizon.");
  const target = ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -spec.origin[1]), new THREE.Vector3());
  if (!target || target.distanceTo(ray.origin) > 100)
    throw new Error("The threshold-height plane is behind the camera or too far away.");
  const direction = target.sub(new THREE.Vector3().fromArray(spec.origin)).setY(0);
  if (direction.lengthSq() < 0.04) throw new Error("Choose an outward point at least 0.2m from the threshold.");
  return {
    ...spec,
    origin: [...spec.origin],
    yaw_degrees: THREE.MathUtils.radToDeg(Math.atan2(direction.x, direction.z)),
  };
}

export function installArchitecturalPicking(options: {
  canvas: HTMLCanvasElement;
  keyboard: EventTarget;
  camera: THREE.PerspectiveCamera;
  spec: ArchitecturalSpec;
  mode: ArchitecturalPickMode;
  isPointerLocked: () => boolean;
  pickSurface: (ray: THREE.Ray) => { point: THREE.Vector3; normal: THREE.Vector3 } | null;
  onPick: (spec: ArchitecturalSpec, mode: ArchitecturalPickMode) => void;
  onError: (message: string) => void;
}): () => void {
  let accepting = true;
  const pointer = (raw: Event) => {
    const event = raw as PointerEvent;
    if (!accepting || event.button !== 0 || options.mode === "inspect" || options.isPointerLocked()) return;
    event.preventDefault();
    event.stopPropagation();
    try {
      const ray = canvasPickRay(options.camera, options.canvas.getBoundingClientRect(), event.clientX, event.clientY);
      const next =
        options.mode === "position"
          ? positionArchitecturalThreshold(options.spec, options.pickSurface(ray), options.camera.position)
          : aimArchitecturalThreshold(options.spec, ray);
      accepting = false;
      options.onError("");
      options.onPick(next, options.mode === "position" ? "direction" : "inspect");
    } catch (reason) {
      options.onError(reason instanceof Error ? reason.message : String(reason));
    }
  };
  const key = (raw: Event) => {
    const event = raw as KeyboardEvent;
    if (options.isPointerLocked() && ["KeyF", "KeyE", "BracketLeft", "BracketRight"].includes(event.code)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }
    if (!accepting || options.mode === "inspect" || options.isPointerLocked() || event.key !== "Escape") return;
    event.preventDefault();
    event.stopPropagation();
    accepting = false;
    options.onError("");
    options.onPick(options.spec, "inspect");
  };
  options.canvas.addEventListener("pointerdown", pointer, true);
  options.keyboard.addEventListener("keydown", key, true);
  return () => {
    accepting = false;
    options.canvas.removeEventListener("pointerdown", pointer, true);
    options.keyboard.removeEventListener("keydown", key, true);
  };
}

export function createArchitecturalGuide(spec: ArchitecturalSpec): { object: THREE.Group; dispose: () => void } {
  const error = architecturalSpecError(spec);
  if (error) throw new Error(error);
  const object = new THREE.Group();
  object.name = "architectural-draft-guide";
  object.userData = { scope: "Authored/inferred positioning aid, not a cut, collider or measured surface" };
  object.position.fromArray(spec.origin);
  object.rotation.y = THREE.MathUtils.degToRad(spec.yaw_degrees);
  const box = (name: string, size: number[], center: number[], color: number) => {
    const source = new THREE.BoxGeometry(...(size as [number, number, number]));
    const geometry = new THREE.EdgesGeometry(source);
    source.dispose();
    const material = new THREE.LineDashedMaterial({
      color,
      dashSize: 0.1,
      gapSize: 0.05,
      depthTest: false,
      depthWrite: false,
    });
    const lines = new THREE.LineSegments(geometry, material);
    lines.computeLineDistances();
    lines.name = name;
    lines.position.fromArray(center);
    lines.renderOrder = 100;
    object.add(lines);
  };
  box(
    "draft-cut-volume",
    [spec.opening_width, spec.opening_height - 0.015, spec.cut_depth],
    [0, (spec.opening_height + 0.015) / 2, 0],
    0xfbbf24,
  );
  box(
    "draft-room-interior",
    [spec.room_width, spec.room_height, spec.room_depth],
    [0, spec.room_height / 2, spec.cut_depth / 2 + spec.room_depth / 2],
    0x22d3ee,
  );
  box(
    "draft-bridge-floor",
    [spec.opening_width + 0.1, spec.wall_thickness, spec.cut_depth + 0.3],
    [0, -spec.wall_thickness / 2, -0.15],
    0x86efac,
  );
  const arrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0.1, 0),
    spec.cut_depth / 2 + Math.min(spec.room_depth / 2, 1),
    0x22d3ee,
    0.2,
    0.12,
  );
  arrow.name = "draft-outward-direction";
  arrow.line.geometry = arrow.line.geometry.clone();
  arrow.cone.geometry = arrow.cone.geometry.clone();
  object.add(arrow);
  object.traverse((child) => {
    const rendered = child as THREE.Mesh;
    if (!rendered.material) return;
    for (const material of Array.isArray(rendered.material) ? rendered.material : [rendered.material]) {
      material.depthTest = false;
      material.depthWrite = false;
    }
    child.renderOrder = 100;
  });
  let disposed = false;
  return {
    object,
    dispose: () => {
      if (disposed) return;
      disposed = true;
      object.removeFromParent();
      object.traverse((child) => {
        const rendered = child as THREE.Mesh;
        rendered.geometry?.dispose();
        if (rendered.material)
          for (const material of Array.isArray(rendered.material) ? rendered.material : [rendered.material])
            material.dispose();
      });
    },
  };
}
